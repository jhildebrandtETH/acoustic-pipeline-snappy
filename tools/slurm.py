"""Durable Slurm coordinator. Only the coordinator submits; case jobs run the state machine.

Jobs are submitted held. A unique intent is persisted BEFORE sbatch, and the job ID
BEFORE release. An ambiguous submission is reconciled by its unique name; absence
of accounting evidence never authorizes a second submission.
"""
from argparse import Namespace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

from tools.common import _atomic_write_json
from tools.process_lock import file_lock
from tools.scheduler import (
    SimulationOrderStore, RuntimeStatusRegistry, LiveBatchDashboard,
    initialize_runtime_queue_states, refresh_dependency_runtime_states,
    block_cases_with_failed_dependencies, runnable_cases, dependency_state,
    reactivate_failed_cases_for_resume, execute_simulation_case,
)

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED', 'SPECIAL_EXIT'}
CASE_TERMINAL = {'postprocessing_done', 'failed', 'blocked'}


def validate_config(args, saved=None):
    config = json.loads(args.slurm_config.read_text()) if args.slurm_config else saved
    if not config:
        raise ValueError('A new Slurm order requires --slurm-config FILE')
    if not isinstance(config, dict):
        raise ValueError('Slurm config must be a JSON object')
    config = dict(config)
    allowed = {'partition', 'account', 'qos', 'constraint', 'time', 'mem',
               'openfoam_bashrc', 'mpi_launcher', 'setup', 'python', 'poll_interval'}
    if set(config) - allowed:
        raise ValueError(f'Unknown Slurm settings: {sorted(set(config) - allowed)}')
    for key in ('time', 'mem', 'openfoam_bashrc'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'Slurm config requires {key}')
    config.setdefault('mpi_launcher', ['mpirun', '-np', '{cores}'])
    launcher = config['mpi_launcher']
    if (not isinstance(launcher, list) or not launcher or
            any(not isinstance(p, str) or not p for p in launcher) or
            '{cores}' not in launcher):
        raise ValueError('mpi_launcher must be an argument list containing {cores}')
    config.setdefault('setup', '')
    config.setdefault('python', sys.executable)
    config.setdefault('poll_interval', 10)
    if not math.isfinite(float(config['poll_interval'])) or float(config['poll_interval']) < 1:
        raise ValueError('Slurm poll_interval must be at least 1 second')
    for key, value in config.items():
        if key not in {'mpi_launcher', 'poll_interval'} and not isinstance(value, str):
            raise ValueError(f'Slurm {key} must be a string')
    if saved and config != saved:
        raise ValueError('Slurm configuration is immutable on resume; use the saved settings')
    return config


def configure_order(store, args):
    order = store.snapshot()
    saved = order.get('slurm')
    config = validate_config(args, saved)
    if not saved:
        # Store only execution options, with all scientific options already restored.
        options = {key: getattr(args, key, None) for key in (
            'field_init', 'turbulence', 'wall_functions', 'aerodynamics_only',
            'acoustic_surface', 'acoustic_sphere_diameter', 'end_on', 'end_on_value',
            'mesh_only', 'allow_bad_mesh')}
        options['live_output'] = True
        store.persist_batch_fields(slurm=config, slurm_worker_options=options)


class SlurmClient:
    @staticmethod
    def run(command, **kwargs):
        # Allocation variables from the small coordinator must not override case
        # resource flags, memory, export policy, or imply an inherited dependency.
        env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        if command[0] == 'sbatch':
            # Keep Slurm connection/auth configuration, but do not export the
            # coordinator's allocation into children (notably conflicting
            # SLURM_MEM_PER_CPU / SLURM_MEM_PER_NODE settings used by srun).
            connection = {'SLURM_CONF', 'SLURM_CONF_SERVER', 'SLURM_JWT'}
            env = {k: v for k, v in env.items() if not k.startswith('SLURM_') or k in connection}
        try:
            return subprocess.run(command, env=env, text=True, capture_output=True,
                                  check=True, timeout=60, **kwargs).stdout.strip()
        except (subprocess.SubprocessError, OSError) as error:
            detail = getattr(error, 'stderr', None) or str(error)
            raise RuntimeError(f'{command[0]} failed: {detail}. Submission identities retained; resume to reconcile.') from error

    def submit(self, script, config, cores, intent, log):
        command = ['sbatch', '--parsable', '--hold', '--no-requeue', '--export=ALL',
                   '--nodes=1', f'--ntasks={cores}', '--cpus-per-task=1',
                   f'--job-name={intent["name"]}', f'--output={log}', f'--error={log}',
                   f'--chdir={script.parent}', f'--time={config["time"]}',
                   f'--mem={config["mem"]}']
        for key in ('partition', 'account', 'qos', 'constraint'):
            if config.get(key):
                command.append(f'--{key}={config[key]}')
        result = self.run(command + [str(script)])
        if not re.fullmatch(r'\d+', result):
            # Federated/remote clusters need explicit identity and command routing;
            # refuse rather than accidentally monitor a different local job ID.
            raise RuntimeError(f'Unexpected sbatch identity: {result!r}; submission intent retained')
        return result

    def records(self, intents):
        if not intents:
            return {}
        names = {item['name'] for item in intents}
        queue = self.run(['squeue', '--me', '--noheader', '--format=%i|%j|%T|%r'])
        start = min(item['created'] for item in intents)
        accounting = self.run(['sacct', '--allocations', '--noheader', '--parsable2',
                               '--starttime', start, '--name', ','.join(sorted(names)),
                               '--format=JobIDRaw,JobName%128,State,ExitCode'])
        records = {}
        for output, queued in ((accounting, False), (queue, True)):
            for line in output.splitlines():
                fields = line.strip().split('|')
                if len(fields) < 4:
                    raise RuntimeError(f'Malformed Slurm response: {line!r}')
                job_id, name, state, detail = fields[:4]
                if name not in names or not job_id.isdigit():
                    continue
                records[job_id] = dict(id=job_id, name=name,
                                       state=state.split()[0].rstrip('+'),
                                       reason=detail if queued else '',
                                       exit_code='' if queued else detail)
        return records

    def release(self, job_id):
        self.run(['scontrol', 'release', job_id])


def runtime_path(root, token):
    return Path(root) / '.slurm' / f'{token}.runtime.json'


class WorkerRegistry(RuntimeStatusRegistry):
    def __init__(self, case, root, token):
        super().__init__([case])
        self.path = runtime_path(root, token)

    def update(self, folder_name, **updates):
        with self._lock:
            super().update(folder_name, **updates)
            _atomic_write_json(self.path, self.get(folder_name))


def refresh_saved_table(store, registry):
    order = store.snapshot()
    active = {c['folder'] for c in order['cases'] if c.get('slurm_job') and
              not c['slurm_job'].get('terminal')}
    refresh_dependency_runtime_states(order, registry, active)
    for case in order['cases']:
        job = case.get('slurm_job')
        if not job or job.get('terminal'):
            continue
        state = job.get('state', 'SUBMITTING')
        path = runtime_path(store.simulations_directory, job['token'])
        if state in {'RUNNING', 'COMPLETING'} or path.is_file():
            registry.update(case['folder'], state='RUNNING', stage='starting',
                            detail=f'Slurm {job.get("id", "?")} {state}')
            if path.is_file():
                registry.update(case['folder'], **{k: v for k, v in json.loads(path.read_text()).items()
                                                  if k != 'folder'})
        else:
            registry.update(case['folder'], state='QUEUED', stage='slurm_queue', progress=None,
                            detail=f'Slurm {job.get("id", "?")} {state}: {job.get("reason", "")}')


def script_text(root, pipeline, config, folder, token):
    command = [config['python'], str(pipeline / 'tools' / 'slurm_worker.py'),
               '--sim-dir', str(root), '--case', folder, '--token', token]
    return ('#!/bin/bash\nset -eo pipefail\n' + config['setup'] +
            '\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec ' +
            shlex.join(command) + '\n')


class Coordinator:
    def __init__(self, store, registry, pipeline, client=None):
        self.store = store
        self.registry = registry
        self.pipeline = Path(pipeline).resolve()
        self.root = store.simulations_directory.resolve()
        self.config = store.snapshot()['slurm']
        self.client = client or SlurmClient()
        (self.root / '.slurm').mkdir(exist_ok=True)

    def reconcile(self):
        cases = self.store.snapshot()['cases']
        intents = [c['slurm_job'] for c in cases if c.get('slurm_job') and not c['slurm_job'].get('terminal')]
        records = self.client.records(intents)
        for case in cases:
            job = case.get('slurm_job')
            if not job or job.get('terminal'):
                continue
            matches = [r for r in records.values() if r['name'] == job['name']]
            if len(matches) > 1:
                raise RuntimeError(f'Multiple Slurm jobs for {job["name"]}; refusing to continue')
            record = records.get(job.get('id')) if job.get('id') else (matches[0] if matches else None)
            if record is None:
                if not job.get('id'):
                    raise RuntimeError(f'Unresolved submission intent {job["name"]}. Check squeue/sacct; no automatic resubmission.')
                job.update(state='ACCOUNTING_WAIT', reason='absent from squeue/sacct; retaining job identity')
                self.store.update_case(case['folder'], slurm_job=job)
                continue
            job.update(record)
            if record['state'] in TERMINAL:
                job['terminal'] = True
                # Re-read after Slurm says the worker has exited, never from the
                # stale snapshot taken before querying accounting.
                current = self.store.get_case(case['folder'])
                if current['status'] not in {'postprocessing_done', 'failed'} or (
                        record['state'] != 'COMPLETED' or record['exit_code'] not in {'', '0:0'}):
                    detail = f'Slurm {job["id"]}: {record["state"]} {record["exit_code"]}; checkpoint {current["status"]}'
                    if current.get('error'):
                        detail += f'; {current["error"]}'
                    self.store.mark_failed(case['folder'],
                        detail,
                        current.get('resume_status') or current['status'])
            self.store.update_case(case['folder'], slurm_job=job)
            if record['state'] == 'PENDING' and record['reason'] == 'JobHeldUser':
                # ID is durable, and release is safe to repeat after a crash.
                self.client.release(job['id'])

    def resume(self):
        self.reconcile()
        active = [c for c in self.store.snapshot()['cases'] if c.get('slurm_job') and not c['slurm_job'].get('terminal')]
        # Reactivation cannot touch a case with a live job, even if it has just
        # recorded a Python failure and is still leaving its allocation.
        if any(c['status'] in {'failed', 'blocked'} for c in active):
            raise RuntimeError('A failed case job is still active in Slurm; resume after it exits')
        active_folders = {c['folder'] for c in active}
        eligible = {c['folder'] for c in self.store.snapshot()['cases']} - active_folders
        reactivate_failed_cases_for_resume(self.store, self.root, eligible_folders=eligible)

    def tick(self):
        self.reconcile()
        block_cases_with_failed_dependencies(self.store, self.registry)
        order = self.store.snapshot()
        active = {c['folder'] for c in order['cases'] if c.get('slurm_job') and not c['slurm_job'].get('terminal')}
        free = int(order['max_parallel_cases']) - len(active)
        ready = [c for c in runnable_cases(order, active) if c.get('depends_on') not in active]
        for case in ready[:max(0, free)]:
            token = uuid.uuid4().hex
            intent = dict(token=token, name=f'acoustic-{token}',
                          created=datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                          state='SUBMITTING', terminal=False)
            script = self.root / '.slurm' / f'{token}.sh'
            script.write_text(script_text(self.root, self.pipeline, self.config, case['folder'], token), encoding='utf-8', newline='\n')
            old = case.get('slurm_job')
            history = list(case.get('slurm_history', [])) + ([old] if old else [])
            self.store.update_case(case['folder'], slurm_job=intent, slurm_history=history)
            job_id = self.client.submit(script, self.config, int(case['allocated_cores']), intent,
                                        self.root / '.slurm' / f'{token}-%j.log')
            intent.update(id=job_id, state='PENDING', reason='JobHeldUser')
            self.store.update_case(case['folder'], slurm_job=intent)
            self.client.release(job_id)
        refresh_saved_table(self.store, self.registry)
        order = self.store.snapshot()
        active = any(c.get('slurm_job') and not c['slurm_job'].get('terminal') for c in order['cases'])
        unfinished = any(c['status'] not in CASE_TERMINAL for c in order['cases'])
        if unfinished and not active and not runnable_cases(order, set()):
            raise RuntimeError('Scheduler deadlock: check case dependencies')
        return active or unfinished


def run_slurm_scheduler(pipeline_main_directory, simulations_directory, order_store,
                        registry, args, **kwargs):
    coordinator = Coordinator(order_store, registry, pipeline_main_directory)
    if args.resume:
        coordinator.resume()
    order = order_store.snapshot()
    dashboard = LiveBatchDashboard(registry, order['total_cores'], order['cores_per_case'])
    dashboard.start()
    try:
        while coordinator.tick():
            time.sleep(float(coordinator.config['poll_interval']))
    finally:
        dashboard.stop()


def show_status(root, watch=False):
    store = SimulationOrderStore(Path(root).resolve())
    order = store.snapshot()
    registry = RuntimeStatusRegistry(order['cases'])
    initialize_runtime_queue_states(order, registry)
    dashboard = LiveBatchDashboard(registry, order['total_cores'], order['cores_per_case'])
    refresh_saved_table(store, registry)
    if not watch:
        print(dashboard.render_text())
        return
    dashboard.start()
    try:
        while True:
            refresh_saved_table(store, registry)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()


def run_worker(root, folder, token):
    from tools.native_openfoam import native_preflight
    from tools.openfoam import find_source_stls
    root = Path(root).resolve()
    store = SimulationOrderStore(root)
    case = store.get_case(folder)
    job = case.get('slurm_job', {})
    if (job.get('token') != token or job.get('id') != os.environ.get('SLURM_JOB_ID') or job.get('terminal')):
        raise RuntimeError('Worker identity does not match the durable Slurm submission')
    with file_lock(root / '.slurm' / f'{token}.worker.lock', blocking=False):
        order = store.snapshot()
        if dependency_state(case, {c['folder']: c['status'] for c in order['cases']})[0] != 'READY':
            raise RuntimeError('Initialization dependency is not complete')
        if int(os.environ.get('SLURM_NTASKS', '0')) != int(case['allocated_cores']):
            raise RuntimeError('Slurm task allocation differs from saved case decomposition')
        config = order['slurm']
        os.environ.update(ACOUSTIC_SLURM_NATIVE='1', ACOUSTIC_OPENFOAM_BASHRC=config['openfoam_bashrc'],
                          ACOUSTIC_MPI_LAUNCHER=json.dumps(config['mpi_launcher']))
        registry = WorkerRegistry(case, root, token)
        try:
            native_preflight(config)
        except Exception as error:
            store.mark_failed(folder, f'Native OpenFOAM preflight failed: {error}', case['status'])
            return False
        return execute_simulation_case(case, Path(__file__).resolve().parent.parent, root,
                                       root / 'STL', find_source_stls(root / 'STL'), store,
                                       registry, Namespace(**order['slurm_worker_options']),
                                       convergence_tolerance=1e-2)

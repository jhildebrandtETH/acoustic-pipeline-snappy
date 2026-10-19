"""Slurm scheduling tests use a fake control plane; no cluster is required."""
from argparse import Namespace
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.cli import create_parser
from tools.common import _atomic_write_json
from tools.scheduler import (SimulationOrderStore, RuntimeStatusRegistry,
                             LiveBatchDashboard, run_parallel_scheduler)
from tools.slurm import (Coordinator, SlurmClient, WorkerRegistry, configure_order,
                         refresh_saved_table, run_worker, script_text, show_status)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = dict(time='01:00:00', mem='8G', runtime='docker', docker_image='microfluidica/openfoam:13',
              python='/env/bin/python', setup='', poll_interval=1)


def update_in_process(root, folder):
    store = SimulationOrderStore(Path(root))
    for i in range(15):
        store.update_case(folder, counter=i)
        store.persist_batch_fields(**{f'field_{folder}': i})


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.submissions = []
        self.releases = []
        self.fail_submit = False

    def records(self, intents):
        return dict(self.jobs)

    def submit(self, script, config, cores, intent, log):
        job_id = str(100 + len(self.submissions))
        self.submissions.append((script, cores, intent.copy()))
        self.jobs[job_id] = dict(id=job_id, name=intent['name'], state='PENDING',
                                 reason='JobHeldUser', exit_code='')
        if self.fail_submit:
            raise TimeoutError('response lost after acceptance')
        return job_id

    def release(self, job_id):
        self.releases.append(job_id)
        self.jobs[job_id] = dict(self.jobs[job_id], reason='Resources')

    def finish(self, job_id, state='COMPLETED', exit_code='0:0'):
        self.jobs[job_id] = dict(self.jobs[job_id], state=state, exit_code=exit_code, reason='')


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cases = [dict(folder=f'case{i}', status='pending', mesh='prop', rpm=3000+i,
                      allocated_cores=2, depends_on=None, mode='AMI', study=False,
                      mesh_only=True) for i in range(3)]
        _atomic_write_json(self.root / 'simulation_order.json', dict(
            cases=cases, total_cores=4, cores_per_case=2, max_parallel_cases=2,
            meshing_backend='snappy-native-v1', slurm=CONFIG, slurm_worker_options={}))
        self.store = SimulationOrderStore(self.root)
        self.registry = RuntimeStatusRegistry(cases)
        self.client = FakeSlurm()
        self.coordinator = Coordinator(self.store, self.registry, ROOT, self.client)

    def test_submission_resource_limit_and_queue_table(self):
        self.assertTrue(self.coordinator.tick())
        self.assertEqual(len(self.client.submissions), 2)
        self.assertEqual([x[1] for x in self.client.submissions], [2, 2])
        saved = self.store.get_case('case0')['slurm_job']
        self.assertEqual(saved['id'], '100')
        self.assertEqual(self.client.releases, ['100', '101'])
        self.coordinator.tick()
        self.assertEqual(len(self.client.submissions), 2)
        table = LiveBatchDashboard(self.registry, 4, 2).render_text()
        self.assertIn('slurm_queue', table)
        self.assertIn('Resources', table)
        text = self.client.submissions[0][0].read_text()
        self.assertTrue(text.startswith('#!/bin/bash\n'))
        self.assertIn('srun --mpi=none', text)
        self.assertIn('--cpu-bind=threads', text)
        self.assertIn('slurm_worker.py', text)

    def test_progress_visible_to_independent_watcher(self):
        self.coordinator.tick()
        case = self.store.get_case('case0')
        worker = WorkerRegistry(case, self.root, case['slurm_job']['token'])
        worker.update('case0', state='RUNNING', stage='snappyHexMesh', progress=42.5)
        watcher = RuntimeStatusRegistry(self.store.snapshot()['cases'])
        refresh_saved_table(SimulationOrderStore(self.root), watcher)
        self.assertEqual(watcher.get('case0')['stage'], 'snappyHexMesh')
        self.assertEqual(watcher.get('case0')['progress'], 42.5)
        self.assertIn('42.50%', LiveBatchDashboard(watcher, 4, 2).render_text())
        with patch('builtins.print') as output:
            show_status(self.root)
        self.assertIn('snappyHexMesh', output.call_args.args[0])

    def test_dependency_waits_for_completed_job_then_submits(self):
        self.store.update_case('case1', depends_on='case0')
        self.store.update_case('case2', depends_on='case1')
        self.coordinator.tick()
        self.assertEqual(len(self.client.submissions), 1)
        self.store.set_status('case0', 'postprocessing_done')
        self.coordinator.tick()  # checkpoint alone does not release a dependency
        self.assertEqual(len(self.client.submissions), 1)
        self.client.finish('100')
        self.coordinator.tick()
        self.assertEqual(len(self.client.submissions), 2)
        self.assertEqual(self.store.get_case('case1')['slurm_job']['id'], '101')

    def test_scheduler_failures_block_descendants_and_preserve_checkpoint(self):
        for state in ('TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'FAILED'):
            with self.subTest(state=state):
                self.setUp()
                self.store.update_case('case1', depends_on='case0')
                self.store.update_case('case2', depends_on='case1')
                self.coordinator.tick()
                self.store.set_status('case0', 'solver_done')
                self.client.finish('100', state, '1:0')
                self.assertFalse(self.coordinator.tick())
                self.assertEqual(self.store.case_status('case0'), 'failed')
                self.assertEqual(self.store.get_case('case0')['resume_status'], 'solver_done')
                self.assertEqual(self.store.case_status('case1'), 'blocked')
                self.assertEqual(self.store.case_status('case2'), 'blocked')

    def test_completed_without_checkpoint_is_failure(self):
        self.coordinator.tick()
        self.client.finish('100')
        self.coordinator.tick()
        self.assertEqual(self.store.case_status('case0'), 'failed')

    def test_python_failure_does_not_restart_automatically(self):
        self.coordinator.tick()
        self.store.mark_failed('case0', 'report failed', 'solver_done')
        self.client.finish('100', 'FAILED', '1:0')
        self.coordinator.tick()
        self.coordinator.tick()
        self.assertEqual(len([c for c in self.client.submissions if c[2]['token'] ==
                             self.store.get_case('case0')['slurm_job']['token']]), 1)
        self.assertEqual(self.store.get_case('case0')['resume_status'], 'solver_done')

    def test_restart_reattaches_pending_running_and_absent_jobs(self):
        self.coordinator.tick()
        self.client.jobs['100']['state'] = 'RUNNING'
        self.client.jobs.pop('101')  # accounting delay must not cause duplication
        restarted = Coordinator(SimulationOrderStore(self.root), self.registry, ROOT, self.client)
        restarted.resume()
        restarted.tick()
        self.assertEqual(len(self.client.submissions), 2)
        self.assertEqual(self.store.get_case('case1')['slurm_job']['state'], 'ACCOUNTING_WAIT')

    def test_resume_cannot_reactivate_job_failing_during_reconciliation(self):
        from tools.scheduler import reactivate_failed_cases_for_resume
        self.coordinator.tick()
        def fail_then_reactivate(store, root, **kwargs):
            # A live worker writes failure after resume inspected active statuses.
            store.mark_failed('case0', 'worker still exiting', 'pending')
            reactivate_failed_cases_for_resume(store, root, **kwargs)
        with patch('tools.slurm.reactivate_failed_cases_for_resume', side_effect=fail_then_reactivate):
            self.coordinator.resume()
        self.assertEqual(self.store.case_status('case0'), 'failed')
        self.assertFalse(self.store.get_case('case0')['slurm_job']['terminal'])

    def test_submission_response_lost_recovers_and_releases_held_job(self):
        self.client.fail_submit = True
        with self.assertRaises(TimeoutError):
            self.coordinator.tick()
        self.assertNotIn('id', self.store.get_case('case0')['slurm_job'])
        self.client.fail_submit = False
        restarted = Coordinator(self.store, self.registry, ROOT, self.client)
        restarted.resume()
        self.assertEqual(self.store.get_case('case0')['slurm_job']['id'], '100')
        self.assertEqual(self.client.releases, ['100'])
        restarted.tick()
        self.assertEqual(len(self.client.submissions), 2)  # only the other case

    def test_unresolved_intent_and_control_plane_outage_fail_closed(self):
        self.client.fail_submit = True
        with self.assertRaises(TimeoutError):
            self.coordinator.tick()
        self.client.jobs.clear()
        with self.assertRaisesRegex(RuntimeError, 'Unresolved submission'):
            self.coordinator.resume()
        self.assertEqual(len(self.client.submissions), 1)
        with patch.object(self.client, 'records', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.coordinator.tick()
        self.assertEqual(len(self.client.submissions), 1)

    def test_resume_retries_failed_checkpoint_and_retains_history(self):
        self.coordinator.tick()
        (self.root / 'case0').mkdir()
        self.store.mark_failed('case0', 'report', 'solver_done')
        self.client.finish('100', 'FAILED', '1:0')
        self.coordinator.reconcile()
        self.coordinator.resume()
        self.assertEqual(self.store.case_status('case0'), 'solver_done')
        self.coordinator.tick()
        case = self.store.get_case('case0')
        self.assertEqual(case['slurm_history'][0]['id'], '100')
        self.assertNotEqual(case['slurm_job']['id'], '100')

    def test_worker_identity_allocation_and_existing_case_executor(self):
        self.coordinator.tick()
        token = self.store.get_case('case0')['slurm_job']['token']
        with patch.dict(os.environ, SLURM_JOB_ID='100', SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='2'), \
                patch('tools.slurm_docker.SlurmDockerRuntime'), \
                patch('tools.openfoam.find_source_stls', return_value={}), \
                patch('tools.slurm.execute_simulation_case', return_value=True) as execute:
            self.assertTrue(run_worker(self.root, 'case0', token))
            self.assertEqual(execute.call_args.args[0]['allocated_cores'], 2)
            self.assertEqual(execute.call_args.args[0]['depends_on'], None)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                run_worker(self.root, 'case0', 'wrong')
            with patch.dict(os.environ, SLURM_CPUS_PER_TASK='1'):
                with self.assertRaisesRegex(RuntimeError, 'allocation'):
                    run_worker(self.root, 'case0', token)

    def test_worker_executes_all_stages_and_passes_previous_case_initialization(self):
        self.store.set_status('case0', 'postprocessing_done')
        self.store.update_case('case1', depends_on='case0')
        self.store.set_status('case2', 'failed')
        options = dict(field_init='on', turbulence='kOmegaSST', wall_functions='yes',
                       aerodynamics_only=True, acoustic_surface=None, acoustic_sphere_diameter=None,
                       end_on='time', end_on_value=1, mesh_only=True, allow_bad_mesh=False,
                       live_output=True)
        self.store.persist_batch_fields(slurm_worker_options=options)
        self.coordinator.tick()
        case = self.store.get_case('case1')
        def prepare(**kwargs):
            kwargs['TARGET_DIRECTORY'].mkdir()
        with patch.dict(os.environ, SLURM_JOB_ID='100', SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='2'), \
                patch('tools.slurm_docker.SlurmDockerRuntime'), \
                patch('tools.openfoam.find_source_stls', return_value={'prop': self.root / 'prop.stl'}), \
                patch('tools.preprocessing.run_preprocessing', side_effect=prepare) as preprocess, \
                patch('openfoamSimulation.openfoamSimulation', return_value=True) as solve, \
                patch('postprocessing.postprocessing') as postprocess:
            self.assertTrue(run_worker(self.root, 'case1', case['slurm_job']['token']))
        self.assertEqual(self.store.case_status('case1'), 'postprocessing_done')
        self.assertTrue(preprocess.call_args.kwargs['INIT_FROM_PREVIOUS'])
        self.assertEqual(preprocess.call_args.kwargs['PREVIOUS_SIMULATION_PATH'], self.root / 'case0')
        self.assertTrue(solve.call_args.kwargs['initialize_from_previous'])
        self.assertEqual(solve.call_args.kwargs['NUMBER_OF_CORES'], 2)
        postprocess.assert_called_once()

    def test_crash_before_release_recovers_saved_job_id(self):
        with patch.object(self.client, 'release', side_effect=RuntimeError('connection lost')):
            with self.assertRaises(RuntimeError):
                self.coordinator.tick()
        self.assertEqual(self.store.get_case('case0')['slurm_job']['id'], '100')
        self.coordinator.resume()
        self.assertEqual(self.client.releases, ['100'])
        self.assertEqual(len(self.client.submissions), 1)

    def test_preflight_failure_records_failure_without_running_stages(self):
        self.coordinator.tick()
        case = self.store.get_case('case0')
        with patch.dict(os.environ, SLURM_JOB_ID='100', SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='2'), \
                patch('tools.slurm_docker.SlurmDockerRuntime', side_effect=RuntimeError('Docker unavailable')), \
                patch('tools.slurm.execute_simulation_case') as execute:
            self.assertFalse(run_worker(self.root, 'case0', case['slurm_job']['token']))
        execute.assert_not_called()
        self.assertEqual(self.store.case_status('case0'), 'failed')
        self.assertIn('Docker unavailable', self.store.get_case('case0')['error'])

    def test_processes_do_not_overwrite_each_others_case_or_batch_updates(self):
        context = multiprocessing.get_context('spawn')
        workers = [context.Process(target=update_in_process, args=(str(self.root), f'case{i}')) for i in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
            self.assertEqual(worker.exitcode, 0)
        order = self.store.snapshot()
        self.assertEqual([c['counter'] for c in order['cases']], [14, 14, 14])
        self.assertEqual([order[f'field_case{i}'] for i in range(3)], [14, 14, 14])

    def test_local_scheduler_still_uses_local_workers_and_dashboard(self):
        args = Namespace(live_output=False)
        def execute(case, *args):
            self.store.set_status(case['folder'], 'postprocessing_done')
        with patch('tools.scheduler.execute_simulation_case', side_effect=execute) as local, \
                patch('tools.scheduler.LiveBatchDashboard') as dashboard, \
                patch('tools.slurm.SlurmClient.submit', side_effect=AssertionError('must stay local')):
            run_parallel_scheduler(ROOT, self.root, self.root / 'STL', {}, self.store,
                                   self.registry, args, scheduler_poll_interval=0.001)
        self.assertEqual(local.call_count, 3)
        dashboard.return_value.start.assert_called_once()
        dashboard.return_value.stop.assert_called_once_with(final_render=True)
        self.assertNotIn('slurm_job', self.store.get_case('case0'))

    def test_sbatch_command_fixed_resources_and_clean_environment(self):
        intent = dict(name='acoustic-token')
        with patch('tools.slurm.subprocess.run', return_value=Namespace(stdout='123\n')) as run, \
                patch.dict(os.environ, SBATCH_MEM_PER_CPU='99G', SBATCH_NTASKS='1',
                           SLURM_MEM_PER_CPU='99G', SLURM_NTASKS='1', SLURM_CONF='/site/slurm.conf'):
            self.assertEqual(SlurmClient().submit(self.root / 'worker.sh', CONFIG, 8, intent, self.root / 'log'), '123')
        command = run.call_args.args[0]
        for flag in ('--hold', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=8G', '--no-requeue'):
            self.assertIn(flag, command)
        self.assertNotIn('SBATCH_MEM_PER_CPU', run.call_args.kwargs['env'])
        self.assertNotIn('SBATCH_NTASKS', run.call_args.kwargs['env'])
        self.assertNotIn('SLURM_MEM_PER_CPU', run.call_args.kwargs['env'])
        self.assertNotIn('SLURM_NTASKS', run.call_args.kwargs['env'])
        self.assertEqual(run.call_args.kwargs['env']['SLURM_CONF'], '/site/slurm.conf')

    def test_queue_and_accounting_parsing_ignores_steps_and_queue_wins(self):
        client = SlurmClient()
        with patch.object(client, 'run', side_effect=[
                '100|acoustic-token|RUNNING|None',
                '100|acoustic-token|PENDING|0:0\n100.batch|acoustic-token|FAILED|1:0']):
            records = client.records([dict(name='acoustic-token', created='2026-10-18T00:00:00')])
        self.assertEqual(records['100']['state'], 'RUNNING')
        self.assertEqual(len(records), 1)

    def test_config_saved_and_immutable(self):
        self.store.persist_batch_fields(slurm=None)
        config_file = self.root / 'cluster.json'
        config_file.write_text(json.dumps(CONFIG))
        args = create_parser().parse_args(['--sim-dir', str(self.root), '--slurm', '--slurm-config', str(config_file)])
        configure_order(self.store, args)
        self.assertEqual(self.store.snapshot()['slurm'], CONFIG)
        args.slurm_config = None
        configure_order(self.store, args)
        config_file.write_text(json.dumps(dict(CONFIG, mem='16G')))
        args.slurm_config = config_file
        with self.assertRaisesRegex(ValueError, 'immutable'):
            configure_order(self.store, args)

    def test_failed_native_order_migrates_without_editing_json_or_losing_history(self):
        self.coordinator.tick()
        old_config = {k: v for k, v in CONFIG.items() if k not in {'runtime', 'docker_image'}}
        old_config.update(openfoam_bashrc='/opt/missing/etc/bashrc', mpi_launcher=['mpirun', '-np', '{cores}'])
        self.store.persist_batch_fields(slurm=old_config)
        for index in range(2):
            folder = f'case{index}'
            self.store.mark_failed(folder, 'missing host bashrc', 'pending')
            self.client.finish(str(100 + index), 'FAILED', '1:0')
        args = create_parser().parse_args(['--sim-dir', str(self.root), '--slurm', '--resume'])
        with patch('tools.slurm.SlurmClient', return_value=self.client):
            configure_order(self.store, args)
        order = self.store.snapshot()
        self.assertEqual(order['slurm'], CONFIG)
        self.assertEqual(order['slurm_config_history'], [old_config])
        self.assertEqual(order['cases'][0]['slurm_job']['id'], '100')
        self.assertTrue(order['cases'][0]['slurm_job']['terminal'])
        self.assertEqual(order['cases'][0]['resume_status'], 'pending')
        restarted = Coordinator(self.store, self.registry, ROOT, self.client)
        restarted.resume()
        restarted.tick()
        retried = self.store.get_case('case0')
        self.assertEqual(retried['slurm_history'][0]['id'], '100')
        self.assertNotEqual(retried['slurm_job']['id'], '100')

    def test_native_migration_refuses_active_or_unknown_jobs(self):
        self.coordinator.tick()
        old_config = {k: v for k, v in CONFIG.items() if k not in {'runtime', 'docker_image'}}
        old_config['openfoam_bashrc'] = '/opt/missing/etc/bashrc'
        self.store.persist_batch_fields(slurm=old_config)
        args = create_parser().parse_args(['--sim-dir', str(self.root), '--slurm', '--resume'])
        with patch('tools.slurm.SlurmClient', return_value=self.client):
            with self.assertRaisesRegex(RuntimeError, 'Cannot migrate'):
                configure_order(self.store, args)
        self.assertEqual(self.store.snapshot()['slurm'], old_config)
        self.assertEqual(len(self.client.submissions), 2)

    def test_legacy_config_file_is_accepted_but_host_mpi_and_bashrc_are_unused(self):
        self.store.persist_batch_fields(slurm=None)
        config_file = self.root / 'cluster.json'
        config_file.write_text(json.dumps(dict(CONFIG, openfoam_bashrc='/does/not/exist',
                                              mpi_launcher=['srun', '--mpi=pmix', '{cores}'])))
        args = create_parser().parse_args(['--sim-dir', str(self.root), '--slurm', '--slurm-config', str(config_file)])
        configure_order(self.store, args)
        self.assertEqual(self.store.snapshot()['slurm'], CONFIG)

    def test_docker_cleanup_failure_preserves_solver_checkpoint(self):
        self.coordinator.tick()
        case = self.store.get_case('case0')
        with patch.dict(os.environ, SLURM_JOB_ID='100', SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='2'), \
                patch('tools.slurm_docker.SlurmDockerRuntime') as runtime, \
                patch('tools.openfoam.find_source_stls', return_value={}), \
                patch('tools.slurm.execute_simulation_case') as execute:
            runtime.return_value.__exit__.side_effect = RuntimeError('cleanup failed')
            execute.side_effect = lambda *args, **kwargs: self.store.set_status('case0', 'solver_done')
            self.assertFalse(run_worker(self.root, 'case0', case['slurm_job']['token']))
        self.assertEqual(self.store.get_case('case0')['resume_status'], 'solver_done')



if __name__ == '__main__':
    unittest.main()

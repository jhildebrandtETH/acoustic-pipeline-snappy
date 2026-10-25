"""Bound a case's existing Docker execution to its single-node Slurm allocation.

Docker is managed by a daemon, so killing the Slurm worker alone does not kill
its containers. Their PID 1 watches a read-only bind of the worker's proc entry;
when the worker disappears (including SIGKILL), PID 1 exits and Docker tears down
the container's remaining processes. Normal and SIGTERM paths also remove it.
"""
from contextvars import ContextVar
import os
from pathlib import Path
import re
import shlex
import signal

from tools.runtime import IMAGE
_RUNTIME = ContextVar('slurm_docker_runtime', default=None)


def memory_bytes(value):
    """Slurm --mem units are binary K/M/G/T; a bare integer means MiB."""
    match = re.fullmatch(r'([0-9]+)([KMGT]?)', str(value), re.IGNORECASE)
    if not match or int(match[1]) == 0:
        raise ValueError('Docker Slurm cases require a positive finite mem value, e.g. 30G')
    return int(match[1]) * 1024 ** {'K': 1, '': 2, 'M': 2, 'G': 3, 'T': 4}[match[2].upper()]


def watchdog_command(proc_path='/pipeline-worker'):
    # Reading the mounted proc inode also prevents a reused host PID from keeping
    # a stale container alive. Zombies count as dead even before their reaping.
    script = f'''while IFS= read -r worker_stat < {shlex.quote(proc_path + '/stat')} 2>/dev/null; do
    fields=${{worker_stat##*) }}
    case "$fields" in Z\\ *|X\\ *) break;; esac
    sleep 1
done
'''
    return ['bash', '-c', script]


def slurm_cgroup_parent(job_id, driver, memberships):
    """Docker's cgroupfs driver can put the container under the Slurm job cap.

    The systemd driver requires a .slice parent, whereas Slurm usually creates
    job directories under a .scope. Do not invent a different systemd slice or
    pass a path that Docker interprets differently on that driver.
    """
    if driver != 'cgroupfs' or not str(job_id).isdigit():
        return None
    parents = set()
    for line in memberships.splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        controllers, path = parts[1:]
        if controllers and not {'cpu', 'cpuset', 'memory'} & set(controllers.split(',')):
            continue
        segments = path.split('/')
        for index, segment in enumerate(segments):
            if segment == f'job_{job_id}':
                parents.add('/'.join(segments[:index + 1]))
                break
    return parents.pop() if len(parents) == 1 else None


def container_options(cores):
    """Only Slurm workers supply options; local Docker calls stay unchanged."""
    runtime = _RUNTIME.get()
    if runtime is None:
        return {}
    if cores != runtime.cores:
        raise RuntimeError('Docker rank count differs from the Slurm case allocation')
    options = dict(image=runtime.config['docker_image'], cpuset_cpus=runtime.cpuset,
                   nano_cpus=cores * 1_000_000_000,
                   mem_limit=runtime.memory, memswap_limit=runtime.memory,
                   command=watchdog_command(),
                   environment={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                'MKL_NUM_THREADS': '1'},
                   volumes={str(runtime.proc): {'bind': '/pipeline-worker', 'mode': 'ro'}},
                   labels={'acoustic-slurm-attempt': runtime.attempt,
                           'acoustic-slurm-job': os.environ['SLURM_JOB_ID']})
    if runtime.cgroup_parent:
        options['cgroup_parent'] = runtime.cgroup_parent
    return options


class SlurmDockerRuntime:
    def __init__(self, config, cores, attempt):
        self.config = config
        self.cores = int(cores)
        self.attempt = attempt
        self.proc = Path('/proc') / str(os.getpid())
        self.client = None
        self._handlers = {}

    def __enter__(self):
        import docker
        if os.environ.get('DOCKER_HOST', '').strip() and not os.environ['DOCKER_HOST'].startswith('unix://'):
            raise RuntimeError('Slurm Docker requires the compute node local Unix socket; remote DOCKER_HOST is unsupported')
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) != self.cores:
            raise RuntimeError(f'Slurm worker affinity has {len(cpus)} CPUs, expected {self.cores}. '
                               'Check srun CPU binding and the cluster task/affinity plugin')
        if not (self.proc / 'stat').is_file():
            raise RuntimeError('Linux /proc worker entry is required for Docker cancellation cleanup')
        self.cpuset = ','.join(map(str, cpus))
        self.memory = memory_bytes(self.config['mem'])
        self.client = docker.from_env(timeout=30)
        try:
            self.client.ping()
            self.client.images.get(self.config['docker_image'])
            info = self.client.info()
            self.cgroup_parent = slurm_cgroup_parent(os.environ['SLURM_JOB_ID'], info.get('CgroupDriver'),
                                                     Path('/proc/self/cgroup').read_text())
            # The image is intentionally not pulled implicitly on a compute node.
            self._token = _RUNTIME.set(self)
            for name in (signal.SIGTERM, signal.SIGINT):
                self._handlers[name] = signal.getsignal(name)
                signal.signal(name, self._terminated)
        except BaseException:
            for name, handler in self._handlers.items():
                signal.signal(name, handler)
            if hasattr(self, '_token'):
                _RUNTIME.reset(self._token)
            self.client.close()
            raise
        return self

    @staticmethod
    def _terminated(signum, frame):
        raise SystemExit(128 + signum)

    def __exit__(self, exc_type, exc, tb):
        cleanup_error = None
        try:
            # Labels scope removal to this attempt, never another case/order.
            for container in self.client.containers.list(
                    all=True, filters={'label': f'acoustic-slurm-attempt={self.attempt}'}):
                try:
                    container.remove(force=True)
                except Exception as error:
                    cleanup_error = error
        except Exception as error:
            cleanup_error = error
        finally:
            for name, handler in self._handlers.items():
                signal.signal(name, handler)
            _RUNTIME.reset(self._token)
            self.client.close()
        if cleanup_error is not None and exc is None:
            raise RuntimeError(f'Docker cleanup failed for Slurm attempt {self.attempt}: {cleanup_error}')

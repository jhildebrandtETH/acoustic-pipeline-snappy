"""Docker allocation/cleanup tests, plus opt-in real-image Linux smoke tests."""
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from tools.slurm_docker import (IMAGE, SlurmDockerRuntime, container_options, memory_bytes,
                                slurm_cgroup_parent, watchdog_command)

CONFIG = dict(docker_image=IMAGE, mem='2G')


class DockerAllocationTests(unittest.TestCase):
    def runtime(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, SLURM_JOB_ID='123', DOCKER_HOST=''))
        stack.enter_context(patch('os.sched_getaffinity', return_value={4, 7}, create=True))
        stack.enter_context(patch.object(Path, 'is_file', return_value=True))
        stack.enter_context(patch.object(Path, 'read_text', return_value='0::/init.scope\n'))
        client = stack.enter_context(patch('docker.from_env')).return_value
        client.containers.list.return_value = []
        return client

    def test_limits_image_and_worker_monitor_apply_only_in_slurm_context(self):
        client = self.runtime()
        self.assertEqual(container_options(2), {})
        with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
            options = container_options(2)
            self.assertEqual(options['image'], IMAGE)
            self.assertEqual(options['cpuset_cpus'], '4,7')
            self.assertEqual(options['nano_cpus'], 2_000_000_000)
            self.assertEqual(options['mem_limit'], 2 * 1024**3)
            self.assertEqual(options['memswap_limit'], options['mem_limit'])
            self.assertEqual(options['labels']['acoustic-slurm-attempt'], 'attempt')
            self.assertEqual(options['labels']['acoustic-slurm-job'], '123')
            self.assertEqual(options['volumes'][f'/proc/{os.getpid()}']['mode'], 'ro')
            self.assertIn('/pipeline-worker/stat', options['command'][2])
            with self.assertRaisesRegex(RuntimeError, 'rank count'):
                container_options(3)
        self.assertEqual(container_options(2), {})
        client.images.get.assert_called_once_with(IMAGE)
        client.containers.list.assert_called_once_with(all=True, filters={'label': 'acoustic-slurm-attempt=attempt'})
        client.close.assert_called_once()

    def test_cleanup_removes_only_attempt_containers_and_propagates_failure(self):
        client = self.runtime()
        containers = [Mock(), Mock()]
        client.containers.list.return_value = containers
        with self.assertRaisesRegex(ValueError, 'worker failure'):
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                raise ValueError('worker failure')
        for container in containers:
            container.remove.assert_called_once_with(force=True)

    def test_sigterm_restores_handlers_and_removes_container(self):
        client = self.runtime()
        container = Mock()
        client.containers.list.return_value = [container]
        handler = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(SystemExit) as terminated:
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertEqual(terminated.exception.code, 128 + signal.SIGTERM)
        self.assertIs(signal.getsignal(signal.SIGTERM), handler)
        container.remove.assert_called_once_with(force=True)
        self.assertEqual(container_options(2), {})

    def test_cleanup_failure_cannot_report_success(self):
        client = self.runtime()
        container = Mock()
        container.remove.side_effect = RuntimeError('daemon down')
        client.containers.list.return_value = [container]
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                pass
        self.assertEqual(container_options(2), {})

    def test_affinity_mismatch_fails_before_container_creation(self):
        client = self.runtime()
        with self.assertRaisesRegex(RuntimeError, 'affinity'):
            with SlurmDockerRuntime(CONFIG, 3, 'attempt'):
                pass
        client.containers.run.assert_not_called()

    def test_remote_docker_cannot_monitor_local_worker(self):
        client = self.runtime()
        with patch.dict(os.environ, DOCKER_HOST='tcp://another-node:2375'):
            with self.assertRaisesRegex(RuntimeError, 'local Unix socket'):
                with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                    pass
        client.containers.run.assert_not_called()

    def test_missing_image_or_daemon_fails_before_case_work(self):
        client = self.runtime()
        client.images.get.side_effect = RuntimeError('image missing')
        with self.assertRaisesRegex(RuntimeError, 'image missing'):
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                pass
        client.close.assert_called_once()
        self.assertEqual(container_options(2), {})

    def test_existing_openfoam_stage_creates_limited_docker_container(self):
        import openfoamSimulation as stage
        client = self.runtime()
        container = client.containers.run.return_value
        container.status = 'running'
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage, 'ensure_case_core_configuration'), \
                patch.object(stage, 'remove_stale_stopped_container'), \
                patch('tools.wall_omega.prepare_wall_omega'), \
                patch.object(stage, 'run_mesh', side_effect=lambda start, *args: bool(start())):
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                self.assertTrue(stage.openfoamSimulation('case', Path(tmp), 1e-2, 4000, 1000,
                                'AMI', 'kOmegaSST', 2, False, True, 'rev', False))
        options = client.containers.run.call_args.kwargs
        self.assertEqual(options['image'], IMAGE)
        self.assertEqual(options['cpuset_cpus'], '4,7')
        self.assertEqual(options['mem_limit'], 2 * 1024**3)
        self.assertEqual(options['volumes'][tmp]['bind'], '/simulation')
        self.assertIn('/pipeline-worker/stat', options['command'][2])
        self.assertEqual(options['labels']['acoustic-pipeline-case'], 'case')
        self.assertEqual(options['labels']['acoustic-slurm-attempt'], 'attempt')
        container.stop.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_local_openfoam_container_arguments_remain_unchanged(self):
        import openfoamSimulation as stage
        with tempfile.TemporaryDirectory() as tmp, \
                patch('docker.from_env') as docker, \
                patch.object(stage, 'ensure_case_core_configuration'), \
                patch.object(stage, 'remove_stale_stopped_container'), \
                patch('tools.wall_omega.prepare_wall_omega'), \
                patch.object(stage, 'run_mesh', side_effect=lambda start, *args: bool(start())):
            docker.return_value.containers.run.return_value.status = 'running'
            self.assertTrue(stage.openfoamSimulation('case', Path(tmp), 1e-2, 4000, 1000,
                            'AMI', 'kOmegaSST', 2, False, True, 'rev', False))
        options = docker.return_value.containers.run.call_args.kwargs
        self.assertEqual(options['image'], IMAGE)
        self.assertEqual(options['command'], 'bash')
        self.assertNotIn('cpuset_cpus', options)
        self.assertNotIn('mem_limit', options)
        self.assertEqual(options['labels'], {'acoustic-pipeline-case': 'case'})
        self.assertEqual(len(options['volumes']), 1)

    def test_slurm_memory_units_and_invalid_values(self):
        for text, expected in [('30G', 30 * 1024**3), ('1024', 1024**3), ('1024M', 1024**3),
                               ('512K', 512 * 1024), ('1t', 1024**4)]:
            self.assertEqual(memory_bytes(text), expected)
        for text in ['0', '0G', '-1G', 'unlimited', '1.5G', '30GB']:
            with self.assertRaises(ValueError):
                memory_bytes(text)

    def test_cgroupfs_container_joins_job_cap_when_slurm_parent_is_identifiable(self):
        membership = '0::/system.slice/slurmstepd.scope/job_123/step_0/user/task_0\n'
        self.assertEqual(slurm_cgroup_parent('123', 'cgroupfs', membership),
                         '/system.slice/slurmstepd.scope/job_123')
        self.assertIsNone(slurm_cgroup_parent('123', 'systemd', membership))
        self.assertIsNone(slurm_cgroup_parent('456', 'cgroupfs', membership))
        self.assertIsNone(slurm_cgroup_parent('123', 'cgroupfs', '0::/init.scope\n'))
        client = self.runtime()
        client.info.return_value = {'CgroupDriver': 'cgroupfs'}
        with patch.object(Path, 'read_text', return_value=membership):
            with SlurmDockerRuntime(CONFIG, 2, 'attempt'):
                self.assertEqual(container_options(2)['cgroup_parent'],
                                 '/system.slice/slurmstepd.scope/job_123')

    @unittest.skipIf(os.name == 'nt', 'watchdog is a Linux container command')
    def test_watchdog_exits_after_worker_disappears_or_is_zombie(self):
        with tempfile.TemporaryDirectory() as tmp:
            stat = Path(tmp) / 'stat'
            stat.write_text('123 (worker with spaces) S 0 0 0\n')
            process = subprocess.Popen(watchdog_command(tmp), stderr=subprocess.DEVNULL)
            try:
                self.assertIsNone(process.poll())
                stat.write_text('123 (worker with spaces) Z 0 0 0\n')
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            stat.unlink()
            self.assertEqual(subprocess.run(watchdog_command(tmp), stderr=subprocess.DEVNULL).returncode, 0)


@unittest.skipUnless(os.environ.get('SLURM_DOCKER_TESTS') == '1' and os.name != 'nt',
                     'opt in with SLURM_DOCKER_TESTS=1; requires local Linux Docker and OpenFOAM image')
class DockerImageSmokeTests(unittest.TestCase):
    def test_real_image_mpi_tools_and_limits_without_native_openfoam(self):
        import docker
        original = os.sched_getaffinity(0)
        cpus = sorted(original)[:2]
        self.assertEqual(len(cpus), 2, 'smoke test needs at least two available CPUs')
        try:
            os.sched_setaffinity(0, cpus)
            with patch.dict(os.environ, SLURM_JOB_ID='smoke'):
                with SlurmDockerRuntime(CONFIG, 2, uuid.uuid4().hex):
                    options = container_options(2)
                    client = docker.from_env()
                    try:
                        container = client.containers.run(detach=True, tty=True, stdin_open=True,
                            user=f'{os.getuid()}:{os.getgid()}', working_dir='/tmp', **options)
                        container.reload()
                        self.assertEqual(container.attrs['HostConfig']['CpusetCpus'], ','.join(map(str, cpus)))
                        self.assertEqual(container.attrs['HostConfig']['Memory'], 2 * 1024**3)
                        result = container.exec_run(['bash', '-c',
                            'source /opt/openfoam13/etc/bashrc && test "$WM_PROJECT_VERSION" = 13 && '
                            'command -v snappyHexMesh && command -v foamRun && '
                            'mpirun --allow-run-as-root --use-hwthread-cpus -np 2 bash -c '
                            '\'echo "rank=$OMPI_COMM_WORLD_RANK"\''])
                        self.assertEqual(result.exit_code, 0, result.output.decode())
                        self.assertIn(b'rank=0', result.output)
                        self.assertIn(b'rank=1', result.output)
                    finally:
                        client.close()
        finally:
            os.sched_setaffinity(0, original)

    def test_real_container_exits_after_worker_sigkill(self):
        import docker
        script = '''import os, signal, uuid
from tools.slurm_docker import SlurmDockerRuntime, container_options, IMAGE
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
os.environ['SLURM_JOB_ID'] = 'smoke-kill'
with SlurmDockerRuntime(dict(docker_image=IMAGE, mem='1G'), 1, uuid.uuid4().hex) as runtime:
    container = runtime.client.containers.run(detach=True, tty=True, stdin_open=True,
        user=f'{os.getuid()}:{os.getgid()}', working_dir='/tmp', **container_options(1))
    print(container.id, flush=True)
    signal.pause()
'''
        process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True)
        client = docker.from_env()
        container = None
        try:
            identity = process.stdout.readline().strip()
            self.assertEqual(len(identity), 64, 'worker did not create a container')
            container = client.containers.get(identity)
            self.assertEqual(container.status, 'running')
            process.kill()
            process.wait(timeout=10)
            self.assertEqual(container.wait(timeout=10)['StatusCode'], 0)
            container.reload()
            self.assertEqual(container.status, 'exited')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            if container is not None:
                container.remove(force=True)
            client.close()


if __name__ == '__main__':
    unittest.main()

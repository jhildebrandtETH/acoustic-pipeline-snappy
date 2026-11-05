"""Local and Slurm status viewers must reflect the worker, without launching it."""
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from argparse import Namespace

from tools.common import _atomic_write_json
from tools.scheduler import (
    SimulationOrderStore, RuntimeStatusRegistry, LiveBatchDashboard,
    initialize_runtime_queue_states, run_parallel_scheduler,
)
from tools.case_monitor import refresh_saved_table, show_status


class CaseMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = dict(folder='10x7E_4000RPM_AMI', mesh='10x7E', rpm=4000,
                         allocated_cores=72, depends_on=None, status='solver_running')
        self.order = dict(cases=[self.case], total_cores=72, cores_per_case=72,
                          max_parallel_cases=1, end_on='rev', end_on_value=30)
        _atomic_write_json(self.root/'simulation_order.json', self.order)
        self.store = SimulationOrderStore(self.root)

    def watcher(self):
        order = self.store.snapshot()
        registry = RuntimeStatusRegistry(order['cases'])
        initialize_runtime_queue_states(order, registry)
        return registry

    def test_single_local_case_uses_72_cores_and_updates_separate_watcher(self):
        writer = RuntimeStatusRegistry([self.case])
        writer.persist_runtime(self.root)
        writer.update(self.case['folder'], state='RUNNING', stage='solving',
                      progress=42.5, detail='t=0.191250/0.450000 s')
        before = (self.root/'simulation_order.json').read_bytes()
        with patch('tools.slurm.SlurmClient.run', side_effect=AssertionError('local viewer must not call Slurm')):
            reader = self.watcher()
            refresh_saved_table(self.store, reader)
            text = LiveBatchDashboard(reader, 72, 72).render_text()
            self.assertIn('Used: 72', text)
            self.assertIn('Running: 1', text)
            self.assertIn('Queued/waiting: 0', text)
            self.assertIn('42.50%', text)
            self.assertIn('t=0.191250/0.450000 s', text)
            # Follow subsequent progress with the same independent registry.
            writer.callback_for(self.case['folder'])(stage='acoustics', progress=55., detail='predicting pressure')
            refresh_saved_table(self.store, reader)
            self.assertEqual(reader.get(self.case['folder'])['stage'], 'acoustics')
            self.assertEqual(reader.get(self.case['folder'])['progress'], 55.)
        self.assertEqual((self.root/'simulation_order.json').read_bytes(), before)

    def test_status_cli_reads_local_progress_in_another_process(self):
        writer = RuntimeStatusRegistry([self.case])
        writer.persist_runtime(self.root)
        writer.update(self.case['folder'], state='RUNNING', stage='snappyHexMesh',
                      progress=60., detail='snapping surface')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'main.py'),
                                 '--sim-dir', str(self.root), '--status'],
                                capture_output=True, text=True, check=True)
        self.assertIn('Used: 72', result.stdout)
        self.assertIn('Running: 1', result.stdout)
        self.assertIn('snappyHexMesh', result.stdout)
        self.assertIn('60.00%', result.stdout)

    def test_progress_writes_are_throttled_but_transitions_are_immediate(self):
        registry = RuntimeStatusRegistry([self.case])
        with patch('tools.scheduler.time.monotonic', return_value=100.) as clock:
            registry.persist_runtime(self.root)
            registry.update(self.case['folder'], state='RUNNING', stage='solving', progress=1.)
            path = self.root/'.runtime/local-status.json'
            for progress in range(2, 50):
                registry.callback_for(self.case['folder'])(stage='solving', progress=float(progress))
            self.assertEqual(json.loads(path.read_text())['cases'][0]['progress'], 1.)
            clock.return_value = 101.
            registry.publish_runtime()
            self.assertEqual(json.loads(path.read_text())['cases'][0]['progress'], 49.)
            registry.update(self.case['folder'], stage='report', progress=75.)
            self.assertEqual(json.loads(path.read_text())['cases'][0]['stage'], 'report')

    def test_existing_launcher_without_snapshot_shows_saved_running_stage(self):
        reader = self.watcher()
        refresh_saved_table(self.store, reader)
        state = reader.get(self.case['folder'])
        self.assertEqual(state['state'], 'RUNNING')
        self.assertEqual(state['stage'], 'solver_running')
        self.assertIn('unavailable', state['detail'])
        self.assertFalse((self.root/'.runtime').exists())

    def test_legacy_solver_log_gives_time_progress_without_slurm_config(self):
        case_path = self.root/self.case['folder']
        case_path.mkdir()
        (case_path/'log.pimpleFoam').write_text('Time = 0.1\nExecutionTime = 42 s\nTime = 2.25e-1\n')
        for slurm in (None, {}):
            self.store.persist_batch_fields(slurm=slurm)
            with self.subTest(slurm=slurm), patch('sys.stdout', new_callable=io.StringIO) as output:
                show_status(self.root)
                text = output.getvalue()
                self.assertIn('Running: 1', text)
                self.assertIn('Used: 72', text)
                self.assertIn('50.00%', text)
                self.assertIn('t=0.225000/0.450000 s', text)

    def test_terminal_order_status_overrides_stale_live_update(self):
        writer = RuntimeStatusRegistry([self.case])
        writer.persist_runtime(self.root)
        writer.update(self.case['folder'], state='RUNNING', stage='solving', progress=70.)
        reader = self.watcher()
        for status, expected in (('postprocessing_done', 'DONE'), ('failed', 'FAILED'), ('blocked', 'BLOCKED')):
            self.store.set_status(self.case['folder'], status, error='test failure' if status != 'postprocessing_done' else None)
            refresh_saved_table(self.store, reader)
            state = reader.get(self.case['folder'])
            self.assertEqual(state['state'], expected)
            self.assertEqual(state['progress'], 100. if expected == 'DONE' else None)

    def test_pending_dependencies_and_failures_are_not_treated_as_running(self):
        source = dict(self.case, status='pending')
        child = dict(source, folder='child', depends_on=source['folder'])
        self.store.persist_batch_fields(cases=[source, child])
        reader = self.watcher()
        refresh_saved_table(self.store, reader)
        self.assertEqual(reader.get(source['folder'])['state'], 'QUEUED')
        self.assertEqual(reader.get('child')['state'], 'WAITING_INIT')
        self.store.set_status(source['folder'], 'failed', error='solver failed')
        refresh_saved_table(self.store, reader)
        self.assertEqual(reader.get('child')['state'], 'BLOCKED')

    def test_corrupt_snapshot_falls_back_and_watcher_does_not_rewrite_it(self):
        path = self.root/'.runtime/local-status.json'
        path.parent.mkdir()
        for content in ('{invalid', '{"cases": null}', '[]'):
            path.write_text(content)
            reader = self.watcher()
            refresh_saved_table(self.store, reader)
            self.assertEqual(reader.get(self.case['folder'])['state'], 'RUNNING')
            self.assertEqual(path.read_text(), content)

    def test_cleanly_stopped_launcher_does_not_remain_running(self):
        writer = RuntimeStatusRegistry([self.case])
        writer.persist_runtime(self.root)
        writer.update(self.case['folder'], state='RUNNING', stage='solving', progress=50.)
        writer.close_runtime()
        reader = self.watcher()
        refresh_saved_table(self.store, reader)
        self.assertEqual(reader.get(self.case['folder'])['state'], 'BLOCKED')
        self.assertIn('resume', reader.get(self.case['folder'])['detail'])

    def test_watch_refreshes_local_progress_and_stops_on_ctrl_c(self):
        writer = RuntimeStatusRegistry([self.case])
        writer.persist_runtime(self.root)
        writer.update(self.case['folder'], state='RUNNING', stage='solving', progress=25.)
        with patch('tools.case_monitor.LiveBatchDashboard') as dashboard, \
                patch('tools.case_monitor.time.sleep', side_effect=KeyboardInterrupt):
            show_status(self.root, watch=True)
        dashboard.return_value.start.assert_called_once()
        dashboard.return_value.stop.assert_called_once()
        reader = dashboard.call_args.args[0]
        self.assertEqual(reader.get(self.case['folder'])['progress'], 25.)

    def test_local_scheduler_publishes_even_with_live_output_enabled(self):
        self.store.set_status(self.case['folder'], 'pending')
        registry = self.watcher()
        def execute(case, *args):
            registry.callback_for(case['folder'])(stage='solving', progress=50., detail='halfway')
            observed = self.watcher()
            refresh_saved_table(self.store, observed)
            self.assertEqual(observed.get(case['folder'])['state'], 'RUNNING')
            self.assertEqual(observed.get(case['folder'])['progress'], 50.)
            self.store.set_status(case['folder'], 'postprocessing_done')
        with patch('tools.scheduler.execute_simulation_case', side_effect=execute) as worker, \
                patch('tools.scheduler.LiveBatchDashboard') as dashboard:
            run_parallel_scheduler(self.root, self.root, self.root/'STL', {}, self.store,
                                   registry, Namespace(live_output=True), scheduler_poll_interval=.001)
        worker.assert_called_once()
        dashboard.return_value.start.assert_not_called()
        snapshot = json.loads((self.root/'.runtime/local-status.json').read_text())
        self.assertFalse(snapshot['active'])
        self.assertEqual(snapshot['cases'][0]['state'], 'DONE')


if __name__ == '__main__':
    unittest.main()

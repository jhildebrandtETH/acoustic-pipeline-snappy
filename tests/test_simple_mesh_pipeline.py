"""Supplied snappy cases preserve inputs and publish reports only after checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.cli import create_parser
from tools.simple_mesh_pipeline import run_simple_mesh_pipeline


class SimpleMeshPipelineTests(unittest.TestCase):
    def args(self, root, *extra):
        source = root/'input'
        (source/'system').mkdir(parents=True)
        for name in ('blockMeshDict', 'snappyHexMeshDict', 'controlDict'):
            (source/'system'/name).write_text('original input')
        return create_parser().parse_args([
            '--mesh-case', str(source), '--mesh-only', '--sim-dir', str(root/'result'),
            *extra])

    def test_failed_mesh_never_publishes_report_or_accepts_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            with patch('tools.simple_mesh_pipeline.Runner') as runner, \
                 patch('tools.report_worker.create_simulation_report') as report:
                runner.return_value.run.side_effect = RuntimeError('mesh failed')
                with self.assertRaisesRegex(RuntimeError, 'mesh failed'):
                    run_simple_mesh_pipeline(args)
            report.assert_not_called()
            status = json.loads((root/'result/mesh-pipeline-status.json').read_text())
            self.assertFalse(status['accepted'])
            self.assertFalse(status['completed'])
            self.assertEqual((root/'input/system/snappyHexMeshDict').read_text(), 'original input')

    def test_report_failure_leaves_mesh_accepted_but_pipeline_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            with patch('tools.simple_mesh_pipeline.Runner'), \
                 patch('tools.report_worker.create_simulation_report', side_effect=RuntimeError('report failed')):
                with self.assertRaisesRegex(RuntimeError, 'report failed'):
                    run_simple_mesh_pipeline(args)
            status = json.loads((root/'result/mesh-pipeline-status.json').read_text())
            self.assertTrue(status['mesh_accepted'])
            self.assertFalse(status['accepted'])
            self.assertFalse(status['completed'])

    def test_solver_modes_do_not_create_output(self):
        for flags in [('--mode', 'AMI'), ('--allow-bad-mesh',), ('--resume',)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.args(root, *flags)
                with self.assertRaises(ValueError):
                    run_simple_mesh_pipeline(args)
                self.assertFalse((root/'result').exists())

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            (root/'result').mkdir()
            marker = root/'result/keep.txt'
            marker.write_text('keep')
            with patch('tools.simple_mesh_pipeline.Runner'):
                with self.assertRaises(FileExistsError):
                    run_simple_mesh_pipeline(args)
            self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

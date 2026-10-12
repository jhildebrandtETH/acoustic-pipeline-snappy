"""Snappy-only CLI, supplied-case orchestration and native utility gates."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.cli import create_parser
from tools.mesh_runner import Runner
from tools.simple_mesh_pipeline import run_simple_mesh_pipeline


class SnappyOnlyTests(unittest.TestCase):
    def test_removed_options_are_rejected(self):
        for flag in ('--strict-layers', '--strict-layers-config',
                     '--strict-layers-executable', '--strict-layers-prepare'):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    create_parser().parse_args(['--sim-dir', 'unused', flag])
        self.assertNotIn('strict', create_parser().format_help())

    def test_supplied_case_runs_snappy_checks_then_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input'
            (source/'system').mkdir(parents=True)
            for name in ('blockMeshDict', 'snappyHexMeshDict', 'controlDict', 'surfaceFeaturesDict'):
                (source/'system'/name).write_text('addLayers true;')
            args = create_parser().parse_args(['--mesh-case', str(source), '--mesh-only',
                                               '--sim-dir', str(root/'result')])
            events = []
            with patch('tools.simple_mesh_pipeline.Runner') as runner, \
                 patch('tools.report_worker.create_simulation_report') as report:
                runner.return_value.run.side_effect = lambda case, program, flags, log: events.append(program)
                report.side_effect = lambda **kwargs: events.append('report')
                result = run_simple_mesh_pipeline(args)
            self.assertEqual(events, ['blockMesh', 'surfaceFeatures',
                                      'snappyHexMesh', 'checkMesh', 'report'])
            self.assertTrue(result['accepted'])
            self.assertTrue(result['layers'])
            self.assertEqual((source/'system/snappyHexMeshDict').read_text(), 'addLayers true;')

    def test_checkmesh_rejects_failure_and_disconnected_regions(self):
        for text, code, passes in [('Mesh OK.\nNumber of regions: 1', 0, True),
                                   ('Failed 1 mesh checks.\nNumber of regions: 1', 0, False),
                                   ('Mesh OK.\nNumber of regions: 2', 0, False),
                                   ('Mesh OK.\nNumber of regions: 1', 1, False)]:
            with self.subTest(text=text, code=code), tempfile.TemporaryDirectory() as tmp:
                def run(command, stdout, **kwargs):
                    stdout.write(text)
                    return SimpleNamespace(returncode=code)
                with patch('tools.mesh_runner.subprocess.run', side_effect=run):
                    if passes:
                        Runner().run(Path(tmp), 'checkMesh', [], 'log.checkMesh')
                    else:
                        with self.assertRaises(RuntimeError):
                            Runner().run(Path(tmp), 'checkMesh', [], 'log.checkMesh')

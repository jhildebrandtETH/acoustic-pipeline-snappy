"""Automatic omega selection and OpenFOAM preparation regressions."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.templates import template_directory
from tools.wall_omega import prepare_wall_omega, wall_omega_command


class WallOmegaTests(unittest.TestCase):
    def fixture(self, path, model='kOmegaSST', walls='no'):
        shutil.copytree(template_directory('AMI', model, walls), path, dirs_exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        shutil.copytree(root/'Parameters', path/'Parameters')

    def test_only_resolved_omega_models_are_prepared(self):
        for model, walls, expected in [('kOmegaSST', 'no', True), ('DES', 'no', True),
                                       ('kOmegaSST', 'yes', False), ('kEpsilon', 'no', False)]:
            with self.subTest(model=model, walls=walls), tempfile.TemporaryDirectory() as d:
                case = Path(d)
                self.fixture(case, model, walls)
                with patch('tools.wall_omega.docker_run') as run:
                    self.assertEqual(prepare_wall_omega(object(), case), expected)
                    self.assertEqual(run.call_count, int(expected))

    def test_manual_mode_leaves_field_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            case = Path(d)
            self.fixture(case)
            before = (case/'0/omega').read_bytes()
            settings = case/'Parameters/wallResolvedDict'
            settings.write_text(settings.read_text().replace('automaticOmega true', 'automaticOmega false'))
            self.assertIsNone(wall_omega_command(case))
            self.assertEqual((case/'0/omega').read_bytes(), before)
            self.assertFalse((case/'mesh/wallOmega').exists())

    def test_older_snapshot_keeps_manual_value(self):
        with tempfile.TemporaryDirectory() as d:
            case = Path(d)
            self.fixture(case)
            (case/'Parameters/wallResolvedDict').write_text('omegaWallValue uniform 19200000;\n')
            self.assertIsNone(wall_omega_command(case))

    def test_preparation_failure_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            case = Path(d)
            self.fixture(case)
            with patch('tools.wall_omega.docker_run', side_effect=RuntimeError('invalid distance')):
                with self.assertRaisesRegex(RuntimeError, 'invalid distance'):
                    prepare_wall_omega(object(), case)

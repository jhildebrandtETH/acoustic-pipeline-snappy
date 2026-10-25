"""Resolve actual case dictionaries to verify physical-time acoustic sampling."""
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.settings_report import read_dictionary, resolve_dictionary, create_settings_report_data


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INTERVAL = 1 / 65536


class AcousticSamplingTests(unittest.TestCase):
    def test_all_supported_templates_resolve_the_measurement_cadence(self):
        templates = sorted((ROOT / 'CoreTemplates').glob('*/*/*/system/controlDict'))
        self.assertEqual(len(templates), 9)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / 'system').mkdir()
            shutil.copytree(ROOT / 'Parameters', case / 'Parameters')
            for template in templates:
                with self.subTest(template=str(template.relative_to(ROOT))):
                    shutil.copyfile(template, case / 'system/controlDict')
                    warnings = []
                    control = resolve_dictionary(read_dictionary(case / 'system/controlDict', case, {}, warnings))
                    self.assertFalse(warnings)
                    # Foundation 13 only aligns function-object writes when the main
                    # Time controller also uses adjustableRunTime.
                    self.assertEqual(control['writeControl'], 'adjustableRunTime')
                    self.assertEqual(float(control['writeInterval']), 10 * SAMPLE_INTERVAL)
                    self.assertEqual(control['adjustTimeStep'], 'yes')
                    self.assertEqual(float(control['deltaT']), 1.2e-8)
                    self.assertEqual(float(control['maxDeltaT']), 1e-5)
                    self.assertEqual(float(control['maxCo']), 5)
                    self.assertEqual(float(control['deltaTFactor']), 1.5)
                    self.assertEqual(float(control['functions']['startupDeltaTCap']['maxCo']), 5)
                    self.assertGreaterEqual(int(control['timePrecision']), 16)
                    for name in ('writePatchFields', 'writePermeableSurfaceFields'):
                        writer = control['functions'][name]
                        self.assertEqual(writer['writeControl'], 'adjustableRunTime')
                        self.assertEqual(float(writer['writeInterval']), SAMPLE_INTERVAL)
                        self.assertEqual(writer['fields'], '(p U)')
                        self.assertEqual(writer['enabled'], 'no')

    def test_sampling_and_snapshot_settings_are_visible_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / 'system').mkdir()
            shutil.copytree(ROOT / 'Parameters', case / 'Parameters')
            shutil.copyfile(ROOT / 'CoreTemplates/AMI/kOmegaSST/wall-functions/system/controlDict',
                            case / 'system/controlDict')
            result = create_settings_report_data(case, 4000, 'AMI', 'kOmegaSST')
            sections = str(result['sections'])
            self.assertIn('Impermeable acoustic sampling', sections)
            self.assertIn('Permeable acoustic sampling', sections)
            self.assertIn('1.52587890625e-5', sections)
            self.assertIn('1.52587890625e-4', sections)
            self.assertIn('deltaTFactor', sections)

    def test_growth_escapes_five_step_alignment_plateau(self):
        # Reproduce Foundation 13 foamRun growth + Time::adjustDeltaT rounding.
        # This checks the controller, not CFD stability or temporal accuracy.
        def intervals(factor, cap=1e-5):
            dt = SAMPLE_INTERVAL / 5
            cycles = []
            for _ in range(6):
                remaining = SAMPLE_INTERVAL
                steps = []
                while remaining > 1e-15:
                    proposed = min(factor * dt, cap)
                    count = int(remaining / proposed + 0.99)
                    dt = remaining / count if count > 0 else proposed
                    steps.append(dt)
                    remaining -= dt
                self.assertAlmostEqual(sum(steps), SAMPLE_INTERVAL, places=15)
                self.assertLessEqual(max(steps), cap * 1.01)
                cycles.append(steps)
            return cycles

        configured = resolve_dictionary(read_dictionary(
            ROOT / 'Parameters/controlDict.cpp', ROOT, {}, []))
        factor = float(configured['deltaTFactor'])
        self.assertEqual(len(intervals(1.2)[-1]), 5)
        self.assertEqual(len(intervals(factor)[-1]), 2)
        self.assertAlmostEqual(intervals(factor)[-1][0], SAMPLE_INTERVAL / 2, places=15)
        # A tighter flow-derived timestep limit must still take precedence.
        self.assertGreaterEqual(len(intervals(factor, SAMPLE_INTERVAL / 8)[-1]), 8)

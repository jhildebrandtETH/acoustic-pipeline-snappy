"""Resolve actual case dictionaries to verify physical-time acoustic sampling."""
from pathlib import Path
import shutil
import json
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
                    self.assertEqual(control['writeControl'], 'runTime')
                    self.assertEqual(float(control['writeInterval']), 10 * SAMPLE_INTERVAL)
                    self.assertEqual(control['adjustTimeStep'], 'yes')
                    self.assertEqual(float(control['deltaT']), 1.2e-8)
                    self.assertEqual(float(control['maxDeltaT']), SAMPLE_INTERVAL)
                    self.assertEqual(float(control['maxCo']), 5)
                    self.assertEqual(float(control['deltaTFactor']), 1.5)
                    self.assertEqual(float(control['functions']['startupDeltaTCap']['maxCo']), 5)
                    self.assertGreaterEqual(int(control['timePrecision']), 16)
                    for name in ('writePatchFields', 'writePermeableSurfaceFields'):
                        writer = control['functions'][name]
                        self.assertEqual(writer['writeControl'], 'timeStep')
                        self.assertEqual(int(writer['writeInterval']), 1)
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

    def test_max_delta_tracks_case_sampling_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / 'system').mkdir()
            shutil.copytree(ROOT / 'Parameters', case / 'Parameters')
            control = case / 'Parameters/controlDict.cpp'
            control.write_text(control.read_text().replace('1.52587890625e-5', '2.0833333333333333e-5'))
            shutil.copyfile(ROOT / 'CoreTemplates/AMI/kOmegaSST/wall-functions/system/controlDict', case / 'system/controlDict')
            actual = resolve_dictionary(read_dictionary(case / 'system/controlDict', case, {}, []))
            self.assertAlmostEqual(float(actual['maxDeltaT']), 1 / 48000)
            self.assertEqual(actual['functions']['writePatchFields']['writeInterval'], '1')

    def test_uniform_grid_metadata_is_included_in_settings_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / 'report').mkdir()
            (case / 'report/acoustic-resampling.json').write_text(json.dumps({
                'status': 'complete', 'target_sampling_frequency_hz': 65536,
                'method': 'linear', 'observer_uniform_samples': 1000}))
            result = create_settings_report_data(case, 4000, 'AMI', 'kOmegaSST')
            sections = str(result['sections'])
            self.assertIn('Acoustic time-grid interpolation', sections)
            self.assertIn('65536', sections)
            self.assertIn('linear', sections)

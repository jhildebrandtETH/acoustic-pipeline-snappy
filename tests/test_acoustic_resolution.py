import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.mesh_case_fixture import write_mesh_case
from tools.acoustic_annotations import minimum_frequency_hz
from tools.acoustic_resolution import (measure_cell_spans, resolution_settings,
    local_cutoff, create_resolution_report_data, append_resolution_report)


class AcousticResolutionTests(unittest.TestCase):
    def test_lower_bound_defaults_override_and_invalid_values(self):
        self.assertEqual(minimum_frequency_hz({}, 6000), 100)
        self.assertEqual(minimum_frequency_hz({'blade_count': 3, 'minimum_frequency_bpf_multiple': 2}, 6000), 600)
        self.assertEqual(minimum_frequency_hz({'minimum_frequency_bpf_multiple': 0}, 6000), 0)
        for value in (-1, float('nan'), float('inf'), True, '0.5'):
            with self.assertRaises(ValueError):
                minimum_frequency_hz({'minimum_frequency_bpf_multiple': value}, 6000)

    def test_anisotropic_geometry_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            mesh = Path(tmp) / 'constant/polyMesh'
            spans, selected = measure_cell_spans(mesh, ['propeller'])
            self.assertEqual(len(spans), 736)
            self.assertGreater(len(selected), 0)
            np.testing.assert_allclose(spans, np.linalg.norm([2/12, 1.2/8, 1/8]))
            self.assertGreater(spans[0], np.cbrt(.003125))
            for name in ('points', 'faces', 'owner', 'neighbour', 'boundary'):
                path = mesh / name
                path.with_suffix('.gz').write_bytes(gzip.compress(path.read_bytes()))
                path.unlink()
            compressed, ids = measure_cell_spans(mesh, ['propeller'])
            np.testing.assert_allclose(compressed, spans)
            np.testing.assert_array_equal(ids, selected)
            with self.assertRaisesRegex(ValueError, 'absent'):
                measure_cell_spans(mesh, ['missing'])

    def write_k(self, root, value=150, time='1'):
        folder = Path(root) / time
        folder.mkdir(exist_ok=True)
        (folder / 'k').write_text('FoamFile { format ascii; class volScalarField; object k; }\n'
                                 f'dimensions [0 2 -2 0 0 0 0]; internalField uniform {value};\n')

    def test_physics_scaling(self):
        velocity, f = local_cutoff([150], [.001])
        self.assertEqual(velocity[0], 10)
        self.assertEqual(f[0], 5000)
        self.assertEqual(local_cutoff([600], [.001])[1][0], 10000)
        self.assertEqual(local_cutoff([150], [.002])[1][0], 2500)
        self.assertEqual(local_cutoff([150], [.001], 4)[1][0], 2500)
        self.assertEqual(local_cutoff([0], [.001])[1][0], 0)
        for k, h in (([-1], [.001]), ([float('nan')], [.001]), ([1], [0]), ([1, 2], [.001])):
            with self.assertRaises(ValueError):
                local_cutoff(k, h)

    def test_saved_k_cap_report_and_stale_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            root = Path(tmp)
            config = {'maximum_frequency_hz': 12000}
            self.assertEqual(create_resolution_report_data(root, config)['status'], 'unavailable')
            self.write_k(root)
            result = create_resolution_report_data(root, config)
            self.assertEqual(result['status'], 'complete')
            self.assertAlmostEqual(result['spatial_resolution_cap_hz'], 10 / (2 * np.cbrt(.003125)))
            self.assertEqual(config['maximum_frequency_hz'], 12000)
            self.assertNotIn('target_checks', result)
            self.assertNotIn('observer_sampling', result)
            self.assertEqual(result['limiting_cell']['k_m2_s2'], 150)
            from reportlab.pdfgen.canvas import Canvas
            from pypdf import PdfReader
            pdf = root / 'report/test.pdf'
            c = Canvas(str(pdf))
            append_resolution_report(c, result)
            c.save()
            reader = PdfReader(pdf)
            self.assertEqual(len(reader.pages), 1)
            text = reader.pages[0].extract_text()
            self.assertIn('Upper plot limits are unchanged', text)
            self.assertIn('Estimated spatial frequency cap:', text)
            self.assertNotIn('12000', text)
            self.assertNotIn('5000', text)
            self.assertIn('sqrt(2 k_i / 3)', text)
            self.assertTrue(Path(result['csv']).is_file())
            (root / '1/k').unlink()
            missing = create_resolution_report_data(root, config)
            self.assertEqual(missing['status'], 'unavailable')
            self.assertIsNone(missing['spatial_resolution_cap_hz'])
            self.assertFalse((root / 'report/acoustic-resolution-cells.csv').exists())
            self.assertEqual(json.loads((root / 'report/acoustic-resolution.json').read_text())['status'], 'unavailable')

    def test_local_minimum_and_zero_are_not_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            root = Path(tmp)
            self.write_k(root)
            _, selected = measure_cell_spans(root / 'constant/polyMesh', ['propeller'])
            values = np.full(736, 150.)
            values[selected[7]] = 1.5
            header = 'FoamFile { format ascii; } dimensions [0 2 -2 0 0 0 0]; '
            def write():
                (root / '1/k').write_text(header + 'internalField nonuniform List<scalar> 736\n(\n' +
                                         '\n'.join(map(str, values)) + '\n);\n')
            write()
            result = create_resolution_report_data(root, {})
            self.assertEqual(result['limiting_cell']['cell_id'], selected[7])
            self.assertAlmostEqual(result['spatial_resolution_cap_hz'], 1 / (2 * np.cbrt(.003125)))
            values[selected[7]] = 0
            write()
            result = create_resolution_report_data(root, {})
            self.assertEqual(result['spatial_resolution_cap_hz'], 0)
            self.assertEqual(result['zero_k_source_cells'], 1)
            values[selected[7]] = -1
            write()
            self.assertEqual(create_resolution_report_data(root, {})['status'], 'unavailable')

    def test_gzip_flow_moving_mesh_and_latest_time(self):
        import re
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            root = Path(tmp)
            self.write_k(root)
            initial = create_resolution_report_data(root, {})
            path = root / '1/k'
            path.with_suffix('.gz').write_bytes(gzip.compress(path.read_bytes()))
            path.unlink()
            self.assertEqual(create_resolution_report_data(root, {})['spatial_resolution_cap_hz'], initial['spatial_resolution_cap_hz'])
            mesh = root / '1/polyMesh'
            mesh.mkdir()
            points = (root / 'constant/polyMesh/points').read_text()
            scaled = re.sub(r'\(([^()]+)\)', lambda m: '(' + ' '.join(str(2 * float(v)) for v in m[1].split()) + ')', points)
            (mesh / 'points').write_text(scaled)
            moved = create_resolution_report_data(root, {})
            self.assertAlmostEqual(moved['spatial_resolution_cap_hz'], initial['spatial_resolution_cap_hz'] / 2)
            (root / '2').mkdir()
            missing = create_resolution_report_data(root, {})
            self.assertEqual(missing['status'], 'unavailable')
            self.assertEqual(missing['time_s'], 2)
            self.assertIsNone(missing['spatial_resolution_cap_hz'])
            self.write_k(root, time='2')
            (root / 'processor0/3').mkdir(parents=True)
            stale = create_resolution_report_data(root, {})
            self.assertEqual(stale['status'], 'unavailable')
            self.assertIn('Processor output is newer', stale['reason'])

    def test_invalid_settings(self):
        for values in ({'cells_per_length_scale': 0}, {'cells_per_length_scale': True},
                       {'cells_per_length_scale': float('nan')}, {'source_patches': []},
                       {'cell_size': 'unknown'}, {'typo': 12}):
            with self.assertRaises(ValueError):
                resolution_settings({'mesh_resolution': values})
        legacy = resolution_settings({'mesh_resolution': {'cells_per_wavelength': 12, 'sound_speed_m_s': 343}})
        self.assertEqual(legacy['cells_per_length_scale'], 2)
        self.assertIn('sound_speed_m_s', legacy['ignored_legacy_settings'])

    def test_simulation_only_lower_bound_and_empty_band(self):
        from tools.acoustic_tonal import write_cached_simulation_tonal
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report'
            report.mkdir()
            fs = 8000
            np.savez(report / 'acoustic-pressure.npz', pressure_pa=np.sin(2*np.pi*500*np.arange(fs)/fs),
                     sampling_frequency_hz=fs, rpm=6000, geometry='test')
            config = dict(floor_db=-120, maximum_frequency_hz=3000, minimum_frequency_bpf_multiple=2)
            result = write_cached_simulation_tonal(tmp, config)
            self.assertEqual(result['covered_hz'], [400, 3000])
            config['minimum_frequency_bpf_multiple'] = 0
            self.assertEqual(write_cached_simulation_tonal(tmp, config)['covered_hz'], [1, 3000])
            config['minimum_frequency_bpf_multiple'] = 20
            with self.assertRaises(ValueError):
                write_cached_simulation_tonal(tmp, config)

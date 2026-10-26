import copy
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools.acoustic_directivity import (frame, transform_position, resolve_observers,
    cache_observers, select_cached_observer, plane_coordinates, directivity_or_status,
    append_directivity_report, elevation_planes, create_directivity)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class DirectivityTests(unittest.TestCase):
    def fixture(self, root):
        case = root / '10x7e_4000RPM_case'
        (case / 'report').mkdir(parents=True)
        rate = 8000
        times = np.arange(rate) / rate
        positions = {'1': [1, 0, .75], '6': [-1, 0, .75], '8': [0, 0, 1.75], '10': [.707, 0, 1.457], '2': [0, 1, .75]}
        channels = {k: dict(channel_position=v, sampling_frequency=rate,
                           pressure=(1 + int(k) / 10) * np.sin(2*np.pi*400*times)) for k, v in positions.items()}
        path = root / '10x7e_2024-11-28-10-00-00_RPM_4000.pkl'
        path.write_bytes(pickle.dumps(dict(propeller_type='10x7e', targeted_rpm=4000, sound={'channels': channels})))
        config = dict(measurements_directory=str(root), microphone_channel='1', maximum_frequency_hz=3000,
                      floor_db=-120, spectrum_bandwidth_hz=20, frequency_bins_hz=[[0, None]], blade_count=2,
                      directivity=dict(enabled=True, experimental_rotation_center_m=[0, 0, .75],
                      planes=[dict(name='XZ', channels=['1', '8', '10', '6'], zero_direction=[1, 0, 0], positive_direction=[0, 0, 1])]))
        ids, raw, mapped = resolve_observers(case, 4000, config)
        result = SimpleNamespace(observer_times_s=Tensor(times), total_pressure_pa=Tensor([channels[k]['pressure'] for k in ids]),
                                 observer_m=Tensor(mapped), maximum_delay_s=np.array([.1, .2, .3, .4, .5]))
        cache_observers(result, case, 4000, config, ids, raw, rotations=100)
        return case, config, path, ids, raw

    def test_unknown_centre_and_invalid_rotations_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Enter'):
            frame({'directivity': {'enabled': True}})
        for rotation in ([[1, 0, 0], [0, 2, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, -1]]):
            with self.assertRaises(ValueError):
                frame({'directivity': dict(experimental_rotation_center_m=[0, 0, 0], experimental_to_simulation_rotation=rotation)})

    def test_coordinate_mapping_and_override(self):
        config = {'directivity': dict(enabled=True, experimental_rotation_center_m=[0, 0, .75],
                   simulation_rotation_center_m=[1, 2, 3], experimental_to_simulation_rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                   microphone_positions_m={'1': [2, 0, .75]})}
        np.testing.assert_allclose(transform_position([9, 9, 9], '1', config), [1, 4, 3])

    def test_all_channels_common_window_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, config, _, ids, _ = self.fixture(Path(temporary))
            with np.load(case / 'report/acoustic-observers.npz') as data:
                self.assertEqual(data['pressure_pa'].shape[0], 5)
                self.assertGreaterEqual(float(data['window_start_s']), .5)
                selected = data['pressure_pa'][ids.index('8')].copy()
            config['microphone_channel'] = '8'
            select_cached_observer(case, config)
            with np.load(case / 'report/acoustic-pressure.npz') as data:
                np.testing.assert_array_equal(data['pressure_pa'], selected)
                np.testing.assert_allclose(data['observer_m'], [0, 0, 1])
            config['directivity']['experimental_rotation_center_m'] = [0, 0, 0]
            with self.assertRaisesRegex(ValueError, 'rerun acoustic'):
                select_cached_observer(case, config)

    def test_plane_angles_and_non_coplanar_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, config, _, ids, raw = self.fixture(Path(temporary))
            plane = config['directivity']['planes'][0]
            output = plane_coordinates(plane, ids, raw, config)
            self.assertEqual(output['channels'], ['1', '10', '8', '6'])
            np.testing.assert_allclose(output['angles_deg'], [0, 45, 90, 180])
            plane['channels'].append('2')
            with self.assertRaisesRegex(ValueError, 'not coplanar'):
                plane_coordinates(plane, ids, raw, config)

    def test_elevation_selection_angles_and_rotated_frame(self):
        # Known meridional positions, plus one observer in the rotor plane
        # outside the requested section. The latter must not be projected.
        points = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        ids = ['1', '2', '3', '4', '5', '6']
        for rotation in (np.eye(3), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])):
            origin = np.array([.1, .2, .75])
            config = {'directivity': dict(experimental_rotation_center_m=origin.tolist(),
                experimental_to_simulation_rotation=rotation.tolist(),
                elevation_planes=[dict(name='Elevation', radial_direction_simulation=[1, 0, 0])])}
            raw = points @ rotation + origin
            planes, omitted = elevation_planes(ids, raw, config)
            self.assertEqual(omitted, [])
            self.assertEqual(planes[0]['channels'], ids[:5])
            np.testing.assert_allclose(planes[0]['angles_deg'], [0, 45, 90, 180, 270])
            normal = np.cross(planes[0]['zero_direction'], planes[0]['positive_direction'])
            self.assertAlmostEqual(float((rotation @ normal)[1]), 0)
            self.assertIn('rotor plane', planes[0]['angle_convention'])

    def test_elevation_rejects_invalid_direction_and_omits_sparse_plane(self):
        config = {'directivity': dict(experimental_rotation_center_m=[0, 0, 0],
            elevation_planes=[dict(name='Elevation', radial_direction_simulation=[0, 1, 0])])}
        with self.assertRaisesRegex(ValueError, 'perpendicular'):
            elevation_planes(['1', '2', '3'], np.eye(3), config)
        config['directivity']['elevation_planes'][0]['radial_direction_simulation'] = [1, 0, 0]
        planes, omitted = elevation_planes(['1', '2', '3'], np.eye(3), config)
        self.assertEqual(planes, [])
        self.assertIn('fewer than three', omitted[0])
        planes, omitted = elevation_planes(['1', '2', '3'], [[1, 0, 0], [2, 0, 0], [-1, 0, 0]], config)
        self.assertEqual(planes, [])
        self.assertIn('distinct sampled angles', omitted[0])

    def test_elevation_report_reuses_all_channel_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, config, _, _, _ = self.fixture(Path(temporary))
            config['directivity']['planes'] = []
            config['directivity']['elevation_planes'] = [
                dict(name='XY elevation', radial_direction_simulation=[1, 0, 0]),
                dict(name='ZY elevation', radial_direction_simulation=[0, 0, 1])]
            result = create_directivity(case, config)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['planes']), 1)
            plane = result['planes'][0]
            self.assertEqual(plane['channels'], ['1', '2', '6'])
            np.testing.assert_allclose(plane['angles_deg'], [0, 90, 180])
            self.assertEqual(len(result['omitted_planes']), 1)
            for filename in [plane['figure'], *plane['comparison_figures'].values()]:
                self.assertTrue((case / 'report' / filename).is_file())
            self.assertTrue((case / 'report/acoustic-directivity.csv').is_file())

    def test_inconsistent_repeats_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, config, path, _, _ = self.fixture(Path(temporary))
            data = pickle.loads(path.read_bytes())
            del data['sound']['channels']['8']
            path.with_name(path.name.replace('10-00-00', '11-00-00')).write_bytes(pickle.dumps(data))
            with self.assertRaisesRegex(ValueError, 'same channels'):
                resolve_observers(case, 4000, config)

    def test_report_energy_and_invalidated_status(self):
        from reportlab.pdfgen.canvas import Canvas
        from tools.acoustic_comparison import comparison_or_status
        with tempfile.TemporaryDirectory() as temporary:
            case, config, _, _, _ = self.fixture(Path(temporary))
            comparison = comparison_or_status(case, config)
            self.assertEqual(comparison['status'], 'complete')
            self.assertTrue(comparison['positions_match'])
            result = json.loads((case / 'report/acoustic-directivity.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertAlmostEqual(result['covered_hz'][0], 4000 / 60)
            self.assertEqual(result['covered_hz'][1], 3000)
            self.assertEqual(len(result['channels']), 5)
            for row in result['channels'].values():
                for kind in ('simulated', 'measured'):
                    power = row[kind]['power_pa2']
                    self.assertAlmostEqual(power['overall'], power['tonal'] + power['broadband'])
            self.assertTrue((case / 'report/acoustic-directivity-plane-1.png').exists())
            self.assertEqual(set(result['planes'][0]['comparison_figures']), {'overall', 'broadband', 'tonal'})
            for filename in result['planes'][0]['comparison_figures'].values():
                self.assertTrue((case / 'report' / filename).is_file())
            canvas = Canvas(str(case / 'report/directivity.pdf'))
            append_directivity_report(canvas, case)
            self.assertEqual(canvas.getPageNumber(), 4)
            canvas.save()
            config['directivity']['planes'][0]['channels'].append('2')
            failed = directivity_or_status(case, config)
            self.assertEqual(failed['status'], 'unavailable')
            self.assertIn('not coplanar', failed['detail'])

    def test_disabled_needs_no_coordinates_or_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            (case / 'report').mkdir()
            self.assertEqual(directivity_or_status(case, {})['status'], 'disabled')

    def test_propagation_batches_every_channel_and_keeps_selected_trace(self):
        import acoustic_propagation as stage
        with tempfile.TemporaryDirectory() as temporary:
            case, config, _, ids, raw = self.fixture(Path(temporary))
            config['microphone_channel'] = '8'
            (case / 'Parameters').mkdir()
            (case / 'Parameters/acousticComparison.json').write_text(json.dumps(config))
            (case / 'Parameters/controlDict.cpp').write_text('acousticWriteInterval 0.000125;')
            for time in map(str, np.arange(801) / 8000):
                directory = case / 'postProcessing/writePatchFields' / time
                directory.mkdir(parents=True)
                (directory / 'propeller.vtk').write_text('placeholder for mocked source loader')
            _, _, mapped = resolve_observers(case, 4000, config)
            times = np.arange(8000) / 8000
            pressures = np.array([(i + 1) * np.sin(2*np.pi*400*times) for i in range(len(ids))])
            import torch
            from acousticSolver.src.foamacoustics.core.result import F1AResult
            values = torch.tensor(pressures, dtype=torch.float64)
            result = F1AResult(observer_times_s=torch.tensor(times, dtype=torch.float64),
                               thickness_pressure_pa=torch.zeros_like(values), loading_pressure_pa=values,
                               total_pressure_pa=values, observer_m=torch.tensor(mapped), rpm=4000,
                               maximum_delay_s=torch.zeros(len(ids)), response_delay_s=None,
                               source_time_count=801, propagation_algorithm='nonuniform')
            solver = Mock()
            solver.predict.return_value = result
            def comparison(case_path, unused_config):
                (case_path / 'report/acoustic-simulated.png').write_bytes(b'test image')
                return {'status': 'complete'}
            with patch.object(stage.F1ASolver, 'from_openfoam_vtk', return_value=solver), \
                 patch.object(stage, 'create_reference_geometry_vtk_series'), \
                 patch.object(stage, 'comparison_or_status', side_effect=comparison):
                stage.run_acoustic_solver('impermeable', case, 4000)
            self.assertEqual(solver.predict.call_count, 1)
            metadata = json.loads((case / 'report/acoustic-resampling.json').read_text())
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(metadata['method'], 'linear')
            self.assertEqual(metadata['target_sampling_frequency_hz'], 8000)
            np.testing.assert_allclose(solver.predict.call_args.args[0].cpu().numpy(), mapped)
            with np.load(case / 'report/acoustic-observers.npz') as multi, np.load(case / 'report/acoustic-pressure.npz') as single:
                self.assertEqual(multi['pressure_pa'].shape[0], 5)
                self.assertEqual(float(multi['sampling_frequency_hz']), 8000)
                np.testing.assert_allclose(np.diff(multi['time_s']), 1 / 8000)
                np.testing.assert_array_equal(single['pressure_pa'], multi['pressure_pa'][ids.index('8')])


if __name__ == '__main__':
    unittest.main()

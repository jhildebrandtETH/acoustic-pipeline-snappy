import json
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from tools.acoustic_comparison import (
    agreement_metrics, cache_result, comparison_or_status, create_comparison, integrate_density,
    load_config, load_channel, match_measurements, resolve_observer_position, validate_config, welch_density,
)


class AcousticComparisonTests(unittest.TestCase):
    def config(self, directory):
        return dict(measurements_directory=str(directory), microphone_channel='1',
                    spectrum_bandwidth_hz=20.,
                    maximum_frequency_hz=3000., floor_db=-120.,
                    frequency_bins_hz=[[0, 1000], [1000, 2000], [2000, None]])

    def test_case_configs_are_isolated_and_reread(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, channel in [('first', '1'), ('second', '6')]:
                parameters = root / name / 'Parameters'
                parameters.mkdir(parents=True)
                (parameters / 'acousticComparison.json').write_text(json.dumps({'microphone_channel': channel}))
            self.assertEqual(load_config(root / 'first')['microphone_channel'], '1')
            self.assertEqual(load_config(root / 'second')['microphone_channel'], '6')
            (root / 'first/Parameters/acousticComparison.json').write_text('{"microphone_channel": "3"}')
            self.assertEqual(load_config(root / 'first')['microphone_channel'], '3')
            self.assertEqual(load_config(root / 'second')['microphone_channel'], '6')

    def test_missing_case_config_never_uses_source_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, 'Missing case acoustic config'):
                load_config(case)
            with self.assertRaisesRegex(FileNotFoundError, 'Missing case acoustic config'):
                create_comparison(case)
            status = comparison_or_status(case)
            self.assertEqual(status['status'], 'unavailable')
            self.assertIn(str(case / 'Parameters/acousticComparison.json'), status['detail'])

    def test_alternate_config_must_belong_to_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / 'case'
            parameters = case / 'Parameters'
            parameters.mkdir(parents=True)
            alternate = parameters / 'alternate.json'
            alternate.write_text('{"microphone_channel": "6"}')
            external = root / 'external.json'
            external.write_text('{"microphone_channel": "1"}')
            self.assertEqual(load_config(case, 'alternate.json')['microphone_channel'], '6')
            self.assertEqual(load_config(case, alternate)['microphone_channel'], '6')
            for path in (external, '../../external.json'):
                with self.assertRaisesRegex(ValueError, 'must be inside'):
                    load_config(case, path)

    def test_area_identity_offset_and_partial_bins(self):
        edges = np.array([10., 900., 1500., 3000.])
        overall, rows = agreement_metrics(edges, np.array([3., 3., 3.]), np.zeros(3),
                                          [[0, 1000], [1000, 2000], [2000, None]])
        self.assertAlmostEqual(overall['area_db_hz'], 3 * 2990)
        self.assertAlmostEqual(sum(r['area_db_hz'] for r in rows), overall['area_db_hz'])
        for row in rows:
            self.assertAlmostEqual(row['mae_db'], 3)
            self.assertAlmostEqual(row['rmse_db'], 3)
        same, _ = agreement_metrics(edges, np.zeros(3), np.zeros(3), [[0, None]])
        self.assertEqual(same['mae_db'], 0)

    def test_signed_errors_do_not_cancel_and_no_overlap_is_unavailable(self):
        total, bins = agreement_metrics(np.array([0., 1., 2.]), np.array([-4., 4.]), np.zeros(2), [[0, 2], [3, 4]])
        self.assertEqual(total['mae_db'], 4)
        self.assertEqual(total['bias_db'], 0)
        self.assertEqual(bins[-1]['status'], 'unavailable')

    def test_welch_sine_energy_at_different_sample_rates(self):
        for rate in [8000, 16000]:
            time = np.arange(rate) / rate
            freq, density = welch_density(np.sqrt(2) * np.sin(2 * np.pi * 500 * time), rate, .1)
            power = integrate_density(freq, density, np.array([0., 1000.]))[0]
            self.assertAlmostEqual(power, 1., places=10)
        with self.assertRaises(ValueError):
            welch_density(np.ones(10), 1000, .1)

    def write_measurement(self, root, stamp='2024-11-28-10-00-00', amplitude=1.):
        rate = 8000
        trace = amplitude * np.sin(2 * np.pi * 500 * np.arange(rate) / rate)
        path = root / f'10x7e_{stamp}_RPM_4000.pkl'
        path.write_bytes(pickle.dumps(dict(propeller_type='10x7e', targeted_rpm=4000,
            sound={'channels': {'1': dict(pressure=trace, sampling_frequency=rate, channel_position=[1., 0., 0.])}})))
        return path, trace, rate

    def test_matching_metadata_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, trace, rate = self.write_measurement(root)
            self.write_measurement(root, '2024-11-28-11-00-00', amplitude=2.)
            self.assertEqual(len(match_measurements(root, '10X7E', 4000)), 2)
            with self.assertRaises(ValueError):
                match_measurements(root, '10x5e', 4000)
            with self.assertRaises(ValueError):
                load_channel(path, '1', '10x7e', 4500)
            with self.assertRaises(ValueError):
                load_channel(path, '2', '10x7e', 4000)
            report = root / 'report'
            report.mkdir()
            np.savez(report / 'acoustic-pressure.npz', pressure_pa=trace * np.sqrt(2.5),
                     sampling_frequency_hz=rate, geometry='10x7e', rpm=4000., observer_m=[1., 0., 0.])
            (root / 'Parameters').mkdir()
            (root / 'Parameters/acousticComparison.json').write_text(json.dumps(self.config(root)))
            result = create_comparison(root)
            self.assertAlmostEqual(result['band_limited_level_difference_db'], 0., places=8)
            self.assertTrue(result['positions_match'])
            self.assertAlmostEqual(result['segment_duration_s'], len(trace) / rate)
            self.assertAlmostEqual(result['fft_spacing_hz'], rate / len(trace))
            self.assertEqual(result['repeat_count'], 2)
            tonal = result['tonal_analysis']['spectra']
            self.assertAlmostEqual(tonal['simulated']['overall_spl_db'], result['simulated_band_limited_spl_db'])
            self.assertAlmostEqual(tonal['measured']['overall_spl_db'], result['measured_band_limited_spl_db'])
            for name in ['acoustic-tonal-simulated.png', 'acoustic-tonal-measured.png', 'acoustic-tonal.json']:
                self.assertTrue((report / name).is_file())
            from reportlab.pdfgen.canvas import Canvas
            from tools.acoustic_comparison import append_comparison_report
            class ReportCanvas(Canvas):
                images = []
                def drawImage(self, image, *args, **kwargs):
                    self.images.append(Path(image).name)
                    return super().drawImage(image, *args, **kwargs)
            canvas = ReportCanvas(str(report / 'test-report.pdf'))
            append_comparison_report(canvas, root)
            self.assertEqual(canvas.images, ['acoustic-psd-overlay.png', 'acoustic-tonal-simulated.png', 'acoustic-tonal-measured.png'])
            self.assertEqual(canvas.getPageNumber(), 4)
            self.assertEqual(result['overall']['covered_hz'], [4000 / 60, 3000.])
            canvas.save()
            for name in ['acoustic-simulated.png', 'acoustic-measured.png', 'acoustic-overlay.png', 'acoustic-comparison.csv']:
                self.assertTrue((report / name).is_file())
            (report / 'acoustic-pressure.npz').unlink()
            result = comparison_or_status(root, self.config(root))
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(json.loads((report / 'acoustic-comparison.json').read_text())['status'], 'unavailable')

    def test_invalid_bins(self):
        config = self.config('.')
        config['frequency_bins_hz'] = [[0, 1000], [900, None]]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_cache_excludes_propagation_startup_and_records_observer(self):
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / '10x7e_4000RPM_AMI'
            (case / 'report').mkdir(parents=True)
            time = np.arange(1000) / 1000
            result = SimpleNamespace(observer_times_s=Tensor(time), total_pressure_pa=Tensor(time),
                                     observer_m=Tensor([1, 0, 0]), maximum_delay_s=.5)
            cache_result(result, case, 4000, rotations=15)
            with np.load(case / 'report/acoustic-pressure.npz') as data:
                self.assertEqual(str(data['geometry']), '10x7e')
                self.assertGreaterEqual(data['pressure_pa'].min(), .5)
                np.testing.assert_equal(data['observer_m'], [1, 0, 0])
                self.assertAlmostEqual(float(data['requested_duration_s']), .225)
                self.assertAlmostEqual(len(data['pressure_pa']) / float(data['sampling_frequency_hz']), .225, delta=.002)
            # A requested window longer than the fully arrived tail is truncated.
            cache_result(result, case, 2000, rotations=30)
            with np.load(case / 'report/acoustic-pressure.npz') as data:
                self.assertAlmostEqual(float(data['requested_duration_s']), .9)
                actual = len(data['pressure_pa']) / float(data['sampling_frequency_hz'])
                self.assertAlmostEqual(actual, .5, delta=.002)
                self.assertGreaterEqual(data['pressure_pa'].min(), .5)
            with self.assertRaises(ValueError):
                cache_result(result, case, 4000, rotations=0)

    def test_silent_trace_has_finite_error(self):
        from tools.acoustic_comparison import levels
        self.assertEqual(float(levels(0., -120)), -120.)
        freq, density = welch_density(np.ones(1000), 1000., .1)
        self.assertTrue(np.all(density == 0))


    def test_observer_follows_selected_channel_and_ignores_old_manual_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.write_measurement(root)
            data = pickle.loads(path.read_bytes())
            data['sound']['channels']['6'] = dict(data['sound']['channels']['1'], channel_position=[1.577, 0., 1.789])
            path.write_bytes(pickle.dumps(data))
            config = self.config(root)
            config['observer_m'] = [999, 999, 999]
            case = root / '10x7e_4000RPM_AMI'
            self.assertEqual(resolve_observer_position(case, 4000, config), [1., 0., 0.])
            config['microphone_channel'] = '6'
            self.assertEqual(resolve_observer_position(case, 4000, config), [1.577, 0., 1.789])
            config['microphone_channel'] = '11'
            with self.assertRaisesRegex(ValueError, 'Microphone 11 missing'):
                resolve_observer_position(case, 4000, config)
            with self.assertRaisesRegex(ValueError, 'No experimental measurements match'):
                resolve_observer_position(case, 4500, config)

    def test_observer_rejects_invalid_and_inconsistent_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_measurement(root)
            path, _, _ = self.write_measurement(root, '2024-11-28-11-00-00')
            config = self.config(root)
            case = root / '10x7e_4000RPM_AMI'
            self.assertEqual(resolve_observer_position(case, 4000, config), [1., 0., 0.])
            for position, message in [([2., 0., 0.], 'different microphone positions'),
                                      ([1., float('nan'), 0.], 'three finite coordinates'),
                                      ([1., 0.], 'three finite coordinates')]:
                data = pickle.loads(path.read_bytes())
                data['sound']['channels']['1']['channel_position'] = position
                path.write_bytes(pickle.dumps(data))
                with self.assertRaisesRegex(ValueError, message):
                    resolve_observer_position(case, 4000, config)

    def test_observer_without_measurements_uses_original_simulation_position(self):
        config = self.config('.')
        config['measurements_directory'] = None
        self.assertEqual(resolve_observer_position('case', 4000, config), [1., 0., 0.])


if __name__ == '__main__':
    unittest.main()

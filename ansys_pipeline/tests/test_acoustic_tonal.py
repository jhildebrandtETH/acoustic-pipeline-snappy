import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.acoustic_tonal import separate_tones, tonal_settings, write_tonal_figures
from tools.acoustic_comparison import welch_density, integrate_density


class TonalTests(unittest.TestCase):
    def test_flat_and_silent_spectra_have_no_tonal_energy(self):
        frequency = np.arange(100.)
        for density in (np.zeros(100), np.ones(100)):
            broadband, tonal, peaks = separate_tones(frequency, density, tonal_settings({}))
            np.testing.assert_array_equal(broadband, density)
            self.assertEqual(tonal.sum(), 0)
            self.assertEqual(len(peaks), 0)

    def test_narrow_tones_preserve_background_and_energy(self):
        frequency = np.arange(200.)
        density = np.ones(200) * 2e-6
        density[49:52] += np.array([.01, .04, .01])
        density[82] += .05
        broadband, tonal, peaks = separate_tones(frequency, density, tonal_settings({}))
        np.testing.assert_array_equal(peaks, [50, 82])
        np.testing.assert_allclose(broadband, 2e-6)
        np.testing.assert_allclose(broadband + tonal, density)
        self.assertTrue(np.all(tonal >= 0))
        self.assertAlmostEqual(integrate_density(frequency, tonal, np.array([1., 199.]))[0], .11)

    def test_hann_tone_energy_is_resolved_before_rebinning(self):
        fs = 8000
        time = np.arange(fs) / fs
        frequency, density = welch_density(np.sqrt(2) * np.sin(2 * np.pi * 500 * time), fs, 1.)
        broadband, tonal, peaks = separate_tones(frequency, density, tonal_settings({}))
        self.assertIn(500, frequency[peaks])
        self.assertAlmostEqual(integrate_density(frequency, tonal, np.array([1., 3000.]))[0], 1., places=8)
        np.testing.assert_allclose(broadband + tonal, density)

    def test_broad_hump_is_not_a_narrow_tone(self):
        frequency = np.arange(200.)
        density = 1 + 100 * np.exp(-.5 * ((frequency - 100) / 12) ** 2)
        _, tonal, peaks = separate_tones(frequency, density, tonal_settings({}))
        self.assertEqual(len(peaks), 0)
        self.assertEqual(tonal.sum(), 0)

    def test_off_bin_hann_tone_and_truncated_band_conserve_energy(self):
        fs = 8000
        time = np.arange(fs) / fs
        frequency, density = welch_density(np.sqrt(2) * np.sin(2 * np.pi * 500.3 * time), fs, 1.)
        background, tonal, peaks = separate_tones(frequency, density, tonal_settings({}))
        self.assertIn(500, frequency[peaks])
        bounds = np.array([499.7, 1000.2])
        original = integrate_density(frequency, density, bounds)[0]
        self.assertAlmostEqual(integrate_density(frequency, background, bounds)[0] +
                               integrate_density(frequency, tonal, bounds)[0], original)
        self.assertGreater(integrate_density(frequency, tonal, np.array([1., 3000.]))[0], .99)

    def test_configuration_and_native_grid_validation(self):
        for options in ({'median_window_bins': 10}, {'half_width_bins': 0},
                        {'threshold_db': float('nan')}, {'extra': 1}):
            with self.assertRaises(ValueError):
                tonal_settings({'tonal_analysis': options})
        with self.assertRaises(ValueError):
            separate_tones([0, 1, 3], [1, 2, 1], tonal_settings({}))

    def test_exports_conserve_energy_and_classify_detected_peaks(self):
        from matplotlib.figure import Figure
        frequency = np.arange(201.)
        density = np.ones(201) * 1e-6
        density[50] += .01
        density[83] += .02
        metadata = dict(rpm=1500, blade_count=2, geometry='synthetic validation', microphone_channel='1')
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary)
            scales = []
            save = Figure.savefig
            def capture(fig, *args, **kwargs):
                scales.append(fig.axes[0].get_xscale())
                return save(fig, *args, **kwargs)
            with patch.object(Figure, 'savefig', capture):
                result = write_tonal_figures(report,
                    {'simulated': [(frequency, density)], 'measured': [(frequency, density), (frequency, density * 2)]},
                    metadata, {'floor_db': -120}, 1., 200.)
            self.assertEqual(scales, ['log', 'log'])
            sim, measured = result['spectra']['simulated'], result['spectra']['measured']
            self.assertEqual(sim['bpf_tones_hz'], [50.])
            self.assertEqual(sim['detected_tones_hz'], [50., 83.])
            for kind, row in result['spectra'].items():
                powers = row['power_pa2']
                self.assertAlmostEqual(powers['tonal'] + powers['broadband'], powers['overall'])
                self.assertTrue((report / f'acoustic-tonal-{kind}.png').is_file())
                data = np.loadtxt(report / f'acoustic-tonal-{kind}.csv', delimiter=',', skiprows=1)
                np.testing.assert_allclose(data[:, 1], data[:, 2] + data[:, 3])
            self.assertAlmostEqual(measured['power_pa2']['overall'], 1.5 * sim['power_pa2']['overall'])
            self.assertEqual(json.loads((report / 'acoustic-tonal.json').read_text())['method'], result['method'])


if __name__ == '__main__':
    unittest.main()

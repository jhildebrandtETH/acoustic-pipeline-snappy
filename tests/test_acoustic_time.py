from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.acoustic_time import case_sample_interval, source_time_statistics, uniform_pressure, uniform_result


class AcousticTimeTests(unittest.TestCase):
    def test_adaptive_startup_and_multichannel_linear_pressure(self):
        step = 1 / 65536
        times = np.r_[0, np.cumsum(np.r_[np.full(12, .001), np.tile([.7, 1., .8, .9], 30)])] * step
        values = np.array([2 * times + 3, -times + 7])
        grid, actual, dt = uniform_pressure(times, values, step)
        np.testing.assert_allclose(actual, [2 * grid + 3, -grid + 7], atol=1e-14)
        np.testing.assert_allclose(np.diff(grid), step, rtol=1e-12)
        self.assertEqual(dt, step)
        self.assertGreaterEqual(grid[0], times[0])
        self.assertLessEqual(grid[-1], times[-1])

    def test_no_endpoint_extrapolation_and_no_missing_history_repair(self):
        grid, _, _ = uniform_pressure([.1, .6, 1.1, 1.6], [0, 1, 2, 3], .5)
        np.testing.assert_array_equal(grid, [.5, 1., 1.5])
        with self.assertRaisesRegex(ValueError, 'gap'):
            uniform_pressure([0, .5, 1.6], [0, 1, 2], .5)
        for times, values, step in [([0, 0], [1, 2], 1), ([0, 1], [1, np.nan], 1),
                                    ([0, 1], [1, 2], 0), ([0, 1], [1], 1)]:
            with self.assertRaises(ValueError):
                uniform_pressure(times, values, step)

    def test_tone_frequency_and_amplitude_survive_adaptive_interpolation(self):
        rate = 65536
        times = np.r_[0, np.cumsum(np.tile([.6, .8, 1., .8], 21000)) / rate]
        grid, pressure, _ = uniform_pressure(times, np.sin(2*np.pi*1000*times), 1/rate)
        pressure = pressure[:rate]
        spectrum = np.abs(np.fft.rfft(pressure)) * 2 / len(pressure)
        self.assertEqual(np.argmax(spectrum), 1000)
        self.assertGreater(spectrum[1000], .998)
        self.assertLess(spectrum[1000], 1.001)

    def test_case_configuration_is_local_and_resolves_sampling_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case/'Parameters').mkdir()
            (case/'system').mkdir()
            (case/'Parameters/controlDict.cpp').write_text('acousticWriteInterval 0.001; maxDeltaT $acousticWriteInterval;')
            (case/'system/controlDict').write_text('#include "../Parameters/controlDict.cpp"\nacousticWriteInterval 0.002;')
            self.assertEqual(case_sample_interval(case), .002)
            self.assertEqual(source_time_statistics([0, .001, .003], .002)['source_sample_count'], 3)
            (case/'system/controlDict').write_text('maxDeltaT 0.1;')
            with self.assertRaisesRegex(ValueError, 'acousticWriteInterval'):
                case_sample_interval(case)

    def test_result_components_share_one_grid_and_keep_metadata(self):
        import torch
        @dataclass(frozen=True)
        class Result:
            observer_times_s: object
            thickness_pressure_pa: object
            loading_pressure_pa: object
            total_pressure_pa: object
            marker: str = 'preserved'
        times = torch.tensor([0, .2, .55, .9, 1.1], dtype=torch.float64)
        loading = torch.stack([2*times, 3*times])
        result = uniform_result(Result(times, loading/2, loading, loading*1.5), .4)
        np.testing.assert_allclose(result.observer_times_s, [0, .4, .8])
        torch.testing.assert_close(result.total_pressure_pa, result.thickness_pressure_pa + result.loading_pressure_pa)
        self.assertEqual(result.marker, 'preserved')


if __name__ == '__main__':
    unittest.main()

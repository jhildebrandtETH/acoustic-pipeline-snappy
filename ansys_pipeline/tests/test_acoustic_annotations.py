import unittest
import numpy as np
from tools.plotting import pyplot as plt
from tools.acoustic_annotations import add_bpf_markers, band_total_db

class AcousticAnnotationsTests(unittest.TestCase):
    def test_harmonics_follow_blades_rpm_and_visible_band(self):
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(100, 1000)
            self.assertEqual(add_bpf_markers(ax, 4000, 3), 200)
            self.assertEqual([line.get_xdata()[0] for line in ax.lines], [200,400,600,800,1000])
            self.assertEqual(ax.get_xlim(), (100,1000))
        finally:
            plt.close(fig)

    def test_total_sums_energy_not_decibels(self):
        self.assertAlmostEqual(band_total_db([60,60]), 63.0102999566)

    def test_invalid_blade_count_rejected(self):
        fig, ax = plt.subplots()
        try:
            for blades in [0, 1.5, True]:
                with self.assertRaises(ValueError):
                    add_bpf_markers(ax, 4000, blades)
        finally:
            plt.close(fig)

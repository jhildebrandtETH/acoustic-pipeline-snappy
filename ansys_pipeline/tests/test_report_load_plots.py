"""Regression checks for report-only load scaling and legend placement."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools import reporting


class ReportLoadPlotTests(unittest.TestCase):
    def test_startup_spike_is_outside_the_full_history_display(self):
        times = np.linspace(0, .1, 201)
        loads = 2 + .05 * np.sin(times * 400)
        loads[0] = 50000

        def inspect(figure, _):
            history = figure.axes[0]
            self.assertEqual(history.lines[0].get_ydata()[0], 50000)
            self.assertLess(history.get_ylim()[1], 3)
            self.assertGreater(history.get_xlim()[1], .1)
            self.assertIn("outside y-range", history.texts[0].get_text())

        with patch.object(reporting, "save_figure", side_effect=inspect):
            reporting.plot_load_history(times, loads, .015, Path("load.png"), "Load", "N")

    def test_thrust_threshold_is_used_in_summary_but_not_drawn(self):
        times = np.linspace(0, .03, 301)
        loads = 2 + np.where(np.arange(times.size) % 2, .008, -.008)
        summary = reporting.evaluate_thrust_convergence(times, loads, .015)
        self.assertEqual(summary["threshold"], .005)
        self.assertTrue(summary["passed"])

        def inspect(figure, path):
            if path.name == "force_convergence.png":
                ax = figure.axes[0]
                self.assertEqual(len(ax.lines), 1)
                self.assertEqual(ax.get_title(loc="right"), "")
                self.assertIsNone(ax.get_legend())
                figure.canvas.draw()

        with patch.object(reporting, "save_figure", side_effect=inspect):
            reporting.create_force_plots(times, loads, Path("."), .015)

    def test_residual_legend_stays_below_the_data_region(self):
        values = pd.DataFrame({"Time": [0, .01, .02], "p": [1e-2, 1e-3, 1e-4]})

        def inspect(figure, _):
            figure.canvas.draw()
            ax = figure.axes[0]
            legend = ax.get_legend().get_window_extent(figure.canvas.get_renderer())
            self.assertLess(legend.y1, ax.bbox.y0)

        with TemporaryDirectory() as directory, \
                patch.object(reporting, "read_residual_dataframe", return_value=values), \
                patch.object(reporting, "save_figure", side_effect=inspect):
            reporting.create_residual_plots(Path(directory) / "residuals.dat", Path(directory), .015, .02)

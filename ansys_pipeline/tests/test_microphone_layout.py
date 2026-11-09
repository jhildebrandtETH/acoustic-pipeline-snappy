import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from reportlab.pdfgen.canvas import Canvas

from tools.microphone_layout import append_microphone_layout, microphone_geometry
from tools.visualization import visualization_settings


class MicrophoneLayoutTests(unittest.TestCase):
    def test_saved_nonplanar_array_is_drawn_even_when_directivity_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            report = case / "report"
            report.mkdir()
            points = np.array([[-1.887, 0, .844], [1.4, .2, -.3], [.2, -1.2, .9], [.3, .7, 1.2]])
            np.savez_compressed(report / "acoustic-observers.npz", rpm=4000,
                                channel_ids=["1", "2", "5", "9"], observer_m=points)
            np.savez_compressed(report / "acoustic-pressure.npz", rpm=4000, observer_m=points[0])
            (report / "acoustic-directivity.json").write_text(json.dumps({
                "status": "unavailable", "detail": "No usable directivity plane",
            }))
            layout = microphone_geometry(case, 4000)
            self.assertEqual(layout["selected"], "1")
            np.testing.assert_array_equal(layout["positions"], points)
            self.assertIsNone(visualization_settings(case, 4000, "permeable")["observer_m"])
            pdf = Canvas(str(report / "test-report.pdf"))
            append_microphone_layout(pdf, case, 4000)
            pdf.save()
            with (report / "microphone-layout.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["channel"] for row in rows], ["1", "2", "5", "9"])
            self.assertEqual([row["used_for_spectrum"] for row in rows], ["True", "False", "False", "False"])
            self.assertTrue((report / "microphone-layout.pdf").read_bytes().startswith(b"%PDF"))
            self.assertTrue((report / "microphone-layout.png").is_file())

    def test_older_single_observer_uses_recorded_array_and_keeps_distinct_observer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "10x7e_4000RPM_case"
            report = case / "report"
            report.mkdir(parents=True)
            parameters = case / "Parameters"
            parameters.mkdir()
            points = {"1": [1, 0, 0], "5": [0, 2, 1]}
            payload = dict(propeller_type="10x7e", targeted_rpm=4000,
                           sound={"channels": {k: {"channel_position": v} for k, v in points.items()}})
            (root / "10x7e_2024-11-28-10-00-00_RPM_4000.pkl").write_bytes(pickle.dumps(payload))
            config = {"measurements_directory": str(root), "microphone_channel": "1"}
            (parameters / "acousticComparison.json").write_text(json.dumps(config))
            np.savez_compressed(report / "acoustic-pressure.npz", rpm=4000, observer_m=[2, 0, 0])
            layout = microphone_geometry(case, 4000)
            self.assertEqual(layout["channels"], ["1", "5"])
            np.testing.assert_array_equal(layout["positions"], list(points.values()))
            np.testing.assert_array_equal(layout["observer"], [2, 0, 0])
            self.assertIsNone(layout["selected"])
            self.assertIsNone(microphone_geometry(case, 3000)["observer"])

    def test_comparison_only_keeps_actual_measured_and_simulated_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            report = case / "report"
            report.mkdir()
            (report / "acoustic-comparison.json").write_text(json.dumps({
                "status": "complete", "rpm": 4000, "microphone_channel": "3",
                "microphone_position_m": [0, 1, 0], "observer_m": [0, 1.1, 0],
                "coordinate_mapping": {"enabled": True, "experimental_rotation_center_m": [0, 0, .75]},
            }))
            layout = microphone_geometry(case, 4000)
            self.assertEqual(layout["channels"], ["3"])
            np.testing.assert_array_equal(layout["positions"], [[0, 1, 0]])
            np.testing.assert_array_equal(layout["observer"], [0, 1.1, 0])
            self.assertIsNone(layout["selected"])

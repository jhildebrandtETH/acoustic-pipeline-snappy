from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from acousticSolver.src.foamacoustics.io.openfoam import OpenFOAMSurfaceSeries


POINTS = ((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
FACES = ((0, 1, 2), (1, 3, 2))


def write_frame(root, name, point_order, face_order, location, *, points=POINTS, faces=FACES):
    directory = root / name
    directory.mkdir()
    point_rows = {point: row for row, point in enumerate(point_order)}
    coordinates = [" ".join(map(str, points[point])) for point in point_order]
    polygons = []
    for face_id in face_order:
        face = faces[face_id]
        if name != "0":
            face = face[1:] + face[:1]
        polygons.append("3 " + " ".join(str(point_rows[point]) for point in face))
    values = [100 * float(name) + 10 * index for index in (
        point_order if location == "POINT_DATA" else face_order
    )]
    count = len(values)
    (directory / "permeableSurface.vtk").write_text(
        "\n".join([
            "# vtk DataFile Version 2.0", "sample", "ASCII", "DATASET POLYDATA",
            "POINTS 4 float", *coordinates, "POLYGONS 2 8", *polygons,
            f"{location} {count}", "FIELD fields 2",
            f"p 1 {count} float", " ".join(map(str, values)),
            f"U 3 {count} float", *(f"{v + 1} {v + 2} {v + 3}" for v in values),
        ]) + "\n",
        encoding="ascii",
    )


class AcousticVtkOrderTests(unittest.TestCase):
    def test_base_sphere_in_study_reads_raw_vtk(self):
        import acoustic_propagation as stage

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / "acoustic-surfaces.json").write_text("{}")
            source = case / "postProcessing/writePermeableSurfaceFields"
            for index in range(9):
                folder = source / str(index * 0.01)
                folder.mkdir(parents=True)
                (folder / "permeableSurface.vtk").touch()
            with (
                patch.object(stage, "load_config", return_value={"microphone_channel": "1"}),
                patch.object(stage, "case_sample_interval", return_value=0.01),
                patch.object(stage, "create_reference_geometry_vtk_series") as rewrite,
                patch.object(stage.F1ASolver, "from_openfoam_vtk", side_effect=RuntimeError("loaded")) as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "loaded"):
                    stage.run_acoustic_solver("permeable", case, 4000)
            rewrite.assert_not_called()
            self.assertEqual(load.call_args.args[0], source)

    def test_reorders_pressure_and_velocity_for_every_frame_and_cache(self):
        for location, pressure in [
            ("CELL_DATA", [0, 10]),
            ("POINT_DATA", [10, 20]),
        ]:
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "frames"
                root.mkdir()
                write_frame(root, "0", (0, 1, 2, 3), (0, 1), location)
                write_frame(root, "1", (2, 0, 3, 1), (1, 0), location)
                write_frame(root, "2", (3, 1, 0, 2), (0, 1), location)
                write_frame(root, "3", (1, 2, 3, 0), (1, 0), location)
                options = dict(
                    surface_file="permeableSurface.vtk",
                    pressure_units="pa",
                    cache_dir=Path(tmp) / "cache",
                )
                series = OpenFOAMSurfaceSeries(root, **options)
                for stage in ("raw", "cached"):
                    if stage == "cached":
                        series.build_cache(workers=2)
                        series = OpenFOAMSurfaceSeries(root, **options)
                        self.assertTrue(series.cache_enabled)
                    frames = list(series.iter_frames(workers=2))
                    self.assertEqual(len(frames), 4)
                    for time, frame in enumerate(frames):
                        expected = torch.tensor(pressure, dtype=torch.float64) + 100 * time
                        torch.testing.assert_close(frame.pressure_pa, expected)
                        torch.testing.assert_close(
                            frame.fluid_velocity_m_s,
                            expected[:, None] + torch.arange(1, 4),
                        )

    def test_rejects_coordinate_changes_reversed_winding_and_duplicate_faces(self):
        for points, faces, message in [
            (POINTS[:-1] + ((2, 1, 1),), FACES, "Point coordinates differ"),
            (POINTS, (FACES[0], FACES[1][::-1]), "Face connectivity or winding differs"),
            (POINTS, (FACES[0], FACES[0]), "Face connectivity or winding differs"),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_frame(root, "0", (0, 1, 2, 3), (0, 1), "CELL_DATA")
                write_frame(
                    root, "1", (2, 0, 3, 1), (1, 0), "CELL_DATA",
                    points=points, faces=faces,
                )
                series = OpenFOAMSurfaceSeries(root, surface_file="permeableSurface.vtk")
                with self.assertRaisesRegex(ValueError, message):
                    list(series.iter_frames())

import math
from pathlib import Path
import tempfile
import unittest

import torch

from acoustic_propagation import _impermeable_ami_rotation, _impermeable_pressure_settings
from acousticSolver.src.foamacoustics import F1ASolver, OpenFOAMSurfaceSeries


POINTS = ((0.02, 0, 0), (0.04, 0, 0), (0.02, 0.02, 0), (0.02, 0, 0.02))
# Deliberately inward-facing.
FACES = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
RPM = 4000
OMEGA = RPM * math.tau / 60


def write_frame(root, time, *, reorder=False, rotate=True, offset=0.0, varied=False, points=POINTS):
    angle = OMEGA * time if rotate else 0.0
    c, s = math.cos(angle), math.sin(angle)
    point_order = (2, 0, 3, 1) if reorder else tuple(range(4))
    face_order = (3, 1, 0, 2) if reorder else tuple(range(4))
    rows = {point: row for row, point in enumerate(point_order)}
    coordinates = []
    for index in point_order:
        x, y, z = points[index]
        coordinates.append(f"{c*x + s*z:.9g} {y:.9g} {-s*x + c*z:.9g}")
    polygons = ["3 " + " ".join(str(rows[vertex]) for vertex in FACES[face])
                for face in face_order]
    pressure = " ".join(str(offset + (face if varied else 0)) for face in face_order)
    directory = root / str(time)
    directory.mkdir()
    (directory / "propeller.vtk").write_text("\n".join([
        "# vtk DataFile Version 2.0", "blade patch", "ASCII", "DATASET POLYDATA",
        "POINTS 4 float", *coordinates, "POLYGONS 4 16", *polygons,
        "CELL_DATA 4", "FIELD fields 1", "p 1 4 float", pressure, "",
    ]), encoding="ascii")


class ImpermeablePropagationTests(unittest.TestCase):
    def test_pressure_settings_resolve_fixed_value_and_total_pressure(self):
        templates = Path(__file__).resolve().parents[1] / "CoreTemplates/AMI"
        for case, expected in (("kOmegaSST/wall-functions", 124123.125),
                               ("DES/wall-resolved", 0.0)):
            with self.subTest(case=case):
                settings = _impermeable_pressure_settings(templates / case, 1.225)
                self.assertEqual(settings["pressure_units"], "kinematic")
                self.assertAlmostEqual(settings["pressure_reference_pa"], expected)

    def test_ami_rotation_comes_from_saved_motion_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            self.assertIsNone(_impermeable_ami_rotation(case))
            (case / "Parameters").mkdir()
            (case / "Parameters/rotational_parameters.cpp").write_text("omega_val 400;")
            (case / "constant").mkdir()
            template = Path(__file__).resolve().parents[1] / "CoreTemplates/AMI/DES/wall-resolved"
            relative = "constant/dynamicMeshDict"
            (case / relative).write_bytes((template / relative).read_bytes().replace(
                b"origin  (0 0 0)", b"origin  (0.1 0 0.2)",
            ).replace(b"axis    (0 1 0)", b"axis    (0 -1 0)"))
            self.assertEqual(_impermeable_ami_rotation(case), (-400.0, (0.1, 0.0, 0.2)))

    def test_stationary_faces_keep_their_gauge_pressure_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / "0").mkdir()
            (case / "0/p").write_text(
                "dimensions [1 -1 -2 0 0 0 0]; "
                "boundaryField { outlet { type fixedValue; value uniform 100000; } }"
            )
            frames = case / "postProcessing/writePatchFields"
            frames.mkdir(parents=True)
            write_frame(frames, 0.0, rotate=False, offset=100000, varied=True)
            write_frame(frames, 0.001, reorder=True, rotate=False, offset=100000, varied=True)
            series = OpenFOAMSurfaceSeries(
                frames, surface_file="propeller.vtk", velocity_field=None, closed_surface=True,
                **_impermeable_pressure_settings(case, 1.2),
            )
            for frame in series.iter_frames():
                torch.testing.assert_close(frame.pressure_pa, torch.arange(4, dtype=torch.float64))

    def test_reorders_rotating_faces_and_recovers_outward_normals_after_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_frame(root, 0.0, varied=True)
            write_frame(root, 0.001, reorder=True, varied=True)
            options = dict(surface_file="propeller.vtk", velocity_field=None, pressure_units="pa",
                           closed_surface=True, geometry_rotation_y=(OMEGA, (0., 0., 0.)),
                           cache_dir=root / "cache")
            # Keep the cache outside the numeric sample directory.
            frames = root / "frames"
            frames.mkdir()
            for name in ("0.0", "0.001"):
                (root / name).rename(frames / name)
            series = OpenFOAMSurfaceSeries(frames, **options)
            for frame in series.iter_frames():
                torch.testing.assert_close(frame.pressure_pa, torch.arange(4, dtype=torch.float64))
            signed_volume = (series.face_centers_m * series.outward_area_vectors_m2).sum() / 3
            self.assertGreater(float(signed_volume), 0)
            self.assertLess(float(series.outward_area_vectors_m2.sum(dim=0).norm()), 1e-15)
            series.build_cache()
            cached = OpenFOAMSurfaceSeries(frames, **options)
            self.assertTrue(cached.cache_enabled)
            torch.testing.assert_close(cached.face_normals, series.face_normals)
            for frame in cached.iter_frames():
                torch.testing.assert_close(frame.pressure_pa, torch.arange(4, dtype=torch.float64))

    def test_uniform_reference_pressure_has_zero_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(21):
                write_frame(root, index / 10000, reorder=index % 2 == 1, offset=101325)
            series = OpenFOAMSurfaceSeries(
                root, surface_file="propeller.vtk", velocity_field=None, closed_surface=True,
                pressure_reference_pa=101325 * 1.225,
                geometry_rotation_y=(OMEGA, (0., 0., 0.)),
            )
            solver = F1ASolver(series, rpm=RPM, permeable=False, moving_surface=True,
                               omega_rad_s=[0., OMEGA, 0.], device="cpu")
            result = solver.predict([-1.887, 0., 0.094])
            self.assertEqual(torch.count_nonzero(result.loading_pressure_pa).item(), 0)
            self.assertGreater(torch.count_nonzero(result.thickness_pressure_pa).item(), 0)

    def test_rejects_frozen_geometry_when_rotation_is_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_frame(root, 0.0)
            write_frame(root, 0.001, rotate=False)
            series = OpenFOAMSurfaceSeries(
                root, surface_file="propeller.vtk", velocity_field=None, closed_surface=True,
                geometry_rotation_y=(OMEGA, (0., 0., 0.)),
            )
            with self.assertRaisesRegex(ValueError, r"0\.001.*propeller\.vtk:.*no unique match"):
                list(series.iter_frames())

    def test_reports_frame_with_ambiguous_rotating_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = (POINTS[0], (0.0200001, 0, 0), *POINTS[2:])
            write_frame(root, 0.0, points=points)
            write_frame(root, 0.001, points=points, reorder=True)
            series = OpenFOAMSurfaceSeries(
                root, surface_file="propeller.vtk", velocity_field=None, closed_surface=True,
                geometry_rotation_y=(OMEGA, (0., 0., 0.)),
            )
            with self.assertRaisesRegex(ValueError, r"0\.001.*propeller\.vtk:.*no unique match"):
                list(series.iter_frames())

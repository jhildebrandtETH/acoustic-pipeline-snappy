"""Sample OpenFOAM volume fields with VTK, without an OpenFOAM runtime."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import numpy as np
import pyvista as pv
import torch
from tqdm.auto import tqdm
from vtkmodules.vtkCommonCore import reference, vtkLogger, vtkVersion
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from ..io.openfoam import OpenFOAMSurfaceSeries, PressureUnits, _read_surface_vtk
from ..io.surface import DTypeName, resolve_dtype
from ..utils.validation import finite_float, positive_float, positive_int
from .geometry import CylinderSurface, SphereSurface, TriangleMesh


SurfaceSpec = SphereSurface | CylinderSurface

_SURFACE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_REGION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIME_VALUE_RE = re.compile(r"^\s*value\s+([^;\s]+)\s*;", re.MULTILINE)
_SAMPLING_SCHEMA_VERSION = 3
_POINT_WEIGHT_CHUNK_CELLS = 50_000
_CELL_GEOMETRY_CHUNK_CELLS = 50_000
_INTERPOLATION_PLAN_CACHE_SIZE = 4
_VTK_GLOBAL_LOCK = threading.RLock()
_KNOWN_VTK_FIELD_MISMATCH = "Number of cells/points in mesh and field do not match"
_FOAM_VECTOR_TRANSLATION = bytes.maketrans(b"()", b"  ")

# Linear VTK volume cells have a fixed local face topology and can therefore be
# evaluated in batches. OpenFOAM's reader normally produces hexahedra, wedges,
# and generic polyhedra, but supporting the other linear primitives costs
# nothing and keeps synthetic/test meshes on the same code path.
_FIXED_LINEAR_CELL_TYPES = frozenset((10, 11, 12, 13, 14, 15, 16))


@dataclass(frozen=True, slots=True)
class _CaseTime:
    """A VTK reader time and its full-precision physical time."""

    name: str
    reader_value: float
    physical_value: float
    directory: Path

    @property
    def output_name(self) -> str:
        value = 0.0 if self.physical_value == 0.0 else self.physical_value
        return f"{value:.17g}"


def _validate_surface_name(value: str) -> str:
    value = str(value)
    if _SURFACE_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            "surface_name must start with a letter and contain only letters, "
            "digits, underscores, or hyphens."
        )
    return value


def _validate_field_name(value: str, name: str) -> str:
    value = str(value)
    if _FIELD_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be an OpenFOAM field name without whitespace or path "
            "separators."
        )
    return value


def _validate_region_name(value: str) -> str:
    value = str(value)
    if _REGION_NAME_RE.fullmatch(value) is None:
        raise ValueError("region must be an OpenFOAM word without path separators.")
    return value


def _foam_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    return compressed if compressed.is_file() else None


def _mesh_dir(case_dir: Path, region: str | None) -> Path:
    path = case_dir / "constant"
    if region is not None:
        path /= region
    return path / "polyMesh"


def _validate_case(case_dir: Path, region: str | None) -> None:
    if not case_dir.is_dir():
        raise FileNotFoundError(f"OpenFOAM case directory does not exist: {case_dir}")

    mesh_dir = _mesh_dir(case_dir, region)
    required = ("points", "faces", "owner", "neighbour", "boundary")
    missing = [name for name in required if _foam_file(mesh_dir / name) is None]
    if missing:
        flattened = case_dir / "polyMesh"
        hint = ""
        if flattened.is_dir():
            hint = (
                f" A mesh exists at {flattened}, but VTK expects the OpenFOAM "
                f"layout at {mesh_dir}."
            )
        raise FileNotFoundError(
            f"Incomplete OpenFOAM mesh at {mesh_dir}; missing "
            f"{', '.join(missing)}.{hint}"
        )


def _time_metadata_files(time_dir: Path, region: str | None) -> tuple[Path, ...]:
    candidates = [time_dir / "uniform" / "time"]
    if region is not None:
        candidates.insert(0, time_dir / region / "uniform" / "time")
    return tuple(
        found
        for candidate in candidates
        if (found := _foam_file(candidate)) is not None
    )


def _physical_time(case_dir: Path, time_name: str, region: str | None) -> float:
    for path in _time_metadata_files(case_dir / time_name, region):
        if path.suffix == ".gz":
            with gzip.open(
                path,
                mode="rt",
                encoding="utf-8",
                errors="replace",
            ) as stream:
                contents = stream.read()
        else:
            contents = path.read_text(encoding="utf-8", errors="replace")
        match = _TIME_VALUE_RE.search(contents)
        if match is None:
            continue
        try:
            value = float(match.group(1))
        except ValueError as error:
            raise ValueError(f"{path}: invalid physical time value.") from error
        if not math.isfinite(value):
            raise ValueError(f"{path}: physical time value must be finite.")
        return value
    try:
        value = float(time_name)
    except ValueError as error:
        raise ValueError(
            f"VTK returned a non-numeric time name {time_name!r}."
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"VTK returned a non-finite time name {time_name!r}.")
    return value


def _number(token: str, selection: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(
            f"Invalid time selection {selection!r}; expected numbers, comma-separated "
            "numbers, or inclusive ranges such as '0.10:0.20'."
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Time selection {selection!r} contains a non-finite value.")
    return value


def _time_token_indices(
    available: Sequence[_CaseTime],
    token: str,
    selection: str,
) -> set[int]:
    """Resolve one exact-time or inclusive-range selection token."""

    if ":" in token:
        if token.count(":") != 1:
            raise ValueError(f"Invalid time range {token!r}.")
        left, right = token.split(":")
        lower = -math.inf if not left else _number(left, selection)
        upper = math.inf if not right else _number(right, selection)
        if lower > upper:
            raise ValueError(f"Time range {token!r} has decreasing bounds.")
        return {
            index
            for index, step in enumerate(available)
            if lower <= step.reader_value <= upper
        }

    requested = _number(token, selection)
    matches = {
        index
        for index, step in enumerate(available)
        if step.name == token
        or math.isclose(
            step.reader_value,
            requested,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    }
    if not matches:
        raise ValueError(f"Requested OpenFOAM time {token!r} was not found.")
    return matches


def _select_times(
    available: Sequence[_CaseTime],
    *,
    time_selection: str | None,
    max_times: int | None,
) -> tuple[_CaseTime, ...]:
    if not available:
        raise ValueError("VTK found no OpenFOAM time directories.")

    if time_selection is None:
        selected = list(available)
    else:
        selection = str(time_selection).strip()
        if not selection:
            raise ValueError("time_selection cannot be empty.")
        chosen: set[int] = set()
        tokens = [token for token in selection.replace(",", " ").split() if token]
        for token in tokens:
            chosen.update(_time_token_indices(available, token, selection))
        selected = [step for index, step in enumerate(available) if index in chosen]

    if max_times is not None:
        limit = positive_int(max_times, "max_times")
        if limit < 2:
            raise ValueError("max_times must be at least two.")
        selected = selected[:limit]
    if len(selected) < 2:
        raise ValueError(
            "At least two OpenFOAM times are required for an F1A surface series."
        )
    physical = [step.physical_value for step in selected]
    if any(right <= left for left, right in zip(physical, physical[1:], strict=False)):
        raise ValueError(
            "Selected physical times must be finite, unique, and increasing."
        )
    return tuple(selected)


@contextmanager
def _capture_vtk_diagnostics() -> Iterator[pv.VtkErrorCatcher]:
    """Capture VTK diagnostics without leaking global logger changes."""

    with _VTK_GLOBAL_LOCK:
        previous_verbosity = vtkLogger.GetCurrentVerbosityCutoff()
        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
        try:
            with pv.VtkErrorCatcher(
                raise_errors=False,
                send_to_logging=False,
                emit_warnings=False,
            ) as catcher:
                yield catcher
        finally:
            vtkLogger.SetStderrVerbosity(previous_verbosity)


def _unexpected_vtk_errors(catcher: pv.VtkErrorCatcher) -> list[str]:
    errors = [str(event) for event in catcher.error_events]
    return [message for message in errors if _KNOWN_VTK_FIELD_MISMATCH not in message]


def _patch_components(name: str) -> tuple[str, ...]:
    return tuple(component for component in name.strip("/").split("/") if component)


def _select_internal_patch(reader: vtkOpenFOAMReader, region: str | None) -> str:
    names = [
        reader.GetPatchArrayName(index)
        for index in range(reader.GetNumberOfPatchArrays())
    ]
    candidates: list[str] = []
    for name in names:
        components = _patch_components(name)
        if components[-1:] != ("internalMesh",):
            continue
        if region is None or components[-2:] == (region, "internalMesh"):
            candidates.append(name)
    if len(candidates) != 1:
        detail = ", ".join(repr(name) for name in names) or "none"
        if region is None and len(candidates) > 1:
            raise ValueError(
                "The case contains multiple internal regions; pass region=... to "
                f"select one. VTK patches: {detail}."
            )
        raise ValueError(
            f"Could not identify one VTK internalMesh patch for region {region!r}. "
            f"VTK patches: {detail}."
        )
    return candidates[0]


def _time_has_topology(time_dir: Path, region: str | None) -> bool:
    roots = [time_dir / "polyMesh"]
    if region is not None:
        roots.insert(0, time_dir / region / "polyMesh")
    topology = ("faces", "owner", "neighbour", "boundary")
    return any(
        _foam_file(root / name) is not None for root in roots for name in topology
    )


def _time_mesh_points_file(time_dir: Path, region: str | None) -> Path | None:
    roots = [time_dir / "polyMesh"]
    if region is not None:
        roots.insert(0, time_dir / region / "polyMesh")
    return next(
        (
            points_file
            for root in roots
            if (points_file := _foam_file(root / "points")) is not None
        ),
        None,
    )


def _dataset_leaves(
    data: pv.DataObject,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], pv.DataSet]]:
    if isinstance(data, pv.MultiBlock):
        for index in range(data.n_blocks):
            child = data[index]
            if child is None:
                continue
            name = data.get_block_name(index) or str(index)
            yield from _dataset_leaves(child, (*path, name))
    elif isinstance(data, pv.DataSet) and data.n_cells:
        yield path, data


class _VTKOpenFOAMReader:
    """Small validated wrapper around VTK's native OpenFOAM reader."""

    def __init__(
        self,
        case_dir: Path,
        *,
        pressure_field: str,
        velocity_field: str,
        region: str | None,
    ) -> None:
        self.case_dir = case_dir
        self.pressure_field = pressure_field
        self.velocity_field = velocity_field
        self.region = region
        reader = vtkOpenFOAMReader()
        reader.SetListTimeStepsByControlDict(False)
        reader.SetSkipZeroTime(False)
        reader.SetCreateCellToPoint(False)
        reader.SetSizeAverageCellToPoint(False)
        reader.SetCacheMesh(True)
        reader.SetReadZones(False)
        reader.SetAddDimensionsToArrayNames(False)
        if hasattr(reader, "SetCopyDataToCellZones"):
            reader.SetCopyDataToCellZones(False)
        if hasattr(reader, "SetSequentialProcessing"):
            reader.SetSequentialProcessing(False)
        # vtkOpenFOAMReader uses the parent directory; the marker need not exist.
        reader.SetFileName(str(case_dir / "foamacoustics.foam"))
        with _capture_vtk_diagnostics() as diagnostics:
            reader.UpdateInformation()
        unexpected = _unexpected_vtk_errors(diagnostics)
        if unexpected:
            raise RuntimeError(
                "VTK could not inspect the OpenFOAM case:\n" + "\n".join(unexpected)
            )

        self.internal_patch = _select_internal_patch(reader, region)
        cell_arrays = {
            reader.GetCellArrayName(index)
            for index in range(reader.GetNumberOfCellArrays())
        }
        missing = [
            field
            for field in (pressure_field, velocity_field)
            if field not in cell_arrays
        ]
        if missing:
            raise ValueError(
                "VTK could not find required cell field(s) "
                f"{', '.join(repr(field) for field in missing)}. Available fields: "
                f"{', '.join(sorted(cell_arrays)) or 'none'}."
            )

        names = reader.GetTimeNames()
        values = reader.GetTimeValues()
        if names.GetNumberOfValues() != values.GetNumberOfValues():
            raise RuntimeError("VTK returned inconsistent OpenFOAM time metadata.")
        times = []
        for index in range(values.GetNumberOfValues()):
            name = str(names.GetValue(index))
            reader_value = float(values.GetValue(index))
            directory = case_dir / name
            times.append(
                _CaseTime(
                    name=name,
                    reader_value=reader_value,
                    physical_value=_physical_time(case_dir, name, region),
                    directory=directory,
                )
            )
        times.sort(key=lambda step: (step.reader_value, step.name))
        self.times = tuple(times)
        self.mesh_topology_static = not any(
            _time_has_topology(step.directory, region) for step in self.times
        )
        self.mesh_points_static = not any(
            _time_mesh_points_file(step.directory, region) is not None
            for step in self.times
        )
        reader.SetCacheMesh(self.mesh_topology_static)

        reader.DisableAllPatchArrays()
        reader.SetPatchArrayStatus(self.internal_patch, 1)
        reader.DisableAllCellArrays()
        reader.SetCellArrayStatus(pressure_field, 1)
        reader.SetCellArrayStatus(velocity_field, 1)
        reader.DisableAllPointArrays()
        self.reader = reader
        self.diagnostics: list[str] = []

    def read(self, step: _CaseTime) -> pv.UnstructuredGrid:
        with _capture_vtk_diagnostics() as diagnostics:
            self.reader.UpdateTimeStep(step.reader_value)
        errors = [str(event) for event in diagnostics.error_events]
        self.diagnostics.extend(errors)
        unexpected = [
            message for message in errors if _KNOWN_VTK_FIELD_MISMATCH not in message
        ]
        if unexpected:
            raise RuntimeError(
                f"VTK failed while reading OpenFOAM time {step.name!r}:\n"
                + "\n".join(unexpected)
            )

        wrapped = pv.wrap(self.reader.GetOutputDataObject(0))
        candidates = [
            (path, leaf)
            for path, leaf in _dataset_leaves(wrapped)
            if isinstance(leaf, pv.UnstructuredGrid)
        ]
        if len(candidates) != 1:
            paths = ["/".join(path) for path, _ in candidates]
            raise RuntimeError(
                f"VTK time {step.name!r} produced {len(candidates)} internal "
                f"unstructured meshes; expected one. Found: {paths or 'none'}."
            )
        _, grid = candidates[0]
        self._validate_fields(grid, step)
        return grid

    def _validate_fields(self, grid: pv.UnstructuredGrid, step: _CaseTime) -> None:
        for field, components in (
            (self.pressure_field, 1),
            (self.velocity_field, 3),
        ):
            if field not in grid.cell_data:
                raise ValueError(
                    f"OpenFOAM time {step.name!r} does not provide cell field "
                    f"{field!r} on the internal mesh."
                )
            values = np.asarray(grid.cell_data[field])
            expected = (
                (grid.n_cells,) if components == 1 else (grid.n_cells, components)
            )
            if values.shape == (grid.n_cells, 1) and components == 1:
                values = values[:, 0]
            if values.shape != expected:
                raise ValueError(
                    f"OpenFOAM field {field!r} at time {step.name!r} has shape "
                    f"{values.shape}; expected {expected}."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"OpenFOAM field {field!r} at time {step.name!r} contains "
                    "non-finite values."
                )


def _cell_fields(
    grid: pv.UnstructuredGrid,
    pressure_field: str,
    velocity_field: str,
) -> np.ndarray:
    pressure = np.asarray(grid.cell_data[pressure_field], dtype=np.float64).reshape(-1)
    velocity = np.asarray(grid.cell_data[velocity_field], dtype=np.float64)
    return np.column_stack((pressure, velocity))


def _selected_cell_fields(
    grid: pv.UnstructuredGrid,
    cell_ids: np.ndarray,
    pressure_field: str,
    velocity_field: str,
) -> np.ndarray:
    """Copy only the cell values required by a prepared interpolation plan."""

    pressure = np.asarray(grid.cell_data[pressure_field]).reshape(-1)[cell_ids]
    velocity = np.asarray(grid.cell_data[velocity_field])[cell_ids]
    return np.column_stack((pressure, velocity)).astype(np.float64, copy=False)


def _inverse_distance_point_fields(
    grid: pv.UnstructuredGrid,
    cell_centers: np.ndarray,
    cell_values: np.ndarray,
) -> np.ndarray:
    """Approximate OpenFOAM volPointInterpolation on internal mesh points."""

    points = np.asarray(grid.points, dtype=np.float64)
    connectivity = np.asarray(grid.cell_connectivity, dtype=np.int64)
    offsets = np.asarray(grid.offset, dtype=np.int64)
    counts = np.diff(offsets)
    if len(counts) != grid.n_cells or offsets[-1] != len(connectivity):
        raise RuntimeError("VTK returned inconsistent cell connectivity.")

    denominator = np.zeros(grid.n_points, dtype=np.float64)
    numerator = np.zeros((grid.n_points, cell_values.shape[1]), dtype=np.float64)
    bounds = np.asarray(grid.bounds, dtype=np.float64)
    scale = max(
        float(np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]])),
        1.0,
    )
    minimum_distance = np.finfo(np.float64).eps * scale
    for first in range(0, grid.n_cells, _POINT_WEIGHT_CHUNK_CELLS):
        stop = min(first + _POINT_WEIGHT_CHUNK_CELLS, grid.n_cells)
        cell_ids = np.repeat(np.arange(first, stop, dtype=np.int64), counts[first:stop])
        point_ids = connectivity[offsets[first] : offsets[stop]]
        distances = np.linalg.norm(
            points[point_ids] - cell_centers[cell_ids],
            axis=1,
        )
        weights = 1.0 / np.maximum(distances, minimum_distance)
        denominator += np.bincount(point_ids, weights, minlength=grid.n_points)
        weighted_values = weights[:, None] * cell_values[cell_ids]
        for component in range(cell_values.shape[1]):
            numerator[:, component] += np.bincount(
                point_ids,
                weighted_values[:, component],
                minlength=grid.n_points,
            )
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise RuntimeError(
            "Could not construct finite cell-to-point interpolation weights."
        )
    result = numerator / denominator[:, None]
    if not np.isfinite(result).all():
        raise RuntimeError("Cell-to-point interpolation produced non-finite values.")
    return result


def _ordered_face_point_ids(face: pv.Cell, *, cell_id: int) -> np.ndarray:
    """Return a cyclic ordering for a VTK face's global point ids.

    VTK polygon faces already expose cyclic point ids, but ``VTK_PIXEL`` faces
    (used by ``VTK_VOXEL`` cells) use image ordering instead. Reconstructing the
    cycle from the face edges handles both without cell-type-specific geometry.
    """

    point_ids = np.asarray(face.point_ids, dtype=np.int64).reshape(-1)
    if len(point_ids) < 3 or len(np.unique(point_ids)) != len(point_ids):
        raise RuntimeError(f"VTK cell {cell_id} contains an invalid face.")

    edge_pairs: list[tuple[int, int]] = []
    for edge in face.edges:
        edge_ids = np.asarray(edge.point_ids, dtype=np.int64).reshape(-1)
        if len(edge_ids) != 2:
            raise RuntimeError(
                f"VTK cell {cell_id} contains a non-linear face edge, which is "
                "not supported for OpenFOAM cellPoint interpolation."
            )
        edge_pairs.append((int(edge_ids[0]), int(edge_ids[1])))

    edge_keys = {tuple(sorted(edge)) for edge in edge_pairs}
    if len(edge_pairs) == len(point_ids) and all(
        tuple(
            sorted(
                (
                    int(point_ids[index]),
                    int(point_ids[(index + 1) % len(point_ids)]),
                )
            )
        )
        in edge_keys
        for index in range(len(point_ids))
    ):
        return point_ids.copy()

    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edge_pairs:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if set(adjacency) != set(int(value) for value in point_ids) or any(
        len(neighbours) != 2 for neighbours in adjacency.values()
    ):
        raise RuntimeError(f"VTK cell {cell_id} contains a non-cyclic face.")

    cycle = [int(point_ids[0])]
    previous: int | None = None
    current = cycle[0]
    for _ in range(1, len(point_ids)):
        candidates = [value for value in adjacency[current] if value != previous]
        if not candidates:
            raise RuntimeError(f"VTK cell {cell_id} contains a broken face cycle.")
        following = candidates[0]
        if following == cycle[0] and len(cycle) < len(point_ids):
            if len(candidates) == 1:
                raise RuntimeError(f"VTK cell {cell_id} contains a broken face cycle.")
            following = candidates[1]
        cycle.append(following)
        previous, current = current, following
    if cycle[-1] not in adjacency[cycle[0]] or set(cycle) != set(adjacency):
        raise RuntimeError(f"VTK cell {cell_id} contains a broken face cycle.")
    return np.asarray(cycle, dtype=np.int64)


def _face_orientation_signs(
    faces: Sequence[np.ndarray],
    *,
    cell_id: int,
) -> np.ndarray:
    """Return signs that make adjacent face cycles traverse edges oppositely."""

    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % len(face)])
            key = (left, right) if left < right else (right, left)
            direction = 1 if (left, right) == key else -1
            edge_uses[key].append((face_index, direction))
    malformed = [
        edge
        for edge, uses in edge_uses.items()
        if len(uses) != 2 or uses[0][0] == uses[1][0]
    ]
    if malformed:
        raise RuntimeError(
            f"VTK cell {cell_id} does not have a closed two-manifold face shell."
        )

    flips = np.zeros(len(faces), dtype=np.int8)
    flips[0] = 1
    queue: deque[int] = deque((0,))
    while queue:
        face_index = queue.popleft()
        face = faces[face_index]
        for index, left_value in enumerate(face):
            left = int(left_value)
            right = int(face[(index + 1) % len(face)])
            key = (left, right) if left < right else (right, left)
            uses = edge_uses[key]
            current_direction = next(
                direction for owner, direction in uses if owner == face_index
            )
            other_index, other_direction = next(
                (owner, direction) for owner, direction in uses if owner != face_index
            )
            required = -current_direction * other_direction * int(flips[face_index])
            if flips[other_index] == 0:
                flips[other_index] = required
                queue.append(other_index)
            elif flips[other_index] != required:
                raise RuntimeError(
                    f"VTK cell {cell_id} has inconsistent face connectivity."
                )
    if np.any(flips == 0):
        raise RuntimeError(f"VTK cell {cell_id} has a disconnected face shell.")
    return flips


def _coherently_orient_face_cycles(
    faces: Sequence[np.ndarray],
    *,
    cell_id: int,
) -> tuple[np.ndarray, ...]:
    """Orient a closed cell shell so adjacent faces traverse edges oppositely."""

    flips = _face_orientation_signs(faces, cell_id=cell_id)
    return tuple(
        face.copy() if flip > 0 else face[::-1].copy()
        for face, flip in zip(faces, flips, strict=True)
    )


def _cell_face_cycles(
    grid: pv.UnstructuredGrid,
    cell_id: int,
) -> tuple[np.ndarray, ...]:
    cell = grid.get_cell(cell_id)
    faces = cell.faces
    if faces is None or len(faces) == 0:
        raise RuntimeError(f"VTK cell {cell_id} has no faces for interpolation.")
    cycles = tuple(_ordered_face_point_ids(face, cell_id=cell_id) for face in faces)
    return _coherently_orient_face_cycles(cycles, cell_id=cell_id)


def _openfoam_face_geometry_batch(
    face_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate OpenFOAM face centres and area vectors for equal-sized faces."""

    if face_points.ndim != 3 or face_points.shape[1] < 3 or face_points.shape[2] != 3:
        raise ValueError(
            "face_points must have shape (face_count, points_per_face, 3)."
        )
    if face_points.shape[1] == 3:
        centres = face_points.mean(axis=1)
        areas = 0.5 * np.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        )
        return centres, areas

    point_averages = face_points.mean(axis=1)
    following = np.roll(face_points, -1, axis=1)
    sub_areas = np.cross(
        following - face_points,
        point_averages[:, None, :] - face_points,
    )
    twice_areas = sub_areas.sum(axis=1)
    area_magnitudes = np.linalg.norm(twice_areas, axis=1)
    centres = point_averages.copy()
    usable = area_magnitudes > np.finfo(np.float64).tiny
    if np.any(usable):
        area_directions = twice_areas[usable] / area_magnitudes[usable, None]
        signed_sub_areas = np.einsum(
            "fpc,fc->fp",
            sub_areas[usable],
            area_directions,
        )
        centre_numerators = np.einsum(
            "fp,fpc->fc",
            signed_sub_areas,
            (face_points[usable] + following[usable] + point_averages[usable, None, :]),
        )
        centres[usable] = centre_numerators / (
            3.0 * signed_sub_areas.sum(axis=1, keepdims=True)
        )
    return centres, 0.5 * twice_areas


def _openfoam_cell_geometry_batch(
    cell_points: np.ndarray,
    local_faces: Sequence[np.ndarray],
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized OpenFOAM face-pyramid geometry for one fixed VTK cell type."""

    face_centres = []
    face_areas = []
    for face in local_faces:
        centres, areas = _openfoam_face_geometry_batch(cell_points[:, face, :])
        face_centres.append(centres)
        face_areas.append(areas)
    centres_by_face = np.stack(face_centres, axis=1)
    areas_by_face = np.stack(face_areas, axis=1)
    estimates = centres_by_face.mean(axis=1)
    pyramid_three_volumes = np.einsum(
        "bfc,bfc->bf",
        areas_by_face,
        centres_by_face - estimates[:, None, :],
    )
    three_volumes = pyramid_three_volumes.sum(axis=1)
    extents = np.max(np.ptp(cell_points, axis=1), axis=1)
    minimum_three_volumes = np.finfo(np.float64).eps * np.maximum(
        extents**3,
        np.finfo(np.float64).tiny,
    )
    valid = np.isfinite(three_volumes) & (np.abs(three_volumes) > minimum_three_volumes)
    if not np.all(valid):
        bad = int(cell_ids[int(np.flatnonzero(~valid)[0])])
        raise RuntimeError(f"VTK cell {bad} has zero or non-finite volume.")
    pyramid_centres = 0.75 * centres_by_face + 0.25 * estimates[:, None, :]
    cell_centres = (
        np.einsum(
            "bf,bfc->bc",
            pyramid_three_volumes,
            pyramid_centres,
        )
        / three_volumes[:, None]
    )
    volumes = np.abs(three_volumes) / 3.0
    if not np.isfinite(cell_centres).all():
        first_bad = int(np.flatnonzero(~np.isfinite(cell_centres).all(axis=1))[0])
        bad = int(cell_ids[first_bad])
        raise RuntimeError(f"VTK cell {bad} has a non-finite volume centroid.")
    return cell_centres, volumes


def _openfoam_cell_geometry_from_faces(
    mesh_points: np.ndarray,
    faces: Sequence[np.ndarray],
    *,
    cell_id: int,
    bincount_reduction: bool = False,
) -> tuple[np.ndarray, float]:
    centres_by_face = []
    areas_by_face = []
    for face in faces:
        centres, areas = _openfoam_face_geometry_batch(mesh_points[face][None, ...])
        centres_by_face.append(centres[0])
        areas_by_face.append(areas[0])
    face_centres = np.asarray(centres_by_face, dtype=np.float64)
    face_areas = np.asarray(areas_by_face, dtype=np.float64)
    owners = np.zeros(len(face_centres), dtype=np.int64)
    estimate = (
        np.asarray(
            [
                np.bincount(
                    owners,
                    weights=face_centres[:, component],
                    minlength=1,
                )[0]
                for component in range(3)
            ]
        )
        / len(face_centres)
        if bincount_reduction
        else face_centres.mean(axis=0)
    )
    pyramid_three_volumes = np.einsum(
        "fc,fc->f",
        face_areas,
        face_centres - estimate,
    )
    three_volume = float(
        np.bincount(
            owners,
            weights=pyramid_three_volumes,
            minlength=1,
        )[0]
        if bincount_reduction
        else pyramid_three_volumes.sum()
    )
    used_points = mesh_points[np.unique(np.concatenate(faces))]
    extent = float(np.max(np.ptp(used_points, axis=0)))
    minimum_three_volume = np.finfo(np.float64).eps * max(
        extent**3,
        np.finfo(np.float64).tiny,
    )
    if not math.isfinite(three_volume) or abs(three_volume) <= minimum_three_volume:
        raise RuntimeError(f"VTK cell {cell_id} has zero or non-finite volume.")
    pyramid_centres = 0.75 * face_centres + 0.25 * estimate
    centre = (
        np.asarray(
            [
                np.bincount(
                    owners,
                    weights=pyramid_three_volumes * pyramid_centres[:, component],
                    minlength=1,
                )[0]
                for component in range(3)
            ]
        )
        if bincount_reduction
        else np.einsum(
            "f,fc->c",
            pyramid_three_volumes,
            pyramid_centres,
        )
    ) / three_volume
    if not np.isfinite(centre).all():
        raise RuntimeError(f"VTK cell {cell_id} has a non-finite volume centroid.")
    return centre, abs(three_volume) / 3.0


def _fixed_cell_face_topology(
    grid: pv.UnstructuredGrid,
    cell_id: int,
    connectivity: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, ...]:
    point_ids = connectivity[offsets[cell_id] : offsets[cell_id + 1]]
    local_ids = {int(point_id): index for index, point_id in enumerate(point_ids)}
    if len(local_ids) != len(point_ids):
        raise RuntimeError(f"VTK cell {cell_id} contains duplicate point ids.")
    try:
        return tuple(
            np.asarray([local_ids[int(point_id)] for point_id in face], dtype=np.int64)
            for face in _cell_face_cycles(grid, cell_id)
        )
    except KeyError as error:
        raise RuntimeError(
            f"VTK cell {cell_id} has face points outside its connectivity."
        ) from error


def _polyhedron_cell_geometry(
    grid: pv.UnstructuredGrid,
    mesh_points: np.ndarray,
    polyhedron_cell_ids: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized face-pyramid geometry for VTK generic polyhedron cells."""

    vtk_faces = grid.GetPolyhedronFaces()
    vtk_locations = grid.GetPolyhedronFaceLocations()
    if vtk_faces is None or vtk_locations is None:
        raise RuntimeError("VTK did not expose generic-polyhedron face connectivity.")

    # Copy these VTK-owned views: vtkOpenFOAMReader may replace the arrays on
    # the next time update, while topology caches can outlive one reader output.
    face_offsets = np.asarray(
        vtk_to_numpy(vtk_faces.GetOffsetsArray()),  # type: ignore[no-untyped-call]
        dtype=np.int64,
    ).copy()
    face_connectivity = np.asarray(
        vtk_to_numpy(vtk_faces.GetConnectivityArray()),  # type: ignore[no-untyped-call]
        dtype=np.int64,
    ).copy()
    location_offsets = np.asarray(
        vtk_to_numpy(vtk_locations.GetOffsetsArray()),  # type: ignore[no-untyped-call]
        dtype=np.int64,
    ).copy()
    location_connectivity = np.asarray(
        vtk_to_numpy(vtk_locations.GetConnectivityArray()),  # type: ignore[no-untyped-call]
        dtype=np.int64,
    ).copy()
    if (
        len(face_offsets) == 0
        or face_offsets[0] != 0
        or face_offsets[-1] != len(face_connectivity)
        or len(location_offsets) != grid.n_cells + 1
        or location_offsets[0] != 0
        or location_offsets[-1] != len(location_connectivity)
    ):
        raise RuntimeError("VTK returned inconsistent polyhedron face connectivity.")

    faces_per_cell = np.diff(location_offsets)
    if not np.array_equal(np.flatnonzero(faces_per_cell), polyhedron_cell_ids):
        raise RuntimeError("VTK returned inconsistent polyhedron face locations.")
    face_point_counts = np.diff(face_offsets)
    if (
        np.any(face_point_counts < 3)
        or np.any(face_connectivity < 0)
        or np.any(face_connectivity >= grid.n_points)
        or np.any(location_connectivity < 0)
        or np.any(location_connectivity >= len(face_point_counts))
    ):
        raise RuntimeError("VTK returned invalid polyhedron faces.")

    # Face connectivity is cyclic but not guaranteed to use mutually coherent
    # winding. Store one sign per cell-face reference, since a face record may
    # in principle be referenced by more than one cell.
    reference_signs = np.ones(len(location_connectivity), dtype=np.int8)
    for cell_id_value in polyhedron_cell_ids:
        cell_id = int(cell_id_value)
        first = int(location_offsets[cell_id])
        stop = int(location_offsets[cell_id + 1])
        face_ids = location_connectivity[first:stop]
        faces = tuple(
            face_connectivity[face_offsets[face_id] : face_offsets[face_id + 1]]
            for face_id in face_ids
        )
        reference_signs[first:stop] = _face_orientation_signs(
            faces,
            cell_id=cell_id,
        )

    face_centres = np.empty((len(face_point_counts), 3), dtype=np.float64)
    face_areas = np.empty((len(face_point_counts), 3), dtype=np.float64)
    for points_per_face_value in np.unique(face_point_counts):
        points_per_face = int(points_per_face_value)
        face_ids = np.flatnonzero(face_point_counts == points_per_face).astype(np.int64)
        local_ids = np.arange(points_per_face, dtype=np.int64)
        for first in range(0, len(face_ids), _CELL_GEOMETRY_CHUNK_CELLS):
            chunk_face_ids = face_ids[first : first + _CELL_GEOMETRY_CHUNK_CELLS]
            point_ids = face_connectivity[
                face_offsets[chunk_face_ids, None] + local_ids[None, :]
            ]
            centres, areas = _openfoam_face_geometry_batch(mesh_points[point_ids])
            face_centres[chunk_face_ids] = centres
            face_areas[chunk_face_ids] = areas

    reference_owners = np.repeat(
        np.arange(grid.n_cells, dtype=np.int64),
        faces_per_cell,
    )
    referenced_centres = face_centres[location_connectivity]
    referenced_areas = face_areas[location_connectivity] * reference_signs[:, None]
    centre_sums = np.column_stack(
        tuple(
            np.bincount(
                reference_owners,
                weights=referenced_centres[:, component],
                minlength=grid.n_cells,
            )
            for component in range(3)
        )
    )
    estimates = np.zeros((grid.n_cells, 3), dtype=np.float64)
    estimates[polyhedron_cell_ids] = (
        centre_sums[polyhedron_cell_ids] / faces_per_cell[polyhedron_cell_ids, None]
    )
    pyramid_three_volumes = np.einsum(
        "rc,rc->r",
        referenced_areas,
        referenced_centres - estimates[reference_owners],
    )
    three_volumes = np.bincount(
        reference_owners,
        weights=pyramid_three_volumes,
        minlength=grid.n_cells,
    )
    pyramid_centres = 0.75 * referenced_centres + 0.25 * estimates[reference_owners]
    moment_sums = np.column_stack(
        tuple(
            np.bincount(
                reference_owners,
                weights=pyramid_three_volumes * pyramid_centres[:, component],
                minlength=grid.n_cells,
            )
            for component in range(3)
        )
    )

    extents = np.empty(len(polyhedron_cell_ids), dtype=np.float64)
    cell_point_counts = np.diff(offsets)
    maximum_points = int(np.max(cell_point_counts[polyhedron_cell_ids]))
    local_point_ids = np.arange(maximum_points, dtype=np.int64)
    for first in range(0, len(polyhedron_cell_ids), _CELL_GEOMETRY_CHUNK_CELLS):
        stop = min(first + _CELL_GEOMETRY_CHUNK_CELLS, len(polyhedron_cell_ids))
        chunk_cell_ids = polyhedron_cell_ids[first:stop]
        chunk_counts = cell_point_counts[chunk_cell_ids]
        valid_points = local_point_ids[None, :] < chunk_counts[:, None]
        connectivity_ids = offsets[chunk_cell_ids, None] + local_point_ids[None, :]
        connectivity_ids = np.minimum(connectivity_ids, len(connectivity) - 1)
        point_ids = connectivity[connectivity_ids]
        point_coordinates = mesh_points[point_ids]
        minimums = np.where(
            valid_points[:, :, None],
            point_coordinates,
            math.inf,
        ).min(axis=1)
        maximums = np.where(
            valid_points[:, :, None],
            point_coordinates,
            -math.inf,
        ).max(axis=1)
        extents[first:stop] = np.max(maximums - minimums, axis=1)

    selected_three_volumes = three_volumes[polyhedron_cell_ids]
    minimum_three_volumes = np.finfo(np.float64).eps * np.maximum(
        extents**3,
        np.finfo(np.float64).tiny,
    )
    valid = np.isfinite(selected_three_volumes) & (
        np.abs(selected_three_volumes) > minimum_three_volumes
    )
    if not np.all(valid):
        bad = int(polyhedron_cell_ids[int(np.flatnonzero(~valid)[0])])
        raise RuntimeError(f"VTK cell {bad} has zero or non-finite volume.")
    centres = moment_sums[polyhedron_cell_ids] / selected_three_volumes[:, None]
    if not np.isfinite(centres).all():
        bad = int(
            polyhedron_cell_ids[
                int(np.flatnonzero(~np.isfinite(centres).all(axis=1))[0])
            ]
        )
        raise RuntimeError(f"VTK cell {bad} has a non-finite volume centroid.")
    return centres, np.abs(selected_three_volumes) / 3.0


def _openfoam_cell_geometry(
    grid: pv.UnstructuredGrid,
    cell_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenFOAM-style centroids and volumes for all or selected cells."""

    mesh_points = np.asarray(grid.points, dtype=np.float64)
    if mesh_points.shape != (grid.n_points, 3) or not np.isfinite(mesh_points).all():
        raise RuntimeError("VTK returned non-finite or malformed mesh points.")
    connectivity = np.asarray(grid.cell_connectivity, dtype=np.int64)
    offsets = np.asarray(grid.offset, dtype=np.int64)
    counts = np.diff(offsets)
    cell_types = np.asarray(grid.celltypes, dtype=np.int64)
    if (
        len(offsets) != grid.n_cells + 1
        or len(counts) != grid.n_cells
        or offsets[-1] != len(connectivity)
        or len(cell_types) != grid.n_cells
    ):
        raise RuntimeError("VTK returned inconsistent cell connectivity.")

    selected = (
        np.arange(grid.n_cells, dtype=np.int64)
        if cell_ids is None
        else np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    )
    if (
        len(selected) == 0
        or np.any(selected < 0)
        or np.any(selected >= grid.n_cells)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("cell_ids must contain unique, in-range cell indices.")

    cell_centres = np.empty((len(selected), 3), dtype=np.float64)
    cell_volumes = np.empty(len(selected), dtype=np.float64)
    selected_types = cell_types[selected]
    for cell_type in np.unique(selected_types):
        output_ids = np.flatnonzero(selected_types == cell_type).astype(np.int64)
        type_cell_ids = selected[output_ids]
        if int(cell_type) == int(pv.CellType.POLYHEDRON):
            if cell_ids is None:
                centres, volumes = _polyhedron_cell_geometry(
                    grid,
                    mesh_points,
                    type_cell_ids,
                    connectivity,
                    offsets,
                )
                cell_centres[output_ids] = centres
                cell_volumes[output_ids] = volumes
            else:
                for output_id, cell_id_value in zip(
                    output_ids,
                    type_cell_ids,
                    strict=True,
                ):
                    cell_id = int(cell_id_value)
                    centre, volume = _openfoam_cell_geometry_from_faces(
                        mesh_points,
                        _cell_face_cycles(grid, cell_id),
                        cell_id=cell_id,
                        bincount_reduction=True,
                    )
                    cell_centres[output_id] = centre
                    cell_volumes[output_id] = volume
        elif int(cell_type) in _FIXED_LINEAR_CELL_TYPES:
            points_per_cell = int(counts[type_cell_ids[0]])
            if not np.all(counts[type_cell_ids] == points_per_cell):
                raise RuntimeError(
                    f"VTK cell type {int(cell_type)} has inconsistent connectivity."
                )
            local_faces = _fixed_cell_face_topology(
                grid,
                int(type_cell_ids[0]),
                connectivity,
                offsets,
            )
            local_point_ids = np.arange(points_per_cell, dtype=np.int64)
            for first in range(0, len(type_cell_ids), _CELL_GEOMETRY_CHUNK_CELLS):
                chunk_cell_ids = type_cell_ids[
                    first : first + _CELL_GEOMETRY_CHUNK_CELLS
                ]
                point_ids = connectivity[
                    offsets[chunk_cell_ids, None] + local_point_ids[None, :]
                ]
                centres, volumes = _openfoam_cell_geometry_batch(
                    mesh_points[point_ids],
                    local_faces,
                    chunk_cell_ids,
                )
                chunk_output_ids = output_ids[
                    first : first + _CELL_GEOMETRY_CHUNK_CELLS
                ]
                cell_centres[chunk_output_ids] = centres
                cell_volumes[chunk_output_ids] = volumes
        else:
            for output_id, cell_id_value in zip(
                output_ids,
                type_cell_ids,
                strict=True,
            ):
                cell_id = int(cell_id_value)
                centre, volume = _openfoam_cell_geometry_from_faces(
                    mesh_points,
                    _cell_face_cycles(grid, cell_id),
                    cell_id=cell_id,
                )
                cell_centres[output_id] = centre
                cell_volumes[output_id] = volume
    return cell_centres, cell_volumes


def _face_triangles(
    grid: pv.UnstructuredGrid,
    cell_id: int,
    cell_center: np.ndarray,
    cell_volume: float,
    *,
    mesh_points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    face_cache: dict[int, tuple[np.ndarray, ...]] | None = None,
    fixed_face_cache: dict[int, tuple[np.ndarray, ...]] | None = None,
) -> np.ndarray:
    """Triangulate cell faces using a non-degenerate cyclic base where possible."""

    faces = None if face_cache is None else face_cache.get(cell_id)
    if faces is None:
        cell_type = int(grid.celltypes[cell_id])
        if cell_type in _FIXED_LINEAR_CELL_TYPES:
            local_faces = (
                None if fixed_face_cache is None else fixed_face_cache.get(cell_type)
            )
            if local_faces is None:
                local_faces = _fixed_cell_face_topology(
                    grid,
                    cell_id,
                    connectivity,
                    offsets,
                )
                if fixed_face_cache is not None:
                    fixed_face_cache[cell_type] = local_faces
            cell_point_ids = connectivity[offsets[cell_id] : offsets[cell_id + 1]]
            faces = tuple(cell_point_ids[face] for face in local_faces)
        else:
            faces = _cell_face_cycles(grid, cell_id)
        if face_cache is not None:
            face_cache[cell_id] = faces
    epsilon = np.finfo(np.float64).eps
    volume = max(float(cell_volume), np.finfo(np.float64).tiny)
    triangles: list[np.ndarray] = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face[None, :])
            continue
        selected: np.ndarray | None = None
        best: np.ndarray | None = None
        best_quality = -math.inf
        for base in range(len(face)):
            rotated = np.roll(face, -base)
            candidate = np.column_stack(
                (
                    np.full(len(face) - 2, rotated[0], dtype=np.int64),
                    rotated[1:-1],
                    rotated[2:],
                )
            )
            matrices = np.transpose(
                mesh_points[candidate] - cell_center[None, None, :],
                (0, 2, 1),
            )
            determinants = np.linalg.det(matrices)
            relative_quality = np.abs(determinants) / volume
            quality = float(np.min(relative_quality))
            if quality > best_quality:
                best_quality = quality
                best = candidate
            consistently_oriented = bool(
                np.all(determinants > epsilon * volume)
                or np.all(determinants < -epsilon * volume)
            )
            if consistently_oriented:
                selected = candidate
                break
        if best is None:
            raise RuntimeError(f"VTK cell {cell_id} has an invalid face triangulation.")
        triangles.append(best if selected is None else selected)
    if not triangles:
        raise RuntimeError(f"VTK cell {cell_id} has no valid face triangles.")
    return np.concatenate(triangles).astype(np.int64, copy=False)


def _point_segment_distance_squared(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= np.finfo(np.float64).tiny:
        difference = point - first
        return float(np.dot(difference, difference))
    fraction = float(np.dot(point - first, edge) / denominator)
    closest = first + min(1.0, max(0.0, fraction)) * edge
    difference = point - closest
    return float(np.dot(difference, difference))


def _point_triangle_distance_squared(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> float:
    """Squared Euclidean distance from a point to a solid triangle."""

    first_edge = second - first
    second_edge = third - first
    normal = np.cross(first_edge, second_edge)
    edge_scale_squared = max(
        float(np.dot(first_edge, first_edge)),
        float(np.dot(second_edge, second_edge)),
        float(np.dot(third - second, third - second)),
    )
    if float(np.dot(normal, normal)) <= np.finfo(np.float64).eps ** 2 * max(
        edge_scale_squared**2,
        np.finfo(np.float64).tiny,
    ):
        return min(
            _point_segment_distance_squared(point, first, second),
            _point_segment_distance_squared(point, second, third),
            _point_segment_distance_squared(point, third, first),
        )

    offset = point - first
    d1 = float(np.dot(first_edge, offset))
    d2 = float(np.dot(second_edge, offset))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.dot(offset, offset))

    offset = point - second
    d3 = float(np.dot(first_edge, offset))
    d4 = float(np.dot(second_edge, offset))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.dot(offset, offset))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        difference = point - (first + fraction * first_edge)
        return float(np.dot(difference, difference))

    offset = point - third
    d5 = float(np.dot(first_edge, offset))
    d6 = float(np.dot(second_edge, offset))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.dot(offset, offset))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        difference = point - (first + fraction * second_edge)
        return float(np.dot(difference, difference))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        difference = point - (second + fraction * (third - second))
        return float(np.dot(difference, difference))

    denominator = 1.0 / (va + vb + vc)
    second_weight = vb * denominator
    third_weight = vc * denominator
    closest = first + second_weight * first_edge + third_weight * second_edge
    difference = point - closest
    return float(np.dot(difference, difference))


def _point_tetrahedron_distance_squared(
    point: np.ndarray,
    tetrahedron: np.ndarray,
) -> float:
    """Squared Euclidean distance from a point to a solid tetrahedron."""

    matrix = np.transpose(tetrahedron[1:] - tetrahedron[0])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) > np.finfo(np.float64).eps:
        vertex_weights = np.linalg.solve(matrix, point - tetrahedron[0])
        weights = np.concatenate(((1.0 - vertex_weights.sum(),), vertex_weights))
        if np.all(weights >= -np.finfo(np.float64).eps):
            return 0.0

    return min(
        _point_triangle_distance_squared(
            point,
            tetrahedron[first],
            tetrahedron[second],
            tetrahedron[third],
        )
        for first, second, third in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
    )


def _tetrahedral_weights(
    position: np.ndarray,
    cell_center: np.ndarray,
    face_points: np.ndarray,
    cell_volume: float,
) -> tuple[np.ndarray, int]:
    """Find OpenFOAM-style cell-centre/face-triangle barycentric weights."""

    cell_volume = float(cell_volume)
    if not math.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError("cell_volume must be finite and positive.")
    matrices = np.transpose(face_points - cell_center[None, None, :], (0, 2, 1))
    determinants = np.linalg.det(matrices)
    tolerance = np.finfo(np.float64).eps
    valid = np.abs(determinants) / cell_volume > tolerance
    # OpenFOAM checks candidate quality relative to the containing-cell volume,
    # but its point-to-barycentric solve uses an independent absolute ``small``
    # determinant test.
    solvable = np.abs(determinants) > tolerance
    weights = np.full((len(face_points), 4), 0.25, dtype=np.float64)
    if np.any(solvable):
        rhs = np.broadcast_to(
            position - cell_center,
            (int(np.count_nonzero(solvable)), 3),
        )
        vertex_weights = np.linalg.solve(matrices[solvable], rhs[..., None])[..., 0]
        weights[solvable, 1:] = vertex_weights
        weights[solvable, 0] = 1.0 - vertex_weights.sum(axis=1)
    inside = np.flatnonzero(valid & np.all(weights >= -tolerance, axis=1))
    if inside.size:
        index = int(inside[0])
        return weights[index], index
    tetrahedra = np.concatenate(
        (
            np.broadcast_to(cell_center, (len(face_points), 1, 3)),
            face_points,
        ),
        axis=1,
    )
    distances = np.fromiter(
        (
            _point_tetrahedron_distance_squared(position, tetrahedron)
            for tetrahedron in tetrahedra
        ),
        dtype=np.float64,
        count=len(tetrahedra),
    )
    index = int(np.argmin(distances))
    return weights[index], index


def _locate_cells(grid: pv.UnstructuredGrid, positions: np.ndarray) -> np.ndarray:
    locator = vtkStaticCellLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()
    cell_ids = np.fromiter(
        (locator.FindCell(position) for position in positions),
        dtype=np.int64,
        count=len(positions),
    )
    # vtkStaticCellLocator is fast but can reject a point on a polyhedron face.
    # vtkUnstructuredGrid.FindCell uses the cell's own tolerance-aware search and
    # is a reliable fallback for those relatively rare interface points.
    missing = np.flatnonzero(cell_ids < 0)
    if missing.size:
        max_cell_points = max(1, int(np.diff(np.asarray(grid.offset)).max()))
        for index in missing:
            sub_id = reference(0)
            parametric = [0.0, 0.0, 0.0]
            weights = [0.0] * max_cell_points
            cell_ids[index] = grid.FindCell(
                positions[index],
                None,
                -1,
                0.0,
                sub_id,
                parametric,
                weights,
            )
    return cell_ids


@dataclass(frozen=True, slots=True)
class _PointWeightChunk:
    """One original cell-order chunk of the local cell-to-point operator."""

    cell_indices: np.ndarray
    point_indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class _StationaryInterpolationPlan:
    """Reusable geometry and weights for a locally stationary sampled surface."""

    grid_cell_count: int
    grid_point_count: int
    required_cell_ids: np.ndarray
    required_cell_types: np.ndarray
    support_point_ids: np.ndarray
    support_point_coordinates: np.ndarray
    surface_positions: np.ndarray
    surface_cell_ids: np.ndarray
    surface_cell_indices: np.ndarray
    surface_point_indices: np.ndarray
    surface_weights: np.ndarray
    point_denominator: np.ndarray
    point_weight_chunks: tuple[_PointWeightChunk, ...]

    def geometry_matches(self, grid: pv.UnstructuredGrid) -> bool:
        """Return whether every cached local geometric dependency is unchanged."""
        if (
            grid.n_cells != self.grid_cell_count
            or grid.n_points != self.grid_point_count
        ):
            return False
        cell_types = np.asarray(grid.celltypes, dtype=np.int64)
        if not np.array_equal(
            cell_types[self.required_cell_ids],
            self.required_cell_types,
        ):
            return False
        points = np.asarray(grid.points, dtype=np.float64)
        if not np.array_equal(
            points[self.support_point_ids],
            self.support_point_coordinates,
        ):
            return False
        return True

    def locate_surface_cells(self, grid: pv.UnstructuredGrid) -> np.ndarray | None:
        """Locate the surface when cached local geometry remains applicable."""

        if not self.geometry_matches(grid):
            return None
        return _locate_cells(grid, self.surface_positions)

    def matches(self, grid: pv.UnstructuredGrid) -> bool:
        """Return whether geometry and VTK containing-cell choices still match."""

        cell_ids = self.locate_surface_cells(grid)
        if cell_ids is None:
            return False
        return np.array_equal(
            cell_ids,
            self.surface_cell_ids,
        )

    def interpolate(
        self,
        grid: pv.UnstructuredGrid,
        *,
        pressure_field: str,
        velocity_field: str,
        time_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the prepared two-stage OpenFOAM-style interpolation operator."""

        cell_values = _selected_cell_fields(
            grid,
            self.required_cell_ids,
            pressure_field,
            velocity_field,
        )
        return self.interpolate_cell_values(cell_values, time_name=time_name)

    def interpolate_cell_values(
        self,
        cell_values: np.ndarray,
        *,
        time_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the plan to required-cell ``p,U`` values in plan order."""

        cell_values = np.asarray(cell_values, dtype=np.float64)
        expected_shape = (len(self.required_cell_ids), 4)
        if cell_values.shape != expected_shape:
            raise ValueError(f"cell_values must have shape {expected_shape}.")
        if not np.isfinite(cell_values).all():
            raise ValueError(
                f"OpenFOAM fields at time {time_name!r} contain non-finite values."
            )
        point_values = np.zeros(
            (len(self.point_denominator), cell_values.shape[1]),
            dtype=np.float64,
        )
        for chunk in self.point_weight_chunks:
            weighted_values = chunk.weights[:, None] * cell_values[chunk.cell_indices]
            for component in range(cell_values.shape[1]):
                point_values[:, component] += np.bincount(
                    chunk.point_indices,
                    weighted_values[:, component],
                    minlength=len(self.point_denominator),
                )
        point_values /= self.point_denominator[:, None]

        values = self.surface_weights[:, :1] * cell_values[self.surface_cell_indices]
        values += np.sum(
            self.surface_weights[:, 1:, None]
            * point_values[self.surface_point_indices],
            axis=1,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Interpolation at OpenFOAM time {time_name!r} produced non-finite values."
            )
        return values[:, 0].copy(), values[:, 1:].copy()


def _surface_cell_ids(
    grid: pv.UnstructuredGrid,
    positions: np.ndarray,
    *,
    time_name: str,
) -> np.ndarray:
    cell_ids = _locate_cells(grid, positions)
    invalid = np.flatnonzero(cell_ids < 0)
    if invalid.size:
        first = positions[int(invalid[0])].tolist()
        target_bounds = (
            positions.min(axis=0).tolist(),
            positions.max(axis=0).tolist(),
        )
        raise ValueError(
            f"Surface is not fully inside the fluid mesh at time {time_name!r}: "
            f"{len(invalid)} of {len(positions)} vertices are outside; first invalid "
            f"vertex is {first}. Surface bounds are {target_bounds}; volume bounds "
            f"are {tuple(float(value) for value in grid.bounds)}."
        )
    return cell_ids


def _support_point_ids(
    connectivity: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    required_cell_ids: np.ndarray,
) -> np.ndarray:
    """Return all mesh points affecting the required cell centroids."""

    required = np.zeros(len(counts), dtype=bool)
    required[required_cell_ids] = True
    point_parts: list[np.ndarray] = []
    for first in range(0, len(counts), _POINT_WEIGHT_CHUNK_CELLS):
        stop = min(first + _POINT_WEIGHT_CHUNK_CELLS, len(counts))
        edge_mask = np.repeat(required[first:stop], counts[first:stop])
        if np.any(edge_mask):
            point_parts.append(connectivity[offsets[first] : offsets[stop]][edge_mask])
    if not point_parts:
        raise RuntimeError("The interpolation plan has no supporting mesh points.")
    return np.unique(np.concatenate(point_parts)).astype(np.int64, copy=False)


def _build_stationary_interpolation_plan(
    grid: pv.UnstructuredGrid,
    positions: np.ndarray,
    *,
    time_name: str,
    face_cache: dict[int, tuple[np.ndarray, ...]] | None = None,
) -> _StationaryInterpolationPlan:
    """Prepare the exact geometry-dependent work for repeated field sampling."""

    cell_ids = _surface_cell_ids(grid, positions, time_name=time_name)
    mesh_points = np.asarray(grid.points, dtype=np.float64)
    connectivity = np.asarray(grid.cell_connectivity, dtype=np.int64)
    offsets = np.asarray(grid.offset, dtype=np.int64)
    counts = np.diff(offsets)
    target_cell_ids = np.unique(cell_ids)
    target_cell_centers, target_cell_volumes = _openfoam_cell_geometry(
        grid,
        target_cell_ids,
    )
    target_cell_indices = np.searchsorted(target_cell_ids, cell_ids)

    if face_cache is None:
        face_cache = {}
    frame_triangle_cache: dict[int, np.ndarray] = {}
    fixed_face_cache: dict[int, tuple[np.ndarray, ...]] = {}
    surface_point_ids = np.empty((len(positions), 3), dtype=np.int64)
    surface_weights = np.empty((len(positions), 4), dtype=np.float64)
    for index, (position, cell_id_value) in enumerate(
        zip(positions, cell_ids, strict=True)
    ):
        cell_id = int(cell_id_value)
        target_index = int(target_cell_indices[index])
        triangles = frame_triangle_cache.get(cell_id)
        if triangles is None:
            triangles = _face_triangles(
                grid,
                cell_id,
                target_cell_centers[target_index],
                target_cell_volumes[target_index],
                mesh_points=mesh_points,
                connectivity=connectivity,
                offsets=offsets,
                face_cache=face_cache,
                fixed_face_cache=fixed_face_cache,
            )
            frame_triangle_cache[cell_id] = triangles
        weights, tetrahedron = _tetrahedral_weights(
            position,
            target_cell_centers[target_index],
            mesh_points[triangles],
            target_cell_volumes[target_index],
        )
        surface_point_ids[index] = triangles[tetrahedron]
        surface_weights[index] = weights

    needed_point_ids = np.unique(surface_point_ids)
    incidence_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for first in range(0, grid.n_cells, _POINT_WEIGHT_CHUNK_CELLS):
        stop = min(first + _POINT_WEIGHT_CHUNK_CELLS, grid.n_cells)
        chunk_cell_ids = np.repeat(
            np.arange(first, stop, dtype=np.int64),
            counts[first:stop],
        )
        chunk_point_ids = connectivity[offsets[first] : offsets[stop]]
        local_point_ids = np.searchsorted(needed_point_ids, chunk_point_ids)
        selected = local_point_ids < len(needed_point_ids)
        selected[selected] &= (
            needed_point_ids[local_point_ids[selected]] == chunk_point_ids[selected]
        )
        if not np.any(selected):
            continue
        selected_cells = chunk_cell_ids[selected]
        selected_points = chunk_point_ids[selected]
        selected_local_points = local_point_ids[selected]
        incidence_chunks.append(
            (selected_cells, selected_points, selected_local_points)
        )

    required_cell_ids = np.unique(
        np.concatenate(
            [cell_ids, *(chunk[0] for chunk in incidence_chunks)],
        )
    ).astype(np.int64, copy=False)
    required_cell_centers, _ = _openfoam_cell_geometry(grid, required_cell_ids)
    point_denominator = np.zeros(len(needed_point_ids), dtype=np.float64)
    global_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    bounds = np.asarray(grid.bounds, dtype=np.float64)
    scale = max(
        float(np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]])),
        1.0,
    )
    minimum_distance = np.finfo(np.float64).eps * scale
    for selected_cells, selected_points, selected_local_points in incidence_chunks:
        required_indices = np.searchsorted(required_cell_ids, selected_cells)
        distances = np.linalg.norm(
            mesh_points[selected_points] - required_cell_centers[required_indices],
            axis=1,
        )
        weights = 1.0 / np.maximum(distances, minimum_distance)
        point_denominator += np.bincount(
            selected_local_points,
            weights,
            minlength=len(needed_point_ids),
        )
        global_chunks.append((selected_cells, selected_local_points, weights))
    if np.any(point_denominator <= 0.0) or not np.isfinite(point_denominator).all():
        raise RuntimeError(
            "Could not construct finite cell-to-point interpolation weights."
        )

    surface_cell_indices = np.searchsorted(required_cell_ids, cell_ids)
    point_weight_chunks = tuple(
        _PointWeightChunk(
            cell_indices=np.searchsorted(required_cell_ids, chunk_cell_ids),
            point_indices=chunk_point_ids.copy(),
            weights=weights.copy(),
        )
        for chunk_cell_ids, chunk_point_ids, weights in global_chunks
    )
    surface_point_indices = np.searchsorted(needed_point_ids, surface_point_ids)
    support_point_ids = _support_point_ids(
        connectivity,
        offsets,
        counts,
        required_cell_ids,
    )
    cell_types = np.asarray(grid.celltypes, dtype=np.int64)
    return _StationaryInterpolationPlan(
        grid_cell_count=grid.n_cells,
        grid_point_count=grid.n_points,
        required_cell_ids=required_cell_ids.copy(),
        required_cell_types=cell_types[required_cell_ids].copy(),
        support_point_ids=support_point_ids.copy(),
        support_point_coordinates=mesh_points[support_point_ids].copy(),
        surface_positions=positions.copy(),
        surface_cell_ids=cell_ids.copy(),
        surface_cell_indices=surface_cell_indices.copy(),
        surface_point_indices=surface_point_indices.copy(),
        surface_weights=surface_weights.copy(),
        point_denominator=point_denominator.copy(),
        point_weight_chunks=point_weight_chunks,
    )


def _openfoam_like_cell_point(
    grid: pv.UnstructuredGrid,
    positions: np.ndarray,
    *,
    pressure_field: str,
    velocity_field: str,
    time_name: str,
    face_cache: dict[int, tuple[np.ndarray, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use VTK location plus an OpenFOAM-inspired ``cellPoint`` reconstruction."""

    cell_ids = _surface_cell_ids(grid, positions, time_name=time_name)

    mesh_points = np.asarray(grid.points, dtype=np.float64)
    connectivity = np.asarray(grid.cell_connectivity, dtype=np.int64)
    offsets = np.asarray(grid.offset, dtype=np.int64)
    cell_centers, cell_volumes = _openfoam_cell_geometry(grid)
    cell_values = _cell_fields(grid, pressure_field, velocity_field)
    point_values = _inverse_distance_point_fields(grid, cell_centers, cell_values)

    values = np.empty((len(positions), cell_values.shape[1]), dtype=np.float64)
    if face_cache is None:
        face_cache = {}
    frame_triangle_cache: dict[int, np.ndarray] = {}
    fixed_face_cache: dict[int, tuple[np.ndarray, ...]] = {}
    for index, (position, cell_id_value) in enumerate(
        zip(positions, cell_ids, strict=True)
    ):
        cell_id = int(cell_id_value)
        triangles = frame_triangle_cache.get(cell_id)
        if triangles is None:
            triangles = _face_triangles(
                grid,
                cell_id,
                cell_centers[cell_id],
                cell_volumes[cell_id],
                mesh_points=mesh_points,
                connectivity=connectivity,
                offsets=offsets,
                face_cache=face_cache,
                fixed_face_cache=fixed_face_cache,
            )
            frame_triangle_cache[cell_id] = triangles
        weights, tetrahedron = _tetrahedral_weights(
            position,
            cell_centers[cell_id],
            mesh_points[triangles],
            cell_volumes[cell_id],
        )
        point_ids = triangles[tetrahedron]
        values[index] = weights[0] * cell_values[cell_id] + np.sum(
            weights[1:, None] * point_values[point_ids], axis=0
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Interpolation at OpenFOAM time {time_name!r} produced non-finite values."
        )
    return values[:, 0].copy(), values[:, 1:].copy()


def _field_path(time_dir: Path, field: str, region: str | None) -> Path:
    root = time_dir if region is None else time_dir / region
    found = _foam_file(root / field)
    if found is None:
        raise FileNotFoundError(
            f"Required OpenFOAM field {field!r} is missing at time {time_dir.name!r} "
            f"under {root}."
        )
    return found


def _read_foam_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, mode="rb") as stream:
            return stream.read()
    return path.read_bytes()


def _uniform_internal_values(
    contents: bytes,
    indices: np.ndarray,
    components: int,
) -> np.ndarray | None:
    """Parse a uniform internal field, if present and conventionally encoded."""

    match = re.search(rb"\binternalField\s+uniform\s+([^;]+);", contents)
    if match is None:
        return None
    flat = np.fromstring(
        match.group(1).translate(_FOAM_VECTOR_TRANSLATION),
        dtype=np.float32,
        sep=" ",
    )
    if len(flat) != components or not np.isfinite(flat).all():
        return None
    if components == 1:
        return np.full(len(indices), flat[0], dtype=np.float32)
    return np.broadcast_to(flat, (len(indices), components)).copy()


def _ascii_list_payload(
    contents: bytes,
    *,
    expected_count: int,
    internal_field: bool,
) -> bytes | None:
    """Extract a conventionally encoded non-uniform OpenFOAM list payload."""

    pattern = (
        rb"\binternalField\s+nonuniform\s+\S+\s+(\d+)\s*\("
        if internal_field
        else rb"(?m)^\s*(\d+)\s*$\s*^\s*\("
    )
    opening = re.search(pattern, contents)
    if opening is None or int(opening.group(1)) != expected_count:
        return None
    closing = re.search(rb"(?m)^\s*\)\s*;?\s*$", contents[opening.end() :])
    if closing is None:
        return None
    return contents[opening.end() : opening.end() + closing.start()]


def _selected_ascii_values(
    payload: bytes,
    indices: np.ndarray,
    *,
    expected_count: int,
    components: int,
) -> np.ndarray | None:
    """Parse only selected list rows when the common line layout permits it."""

    lines = payload.splitlines()
    first_line = next(
        (index for index in range(len(lines)) if lines[index].strip()),
        len(lines),
    )
    last_line = next(
        (
            index + 1
            for index in range(len(lines) - 1, first_line - 1, -1)
            if lines[index].strip()
        ),
        first_line,
    )
    if last_line - first_line == expected_count:
        selected_payload = b"\n".join(
            lines[first_line + int(index)] for index in indices
        )
        flat = np.fromstring(
            selected_payload.translate(_FOAM_VECTOR_TRANSLATION),
            dtype=np.float32,
            sep=" ",
        )
        return flat if len(flat) == len(indices) * components else None

    flat = np.fromstring(
        payload.translate(_FOAM_VECTOR_TRANSLATION),
        dtype=np.float32,
        sep=" ",
    )
    if len(flat) != expected_count * components:
        return None
    selected_values: np.ndarray = flat.reshape(expected_count, components)[indices]
    return selected_values.reshape(-1)


def _read_selected_ascii_list(
    path: Path,
    indices: np.ndarray,
    *,
    expected_count: int,
    components: int,
    internal_field: bool,
) -> np.ndarray | None:
    """Read selected entries from a standard ASCII OpenFOAM list.

    ``None`` requests the VTK fallback for binary or unusual encodings. The
    common one-entry-per-line representation avoids parsing unused values.
    """

    try:
        contents = _read_foam_bytes(path)
    except OSError:
        return None
    if re.search(rb"\bformat\s+ascii\s*;", contents[:4096]) is None:
        return None

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if (
        components not in (1, 3)
        or np.any(indices < 0)
        or np.any(indices >= expected_count)
    ):
        return None

    if internal_field:
        uniform_values = _uniform_internal_values(contents, indices, components)
        if uniform_values is not None:
            return uniform_values
    payload = _ascii_list_payload(
        contents,
        expected_count=expected_count,
        internal_field=internal_field,
    )
    if payload is None:
        return None
    flat = _selected_ascii_values(
        payload,
        indices,
        expected_count=expected_count,
        components=components,
    )
    if flat is None:
        return None
    if not np.isfinite(flat).all():
        raise ValueError(f"OpenFOAM list {path} contains non-finite values.")
    return flat if components == 1 else flat.reshape(len(indices), components)


def _read_stationary_ascii_cell_values(
    step: _CaseTime,
    plan: _StationaryInterpolationPlan,
    *,
    pressure_field: str,
    velocity_field: str,
    region: str | None,
    mesh_points_static: bool,
    assume_stationary_mesh: bool = False,
) -> tuple[bool, np.ndarray | None]:
    """Try a minimal direct read, returning geometry validity and cell values."""

    points_file = _time_mesh_points_file(step.directory, region)
    if not assume_stationary_mesh and points_file is not None:
        support_coordinates = _read_selected_ascii_list(
            points_file,
            plan.support_point_ids,
            expected_count=plan.grid_point_count,
            components=3,
            internal_field=False,
        )
        if support_coordinates is None:
            return True, None
        if not np.array_equal(
            support_coordinates.astype(np.float64),
            plan.support_point_coordinates,
        ):
            return False, None
    elif not assume_stationary_mesh and not mesh_points_static:
        # Sparse moving-mesh writes require OpenFOAM's time-state resolution.
        return True, None

    pressure = _read_selected_ascii_list(
        _field_path(step.directory, pressure_field, region),
        plan.required_cell_ids,
        expected_count=plan.grid_cell_count,
        components=1,
        internal_field=True,
    )
    velocity = _read_selected_ascii_list(
        _field_path(step.directory, velocity_field, region),
        plan.required_cell_ids,
        expected_count=plan.grid_cell_count,
        components=3,
        internal_field=True,
    )
    if pressure is None or velocity is None:
        return True, None
    return True, np.column_stack((pressure, velocity)).astype(np.float64, copy=False)


def _source_signatures(
    case_dir: Path,
    times: Sequence[_CaseTime],
    *,
    pressure_field: str,
    velocity_field: str,
    region: str | None,
) -> list[dict[str, int | str]]:
    paths: set[Path] = set()
    mesh_dir = _mesh_dir(case_dir, region)
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        path = _foam_file(mesh_dir / name)
        if path is not None:
            paths.add(path)
    for step in times:
        paths.add(_field_path(step.directory, pressure_field, region))
        paths.add(_field_path(step.directory, velocity_field, region))
        paths.update(_time_metadata_files(step.directory, region))
        for candidate in (
            step.directory / "polyMesh",
            step.directory / "fvMesh",
            step.directory / region / "polyMesh" if region is not None else None,
            step.directory / region / "fvMesh" if region is not None else None,
        ):
            if candidate is None or not candidate.exists():
                continue
            if candidate.is_file():
                paths.add(candidate)
            else:
                paths.update(path for path in candidate.rglob("*") if path.is_file())
    signatures: list[dict[str, int | str]] = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(case_dir))):
        stat = path.stat()
        signatures.append(
            {
                "path": path.relative_to(case_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signatures


def _mesh_fingerprint(mesh: TriangleMesh) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(mesh.vertices_m.numpy().tobytes())
    digest.update(mesh.triangles.numpy().tobytes())
    return digest.hexdigest()


def _cache_payload(
    surface: SurfaceSpec,
    mesh: TriangleMesh,
    times: Sequence[_CaseTime],
    signatures: list[dict[str, int | str]],
    *,
    pressure_field: str,
    velocity_field: str,
    pressure_units: PressureUnits,
    density_kg_m3: float,
    pressure_reference_pa: float,
    region: str | None,
    dtype: DTypeName | torch.dtype,
    assume_stationary_mesh: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": _SAMPLING_SCHEMA_VERSION,
        "implementation": "vtk-openfoam-volume-centroid-cellPoint-v3",
        "vtk_version": vtkVersion.GetVTKVersion(),
        "pyvista_version": pv.__version__,
        "surface_type": type(surface).__name__,
        "surface": asdict(surface),
        "surface_mesh": _mesh_fingerprint(mesh),
        "point_count": len(mesh.vertices_m),
        "panel_count": len(mesh.triangles),
        "pressure_field": pressure_field,
        "velocity_field": velocity_field,
        "pressure_units": pressure_units,
        "density_kg_m3": density_kg_m3,
        "pressure_reference_pa": pressure_reference_pa,
        "region": region,
        "dtype": str(dtype),
        "times": [
            {
                "name": step.name,
                "reader_value": step.reader_value,
                "physical_value": step.physical_value,
                "output_name": step.output_name,
            }
            for step in times
        ],
        "sources": signatures,
    }
    if assume_stationary_mesh:
        payload["assume_stationary_mesh"] = True
    return payload


def _cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_values(stream: TextIO, values: np.ndarray, *, per_line: int = 8) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    for first in range(0, len(flat), per_line):
        stream.write(
            " ".join(f"{value:.17g}" for value in flat[first : first + per_line])
        )
        stream.write("\n")


def _surface_vtk_geometry_text(mesh: TriangleMesh) -> str:
    """Format the time-invariant legacy-VTK geometry once per sampling run."""

    points = mesh.vertices_m.numpy()
    triangles = mesh.triangles.numpy()
    lines = [
        "# vtk DataFile Version 2.0\n",
        "FoamAcoustics VTK sampled surface\n",
        "ASCII\n",
        "DATASET POLYDATA\n",
        f"POINTS {len(points)} double\n",
    ]
    lines.extend(
        " ".join(f"{value:.17g}" for value in point) + "\n" for point in points
    )
    lines.append(f"POLYGONS {len(triangles)} {4 * len(triangles)}\n")
    lines.extend(
        f"3 {int(triangle[0])} {int(triangle[1])} {int(triangle[2])}\n"
        for triangle in triangles
    )
    return "".join(lines)


def _write_surface_vtk(
    path: Path,
    mesh: TriangleMesh,
    pressure: np.ndarray,
    velocity: np.ndarray,
    *,
    pressure_field: str,
    velocity_field: str,
    geometry_text: str | None = None,
) -> None:
    points = mesh.vertices_m.numpy()
    pressure = np.asarray(pressure, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if pressure.shape != (len(points),) or velocity.shape != (len(points), 3):
        raise ValueError(
            "Sampled surface fields do not match the surface vertex count."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(
                _surface_vtk_geometry_text(mesh)
                if geometry_text is None
                else geometry_text
            )
            stream.write(f"POINT_DATA {len(points)}\n")
            stream.write("FIELD attributes 2\n")
            stream.write(f"{pressure_field} 1 {len(points)} double\n")
            _write_values(stream, pressure)
            stream.write(f"{velocity_field} 3 {len(points)} double\n")
            _write_values(stream, velocity)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _output_file_signatures(
    output_dir: Path,
    *,
    surface_file: str,
    times: Sequence[_CaseTime],
) -> list[dict[str, int | str]] | None:
    signatures: list[dict[str, int | str]] = []
    try:
        for step in times:
            path = output_dir / step.output_name / surface_file
            if not path.is_file():
                return None
            stat = path.stat()
            signatures.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    except OSError:
        return None
    return signatures


def _cached_output_valid(
    manifest_file: Path,
    *,
    key: str,
    output_dir: Path,
    surface_file: str,
    time_list_file: Path,
    times: Sequence[_CaseTime],
) -> bool:
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if manifest.get("cache_key") != key:
        return False
    expected = [step.output_name for step in times]
    if manifest.get("completed_times") != expected:
        return False
    if manifest.get("surface_file") != surface_file:
        return False
    expected_time_list = "".join(f"{name}\n" for name in expected).encode("utf-8")
    try:
        if time_list_file.read_bytes() != expected_time_list:
            return False
    except OSError:
        return False
    output_signatures = _output_file_signatures(
        output_dir,
        surface_file=surface_file,
        times=times,
    )
    return (
        output_signatures is not None
        and manifest.get("output_signatures") == output_signatures
    )


def _validated_sampling_options(
    folder: str | Path,
    surface: SurfaceSpec,
    *,
    surface_name: str,
    pressure_field: str,
    velocity_field: str,
    pressure_units: PressureUnits,
    density_kg_m3: float,
    pressure_reference_pa: float,
    dtype: DTypeName | torch.dtype,
    region: str | None,
    assume_stationary_mesh: bool,
    workers: int,
    read_attempts: int,
) -> tuple[Path, str, str, str, float, float, torch.dtype, str | None, bool, int, int]:
    """Validate public sampling options before any expensive VTK work."""

    case_dir = Path(folder).expanduser().resolve()
    if not isinstance(surface, (SphereSurface, CylinderSurface)):
        raise TypeError("surface must be a SphereSurface or CylinderSurface.")
    if pressure_units not in {"pa", "kinematic"}:
        raise ValueError("pressure_units must be either 'pa' or 'kinematic'.")
    if not isinstance(assume_stationary_mesh, (bool, np.bool_)):
        raise TypeError("assume_stationary_mesh must be a boolean.")
    validated_region = None if region is None else _validate_region_name(region)
    validated_density = positive_float(density_kg_m3, "density_kg_m3")
    validated_reference = finite_float(
        pressure_reference_pa,
        "pressure_reference_pa",
    )
    validated_workers = positive_int(workers, "workers")
    validated_attempts = positive_int(read_attempts, "read_attempts")
    _validate_case(case_dir, validated_region)
    return (
        case_dir,
        _validate_surface_name(surface_name),
        _validate_field_name(pressure_field, "pressure_field"),
        _validate_field_name(velocity_field, "velocity_field"),
        validated_density,
        validated_reference,
        resolve_dtype(dtype),
        validated_region,
        bool(assume_stationary_mesh),
        validated_workers,
        validated_attempts,
    )


def interpolate_openfoam_fields(
    folder: str | Path,
    surface: SurfaceSpec,
    *,
    surface_name: str = "acousticSurface",
    pressure_field: str = "p",
    velocity_field: str = "U",
    pressure_units: PressureUnits = "kinematic",
    density_kg_m3: float = 1.225,
    pressure_reference_pa: float = 0.0,
    time_selection: str | None = None,
    region: str | None = None,
    dtype: DTypeName | torch.dtype = "float64",
    max_times: int | None = None,
    read_attempts: int = 4,
    cache_dir: str | Path | None = None,
    assume_stationary_mesh: bool = False,
    workers: int = 1,
    progress: bool = True,
) -> OpenFOAMSurfaceSeries:
    """Interpolate OpenFOAM ``p`` and ``U`` onto an analytic surface using VTK.

    No OpenFOAM executable or ``controlDict`` is used. VTK reads the volume
    mesh and fields directly. FoamAcoustics reconstructs OpenFOAM-style face
    geometry, face-pyramid cell-volume centroids, inverse-distance point
    values, and cell-centre/face-vertex tetrahedral weights. VTK still controls
    cell location and topology, so physical-boundary and non-conformal point
    constraints are not bitwise equivalent to OpenFOAM. The returned series is
    accepted by :class:`foamacoustics.F1ASolver` exactly like an existing
    sampled-surface VTK series. When topology is static, geometry and weights
    for the surface's local interpolation stencil are prepared once and reused
    until motion is detected in that stencil. Set ``assume_stationary_mesh``
    only when that stencil is known to remain fixed; this skips its per-time
    coordinate validation and pins first-step containing-cell choices for
    maximum throughput. With that option enabled, ``workers`` parallelizes
    direct reads and interpolation of standard ASCII fields after the stationary
    interpolation plan has been prepared. VTK fallbacks and moving-mesh plan
    updates remain serialized because VTK's OpenFOAM reader uses process-global
    state.

    Args:
        folder: OpenFOAM case directory containing ``constant/polyMesh``.
        surface: Analytic sphere or capped cylinder to sample.
        surface_name: Cache/output stem for the sampled surface.
        pressure_field: OpenFOAM scalar pressure field name.
        velocity_field: OpenFOAM vector velocity field name.
        pressure_units: ``"pa"`` or kinematic pressure units.
        density_kg_m3: Positive density used for kinematic-pressure conversion.
        pressure_reference_pa: Pascal reference subtracted after conversion.
        time_selection: Optional comma/space-separated times or inclusive ranges.
        region: Optional OpenFOAM region name.
        dtype: Floating-point dtype used by the returned surface series.
        max_times: Optional number of leading selected times, at least two.
        read_attempts: VTK read attempts used by the returned series.
        cache_dir: Optional binary-cache directory for the returned series.
        assume_stationary_mesh: Skip per-time local-coordinate validation when the
            interpolation stencil is known to remain fixed.
        workers: Number of parallel ASCII-field readers for a stationary mesh.
        progress: Display a time-step progress bar.

    Returns:
        A surface series ready to pass to :class:`foamacoustics.F1ASolver`.

    Raises:
        FileNotFoundError: If the case or required mesh/field files are absent.
        TypeError: If ``surface`` or a strict boolean/integer option is invalid.
        ValueError: If names, times, fields, geometry, or interpolation are invalid.
        RuntimeError: If source files change during sampling or VTK reports an error.
    """

    (
        case_dir,
        surface_name,
        pressure_field,
        velocity_field,
        density_kg_m3,
        pressure_reference_pa,
        resolved_dtype,
        region,
        assume_stationary_mesh,
        workers,
        read_attempts,
    ) = _validated_sampling_options(
        folder,
        surface,
        surface_name=surface_name,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        pressure_units=pressure_units,
        density_kg_m3=density_kg_m3,
        pressure_reference_pa=pressure_reference_pa,
        dtype=dtype,
        region=region,
        assume_stationary_mesh=assume_stationary_mesh,
        workers=workers,
        read_attempts=read_attempts,
    )

    mesh = surface.triangulate()
    positions = np.asarray(mesh.vertices_m.numpy(), dtype=np.float64)
    vtk_reader = _VTKOpenFOAMReader(
        case_dir,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        region=region,
    )
    times = _select_times(
        vtk_reader.times,
        time_selection=time_selection,
        max_times=max_times,
    )
    signatures = _source_signatures(
        case_dir,
        times,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        region=region,
    )
    payload = _cache_payload(
        surface,
        mesh,
        times,
        signatures,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        pressure_units=pressure_units,
        density_kg_m3=density_kg_m3,
        pressure_reference_pa=pressure_reference_pa,
        region=region,
        dtype=dtype,
        assume_stationary_mesh=assume_stationary_mesh,
    )
    key = _cache_key(payload)
    run_dir = case_dir / ".foamacoustics" / "sampled" / f"{surface_name}-{key[:16]}"
    output_dir = run_dir / "surfaceData"
    surface_file = f"{surface_name}.vtk"
    time_list_file = run_dir / "foamacoustics-times.txt"
    manifest_file = run_dir / "manifest.json"

    if not _cached_output_valid(
        manifest_file,
        key=key,
        output_dir=output_dir,
        surface_file=surface_file,
        time_list_file=time_list_file,
        times=times,
    ):
        face_cache: dict[int, tuple[np.ndarray, ...]] | None = (
            {} if getattr(vtk_reader, "mesh_topology_static", False) else None
        )
        use_stationary_plan = bool(getattr(vtk_reader, "mesh_topology_static", False))
        stationary_plan: _StationaryInterpolationPlan | None = None
        stationary_plan_cache: dict[bytes, _StationaryInterpolationPlan] = {}

        def remember_plan(plan: _StationaryInterpolationPlan) -> None:
            assignment = plan.surface_cell_ids.tobytes()
            stationary_plan_cache.pop(assignment, None)
            stationary_plan_cache[assignment] = plan
            while len(stationary_plan_cache) > _INTERPOLATION_PLAN_CACHE_SIZE:
                stationary_plan_cache.pop(next(iter(stationary_plan_cache)))

        geometry_text = _surface_vtk_geometry_text(mesh)

        def write_frame(
            step: _CaseTime,
            pressure: np.ndarray,
            velocity: np.ndarray,
        ) -> Path:
            output_file = output_dir / step.output_name / surface_file
            _write_surface_vtk(
                output_file,
                mesh,
                pressure,
                velocity,
                pressure_field=pressure_field,
                velocity_field=velocity_field,
                geometry_text=geometry_text,
            )
            return output_file

        def sample_step(step: _CaseTime, *, validate_output: bool = False) -> None:
            nonlocal stationary_plan, use_stationary_plan
            direct_cell_values: np.ndarray | None = None
            if stationary_plan is not None and assume_stationary_mesh:
                geometry_matches, direct_cell_values = (
                    _read_stationary_ascii_cell_values(
                        step,
                        stationary_plan,
                        pressure_field=pressure_field,
                        velocity_field=velocity_field,
                        region=region,
                        mesh_points_static=bool(
                            getattr(vtk_reader, "mesh_points_static", False)
                        ),
                        assume_stationary_mesh=assume_stationary_mesh,
                    )
                )
                if not geometry_matches:
                    vtk_reader.diagnostics.append(
                        "Local mesh geometry changed near the sampled surface; "
                        "disabled stationary interpolation-plan reuse."
                    )
                    stationary_plan = None
                    use_stationary_plan = False

            if stationary_plan is not None and direct_cell_values is not None:
                pressure, velocity = stationary_plan.interpolate_cell_values(
                    direct_cell_values,
                    time_name=step.name,
                )
            else:
                grid = vtk_reader.read(step)
                if stationary_plan is None and use_stationary_plan:
                    stationary_plan = _build_stationary_interpolation_plan(
                        grid,
                        positions,
                        time_name=step.name,
                        face_cache=face_cache,
                    )
                    remember_plan(stationary_plan)
                elif stationary_plan is not None:
                    current_cell_ids = stationary_plan.locate_surface_cells(grid)
                    assignment_matches = (
                        current_cell_ids is not None
                        and np.array_equal(
                            current_cell_ids,
                            stationary_plan.surface_cell_ids,
                        )
                    )
                    if not assignment_matches:
                        cached_plan = (
                            stationary_plan_cache.get(current_cell_ids.tobytes())
                            if current_cell_ids is not None
                            else None
                        )
                        if cached_plan is not None and cached_plan.geometry_matches(
                            grid
                        ):
                            stationary_plan = cached_plan
                            remember_plan(stationary_plan)
                        else:
                            vtk_reader.diagnostics.append(
                                "The local mesh geometry or VTK containing-cell "
                                "assignment changed near the sampled surface; rebuilt "
                                "the interpolation plan."
                            )
                            stationary_plan = _build_stationary_interpolation_plan(
                                grid,
                                positions,
                                time_name=step.name,
                                face_cache=face_cache,
                            )
                            remember_plan(stationary_plan)

                if stationary_plan is not None:
                    pressure, velocity = stationary_plan.interpolate(
                        grid,
                        pressure_field=pressure_field,
                        velocity_field=velocity_field,
                        time_name=step.name,
                    )
                else:
                    pressure, velocity = _openfoam_like_cell_point(
                        grid,
                        positions,
                        pressure_field=pressure_field,
                        velocity_field=velocity_field,
                        time_name=step.name,
                        face_cache=face_cache,
                    )
            output_file = write_frame(step, pressure, velocity)
            if validate_output:
                _read_surface_vtk(
                    output_file,
                    pressure_field=pressure_field,
                    velocity_field=velocity_field,
                    expected_cells=len(mesh.triangles),
                    expected_geometry_fingerprint=None,
                    include_geometry=False,
                    field_dtype=torch.float64,
                    read_attempts=1,
                )

        def sample_stationary_ascii_step(
            step: _CaseTime,
            plan: _StationaryInterpolationPlan,
        ) -> bool:
            geometry_matches, direct_cell_values = _read_stationary_ascii_cell_values(
                step,
                plan,
                pressure_field=pressure_field,
                velocity_field=velocity_field,
                region=region,
                mesh_points_static=bool(
                    getattr(vtk_reader, "mesh_points_static", False)
                ),
                assume_stationary_mesh=assume_stationary_mesh,
            )
            if not geometry_matches or direct_cell_values is None:
                return False
            pressure, velocity = plan.interpolate_cell_values(
                direct_cell_values,
                time_name=step.name,
            )
            write_frame(step, pressure, velocity)
            return True

        with tqdm(
            total=len(times),
            desc="Interpolating OpenFOAM fields",
            unit="time",
            disable=not progress,
        ) as progress_bar:
            sample_step(times[0], validate_output=True)
            progress_bar.update()
            remaining = times[1:]
            parallel_plan = stationary_plan
            can_sample_in_parallel = (
                workers > 1 and parallel_plan is not None and assume_stationary_mesh
            )
            if not can_sample_in_parallel:
                for step in remaining:
                    sample_step(step)
                    progress_bar.update()
            else:
                assert parallel_plan is not None
                fallback_steps: list[_CaseTime] = []
                max_pending = min(2 * workers, len(remaining))
                next_submit = 0
                next_result = 0
                pending: dict[int, Future[bool]] = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    while next_submit < len(remaining) and len(pending) < max_pending:
                        pending[next_submit] = executor.submit(
                            sample_stationary_ascii_step,
                            remaining[next_submit],
                            parallel_plan,
                        )
                        next_submit += 1
                    while next_result < len(remaining):
                        sampled = pending.pop(next_result).result()
                        if sampled:
                            progress_bar.update()
                        else:
                            fallback_steps.append(remaining[next_result])
                        next_result += 1
                        while (
                            next_submit < len(remaining) and len(pending) < max_pending
                        ):
                            pending[next_submit] = executor.submit(
                                sample_stationary_ascii_step,
                                remaining[next_submit],
                                parallel_plan,
                            )
                            next_submit += 1
                for step in fallback_steps:
                    sample_step(step)
                    progress_bar.update()

        completed_times = [step.output_name for step in times]
        output_signatures = _output_file_signatures(
            output_dir,
            surface_file=surface_file,
            times=times,
        )
        if output_signatures is None:
            raise RuntimeError("Sampled VTK outputs disappeared before publication.")
        try:
            final_signatures = _source_signatures(
                case_dir,
                times,
                pressure_field=pressure_field,
                velocity_field=velocity_field,
                region=region,
            )
        except OSError as error:
            raise RuntimeError(
                "OpenFOAM source files changed while sampling; no cache manifest "
                "was published. Retry after the case is stable."
            ) from error
        if final_signatures != signatures:
            raise RuntimeError(
                "OpenFOAM source files changed while sampling; no cache manifest "
                "was published. Retry after the case is stable."
            )

        _atomic_write_text(
            time_list_file, "".join(f"{name}\n" for name in completed_times)
        )
        manifest = {
            **payload,
            "cache_key": key,
            "surface_file": surface_file,
            "completed_times": completed_times,
            "output_signatures": output_signatures,
            "vtk_diagnostics": sorted(set(vtk_reader.diagnostics)),
        }
        _atomic_write_text(
            manifest_file,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    series_cache = (
        run_dir / "binary-cache" if cache_dir is None else Path(cache_dir).expanduser()
    )
    return OpenFOAMSurfaceSeries(
        output_dir,
        surface_file=surface_file,
        time_list=time_list_file,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        pressure_units=pressure_units,
        density_kg_m3=density_kg_m3,
        pressure_reference_pa=pressure_reference_pa,
        dtype=resolved_dtype,
        read_attempts=read_attempts,
        cache_dir=series_cache,
    )


__all__ = ["interpolate_openfoam_fields"]

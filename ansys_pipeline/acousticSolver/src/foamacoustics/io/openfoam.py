"""Read legacy VTK surface time series produced by OpenFOAM sampling."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch

from ..utils.validation import finite_float, positive_float, positive_int
from .surface import (
    DTypeName,
    INVALID_TIME_POLICIES as INVALID_TIME_POLICIES,
    InvalidTimePolicy as InvalidTimePolicy,
    PressureUnits as PressureUnits,
    ProgressCallback,
    SurfaceChunk,
    SurfaceFrame,
    chunks_from_frames,
    resolve_dtype,
    validate_invalid_time_policy,
)


POINTS_RE = re.compile(rb"^POINTS\s+(\d+)\s+\S+\s*$", re.MULTILINE)
POLYGONS_RE = re.compile(rb"^POLYGONS\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)
CELL_DATA_RE = re.compile(rb"^CELL_DATA\s+(\d+)\s*$", re.MULTILINE)
POINT_DATA_RE = re.compile(rb"^POINT_DATA\s+(\d+)\s*$", re.MULTILINE)

VTK_READ_ATTEMPTS = 4
VTK_READ_RETRY_DELAY_SECONDS = 0.25
CACHE_VERSION = 2


class VTKSectionError(ValueError):
    """A required legacy-VTK section or field is missing or truncated."""


OpenFOAMSurfaceFrame = SurfaceFrame


@dataclass(frozen=True, slots=True)
class _PointDataMapping:
    """Precomputed mapping used to average VTK point values onto panels."""

    point_indices: torch.Tensor
    cell_indices: torch.Tensor
    vertex_counts: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ParsedSurfaceVTK:
    points: torch.Tensor | None
    polygons: torch.Tensor | None
    pressure: torch.Tensor
    fluid_velocity: torch.Tensor | None
    cell_count: int
    geometry_fingerprint: bytes
    point_data_mapping: _PointDataMapping | None = None


@dataclass(frozen=True, slots=True)
class _FrameReadResult:
    index: int
    time_name: str
    absolute_time_s: float
    frame: OpenFOAMSurfaceFrame | None
    error: str | None


def _time_key(path: Path) -> tuple[float, str]:
    try:
        value = float(path.name)
    except ValueError as error:
        raise ValueError(f"Time directory is not numeric: {path}") from error
    if not math.isfinite(value):
        raise ValueError(f"Time directory is not finite: {path}")
    return value, path.name


def _time_directories(input_dir: Path, time_list: Path | None) -> list[Path]:
    input_dir = Path(input_dir)
    if time_list is not None:
        names = [
            line.strip()
            for line in Path(time_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        time_dirs = [
            path if (path := Path(name)).is_absolute() else input_dir / name
            for name in names
        ]
    else:
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)
        with os.scandir(input_dir) as entries:
            time_dirs = [
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ]

    time_dirs.sort(key=_time_key)
    if not time_dirs:
        raise ValueError("No time directories found.")

    numeric_times = [_time_key(path)[0] for path in time_dirs]
    duplicate_indices = [
        index
        for index in range(len(numeric_times) - 1)
        if numeric_times[index + 1] == numeric_times[index]
    ]
    if duplicate_indices:
        duplicate_names = [
            f"{time_dirs[index].name!r}/{time_dirs[index + 1].name!r}"
            for index in duplicate_indices
        ]
        raise ValueError(
            "Time directories contain duplicate numeric values: "
            + ", ".join(duplicate_names)
        )
    return time_dirs


def _source_file_signature(path: Path) -> tuple[int, int] | None:
    """Return a cheap cache-invalidating signature, or ``None`` if absent."""

    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _payload_start(text: bytes, match: re.Match[bytes]) -> int:
    start = text.find(b"\n", match.start())
    if start < 0:
        raise VTKSectionError("Malformed VTK section header.")
    return start + 1


def _required_match(
    pattern: re.Pattern[bytes],
    text: bytes,
    section: str,
    path: Path,
) -> re.Match[bytes]:
    match = pattern.search(text)
    if match is None:
        raise VTKSectionError(f"{path}: missing VTK {section} section.")
    return match


def _field_re(
    field_name: str,
    component_count: int,
    value_count: int,
) -> re.Pattern[bytes]:
    field = re.escape(field_name).encode("ascii")
    components = str(component_count).encode("ascii")
    count = str(value_count).encode("ascii")
    return re.compile(
        rb"^" + field + rb"\s+" + components + rb"\s+" + count + rb"\s+\S+\s*$",
        re.MULTILINE,
    )


def _read_field_array(
    text: bytes,
    *,
    path: Path,
    field_name: str,
    component_count: int,
    value_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    field_match = _field_re(field_name, component_count, value_count).search(text)
    if field_match is None:
        raise VTKSectionError(
            f"{path}: missing FIELD array {field_name!r} with {component_count} "
            f"component(s) and {value_count} values."
        )

    flat_count = component_count * value_count
    field_text = text[_payload_start(text, field_match) :]
    tokens = field_text.split(None, flat_count)
    values = torch.tensor([float(value) for value in tokens[:flat_count]], dtype=dtype)
    if values.numel() != flat_count:
        raise VTKSectionError(
            f"{path}: expected {flat_count} values for FIELD array "
            f"{field_name!r}, got {values.numel()}."
        )
    if not torch.isfinite(values).all().item():
        raise ValueError(
            f"{path}: FIELD array {field_name!r} contains non-finite values."
        )
    if component_count == 1:
        return values
    return values.reshape(value_count, component_count)


def _geometry_fingerprint(
    text: bytes,
    points_match: re.Match[bytes],
    data_start: int,
) -> bytes:
    geometry_bytes = text[points_match.start() : data_start]
    return hashlib.blake2b(geometry_bytes, digest_size=16).digest()


def _parse_geometry(
    text: bytes,
    *,
    path: Path,
    points_match: re.Match[bytes],
    polygons_match: re.Match[bytes],
    data_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_count = int(points_match.group(1))
    polygon_entry_count = int(polygons_match.group(2))

    points_text = text[_payload_start(text, points_match) : polygons_match.start()]
    points = torch.tensor(
        [float(value) for value in points_text.split()], dtype=torch.float64
    )
    if points.numel() != 3 * point_count:
        raise VTKSectionError(
            f"{path}: expected {3 * point_count} point coordinates, got {points.numel()}."
        )
    points = points.reshape(point_count, 3)

    polygons_text = text[_payload_start(text, polygons_match) : data_start]
    polygons = torch.tensor(
        [int(value) for value in polygons_text.split()], dtype=torch.int64
    )
    if polygons.numel() != polygon_entry_count:
        raise VTKSectionError(
            f"{path}: expected {polygon_entry_count} polygon entries, got "
            f"{polygons.numel()}."
        )
    return points, polygons


def _point_data_mapping(
    polygons: torch.Tensor,
    cell_count: int,
) -> _PointDataMapping:
    counts, _, point_indices, _ = _polygon_edges(polygons, cell_count)
    return _PointDataMapping(
        point_indices=point_indices,
        cell_indices=torch.repeat_interleave(torch.arange(cell_count), counts),
        vertex_counts=counts,
    )


def _point_values_to_cells(
    values: torch.Tensor,
    mapping: _PointDataMapping,
) -> torch.Tensor:
    selected = values.index_select(0, mapping.point_indices)
    output_shape = (mapping.vertex_counts.numel(), *values.shape[1:])
    output = torch.zeros(output_shape, dtype=values.dtype)
    output.index_add_(0, mapping.cell_indices, selected)
    divisor = mapping.vertex_counts.to(dtype=values.dtype)
    if values.ndim > 1:
        divisor = divisor[:, None]
    return output / divisor


def _vtk_data_sections(
    text: bytes,
    *,
    path: Path,
    point_count: int,
    cell_count: int,
) -> list[tuple[Literal["cell", "point"], bytes, int, int]]:
    """Return VTK field-data sections as location, bytes, count, and start."""

    matches: list[tuple[Literal["cell", "point"], re.Match[bytes], int]] = []
    cell_match = CELL_DATA_RE.search(text)
    if cell_match is not None:
        declared = int(cell_match.group(1))
        if declared != cell_count:
            raise VTKSectionError(
                f"{path}: POLYGONS count {cell_count} does not match "
                f"CELL_DATA count {declared}."
            )
        matches.append(("cell", cell_match, cell_count))
    point_match = POINT_DATA_RE.search(text)
    if point_match is not None:
        declared = int(point_match.group(1))
        if declared != point_count:
            raise VTKSectionError(
                f"{path}: POINTS count {point_count} does not match "
                f"POINT_DATA count {declared}."
            )
        matches.append(("point", point_match, point_count))
    if not matches:
        raise VTKSectionError(f"{path}: missing VTK CELL_DATA or POINT_DATA section.")

    matches.sort(key=lambda item: item[1].start())
    sections: list[tuple[Literal["cell", "point"], bytes, int, int]] = []
    for index, (location, match, value_count) in enumerate(matches):
        stop = matches[index + 1][1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            (location, text[match.start() : stop], value_count, match.start())
        )
    return sections


def _read_surface_fields(
    sections: list[tuple[Literal["cell", "point"], bytes, int, int]],
    *,
    path: Path,
    pressure_field: str,
    velocity_field: str | None,
    dtype: torch.dtype,
) -> tuple[Literal["cell", "point"], torch.Tensor, torch.Tensor | None]:
    for location, section, value_count, _ in sections:
        if _field_re(pressure_field, 1, value_count).search(section) is None:
            continue
        pressure = _read_field_array(
            section,
            path=path,
            field_name=pressure_field,
            component_count=1,
            value_count=value_count,
            dtype=dtype,
        )
        velocity = (
            _read_field_array(
                section,
                path=path,
                field_name=velocity_field,
                component_count=3,
                value_count=value_count,
                dtype=dtype,
            )
            if velocity_field is not None
            else None
        )
        return location, pressure, velocity
    locations = ", ".join(location.upper() + "_DATA" for location, *_ in sections)
    raise VTKSectionError(
        f"{path}: missing FIELD array {pressure_field!r} in {locations}."
    )


def _read_surface_vtk_once(
    path: Path,
    *,
    pressure_field: str,
    velocity_field: str | None,
    expected_cells: int | None,
    expected_geometry_fingerprint: bytes | None,
    include_geometry: bool,
    field_dtype: torch.dtype,
    point_data_mapping: _PointDataMapping | None,
) -> _ParsedSurfaceVTK:
    text = Path(path).read_bytes()
    points_match = _required_match(POINTS_RE, text, "POINTS", path)
    polygons_match = _required_match(POLYGONS_RE, text, "POLYGONS", path)
    point_count = int(points_match.group(1))
    cell_count = int(polygons_match.group(1))
    if expected_cells is not None and cell_count != expected_cells:
        raise ValueError(
            f"{path}: expected {expected_cells} cells, found {cell_count}."
        )

    sections = _vtk_data_sections(
        text,
        path=path,
        point_count=point_count,
        cell_count=cell_count,
    )
    data_start = min(section[3] for section in sections)
    fingerprint = _geometry_fingerprint(text, points_match, data_start)
    if (
        expected_geometry_fingerprint is not None
        and fingerprint != expected_geometry_fingerprint
    ):
        raise ValueError(
            f"{path}: surface geometry, polygon order, or topology differs from "
            "the first VTK frame."
        )

    location, pressure, fluid_velocity = _read_surface_fields(
        sections,
        path=path,
        pressure_field=pressure_field,
        velocity_field=velocity_field,
        dtype=field_dtype,
    )
    points: torch.Tensor | None = None
    polygons: torch.Tensor | None = None
    if include_geometry or (location == "point" and point_data_mapping is None):
        points, polygons = _parse_geometry(
            text,
            path=path,
            points_match=points_match,
            polygons_match=polygons_match,
            data_start=data_start,
        )
    if location == "point":
        mapping = point_data_mapping
        if mapping is None:
            if polygons is None:
                raise RuntimeError(
                    "POINT_DATA conversion requires polygon connectivity."
                )
            mapping = _point_data_mapping(polygons, cell_count)
        if int(mapping.point_indices.max()) >= point_count:
            raise ValueError(f"{path}: POINT_DATA mapping exceeds the VTK point count.")
        pressure = _point_values_to_cells(pressure, mapping)
        if fluid_velocity is not None:
            fluid_velocity = _point_values_to_cells(fluid_velocity, mapping)
        point_data_mapping = mapping
    return _ParsedSurfaceVTK(
        points=points,
        polygons=polygons,
        pressure=pressure,
        fluid_velocity=fluid_velocity,
        cell_count=cell_count,
        geometry_fingerprint=fingerprint,
        point_data_mapping=point_data_mapping,
    )


def _read_surface_vtk(
    path: Path,
    *,
    pressure_field: str,
    velocity_field: str | None,
    expected_cells: int | None,
    expected_geometry_fingerprint: bytes | None,
    include_geometry: bool,
    field_dtype: torch.dtype,
    point_data_mapping: _PointDataMapping | None = None,
    read_attempts: int = VTK_READ_ATTEMPTS,
) -> _ParsedSurfaceVTK:
    read_attempts = positive_int(read_attempts, "read_attempts")
    for attempt in range(read_attempts):
        try:
            return _read_surface_vtk_once(
                path,
                pressure_field=pressure_field,
                velocity_field=velocity_field,
                expected_cells=expected_cells,
                expected_geometry_fingerprint=expected_geometry_fingerprint,
                include_geometry=include_geometry,
                field_dtype=field_dtype,
                point_data_mapping=point_data_mapping,
            )
        except (OSError, VTKSectionError):
            if attempt + 1 == read_attempts:
                raise
            time.sleep(VTK_READ_RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")


def _polygon_edges(
    polygons: torch.Tensor,
    cell_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.empty(cell_count, dtype=torch.int64)
    starts = torch.empty(cell_count, dtype=torch.int64)
    cursor = 0
    flat_size = 0
    for cell_index in range(cell_count):
        if cursor >= polygons.numel():
            raise ValueError("Polygon connectivity ended unexpectedly.")
        count = int(polygons[cursor])
        if count < 3:
            raise ValueError(f"Cell {cell_index} has fewer than three vertices.")
        counts[cell_index] = count
        starts[cell_index] = flat_size
        cursor += 1 + count
        flat_size += count
    if cursor != polygons.numel():
        raise ValueError("Polygon connectivity contains trailing entries.")

    vertex_indices = torch.empty(flat_size, dtype=torch.int64)
    next_vertex_indices = torch.empty(flat_size, dtype=torch.int64)
    cursor = 0
    flat_cursor = 0
    for count in counts:
        cell_indices = polygons[cursor + 1 : cursor + 1 + count]
        end = flat_cursor + count
        vertex_indices[flat_cursor:end] = cell_indices
        next_vertex_indices[flat_cursor : end - 1] = cell_indices[1:]
        next_vertex_indices[end - 1] = cell_indices[0]
        cursor += 1 + count
        flat_cursor = end

    return counts, starts, vertex_indices, next_vertex_indices


def _surface_geometry(
    points: torch.Tensor,
    polygons: torch.Tensor,
    cell_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts, starts, vertex_indices, next_vertex_indices = _polygon_edges(
        polygons,
        cell_count,
    )
    vertices = points[vertex_indices]

    cell_indices = torch.repeat_interleave(torch.arange(cell_count), counts)
    centers = torch.zeros((cell_count, 3), dtype=points.dtype)
    centers.index_add_(0, cell_indices, vertices)
    centers /= counts[:, None]
    area_vectors = torch.zeros_like(centers)
    area_vectors.index_add_(
        0,
        cell_indices,
        torch.linalg.cross(vertices, points[next_vertex_indices]),
    )
    area_vectors *= 0.5

    # Orient every face away from the mean surface centre.
    surface_center = centers.mean(dim=0)
    inward = torch.einsum("ij,ij->i", centers - surface_center, area_vectors) <= 0.0
    area_vectors[inward] *= -1.0

    areas = torch.linalg.vector_norm(area_vectors, dim=1)
    if torch.any(areas <= 0.0).item():
        raise ValueError("At least one VTK cell has zero area.")
    normals = area_vectors / areas[:, None]
    return centers, normals, areas, area_vectors


class OpenFOAMSurfaceSeries:
    """Ordered, bounded-memory reader for an OpenFOAM VTK surface series.

    Static geometry is parsed once. Each iteration then yields pressure and,
    when requested, fluid velocity for one time, while a raw geometry
    fingerprint verifies that VTK cell order and topology remain identical to
    the first frame.
    """

    def __init__(
        self,
        input_dir: str | Path,
        *,
        surface_file: str = "Cylx15.vtk",
        time_list: str | Path | None = None,
        pressure_field: str = "p",
        velocity_field: str | None = "U",
        pressure_units: PressureUnits = "kinematic",
        density_kg_m3: float = 1.225,
        pressure_reference_pa: float = 0.0,
        dtype: DTypeName | torch.dtype = torch.float64,
        max_times: int | None = None,
        read_attempts: int = VTK_READ_ATTEMPTS,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Index a legacy-VTK surface time series.

        Args:
            input_dir: Directory containing numeric OpenFOAM time directories.
            surface_file: VTK filename present in every time directory.
            time_list: Optional text file listing time directories explicitly.
            pressure_field: Scalar VTK FIELD array containing pressure.
            velocity_field: Optional three-component VTK FIELD array containing
                inertial-frame fluid velocity. Pass ``None`` for pressure-only
                impermeable calculations.
            pressure_units: Pressure field units: pascals or kinematic pressure.
            density_kg_m3: Positive density used to convert kinematic pressure.
            pressure_reference_pa: Reference pressure subtracted after conversion.
            dtype: Dynamic-field dtype, ``float32`` or ``float64``.
            max_times: Optional maximum number of ordered frames, at least two.
            read_attempts: VTK read attempts before reporting an invalid frame.
            cache_dir: Optional memory-mapped binary-cache directory.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            ValueError: If configuration, time directories, or geometry is invalid.
        """
        self.input_dir = Path(input_dir).resolve()
        self.surface_file = str(surface_file)
        self.pressure_field = str(pressure_field)
        self.velocity_field = None if velocity_field is None else str(velocity_field)
        self.pressure_units = pressure_units
        self.density_kg_m3 = positive_float(density_kg_m3, "density_kg_m3")
        self.pressure_reference_pa = finite_float(
            pressure_reference_pa,
            "pressure_reference_pa",
        )
        self.dtype = resolve_dtype(dtype)
        self.read_attempts = positive_int(read_attempts, "read_attempts")
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.is_moving_surface = False
        self.uses_surface_kinematics = False
        self._cached_pressure: torch.Tensor | None = None
        self._cached_velocity: torch.Tensor | None = None
        self._point_data_mapping: _PointDataMapping | None = None

        if pressure_units not in {"pa", "kinematic"}:
            raise ValueError("pressure_units must be either 'pa' or 'kinematic'.")
        self.pressure_scale_to_pa = (
            1.0 if pressure_units == "pa" else self.density_kg_m3
        )
        time_list_path = None if time_list is None else Path(time_list)
        time_dirs = _time_directories(self.input_dir, time_list_path)
        if max_times is not None:
            max_times = positive_int(max_times, "max_times")
            if max_times < 2:
                raise ValueError("max_times must be at least two.")
            time_dirs = time_dirs[:max_times]
        if len(time_dirs) < 2:
            raise ValueError("At least two VTK time directories are required.")

        self.time_directories = tuple(time_dirs)
        self.time_names = tuple(path.name for path in time_dirs)
        self.absolute_times_s = torch.tensor(
            [float(path.name) for path in time_dirs],
            dtype=torch.float64,
        )
        self.source_times_s = self.absolute_times_s - self.absolute_times_s[0]
        self._source_signatures = tuple(
            _source_file_signature(path / self.surface_file)
            for path in self.time_directories
        )

        if self.cache_dir is not None and self._load_cache():
            return

        first_path = time_dirs[0] / self.surface_file
        first = _read_surface_vtk(
            first_path,
            pressure_field=self.pressure_field,
            velocity_field=self.velocity_field,
            expected_cells=None,
            expected_geometry_fingerprint=None,
            include_geometry=True,
            field_dtype=self.dtype,
            read_attempts=self.read_attempts,
        )
        if first.points is None or first.polygons is None:
            raise RuntimeError("The first VTK frame did not provide geometry.")

        self.points_m = first.points
        self.polygons = first.polygons
        (
            self.face_centers_m,
            self.face_normals,
            self.face_areas_m2,
            self.outward_area_vectors_m2,
        ) = _surface_geometry(first.points, first.polygons, first.cell_count)
        self.face_ids = torch.arange(first.cell_count, dtype=torch.int32)
        self.geometry_fingerprint = first.geometry_fingerprint
        self._point_data_mapping = first.point_data_mapping
        self._first_frame = self._make_frame(0, first)

    @property
    def cache_enabled(self) -> bool:
        """Whether dynamic fields are currently backed by the binary cache."""

        return self._cached_pressure is not None

    def _cache_metadata(self) -> dict[str, object]:
        return {
            "version": CACHE_VERSION,
            "input_dir": str(self.input_dir),
            "surface_file": self.surface_file,
            "pressure_field": self.pressure_field,
            "velocity_field": self.velocity_field,
            "pressure_units": self.pressure_units,
            "density_kg_m3": self.density_kg_m3,
            "pressure_reference_pa": self.pressure_reference_pa,
            "dtype": str(self.dtype),
            "time_names": self.time_names,
            "absolute_times_s": self.absolute_times_s,
            "panel_count": self.panel_count,
            "geometry_fingerprint": self.geometry_fingerprint,
            "source_signatures": self._source_signatures,
        }

    def _cache_matches(self, metadata: dict[str, object]) -> bool:
        cached_time_names = metadata.get("time_names")
        cached_source_signatures = metadata.get("source_signatures")
        if not isinstance(cached_time_names, (list, tuple)):
            return False
        if not isinstance(cached_source_signatures, (list, tuple)):
            return False
        return (
            metadata.get("version") == CACHE_VERSION
            and metadata.get("input_dir") == str(self.input_dir)
            and metadata.get("surface_file") == self.surface_file
            and metadata.get("pressure_field") == self.pressure_field
            and metadata.get("velocity_field") == self.velocity_field
            and metadata.get("pressure_units") == self.pressure_units
            and metadata.get("density_kg_m3") == self.density_kg_m3
            and metadata.get("pressure_reference_pa") == self.pressure_reference_pa
            and metadata.get("dtype") == str(self.dtype)
            and tuple(cached_time_names) == self.time_names
            and tuple(cached_source_signatures) == self._source_signatures
            and torch.equal(
                torch.as_tensor(metadata.get("absolute_times_s")),
                self.absolute_times_s,
            )
        )

    def _load_cache(self) -> bool:
        if self.cache_dir is None:
            return False
        metadata_path = self.cache_dir / "metadata.pt"
        geometry_path = self.cache_dir / "geometry.pt"
        pressure_path = self.cache_dir / "pressure.bin"
        velocity_path = self.cache_dir / "velocity.bin"
        if not all(
            path.is_file()
            for path in (
                metadata_path,
                geometry_path,
                pressure_path,
                velocity_path,
            )
        ):
            return False
        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
            if not isinstance(metadata, dict):
                return False
            if not self._cache_matches(metadata):
                return False
            geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
            if not isinstance(geometry, dict):
                return False
            panel_count = int(metadata["panel_count"])
            frame_count = len(self.time_names)
            element_size = torch.empty((), dtype=self.dtype).element_size()
            expected_pressure_bytes = frame_count * panel_count * element_size
            expected_velocity_bytes = (
                0 if self.velocity_field is None else 3 * expected_pressure_bytes
            )
            if pressure_path.stat().st_size != expected_pressure_bytes:
                return False
            if velocity_path.stat().st_size != expected_velocity_bytes:
                return False
            self.points_m = geometry["points_m"]
            self.polygons = geometry["polygons"]
            self.face_centers_m = geometry["face_centers_m"]
            self.face_normals = geometry["face_normals"]
            self.face_areas_m2 = geometry["face_areas_m2"]
            self.outward_area_vectors_m2 = geometry["outward_area_vectors_m2"]
            self.face_ids = torch.arange(panel_count, dtype=torch.int32)
            self.geometry_fingerprint = metadata["geometry_fingerprint"]
            self._cached_pressure = torch.from_file(
                str(pressure_path),
                shared=False,
                size=frame_count * panel_count,
                dtype=self.dtype,
            ).reshape(frame_count, panel_count)
            self._cached_velocity = (
                None
                if self.velocity_field is None
                else torch.from_file(
                    str(velocity_path),
                    shared=False,
                    size=frame_count * panel_count * 3,
                    dtype=self.dtype,
                ).reshape(frame_count, panel_count, 3)
            )
            self._first_frame = self._cached_frame(0)
            return True
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            return False

    def _cached_frame(self, index: int) -> OpenFOAMSurfaceFrame:
        if self._cached_pressure is None:
            raise RuntimeError("The binary surface cache is not loaded.")
        return OpenFOAMSurfaceFrame(
            index=index,
            time_name=self.time_names[index],
            absolute_time_s=float(self.absolute_times_s[index]),
            source_time_s=float(self.source_times_s[index]),
            pressure_pa=self._cached_pressure[index],
            fluid_velocity_m_s=(
                None if self._cached_velocity is None else self._cached_velocity[index]
            ),
        )

    def build_cache(
        self,
        *,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> None:
        """Build and activate a memory-mapped cache of all dynamic fields.

        Args:
            workers: Number of bounded VTK-reader worker threads.
            invalid_time_policy: Raise on invalid frames or interpolate them.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Successful-frame interval between callbacks.

        Raises:
            ValueError: If no cache directory was configured or options are invalid.
            RuntimeError: If the completed cache cannot be loaded.
        """

        if self.cache_dir is None:
            raise ValueError("Set cache_dir when constructing the surface series.")
        if self.cache_enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pressure_tmp = self.cache_dir / "pressure.bin.tmp"
        velocity_tmp = self.cache_dir / "velocity.bin.tmp"
        metadata_tmp = self.cache_dir / "metadata.pt.tmp"
        geometry_tmp = self.cache_dir / "geometry.pt.tmp"
        try:
            with (
                pressure_tmp.open("wb") as pressure_file,
                velocity_tmp.open("wb") as velocity_file,
            ):
                for frame in self.iter_frames(
                    workers=workers,
                    invalid_time_policy=invalid_time_policy,
                    progress_callback=progress_callback,
                    progress_every=progress_every,
                ):
                    if (
                        self.velocity_field is not None
                        and frame.fluid_velocity_m_s is None
                    ):
                        raise RuntimeError(
                            "An OpenFOAM frame did not provide fluid velocity."
                        )
                    pressure_file.write(
                        frame.pressure_pa.contiguous().numpy().tobytes()
                    )
                    if frame.fluid_velocity_m_s is not None:
                        velocity_file.write(
                            frame.fluid_velocity_m_s.contiguous().numpy().tobytes()
                        )
            geometry = {
                "points_m": self.points_m,
                "polygons": self.polygons,
                "face_centers_m": self.face_centers_m,
                "face_normals": self.face_normals,
                "face_areas_m2": self.face_areas_m2,
                "outward_area_vectors_m2": self.outward_area_vectors_m2,
            }
            torch.save(geometry, geometry_tmp)
            torch.save(self._cache_metadata(), metadata_tmp)
            os.replace(pressure_tmp, self.cache_dir / "pressure.bin")
            os.replace(velocity_tmp, self.cache_dir / "velocity.bin")
            os.replace(geometry_tmp, self.cache_dir / "geometry.pt")
            os.replace(metadata_tmp, self.cache_dir / "metadata.pt")
        finally:
            for path in (pressure_tmp, velocity_tmp, metadata_tmp, geometry_tmp):
                path.unlink(missing_ok=True)
        if not self._load_cache():
            raise RuntimeError(
                "The binary surface cache was written but could not be loaded."
            )

    def __len__(self) -> int:
        return len(self.time_directories)

    @property
    def panel_count(self) -> int:
        """Number of faces in each sampled surface frame."""

        return int(self.face_ids.numel())

    def _make_frame(
        self,
        index: int,
        parsed: _ParsedSurfaceVTK,
        *,
        repair_note: str | None = None,
    ) -> OpenFOAMSurfaceFrame:
        pressure_pa = parsed.pressure.to(dtype=self.dtype).clone()
        pressure_pa *= self.pressure_scale_to_pa
        pressure_pa -= self.pressure_reference_pa
        return OpenFOAMSurfaceFrame(
            index=index,
            time_name=self.time_names[index],
            absolute_time_s=float(self.absolute_times_s[index]),
            source_time_s=float(self.source_times_s[index]),
            pressure_pa=pressure_pa,
            fluid_velocity_m_s=(
                None
                if parsed.fluid_velocity is None
                else parsed.fluid_velocity.to(dtype=self.dtype)
            ),
            repair_note=repair_note,
        )

    def _read_frame_result(self, index: int) -> _FrameReadResult:
        time_dir = self.time_directories[index]
        try:
            parsed = _read_surface_vtk(
                time_dir / self.surface_file,
                pressure_field=self.pressure_field,
                velocity_field=self.velocity_field,
                expected_cells=self.panel_count,
                expected_geometry_fingerprint=self.geometry_fingerprint,
                include_geometry=False,
                field_dtype=self.dtype,
                point_data_mapping=self._point_data_mapping,
                read_attempts=self.read_attempts,
            )
            frame = self._make_frame(index, parsed)
            return _FrameReadResult(
                index=index,
                time_name=time_dir.name,
                absolute_time_s=float(self.absolute_times_s[index]),
                frame=frame,
                error=None,
            )
        except (OSError, VTKSectionError) as error:
            return _FrameReadResult(
                index=index,
                time_name=time_dir.name,
                absolute_time_s=float(self.absolute_times_s[index]),
                frame=None,
                error=str(error),
            )

    def _ordered_results(self, workers: int) -> Iterator[_FrameReadResult]:
        if self.cache_enabled:
            for index in range(len(self)):
                yield _FrameReadResult(
                    index=index,
                    time_name=self.time_names[index],
                    absolute_time_s=float(self.absolute_times_s[index]),
                    frame=self._cached_frame(index),
                    error=None,
                )
            return
        yield _FrameReadResult(
            index=0,
            time_name=self.time_names[0],
            absolute_time_s=float(self.absolute_times_s[0]),
            frame=self._first_frame,
            error=None,
        )
        if workers == 1:
            for index in range(1, len(self)):
                yield self._read_frame_result(index)
            return

        max_pending = min(2 * workers, len(self) - 1)
        next_submit = 1
        next_yield = 1
        pending: dict[int, Future[_FrameReadResult]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while next_submit < len(self) and len(pending) < max_pending:
                pending[next_submit] = executor.submit(
                    self._read_frame_result,
                    next_submit,
                )
                next_submit += 1
            while next_yield < len(self):
                future = pending.pop(next_yield)
                yield future.result()
                next_yield += 1
                while next_submit < len(self) and len(pending) < max_pending:
                    pending[next_submit] = executor.submit(
                        self._read_frame_result,
                        next_submit,
                    )
                    next_submit += 1

    def _interpolated_frame(
        self,
        result: _FrameReadResult,
        left: OpenFOAMSurfaceFrame,
        right: OpenFOAMSurfaceFrame,
    ) -> OpenFOAMSurfaceFrame:
        denominator = right.absolute_time_s - left.absolute_time_s
        alpha = (
            0.0
            if denominator == 0.0
            else (result.absolute_time_s - left.absolute_time_s) / denominator
        )
        pressure = (1.0 - alpha) * left.pressure_pa + alpha * right.pressure_pa
        if left.fluid_velocity_m_s is None and right.fluid_velocity_m_s is None:
            velocity = None
        elif left.fluid_velocity_m_s is None or right.fluid_velocity_m_s is None:
            raise RuntimeError(
                "OpenFOAM interpolation received inconsistent velocity fields."
            )
        else:
            velocity = (
                1.0 - alpha
            ) * left.fluid_velocity_m_s + alpha * right.fluid_velocity_m_s
        note = (
            f"Interpolated invalid VTK frame from {left.time_name} to "
            f"{right.time_name}: {result.error}"
        )
        return OpenFOAMSurfaceFrame(
            index=result.index,
            time_name=result.time_name,
            absolute_time_s=result.absolute_time_s,
            source_time_s=result.absolute_time_s - float(self.absolute_times_s[0]),
            pressure_pa=pressure.to(dtype=self.dtype),
            fluid_velocity_m_s=(
                None if velocity is None else velocity.to(dtype=self.dtype)
            ),
            repair_note=note,
        )

    def iter_chunks(
        self,
        *,
        chunk_size: int,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[SurfaceChunk]:
        """Yield contiguous source arrays, slicing the binary cache when active.

        Args:
            chunk_size: Maximum number of contiguous frames per yielded chunk.
            workers: Number of bounded VTK-reader worker threads.
            invalid_time_policy: Raise on invalid VTK frames or interpolate them.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Successful-frame interval between callbacks.

        Yields:
            Ordered source chunks. Cached chunks are zero-copy tensor views;
            uncached frames are stacked into equivalent tensors.

        Raises:
            TypeError: If an integer option is not an integer.
            ValueError: If an integer option or policy is invalid.
            VTKSectionError: If a VTK frame is invalid under the error policy.
        """

        chunk_size = positive_int(chunk_size, "chunk_size")
        workers = positive_int(workers, "workers")
        progress_every = positive_int(progress_every, "progress_every")
        validate_invalid_time_policy(invalid_time_policy)

        if not self.cache_enabled:
            yield from chunks_from_frames(
                self.iter_frames(
                    workers=workers,
                    invalid_time_policy=invalid_time_policy,
                    progress_callback=progress_callback,
                    progress_every=progress_every,
                ),
                chunk_size,
            )
            return

        if self._cached_pressure is None:
            raise RuntimeError("The binary pressure cache is not loaded.")
        total = len(self)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            yield SurfaceChunk(
                start_index=start,
                absolute_times_s=self.absolute_times_s[start:stop],
                pressure_pa=self._cached_pressure[start:stop],
                fluid_velocity_m_s=(
                    None
                    if self._cached_velocity is None
                    else self._cached_velocity[start:stop]
                ),
            )
            if progress_callback is not None:
                for completed in range(start + 1, stop + 1):
                    if completed % progress_every == 0 or completed == total:
                        progress_callback(
                            completed,
                            total,
                            "Read "
                            f"{completed}/{total} cached VTK frames "
                            f"({self.time_names[completed - 1]})",
                        )

    def iter_frames(
        self,
        *,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[OpenFOAMSurfaceFrame]:
        """Yield frames in numeric time order with bounded threaded prefetch.

        Args:
            workers: Number of VTK-reader worker threads.
            invalid_time_policy: Raise on invalid VTK frames or interpolate them.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Successful-frame interval between callbacks.

        Yields:
            Surface frames in their original numeric time order.

        Raises:
            TypeError: If integer options are not integers.
            ValueError: If integer options or ``invalid_time_policy`` are invalid.
            VTKSectionError: If a VTK frame is invalid under the error policy.
        """

        workers = positive_int(workers, "workers")
        progress_every = positive_int(progress_every, "progress_every")
        validate_invalid_time_policy(invalid_time_policy)

        previous_valid: OpenFOAMSurfaceFrame | None = None
        invalid_results: list[_FrameReadResult] = []
        emitted = 0
        total = len(self)
        for result in self._ordered_results(workers):
            if result.frame is None:
                if invalid_time_policy == "error":
                    raise VTKSectionError(result.error)
                invalid_results.append(result)
                continue

            frame = result.frame
            if invalid_results:
                if previous_valid is None:
                    previous_valid = frame
                for invalid in invalid_results:
                    repaired = self._interpolated_frame(
                        invalid,
                        previous_valid,
                        frame,
                    )
                    emitted += 1
                    if progress_callback is not None:
                        progress_callback(emitted, total, repaired.repair_note or "")
                    yield repaired
                invalid_results.clear()

            emitted += 1
            if progress_callback is not None and (
                emitted % progress_every == 0 or emitted == total
            ):
                progress_callback(
                    emitted,
                    total,
                    f"Read {emitted}/{total} VTK frames ({frame.time_name})",
                )
            yield frame
            previous_valid = frame

        if invalid_results:
            if previous_valid is None:
                raise ValueError(
                    "No valid VTK frames were available for interpolation."
                )
            for invalid in invalid_results:
                repaired = self._interpolated_frame(
                    invalid,
                    previous_valid,
                    previous_valid,
                )
                emitted += 1
                if progress_callback is not None:
                    progress_callback(emitted, total, repaired.repair_note or "")
                yield repaired

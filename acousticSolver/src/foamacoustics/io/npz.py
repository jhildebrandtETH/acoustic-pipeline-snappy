"""Reader for compact and legacy NumPy surface-pressure archives."""

from __future__ import annotations

import ast
import math
import struct
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

from ..utils.validation import finite_float, positive_float, positive_int
from .surface import (
    DTypeName,
    InvalidTimePolicy,
    ProgressCallback,
    SurfaceChunk,
    SurfaceFrame,
    resolve_dtype,
    validate_invalid_time_policy,
)


NPZPressureUnits = Literal["pa", "kinematic", "auto"]
VectorInput: TypeAlias = Sequence[float] | npt.NDArray[np.generic] | torch.Tensor
FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntArray: TypeAlias = npt.NDArray[np.int64]

_PRESSURE_FIELDS = ("p_surface_pred", "p_surface_true")
_NORMAL_MINIMUM_NORM = 1.0e-12
_NPY_HEADER_MAX_SIZE = 10_000


def _read_exact_into(
    stream: IO[bytes],
    buffer: bytearray,
    description: str,
) -> None:
    view = memoryview(buffer)
    offset = 0
    while offset < len(buffer):
        chunk = stream.read(len(buffer) - offset)
        if not chunk:
            raise ValueError(f"Legacy data.npy is truncated in {description}.")
        view[offset : offset + len(chunk)] = chunk
        offset += len(chunk)


def _read_exact_bytes(
    stream: IO[bytes],
    size: int,
    description: str,
) -> bytes:
    buffer = bytearray(size)
    _read_exact_into(stream, buffer, description)
    return bytes(buffer)


def _read_npy_v3_header(
    stream: IO[bytes],
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    length_bytes = _read_exact_bytes(stream, 4, "array header length")
    header_length = struct.unpack("<I", length_bytes)[0]
    if header_length > _NPY_HEADER_MAX_SIZE:
        raise ValueError("Legacy data.npy header exceeds the safe 10000-byte limit.")
    header_bytes = _read_exact_bytes(stream, header_length, "array header")
    try:
        header = ast.literal_eval(header_bytes.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("Legacy data.npy has an invalid version 3 header.") from error
    if not isinstance(header, dict):
        raise ValueError("Legacy data.npy header must be a dictionary.")
    expected_keys = {"descr", "fortran_order", "shape"}
    if set(header) != expected_keys:
        raise ValueError(
            "Legacy data.npy header must contain only descr, fortran_order, and shape."
        )

    raw_shape = header["shape"]
    if not isinstance(raw_shape, tuple) or not all(
        type(dimension) is int and dimension >= 0 for dimension in raw_shape
    ):
        raise ValueError("Legacy data.npy header contains an invalid shape.")
    raw_order = header["fortran_order"]
    if not isinstance(raw_order, bool):
        raise ValueError("Legacy data.npy header contains an invalid storage order.")
    try:
        dtype = np.lib.format.descr_to_dtype(header["descr"])
    except (TypeError, ValueError) as error:
        raise ValueError("Legacy data.npy header contains an invalid dtype.") from error
    return raw_shape, raw_order, dtype


def _read_npy_header(
    stream: IO[bytes],
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=_NPY_HEADER_MAX_SIZE,
            )
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=_NPY_HEADER_MAX_SIZE,
            )
        if version == (3, 0):
            return _read_npy_v3_header(stream)
    except (EOFError, UnicodeError, ValueError) as error:
        raise ValueError("Legacy data.npy has an invalid NPY header.") from error
    raise ValueError(
        f"Legacy data.npy uses unsupported NPY format version {version!r}."
    )


def _legacy_array_layout(
    stream: IO[bytes],
    member: zipfile.ZipInfo,
    *,
    time_count: int,
    column_count: int,
) -> tuple[np.dtype[Any], int, int]:
    """Validate the legacy data array and return dtype, faces, and row size."""

    shape, fortran_order, data_dtype = _read_npy_header(stream)
    if len(shape) != 3:
        raise ValueError("data must have 3 dimensions.")
    if not np.issubdtype(data_dtype, np.number):
        raise ValueError("data must be numeric.")
    if fortran_order:
        raise ValueError("Legacy data must be stored in C order.")
    if shape[0] != time_count:
        raise ValueError("Legacy data must contain one table for every entry in times.")
    if shape[2] != column_count:
        raise ValueError("Legacy data column count does not match columns.")
    expected_payload_size = math.prod(shape) * data_dtype.itemsize
    if member.file_size - stream.tell() != expected_payload_size:
        raise ValueError("Legacy data.npy payload size does not match its header.")
    face_count = shape[1]
    return data_dtype, face_count, face_count * column_count


def _decode_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} must contain strings.")


def _string_vector(value: npt.NDArray[np.generic], name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional string array.")
    return tuple(_decode_text(item, name) for item in array.tolist())


def _scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    array = np.asarray(archive[key])
    if array.size != 1:
        raise ValueError(f"NPZ metadata {key!r} must be scalar.")
    return _decode_text(array.reshape(-1)[0], key)


def _scalar_float(
    archive: np.lib.npyio.NpzFile,
    key: str,
    *,
    positive: bool = False,
) -> float:
    array = np.asarray(archive[key])
    if array.size != 1:
        raise ValueError(f"NPZ metadata {key!r} must be scalar.")
    try:
        value = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"NPZ metadata {key!r} must be numeric.") from error
    if positive:
        return positive_float(value, key)
    return finite_float(value, key)


def _vector3(value: VectorInput, name: str) -> FloatArray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric three-vector.") from error
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(result, dtype=np.float64, copy=True)


def _metadata_vector3(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> FloatArray:
    try:
        return _vector3(np.asarray(archive[key]), key)
    except ValueError as error:
        raise ValueError(f"Invalid NPZ metadata {key!r}: {error}") from error


def _column_indices(
    columns: tuple[str, ...],
    required: Sequence[str],
    *,
    table_name: str,
) -> list[int]:
    if len(set(columns)) != len(columns):
        raise ValueError(f"{table_name} contains duplicate column names.")
    lookup = {column: index for index, column in enumerate(columns)}
    missing = [column for column in required if column not in lookup]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: "
            + ", ".join(repr(column) for column in missing)
        )
    return [lookup[column] for column in required]


def _numeric_array(
    value: npt.NDArray[np.generic],
    name: str,
    *,
    ndim: int,
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, dtype=np.float64, copy=True)


def _face_ids(
    value: npt.NDArray[np.generic] | None,
    face_count: int,
) -> IntArray:
    if value is None:
        return np.arange(face_count, dtype=np.int64)
    array = np.asarray(value)
    if array.shape != (face_count,):
        raise ValueError(f"face ids must have shape ({face_count},).")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("face ids must be integer-valued.")
    if np.issubdtype(array.dtype, np.integer):
        result = np.asarray(array, dtype=np.int64)
    else:
        try:
            numeric = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("face ids must be integer-valued.") from error
        if (
            not np.isfinite(numeric).all()
            or not np.equal(numeric, np.rint(numeric)).all()
        ):
            raise ValueError("face ids must be finite integer values.")
        result = np.asarray(numeric, dtype=np.int64)
    if np.unique(result).size != face_count:
        raise ValueError("face ids must be unique.")
    return np.array(result, dtype=np.int64, copy=True)


def _torch_float(array: FloatArray, dtype: torch.dtype) -> torch.Tensor:
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    contiguous = np.array(array, dtype=numpy_dtype, order="C", copy=True)
    return torch.from_numpy(contiguous)


def _torch_int(array: IntArray) -> torch.Tensor:
    return torch.from_numpy(np.array(array, dtype=np.int64, order="C", copy=True))


def _resolve_pressure_units(
    archive: np.lib.npyio.NpzFile,
    requested: NPZPressureUnits,
    *,
    compact: bool,
) -> Literal["pa", "kinematic"]:
    if requested not in {"pa", "kinematic", "auto"}:
        raise ValueError("pressure_units must be 'pa', 'kinematic', or 'auto'.")
    if requested != "auto":
        return requested
    if not compact:
        raise ValueError(
            "pressure_units='auto' is unavailable for legacy NPZ archives "
            "(data/columns/times) because they have no reliable pressure-unit "
            "metadata. Pass pressure_units='pa' or 'kinematic' explicitly."
        )
    if "pressure_units" not in archive.files:
        raise ValueError(
            "pressure_units='auto' requires compact NPZ metadata named "
            "'pressure_units'. Pass 'pa' or 'kinematic' explicitly."
        )
    metadata = _scalar_text(archive, "pressure_units").strip().lower()
    normalized = metadata.replace(" ", "").replace("^", "")
    pa_units = {
        "pa",
        "pascal",
        "pascals",
        "pa_gauge",
        "pa_absolute",
    }
    kinematic_units = {
        "kinematic",
        "m2/s2",
        "m2/s2_kinematic",
        "m2/s2_kinematic_gauge",
        "m2s-2",
    }
    if normalized in pa_units:
        return "pa"
    if normalized in kinematic_units:
        return "kinematic"
    raise ValueError(
        "Unrecognized compact NPZ pressure_units metadata "
        f"{metadata!r}; pass pressure_units='pa' or 'kinematic' explicitly."
    )


def _detect_compact_schema(keys: set[str]) -> bool:
    """Identify exactly one supported archive schema."""

    compact = {
        "surface_geometry_ref",
        "surface_geometry_columns",
        "time",
    }.issubset(keys)
    legacy = {"data", "columns", "times"}.issubset(keys)
    if compact and legacy:
        raise ValueError(
            "NPZ archive ambiguously contains both compact and legacy surface schemas."
        )
    if not compact and not legacy:
        raise ValueError(
            "Unsupported NPZ surface schema. Expected compact keys "
            "surface_geometry_ref/surface_geometry_columns/time or "
            "legacy keys data/columns/times."
        )
    return compact


def _read_times(
    archive: np.lib.npyio.NpzFile,
    *,
    compact: bool,
    max_times: int | None,
) -> tuple[FloatArray, FloatArray, int]:
    """Read ordered archive times and select the requested leading frames."""

    time_key = "time" if compact else "times"
    all_times = _numeric_array(
        np.asarray(archive[time_key]),
        time_key,
        ndim=1,
    )
    if all_times.size < 2:
        raise ValueError("At least two NPZ surface times are required.")
    if np.any(np.diff(all_times) <= 0.0):
        raise ValueError("NPZ surface times must be strictly increasing.")
    retained_count = (
        int(all_times.size)
        if max_times is None
        else min(max_times, int(all_times.size))
    )
    selected_times = np.array(
        all_times[:retained_count],
        dtype=np.float64,
        copy=True,
    )
    return all_times, selected_times, retained_count


def _reference_geometry(
    geometry: FloatArray,
    columns: tuple[str, ...],
    *,
    flip_normals: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract and validate face centres, unit normals, and areas."""

    if geometry.shape[0] < 1:
        raise ValueError("NPZ surface geometry must contain at least one face.")
    center_indices = _column_indices(
        columns,
        ("x", "y", "z"),
        table_name="surface geometry columns",
    )
    normal_indices = _column_indices(
        columns,
        ("normal_x", "normal_y", "normal_z"),
        table_name="surface geometry columns",
    )
    area_index = _column_indices(
        columns,
        ("area",),
        table_name="surface geometry columns",
    )[0]
    centers = np.array(geometry[:, center_indices], dtype=np.float64, copy=True)
    normals = np.array(geometry[:, normal_indices], dtype=np.float64, copy=True)
    areas = np.array(geometry[:, area_index], dtype=np.float64, copy=True)
    if not (
        np.isfinite(centers).all()
        and np.isfinite(normals).all()
        and np.isfinite(areas).all()
    ):
        raise ValueError(
            "NPZ reference surface geometry must contain only finite values."
        )
    if np.any(areas <= 0.0):
        raise ValueError("NPZ surface face areas must be positive.")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms <= _NORMAL_MINIMUM_NORM):
        raise ValueError("NPZ surface face normals must be nonzero.")
    normals /= normal_norms[:, None]
    if flip_normals:
        normals *= -1.0
    return centers, normals, areas


def _converted_pressure(
    pressure: FloatArray,
    *,
    field_name: str,
    full_time_count: int,
    retained_time_count: int,
    face_count: int,
    compact: bool,
    scale_to_pa: float,
    reference_pa: float,
) -> FloatArray:
    """Validate, select, and convert one archived pressure field to pascals."""

    expected_time_count = full_time_count if compact else retained_time_count
    if pressure.shape != (expected_time_count, face_count):
        raise ValueError(
            f"Pressure field {field_name!r} must have shape "
            f"({expected_time_count}, {face_count}); got {pressure.shape}."
        )
    selected = (
        np.array(
            pressure[:retained_time_count],
            dtype=np.float64,
            copy=True,
        )
        if compact
        else pressure
    )
    selected *= scale_to_pa
    selected -= reference_pa
    if not np.isfinite(selected).all():
        raise ValueError("Converted NPZ pressure contains non-finite values.")
    return selected


def _read_time_names(
    archive: np.lib.npyio.NpzFile,
    selected_times: FloatArray,
    *,
    full_time_count: int,
) -> tuple[str, ...]:
    """Read optional source-time labels or format deterministic defaults."""

    if "time_names" not in archive.files:
        return tuple(
            np.format_float_positional(value, trim="-") for value in selected_times
        )
    all_names = _string_vector(np.asarray(archive["time_names"]), "time_names")
    if len(all_names) != full_time_count:
        raise ValueError("time_names must contain one entry per NPZ surface time.")
    return all_names[: selected_times.size]


def _resolve_density(
    archive: np.lib.npyio.NpzFile,
    keys: set[str],
    explicit: float | None,
) -> float:
    """Resolve density from an explicit value, archive metadata, or the default."""

    if explicit is not None:
        return positive_float(explicit, "density_kg_m3")
    if "rho" in keys:
        return _scalar_float(archive, "rho", positive=True)
    return 1.225


def _resolve_rotation_center(
    archive: np.lib.npyio.NpzFile,
    keys: set[str],
    explicit: VectorInput | None,
) -> FloatArray:
    """Resolve the fixed rotation centre from arguments or archive metadata."""

    if explicit is not None:
        return _vector3(explicit, "rotation_center_m")
    if "rotation_origin" in keys:
        return _metadata_vector3(archive, "rotation_origin")
    return np.zeros(3, dtype=np.float64)


def _rpm_from_metadata(
    archive: np.lib.npyio.NpzFile,
    keys: set[str],
    omega: FloatArray | None,
    *,
    explicit_omega: bool,
) -> float:
    """Resolve RPM metadata while preserving explicit-vector precedence."""

    radians_to_rpm = 60.0 / (2.0 * math.pi)
    if explicit_omega:
        if omega is None:
            raise RuntimeError("Explicit angular velocity was not resolved.")
        return float(np.linalg.norm(omega)) * radians_to_rpm
    if "rpm" in keys:
        return abs(_scalar_float(archive, "rpm"))
    if omega is not None:
        return float(np.linalg.norm(omega)) * radians_to_rpm
    if "omega" in keys and np.asarray(archive["omega"]).size == 1:
        return abs(_scalar_float(archive, "omega")) * radians_to_rpm
    return 0.0


def _optional_scalar_text(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> str | None:
    """Return optional scalar string metadata."""

    return _scalar_text(archive, key) if key in archive.files else None


class NPZSurfaceSeries:
    """Surface-pressure series backed by one NumPy ``.npz`` archive.

    The compact schema stores one reference geometry and separate pressure
    arrays. The legacy schema stores a time-by-face table; only its first
    geometry is used, because subsequent geometry is reconstructed exactly from
    the configured rigid rotation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pressure_field: str | None = None,
        pressure_units: NPZPressureUnits = "auto",
        density_kg_m3: float | None = None,
        pressure_reference_pa: float = 0.0,
        rotation_center_m: VectorInput | None = None,
        omega_rad_s: VectorInput | None = None,
        rotation_reference_time_s: float | None = None,
        flip_normals: bool = False,
        dtype: DTypeName | torch.dtype = torch.float64,
        max_times: int | None = None,
    ) -> None:
        """Read and validate a rotating surface-pressure archive.

        Args:
            path: Path to a compact or legacy ``.npz`` archive.
            pressure_field: ``p_surface_pred`` or ``p_surface_true``.
            pressure_units: Pressure units, or ``"auto"`` for compact metadata.
            density_kg_m3: Density used for kinematic-pressure conversion.
            pressure_reference_pa: Pascal reference subtracted after conversion.
            rotation_center_m: Fixed center of rotation in metres.
            omega_rad_s: Angular-velocity vector in radians per second.
            rotation_reference_time_s: Time represented by the stored geometry.
            flip_normals: Reverse all declared reference normals.
            dtype: Floating-point dtype of geometry and dynamic fields.
            max_times: Optional number of leading frames to retain.
        """
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.dtype = resolve_dtype(dtype)
        self.pressure_reference_pa = finite_float(
            pressure_reference_pa,
            "pressure_reference_pa",
        )
        if max_times is not None:
            max_times = positive_int(max_times, "max_times")
            if max_times < 2:
                raise ValueError("max_times must be at least two.")

        loaded = np.load(self.path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(f"Expected an NPZ archive: {self.path}")
        with loaded as archive:
            keys = set(archive.files)
            compact = _detect_compact_schema(keys)
            self.schema = "compact" if compact else "legacy"
            resolved_units = _resolve_pressure_units(
                archive,
                pressure_units,
                compact=compact,
            )
            self.pressure_units: Literal["pa", "kinematic"] = resolved_units

            density = _resolve_density(archive, keys, density_kg_m3)
            self.density_kg_m3 = density

            all_times, selected_times, retained_count = _read_times(
                archive,
                compact=compact,
                max_times=max_times,
            )

            geometry, columns, pressure, self.pressure_field = self._read_payload(
                archive,
                path=self.path,
                compact=compact,
                pressure_field=pressure_field,
                full_time_count=int(all_times.size),
                retained_time_count=retained_count,
            )
            geometry_columns = (
                _string_vector(
                    np.asarray(archive["surface_geometry_columns"]),
                    "surface_geometry_columns",
                )
                if compact
                else columns
            )
            centers, normals, areas = _reference_geometry(
                geometry,
                geometry_columns,
                flip_normals=flip_normals,
            )
            face_count = int(centers.shape[0])

            face_id_key = "face_id" if compact else "face_ids"
            ids_value = (
                np.asarray(archive[face_id_key]) if face_id_key in keys else None
            )
            ids = _face_ids(ids_value, face_count)
            selected_pressure = _converted_pressure(
                pressure,
                field_name=self.pressure_field,
                full_time_count=int(all_times.size),
                retained_time_count=retained_count,
                face_count=face_count,
                compact=compact,
                scale_to_pa=1.0 if resolved_units == "pa" else density,
                reference_pa=self.pressure_reference_pa,
            )
            time_names = _read_time_names(
                archive,
                selected_times,
                full_time_count=int(all_times.size),
            )

            rotation_center = _resolve_rotation_center(
                archive,
                keys,
                rotation_center_m,
            )

            omega = self._resolve_omega(
                archive,
                explicit=omega_rad_s,
            )
            rpm = _rpm_from_metadata(
                archive,
                keys,
                omega,
                explicit_omega=omega_rad_s is not None,
            )
            self.geometry_frame = _optional_scalar_text(archive, "geometry_frame")
            self.normal_convention = _optional_scalar_text(
                archive,
                "normal_convention",
            )

        self.time_names = tuple(time_names)
        self._absolute_times_np = selected_times
        self._source_times_np = selected_times - selected_times[0]
        self.absolute_times_s = torch.from_numpy(
            np.array(selected_times, dtype=np.float64, copy=True)
        )
        self.source_times_s = torch.from_numpy(
            np.array(self._source_times_np, dtype=np.float64, copy=True)
        )

        self._face_centers_np = centers
        self._face_normals_np = normals
        self._pressure_pa_np = np.asarray(
            selected_pressure,
            dtype=np.float32 if self.dtype == torch.float32 else np.float64,
        )
        self._pressure_pa = torch.from_numpy(self._pressure_pa_np)
        self._rotation_center_np = rotation_center
        self._omega_np = omega
        self._reference_radius_np = centers - rotation_center[None, :]
        self._omega_norm = 0.0 if omega is None else float(np.linalg.norm(omega))
        self._omega_axis_np = (
            None
            if omega is None or self._omega_norm == 0.0
            else omega / self._omega_norm
        )

        self.face_centers_m = _torch_float(centers, self.dtype)
        self.face_normals = _torch_float(normals, self.dtype)
        self.face_areas_m2 = _torch_float(areas, self.dtype)
        outward_area_vectors = normals * areas[:, None]
        self.outward_area_vectors_m2 = _torch_float(
            outward_area_vectors,
            self.dtype,
        )
        self.face_ids = _torch_int(ids)
        self.rotation_center_m = _torch_float(rotation_center, self.dtype)
        self.omega_rad_s = None if omega is None else _torch_float(omega, self.dtype)
        self.is_moving_surface = self._omega_norm > 0.0
        self.uses_surface_kinematics = omega is not None
        self.rpm = rpm
        self.rotation_reference_time_s = (
            float(selected_times[0])
            if rotation_reference_time_s is None
            else finite_float(
                rotation_reference_time_s,
                "rotation_reference_time_s",
            )
        )

    @staticmethod
    def _selected_pressure_field(
        columns: tuple[str, ...],
        *,
        compact: bool,
        pressure_field: str | None,
        archive_keys: set[str],
    ) -> str:
        if pressure_field is not None and pressure_field not in _PRESSURE_FIELDS:
            raise ValueError(
                "pressure_field must be 'p_surface_pred' or 'p_surface_true'."
            )
        if pressure_field is not None:
            return pressure_field
        if compact:
            if "p_surface_pred" in archive_keys:
                return "p_surface_pred"
            return "p_surface_true"
        if "p_surface_pred" in columns:
            return "p_surface_pred"
        if "p_surface_true" in columns:
            return "p_surface_true"
        if "p_surface_pred" in archive_keys:
            return "p_surface_pred"
        return "p_surface_true"

    @classmethod
    def _read_payload(
        cls,
        archive: np.lib.npyio.NpzFile,
        *,
        path: Path,
        compact: bool,
        pressure_field: str | None,
        full_time_count: int,
        retained_time_count: int,
    ) -> tuple[FloatArray, tuple[str, ...], FloatArray, str]:
        if compact:
            columns = _string_vector(
                np.asarray(archive["surface_geometry_columns"]),
                "surface_geometry_columns",
            )
            geometry = _numeric_array(
                np.asarray(archive["surface_geometry_ref"]),
                "surface_geometry_ref",
                ndim=2,
            )
            if geometry.shape[1] != len(columns):
                raise ValueError(
                    "surface_geometry_ref column count does not match "
                    "surface_geometry_columns."
                )
            selected_field = cls._selected_pressure_field(
                columns,
                compact=True,
                pressure_field=pressure_field,
                archive_keys=set(archive.files),
            )
            if selected_field not in archive.files:
                hint = (
                    " Specify pressure_field='p_surface_true'."
                    if selected_field == "p_surface_pred"
                    and "p_surface_true" in archive.files
                    else ""
                )
                raise ValueError(
                    f"Compact NPZ archive has no pressure field "
                    f"{selected_field!r}.{hint}"
                )
            pressure = _numeric_array(
                np.asarray(archive[selected_field]),
                selected_field,
                ndim=2,
            )
            return geometry, columns, pressure, selected_field

        columns = _string_vector(
            np.asarray(archive["columns"]),
            "columns",
        )
        selected_field = cls._selected_pressure_field(
            columns,
            compact=False,
            pressure_field=pressure_field,
            archive_keys=set(archive.files),
        )
        pressure_index = _column_indices(
            columns,
            (selected_field,),
            table_name="legacy columns",
        )[0]
        geometry, pressure = cls._stream_legacy_data(
            path,
            full_time_count=full_time_count,
            retained_time_count=retained_time_count,
            column_count=len(columns),
            pressure_index=pressure_index,
            pressure_field=selected_field,
        )
        return geometry, columns, pressure, selected_field

    @staticmethod
    def _stream_legacy_data(
        path: Path,
        *,
        full_time_count: int,
        retained_time_count: int,
        column_count: int,
        pressure_index: int,
        pressure_field: str,
    ) -> tuple[FloatArray, FloatArray]:
        with zipfile.ZipFile(path, mode="r") as zip_archive:
            members = [
                member
                for member in zip_archive.infolist()
                if member.filename == "data.npy"
            ]
            if len(members) != 1:
                raise ValueError(
                    "Legacy NPZ archive must contain exactly one data.npy member."
                )
            member_info = members[0]
            with zip_archive.open(member_info, mode="r") as stream:
                data_dtype, face_count, row_element_count = _legacy_array_layout(
                    stream,
                    member_info,
                    time_count=full_time_count,
                    column_count=column_count,
                )
                row_buffer = bytearray(row_element_count * data_dtype.itemsize)
                pressure = np.empty(
                    (retained_time_count, face_count),
                    dtype=np.float64,
                )
                geometry: FloatArray | None = None
                for time_index in range(retained_time_count):
                    _read_exact_into(
                        stream,
                        row_buffer,
                        f"surface table {time_index}",
                    )
                    table = np.frombuffer(
                        row_buffer,
                        dtype=data_dtype,
                        count=row_element_count,
                    ).reshape(face_count, column_count)
                    if time_index == 0:
                        geometry = np.array(
                            table,
                            dtype=np.float64,
                            copy=True,
                        )
                    pressure_row = np.asarray(
                        table[:, pressure_index],
                        dtype=np.float64,
                    )
                    if not np.isfinite(pressure_row).all():
                        raise ValueError(
                            f"Pressure field {pressure_field!r} contains "
                            "non-finite values."
                        )
                    pressure[time_index] = pressure_row

        if geometry is None:
            raise ValueError("Legacy data contains no reference geometry.")
        return geometry, pressure

    @staticmethod
    def _resolve_omega(
        archive: np.lib.npyio.NpzFile,
        *,
        explicit: VectorInput | None,
    ) -> FloatArray | None:
        if explicit is not None:
            return _vector3(explicit, "omega_rad_s")
        if "omega" not in archive.files:
            return None
        raw_omega = np.asarray(archive["omega"])
        if raw_omega.shape == (3,):
            return _metadata_vector3(archive, "omega")
        if raw_omega.size != 1:
            raise ValueError(
                "NPZ omega metadata must be scalar; pass omega_rad_s as an "
                "explicit three-vector."
            )
        if "rotation_axis" not in archive.files:
            return None
        magnitude = _scalar_float(archive, "omega")
        axis = _metadata_vector3(archive, "rotation_axis")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= _NORMAL_MINIMUM_NORM:
            raise ValueError("NPZ rotation_axis metadata must be nonzero.")
        return np.asarray(magnitude * axis / axis_norm, dtype=np.float64)

    def __len__(self) -> int:
        return int(self._absolute_times_np.size)

    @property
    def panel_count(self) -> int:
        """Number of faces in each archived surface frame."""

        return int(self.face_ids.numel())

    def _geometry_at(
        self,
        absolute_time_s: float,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray | None,
        FloatArray | None,
    ]:
        if self._omega_np is None:
            return (
                np.array(self._face_centers_np, dtype=np.float64, copy=True),
                np.array(self._face_normals_np, dtype=np.float64, copy=True),
                None,
                None,
            )
        radius = self._reference_radius_np
        if self._omega_axis_np is None:
            rotated_radius = np.array(radius, dtype=np.float64, copy=True)
            normals = np.array(
                self._face_normals_np,
                dtype=np.float64,
                copy=True,
            )
        else:
            angle = math.remainder(
                self._omega_norm * (absolute_time_s - self.rotation_reference_time_s),
                2.0 * math.pi,
            )
            cosine = math.cos(angle)
            sine = math.sin(angle)
            axis = self._omega_axis_np
            rotated_radius = (
                radius * cosine
                + np.cross(axis, radius) * sine
                + np.outer(radius @ axis, axis) * (1.0 - cosine)
            )
            normals = (
                self._face_normals_np * cosine
                + np.cross(axis, self._face_normals_np) * sine
                + np.outer(self._face_normals_np @ axis, axis) * (1.0 - cosine)
            )
        centers = rotated_radius + self._rotation_center_np[None, :]
        surface_velocity = np.cross(self._omega_np, rotated_radius)
        surface_acceleration = np.cross(self._omega_np, surface_velocity)
        return centers, normals, surface_velocity, surface_acceleration

    def iter_chunks(
        self,
        *,
        chunk_size: int,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[SurfaceChunk]:
        """Yield direct pressure-array slices for chunked solver evaluation.

        Args:
            chunk_size: Maximum number of contiguous frames per yielded chunk.
            workers: Reader-worker count accepted for ``SurfaceSeries`` parity.
                NPZ array slicing itself is synchronous.
            invalid_time_policy: Invalid-frame policy accepted for reader parity;
                validated NPZ arrays contain no repairable frames.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Frame interval between progress callbacks.

        Yields:
            Contiguous chunks backed directly by the loaded pressure array.

        Raises:
            TypeError: If an integer option is not an integer.
            ValueError: If an integer option or policy is invalid.
        """

        chunk_size = positive_int(chunk_size, "chunk_size")
        positive_int(workers, "workers")
        progress_every = positive_int(progress_every, "progress_every")
        validate_invalid_time_policy(invalid_time_policy)

        total = len(self)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            yield SurfaceChunk(
                start_index=start,
                absolute_times_s=self.absolute_times_s[start:stop],
                pressure_pa=self._pressure_pa[start:stop],
                fluid_velocity_m_s=None,
            )
            if progress_callback is not None:
                for completed in range(start + 1, stop + 1):
                    if completed % progress_every == 0 or completed == total:
                        progress_callback(
                            completed,
                            total,
                            f"Read {completed}/{total} NPZ frames "
                            f"({self.time_names[completed - 1]})",
                        )

    def iter_frames(
        self,
        *,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[SurfaceFrame]:
        """Yield pressure plus any resolvable rigid-body kinematics.

        Args:
            workers: Reader-worker count accepted for ``SurfaceSeries`` parity.
                NPZ frame generation itself is synchronous.
            invalid_time_policy: Invalid-frame policy accepted for reader parity;
                validated NPZ arrays contain no repairable frames.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Frame interval between progress callbacks.

        Yields:
            Frames in archive order, including rotated geometry and kinematics
            when angular-velocity metadata is available.

        Raises:
            TypeError: If an integer option is not an integer.
            ValueError: If an integer option or policy is invalid.
        """
        positive_int(workers, "workers")
        progress_every = positive_int(progress_every, "progress_every")
        validate_invalid_time_policy(invalid_time_policy)

        total = len(self)
        for index in range(total):
            absolute_time_s = float(self._absolute_times_np[index])
            (
                centers,
                normals,
                surface_velocity,
                surface_acceleration,
            ) = self._geometry_at(absolute_time_s)
            frame = SurfaceFrame(
                index=index,
                time_name=self.time_names[index],
                absolute_time_s=absolute_time_s,
                source_time_s=float(self._source_times_np[index]),
                pressure_pa=_torch_float(
                    self._pressure_pa_np[index],
                    self.dtype,
                ),
                fluid_velocity_m_s=None,
                face_centers_m=_torch_float(centers, self.dtype),
                face_normals=_torch_float(normals, self.dtype),
                face_areas_m2=self.face_areas_m2,
                surface_velocity_m_s=(
                    None
                    if surface_velocity is None
                    else _torch_float(
                        surface_velocity,
                        self.dtype,
                    )
                ),
                surface_acceleration_m_s2=(
                    None
                    if surface_acceleration is None
                    else _torch_float(
                        surface_acceleration,
                        self.dtype,
                    )
                ),
            )
            completed = index + 1
            if progress_callback is not None and (
                completed % progress_every == 0 or completed == total
            ):
                progress_callback(
                    completed,
                    total,
                    f"Read {completed}/{total} NPZ frames ({self.time_names[index]})",
                )
            yield frame

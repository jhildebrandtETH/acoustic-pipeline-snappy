"""Shared surface-source models used by readers and the acoustic solver."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Callable, Literal, Protocol, cast

import torch

from ..utils.validation import positive_int


PressureUnits = Literal["pa", "kinematic"]
InvalidTimePolicy = Literal["error", "interpolate"]
DTypeName = Literal["float32", "float64"]
ProgressCallback = Callable[[int, int, str], None]
INVALID_TIME_POLICIES: tuple[InvalidTimePolicy, ...] = ("error", "interpolate")


def resolve_dtype(dtype: DTypeName | torch.dtype) -> torch.dtype:
    """Resolve a supported dtype name or Torch dtype.

    Args:
        dtype: ``"float32"``, ``"float64"``, or the corresponding Torch dtype.

    Returns:
        The normalized :class:`torch.dtype`.

    Raises:
        ValueError: If ``dtype`` is not a supported floating-point dtype.
    """

    resolved = (
        {"float32": torch.float32, "float64": torch.float64}.get(dtype)
        if isinstance(dtype, str)
        else dtype
    )
    if resolved not in {torch.float32, torch.float64}:
        raise ValueError(
            "dtype must be 'float32', 'float64', torch.float32, or torch.float64."
        )
    return resolved


def validate_invalid_time_policy(policy: InvalidTimePolicy) -> None:
    """Reject unsupported invalid-frame handling policies."""

    if policy not in INVALID_TIME_POLICIES:
        choices = ", ".join(repr(choice) for choice in INVALID_TIME_POLICIES)
        raise ValueError(f"invalid_time_policy must be one of: {choices}")


@dataclass(frozen=True, slots=True)
class SurfaceFrame:
    """Face-sampled source fields and optional moving geometry at one time.

    ``fluid_velocity_m_s`` is the inertial-frame fluid velocity that supplies
    permeable FW-H source terms.
    ``surface_velocity_m_s`` and ``surface_acceleration_m_s2`` describe source
    motion. They are deliberately separate because a moving permeable surface
    can require both.

    Attributes:
        index: Zero-based frame index in the source series.
        time_name: Reader-specific name of the source time.
        absolute_time_s: Absolute source time in seconds.
        source_time_s: Time in seconds relative to the first source frame.
        pressure_pa: Face pressure with shape ``(faces,)``.
        fluid_velocity_m_s: Optional fluid velocity with shape ``(faces, 3)``.
        repair_note: Description of interpolation used to repair an invalid frame.
        face_centers_m: Optional time-dependent centres with shape ``(faces, 3)``.
        face_normals: Optional time-dependent unit normals with shape ``(faces, 3)``.
        face_areas_m2: Optional time-dependent areas with shape ``(faces,)``.
        surface_velocity_m_s: Optional source velocity with shape ``(faces, 3)``.
        surface_acceleration_m_s2: Optional source acceleration with shape
            ``(faces, 3)``.
    """

    index: int
    time_name: str
    absolute_time_s: float
    source_time_s: float
    pressure_pa: torch.Tensor
    fluid_velocity_m_s: torch.Tensor | None
    repair_note: str | None = None
    face_centers_m: torch.Tensor | None = None
    face_normals: torch.Tensor | None = None
    face_areas_m2: torch.Tensor | None = None
    surface_velocity_m_s: torch.Tensor | None = None
    surface_acceleration_m_s2: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class SurfaceChunk:
    """Contiguous source fields for a range of surface times.

    Readers with array-backed storage can expose slices directly through this
    model, avoiding one Python object and one stacking copy per source frame.
    Optional geometry and motion fields are needed only when rigid-body motion
    cannot be reconstructed from series metadata.

    Attributes:
        start_index: Index of the first frame in the source series.
        absolute_times_s: Absolute times with shape ``(times,)``.
        pressure_pa: Face pressure with shape ``(times, faces)``.
        fluid_velocity_m_s: Optional fluid velocity with shape
            ``(times, faces, 3)``.
        face_centers_m: Optional time-dependent centres.
        face_normals: Optional time-dependent normals.
        face_areas_m2: Optional time-dependent areas.
        surface_velocity_m_s: Optional source velocity.
        surface_acceleration_m_s2: Optional source acceleration.
    """

    start_index: int
    absolute_times_s: torch.Tensor
    pressure_pa: torch.Tensor
    fluid_velocity_m_s: torch.Tensor | None
    face_centers_m: torch.Tensor | None = None
    face_normals: torch.Tensor | None = None
    face_areas_m2: torch.Tensor | None = None
    surface_velocity_m_s: torch.Tensor | None = None
    surface_acceleration_m_s2: torch.Tensor | None = None

    @property
    def frame_count(self) -> int:
        """Number of source frames in this chunk."""

        return int(self.absolute_times_s.numel())


class SurfaceSeries(Protocol):
    """Structural interface consumed by :class:`foamacoustics.F1ASolver`."""

    density_kg_m3: float
    dtype: torch.dtype
    absolute_times_s: torch.Tensor
    source_times_s: torch.Tensor
    face_centers_m: torch.Tensor
    face_normals: torch.Tensor
    face_areas_m2: torch.Tensor
    face_ids: torch.Tensor

    def __len__(self) -> int: ...

    @property
    def panel_count(self) -> int:
        """Number of faces in each source frame."""
        ...

    def iter_frames(
        self,
        *,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[SurfaceFrame]:
        """Yield source frames in increasing index order."""
        ...


class _SurfaceChunkIterator(Protocol):
    def __call__(
        self,
        *,
        chunk_size: int,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> Iterator[SurfaceChunk]: ...


def _stack_optional_frame_field(
    frames: Sequence[SurfaceFrame],
    name: str,
) -> torch.Tensor | None:
    values = [getattr(frame, name) for frame in frames]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            f"Surface frames must either all provide {name} or all omit it."
        )
    return torch.stack(
        [cast(torch.Tensor, value) for value in values],
        dim=0,
    )


def surface_chunk_from_frames(frames: Sequence[SurfaceFrame]) -> SurfaceChunk:
    """Stack a non-empty, contiguous sequence of frames into one source chunk."""

    if not frames:
        raise ValueError("A surface chunk requires at least one frame.")
    start_index = frames[0].index
    expected_indices = range(start_index, start_index + len(frames))
    if any(
        frame.index != expected for frame, expected in zip(frames, expected_indices)
    ):
        raise ValueError("Surface frames in a chunk must have contiguous indices.")
    return SurfaceChunk(
        start_index=start_index,
        absolute_times_s=torch.tensor(
            [frame.absolute_time_s for frame in frames],
            dtype=torch.float64,
        ),
        pressure_pa=torch.stack([frame.pressure_pa for frame in frames], dim=0),
        fluid_velocity_m_s=_stack_optional_frame_field(
            frames,
            "fluid_velocity_m_s",
        ),
        face_centers_m=_stack_optional_frame_field(frames, "face_centers_m"),
        face_normals=_stack_optional_frame_field(frames, "face_normals"),
        face_areas_m2=_stack_optional_frame_field(frames, "face_areas_m2"),
        surface_velocity_m_s=_stack_optional_frame_field(
            frames,
            "surface_velocity_m_s",
        ),
        surface_acceleration_m_s2=_stack_optional_frame_field(
            frames,
            "surface_acceleration_m_s2",
        ),
    )


def chunks_from_frames(
    frames: Iterable[SurfaceFrame],
    chunk_size: int,
) -> Iterator[SurfaceChunk]:
    """Stack an ordered frame stream into bounded, contiguous chunks."""

    size = positive_int(chunk_size, "chunk_size")
    for frame_batch in batched(frames, size):
        yield surface_chunk_from_frames(frame_batch)


def iter_surface_chunks(
    series: SurfaceSeries,
    *,
    chunk_size: int,
    workers: int = 1,
    invalid_time_policy: InvalidTimePolicy = "error",
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1,
) -> Iterator[SurfaceChunk]:
    """Use a reader's bulk path when available, otherwise stack its frames."""

    bulk_iterator = cast(
        _SurfaceChunkIterator | None,
        getattr(series, "iter_chunks", None),
    )
    if bulk_iterator is not None:
        yield from bulk_iterator(
            chunk_size=chunk_size,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
            progress_callback=progress_callback,
            progress_every=progress_every,
        )
        return

    yield from chunks_from_frames(
        series.iter_frames(
            workers=workers,
            invalid_time_policy=invalid_time_policy,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ),
        chunk_size,
    )

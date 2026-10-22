"""Torch implementations of fixed- and moving-surface Farassat 1A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

import torch

from ..io.openfoam import (
    OpenFOAMSurfaceSeries,
)
from ..io.surface import (
    DTypeName,
    InvalidTimePolicy,
    PressureUnits,
    ProgressCallback,
    SurfaceChunk,
    SurfaceSeries,
    iter_surface_chunks,
)
from ..utils.validation import (
    finite_float,
    nonnegative_float,
    positive_float,
    positive_int,
)
from .result import F1AResult


FOUR_PI = 4.0 * math.pi
FOAM_GREAT = 1.0e30

PropagationAlgorithm = Literal["auto", "uniform", "nonuniform"]
TensorLike = Sequence[float] | Sequence[Sequence[float]] | torch.Tensor
_FixedPropagationState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _constant_spacing(values: torch.Tensor) -> float | None:
    differences = torch.diff(values)
    if differences.numel() == 0 or torch.any(differences <= 0.0).item():
        return None
    spacing = float(differences[0])
    scale = max(1.0, float(torch.max(torch.abs(values))))
    tolerance = 64.0 * torch.finfo(values.dtype).eps * scale
    if torch.all(torch.abs(differences - spacing) <= tolerance).item():
        return spacing
    return None


def _resolve_device(value: str | torch.device | None) -> torch.device:
    """Resolve and validate a supported Torch execution device."""

    device = torch.device("cpu" if value is None else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Torch installation cannot access CUDA."
        )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must resolve to a CPU or CUDA device.")
    return device


@dataclass(frozen=True, slots=True)
class _TorchUniformDelays:
    face_order: torch.Tensor
    output_offsets_per_face: torch.Tensor
    grouped_output_offsets: torch.Tensor | None
    faces_per_group: torch.Tensor | None
    weight_at_offset: torch.Tensor
    weight_at_next_offset: torch.Tensor
    initial_weight_at_offset: torch.Tensor
    maximum_output_offset: int


@dataclass(frozen=True, slots=True)
class _TorchPackedUniformDelays:
    packed_face_indices: torch.Tensor | None
    group_id_per_face: torch.Tensor
    group_observer: torch.Tensor
    grouped_output_offsets: torch.Tensor
    weight_at_offset: torch.Tensor
    weight_at_next_offset: torch.Tensor
    initial_weight_at_offset: torch.Tensor
    maximum_output_offset: int


@dataclass(frozen=True, slots=True)
class _FixedObserverCoefficients:
    thickness: torch.Tensor
    loading_rate: torch.Tensor
    loading: torch.Tensor


class _TorchFiniteDifferenceDerivative:
    """Stateful finite-difference derivative for chunked tensor streams."""

    def __init__(
        self,
        *,
        adjust_time_step: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.adjust_time_step = bool(adjust_time_step)
        self.device = device
        self.dtype = dtype
        self.sample_count = 0
        self.previous_time_s = torch.zeros((), device=device, dtype=dtype)
        self.delta_time_s = torch.full(
            (),
            FOAM_GREAT,
            device=device,
            dtype=dtype,
        )
        self.old: torch.Tensor | None = None
        self.old_old: torch.Tensor | None = None

    def step_chunk(
        self,
        values: torch.Tensor,
        absolute_times_s: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError(
                "Derivative values must have shape (times, faces, fields)."
            )
        if absolute_times_s.shape != (values.shape[0],):
            raise ValueError("Derivative times do not match the chunk length.")

        chunk_count = values.shape[0]
        delta_times = torch.empty(
            chunk_count,
            device=self.device,
            dtype=self.dtype,
        )
        first_delta = absolute_times_s[0] - self.previous_time_s
        delta_times[0] = torch.where(
            first_delta > 0.0,
            first_delta,
            self.delta_time_s,
        )
        if chunk_count > 1:
            delta_times[1:] = torch.diff(absolute_times_s)

        prefix_old = torch.zeros_like(values[0]) if self.old is None else self.old
        if self.adjust_time_step:
            derivative = values.clone()
            derivative[0].sub_(prefix_old)
            if chunk_count > 1:
                derivative[1:].sub_(values[:-1])
            derivative.div_(delta_times[:, None, None])
        else:
            prefix_old_old = (
                torch.zeros_like(values[0]) if self.old_old is None else self.old_old
            )
            extended = torch.cat(
                (
                    prefix_old_old[None, ...],
                    prefix_old[None, ...],
                    values,
                ),
                dim=0,
            )
            previous = extended[1 : chunk_count + 1]
            previous_previous = extended[:chunk_count]
            derivative = (values - previous) / delta_times[:, None, None]
            global_indices = torch.arange(
                self.sample_count,
                self.sample_count + chunk_count,
                device=self.device,
            )
            bdf2 = (3.0 * values - 4.0 * previous + previous_previous) / (
                2.0 * delta_times[:, None, None]
            )
            use_bdf2 = global_indices >= 2
            derivative = torch.where(use_bdf2[:, None, None], bdf2, derivative)

        previous_old = self.old
        if chunk_count >= 2:
            self.old_old = values[-2].detach().clone()
        elif previous_old is not None:
            self.old_old = previous_old.detach().clone()
        self.old = values[-1].detach().clone()
        self.previous_time_s = absolute_times_s[-1].detach()
        self.delta_time_s = delta_times[-1].detach()
        self.sample_count += chunk_count
        return derivative


class F1ASolver:
    """Farassat 1A solver with independent surface and motion models.

    ``permeable`` selects the FW-H source formulation, while ``moving_surface``
    selects fixed or time-dependent source geometry. These choices are
    independent of the reader that supplied the pressure and velocity fields.
    Moving surfaces include source Mach/Doppler terms and time-dependent
    retarded arrival times. Quadrupole volume sources and mean-flow convection
    are neglected. The permeable reduction assumes constant source and ambient
    density and uses pressure-only, inviscid surface loading.
    """

    def __init__(
        self,
        series: SurfaceSeries,
        *,
        rpm: float,
        permeable: bool = True,
        moving_surface: bool = False,
        rotation_center_m: Sequence[float] | torch.Tensor | None = None,
        omega_rad_s: Sequence[float] | torch.Tensor | None = None,
        rotation_reference_time_s: float | None = None,
        density_kg_m3: float | None = None,
        sound_speed_m_s: float = 343.0,
        response_delay_s: float | None = 1.0e-5,
        adjust_time_step: bool = True,
        solver_delta_t_s: float | Sequence[float] | torch.Tensor | None = None,
        device: str | torch.device | None = None,
        chunk_size: int = 64,
        pin_memory: bool = True,
    ) -> None:
        """Configure a solver for an already indexed surface series.

        Args:
            series: Surface series supplying face geometry and source fields.
            rpm: Non-negative rotor speed used as result metadata.
            permeable: Use permeable-surface FW-H source terms. This requires
                fluid velocity in every source frame. ``False`` uses the
                impermeable pressure-loading formulation.
            moving_surface: Use time-dependent source geometry and the moving
                Farassat 1A propagation terms. ``False`` holds the reference
                mesh fixed.
            rotation_center_m: Center of rigid rotation in metres. Explicit
                values override reader metadata.
            omega_rad_s: Right-hand-rule angular-velocity vector in radians per
                second. Explicit values override reader metadata.
            rotation_reference_time_s: Absolute time represented by the
                reference mesh. Explicit values override reader metadata.
            density_kg_m3: Fluid density. It must match the density used while
                reading kinematic pressure from ``series``.
            sound_speed_m_s: Positive acoustic propagation speed.
            response_delay_s: Optional non-negative delay added after shifting
                each observer's nearest-face arrival to zero.
            adjust_time_step: Use backward Euler, matching an OpenFOAM run with
                adjustable time steps. ``False`` enables BDF2 after startup.
            solver_delta_t_s: OpenFOAM solver ``deltaT`` as one scalar or one
                value per source frame. By default it is inferred from times.
            device: Torch CPU or CUDA device. Defaults to CPU.
            chunk_size: Number of source frames processed together.
            pin_memory: Pin host tensors for asynchronous CUDA transfers.

        Raises:
            ValueError: If configuration, surface dtype, or geometry is invalid.
            RuntimeError: If CUDA is requested but unavailable.
        """
        self.series = series
        self.rpm = nonnegative_float(rpm, "rpm")
        self.permeable = bool(permeable)
        self.moving_surface = bool(moving_surface)
        self.is_moving_surface = self.moving_surface
        self.density_kg_m3 = positive_float(
            series.density_kg_m3 if density_kg_m3 is None else density_kg_m3,
            "density_kg_m3",
        )
        self.sound_speed_m_s = positive_float(
            sound_speed_m_s,
            "sound_speed_m_s",
        )
        self.response_delay_s = (
            None
            if response_delay_s is None
            else nonnegative_float(response_delay_s, "response_delay_s")
        )
        self.adjust_time_step = bool(adjust_time_step)
        self.device = _resolve_device(device)
        self.torch_dtype = series.dtype
        self.chunk_size = positive_int(chunk_size, "chunk_size")
        self.pin_memory = bool(pin_memory) and self.device.type == "cuda"

        if self.density_kg_m3 != series.density_kg_m3:
            raise ValueError(
                "density_kg_m3 must match the density used by the surface reader; "
                "pressure conversion and momentum terms require one consistent "
                "density."
            )
        if series.dtype not in {torch.float32, torch.float64}:
            raise ValueError("F1ASolver requires a float32 or float64 surface series.")
        self._absolute_times_s = self._to_torch(series.absolute_times_s)
        self._solver_delta_t_s = self._coerce_solver_delta_t(solver_delta_t_s)
        self._query_times_s = self._absolute_times_s + self._solver_delta_t_s * 1.0e-6
        if torch.any(torch.diff(self._query_times_s) <= 0.0).item():
            raise ValueError("Acoustic query times must be strictly increasing.")

        (
            self._face_centers_m,
            self._face_normals,
            self._face_areas_m2,
        ) = self._validated_reference_geometry()
        self._configure_rotation(
            rotation_center_m=rotation_center_m,
            omega_rad_s=omega_rad_s,
            rotation_reference_time_s=rotation_reference_time_s,
        )
        self.maximum_delay_s = math.nan

    @classmethod
    def from_openfoam_vtk(
        cls,
        input_dir: str | Path,
        *,
        rpm: float,
        permeable: bool = True,
        moving_surface: bool = False,
        rotation_center_m: Sequence[float] | torch.Tensor | None = None,
        omega_rad_s: Sequence[float] | torch.Tensor | None = None,
        rotation_reference_time_s: float | None = None,
        surface_file: str = "Cylx15.vtk",
        time_list: str | Path | None = None,
        pressure_field: str = "p",
        velocity_field: str | None = "U",
        pressure_units: PressureUnits = "kinematic",
        density_kg_m3: float = 1.225,
        pressure_reference_pa: float = 0.0,
        sound_speed_m_s: float = 343.0,
        response_delay_s: float | None = 1.0e-5,
        adjust_time_step: bool = True,
        solver_delta_t_s: float | Sequence[float] | torch.Tensor | None = None,
        dtype: DTypeName | torch.dtype = "float64",
        max_times: int | None = None,
        read_attempts: int = 4,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        chunk_size: int = 64,
        pin_memory: bool = True,
    ) -> F1ASolver:
        """Construct a surface reader and solver in one call.

        Reader arguments mirror :class:`OpenFOAMSurfaceSeries`; solver arguments
        are forwarded to :class:`F1ASolver`.

        Args:
            input_dir: Directory containing numeric OpenFOAM time directories.
            rpm: Non-negative rotor speed used as result metadata.
            permeable: Select permeable or impermeable FW-H source terms.
            moving_surface: Select moving or fixed source geometry.
            rotation_center_m: Optional center of rigid source rotation.
            omega_rad_s: Optional angular-velocity vector for rigid rotation.
            rotation_reference_time_s: Time represented by the reference mesh.
            surface_file: Legacy VTK filename inside each time directory.
            time_list: Optional file listing time-directory paths.
            pressure_field: Scalar VTK pressure field name.
            velocity_field: Three-component VTK velocity field name. It is read
                only when ``permeable=True``.
            pressure_units: ``"pa"`` or kinematic pressure units.
            density_kg_m3: Positive fluid density.
            pressure_reference_pa: Gauge-pressure reference subtracted in pascals.
            sound_speed_m_s: Positive acoustic propagation speed.
            response_delay_s: Optional non-negative response delay.
            adjust_time_step: Select backward Euler for adjustable time steps.
            solver_delta_t_s: Scalar or framewise OpenFOAM ``deltaT`` values.
            dtype: Floating-point dtype used for reader and solver tensors.
            max_times: Optional maximum number of time directories to read.
            read_attempts: VTK read attempts before failure.
            cache_dir: Optional binary-cache directory.
            device: Torch CPU or CUDA device.
            chunk_size: Number of source frames processed per solver chunk.
            pin_memory: Pin host tensors for CUDA transfers.

        Returns:
            Configured solver whose surface series is available as ``series``.
        """
        series = OpenFOAMSurfaceSeries(
            input_dir,
            surface_file=surface_file,
            time_list=time_list,
            pressure_field=pressure_field,
            velocity_field=velocity_field if permeable else None,
            pressure_units=pressure_units,
            density_kg_m3=density_kg_m3,
            pressure_reference_pa=pressure_reference_pa,
            dtype=dtype,
            max_times=max_times,
            read_attempts=read_attempts,
            cache_dir=cache_dir,
        )
        return cls(
            series,
            rpm=rpm,
            permeable=permeable,
            moving_surface=moving_surface,
            rotation_center_m=rotation_center_m,
            omega_rad_s=omega_rad_s,
            rotation_reference_time_s=rotation_reference_time_s,
            density_kg_m3=density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
            response_delay_s=response_delay_s,
            adjust_time_step=adjust_time_step,
            solver_delta_t_s=solver_delta_t_s,
            device=device,
            chunk_size=chunk_size,
            pin_memory=pin_memory,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        rpm: float | None = None,
        permeable: bool,
        moving_surface: bool,
        pressure_field: str | None = None,
        pressure_units: Literal["auto", "pa", "kinematic"] = "auto",
        density_kg_m3: float | None = None,
        pressure_reference_pa: float = 0.0,
        rotation_center_m: Sequence[float] | torch.Tensor | None = None,
        omega_rad_s: Sequence[float] | torch.Tensor | None = None,
        rotation_reference_time_s: float | None = None,
        flip_normals: bool = False,
        sound_speed_m_s: float = 343.0,
        response_delay_s: float | None = 1.0e-5,
        adjust_time_step: bool = True,
        solver_delta_t_s: float | Sequence[float] | torch.Tensor | None = None,
        dtype: DTypeName | torch.dtype = "float64",
        max_times: int | None = None,
        device: str | torch.device | None = None,
        chunk_size: int = 64,
        pin_memory: bool = True,
    ) -> F1ASolver:
        """Construct an NPZ surface reader and solver in one call.

        The compact archive schema may supply rotation origin, axis, angular
        speed, density, and pressure units. Explicit arguments override those
        metadata. ``permeable`` and ``moving_surface`` are required here so the
        archive format never implicitly selects the acoustic formulation.
        ``omega_rad_s`` is a right-hand-rule vector in radians/second.

        Args:
            path: Compact or legacy NPZ surface archive.
            rpm: Optional rotor speed; archive metadata is used by default.
            permeable: Select permeable or impermeable FW-H source terms.
            moving_surface: Select moving or fixed source geometry.
            pressure_field: ``p_surface_pred`` or ``p_surface_true``.
            pressure_units: Explicit units or ``"auto"`` for compact metadata.
            density_kg_m3: Density for kinematic-pressure conversion.
            pressure_reference_pa: Gauge-pressure reference subtracted in pascals.
            rotation_center_m: Optional centre of rigid source rotation.
            omega_rad_s: Optional angular-velocity vector for rigid rotation.
            rotation_reference_time_s: Time represented by the reference geometry.
            flip_normals: Reverse the archive's declared face normals.
            sound_speed_m_s: Positive acoustic propagation speed.
            response_delay_s: Optional non-negative response delay.
            adjust_time_step: Select backward Euler for adjustable time steps.
            solver_delta_t_s: Scalar or framewise OpenFOAM ``deltaT`` values.
            dtype: Floating-point dtype of source and solver tensors.
            max_times: Optional number of leading archive times to load.
            device: Torch CPU or CUDA device.
            chunk_size: Number of source frames processed per solver chunk.
            pin_memory: Pin host tensors for CUDA transfers.

        Returns:
            A configured solver backed by :class:`~foamacoustics.NPZSurfaceSeries`.
        """

        from ..io.npz import NPZSurfaceSeries

        series = NPZSurfaceSeries(
            path,
            pressure_field=pressure_field,
            pressure_units=pressure_units,
            density_kg_m3=density_kg_m3,
            pressure_reference_pa=pressure_reference_pa,
            rotation_center_m=rotation_center_m,
            omega_rad_s=omega_rad_s,
            rotation_reference_time_s=rotation_reference_time_s,
            flip_normals=flip_normals,
            dtype=dtype,
            max_times=max_times,
        )
        resolved_rpm = series.rpm if rpm is None else nonnegative_float(rpm, "rpm")
        return cls(
            series,
            rpm=resolved_rpm,
            permeable=permeable,
            moving_surface=moving_surface,
            rotation_center_m=rotation_center_m,
            omega_rad_s=omega_rad_s,
            rotation_reference_time_s=rotation_reference_time_s,
            density_kg_m3=series.density_kg_m3,
            sound_speed_m_s=sound_speed_m_s,
            response_delay_s=response_delay_s,
            adjust_time_step=adjust_time_step,
            solver_delta_t_s=solver_delta_t_s,
            device=device,
            chunk_size=chunk_size,
            pin_memory=pin_memory,
        )

    @property
    def panel_count(self) -> int:
        """Number of faces on the acoustic control surface."""
        return self.series.panel_count

    def _validated_reference_geometry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy reference geometry to the device and validate solver invariants."""

        centers = self._to_torch(self.series.face_centers_m)
        normals = self._to_torch(self.series.face_normals)
        areas = self._to_torch(self.series.face_areas_m2)
        if centers.shape != (self.panel_count, 3):
            raise ValueError("face_centers_m must have shape (faces, 3).")
        if normals.shape != (self.panel_count, 3):
            raise ValueError("face_normals must have shape (faces, 3).")
        if areas.shape != (self.panel_count,):
            raise ValueError("face_areas_m2 must have shape (faces,).")
        if not all(
            torch.isfinite(values).all().item() for values in (centers, normals, areas)
        ):
            raise ValueError("Surface geometry must contain only finite values.")
        if torch.any(areas <= 0.0).item():
            raise ValueError("Surface face areas must be positive.")
        if torch.any(torch.linalg.vector_norm(normals, dim=1) <= 0.0).item():
            raise ValueError("Surface face normals must be nonzero.")
        return centers, normals, areas

    def _coerce_vector3(self, value: object, name: str) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=self.torch_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a numeric three-vector.") from error
        if tensor.shape != (3,):
            raise ValueError(f"{name} must have shape (3,).")
        tensor = self._to_torch(tensor)
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values.")
        return tensor

    def _configure_rotation(
        self,
        *,
        rotation_center_m: Sequence[float] | torch.Tensor | None,
        omega_rad_s: Sequence[float] | torch.Tensor | None,
        rotation_reference_time_s: float | None,
    ) -> None:
        series_omega = getattr(self.series, "omega_rad_s", None)
        raw_omega = omega_rad_s if omega_rad_s is not None else series_omega
        if raw_omega is None:
            if rotation_center_m is not None or rotation_reference_time_s is not None:
                raise ValueError(
                    "rotation_center_m and rotation_reference_time_s require "
                    "omega_rad_s or angular-velocity metadata on the surface series."
                )
            self._omega_rad_s: torch.Tensor | None = None
            self._rotation_center_m: torch.Tensor | None = None
            self._rotation_reference_time_s: float | None = None
            self.omega_rad_s: torch.Tensor | None = None
            self.rotation_center_m: torch.Tensor | None = None
            self.rotation_reference_time_s: float | None = None
            return

        effective_omega = self._coerce_vector3(raw_omega, "omega_rad_s")
        series_center = getattr(self.series, "rotation_center_m", None)
        raw_center = (
            rotation_center_m if rotation_center_m is not None else series_center
        )
        effective_center = (
            torch.zeros(3, device=self.device, dtype=self.torch_dtype)
            if raw_center is None
            else self._coerce_vector3(raw_center, "rotation_center_m")
        )
        series_reference_time = getattr(
            self.series,
            "rotation_reference_time_s",
            None,
        )
        raw_reference_time = (
            rotation_reference_time_s
            if rotation_reference_time_s is not None
            else series_reference_time
        )
        effective_reference_time = (
            float(self._absolute_times_s[0].detach().cpu())
            if raw_reference_time is None
            else finite_float(
                raw_reference_time,
                "rotation_reference_time_s",
            )
        )

        self._omega_rad_s = effective_omega
        self._rotation_center_m = effective_center
        self._rotation_reference_time_s = effective_reference_time
        self.omega_rad_s = effective_omega
        self.rotation_center_m = effective_center
        self.rotation_reference_time_s = effective_reference_time

    def _coerce_solver_delta_t(
        self,
        value: float | Sequence[float] | torch.Tensor | None,
    ) -> torch.Tensor:
        count = len(self.series)
        if value is None:
            differences = torch.diff(self._absolute_times_s)
            delta_t = torch.empty(count, device=self.device, dtype=self.torch_dtype)
            delta_t[0] = differences[0]
            delta_t[1:] = differences
        elif isinstance(value, (int, float)):
            delta_t = torch.full(
                (count,),
                positive_float(value, "solver_delta_t_s"),
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            delta_t = self._to_torch(value)
            if delta_t.shape != (count,):
                raise ValueError(
                    f"solver_delta_t_s must have shape ({count},), got {delta_t.shape}."
                )
            if (
                not torch.isfinite(delta_t).all().item()
                or torch.any(delta_t <= 0.0).item()
            ):
                raise ValueError("solver_delta_t_s values must be finite and positive.")
        return delta_t

    def _validated_observers(
        self,
        observers_m: Sequence[Sequence[float]] | torch.Tensor,
    ) -> torch.Tensor:
        observers = self._to_torch(observers_m)
        if (
            observers.ndim != 2
            or observers.shape[1] != 3
            or observers.shape[0] == 0
            or not torch.isfinite(observers).all().item()
        ):
            raise ValueError(
                "observers_m must have shape (n_observers, 3) and contain "
                "only finite coordinates."
            )
        return observers

    def _observer_geometries(
        self,
        observers_m: Sequence[Sequence[float]] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observers = self._validated_observers(observers_m)
        source_to_observer = observers[:, None, :] - self._face_centers_m[None, :, :]
        distance = torch.linalg.vector_norm(source_to_observer, dim=2)
        if torch.any(distance <= 0.0).item():
            raise ValueError("An observer cannot coincide with a face centre.")
        direction = source_to_observer / distance[..., None]
        if self.response_delay_s is None:
            delay = distance / self.sound_speed_m_s
        else:
            delay = (
                distance - distance.min(dim=1, keepdim=True).values
            ) / self.sound_speed_m_s + self.response_delay_s
        return observers, distance, direction, delay

    def _resolve_algorithm(
        self,
        requested: PropagationAlgorithm,
    ) -> tuple[Literal["uniform", "nonuniform"], float | None]:
        if requested not in {"auto", "uniform", "nonuniform"}:
            raise ValueError("algorithm must be 'auto', 'uniform', or 'nonuniform'.")
        if self.moving_surface:
            if requested == "uniform":
                raise ValueError(
                    "algorithm='uniform' is not available for moving surfaces "
                    "because each face's propagation delay changes with time."
                )
            return "nonuniform", None
        spacing = _constant_spacing(self._absolute_times_s)
        # A constant array has zero differences, so check it directly.
        epsilon = self._solver_delta_t_s * 1.0e-6
        epsilon_constant = torch.all(epsilon == epsilon[0]).item() or torch.allclose(
            epsilon, epsilon[0], rtol=0.0, atol=torch.finfo(epsilon.dtype).eps
        )
        can_use_uniform = spacing is not None and epsilon_constant
        if requested == "uniform" and not can_use_uniform:
            raise ValueError(
                "algorithm='uniform' requires exactly uniform source times and "
                "a constant solver deltaT."
            )
        if requested == "uniform" or (requested == "auto" and can_use_uniform):
            return "uniform", spacing
        return "nonuniform", None

    def _to_torch(self, values: TensorLike) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=self.torch_dtype)
        if self.pin_memory and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        return tensor.to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=self.pin_memory,
        )

    def _source_terms_torch_chunk(
        self,
        chunk: SurfaceChunk,
        derivative: _TorchFiniteDifferenceDerivative,
        face_normals: torch.Tensor,
        coefficients: _FixedObserverCoefficients,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_count = chunk.frame_count
        pressure = chunk.pressure_pa
        if pressure.shape != (time_count, self.panel_count):
            raise ValueError("A surface chunk has an unexpected pressure shape.")

        pressure = self._to_torch(pressure)
        absolute_times_s = self._absolute_times_s[
            chunk.start_index : chunk.start_index + time_count
        ]
        if self.permeable:
            if chunk.fluid_velocity_m_s is None:
                raise ValueError(
                    "permeable=True requires fluid velocity in every source frame."
                )
            fluid_velocity = chunk.fluid_velocity_m_s
            if fluid_velocity.shape != (time_count, self.panel_count, 3):
                raise ValueError(
                    "A surface chunk has an unexpected fluid-velocity shape."
                )
            fluid_velocity = self._to_torch(fluid_velocity)
            normal_velocity = torch.einsum(
                "tfi,fi->tf",
                fluid_velocity,
                face_normals,
            )
            loading_vector = (
                pressure[..., None] * face_normals[None, ...]
                + self.density_kg_m3 * fluid_velocity * normal_velocity[..., None]
            )
        else:
            normal_velocity = torch.zeros_like(pressure)
            loading_vector = pressure[..., None] * face_normals[None, ...]
        state = torch.cat((normal_velocity[..., None], loading_vector), dim=2)
        state_rate = derivative.step_chunk(state, absolute_times_s)

        thickness = state_rate[:, None, :, 0] * coefficients.thickness[None, :, :]
        loading = torch.einsum(
            "tfi,ofi->tof",
            state_rate[..., 1:],
            coefficients.loading_rate,
        ) + torch.einsum(
            "tfi,ofi->tof",
            loading_vector,
            coefficients.loading,
        )
        return thickness, loading, absolute_times_s

    def _rigid_rotation_kinematics(
        self,
        absolute_times_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._omega_rad_s is None
            or self._rotation_center_m is None
            or self._rotation_reference_time_s is None
        ):
            raise RuntimeError("Incomplete rigid-rotation configuration.")
        omega = self._omega_rad_s
        center = self._rotation_center_m
        time_count = absolute_times_s.numel()
        reference_radius = (self._face_centers_m - center[None, :])[None, ...]
        reference_normals = self._face_normals[None, ...]
        omega_norm = torch.linalg.vector_norm(omega)
        if float(omega_norm.detach().cpu()) == 0.0:
            centers = (reference_radius + center[None, None, :]).expand(
                time_count,
                -1,
                -1,
            )
            normals = reference_normals.expand(time_count, -1, -1)
            velocity = torch.zeros_like(centers)
            acceleration = torch.zeros_like(centers)
            return centers, normals, velocity, acceleration

        axis = omega / omega_norm
        raw_angles = omega_norm * (absolute_times_s - self._rotation_reference_time_s)
        angles = torch.remainder(raw_angles + math.pi, 2.0 * math.pi) - math.pi
        cosine = torch.cos(angles)[:, None, None]
        sine = torch.sin(angles)[:, None, None]
        axis_for_radius = axis[None, None, :].expand_as(reference_radius)
        rotated_radius = (
            reference_radius * cosine
            + torch.linalg.cross(
                axis_for_radius,
                reference_radius,
                dim=2,
            )
            * sine
            + torch.einsum(
                "tfi,i->tf",
                reference_radius,
                axis,
            )[..., None]
            * axis[None, None, :]
            * (1.0 - cosine)
        )
        axis_for_normals = axis[None, None, :].expand_as(reference_normals)
        normals = (
            reference_normals * cosine
            + torch.linalg.cross(
                axis_for_normals,
                reference_normals,
                dim=2,
            )
            * sine
            + torch.einsum(
                "tfi,i->tf",
                reference_normals,
                axis,
            )[..., None]
            * axis[None, None, :]
            * (1.0 - cosine)
        )
        centers = rotated_radius + center[None, None, :]
        velocity = torch.linalg.cross(
            omega[None, None, :].expand_as(rotated_radius),
            rotated_radius,
            dim=2,
        )
        acceleration = torch.linalg.cross(
            omega[None, None, :].expand_as(velocity),
            velocity,
            dim=2,
        )
        return centers, normals, velocity, acceleration

    def _surface_kinematics_torch_chunk(
        self,
        chunk: SurfaceChunk,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Resolve moving geometry from rigid-rotation options or frame fields."""

        time_count = chunk.frame_count
        absolute_times_s = self._absolute_times_s[
            chunk.start_index : chunk.start_index + time_count
        ]
        if self._omega_rad_s is not None:
            centers, normals, velocity, acceleration = self._rigid_rotation_kinematics(
                absolute_times_s
            )
            face_areas_m2 = self._face_areas_m2[None, :].expand(
                time_count,
                -1,
            )
        else:
            required_geometry = {
                "face_centers_m": chunk.face_centers_m,
                "face_normals": chunk.face_normals,
                "surface_velocity_m_s": chunk.surface_velocity_m_s,
                "surface_acceleration_m_s2": chunk.surface_acceleration_m_s2,
            }
            missing = [
                name for name, values in required_geometry.items() if values is None
            ]
            if missing:
                raise ValueError(
                    "moving_surface=True requires omega_rad_s for rigid "
                    "rotation or moving frame fields: " + ", ".join(missing)
                )
            centers = self._to_torch(cast(torch.Tensor, chunk.face_centers_m))
            normals = self._to_torch(cast(torch.Tensor, chunk.face_normals))
            velocity = self._to_torch(cast(torch.Tensor, chunk.surface_velocity_m_s))
            acceleration = self._to_torch(
                cast(torch.Tensor, chunk.surface_acceleration_m_s2)
            )
            if chunk.face_areas_m2 is not None:
                face_areas_m2 = self._to_torch(chunk.face_areas_m2)
            else:
                face_areas_m2 = self._face_areas_m2[None, :].expand(
                    time_count,
                    -1,
                )

        expected_vectors = (time_count, self.panel_count, 3)
        for name, values in (
            ("face centers", centers),
            ("face normals", normals),
            ("surface velocity", velocity),
            ("surface acceleration", acceleration),
        ):
            if values.shape != expected_vectors:
                raise ValueError(f"A moving chunk has unexpected {name} shape.")
        if face_areas_m2.shape != (time_count, self.panel_count):
            raise ValueError("A moving chunk has an unexpected face-area shape.")
        if (
            not torch.isfinite(centers).all().item()
            or not torch.isfinite(normals).all().item()
            or not torch.isfinite(velocity).all().item()
            or not torch.isfinite(acceleration).all().item()
            or not torch.isfinite(face_areas_m2).all().item()
        ):
            raise ValueError("Moving surface fields must contain only finite values.")
        if torch.any(face_areas_m2 <= 0.0).item():
            raise ValueError("Moving surface face areas must be positive.")
        if torch.any(torch.linalg.vector_norm(normals, dim=2) <= 0.0).item():
            raise ValueError("Moving surface face normals must be nonzero.")
        return (
            centers,
            normals,
            face_areas_m2,
            velocity,
            acceleration,
            absolute_times_s,
        )

    def _moving_source_terms_torch_chunk(
        self,
        chunk: SurfaceChunk,
        derivative: _TorchFiniteDifferenceDerivative,
        observers_m: torch.Tensor,
        reference_distance_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate moving-surface Formulation 1A source terms."""

        pressure = chunk.pressure_pa
        time_count = chunk.frame_count
        if pressure.shape != (time_count, self.panel_count):
            raise ValueError("A moving chunk has an unexpected pressure shape.")
        pressure = self._to_torch(pressure)
        (
            centers,
            normals,
            face_areas_m2,
            velocity,
            acceleration,
            absolute_times_s,
        ) = self._surface_kinematics_torch_chunk(chunk)
        if not torch.isfinite(pressure).all().item():
            raise ValueError("Moving surface pressure must contain only finite values.")

        surface_normal_velocity = torch.einsum(
            "tfi,tfi->tf",
            velocity,
            normals,
        )
        if self.permeable:
            if chunk.fluid_velocity_m_s is None:
                raise ValueError(
                    "permeable=True requires fluid velocity in every source frame."
                )
            fluid_velocity = chunk.fluid_velocity_m_s
            if fluid_velocity.shape != (time_count, self.panel_count, 3):
                raise ValueError(
                    "A moving chunk has an unexpected fluid-velocity shape."
                )
            fluid_velocity = self._to_torch(fluid_velocity)
            if not torch.isfinite(fluid_velocity).all().item():
                raise ValueError("Fluid velocity must contain only finite values.")
            fluid_normal_velocity = torch.einsum(
                "tfi,tfi->tf",
                fluid_velocity,
                normals,
            )
            normal_velocity = fluid_normal_velocity
            loading_vector = (
                pressure[..., None] * normals
                + self.density_kg_m3
                * fluid_velocity
                * (fluid_normal_velocity - surface_normal_velocity)[..., None]
            )
        else:
            normal_velocity = surface_normal_velocity
            loading_vector = pressure[..., None] * normals
        state = torch.cat((normal_velocity[..., None], loading_vector), dim=2)
        state_rate = derivative.step_chunk(state, absolute_times_s)

        source_to_observer = observers_m[None, :, None, :] - centers[:, None, :, :]
        distance_m = torch.linalg.vector_norm(source_to_observer, dim=3)
        if torch.any(distance_m <= 0.0).item():
            raise ValueError("An observer cannot coincide with a moving face centre.")
        direction = source_to_observer / distance_m[..., None]

        mach = velocity / self.sound_speed_m_s
        mach_squared = torch.einsum("tfi,tfi->tf", mach, mach)
        if torch.any(mach_squared >= 1.0).item():
            raise ValueError(
                "Moving-surface Farassat 1A requires every face speed to be "
                "strictly subsonic."
            )
        radial_mach = torch.einsum("tfi,tofi->tof", mach, direction)
        doppler_factor = 1.0 - radial_mach
        if torch.any(doppler_factor <= 0.0).item():
            raise ValueError("A moving face has a non-positive Doppler factor.")

        mach_rate_radial = torch.einsum(
            "tfi,tofi->tof",
            acceleration / self.sound_speed_m_s,
            direction,
        )
        kinematic_factor = distance_m * mach_rate_radial + self.sound_speed_m_s * (
            radial_mach - mach_squared[:, None, :]
        )
        loading_radial = torch.einsum(
            "tfi,tofi->tof",
            loading_vector,
            direction,
        )
        loading_rate_radial = torch.einsum(
            "tfi,tofi->tof",
            state_rate[..., 1:],
            direction,
        )
        loading_mach = torch.einsum("tfi,tfi->tf", loading_vector, mach)

        area = face_areas_m2[:, None, :]
        thickness = (
            area
            * self.density_kg_m3
            * (
                state_rate[:, None, :, 0] / (distance_m * doppler_factor**2)
                + normal_velocity[:, None, :]
                * kinematic_factor
                / (distance_m**2 * doppler_factor**3)
            )
        )
        loading = area * (
            loading_rate_radial
            / (self.sound_speed_m_s * distance_m * doppler_factor**2)
            + (loading_radial - loading_mach[:, None, :])
            / (distance_m**2 * doppler_factor**2)
            + loading_radial
            * kinematic_factor
            / (self.sound_speed_m_s * distance_m**2 * doppler_factor**3)
        )

        if self.response_delay_s is None:
            delay_s = distance_m / self.sound_speed_m_s
        else:
            delay_s = (
                distance_m - reference_distance_m[None, :, None]
            ) / self.sound_speed_m_s + self.response_delay_s
        arrival_times_s = absolute_times_s[:, None, None] + delay_s
        return thickness, loading, absolute_times_s, arrival_times_s

    def _torch_uniform_delays(
        self,
        delay_s: torch.Tensor,
        spacing_s: float,
        epsilon_s: float,
    ) -> _TorchUniformDelays:
        delay_samples = (delay_s - epsilon_s) / spacing_s
        valid_faces = torch.nonzero(delay_samples >= 0.0, as_tuple=False).flatten()
        if valid_faces.numel() == 0:
            empty_i = torch.empty(0, device=self.device, dtype=torch.int64)
            empty_f = torch.empty(0, device=self.device, dtype=self.torch_dtype)
            return _TorchUniformDelays(
                empty_i,
                empty_i,
                None,
                None,
                empty_f,
                empty_f,
                empty_f,
                0,
            )
        offsets = torch.floor(delay_samples[valid_faces]).to(torch.int64)
        fraction = delay_samples[valid_faces] - offsets
        sorting = torch.argsort(offsets, stable=True)
        face_order = valid_faces[sorting]
        offsets = offsets[sorting]
        fraction = fraction[sorting]
        grouped_offsets, faces_per_group = torch.unique_consecutive(
            offsets,
            return_counts=True,
        )
        use_grouped_reduction = grouped_offsets.numel() * 4 <= offsets.numel()
        return _TorchUniformDelays(
            face_order=face_order,
            output_offsets_per_face=offsets,
            grouped_output_offsets=(grouped_offsets if use_grouped_reduction else None),
            faces_per_group=faces_per_group if use_grouped_reduction else None,
            weight_at_offset=1.0 - fraction,
            weight_at_next_offset=fraction,
            initial_weight_at_offset=torch.where(
                fraction == 0.0, torch.ones_like(fraction), torch.zeros_like(fraction)
            ),
            maximum_output_offset=int(offsets[-1].detach().cpu()),
        )

    def _torch_packed_uniform_delays(
        self,
        delay_s: torch.Tensor,
        spacing_s: float,
        epsilon_s: float,
    ) -> _TorchPackedUniformDelays | None:
        """Pack strongly repeated observer/face delays into global delay groups."""

        observer_count, face_count = delay_s.shape
        delay_samples = (delay_s - epsilon_s) / spacing_s
        packed_face_indices: list[torch.Tensor] = []
        group_ids: list[torch.Tensor] = []
        group_observers: list[torch.Tensor] = []
        grouped_offsets: list[torch.Tensor] = []
        lower_weights: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        initial_weights: list[torch.Tensor] = []
        valid_face_count = 0
        group_count = 0

        for observer_index in range(observer_count):
            observer_samples = delay_samples[observer_index]
            valid_faces = torch.nonzero(
                observer_samples >= 0.0,
                as_tuple=False,
            ).flatten()
            if valid_faces.numel() == 0:
                continue
            offsets = torch.floor(observer_samples[valid_faces]).to(torch.int64)
            fraction = observer_samples[valid_faces] - offsets
            unique_offsets, local_group_ids = torch.unique(
                offsets,
                sorted=True,
                return_inverse=True,
            )
            packed_face_indices.append(observer_index * face_count + valid_faces)
            group_ids.append(local_group_ids + group_count)
            group_observers.append(
                torch.full_like(unique_offsets, observer_index),
            )
            grouped_offsets.append(unique_offsets)
            lower_weights.append(1.0 - fraction)
            next_weights.append(fraction)
            initial_weights.append(
                torch.where(
                    fraction == 0.0,
                    torch.ones_like(fraction),
                    torch.zeros_like(fraction),
                )
            )
            valid_face_count += valid_faces.numel()
            group_count += unique_offsets.numel()

        if valid_face_count == 0 or group_count * 4 > valid_face_count:
            return None

        combined_offsets = torch.cat(grouped_offsets)
        all_faces_valid = valid_face_count == observer_count * face_count
        return _TorchPackedUniformDelays(
            packed_face_indices=(
                None if all_faces_valid else torch.cat(packed_face_indices)
            ),
            group_id_per_face=torch.cat(group_ids),
            group_observer=torch.cat(group_observers),
            grouped_output_offsets=combined_offsets,
            weight_at_offset=torch.cat(lower_weights),
            weight_at_next_offset=torch.cat(next_weights),
            initial_weight_at_offset=torch.cat(initial_weights),
            maximum_output_offset=int(combined_offsets.max().detach().cpu()),
        )

    @staticmethod
    def _scatter_add_valid(
        output: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        all_indices_valid: bool,
    ) -> None:
        if all_indices_valid:
            output.scatter_add_(0, indices, values)
            return
        valid = indices < output.numel()
        output.scatter_add_(0, indices[valid], values[valid])

    @staticmethod
    def _reduce_packed_uniform_source(
        source: torch.Tensor,
        group_id_per_face: torch.Tensor,
        group_count: int,
        weights: torch.Tensor,
        *,
        initial_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grouped = torch.zeros(
            (source.shape[0], group_count),
            device=source.device,
            dtype=source.dtype,
        )
        if initial_weights is None:
            grouped.scatter_add_(
                1,
                group_id_per_face[None, :].expand(source.shape[0], -1),
                source * weights[None, :],
            )
            return grouped

        grouped[0].scatter_add_(
            0,
            group_id_per_face,
            source[0] * initial_weights,
        )
        if source.shape[0] > 1:
            grouped[1:].scatter_add_(
                1,
                group_id_per_face[None, :].expand(source.shape[0] - 1, -1),
                source[1:] * weights[None, :],
            )
        return grouped

    @staticmethod
    def _scatter_add_packed_uniform(
        output: torch.Tensor,
        time_indices: torch.Tensor,
        group_observer: torch.Tensor,
        values: torch.Tensor,
        *,
        all_indices_valid: bool,
    ) -> None:
        output_length = output.shape[1]
        output_indices = time_indices + group_observer[None, :] * output_length
        output_flat = output.reshape(-1)
        if all_indices_valid:
            output_flat.scatter_add_(
                0,
                output_indices.reshape(-1),
                values.reshape(-1),
            )
            return
        valid = time_indices < output_length
        output_flat.scatter_add_(
            0,
            output_indices[valid],
            values[valid],
        )

    def _deposit_packed_uniform_torch_chunk(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        thickness_source: torch.Tensor,
        loading_source: torch.Tensor,
        source_start_index: int,
        delays: _TorchPackedUniformDelays,
    ) -> None:
        time_count = thickness_source.shape[0]
        flat_thickness = thickness_source.reshape(time_count, -1)
        flat_loading = loading_source.reshape(time_count, -1)
        if delays.packed_face_indices is not None:
            flat_thickness = flat_thickness.index_select(
                1,
                delays.packed_face_indices,
            )
            flat_loading = flat_loading.index_select(
                1,
                delays.packed_face_indices,
            )

        group_count = delays.grouped_output_offsets.numel()
        initial_weights = (
            delays.initial_weight_at_offset if source_start_index == 0 else None
        )
        lower_thickness = self._reduce_packed_uniform_source(
            flat_thickness,
            delays.group_id_per_face,
            group_count,
            delays.weight_at_offset,
            initial_weights=initial_weights,
        )
        lower_loading = self._reduce_packed_uniform_source(
            flat_loading,
            delays.group_id_per_face,
            group_count,
            delays.weight_at_offset,
            initial_weights=initial_weights,
        )
        next_thickness = self._reduce_packed_uniform_source(
            flat_thickness,
            delays.group_id_per_face,
            group_count,
            delays.weight_at_next_offset,
        )
        next_loading = self._reduce_packed_uniform_source(
            flat_loading,
            delays.group_id_per_face,
            group_count,
            delays.weight_at_next_offset,
        )

        source_indices = torch.arange(
            source_start_index,
            source_start_index + time_count,
            device=self.device,
            dtype=torch.int64,
        )
        lower_time_indices = (
            source_indices[:, None] + delays.grouped_output_offsets[None, :]
        )
        final_lower_index = (
            source_start_index + time_count - 1 + delays.maximum_output_offset
        )
        all_lower_indices_valid = final_lower_index < thickness_output.shape[1]
        self._scatter_add_packed_uniform(
            thickness_output,
            lower_time_indices,
            delays.group_observer,
            lower_thickness,
            all_indices_valid=all_lower_indices_valid,
        )
        self._scatter_add_packed_uniform(
            loading_output,
            lower_time_indices,
            delays.group_observer,
            lower_loading,
            all_indices_valid=all_lower_indices_valid,
        )

        next_time_indices = lower_time_indices + 1
        all_next_indices_valid = final_lower_index + 1 < thickness_output.shape[1]
        self._scatter_add_packed_uniform(
            thickness_output,
            next_time_indices,
            delays.group_observer,
            next_thickness,
            all_indices_valid=all_next_indices_valid,
        )
        self._scatter_add_packed_uniform(
            loading_output,
            next_time_indices,
            delays.group_observer,
            next_loading,
            all_indices_valid=all_next_indices_valid,
        )

    def _deposit_uniform_torch_chunk(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        thickness_source: torch.Tensor,
        loading_source: torch.Tensor,
        source_start_index: int,
        delays: _TorchUniformDelays,
    ) -> None:
        if delays.face_order.numel() == 0:
            return
        source_indices = torch.arange(
            source_start_index,
            source_start_index + thickness_source.shape[0],
            device=self.device,
            dtype=torch.int64,
        )
        ordered_thickness = thickness_source.index_select(1, delays.face_order)
        ordered_loading = loading_source.index_select(1, delays.face_order)

        lower_weights = delays.weight_at_offset[None, :].expand_as(ordered_thickness)
        if source_start_index == 0:
            lower_weights = lower_weights.clone()
            lower_weights[0] = delays.initial_weight_at_offset
        next_weights = delays.weight_at_next_offset[None, :]
        if (
            delays.grouped_output_offsets is not None
            and delays.faces_per_group is not None
        ):
            output_offsets = delays.grouped_output_offsets
            group_lengths = delays.faces_per_group[None, :].expand(
                ordered_thickness.shape[0],
                -1,
            )
            lower_thickness = torch.segment_reduce(
                ordered_thickness * lower_weights,
                reduce="sum",
                lengths=group_lengths,
                axis=1,
            )
            lower_loading = torch.segment_reduce(
                ordered_loading * lower_weights,
                reduce="sum",
                lengths=group_lengths,
                axis=1,
            )
            next_thickness = torch.segment_reduce(
                ordered_thickness * next_weights,
                reduce="sum",
                lengths=group_lengths,
                axis=1,
            )
            next_loading = torch.segment_reduce(
                ordered_loading * next_weights,
                reduce="sum",
                lengths=group_lengths,
                axis=1,
            )
        else:
            output_offsets = delays.output_offsets_per_face
            lower_thickness = ordered_thickness * lower_weights
            lower_loading = ordered_loading * lower_weights
            next_thickness = ordered_thickness * next_weights
            next_loading = ordered_loading * next_weights
        output_indices = source_indices[:, None] + output_offsets[None, :]
        final_lower_index = (
            source_start_index
            + thickness_source.shape[0]
            - 1
            + delays.maximum_output_offset
        )
        all_lower_indices_valid = final_lower_index < thickness_output.numel()
        self._scatter_add_valid(
            thickness_output,
            output_indices.reshape(-1),
            lower_thickness.reshape(-1),
            all_indices_valid=all_lower_indices_valid,
        )
        self._scatter_add_valid(
            loading_output,
            output_indices.reshape(-1),
            lower_loading.reshape(-1),
            all_indices_valid=all_lower_indices_valid,
        )

        next_indices = output_indices + 1
        all_next_indices_valid = final_lower_index + 1 < thickness_output.numel()
        self._scatter_add_valid(
            thickness_output,
            next_indices.reshape(-1),
            next_thickness.reshape(-1),
            all_indices_valid=all_next_indices_valid,
        )
        self._scatter_add_valid(
            loading_output,
            next_indices.reshape(-1),
            next_loading.reshape(-1),
            all_indices_valid=all_next_indices_valid,
        )

    def _deposit_initial_exact_torch(
        self,
        output: torch.Tensor,
        source: torch.Tensor,
        arrival_s: torch.Tensor,
        query_times_s: torch.Tensor,
    ) -> None:
        indices = torch.searchsorted(query_times_s, arrival_s, right=False)
        inside = indices < query_times_s.numel()
        safe_indices = indices.clamp_max(query_times_s.numel() - 1)
        exact = inside & (query_times_s[safe_indices] == arrival_s)
        output.scatter_add_(0, indices[exact], source[exact])

    @staticmethod
    def _deposit_interpolated_torch_intervals(
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        previous_thickness: torch.Tensor,
        current_thickness: torch.Tensor,
        previous_loading: torch.Tensor,
        current_loading: torch.Tensor,
        previous_arrival_s: torch.Tensor,
        current_arrival_s: torch.Tensor,
        interpolation_interval_s: torch.Tensor,
        current_source_indices: torch.Tensor,
        query_times_s: torch.Tensor,
    ) -> None:
        first_output = torch.searchsorted(
            query_times_s,
            previous_arrival_s,
            right=True,
        )
        output_stop = torch.searchsorted(
            query_times_s,
            current_arrival_s,
            right=True,
        )
        counts = output_stop - first_output
        maximum_count = int(counts.max().item()) if counts.numel() else 0
        for offset in range(maximum_count):
            output_indices = first_output + offset
            valid = (
                (output_indices < output_stop)
                & (output_indices >= current_source_indices[:, None])
                & (output_indices < query_times_s.numel())
            )
            safe_indices = output_indices.clamp(0, query_times_s.numel() - 1)
            interpolation_fraction = (
                query_times_s[safe_indices] - previous_arrival_s
            ) / interpolation_interval_s
            thickness = (
                1.0 - interpolation_fraction
            ) * previous_thickness + interpolation_fraction * current_thickness
            loading = (
                1.0 - interpolation_fraction
            ) * previous_loading + interpolation_fraction * current_loading
            thickness_output.scatter_add_(
                0,
                output_indices[valid],
                thickness[valid],
            )
            loading_output.scatter_add_(
                0,
                output_indices[valid],
                loading[valid],
            )

    def _deposit_nonuniform_torch_intervals(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        previous_thickness: torch.Tensor,
        current_thickness: torch.Tensor,
        previous_loading: torch.Tensor,
        current_loading: torch.Tensor,
        previous_times_s: torch.Tensor,
        current_times_s: torch.Tensor,
        delay_s: torch.Tensor,
        current_source_indices: torch.Tensor,
        query_times_s: torch.Tensor,
    ) -> None:
        self._deposit_interpolated_torch_intervals(
            thickness_output,
            loading_output,
            previous_thickness,
            current_thickness,
            previous_loading,
            current_loading,
            previous_times_s[:, None] + delay_s[None, :],
            current_times_s[:, None] + delay_s[None, :],
            (current_times_s - previous_times_s)[:, None],
            current_source_indices,
            query_times_s,
        )

    def _deposit_moving_torch_intervals(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        previous_thickness: torch.Tensor,
        current_thickness: torch.Tensor,
        previous_loading: torch.Tensor,
        current_loading: torch.Tensor,
        previous_arrival_s: torch.Tensor,
        current_arrival_s: torch.Tensor,
        current_source_indices: torch.Tensor,
        query_times_s: torch.Tensor,
    ) -> None:
        previous_arrival_s = previous_arrival_s.contiguous()
        current_arrival_s = current_arrival_s.contiguous()
        arrival_interval_s = current_arrival_s - previous_arrival_s
        if torch.any(arrival_interval_s <= 0.0).item():
            raise ValueError(
                "Moving-face arrival times must be strictly increasing. "
                "Check that the surface motion is subsonic and times are ordered."
            )
        self._deposit_interpolated_torch_intervals(
            thickness_output,
            loading_output,
            previous_thickness,
            current_thickness,
            previous_loading,
            current_loading,
            previous_arrival_s,
            current_arrival_s,
            arrival_interval_s,
            current_source_indices,
            query_times_s,
        )

    def _deposit_fixed_uniform_chunk(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        thickness_source: torch.Tensor,
        loading_source: torch.Tensor,
        source_start_index: int,
        packed_delays: _TorchPackedUniformDelays | None,
        observer_delays: list[_TorchUniformDelays] | None,
    ) -> None:
        """Deposit one fixed-surface chunk with precomputed uniform delays."""

        if packed_delays is not None:
            self._deposit_packed_uniform_torch_chunk(
                thickness_output,
                loading_output,
                thickness_source,
                loading_source,
                source_start_index,
                packed_delays,
            )
            return
        if observer_delays is None:
            raise RuntimeError("Uniform Torch delays were not created.")
        for observer_index, delays in enumerate(observer_delays):
            self._deposit_uniform_torch_chunk(
                thickness_output[observer_index],
                loading_output[observer_index],
                thickness_source[:, observer_index, :],
                loading_source[:, observer_index, :],
                source_start_index,
                delays,
            )

    def _deposit_fixed_nonuniform_chunk(
        self,
        thickness_output: torch.Tensor,
        loading_output: torch.Tensor,
        thickness_source: torch.Tensor,
        loading_source: torch.Tensor,
        absolute_times_s: torch.Tensor,
        delay_s: torch.Tensor,
        source_start_index: int,
        previous: _FixedPropagationState | None,
    ) -> _FixedPropagationState:
        """Deposit one fixed-surface chunk with framewise interpolation."""

        current_indices = torch.arange(
            source_start_index,
            source_start_index + thickness_source.shape[0],
            device=self.device,
            dtype=torch.int64,
        )
        for observer_index in range(thickness_output.shape[0]):
            observer_thickness = thickness_source[:, observer_index, :]
            observer_loading = loading_source[:, observer_index, :]
            if previous is None:
                initial_arrival = absolute_times_s[0] + delay_s[observer_index]
                self._deposit_initial_exact_torch(
                    thickness_output[observer_index],
                    observer_thickness[0],
                    initial_arrival,
                    self._query_times_s,
                )
                self._deposit_initial_exact_torch(
                    loading_output[observer_index],
                    observer_loading[0],
                    initial_arrival,
                    self._query_times_s,
                )
                if observer_thickness.shape[0] > 1:
                    self._deposit_nonuniform_torch_intervals(
                        thickness_output[observer_index],
                        loading_output[observer_index],
                        observer_thickness[:-1],
                        observer_thickness[1:],
                        observer_loading[:-1],
                        observer_loading[1:],
                        absolute_times_s[:-1],
                        absolute_times_s[1:],
                        delay_s[observer_index],
                        current_indices[1:],
                        self._query_times_s,
                    )
                continue

            previous_thickness, previous_loading, previous_time_s = previous
            self._deposit_nonuniform_torch_intervals(
                thickness_output[observer_index],
                loading_output[observer_index],
                torch.cat(
                    (
                        previous_thickness[observer_index][None, :],
                        observer_thickness[:-1],
                    ),
                    dim=0,
                ),
                observer_thickness,
                torch.cat(
                    (
                        previous_loading[observer_index][None, :],
                        observer_loading[:-1],
                    ),
                    dim=0,
                ),
                observer_loading,
                torch.cat((previous_time_s[None], absolute_times_s[:-1]), dim=0),
                absolute_times_s,
                delay_s[observer_index],
                current_indices,
                self._query_times_s,
            )
        return (
            thickness_source[-1].detach(),
            loading_source[-1].detach(),
            absolute_times_s[-1].detach(),
        )

    def _moving_distance_extrema(
        self,
        observers_m: torch.Tensor,
        *,
        workers: int,
        invalid_time_policy: InvalidTimePolicy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        minimum = torch.full(
            (observers_m.shape[0],),
            torch.inf,
            device=self.device,
            dtype=self.torch_dtype,
        )
        maximum = torch.zeros_like(minimum)

        def include_centers(centers: torch.Tensor) -> None:
            nonlocal minimum, maximum
            if centers.ndim != 3 or centers.shape[1:] != (
                self.panel_count,
                3,
            ):
                raise ValueError("A moving chunk has unexpected face-centre shape.")
            distances = torch.linalg.vector_norm(
                observers_m[None, :, None, :] - centers[:, None, :, :],
                dim=3,
            )
            if torch.any(distances <= 0.0).item():
                raise ValueError("An observer cannot coincide with a face centre.")
            minimum = torch.minimum(minimum, distances.amin(dim=(0, 2)))
            maximum = torch.maximum(maximum, distances.amax(dim=(0, 2)))

        if self._omega_rad_s is not None:
            for start in range(0, len(self.series), self.chunk_size):
                times = self._absolute_times_s[start : start + self.chunk_size]
                centers, _, _, _ = self._rigid_rotation_kinematics(times)
                include_centers(centers)
            return minimum, maximum

        frame_count = 0
        for chunk in iter_surface_chunks(
            self.series,
            chunk_size=self.chunk_size,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
        ):
            if chunk.start_index != frame_count:
                raise ValueError(
                    "Surface frames were not yielded in their original index order."
                )
            centers, _, _, _, _, _ = self._surface_kinematics_torch_chunk(chunk)
            include_centers(centers)
            frame_count += chunk.frame_count
        if frame_count != len(self.series):
            raise ValueError(
                f"Expected {len(self.series)} moving frames, received {frame_count}."
            )
        return minimum, maximum

    def _finalize_result(
        self,
        observers_m: torch.Tensor,
        thickness_pressure_pa: torch.Tensor,
        loading_pressure_pa: torch.Tensor,
        maximum_delay_s: torch.Tensor,
        *,
        sample_count: int,
        propagation_algorithm: Literal["uniform", "nonuniform"],
        remove_mean: bool,
    ) -> F1AResult:
        """Apply common pressure scaling and construct solver result metadata."""

        thickness_pressure_pa /= FOUR_PI
        loading_pressure_pa /= FOUR_PI
        if remove_mean:
            thickness_pressure_pa -= thickness_pressure_pa.mean(
                dim=1,
                keepdim=True,
            )
            loading_pressure_pa -= loading_pressure_pa.mean(dim=1, keepdim=True)
        return F1AResult(
            observer_times_s=self._to_torch(self.series.source_times_s),
            thickness_pressure_pa=thickness_pressure_pa,
            loading_pressure_pa=loading_pressure_pa,
            total_pressure_pa=thickness_pressure_pa + loading_pressure_pa,
            observer_m=observers_m,
            rpm=self.rpm,
            maximum_delay_s=maximum_delay_s,
            response_delay_s=self.response_delay_s,
            source_time_count=sample_count,
            propagation_algorithm=propagation_algorithm,
            device=str(self.device),
            chunk_size=self.chunk_size,
        )

    def _run_moving(
        self,
        observers_m: Sequence[Sequence[float]] | torch.Tensor,
        *,
        workers: int,
        invalid_time_policy: InvalidTimePolicy,
        progress_callback: ProgressCallback | None,
        progress_every: int,
        remove_mean: bool,
        algorithm: PropagationAlgorithm,
    ) -> F1AResult:
        observers = self._validated_observers(observers_m)
        resolved_algorithm, _ = self._resolve_algorithm(algorithm)
        minimum_distance, maximum_distance = self._moving_distance_extrema(
            observers,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
        )
        if self.response_delay_s is None:
            maximum_delays = maximum_distance / self.sound_speed_m_s
            reference_distance = torch.zeros_like(minimum_distance)
        else:
            maximum_delays = (
                maximum_distance - minimum_distance
            ) / self.sound_speed_m_s + self.response_delay_s
            reference_distance = minimum_distance
        self.maximum_delay_s = float(maximum_delays.max())

        observer_count = observers.shape[0]
        sample_count = len(self.series)
        query_times = self._query_times_s
        thickness_output = torch.zeros(
            (observer_count, sample_count),
            device=self.device,
            dtype=self.torch_dtype,
        )
        loading_output = torch.zeros_like(thickness_output)
        derivative = _TorchFiniteDifferenceDerivative(
            adjust_time_step=self.adjust_time_step,
            device=self.device,
            dtype=self.torch_dtype,
        )

        previous_thickness: torch.Tensor | None = None
        previous_loading: torch.Tensor | None = None
        previous_arrival_s: torch.Tensor | None = None
        frame_count = 0

        def process_chunk(chunk: SurfaceChunk) -> None:
            nonlocal previous_thickness, previous_loading, previous_arrival_s
            nonlocal frame_count
            if chunk.start_index != frame_count:
                raise ValueError(
                    "Surface frames were not yielded in their original index order."
                )
            source_start_index = chunk.start_index
            thickness, loading, _, arrival_s = self._moving_source_terms_torch_chunk(
                chunk,
                derivative,
                observers,
                reference_distance,
            )
            current_indices = torch.arange(
                source_start_index,
                source_start_index + thickness.shape[0],
                device=self.device,
                dtype=torch.int64,
            )
            for observer_index in range(observer_count):
                observer_thickness = thickness[:, observer_index, :]
                observer_loading = loading[:, observer_index, :]
                observer_arrival = arrival_s[:, observer_index, :]
                if previous_thickness is None:
                    self._deposit_initial_exact_torch(
                        thickness_output[observer_index],
                        observer_thickness[0],
                        observer_arrival[0],
                        query_times,
                    )
                    self._deposit_initial_exact_torch(
                        loading_output[observer_index],
                        observer_loading[0],
                        observer_arrival[0],
                        query_times,
                    )
                    if observer_thickness.shape[0] > 1:
                        self._deposit_moving_torch_intervals(
                            thickness_output[observer_index],
                            loading_output[observer_index],
                            observer_thickness[:-1],
                            observer_thickness[1:],
                            observer_loading[:-1],
                            observer_loading[1:],
                            observer_arrival[:-1],
                            observer_arrival[1:],
                            current_indices[1:],
                            query_times,
                        )
                else:
                    if previous_loading is None or previous_arrival_s is None:
                        raise RuntimeError("Incomplete moving propagation state.")
                    self._deposit_moving_torch_intervals(
                        thickness_output[observer_index],
                        loading_output[observer_index],
                        torch.cat(
                            (
                                previous_thickness[observer_index][None, :],
                                observer_thickness[:-1],
                            ),
                            dim=0,
                        ),
                        observer_thickness,
                        torch.cat(
                            (
                                previous_loading[observer_index][None, :],
                                observer_loading[:-1],
                            ),
                            dim=0,
                        ),
                        observer_loading,
                        torch.cat(
                            (
                                previous_arrival_s[observer_index][None, :],
                                observer_arrival[:-1],
                            ),
                            dim=0,
                        ),
                        observer_arrival,
                        current_indices,
                        query_times,
                    )
            previous_thickness = thickness[-1].detach()
            previous_loading = loading[-1].detach()
            previous_arrival_s = arrival_s[-1].detach()
            frame_count += chunk.frame_count

        for chunk in iter_surface_chunks(
            self.series,
            chunk_size=self.chunk_size,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ):
            process_chunk(chunk)
        if frame_count != sample_count:
            raise ValueError(
                f"Expected {sample_count} moving frames, received {frame_count}."
            )

        return self._finalize_result(
            observers,
            thickness_output,
            loading_output,
            maximum_delays,
            sample_count=sample_count,
            propagation_algorithm=resolved_algorithm,
            remove_mean=remove_mean,
        )

    def _run(
        self,
        observers_m: Sequence[Sequence[float]] | torch.Tensor,
        *,
        workers: int,
        invalid_time_policy: InvalidTimePolicy,
        progress_callback: ProgressCallback | None,
        progress_every: int,
        remove_mean: bool,
        algorithm: PropagationAlgorithm,
    ) -> F1AResult:
        if self.moving_surface:
            return self._run_moving(
                observers_m,
                workers=workers,
                invalid_time_policy=invalid_time_policy,
                progress_callback=progress_callback,
                progress_every=progress_every,
                remove_mean=remove_mean,
                algorithm=algorithm,
            )
        observers, distance, direction, delay = self._observer_geometries(observers_m)
        maximum_delays = delay.max(dim=1).values
        self.maximum_delay_s = float(maximum_delays.max())
        resolved_algorithm, uniform_spacing = self._resolve_algorithm(algorithm)
        observer_count = observers.shape[0]
        sample_count = len(self.series)

        face_normals = self._face_normals
        face_areas_m2 = self._face_areas_m2
        observer_area = face_areas_m2[None, :, None]
        observer_distance = distance[..., None]
        coefficients = _FixedObserverCoefficients(
            thickness=(face_areas_m2[None, :] * self.density_kg_m3 / distance),
            loading_rate=(
                observer_area * direction / (self.sound_speed_m_s * observer_distance)
            ),
            loading=observer_area * direction / observer_distance**2,
        )
        del distance, direction, observer_area, observer_distance
        thickness_output = torch.zeros(
            (observer_count, sample_count),
            device=self.device,
            dtype=self.torch_dtype,
        )
        loading_output = torch.zeros_like(thickness_output)
        derivative = _TorchFiniteDifferenceDerivative(
            adjust_time_step=self.adjust_time_step,
            device=self.device,
            dtype=self.torch_dtype,
        )

        torch_uniform_delays: list[_TorchUniformDelays] | None = None
        packed_uniform_delays: _TorchPackedUniformDelays | None = None
        if resolved_algorithm == "uniform":
            if uniform_spacing is None:
                raise RuntimeError("Uniform spacing was not resolved.")
            epsilon_s = float(self._solver_delta_t_s[0]) * 1.0e-6
            packed_uniform_delays = self._torch_packed_uniform_delays(
                delay,
                uniform_spacing,
                epsilon_s,
            )
            if packed_uniform_delays is None:
                torch_uniform_delays = [
                    self._torch_uniform_delays(
                        delay[index],
                        uniform_spacing,
                        epsilon_s,
                    )
                    for index in range(observer_count)
                ]

        previous_state: _FixedPropagationState | None = None
        frame_count = 0

        for chunk in iter_surface_chunks(
            self.series,
            chunk_size=self.chunk_size,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
            progress_callback=progress_callback,
            progress_every=progress_every,
        ):
            if chunk.start_index != frame_count:
                raise ValueError(
                    "Surface frames were not yielded in their original index order."
                )
            source_start_index = chunk.start_index
            thickness, loading, absolute_times = self._source_terms_torch_chunk(
                chunk,
                derivative,
                face_normals,
                coefficients,
            )
            if resolved_algorithm == "uniform":
                self._deposit_fixed_uniform_chunk(
                    thickness_output,
                    loading_output,
                    thickness,
                    loading,
                    source_start_index,
                    packed_uniform_delays,
                    torch_uniform_delays,
                )
            else:
                previous_state = self._deposit_fixed_nonuniform_chunk(
                    thickness_output,
                    loading_output,
                    thickness,
                    loading,
                    absolute_times,
                    delay,
                    source_start_index,
                    previous_state,
                )
            frame_count += chunk.frame_count
        if frame_count != sample_count:
            raise ValueError(
                f"Expected {sample_count} VTK frames, received {frame_count}."
            )

        return self._finalize_result(
            observers,
            thickness_output,
            loading_output,
            maximum_delays,
            sample_count=sample_count,
            propagation_algorithm=resolved_algorithm,
            remove_mean=remove_mean,
        )

    def predict(
        self,
        observer_m: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor,
        *,
        workers: int = 1,
        invalid_time_policy: InvalidTimePolicy = "error",
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
        remove_mean: bool = False,
        algorithm: PropagationAlgorithm = "auto",
    ) -> F1AResult:
        """Evaluate acoustic pressure at one or more observers.

        Args:
            observer_m: One ``(x, y, z)`` coordinate or an array with shape
                ``(n_observers, 3)``, in metres.
            workers: Number of bounded VTK-reader worker threads.
            invalid_time_policy: Raise on invalid VTK frames or interpolate them.
            progress_callback: Optional ``(completed, total, message)`` callback.
            progress_every: Successful-frame interval between progress callbacks.
            remove_mean: Remove each pressure component's temporal mean.
            algorithm: Arrival interpolation path. ``"auto"`` selects the
                uniform path only when source times and solver ``deltaT`` permit.

        Returns:
            A single-observer result for one coordinate, otherwise a batched
            result whose pressure arrays have shape ``(observers, times)``.

        Raises:
            ValueError: If observer coordinates or runtime options are invalid.
        """
        workers = positive_int(workers, "workers")
        progress_every = positive_int(progress_every, "progress_every")
        observers = self._to_torch(observer_m)
        single_observer = observers.ndim == 1
        if single_observer:
            observers = observers[None, :]
        result = self._run(
            observers,
            workers=workers,
            invalid_time_policy=invalid_time_policy,
            progress_callback=progress_callback,
            progress_every=progress_every,
            remove_mean=remove_mean,
            algorithm=algorithm,
        )
        return result.for_observer(0) if single_observer else result

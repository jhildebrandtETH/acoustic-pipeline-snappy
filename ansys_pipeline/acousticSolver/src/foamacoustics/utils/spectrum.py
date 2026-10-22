"""Pressure-level and single-sided SPL spectrum utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .validation import positive_float


REFERENCE_PRESSURE_PA = 20.0e-6
PressureLike = torch.Tensor | Sequence[float]
SpectrumLike = PressureLike | Sequence[Sequence[float]]
_DECIBEL_POWER_SCALE = math.log(10.0) / 10.0


@dataclass(frozen=True, slots=True)
class PressureSpectrum:
    """Single-sided RMS pressure spectrum and its SPL representation.

    Attributes:
        frequencies_hz: Frequency at each real-FFT bin.
        pressure_rms_pa: RMS pressure amplitude in each bin.
        spl_db: Bin levels in decibels relative to the reference pressure.
        frequency_resolution_hz: Spacing between adjacent frequency bins.
    """

    frequencies_hz: torch.Tensor
    pressure_rms_pa: torch.Tensor
    spl_db: torch.Tensor
    frequency_resolution_hz: float


def _pressure_trace(pressure_pa: PressureLike) -> torch.Tensor:
    pressure = torch.as_tensor(pressure_pa, dtype=torch.float64)
    if pressure.ndim != 1 or pressure.numel() < 2:
        raise ValueError(
            "pressure_pa must be a one-dimensional array with at least two samples."
        )
    if not torch.isfinite(pressure).all().item():
        raise ValueError("pressure_pa must contain only finite values.")
    return pressure


def pressure_to_oaspl(
    pressure_pa: PressureLike,
    *,
    reference_pressure_pa: float = REFERENCE_PRESSURE_PA,
    remove_mean: bool = True,
) -> torch.Tensor:
    """Convert a pressure history in pascals to OASPL.

    Args:
        pressure_pa: One-dimensional pressure history in pascals.
        reference_pressure_pa: Positive reference pressure in pascals.
        remove_mean: Remove the temporal mean before computing RMS.

    Returns:
        Scalar OASPL tensor in decibels.

    Raises:
        ValueError: If the trace or reference pressure is invalid.
    """
    pressure = _pressure_trace(pressure_pa)
    if remove_mean:
        pressure = pressure - pressure.mean()
    reference = positive_float(reference_pressure_pa, "reference_pressure_pa")
    rms = torch.sqrt(torch.mean(pressure**2))
    return 20.0 * torch.log10(rms / reference)


def spl_spectrum_to_oaspl(
    spl_db: SpectrumLike,
    frequencies_hz: PressureLike,
    *,
    frequency_ceiling_hz: float | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Combine an SPL spectrum into an overall sound pressure level.

    Args:
        spl_db: Per-bin sound pressure levels in decibels.
        frequencies_hz: One-dimensional frequency at each spectrum bin.
        frequency_ceiling_hz: Include only bins at or below this frequency.
            ``None`` includes every bin.
        dim: Frequency-bin dimension to combine.

    Returns:
        OASPL in decibels with the frequency-bin dimension removed.

    Raises:
        ValueError: If the spectrum or frequencies are invalid, their bin
            counts differ, or the ceiling excludes every bin.

    Negative-infinity values are accepted so zero-energy spectrum bins can be
    represented naturally.
    """
    levels, frequencies, normalized_dim = _prepare_spl_spectrum(
        spl_db,
        frequencies_hz,
        frequency_ceiling_hz=frequency_ceiling_hz,
        dim=dim,
    )
    return _combine_decibel_levels(levels, normalized_dim)


def spl_spectrum_to_a_weighted_oaspl(
    spl_db: SpectrumLike,
    frequencies_hz: PressureLike,
    *,
    frequency_ceiling_hz: float | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Combine an SPL spectrum into an A-weighted overall level.

    Args:
        spl_db: Per-bin sound pressure levels in decibels.
        frequencies_hz: One-dimensional frequency at each spectrum bin.
        frequency_ceiling_hz: Include only bins at or below this frequency.
            ``None`` includes every bin.
        dim: Frequency-bin dimension to combine.

    Returns:
        A-weighted OASPL in decibels with the frequency-bin dimension removed.

    Raises:
        ValueError: If the spectrum or frequencies are invalid, their bin
            counts differ, or the ceiling excludes every bin.

    The weighting follows the standard analog A-weighting response. The
    zero-frequency bin receives negative-infinite weight and contributes no
    energy to the result.
    """
    levels, frequencies, normalized_dim = _prepare_spl_spectrum(
        spl_db,
        frequencies_hz,
        frequency_ceiling_hz=frequency_ceiling_hz,
        dim=dim,
    )
    frequency_squared = frequencies**2
    response = (
        (12194.0**2)
        * frequency_squared**2
        / (
            (frequency_squared + 20.6**2)
            * torch.sqrt(
                (frequency_squared + 107.7**2) * (frequency_squared + 737.9**2)
            )
            * (frequency_squared + 12194.0**2)
        )
    )
    weighting_db = 20.0 * torch.log10(response) + 2.0
    weighting_shape = [1] * levels.ndim
    weighting_shape[normalized_dim] = frequencies.numel()
    weighted_levels = levels + weighting_db.reshape(weighting_shape)

    return _combine_decibel_levels(weighted_levels, normalized_dim)


def _combine_decibel_levels(levels: torch.Tensor, dim: int) -> torch.Tensor:
    """Combine independent decibel levels by summing their linear energy."""

    return (
        torch.logsumexp(levels * _DECIBEL_POWER_SCALE, dim=dim) / _DECIBEL_POWER_SCALE
    )


def _prepare_spl_spectrum(
    spl_db: SpectrumLike,
    frequencies_hz: PressureLike,
    *,
    frequency_ceiling_hz: float | None,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    levels = torch.as_tensor(spl_db)
    if not levels.is_floating_point():
        levels = levels.to(dtype=torch.float64)
    if levels.ndim == 0:
        raise ValueError("spl_db must have at least one dimension.")
    if not -levels.ndim <= dim < levels.ndim:
        raise IndexError(
            f"dim must be in [{-levels.ndim}, {levels.ndim - 1}], but got {dim}."
        )
    normalized_dim = dim % levels.ndim
    if levels.shape[normalized_dim] == 0:
        raise ValueError("spl_db must contain at least one frequency bin.")
    if torch.isnan(levels).any().item() or torch.isposinf(levels).any().item():
        raise ValueError("spl_db must not contain NaN or positive-infinity values.")

    frequencies = torch.as_tensor(
        frequencies_hz,
        device=levels.device,
        dtype=levels.dtype,
    )
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional.")
    if frequencies.numel() != levels.shape[normalized_dim]:
        raise ValueError("frequencies_hz must contain one value for each spectrum bin.")
    if (
        not torch.isfinite(frequencies).all().item()
        or torch.any(frequencies < 0.0).item()
    ):
        raise ValueError("frequencies_hz must contain finite, nonnegative values.")

    if frequency_ceiling_hz is not None:
        ceiling = positive_float(frequency_ceiling_hz, "frequency_ceiling_hz")
        included_indices = torch.nonzero(
            frequencies <= ceiling,
            as_tuple=False,
        ).flatten()
        if included_indices.numel() == 0:
            raise ValueError("frequency_ceiling_hz excludes every spectrum bin.")
        frequencies = frequencies.index_select(0, included_indices)
        levels = levels.index_select(normalized_dim, included_indices)

    return levels, frequencies, normalized_dim


def pressure_to_spl_spectrum(
    pressure_pa: PressureLike,
    sampling_rate_hz: float,
    *,
    reference_pressure_pa: float = REFERENCE_PRESSURE_PA,
    remove_mean: bool = True,
) -> PressureSpectrum:
    """Return a single-sided RMS-per-bin SPL spectrum from uniform samples.

    Args:
        pressure_pa: One-dimensional, uniformly sampled pressure in pascals.
        sampling_rate_hz: Positive sample rate in hertz.
        reference_pressure_pa: Positive reference pressure in pascals.
        remove_mean: Remove the temporal mean before the FFT.

    Returns:
        Frequencies, RMS pressure amplitudes, SPL values, and bin resolution.

    Raises:
        ValueError: If the trace, sample rate, or reference pressure is invalid.

    The interior FFT bins are multiplied by ``sqrt(2)`` so the returned
    pressure amplitudes are RMS values for real-valued sinusoidal components.
    No window is applied; use an integer-period record for tonal analysis.
    """
    pressure = _pressure_trace(pressure_pa)
    sampling_rate = positive_float(sampling_rate_hz, "sampling_rate_hz")
    reference = positive_float(reference_pressure_pa, "reference_pressure_pa")
    if remove_mean:
        pressure = pressure - pressure.mean()

    sample_count = pressure.numel()
    pressure_rms = torch.abs(torch.fft.rfft(pressure)) / sample_count
    if sample_count % 2 == 0:
        pressure_rms[1:-1] *= 2.0**0.5
    else:
        pressure_rms[1:] *= 2.0**0.5
    frequencies = torch.fft.rfftfreq(
        sample_count,
        d=1.0 / sampling_rate,
        device=pressure.device,
        dtype=pressure.dtype,
    )
    spl = 20.0 * torch.log10(pressure_rms / reference)
    return PressureSpectrum(
        frequencies_hz=frequencies,
        pressure_rms_pa=pressure_rms,
        spl_db=spl,
        frequency_resolution_hz=sampling_rate / sample_count,
    )

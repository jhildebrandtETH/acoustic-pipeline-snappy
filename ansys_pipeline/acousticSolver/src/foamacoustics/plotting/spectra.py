"""Sound-pressure-level spectrum plotting helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.result import F1AResult
from ..utils.spectrum import REFERENCE_PRESSURE_PA, PressureSpectrum
from ..utils.validation import positive_float, positive_int
from ._common import (
    CLASSIC_COLORS,
    ArrayLike,
    FigureSize,
    FrequencyLimits,
    finite_1d,
    line_properties,
    validated_figsize,
)


class SpectrumLike(Protocol):
    """Structural type accepted by spectrum comparison helpers."""

    @property
    def frequencies_hz(self) -> torch.Tensor:
        """Frequency coordinate for every spectral bin."""
        ...

    @property
    def spl_db(self) -> torch.Tensor:
        """Sound-pressure level at every spectral bin."""
        ...


@dataclass(frozen=True, slots=True)
class SPLPlotResult:
    """Computed multi-observer spectra and the axes that display them."""

    axes: tuple[Axes, ...]
    frequencies_hz: torch.Tensor
    amplitude_pa: torch.Tensor
    spl_db: torch.Tensor
    frequency_resolution_hz: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class ObserverSpectrumPlots:
    """One figure and axes for each observer in a spectrum comparison."""

    figures: tuple[Figure, ...]
    axes: tuple[Axes, ...]


@dataclass(frozen=True, slots=True)
class _ComputedSpectra:
    frequencies_hz: torch.Tensor
    amplitude_pa: torch.Tensor
    spl_db: torch.Tensor
    frequency_resolution_hz: float
    sample_count: int


def plot_spl_spectrum(
    data: PressureSpectrum | F1AResult,
    *,
    rotations: float | None = None,
    blade_count: int = 2,
    observer_elevation_deg: float | None = None,
    blade_passing_frequency_hz: float | None = None,
    xlim_hz: FrequencyLimits = (None, 5000.0),
    figsize: FigureSize = (4.5, 3.0),
    reference_pressure_pa: float = REFERENCE_PRESSURE_PA,
    ax: Axes | None = None,
    logarithmic_frequency: bool = True,
) -> Axes | SPLPlotResult:
    """Plot one spectrum or spectra computed directly from an F1A result.

    Args:
        data: A precomputed spectrum or an F1A solver result.
        rotations: Final rotor rotations to use when ``data`` is an F1A result.
        blade_count: Number of blades used to place BPF harmonic lines.
        observer_elevation_deg: Legend elevation for a precomputed spectrum.
        blade_passing_frequency_hz: BPF for a precomputed spectrum. For an F1A
            result it is inferred from ``rpm`` and ``blade_count``.
        xlim_hz: Left and right frequency limits. ``None`` selects the first
            positive bin or Nyquist respectively.
        figsize: Figure size in inches when axes are created.
        reference_pressure_pa: Reference pressure used for result spectra.
        ax: Existing axes for a single spectrum, or ``None`` to create axes.
        logarithmic_frequency: Use a logarithmic frequency axis and omit DC.

    Returns:
        Axes for a precomputed spectrum. For an F1A result, returns the
        computed spectra and one axes per observer.

    Raises:
        ValueError: If the inputs, result time grid, or requested final window
            are invalid.
    """
    validated_size = validated_figsize(figsize)
    if isinstance(data, PressureSpectrum):
        if data.spl_db.ndim != 1:
            raise ValueError("A precomputed spectrum must contain one SPL curve.")
        if ax is None:
            _, ax = plt.subplots(figsize=validated_size, constrained_layout=True)
        _plot_spectrum_curve(
            ax,
            data.frequencies_hz.detach().cpu(),
            data.spl_db.detach().cpu(),
            elevation_deg=observer_elevation_deg,
            line_index=0,
            blade_passing_frequency_hz=blade_passing_frequency_hz,
            xlim_hz=xlim_hz,
            logarithmic_frequency=logarithmic_frequency,
        )
        return ax

    if rotations is None:
        raise ValueError("rotations is required when plotting an F1A result.")
    spectra = _result_spl_spectra(
        data,
        rotations=rotations,
        reference_pressure_pa=reference_pressure_pa,
    )
    observer_count = spectra.spl_db.shape[0]
    if ax is not None and observer_count != 1:
        raise ValueError("ax can only be supplied for a single-observer result.")

    elevations = _observer_elevations_deg(data)
    if blade_passing_frequency_hz is None:
        bpf_hz = (
            positive_float(data.rpm, "result.rpm")
            / 60.0
            * positive_int(blade_count, "blade_count")
        )
    else:
        bpf_hz = positive_float(
            blade_passing_frequency_hz,
            "blade_passing_frequency_hz",
        )
    axes: list[Axes] = []
    for observer_index in range(observer_count):
        observer_ax = ax
        if observer_ax is None:
            _, observer_ax = plt.subplots(
                figsize=validated_size,
                constrained_layout=True,
            )
        _plot_spectrum_curve(
            observer_ax,
            spectra.frequencies_hz,
            spectra.spl_db[observer_index],
            elevation_deg=float(elevations[observer_index]),
            line_index=0,
            blade_passing_frequency_hz=bpf_hz,
            xlim_hz=xlim_hz,
            logarithmic_frequency=logarithmic_frequency,
        )
        axes.append(observer_ax)

    return SPLPlotResult(
        axes=tuple(axes),
        frequencies_hz=spectra.frequencies_hz,
        amplitude_pa=spectra.amplitude_pa,
        spl_db=spectra.spl_db,
        frequency_resolution_hz=spectra.frequency_resolution_hz,
        sample_count=spectra.sample_count,
    )


def plot_spectrum_comparison(
    spectra: Mapping[str, SpectrumLike],
    *,
    xlim_hz: FrequencyLimits = (None, 5000.0),
    blade_passing_frequency_hz: float | None = None,
    bpf_label: str | None = None,
    bpf_color: str = "tab:red",
    colors: Mapping[str, str] | None = None,
    linewidths: Mapping[str, float] | None = None,
    alphas: Mapping[str, float] | None = None,
    title: str | None = None,
    ylabel: str = "SPL [dB re 20 µPa, RMS/bin]",
    legend_location: str = "lower left",
    figsize: FigureSize = (10.0, 4.2),
    logarithmic_frequency: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot named SPL spectra on one set of axes.

    Args:
        spectra: Mapping from legend label to spectrum-like objects. Each
            object needs ``frequencies_hz`` and ``spl_db`` attributes.
        xlim_hz: Requested frequency bounds. ``None`` uses the common data
            bound across all spectra.
        blade_passing_frequency_hz: Optional BPF marker frequency.
        bpf_label: Legend text for the BPF marker. A frequency label is used
            when omitted.
        bpf_color: Matplotlib color for the BPF marker.
        colors: Optional curve colors keyed by label.
        linewidths: Optional curve line widths keyed by label.
        alphas: Optional curve opacities keyed by label.
        title: Optional axes title.
        ylabel: Vertical-axis label.
        legend_location: Matplotlib legend-location string.
        figsize: Figure size in inches when axes are created.
        logarithmic_frequency: Plot frequency on a logarithmic scale.
        ax: Existing axes, or ``None`` to create one.

    Returns:
        Axes containing the comparison.

    Raises:
        ValueError: If no spectra are supplied or their data are invalid.
    """
    if not spectra:
        raise ValueError("spectra must contain at least one curve.")

    validated_spectra: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for label, spectrum in spectra.items():
        frequencies = finite_1d(
            spectrum.frequencies_hz, f"spectra[{label!r}].frequencies_hz"
        )
        levels = finite_1d(spectrum.spl_db, f"spectra[{label!r}].spl_db")
        if frequencies.numel() != levels.numel():
            raise ValueError(f"Spectrum {label!r} has mismatched data lengths.")
        if torch.any(frequencies < 0.0).item():
            raise ValueError(f"Spectrum {label!r} contains negative frequencies.")
        validated_spectra[label] = frequencies, levels

    left_hz, right_hz = _common_frequency_limits(
        tuple(frequencies for frequencies, _ in validated_spectra.values()),
        xlim_hz,
        logarithmic_frequency=logarithmic_frequency,
    )
    if ax is None:
        _, ax = plt.subplots(
            figsize=validated_figsize(figsize), constrained_layout=True
        )

    for index, (label, (frequencies, levels)) in enumerate(validated_spectra.items()):
        plotted = (frequencies >= left_hz) & (frequencies <= right_hz)
        if not torch.any(plotted).item():
            raise ValueError(
                f"Spectrum {label!r} has no samples within the frequency limits."
            )
        default_color, _ = line_properties(index)
        plot = ax.semilogx if logarithmic_frequency else ax.plot
        plot(
            frequencies[plotted],
            levels[plotted],
            color=(colors or {}).get(label, default_color),
            linewidth=(linewidths or {}).get(label, 1.0),
            alpha=(alphas or {}).get(label, 1.0),
            label=label,
        )

    ax.set_xlim(left_hz, right_hz)
    if blade_passing_frequency_hz is not None:
        bpf_hz = positive_float(
            blade_passing_frequency_hz, "blade_passing_frequency_hz"
        )
        ax.axvline(
            bpf_hz,
            color=bpf_color,
            linestyle="--",
            linewidth=1.0,
            label=bpf_label or f"BPF ({bpf_hz:g} Hz)",
        )
    ax.set(title=title, xlabel="Frequency [Hz]", ylabel=ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc=cast("Any", legend_location))
    return ax


def plot_observer_spectra(
    spectra_by_case: Mapping[str, Sequence[SpectrumLike]],
    *,
    observer_titles: Sequence[str] | None = None,
    xlim_hz: FrequencyLimits = (None, 5000.0),
    blade_passing_frequency_hz: float | None = None,
    colors: Mapping[str, str] | None = None,
    linewidths: Mapping[str, float] | None = None,
    alphas: Mapping[str, float] | None = None,
    legend_order: Sequence[str] | None = None,
    figsize: FigureSize = (10.0, 4.2),
) -> ObserverSpectrumPlots:
    """Create one comparison figure for every observer.

    Args:
        spectra_by_case: Named spectrum sequences with one entry per observer.
        observer_titles: Optional title for each observer figure.
        xlim_hz: Frequency limits shared by every figure.
        blade_passing_frequency_hz: Optional BPF marker frequency.
        colors: Optional curve colors keyed by case label.
        linewidths: Optional curve widths keyed by case label.
        alphas: Optional curve opacities keyed by case label.
        legend_order: Optional case order. It must contain every case exactly
            once; otherwise mapping insertion order is used.
        figsize: Size in inches of each observer figure.

    Returns:
        Tuples containing the created figures and axes in observer order.

    Raises:
        ValueError: If cases are missing, have unequal observer counts, or the
            titles/order do not match the supplied data.
    """
    if not spectra_by_case:
        raise ValueError("spectra_by_case must contain at least one case.")
    labels = tuple(spectra_by_case) if legend_order is None else tuple(legend_order)
    if set(labels) != set(spectra_by_case) or len(labels) != len(spectra_by_case):
        raise ValueError("legend_order must contain every case exactly once.")
    observer_counts = {len(spectra_by_case[label]) for label in labels}
    if len(observer_counts) != 1:
        raise ValueError("Every case must contain the same number of observers.")
    observer_count = observer_counts.pop()
    if observer_count == 0:
        raise ValueError("Each case must contain at least one spectrum.")
    if observer_titles is None:
        titles: tuple[str | None, ...] = (None,) * observer_count
    else:
        if len(observer_titles) != observer_count:
            raise ValueError("observer_titles must match the observer count.")
        titles = tuple(observer_titles)

    figures: list[Figure] = []
    axes: list[Axes] = []
    for observer_index, title in enumerate(titles):
        figure, ax = plt.subplots(
            figsize=validated_figsize(figsize), constrained_layout=True
        )
        plot_spectrum_comparison(
            {label: spectra_by_case[label][observer_index] for label in labels},
            xlim_hz=xlim_hz,
            blade_passing_frequency_hz=blade_passing_frequency_hz,
            colors=colors,
            linewidths=linewidths,
            alphas=alphas,
            title=title,
            ax=ax,
        )
        figures.append(figure)
        axes.append(ax)
    return ObserverSpectrumPlots(figures=tuple(figures), axes=tuple(axes))


def _result_spl_spectra(
    result: F1AResult,
    *,
    rotations: float,
    reference_pressure_pa: float,
) -> _ComputedSpectra:
    requested_rotations = positive_float(rotations, "rotations")
    rotation_frequency_hz = positive_float(result.rpm, "result.rpm") / 60.0
    reference_pressure = positive_float(
        reference_pressure_pa,
        "reference_pressure_pa",
    )
    times = result.observer_times_s
    pressure = result.total_pressure_pa
    if times.ndim != 1 or times.numel() < 3:
        raise ValueError("result.observer_times_s must contain at least three samples.")
    if pressure.ndim == 1:
        pressure = pressure[None, :]
    if pressure.ndim != 2 or pressure.shape[1] != times.numel():
        raise ValueError(
            "result.total_pressure_pa must have shape (times,) or (observers, times)."
        )
    if (
        not torch.isfinite(times).all().item()
        or not torch.isfinite(pressure).all().item()
    ):
        raise ValueError("result times and pressures must contain only finite values.")
    time_steps = torch.diff(times)
    if not torch.all(time_steps > 0.0).item():
        raise ValueError("result.observer_times_s must be strictly increasing.")

    maximum_delays = torch.as_tensor(
        result.maximum_delay_s,
        device=times.device,
        dtype=times.dtype,
    ).reshape(-1)
    if maximum_delays.numel() == 1:
        maximum_delays = maximum_delays.expand(pressure.shape[0])
    if maximum_delays.numel() != pressure.shape[0]:
        raise ValueError("result.maximum_delay_s must match the observer count.")

    spectrum_end_s = float(times[-1])
    spectrum_start_s = spectrum_end_s - requested_rotations / rotation_frequency_hz
    if spectrum_start_s < float(maximum_delays.max()):
        raise ValueError(
            "The fully arrived result tail is too short for the requested rotations."
        )
    spectrum_window = (times >= spectrum_start_s) & (times <= spectrum_end_s)
    spectrum_time_s = times[spectrum_window]
    spectrum_pressure_pa = pressure[:, spectrum_window]
    if spectrum_time_s.numel() < 3:
        raise ValueError("At least three fully arrived samples are required.")

    spectrum_dt_s = float(torch.median(torch.diff(spectrum_time_s)))
    uniform_time_s = torch.arange(
        float(spectrum_time_s[0]),
        float(spectrum_time_s[-1]) + 0.5 * spectrum_dt_s,
        spectrum_dt_s,
        device=spectrum_time_s.device,
        dtype=spectrum_time_s.dtype,
    )
    right = torch.searchsorted(spectrum_time_s, uniform_time_s).clamp(
        1,
        spectrum_time_s.numel() - 1,
    )
    left = right - 1
    weight = (uniform_time_s - spectrum_time_s[left]) / (
        spectrum_time_s[right] - spectrum_time_s[left]
    )
    uniform_pressure_pa = spectrum_pressure_pa[:, left] + weight[None, :] * (
        spectrum_pressure_pa[:, right] - spectrum_pressure_pa[:, left]
    )
    uniform_pressure_pa = uniform_pressure_pa - uniform_pressure_pa.mean(
        dim=1,
        keepdim=True,
    )

    window = torch.hann_window(
        uniform_pressure_pa.shape[-1],
        periodic=False,
        device=uniform_pressure_pa.device,
        dtype=uniform_pressure_pa.dtype,
    )
    frequencies_hz = torch.fft.rfftfreq(
        uniform_pressure_pa.shape[-1],
        d=spectrum_dt_s,
        device=uniform_pressure_pa.device,
        dtype=uniform_pressure_pa.dtype,
    )
    amplitude_pa = (
        2.0
        * torch.abs(torch.fft.rfft(uniform_pressure_pa * window, dim=-1))
        / window.sum()
    )
    amplitude_pa[:, 0] *= 0.5
    if uniform_pressure_pa.shape[-1] % 2 == 0:
        amplitude_pa[:, -1] *= 0.5
    spl_db = 20.0 * torch.log10(
        amplitude_pa.clamp_min(torch.finfo(amplitude_pa.dtype).tiny)
        / reference_pressure
    )

    return _ComputedSpectra(
        frequencies_hz=frequencies_hz.detach().cpu(),
        amplitude_pa=amplitude_pa.detach().cpu(),
        spl_db=spl_db.detach().cpu(),
        frequency_resolution_hz=1.0 / (uniform_time_s.numel() * spectrum_dt_s),
        sample_count=uniform_time_s.numel(),
    )


def _plot_spectrum_curve(
    ax: Axes,
    frequencies_hz: ArrayLike,
    spl_db: ArrayLike,
    *,
    elevation_deg: float | None,
    line_index: int,
    blade_passing_frequency_hz: float | None,
    xlim_hz: FrequencyLimits,
    logarithmic_frequency: bool,
) -> None:
    frequencies = finite_1d(frequencies_hz, "frequencies_hz")
    levels = finite_1d(spl_db, "spl_db")
    if frequencies.numel() != levels.numel():
        raise ValueError("frequencies_hz and spl_db must have the same length.")
    if torch.any(frequencies < 0.0).item():
        raise ValueError("frequencies_hz must be nonnegative.")

    start = 1 if logarithmic_frequency and frequencies[0] == 0.0 else 0
    plot_frequencies = frequencies[start:]
    plot_levels = levels[start:]
    if plot_frequencies.numel() == 0:
        raise ValueError("The spectrum must contain a positive-frequency bin.")
    left_hz, right_hz = _resolved_frequency_limits(
        plot_frequencies,
        xlim_hz,
        logarithmic_frequency=logarithmic_frequency,
    )
    color, linestyle = line_properties(line_index)
    label = (
        "SPL spectrum" if elevation_deg is None else f"Elevation {elevation_deg:+g}°"
    )
    ax.plot(
        plot_frequencies,
        plot_levels,
        color=color,
        linestyle=linestyle,
        linewidth=1.0,
        label=label,
    )

    if logarithmic_frequency:
        ax.set_xscale("log")
    ax.set_xlim(left_hz, right_hz)
    if blade_passing_frequency_hz is not None:
        bpf_hz = positive_float(
            blade_passing_frequency_hz,
            "blade_passing_frequency_hz",
        )
        first_harmonic = max(1, math.ceil(left_hz / bpf_hz))
        last_harmonic = math.floor(right_hz / bpf_hz + 1.0e-12)
        for harmonic in range(first_harmonic, last_harmonic + 1):
            ax.axvline(
                harmonic * bpf_hz,
                color=CLASSIC_COLORS[-1],
                linestyle=":",
                linewidth=0.8,
                alpha=0.65,
                label="BPF harmonics" if harmonic == first_harmonic else "_nolegend_",
            )

    ax.set(xlabel="Frequency [Hz]", ylabel="SPL [dB re 20 µPa]")
    ax.set_title("")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)


def _observer_elevations_deg(result: F1AResult) -> torch.Tensor:
    observers = result.observer_m
    if observers.ndim == 1:
        observers = observers[None, :]
    if observers.ndim != 2 or observers.shape[1] != 3:
        raise ValueError("result.observer_m must have shape (3,) or (observers, 3).")
    if observers.shape[0] != (
        1 if result.total_pressure_pa.ndim == 1 else result.total_pressure_pa.shape[0]
    ):
        raise ValueError("result.observer_m must match the observer count.")
    return torch.rad2deg(torch.atan2(observers[:, 1], observers[:, 0])).detach().cpu()


def _resolved_frequency_limits(
    frequencies_hz: torch.Tensor,
    xlim_hz: FrequencyLimits,
    *,
    logarithmic_frequency: bool,
) -> tuple[float, float]:
    requested_left, requested_right = xlim_hz
    available_left = float(frequencies_hz[0])
    available_right = float(frequencies_hz[-1])
    left_hz = (
        available_left
        if requested_left is None
        else positive_float(requested_left, "xlim_hz[0]")
    )
    right_hz = (
        available_right
        if requested_right is None
        else min(positive_float(requested_right, "xlim_hz[1]"), available_right)
    )
    if logarithmic_frequency and left_hz <= 0.0:
        raise ValueError("The left logarithmic frequency limit must be positive.")
    if right_hz <= left_hz:
        raise ValueError("The right frequency limit must exceed the left limit.")
    return left_hz, right_hz


def _common_frequency_limits(
    frequencies: Sequence[torch.Tensor],
    requested: FrequencyLimits,
    *,
    logarithmic_frequency: bool,
) -> tuple[float, float]:
    left_candidates = []
    right_candidates = []
    for values in frequencies:
        plottable = values[values > 0.0] if logarithmic_frequency else values
        if plottable.numel() == 0:
            raise ValueError("Every spectrum must contain a plottable frequency.")
        left_candidates.append(float(plottable[0]))
        right_candidates.append(float(plottable[-1]))
    available_left = max(left_candidates)
    available_right = min(right_candidates)
    left = (
        available_left
        if requested[0] is None
        else positive_float(requested[0], "xlim_hz[0]")
    )
    right = (
        available_right
        if requested[1] is None
        else min(positive_float(requested[1], "xlim_hz[1]"), available_right)
    )
    if logarithmic_frequency and left <= 0.0:
        raise ValueError("The left logarithmic frequency limit must be positive.")
    if right <= left:
        raise ValueError("The common right frequency limit must exceed the left limit.")
    return left, right

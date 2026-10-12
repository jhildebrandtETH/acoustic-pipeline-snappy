"""Reusable diagnostic plots extracted from the project notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]

from ..utils.validation import positive_float, positive_int
from ._common import (
    ArrayLike,
    FigureSize,
    finite_1d,
    finite_numpy_1d,
    validated_figsize,
)
from .pressure import MultiAxesPlot
from .spectra import SpectrumLike


def plot_surface_pressure_preview(
    face_centers_m: ArrayLike,
    pressure_pa: ArrayLike,
    *,
    title: str,
    max_points: int = 7000,
    figsize: FigureSize = (8.0, 6.0),
) -> Axes3D:
    """Plot a decimated three-dimensional preview of sampled face pressure.

    Args:
        face_centers_m: Face-center coordinates with shape ``(faces, 3)``.
        pressure_pa: Pressure value for every face.
        title: Axes title identifying the sampled surface.
        max_points: Approximate maximum number of markers to render.
        figsize: Figure size in inches.

    Returns:
        The created three-dimensional axes.

    Raises:
        ValueError: If arrays are invalid or have inconsistent lengths.
    """
    centers = _finite_numpy(face_centers_m, "face_centers_m", ndim=2)
    pressure = finite_numpy_1d(pressure_pa, "pressure_pa")
    if centers.shape[1] != 3:
        raise ValueError("face_centers_m must have shape (faces, 3).")
    if centers.shape[0] != pressure.size:
        raise ValueError("pressure_pa must contain one value per face center.")
    if max_points < 1:
        raise ValueError("max_points must be positive.")
    stride = max(1, int(np.ceil(centers.shape[0] / max_points)))
    centers = centers[::stride]
    pressure = pressure[::stride]

    figure = plt.figure(figsize=validated_figsize(figsize), constrained_layout=True)
    ax = cast("Axes3D", figure.add_subplot(projection="3d"))
    points = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=pressure,
        s=4,
        cmap="coolwarm",
    )
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=title)
    figure.colorbar(points, ax=ax, shrink=0.75, label="Pressure [Pa]")
    return ax


def plot_blade_metric(
    values: ArrayLike,
    *,
    radius_over_r: ArrayLike,
    local_chord_m: ArrayLike,
    surface_side: ArrayLike,
    blade_id: ArrayLike,
    side_names: Sequence[str],
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    symmetric: bool = False,
    percentile: float = 99.5,
    figsize: FigureSize = (12.0, 7.5),
) -> MultiAxesPlot:
    """Map a facewise scalar metric over both sides of every blade.

    Args:
        values: Scalar value for every face.
        radius_over_r: Normalized face-center radius ``r/R``.
        local_chord_m: Aligned local chord coordinate in metres.
        surface_side: Integer side label, starting at zero.
        blade_id: Integer blade label, starting at zero.
        side_names: Display name for each surface-side row.
        title: Figure-level title.
        colorbar_label: Label for the shared colorbar.
        cmap: Matplotlib colormap name.
        symmetric: Center color limits around zero.
        percentile: Robust upper percentile used for color limits.
        figsize: Overall figure size in inches.

    Returns:
        The created figure and flattened axes in row-major order.

    Raises:
        ValueError: If arrays differ in length, labels are invalid, or the
            selected percentile is outside ``(50, 100]``.
    """
    data = finite_numpy_1d(values, "values")
    radius = finite_numpy_1d(radius_over_r, "radius_over_r")
    chord = finite_numpy_1d(local_chord_m, "local_chord_m")
    sides = _integer_labels(surface_side, "surface_side")
    blades = _integer_labels(blade_id, "blade_id")
    _require_same_length(data, radius, chord, sides, blades)
    if not 50.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (50, 100].")
    side_count = int(sides.max()) + 1
    blade_count = int(blades.max()) + 1
    if len(side_names) != side_count:
        raise ValueError("side_names must contain one label per surface side.")
    vmin, vmax = _color_limits(data, symmetric=symmetric, percentile=percentile)

    figure, axes_array = plt.subplots(
        side_count,
        blade_count,
        figsize=validated_figsize(figsize),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    plotted = None
    for side in range(side_count):
        for blade in range(blade_count):
            ax = axes_array[side, blade]
            mask = (sides == side) & (blades == blade)
            plotted = ax.scatter(
                radius[mask],
                1.0e3 * chord[mask],
                c=data[mask],
                s=5,
                linewidths=0.0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_title(f"Blade {blade}, {side_names[side]}")
            ax.set_xlim(0.0, 1.02)
            ax.set_ylabel("Aligned local chord [mm]")
            ax.set_aspect("auto")
    for ax in axes_array[-1, :]:
        ax.set_xlabel("Face-center radius r/R")
    figure.suptitle(title)
    if plotted is not None:
        figure.colorbar(
            plotted,
            ax=axes_array,
            shrink=0.88,
            label=colorbar_label,
        )
    axes = tuple(ax for row in axes_array for ax in row)
    return MultiAxesPlot(figure=figure, axes=axes)


def plot_blade_pressure_snapshot(
    true_pressure_pa: ArrayLike,
    predicted_pressure_pa: ArrayLike,
    *,
    radius_over_r: ArrayLike,
    local_chord_m: ArrayLike,
    surface_side: ArrayLike,
    blade_id: ArrayLike,
    selected_blade: int,
    side_names: Sequence[str],
    title: str,
    percentile: float = 99.5,
    figsize: FigureSize = (13.0, 7.2),
) -> MultiAxesPlot:
    """Compare true, predicted, and error pressure maps for one blade.

    Args:
        true_pressure_pa: True face pressure at one phase.
        predicted_pressure_pa: Predicted face pressure at the same phase.
        radius_over_r: Normalized face-center radius ``r/R``.
        local_chord_m: Aligned local chord coordinate in metres.
        surface_side: Integer side label for each face.
        blade_id: Integer blade label for each face.
        selected_blade: Blade to display.
        side_names: Display name for each surface-side row.
        title: Figure-level title, typically including phase information.
        percentile: Robust symmetric/error and shared-pressure limit percentile.
        figsize: Overall figure size in inches.

    Returns:
        The created figure and axes in row-major order.

    Raises:
        ValueError: If inputs are invalid or the selected blade has no faces.
    """
    true = finite_numpy_1d(true_pressure_pa, "true_pressure_pa")
    predicted = finite_numpy_1d(predicted_pressure_pa, "predicted_pressure_pa")
    radius = finite_numpy_1d(radius_over_r, "radius_over_r")
    chord = finite_numpy_1d(local_chord_m, "local_chord_m")
    sides = _integer_labels(surface_side, "surface_side")
    blades = _integer_labels(blade_id, "blade_id")
    _require_same_length(true, predicted, radius, chord, sides, blades)
    if not 50.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (50, 100].")
    blade_mask = blades == selected_blade
    if not np.any(blade_mask):
        raise ValueError(f"selected_blade {selected_blade} has no faces.")
    side_count = int(sides.max()) + 1
    if len(side_names) != side_count:
        raise ValueError("side_names must contain one label per surface side.")

    error = predicted - true
    shared_values = np.concatenate((true[blade_mask], predicted[blade_mask]))
    tail = 0.5 * (100.0 - percentile)
    shared_vmin, shared_vmax = np.percentile(shared_values, [tail, 100.0 - tail])
    error_limit = float(np.percentile(np.abs(error[blade_mask]), percentile))
    fields = (
        (true, "True pressure", "viridis", shared_vmin, shared_vmax, "Pressure [Pa]"),
        (
            predicted,
            "Predicted pressure",
            "viridis",
            shared_vmin,
            shared_vmax,
            "Pressure [Pa]",
        ),
        (
            error,
            "Predicted − true",
            "coolwarm",
            -error_limit,
            error_limit,
            "Error [Pa]",
        ),
    )
    figure, axes_array = plt.subplots(
        side_count,
        3,
        figsize=validated_figsize(figsize),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for side in range(side_count):
        for column, (values, label, cmap, vmin, vmax, colorbar_label) in enumerate(
            fields
        ):
            ax = axes_array[side, column]
            mask = blade_mask & (sides == side)
            scatter = ax.scatter(
                radius[mask],
                1.0e3 * chord[mask],
                c=values[mask],
                s=6,
                linewidths=0.0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_title(f"{label}\n{side_names[side]}")
            ax.set_ylabel("Aligned local chord [mm]")
            figure.colorbar(scatter, ax=ax, label=colorbar_label)
    for ax in axes_array[-1, :]:
        ax.set_xlabel("Face-center radius r/R")
    figure.suptitle(title)
    return MultiAxesPlot(
        figure=figure,
        axes=tuple(ax for row in axes_array for ax in row),
    )


def plot_error_concentration(
    cumulative_face_fraction: ArrayLike,
    cumulative_area_fraction: ArrayLike,
    cumulative_error_fraction: ArrayLike,
    *,
    annotation_percentages: Sequence[float] = (1.0, 5.0, 10.0),
    figsize: FigureSize = (7.5, 4.8),
    ax: Axes | None = None,
) -> Axes:
    """Plot cumulative area-weighted error against face count and area.

    Fractions are expected on a zero-to-one scale; axes and annotations are
    displayed as percentages.

    Args:
        cumulative_face_fraction: Ranked cumulative fraction of faces.
        cumulative_area_fraction: Ranked cumulative fraction of physical area.
        cumulative_error_fraction: Ranked cumulative fraction of error.
        annotation_percentages: Face-count percentages to annotate.
        figsize: Figure size in inches when axes are created.
        ax: Existing axes, or ``None`` to create one.

    Returns:
        Axes containing both cumulative curves.

    Raises:
        ValueError: If inputs differ in length or do not lie in ``[0, 1]``.
    """
    face_fraction = finite_numpy_1d(
        cumulative_face_fraction, "cumulative_face_fraction"
    )
    area_fraction = finite_numpy_1d(
        cumulative_area_fraction, "cumulative_area_fraction"
    )
    error_fraction = finite_numpy_1d(
        cumulative_error_fraction, "cumulative_error_fraction"
    )
    _require_same_length(face_fraction, area_fraction, error_fraction)
    for name, values in (
        ("cumulative_face_fraction", face_fraction),
        ("cumulative_area_fraction", area_fraction),
        ("cumulative_error_fraction", error_fraction),
    ):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1].")
    if ax is None:
        _, ax = plt.subplots(
            figsize=validated_figsize(figsize), constrained_layout=True
        )
    face_percent = 100.0 * face_fraction
    error_percent = 100.0 * error_fraction
    ax.plot(face_percent, error_percent, label="vs. face count")
    ax.plot(100.0 * area_fraction, error_percent, label="vs. physical area")
    for percentage in annotation_percentages:
        if not 0.0 < percentage <= 100.0:
            raise ValueError("annotation_percentages must lie in (0, 100].")
        index = min(
            int(np.searchsorted(face_percent, percentage)), len(face_percent) - 1
        )
        ax.scatter(face_percent[index], error_percent[index], s=28, zorder=3)
        ax.annotate(
            f"top {percentage:g}% faces → {error_percent[index]:.1f}% error",
            (face_percent[index], error_percent[index]),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=8,
        )
    ax.plot(
        [0.0, 100.0],
        [0.0, 100.0],
        color="0.5",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set(
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        xlabel="Cumulative selected faces or surface area [%]",
        ylabel="Cumulative area-weighted pressure-error MSE [%]",
        title="Pressure-error concentration",
    )
    ax.legend()
    return ax


def plot_face_pressure_histories(
    phase_deg: ArrayLike,
    true_pressure_pa: ArrayLike,
    predicted_pressure_pa: ArrayLike,
    *,
    face_titles: Sequence[str],
    figsize: FigureSize = (10.0, 8.5),
) -> MultiAxesPlot:
    """Plot true and predicted phase histories for selected faces.

    Args:
        phase_deg: Rotor phase coordinates.
        true_pressure_pa: Array with shape ``(phase, selected_faces)``.
        predicted_pressure_pa: Array with the same shape as true pressure.
        face_titles: Title for each selected-face column.
        figsize: Overall figure size in inches.

    Returns:
        The created figure and one axes per selected face.

    Raises:
        ValueError: If arrays or titles have inconsistent shapes.
    """
    phase = finite_numpy_1d(phase_deg, "phase_deg")
    true = _finite_numpy(true_pressure_pa, "true_pressure_pa", ndim=2)
    predicted = _finite_numpy(predicted_pressure_pa, "predicted_pressure_pa", ndim=2)
    if true.shape != predicted.shape or true.shape[0] != phase.size:
        raise ValueError("Pressure arrays must share shape (phase, selected_faces).")
    if true.shape[1] != len(face_titles) or not face_titles:
        raise ValueError("face_titles must contain one title per selected face.")
    figure, axes_array = plt.subplots(
        true.shape[1],
        1,
        figsize=validated_figsize(figsize),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = tuple(axes_array[:, 0])
    for face_index, (ax, title) in enumerate(zip(axes, face_titles, strict=True)):
        ax.plot(
            phase,
            true[:, face_index],
            color="black",
            linewidth=1.0,
            label="true",
        )
        ax.plot(
            phase,
            predicted[:, face_index],
            color="tab:blue",
            linewidth=1.0,
            label="predicted",
        )
        ax.set_ylabel("p [Pa]")
        ax.set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel("Rotor phase [deg]")
    return MultiAxesPlot(figure=figure, axes=axes)


def plot_harmonic_levels(
    harmonic_orders: ArrayLike,
    levels_by_case_db: Mapping[str, ArrayLike],
    *,
    blade_count: int,
    title: str,
    ylabel: str = "Compact-proxy SPL [dB re 20 µPa RMS]",
    colors: Mapping[str, str] | None = None,
    figsize: FigureSize = (9.0, 4.8),
    ax: Axes | None = None,
) -> Axes:
    """Compare tonal levels by shaft order and highlight BPF harmonics.

    Args:
        harmonic_orders: Positive integer shaft orders.
        levels_by_case_db: Named level curve for each shaft order.
        blade_count: Physical blade count used to identify BPF harmonics.
        title: Axes title.
        ylabel: Vertical-axis label.
        colors: Optional curve colors keyed by case label.
        figsize: Figure size in inches when axes are created.
        ax: Existing axes, or ``None`` to create one.

    Returns:
        Axes containing the harmonic comparison.

    Raises:
        ValueError: If orders, blade count, or curve lengths are invalid.
    """
    orders = finite_numpy_1d(harmonic_orders, "harmonic_orders")
    if np.any(orders <= 0.0) or np.any(orders != np.round(orders)):
        raise ValueError("harmonic_orders must contain positive integers.")
    if blade_count < 1:
        raise ValueError("blade_count must be positive.")
    if not levels_by_case_db:
        raise ValueError("levels_by_case_db must contain at least one curve.")
    if ax is None:
        _, ax = plt.subplots(
            figsize=validated_figsize(figsize), constrained_layout=True
        )
    default_colors = ("black", "tab:blue", "tab:red")
    for index, (label, values) in enumerate(levels_by_case_db.items()):
        levels = finite_numpy_1d(values, f"levels_by_case_db[{label!r}]")
        if levels.size != orders.size:
            raise ValueError(f"Curve {label!r} must match harmonic_orders.")
        ax.plot(
            orders,
            levels,
            "o-",
            color=(colors or {}).get(label, default_colors[index % 3]),
            label=label,
        )
    for order in orders.astype(np.int64):
        if order % blade_count == 0:
            ax.axvline(order, color="tab:orange", alpha=0.16, linewidth=1.0)
    ax.set(
        xlabel="Shaft order",
        ylabel=ylabel,
        title=title,
        xticks=orders,
    )
    ax.legend()
    return ax


def plot_signed_contributions(
    labels: Sequence[str],
    contributions_by_title: Mapping[str, ArrayLike],
    *,
    ylabel: str = "Signed contribution [%]",
    label_rotation_deg: float = 40.0,
    label_fontsize: float = 8.0,
    figsize: FigureSize | None = None,
) -> MultiAxesPlot:
    """Plot one or more signed regional-contribution bar charts.

    Args:
        labels: Region label for every bar.
        contributions_by_title: Mapping from panel title to percentage values.
        ylabel: Shared vertical-axis label.
        label_rotation_deg: Rotation of region labels.
        label_fontsize: Font size for region labels.
        figsize: Overall figure size. The default scales with panel count.

    Returns:
        The created figure and one axes per contribution series.

    Raises:
        ValueError: If no series are supplied or a series length differs from
            the region-label count.
    """
    if not labels:
        raise ValueError("labels must contain at least one region.")
    if not contributions_by_title:
        raise ValueError("contributions_by_title must contain at least one series.")
    series_count = len(contributions_by_title)
    figure_size = figsize or (6.5 * series_count, 4.8)
    figure, axes_array = plt.subplots(
        1,
        series_count,
        figsize=validated_figsize(figure_size),
        squeeze=False,
        constrained_layout=True,
    )
    axes = tuple(axes_array[0])
    positions = np.arange(len(labels))
    for ax, (title, raw_values) in zip(
        axes, contributions_by_title.items(), strict=True
    ):
        values = finite_numpy_1d(raw_values, f"contributions_by_title[{title!r}]")
        if values.size != len(labels):
            raise ValueError(f"Contribution series {title!r} must match labels.")
        ax.bar(
            positions,
            values,
            color=np.where(values >= 0.0, "tab:orange", "tab:purple"),
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(
            xticks=positions,
            xticklabels=labels,
            ylabel=ylabel,
            title=title,
        )
        ax.tick_params(
            axis="x",
            labelrotation=label_rotation_deg,
            labelsize=label_fontsize,
        )
    return MultiAxesPlot(figure=figure, axes=axes)


def plot_trace_spectrum_comparison(
    time_values_s: ArrayLike,
    traces_pa: Mapping[str, ArrayLike],
    spectra: Mapping[str, SpectrumLike],
    *,
    oaspl_db: Mapping[str, float] | None = None,
    shaft_frequency_hz: float | None = None,
    blade_passing_frequency_hz: float | None = None,
    bpf_harmonic_count: int = 4,
    figsize: FigureSize = (13.0, 4.5),
) -> MultiAxesPlot:
    """Compare time signatures and spectra on adjacent axes.

    Args:
        time_values_s: Shared time coordinates for every pressure trace.
        traces_pa: Named time-domain traces.
        spectra: Named spectra, normally corresponding to ``traces_pa``.
        oaspl_db: Optional OASPL values appended to spectrum legend labels.
        shaft_frequency_hz: Optional shaft-frequency marker.
        blade_passing_frequency_hz: Optional BPF harmonic spacing.
        bpf_harmonic_count: Number of BPF harmonics to mark.
        figsize: Overall figure size in inches.

    Returns:
        The created figure with time and spectrum axes.

    Raises:
        ValueError: If mappings are empty, keys differ, or arrays are invalid.
    """
    if not traces_pa or set(traces_pa) != set(spectra):
        raise ValueError("traces_pa and spectra must have the same nonempty keys.")
    times = finite_1d(time_values_s, "time_values_s")
    figure, axes_array = plt.subplots(
        1,
        2,
        figsize=validated_figsize(figsize),
        constrained_layout=True,
    )
    time_ax, spectrum_ax = axes_array
    for label, raw_trace in traces_pa.items():
        trace = finite_1d(raw_trace, f"traces_pa[{label!r}]")
        if trace.numel() != times.numel():
            raise ValueError(f"Trace {label!r} does not match time_values_s.")
        time_ax.plot(times, trace, label=label)
        spectrum = spectra[label]
        frequencies = finite_1d(
            spectrum.frequencies_hz, f"spectra[{label!r}].frequencies_hz"
        )
        levels = finite_1d(spectrum.spl_db, f"spectra[{label!r}].spl_db")
        if frequencies.numel() != levels.numel():
            raise ValueError(f"Spectrum {label!r} has mismatched data lengths.")
        if torch.any(frequencies < 0.0).item():
            raise ValueError(f"Spectrum {label!r} contains negative frequencies.")
        positive = frequencies > 0.0
        if not torch.any(positive).item():
            raise ValueError(f"Spectrum {label!r} has no positive frequencies.")
        legend_label = label
        if oaspl_db is not None and label in oaspl_db:
            legend_label = f"{label} ({oaspl_db[label]:.1f} dB OASPL)"
        spectrum_ax.plot(frequencies[positive], levels[positive], label=legend_label)
    time_ax.set(
        title="Common fully arrived signatures",
        xlabel="Common-window time [s]",
        ylabel="Pressure [Pa]",
    )
    spectrum_ax.set(
        title="Acoustic spectra",
        xlabel="Frequency [Hz]",
        ylabel="SPL [dB re 20 µPa]",
        xscale="log",
    )
    if shaft_frequency_hz is not None:
        shaft_hz = positive_float(shaft_frequency_hz, "shaft_frequency_hz")
        spectrum_ax.axvline(
            shaft_hz,
            color="0.4",
            linestyle="--",
            label="Shaft frequency",
        )
    if blade_passing_frequency_hz is not None:
        bpf_hz = positive_float(
            blade_passing_frequency_hz,
            "blade_passing_frequency_hz",
        )
        harmonic_count = positive_int(bpf_harmonic_count, "bpf_harmonic_count")
        for harmonic in range(1, harmonic_count + 1):
            spectrum_ax.axvline(
                harmonic * bpf_hz,
                color="0.7",
                linestyle=":",
                label="BPF harmonics" if harmonic == 1 else None,
            )
    for ax in (time_ax, spectrum_ax):
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    return MultiAxesPlot(figure=figure, axes=(time_ax, spectrum_ax))


def _finite_numpy(
    values: ArrayLike,
    name: str,
    *,
    ndim: int,
) -> npt.NDArray[np.float64]:
    if isinstance(values, torch.Tensor):
        data = values.detach().cpu().numpy()
    else:
        data = np.asarray(values)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != ndim or data.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain only finite values.")
    return data


def _integer_labels(values: ArrayLike, name: str) -> npt.NDArray[np.int64]:
    labels = finite_numpy_1d(values, name)
    if np.any(labels < 0.0) or np.any(labels != np.round(labels)):
        raise ValueError(f"{name} must contain nonnegative integer labels.")
    return labels.astype(np.int64)


def _require_same_length(*values: npt.NDArray[np.generic]) -> None:
    if len({array.shape[0] for array in values}) != 1:
        raise ValueError("All facewise arrays must have the same length.")


def _color_limits(
    values: npt.NDArray[np.float64],
    *,
    symmetric: bool,
    percentile: float,
) -> tuple[float, float]:
    if symmetric:
        limit = float(np.percentile(np.abs(values), percentile))
        if limit == 0.0:
            limit = float(np.finfo(np.float64).eps)
        return -limit, limit
    vmin = float(np.percentile(values, 100.0 - percentile))
    vmax = float(np.percentile(values, percentile))
    if vmin == vmax:
        vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + float(np.finfo(np.float64).eps)
    return vmin, vmax

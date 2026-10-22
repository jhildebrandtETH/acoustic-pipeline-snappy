"""Pressure-history plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.result import F1AResult
from ._common import (
    ArrayLike,
    FigureSize,
    finite_1d,
    line_properties,
    validated_figsize,
)


@dataclass(frozen=True, slots=True)
class MultiAxesPlot:
    """A figure and its ordered axes, returned by multi-panel helpers."""

    figure: Figure
    axes: tuple[Axes, ...]


def plot_pressure_history(
    result: F1AResult,
    *,
    figsize: FigureSize = (8.0, 4.0),
    annotate_arrivals: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot thickness, loading, and total pressure for one observer.

    Args:
        result: Single-observer solver result.
        figsize: Figure size in inches when axes are created.
        annotate_arrivals: Mark the incomplete-arrival interval, nearest-face
            arrival, and time when all faces are available.
        title: Optional axes title.
        ax: Existing Matplotlib axes, or ``None`` to create one.

    Returns:
        Axes containing the pressure histories.

    Raises:
        ValueError: If ``result`` contains multiple observers.
    """
    if result.total_pressure_pa.ndim != 1:
        raise ValueError(
            "Select one observer with result.for_observer(index) before plotting."
        )
    if ax is None:
        _, ax = plt.subplots(
            figsize=validated_figsize(figsize),
            constrained_layout=True,
        )
    times = result.observer_times_s.detach().cpu()
    series = (
        ("Thickness", result.thickness_pressure_pa),
        ("Loading", result.loading_pressure_pa),
        ("Total", result.total_pressure_pa),
    )
    for index, (label, pressure_pa) in enumerate(series):
        color, linestyle = line_properties(index)
        ax.plot(
            times,
            pressure_pa.detach().cpu(),
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ax.set(xlabel="Observer Time [s]", ylabel="Pressure [Pa]")
    if annotate_arrivals:
        maximum_delay_s = float(torch.as_tensor(result.maximum_delay_s))
        ax.axvspan(
            0.0,
            maximum_delay_s,
            color="0.85",
            alpha=0.5,
            label="Incomplete arrival history",
        )
        if result.response_delay_s is not None:
            ax.axvline(
                result.response_delay_s,
                color="k",
                linestyle=":",
                alpha=0.8,
                label="Nearest-face arrival",
            )
        ax.axvline(
            maximum_delay_s,
            color="m",
            linestyle=":",
            alpha=0.8,
            label="All faces available",
        )
        ax.grid(alpha=0.25)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_pressure_traces(
    time_values: ArrayLike,
    traces: Mapping[str, ArrayLike],
    *,
    time_scale: float = 1.0,
    xlabel: str = "Observer time [s]",
    ylabel: str = "Pressure [Pa]",
    title: str | None = None,
    colors: Mapping[str, str] | None = None,
    linewidths: Mapping[str, float] | None = None,
    figsize: FigureSize = (10.0, 4.5),
    ax: Axes | None = None,
) -> Axes:
    """Plot named pressure traces on a shared time grid.

    Args:
        time_values: One-dimensional time coordinates.
        traces: Mapping from legend labels to pressure values.
        time_scale: Factor applied to time coordinates before plotting.
        xlabel: Horizontal-axis label, including the scaled time unit.
        ylabel: Vertical-axis label.
        title: Optional axes title.
        colors: Optional color override for each trace label.
        linewidths: Optional line-width override for each trace label.
        figsize: Figure size in inches when axes are created.
        ax: Existing axes, or ``None`` to create one.

    Returns:
        Axes containing all traces.

    Raises:
        ValueError: If traces are empty, non-finite, or differ in length.
    """
    times = finite_1d(time_values, "time_values")
    if not traces:
        raise ValueError("traces must contain at least one pressure history.")
    if ax is None:
        _, ax = plt.subplots(
            figsize=validated_figsize(figsize), constrained_layout=True
        )
    for index, (label, values) in enumerate(traces.items()):
        pressure = finite_1d(values, f"traces[{label!r}]")
        if pressure.numel() != times.numel():
            raise ValueError(f"Trace {label!r} does not match the time grid length.")
        default_color, _ = line_properties(index)
        ax.plot(
            times * time_scale,
            pressure,
            color=(colors or {}).get(label, default_color),
            linewidth=(linewidths or {}).get(label, 1.0),
            label=label,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    return ax


def plot_result_pressure_comparison(
    results: Mapping[str, F1AResult],
    *,
    observer_labels: Sequence[str] | None = None,
    colors: Mapping[str, str] | None = None,
    time_scale: float = 1.0,
    time_unit: str = "s",
    ylabel: str = "Acoustic pressure [Pa]",
    shade_incomplete: bool = False,
    figsize: FigureSize | None = None,
) -> MultiAxesPlot:
    """Compare total-pressure histories from several results by observer.

    Args:
        results: Named results with the same observer count.
        observer_labels: Optional title for each observer panel. Coordinates
            from the first result are used when omitted.
        colors: Optional line colors keyed by result label.
        time_scale: Factor applied to observer time before plotting.
        time_unit: Unit shown in the horizontal-axis label.
        ylabel: Shared vertical-axis label.
        shade_incomplete: Shade the interval before every surface face has
            reached an observer, using the worst delay across the results.
        figsize: Overall figure size. When omitted, panel height scales with
            the observer count.

    Returns:
        The created figure and one axes per observer.

    Raises:
        ValueError: If results are empty or their observer counts differ.
    """
    if not results:
        raise ValueError("results must contain at least one solver result.")
    first_result = next(iter(results.values()))
    observer_count = _observer_count(first_result)
    for label, result in results.items():
        if _observer_count(result) != observer_count:
            raise ValueError(f"Result {label!r} has a different observer count.")

    if observer_labels is None:
        observers = first_result.observer_m
        if observers.ndim == 1:
            observers = observers[None, :]
        labels = tuple(
            f"Observer {index}: {position.detach().cpu().tolist()} m"
            for index, position in enumerate(observers)
        )
    else:
        labels = tuple(observer_labels)
        if len(labels) != observer_count:
            raise ValueError("observer_labels must match the observer count.")

    figure_size = figsize or (10.0, max(3.5, 3.5 * observer_count))
    figure, axes_array = plt.subplots(
        observer_count,
        1,
        figsize=validated_figsize(figure_size),
        sharex=observer_count > 1,
        squeeze=False,
        constrained_layout=True,
    )
    axes = tuple(axes_array[:, 0])
    for observer_index, (ax, observer_label) in enumerate(
        zip(axes, labels, strict=True)
    ):
        for result_index, (case_label, result) in enumerate(results.items()):
            selected = result.for_observer(observer_index)
            default_color, _ = line_properties(result_index)
            ax.plot(
                selected.observer_times_s.detach().cpu() * time_scale,
                selected.total_pressure_pa.detach().cpu(),
                color=(colors or {}).get(case_label, default_color),
                linewidth=0.9,
                label=case_label,
            )
        if shade_incomplete:
            maximum_delay_s = max(
                float(
                    torch.as_tensor(result.for_observer(observer_index).maximum_delay_s)
                )
                for result in results.values()
            )
            delay = maximum_delay_s * time_scale
            ax.axvspan(
                0.0,
                delay,
                color="tab:red",
                alpha=0.08,
                label="Incomplete arrival history",
            )
            ax.axvline(
                delay,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
            )
        ax.set(title=observer_label, ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel(f"Observer time [{time_unit}]")
    return MultiAxesPlot(figure=figure, axes=axes)


def _observer_count(result: F1AResult) -> int:
    return (
        1 if result.total_pressure_pa.ndim == 1 else result.total_pressure_pa.shape[0]
    )

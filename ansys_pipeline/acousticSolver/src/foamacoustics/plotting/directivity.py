"""Polar acoustic-directivity plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

from ._common import (
    ArrayLike,
    FigureSize,
    finite_1d,
    line_properties,
    validated_figsize,
)


@dataclass(frozen=True, slots=True)
class DirectivityPlot:
    """A polar directivity figure and its metric axes."""

    figure: Figure
    axes: tuple[PolarAxes, ...]


def plot_oaspl_directivity(
    elevation_deg: ArrayLike,
    oaspl_db: ArrayLike,
    *,
    a_weighted_oaspl_db: ArrayLike | None = None,
    figsize: FigureSize = (6.0, 4.0),
    ax: PolarAxes | None = None,
) -> PolarAxes:
    """Plot overall SPL directivity against observer elevation.

    Args:
        elevation_deg: One-dimensional observer elevations in degrees.
        oaspl_db: Unweighted overall SPL at each elevation.
        a_weighted_oaspl_db: Optional A-weighted overall SPL at each elevation.
        figsize: Figure size in inches when axes are created.
        ax: Existing polar axes, or ``None`` to create one.

    Returns:
        Polar axes containing the directivity curves.

    Raises:
        ValueError: If an input is not finite and one-dimensional, the input
            lengths differ, or ``ax`` is not a polar axes.
    """
    elevations = finite_1d(elevation_deg, "elevation_deg")
    unweighted_levels = finite_1d(oaspl_db, "oaspl_db")
    if elevations.numel() != unweighted_levels.numel():
        raise ValueError("elevation_deg and oaspl_db must have the same length.")

    weighted_levels = None
    if a_weighted_oaspl_db is not None:
        weighted_levels = finite_1d(
            a_weighted_oaspl_db,
            "a_weighted_oaspl_db",
        )
        if elevations.numel() != weighted_levels.numel():
            raise ValueError(
                "elevation_deg and a_weighted_oaspl_db must have the same length."
            )

    if ax is None:
        _, created_ax = plt.subplots(
            figsize=validated_figsize(figsize),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )
        ax = cast("PolarAxes", created_ax)
    elif ax.name != "polar":
        raise ValueError("ax must use the polar projection.")

    angles_rad = torch.deg2rad(elevations)
    color, linestyle = line_properties(0)
    ax.plot(
        angles_rad,
        unweighted_levels,
        color=color,
        linestyle=linestyle,
        marker="o",
        linewidth=1.5,
        label="OSPL",
    )

    all_levels = unweighted_levels
    if weighted_levels is not None:
        color, linestyle = line_properties(1)
        ax.plot(
            angles_rad,
            weighted_levels,
            color=color,
            linestyle=linestyle,
            marker="s",
            linewidth=1.5,
            label="A-weighted OSPL",
        )
        all_levels = torch.cat((unweighted_levels, weighted_levels))

    radial_min_db = 5.0 * torch.floor((all_levels.min() - 5.0) / 5.0)
    radial_max_db = 5.0 * torch.ceil((all_levels.max() + 5.0) / 5.0)
    configure_polar_axis(
        ax,
        (all_levels,),
        (float(radial_min_db), float(radial_max_db)),
    )
    ax.legend(loc="upper right", fontsize=8)
    return ax


def configure_polar_axis(
    ax: PolarAxes,
    all_levels_db: Sequence[ArrayLike],
    radial_limits_db: tuple[float, float],
) -> None:
    """Apply the shared propeller-plane polar-axis presentation.

    Args:
        ax: Matplotlib axes using the polar projection.
        all_levels_db: Level curves shown on the axes. They are validated so
            empty plot inputs fail before the axes are configured.
        radial_limits_db: Lower and upper radial limits in decibels.

    Raises:
        ValueError: If the axes are not polar, no levels are supplied, or the
            radial limits are invalid.
    """
    if ax.name != "polar":
        raise ValueError("ax must use the polar projection.")
    if not all_levels_db:
        raise ValueError("At least one directivity level is required.")
    levels = torch.cat(
        tuple(
            finite_1d(values, f"all_levels_db[{index}]")
            for index, values in enumerate(all_levels_db)
        )
    )
    if levels.numel() == 0:
        raise ValueError("At least one directivity level is required.")
    radial_min, radial_max = radial_limits_db
    if not radial_min < radial_max:
        raise ValueError("radial_limits_db must be strictly increasing.")
    ax.set_thetamin(-90.0)
    ax.set_thetamax(90.0)
    ax.set_rlim(radial_min, radial_max)
    ax.set_rlabel_position(75.0)
    ax.set_title("")
    ax.set_ylabel("Level [dB re 20 µPa]", labelpad=8)
    ax.grid(alpha=0.3)


def plot_directivity_metrics(
    observer_angles_deg: ArrayLike,
    metrics_by_case: Mapping[str, ArrayLike],
    *,
    metric_titles: Sequence[str],
    radial_limits_db: tuple[float, float] = (30.0, 75.0),
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    figure_title: str | None = None,
    figsize: FigureSize | None = None,
) -> DirectivityPlot:
    """Plot several acoustic metrics as side-by-side polar directivities.

    Args:
        observer_angles_deg: Observer elevation angles in degrees.
        metrics_by_case: Mapping of case label to a two-dimensional array with
            shape ``(observers, metrics)``.
        metric_titles: Title for each metric column.
        radial_limits_db: Shared radial range in decibels.
        colors: Optional curve colors keyed by case label.
        markers: Optional marker styles keyed by case label.
        figure_title: Optional figure-level title.
        figsize: Overall figure size. The default scales with metric count.

    Returns:
        The created figure and one polar axes per metric.

    Raises:
        ValueError: If data are empty, non-finite, or their dimensions do not
            agree with the angles and metric titles.
    """
    angles = finite_1d(observer_angles_deg, "observer_angles_deg")
    if not metrics_by_case:
        raise ValueError("metrics_by_case must contain at least one case.")
    metric_count = len(metric_titles)
    if metric_count == 0:
        raise ValueError("metric_titles must contain at least one title.")

    normalized_metrics: dict[str, torch.Tensor] = {}
    for label, values in metrics_by_case.items():
        data = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
        if data.ndim != 2 or data.shape != (angles.numel(), metric_count):
            raise ValueError(
                f"metrics_by_case[{label!r}] must have shape "
                f"({angles.numel()}, {metric_count})."
            )
        if not torch.isfinite(data).all().item():
            raise ValueError(f"metrics_by_case[{label!r}] must be finite.")
        normalized_metrics[label] = data

    figure_size = figsize or (4.5 * metric_count, 4.2)
    figure, axes_array = plt.subplots(
        1,
        metric_count,
        figsize=validated_figsize(figure_size),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
        squeeze=False,
    )
    axes = tuple(cast("PolarAxes", ax) for ax in axes_array[0])
    angles_rad = torch.deg2rad(angles)
    for metric_index, (ax, title) in enumerate(zip(axes, metric_titles, strict=True)):
        levels_for_axis: list[torch.Tensor] = []
        for case_index, (label, metrics) in enumerate(normalized_metrics.items()):
            levels_db = metrics[:, metric_index]
            levels_for_axis.append(levels_db)
            default_color, _ = line_properties(case_index)
            ax.plot(
                angles_rad,
                levels_db,
                color=(colors or {}).get(label, default_color),
                marker=(markers or {}).get(label, "o"),
                linewidth=1.5,
                markersize=4.0,
                label=label,
            )
        configure_polar_axis(ax, levels_for_axis, radial_limits_db)
        ax.set_title(title, pad=14)
    axes[-1].legend(loc="upper right", fontsize=8)
    if figure_title is not None:
        figure.suptitle(figure_title)
    return DirectivityPlot(figure=figure, axes=axes)

"""Reporting helpers for the acoustic pipeline."""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from .mesh_quality import latest_mesh_log
from .plotting import COLORS, pyplot as plt, save_figure

def create_reference_geometry_vtk_series(
    source_directory: Path,
    output_directory: Path,
    surface_file: str = "cubeWall.vtk",
) -> Path:
    time_directories = sorted(
        (
            path
            for path in source_directory.iterdir()
            if path.is_dir()
        ),
        key=lambda path: float(path.name),
    )

    if not time_directories:
        raise FileNotFoundError(
            f"No timestep directories found in {source_directory}"
        )

    reference_path = time_directories[0] / surface_file
    reference_data = reference_path.read_bytes()

    def split_vtk(data: bytes) -> tuple[bytes, bytes, bytes]:
        points_match = re.search(
            rb"(?m)^POINTS\s+\d+\s+\S+\s*$",
            data,
        )
        field_match = re.search(
            rb"(?m)^(?:CELL_DATA|POINT_DATA)\s+\d+\s*$",
            data,
        )

        if points_match is None or field_match is None:
            raise ValueError("Unsupported VTK file structure")

        header = data[:points_match.start()]
        geometry = data[points_match.start():field_match.start()]
        fields = data[field_match.start():]

        return header, geometry, fields

    _, reference_geometry, _ = split_vtk(reference_data)

    for time_directory in time_directories:
        source_path = time_directory / surface_file
        current_data = source_path.read_bytes()

        current_header, _, current_fields = split_vtk(current_data)

        target_directory = output_directory / time_directory.name
        target_directory.mkdir(parents=True, exist_ok=True)

        target_path = target_directory / surface_file
        target_path.write_bytes(
            current_header
            + reference_geometry
            + current_fields
        )

    return output_directory


def create_yplus_distribution_plot(case_path, report_dir, patch_name="cubeWall"):
    """Area-primary saved-snapshot coverage, with explicit face diagnostics."""
    from .yplus_reporting import create_yplus_report
    return create_yplus_report(case_path, report_dir, patch_name)


def read_mesh_element_types(case_path):
    log_checkmesh = case_path / "log.checkMesh"

    element_types = {
        "hexahedra": 0,
        "prisms": 0,
        "wedges": 0,
        "pyramids": 0,
        "tet wedges": 0,
        "tetrahedra": 0,
        "polyhedra": 0,
    }

    if not log_checkmesh.exists():
        return element_types

    text = latest_mesh_log(case_path)

    patterns = {
        "hexahedra": r"hexahedra:\s*([0-9]+)",
        "prisms": r"prisms:\s*([0-9]+)",
        "wedges": r"(?m)^\s*wedges:\s*([0-9]+)",
        "pyramids": r"pyramids:\s*([0-9]+)",
        "tet wedges": r"tet wedges:\s*([0-9]+)",
        "tetrahedra": r"tetrahedra:\s*([0-9]+)",
        "polyhedra": r"polyhedra:\s*([0-9]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            element_types[key] = int(match.group(1))

    return element_types


def create_mesh_element_plot(element_types, report_dir):

    nonzero = {
        key: value
        for key, value in element_types.items()
        if value > 0
    }

    if not nonzero:
        return None

    total = sum(nonzero.values())

    labels = list(nonzero.keys())
    values = [100.0 * value / total for value in nonzero.values()]

    mesh_plot = report_dir / "mesh_element_types.png"

    fig, ax = plt.subplots(figsize=(7.4, 3.5), layout="constrained")
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS[0])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set(xlabel="Cell share [%]", title="Mesh element composition", xlim=(0, 117))
    ax.grid(axis="x")
    save_figure(fig, mesh_plot)
    plt.close(fig)

    return mesh_plot


def read_mesh_information(case_path):
    log_checkmesh = case_path / "log.checkMesh"

    mesh_info = {
        "mesh_ok": False,
        "cells": None,
        "faces": None,
        "points": None,
        "boundary_patches": None,
        "max_aspect_ratio": None,
        "max_skewness": None,
        "max_non_orthogonality": None,
    }

    if not log_checkmesh.exists():
        mesh_info["status"] = "log.checkMesh not found"
        return mesh_info

    text = latest_mesh_log(case_path)

    endings = list(re.finditer(r"Mesh OK\.?|Failed\s+\d+\s+mesh checks?", text))
    mesh_info["mesh_ok"] = bool(endings and endings[-1][0].startswith("Mesh OK"))
    mesh_info["status"] = "Mesh OK" if mesh_info["mesh_ok"] else "Mesh check failed / not confirmed"

    patterns = {
        "points": r"points:\s*([0-9]+)",
        "faces": r"faces:\s*([0-9]+)",
        "cells": r"cells:\s*([0-9]+)",
        "boundary_patches": r"boundary patches:\s*([0-9]+)",
        "max_aspect_ratio": r"Max aspect ratio\s*=\s*([0-9.eE+-]+)",
        "max_skewness": r"Max skewness\s*=\s*([0-9.eE+-]+)",
        "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
        "mean_non_orthogonality": r"Mesh non-orthogonality Max:[^\n]*average:\s*([0-9.eE+-]+)",
        "min_volume": r"Min volume\s*=\s*([0-9.eE+-]+)",
        "min_determinant": r"Cell determinant[^\n]*minimum:\s*([0-9.eE+-]+)",
    }

    for key, pattern in patterns.items():
        pattern = pattern.replace("[0-9.eE+-]+", r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            try:
                mesh_info[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                mesh_info[key] = value

    return mesh_info


def format_seconds(seconds):
    if seconds is None:
        return "Not found"

    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def format_optional_number(value, format_spec=".6e"):
    if value is None:
        return "Not found"

    try:
        return format(float(value), format_spec)
    except (TypeError, ValueError):
        return str(value)


def format_optional_bool(value):
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "Not found"


def evaluate_thrust_convergence(times, thrusts, rev_time, threshold=0.005):
    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time
    idx_start = np.searchsorted(times, last_rev_start, side="left")

    window_times = times[idx_start:]
    window_thrusts = thrusts[idx_start:]

    if len(window_thrusts) == 0:
        return {
            "passed": False,
            "reason": "No thrust samples found in final revolution window.",
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "mean_N": None,
            "std_N": None,
            "relative_std": None,
            "threshold": threshold,
            "n_samples": 0,
        }

    mean_thrust = float(np.mean(window_thrusts))
    std_thrust = float(np.std(window_thrusts, ddof=0))
    relative_std = std_thrust / max(abs(mean_thrust), 1e-12)

    return {
        "passed": bool(relative_std < threshold),
        "reason": None,
        "window_start_s": float(window_times[0]),
        "window_end_s": latest_time,
        "mean_N": mean_thrust,
        "std_N": std_thrust,
        "relative_std": float(relative_std),
        "threshold": float(threshold),
        "n_samples": int(len(window_thrusts)),
    }


def evaluate_moments(times, moments, rev_time):
    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time
    idx_start = np.searchsorted(times, last_rev_start, side="left")

    window_times = times[idx_start:]
    window_moments = moments[idx_start:]

    if len(window_moments) == 0:
        return {
            "passed": False,
            "reason": "No moments samples found in final revolution window.",
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "mean_N": None,
            "std_N": None,
            "relative_std": None
        }

    mean_moment = float(np.mean(window_moments))
    std_moment = float(np.std(window_moments, ddof=0))
    relative_std = std_moment / max(abs(mean_moment), 1e-12)

    return {
        "passed": bool,
        "reason": None,
        "window_start_s": float(window_times[0]),
        "window_end_s": latest_time,
        "mean_N": mean_moment,
        "std_N": std_moment,
        "relative_std": float(relative_std)
    }


def compute_thrust_stability_history(times, thrusts, rev_time):
    """
    Computes the relative thrust fluctuation over a sliding one-revolution window.

    metric(t) = std(F_window) / |mean(F_window)|

    The value at time t uses all force samples within [t - T_rev, t].
    """
    metric = np.full(len(times), np.nan, dtype=float)
    window_mean = np.full(len(times), np.nan, dtype=float)
    window_std = np.full(len(times), np.nan, dtype=float)
    sample_count = np.zeros(len(times), dtype=int)

    for i, time_value in enumerate(times):
        window_start = time_value - rev_time

        if window_start < times[0]:
            continue

        j = np.searchsorted(times, window_start, side="left")
        window_values = thrusts[j : i + 1]

        if len(window_values) < 2:
            continue

        mean_value = float(np.mean(window_values))
        std_value = float(np.std(window_values, ddof=0))

        window_mean[i] = mean_value
        window_std[i] = std_value
        metric[i] = std_value / max(abs(mean_value), 1e-12)
        sample_count[i] = int(len(window_values))

    return {
        "time": times,
        "relative_std": metric,
        "mean_N": window_mean,
        "std_N": window_std,
        "sample_count": sample_count,
    }


def clip_load_axis(ax, values, reference):
    """Limit the display when a transient would obscure the load history."""
    if not len(reference):
        return None
    if len(reference) >= 20:
        lo, hi = np.percentile(reference, [1, 99])
    else:
        lo, hi = reference.min(), reference.max()
    span = max(float(hi - lo), abs(float(lo)) * .002, 1e-9)
    if np.ptp(values) <= 5 * span:
        return None
    limits = (lo - .15 * span, hi + .15 * span)
    ax.set_ylim(*limits)
    outside = int(np.sum((values < limits[0]) | (values > limits[1])))
    ax.text(.98, .96, f"{outside} samples outside y-range", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    return limits


def plot_load_history(times, values, rev_time, path, title, ylabel):
    """Display the full time span and zoom into the final revolution."""
    end = float(times[-1])
    start = max(float(times[0]), end - rev_time)
    final = times >= start
    fig, (history, detail) = plt.subplots(
        1, 2, figsize=(9, 3.8), width_ratios=(1.45, 1), layout="constrained")
    try:
        history.plot(times, values, color=COLORS[0])
        history.axvspan(start, end, color=COLORS[0], alpha=.13, lw=0)
        history.set(xlabel="Physical time [s]", ylabel=ylabel, title="(a) Full history")
        # Keep the startup samples in the series, but set display limits from
        # later samples so a startup spike cannot flatten the rest of the run.
        reference = values[times > float(times[0]) + .001]
        clip_load_axis(history, values, reference if len(reference) else values[1:])
        phase = (times[final] - (end - rev_time)) * (360 / rev_time)
        window = values[final]
        detail.plot(phase, window, color=COLORS[0], marker="." if len(window) < 2 else None)
        mean = float(np.mean(window))
        limits = clip_load_axis(detail, window, window) if len(window) >= 20 else None
        if limits is None or limits[0] <= mean <= limits[1]:
            detail.axhline(mean, color=COLORS[1], ls="--", lw=1, label="Sample mean")
            detail.legend(loc="best")
        label = "(b) Final revolution" if end - float(times[0]) >= .99 * rev_time else "(b) Partial revolution"
        detail.set(xlabel="Angle into final revolution [deg]", ylabel=ylabel,
                   title=label, xlim=(0, 360), xticks=[0, 90, 180, 270, 360])
        for ax in (history, detail):
            ax.grid(axis="y")
        fig.suptitle(title, weight="bold")
        save_figure(fig, path)
    finally:
        plt.close(fig)


def create_force_plots(times, thrusts, report_dir, rev_time, thrust_convergence=None):
    """Plot thrust and variability; accept the legacy summary argument."""
    force_plot = report_dir / "force_plot.png"
    conv_plot = report_dir / "force_convergence.png"
    plot_load_history(times, thrusts, rev_time, force_plot, "Axial thrust", r"Thrust, $F_y$ [N]")

    stability_history = compute_thrust_stability_history(times, thrusts, rev_time)
    metric_time = stability_history["time"]
    metric = stability_history["relative_std"]
    valid = np.isfinite(metric)
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    try:
        if np.any(valid):
            ax.plot(metric_time[valid], metric[valid])
        ax.axvspan(max(float(times[0]), float(times[-1]) - rev_time), float(times[-1]), color=COLORS[0], alpha=.13, lw=0)
        ax.set(xlabel="Physical time [s]", ylabel=r"$\sigma_F / |\overline{F}|$ [-]",
               title="Thrust variability over one revolution")
        if np.any(metric[valid] > 0):
            # The linear region keeps genuinely zero-variance windows visible.
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(axis="y")
        save_figure(fig, conv_plot)
    finally:
        plt.close(fig)
    return force_plot, conv_plot, stability_history


def create_moments_plots(times, moments, report_dir, rev_time):
    moments_plot = report_dir / "moments_plot.png"
    plot_load_history(times, moments, rev_time, moments_plot, "Shaft torque", r"Torque, $M_y$ [N m]")
    return moments_plot


def read_residual_dataframe(residual_file):
    if not residual_file.exists():
        return None

    with open(residual_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    header = lines[1].lstrip("#").split()

    df = pd.read_csv(
        residual_file,
        sep=r"\s+",
        names=header,
        skiprows=2,
        engine="python",
    )

    if "Time" not in df.columns:
        return None

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time")

    return df


def evaluate_residual_slopes(df, rev_time, latest_time):
    from tools.openfoam import slope_bounds
    if df is None or len(df) == 0:
        return None

    last_rev_start = latest_time - rev_time
    window = df[(df["Time"] >= last_rev_start) & (df["Time"] <= latest_time)].copy()
    passed_fields = dict.fromkeys(slope_bounds, False)

    if len(window) < 2:
        return {
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "n_samples": int(len(window)),
            "slopes_per_rev": {},
            "passed_fields": passed_fields,
            "end_residuals": {},
            "mean_residuals": {},
            "reason": "Not enough residual samples in final revolution window.",
        }

    # Independent variable in revolutions relative to the start of the final window.
    x_rev = (window["Time"].to_numpy(dtype=float) - last_rev_start) / rev_time

    slopes_per_rev = {}
    end_residuals = {}
    mean_residuals = {}

    for col in window.columns:
        if col == "Time":
            continue

        values = window[col].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0.0) & np.isfinite(x_rev)

        if np.count_nonzero(valid) < 2:
            slopes_per_rev[col] = None
            end_residuals[col] = None
            mean_residuals[col] = None
            continue

        y_log = np.log10(values[valid])
        x_valid = x_rev[valid]

        # Slope of log10(residual) per propeller revolution.
        slope, _intercept = np.polyfit(x_valid, y_log, 1)

        slopes_per_rev[col] = float(slope)
        if col in slope_bounds:
            lower_bound, upper_bound = slope_bounds[col]
            passed_fields[col] = bool(
                latest_time > rev_time
                and np.count_nonzero(valid) >= 10
                and lower_bound <= slope <= upper_bound
            )
        end_residuals[col] = float(values[valid][-1])
        mean_residuals[col] = float(np.mean(values[valid]))

    return {
        "window_start_s": float(window["Time"].iloc[0]),
        "window_end_s": float(window["Time"].iloc[-1]),
        "n_samples": int(len(window)),
        "slopes_per_rev": slopes_per_rev,
        "passed_fields": passed_fields,
        "end_residuals": end_residuals,
        "mean_residuals": mean_residuals,
        "reason": None,
    }


def create_residual_plots(residual_file, report_dir, rev_time, latest_time):

    residual_plot = report_dir / "residuals.png"

    df = read_residual_dataframe(residual_file)

    if df is None:
        return None, None

    last_rev_start = latest_time - rev_time

    fig, ax = plt.subplots(figsize=(9, 4.1), layout="constrained")
    try:
        styles = ("-", "--", "-.", ":")
        for index, col in enumerate(df.columns.drop("Time")):
            values = df[col].where(df[col] > 0)
            ax.plot(df["Time"], values, label=col, ls=styles[index % len(styles)])
        ax.axvspan(last_rev_start, latest_time, color=COLORS[0], alpha=.13, lw=0)
        ax.set(yscale="log", xlabel="Physical time [s]", ylabel="Solver residual [-]",
               title="Residual histories")
        ax.grid(axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=3)
        save_figure(fig, residual_plot)
    finally:
        plt.close(fig)

    residual_slope_info = evaluate_residual_slopes(df, rev_time, latest_time)

    return residual_plot, residual_slope_info


def draw_courant_summary(c, title, summary, y_position):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, title)
    y_position -= 20

    c.setFont("Helvetica", 10)

    if summary is None:
        c.drawString(50, y_position, "No matching entries found in solver log.")
        return y_position - 26

    c.drawString(
        50,
        y_position,
        f"Samples: {summary['samples']} | "
        "Logged mean Co, average / maximum: "
        f"{format_optional_number(summary['mean_co_average'], '.4g')} / "
        f"{format_optional_number(summary['mean_co_max'], '.4g')}",
    )
    y_position -= 18

    c.drawString(
        50,
        y_position,
        "Logged maximum Co, average / peak: "
        f"{format_optional_number(summary['max_co_average'], '.4g')} / "
        f"{format_optional_number(summary['peak_max_co'], '.4g')}",
    )
    y_position -= 18

    exceedance_count = summary[
        "configured_max_co_exceedance_count"
    ]
    exceedance_percent = summary[
        "configured_max_co_exceedance_percent"
    ]

    if exceedance_count is not None:
        c.drawString(
            50,
            y_position,
            "Samples above configured maxCo: "
            f"{exceedance_count} "
            f"({format_optional_number(exceedance_percent, '.2f')}%)",
        )
        y_position -= 18

    return y_position - 18

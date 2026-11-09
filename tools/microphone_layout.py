"""Microphone geometry for acoustic reports, independent of planar directivity."""
import csv
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def _read_coordinates(report, filename, keys, rpm):
    with np.load(report / filename, allow_pickle=False) as archive:
        if "rpm" in archive and not np.isclose(float(archive["rpm"]), rpm):
            raise ValueError(f"{filename} is for a different RPM")
        return {key: archive[key] for key in keys if key in archive}


def microphone_geometry(case, rpm):
    """Return recorded channels in simulation coordinates and the actual spectrum observer."""
    from tools.acoustic_comparison import load_config
    from tools.acoustic_directivity import frame, recorded_microphones, settings, transform_position, vector

    case = Path(case)
    report = case / "report"
    errors = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, EOFError, BadZipFile)
    try:
        config = load_config(case)
    except errors:
        config = {}
    try:
        comparison = json.loads((report / "acoustic-comparison.json").read_text(encoding="utf-8"))
        if comparison["status"] != "complete" or not np.isclose(float(comparison["rpm"]), rpm):
            comparison = {}
    except errors:
        comparison = {}

    channels, positions = [], np.empty((0, 3))
    center = np.zeros(3)
    source = None
    try:
        archive = _read_coordinates(report, "acoustic-observers.npz",
                                    ("channel_ids", "observer_m", "coordinate_frame"), rpm)
        ids = archive["channel_ids"].astype(str)
        points = np.asarray(archive["observer_m"], dtype=float)
        if (ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids)
                or points.shape != (len(ids), 3) or not np.isfinite(points).all()):
            raise ValueError("Invalid observer coordinates or channel IDs")
        if "coordinate_frame" in archive:
            center = vector(json.loads(str(archive["coordinate_frame"]))[1], "rotation center")
        channels, positions, source = ids.tolist(), points, "Saved FW-H observer archive"
    except errors:
        if config.get("measurements_directory"):
            try:
                recorded = recorded_microphones(case, rpm, config)
                ids = sorted(recorded, key=lambda channel: (not channel.isdigit(),
                             int(channel) if channel.isdigit() else channel))
                points = np.array([transform_position(recorded[channel], channel, config) for channel in ids])
                if settings(config).get("enabled", False):
                    center = frame(config)[1]
                channels, positions, source = ids, points, "Matched measurement metadata"
            except errors:
                pass

    observer = None
    try:
        archive = _read_coordinates(report, "acoustic-pressure.npz", ("observer_m",), rpm)
        observer = vector(archive["observer_m"], "spectrum observer")
    except errors:
        if comparison:
            try:
                observer = vector(comparison["observer_m"], "spectrum observer")
            except errors:
                pass

    if not channels and comparison:
        try:
            # Older cases may only have the selected channel in their comparison.
            channel = str(comparison["microphone_channel"])
            saved_config = {"directivity": comparison.get("coordinate_mapping") or {}}
            # Comparison metadata already stores this point in simulation axes.
            point = vector(comparison["microphone_position_m"], "measurement position")
            if settings(saved_config).get("enabled", False):
                center = frame(saved_config)[1]
            channels, positions = [channel], np.array([point])
            source = "Saved spectrum comparison (single channel)"
        except errors:
            pass

    selected = None
    if observer is not None:
        matches = [channel for channel, point in zip(channels, positions)
                   if np.allclose(point, observer, atol=1e-6, rtol=0)]
        preferred = str(comparison.get("microphone_channel", config.get("microphone_channel")))
        if preferred in matches:
            selected = preferred
        elif len(matches) == 1:
            selected = matches[0]
    return dict(channels=channels, positions=positions, center=center,
                observer=observer, selected=selected, source=source)


def append_microphone_layout(canvas, case, rpm):
    """Write a standalone figure and embed it in the report when coordinates exist."""
    from matplotlib.ticker import MaxNLocator
    from tools.plotting import COLORS, pyplot as plt, save_figure

    report = Path(case) / "report"
    path = report / "microphone-layout.png"
    table = path.with_suffix(".csv")
    layout = microphone_geometry(case, rpm)
    ids, points = layout["channels"], layout["positions"]
    center, observer, selected = layout["center"], layout["observer"], layout["selected"]
    if not ids and observer is None:
        for stale in (path, path.with_suffix(".pdf"), table):
            stale.unlink(missing_ok=True)
        return

    extent = np.vstack([center, points, *([] if observer is None else [observer])])
    mid = (extent.max(axis=0) + extent.min(axis=0)) / 2
    half = max(.1, .63 * np.ptp(extent, axis=0).max())
    shaft_half = .14 * half
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), subplot_kw={"projection": "3d"})
    try:
        fig.subplots_adjust(left=.03, right=.97, bottom=.16, top=.83, wspace=.04)
        fig.suptitle("Microphone positions around the rotor", fontsize=17, weight="bold", x=.06, ha="left")
        for ax, angles, title in zip(axes, [(25, -55), (24, 46)], ["(a) Oblique", "(b) Alternate azimuth"]):
            ax.view_init(*angles)
            ax.set_proj_type("ortho")
            ax.set_box_aspect((1, 1, 1), zoom=.92)
            ax.set(xlim=(mid[0] - half, mid[0] + half), ylim=(mid[1] - half, mid[1] + half),
                   zlim=(mid[2] - half, mid[2] + half),
                   xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=title)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
                axis.pane.set_facecolor("#F3F6F8")
            ax.tick_params(labelsize=8, pad=1)
            ax.plot([center[0]] * 2, center[1] + np.array([-shaft_half, shaft_half]), [center[2]] * 2,
                    color=COLORS[3], lw=2, label="Rotor shaft (y)")
            ax.scatter(*center, color=COLORS[5], marker="x", s=36, label="Rotor center", depthshade=False)
            for index, (channel, point) in enumerate(zip(ids, points)):
                highlight = channel == selected
                ax.scatter(*point, color=COLORS[1] if highlight else COLORS[0],
                           marker="D" if highlight else "o", s=52 if highlight else 32,
                           edgecolor="white", linewidth=.5, depthshade=False,
                           label="Spectrum microphone" if highlight else "Microphone")
                ax.text(point[0], point[1], point[2] + (.05 if index % 2 else .025) * half,
                        f"Ch {channel}", fontsize=8, color="#182735", ha="center")
            if observer is not None and selected is None:
                ax.scatter(*observer, color=COLORS[1], marker="D", s=54, depthshade=False,
                           label="Saved FW-H spectrum observer")
                ax.text(observer[0], observer[1], observer[2] + .05 * half, "FW-H", fontsize=8, ha="center")
        handles, labels = axes[0].get_legend_handles_labels()
        legend = dict(zip(labels, handles))
        fig.legend(legend.values(), legend.keys(), loc="lower center", bbox_to_anchor=(.5, .065), ncol=4)
        fig.text(.06, .028, (layout["source"] or "Saved spectrum observer (single point)")
                 + " | Simulation coordinates, equal axis scale", fontsize=9)
        save_figure(fig, path)
    finally:
        plt.close(fig)

    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "x_m", "y_m", "z_m", "used_for_spectrum"])
        for channel, point in zip(ids, points):
            writer.writerow([channel, *point, channel == selected])
        if observer is not None and selected is None:
            writer.writerow(["FW-H observer", *observer, True])

    width, height = canvas._pagesize
    if canvas._code:
        canvas.showPage()
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(40, height - 48, "Microphone geometry")
    canvas.drawImage(str(path), 25, height - 420, width=width - 50, height=340,
                     preserveAspectRatio=True, anchor="c")
    canvas.setFont("Helvetica", 10)
    count = len(ids)
    caption = (f"Recorded microphone channels shown: {count}. Layout is independent of planar directivity."
               if count else "This case retains the spectrum observer coordinates only.")
    canvas.drawString(40, height - 433, caption)
    canvas.drawString(40, height - 449, "Coordinates in metres: report/microphone-layout.csv.")

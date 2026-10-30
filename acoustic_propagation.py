import json
import math
import shutil

from tools.acoustic_time import case_sample_interval, source_time_statistics, uniform_result

from tools.acoustic_comparison import cache_result, comparison_or_status, load_config, resolve_observer_position
from tools.acoustic_directivity import settings as directivity_settings, frame, resolve_observers, cache_observers

from tools.plotting import pyplot as plt
import torch

from acousticSolver.src.foamacoustics import F1ASolver
from acousticSolver.src.foamacoustics.plotting import plot_spl_spectrum
from tools import MATPLOTLIB_LOCK
from tools import create_reference_geometry_vtk_series
from tools import emit_status


# One analysis-window setting for both comparison and simulation-only spectra.
ACOUSTIC_ANALYSIS_ROTATIONS = 15.0


def run_acoustic_solver(
    ACOUSTIC_SURFACE,
    SIMULATION_WORKING_DIRECTORY,
    RPM,
    STATUS_CALLBACK=None,
):
    comparison_status = SIMULATION_WORKING_DIRECTORY / "report/acoustic-comparison.json"
    comparison_status.parent.mkdir(parents=True, exist_ok=True)
    (comparison_status.parent / "acoustic-resampling.json").write_text(json.dumps(
        {"status": "unavailable", "detail": "No uniform pressure grid from the current acoustic run yet."}))
    (comparison_status.parent / "acoustic-pressure.npz").unlink(missing_ok=True)
    (comparison_status.parent / "acoustic-observers.npz").unlink(missing_ok=True)
    (comparison_status.parent / "acoustic-directivity.json").write_text(json.dumps(
        {"status": "unavailable", "detail": "No directivity from the current acoustic run yet."}))
    comparison_status.write_text(json.dumps({"status": "unavailable", "detail": "No comparison from the current acoustic run yet."}))
    try:
        comparison_config = load_config(SIMULATION_WORKING_DIRECTORY)
        multiple = directivity_settings(comparison_config).get('enabled', False)
        if multiple:
            channels, raw_positions, observer_coordinates = resolve_observers(SIMULATION_WORKING_DIRECTORY, RPM, comparison_config)
        else:
            observer_coordinates = resolve_observer_position(SIMULATION_WORKING_DIRECTORY, RPM, comparison_config)
    except Exception as exc:
        comparison_status.write_text(json.dumps({"status": "unavailable", "detail": str(exc)}, indent=2))
        raise
    print(f"Acoustic observers ({len(channels) if multiple else 1}); spectrum channel {comparison_config.get('microphone_channel')}: {observer_coordinates} m", flush=True)
    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="preparing acoustic surface data",
        progress=0.0,
    )

    if ACOUSTIC_SURFACE == "impermeable":
        surface_directory = (
            SIMULATION_WORKING_DIRECTORY / "postProcessing" / "writePatchFields"
        )
        acoustic_surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePatchFields_referenceGeometry"
        )
        surface_file = "propeller.vtk"
        permeable = False
    else:
        surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePermeableSurfaceFields"
        )
        acoustic_surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePermeableSurfaceFields_referenceGeometry"
        )
        surface_file = "permeableSurface.vtk"
        permeable = True

    sample_times = sorted(
        float(directory.name)
        for directory in surface_directory.iterdir()
        if directory.is_dir() and (directory / surface_file).is_file()
    )
    if not sample_times:
        raise ValueError(f"No acoustic surface samples found in {surface_directory}")
    minimum_duration = 5.0 * 60.0 / RPM
    sampled_duration = sample_times[-1] - sample_times[0]
    status_path = SIMULATION_WORKING_DIRECTORY / "report/acoustic-status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if len(sample_times) < 3 or sampled_duration < minimum_duration:
        detail = (
            f"Acoustic spectrum unavailable: sampled {sampled_duration:g} s; "
            f"five rotations require at least {minimum_duration:g} s, "
            "plus propagation delay. Extend the flow run for a spectrum."
        )
        status_path.write_text(
            json.dumps(
                {
                    "status": "insufficient_duration",
                    "sample_count": len(sample_times),
                    "sampled_duration_s": sampled_duration,
                    "minimum_source_duration_s": minimum_duration,
                    "detail": detail,
                },
                indent=2,
            )
            + "\n"
        )
        print(detail, flush=True)
        emit_status(STATUS_CALLBACK, stage="acoustics", detail=detail, progress=100.0)
        return None

    sample_interval = case_sample_interval(SIMULATION_WORKING_DIRECTORY)
    resampling_path = SIMULATION_WORKING_DIRECTORY / "report/acoustic-resampling.json"
    try:
        source_stats = source_time_statistics(sample_times, sample_interval)
    except ValueError as exc:
        failed = {"status": "invalid_sampling", "detail": str(exc)}
        resampling_path.write_text(json.dumps(failed, indent=2))
        status_path.write_text(json.dumps(failed, indent=2))
        raise
    resampling = dict(status="source_validated", method="linear", target_interval_s=sample_interval,
                      target_sampling_frequency_hz=1 / sample_interval,
                      source_time_handling="FW-H uses actual adaptive timestamps",
                      **source_stats)
    resampling_path.write_text(json.dumps(resampling, indent=2))

    create_reference_geometry_vtk_series(
        surface_directory,
        acoustic_surface_directory,
        surface_file=surface_file,
    )

    rpm = RPM
    omega_rad_s = rpm * 2.0 * math.pi / 60.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail=f"loading FW-H data on {device}",
        progress=25.0,
    )

    cache_name = (
        "writePermeableSurfaceFields_referenceGeometry"
        if permeable
        else "writePatchFields_referenceGeometry"
    )

    solver = F1ASolver.from_openfoam_vtk(
        acoustic_surface_directory,
        surface_file=surface_file,
        rpm=rpm,
        permeable=permeable,
        # The sampled permeable sphere is fixed in the inertial frame.
        moving_surface=not permeable,
        rotation_center_m=frame(comparison_config)[1].tolist() if multiple else [0.0, 0.0, 0.0],
        omega_rad_s=[0.0, omega_rad_s if not permeable else 0.0, 0.0],
        device=device,
        cache_dir=SIMULATION_WORKING_DIRECTORY / ".cache" / cache_name,
    )

    observer_m = torch.tensor(observer_coordinates, device=device)

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail=f"predicting pressure at {len(channels) if multiple else 1} observer(s)",
        progress=55.0,
    )
    result = solver.predict(observer_m)
    result = uniform_result(result, sample_interval)
    resampling.update(status="complete", observer_uniform_samples=len(result.observer_times_s))
    resampling_path.write_text(json.dumps(resampling, indent=2))
    multi_result = result if multiple else None
    if multiple:
        result = multi_result.for_observer(channels.index(str(comparison_config['microphone_channel'])))

    output_path = SIMULATION_WORKING_DIRECTORY / "report" / "spl_spectrum.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib uses global process state. The CFD cases may run concurrently,
    # but plotting is serialized to prevent figures from different worker
    # threads interfering with each other.
    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="creating SPL spectrum",
        progress=85.0,
    )
    try:
        if multiple:
            cache_observers(multi_result, SIMULATION_WORKING_DIRECTORY, RPM, comparison_config,
                            channels, raw_positions, rotations=ACOUSTIC_ANALYSIS_ROTATIONS, sample_interval_s=sample_interval)
        else:
            cache_result(result, SIMULATION_WORKING_DIRECTORY, RPM, rotations=ACOUSTIC_ANALYSIS_ROTATIONS, sample_interval_s=sample_interval)
        comparison = comparison_or_status(SIMULATION_WORKING_DIRECTORY, comparison_config)
        if comparison['status'] != 'complete':
            from tools.acoustic_tonal import write_cached_simulation_tonal
            write_cached_simulation_tonal(SIMULATION_WORKING_DIRECTORY, comparison_config)
    except Exception as exc:
        comparison = {"status": "unavailable", "detail": str(exc)}
        comparison_status.write_text(json.dumps(comparison, indent=2))
    if comparison["status"] == "complete":
        shutil.copyfile(output_path.parent / "acoustic-simulated.png", output_path)
    else:
        print(f"Acoustic comparison unavailable: {comparison['detail']}", flush=True)
        with MATPLOTLIB_LOCK:
            available_rotations = (float(result.observer_times_s[-1]) - float(result.maximum_delay_s)) * RPM / 60
            ax = plot_spl_spectrum(
                result, rotations=min(ACOUSTIC_ANALYSIS_ROTATIONS, available_rotations * 0.999),
                blade_count=comparison_config.get("blade_count", 2), xlim_hz=(None, None),
            )
            fig = ax.axes[0].figure if hasattr(ax, "axes") and isinstance(ax.axes, tuple) else ax.figure
            from tools.acoustic_annotations import add_bpf_markers, band_total_db
            for index, axis in enumerate(ax.axes):
                # Replace the library's unlabelled harmonic lines with labelled references.
                for line in list(axis.lines)[1:]:
                    line.remove()
                frequencies = ax.frequencies_hz.detach().cpu().numpy()
                levels = ax.spl_db[index].detach().cpu().numpy()
                from tools.acoustic_annotations import minimum_frequency_hz
                low = max(float(frequencies[frequencies > 0][0]), minimum_frequency_hz(comparison_config, RPM))
                high = min(float(frequencies[-1]), comparison_config['maximum_frequency_hz'])
                if low >= high:
                    raise ValueError('No acoustic frequency coverage above the configured BPF lower limit')
                axis.set_xlim(left=low, right=high)
                add_bpf_markers(axis, RPM, comparison_config.get("blade_count", 2))
                positive = (frequencies > 0) & (frequencies >= low) & (frequencies <= high)
                total = band_total_db(levels[positive])
                axis.set_title(f"{RPM:g} RPM | band total {total:.2f} dB ({frequencies[positive][0]:.1f}-{frequencies[positive][-1]:.1f} Hz)\n"
                               f"Unweighted | FFT spacing {ax.frequency_resolution_hz:.2f} Hz", fontsize=10)
                axis.legend(loc="lower left", fontsize=8)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_count": len(sample_times),
                "sampled_duration_s": sampled_duration,
                "spectrum": str(output_path),
                "observer_count": len(channels) if multiple else 1,
                "spectrum_channel": str(comparison_config['microphone_channel']),
            },
            indent=2,
        )
        + "\n"
    )

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="acoustic spectrum saved",
        progress=100.0,
    )

    return None

"""Compare blade-surface and cylinder-surface acoustic directivity.

Both predictions are reduced with the same settings: the final nine rotor
rotations, 360 samples per rotation, no spectral window, and a 5 kHz OSPL
ceiling.  Observer elevations -80 and -90 degrees are omitted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from foamacoustics import (
    F1AResult,
    F1ASolver,
    OpenFOAMSurfaceSeries,
    PressureSpectrum,
)
from foamacoustics import pressure_to_spl_spectrum
from Scripts.plot_propeller_plane_directivity import (
    BLADE_COUNT,
    OBSERVER_RADIUS_M,
    RPM,
    SAMPLES_PER_ROTATION,
    SPECTRUM_ROTATIONS,
    plot_directivity,
    resample_pressure,
    spectrum_metrics,
)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blade-npz",
        type=Path,
        default=repo_dir.parent / "p_surface_10x7E_7000RPM_AMI.npz",
    )
    parser.add_argument(
        "--cylinder-vtk-dir", type=Path, default=repo_dir.parent / "cylinder15r"
    )
    parser.add_argument(
        "--cylinder-cache-dir",
        type=Path,
        default=repo_dir / ".cache" / "cylinder15r_float64",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_dir
            / "propeller_plane_spl_comparison_figures"
            / "directivity_blade_vs_cylinder.png"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def solve_case(label: str, solver: F1ASolver, observers_m: torch.Tensor) -> F1AResult:
    progress = tqdm(total=len(solver.series), desc=label, unit="frame")

    def report_progress(done: int, total: int, message: str) -> None:
        del total, message
        progress.update(done - progress.n)

    try:
        return solver.predict(
            observers_m,
            workers=10,
            invalid_time_policy="interpolate",
            progress_callback=report_progress,
            progress_every=25,
            remove_mean=False,
            algorithm="auto",
        )
    finally:
        progress.close()


def spectra_for_result(
    result: F1AResult, sampling_rate_hz: float, sample_count: int
) -> list[PressureSpectrum]:
    end_s = float(result.observer_times_s[-1])
    start_s = end_s - SPECTRUM_ROTATIONS / (RPM / 60.0)
    maximum_delay_s = float(torch.as_tensor(result.maximum_delay_s).max())
    if start_s < maximum_delay_s:
        raise RuntimeError(
            "The fully arrived pressure history is too short for nine rotations."
        )
    uniform_time_s = (
        start_s + torch.arange(sample_count, dtype=torch.float64) / sampling_rate_hz
    )
    return [
        pressure_to_spl_spectrum(
            pressure_pa,
            sampling_rate_hz=sampling_rate_hz,
            remove_mean=True,
        )
        for pressure_pa in resample_pressure(result, uniform_time_s)
    ]


def main() -> None:
    args = parse_args()
    if not args.blade_npz.is_file():
        raise FileNotFoundError(args.blade_npz)
    if not args.cylinder_vtk_dir.is_dir():
        raise FileNotFoundError(args.cylinder_vtk_dir)

    device = torch.device(args.device)
    rotation_frequency_hz = RPM / 60.0
    observer_angles_deg = torch.arange(90.0, -71.0, -10.0, dtype=torch.float64)
    observer_angles_rad = torch.deg2rad(observer_angles_deg).to(device)
    observers_m = OBSERVER_RADIUS_M * torch.stack(
        (
            torch.cos(observer_angles_rad),
            torch.sin(observer_angles_rad),
            torch.zeros_like(observer_angles_rad),
        ),
        dim=1,
    )

    blade_solver = F1ASolver.from_npz(
        args.blade_npz,
        pressure_field="p_surface_true",
        pressure_units="kinematic",
        density_kg_m3=1.225,
        pressure_reference_pa=101325.0 * 1.225,
        flip_normals=True,
        rpm=RPM,
        permeable=False,
        moving_surface=True,
        rotation_center_m=[0.0, 0.0, 0.0],
        omega_rad_s=[0.0, 2.0 * math.pi * rotation_frequency_hz, 0.0],
        sound_speed_m_s=343.0,
        response_delay_s=1.0e-5,
        adjust_time_step=True,
        device=device,
        chunk_size=64,
    )
    cylinder_series = OpenFOAMSurfaceSeries(
        args.cylinder_vtk_dir,
        surface_file="Cylx15.vtk",
        pressure_field="p",
        velocity_field="U",
        pressure_units="kinematic",
        density_kg_m3=1.225,
        pressure_reference_pa=0.0,
        dtype="float64",
        cache_dir=args.cylinder_cache_dir,
    )
    if not cylinder_series.cache_enabled:
        raise RuntimeError(
            f"The cylinder binary cache is unavailable: {args.cylinder_cache_dir}"
        )
    cylinder_solver = F1ASolver(
        cylinder_series,
        rpm=RPM,
        density_kg_m3=1.225,
        sound_speed_m_s=343.0,
        response_delay_s=1.0e-5,
        adjust_time_step=True,
        device=device,
        chunk_size=64,
    )

    results = {
        "Blade": solve_case("Blade", blade_solver, observers_m),
        "Cylinder": solve_case("Cylinder", cylinder_solver, observers_m),
    }
    sampling_rate_hz = SAMPLES_PER_ROTATION * rotation_frequency_hz
    sample_count = SPECTRUM_ROTATIONS * SAMPLES_PER_ROTATION
    blade_passing_frequency_hz = BLADE_COUNT * rotation_frequency_hz
    metrics_by_case = {
        label: spectrum_metrics(
            spectra_for_result(result, sampling_rate_hz, sample_count),
            blade_passing_frequency_hz,
        )
        for label, result in results.items()
    }
    plot_directivity(observer_angles_deg, metrics_by_case, args.output, args.dpi)
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

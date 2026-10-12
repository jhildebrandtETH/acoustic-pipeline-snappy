"""Plot measured, CFD, and GNO propeller-plane acoustic directivity.

The processing matches ``propeller_plane_spl_comparison.ipynb``: the final
nine rotations are resampled to 360 samples/revolution, spectra are computed
without a window, and OSPL is integrated up to 5 kHz.  The harmonic panels
show the largest spectral bin within half a shaft-frequency of 1x and 2x BPF.

Run this script from the FoamAcoustics repository root::

    python plot_propeller_plane_directivity.py
"""

from __future__ import annotations

import argparse
import importlib
import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from tqdm.auto import tqdm

from foamacoustics import F1AResult, F1ASolver, PressureSpectrum
from foamacoustics import pressure_to_spl_spectrum
from foamacoustics.plotting import plot_directivity_metrics
from foamacoustics.utils import spl_spectrum_to_oaspl

importlib.import_module("scienceplots")


RPM = 7000.0
BLADE_COUNT = 2
OBSERVER_RADIUS_M = 1.88
SPECTRUM_ROTATIONS = 9
SAMPLES_PER_ROTATION = 360
FREQUENCY_CEILING_HZ = 5000.0
MEASUREMENT_RECORD_KEY = 10


class MeasurementRecord(TypedDict):
    """Pressure fields read from one measurement archive entry."""

    unit: str
    scaled_samples: npt.NDArray[np.float64]
    sample_rate: float


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfd-npz",
        type=Path,
        default=repo_dir.parent / "p_surface_10x7E_7000RPM_AMI.npz",
    )
    parser.add_argument(
        "--gno-npz",
        type=Path,
        default=repo_dir.parent / "10X7E_7000_surface_time_prediction.npz",
    )
    parser.add_argument("--measurement-dir", type=Path, default=repo_dir / "actual")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_dir
            / "propeller_plane_spl_comparison_figures"
            / "directivity_ospl_bpf_harmonics.png"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def observer_matrix(values: torch.Tensor) -> torch.Tensor:
    return values[None, :] if values.ndim == 1 else values


def build_solver(
    path: Path,
    *,
    pressure_field: str,
    pressure_units: Literal["kinematic", "auto"],
    pressure_reference_pa: float,
    flip_normals: bool,
    device: torch.device,
) -> F1ASolver:
    """Create a blade-surface solver with the shared comparison settings."""
    rotation_frequency_hz = RPM / 60.0
    return F1ASolver.from_npz(
        path,
        pressure_field=pressure_field,
        pressure_units=pressure_units,
        density_kg_m3=1.225,
        pressure_reference_pa=pressure_reference_pa,
        flip_normals=flip_normals,
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


def solve_case(label: str, solver: F1ASolver, observers_m: torch.Tensor) -> F1AResult:
    progress = tqdm(total=len(solver.series), desc=label, unit="frame")

    def report_progress(done: int, total: int, message: str) -> None:
        del total, message
        progress.update(done - progress.n)

    try:
        return solver.predict(
            observers_m,
            progress_callback=report_progress,
            progress_every=25,
            remove_mean=False,
            algorithm="auto",
        )
    finally:
        progress.close()


def resample_pressure(result: F1AResult, target_time_s: torch.Tensor) -> torch.Tensor:
    source_time_s = result.observer_times_s.detach().cpu().contiguous()
    source_pressure_pa = observer_matrix(result.total_pressure_pa).detach().cpu()
    target_time_s = target_time_s.contiguous()
    if target_time_s[0] < source_time_s[0] or target_time_s[-1] > source_time_s[-1]:
        raise ValueError("The requested spectrum grid is outside the pressure history.")
    right = torch.searchsorted(source_time_s, target_time_s).clamp(
        1, source_time_s.numel() - 1
    )
    left = right - 1
    weight = (target_time_s - source_time_s[left]) / (
        source_time_s[right] - source_time_s[left]
    )
    return source_pressure_pa[:, left] + weight[None, :] * (
        source_pressure_pa[:, right] - source_pressure_pa[:, left]
    )


def measurement_path(measurement_dir: Path, elevation_deg: float) -> Path:
    elevation = round(elevation_deg)
    token = f"p{elevation}" if elevation > 0 else f"m{abs(elevation)}"
    if elevation == 0:
        token = "00"
    return measurement_dir / f"10x7E_{token}_actual.pkl"


def resample_measurement_window(
    pressure_pa: torch.Tensor | npt.NDArray[np.float64],
    source_sampling_rate_hz: float,
    target_sampling_rate_hz: float,
    target_sample_count: int,
    filter_taps: int = 257,
) -> torch.Tensor:
    """Return a centered, anti-aliased measurement window on the target grid."""
    pressure_pa = torch.as_tensor(pressure_pa, dtype=torch.float64)
    duration_s = target_sample_count / target_sampling_rate_hz
    center_s = pressure_pa.numel() / (2.0 * source_sampling_rate_hz)
    target_time_s = (
        center_s
        - 0.5 * duration_s
        + torch.arange(target_sample_count, dtype=torch.float64)
        / target_sampling_rate_hz
    )

    half_width = filter_taps // 2
    cutoff_hz = 0.45 * target_sampling_rate_hz
    offsets = torch.arange(-half_width, half_width + 1, dtype=torch.float64)
    kernel = (
        2.0
        * cutoff_hz
        / source_sampling_rate_hz
        * torch.sinc(2.0 * cutoff_hz * offsets / source_sampling_rate_hz)
    )
    kernel *= torch.hann_window(filter_taps, periodic=False, dtype=torch.float64)
    kernel /= kernel.sum()

    first = max(
        0,
        math.floor(float(target_time_s[0]) * source_sampling_rate_hz) - half_width - 2,
    )
    last = min(
        pressure_pa.numel(),
        math.ceil(float(target_time_s[-1]) * source_sampling_rate_hz) + half_width + 3,
    )
    source_segment = pressure_pa[first:last]
    filtered = torch.nn.functional.conv1d(
        source_segment[None, None, :],
        kernel[None, None, :],
        padding=half_width,
    )[0, 0]
    source_time_s = (
        torch.arange(first, last, dtype=torch.float64) / source_sampling_rate_hz
    )
    right = torch.searchsorted(source_time_s, target_time_s).clamp(
        1, source_time_s.numel() - 1
    )
    left = right - 1
    weight = (target_time_s - source_time_s[left]) / (
        source_time_s[right] - source_time_s[left]
    )
    return filtered[left] + weight * (filtered[right] - filtered[left])


def load_measurement_spectra(
    measurement_dir: Path,
    observer_angles_deg: torch.Tensor,
    sampling_rate_hz: float,
    sample_count: int,
) -> list[PressureSpectrum]:
    spectra: list[PressureSpectrum] = []
    for elevation_deg in observer_angles_deg:
        path = measurement_path(measurement_dir, float(elevation_deg))
        with path.open("rb") as stream:
            archive = cast("dict[int, MeasurementRecord]", pickle.load(stream))
        measurement = archive[MEASUREMENT_RECORD_KEY]
        if measurement.get("unit") != "Pa":
            raise ValueError(f"Expected pressure in Pa in {path}.")
        pressure_window_pa = resample_measurement_window(
            measurement["scaled_samples"],
            float(measurement["sample_rate"]),
            sampling_rate_hz,
            sample_count,
        )
        spectra.append(
            pressure_to_spl_spectrum(
                pressure_window_pa,
                sampling_rate_hz=sampling_rate_hz,
                remove_mean=True,
            )
        )
    return spectra


def spectrum_metrics(
    spectra: Sequence[PressureSpectrum], blade_passing_frequency_hz: float
) -> torch.Tensor:
    """Return columns containing <=5 kHz OSPL, 1x BPF, and 2x BPF peak."""
    metrics: list[torch.Tensor] = []
    shaft_frequency_hz = RPM / 60.0
    for spectrum in spectra:
        frequencies_hz = spectrum.frequencies_hz.detach().cpu()
        spl_db = spectrum.spl_db.detach().cpu()
        ospl_db = spl_spectrum_to_oaspl(
            spl_db[1:],
            frequencies_hz[1:],
            frequency_ceiling_hz=FREQUENCY_CEILING_HZ,
        )
        harmonic_peaks: list[torch.Tensor] = []
        for harmonic in (1, 2):
            center_hz = harmonic * blade_passing_frequency_hz
            in_peak_band = torch.abs(frequencies_hz - center_hz) <= (
                0.5 * shaft_frequency_hz
            )
            if not torch.any(in_peak_band):
                raise ValueError(f"No spectral bin found near {harmonic}x BPF.")
            harmonic_peaks.append(spl_db[in_peak_band].max())
        metrics.append(torch.stack((torch.as_tensor(ospl_db), *harmonic_peaks)))
    return torch.stack(metrics)


def plot_directivity(
    observer_angles_deg: torch.Tensor,
    metrics_by_case: dict[str, torch.Tensor],
    output: Path,
    dpi: int,
    radial_limits_db: tuple[float, float] = (30.0, 75.0),
) -> None:
    """Render and save the three propeller-plane directivity metrics."""
    plt.style.use(["science", "no-latex"])
    colors = {
        "Measurement": "tab:orange",
        "CFD (AMI)": "black",
        "GNO": "tab:blue",
        "Blade": "black",
        "Cylinder": "tab:blue",
    }
    markers = {
        "Measurement": "o",
        "CFD (AMI)": "s",
        "GNO": "^",
        "Blade": "s",
        "Cylinder": "o",
    }
    metric_titles = (
        f"OSPL (0–{FREQUENCY_CEILING_HZ / 1000:g} kHz)",
        "1st harmonic peak (1× BPF)",
        "2nd harmonic peak (2× BPF)",
    )
    directivity = plot_directivity_metrics(
        observer_angles_deg,
        metrics_by_case,
        metric_titles=metric_titles,
        radial_limits_db=radial_limits_db,
        colors=colors,
        markers=markers,
        figure_title=(f"APC 10×7E at {RPM:g} RPM — propeller-plane directivity"),
        figsize=(13.5, 4.2),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    directivity.figure.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output.resolve()}")


def main() -> None:
    args = parse_args()
    for path in (args.cfd_npz, args.gno_npz):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.measurement_dir.is_dir():
        raise FileNotFoundError(args.measurement_dir)

    device = torch.device(args.device)
    rotation_frequency_hz = RPM / 60.0
    blade_passing_frequency_hz = BLADE_COUNT * rotation_frequency_hz
    # The two most negative elevations are intentionally omitted from the
    # directivity comparison.
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

    solvers = {
        "CFD (AMI)": build_solver(
            args.cfd_npz,
            pressure_field="p_surface_true",
            pressure_units="kinematic",
            pressure_reference_pa=101325.0 * 1.225,
            flip_normals=True,
            device=device,
        ),
        "GNO": build_solver(
            args.gno_npz,
            pressure_field="p_surface_pred",
            pressure_units="auto",
            pressure_reference_pa=0.0,
            flip_normals=False,
            device=device,
        ),
    }
    results = {
        label: solve_case(label, solver, observers_m)
        for label, solver in solvers.items()
    }

    sampling_rate_hz = SAMPLES_PER_ROTATION * rotation_frequency_hz
    sample_count = SPECTRUM_ROTATIONS * SAMPLES_PER_ROTATION
    common_end_s = min(
        float(result.observer_times_s[-1]) for result in results.values()
    )
    common_start_s = common_end_s - SPECTRUM_ROTATIONS / rotation_frequency_hz
    maximum_delay_s = max(
        float(torch.as_tensor(result.maximum_delay_s).max())
        for result in results.values()
    )
    if common_start_s < maximum_delay_s:
        raise RuntimeError(
            "The fully arrived histories are too short for nine rotations."
        )
    uniform_time_s = (
        common_start_s
        + torch.arange(sample_count, dtype=torch.float64) / sampling_rate_hz
    )

    spectra_by_case = {
        label: [
            pressure_to_spl_spectrum(
                pressure_pa,
                sampling_rate_hz=sampling_rate_hz,
                remove_mean=True,
            )
            for pressure_pa in resample_pressure(result, uniform_time_s)
        ]
        for label, result in results.items()
    }
    spectra_by_case = {
        "Measurement": load_measurement_spectra(
            args.measurement_dir,
            observer_angles_deg,
            sampling_rate_hz,
            sample_count,
        ),
        **spectra_by_case,
    }
    metrics_by_case = {
        label: spectrum_metrics(spectra, blade_passing_frequency_hz)
        for label, spectra in spectra_by_case.items()
    }
    plot_directivity(
        observer_angles_deg,
        metrics_by_case,
        args.output,
        args.dpi,
        radial_limits_db=(30.0, 80.0),
    )
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

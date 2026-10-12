"""Plot measured, CFD, and GNO spectra at the measurement microphone angles.

The FFT processing and plot styling match ``propeller_plane_spl_comparison``:
the final nine rotations are sampled at 360 samples/revolution, no FFT window
is applied, and spectra are shown from 100 Hz to 5 kHz.  Only propeller/RPM
cases present in all three input directories are processed.

For cases with repeated measurement runs, the first chronologically named
archive is used.  The measurement array is centred at (0, 0, 0.75) m and its
coordinates are translated directly into the CFD coordinate system, whose
rotation axis is +Y.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from foamacoustics import F1ASolver, pressure_to_spl_spectrum
from foamacoustics.plotting import plot_spectrum_comparison


PROPELLERS = ("10x5E", "10x7E", "10x8E")
BLADE_COUNT = 2
SPECTRUM_ROTATIONS = 9
SAMPLES_PER_ROTATION = 360
FREQUENCY_CEILING_HZ = 5000.0
MEASUREMENT_ARRAY_CENTER_M = torch.tensor([0.0, 0.0, 0.75])


@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: torch.Tensor
    spl_db: torch.Tensor
    frequency_resolution_hz: float


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-dir",
        type=Path,
        default=Path(
            "/media/inspire-data/Teams/Groups/ThermalSim/Projects/Public/"
            "BAZL Propeller/Datasets/Measurements"
        ),
    )
    parser.add_argument(
        "--cfd-dir", type=Path, default=Path("/home/pkaftan/CFD_Aeroacoustics_AMI")
    )
    parser.add_argument(
        "--gno-dir",
        type=Path,
        default=Path("/home/pkaftan/GNO_Aeroacoustics_AMI/Blade_GNO"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_dir / "spectrum_comparisons"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List common cases and microphone geometry without solving them.",
    )
    return parser.parse_args()


def normalize_propeller(token: str) -> str:
    wanted = token.casefold()
    for propeller in PROPELLERS:
        if propeller.casefold() == wanted:
            return propeller
    raise ValueError(f"Unsupported propeller token {token!r}.")


def discover_cases(
    measurement_dir: Path, cfd_dir: Path, gno_dir: Path
) -> dict[str, dict[int, tuple[list[Path], Path, Path]]]:
    measurement_pattern = re.compile(r"^(10x[578]e)_.+_RPM_(\d+)\.pkl$", re.IGNORECASE)
    cfd_pattern = re.compile(
        r"^p_surface_(10x[578]E)_(\d+)RPM_AMI\.npz$", re.IGNORECASE
    )
    gno_pattern = re.compile(
        r"^(10x[578]E)_(\d+)_surface_time_prediction\.npz$", re.IGNORECASE
    )

    measurements: dict[tuple[str, int], list[Path]] = {}
    cfd: dict[tuple[str, int], Path] = {}
    gno: dict[tuple[str, int], Path] = {}
    for path in measurement_dir.glob("*.pkl"):
        if match := measurement_pattern.match(path.name):
            key = (normalize_propeller(match.group(1)), int(match.group(2)))
            measurements.setdefault(key, []).append(path)
    for path in cfd_dir.glob("*.npz"):
        if match := cfd_pattern.match(path.name):
            cfd[(normalize_propeller(match.group(1)), int(match.group(2)))] = path
    for path in gno_dir.glob("*.npz"):
        if match := gno_pattern.match(path.name):
            gno[(normalize_propeller(match.group(1)), int(match.group(2)))] = path

    common = set(measurements) & set(cfd) & set(gno)
    cases: dict[str, dict[int, tuple[list[Path], Path, Path]]] = {}
    for propeller, rpm in sorted(common):
        cases.setdefault(propeller, {})[rpm] = (
            sorted(measurements[(propeller, rpm)]),
            cfd[(propeller, rpm)],
            gno[(propeller, rpm)],
        )
    return cases


def load_measurement_geometry(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with path.open("rb") as stream:
        archive = pickle.load(stream)
    channels = archive["sound"]["channels"]
    positions = torch.tensor(
        [channels[key]["channel_position"] for key in sorted(channels, key=int)],
        dtype=torch.float64,
    )
    relative = positions - MEASUREMENT_ARRAY_CENTER_M.to(dtype=torch.float64)
    radial_from_y = torch.linalg.vector_norm(relative[:, (0, 2)], dim=1)
    polar_angles_deg = torch.rad2deg(torch.atan2(radial_from_y, relative[:, 1]))
    return relative, polar_angles_deg


def resample_measurement_window(
    pressure_pa: torch.Tensor,
    source_sampling_rate_hz: float,
    target_sampling_rate_hz: float,
    target_sample_count: int,
    filter_taps: int = 257,
) -> torch.Tensor:
    """Return a centred, anti-aliased measurement window on the FFT grid."""
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
    path: Path, sampling_rate_hz: float, sample_count: int
) -> list[Spectrum]:
    with path.open("rb") as stream:
        archive = pickle.load(stream)
    channels = archive["sound"]["channels"]
    spectra = []
    for key in tqdm(sorted(channels, key=int), desc="Measurement", unit="channel"):
        channel = channels[key]
        pressure_window = resample_measurement_window(
            channel["pressure"],
            float(channel["sampling_frequency"]),
            sampling_rate_hz,
            sample_count,
        )
        spectrum = pressure_to_spl_spectrum(
            pressure_window,
            sampling_rate_hz=sampling_rate_hz,
            remove_mean=True,
        )
        spectra.append(
            Spectrum(
                spectrum.frequencies_hz.detach().cpu(),
                spectrum.spl_db.detach().cpu(),
                spectrum.frequency_resolution_hz,
            )
        )
    return spectra


def observer_matrix(values: torch.Tensor) -> torch.Tensor:
    return values[None, :] if values.ndim == 1 else values


def build_solver(
    path: Path,
    *,
    label: Literal["CFD", "GNO"],
    rpm: int,
    device: torch.device,
) -> F1ASolver:
    """Create a blade solver with shared settings for one comparison case."""
    pressure_field: str
    pressure_units: Literal["kinematic", "auto"]
    pressure_reference_pa: float
    flip_normals: bool
    if label == "CFD":
        pressure_field = "p_surface_true"
        pressure_units = "kinematic"
        pressure_reference_pa = 101325.0 * 1.225
        flip_normals = True
    else:
        pressure_field = "p_surface_pred"
        pressure_units = "auto"
        pressure_reference_pa = 0.0
        flip_normals = False
    rotation_frequency_hz = rpm / 60.0
    return F1ASolver.from_npz(
        path,
        pressure_field=pressure_field,
        pressure_units=pressure_units,
        density_kg_m3=1.225,
        pressure_reference_pa=pressure_reference_pa,
        flip_normals=flip_normals,
        rpm=float(rpm),
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


def solve_spectra(
    path: Path,
    label: Literal["CFD", "GNO"],
    rpm: int,
    observers_m: torch.Tensor,
    device: torch.device,
    sampling_rate_hz: float,
    sample_count: int,
) -> list[Spectrum]:
    rotation_frequency_hz = rpm / 60.0
    solver = build_solver(path, label=label, rpm=rpm, device=device)
    progress = tqdm(total=len(solver.series), desc=label, unit="frame")

    def report_progress(done: int, total: int, message: str) -> None:
        del total, message
        progress.update(done - progress.n)

    try:
        result = solver.predict(
            observers_m.to(device),
            progress_callback=report_progress,
            progress_every=25,
            remove_mean=False,
            algorithm="auto",
        )
    finally:
        progress.close()

    common_end_s = float(result.observer_times_s[-1])
    common_start_s = common_end_s - SPECTRUM_ROTATIONS / rotation_frequency_hz
    maximum_delay_s = float(torch.as_tensor(result.maximum_delay_s).max())
    if common_start_s < maximum_delay_s:
        raise RuntimeError(f"{label} history is too short for nine rotations.")
    target_time_s = (
        common_start_s
        + torch.arange(sample_count, dtype=torch.float64) / sampling_rate_hz
    )
    source_time_s = result.observer_times_s.detach().cpu().contiguous()
    source_pressure_pa = observer_matrix(result.total_pressure_pa).detach().cpu()
    right = torch.searchsorted(source_time_s, target_time_s).clamp(
        1, source_time_s.numel() - 1
    )
    left = right - 1
    weight = (target_time_s - source_time_s[left]) / (
        source_time_s[right] - source_time_s[left]
    )
    uniform_pressure_pa = source_pressure_pa[:, left] + weight[None, :] * (
        source_pressure_pa[:, right] - source_pressure_pa[:, left]
    )
    output = []
    for pressure_pa in uniform_pressure_pa:
        spectrum = pressure_to_spl_spectrum(
            pressure_pa, sampling_rate_hz=sampling_rate_hz, remove_mean=True
        )
        output.append(
            Spectrum(
                spectrum.frequencies_hz.detach().cpu(),
                spectrum.spl_db.detach().cpu(),
                spectrum.frequency_resolution_hz,
            )
        )
    return output


def plot_case(
    propeller: str,
    rpm: int,
    observers_m: torch.Tensor,
    spectra_by_case: dict[str, list[Spectrum]],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> list[Path]:
    """Save one measured/CFD/GNO spectrum comparison per observer."""
    colors = {"Measurement": "tab:orange", "CFD": "black", "GNO": "tab:blue"}
    widths = {"Measurement": 0.9, "CFD": 1.1, "GNO": 1.1}
    alphas = {"Measurement": 0.85, "CFD": 1.0, "GNO": 1.0}
    blade_passing_frequency_hz = BLADE_COUNT * rpm / 60.0
    case_dir = output_dir / propeller / f"{rpm}RPM"
    case_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for observer_index, observer_m in enumerate(observers_m):
        coordinate_text = ", ".join(
            f"{float(value):.3f}".rstrip("0").rstrip(".") for value in observer_m
        )
        resolution_hz = spectra_by_case["CFD"][observer_index].frequency_resolution_hz
        ax = plot_spectrum_comparison(
            {
                label: spectra_by_case[label][observer_index]
                for label in ("CFD", "GNO", "Measurement")
            },
            xlim_hz=(max(resolution_hz, 100.0), FREQUENCY_CEILING_HZ),
            blade_passing_frequency_hz=blade_passing_frequency_hz,
            colors=colors,
            linewidths=widths,
            alphas=alphas,
            title=(
                f"APC {propeller} @ {rpm} RPM — microphone {observer_index + 1}, "
                f"[{coordinate_text}] m"
            ),
            figsize=(10.0, 4.2),
        )
        figure = cast("Figure", ax.figure)
        output = case_dir / f"spectrum_microphone_{observer_index + 1:02d}.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        outputs.append(output)
        if show:
            plt.show()
        else:
            plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    for directory in (args.measurement_dir, args.cfd_dir, args.gno_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    cases = discover_cases(args.measurement_dir, args.cfd_dir, args.gno_dir)
    missing_propellers = set(PROPELLERS) - set(cases)
    if missing_propellers:
        raise RuntimeError(f"No common cases for: {sorted(missing_propellers)}")

    first_measurement = next(iter(next(iter(cases.values())).values()))[0][0]
    observers_m, angles_deg = load_measurement_geometry(first_measurement)
    print("Common cases:")
    for propeller in PROPELLERS:
        print(f"  {propeller}: {', '.join(map(str, sorted(cases[propeller])))} RPM")
    print(
        "Microphone polar angles from +Y: " + ", ".join(f"{x:.1f}°" for x in angles_deg)
    )
    if args.dry_run:
        return

    device = torch.device(args.device)
    all_outputs = []
    for propeller in PROPELLERS:
        for rpm, (measurement_paths, cfd_path, gno_path) in sorted(
            cases[propeller].items()
        ):
            print(f"\n{propeller} at {rpm} RPM")
            sampling_rate_hz = SAMPLES_PER_ROTATION * rpm / 60.0
            sample_count = SPECTRUM_ROTATIONS * SAMPLES_PER_ROTATION
            spectra_by_case = {
                "Measurement": load_measurement_spectra(
                    measurement_paths[0], sampling_rate_hz, sample_count
                ),
                "CFD": solve_spectra(
                    cfd_path,
                    "CFD",
                    rpm,
                    observers_m,
                    device,
                    sampling_rate_hz,
                    sample_count,
                ),
                "GNO": solve_spectra(
                    gno_path,
                    "GNO",
                    rpm,
                    observers_m,
                    device,
                    sampling_rate_hz,
                    sample_count,
                ),
            }
            all_outputs.extend(
                plot_case(
                    propeller,
                    rpm,
                    observers_m,
                    spectra_by_case,
                    args.output_dir,
                    args.dpi,
                    args.show,
                )
            )
    print(f"\nSaved {len(all_outputs)} spectra under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

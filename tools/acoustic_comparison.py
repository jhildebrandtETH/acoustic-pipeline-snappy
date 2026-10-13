"""Experimental/FW-H agreement using common narrow-band energy spectra.

No solver dependency: cached pressure traces can be re-reported independently.
"""
from pathlib import Path
import argparse
import json
import pickle
import re

import numpy as np

P_REF = 20e-6
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'Parameters/acousticComparison.json'


def load_config(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))



def case_geometry(case_path):
    match = re.match(r'^(.*?)_\d+(?:\.\d+)?RPM(?:_|$)', Path(case_path).name, re.I)
    if not match:
        raise ValueError('Case folder must start with <geometry>_<rpm>RPM_ for automatic matching')
    return match[1]


def resolve_observer_position(case_path, rpm, config):
    """Use the selected experimental channel, checking every matching repeat.

    Without experimental comparison, retain the original simulation-only observer.
    With measurements configured, missing or inconsistent positions are errors:
    never silently simulate another point or use a stale manual observer setting.
    """
    if not config.get('measurements_directory'):
        return [1.0, 0.0, 0.0]
    geometry = case_geometry(case_path)
    paths = match_measurements(config['measurements_directory'], geometry, rpm)
    reference = None
    for path in paths:
        pressure, _, position = load_channel(path, config['microphone_channel'], geometry, rpm)
        del pressure
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f'Microphone {config["microphone_channel"]} must have three finite coordinates: {path}')
        if reference is None:
            reference = position
        elif not np.allclose(reference, position, atol=1e-6, rtol=0):
            raise ValueError('Matched repeats use different microphone positions; select a consistent measurement directory')
    return reference.tolist()


def cache_result(result, case_path, rpm, *, rotations):
    """Retain the requested final rotations, limited to the fully arrived tail."""
    if not np.isfinite(rotations) or rotations <= 0 or not np.isfinite(rpm) or rpm <= 0:
        raise ValueError('Analysis rotations and RPM must be positive and finite')
    times = result.observer_times_s.detach().cpu().numpy().astype(float)
    pressure = result.total_pressure_pa.detach().cpu().numpy().astype(float)
    if times.ndim != 1 or pressure.shape != times.shape or len(times) < 8:
        raise ValueError('Expected a single-observer pressure trace')
    if not np.isfinite(times).all() or not np.isfinite(pressure).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Observer time and pressure must be finite; time must increase')
    start = max(float(result.maximum_delay_s), times[-1] - rotations * 60 / rpm)
    mask = times >= start
    times, pressure = times[mask], pressure[mask]
    if len(times) < 8:
        raise ValueError('Too few fully arrived acoustic samples')
    step = np.median(np.diff(times))
    if np.max(np.diff(times)) > 1.5 * step:
        raise ValueError('Observer trace has sampling gaps; cannot form a reliable spectrum')
    grid = np.arange(times[0], times[-1], step)
    pressure = np.interp(grid, times, pressure)
    case_path = Path(case_path)
    geometry = case_geometry(case_path)
    np.savez_compressed(case_path / 'report/acoustic-pressure.npz', pressure_pa=pressure,
                        sampling_frequency_hz=1 / step, rpm=rpm, geometry=geometry,
                        requested_rotations=rotations, requested_duration_s=rotations * 60 / rpm,
                        observer_m=result.observer_m.detach().cpu().numpy())


def geometry_key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def match_measurements(directory, geometry, rpm):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Measurement directory does not exist: {directory}')
    pattern = re.compile(r'^(.*?)_\d{4}-\d{2}-\d{2}.*_RPM_(\d+(?:\.\d+)?)$', re.I)
    matches = []
    for path in sorted(directory.rglob('*.pkl')):
        match = pattern.fullmatch(path.stem)
        if match and geometry_key(match[1]) == geometry_key(geometry) and float(match[2]) == float(rpm):
            matches.append(path)
    if not matches:
        raise ValueError(f'No experimental measurements match {geometry} at {rpm:g} RPM')
    return matches


class MeasurementUnpickler(pickle.Unpickler):
    """Only permit NumPy array reconstruction, not arbitrary pickle classes."""
    def find_class(self, module, name):
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(np._core.multiarray if hasattr(np, '_core') else np.core.multiarray, name)
        raise pickle.UnpicklingError(f'Unsupported measurement class: {module}.{name}')


def load_channel(path, channel, geometry, rpm):
    with Path(path).open('rb') as stream:
        data = MeasurementUnpickler(stream).load()
    if geometry_key(data['propeller_type']) != geometry_key(geometry) or float(data['targeted_rpm']) != float(rpm):
        raise ValueError(f'Filename and measurement metadata disagree: {path}')
    channels = data['sound']['channels']
    if str(channel) not in channels:
        raise ValueError(f'Microphone {channel} missing in {path}')
    selected = channels[str(channel)]
    return (np.asarray(selected['pressure'], dtype=float),
            float(selected['sampling_frequency']),
            np.asarray(selected['channel_position'], dtype=float))


def welch_density(pressure, fs, duration):
    """Periodic Hann, 50% overlap, per-segment mean removal; units Pa²/Hz."""
    pressure = np.asarray(pressure, dtype=float)
    if pressure.ndim != 1 or pressure.size < 8 or not np.isfinite(pressure).all():
        raise ValueError('Pressure must contain at least eight finite scalar samples')
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Sampling frequency and segment duration must be positive')
    count = round(fs * duration)
    if count < 8 or count > len(pressure):
        raise ValueError('Trace is too short for the requested spectral segment duration')
    window = np.hanning(count + 1)[:-1]
    total = np.zeros(count // 2 + 1)
    segments = 0
    for start in range(0, len(pressure) - count + 1, count // 2):
        trace = pressure[start:start + count]
        power = abs(np.fft.rfft((trace - trace.mean()) * window)) ** 2
        power /= fs * np.sum(window ** 2)
        power[1:-1 if count % 2 == 0 else None] *= 2
        total += power
        segments += 1
    return np.fft.rfftfreq(count, 1 / fs), total / segments


def integrate_density(freq, density, edges):
    """Exact integral of the piecewise-linear PSD; no extrapolation."""
    if edges[0] < freq[0] or edges[-1] > freq[-1]:
        raise ValueError('Requested frequencies exceed available spectrum')
    grid = np.unique(np.concatenate((freq, edges)))
    values = np.interp(grid, freq, density)
    integral = np.r_[0., np.cumsum(np.diff(grid) * (values[:-1] + values[1:]) / 2)]
    return np.diff(np.interp(edges, grid, integral))


def levels(power, floor_db):
    return 10 * np.log10(np.maximum(power / P_REF ** 2, 10 ** (floor_db / 10)))


def agreement_metrics(edges, simulated_db, measured_db, bins):
    """Integrate the displayed step curves exactly, including partial bins."""
    difference = np.asarray(simulated_db) - measured_db
    def measure(low, high):
        widths = np.maximum(0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))
        width = widths.sum()
        if width <= 0:
            return {'status': 'unavailable', 'requested_hz': [low, high]}
        area = float(np.sum(widths * abs(difference)))
        return {'status': 'complete', 'requested_hz': [low, high],
                'covered_hz': [max(float(edges[0]), low), min(float(edges[-1]), high)],
                'width_hz': float(width), 'area_db_hz': area,
                'mae_db': area / width,
                'rmse_db': float(np.sqrt(np.sum(widths * difference ** 2) / width)),
                'bias_db': float(np.sum(widths * difference) / width)}
    maximum = float(edges[-1])
    rows = [measure(float(low), maximum if high is None else float(high)) for low, high in bins]
    return measure(float(edges[0]), maximum), rows


def validate_config(config):
    for name in ('spectrum_bandwidth_hz', 'maximum_frequency_hz'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not np.isfinite(config['floor_db']):
        raise ValueError('floor_db must be finite')
    bins = config['frequency_bins_hz']
    if not bins or bins[0][0] != 0 or bins[-1][1] is not None:
        raise ValueError('Frequency bins must start at zero and end with null (remaining frequencies)')
    for index, (low, high) in enumerate(bins):
        if not np.isfinite(low) or low < 0 or (high is not None and (not np.isfinite(high) or high <= low)):
            raise ValueError('Frequency bins must have increasing finite boundaries')
        if index < len(bins) - 1 and high != bins[index + 1][0]:
            raise ValueError('Frequency bins must be contiguous and non-overlapping')


def create_comparison(case_path, config=None):
    """Read cached simulation, select all matching repeats, write plots and metrics."""
    config = load_config() if config is None else config
    validate_config(config)
    report = Path(case_path) / 'report'
    with np.load(report / 'acoustic-pressure.npz', allow_pickle=False) as data:
        pressure, fs = data['pressure_pa'], float(data['sampling_frequency_hz'])
        geometry, rpm = str(data['geometry']), float(data['rpm'])
        observer = data['observer_m']
        source_note = str(data['source_note']) if 'source_note' in data else ''
        requested_duration = float(data['requested_duration_s']) if 'requested_duration_s' in data else None
    paths = match_measurements(config['measurements_directory'], geometry, rpm)
    # The cached trace already represents the propagation rotation window.
    # N/fs is the DFT segment duration; (N-1)/fs is only the timestamp span.
    # Match this physical duration in the experiment, to within one sample.
    duration = len(pressure) / fs
    sim_f, sim_psd = welch_density(pressure, fs, duration)
    spectra, positions = [], []
    for path in paths:
        trace, rate, position = load_channel(path, config['microphone_channel'], geometry, rpm)
        frequency, density = welch_density(trace, rate, duration)
        spectra.append((frequency, density))
        positions.append(position.tolist())
    if any(np.shape(pos) != (3,) or not np.isfinite(pos).all() for pos in positions):
        raise ValueError('Each microphone position must contain three finite coordinates')
    if any(not np.allclose(pos, positions[0], atol=1e-6) for pos in positions):
        raise ValueError('Matched repeats use different microphone positions; select a consistent measurement directory')
    low = max(sim_f[1], *(freq[1] for freq, _ in spectra))
    high = min(config['maximum_frequency_hz'], sim_f[-1], *(freq[-1] for freq, _ in spectra))
    bandwidth = max(config['spectrum_bandwidth_hz'], 2 / duration)
    if high - low < bandwidth:
        raise ValueError('Insufficient common frequency coverage for one comparison band')
    edges = np.r_[np.arange(low, high, bandwidth), high]
    sim_power = integrate_density(sim_f, sim_psd, edges)
    repeated_power = np.array([integrate_density(f, p, edges) for f, p in spectra])
    measured_power = repeated_power.mean(axis=0)
    sim_db = levels(sim_power, config['floor_db'])
    exp_db = levels(measured_power, config['floor_db'])
    repeat_db = levels(repeated_power, config['floor_db'])
    overall, bins = agreement_metrics(edges, sim_db, exp_db, config['frequency_bins_hz'])
    warnings = [
        'Pressure is assumed calibrated in Pa, unweighted; no background-noise correction is applied.',
        'Coordinates are used as supplied, with origin at the propeller centre and rotation about y.',
        'Low frequencies below the first positive spectral sample and frequencies beyond common coverage are excluded.',
        'Broadband differences can dominate this metric; it is not a tonal-peak or perceptual score.',
    ]
    position_match = np.allclose(observer, positions[0], atol=1e-3, rtol=0)
    if not position_match:
        warnings.insert(0, 'PROVISIONAL: simulated observer and measurement microphone positions differ.')
    if source_note:
        warnings.insert(0, source_note)
    metadata = {
        'status': 'complete', 'geometry': geometry, 'rpm': rpm,
        'source_note': source_note,
        'microphone_channel': str(config['microphone_channel']), 'observer_m': observer.tolist(),
        'microphone_position_m': positions[0], 'positions_match': bool(position_match),
        'measurement_files': [str(path) for path in paths], 'repeat_count': len(paths),
        'segment_duration_s': duration, 'nominal_bandwidth_hz': bandwidth,
        'requested_duration_s': requested_duration, 'effective_rotations': duration * rpm / 60,
        'fft_spacing_hz': fs / len(pressure),
        'floor_db': config['floor_db'], 'frequency_bins_hz': config['frequency_bins_hz'],
        'overall': overall, 'bins': bins, 'warnings': warnings,
        'simulated_band_limited_spl_db': float(levels(sim_power.sum(), config['floor_db'])),
        'measured_band_limited_spl_db': float(levels(measured_power.sum(), config['floor_db'])),
    }
    metadata['band_limited_level_difference_db'] = (metadata['simulated_band_limited_spl_db'] - metadata['measured_band_limited_spl_db'])
    plot_comparison(report, edges, sim_db, exp_db, repeat_db, metadata)
    np.savetxt(report / 'acoustic-comparison.csv', np.c_[edges[:-1], edges[1:], sim_db, exp_db, sim_db - exp_db],
               delimiter=',', header='lower_hz,upper_hz,simulated_spl_db,measured_spl_db,difference_db', comments='')
    (report / 'acoustic-comparison.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return metadata


def plot_comparison(report, edges, simulated, measured, repeats, metadata):
    from tools.plotting import pyplot as plt
    from tools import MATPLOTLIB_LOCK
    blue, orange = '#2463a6', '#db7525'
    extend = lambda values: np.r_[values, values[-1]]
    ymin = min(simulated.min(), repeats.min()) - 5
    ymax = max(simulated.max(), repeats.max()) + 5
    with MATPLOTLIB_LOCK, plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        for kind, title in [('simulated', 'Simulated spectrum'), ('measured', 'Experimental spectrum'), ('overlay', 'Acoustic agreement')]:
            fig, ax = plt.subplots(figsize=(10.5, 5.4))
            fig.subplots_adjust(left=.09, right=.98, top=.81, bottom=.20)
            fig.suptitle(title, x=.09, ha='left', fontsize=18, fontweight='bold')
            ax.set_title(f"{metadata['geometry']}  |  {metadata['rpm']:g} RPM  |  microphone {metadata['microphone_channel']}  |  {metadata['repeat_count']} repeats", loc='left', fontsize=10, pad=14)
            if kind != 'measured':
                ax.step(edges, extend(simulated), where='post', color=blue, lw=1.2, label='Simulated')
            if kind != 'simulated':
                ax.fill_between(edges, extend(repeats.min(axis=0)), extend(repeats.max(axis=0)), step='post', color=orange, alpha=.14, label='Experimental repeat range')
                ax.step(edges, extend(measured), where='post', color=orange, lw=1.2, label='Experimental energy mean')
            if kind == 'overlay':
                colors = ['#70b7ab', '#c7a1cd', '#e7bc66', '#97afd8']
                for index, row in enumerate(metadata['bins']):
                    if row['status'] != 'complete':
                        continue
                    lo, hi = row['covered_hz']
                    grid = np.unique(np.r_[lo, edges[(edges > lo) & (edges < hi)], hi])
                    indices = np.clip(np.searchsorted(edges, grid, side='right') - 1, 0, len(simulated) - 1)
                    ax.fill_between(grid, simulated[indices], measured[indices], step='post', color=colors[index % len(colors)], alpha=.35)
                    ax.axvline(lo, color='#777777', lw=.7, ls=':')
                fig.text(.09, .055, f"Mean absolute difference: {metadata['overall']['mae_db']:.2f} dB   |   RMSE: {metadata['overall']['rmse_db']:.2f} dB   |   lower is better", fontsize=11)
            else:
                fig.text(.09, .055, f"Common Hann/Welch processing  |  nominal band width {metadata['nominal_bandwidth_hz']:.2f} Hz  |  unweighted", fontsize=10)
            ax.set(xlabel='Frequency [Hz]', ylabel='Narrow-band SPL [dB re 20 µPa]', xlim=(edges[0], edges[-1]), ylim=(ymin, ymax))
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.18)
            ax.legend(loc='upper right', fontsize=8)
            if metadata.get('source_note'):
                fig.text(.09, .012, metadata['source_note'], fontsize=9, color='#a22b2b')
            elif not metadata['positions_match']:
                fig.text(.09, .012, 'PROVISIONAL: observer and microphone positions differ.', fontsize=9, color='#a22b2b')
            fig.savefig(report / f'acoustic-{kind}.png', dpi=160)
            plt.close(fig)


def append_comparison_report(canvas, case_path):
    """Append three spectrum pages and a paginated metrics/provenance section."""
    from reportlab.lib.utils import simpleSplit
    report = Path(case_path) / 'report'
    path = report / 'acoustic-comparison.json'
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    width, height = canvas._pagesize
    def page(title):
        if canvas._code:
            canvas.showPage()
        canvas.setFont('Helvetica-Bold', 17)
        canvas.drawString(45, height - 50, title)
        canvas.setFont('Helvetica', 10)
    if data['status'] != 'complete':
        page('Acoustic comparison unavailable')
        for index, line in enumerate(simpleSplit(data.get('detail', ''), 'Helvetica', 10, width - 90)):
            canvas.drawString(45, height - 85 - index * 15, line)
        return
    for kind, title in [('simulated', 'Simulated SPL'), ('measured', 'Experimental SPL'), ('overlay', 'Simulated vs experimental SPL')]:
        page(title)
        canvas.drawImage(str(report / f'acoustic-{kind}.png'), 30, height - 365, width=width - 60, height=280, preserveAspectRatio=True)
        canvas.drawString(45, height - 390, f"Common evaluated range: {data['overall']['covered_hz'][0]:.1f} - {data['overall']['covered_hz'][1]:.1f} Hz")
        canvas.drawString(45, height - 410, f"Simulation position: {data['observer_m']} m; microphone: {data['microphone_position_m']} m")
    page('Acoustic agreement metrics')
    y = height - 85
    def line(text, bold=False):
        nonlocal y
        font = 'Helvetica-Bold' if bold else 'Helvetica'
        # Split long paths as well as prose to keep every provenance line on-page.
        parts = []
        for paragraph in simpleSplit(text, font, 10, width - 90):
            current = ''
            for char in paragraph:
                if canvas.stringWidth(current + char, font, 10) > width - 90:
                    parts.append(current)
                    current = ''
                current += char
            parts.append(current)
        for part in parts:
            if y < 55:
                page('Acoustic agreement - continued')
                y = height - 85
            canvas.setFont(font, 10)
            canvas.drawString(45, y, part)
            y -= 15
        y -= 5
    line('Mean absolute difference = shaded area / evaluated frequency width. Zero is perfect; lower is better.', True)
    for name, row in [('Overall', data['overall'])] + [(f'Bin {i + 1}', row) for i, row in enumerate(data['bins'])]:
        if row['status'] != 'complete':
            line(f"{name}: unavailable (outside common coverage)")
            continue
        lo, hi = row['covered_hz']
        line(f"{name}: {lo:.1f}-{hi:.1f} Hz | mean difference {row['mae_db']:.2f} dB | area {row['area_db_hz']:.1f} dB Hz", True)
        line(f"RMSE {row['rmse_db']:.2f} dB; signed bias {row['bias_db']:+.2f} dB (positive = simulation louder).")
    line(f"Band-limited total SPL: simulation {data['simulated_band_limited_spl_db']:.2f} dB; experimental {data['measured_band_limited_spl_db']:.2f} dB; difference {data['band_limited_level_difference_db']:+.2f} dB.")
    line(f"Processing: periodic Hann, 50% overlap, mean removed per segment; segment {data['segment_duration_s']:.5f} s; nominal narrow-band width {data['nominal_bandwidth_hz']:.2f} Hz; numerical floor {data['floor_db']:g} dB.")
    for warning in data['warnings']:
        line(warning)
    line('Experimental files (equal weight in linear acoustic energy):', True)
    for filename in data['measurement_files']:
        line(filename)


def comparison_or_status(case_path, config=None):
    report = Path(case_path) / 'report'
    report.mkdir(parents=True, exist_ok=True)
    try:
        config = load_config() if config is None else config
        if not config.get('measurements_directory'):
            raise ValueError('Set measurements_directory in Parameters/acousticComparison.json to enable comparison.')
        return create_comparison(case_path, config)
    except Exception as exc:
        status = {'status': 'unavailable', 'detail': str(exc)}
        (report / 'acoustic-comparison.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--measurements', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.measurements:
        config['measurements_directory'] = str(args.measurements)
    result = comparison_or_status(args.case, config)
    if result['status'] != 'complete':
        parser.exit(1, result['detail'] + '\n')
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import A4
    output = args.case / 'report/acoustic-comparison.pdf'
    canvas = Canvas(str(output), pagesize=A4)
    append_comparison_report(canvas, args.case)
    canvas.save()
    print(output)


if __name__ == '__main__':
    main()

"""Configured microphone coordinates, batched caches and planar directivity."""
import csv
import json
from pathlib import Path

import numpy as np


def settings(config):
    value = config.get('directivity', {})
    if not isinstance(value, dict) or not isinstance(value.get('enabled', False), bool):
        raise ValueError('directivity.enabled must be boolean')
    return value


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must have three finite coordinates')
    return result


def frame(config):
    value = settings(config)
    if value.get('experimental_rotation_center_m') is None:
        raise ValueError('Enter directivity.experimental_rotation_center_m before enabling multi-observer propagation')
    origin = vector(value['experimental_rotation_center_m'], 'experimental_rotation_center_m')
    centre = vector(value.get('simulation_rotation_center_m', [0, 0, 0]), 'simulation_rotation_center_m')
    rotation = np.asarray(value.get('experimental_to_simulation_rotation', np.eye(3)), dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError('experimental_to_simulation_rotation must be a proper orthonormal 3x3 rotation matrix')
    return origin, centre, rotation


def transform_position(position, channel, config):
    if not settings(config).get('enabled', False):
        return vector(position, 'microphone position')
    origin, centre, rotation = frame(config)
    raw = settings(config).get('microphone_positions_m', {}).get(str(channel), position)
    return centre + rotation @ (vector(raw, f'M{channel} position') - origin)


def measurement_channels(path, geometry, rpm):
    from tools.acoustic_comparison import MeasurementUnpickler, geometry_key
    with Path(path).open('rb') as stream:
        data = MeasurementUnpickler(stream).load()
    if geometry_key(data['propeller_type']) != geometry_key(geometry) or float(data['targeted_rpm']) != float(rpm):
        raise ValueError(f'Filename and measurement metadata disagree: {path}')
    return data['sound']['channels']


def resolve_observers(case, rpm, config):
    from tools.acoustic_comparison import case_geometry, match_measurements
    frame(config)
    paths = match_measurements(config['measurements_directory'], case_geometry(case), rpm)
    positions = None
    for path in paths:
        channels = measurement_channels(path, case_geometry(case), rpm)
        current = {str(k): vector(v['channel_position'], f'M{k} position') for k, v in channels.items()}
        if positions is None:
            positions = current
        elif set(current) != set(positions) or any(not np.allclose(current[k], positions[k], atol=1e-6, rtol=0) for k in current):
            raise ValueError('Matched repeats must have the same channels and microphone positions')
    if not positions or str(config['microphone_channel']) not in positions:
        raise ValueError('Selected microphone channel is missing from the measurements')
    overrides = settings(config).get('microphone_positions_m', {})
    if not isinstance(overrides, dict) or set(overrides) - set(positions):
        raise ValueError('microphone_positions_m must map existing channel IDs to coordinates')
    ids = sorted(positions, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    raw = np.array([vector(overrides.get(k, positions[k]), f'M{k} position') for k in ids])
    mapped = np.array([transform_position(positions[k], k, config) for k in ids])
    if np.any(np.linalg.norm(mapped - frame(config)[1], axis=1) <= 1e-9):
        raise ValueError('Microphones cannot coincide with the rotation centre')
    return ids, raw, mapped


def cache_observers(result, case, rpm, config, channels, raw_positions, rotations):
    """Use one fully arrived time window for all microphones and the legacy plot."""
    from tools.acoustic_comparison import case_geometry
    times = result.observer_times_s.detach().cpu().numpy().astype(float)
    pressure = result.total_pressure_pa.detach().cpu().numpy().astype(float)
    positions = result.observer_m.detach().cpu().numpy().astype(float)
    delay = np.asarray(result.maximum_delay_s.detach().cpu() if hasattr(result.maximum_delay_s, 'detach') else result.maximum_delay_s)
    if (times.ndim != 1 or len(times) < 8 or pressure.shape != (len(channels), len(times))
            or positions.shape != (len(channels), 3) or not np.isfinite(pressure).all()
            or not np.isfinite(times).all() or not np.isfinite(delay).all()
            or not np.isfinite(rotations) or rotations <= 0 or not np.isfinite(rpm) or rpm <= 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid multi-observer pressure trace or analysis duration')
    mask = times >= max(float(delay.max()), times[-1] - rotations * 60 / rpm)
    retained = times[mask]
    if len(retained) < 8:
        raise ValueError('Too few fully arrived samples common to every observer')
    step = np.median(np.diff(retained))
    if np.max(np.diff(retained)) > 1.5 * step:
        raise ValueError('Observer trace has sampling gaps')
    grid = np.arange(retained[0], retained[-1], step)
    values = np.array([np.interp(grid, retained, row[mask]) for row in pressure])
    payload = dict(pressure_pa=values, sampling_frequency_hz=1 / step, rpm=rpm,
                   geometry=case_geometry(case), channel_ids=np.array(channels), observer_m=positions,
                   experimental_positions_m=raw_positions, requested_duration_s=rotations * 60 / rpm,
                   coordinate_frame=json.dumps([part.tolist() for part in frame(config)]),
                   requested_rotations=rotations, window_start_s=grid[0], window_end_s=grid[-1])
    np.savez_compressed(Path(case) / 'report/acoustic-observers.npz', **payload)
    select_cached_observer(case, config)


def read_cache(case, config):
    with np.load(Path(case) / 'report/acoustic-observers.npz', allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    ids, raw, mapped = resolve_observers(case, float(payload['rpm']), config)
    if (str(payload['coordinate_frame']) != json.dumps([part.tolist() for part in frame(config)])
            or ids != payload['channel_ids'].tolist() or mapped.shape != payload['observer_m'].shape
            or not np.allclose(mapped, payload['observer_m'], atol=1e-8, rtol=0)
            or not np.allclose(raw, payload['experimental_positions_m'], atol=1e-8, rtol=0)):
        raise ValueError('Observer settings changed: rerun acoustic propagation before rebuilding directivity')
    return payload


def select_cached_observer(case, config):
    payload = read_cache(case, config)
    index = payload['channel_ids'].tolist().index(str(config['microphone_channel']))
    single = {k: payload[k] for k in ('sampling_frequency_hz', 'rpm', 'geometry', 'requested_duration_s', 'requested_rotations')}
    single.update(pressure_pa=payload['pressure_pa'][index], observer_m=payload['observer_m'][index])
    np.savez_compressed(Path(case) / 'report/acoustic-pressure.npz', **single)


def plane_coordinates(plane, channels, raw, config):
    origin, _, _ = frame(config)
    ids = [str(k) for k in plane['channels']]
    if len(ids) < 3 or len(set(ids)) != len(ids) or set(ids) - set(channels):
        raise ValueError('Each directivity plane needs at least three distinct, available channels')
    zero = vector(plane['zero_direction'], 'plane zero_direction')
    positive = vector(plane['positive_direction'], 'plane positive_direction')
    if min(np.linalg.norm(zero), np.linalg.norm(positive)) <= 1e-9:
        raise ValueError('Plane direction vectors must be nonzero')
    zero, positive = zero / np.linalg.norm(zero), positive / np.linalg.norm(positive)
    if abs(zero @ positive) > 1e-6:
        raise ValueError('Plane angle directions must be perpendicular')
    points = raw[[channels.index(k) for k in ids]] - origin
    tolerance = float(settings(config).get('plane_tolerance_m', .005))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('plane_tolerance_m must be positive')
    residual = abs(points @ np.cross(zero, positive))
    if np.any(residual > tolerance):
        raise ValueError(f"Plane {plane['name']}: microphones are not coplanar with the configured centre (max offset {residual.max():.4f} m)")
    angles = np.mod(np.degrees(np.arctan2(points @ positive, points @ zero)), 360)
    angles[np.isclose(angles, 360)] = 0
    order = np.argsort(angles)
    return dict(name=str(plane['name']), channels=[ids[i] for i in order],
                angles_deg=angles[order].tolist(), radii_m=np.linalg.norm(points, axis=1)[order].tolist(),
                zero_direction=zero.tolist(), positive_direction=positive.tolist())


def integrated_components(frequency, density, bounds, config):
    from tools.acoustic_tonal import separate_tones, tonal_settings
    from tools.acoustic_comparison import integrate_density
    broadband, tonal, _ = separate_tones(frequency, density, tonal_settings(config))
    return {name: float(integrate_density(frequency, psd, np.array(bounds))[0])
            for name, psd in [('overall', density), ('broadband', broadband), ('tonal', tonal)]}


def create_directivity(case, config):
    from tools.acoustic_comparison import match_measurements, welch_density, levels
    data = read_cache(case, config)
    ids = data['channel_ids'].tolist()
    planes = [plane_coordinates(p, ids, data['experimental_positions_m'], config)
              for p in settings(config).get('planes', [])]
    if not planes:
        raise ValueError('Configure at least one directivity plane')
    duration = data['pressure_pa'].shape[1] / float(data['sampling_frequency_hz'])
    spectra = {k: {'simulated': [welch_density(row, float(data['sampling_frequency_hz']), duration)], 'measured': []}
               for k, row in zip(ids, data['pressure_pa'])}
    paths = match_measurements(config['measurements_directory'], str(data['geometry']), float(data['rpm']))
    for path in paths:
        measurements = measurement_channels(path, str(data['geometry']), float(data['rpm']))
        for channel in ids:
            measurement = measurements[channel]
            spectra[channel]['measured'].append(welch_density(measurement['pressure'], float(measurement['sampling_frequency']), duration))
    all_spectra = [s for groups in spectra.values() for group in groups.values() for s in group]
    bounds = [max(float(f[1]) for f, _ in all_spectra),
              min(config['maximum_frequency_hz'], *(float(f[-1]) for f, _ in all_spectra))]
    if bounds[0] >= bounds[1]:
        raise ValueError('No common positive-frequency range for directivity')
    summary = dict(status='complete', geometry=str(data['geometry']), rpm=float(data['rpm']),
                   covered_hz=bounds, segment_duration_s=duration, planes=planes, channels={},
                   settings=settings(config), tonal_settings=config.get('tonal_analysis', {}),
                   measurement_files=[str(p) for p in paths],
                   note='Raw SPL at microphone distances; no distance normalization. Sparse points are not a resolved directivity pattern.')
    for index, channel in enumerate(ids):
        row = dict(observer_m=data['observer_m'][index].tolist(), experimental_position_m=data['experimental_positions_m'][index].tolist())
        for kind, group in spectra[channel].items():
            powers = [integrated_components(f, p, bounds, config) for f, p in group]
            mean = {key: float(np.mean([p[key] for p in powers])) for key in powers[0]}
            row[kind] = dict(power_pa2=mean, **{key + '_spl_db': float(levels(value, config['floor_db'])) if value > 0 else None for key, value in mean.items()})
        summary['channels'][channel] = row
    report = Path(case) / 'report'
    plot_directivity(report, summary)
    with (report / 'acoustic-directivity.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['channel', 'source', 'x_m', 'y_m', 'z_m', 'overall_spl_db', 'broadband_spl_db', 'tonal_spl_db'])
        for channel, row in summary['channels'].items():
            for kind in ('simulated', 'measured'):
                writer.writerow([channel, kind, *row['observer_m'], *[row[kind][key + '_spl_db'] for key in ('overall', 'broadband', 'tonal')]])
    return summary


def plot_directivity(report, summary):
    from tools.plotting import pyplot as plt
    from tools import MATPLOTLIB_LOCK
    maximum_gap = float(summary['settings'].get('maximum_connected_gap_deg', 60))
    if not np.isfinite(maximum_gap) or not 0 < maximum_gap <= 180:
        raise ValueError('maximum_connected_gap_deg must be in (0, 180]')
    for index, plane in enumerate(summary['planes']):
        with MATPLOTLIB_LOCK:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})
            fig.subplots_adjust(left=.07, right=.93, top=.82, bottom=.20, wspace=.30)
            fig.suptitle(f"{summary['geometry']} | {summary['rpm']:g} RPM | {plane['name']}", fontsize=16, y=.97)
            angles = np.array(plane['angles_deg'])
            values = [row[kind][key + '_spl_db'] for ch in plane['channels'] for row in [summary['channels'][ch]]
                      for kind in ('simulated', 'measured') for key in ('overall', 'broadband', 'tonal')]
            finite = [v for v in values if v is not None and np.isfinite(v)]
            limits = (np.floor(min(finite) / 10) * 10 - 5, np.ceil(max(finite) / 10) * 10 + 10) if finite else (-10, 10)
            for ax, kind in zip(axes, ('simulated', 'measured')):
                ax.set_title('Simulation' if kind == 'simulated' else 'Experiment', pad=22)
                ax.set_ylim(*limits)
                if np.all(angles <= 180 + 1e-6):
                    ax.set_thetamin(0)
                    ax.set_thetamax(180)
                for key, color, style in [('overall', '#247eaf', '-'), ('broadband', '#ff7f0e', '--'), ('tonal', '#2ca02c', ':')]:
                    levels = np.array([summary['channels'][ch][kind][key + '_spl_db'] if summary['channels'][ch][kind][key + '_spl_db'] is not None else np.nan for ch in plane['channels']])
                    # Break lines across unsampled sectors; never close the curve.
                    x, y = [], []
                    for i, angle in enumerate(angles):
                        if i and angle - angles[i - 1] > maximum_gap:
                            x.append(np.nan); y.append(np.nan)
                        x.append(np.radians(angle)); y.append(levels[i])
                    ax.plot(x, y, color=color, ls=style, lw=2, marker='o', ms=4, label=key.capitalize() + ' SPL')
                for angle, ch in zip(angles, plane['channels']):
                    value = summary['channels'][ch][kind]['overall_spl_db']
                    if value is not None:
                        ax.annotate('M' + ch, (np.radians(angle), value), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
                ax.set_rlabel_position(0)
                ax.grid(alpha=.4)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .92), ncol=3, frameon=False)
            radii = plane['radii_m']
            fig.text(.5, .17, 'SPL [dB re 20 µPa] — common frequency range %.1f–%.1f Hz' % tuple(summary['covered_hz']), ha='center')
            fig.text(.5, .115, f"Microphone radii {min(radii):.3f}–{max(radii):.3f} m | raw levels; no distance normalization", ha='center', fontsize=9)
            fig.text(.5, .07, 'Markers are computed/measured positions; lines only guide the eye. Tonal/broadband split is estimated.', ha='center', fontsize=9)
            fig.text(.5, .025, f"0° along {plane['zero_direction']}; 90° along {plane['positive_direction']} (experimental axes)", ha='center', fontsize=8)
            filename = f'acoustic-directivity-plane-{index + 1}.png'
            fig.savefig(Path(report) / filename, dpi=160)
            plt.close(fig)
            plane['figure'] = filename
        plane['comparison_figures'] = plot_component_comparisons(report, summary, plane, index, limits, maximum_gap)


def plot_component_comparisons(report, summary, plane, index, limits, maximum_gap):
    """Overlay simulation and experiment separately for each integrated SPL."""
    from tools.plotting import pyplot as plt
    from tools import MATPLOTLIB_LOCK
    figures = {}
    angles = np.asarray(plane['angles_deg'])
    for component in ('overall', 'broadband', 'tonal'):
        with MATPLOTLIB_LOCK:
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
            fig.subplots_adjust(left=.12, right=.88, top=.82, bottom=.23)
            fig.suptitle(f'{component.capitalize()} SPL — simulation / experiment', fontsize=15, y=.97)
            fig.text(.5, .91, f"{summary['geometry']} | {summary['rpm']:g} RPM | {plane['name']}", ha='center', fontsize=10)
            ax.set_ylim(*limits)
            if np.all(angles <= 180 + 1e-6):
                ax.set_thetamin(0)
                ax.set_thetamax(180)
            for kind, label, color, style, marker in (
                ('simulated', 'Simulation', '#2463a6', '-', 'o'),
                ('measured', 'Experiment', '#c73546', '--', 's'),
            ):
                x, y = [], []
                for i, (angle, channel) in enumerate(zip(angles, plane['channels'])):
                    if i and angle - angles[i - 1] > maximum_gap:
                        x.append(np.nan); y.append(np.nan)
                    value = summary['channels'][channel][kind][component + '_spl_db']
                    x.append(np.radians(angle)); y.append(value if value is not None else np.nan)
                ax.plot(x, y, color=color, ls=style, lw=2, marker=marker, ms=5,
                        markerfacecolor=color if kind == 'simulated' else 'white', label=label)
            for angle, channel in zip(angles, plane['channels']):
                values = [summary['channels'][channel][kind][component + '_spl_db'] for kind in ('simulated', 'measured')]
                valid = [value for value in values if value is not None and np.isfinite(value)]
                if valid:
                    ax.annotate('M' + channel, (np.radians(angle), max(valid)), xytext=(0, 10),
                                textcoords='offset points', ha='center', fontsize=9)
            ax.set_rlabel_position(0)
            ax.grid(alpha=.4)
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .87), ncol=2, frameon=False)
            fig.text(.5, .19, 'SPL [dB re 20 µPa] | %.1f–%.1f Hz' % tuple(summary['covered_hz']), ha='center', fontsize=10)
            fig.text(.5, .14, f"Microphone radii {min(plane['radii_m']):.3f}–{max(plane['radii_m']):.3f} m; no distance normalization", ha='center', fontsize=9)
            fig.text(.5, .10, 'Same radial scale across all components; lines only guide the eye.', ha='center', fontsize=9)
            if component != 'overall':
                fig.text(.5, .065, 'Tonal / broadband separation is estimated.', ha='center', fontsize=9)
            fig.text(.5, .025, f"0°: {plane['zero_direction']}; 90°: {plane['positive_direction']} (experimental axes)", ha='center', fontsize=8)
            filename = f'acoustic-directivity-plane-{index + 1}-{component}.png'
            fig.savefig(Path(report) / filename, dpi=160)
            plt.close(fig)
            figures[component] = filename
    return figures


def directivity_or_status(case, config):
    try:
        if not settings(config).get('enabled', False):
            result = dict(status='disabled', detail='Multi-observer directivity is disabled; enter coordinate settings and enable it.')
        else:
            result = create_directivity(case, config)
    except Exception as exc:
        result = dict(status='unavailable', detail=str(exc))
    (Path(case) / 'report/acoustic-directivity.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def append_directivity_report(canvas, case):
    path = Path(case) / 'report/acoustic-directivity.json'
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] == 'disabled':
        return
    width, height = canvas._pagesize
    if data['status'] != 'complete':
        from reportlab.lib.utils import simpleSplit
        if canvas._code:
            canvas.showPage()
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(40, height - 50, 'Multi-observer directivity unavailable')
        canvas.setFont('Helvetica', 10)
        for index, line in enumerate(simpleSplit(data['detail'], 'Helvetica', 10, width - 80)):
            canvas.drawString(40, height - 85 - 15 * index, line)
        return
    for plane in data['planes']:
        if canvas._code:
            canvas.showPage()
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(40, height - 50, 'Multi-observer acoustic directivity')
        canvas.drawImage(str(path.parent / plane['figure']), 25, height - 400, width=width - 50, height=310, preserveAspectRatio=True)
        for component, filename in plane.get('comparison_figures', {}).items():
            canvas.showPage()
            canvas.setFont('Helvetica-Bold', 16)
            canvas.drawString(40, height - 50, f'{component.capitalize()} SPL: simulation / experiment')
            canvas.drawImage(str(path.parent / filename), 25, height - 510,
                             width=width - 50, height=420, preserveAspectRatio=True)

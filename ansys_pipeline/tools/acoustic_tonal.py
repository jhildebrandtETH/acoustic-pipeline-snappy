"""Descriptive narrow-tone separation, not a standardized tonality assessment.

Detect narrow peaks above a running median, replace their Hann main-lobe
neighbourhood with a linear baseline, and assign only excess energy to tones.
The pointwise split conserves the original PSD before any dB display floor.
"""
import json
from pathlib import Path

import numpy as np


def write_cached_simulation_tonal(case_path, config):
    """Supply the same figure when experimental comparison is unavailable."""
    from tools.acoustic_comparison import welch_density
    report = Path(case_path) / 'report'
    with np.load(report / 'acoustic-pressure.npz', allow_pickle=False) as data:
        pressure = data['pressure_pa']
        fs = float(data['sampling_frequency_hz'])
        metadata = dict(rpm=float(data['rpm']), geometry=str(data['geometry']),
                        blade_count=config.get('blade_count', 2),
                        microphone_channel=config.get('microphone_channel', 'simulation only'),
                        source_note=str(data['source_note']) if 'source_note' in data else '')
    frequency, density = welch_density(pressure, fs, len(pressure) / fs)
    from tools.acoustic_annotations import minimum_frequency_hz
    return write_tonal_figures(report, {'simulated': [(frequency, density)]}, metadata,
                              config, max(frequency[1], minimum_frequency_hz(config, metadata['rpm'])),
                              min(frequency[-1], config['maximum_frequency_hz']))


def tonal_settings(config):
    settings = dict(median_window_bins=21, threshold_db=8.0, half_width_bins=2)
    supplied = config.get('tonal_analysis', {})
    if not isinstance(supplied, dict) or set(supplied) - set(settings):
        raise ValueError('tonal_analysis must contain only median_window_bins, threshold_db, half_width_bins')
    settings.update(supplied)
    for key in ('median_window_bins', 'half_width_bins'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'tonal_analysis.{key} must be a positive integer')
    if settings['median_window_bins'] % 2 != 1 or settings['median_window_bins'] < 4 * settings['half_width_bins'] + 3:
        raise ValueError('tonal median window must be odd and at least 4 * half_width_bins + 3')
    if not np.isfinite(settings['threshold_db']) or settings['threshold_db'] <= 0:
        raise ValueError('tonal threshold_db must be positive and finite')
    return settings


def separate_tones(frequency, density, settings):
    """Return broadband PSD, tonal PSD and detected native-grid peak indices.

    Only interior, resolved local maxima are detected. Broad features wider than
    the configured tone neighbourhood at half peak power remain broadband.
    Peaks at spectrum endpoints cannot be resolved and remain broadband too.
    """
    frequency, density = np.asarray(frequency, float), np.asarray(density, float)
    if (frequency.ndim != 1 or density.shape != frequency.shape or len(frequency) < 3
            or not np.isfinite(frequency).all() or not np.isfinite(density).all()
            or np.any(np.diff(frequency) <= 0) or np.any(density < 0)):
        raise ValueError('Expected finite increasing frequencies and nonnegative PSD')
    step = np.diff(frequency)
    if not np.allclose(step, step[0], rtol=1e-5, atol=1e-10):
        raise ValueError('Tone detection requires a uniform native FFT grid')
    width = settings['median_window_bins']
    # Chunk the median calculation so long recordings do not allocate N*width.
    padded = np.pad(density, width // 2, mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    baseline = np.empty_like(density)
    for start in range(0, len(density), 4096):
        baseline[start:start + 4096] = np.median(windows[start:start + 4096], axis=1)
    # Suppress floating-point FFT residue relative to the strongest bin.
    numerical_floor = max(float(density.max()) * 1e-12, np.finfo(float).tiny)
    ratio = density / np.maximum(baseline, numerical_floor)
    candidates = np.flatnonzero((density[1:-1] > density[:-2]) &
                               (density[1:-1] >= density[2:]) &
                               (ratio[1:-1] >= 10 ** (settings['threshold_db'] / 10))) + 1
    half = settings['half_width_bins']
    mask = np.zeros(len(density), dtype=bool)
    peaks = []
    for peak in candidates:
        if peak - half < 1 or peak + half >= len(density) - 1:
            continue
        left, right = peak, peak
        while left > 0 and density[left - 1] > density[peak] / 2:
            left -= 1
        while right < len(density) - 1 and density[right + 1] > density[peak] / 2:
            right += 1
        if right - left + 1 > 2 * half + 1:
            continue
        peaks.append(peak)
        mask[peak - half:peak + half + 1] = True
    broadband = density.copy()
    # Interpolate across merged regions once, avoiding double-counted tones.
    broadband[mask] = np.minimum(density[mask], np.interp(
        frequency[mask], frequency[~mask], density[~mask]))
    return broadband, density - broadband, np.asarray(peaks, dtype=int)


def write_tonal_figures(report, groups, metadata, config, low, high):
    """Analyse native PSDs, average repeat components in energy, export figures.

    groups maps simulated/measured to lists of (native frequency, PSD) pairs.
    The union plotting grid retains every integration knot across repeat grids.
    """
    from tools.acoustic_comparison import integrate_density, levels
    from tools.plotting import pyplot as plt
    from tools import MATPLOTLIB_LOCK

    settings = tonal_settings(config)
    report = Path(report)
    bpf = metadata['rpm'] * metadata.get('blade_count', 2) / 60
    if not np.isfinite(bpf) or bpf <= 0 or not 0 < low < high:
        raise ValueError('Positive BPF and frequency coverage are required')
    floor = config['floor_db']
    summary = dict(method='local_median_narrow_peak_excess_v1', settings=settings,
                   covered_hz=[float(low), float(high)], blade_passing_frequency_hz=bpf,
                   note='Estimated separation; resolution and threshold dependent. No motor subtraction.',
                   spectra={})
    for kind, spectra in groups.items():
        grid = np.unique(np.concatenate(([low, high], *[
            f[(f > low) & (f < high)] for f, _ in spectra])))
        original, background = np.zeros_like(grid), np.zeros_like(grid)
        tone_frequencies, spacings = [], []
        for frequency, density in spectra:
            broadband, _, peaks = separate_tones(frequency, density, settings)
            if low < frequency[0] or high > frequency[-1]:
                raise ValueError('Tonal coverage exceeds available PSD')
            original += np.interp(grid, frequency, density) / len(spectra)
            background += np.interp(grid, frequency, broadband) / len(spectra)
            tone_frequencies.extend(float(frequency[p]) for p in peaks if low <= frequency[p] <= high)
            spacings.append(float(frequency[1] - frequency[0]))
        tonal = np.maximum(original - background, 0)
        powers = {name: float(integrate_density(grid, values, np.array([low, high]))[0])
                  for name, values in [('tonal', tonal), ('broadband', background), ('overall', original)]}
        totals = {name + '_spl_db': (float(levels(value, floor)) if value > 0 else None)
                  for name, value in powers.items()}
        detected = np.unique(tone_frequencies)
        orders = np.rint(detected / bpf)
        # Classification tolerance is limited to avoid calling every coarse bin a harmonic.
        tolerance = min(max(spacings), .1 * bpf)
        harmonic = (orders >= 1) & (abs(detected - orders * bpf) <= tolerance)
        row = dict(**totals, power_pa2=powers, detected_tones_hz=detected.tolist(),
                   bpf_tones_hz=detected[harmonic].tolist(), bpf_tolerance_hz=tolerance,
                   fft_spacing_hz=spacings, repeat_count=len(spectra))
        summary['spectra'][kind] = row
        np.savetxt(report / f'acoustic-tonal-{kind}.csv',
                   np.c_[grid, original, background, tonal], delimiter=',',
                   header='frequency_hz,original_psd_pa2_per_hz,broadband_psd_pa2_per_hz,tonal_psd_pa2_per_hz', comments='')
        with MATPLOTLIB_LOCK, plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
            fig, ax = plt.subplots(figsize=(10.5, 6.5))
            fig.subplots_adjust(left=.10, right=.97, bottom=.30, top=.76)
            fig.suptitle(f"{'Simulated' if kind == 'simulated' else 'Experimental'} tonal / broadband spectrum",
                         x=.10, y=.97, ha='left', fontsize=17, fontweight='bold')
            fig.text(.10, .91, f"{metadata['geometry']} | {metadata['rpm']:g} RPM | "
                     f"microphone {metadata['microphone_channel']} | {len(spectra)} recording(s)")
            ax.set(xscale='log', xlim=(low, high), xlabel='Frequency [Hz]',
                   ylabel='PSD [dB re (20 µPa)²/Hz]', facecolor='#f0f0f0')
            ax.plot(grid, levels(background, floor), color='#247eaf', lw=2.5, label='Estimated broadband PSD')
            ax.plot(grid, levels(original, floor), color='#202020', ls=':', lw=1.1,
                    label='Total simulated PSD' if kind == 'simulated' else 'Total experimental PSD')
            ax.fill_between(grid, levels(background, floor), levels(original, floor),
                            where=tonal > 0, color='#c73546', alpha=.22, interpolate=True)
            ax.vlines(detected, levels(np.interp(detected, grid, background), floor),
                      levels(np.interp(detected, grid, original), floor),
                      color='#b92335', lw=.85, alpha=.8, zorder=3)
            for selected, label, filled in [(harmonic, 'Detected BPF harmonics', True),
                                             (~harmonic, 'Other detected tones', False)]:
                positions = detected[selected]
                ax.scatter(positions, levels(np.interp(positions, grid, original), floor),
                           s=32, facecolors='#d62728' if filled else 'none', edgecolors='#d62728',
                           linewidths=1.2, label=label, zorder=4)
            top = ax.secondary_xaxis('top', functions=(lambda f: f / bpf, lambda order: order * bpf))
            top.set_xlabel('Frequency / BPF', color='#d62728')
            top.tick_params(axis='x', colors='#d62728')
            top.spines['top'].set_color('#d62728')
            ax.grid(which='major', alpha=.3)
            ax.legend(loc='best', fontsize=8)
            for y, key, color in [(.22, 'tonal', '#d62728'), (.18, 'broadband', '#247eaf'), (.14, 'overall', '#202020')]:
                value = totals[key + '_spl_db']
                label = f'{value:.1f} dB' if value is not None else 'no resolved energy'
                fig.text(.10, y, f'{key.capitalize()} SPL: {label}', color=color, fontsize=13, fontweight='bold')
            fig.text(.10, .095, f"Unweighted totals: {low:.1f}–{high:.1f} Hz | BPF {bpf:.2f} Hz | estimated separation", fontsize=9)
            fig.text(.10, .06, f"Local median {settings['median_window_bins']} bins | threshold {settings['threshold_db']:g} dB | "
                     f"tone half-width {settings['half_width_bins']} bins | no motor subtraction", fontsize=8)
            note = metadata.get('source_note') or ('PROVISIONAL: observer and microphone positions differ.'
                                                  if not metadata.get('positions_match', True) else '')
            if note:
                fig.text(.10, .02, note, fontsize=8, color='#a22b2b', wrap=True)
            fig.savefig(report / f'acoustic-tonal-{kind}.png', dpi=160)
            plt.close(fig)
    (report / 'acoustic-tonal.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary

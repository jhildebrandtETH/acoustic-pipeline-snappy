"""Shared native-PSD overlay and compact acoustic report pages."""
import json
from pathlib import Path

import numpy as np


def plot_psd_overlay(report, experimental_spectra, metadata):
    from tools.acoustic_comparison import levels
    from tools.plotting import pyplot as plt
    from tools import MATPLOTLIB_LOCK
    report = Path(report)
    sources = {kind: np.loadtxt(report / f'acoustic-tonal-{kind}.csv', delimiter=',', skiprows=1)
               for kind in ('simulated', 'measured')}
    measured = sources['measured']
    repeats = np.array([np.interp(measured[:, 0], f, p) for f, p in experimental_spectra])
    floor = metadata['floor_db']
    with MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        fig.subplots_adjust(left=.10, right=.98, top=.80, bottom=.24)
        fig.suptitle('Simulation / experiment — total PSD', x=.10, ha='left', fontsize=17, fontweight='bold')
        ax.set_title(f"{metadata['geometry']} | {metadata['rpm']:g} RPM | microphone {metadata['microphone_channel']} | "
                     f"{metadata['repeat_count']} experimental repeats", loc='left', fontsize=10, pad=12)
        ax.fill_between(measured[:, 0], levels(repeats.min(axis=0), floor), levels(repeats.max(axis=0), floor),
                        color='#db7525', alpha=.14, label='Experimental repeat range')
        for kind, color, label, marker in [('simulated', '#2463a6', 'Simulation', '^'),
                                           ('measured', '#db7525', 'Experimental energy mean', 'o')]:
            data = sources[kind]
            ax.plot(data[:, 0], levels(data[:, 1], floor), color=color, lw=1, label=label)
            tones = np.array(metadata['tonal_analysis']['spectra'][kind]['detected_tones_hz'])
            ax.scatter(tones, levels(np.interp(tones, data[:, 0], data[:, 1]), floor),
                       marker=marker, s=25, facecolors='#b92335' if kind == 'simulated' else 'none',
                       edgecolors='#b92335', linewidths=1, zorder=4,
                       label='Simulated tones' if kind == 'simulated' else 'Experimental tones')
        ax.set(xscale='log', xlim=metadata['tonal_analysis']['covered_hz'],
               xlabel='Frequency [Hz]', ylabel='PSD [dB re (20 µPa)²/Hz]')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
        fig.text(.10, .12, f"Band total: simulation {metadata['simulated_band_limited_spl_db']:.2f} dB | "
                 f"experiment {metadata['measured_band_limited_spl_db']:.2f} dB", fontsize=10)
        fig.text(.10, .075, 'Native PSD before tonal separation; red markers indicate estimated tones.', fontsize=9)
        note = metadata.get('source_note') or ('PROVISIONAL: observer and microphone positions differ.' if not metadata['positions_match'] else '')
        if note:
            fig.text(.10, .025, note, color='#a22b2b', fontsize=8, wrap=True)
        fig.savefig(report / 'acoustic-psd-overlay.png', dpi=160)
        plt.close(fig)


def append_summary(canvas, case):
    from reportlab.lib.utils import simpleSplit
    from tools.acoustic_directivity import append_directivity_report
    report = Path(case) / 'report'
    from tools.acoustic_resolution import create_resolution_report_data, append_resolution_report
    append_resolution_report(canvas, create_resolution_report_data(case))
    path = report / 'acoustic-comparison.json'
    if not path.exists():
        append_directivity_report(canvas, case)
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    width, height = canvas._pagesize
    def page(title):
        if canvas._code:
            canvas.showPage()
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(40, height - 48, title)
    if data['status'] != 'complete':
        page('Acoustic comparison unavailable')
        canvas.setFont('Helvetica', 10)
        for i, line in enumerate(simpleSplit(data.get('detail', ''), 'Helvetica', 10, width - 80)):
            canvas.drawString(40, height - 85 - i * 15, line)
        append_directivity_report(canvas, case)
        return
    page('Simulation / experiment comparison')
    overlay = report / 'acoustic-psd-overlay.png'
    if overlay.is_file():
        canvas.drawImage(str(overlay), 25, height - 355, width=width - 50, height=280, preserveAspectRatio=True)
    else:
        canvas.setFont('Helvetica', 10)
        canvas.drawString(40, height - 95, 'Rebuild the acoustic comparison to generate the shared PSD overlay.')
    y = height - 385
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(40, y, 'Agreement summary — existing narrow-band SPL metrics')
    y -= 20
    columns = [40, 120, 285, 375, 465]
    headers = ['Range', 'Frequency [Hz]', 'MAE [dB]', 'RMSE [dB]', 'Bias [dB]']
    def heading():
        canvas.setFont('Helvetica-Bold', 9)
        for x, text in zip(columns, headers):
            canvas.drawString(x, y, text)
    heading()
    y -= 19
    rows = [('Overall', data['overall'])] + [(f'Bin {i+1}', r) for i, r in enumerate(data['bins'])]
    for label, row in rows:
        if y < 105:
            page('Comparison summary — continued')
            y = height - 85
            heading()
            y -= 19
        if row['status'] == 'complete':
            lo, hi = row['covered_hz']
            values = [label, f'{lo:.1f}–{hi:.1f}', f"{row['mae_db']:.2f}", f"{row['rmse_db']:.2f}", f"{row['bias_db']:+.2f}"]
        else:
            values = [label, 'Outside common coverage', '—', '—', '—']
        canvas.setFont('Helvetica', 9)
        for x, text in zip(columns, values):
            canvas.drawString(x, y, text)
        y -= 18
    notes = [
        f"Metrics use {data['nominal_bandwidth_hz']:.2f} Hz nominal bands; the figure shows native PSD.",
        'MAE/RMSE: lower is better. Positive bias means the simulation is louder.',
        f"Hann/Welch duration {data['segment_duration_s']:.4f} s; simulation FFT spacing {data['fft_spacing_hz']:.2f} Hz.",
        f"Simulation observer: {data['observer_m']} m; measurement position: {data['microphone_position_m']} m.",
        'Full metrics, processing settings and measurement provenance: acoustic-comparison.json.',
    ]
    if data.get('source_note'):
        notes.insert(0, data['source_note'])
    if not data['positions_match']:
        notes.insert(0, 'PROVISIONAL: simulated and measured observer positions differ.')
    for note in notes:
        for text in simpleSplit(note, 'Helvetica', 9, width - 80):
            if y < 45:
                page('Comparison notes — continued')
                y = height - 85
            canvas.setFont('Helvetica', 9)
            canvas.drawString(40, y, text)
            y -= 13
        y -= 3
    for kind in data.get('tonal_analysis', {}).get('spectra', {}):
        page(f"Estimated tonal / broadband SPL — {kind}")
        canvas.drawImage(str(report / f'acoustic-tonal-{kind}.png'), 25, height - 430,
                         width=width - 50, height=350, preserveAspectRatio=True)
        canvas.setFont('Helvetica', 10)
        canvas.drawString(40, height - 455, 'Separation depends on threshold and resolution; no motor subtraction.')
    append_directivity_report(canvas, case)

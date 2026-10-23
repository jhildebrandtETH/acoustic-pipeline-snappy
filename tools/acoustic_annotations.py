"""Readable BPF reference markers; these are expected frequencies, not detected tones."""
import math
import numpy as np


def minimum_frequency_hz(config, rpm):
    """Configurable lower analysis/display bound; zero restores native coverage."""
    multiple = config.get('minimum_frequency_bpf_multiple', 0.5)
    blades = config.get('blade_count', 2)
    if (isinstance(multiple, bool) or not isinstance(multiple, (int, float))
            or not math.isfinite(multiple) or multiple < 0):
        raise ValueError('minimum_frequency_bpf_multiple must be finite and nonnegative')
    if isinstance(blades, bool) or not isinstance(blades, int) or blades < 1:
        raise ValueError('blade_count must be a positive integer')
    if not math.isfinite(rpm) or rpm <= 0:
        raise ValueError('RPM must be positive and finite')
    return multiple * rpm * blades / 60


def add_bpf_markers(ax, rpm, blade_count=2):
    if isinstance(blade_count, bool) or int(blade_count) != blade_count or blade_count < 1:
        raise ValueError("blade_count must be a positive integer")
    bpf = float(rpm) * blade_count / 60
    if not math.isfinite(bpf) or bpf <= 0:
        raise ValueError("RPM must be positive and finite")
    low, high = ax.get_xlim()
    first, last = max(1, math.ceil(low / bpf)), math.floor(high / bpf)
    # Label a readable subset; keep every harmonic reference line.
    stride = max(1, math.ceil((last-first+1) / 10))
    for order in range(first, last+1):
        x = order*bpf
        ax.axvline(x, color="#666666", ls=":", lw=.65, alpha=.4,
                   label=f"BPF = {bpf:.2f} Hz ({blade_count} blades)" if order == first else "_nolegend_")
        if order <= 3 or order % stride == 0:
            ax.text(x, .98, "BPF" if order == 1 else f"{order}x", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=7, color="#555555")
    return bpf


def band_total_db(levels):
    values = np.asarray(levels, dtype=float)
    peak = values.max()
    return float(peak + 10*np.log10(np.power(10., (values-peak)/10).sum()))

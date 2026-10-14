# Estimated tonal and broadband SPL

Additional `acoustic-tonal-simulated.png` and `acoustic-tonal-measured.png` figures
show native Welch PSD (black dotted), estimated broadband PSD (blue), detected
tones (red), and a frequency/BPF top axis. Both frequency axes are logarithmic.
Filled red markers indicate detected peaks within one FFT-bin spacing of a BPF
harmonic, capped at 10% of BPF; hollow markers indicate other detected peaks.
Expected harmonics without detected peaks are not marked as tones. These are
separate simulated and experimental views at the configured microphone, not
invented z/D stations. Motor PSD/SPL is omitted because no motor-only input exists.

## Method and limits

The default descriptive separation operates on each original, uniformly spaced
FFT grid, before the comparison's narrow-band rebinning:

1. Estimate the local baseline using a reflected, 21-bin running median of linear PSD.
2. Detect interior local maxima at least 8 dB above that baseline. Reject features
   wider than 5 bins at half peak power and peaks without two bins plus an
   interpolation anchor on both sides. Unresolved endpoint tones remain broadband.
3. Replace the two bins either side of each accepted peak (including the peak)
   with linear interpolation between the outside PSD samples, capped at the
   original PSD. Merge overlapping regions. Only the removed excess is tonal PSD;
   all remaining energy is broadband. There is no subtraction outside these regions.
4. Average experimental repeat components in linear energy with equal weights.
   Integrate over the same common positive-frequency range used for comparison,
   preserving every native-grid integration knot. Convert integrated Pa² to SPL
   relative to (20 µPa)². Tonal energy + broadband energy = overall energy.

This is a threshold- and resolution-dependent **estimate**, not a standardized
tonality test or an exact reproduction of the reference image's unknown method.
Short traces, leakage outside the selected neighbourhood, RPM drift, close tones,
and random narrow noise peaks can affect the split. A single simulated segment
has greater spectral variance than a many-segment experimental average. Review
the highlighted peaks and vary the settings when interpreting results.
For detection only, the baseline is bounded below at 120 dB beneath the strongest
PSD bin to suppress floating-point FFT residue. This does not change PSD energy.

## Configuration and outputs

Optional settings in each case's `Parameters/acousticComparison.json` (older case
files use these same defaults):

```json
"tonal_analysis": {
  "median_window_bins": 21,
  "threshold_db": 8.0,
  "half_width_bins": 2
}
```

The median window must be odd and at least `4 * half_width_bins + 3`; thresholds
and half-width must be positive. Changing these settings only requires rebuilding
from cached pressure, as described in [the comparison guide](acoustic-comparison.md).
Half-width is in native FFT bins, not comparison bands.
The running-median definition follows the usual local median operation (see
[SciPy's reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html));
this implementation uses NumPy and introduces no additional dependency.

Each figure annotates estimated tonal, broadband and overall **band-limited SPL**,
in dB re 20 µPa. The plotted ordinate is **PSD**, in dB re (20 µPa)²/Hz; peak PSD
heights are not integrated tone SPLs. Exact zero energy is saved as a null SPL
and labelled "no resolved energy"; the numerical display floor does not add energy.

`acoustic-tonal.json` and the comparison JSON retain settings, coverage, linear
powers, integrated levels, detected frequencies, and BPF classification tolerance.
`acoustic-tonal-{simulated,measured}.csv` retain original, broadband and tonal PSD
in Pa²/Hz. The additional figures are embedded in comparison and full simulation
reports. When comparison is disabled/unavailable, a successful acoustic run also
saves the simulated-only tonal figure and embeds it in the full report.

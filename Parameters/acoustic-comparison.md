# Acoustic agreement report

The acoustic stage now saves its fully arrived final pressure trace, automatically
matches experimental files by geometry and exact target RPM, and adds comparison
pages to `report/simulation_report.pdf`. No CFD rerun is needed when only the
comparison settings change and `report/acoustic-pressure.npz` already exists.

## Settings

Edit `<case>/Parameters/acousticComparison.json`. Repository `Parameters` seed
new cases only. Existing cases read their own copy, including when rebuilding
reports from cached pressure. A missing case config is an explicit error; there
is no fallback to the source folder.

Settings are loaded when the acoustic stage starts and kept consistent for that
propagation and comparison. Edit the case file before that stage begins. Changes
during propagation take effect on the next acoustic run or report rebuild, not
halfway through the current calculation. Changing the microphone requires a new
acoustic run; changing comparison bins only requires a report rebuild.

Other case parameters are also consumed by their corresponding stages. Generated
mesh settings require remeshing; this does not make every parameter hot-reloadable.
OpenFOAM runtime rereading remains governed by its dictionaries and supported controls.

Settings:

- `measurements_directory`: recursively searched measurement folder. Set to null
  to disable comparison. The initial value is the supplied local Measurements folder.
- `microphone_channel`: the only observer-selection setting, initially `"1"`.
  Before propagation, the pipeline reads this channel's `channel_position` from
  every geometry/RPM-matched `.pkl` file and uses that coordinate for the simulated
  observer. There is no separate `observer_m` setting. Missing channels, invalid
  coordinates or inconsistent positions across repeats stop the acoustic stage
  with an explicit error rather than silently choosing another observer.
  For example, set `"microphone_channel": "6"` to simulate and compare microphone 6.
  Coordinates are used directly in metres, assuming a shared propeller-centred
  origin and y rotation axis. No coordinate transform is applied.
  Changing the channel requires rerunning acoustic propagation, but not CFD when
  the saved source surface data are available. Merely regenerating a report does
  not change the observer in an existing cached pressure trace.
  When `measurements_directory` is null, comparison is disabled and the original
  simulation-only observer `[1, 0, 0]` m is used.
- `frequency_bins_hz`: initially `[[0, 1000], [1000, 2000], [2000, null]]`.
  Use contiguous increasing boundaries; null means the remaining common range.
- `maximum_frequency_hz`: initially 5000 Hz; use another positive ceiling as needed.
- `spectrum_bandwidth_hz`: displayed narrow-band width, initially 20 Hz.
  The actual width is at least twice the inverse segment duration.
- `floor_db`: numerical floor, initially -120 dB; this is not a measured noise floor.

Matching accepts names such as `10x7e_2024-11-28-13-14-55_RPM_4000.pkl` for a
`10x7e_4000RPM_...` case. Geometry comparison ignores case and punctuation, but
does not guess aliases or choose a nearby RPM. Metadata inside each file must
agree with its name. Every matched repeat receives equal weight in linear energy.
Repeats with inconsistent microphone coordinates are rejected rather than mixed.
The reader supports the supplied NumPy-array measurement format and rejects
unrecognised pickle classes. Keep the folder restricted to trusted measurement data.

## Automatic segment duration

`ACOUSTIC_ANALYSIS_ROTATIONS` in `acoustic_propagation.py` is the single analysis
window setting (default 15 rotations). Requested duration is rotations � 60 / RPM.
For 15 rotations at 4000 RPM this is 0.225 s, with FFT-bin spacing approximately
1 / 0.225 = 4.44 Hz. Hann-window peak resolution is broader than the bin spacing.
The cached trace uses the final requested rotations, excluding the propagation
startup; a shorter fully arrived tail limits the actual duration.

The comparison derives segment duration from the actual cached sample count and
sample rate (N/fs), and applies the same physical duration to experimental Welch
segments, rounded to the nearest experimental sample. The simulation uses one
segment; longer experiments average multiple segments with 50% overlap. A shorter
experimental recording is reported as unavailable rather than silently changing
its resolution. JSON reports record actual duration, effective rotations and FFT
spacing. A legacy `segment_duration_s` config entry is ignored and can be removed.
The displayed `spectrum_bandwidth_hz` remains a separate smoothing/rebinning setting.
Changing rotation count requires rerunning acoustic propagation to refresh the
cached trace; rebuilding a report alone uses the existing cached window.

## Rebuild from cached acoustic pressure

From the repository root, in the pipeline Python environment:

```powershell
python -m tools.acoustic_comparison "path/to/10x7e_4000RPM_case"
```

Override the measurement directory with `--measurements "path/to/Measurements"`
or select another JSON file inside the case Parameters folder with
`--config "settings.json"`. External config paths are rejected.
This produces a standalone `report/acoustic-comparison.pdf`. Regenerating the full
simulation report afterwards embeds the updated comparison pages.

Outputs also include `acoustic-simulated.png`, `acoustic-measured.png`,
`acoustic-overlay.png`, `acoustic-comparison.csv`, and `acoustic-comparison.json`.
The JSON includes provenance, actual frequency coverage, coordinates and all metrics.
Missing matches or invalid measurements appear explicitly as an unavailable
comparison in the report; the simulation-only plot remains available.

## Spectra and agreement

Both pressure traces use the same physical segment duration, periodic Hann window,
50% overlap, per-segment mean removal and one-sided power spectral density in Pa²/Hz.
PSD is integrated into common narrow bands before conversion to dB re 20 µPa.
The last narrow band may be shorter; its exact width is used in all area metrics.
This differs from coherent-gain-corrected single-tone FFT amplitudes in the old
measurement plotter. The comparison plots are deliberately labelled narrow-band SPL.
The normalization follows the density convention documented by
[SciPy's Welch reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html);
the implementation uses NumPy and adds no SciPy dependency.

All three plots share frequency and SPL limits. Experimental shading is the
minimum-to-maximum repeat range, not a confidence interval. Coloured shading in
the overlay is the absolute difference between the displayed step curves, split
at the configurable reporting-bin boundaries.

For each reporting bin and the entire common frequency range:

- **Area (dB Hz)** = integral of absolute simulated-minus-measured SPL difference.
- **Mean absolute difference (dB)** = area divided by evaluated frequency width.
  This is the primary agreement metric: zero is identical, lower is better.
  Unlike raw area it allows comparisons between bins of unequal widths.
- **RMSE (dB)** weights larger differences more strongly.
- **Signed bias (dB)** is positive when the simulation is louder on average.
- **Band-limited total SPL difference** compares total linear energy over the
  evaluated range. It can hide spectral-shape differences, so it is supplemental.

These are descriptive errors, not an arbitrary agreement percentage. Overall MAE
is weighted by frequency width, not an unweighted average of the reporting bins.
DC and frequencies below the first positive spectral sample are excluded. There
is no extrapolation beyond the lower Nyquist limit or configured frequency ceiling.
Consequently a requested 0–1000 Hz bin reports its actual evaluated lower bound.
Bins without coverage are unavailable, not zero error.

Assumptions: calibrated pressure in Pa, unweighted spectra, consistent microphone
and observer coordinates, no background-noise subtraction. Broadband discrepancies
can dominate the area; a separate tonal/BPF metric may be useful in future.

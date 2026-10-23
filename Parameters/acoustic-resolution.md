# Spatial mesh frequency cap

Acoustic reports end with one value: **Estimated spatial frequency cap: X Hz**.
There are no comparisons against preset 5 kHz or 12 kHz targets. This value is
printed in the report only; it does not change upper plot limits. For the
initial simulation runs, `maximum_frequency_hz` is manually set to `5000.0`
in the repository defaults. This is a temporary analysis/plot limit and is
independent of the calculated spatial cap. Existing cases must update their
own `Parameters/acousticComparison.json` to use the same limit.

## Local cutoff model

The model uses local turbulent velocity and local cell size:

```
u_turb,i = sqrt(2 k_i / 3)
f_MC,i   = u_turb,i / (N Delta_i)
f_cap    = min(f_MC,i) over selected source cells
```

`k_i` is the saved OpenFOAM turbulent kinetic energy in m2/s2. The velocity
scale is a local isotropic turbulent fluctuation scale, not sound speed, blade
speed or the mean velocity magnitude. Scalar k is unchanged by a rotating-frame
velocity transformation. The default `N=2` is the shortest captured turbulent
length scale in the mesh-frequency-cutoff model. Increasing N lowers the cap.
This is an engineering estimate of a turbulence-based spatial cutoff, not an
accuracy guarantee or a Nyquist limit based on time sampling.

The formula is documented in
[Minisci, Scardigli and Gallizio, APG Acoustic Symposium 2019, slide 7](https://strathprints.strath.ac.uk/70275/1/Minisci_APGAS2019_Propagation_of_boundary_and_geometrical_uncertainties_for_the_aeroacoustics_analysis.pdf).
The single-number aggregation (minimum across the selected cells) is this
pipeline's conservative policy, not a universal aggregation from that source.
For example, a local k of 150 m2/s2 gives a turbulent velocity of 10 m/s. With
Delta = 1 mm and N = 2, that cell's cutoff is 5000 Hz. Other selected cells can
lower the final cap. The code does not tune the result towards 5000 Hz.

## Settings and scope

Configure `mesh_resolution` in each case's `Parameters/acousticComparison.json`:

```json
"mesh_resolution": {
  "source_patches": ["propeller"],
  "cells_per_length_scale": 2.0,
  "cell_size": "volume_cuberoot"
}
```

The selected source region consists of the internal cells adjacent to those
boundary patches. The strict minimum includes every selected cell, including
stagnant/low-k cells. A zero k gives a zero cutoff; nothing is silently removed
to produce a more attractive value. This value applies to this region only,
not automatically to the rest of the wake or every possible noise mechanism.
The report identifies the limiting cell, its k, turbulent velocity, and Delta.

The default `volume_cuberoot` uses Delta = V^(1/3), a nominal local grid scale.
For stretched meshes, select `bbox_diagonal` for a conservative upper bound on
cell diameter instead. Both choices and the factor N are printed in the report.
A minimum over near-wall cells may be much lower than a typical source-region
cutoff; it is deliberately not replaced with a mean or a high percentile.

The most recent reconstructed flow snapshot above time zero supplies k. Mesh
files are resolved at that same time, including earlier saved mesh changes.
ASCII and gzip are supported. Missing/binary/inconsistent k or mesh data,
negative/nonfinite k, and newer unreconstructed processor output produce
**cap unavailable**, never a fallback sound-speed estimate or a stale value.
A mesh-only case without flow data cannot provide this flow-dependent cap.

`report/acoustic-resolution.json` records the cap and provenance.
`report/acoustic-resolution-cells.csv` stores each selected cell's Delta, k,
turbulent velocity scale and local cutoff. Both are regenerated with the report.
Legacy sound-speed, convective-speed, samples-per-period and acoustic
cells-per-wavelength settings are ignored and listed as obsolete in the report.

For RANS this is a modeled turbulence estimate. For DES/LES, the saved k field
may represent modeled/subgrid energy rather than total resolved-plus-modeled
energy; that limitation is printed in the report. The cap does not establish
that a steady/frozen-loading calculation resolves broadband noise, and it does
not impose a cutoff on deterministic BPF harmonics. Timestep and acoustic source
sampling checks are separate from this spatial cap.

## Lower acoustic plot bound

`minimum_frequency_bpf_multiple` still defaults to `0.5`: the lower bound is
`multiple * blade_count * RPM / 60`, or the first usable positive frequency if
higher. Use `0` to restore native lower coverage. Comparison, PSD, tonal and
simulation-only spectra, and bands integrated into directivity plots, use this
bound. Displayed totals and comparison metrics follow the same retained band;
cached pressure and upper plot limits are unchanged.

Existing cases use their own Parameters copy. Rebuild reports to add the cap;
rerun acoustic postprocessing from cached pressure to regenerate figures.

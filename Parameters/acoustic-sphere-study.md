# Acoustic sphere diameter studies

The root snappy pipeline records pressure **and velocity** on several closed
permeable spheres in one CFD run, then generates separate acoustic results and
a simulation PDF for each diameter. Factors multiply the measured propeller span;
they are not radii or metres. The independent `ansys_pipeline/` is unchanged.

## One CFD run: 2.5D to 1.1D in steps of 0.1D

Run from the repository root in the normal Linux/WSL environment:

```sh
python main.py --sim-dir output/sphere-study --rpms 6000 \
  --mode AMI --turbulence kOmegaSST --wall-functions yes \
  --total-cores 8 --cores-per-case 8 \
  --acoustic-surface permeable --acoustic-sphere-range 2.5 1.1 -0.1 \
  --end-on rev 30
```

Replace RPM, turbulence, allocation and duration with your simulation settings.
Put the intended STL in `output/sphere-study/STL/` first; otherwise the bundled
example propeller is used. This creates **15 sampling spheres**, sharing one mesh
and one flow solve per RPM/geometry case.

Thirty revolutions is an example duration, not proof of convergence. Allow for
startup, a settled flow record and physical source-to-observer delay. Analysis
uses up to the last 15 rotations, and the cross-sphere comparison requires at
least five common fully-arrived observer rotations. Verify the tail is settled.

For a pilot, replace the range option with:

```sh
--acoustic-sphere-diameters 2.5 1.8 1.1
```

Choose exactly one of `--acoustic-sphere-diameter` (legacy single sphere),
`--acoustic-sphere-diameters`, or `--acoustic-sphere-range`. Ranges include both
endpoints and must reach STOP exactly. Factors must be unique, finite and greater
than one. `refinementSphereDiameter` in `Parameters/snappyGeometry.json` sets
the mesh refinement sphere to 2.1D by default. A single permeable sampling
sphere uses that diameter unless specified separately. The domain and the
inner and outer refinement cylinders scale with the propeller span using
`domainAndCylinderSizingDiameter`, which defaults to 2.5D. Older parameter
files without this key use their configured `refinementSphereDiameter` for
both purposes. Changing the sampling diameter or mode keeps the same mesh
inputs. Every sampling sphere must fit inside the CFD domain and enclose the
complete propeller. Keep observers outside all selected spheres.

Selections are saved in `simulation_order.json` and restored on `--resume`,
including Slurm runs. Resume does not add new surfaces to a completed flow record.

## Output and reports

Each sphere writes `(p U)` every adaptive CFD timestep with `cellPoint` sampling.
The largest retains `postProcessing/writePermeableSurfaceFields/`; others use
`postProcessing/write_sphere_2p4D/`, etc. `acoustic-surfaces.json` maps factors,
physical radii, STL filenames and writer directories. `purgeWrite` controls
volume snapshots only and does not purge these surface histories.

Fifteen spheres increase sampling, disk and acoustic processing costs. Each uses
a closed subdivision-4 icosphere. Check panel-resolution sensitivity separately,
especially on the largest sphere, before attributing all differences to location.

Normal postprocessing automatically creates:

```text
<case>/
  report/simulation_report.pdf
  acoustic-study/recorded/
    study.json
    sphere-comparison.png
    sphere-comparison.csv
    sphere-comparison.npz
    sphere-comparison.json
    sphere_2p5D/<case-name>/report/
      simulation_report.pdf
      acoustic-sphere.json
      acoustic-pressure.npz
      spl_spectrum.png
      ...
    sphere_2p4D/<case-name>/report/...
    ...
    sphere_1p1D/<case-name>/report/...
```

Each PDF identifies its diameter and physical radius. Acoustic plots, caches,
measurement comparisons and enabled tonal/directivity results are independent.
The study case views share CFD inputs via absolute symlinks; keep the original
case at its saved location and do not launch CFD from the study views. Mesh/flow atlas images are reused. The
original sphere's acoustic atlas views are omitted to avoid mislabelling them.

The cross-sphere plot uses the configured spectrum microphone and a common
fully-arrived interval, uniform grid, mean removal and Hann window. It selects
an integer number of shaft rotations, then quantises duration to whole samples.
The CSV gives unwindowed, mean-removed time-domain OASPL over that same interval.
Spectra are SPL per FFT bin, not PSD per Hz. The NPZ stores frequency bins and
SPL curves. The JSON records the window or explains an unavailable comparison.
Individual PDFs can still report insufficient acoustic duration.

Study propagation explicitly sets `response_delay_s=None` to preserve physical
delay and align arrival times between surfaces. Individual PDF spectra retain
the existing per-run tail selection; use `sphere-comparison.*` for the explicitly
common-window comparison.

Rerun propagation and reports afterward, without CFD:

```sh
python -m tools.acoustic_study --case output/sphere-study/10x7E_6000RPM_AMI \
  --source recorded --rpm 6000 --mode AMI --turbulence kOmegaSST
```

Replace the example case folder with its actual name in the saved order.
Optionally select a recorded subset with `--diameters 2.5 1.5 1.1` or
`--range 2.5 1.1 -0.1`. Unrecorded factors are rejected. The command acquires the
order lock to prevent overlapping pipeline writes.

## New diameters from saved volume fields

The existing acoustic library supports `SphereSurface` and
`interpolate_openfoam_fields`. The study command uses that sampler directly:

```sh
python -m tools.acoustic_study --case output/my-order/10x7E_6000RPM_AMI \
  --source volume --range 2.5 1.1 -0.1 --time-selection 0.10:0.30 \
  --rpm 6000 --mode AMI --turbulence kOmegaSST
```

No CFD is launched. You need the reconstructed base mesh, mesh motion when
applicable, and a sufficiently long retained `p`/`U` volume history. Every sphere
must lie inside the fluid mesh. This expects a normal pipeline case, with geometry,
parameters and flow diagnostics, so full simulation reports can be rebuilt.
Results go under `acoustic-study/volume/`, separately from recorded-surface results.

The volume path uses the library's default kinematic-pressure conversion at
1.225 kg/m3, matching the pipeline acoustic reader, and sound speed 343 m/s.
Adapt these consistently if the physical setup differs.

The default uniform observer interval is the maximum saved source interval.
Override with `--sample-interval SECONDS`; an interval smaller than the largest
source gap is rejected. Actual intervals and gaps appear in each report's
`acoustic-resampling.json`. A finer grid cannot recover missing bandwidth.
Original CFD settings stay unchanged.

For future full-volume studies, set `purgeWrite 0` in the active case settings
before recording, and choose a volume write interval suitable for the target
frequencies. Current defaults retain ten volume snapshots at roughly ten times
the acoustic target interval. Deleted volume fields cannot be recovered from the
original sphere, even though that sphere's every-step history is retained.

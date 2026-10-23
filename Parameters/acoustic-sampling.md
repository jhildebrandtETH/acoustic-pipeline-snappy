# Acoustic sampling at the measurement rate

Both the normal snappy pipeline and the ANSYS import pipeline permanently use
**65536 Hz** as the default acoustic surface sampling rate. A local audit of
98 measurement files (980 microphone channels) found this rate in every channel;
the recorded time arrays also had a median spacing of `1 / 65536` seconds.

In each case's `Parameters/controlDict.cpp`:

```cpp
// Acoustic surface samples: 15.2587890625 microseconds apart.
acousticWriteInterval 1.52587890625e-5;

// Full-field snapshots: ten acoustic intervals, 152.587890625 microseconds.
writeInterval 1.52587890625e-4;
```

In both `functions.writePatchFields` (impermeable) and
`functions.writePermeableSurfaceFields` (permeable), the templates use:

```cpp
writeControl  adjustableRunTime;
writeInterval $acousticWriteInterval;
```

Only the writer selected by the acoustic-surface option is enabled. Each saved
VTK surface contains **p and U**, as required by the FW-H solver. These are source
surface samples, not experimental microphone recordings or full-volume snapshots.

## Solver timesteps and full-field snapshots

Pressure is scheduled by **physical simulation time**, not by a fixed number of
solver steps. The solver can take many smaller steps between surface samples.
`deltaT`, `maxDeltaT`, `maxCo`, adaptive stepping and the startup timestep cap
retain their existing values. Do not enlarge the solver timestep just to match
the measurement sample interval: the mesh and flow still determine its limit.

An important Foundation OpenFOAM 13 detail is that `Time::setDeltaT()` calls its
write-time alignment logic only when the **main** `writeControl` is also
`adjustableRunTime`. The main controlDict therefore uses that setting too;
changing only the surface function objects would not guarantee exact write times.
See the [OpenFOAM 13 Time implementation](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/master/src/OpenFOAM/db/Time/Time.C)
and [function-object time controls](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/master/src/OpenFOAM/db/functionObjects/functionObject/functionObject.H).

Consequently, the top-level `writeInterval` is now expressed in **seconds**.
Its default `1.52587890625e-4` writes a full field every ten acoustic samples;
it no longer means every ten solver timesteps. You can change full-field cadence
independently of acoustic cadence. Keep an integer multiple of the acoustic
interval for convenient alignment. `purgeWrite` still controls retention of
full-field snapshots; it does not purge the acoustic surface history.

All supported templates use `timePrecision 16` to preserve the sample timestamps
in output directory names. Surface samples may include a startup sample at the
initial time; comparisons should use the fully arrived analysis window. Source
sampling does not guarantee a particular spatial resolution or a 32768 Hz useful
acoustic bandwidth. The temporary plot upper limit remains 5000 Hz, and the
configured lower BPF multiple and spatial-cap report remain independent.

The FW-H observer pressure is computed from these source samples with propagation
delays. Its usable duration and sample count need not equal the approximately
20-second experimental recordings; comparison processing uses matched segment
durations. Matching sampling rates alone does not establish matching FFT spacing.

## Existing cases

New cases receive these defaults automatically in every supported AMI/MRF and
turbulence/wall-treatment template. Existing generated cases and resumed
checkpoints retain their saved dictionaries; updating the repository does not
rewrite an already running simulation. For an existing case, apply the parameter
entries above, change the main and active surface `writeControl` to
`adjustableRunTime`, and set `timePrecision 16` in its `system/controlDict` before
continuing. Previously saved every-step pressure samples are not converted to
the new cadence. For a clean comparison, start a fresh case or analyze a separate
uniformly sampled segment after the change. Legacy templates are not migrated.

The PDF settings report records the resolved impermeable/permeable writer
settings separately from the solver and full-field output controls.

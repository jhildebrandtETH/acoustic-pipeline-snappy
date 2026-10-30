# Adaptive CFD steps and a uniform acoustic grid

New snappy-pipeline cases use Courant-controlled adaptive steps without forcing
the solver to land on acoustic output times. The maximum step is the desired
acoustic sample **interval**, not the sample rate in Hz:

```cpp
// Parameters/controlDict.cpp
acousticWriteInterval 1.52587890625e-5; // target interval: 1/65536 s
maxDeltaT $acousticWriteInterval;
adjustTimeStep yes;
maxCo 5;
deltaTFactor 1.5;
```

The existing initial timestep and startup cap remain in place. The mesh/flow
and startup controls can require smaller steps. A maximum of 15.2587890625 µs
does not guarantee that the solver can reach that value or that a larger step
is temporally accurate for every flow feature.

The parameter keeps its existing name `acousticWriteInterval` for compatibility,
but now specifies the **uniform postprocessing grid and maximum CFD step**;
it does not schedule the surface writer.

## Record the actual adaptive steps

Both impermeable and permeable p/U surface writers use:

```cpp
writeControl timeStep;
writeInterval 1;
```

Only the selected acoustic-surface writer is enabled. Save every adaptive step:
the source timestamps can be nonuniform, but adjacent samples remain no farther
apart than the target interval. Both pressure and velocity are retained because
the FW-H formulations require them. `timePrecision 16` preserves timestamps in
the VTK directory names. Every-step output can increase surface-history disk
usage compared with the previous fixed-cadence writer, especially when Co forces
small CFD steps. Aerodynamics-only cases keep both acoustic writers disabled.

Full-volume fields use top-level `writeControl runTime` with the existing
`writeInterval 1.52587890625e-4` seconds. These snapshots occur on an available
solver step after a scheduled interval is crossed. They do not shorten the CFD
step to hit an exact write time. `purgeWrite` controls full-field retention and
does not purge acoustic surface history.

The previous `adjustableRunTime` write controls aligned solver steps to output
times and could quantise the achieved timestep. That alignment has been removed;
the 1.5 growth factor remains configured, but is no longer an output-alignment
workaround. Foundation implementation references:
[Time](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/OpenFOAM/db/Time/Time.C)
and [function-object time controls](https://github.com/OpenFOAM/OpenFOAM-13/blob/master/src/OpenFOAM/db/functionObjects/timeControl/timeControl.C).

## Postprocessing

1. Read the requested interval from the saved case's dictionaries, including
   ordinary local overrides. Repository defaults are not substituted for a
   missing case setting.
2. Check actual source timestamps for finite values, strict ordering and gaps
   larger than the target interval. Missing or overly sparse history is rejected
   rather than silently filled across large gaps.
3. Run FW-H using the **actual adaptive source times**, preserving the timing of
   derivatives, moving geometry and propagation delays.
4. Linearly interpolate the resulting observer-pressure components onto a
   uniform grid at the requested rate. Thickness, loading and total pressure
   share the same grid, including for multiple microphones. No endpoint
   extrapolation is used.
5. Use this grid for pressure caches, spectra, tonal/directivity analysis and
   reports. Cache files retain `time_s`, `sampling_frequency_hz` and interpolation
   metadata. Fully arrived propagation windows and minimum-duration requirements
   still apply.

`report/acoustic-resampling.json` records the requested rate, source interval
statistics, interpolation method and output count. These settings are also
included in the PDF's configuration overview. There is no pressure resampling
during CFD itself.

Interpolation does not restore information from missing samples, improve CFD
temporal accuracy, or act as an anti-alias filter. The configured 5000 Hz analysis
limit and spatial-resolution assessment remain independent. A 65536 Hz grid alone
does not establish a useful 32768 Hz acoustic bandwidth.

## Existing cases and runs

The new defaults apply to newly prepared cases. Existing case dictionaries,
order snapshots and running jobs are not rewritten automatically. To migrate a
**stopped** case, set its `maxDeltaT $acousticWriteInterval`, top-level
`writeControl runTime`, and both acoustic surface writers to `timeStep` / `1`.
Retain its intended target interval and other flow controls. Resume from a saved
time and keep the changed-settings segment identifiable for comparisons.

Previously saved uniform source histories remain usable when their spacing is
no larger than the configured interval. Histories with larger gaps require a
different justified target rate or a new run; interpolation cannot reconstruct
them. The ANSYS branch uses the same adaptive-time acoustic handling. Legacy templates are not migrated; select an explicit supported wall treatment.

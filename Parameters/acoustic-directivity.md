# Multiple observers and polar SPL diagrams

Edit `directivity` in each case's `Parameters/acousticComparison.json`. Repository
settings seed new cases only. The repository configuration enables multi-observer
comparison with the recorded APC array centre `[0, 0, 0.75]` m and identity
rotation. The CFD rotor centre stays at `[0, 0, 0]` m. Verify these settings
against the measurement coordinates when using another array.

## Coordinate settings

- `experimental_rotation_center_m`: the propeller centre in the experimental
  coordinate system, in metres. Required when enabled.
- `simulation_rotation_center_m`: the physical rotor centre in simulation
  coordinates, normally `[0, 0, 0]` for the existing geometry.
- `experimental_to_simulation_rotation`: a proper orthonormal 3x3 matrix. Identity
  is correct only if the axes match. The solver shaft remains the simulation y
  axis. This setting rotates microphone coordinates, not the CFD geometry.
- `microphone_positions_m`: optional channel-ID-to-XYZ overrides in experimental
  coordinates, for example `{"1": [1.2, 0, 0.8]}`. By default all coordinates come
  from each measurement's `channel_position`; no microphone needs to be retyped.

The mapping for a microphone is:

`simulation_position = simulation_rotation_center_m + rotation @ (experimental_position - experimental_rotation_center_m)`

Check the source geometry and axis conventions before enabling. Changing the
simulation centre must agree with the actual rotating geometry, not merely shift
the microphone array. The previous single-channel mode continues unchanged while
directivity is disabled; its existing coordinate assumptions still apply.

## Plane settings

The default XZ rotor-plane azimuth plot uses microphones M6, M10, M8 and
M1. Angles increase from +X toward +Z. The APC array has no three microphones
in a common Y-shaft plane at the configured 5 mm tolerance. For other arrays,
optional `elevation_planes` select measured microphones in shaft-containing
planes using the recorded positions and the configured coordinate mapping.

Each entry has a `name` and `radial_direction_simulation`, a nonzero direction
perpendicular to the Y shaft. For example:

```json
"elevation_planes": [
  {"name": "Elevation in simulation XY plane", "radial_direction_simulation": [1, 0, 0]},
  {"name": "Elevation in simulation ZY plane", "radial_direction_simulation": [0, 0, 1]}
]
```

The plotted meridional angle is 0 degrees along the radial direction, 90 degrees
along +shaft, 180 degrees along the opposite radial direction, and 270 degrees
along -shaft. Thus 0/180 degrees lie in the rotor plane. This is an oriented
angle around the shaft-containing section, not azimuth around the shaft.
Planes need at least three distinct sampled angles. Unsupported planes are listed
in JSON `omitted_planes`; if none remain and no manual planes exist, directivity
is explicitly unavailable. Off-plane microphones are never projected onto a plane.
The 1.88 m radius in the schematic is not imposed on measured coordinates.

For existing APC cases, update `experimental_rotation_center_m` and
`planes` in the case-local configuration from the repository defaults. A
change to the experimental centre requires a fresh FW-H propagation with the
saved source surfaces. Plane-only changes can reuse a cache with matching
observer coordinates.

Add further entries to `planes`, each with `name`, `channels`, `zero_direction`
and `positive_direction`. Direction vectors must be nonzero and perpendicular.
Every plane must contain at least three distinct microphones, and all positions
must lie within `plane_tolerance_m` of that plane through the configured centre.
The default tolerance is 0.005 m for rounded coordinates. Invalid plane groups
produce an explicit unavailable result, not a projection of off-plane observers.

Angles are calculated from coordinates and sorted. Each figure has simulation
and experimental panels showing overall, broadband and tonal SPL on the same
radial scale, labelled with microphone IDs. Only sampled positions are shown.
Lines are broken across gaps above `maximum_connected_gap_deg` (default 60), and
are never closed or mirrored to invent missing angular coverage. Four microphone
positions give four samples, not a smooth resolved directivity pattern.

Radii are displayed. These are raw SPLs at the actual microphone distances; there
is no automatic distance correction. Prefer equal-radius microphone groups when
interpreting angular directivity. Tonal/broadband decomposition is the same
estimated method documented in [acoustic-tonal.md](acoustic-tonal.md).

## Propagation, caches and reports

When enabled, the solver evaluates **every experimental channel** in one batched
prediction, including channels not yet assigned to a polar plane. Matched repeats
must contain the same channel IDs and positions. All observers use one common,
fully arrived tail window (bounded by the latest propagation delay).

`report/acoustic-observers.npz` retains all pressure traces, mapped and original
coordinates, channel IDs, frame settings and the common window. The selected
`microphone_channel` still controls all existing spectrum/comparison figures.
Its trace is also written to the legacy `acoustic-pressure.npz` file.

Polar totals use the common positive-frequency coverage across every observer
and experimental recording. Experimental repeats are averaged in linear energy.
Total SPL is integrated before conversion to dB; tonal plus broadband power equals
overall power. Exact zero energy is null, rather than a fabricated finite level.

The matched microphone comparison uses every cached channel, independently
of the available polar planes. It shows overall SPL, estimated broadband SPL
and the first two BPF bands. Each band is centred at its expected harmonic and
extends two FFT bins either side, capped at one quarter of the fundamental BPF.
Bands outside the common spectral coverage are omitted. The measurements use
an energy mean and show the repeat range. Figure PNG/PDF and the per-channel
values and signed differences are exported as `acoustic-microphone-comparison.*`.

New outputs: `acoustic-directivity-plane-1.png` (and subsequent planes),
`acoustic-directivity.csv`, and `acoustic-directivity.json`. Figures are included
in the standalone acoustic PDF and full simulation report. The JSON records
settings, coordinates, angles, radii, coverage, levels and linear powers.

After entering coordinates and enabling, rerun acoustic propagation using the
saved CFD source fields. Old single-observer caches cannot produce uncomputed
observers. You do not need to rerun CFD if the saved acoustic surfaces suffice.
After that, change planes, tonal settings or the selected spectrum channel and
rebuild with `python -m tools.acoustic_comparison "path/to/case"`.
Changing the coordinate mapping requires another acoustic propagation run;
the cache check rejects mismatched settings. A failed directivity rebuild is
recorded separately and does not discard a valid selected-channel comparison.

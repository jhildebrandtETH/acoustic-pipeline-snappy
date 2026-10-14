# Multiple observers and polar SPL diagrams

Edit `directivity` in each case's `Parameters/acousticComparison.json`. Repository
settings seed new cases only. The feature is disabled by default: fill in the
coordinate mapping before setting `enabled` to `true`. The unknown experimental
centre is deliberately `null`; no fitted centre is silently assumed.

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

The starter plane contains M1, M8, M10 and M6. Their recorded y coordinates are
zero. `zero_direction: [1, 0, 0]` defines 0 degrees along experimental +x;
`positive_direction: [0, 0, 1]` defines 90 degrees along experimental +z.
Both directions refer to the experimental frame, relative to its propeller centre.

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

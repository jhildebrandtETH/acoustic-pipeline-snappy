# Directional sliver screen for native Fluent H5 imports

New ANSYS pipeline cases copy `Parameters/sliverScreen.json`. For H5 imports, `tools/ansys_mesh.py` calls the screen before conversion or flow solving. Existing cases need that settings file copied in or a standalone run. Legacy ASCII mesh inputs and the separate native Snappy meshing path are not screened by this integration.

The screen measures every rotor cell with signed pyramid volume moments and projected polygon-centroid weights. It reports effective tangential width and the smallest effective width tested along each face normal: `2 V / sum(abs(Sf dot direction))`. Arbitrarily oriented thin cells can therefore be found even if Cartesian bounding boxes, skewness or cell volume appear acceptable.

`report.json` lists the worst 100 flags; `flagged-cells.csv` lists every flag with native one-based cell ID, coordinates, volume, widths and wall adjacency. Coordinates in the report are mm after applying the native-to-metre scale. The current pipeline frame is rotation about Y through the origin.

The tangential rotation-only Co proxy uses rotation speed, radius, requested target time step and directional width. It is a ranking proxy, not a fluid-relative Courant prediction or bound: near-wall fluid velocity and motion flux can cancel strongly. The pipeline obtains RPM from the case's numeric `omega_val`; standalone runs use the JSON RPM.

Intended thin wall-normal inflation alone does not flag a boundary-adjacent rotor cell. Tangential thinness can still flag it. Non-wall cells also have a minimum face-normal directional-width threshold. Geometry thresholds are configurable and should be calibrated for the intended wall-function or wall-resolved mesh family.

Modes in `sliverScreen.json`:

- `warn` (default): write the diagnostics and continue with the independent standard checkMesh checks.
- `reject`: write diagnostics and stop before conversion/flow if any threshold flags remain. This explicit gate is independent of `allow_bad_mesh`.
- `off`: disable this screen.

Example from the ANSYS pipeline directory, with the pipeline Python environment active:

```bash
python tools/mesh_sliver_screen.py /path/candidate.msh.h5 --output /path/sliver-check --config Parameters/sliverScreen.json
```

The detector does not alter meshes or guarantee a usable time step. It cannot prevent Fluent creating slivers; the rejection gate can prevent accepting them into a flow run. Remeshing, cell unions or internal-node repairs remain separate operations and must retain independent geometry and checkMesh verification.

Tests include a rotated thin cuboid with analytical volume/width, intended thin wall-normal cells, invalid thresholds and diagnostic writing before rejection.

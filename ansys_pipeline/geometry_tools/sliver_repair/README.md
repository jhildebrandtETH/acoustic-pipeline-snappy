# Iterative repair of small rotor transition cells

This is a mesh-generation utility, not a flow solver or a new simulation default. It reads a native Fluent `.msh.h5`, scans the rotor, adjusts internal nodes in small cavities, rescans, and writes a distinct candidate. It never modifies the input mesh or launches CFD.

Requirements: Python, NumPy, SciPy and h5py. The mesh must use millimetres and the pipeline frame (rotation about +Y at the origin), with `propeller_blade` walls and the configured rotor cell-zone name. Rotor-interface and physical boundary vertices stay fixed. The present sanity check supports propeller radii between 10 and 1000 mm.

```bash
python repair.py original.msh.h5 new-output/candidate.msh.h5 --settings settings.json
```

Use a fresh output directory. `--scan-only` writes a scan without changing a mesh. `--start-from checkpoint.msh.h5` continues a topology-identical checkpoint while retaining the original input's displacement bounds and reference wall distances. Connectivity and fixed coordinates are checked. `--require-clear` returns exit code 3 when any screened non-wall flags remain. A `STOP` file in the output directory requests a checkpoint and stop at the next safe iteration boundary.

The scan measures `2V/sum(abs(Sf dot direction))` tangentially and along every face normal. Radius, RPM, target timestep and a configurable margin define target widths. This is a geometry screen: neither the rotation-only proxy nor a clean scan establishes actual fluid-relative Courant numbers. Intentionally thin wall-normal layers are not automatically treated as off-wall slivers.

Cells near the outer leading-edge transition receive priority. This family is detected from nearest blade-face positions and normals on both blades, not a hard-coded tip coordinate. Other flagged non-wall rotor cells are included. Small neighbouring risk cells can be repaired jointly; optional vertex-connected clusters are useful for stubborn cavities.

Each proposed repair checks signed volumes, owner/neighbour face pyramids, orthogonality, local OpenFOAM-formula skewness, neighbouring widths, and changes in wall-to-cell-centre distance. The current settings limit the latter to 10% of the original input. `max_node_motion_mm` is an absolute per-coordinate bound relative to the original input; the actual vector movement is recorded separately. The input's existing bad skewness faces are not silently classified as repaired. Independent standard checkMesh remains mandatory.

Safeguards and limits:

- Failed local attempts are rolled back. Source geometry and interfaces are fixed.
- The neighbour guard prevents reducing neighbours below their original narrow width or the configured target, apart from numerical acceptance tolerance.
- Every pass rescans the mesh, including newly flagged neighbours.
- Achieved widths become protected floors for subsequent repairs, preventing a repaired cell from giving its gained width back to a neighbour.
- Pass, attempt, displacement and evaluation budgets prevent unbounded iteration.
- Blocked cells remain in `latest-scan.json`; do not label an output sliver-free unless the scan says so.
- Some geometries cannot meet all width targets with fixed boundaries and layer-distance constraints. These need remeshing or changed mesh controls, rather than forced node movement.

Outputs include `summary.json`, `initial-scan.json`, `latest-scan.json`, `repairs.jsonl`, `node-changes.npz`, settings and the mesh checkpoint. `stop_reason` and `independent_checkMesh_required` distinguish a bounded experimental candidate from a verified mesh. This tool does not certify yPlus, thrust, stability or achieved timestep.

Nine analytic and workflow tests cover rotated slivers, polygon geometry, both blade sides, radial scaling, fixed-node protection, skipped protected targets, checkpoint connectivity fingerprints, invalid settings and retention of achieved widths. Run `python test_repair.py`.

For the grapes-derived candidate, the archived first, second and polish runner versions/settings record the actual staged experiment. The reusable version has additional checkpoint checks, strict neighbour floors, achieved-width locking and configurable vertex-connected clusters. Keep those records with the generated candidate when reproducing that exact experiment.

# AMI dimension study

`--ami_dimension_study` (alias `--ami-dimension-study`) creates one order with
16 independent rotating-interface cases for one STL and one RPM. The normal
pipeline uses its existing automatic geometry scaling when the flag is absent.
The study uses the existing local or Slurm scheduler, preprocessing, meshing,
solver, acoustics, and reporting.

## Geometry matrix

**D** is the propeller span measured from the scaled STL. Each diameter is paired
with all four axial configurations. The table shows the caps for a nominal
10-inch propeller and the default `rotorHalfLength` of 25 mm.

| Index | AMI diameter / D | Axial interpolation fraction | Upstream cap y/D | Downstream cap y/D |
|---|---:|---:|---:|---:|
| 1 | 1.100 | 0 | +0.0984 | −0.0984 |
| 2 | 1.267 | 1/3 | +0.1656 | −0.2656 |
| 3 | 1.433 | 2/3 | +0.2328 | −0.4328 |
| 4 | 1.600 | 1 | +0.3000 | −0.6000 |

The first axial configuration uses the configured symmetric caps in metres.
The last uses the actual upstream and downstream ends of the generated
`outerCylinder`. Each cap is interpolated independently between these endpoints.
The cylinder stays coaxial with the propeller and its centre shifts downstream.
The normal default AMI diameter of 1.20D falls between study values.

## Common settings

| Setting | Study value |
|---|---|
| Domain bounds | Original 2.5D reference sizing, with the configured margins and inlet fraction |
| Background cells | `(18 52 18)` in x, y, z for each case |
| Mesh refinement sphere | Diameter 2.1D, generated as `permeableSurface.stl` for the meshing templates |
| Permeable FW–H sphere | Diameter 2.1D, matching the mesh refinement sphere and generated separately as `permeableSamplingSurface.stl` |
| `innerCylinder` | Radius 0.75D, caps at +0.175D and −0.475D, volume refinement level 2 |
| `outerCylinder` | Radius 1.00D, caps at +0.300D and −0.600D, existing refinement setting |

The fixed cylinders and box continue to use the original 2.5D reference.
The standard pipeline also defaults to 2.1D for both spheres and uses a
separate 2.5D sizing reference for the domain and fixed cylinders. The study
records these factors in its own order so its design is explicit.
Other mesh controls, including the rotating surface and volume refinement
levels and the blade refinements, come from the normal configuration. The
smaller sphere changes the refined portion of the domain while preserving the
outer boundaries. The study uses the existing mesh quality and non-conformal
interface coverage checks.

## Run and resume

Put one propeller STL under `STL/` in a dedicated study order directory. Run
the following from the pipeline repository on Linux or WSL.

```sh
python main.py --sim-dir output/ami-dimension-study \
  --ami_dimension_study --rpms 4000 --mode AMI \
  --turbulence kOmegaSST --wall-functions yes \
  --total-cores 16 --cores-per-case 8 --acoustic-surface permeable
```

Adjust the solver, end condition, and resource arguments to the study run.
`--mode AMI` is implied if omitted. The study uses `--field-init off` and
can also run with `--mesh-only` or `--aerodynamics-only`. For Slurm, add
`--slurm --slurm-config /shared/path/cluster.json` as for other orders.
Cases queue within the selected core budget.

```sh
python main.py --sim-dir output/ami-dimension-study --status
python main.py --sim-dir output/ami-dimension-study --resume
```

For Slurm orders, add `--slurm` to the resume command. The order records the
matrix and case-specific settings in `simulation_order.json`. Case names end
with `_ami_dim_r1_a1` through `_ami_dim_r4_a4`. The effective dimensions in
metres and the normalized cap coordinates are in each case's
`mesh/geometry.json`, alongside the mesh and acoustic sphere radii.
The block resolution and level-2 override are written into each case's copy
of the parameter dictionaries.

## Remove the study

To remove results, delete the dedicated order directory once its local or
Slurm jobs have finished and any required outputs have been archived.

To remove the mode from the code, remove `tools/ami_dimension_study.py`,
the study hooks in `tools/cli.py`, `main.py`, `tools/orders.py`,
`tools/openfoam.py`, `preprocessing.py`, and `tools/snappy_pipeline.py`,
and this documentation, its README link, and `tests/test_ami_dimension_study.py`.
The standard scheduler, Slurm workers, templates, and source parameter
dictionaries are shared with normal runs.

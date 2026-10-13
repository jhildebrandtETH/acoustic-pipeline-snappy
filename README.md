# Acoustic pipeline — snappy only

STL input → blockMesh → surfaceFeatures → snappyHexMesh (castellation, snapping and optional native layers) → rotating cell zone and NCC interface → OpenFOAM 13 `foamRun -solver incompressibleFluid` → reports and optional acoustics.

Native snappy layers are enabled by `addLayers true` in `Parameters/snappyHexMeshDict.cpp`. Set it to `false` to disable them. All layer controls are in that same file; there is no separate layer tool or CLI switch.

## Setup

Run under Linux or WSL with Docker available and the `microfluidica/openfoam:13` image installed. From the repository root:

```sh
conda env create -f of_pipeline_env.yml
conda activate of_pipeline_env
docker pull microfluidica/openfoam:13
```

The solver and meshing utilities all run in the same Foundation OpenFOAM 13 container. No host mesher installation is needed. The current acousticSolver source is included directly so this copy is self-contained.

## Start an order

Put one or more closed propeller STL files in a dedicated order directory under `STL/`. Shaft axis is Y through the origin; inputs use metres by default. Set `scale` in `Parameters/snappyGeometry.json` for other units. STL names identify cases; no external feature OBJ files are required. Geometry must be one consistently wound closed solid, centred on the configured shaft.

```sh
python main.py --sim-dir output/my-order --rpms 4000 5000 \
  --mode AMI --turbulence kOmegaSST --wall-functions yes \
  --total-cores 8 --cores-per-case 4 --aerodynamics-only --end-on rev 2
```

Use `--mesh-only` to stop after mesh checks and NCC coupling and produce a mesh report. Use `--live-output` for command output instead of the dashboard. For acoustic analysis, omit `--aerodynamics-only` and specify `--acoustic-surface impermeable`, or `--acoustic-surface permeable --acoustic-sphere-diameter 2`.

The scheduler preserves per-case core allocations, runs independent cases concurrently within the total budget, and records checkpoints in `simulation_order.json`. `--field-init on` chains ascending RPM cases using the previous solution. `--end-on` accepts `time SECONDS`, `rev REVOLUTIONS`, `convergence`, `force_convergence`, or `residual_convergence`.

```sh
python main.py --sim-dir output/my-order --resume
```

Resume uses stored settings and allocations. Native-snappy and previous unlayered-snappy orders can resume; other backend orders require a fresh order directory. Failed meshing restarts from a fresh case while preserving the previous case in a timestamped sibling. Completed mesh/solve checkpoints allow report recovery without repeating the solver. A file lock prevents simultaneous launchers from modifying one order.

## Templates and mesh settings

`CoreTemplates/` contains both wall treatments for kOmegaSST and kEpsilon in AMI and MRF, plus wall-resolved DES in AMI: nine combinations. `--wall-functions yes|no` is required for new orders. See [template details](CoreTemplates/README.md).

Wall treatment configures the turbulence boundary conditions; it does not generate layers or establish adequate wall resolution. An unlayered mesh may not satisfy the selected model's near-wall requirements. Check solved y+ and first-cell spacing. For resolved SST/DES, adjust `Parameters/wallResolvedDict` to the actual wall-to-cell-centre distance and viscosity.

- `Parameters/blockMeshDict.cpp`: background block resolution.
- `Parameters/snappyHexMeshDict.cpp`: surface/distance/volume refinement, snapping and quality limits.
- `Parameters/snappyGeometry.json`: STL scale, rotating cylinder, refinement sphere, relative domain margins, volume and coupling acceptance.
- `Parameters/controlDict.cpp`: time stepping and output.
- `Parameters/wallResolvedDict`: prescribed resolved-wall omega.
- `Parameters/meshReport.json`: wall patch and optional diagnostic thickness target.

Geometry preparation measures the STL span, creates the cylinder and refinement surfaces, and sizes the domain around the sphere. Inlet is +Y and the longer wake region is -Y. A refinement sphere exists even when acoustic sampling is disabled. Generated domain bounds and mesh location are stored in each case's Parameters directory; provenance is recorded in `mesh/geometry.json`.

Studies use `--study --study-file snappyHexMeshDict --study-parameter propellerSurfaceRefinementLevel --study-values '(3 3)...(4 4)'` with one STL and one RPM. JSON geometry settings also support studies. The `addLayers` switch can also be varied with a parameter study.

Serial cases mesh and solve serially. Parallel cases use their allocated MPI cores for snappy, reconstruct the root mesh for checks and coupling, and decompose the coupled mesh afresh before solving. Standard checkMesh acceptance is enforced before and after NCC; extended geometry/topology diagnostics are retained separately. `--allow-bad-mesh` explicitly overrides standard quality checks, but never missing patches, invalid volume or inadequate interface coverage.

## Reports and diagnostics

The pipeline retains the dashboard, case history, isolated preprocessing/report workers, settings provenance, aerodynamic plots, y+ analysis, visualization, acoustic processing, mesh quality, cell-volume and Courant diagnostics. First-cell thickness diagnostics measure actual wall-adjacent cells; they do not claim that an extruded layer stack exists.

Case outputs include `sim.foam`, `mesh/geometry.json`, `mesh/mesh-status.json`, `mesh/ncc-status.json`, command logs, and `report/`. Settings reports read the case's saved snappy and solver dictionaries, not current repository defaults. See `Parameters/courantDiagnosticsGuide.md` for offline diagnostics.

## Supplied snappy cases

For a non-rotating input case with its own system dictionaries and triSurface geometry:

```sh
python main.py --mesh-case /path/to/input-case --sim-dir output/simple-case --mesh-only
```

This serial workflow respects the supplied case�s layer settings, checks the mesh and creates a PDF report. Add `--mesh-native` to use a locally sourced OpenFOAM 13 installation.

## Validation

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests
```

The regression suite covers case preservation, scheduler/resume, solver completion, templates, geometry, native meshing orchestration, and report contents. Optional visualization checks need the corresponding rendering libraries. Integration with OpenFOAM requires the Docker image above.

## Custom OBJ feature refinement

Put explicit curves alongside the order's STL inputs:

```text
my-order/
  STL/10x7E.stl
  FEATURES/10x7E_le.obj
  FEATURES/10x7E_te.obj
  FEATURES/10x7E_tip.obj
  FEATURES/10x7E_hub.obj
  FEATURES/10x7E_other.obj
```

Each filename uses the STL stem plus a configured group suffix. The `other`
group supports the older tip/other naming. Files must contain OBJ `v` vertices
and `l` lines or polylines, in the same coordinate system and units as the STL.
Face-only exports are rejected. The STL scale is also applied to feature curves.
Missing groups are warned about and skipped, so STL-only orders still work.

Edit `Parameters/snappyFeatures.json` to select groups and independently set
`teLevel`, `teRefinementThickness`, `tipLevel`, and the corresponding keys for
each group. Thickness is the refinement distance from a curve **in metres**;
zero requests edge-intersection refinement only. Defaults reproduce the earlier
feature settings: level 6 and 0.0005 m. With the current level-6 propeller surface,
use a higher feature level (for example `teLevel: 7`) when you want finer cells
near that curve. Levels are absolute snappy refinement levels relative to the
background mesh; they are not added to the surface level. Other overlapping
refinements can already request equally fine or finer cells.

Set `enabled` to false to disable custom features, or remove selected names from
`groups`. Additional groups use `<group>Level` and `<group>RefinementThickness`.
No extra command-line switch is needed. A trailing-edge refinement study uses:

```sh
--study --study-file snappyFeatures.json --study-parameter teLevel --study-values '6...7...8'
```

Preparation snapshots the original OBJ bytes in each case's `FEATURES/` folder,
writes scaled `custom_<group>.eMesh` files to `constant/triSurface`, and includes
their entries in snappy's `features` list. These curves also participate in the
existing explicit feature snapping. `mesh/features.json`, geometry provenance,
and the PDF settings pages record active/missing groups, levels, distances and
source hashes. Start a fresh order to apply new controls reliably; already-meshed
resume checkpoints do not remesh simply because an input curve was edited.

The implementation uses OpenFOAM 13's native [feature-distance refinement](https://cpp.openfoam.org/v13/refinementFeatures_8C_source.html).
It requires no external mesher and works with native snappy layers enabled or disabled.

To run the optional real OpenFOAM comparison test:

```sh
OPENFOAM_FEATURE_TESTS=1 python -m unittest discover -s tests -p test_snappy_features.py
```


## Native snappy layers

Edit `Parameters/snappyHexMeshDict.cpp` before preparing a new case:

- `addLayers true` enables snappy layer addition; `false` disables it.
- `addLayersControls/layers/propeller/nSurfaceLayers` requests 3 layers initially.
- `expansionRatio` starts at 1.2; `finalLayerThickness` starts at 0.3.
- `relativeSizes true` makes thickness relative to the local cell outside the
  layer stack. Use `false` and suitable metre values for absolute thicknesses.
- `minThickness` and the smoothing/iteration controls are editable in the same block.

Only the propeller wall is selected by default. Native snappy may reduce or omit
layers locally to satisfy quality checks. The report's layer switch records the
request, not guaranteed coverage; inspect `log.snappyHexMesh` and the mesh.
Use exactly two thickness controls from expansion ratio, first-layer thickness,
final-layer thickness and total thickness. See the [OpenFOAM 13 layer controls](https://doc.cfd.direct/openfoam/user-guide-v13/snappyhexmesh).
Already prepared cases keep their copied parameters; edit the case copy or start
a fresh order. A completed mesh is not regenerated merely by resuming an order.

## Experimental acoustic agreement

The acoustic stage compares matching experimental geometry/RPM recordings with the simulated observer pressure. Configure the measurement folder, microphone channel and frequency bins in Parameters/acousticComparison.json. See [the acoustic comparison guide](Parameters/acoustic-comparison.md) for metrics, assumptions and regenerating reports without rerunning CFD.

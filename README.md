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

## Slurm case jobs (optional)

Without `--slurm`, the local Docker scheduler, meshing and dashboard behave as before.
With `--slurm`, submit your usual small coordinator batch script containing the
normal command with `--slurm --slurm-config /shared/path/cluster.json` added.
See [coordinator example](examples/slurm-coordinator.sh) and
[cluster settings](examples/slurm-config.json). Adapt all example paths, partition,
account and environment setup to your site before submitting:

```sh
sbatch examples/slurm-coordinator.sh
```

The coordinator submits a separate batch job for each ready case. Each job keeps
preprocessing, Docker snappy meshing, solving, reconstruction and postprocessing
together and releases its fixed allocation when it exits. `--cores-per-case`
specifies MPI ranks; each job requests one node, one task, and that many CPUs per
task. `mem` is the Slurm memory request and Docker memory cap per case; `time` is
the case job's wall limit. The order's
`--total-cores` remains an admission budget, not the coordinator allocation:
`--total-cores 32 --cores-per-case 8` allows at most four submitted/running case
jobs. Waiting jobs count toward this limit. Slurm decides when those jobs run.
The coordinator itself can request one CPU and small memory; size its wall time
for the entire order, including queue waits. It must be allowed to call `sbatch`,
`squeue`, `sacct` and `scontrol` from its compute node.

Slurm mode uses **the same Docker Foundation OpenFOAM 13 image as local mode**:
`microfluidica/openfoam:13`. Host-native OpenFOAM and MPI installations are not
required. Each case's Python worker runs once on its allocated node, drives the
normal preprocessing/mesh/solve/postprocessing state machine, and starts the
existing OpenFOAM container for meshing and solving. The simple `--mesh-case`
workflow remains local.

Minimal configuration, which can live inside the order directory alongside `STL/`:

```json
{
  "partition": "debug",
  "time": "2-00:00:00",
  "mem": "30G",
  "runtime": "docker",
  "docker_image": "microfluidica/openfoam:13",
  "setup": "",
  "poll_interval": 10
}
```

`time` and `mem` are required; `mem` must be positive and finite (a bare number
means MiB, or use K/M/G/T). `runtime` defaults to `docker`; `docker_image` defaults
to the image above. A custom image must provide the same Foundation 13 paths and
tools as that image. `python` defaults to the coordinator's active Python executable
and is saved as an absolute path. `setup` is an optional trusted shell snippet run
in each case script for Python/Docker environment setup. `partition`, `account`,
`qos` and `constraint` are optional. Scientific settings and cluster configuration
are saved in the order for resume; allocation/configuration changes require a new
order. Invalid settings are rejected before a new order is created.

Every case node must run a local Linux Docker daemon accessible to the worker's
user, with the image already installed (`docker pull microfluidica/openfoam:13`).
The coordinator does not need Docker; availability is checked on each case node.
Remote `DOCKER_HOST` connections are unsupported, since worker-lifetime monitoring
requires the container daemon and worker to share the node's `/proc` filesystem.

Meshing and solving retain the existing in-container `mpirun` commands and use
`allocated_cores` for both `numberOfSubdomains` and MPI rank count. One-core cases
run serially. The generated script uses **one** `srun --mpi=none` task with CPU
binding to launch Python; host MPI/PMIx is not involved. Do not wrap the coordinator
in a multi-task `srun`. Docker receives the worker's assigned CPU mask, a matching
CPU quota, and explicit memory/swap limits. Worker preflight rejects a CPU affinity
mask that does not match the saved core count rather than silently selecting
arbitrary node CPUs. The cluster must support `srun --cpu-bind=threads` with a
working task/affinity setup.

When Docker uses the `cgroupfs` driver and the worker exposes an identifiable
Slurm `job_<id>` cgroup, the container also joins that parent, sharing the job's
aggregate resource limits/accounting. With other layouts/drivers, explicit Docker
caps still apply, but Slurm accounting and combined Python/container memory
enforcement depend on site integration; Docker's daemon-managed cgroup is not
automatically part of the Slurm job. Verify this on your server. See
[Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)
and [Slurm CPU binding](https://slurm.schedmd.com/cpu_management.html).

Containers carry labels identifying their Slurm job and attempt. Worker exit,
including SIGTERM, removes its attempt's containers. For SIGKILL, a small PID-1
watchdog inside the container monitors a read-only bind of `/proc/<worker-pid>`;
it exits when the worker disappears or becomes a zombie, stopping remaining
container processes within about one second. This does not require native OpenFOAM,
a privileged container, or a Docker restart policy. Normal solver cleanup still
stops and removes the container as before. After SIGKILL an exited container may
remain on that node for inspection, but it no longer runs a solver.

Repository, order, STL/FEATURES inputs and Python environment must be accessible
at identical absolute paths on coordinator and case nodes.
Use shared storage with working cross-node POSIX `flock` locks and atomic rename;
node-local temporary directories and filesystems without these guarantees are
unsupported. `SimulationOrderStore` locks a separate state-lock file and reloads
JSON inside every transaction. Workers write stage/progress records to separate
per-attempt files under `.slurm/`, without competing over a shared runtime JSON.
The normal `.pipeline.lock` permits one coordinator, while per-attempt worker
locks prevent overlapping copies of one case job. This implementation targets
one Slurm cluster, with accounting enabled, and one node per case.

### Viewing status and resuming

The usual case table appears in coordinator batch output (without terminal escape
codes). From another terminal, watch the same saved stages and progress without
starting a coordinator or acquiring its launcher lock:

```sh
python main.py --sim-dir /shared/orders/propeller --watch
# One snapshot:
python main.py --sim-dir /shared/orders/propeller --status
```

The queued table uses `slurm_queue` and shows job ID, Slurm state and queue reason;
initialization dependencies retain their waiting state. Workers' stage/progress
updates remain visible even if the coordinator stops. Queue/accounting state is
refreshed by the coordinator, so it can be stale while that coordinator is down.
Ctrl-C exits the viewer. Worker command output is retained in
`.slurm/<attempt>-%j.log`; normal per-case logs and error tracebacks remain.
`--live-output` continues to control local runs; Slurm workers always log command
output while the coordinator retains its table.

To restart after coordinator timeout/interruption or to retry failed cases, put
this command in your usual small batch script and submit it:

```sh
python main.py --sim-dir /shared/orders/propeller --slurm --resume
```

Orders/config files saved by the initial native-only Slurm implementation are
automatically migrated to Docker on resume **after existing job identities have
been reconciled and all old jobs have terminated**. The unused `openfoam_bashrc`
and host `mpi_launcher` fields are removed; the old configuration is kept in
`slurm_config_history`. Core allocations, checkpoints, job IDs/history, and inputs
are preserved. No manual order-JSON editing or new study directory is needed for
orders that failed because `/opt/openfoam13/etc/bashrc` was absent on the host.
If an old job remains active or unresolved, migration stops without resubmission.

The coordinator reconciles existing jobs before retrying failures. Pending/running
jobs are reattached, completed cases stay complete, and failed cases resume from
the existing validated checkpoints (including report-only recovery). Dependencies
are submitted only after the source case and its Slurm job both finish successfully;
failed sources block descendants. Cancellation, timeout, OOM, node failure and
nonzero exit are recorded as failures even if no Python exception was written.
A zero-exit Slurm job missing the final pipeline checkpoint is also a failure.
Failures are retried only on explicit `--resume`, not in a loop.

Before `sbatch`, the coordinator saves a unique submission intent; it submits
held, persists the returned ID, then releases the job. After a crash between
submission and saving the ID, it finds the held job by its unique name in
`squeue`/`sacct`. An unknown ID or accounting delay never triggers resubmission.
If an intent cannot be resolved, the coordinator stops with its name and leaves
the intent intact. Inspect queue/accounting records and submission errors before
manual repair; **do not delete job identities while a matching job may exist**.
After a definitively rejected submission with no matching job, an operator may
remove only that case's unresolved `slurm_job` entry from `simulation_order.json`
while no coordinator is running, then resume. Query failures retain all identities.
Automatic Slurm requeue is disabled. Cancelling the coordinator does not cancel
its case jobs: they can finish, and a replacement coordinator can reattach.

### Cluster validation still required

Tests simulate Slurm responses and process failures locally; they do not run a
Slurm daemon or CFD workload. On your server, first run a small mesh-only order,
then a short solve and a two-RPM initialization chain. Verify MPI rank placement,
decomposition, NCC, reconstruction, memory/CPU enforcement, compute-node Docker access,
report dependencies and shared filesystem locking across different nodes. Exercise
queued jobs, `scancel`, wall timeout, OOM/node failure, coordinator restart during
submission, accounting delay, and report-only resume. Confirm allocations are
released when each case job ends, and that containers exit after cancellation,
timeout or a killed worker. Check cgroup parent integration/accounting on your
site's Docker/Slurm combination. Local mode remains covered by the existing meshing,
solver and scheduler regression tests.

Implementation verification in WSL includes simulated submissions, lost responses,
restart/reattachment, dependency and worker failures, checkpoint recovery, progress
viewing, concurrent process writes, migration of failed native orders, Docker
resource/cleanup options and unchanged local dispatch. Real Docker smoke tests
also verified the installed OpenFOAM image, two in-container MPI ranks, actual
container CPU/memory settings and container exit after worker SIGKILL:

```sh
SLURM_DOCKER_TESTS=1 python -m unittest discover -s tests -p 'test_slurm*.py'
```

No Slurm server or complete rotor mesh/solve was exercised locally. The previously
observed feature-refinement test failure (`test_geometry_study_and_report_include_features`,
`KeyError: 'te'`) also reproduced on the unchanged baseline.

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

The acoustic stage compares matching experimental geometry/RPM recordings with the simulated observer pressure. Configure the measurement folder, microphone channel and frequency bins in each case's `Parameters/acousticComparison.json`. Repository parameters seed new cases only; acoustic runs and report rebuilds read the copied case config without a source fallback. See [the acoustic comparison guide](Parameters/acoustic-comparison.md) for metrics, assumptions and regenerating reports without rerunning CFD.

Future flow atlases include uncropped XY/YZ sections through the shaft, covering the full inlet and wake domain, plus labelled wake and rotor detail crops. The latest full-domain quantity/plane views are retained even if they exceed the report chapter page budget. Vortex extraction also uses the full domain.

Acoustic report spectra mark BPF and its harmonics using `blade_count` in `Parameters/acousticComparison.json` (default 2 for older cases). Markers are expected frequencies, not detected peaks. Comparisons also show unweighted band-limited total SPL, FFT spacing and segment duration; the existing experimental repeat envelope is retained. FFT spacing is not the effective Hann/Welch spectral resolution.

Flow-field slice and blade-section images use ParaView Cell to Point Data interpolation for smooth colors. Interpolation is applied only for rendering and can reduce local extrema; saved simulation fields are unchanged. Courant-number maps and blade-wall diagnostics retain native cell values.

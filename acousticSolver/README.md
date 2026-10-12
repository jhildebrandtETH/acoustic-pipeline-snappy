# FoamAcoustics

`foamacoustics` predicts Farassat 1A observer pressure from OpenFOAM surface
data or rotating-blade NumPy archives and provides small, explicit utilities
for inspecting the result.

```python
import torch

from foamacoustics import F1ASolver, OpenFOAMSurfaceSeries
from foamacoustics.plotting import plot_spl_spectrum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solver = F1ASolver.from_openfoam_vtk(
    "postProcessing/surface",
    rpm=7000.0,
    device=device,
    cache_dir=".cache/surface",
)
observer_m = torch.tensor([1.0, 0.0, 0.0], device=device)
result = solver.predict(observer_m)
plot_spl_spectrum(result, rotations=1.0, blade_count=2)
```

For repeated predictions, build the binary cache once before calling the solver:

```python
series = OpenFOAMSurfaceSeries(
    "postProcessing/surface",
    cache_dir=".cache/surface",
)
series.build_cache(workers=4)
solver = F1ASolver(series, rpm=7000.0, device=device)
```

The first build still reads every ASCII VTK file. Later series instances validate
and memory-map `pressure.bin` and `velocity.bin`, avoiding repeated VTK parsing.

Multiple observers are evaluated during the same pass through the VTK frames:

```python
result = solver.predict(torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    device=device,
))

# Multi-observer pressure arrays have shape (observers, times).
first_observer = result.for_observer(0)
```

## NPZ blade surfaces

A compact archive stores one reference mesh and a pressure history. The
geometry table must contain `x`, `y`, `z`, `normal_x`, `normal_y`, `normal_z`,
and `area`; each pressure array has shape `(times, faces)`.

```python
import numpy as np
import torch

from foamacoustics import F1ASolver, NPZSurfaceSeries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.savez(
    "blade_surface.npz",
    surface_geometry_ref=face_table,
    surface_geometry_columns=np.array([
        "x", "y", "z", "normal_x", "normal_y", "normal_z", "area",
    ]),
    time=time_s,
    p_surface_pred=predicted_pressure_pa,
    p_surface_true=reference_pressure_pa,
    pressure_units=np.array("pa"),
    rho=np.array(1.225),
)

rotation_center_m = [0.0, 0.0, 0.0]
omega_rad_s = [0.0, 0.0, 7000.0 * 2.0 * np.pi / 60.0]

# Build the reader explicitly when its geometry or frames are also useful.
series = NPZSurfaceSeries(
    "blade_surface.npz",
    pressure_field="p_surface_pred",  # or "p_surface_true"
)
solver = F1ASolver(
    series,
    rpm=7000.0,
    permeable=False,
    moving_surface=True,
    rotation_center_m=rotation_center_m,
    omega_rad_s=omega_rad_s,
    device=device,
)

# Or construct the same reader and solver in one call.
solver = F1ASolver.from_npz(
    "blade_surface.npz",
    pressure_field="p_surface_true",
    permeable=False,
    moving_surface=True,
    rotation_center_m=rotation_center_m,
    omega_rad_s=omega_rad_s,
    device=device,
)
result = solver.predict(torch.tensor([1.0, 0.0, 0.0], device=device))
```

The source formulation is selected by the solver, independently of whether the
data came from NPZ or OpenFOAM:

- `permeable=True` uses the fluid velocity field and requires one in every
  frame. `permeable=False` uses the impermeable formulation, so pressure-only
  surface data is sufficient. Fluid velocity must be expressed in the inertial
  frame. `from_openfoam_vtk(..., permeable=False)` therefore does not require
  or read a `U` field.
- `moving_surface=True` enables moving-surface geometry and retarded-time
  handling. `moving_surface=False` holds the reference mesh fixed.

For rigid rotation, pass `rotation_center_m` and `omega_rad_s` to the solver.
The solver can instead fall back to compatible archive metadata when those
arguments are omitted. Metadata `omega` may be a three-vector or a scalar
paired with `rotation_axis`; if `rpm` is not stored, it is inferred from the
magnitude of `omega`. Explicit solver motion arguments override archive
metadata.

Coordinates are in metres, areas in m², times in seconds, and `omega_rad_s`
uses the right-hand rule. Pressure may be pascals or kinematic pressure; the
latter is multiplied by `rho`, then `pressure_reference_pa` is subtracted.
For compact archives, `pressure_units` and `rho` can be inferred from metadata;
explicit reader arguments override them. Normals are normalized as supplied,
so use `flip_normals=True` if the archive points inward. With rigid rotation
enabled, the solver rotates the reference face centers and normals and computes
face velocity from `omega × (position - rotation center)`.

Moving surfaces use dynamic, nonuniform retarded-time interpolation regardless
of their data source. Any fixed/moving and permeable/impermeable combination is
available when the required fields or motion parameters are present. Legacy
`data`/`columns`/`times` archives have no reliable unit metadata, so they
require explicit `pressure_units` and will commonly also need
`pressure_reference_pa`.

For the legacy `p_surface_10x7E_6000RPM_AMI.npz` archive, the matching setup is:

```python
legacy_series = NPZSurfaceSeries(
    "p_surface_10x7E_6000RPM_AMI.npz",
    pressure_units="kinematic",
    density_kg_m3=1.225,
    pressure_reference_pa=101325.0 * 1.225,
    flip_normals=True,
)
legacy_solver = F1ASolver(
    legacy_series,
    rpm=6000.0,
    permeable=False,
    moving_surface=True,
    rotation_center_m=[0.0, 0.0, 0.0],
    omega_rad_s=[0.0, 628.3185307179587, 0.0],
)
```

## Sampling an OpenFOAM volume field

The F1A solver can use a fixed, closed permeable surface sampled directly from
the OpenFOAM volume fields. `SphereSurface` and `CylinderSurface` generate a
watertight triangular surface. `interpolate_openfoam_fields` takes the case
folder and shape, interpolates `p` and `U`, and returns the same
`OpenFOAMSurfaceSeries` accepted by the existing solver:

```python
import torch

from foamacoustics import (
    CylinderSurface,
    F1ASolver,
    interpolate_openfoam_fields,
)

series = interpolate_openfoam_fields(
    "path/to/propellerCase",
    CylinderSurface(
        radius_m=0.15,
        length_m=0.60,
        center_m=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        axial_divisions=32,
        azimuthal_divisions=96,
        cap_radial_divisions=8,
    ),
    surface_name="permeableCylinder",
    time_selection="0.10:0.20",
    pressure_units="kinematic",
    density_kg_m3=1.225,
    assume_stationary_mesh=True,
    workers=4,
    progress=True,  # Set to False to hide the per-time-step progress bar.
)

series.build_cache(workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solver = F1ASolver(
    series,
    rpm=7000.0,
    response_delay_s=None,
    device=device,
)
result = solver.predict(torch.tensor([1.0, 0.0, 0.0], device=device))
```

`response_delay_s=None` keeps the physical source-to-observer propagation
delay, which is required when time signatures from different surfaces must
share one arrival-time reference. The source record must extend beyond the
nearest-face propagation delay before any physical arrival can be observed,
and beyond the farthest-face delay plus the desired analysis window for a
fully arrived spectrum. For an interpolation or solver preview only, a small
explicit normalized delay such as `response_delay_s=1e-5` can bring the first
arrival onto a short record; such normalized traces are not physically aligned
between different surfaces.

Use `SphereSurface(radius_m=..., center_m=...)` for a sphere. The cylinder is
closed with end caps, its `length_m` is the full cap-to-cap length, and `axis`
may point in any direction. The entire surface must lie inside the fluid mesh.
There is one interpolation implementation: VTK reads the OpenFOAM volume mesh
and locates the cell containing each surface vertex. FoamAcoustics then
reconstructs OpenFOAM-style face centres and face-pyramid cell-volume
centroids, inverse-distance cell-to-vertex values, and cell-centre/face-vertex
tetrahedral weights. Candidate tetrahedra use a cell-volume-relative tolerance
and the same Euclidean nearest-tetrahedron fallback as OpenFOAM `cellPoint`.
The result is reduced to one value per triangular F1A panel by the existing
surface-series reader. It is not expected to be bitwise identical to OpenFOAM:
VTK still controls cell location and face topology, and physical-boundary and
non-conformal point constraints are not reconstructed by this interior-field
sampler.

The input must contain an OpenFOAM-readable base mesh and field time
directories, not only copied field files. At a minimum it needs this layout
(additional moving-mesh data can remain in the appropriate time directories):

```text
propellerCase/
|-- constant/
|   `-- polyMesh/
|       |-- points
|       |-- faces
|       |-- owner
|       |-- neighbour
|       `-- boundary
`-- <time>/
    |-- p
    `-- U
```

No OpenFOAM installation, subprocess, `.foam` marker, sampling dictionary, or
`system/controlDict` is needed. The VTK reader uses the numeric time
directories directly; when `<time>/uniform/time` (plain or gzip-compressed) is
present, FoamAcoustics preserves its higher-precision physical value on the
returned acoustic time axis. A flattened `polyMesh/` folder is not sufficient:
the base topology must remain under `constant/polyMesh/`. Moving-mesh
`polyMesh/points` files may remain in their normal time directories and are
re-read at every selected time.

For the usual incompressible OpenFOAM kinematic pressure, keep
`pressure_units="kinematic"` and provide the simulation density with
`density_kg_m3`; the reader multiplies `p` by that density to obtain pascals.
Use `pressure_units="pa"` if the stored field is already in pascals.
`pressure_reference_pa` may be supplied when a constant pressure reference
must be removed after conversion.

The function is a Python API intended for scripts and notebook cells; there is
no CLI or backend selector. PyVista and VTK are package dependencies and all
interpolation runs in the Python process. Every surface definition and source
signature receives a source-signature-addressed sampled-VTK directory and
binary cache, so rerunning an unchanged request reuses its result while several
radii and shapes can coexist.

For cases with static topology, interpolation geometry is prepared once and
reused across the selected times. Only the cells and mesh points that can
influence the sampled surface are retained in this plan. Their coordinates are
checked at every time, so a rotating or deforming zone elsewhere does not
disable the fast path; if motion reaches the surface's interpolation stencil,
FoamAcoustics automatically falls back to the full per-time reconstruction.
When that stencil is known to remain stationary, set
`assume_stationary_mesh=True` to skip these coordinate checks and read only the
contributing pressure and velocity entries after the first time step. This also
pins the first-step containing-cell choice for vertices exactly on cell
interfaces, avoiding time-dependent VTK locator jitter. Leave the option false
when reproducing those per-time VTK interface choices bit-for-bit is required.

For a meaningful surface comparison, use the same source times, observer
positions, atmosphere, and FFT window for every run. Give the surfaces
comparable panel sizes, ensure each one encloses the propeller and relevant flow
sources, and choose a common fully-arrived observer-time interval. A uniformly
sampled window containing an integer number of shaft revolutions reduces
spectral leakage. The short raw-field subset in `field data` is sufficient for
interpolation checks, but it is shorter than the physical propagation delay to
the example observer and therefore cannot produce a physical-delay observer
trace or a resolved propeller spectrum. New surface comparisons need the full
volume-field time record.

Acoustic propagation currently assumes constant density and zero mean-flow
convection. The supported fixed/moving and permeable/impermeable source
combinations, including their required fields, are described above.

## Plotting workflows

Reusable Matplotlib helpers live in the `foamacoustics.plotting` package:

- `pressure` plots solver histories and compares cases by observer.
- `spectra` computes or compares SPL spectra and multi-observer results.
- `directivity` draws OASPL and multi-metric polar directivity plots.
- `diagnostics` contains the surface, blade, harmonic, contribution, and
  error-attribution plots used by the example notebooks.

The package namespace re-exports the public helpers, so common workflows need
only one import:

```python
from foamacoustics.plotting import (
    plot_result_pressure_comparison,
    plot_spectrum_comparison,
)
```

Plotting functions return their Matplotlib axes or a small typed result object,
which makes further styling and saving explicit. Their typed signatures and
docstrings also expose parameter guidance in IDE tooltips. The repository
notebooks call these helpers instead of carrying separate plotting
implementations, and are stored without generated output so reruns remain
small and reviewable.

## Layout

- `foamacoustics.core`: the F1A solver and result model.
- `foamacoustics.io`: OpenFOAM legacy-VTK surface reading.
- `foamacoustics.sampling`: sphere/cylinder geometry and native VTK
  interpolation of OpenFOAM fields.
- `foamacoustics.utils`: pressure/SPL conversion and validation helpers.
- `foamacoustics.plotting`: typed Matplotlib presentation and diagnostics.

Pressure inputs and outputs are in pascals. SPL utilities use a reference
pressure of 20 µPa by default.

Install notebook support with `pip install -e ".[notebook]"`. Development
checks are available through `pip install -e ".[dev]"`; run `pytest`,
`ruff check src tests plot_*.py *.ipynb`,
`ruff format --check src tests plot_*.py *.ipynb`, and `mypy` before committing
changes.

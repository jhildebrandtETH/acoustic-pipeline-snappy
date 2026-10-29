# Native Snappy defaults with the stable pressure inlet

Updated 28 October 2026 at the user's request. This applies to all nine supported
ANSYS AMI/MRF turbulence/wall-treatment templates. The ANSYS import path and the
normal native Snappy pipeline are not changed.

| Setting | New ANSYS default |
|---|---|
| Pressure / pressure correction | Native Snappy GAMG, DICGaussSeidel, cacheAgglomeration no, maxIter 50 |
| Pressure tolerances, PIMPLE and convection/diffusion schemes | Same as the corresponding native Snappy template |
| Wall distance | meshWave; Poisson yPsi entries removed |
| SST initial/background omega | 0.0495 1/s, also used by inlet and outlet backflow through internalField |
| Outer-wall fixed omega | 1000 1/s, matching native Snappy |
| Pressure inlet | Existing totalPressure retained, rather than restoring the RANS zeroGradient inlet |
| Other field boundaries and model choices | Corresponding native Snappy settings |

For the eight RANS templates, totalPressure keeps p0/value=101325 with the same
kinematic-pressure reference as the outlet and initial field. DES retains its
existing native combined outer-boundary totalPressure configuration and zero
reference. Wall-function omegaWallFunction and wall-resolved fixedValue treatments
remain distinct. Automatic mesh-based wall omega preparation for resolved cases
is retained; this change concerns background omega, not that mesh-dependent wall
calculation.

The following newer features are deliberately retained: maxDeltaT tied to
1/65536s acoustic sampling; adaptive timesteps constrained by maxCo; acoustic
p/U output every solver step with uniform-grid interpolation during postprocessing;
full-field writeControl runTime; enabled yPlus field output; the blade areaAverage
monitor and area-weighted reporting; and imported blade/hub patch handling.
No runtime image/version, meshing controls, rank count, force monitoring or
postprocessing defaults were changed.

## Evidence and scope

The tested zeroGradient pressure inlet diverged with GAMG+meshWave,
GAMG+Poisson and PCG+Poisson. The totalPressure PCG+Poisson comparison completed
0.1s; the downloaded GAMG+meshWave totalPressure log was stable through0.0219s.
These observations support retaining totalPressure while restoring native solver
settings. They do not establish every possible combination or long-run stability.
Earlier platform/runtime crashes are separate observations; the restored defaults
are not a claim that such failures are impossible.

## Applying the defaults

Create a **new order** after updating this repository. Orders capture their own
preparation templates and parameters; resuming an existing order retains its old
snapshot. Generated cases and active runs retain their own dictionaries. Neither
was edited by this update. Initialization from an earlier case may also replace
initial internal fields by design; use a clean start for controlled comparisons.

For a server running its own pipeline checkout, transfer the updated repository
files before creating that new order. Uploading only a mesh does not change an
older server checkout's template defaults. No flow solver was launched during
this update; regression checks cover all nine template presets and retained
sampling/reporting controls.

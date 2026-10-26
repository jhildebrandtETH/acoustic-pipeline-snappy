# Temporary runtime crash workarounds

## Y-plus restored — 26 October 2026

The yPlus function is enabled again in all nine ANSYS templates at the user request. It calculates every timestep and writes its field at normal write times. A separate propellerYPlusMean surfaceFieldValue function records the propeller area-average every timestep, preserving its time history despite purgeWrite. This supersedes the disabled-yPlus workaround below. It does not prove the earlier wall-distance crash is fixed; the newly enabled diagnostic requires validation in subsequent runs. Existing cases and order snapshots are unchanged.

Applied to the normal ANSYS pipeline templates on 23 October 2026. These settings apply to all nine AMI/MRF turbulence/wall-treatment templates under `CoreTemplates`. They are diagnostic workarounds motivated by crashes in the roundedV2 rotating-mesh case; they do not establish mesh validity, physical accuracy, acoustic accuracy, or long-run stability.

## Changes and reasons

| Location in each template | Current workaround | Reason |
|---|---|---|
| `system/controlDict`, `functions/yPlus` | `enabled false;` | A segmentation fault occurred inside the built-in yPlus calculation's `nearWallDist` path. That diagnostic performs a separate wave-based distance calculation even when the turbulence wall-distance method is Poisson. |
| `system/fvSolution`, `solvers/"pcorr.*"` | `solver PCG; preconditioner DIC; maxIter 1000;` | A separate crash occurred in GAMG agglomeration during a pressure solve. The pressure matrix in the tested case was symmetric. PCG/DIC bypasses pressure multigrid agglomeration. |

`p` inherits the pressure-correction solver using `$pcorr;`; `pFinal` inherits `p` using `$p;`. Thus all pressure solves use PCG/DIC. Their original tolerances and relative tolerances remain unchanged. The iteration cap was raised from 50 to 1000 because Krylov iterations are not equivalent in cost or convergence rate to GAMG cycles. PCG/DIC may require more iterations and increase runtime.

The attempted PBiCGStab/DILU combination is **not** the default: DILU was rejected for the symmetric pressure matrix. PCG/DIC was subsequently exercised in the test case. A later crash was in yPlus, and after disabling that diagnostic the run passed the previous crash location but later developed an unphysical tip disturbance. The workarounds therefore do not solve every numerical or mesh issue.

## Settings retained separately

- Poisson wall distance and its complete `yPsi` / `yPsiFinal` settings remain enabled in all nine templates. The Poisson `yPsi` equation still uses GAMG; this change targets pressure GAMG only.
- Initial/background SST omega remains 1000 in the four SST templates. Its physical suitability still depends on the intended inlet/background turbulence conditions.
- Mesh import accepts the two tip labels and merges those faces into the physical `propeller` wall. This is independent of the runtime workarounds.
- The Courant limit, time-step settings, convection schemes, PIMPLE correction counts and acoustic sampling settings are unchanged by this update.

## Effect on y+ reporting

The built-in yPlus diagnostic no longer calculates, logs or writes new y+ results in new cases. Any initial zero `yPlus` field or old retained y+ output is not a measurement of the current run. A dependent yPlus mean monitor must remain disabled unless a working yPlus producer is restored; the normal templates do not add that mean monitor.

Wall shear reporting remains configured. A replacement y+ diagnostic or separate post-processing method requires its own validation; none is introduced by this workaround.

## Revert only the pressure workaround

In the relevant template's `system/fvSolution`, replace the complete `"pcorr.*"` block with the previous settings:

```foam
    "pcorr.*"
    {
        solver          GAMG;
        tolerance       1e-2;
        relTol          0;
        smoother        DICGaussSeidel;
        cacheAgglomeration no;
        maxIter         50;
    }
```

Do not leave `preconditioner DIC;` in that reverted block. Keep the existing `p` and `pFinal` inheritance/tolerances. This restores the previous pressure configuration without removing Poisson wall distance, changing SST initialization, or undoing tip-label support. Retest before adopting it for production runs.

## Revert only the y+ workaround

In the relevant template's `system/controlDict`, change the `yPlus` function entry to:

```foam
    yPlus
    {
        enabled         true;
        // Keep the existing type, libraries and other entries.
    }
```

Edit the existing dictionary; the abbreviated block above is not a replacement for the complete function. Alternatively remove only its `enabled false;` line to restore the prior default. If a particular generated case has a separate `yPlusMean`, enable that only after the producer is verified. Re-enabling built-in yPlus reintroduces the path implicated by the crash, so validate it on the intended mesh/version first.

## Existing orders and cases

New simulation orders snapshot `CoreTemplates` and `Parameters` when created. Existing orders retain their saved preparation snapshot, and generated cases retain their own dictionaries. Editing repository templates does not update either retroactively. Use a fresh order to receive these defaults. For an existing case or deliberate rollback, edit its own dictionaries explicitly and record that configuration with the results. Do not restore entire template files from an old revision merely to undo one workaround; that could also remove unrelated fixes.


## Explicit omega updates (24 October 2026)

**Status: rolled back later on 24 October at the user's request.** The blocks below describe the historical test configuration, not the current normal defaults.

Earlier on 24 October, kOmegaSST templates were changed to separate `omega` and `omegaFinal` linear solver settings in all four AMI/MRF wall-resolved/wall-function templates. The AMI DES template uses the same controls because its turbulence model also solves omega. The kEpsilon templates do not solve omega and are unchanged by this addition.

```foam
    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
        minIter         1;
    }
    omegaFinal
    {
        $omega;
    }
```

Omega is removed from the shared `"(U|k|omega)"` / `"(U|k|omega)Final"` regular expressions; they now cover only U and k. Their existing tolerances remain unchanged. Explicit omega entries avoid dependence on overlapping regular-expression selection. Relaxation settings are unchanged.

The original v3 checkpoint-to-500 microsecond server test skipped every omega iteration: the initial residual was already below 1e-6. The controlled test using these explicit settings completed normally with one omega iteration in all 591 steps. Forces and maximum speed remained close to the previous result, while maximum k changed materially. This is a sensitivity-tested numerical default, not proof of physical accuracy or long-run stability. It did not increase the timestep; the same tip cell remained restrictive. The four SST modes and DES still require validation for their intended simulations.

The existing normal runtime default remains `acoustic-pipeline/openfoam:13-20260624`, and `Parameters/controlDict.cpp` retains maxCo=5. This addition does not change background omega, boundary values, first-layer geometry, pressure settings, wall-distance method or yPlus reporting.

### Revert only the explicit omega controls

In the affected template's fvSolution, remove the standalone `omega` and `omegaFinal` blocks above. Change `"(U|k)"` back to `"(U|k|omega)"` and `"(U|k)Final"` back to `"(U|k|omega)Final"`. Retain the original shared block values: tolerance 1e-6, relTol 0.1 for ordinary solves and relTol 0 for Final. This restores the previous omega solver behaviour without removing other runtime fixes.

Use a fresh simulation order to receive these settings. Existing orders, saved preparation snapshots and already-generated cases keep their previous dictionaries, including when resumed. Do not overwrite those snapshots silently. A server running the normal repository pipeline needs this updated code transferred before creating its new order; the standalone server omega-test bundle already has equivalent controls. Nothing is pushed to GitHub automatically.


## Omega rollback (24 October 2026)

At the user's request, the explicit omega/omegaFinal blocks above were removed from all five affected normal templates and from `output/roundedV3-tip-refinement/roundedV3_4000RPM_AMI/system/fvSolution`. The shared solver patterns are again `"(U|k|omega)"` and `"(U|k|omega)Final"`: tolerance=1e-6, ordinary relTol=0.1, Final relTol=0, and no forced minIter. This permits zero omega iterations when the initial residual meets the existing tolerance. It reproduces the prior solver configuration; it is not a physical-accuracy improvement by itself.

The named live case was already divergent when its dictionary was reverted. This file edit does not repair its existing fields or provide a clean comparison. The rollback did not stop, restart, or delete a simulation. Start a clean case from its own initial fields for comparison; do not resume a blown-up state. A running solver may reread the reverted dictionary because runTimeModifiable is enabled.

Pressure PCG/DIC, Poisson wall distance, disabled built-in yPlus, background omega=1000, tip-label support, Co=5 and the patched OpenFOAM 13 runtime are retained. No other solver equation controls or mesh settings were reverted. Existing immutable order/preparation snapshots were not rewritten; create a fresh order to receive the reverted template defaults. Resuming an old order may still use its saved explicit omega controls.

This rollback follows early-startup divergence on v5 and on the user's restarted v3 case. Those observations do not isolate the omega controls as the cause. The earlier successful server comparisons started from a 152.6 microsecond checkpoint, while these new runs started from zero, which is a different stability test.

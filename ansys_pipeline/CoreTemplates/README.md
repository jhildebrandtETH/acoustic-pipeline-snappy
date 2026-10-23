# Core templates

Select a complete case with `--mode AMI|MRF`, `--turbulence kOmegaSST|kEpsilon|DES`,
and `--wall-functions yes|no`. Templates are arranged as
`<mode>/<turbulence>/wall-functions/` or `<mode>/<turbulence>/wall-resolved/`.

There are nine supported combinations: both wall treatments for SST and k-epsilon
in both rotation modes, plus wall-resolved DES in AMI only.

SST and k-epsilon use `totalPressure` at the inlet with
`pressureInletOutletVelocity` for velocity. Inlet `p0`, initial pressure and the
fixed outlet pressure retain the same `101325` reference. Here `p` is kinematic
pressure (`m2/s2`); this is a shared reference offset, not pressure in pascals.
DES retains its existing total-pressure boundaries and zero reference.

Start a fresh order from time zero to apply these template changes. Existing
orders use saved template snapshots; `--resume` does not update their boundary
conditions. Do not initialize the new test from a diverged solution.

- AMI: moving rotor mesh, NCC coupling.
- MRF: stationary mesh, NCC coupling, rotaryRegion MRF zone, MRFnoSlip propeller.
- Wall functions: nutkWallFunction, kqRWallFunction and omegaWallFunction or epsilonWallFunction.
- Resolved SST/DES: native fixedValue conditions, zero wall k/nut and prescribed wall omega.
- Resolved k-epsilon: LaunderSharmaKE low-Re formulation, zero wall k/modified epsilon/nut.

Boundary layers are imported from Fluent. These templates do not generate
or modify layer thickness, growth rate or surface sizing.
The mesh must resolve the viscous sublayer for the resolved templates; inspect y+.
Wall treatment changes only OpenFOAM dictionaries. Both choices use the same
preprocessing, coupling and solver workflow, with no custom boundary code or
runtime compilation.

For resolved SST/DES, `Parameters/wallResolvedDict` enables `automaticOmega`.
New-case preparation calculates `60*nu/(0.075*y1^2)` separately for every propeller
face using OpenFOAM 13 `nearWallDist` and the case's `transportProperties` viscosity.
This retains the existing fixedValue wall treatment; it does not switch to
omegaWallFunction or constrain the adjacent internal omega cells. The initial
`omegaWallValue` is a placeholder for NCC field loading, then the utility writes
mesh-dependent values to `0/omega` before decomposition. Values are reapplied
after optional field mapping. Inspect `log.wallOmega` for distance/value ranges.

The preparation utility is built with OpenFOAM's bundled `wmake` in the case's
`mesh/wallOmega` directory; no Python dependency or runtime boundary compilation
is needed. Failure to calculate valid positive finite distances stops preparation.
Rigid rotor motion preserves these distances; this does not support deforming
or remeshing walls without recalculation. Wall-function and k-epsilon cases skip it.

Set `automaticOmega false` to retain a manually prescribed `omegaWallValue`.
Changing mesh or viscosity requires preparing a new case. Existing order snapshots
and resumed cases are not rewritten; start a fresh order from zero to adopt this
change. Do not initialize from the already corrupted fifteen-layer solution.

Model and boundary-condition references:
- https://doc.cfd.direct/notes/cfd-general-principles/low-re-k-epsilon-models
- https://cpp.openfoam.org/v13/classFoam_1_1fixedValueFvPatchField.html
- https://cpp.openfoam.org/v13/MRFnoSlipFvPatchVectorField_8C_source.html

MRF uses a frozen rotor orientation. It does not capture sliding-mesh blade-passing
interactions. Impermeable acoustics retains the existing rotating-source reconstruction
from reference geometry; using MRF loads makes this a frozen-loading approximation.
Use AMI for time-resolved aeroacoustic analysis.

All supported templates sample acoustic surface p/U at 65536 Hz using physical
write times. The main output controller also uses adjustableRunTime so OpenFOAM
13 aligns adaptive steps with the surface writes. The parameter writeInterval
now means seconds (default: ten acoustic samples), and timePrecision is 16.
See [acoustic sampling](../Parameters/acoustic-sampling.md), including migration
instructions for already generated cases.

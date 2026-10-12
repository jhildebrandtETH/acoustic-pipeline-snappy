# Snappy-only repository copy

Created 2026-10-12 from the current working tree of C:/repos/acoustic-pipeline
(base Git revision dbd2990aed223a2b62b57e579a22f7594c73a306), including
uncommitted pipeline features and settings. The original repository is unchanged.

This is an independent source snapshot with fresh Git history. The acousticSolver
source is included directly; no submodule points back to the original checkout.
Simulation outputs, temporary experiments, build artifacts and Python caches are
excluded. Supply your propeller STL in the new order's STL directory.

Meshing uses only blockMesh, surfaceFeatures and snappyHexMesh, followed by the
existing rotor/NCC and mesh-quality checks. Native snappy layers are now configurable and enabled by default in
Parameters/snappyHexMeshDict.cpp.
Layer generators, preparation/repair tools, experimental candidate workflows,
external layer configuration/CLI switches and layer-specific report pages were removed.

Retained: parallel case scheduling, resume/checkpoints, parameter studies,
solver controls, turbulence/wall-treatment templates, acoustics, visualization,
PDF reports, settings provenance, mesh quality, cell volumes, first-cell spacing,
y+ and Courant diagnostics. First-cell spacing reporting measures the actual
mesh; it does not generate boundary layers.

Custom FEATURES/<geometry>_<group>.obj refinement is restored for le, te, tip,
hub and the legacy other group. Parameters/snappyFeatures.json controls levels
and refinement distances; OBJ curves are converted to native snappy edge meshes.

Validation after feature restoration: 102 regression tests passed, with 4
optional tests skipped in that run. The custom-feature OpenFOAM integration test
was then run separately and passed: 5,793 cells without custom refinement versus
7,558 cells with it. The other 3 optional tests (NCC coupling and two ParaView
renders) remain unrun. No production propeller solve was performed.

Native layer validation: real OpenFOAM meshing added layer cells with custom OBJ
refinement enabled (7,558 cells without layers; 9,156 with layers).

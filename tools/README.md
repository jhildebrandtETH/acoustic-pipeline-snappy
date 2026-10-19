# Pipeline helpers

`cli.py` and `orders.py` validate options and create or resume orders.
`order_files.py` locks orders and preserves input handling.
`scheduler.py` manages CPU budgets, dashboard, checkpoints and case workers.
`slurm.py` submits and reconciles durable case jobs; `slurm_docker.py` applies
their Docker CPU/memory limits, job cgroup parent when supported, and worker-exit cleanup.
`preprocessing.py` isolates geometry preparation and preserves previous cases.
`snappy_pipeline.py` prepares STL geometry, runs standard OpenFOAM meshing and validates the coupled mesh with configurable native layers.
`simple_mesh_pipeline.py` and `mesh_runner.py` run supplied snappy cases and their reports with Docker or native OpenFOAM.
`templates.py` selects mode, turbulence and wall-treatment dictionaries.
`openfoam.py` contains solver monitoring, reconstruction and order helpers.
`report_worker.py`, `reporting.py`, `settings_report.py`, `visualization.py`,
`mesh_quality.py`, `mesh_volumes.py`, `layer_thickness.py` and
`courant_distribution.py` retain the report and diagnostic workflow.

`snappy_features.py` stages named OBJ curves from the order's FEATURES directory,
converts explicit polylines into scaled native edge meshes, and records custom
refinement controls and provenance for reports.

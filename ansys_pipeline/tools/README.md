# ANSYS pipeline helpers

`cli.py` and `orders.py` validate options and create or resume orders.
`order_files.py` locks orders, snapshots configuration and preserves inputs.
`scheduler.py` manages CPU budgets, dashboard, checkpoints and case workers.
`slurm.py` submits and reconciles durable case jobs; `slurm_docker.py` applies
their Docker CPU/memory limits, job cgroup parent when supported, and worker-exit cleanup.
`ansys_mesh.py` imports fixed Fluent meshes, maps zones and patches, checks the
mesh and prepares the rotor interface. `fluent_hdf5.py` converts Fluent CFF HDF5
input to the ASCII Fluent representation consumed by OpenFOAM 13.
`snappy_pipeline.py` retains its compatibility name for shared path safety,
OpenFOAM execution and NCC validation. Its `run_mesh` entry point delegates to
the Fluent importer; it does not run Snappy or generate geometry.
`mesh_point_format.py` preserves coordinates when OpenFOAM utilities rewrite
small mesh faces. The top-level `preprocessing.py` stages cases and preserves
previous runs.
`templates.py` selects mode, turbulence and wall-treatment dictionaries.
`openfoam.py` contains solver monitoring, reconstruction and order helpers.
`report_worker.py`, `reporting.py`, `settings_report.py`, `visualization.py`,
`mesh_quality.py`, `mesh_volumes.py`, `layer_thickness.py` and
`courant_distribution.py` retain the report and diagnostic workflow.

Shared settings readers retain support for older Snappy case dictionaries so
existing diagnostic fixtures and reports can still be read. New ANSYS cases
derive mesh provenance from `mesh/import.json`. The top-level
`diagnoseCourant.py` collects Courant hotspots and their import configuration.

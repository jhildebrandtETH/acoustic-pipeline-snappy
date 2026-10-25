# Preparation performance workaround — 25 October 2026

The normal pipeline now scans ASCII mesh point records in 4 MiB blocks, rather than using small buffered line reads across the Windows/WSL filesystem boundary. Blocks without possible explicit positive coordinate signs are not split into individual Python records. Incomplete records are carried across block boundaries.

Successfully validated files are cached for the lifetime of the Python process by resolved path, device, inode, size, nanosecond modification time and change time. A changed or replaced points file is rechecked. Repairs still retain the original bytes and append the existing repair audit. Mesh quality checks, baffle operations, NCC configuration, solver parameters and turbulence values are unchanged.

Each completed utility wrapper now appends preparation-timings.jsonl in the case directory. It records the command, log name, start epoch, utility success, total wall duration and separate before/after point-check and utility durations. Progress callbacks name the point-format checks instead of leaving them hidden between utility stages. Timing-file I/O failure does not fail a successful utility.

Measured on the same existing 356,199,351-byte V6 points file from Ubuntu through /mnt/c: original line scan 30.04 s; new first validation 6.09 s; unchanged cached validation 0.016 s. These are individual scan measurements, not end-to-end pipeline timings. Nine focused tests cover sign preservation, malformed records, binary files, writer changes, cache invalidation by replacement, block boundaries and timing records.

This update applies when a new Python pipeline process imports the changed modules. It does not update a process already preparing/running a case. No existing simulation folders need deleting, and the current run must not be restarted solely to pick up this change.

Remaining costs include serial Fluent conversion, full mesh reads/writes, four validation commands, wall-omega preparation and serial decomposition. Moving active cases to native Linux storage and implementing a coherent early-decomposition/MPI preparation path are separate changes requiring validation.

## Reverting only this change

Restore the previous versions of tools/mesh_point_format.py, tools/snappy_pipeline.py and tests/test_mesh_point_format.py from the task's outputs/preparation-performance-fix/before folder. Do not use a blanket git restore: the repository already contained unrelated uncommitted fixes. Remove this document only if desired. Existing preparation-timings.jsonl files are diagnostic records and do not affect solver inputs.

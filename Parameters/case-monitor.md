# Case monitor for local and Slurm orders

Use the same commands in another terminal for either execution mode:

```sh
python main.py --sim-dir ~/run/multi_sphere_study --watch
# One snapshot:
python main.py --sim-dir ~/run/multi_sphere_study --status
```

Neither command needs `--slurm` or a cluster config. The monitor detects the
saved execution mode and does not launch, resume, submit or cancel work. This
includes a local pipeline launched inside a manually submitted batch allocation.

Local launchers publish stage, progress, detail and case state to
`<order>/.runtime/local-status.json`. Writes are atomic, throttled to once per
second for progress/heartbeat, with immediate stage/state transitions. The
launcher owns this file; watchers only read it. It is separate from durable
solver checkpoints in `simulation_order.json`. Detailed updates work with both
the launcher's dashboard and `--live-output`.

A single local case using 72 cores appears in RUNNING CASES with Used: 72.
Dependency-waiting cases remain queued/waiting. Durable completion, failure and
blocked status override older runtime snapshots. A cleanly stopped launcher with
an incomplete running case is shown as blocked and requiring resume. The running
count represents saved pipeline worker state; it is not a query of node CPU load.

For Slurm orders, the existing per-attempt `.slurm/*.runtime.json` files and saved
job identities supply the table. `--watch` does not query the cluster itself;
queue/accounting details can be stale while the coordinator is stopped.

An already-running local launcher started before this fix cannot retroactively
publish detailed callbacks. In that case the monitor recognises the saved
`solver_running` checkpoint and reads a bounded tail of `log.pimpleFoam` for
physical-time progress. The row explicitly says it uses saved log/stage data;
this fallback does not prove process liveness. Other old checkpoints retain
their saved state when no finer information is available. Full stage updates
become available on the next normal launcher start/resume; no CFD restart is
needed merely to use the fallback viewer.

"""Read-only case overview for both local launchers and Slurm case jobs."""
import json
import math
from pathlib import Path
import re
import time

from tools.scheduler import (
    SimulationOrderStore, RuntimeStatusRegistry, LiveBatchDashboard,
    initialize_runtime_queue_states, refresh_dependency_runtime_states,
)


def _read_snapshot(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        # A missing snapshot is normal for launchers predating local persistence.
        return {}


def _legacy_progress(root, case, order):
    """Bounded log read for an already-running launcher without a live snapshot."""
    fields = dict(state='RUNNING', stage=case['status'], progress=None,
                  detail='saved stage; live runtime updates unavailable', error=None)
    path = (root / case['folder']).resolve()
    if path.parent != root.resolve():
        return fields
    try:
        with (path/'log.pimpleFoam').open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-131072))
            tail = handle.read().decode('utf-8', errors='replace')
        matches = re.findall(r'^\s*Time\s*=\s*([-+\d.eE]+)\s*$', tail, flags=re.M)
        if not matches:
            return fields
        latest = float(matches[-1])
        if not math.isfinite(latest) or latest < 0:
            return fields
        fields.update(stage='solving', detail=f't={latest:.6f} s (saved solver log)')
        mode = case.get('end_on', order.get('end_on'))
        value = case.get('end_on_value', order.get('end_on_value'))
        if mode in {'rev', 'time'} and value is not None:
            target = float(value)
            if mode == 'rev':
                target *= 60 / float(case['rpm'])
            if math.isfinite(target) and target > 0:
                fields.update(progress=min(100., max(0., 100*latest/target)),
                              detail=f't={latest:.6f}/{target:.6f} s (saved log)')
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError):
        pass
    return fields


def refresh_local_table(store, registry, order=None):
    order = order if order is not None else store.snapshot()
    payload = _read_snapshot(store.simulations_directory/'.runtime/local-status.json')
    records = payload.get('cases', [])
    if not isinstance(records, list):
        records = []
    saved = {item['folder']: item for item in records
             if isinstance(item, dict) and isinstance(item.get('folder'), str)}
    terminal = {'postprocessing_done', 'failed', 'blocked'}
    active = {case['folder'] for case in order['cases']
              if case.get('status') not in terminal
              and (saved.get(case['folder'], {}).get('state') == 'RUNNING'
                   or case.get('status') == 'solver_running')}
    refresh_dependency_runtime_states(order, registry, active)
    for case in order['cases']:
        folder = case['folder']
        status = case.get('status', 'pending')
        # Durable completion/failure always wins over an older runtime snapshot.
        if status in terminal:
            if status == 'postprocessing_done':
                registry.update(folder, state='DONE', stage='complete', detail='finished',
                                progress=100., error=None)
            else:
                registry.update(folder, state='FAILED' if status == 'failed' else 'BLOCKED',
                                stage=status, detail=case.get('error') or status,
                                progress=None, error=case.get('error'))
            continue
        item = saved.get(folder)
        if item and item.get('state') not in {'DONE', 'FAILED', 'BLOCKED'}:
            # Identity, core allocation and dependencies come from the order.
            fields = {key: item[key] for key in ('state', 'stage', 'progress', 'detail', 'error') if key in item}
            if payload.get('active') is False and fields.get('state') == 'RUNNING':
                fields.update(state='BLOCKED', detail='local launcher stopped; resume required',
                              error='local launcher stopped before case completion')
            registry.update(folder, **fields)
        elif status == 'solver_running':
            registry.update(folder, **_legacy_progress(store.simulations_directory, case, order))


def refresh_saved_table(store, registry):
    order = store.snapshot()
    if order.get('slurm') or any(case.get('slurm_job') for case in order['cases']):
        from tools.slurm import refresh_saved_table as refresh_slurm
        refresh_slurm(store, registry)
    else:
        refresh_local_table(store, registry, order)


def show_status(root, watch=False):
    store = SimulationOrderStore(Path(root).resolve())
    order = store.snapshot()
    registry = RuntimeStatusRegistry(order['cases'])
    initialize_runtime_queue_states(order, registry)
    dashboard = LiveBatchDashboard(registry, order['total_cores'], order['cores_per_case'],
                                   max_cores_per_case=order.get('max_cores_per_case'))
    refresh_saved_table(store, registry)
    if not watch:
        print(dashboard.render_text())
        return
    dashboard.start()
    try:
        while True:
            refresh_saved_table(store, registry)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()

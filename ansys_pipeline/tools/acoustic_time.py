"""Case-local acoustic cadence and interpolation of adaptive-time pressure traces."""
from dataclasses import replace
from pathlib import Path

import numpy as np


def case_sample_interval(case):
    """Read the target interval from this case, including ordinary overrides."""
    from tools.settings_report import read_dictionary, resolve_dictionary
    case = Path(case).resolve()
    path = case / 'system/controlDict'
    if not path.is_file():
        path = case / 'Parameters/controlDict.cpp'
    if not path.is_file():
        raise FileNotFoundError(f'Missing case time controls: {path}')
    warnings = []
    control = resolve_dictionary(read_dictionary(path, case, {}, warnings))
    value = control.get('acousticWriteInterval')
    try:
        step = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Set acousticWriteInterval (seconds) in this case for the uniform acoustic grid') from exc
    if not np.isfinite(step) or step <= 0:
        raise ValueError('acousticWriteInterval must be positive and finite')
    return step


def source_time_statistics(times, interval):
    """Reject missing/coarsely sampled history rather than filling long gaps."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Expected at least two finite acoustic timestamps')
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('Acoustic sample interval must be positive and finite')
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError('Acoustic timestamps must increase strictly')
    if gaps.max() > interval * (1 + 1e-6):
        raise ValueError(
            f'Acoustic sampling gap {gaps.max():.9g} s exceeds target interval '
            f'{interval:.9g} s; interpolation cannot recover missing samples. '
            'Use every-step p/U output and maxDeltaT <= acousticWriteInterval.')
    return dict(source_sample_count=len(times), source_min_interval_s=float(gaps.min()),
                source_median_interval_s=float(np.median(gaps)), source_max_interval_s=float(gaps.max()))


def uniform_pressure(times, pressure, interval=None):
    """Linearly interpolate one or many traces onto a zero-phase uniform grid.

    The last pressure axis is time. No endpoint extrapolation or gap repair is
    performed. Without an explicit interval, retain the legacy median cadence.
    This interpolation is not an anti-alias filter or a temporal-accuracy claim.
    """
    times = np.asarray(times, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or pressure.ndim < 1
            or pressure.shape[-1] != len(times) or not np.isfinite(pressure).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Pressure and time must be finite with matching shapes and strictly increasing time')
    if interval is None:
        interval = float(np.median(np.diff(times)))
        if np.max(np.diff(times)) > 1.5 * interval:
            raise ValueError('Observer trace has sampling gaps')
    else:
        interval = float(interval)
        source_time_statistics(times, interval)
    first = int(np.ceil(times[0] / interval))
    last = int(np.floor(times[-1] / interval))
    grid = np.arange(first, last + 1, dtype=float) * interval
    grid = grid[(grid >= times[0]) & (grid <= times[-1])]
    if len(grid) < 2:
        raise ValueError('Too few samples on the requested uniform acoustic grid')
    rows = pressure.reshape(-1, len(times))
    values = np.array([np.interp(grid, times, row) for row in rows])
    return grid, values.reshape((*pressure.shape[:-1], len(grid))), interval


def uniform_result(result, interval):
    """Resample every FW-H pressure component consistently, preserving metadata."""
    import torch
    fields = ('thickness_pressure_pa', 'loading_pressure_pa', 'total_pressure_pa')
    times = result.observer_times_s.detach().cpu().numpy()
    arrays = np.stack([getattr(result, name).detach().cpu().numpy() for name in fields])
    grid, values, _ = uniform_pressure(times, arrays, interval)
    updated = {name: torch.as_tensor(values[i], dtype=getattr(result, name).dtype,
                                     device=getattr(result, name).device)
               for i, name in enumerate(fields)}
    updated['observer_times_s'] = torch.as_tensor(grid, dtype=result.observer_times_s.dtype,
                                                 device=result.observer_times_s.device)
    return replace(result, **updated)

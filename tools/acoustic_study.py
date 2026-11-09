"""Record concentric FW-H spheres once, or sample retained volume fields later.

Run post-solver studies with ``python -m tools.acoustic_study --help``.
Study case views link shared CFD inputs; each owns its parameters, report and cache.
Like the CFD pipeline, this workflow runs on Linux/WSL (directory symlinks).
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import shutil


def diameter_factors(values=None, sweep=None):
    if sweep is not None:
        try:
            start, stop, step = map(lambda x: Decimal(str(x)), sweep)
            if not all(x.is_finite() for x in (start, stop, step)) or not step or (stop-start)*step < 0:
                raise ValueError('Range step must point from START to STOP and be nonzero')
            count = (stop-start)/step
            if count != count.to_integral_value() or count > 999:
                raise ValueError('Range must reach STOP exactly, with at most 1000 diameters')
            values = [float(start+i*step) for i in range(int(count)+1)]
        except (InvalidOperation, TypeError) as exc:
            raise ValueError('Invalid sphere range') from exc
    result = [float(x) for x in (values or [])]
    if not result or any(not math.isfinite(x) or x <= 1 for x in result):
        raise ValueError('Sphere diameter factors must be finite and greater than 1')
    if len(set(result)) != len(result):
        raise ValueError('Sphere diameter factors must be unique')
    if len({sphere_key(x) for x in result}) != len(result):
        raise ValueError('Sphere diameter factors are too close to distinguish')
    return result


def sphere_key(factor):
    return 'sphere_' + format(factor, '.10g').replace('.', 'p') + 'D'


def configure_recording(case, factors, diameter):
    """Keep the original largest-sphere writer, add independent p/U writers."""
    import trimesh
    case = Path(case)
    factors = diameter_factors(factors)
    control = case / 'system/controlDict'
    text = control.read_text()
    match = re.search(r'\bwritePermeableSurfaceFields\s*\{', text)
    if not match:
        raise ValueError('Template is missing writePermeableSurfaceFields')
    depth, end = 1, match.end()
    while depth and end < len(text):
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    if depth:
        raise ValueError('Unbalanced acoustic writer dictionary')
    original = text[match.start():end]
    additions, surfaces = [], []
    largest = max(factors)
    for factor in factors:
        key = sphere_key(factor)
        writer = 'writePermeableSurfaceFields' if factor == largest else 'write_' + key
        stl = 'permeableSurface.stl' if factor == largest else key + '.stl'
        radius = .5 * factor * diameter
        if factor != largest:
            trimesh.creation.icosphere(subdivisions=4, radius=radius).export(case/'constant/triSurface'/stl)
            additions.append(original.replace('writePermeableSurfaceFields', writer, 1)
                             .replace('permeableSurface.stl', stl))
        surfaces.append(dict(factor=factor, radius_m=radius, writer=writer,
                             surface_file='permeableSurface.vtk', stl=stl))
    control.write_text(text[:end] + '\n' + '\n'.join(additions) + text[end:])
    (case/'acoustic-surfaces.json').write_text(json.dumps(
        dict(propeller_diameter_m=diameter, surfaces=surfaces), indent=2)+'\n')


def _link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f'Unexpected study link: {destination}')
    elif destination.exists():
        raise ValueError(f'Refusing to replace existing study input: {destination}')
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_view(case, item, source):
    """A separately writable acoustic case, without duplicating CFD fields."""
    case = Path(case).resolve()
    view = case/'acoustic-study'/source/sphere_key(item['factor'])/case.name
    view.mkdir(parents=True, exist_ok=True)
    excluded = {'acoustic-study', 'postProcessing', 'report', '.cache', 'Parameters', 'system', 'constant', 'mesh'}
    for entry in case.iterdir():
        if entry.name not in excluded:
            _link(entry, view/entry.name)
    # Config readers require Parameters to be physically inside the case.
    shutil.copytree(case/'Parameters', view/'Parameters', dirs_exist_ok=True)
    shutil.copytree(case/'system', view/'system', dirs_exist_ok=True)
    # Dictionary readers reject includes outside the case. Keep dictionaries
    # local while sharing the large volume mesh and geometry directories.
    (view/'constant').mkdir(exist_ok=True)
    for entry in (case/'constant').iterdir():
        if entry.is_dir():
            _link(entry, view/'constant'/entry.name)
        else:
            shutil.copy2(entry, view/'constant'/entry.name)
    (view/'mesh').mkdir(exist_ok=True)
    for entry in (case/'mesh').iterdir():
        if entry.name != 'geometry.json':
            _link(entry, view/'mesh'/entry.name)
    geometry = json.loads((case/'mesh/geometry.json').read_text())
    geometry['acoustic_sphere_radius_m'] = item['radius_m']
    (view/'mesh/geometry.json').write_text(json.dumps(geometry, indent=2))
    (view/'postProcessing').mkdir(exist_ok=True)
    for entry in (case/'postProcessing').iterdir():
        if not entry.name.startswith(('writePermeableSurfaceFields', 'write_sphere_')):
            _link(entry, view/'postProcessing'/entry.name)
    if source == 'recorded':
        _link(case/'postProcessing'/item['writer'], view/'postProcessing/writePermeableSurfaceFields')
    (view/'report').mkdir(exist_ok=True)
    # Reuse the expensive shared flow/mesh atlas; acoustic figures stay independent.
    visuals = case/'report/visuals'
    if visuals.is_dir():
        target = view/'report/visuals'
        target.mkdir(exist_ok=True)
        for entry in visuals.iterdir():
            if entry.name == 'manifest.json':
                shutil.copy2(entry, target/entry.name)
            else:
                _link(entry, target/entry.name)
    atlas = view/'report/visuals/manifest.json'
    if atlas.is_file():
        from tools.visualization import visualization_chapter
        data = json.loads(atlas.read_text())
        data['views'] = [v for v in data.get('views', []) if visualization_chapter(v) != 'acoustic']
        data.setdefault('warnings', []).append('Shared CFD mesh/flow views; original-sphere acoustic atlas views omitted for this diameter study.')
        atlas.write_text(json.dumps(data, indent=2))
    metadata = dict(item, source=source, cfd_case=str(case), response_delay_s=None)
    (view/'report/acoustic-sphere.json').write_text(json.dumps(metadata, indent=2))
    return view


def compare_cached(entries, output, rpm):
    """Compare the same fully-arrived observer interval for every diameter."""
    import numpy as np
    from tools.plotting import pyplot as plt, save_figure
    output = Path(output)
    traces = []
    for entry in entries:
        path = Path(entry['report'])/'acoustic-pressure.npz'
        if not path.is_file():
            raise ValueError(f'No acoustic pressure cache for {entry["factor"]:g}D: {path}')
        with np.load(path) as data:
            traces.append({k: data[k].copy() for k in ('time_s', 'pressure_pa', 'sampling_frequency_hz', 'observer_m')})
    observer = traces[0]['observer_m']
    if any(not np.allclose(t['observer_m'], observer) for t in traces):
        raise ValueError('Sphere comparison requires the same observer')
    step = max(1/float(t['sampling_frequency_hz']) for t in traces)
    start = max(float(t['time_s'][0]) for t in traces)
    stop = min(float(t['time_s'][-1]) for t in traces)
    rotations = math.floor((stop-start)*rpm/60 + 1e-9)
    if rotations < 5:
        raise ValueError('Sphere comparison needs at least five common fully-arrived revolutions')
    count = int(math.floor(rotations*60/rpm/step))
    grid = stop - np.arange(count-1, -1, -1)*step
    frequencies = np.fft.rfftfreq(count, step)
    window = np.hanning(count)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    rows, spectra = [], {}
    for entry, trace in zip(entries, traces):
        pressure = np.interp(grid, trace['time_s'], trace['pressure_pa'])
        pressure -= pressure.mean()
        power = np.abs(np.fft.rfft(pressure*window))**2 / (count*np.sum(window**2))
        power[1:-1 if count % 2 == 0 else None] *= 2
        level = 10*np.log10(np.maximum(power, 1e-30)/(20e-6)**2)
        ax.semilogx(frequencies[1:], level[1:], label=f'{entry["factor"]:g}D', linewidth=.8)
        spectra[sphere_key(entry['factor'])] = level
        rows.append(dict(factor=entry['factor'], report=entry['report'],
                         oaspl_db=float(10*np.log10(max(np.mean(pressure**2), 1e-30)/(20e-6)**2))))
    ax.set(xlabel='Frequency [Hz]', ylabel='SPL per FFT bin [dB re 20 µPa]', title='Acoustic sphere diameter comparison')
    ax.grid(True, alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, output/'sphere-comparison.png')
    plt.close(fig)
    np.savez_compressed(output/'sphere-comparison.npz', frequencies_hz=frequencies, **spectra)
    result = dict(status='complete', observer_m=observer.tolist(), common_start_s=float(grid[0]),
                  common_stop_s=float(grid[-1]), sample_interval_s=step, requested_rotations=rotations,
                  frequency_resolution_hz=1/(count*step), window='Hann; mean removed', surfaces=rows)
    (output/'sphere-comparison.json').write_text(json.dumps(result, indent=2))
    import csv
    with (output/'sphere-comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['factor', 'report', 'oaspl_db'])
        writer.writeheader()
        writer.writerows(rows)
    return result


def run_study(case, rpm, mode, turbulence, *, source='recorded', factors=None,
              time_selection=None, sample_interval=None, status_callback=None):
    from acoustic_propagation import run_acoustic_solver
    from tools.report_worker import create_simulation_report
    from tools.common import MATPLOTLIB_LOCK
    case = Path(case).resolve()
    if not math.isfinite(rpm) or rpm <= 0:
        raise ValueError('Study RPM must be positive and finite')
    if sample_interval is not None and (not math.isfinite(sample_interval) or sample_interval <= 0):
        raise ValueError('Sample interval must be positive and finite')
    if source == 'recorded':
        manifest = json.loads((case/'acoustic-surfaces.json').read_text())
        surfaces = manifest['surfaces']
        if factors:
            requested = diameter_factors(factors)
            surfaces = [s for s in surfaces if s['factor'] in requested]
            if len(surfaces) != len(requested):
                raise ValueError('Requested diameter was not recorded; use --source volume if its volume history exists')
        # Validate all histories before starting expensive propagation.
        histories = []
        for item in surfaces:
            folder = case/'postProcessing'/item['writer']
            times = sorted(float(p.name) for p in folder.iterdir()
                           if p.is_dir() and (p/item['surface_file']).is_file())
            if not times:
                raise ValueError(f'No recorded history for {item["factor"]:g}D')
            histories.append(times)
        if any(times != histories[0] for times in histories[1:]):
            raise ValueError('Recorded sphere histories must have identical source timestamps')
    elif source == 'volume':
        import numpy as np
        geometry = json.loads((case/'mesh/geometry.json').read_text())
        diameter = geometry['diameter_m']
        surfaces = [dict(factor=f, radius_m=.5*f*diameter) for f in diameter_factors(factors)]
        # Readability, enclosing geometry and domain containment are checked by the sampler/geometry check below.
        import trimesh
        propeller = trimesh.load(case/'constant/triSurface/propeller.stl', force='mesh')
        if any(item['radius_m'] <= np.linalg.norm(propeller.vertices, axis=1).max() for item in surfaces):
            raise ValueError('Every study sphere must enclose the complete propeller')
    else:
        raise ValueError('Source must be recorded or volume')
    if not surfaces:
        raise ValueError('No study surfaces selected')
    entries = []
    output = case/'acoustic-study'/source
    output.mkdir(parents=True, exist_ok=True)
    for name in ('sphere-comparison.png', 'sphere-comparison.pdf', 'sphere-comparison.npz', 'sphere-comparison.csv'):
        (output/name).unlink(missing_ok=True)
    (output/'sphere-comparison.json').write_text(json.dumps(dict(status='running')))
    (output/'study.json').write_text(json.dumps(dict(source=source, surfaces=[])))
    for item in surfaces:
        view = prepare_view(case, item, source)
        series = None
        interval = sample_interval
        if source == 'volume':
            import numpy as np
            from acousticSolver.src.foamacoustics import SphereSurface, interpolate_openfoam_fields
            series = interpolate_openfoam_fields(case, SphereSurface(radius_m=item['radius_m']),
                surface_name=sphere_key(item['factor']), time_selection=time_selection)
            times = sorted(float(p.name) for p in series.input_dir.iterdir()
                           if p.is_dir() and (p/series.surface_file).is_file())
            if len(times) < 3:
                raise ValueError('Retained volume history needs at least three samples and five revolutions plus propagation delay')
            interval = interval or float(np.max(np.diff(times)))
        if interval is not None:
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError('Sample interval must be positive and finite')
        metadata_path = view/'report/acoustic-sphere.json'
        metadata = json.loads(metadata_path.read_text())
        metadata.update(sample_interval_override_s=interval, time_selection=time_selection)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        for image in ([image for suffix in ('png', 'pdf') for image in (view/'report').glob(f'acoustic-*.{suffix}')] +
                [view/'report'/f'spl_spectrum.{suffix}' for suffix in ('png', 'pdf')]):
            image.unlink(missing_ok=True)
        run_acoustic_solver('permeable', view, rpm, STATUS_CALLBACK=status_callback,
                            SURFACE_SERIES=series, RESPONSE_DELAY_S=None, SAMPLE_INTERVAL_S=interval)
        with MATPLOTLIB_LOCK:
            create_simulation_report(case_path=view, rpm=rpm, mode=mode, turbulence_model=turbulence,
                                     quiet=True, status_callback=status_callback)
        entry = dict(factor=item['factor'], radius_m=item['radius_m'], report=str(view/'report'))
        entries.append(entry)
        (output/'study.json').write_text(json.dumps(dict(source=source, surfaces=entries), indent=2))
    with MATPLOTLIB_LOCK:
        try:
            compare_cached(entries, output, rpm)
        except ValueError as exc:
            (output/'sphere-comparison.json').write_text(json.dumps(dict(status='unavailable', detail=str(exc)), indent=2))
            print(f'Sphere comparison unavailable: {exc}', flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--source', choices=['recorded', 'volume'], default='recorded')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--diameters', nargs='+', type=float)
    group.add_argument('--range', nargs=3, metavar=('START', 'STOP', 'STEP'))
    parser.add_argument('--time-selection', help='Volume source time range, e.g. 0.10:0.30')
    parser.add_argument('--sample-interval', type=float, help='Uniform observer grid in seconds; volume default is maximum saved source interval')
    parser.add_argument('--rpm', type=float, required=True)
    parser.add_argument('--mode', choices=['AMI', 'MRF'], required=True)
    parser.add_argument('--turbulence', required=True)
    args = parser.parse_args()
    try:
        factors = diameter_factors(args.diameters, args.range) if args.diameters or args.range else None
        from tools.order_files import order_lock
        with order_lock(args.case.resolve().parent):
            entries = run_study(args.case, args.rpm, args.mode, args.turbulence, source=args.source,
                factors=factors, time_selection=args.time_selection, sample_interval=args.sample_interval)
        for entry in entries:
            print(f'{entry["factor"]:g}D: {entry["report"]}/simulation_report.pdf')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Acoustic study failed: {exc}\n')


if __name__ == '__main__':
    main()

"""Report-only turbulence/mesh cutoff: f_MC = sqrt(2 k / 3) / (N Delta)."""
import json
import gzip
from pathlib import Path
import re

import numpy as np

from .layer_thickness import _read_list


REFERENCE = ('https://strathprints.strath.ac.uk/70275/1/'
             'Minisci_APGAS2019_Propagation_of_boundary_and_geometrical_uncertainties_for_the_aeroacoustics_analysis.pdf')


def resolution_settings(config):
    result = dict(source_patches=['propeller'], cells_per_length_scale=2.,
                  cell_size='volume_cuberoot')
    supplied = config.get('mesh_resolution', {})
    # Previous report-only acoustic-wave options no longer enter this model.
    legacy = {'sound_speed_m_s', 'cells_per_wavelength', 'convective_speed_m_s', 'samples_per_period'}
    if not isinstance(supplied, dict) or set(supplied) - set(result) - legacy:
        raise ValueError('Unknown mesh_resolution settings')
    result.update({key: value for key, value in supplied.items() if key not in legacy})
    result['ignored_legacy_settings'] = sorted(set(supplied) & legacy)
    patches = result['source_patches']
    if (not isinstance(patches, list) or not patches
            or any(not isinstance(p, str) or not re.fullmatch(r'[\w.-]+', p) for p in patches)):
        raise ValueError('mesh_resolution.source_patches must be a nonempty list of patch names')
    value = result['cells_per_length_scale']
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < 2):
        raise ValueError('mesh_resolution.cells_per_length_scale must be finite and at least two')
    if result['cell_size'] not in ('volume_cuberoot', 'bbox_diagonal'):
        raise ValueError('mesh_resolution.cell_size must be volume_cuberoot or bbox_diagonal')
    return result


def measure_cell_spans(mesh, patches, mesh_files=None):
    """Cell bounding-box diagonal: upper bound on cell diameter, in metres.

    Unlike V**(1/3) or wall-normal depth, this does not hide stretched cells.
    All vertices of each cell contribute, including its neighbour-side faces.
    """
    mesh = Path(mesh)
    mesh_files = mesh_files or {}
    def read(name):
        return _read_list(mesh_files.get(name, mesh / name))
    count, text = read('points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ')
    if points.size != count * 3 or not np.isfinite(points).all():
        raise ValueError('Invalid mesh points')
    points = points.reshape(-1, 3)
    nfaces, text = read('faces')
    sizes = np.array([int(v) for v in re.findall(r'(\d+)\s*\(', text)], dtype=np.int64)
    vertices = np.fromstring(re.sub(r'\d+\s*\(', ' ', text).replace(')', ' '), sep=' ', dtype=np.int64)
    if (len(sizes) != nfaces or sizes.sum() != len(vertices) or np.any(sizes < 3)
            or np.any(vertices < 0) or np.any(vertices >= count)):
        raise ValueError('Invalid mesh faces')
    offsets = np.r_[0, np.cumsum(sizes)]
    nowner, text = read('owner')
    owner = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    nneighbour, text = read('neighbour')
    neighbour = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    if (not nfaces or nowner != nfaces or len(owner) != nfaces or len(neighbour) != nneighbour
            or nneighbour > nfaces or np.any(owner < 0) or np.any(neighbour < 0)):
        raise ValueError('Invalid cell addressing')
    labels = np.union1d(owner, neighbour)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError('Cell labels are not contiguous')
    lower = np.full((len(labels), 3), np.inf)
    upper = np.full((len(labels), 3), -np.inf)
    for start in range(0, nfaces, 50000):
        end = min(start + 50000, nfaces)
        local = offsets[start:end+1] - offsets[start]
        coordinates = points[vertices[offsets[start]:offsets[end]]]
        face_low = np.minimum.reduceat(coordinates, local[:-1])
        face_high = np.maximum.reduceat(coordinates, local[:-1])
        np.minimum.at(lower, owner[start:end], face_low)
        np.maximum.at(upper, owner[start:end], face_high)
        stop = min(end, nneighbour)
        if stop > start:
            np.minimum.at(lower, neighbour[start:stop], face_low[:stop-start])
            np.maximum.at(upper, neighbour[start:stop], face_high[:stop-start])
    spans = np.linalg.norm(upper - lower, axis=1)
    if not np.isfinite(spans).all() or np.any(spans <= 0):
        raise ValueError('Nonpositive or nonfinite cell dimensions')
    _, boundary = read('boundary')
    selected = []
    for patch in patches:
        match = re.search(r'(?<!\w)"?' + re.escape(patch) + r'"?\s*\{([^{}]*)\}', boundary)
        if not match:
            raise ValueError(f'Source patch {patch!r} is absent from the initial mesh')
        start = re.search(r'\bstartFace\s+(\d+)', match[1])
        count = re.search(r'\bnFaces\s+(\d+)', match[1])
        if not start or not count:
            raise ValueError('Missing source patch addressing')
        start, count = int(start[1]), int(count[1])
        if count <= 0 or start < nneighbour or start + count > nfaces:
            raise ValueError('Empty or invalid source patch')
        selected.extend(owner[start:start+count])
    return spans, np.unique(selected)



def local_cutoff(k, cell_size, cells_per_length_scale=2.):
    """Local turbulent velocity scale divided by the shortest retained scale."""
    k, cell_size = np.asarray(k, dtype=float), np.asarray(cell_size, dtype=float)
    if (k.shape != cell_size.shape or k.ndim != 1 or not k.size
            or not np.isfinite(k).all() or np.any(k < 0)
            or not np.isfinite(cell_size).all() or np.any(cell_size <= 0)):
        raise ValueError('Cutoff requires matching finite nonnegative k and positive cell sizes')
    if not np.isfinite(cells_per_length_scale) or cells_per_length_scale < 2:
        raise ValueError('cells_per_length_scale must be finite and at least two')
    velocity = np.sqrt(2. / 3. * k)
    return velocity, velocity / (cells_per_length_scale * cell_size)


def latest_snapshot(case):
    from .courant_distribution import _times
    times = _times(case)
    if not times:
        raise ValueError('No reconstructed flow snapshot above time zero; a saved k field is required')
    time, directory = times[-1]
    for processor in case.glob('processor*'):
        if processor.is_dir():
            newer = _times(processor)
            if newer and newer[-1][0] > time:
                raise ValueError('Processor output is newer; reconstruct the latest flow fields first')
    files = {}
    for name in ('points', 'faces', 'owner', 'neighbour', 'boundary'):
        candidates = [case / 'constant/polyMesh' / name] + [p / 'polyMesh' / name for _, p in times]
        files[name] = next((p for p in reversed(candidates)
                            if p.is_file() or p.with_name(name + '.gz').is_file()), None)
        if files[name] is None:
            raise ValueError(f'Missing reconstructed mesh file: {name}')
    return time, directory, files


def create_resolution_report_data(case, config=None):
    """Return one conservative source-region cap from the latest saved k field."""
    from diagnoseCourant import read_ascii_internal
    from .mesh_volumes import measure_cell_volumes
    case = Path(case)
    report = case / 'report'
    report.mkdir(parents=True, exist_ok=True)
    csv = report / 'acoustic-resolution-cells.csv'
    csv.unlink(missing_ok=True)
    result = dict(status='unavailable', method='turbulent_mesh_cutoff_v2', report_only=True,
                  spatial_resolution_cap_hz=None, reference=REFERENCE,
                  formula='min_i sqrt(2*k_i/3)/(N*Delta_i)', aggregation='minimum over all selected source cells')
    try:
        if config is None:
            from .acoustic_comparison import load_config
            config = load_config(case)
        settings = resolution_settings(config)
        result['settings'] = settings
        time, directory, mesh_files = latest_snapshot(case)
        result.update(time_s=time, k_field=str(directory / 'k'),
                      mesh_files={name: str(path) for name, path in mesh_files.items()})
        spans, selected = measure_cell_spans(case / 'constant/polyMesh', settings['source_patches'], mesh_files)
        field = directory / 'k'
        if not field.is_file():
            field = directory / 'k.gz'
        result['k_field'] = str(field)
        opener = gzip.open if field.suffix == '.gz' else open
        with opener(field, 'rt', encoding='utf-8') as stream:
            header = re.sub(r'/\*.*?\*/|//[^\n]*', '', stream.read(8192), flags=re.S)
        dimensions = re.search(r'\bdimensions\s*\[([^\]]+)\]', header)
        if not dimensions or dimensions[1].split() != ['0', '2', '-2', '0', '0', '0', '0']:
            raise ValueError('k must have turbulent kinetic energy dimensions [0 2 -2 0 0 0 0]')
        k = read_ascii_internal(field, len(spans), 1)
        if not np.isfinite(k).all() or np.any(k < 0):
            raise ValueError('Saved k field contains nonfinite or negative turbulent kinetic energy')
        if settings['cell_size'] == 'volume_cuberoot':
            volumes = measure_cell_volumes(case / 'constant/polyMesh', mesh_files)
            if not np.isfinite(volumes).all() or np.any(volumes <= 0):
                raise ValueError('Cell volumes must be finite and positive')
            size = np.cbrt(volumes)
        else:
            size = spans
        velocity, frequencies = local_cutoff(k[selected], size[selected], settings['cells_per_length_scale'])
        index = int(np.argmin(frequencies))
        cell = int(selected[index])
        result.update(status='complete', spatial_resolution_cap_hz=float(frequencies[index]),
                      source_cells=len(selected), mesh_cells=len(spans),
                      limiting_cell=dict(cell_id=cell, delta_m=float(size[cell]), k_m2_s2=float(k[cell]),
                                         turbulent_velocity_m_s=float(velocity[index])),
                      zero_k_source_cells=int(np.count_nonzero(k[selected] == 0)), csv=str(csv))
        np.savetxt(csv, np.c_[selected, size[selected], k[selected], velocity, frequencies], delimiter=',',
                   header='cell_id,delta_m,k_m2_s2,turbulent_velocity_m_s,local_cutoff_hz', comments='')
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        result.update(status='unavailable', spatial_resolution_cap_hz=None,
                      reason=str(exc).replace('First-cell measurement', 'Mesh frequency cutoff'))
    (report / 'acoustic-resolution.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def append_resolution_report(canvas, result):
    """Print one spatial cap, without applying it to any acoustic plot."""
    from reportlab.lib.utils import simpleSplit
    width, height = canvas._pagesize
    if canvas._code:
        canvas.showPage()
    canvas.setFont('Helvetica-Bold', 16)
    canvas.drawString(40, height - 48, 'Spatial mesh frequency cap')
    y = height - 82

    def paragraph(text):
        nonlocal y
        canvas.setFont('Helvetica', 10)
        for line in simpleSplit(text, 'Helvetica', 10, width - 80):
            if y < 50:
                canvas.showPage()
                canvas.setFont('Helvetica', 10)
                y = height - 60
            canvas.drawString(40, y, line)
            y -= 15
        y -= 12

    paragraph('Local mesh cutoff: f_MC,i = sqrt(2 k_i / 3) / (N Delta_i). '
              'The local turbulent velocity scale comes from the saved k field. '
              'The default N = 2 represents the shortest turbulent scale in this cutoff model.')
    if result['status'] == 'complete':
        s, cell = result['settings'], result['limiting_cell']
        paragraph(f"Saved flow snapshot: t = {result['time_s']:g} s. Source region: cells touching "
                  f"{', '.join(s['source_patches'])}; {result['source_cells']:,} of {result['mesh_cells']:,} cells.")
        paragraph('Cell size Delta = ' + ('V^(1/3), using local cell volume.'
                  if s['cell_size'] == 'volume_cuberoot' else 'cell bounding-box diagonal, a conservative size for stretched cells.'))
        paragraph(f"N = {s['cells_per_length_scale']:g}. Evaluate each cell with its own k and Delta, then take "
                  'the minimum local frequency across the selected source cells. No cells or low-k locations are discarded.')
        paragraph(f"Limiting cell ID: {cell['cell_id']} (zero-based). Delta = {1000 * cell['delta_m']:.6g} mm; "
                  f"k = {cell['k_m2_s2']:.6g} m2/s2; sqrt(2k/3) = {cell['turbulent_velocity_m_s']:.6g} m/s.")
        if result['zero_k_source_cells']:
            paragraph(f"{result['zero_k_source_cells']} selected cells have k = 0, so the strict spatial cap is zero.")
        if s['ignored_legacy_settings']:
            paragraph('Obsolete acoustic-wave settings ignored: ' + ', '.join(s['ignored_legacy_settings']) + '.')
    else:
        paragraph('Required flow or mesh data unavailable: ' + result['reason'])
    paragraph('This is a turbulence-based spatial cutoff estimate for the stated region and snapshot. '
              'It does not certify spectral accuracy, establish a timestep limit, or limit deterministic BPF tones. '
              'Saved k is the turbulence model field; in DES/LES it need not include resolved fluctuation energy. '
              'V^(1/3) is a nominal size and can be optimistic for stretched cells; bbox_diagonal is a configurable conservative alternative. '
              'The rest of the wake is outside the selected scope.')
    paragraph('Report only. Upper plot limits are unchanged. Cellwise values and provenance are saved in '
              'acoustic-resolution-cells.csv and acoustic-resolution.json.')
    canvas.linkURL(REFERENCE, (40, y - 25, width - 40, y + 10), relative=0)
    paragraph('Reference: Minisci, Scardigli and Gallizio, APG Acoustic Symposium 2019, '
              'Mesh frequency cut-off (slide 7; click to open).')
    if y < 95:
        canvas.showPage()
        y = height - 65
    canvas.setFont('Helvetica-Bold', 15)
    cap = result['spatial_resolution_cap_hz']
    label = f'{cap:,.3g} Hz' if cap is not None else 'unavailable (saved flow data required)'
    for line in simpleSplit('Estimated spatial frequency cap: ' + label, 'Helvetica-Bold', 15, width - 80):
        canvas.drawString(40, y, line)
        y -= 20

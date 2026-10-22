"""Import a fixed Fluent mesh, preserve cell zones, and prepare OF13 NCC motion."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

BACKEND = 'ansys-import-v1'
COMPATIBLE_BACKENDS = (BACKEND,)
DEFAULTS = dict(scale=0.001, rotorCellZone='rotaryregion',
                rotorFaceZone='rotor_interface',
                patches=dict(propeller='propeller', inlet='inlet', outlet='outlet', walls='walls'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def find_meshes(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Place Fluent .msh.h5 or legacy ASCII .msh files in {directory}')
    result = {}
    for path in sorted(directory.iterdir()):
        suffix = next((s for s in ('.msh.h5', '.msh') if path.name.lower().endswith(s)), None)
        if not path.is_file() or not suffix:
            continue
        name = path.name[:-len(suffix)]
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or re.search(r'_\d+RPM_', name):
            raise ValueError(f'Use a simple distinct mesh filename: {path.name}')
        if name in result:
            raise ValueError(f'Duplicate mesh identity {name}: keep one format per input')
        result[name] = path
    if not result:
        raise FileNotFoundError(f'No Fluent .msh.h5 or .msh files in {directory}')
    return result


def read_config(parameters, source):
    config = dict(DEFAULTS, patches=dict(DEFAULTS['patches']))
    name = Path(source).name
    stem = name[:-7] if name.lower().endswith('.msh.h5') else Path(source).stem
    for path in (Path(parameters)/'ansysMesh.json', Path(source).with_name(stem+'.json')):
        if path.is_file():
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or set(value)-set(DEFAULTS):
                raise ValueError(f'Unknown imported-mesh settings in {path}')
            config.update(value)
    scale = config['scale']
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Imported mesh scale must be finite and positive')
    if set(config['patches']) != set(DEFAULTS['patches']):
        raise ValueError('Patch mapping must specify propeller, inlet, outlet and walls')
    if len(set(config['patches'].values())) != 4:
        raise ValueError('Source patch names must be distinct')
    for value in [config['rotorCellZone'], config['rotorFaceZone'], *config['patches'].values()]:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', value):
            raise ValueError('Zone and patch names must be simple OpenFOAM identifiers')
    return config


def input_fingerprints(directory, parameters):
    result = {}
    for name, source in find_meshes(directory).items():
        result[name] = dict(sha256=digest(source), config=read_config(parameters, source))
    return result


def preflight(args):
    from tools.snappy_pipeline import preflight as common_preflight
    if args.mesh_case or args.mesh_native:
        raise ValueError('This copy imports Fluent meshes; --mesh-case/--mesh-native are not supported')
    if args.study and args.study_file in ('snappyHexMeshDict', 'snappyHexMeshDict.cpp',
                                         'blockMeshDict', 'blockMeshDict.cpp',
                                         'snappyGeometry.json', 'snappyFeatures.json', 'ansysMesh.json'):
        raise ValueError('Fixed meshes cannot use meshing/geometry parameter studies; use separate MESH inputs')
    sources = find_meshes(args.sim_dir/'MESH')
    if any(p.name.lower().endswith('.msh.h5') for p in sources.values()):
        import h5py  # fail before creating the order, not during a worker
    common_preflight(args)


def prepare_input(case, source, acoustic_surface, acoustic_diameter, study=None):
    from tools.snappy_pipeline import write, parameter_file
    from tools import update_parameter
    case, source = Path(case), Path(source)
    if study:
        filename, entry, value = study
        path = parameter_file(case/'Parameters', filename)
        if path.suffix == '.json':
            values = json.loads(path.read_text())
            if entry not in values:
                raise ValueError(f'Unknown study parameter {entry}')
            values[entry] = json.loads(str(value))
            write(path, json.dumps(values, indent=2)+'\n')
        elif not update_parameter(path, entry, value, quiet=True):
            raise ValueError(f'Cannot update study parameter {entry}')
    config = read_config(case/'Parameters', source)
    order_file = case.parent/'simulation_order.json'
    if order_file.is_file():
        saved = json.loads(order_file.read_text()).get('mesh_inputs', {})
        mesh_name = source.name[:-7] if source.name.lower().endswith('.msh.h5') else source.stem
        if mesh_name in saved and saved[mesh_name] != dict(sha256=digest(source), config=config):
            raise ValueError('Imported mesh or configuration changed since order creation')
    destination = case/'mesh/input'/source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record = dict(backend=BACKEND, source=str(source.resolve()), source_sha256=digest(destination),
                  input_file=destination.relative_to(case).as_posix(), config=config,
                  acoustic_surface=acoustic_surface, acoustic_diameter=acoustic_diameter,
                  layers=None, boundary_layer_flagged_cells=None)
    write(case/'mesh/import.json', json.dumps(record, indent=2)+'\n')
    control = case/'system/controlDict'
    control.write_text(re.sub(r'\bwritePrecision\s+\d+\s*;', 'writePrecision 17;', control.read_text()))
    # These generator dictionaries are not used and must not masquerade as
    # provenance for an imported mesh in the settings report.
    for name in ('snappyHexMeshDict', 'surfaceFeaturesDict'):
        (case/'system'/name).unlink(missing_ok=True)


def _zone_entries(path):
    text = Path(path).read_text()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    result = {}
    position = 0
    entry = re.compile(r'(?m)^\s*([A-Za-z_][\w.-]*)\s*\{')
    braces = re.compile(r'[{}]')
    while (match := entry.search(text, position)) is not None:
        depth = 1
        for brace in braces.finditer(text, match.end()):
            depth += 1 if brace[0] == '{' else -1
            if depth == 0:
                if match[1] != 'FoamFile':
                    result[match[1]] = text[match.end():brace.start()]
                position = brace.end()
                break
        else:
            raise ValueError(f'Unclosed zone dictionary in {path}')
    return result


def _labels(body, kind):
    match = re.search(r'\b'+kind+r'\s+(?:List<label>\s*)?(\d+)\s*\((.*?)\)\s*;', body, re.S)
    if not match:
        raise ValueError(f'Missing ASCII {kind} zone labels')
    labels = np.fromstring(match[2], sep=' ', dtype=np.int64)
    if len(labels) != int(match[1]):
        raise ValueError(f'Incorrect {kind} zone length')
    return labels


def _rename_entry(path, old, new):
    if old == new:
        return
    text = Path(path).read_text()
    if new in _zone_entries(path):
        if old not in _zone_entries(path):
            return
        raise ValueError(f'Cannot rename {old}: {new} already exists')
    text, count = re.subn(r'(?m)^(\s*)'+re.escape(old)+r'(?=\s*\{)', r'\g<1>'+new, text)
    if count != 1:
        raise ValueError(f'Expected one named entry {old} in {path}')
    Path(path).write_text(text)


def mesh_arrays(case):
    from tools.layer_thickness import _read_list
    mesh = Path(case)/'constant/polyMesh'
    count, text = _read_list(mesh/'points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ').reshape(-1, 3)
    if len(points) != count or not np.isfinite(points).all():
        raise ValueError('Invalid imported mesh points')
    count, text = _read_list(mesh/'faces')
    sizes = np.array([int(v) for v in re.findall(r'(\d+)\s*\(', text)], dtype=np.int64)
    vertices = np.fromstring(re.sub(r'\d+\s*\(', ' ', text).replace(')', ' '), sep=' ', dtype=np.int64)
    if len(sizes) != count or np.any(sizes < 3) or int(sizes.sum()) != len(vertices):
        raise ValueError('Invalid imported face list')
    if np.any(vertices < 0) or np.any(vertices >= len(points)):
        raise ValueError('Imported face node outside point range')
    offsets = np.r_[0, np.cumsum(sizes)]
    count, text = _read_list(mesh/'owner')
    owner = np.fromstring(text, sep=' ', dtype=np.int64)
    if count != len(sizes) or len(owner) != count:
        raise ValueError('Invalid owner addressing')
    count, text = _read_list(mesh/'neighbour')
    neighbour = np.fromstring(text, sep=' ', dtype=np.int64)
    if len(neighbour) != count:
        raise ValueError('Invalid neighbour addressing')
    return points, sizes, vertices, offsets, owner, neighbour


def cell_count(case):
    """The highest cell label can occur only in neighbour, not in owner."""
    from tools.layer_thickness import _read_list
    largest = -1
    for name in ('owner', 'neighbour'):
        count, text = _read_list(Path(case)/'constant/polyMesh'/name)
        values = np.fromstring(text, sep=' ', dtype=np.int64)
        if len(values) != count or np.any(values < 0):
            raise ValueError(f'Invalid {name} addressing before point cleanup')
        largest = max(largest, int(values.max(initial=-1)))
    if largest < 0:
        raise ValueError('No cells in imported mesh')
    return largest+1


def prepare_converted_mesh(case):
    """Validate the actual imported regions and derive geometry for reporting."""
    import trimesh
    from tools.snappy_pipeline import write, header
    case = Path(case)
    record = json.loads((case/'mesh/import.json').read_text())
    config = record['config']
    mesh = case/'constant/polyMesh'
    for canonical, original in config['patches'].items():
        _rename_entry(mesh/'boundary', original, canonical)
    _rename_entry(mesh/'cellZones', config['rotorCellZone'], 'rotaryRegion')
    zones = _zone_entries(mesh/'cellZones')
    rotor = _labels(zones['rotaryRegion'], 'cellLabels')
    points, sizes, vertices, offsets, owner, neighbour = mesh_arrays(case)
    ncells = int(max(owner.max(), neighbour.max(initial=-1)))+1
    if not len(rotor) or np.any(rotor < 0) or np.any(rotor >= ncells) or len(np.unique(rotor)) != len(rotor) or len(rotor) == ncells:
        raise ValueError('Rotor zone must be a nonempty, proper subset of all cells')
    mask = np.zeros(ncells, dtype=bool)
    mask[rotor] = True
    fzones = _zone_entries(mesh/'faceZones')
    interface_name = config['rotorFaceZone']
    if interface_name not in fzones:
        raise ValueError(f'Missing internal rotor faceZone {interface_name}; preserve it in Fluent')
    interface = _labels(fzones[interface_name], 'faceLabels')
    if not len(interface) or np.any(interface < 0) or np.any(interface >= len(neighbour)):
        raise ValueError('Rotor faceZone must consist only of internal faces')
    separating = np.flatnonzero(mask[owner[:len(neighbour)]] != mask[neighbour])
    if not np.array_equal(np.sort(interface), separating):
        raise ValueError('Rotor interface does not exactly enclose the rotating cell zone')
    patches = _zone_entries(mesh/'boundary')
    empty = [name for name, body in patches.items()
             if re.search(r'\bnFaces\s+0\s*;', body)]
    if empty:
        patches = {name: body for name, body in patches.items() if name not in empty}
        write(mesh/'boundary', header('boundary').replace('class dictionary;', 'class polyBoundaryMesh;')
              +str(len(patches))+'\n(\n'+''.join(name+'\n{'+body+'}\n' for name,body in patches.items())+')\n')
    if set(patches) != {'propeller','inlet','outlet','walls'}:
        raise ValueError(f'Expected only propeller/inlet/outlet/walls before baffles, got {list(patches)}')
    def patch_range(name):
        body = patches[name]
        start = int(re.search(r'\bstartFace\s+(\d+)', body)[1])
        count = int(re.search(r'\bnFaces\s+(\d+)', body)[1])
        if count < 1 or start < len(neighbour) or start+count > len(owner):
            raise ValueError(f'Empty or invalid patch {name}')
        return range(start, start+count)
    wall_faces = patch_range('propeller')
    if not np.all(mask[owner[list(wall_faces)]]):
        raise ValueError('Every propeller wall face must belong to the rotating cell zone')
    volume = 0.0
    blade_triangles = []
    # Oriented boundary integrals provide a volume independent of checkMesh.
    for name in patches:
        for face in patch_range(name):
            polygon = points[vertices[offsets[face]:offsets[face+1]]]
            centre = polygon.mean(axis=0)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                volume += float(np.dot(centre, np.cross(a, b)))/6
                if name == 'propeller' and np.linalg.norm(np.cross(a-centre, b-centre)) > 0:
                    blade_triangles.append([centre, b, a])
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError('Imported boundary has nonpositive enclosed fluid volume')
    triangles = np.asarray(blade_triangles)
    blade_points = triangles.reshape(-1, 3)
    diameter = float(np.ptp(blade_points, axis=0).max())
    if not .02 < diameter < 2:
        raise ValueError('Check ansysMesh.json scale: propeller span must be 0.02–2 m')
    source_cells = record.get('source_counts', {}).get('cells')
    if source_cells is not None and ncells != source_cells:
        raise ValueError('Import changed the source cell count')
    lower, upper = points.min(axis=0), points.max(axis=0)
    config_geometry = json.loads((case/'Parameters/snappyGeometry.json').read_text())
    radius = .5 * (record['acoustic_diameter'] if record['acoustic_surface']=='permeable'
                    else config_geometry['refinementSphereDiameter']) * diameter
    tri_dir = case/'constant/triSurface'
    tri_dir.mkdir(parents=True, exist_ok=True)
    # This surface is only a visualisation/acoustics reference, never remeshed.
    tris = trimesh.Trimesh(vertices=blade_points,
        faces=np.arange(len(blade_points)).reshape(-1,3), process=False)
    tris.export(tri_dir/'propeller.stl')
    if record['acoustic_surface']=='permeable':
        if (not math.isfinite(radius) or radius <= np.linalg.norm(blade_points,axis=1).max()
                or np.any(lower >= -radius) or np.any(upper <= radius)):
            raise ValueError('Permeable sampling sphere must enclose the blade and fit inside the imported domain; choose a smaller --acoustic-sphere-diameter')
        trimesh.creation.icosphere(subdivisions=4, radius=radius).export(tri_dir/'permeableSurface.stl')
    x0,y0,z0 = lower; x1,y1,z1 = upper
    corners = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
               (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    write(case/'Parameters/blockMesh.generated', 'vertices\n(\n'+'\n'.join(
        '('+' '.join(format(float(v),'.17g') for v in p)+')' for p in corners)+'\n);\n')
    geometry = dict(backend=BACKEND, source=record['source'], source_sha256=record['source_sha256'],
        scale=config['scale'], diameter_m=diameter, box_min=lower.tolist(), box_max=upper.tolist(),
        axis=[0,1,0], origin=[0,0,0], expected_fluid_volume_m3=volume,
        layers=record['layers'], boundary_layer_flagged_cells=record.get('boundary_layer_flagged_cells'),
        acoustic_sphere_radius_m=radius if record['acoustic_surface']=='permeable' else None,
        settings=config_geometry, input_cells=ncells, rotor_cells=int(len(rotor)),
        reference_surface='Extracted from the imported mesh; no original STL required')
    write(case/'mesh/geometry.json', json.dumps(geometry,indent=2)+'\n')
    write(case/'system/createBafflesDict', header('createBafflesDict')+f'''
internalFacesOnly true;
baffles
{{
    rotor
    {{
        type faceZone;
        zoneName {interface_name};
        owner {{ name rotaryRegion; type wall; }}
        neighbour {{ name rotaryRegion_slave; type wall; }}
    }}
}}
''')
    return ncells


def run_mesh(container, case, cores, allow_bad, callback=None, live=False):
    """Never invokes blockMesh, surfaceFeatures or snappyHexMesh."""
    from tools.snappy_pipeline import docker_run, finish_rotor_mesh, header, write
    case = Path(case)
    if callable(container):
        container = container()
    def run(command, log):
        docker_run(container, case, command, log, callback, live)
    path = case/'mesh/import.json'
    record = json.loads(path.read_text())
    source = case/record['input_file']
    if digest(source) != record['source_sha256']:
        raise ValueError('Case input mesh checksum mismatch')
    if source.name.lower().endswith('.msh.h5'):
        from tools.fluent_hdf5 import export_mesh
        destination = case/'mesh/input.msh'
        counts = export_mesh(source, destination)
        record.update(source_counts=counts, layers=counts['layers'],
                      boundary_layer_flagged_cells=counts['boundary_layer_flagged_cells'])
        write(path, json.dumps(record, indent=2)+'\n')
    else:
        destination = source
        with source.open('rb') as stream:
            beginning = stream.read(2048)
        if b'\x00' in beginning or b'(3010' in beginning or b'(2010' in beginning:
            raise ValueError('Legacy Fluent meshes must be ASCII; binary files are unsupported')
    run(['fluent3DMeshToFoam', destination.relative_to(case).as_posix(),
         '-scale', str(record['config']['scale'])], 'log.fluent3DMeshToFoam')
    # Preserve double precision when OpenFOAM writes this imported mesh.
    # All-cell subsetting removes only orphaned export points, not cells.
    ncells = cell_count(case)
    all_cells = case/'constant/polyMesh/sets/importAllCells'
    write(all_cells, header('importAllCells').replace('class dictionary;', 'class cellSet;')
          +str(ncells)+'\n(\n'+''.join(str(i)+'\n' for i in range(ncells))+')\n')
    run(['subsetMesh','-cellSet','importAllCells','-noFields'], 'log.removeUnusedPoints')
    validated_cells = prepare_converted_mesh(case)
    if validated_cells != ncells:
        raise ValueError('Point cleanup changed the cell count')
    run(['checkMesh'], 'log.checkMesh.import')
    from tools.snappy_pipeline import mesh_ok
    if not mesh_ok((case/'log.checkMesh.import').read_text()) and not allow_bad:
        raise ValueError('Imported mesh fails checkMesh; no solver was started')
    run(['createBaffles'], 'log.createBaffles')
    run(['splitBaffles'], 'log.splitBaffles')
    return finish_rotor_mesh(container,case,allow_bad,callback,live)

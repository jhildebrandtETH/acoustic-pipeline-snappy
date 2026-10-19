"""Foundation OpenFOAM 13 snappy meshing with configurable native layer generation."""
import json
import math
import os
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parent.parent
BACKEND = "snappy-native-v1"
COMPATIBLE_BACKENDS = (BACKEND, "snappy-no-layers-v1")


def safe_path(path):
    path = Path(path).expanduser().resolve()
    def key(p):
        value = p.as_posix().rstrip('/')
        return value.casefold() if os.name == 'nt' or re.match(r'^/mnt/[a-zA-Z]/', value) else value
    for name in ('.git', '.codex', '.agents', 'tools', 'tests', 'examples',
                 'Parameters', 'CoreTemplates', 'acousticSolver'):
        protected = key((ROOT/name).resolve())
        if key(path) == protected or key(path).startswith(protected + '/'):
            raise ValueError(f'Source directory cannot contain simulation output: {ROOT/name}')
    if path == ROOT or path == Path(path.anchor):
        raise ValueError('Choose a dedicated simulation directory, not the pipeline or filesystem root')
    return path


def parameter_file(parameters, name, *, read_only=False):
    if not name or Path(name).name != name or '\\' in name or name in {'.', '..'}:
        raise ValueError('Study file must be a filename in Parameters, without a path')
    for candidate in (Path(parameters)/name, Path(parameters)/(name+'.cpp')):
        if not read_only:
            safe_path(candidate)
        if candidate.is_file():
            return candidate
    raise ValueError(f'Study dictionary not found in {parameters}: {name}')


def preflight(args):
    safe_path(args.sim_dir)
    from tools.plotting import pyplot
    if not getattr(args, 'slurm', False):
        import docker
        client = docker.from_env()
        try:
            client.ping()
            client.images.get('microfluidica/openfoam:13')
        finally:
            client.close()
    if not args.mesh_only:
        import postprocessing
    if args.study:
        path = parameter_file(ROOT/'Parameters', args.study_file, read_only=True)
        if path.suffix == '.json':
            if args.study_parameter not in json.loads(path.read_text()):
                raise ValueError(f'Unknown study parameter: {args.study_parameter}')
        else:
            from tools.settings_report import read_dictionary, resolve_dictionary
            data = resolve_dictionary(read_dictionary(path, ROOT, {}, []))
            if args.study_parameter not in data:
                raise ValueError(f'Unknown study parameter: {args.study_parameter}')


def write(path, text):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def header(name):
    return f'FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n'


def configured_layers(case):
    """Report the requested switch, not a guarantee of complete layer coverage."""
    from tools.settings_report import read_dictionary, resolve_dictionary
    path = Path(case)/'system/snappyHexMeshDict'
    if not path.is_file():
        return False
    data = resolve_dictionary(read_dictionary(path, case, {}, []))
    value = data.get('addLayers', 'false').lower()
    if value not in ('true', 'yes', 'on', '1', 'false', 'no', 'off', '0'):
        raise ValueError(f'Cannot resolve addLayers in {path}: {value}')
    return value in ('true', 'yes', 'on', '1')


def prepare_geometry(case, source, acoustic_surface, acoustic_diameter, study=None):
    import numpy as np
    import trimesh
    from tools.geometry import read_stl
    from tools import update_parameter
    case = safe_path(case)
    if study:
        filename, entry, value = study
        path = parameter_file(case/'Parameters', filename)
        if path.suffix == '.json':
            data = json.loads(path.read_text())
            if entry not in data:
                raise ValueError(f'Unknown study parameter: {entry}')
            data[entry] = json.loads(str(value))
            write(path, json.dumps(data, indent=2)+'\n')
        elif not update_parameter(path, entry, value, quiet=True):
            raise ValueError(f'Cannot update study parameter {entry} in {path}')
    config = json.loads((case/'Parameters/snappyGeometry.json').read_text())
    positive = ('scale', 'rotorRadiusFactor', 'rotorHalfLength', 'refinementSphereDiameter',
                'lateralMargin', 'inletMargin', 'maxRelativeVolumeError')
    if any(not math.isfinite(config[k]) or config[k] <= 0 for k in positive):
        raise ValueError('Geometry dimensions, margins and volume tolerance must be finite and positive')
    segments = config['cylinderSegments']
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 32:
        raise ValueError('cylinderSegments must be an integer of at least 32')
    if not 0 < config['inletFraction'] < 1:
        raise ValueError('inletFraction must be between zero and one')
    for key in ('minimumFaceCoverage', 'minimumAverageCoverage'):
        if not 0 < config[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    vertices, faces, volume, digest = read_stl(Path(source), config['scale'])
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    diameter = float(np.ptp(vertices, axis=0).max())
    if not .02 < diameter < 2:
        raise ValueError('Check STL scale: measured propeller span must be 0.02–2 m')
    radius = config['rotorRadiusFactor'] * diameter
    half = config['rotorHalfLength']
    if np.hypot(vertices[:, 0], vertices[:, 2]).max() >= radius*math.cos(math.pi/segments):
        raise ValueError('Propeller swept radius does not fit inside rotor cylinder')
    if np.abs(vertices[:, 1]).max() >= half:
        raise ValueError('Propeller does not fit axially inside rotor cylinder')
    multiplier = acoustic_diameter if acoustic_surface == 'permeable' else config['refinementSphereDiameter']
    if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Sphere diameter multiplier must be finite and positive')
    sphere_radius = .5 * multiplier * diameter
    if sphere_radius <= np.linalg.norm(vertices, axis=1).max():
        raise ValueError('Acoustic/refinement sphere must enclose the complete propeller')
    lateral = sphere_radius * (1 + config['lateralMargin'])
    inlet = sphere_radius * (1 + config['inletMargin'])
    outlet = inlet * (1-config['inletFraction'])/config['inletFraction']
    lower, upper = [-lateral, -outlet, -lateral], [lateral, inlet, lateral]
    if min(lateral, inlet, outlet) <= sphere_radius or min(inlet, outlet) <= half or lateral <= radius:
        raise ValueError('Domain must enclose the rotor and refinement sphere')
    surface = case/'constant/triSurface'
    surface.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(surface/'propeller.stl')
    rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    for name, r, h, offset in (
        ('rotaryCylinder', radius, 2*half, 0),
        ('innerCylinder', .6*sphere_radius, .52*sphere_radius, -.12*sphere_radius),
        ('outerCylinder', .8*sphere_radius, .72*sphere_radius, -.12*sphere_radius),
    ):
        cylinder = trimesh.creation.cylinder(radius=r, height=h, sections=segments)
        cylinder.apply_transform(rotation)
        cylinder.apply_translation([0, offset, 0])
        cylinder.export(surface/(name+'.stl'))
    # This is always a refinement surface; sampling is independently enabled.
    trimesh.creation.icosphere(subdivisions=4, radius=sphere_radius).export(surface/'permeableSurface.stl')
    x0,y0,z0 = lower
    x1,y1,z1 = upper
    corners = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
               (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    block = 'scale 1;\nvertices\n(\n' + '\n'.join('('+' '.join(map(str,p))+')' for p in corners)
    block += '\n);\nblocks (hex (0 1 2 3 4 5 6 7) $blocks_resolution simpleGrading (1 1 1));\nedges ();\n'
    block += '''boundary (
walls { type wall; faces ((2 6 5 1) (0 3 2 1) (0 4 7 3) (4 5 6 7)); }
inlet { type patch; faces ((3 7 6 2)); }
outlet { type patch; faces ((1 5 4 0)); }
);
'''
    write(case/'Parameters/blockMesh.generated', block)
    location = (.013*lateral, -.5*(outlet+half), .017*lateral)
    write(case/'Parameters/snappyGeometry.generated', 'locationInMesh ('+' '.join(map(str,location))+');\n')
    from tools.snappy_features import prepare_features
    features = prepare_features(case, source, config['scale'])
    write(case/'mesh/geometry.json', json.dumps(dict(
        backend=BACKEND, layers=configured_layers(case), source=str(source), source_sha256=digest,
        scale=config['scale'], diameter_m=diameter, box_min=lower, box_max=upper,
        rotor_radius_m=radius, rotor_half_length_m=half, axis=[0,1,0], origin=[0,0,0],
        expected_fluid_volume_m3=float(np.prod(np.array(upper)-lower))-abs(volume),
        acoustic_sphere_radius_m=sphere_radius if acoustic_surface == 'permeable' else None,
        refinement_sphere_radius_m=sphere_radius, settings=config, features=features,
    ), indent=2)+'\n')
    (case/'sim.foam').touch()


def docker_run(container, case, command, log_name, callback=None, live=False):
    from tools import safe_exec, report_case_stage
    from tools.mesh_point_format import normalize_point_signs
    points = Path(case)/'constant/polyMesh/points'
    if points.is_file():
        normalize_point_signs(points)
    cmd = 'source /opt/openfoam13/etc/bashrc && set -o pipefail && cd /simulation && '
    cmd += shlex.join(command)+' 2>&1 | tee '+shlex.quote(log_name)
    report_case_stage(callback, command[0], shlex.join(command))
    if not safe_exec(container, ['bash', '-c', cmd], command[0],
                     print_output=live, status_callback=callback):
        raise RuntimeError(f'{command[0]} failed; see {Path(case)/log_name}')
    if points.is_file():
        normalize_point_signs(points)


def patch_info(case):
    text = (Path(case)/'constant/polyMesh/boundary').read_text()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    patches = {}
    for name, body in re.findall(r'(\w+)\s*\{([^{}]*)\}', text):
        count = re.search(r'nFaces\s+(\d+)', body)
        if count:
            patches[name] = dict(nFaces=int(count[1]), type=re.search(r'type\s+(\w+)', body)[1])
    return patches


def mesh_ok(text):
    return 'Mesh OK.' in text and not re.search(r'Failed\s+\d+\s+mesh checks', text)


def run_mesh(container, case, cores, allow_bad, callback=None, live=False):
    case = safe_path(case)
    if callable(container):
        container = container()
    def run(command, log):
        docker_run(container, case, command, log, callback, live)
    run(['blockMesh'], 'log.blockMesh')
    run(['surfaceFeatures'], 'log.surfaceFeatures')
    if cores > 1:
        run(['decomposePar', '-copyZero'], 'log.decomposePar.mesh')
        run(['mpirun', '--allow-run-as-root', '--use-hwthread-cpus', '-np', str(cores),
             'snappyHexMesh', '-parallel', '-overwrite'], 'log.snappyHexMesh')
        run(['reconstructPar', '-constant', '-noFields'], 'log.reconstructParMesh')
    else:
        run(['snappyHexMesh', '-overwrite'], 'log.snappyHexMesh')
    # Keep interface points shared through snapping, then release rotor motion.
    run(['splitBaffles', '-overwrite'], 'log.splitBaffles')
    return finish_rotor_mesh(container,case,allow_bad,callback,live)


def finish_rotor_mesh(container, case, allow_bad, callback=None, live=False):
    """Validate mesh quality and couple the rotating snappy interface."""
    def run(command, log):
        docker_run(container, case, command, log, callback, live)
    patches = patch_info(case)
    expected = {'inlet', 'outlet', 'walls', 'propeller', 'rotaryRegion', 'rotaryRegion_slave'}
    if {name for name, data in patches.items() if data['nFaces']} != expected:
        raise ValueError(f'Unexpected snappy patches: {patches}')
    if not re.search(r'\brotaryRegion\b', (case/'constant/polyMesh/cellZones').read_text()):
        raise ValueError('snappy mesh has no rotaryRegion cellZone')
    run(['checkMesh', '-allGeometry', '-allTopology'], 'log.checkMesh.extended')
    run(['checkMesh'], 'log.checkMesh')
    text = (case/'log.checkMesh').read_text()
    geometry = json.loads((case/'mesh/geometry.json').read_text())
    settings = geometry['settings']
    measured = re.search(r'Total volume\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', text)
    if not measured or 'Zero or negative cell volume detected' in text:
        raise ValueError('Missing or invalid fluid volume; inspect log.checkMesh')
    volume = float(measured[1])
    expected_volume = geometry['expected_fluid_volume_m3']
    if not math.isfinite(volume) or abs(volume-expected_volume)/expected_volume > settings['maxRelativeVolumeError']:
        raise ValueError('Mesh fluid volume differs from the input geometry; inspect log.checkMesh')
    quality = mesh_ok(text)
    status = dict(backend=geometry.get('backend',BACKEND), layers=configured_layers(case), mesh_quality_ok=quality,
                  extended_mesh_diagnostics_ok=mesh_ok((case/'log.checkMesh.extended').read_text()),
                  allow_bad_mesh=allow_bad, patches=patches)
    write(case/'mesh/mesh-status.json', json.dumps(status, indent=2)+'\n')
    if not quality and not allow_bad:
        raise ValueError('Mesh quality checks failed; inspect log.checkMesh. No solver was started.')
    run(['createNonConformalCouples', '-fields', 'rotaryRegion_slave', 'rotaryRegion'], 'log.createNonConformalCouples')
    run(['checkMesh'], 'log.checkMesh.NCC')
    coupling = (case/'log.createNonConformalCouples').read_text()
    coverage = re.findall(r'(?:Source|Target) min/average/max coverage = ([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)', coupling)
    coverage_ok = len(coverage) == 2 and all(
        float(lo) >= settings['minimumFaceCoverage'] and float(avg) >= settings['minimumAverageCoverage']
        for lo, avg, hi in coverage)
    write(case/'mesh/ncc-status.json', json.dumps(dict(coverage=coverage, coverage_ok=coverage_ok), indent=2)+'\n')
    if not coverage_ok:
        raise ValueError('NCC interface coverage is inadequate; inspect mesh/ncc-status.json')
    if sum(p['type'] == 'nonConformalCyclic' for p in patch_info(case).values()) != 2:
        raise ValueError('Expected exactly two nonConformalCyclic patches after coupling')
    ncc_ok = mesh_ok((case/'log.checkMesh.NCC').read_text())
    status.update(ncc_mesh_quality_ok=ncc_ok, mesh_quality_ok=quality and ncc_ok)
    write(case/'mesh/mesh-status.json', json.dumps(status, indent=2)+'\n')
    if not ncc_ok and not allow_bad:
        raise ValueError('Post-NCC mesh checks failed; inspect log.checkMesh.NCC')
    (case/'sim.foam').touch()
    return quality and ncc_ok

"""Shared case-safety and NCC helpers retained from the original pipeline.

The run_mesh compatibility entry point imports Fluent meshes in this copy.
"""
import json
import math
import os
from pathlib import Path
from tools.runtime import IMAGE
import re
import shlex

ROOT = Path(__file__).resolve().parent.parent
BACKEND = "ansys-import-v1"
COMPATIBLE_BACKENDS = (BACKEND,)


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
            client.images.get(IMAGE)
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
    imported = Path(case)/'mesh/import.json'
    if imported.is_file():
        return json.loads(imported.read_text()).get('layers')
    from tools.settings_report import read_dictionary, resolve_dictionary
    path = Path(case)/'system/snappyHexMeshDict'
    if not path.is_file():
        return False
    data = resolve_dictionary(read_dictionary(path, case, {}, []))
    value = data.get('addLayers', 'false').lower()
    if value not in ('true', 'yes', 'on', '1', 'false', 'no', 'off', '0'):
        raise ValueError(f'Cannot resolve addLayers in {path}: {value}')
    return value in ('true', 'yes', 'on', '1')


def docker_run(container, case, command, log_name, callback=None, live=False):
    import time
    from tools import safe_exec, report_case_stage
    from tools.mesh_point_format import normalize_point_signs
    points = Path(case)/'constant/polyMesh/points'
    started = time.time()
    elapsed_start = time.perf_counter()
    phases = {}
    if points.is_file():
        report_case_stage(callback, 'meshPointFormat', 'checking point format before '+command[0])
        normalize_point_signs(points)
    phases['point_check_before_s'] = time.perf_counter() - elapsed_start
    cmd = 'source /opt/openfoam13/etc/bashrc && set -o pipefail && cd /simulation && '
    cmd += shlex.join(command)+' 2>&1 | tee '+shlex.quote(log_name)
    report_case_stage(callback, command[0], shlex.join(command))
    command_started = time.perf_counter()
    succeeded = False
    try:
        succeeded = safe_exec(container, ['bash', '-c', cmd], command[0],
                              print_output=live, status_callback=callback)
        phases['utility_s'] = time.perf_counter() - command_started
        if not succeeded:
            raise RuntimeError(f'{command[0]} failed; see {Path(case)/log_name}')
        post_started = time.perf_counter()
        if points.is_file():
            report_case_stage(callback, 'meshPointFormat', 'checking point format after '+command[0])
            normalize_point_signs(points)
        phases['point_check_after_s'] = time.perf_counter() - post_started
    finally:
        phases.setdefault('utility_s', time.perf_counter() - command_started)
        total = time.perf_counter() - elapsed_start
        # Timing diagnostics must never turn a successful utility into a failure.
        try:
            with (Path(case)/'preparation-timings.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(command=command, log=log_name,
                    started_epoch_s=started, total_s=total,
                    utility_succeeded=bool(succeeded), **phases))+'\n')
        except OSError:
            pass
    report_case_stage(callback, command[0], f'{command[0]} complete in {total:.2f}s')


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
    from tools.ansys_mesh import run_mesh as import_mesh
    return import_mesh(container, case, cores, allow_bad, callback, live)


def finish_rotor_mesh(container, case, allow_bad, callback=None, live=False):
    """Validate mesh quality and couple the rotating snappy interface."""
    def run(command, log):
        docker_run(container, case, command, log, callback, live)
    patches = patch_info(case)
    expected = {'inlet', 'outlet', 'walls', 'propeller', 'rotaryRegion', 'rotaryRegion_slave'}
    if {name for name, data in patches.items() if data['nFaces']} != expected:
        raise ValueError(f'Unexpected rotor mesh patches: {patches}')
    if not re.search(r'\brotaryRegion\b', (case/'constant/polyMesh/cellZones').read_text()):
        raise ValueError('Imported mesh has no rotaryRegion cellZone')
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

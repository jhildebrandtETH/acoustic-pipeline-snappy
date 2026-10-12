"""Snappy-only meshing and reports for supplied non-rotating cases."""
import shutil

from tools.mesh_runner import Runner
from tools.common import _atomic_write_json as write_json


def run_simple_mesh_pipeline(args):
    if not args.mesh_case or not args.mesh_only:
        raise ValueError('--mesh-case and --mesh-only are required for supplied-case meshing')
    if args.resume or args.allow_bad_mesh or args.study or args.rpms or args.mode or args.turbulence or args.wall_functions or args.aerodynamics_only or args.total_cores or args.cores_per_case or args.field_init != 'off':
        raise ValueError('Simple mesh mode is serial and mesh-only; resume, quality overrides and solver/order options are unsupported')
    source, output = args.mesh_case.resolve(), args.sim_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input case and output directory must be separate and non-nested')
    for name in ('blockMeshDict', 'snappyHexMeshDict', 'controlDict'):
        if not (source/'system'/name).is_file():
            raise ValueError(f'Missing input dictionary: {source / "system" / name}')
    runner = Runner(docker=not args.mesh_native)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(completed=False, accepted=False, stage='copying inputs', source=str(source),
                  backend='snappy-simple-v1')
    def stage(name):
        status['stage'] = name
        write_json(output/'mesh-pipeline-status.json', status)
        print(name, flush=True)
    try:
        stage('copying inputs')
        base = output/'base'
        base.mkdir()
        shutil.copytree(source/'system', base/'system')
        if (source/'constant').is_dir():
            shutil.copytree(source/'constant', base/'constant', ignore=shutil.ignore_patterns('polyMesh'))
        if (source/'Parameters').is_dir():
            shutil.copytree(source/'Parameters', base/'Parameters')
        from tools.snappy_pipeline import configured_layers
        status['layers'] = configured_layers(base)
        stage('running snappyHexMesh')
        runner.run(base, 'blockMesh', [], 'log.blockMesh')
        if (base/'system/surfaceFeaturesDict').is_file():
            runner.run(base, 'surfaceFeatures', [], 'log.surfaceFeatures')
        runner.run(base, 'snappyHexMesh', ['-overwrite'], 'log.snappyHexMesh')
        (base/'sim.foam').touch()
        final = base
        runner.run(final, 'checkMesh', ['-constant', '-allGeometry', '-allTopology'], 'log.checkMesh')
        status.update(mesh_accepted=True, final_case=str(final))
        stage('generating mesh quality report')
        parameters = final/'Parameters'
        parameters.mkdir(exist_ok=True)
        from tools.report_worker import create_simulation_report
        pdf = final/'report/simulation_report.pdf'
        create_simulation_report(case_path=final, rpm=0, mode='mesh-only',
                                 turbulence_model='none', mesh_only=True, quiet=True,
                                 output_pdf=pdf, latest_report_path=output/'mesh_report.pdf')
        status.update(completed=True, accepted=True, report=str(output/'mesh_report.pdf'))
        stage('complete')
        print(f"Mesh: {final / 'sim.foam'}\nReport: {output / 'mesh_report.pdf'}", flush=True)
        return status
    except Exception as exc:
        status.update(completed=False, accepted=False, error=str(exc))
        write_json(output/'mesh-pipeline-status.json', status)
        raise

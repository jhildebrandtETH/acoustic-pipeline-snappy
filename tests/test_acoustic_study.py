"""Multi-surface recording, post-solver isolation and common-window comparison."""
import json
import io
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import trimesh

from tools.acoustic_study import (
    diameter_factors, sphere_key, configure_recording, prepare_view,
    compare_cached, run_study,
)
from tools.cli import create_parser
from tools.common import validate_acoustic_arguments
from tools.settings_report import read_dictionary, resolve_dictionary
from tools.snappy_pipeline import ROOT, prepare_geometry


class AcousticStudyTests(unittest.TestCase):
    @staticmethod
    def fixture_link(source, destination):
        if destination.exists():
            return
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    def test_exact_descending_range_and_validation(self):
        values = diameter_factors(sweep=['2.5', '1.1', '-0.1'])
        self.assertEqual(len(values), 15)
        self.assertEqual(values, [i/10 for i in range(25, 10, -1)])
        for sweep in (['2.5', '1.1', '.1'], ['2', '1.1', '-.2'], ['2', '1', '0'], ['nan', '2', '.1']):
            with self.subTest(sweep=sweep), self.assertRaises(ValueError):
                diameter_factors(sweep=sweep)
        for values in ([1.0], [2, 2], [float('inf')], [0], []):
            with self.assertRaises(ValueError):
                diameter_factors(values)

    def test_cli_uses_largest_factor_for_shared_mesh(self):
        parser = create_parser()
        args = parser.parse_args(['--sim-dir', 'output/study', '--acoustic-surface', 'permeable',
                                  '--acoustic-sphere-range', '2.5', '1.1', '-0.1'])
        validate_acoustic_arguments(parser, args)
        self.assertEqual(args.acoustic_sphere_diameter, 2.5)
        self.assertEqual(len(args.acoustic_sphere_diameters), 15)
        for extra in (['--acoustic-sphere-diameter', '2'], ['--aerodynamics-only']):
            args = parser.parse_args(['--sim-dir', 'output/study', '--acoustic-surface', 'permeable',
                                      '--acoustic-sphere-diameters', '2.5', '1.1'] + extra)
            with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                validate_acoustic_arguments(parser, args)

    def test_each_template_records_independent_every_step_p_u(self):
        factors = diameter_factors(sweep=['2.5', '1.1', '-.1'])
        for template in (ROOT/'CoreTemplates').glob('*/*/*/system/controlDict'):
            with self.subTest(template=template), tempfile.TemporaryDirectory() as tmp:
                case = Path(tmp)
                (case/'system').mkdir()
                (case/'constant/triSurface').mkdir(parents=True)
                shutil.copytree(ROOT/'Parameters', case/'Parameters')
                shutil.copyfile(template, case/'system/controlDict')
                control = case/'Parameters/controlDict.cpp'
                control.write_text(control.read_text().replace('permeableEnabled no', 'permeableEnabled yes'))
                configure_recording(case, factors, .254)
                data = resolve_dictionary(read_dictionary(case/'system/controlDict', case, {}, []))
                manifest = json.loads((case/'acoustic-surfaces.json').read_text())
                self.assertEqual(len(manifest['surfaces']), 15)
                for item in manifest['surfaces']:
                    writer = data['functions'][item['writer']]
                    self.assertEqual(writer['enabled'], 'yes')
                    self.assertEqual(writer['writeControl'], 'timeStep')
                    self.assertEqual(writer['writeInterval'], '1')
                    self.assertEqual(writer['fields'], '(p U)')
                    self.assertEqual(writer['surfaces']['permeableSurface']['surface'], item['stl'])
                    if item['factor'] != 2.5:
                        mesh = trimesh.load(case/'constant/triSurface'/item['stl'], force='mesh')
                        self.assertTrue(mesh.is_watertight)
                        self.assertAlmostEqual(np.linalg.norm(mesh.vertices, axis=1).max(), item['radius_m'], places=6)

    def fixture(self, root):
        case = root/'blade_6000RPM_AMI'
        shutil.copytree(ROOT/'CoreTemplates/AMI/kOmegaSST/wall-functions', case)
        shutil.copytree(ROOT/'Parameters', case/'Parameters')
        source = root/'blade.stl'
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.08)
        mesh.vertices[:, 1] *= .1
        mesh.export(source)
        prepare_geometry(case, source, 'permeable', 2.5, acoustic_diameters=[2.5, 1.1])
        (case/'report').mkdir()
        (case/'postProcessing').mkdir()
        manifest = json.loads((case/'acoustic-surfaces.json').read_text())
        for item in manifest['surfaces']:
            for time in ('0.1', '0.2', '0.3'):
                folder = case/'postProcessing'/item['writer']/time
                folder.mkdir(parents=True)
                (folder/'permeableSurface.vtk').write_text(str(item['factor']))
        return case, manifest

    def test_small_sphere_must_enclose_entire_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, _ = self.fixture(Path(tmp))
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=.08)
            mesh.vertices[:, 1] *= .1
            mesh.apply_translation([.05, 0, 0])
            source = Path(tmp)/'offset.stl'
            mesh.export(source)
            geometry = case/'Parameters/snappyGeometry.json'
            config = json.loads(geometry.read_text())
            config['rotorRadiusFactor'] = 1.0
            geometry.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, '1.1D must enclose'):
                prepare_geometry(case, source, 'permeable', 2.5, acoustic_diameters=[2.5, 1.1])

    def test_study_routes_each_history_to_own_report_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, manifest = self.fixture(Path(tmp))
            (case/'report/acoustic-pressure.npz').write_bytes(b'base report unchanged')
            # Windows test hosts may lack symlink permission. Small fixture copies
            # exercise the same path routing without requiring that privilege.
            with patch('tools.acoustic_study._link', side_effect=self.fixture_link), \
                    patch('acoustic_propagation.run_acoustic_solver') as solve, \
                    patch('tools.report_worker.create_simulation_report') as report, \
                    patch('tools.acoustic_study.compare_cached') as compare:
                entries = run_study(case, 6000, 'AMI', 'kOmegaSST')
            self.assertEqual(solve.call_count, 2)
            self.assertEqual(report.call_count, 2)
            self.assertEqual(compare.call_count, 1)
            for call, item in zip(solve.call_args_list, manifest['surfaces']):
                view = call.args[1]
                data = view/'postProcessing/writePermeableSurfaceFields/0.1/permeableSurface.vtk'
                self.assertEqual(data.read_text(), str(item['factor']))
                self.assertIsNone(call.kwargs['RESPONSE_DELAY_S'])
                config = (view/'Parameters/acousticComparison.json').resolve()
                self.assertTrue(config.is_relative_to(view))
                self.assertEqual(json.loads((view/'mesh/geometry.json').read_text())['acoustic_sphere_radius_m'], item['radius_m'])
            self.assertNotEqual(entries[0]['report'], entries[1]['report'])
            self.assertEqual((case/'report/acoustic-pressure.npz').read_bytes(), b'base report unchanged')

    def test_volume_workflow_uses_existing_sampler_and_retained_field_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, _ = self.fixture(Path(tmp))
            sampled = Path(tmp)/'sampled'
            for name in ('0.1', '0.11', '0.12'):
                folder = sampled/name
                folder.mkdir(parents=True)
                (folder/'volume.vtk').write_text('sample')
            series = SimpleNamespace(input_dir=sampled, surface_file='volume.vtk')
            with patch('tools.acoustic_study._link', side_effect=self.fixture_link), \
                    patch('acousticSolver.src.foamacoustics.interpolate_openfoam_fields', return_value=series) as sample, \
                    patch('acoustic_propagation.run_acoustic_solver') as solve, \
                    patch('tools.report_worker.create_simulation_report') as report, \
                    patch('tools.acoustic_study.compare_cached'):
                entries = run_study(case, 6000, 'AMI', 'kOmegaSST', source='volume', factors=[1.5], time_selection='0.1:0.12')
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(sample.call_args.kwargs['time_selection'], '0.1:0.12')
            self.assertAlmostEqual(sample.call_args.args[1].radius_m, .12)
            self.assertIs(solve.call_args.kwargs['SURFACE_SERIES'], series)
            self.assertAlmostEqual(solve.call_args.kwargs['SAMPLE_INTERVAL_S'], .01)
            self.assertIsNone(solve.call_args.kwargs['RESPONSE_DELAY_S'])
            view = solve.call_args.args[1]
            self.assertEqual((view/'Parameters/controlDict.cpp').read_bytes(), (case/'Parameters/controlDict.cpp').read_bytes())
            self.assertIn('volume', entries[0]['report'])
            report.assert_called_once()

    def test_normal_postprocessing_automatically_builds_study_reports(self):
        import postprocessing
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case/'acoustic-surfaces.json').write_text('{}')
            with patch.object(postprocessing, 'create_simulation_report'), \
                    patch.object(postprocessing, 'run_visualization'), \
                    patch.object(postprocessing, 'merge_postprocessing_dat_files'), \
                    patch('acoustic_propagation.run_acoustic_solver'), \
                    patch('tools.acoustic_study.run_study') as study:
                postprocessing.postprocessing('permeable', case, 6000, 'AMI', 'kOmegaSST')
            study.assert_called_once_with(case, 6000, 'AMI', 'kOmegaSST', status_callback=None)

    def test_prebuilt_volume_series_reaches_solver_without_reference_geometry_rewrite(self):
        import acoustic_propagation as stage
        import torch
        from acousticSolver.src.foamacoustics.core.result import F1AResult
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'blade_6000RPM_AMI'
            case.mkdir()
            sampled = Path(tmp)/'volume-samples'
            times = np.arange(81)/800
            for time in times:
                folder = sampled/str(time)
                folder.mkdir(parents=True)
                (folder/'sample.vtk').write_text('mock volume sample')
            series = SimpleNamespace(input_dir=sampled, surface_file='sample.vtk')
            values = torch.tensor(np.sin(2*np.pi*200*times), dtype=torch.float64)
            result = F1AResult(observer_times_s=torch.tensor(times, dtype=torch.float64),
                thickness_pressure_pa=torch.zeros_like(values), loading_pressure_pa=values,
                total_pressure_pa=values, observer_m=torch.tensor([1.,0.,0.]), rpm=6000,
                maximum_delay_s=torch.tensor(.004), response_delay_s=None,
                source_time_count=len(times), propagation_algorithm='nonuniform')
            solver = Mock()
            solver.predict.return_value = result
            def comparison(case_path, config):
                (case_path/'report/acoustic-simulated.png').write_bytes(b'mock spectrum')
                return {'status': 'complete'}
            with patch.object(stage, 'load_config', return_value={'microphone_channel': '1'}), \
                    patch.object(stage, 'F1ASolver', return_value=solver) as constructor, \
                    patch.object(stage, 'create_reference_geometry_vtk_series') as rewrite, \
                    patch.object(stage, 'comparison_or_status', side_effect=comparison):
                stage.run_acoustic_solver('permeable', case, 6000, SURFACE_SERIES=series,
                                          RESPONSE_DELAY_S=None, SAMPLE_INTERVAL_S=1/800)
            self.assertIs(constructor.call_args.args[0], series)
            self.assertIsNone(constructor.call_args.kwargs['response_delay_s'])
            self.assertFalse(constructor.call_args.kwargs['moving_surface'])
            rewrite.assert_not_called()
            self.assertEqual(json.loads((case/'report/acoustic-status.json').read_text())['status'], 'complete')
            with np.load(case/'report/acoustic-pressure.npz') as cache:
                self.assertAlmostEqual(float(cache['sampling_frequency_hz']), 800.)

    def test_mismatched_histories_rejected_before_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, manifest = self.fixture(Path(tmp))
            (case/'postProcessing'/manifest['surfaces'][1]['writer']/'0.1/permeableSurface.vtk').unlink()
            with patch('acoustic_propagation.run_acoustic_solver') as solve:
                with self.assertRaisesRegex(ValueError, 'identical source timestamps'):
                    run_study(case, 6000, 'AMI', 'kOmegaSST')
                solve.assert_not_called()

    def test_common_window_comparison_of_known_tones(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            entries = []
            for factor, amplitude, first in ((2.5, 1., 0.), (1.1, 2., .02)):
                report = output/sphere_key(factor)
                report.mkdir()
                times = np.arange(round(first*10000), 2001)/10000
                np.savez(report/'acoustic-pressure.npz', time_s=times,
                         pressure_pa=amplitude*np.sin(2*math.pi*500*times),
                         sampling_frequency_hz=10000., observer_m=[1.,0.,0.])
                entries.append(dict(factor=factor, report=str(report)))
            result = compare_cached(entries, output, 6000)
            self.assertGreaterEqual(result['common_start_s'], .02)
            self.assertAlmostEqual(result['surfaces'][1]['oaspl_db']-result['surfaces'][0]['oaspl_db'], 20*math.log10(2), places=6)
            with np.load(output/'sphere-comparison.npz') as data:
                peak = data['frequencies_hz'][np.argmax(data[sphere_key(2.5)])]
                self.assertAlmostEqual(peak, 500, delta=result['frequency_resolution_hz'])
            self.assertTrue((output/'sphere-comparison.png').is_file())


if __name__ == '__main__':
    unittest.main()

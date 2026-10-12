"""Explicit OBJ feature refinements, snapshots, scaling and native snappy use."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch

from tools.snappy_features import prepare_features, read_obj_lines, load_settings
from tools.snappy_pipeline import ROOT, prepare_geometry
from tools.templates import template_directory


class FeatureTests(unittest.TestCase):
    def setup_case(self, root):
        case = root/'case'
        (case/'Parameters').mkdir(parents=True)
        (root/'STL').mkdir()
        (root/'FEATURES').mkdir()
        settings = {'enabled': True, 'groups': ['te'], 'teLevel': 7,
                    'teRefinementThickness': .0005}
        (case/'Parameters/snappyFeatures.json').write_text(json.dumps(settings))
        source = root/'STL/blade.stl'
        curve = root/'FEATURES/blade_te.obj'
        curve.write_text('v 0 0 0\nv 1 0 0\nv 2 0 0\nl 1 2 3\n')
        return case, source, curve, settings

    def test_polyline_scaling_negative_indices_and_unused_vertices(self):
        points, edges = read_obj_lines(b'v 0 0 0\nv 10 0 0\nl -2/1 -1/2\nv 999 9 9\n', .001)
        self.assertEqual(points, [(0.,0.,0.),(.01,0.,0.)])
        self.assertEqual(edges, [(0,1)])

    def test_invalid_curves_fail(self):
        for raw in [b'v 0 0 0\nf 1 1 1', b'v nan 0 0\nl 1 1',
                    b'v 0 0 0\nl 0 1', b'v 0 0 0\nl 1 2',
                    b'v 0 0 0\nv 0 0 0\nl 1 2', b'v 0 0 0\nl -2 -1']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                read_obj_lines(raw, 1)

    def test_native_distance_entries_and_raw_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, source, curve, settings = self.setup_case(Path(tmp))
            report = prepare_features(case, source, .001)
            entry = report['groups']['te']
            self.assertEqual(entry['status'], 'active')
            self.assertEqual(entry['edges'], 2)
            self.assertEqual((case/entry['snapshot']).read_bytes(), curve.read_bytes())
            self.assertIn('(0.002 0 0)', (case/entry['edge_file']).read_text())
            generated = (case/'Parameters/snappyFeatures.generated').read_text()
            self.assertIn('custom_te.eMesh', generated)
            self.assertIn('levels ((0.00050000000000000001 7))', generated)
            settings['teRefinementThickness'] = 0
            (case/'Parameters/snappyFeatures.json').write_text(json.dumps(settings))
            prepare_features(case, source, 1)
            self.assertIn('level 7;', (case/'Parameters/snappyFeatures.generated').read_text())

    def test_missing_disabled_and_legacy_cases_clear_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, source, curve, settings = self.setup_case(Path(tmp))
            curve.unlink()
            with warnings.catch_warnings(record=True) as caught:
                report = prepare_features(case, source, 1)
            self.assertEqual(len(caught), 1)
            self.assertEqual(report['groups']['te']['status'], 'missing')
            settings['enabled'] = False
            (case/'Parameters/snappyFeatures.json').write_text(json.dumps(settings))
            self.assertFalse(prepare_features(case, source, 1)['enabled'])
            self.assertNotIn('file ', (case/'Parameters/snappyFeatures.generated').read_text())
            (case/'Parameters/snappyFeatures.json').unlink()
            self.assertFalse(prepare_features(case, source, 1)['enabled'])

    def test_bad_controls_rejected(self):
        for key, value in [('teLevel', -1), ('teLevel', True), ('teLevel', 1.5),
                           ('teRefinementThickness', -1), ('teRefinementThickness', float('nan')),
                           ('groups', ['../te']), ('groups', ['te','te']), ('enabled','true')]:
            with self.subTest(key=key,value=value), tempfile.TemporaryDirectory() as tmp:
                case, _, _, settings = self.setup_case(Path(tmp))
                settings[key]=value
                (case/'Parameters/snappyFeatures.json').write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    load_settings(case/'Parameters/snappyFeatures.json')

    def test_geometry_study_and_report_include_features(self):
        import trimesh
        from tools.settings_report import create_settings_report_data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            case, source, curve, _ = self.setup_case(root)
            shutil.copytree(template_directory('AMI','kOmegaSST','yes'), case, dirs_exist_ok=True)
            shutil.copytree(ROOT/'Parameters', case/'Parameters', dirs_exist_ok=True)
            solid=trimesh.creation.icosphere(subdivisions=1,radius=.08)
            solid.vertices[:,1] *= .1
            solid.export(source)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                prepare_geometry(case,source,None,None,study=('snappyFeatures.json','teLevel','7'))
            geometry=json.loads((case/'mesh/geometry.json').read_text())
            self.assertEqual(geometry['features']['groups']['te']['level'],7)
            self.assertTrue(geometry['layers'])
            report=create_settings_report_data(case,4000,'AMI','kOmegaSST',mesh_only=True)
            self.assertIn('custom_te.eMesh', json.dumps(report))
            self.assertIn('Custom OBJ feature refinement', json.dumps(report))
            for path in (ROOT/'CoreTemplates').rglob('snappyHexMeshDict'):
                self.assertIn('snappyFeatures.generated',path.read_text())


@unittest.skipUnless(os.environ.get('OPENFOAM_FEATURE_TESTS') == '1', 'requires Docker/OpenFOAM 13')
class NativeFeatureTests(unittest.TestCase):
    def test_custom_curve_increases_snappy_refinement(self):
        import trimesh
        from tools.mesh_runner import Runner
        from tools import update_parameter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'STL').mkdir()
            (root/'FEATURES').mkdir()
            solid=trimesh.creation.icosphere(subdivisions=1,radius=.08)
            solid.vertices[:,1] *= .1
            source=root/'STL/blade.stl'
            solid.export(source)
            edge=solid.faces[0][:2]
            curve=''.join('v '+' '.join(map(str,solid.vertices[i]))+'\n' for i in edge)+'l 1 2\n'
            (root/'FEATURES/blade_te.obj').write_text(curve)
            counts=[]
            for enabled, layers in ((False,False), (True,False), (True,True)):
                case=root/(str(enabled)+'-'+str(layers))
                shutil.copytree(template_directory('AMI','kOmegaSST','yes'),case)
                shutil.copytree(ROOT/'Parameters',case/'Parameters')
                (case/'Parameters/snappyFeatures.json').write_text(json.dumps(dict(
                    enabled=enabled,groups=['te'],teLevel=4,teRefinementThickness=.015)))
                mesh=case/'Parameters/snappyHexMeshDict.cpp'
                for key,value in [('addLayers', str(layers).lower()), ('propellerSurfaceRefinementLevel','(2 2)'),
                                  ('propellerRefinementRegionLevel','((0.0045 2))'),
                                  ('rotaryRegionSurfaceRefinementLevel','(1 1)'),
                                  ('rotaryRegionRefinementRegionLevel','((1E15 1))'),
                                  ('innerCylinderRefinementRegionLevel','((1E15 1))'),
                                  ('outerCylinderRefinementRegionLevel','((1E15 1))'),
                                  ('acousticSurfaceRefinementRegionLevel','((1E15 1))')]:
                    update_parameter(mesh,key,value,quiet=True)
                update_parameter(case/'Parameters/blockMeshDict.cpp','blocks_resolution','(8 12 8)',quiet=True)
                # Isolate custom refinement from the automatically extracted edge level.
                p=case/'system/snappyHexMeshDict'
                p.write_text(p.read_text().replace('level       4;','level       2;'))
                prepare_geometry(case,source,None,None)
                runner=Runner(docker=True)
                for program,flags in [('blockMesh',[]),('surfaceFeatures',[]),('snappyHexMesh',['-overwrite'])]:
                    runner.run(case,program,flags,'log.'+program)
                log=(case/'log.snappyHexMesh').read_text()
                counts.append(int(re.findall(r'cells:\s*(\d+)',log,re.S)[-1]))
                if enabled:
                    self.assertIn('custom_te.eMesh',log)
                    self.assertIn('distance',log.lower())
                if layers:
                    self.assertIn('Layer', log)
                    self.assertRegex(log, r'Added\s+[1-9][0-9]*\s+(?:out of.*? )?cells')
            self.assertGreater(counts[1],counts[0])
            print('Native snappy cell counts, baseline/custom/custom with layers:',counts)

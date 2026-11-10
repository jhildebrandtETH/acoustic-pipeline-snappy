"""Native snappy geometry and meshing regressions without requiring Docker."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import trimesh
from tools import snappy_pipeline as mesh
from tools.templates import template_directory


class SnappyTests(unittest.TestCase):
    def test_geometry_modes_scale_and_studies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid = trimesh.creation.icosphere(subdivisions=1, radius=80)
            solid.vertices[:, 1] *= .1
            source = root/'arbitrary-name.stl'
            solid.export(source)
            inputs = ('Parameters/blockMesh.generated', 'Parameters/snappyGeometry.generated',
                      'constant/triSurface/permeableSurface.stl', 'constant/triSurface/innerCylinder.stl',
                      'constant/triSurface/outerCylinder.stl')
            for refinement_override in (None, 2.2):
                expected_inputs = None
                for mode, diameter in ((None, None), ('impermeable', None),
                                       ('permeable', None), ('permeable', 1.8)):
                    with self.subTest(refinement_override=refinement_override, mode=mode, diameter=diameter):
                        case = root/f'{refinement_override}-{mode}-{diameter}'
                        shutil.copytree(template_directory('AMI', 'kOmegaSST', 'yes'), case)
                        shutil.copytree(mesh.ROOT/'Parameters', case/'Parameters')
                        if refinement_override is not None:
                            path = case/'Parameters/snappyGeometry.json'
                            config = json.loads(path.read_text())
                            config['refinementSphereDiameter'] = refinement_override
                            path.write_text(json.dumps(config))
                        mesh.prepare_geometry(case, source, mode, diameter,
                                              study=('snappyGeometry.json', 'scale', '0.001'))
                        data = json.loads((case/'mesh/geometry.json').read_text())
                        self.assertAlmostEqual(data['diameter_m'], .16)
                        self.assertEqual(data['layers'], mesh.configured_layers(case))
                        refinement_factor = refinement_override or 2.5
                        self.assertAlmostEqual(data['refinement_sphere_radius_m'], .08 * refinement_factor)
                        actual_inputs = [(case/name).read_bytes() for name in inputs]
                        if expected_inputs is None:
                            expected_inputs = actual_inputs
                        self.assertEqual(actual_inputs, expected_inputs)
                        sampling = case/'constant/triSurface/permeableSamplingSurface.stl'
                        self.assertFalse((case/'acoustic-surfaces.json').exists())
                        if mode == 'permeable':
                            expected_radius = .08 * (diameter or refinement_factor)
                            self.assertAlmostEqual(data['acoustic_sphere_radius_m'], expected_radius)
                            self.assertAlmostEqual(trimesh.load(sampling).extents.max(), 2 * expected_radius)
                            self.assertIn(sampling.name, (case/'system/controlDict').read_text())
                        else:
                            self.assertIsNone(data['acoustic_sphere_radius_m'])
                            self.assertFalse(sampling.exists())

    def test_layer_switch_study_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            shutil.copytree(template_directory('AMI','kOmegaSST','yes'), case)
            shutil.copytree(mesh.ROOT/'Parameters', case/'Parameters')
            solid = trimesh.creation.icosphere(subdivisions=1, radius=.08)
            solid.vertices[:,1] *= .1
            source = Path(tmp)/'blade.stl'
            solid.export(source)
            for value, enabled in [('false',False), ('true',True)]:
                mesh.prepare_geometry(case,source,None,None,
                                      study=('snappyHexMeshDict','addLayers',value))
                self.assertEqual(json.loads((case/'mesh/geometry.json').read_text())['layers'], enabled)

    def run_fixture(self, cores=1, quality=True, allow_bad=False, coverage=1.0, layers=False):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case/'system').mkdir()
            (case/'system/snappyHexMeshDict').write_text('addLayers '+str(layers).lower()+';')
            (case/'mesh').mkdir()
            (case/'constant/polyMesh').mkdir(parents=True)
            (case/'constant/polyMesh/cellZones').write_text('rotaryRegion {}')
            (case/'mesh/geometry.json').write_text(json.dumps(dict(expected_fluid_volume_m3=1,
                settings=dict(maxRelativeVolumeError=.02,minimumFaceCoverage=.95,minimumAverageCoverage=.99))))
            commands=[]
            coupled=False
            def patches(_):
                result={name:dict(nFaces=10,type='patch') for name in
                        ('inlet','outlet','walls','propeller','rotaryRegion','rotaryRegion_slave')}
                if coupled:
                    result.update(ncc1=dict(nFaces=0,type='nonConformalCyclic'),
                                  ncc2=dict(nFaces=0,type='nonConformalCyclic'))
                return result
            def run(container, case, command, log, callback, live):
                nonlocal coupled
                commands.append(command)
                if command[0]=='checkMesh':
                    (case/log).write_text('Total volume = 1.0.\n'+('Mesh OK.' if quality else 'Failed 1 mesh checks.'))
                elif command[0]=='createNonConformalCouples':
                    coupled=True
                    (case/log).write_text('\n'.join(f'{role} min/average/max coverage = {coverage}/{coverage}/1'
                                                   for role in ('Source','Target')))
            with patch.object(mesh,'docker_run',side_effect=run), patch.object(mesh,'patch_info',side_effect=patches):
                result=mesh.run_mesh(object(),case,cores,allow_bad)
            return result,commands,json.loads((case/'mesh/mesh-status.json').read_text())

    def test_serial_and_parallel_order_and_no_layers(self):
        for cores in (1,2):
            ok,commands,status=self.run_fixture(cores)
            self.assertTrue(ok)
            self.assertFalse(status['layers'])
            self.assertEqual(commands[0], ['blockMesh'])
            self.assertNotIn('foamDictionary', [c[0] for c in commands])
            names=[c[0] for c in commands]
            self.assertEqual('reconstructPar' in names,cores>1)
            if cores>1:
                self.assertLess(names.index('reconstructPar'),names.index('createNonConformalCouples'))
                self.assertIn(['reconstructPar','-constant','-noFields'],commands)
            self.assertIn(['splitBaffles', '-overwrite'], commands)
            self.assertLess(names.index('splitBaffles'), names.index('createNonConformalCouples'))
            if cores > 1:
                self.assertLess(names.index('reconstructPar'), names.index('splitBaffles'))
            self.assertIn(['createNonConformalCouples','-fields','rotaryRegion_slave','rotaryRegion'],commands)

    def test_quality_gate_and_override(self):
        with self.assertRaisesRegex(ValueError,'Mesh quality checks failed'):
            self.run_fixture(quality=False)
        ok,_,status=self.run_fixture(quality=False,allow_bad=True)
        self.assertFalse(ok)
        self.assertFalse(status['mesh_quality_ok'])
        with self.assertRaisesRegex(ValueError,'coverage is inadequate'):
            self.run_fixture(allow_bad=True,coverage=.8)

    def test_all_templates_use_editable_layers(self):
        paths=list((mesh.ROOT/'CoreTemplates').rglob('snappyHexMeshDict'))
        self.assertGreaterEqual(len(paths),9)
        for path in paths:
            text=path.read_text()
            self.assertRegex(text,r'addLayers\s+\$addLayers;')
            self.assertNotIn('addLayersControls {}',text)
            self.assertNotIn('nSurfaceLayers',text)

    def test_enabled_layers_are_recorded_without_overriding_dictionary(self):
        for cores in (1,2):
            ok, commands, status = self.run_fixture(cores=cores,layers=True)
            self.assertTrue(status['layers'])
            self.assertNotIn('foamDictionary', [c[0] for c in commands])

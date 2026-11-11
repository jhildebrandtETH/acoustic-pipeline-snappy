"""Exercise the AMI study order and generated geometry without OpenFOAM."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

from preprocessing import preprocessing
from tools.cli import create_parser
from tools.orders import prepare_simulation_order
from tools.snappy_pipeline import ROOT


class AmiDimensionStudyTests(unittest.TestCase):
    def test_full_order_geometry_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch('tools.orders.preflight'):
            root = Path(tmp)
            (root / 'STL').mkdir()
            blade = trimesh.creation.icosphere(subdivisions=1, radius=.127)
            blade.vertices[:, 1] *= .1
            blade.export(root / 'STL/blade.stl')
            parser = create_parser()
            options = [
                '--sim-dir', tmp, '--ami_dimension_study', '--rpms', '4000',
                '--turbulence', 'kOmegaSST', '--wall-functions', 'yes',
                '--total-cores', '4', '--cores-per-case', '2',
                '--acoustic-surface', 'permeable',
            ]
            _, _, _, _, order = prepare_simulation_order(parser.parse_args(options), parser)
            self.assertEqual(len(order['cases']), 16)
            self.assertEqual(order['max_parallel_cases'], 2)
            self.assertEqual(order['acoustic_sphere_diameter'], 2.1)
            self.assertEqual(len({case['folder'] for case in order['cases']}), 16)
            _, _, _, _, resumed = prepare_simulation_order(
                parser.parse_args(['--sim-dir', tmp, '--resume']), parser,
            )
            self.assertEqual(resumed['cases'], order['cases'])

            def prepare(name, sampling_diameter=None):
                target = root / name
                preprocessing(name, 4000, ROOT, target, 2, 'AMI', False, None,
                              'kOmegaSST', 'permeable', sampling_diameter, WALL_FUNCTIONS='yes')
                data = json.loads((target / 'mesh/geometry.json').read_text())
                return target, data

            for index, case in enumerate(order['cases']):
                target, data = prepare(case['folder'], order['acoustic_sphere_diameter'])
                dimensions = data['ami_dimension_study']
                diameter = data['diameter_m']
                radial_step, axial_step = divmod(index, 4)
                fraction = axial_step / 3
                self.assertAlmostEqual(2 * data['rotor_radius_m'] / diameter,
                                       1.1 + radial_step / 6)
                self.assertAlmostEqual(dimensions['upstream_cap_y_m'],
                                       .025 * (1 - fraction) + .30 * diameter * fraction)
                self.assertAlmostEqual(dimensions['downstream_cap_y_m'],
                                       -.025 * (1 - fraction) - .60 * diameter * fraction)
                self.assertAlmostEqual(2 * data['refinement_sphere_radius_m'] / diameter, 2.1)
                self.assertEqual(data['acoustic_sphere_radius_m'], data['refinement_sphere_radius_m'])
                surfaces = target / 'constant/triSurface'
                self.assertEqual((surfaces / 'permeableSurface.stl').read_bytes(),
                                 (surfaces / 'permeableSamplingSurface.stl').read_bytes())
                self.assertIn('blocks_resolution (18 52 18);',
                              (target / 'Parameters/blockMeshDict.cpp').read_text())
                self.assertIn('innerCylinderRefinementRegionLevel ((1E15 2));',
                              (target / 'Parameters/snappyHexMeshDict.cpp').read_text())
                fixed_files = ('Parameters/blockMesh.generated',
                               'constant/triSurface/innerCylinder.stl',
                               'constant/triSurface/outerCylinder.stl')
                fixed_geometry = tuple((target / path).read_bytes() for path in fixed_files)
                if index == 0:
                    reference = fixed_geometry
                self.assertEqual(fixed_geometry, reference)
                if axial_step == 3:
                    rotor = trimesh.load_mesh(surfaces / 'rotaryCylinder.stl')
                    outer = trimesh.load_mesh(surfaces / 'outerCylinder.stl')
                    self.assertEqual(list(rotor.bounds[:, 1]), list(outer.bounds[:, 1]))

            (root / 'simulation_order.json').unlink()
            normal, data = prepare('blade_4000RPM_AMI')
            self.assertNotIn('ami_dimension_study', data)
            self.assertAlmostEqual(2 * data['refinement_sphere_radius_m'] / data['diameter_m'], 2.1)
            self.assertEqual(data['acoustic_sphere_radius_m'], data['refinement_sphere_radius_m'])
            self.assertEqual(tuple((normal / path).read_bytes() for path in fixed_files), reference)
            for name in ('blockMeshDict.cpp', 'snappyHexMeshDict.cpp'):
                self.assertEqual((normal / 'Parameters' / name).read_bytes(),
                                 (ROOT / 'Parameters' / name).read_bytes())

"""Import integrity and fixed-mesh order tests; no Fluent license required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from tools import ansys_mesh as am
from tools.fluent_hdf5 import export_mesh


def write_hdf5(path):
    """Two tetrahedra, separate cell zones and one internal rotor face."""
    with h5py.File(path, 'w') as file:
        mesh = file.create_group('meshes/1')
        def topology(kind, names, ids, counts, types):
            group = mesh.create_group(kind+'/zoneTopology')
            group['name'] = np.array([';'.join(names).encode()])
            group['id'] = ids
            group['minId'] = np.r_[1, 1+np.cumsum(counts)[:-1]]
            group['maxId'] = np.cumsum(counts)
            group['dimension'] = [3]*len(ids)
            group['zoneType' if kind != 'cells' else 'cellType'] = types
        topology('nodes',['points'],[1],[5],[1])
        topology('cells',['rotaryregion','stationary_fluid'],[10,11],[1,1],[0,0])
        topology('faces',['propeller','walls','outlet','inlet','rotor_interface'],
                 [20,21,22,23,24],[3,1,1,1,1],[3,3,5,4,15])
        xyz = mesh.create_dataset('nodes/coords/1',data=[
            [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
        xyz.attrs['minId'] = [1]; xyz.attrs['maxId'] = [5]
        mesh['cells/ctype/1/cell-types'] = [2]
        mesh['cells/ctype/2/cell-types'] = [2]
        polygons = [[[1,2,4],[2,3,4],[3,1,4]],[[1,5,2]],[[2,5,3]],[[3,5,1]],[[1,3,2]]]
        start = 1
        for i, faces in enumerate(polygons,1):
            group = mesh.create_group(f'faces/nodes/{i}')
            group.attrs['minId']=[start];group.attrs['maxId']=[start+len(faces)-1]
            group['nnodes']=[3]*len(faces)
            group['nodes']=np.array(faces).ravel()
            mesh[f'faces/c0/{i}']=[1 if i in (1,5) else 2]*len(faces)
            mesh[f'faces/c1/{i}']=[2 if i==5 else 0]*len(faces)
            start += len(faces)
        file['special/boundary_layer_flags/cells/10/data']=[1]


class ImportTests(unittest.TestCase):
    def test_hdf5_export_preserves_connectivity_units_and_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'mesh.msh.h5', Path(tmp)/'mesh.msh'
            write_hdf5(source)
            before=am.digest(source)
            counts=export_mesh(source,output)
            self.assertEqual(counts,dict(points=5,faces=7,cells=2,
                boundary_layer_flagged_cells=1,layers=True))
            text=output.read_text()
            self.assertIn('(10 (1 1 5 1 3)',text)
            self.assertIn('3 1 3 2 1 2',text)
            self.assertIn('(45 (24 interior rotor_interface)',text)
            self.assertEqual(before,am.digest(source))

    def test_uniform_polyhedral_zones_without_cell_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'mesh.msh.h5', Path(tmp)/'mesh.msh'
            write_hdf5(source)
            with h5py.File(source, 'r+') as file:
                mesh = file['meshes/1']
                mesh['cells/zoneTopology/cellType'][...] = [7, 7]
                for i in (1, 2):
                    del mesh[f'cells/ctype/{i}/cell-types']
                    mesh[f'cells/ctype/{i}'].attrs['elementType'] = [7]
            counts = export_mesh(source, output)
            text = output.read_text()
            self.assertEqual(counts['cells'], 2)
            self.assertIn('(12 (a 1 1 1 7))', text)
            self.assertIn('(12 (b 2 2 1 7))', text)
            self.assertIn('3 1 3 2 1 2', text)

    def test_mixed_zone_without_cell_types_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'bad.msh.h5'
            write_hdf5(source)
            with h5py.File(source, 'r+') as file:
                del file['meshes/1/cells/ctype/1/cell-types']
            with self.assertRaisesRegex(ValueError, 'Mixed.*missing cell-types'):
                export_mesh(source, Path(tmp)/'out.msh')

    def test_reject_unpaired_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'bad.msh.h5'
            write_hdf5(source)
            with h5py.File(source,'r+') as file:
                file['meshes/1/faces/c1/5'][0]=0
            with self.assertRaisesRegex(ValueError,'unpaired'):
                export_mesh(source,Path(tmp)/'out.msh')

    def test_reject_nonconformal_and_nonfinite(self):
        for mode in ('interface','nonfinite'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/'bad.msh.h5'
                write_hdf5(source)
                with h5py.File(source,'r+') as file:
                    if mode=='interface':
                        file['meshes/1/faces/zoneTopology/zoneType'][4]=24
                    else:
                        file['meshes/1/nodes/coords/1'][0,0]=np.nan
                with self.assertRaises(ValueError):
                    export_mesh(source,Path(tmp)/'out.msh')

    def test_discovery_no_stl_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'mesh.msh.h5').write_bytes(b'h5')
            (root/'ignored.stl').write_text('not an input')
            self.assertEqual(list(am.find_meshes(root)),['mesh'])
            (root/'mesh.msh').write_bytes(b'ascii')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                am.find_meshes(root)

    def test_per_mesh_units_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'mesh.msh.h5'
            source.touch()
            (root/'mesh.json').write_text(json.dumps(dict(scale=1.0)))
            self.assertEqual(am.read_config(root,source)['scale'],1.0)
            (root/'mesh.json').write_text(json.dumps(dict(scale=0)))
            with self.assertRaisesRegex(ValueError,'positive'):
                am.read_config(root,source)

    def test_checksum_and_configuration_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'mesh.msh'
            source.write_text('original')
            initial=am.input_fingerprints(root,root)
            source.write_text('modified')
            self.assertNotEqual(initial,am.input_fingerprints(root,root))
            source.write_text('original')
            (root/'mesh.json').write_text('{"scale":1}')
            self.assertNotEqual(initial,am.input_fingerprints(root,root))

    def test_order_snapshots_preparation_and_backend(self):
        from tools.cli import create_parser
        from tools.openfoam import create_simulation_order
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'MESH').mkdir()
            (root/'MESH/mesh.msh').write_text('mesh')
            args=create_parser().parse_args(['--sim-dir',str(root),'--rpms','4000',
                '--mode','AMI','--turbulence','kOmegaSST','--wall-functions','yes',
                '--total-cores','2','--cores-per-case','1','--mesh-only'])
            args.meshes=['mesh'];args.end_on='rev';args.end_on_value=1
            create_simulation_order(args,root)
            order=json.loads((root/'simulation_order.json').read_text())
            self.assertEqual(order['meshing_backend'],am.BACKEND)
            snapshot=root/order['preparation_snapshot']
            self.assertTrue((snapshot/'Parameters/ansysMesh.json').is_file())
            self.assertEqual(order['mesh_inputs'],am.input_fingerprints(root/'MESH',snapshot/'Parameters'))

    def test_zone_labels_and_renaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'cellZones'
            source.write_text('FoamFile { class regIOobject; }\n1\n(\nrotaryregion\n{ type cellZone; cellLabels List<label> 2(0 1); flipMap List<bool> 2{false}; }\n)')
            am._rename_entry(source,'rotaryregion','rotaryRegion')
            body=am._zone_entries(source)['rotaryRegion']
            self.assertEqual(am._labels(body,'cellLabels').tolist(),[0,1])
            with self.assertRaisesRegex(ValueError,'Incorrect'):
                am._labels('cellLabels 3(0 1);','cellLabels')

    def test_import_driver_never_calls_a_mesher(self):
        import inspect
        source=inspect.getsource(am.run_mesh)
        for forbidden in ("run(['blockMesh'", "run(['surfaceFeatures'", "run(['snappyHexMesh'"):
            self.assertNotIn(forbidden,source)
        self.assertIn("run(['createBaffles']",source)
        self.assertIn("run(['splitBaffles']",source)

    def test_cell_count_includes_neighbour_only_last_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh=Path(tmp)/'constant/polyMesh'
            mesh.mkdir(parents=True)
            (mesh/'owner').write_text('FoamFile { format ascii; }\n3\n(0 0 0)')
            (mesh/'neighbour').write_text('FoamFile { format ascii; }\n1\n(1)')
            self.assertEqual(am.cell_count(tmp),2)

    def test_tip_wall_merge_accepts_fluent_names_and_preserves_face_count(self):
        for tips in [('tip_negative', 'tip_positive'),
                     ('propeller-tip_negative-rotaryregion', 'propeller-tip_positive-rotaryregion'),
                     ('propeller_tip_negative_rotaryregion', 'propeller_tip_positive_rotaryregion')]:
            with self.subTest(tips=tips), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);mesh=root/'constant/polyMesh';mesh.mkdir(parents=True)
                (root/'mesh').mkdir()
                (root/'mesh/import.json').write_text(json.dumps({'config':am.DEFAULTS}))
                (mesh/'owner').write_text('FoamFile { format ascii; }\n3\n(0 0 0)')
                (mesh/'neighbour').write_text('FoamFile { format ascii; }\n1\n(1)')
                (mesh/'boundary').write_text('\n'.join(
                    f'{name}\n{{type wall; nFaces {count}; startFace 1;}}'
                    for name,count in [('propeller',10),(tips[0],2),(tips[1],3),('walls',4)]))
                def native_merge(command, log):
                    self.assertEqual(command[0],'createPatch')
                    self.assertIn('-dict',command)
                    self.assertEqual(log,'log.mergePropellerTips')
                    self.assertTrue((root/'system/createPropellerPatchDict').is_file())
                    (mesh/'boundary').write_text('propeller\n{type wall; nFaces 15; startFace 1;}\nwalls\n{type wall; nFaces 4; startFace 16;}')
                am.merge_propeller_tip_patches(root,native_merge)
                merged=json.loads((root/'mesh/import.json').read_text())['propeller_patch_merge']
                self.assertEqual(merged['merged_faces'],15)
                self.assertEqual(merged['source_face_counts'],{'propeller':10,tips[0]:2,tips[1]:3})
                self.assertEqual(merged['cells'],2)

    def test_tip_wall_merge_does_not_accept_nonwall_or_unrelated_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh=root/'constant/polyMesh';mesh.mkdir(parents=True)
            (root/'mesh').mkdir()
            (root/'mesh/import.json').write_text(json.dumps({'config':am.DEFAULTS}))
            (mesh/'boundary').write_text('propeller\n{type wall; nFaces 10;}\ntip_negative\n{type patch; nFaces 2;}')
            with self.assertRaisesRegex(ValueError,'must be a wall'):
                am.merge_propeller_tip_patches(root,lambda *args:self.fail('Must reject before merging'))
            # Old meshes and unrelated names do not trigger a merge. Normal
            # import validation continues to reject unexpected extra patches.
            (mesh/'boundary').write_text('propeller\n{type wall; nFaces 10;}\ntip_unknown\n{type wall; nFaces 2;}')
            am.merge_propeller_tip_patches(root,lambda *args:self.fail('Unrelated patch must not be merged'))
            self.assertNotIn('propeller_patch_merge',json.loads((root/'mesh/import.json').read_text()))

    def test_tip_wall_merge_rejects_lost_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh=root/'constant/polyMesh';mesh.mkdir(parents=True)
            (root/'mesh').mkdir()
            (root/'mesh/import.json').write_text(json.dumps({'config':am.DEFAULTS}))
            (mesh/'owner').write_text('FoamFile { format ascii; }\n3\n(0 0 0)')
            (mesh/'neighbour').write_text('FoamFile { format ascii; }\n1\n(1)')
            (mesh/'boundary').write_text('propeller\n{type wall; nFaces 10;}\ntip_negative\n{type wall; nFaces 2;}')
            def lose_faces(*args):
                (mesh/'boundary').write_text('propeller\n{type wall; nFaces 10;}')
            with self.assertRaisesRegex(ValueError,'preserve all wall faces'):
                am.merge_propeller_tip_patches(root,lose_faces)

    def test_visualization_uses_measured_import_diameter(self):
        from tools.visualization import visualization_settings
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'mesh').mkdir()
            (root/'mesh/geometry.json').write_text(json.dumps(dict(backend=am.BACKEND,diameter_m=.254)))
            self.assertEqual(visualization_settings(root,4000,None)['diameter_m'],.254)


if __name__=='__main__':
    unittest.main()

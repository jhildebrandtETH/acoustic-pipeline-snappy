"""Settings safeguards and actual CAD trimming invariants on known wedge blades."""
import contextlib, copy, hashlib, io, json, math, tempfile, unittest
from pathlib import Path
import numpy as np
from trim_step import frame_vectors, load_settings, run

BASE={
    'schema_version':1,'input_step':'input.step','output_directory':'result',
    'setback_mm':.2,'frame':{'origin_mm':[0,0,0],'rotation_axis':[0,1,0],
        'positive_blade_axis':[0,0,1],'positive_blade_trailing_edge':'min_chord_coordinate'},
    'hub':{'radius_mm':'auto'},'station_spacing_mm':2.,'section_deflection_mm':.001,
    'validation_tolerance_mm':.015,'overlay_radius_fractions':[.3,.5,.7,.85,.95,.98],
    'overwrite':False}


class SettingsTests(unittest.TestCase):
    def test_frame_is_orthonormal(self):
        f=copy.deepcopy(BASE['frame']);f['rotation_axis']=[0,7,0]
        _,a,s,c=frame_vectors(f)
        np.testing.assert_allclose(np.cross(c,a),s)
        self.assertAlmostEqual(np.linalg.norm(a),1)

    def test_invalid_axes_rejected(self):
        for vector in [[0,0,0],[0,1,0],[0,1,1]]:
            f=copy.deepcopy(BASE['frame']);f['positive_blade_axis']=vector
            with self.assertRaises(ValueError):frame_vectors(f)

    def test_json_paths_relative_to_settings(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'settings.json';p.write_text(json.dumps(BASE));cfg=load_settings(p)
            self.assertEqual(Path(cfg['input_step']),Path(td)/'input.step')

    def test_unknown_and_unsafe_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'settings.json'
            for key,value in [('setback_mm',0),('setback_mm',-.2),('station_spacing_mm',0),('typo',1)]:
                cfg=copy.deepcopy(BASE);cfg[key]=value;p.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load_settings(p)

    def test_schema_version_preserves_legacy_strategy(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'settings.json';cfg=copy.deepcopy(BASE)
            p.write_text(json.dumps(cfg));self.assertEqual(load_settings(p)['station_strategy'],'uniform')
            cfg['schema_version']=2;p.write_text(json.dumps(cfg));self.assertEqual(load_settings(p)['station_strategy'],'cad_seams')


class CadTests(unittest.TestCase):
    def make_wedge_propeller(self,path,rotated=False,partitioned=False):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon,BRepBuilderAPI_MakeFace,BRepBuilderAPI_Transform
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism,BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.gp import gp_Pnt,gp_Dir,gp_Vec,gp_Ax1,gp_Ax2,gp_Trsf
        from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
        poly=BRepBuilderAPI_MakePolygon()
        for point in [(-2,0,0),(2,-.3,0),(2,.3,0)]:poly.Add(gp_Pnt(*point))
        poly.Close();face=BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        if partitioned:
            from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
            loft=BRepOffsetAPI_ThruSections(True,True,1e-7);loft.CheckCompatibility(False)
            for z in [0,3.5,7.2,12]:
                tr=gp_Trsf();tr.SetTranslation(gp_Vec(0,0,z))
                wire=BRepBuilderAPI_Transform(poly.Wire(),tr,True).Shape()
                from OCP.TopoDS import TopoDS
                wire=(getattr(TopoDS,'Wire_s',None) or TopoDS.Wire)(wire)
                loft.AddWire(wire)
            loft.Build();blade=loft.Shape()
        else:blade=BRepPrimAPI_MakePrism(face,gp_Vec(0,0,12)).Shape()
        tr=gp_Trsf();tr.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,1,0)),math.pi)
        other=BRepBuilderAPI_Transform(blade,tr,True).Shape()
        hub=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,-1,0),gp_Dir(0,1,0)),2,2).Shape()
        solid=BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(hub,blade).Shape(),other).Shape()
        cfg=copy.deepcopy(BASE)
        if rotated:
            angle=.6;tr=gp_Trsf();tr.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,1,0)),angle)
            tr.SetTranslationPart(gp_Vec(1,2,3));solid=BRepBuilderAPI_Transform(solid,tr,True).Shape()
            cfg['frame']['origin_mm']=[1,2,3]
            cfg['frame']['positive_blade_axis']=[math.sin(angle),0,math.cos(angle)]
        writer=STEPControl_Writer();writer.Transfer(solid,STEPControl_AsIs);writer.Write(str(path))
        cfg['input_step']=str(path);return cfg

    def test_real_boolean_and_rotated_frame(self):
        for rotated in [False,True]:
            with self.subTest(rotated=rotated),tempfile.TemporaryDirectory() as td:
                folder=Path(td);inp=folder/'wedge.step';cfg=self.make_wedge_propeller(inp,rotated)
                p=folder/'settings.json';p.write_text(json.dumps(cfg))
                original_hash=hashlib.sha256(inp.read_bytes()).hexdigest()
                with contextlib.redirect_stdout(io.StringIO()):report=run(p)
                self.assertTrue(report['export_readback_valid'])
                self.assertEqual(report['solids_after'],1)
                self.assertLess(report['hub_removed_volume_mm3'],1e-6)
                self.assertLess(report['added_volume_mm3'],1e-6)
                self.assertGreater(report['removed_volume_mm3'],0)
                for m in report['measurements']:
                    self.assertAlmostEqual(m['setback_measured_mm'],.2,delta=.001)
                    self.assertAlmostEqual(m['closure_thickness_mm'],.03,delta=.003)
                    self.assertLess(m['retained_contour_max_gap_mm'],.001)
                self.assertEqual(hashlib.sha256(inp.read_bytes()).hexdigest(),original_hash)
                with self.assertRaises(FileExistsError):run(p)

    def test_native_seams_align_without_uniform_grid(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);cfg=self.make_wedge_propeller(folder/'segmented.step',partitioned=True)
            cfg.update(schema_version=2,station_strategy='cad_seams')
            p=folder/'settings.json';p.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):report=run(p)
            self.assertLess(report['seam_alignment']['maximum_station_offset_mm'],1e-6)
            self.assertGreater(report['seam_alignment']['checked_short_cross_edges'],0)
            frames=json.loads((folder/'result/cutting-frames.json').read_text())
            np.testing.assert_allclose([r['span_mm'] for r in frames['1']],[0,3.5,7.2,12],atol=1e-6)


if __name__=='__main__':unittest.main()

from pathlib import Path
from types import SimpleNamespace
import json,tempfile,unittest,shutil
import numpy as np,h5py
from repair import widths,face_geometry,targets,scan,save_mesh,local_problem,topology_digest,validate_settings,promote_floor

CFG=json.loads((Path(__file__).resolve().parent/'settings.json').read_text())

class RepairTests(unittest.TestCase):
 def test_rotated_sliver_width_is_not_hidden_by_global_axes(self):
  dx,dy,dz=.003,.2,.3
  area=np.array([[dy*dz,0,0],[-dy*dz,0,0],[0,dx*dz,0],[0,-dx*dz,0],[0,0,dx*dy],[0,0,-dx*dy]])
  a=.63;rot=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
  result=widths(area@rot.T,dx*dy*dz,np.array([30,0,100]))
  self.assertAlmostEqual(result[1],dx,places=12)

 def test_projected_polygon_geometry_translation(self):
  xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)+[7,11,19]
  calculate,_,_,_=face_geometry(xyz,[np.arange(4)])
  centre,area=calculate();np.testing.assert_allclose(centre,[[7.5,11.5,19]]);np.testing.assert_allclose(area,[[0,0,-1]])

 def test_scan_finds_same_family_on_both_blades(self):
  area=np.array([[.01,0,0],[-.01,0,0],[0,.002,0],[0,-.002,0],[0,0,.002],[0,0,-.002]])
  sf=np.vstack([area,area,[-1,0,0],[1,0,0]])
  fc=np.zeros((14,3));fc[12]=[3,0,100];fc[13]=[-3,0,-100]
  centre=np.array([[0,0,0],[4,0,100],[-4,0,-100],[3,0,100],[-3,0,-100]],float)
  mask=np.zeros(14,bool);mask[-2:]=True
  mesh=SimpleNamespace(lo=1,hi=4,wall_cells={3,4},centre=centre,vol=np.array([0,.0002,.0002,.0002,.0002]),sf=sf,fc=fc,blade_mask=mask,cf={1:set(range(6)),2:set(range(6,12))})
  risk=scan(mesh,CFG,{})
  self.assertEqual(set(risk),{1,2});self.assertTrue(all(r['leading_transition'] for r in risk.values()))

 def test_speed_target_scales_with_radius(self):
  result=targets(np.array([[0,0,50],[0,0,100]],float),CFG)
  self.assertAlmostEqual(result[1,0]/result[0,0],2)

 def test_fixed_nodes_rejected_before_writing_checkpoint(self):
  mesh=SimpleNamespace(fixed={0},xyz=np.array([[1.,0,0]]),original_xyz=np.zeros((1,3)))
  with self.assertRaisesRegex(RuntimeError,'Fixed physical'):
   save_mesh(mesh,Path('unused'),Path('unused-output'),Path('unused-progress'))

 def test_protected_target_is_skipped_without_moving_nodes(self):
  mesh=SimpleNamespace(cf={1:{0}},owner=np.array([1]),nei=np.array([0]),polys={0:np.array([0,1,2])},fixed={0,1,2})
  result,reason=local_problem(mesh,1,{1:{'score':2}},CFG,{},{});
  self.assertIsNone(result);self.assertEqual(reason,'protected_nodes')

 def test_resume_fingerprint_ignores_coordinates_but_checks_connectivity(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
   a=Path(tmp)/'a.h5';b=Path(tmp)/'b.h5'
   with h5py.File(a,'w') as h:
    h.create_dataset('meshes/1/faces/owner',data=[1,2]);h.create_dataset('meshes/1/cells/name',data=['rotor'],dtype=h5py.string_dtype());h.create_dataset('meshes/1/nodes/coords/1',data=[[0.,0,0]])
   shutil.copy2(a,b)
   with h5py.File(b,'r+') as h:h['meshes/1/nodes/coords/1'][0]=[.1,0,0]
   self.assertEqual(topology_digest(a),topology_digest(b))
   with h5py.File(b,'r+') as h:h['meshes/1/faces/owner'][0]=2
   self.assertNotEqual(topology_digest(a),topology_digest(b))

 def test_invalid_budget_or_threshold_is_rejected(self):
  for name,value in [('max_passes',True),('target_delta_t_s',float('nan')),('max_layer_distance_change_fraction',1)]:
   with self.assertRaises(ValueError):validate_settings(dict(CFG,**{name:value}))

 def test_achieved_width_cannot_be_given_back_by_later_neighbor_repair(self):
  goal=np.array([.12,.06]);original=np.array([.03,.02]);repaired=np.array([.13,.061])
  floor=promote_floor(original,repaired,goal);np.testing.assert_allclose(floor,goal)
  np.testing.assert_allclose(promote_floor(floor,np.array([.08,.04]),goal),goal)

if __name__=='__main__':unittest.main()

import json,tempfile,unittest
from pathlib import Path
import numpy as np,h5py
from tools.mesh_sliver_screen import measure_native,assess_geometry,check_native_mesh,settings

def box(path,lengths=(.001,.2,.3),angle=.61):
 xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
 xyz=(xyz-.5)*lengths;c=np.cos(angle);s=np.sin(angle);xyz=xyz@np.array([[c,0,-s],[0,1,0],[s,0,c]])+np.array([30,0,100])
 faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
 with h5py.File(path,'w') as h:
  m=h.create_group('meshes/1');m.attrs['cellCount']=[1];m.attrs['faceCount']=[6];m.create_dataset('nodes/coords/1',data=xyz)
  ct=m.create_group('cells/zoneTopology');ct.create_dataset('name',data=[b'rotaryregion']);ct.create_dataset('minId',data=[1]);ct.create_dataset('maxId',data=[1])
  ft=m.create_group('faces/zoneTopology');ft.create_dataset('name',data=[b'propeller_blade']);ft.create_dataset('minId',data=[1]);ft.create_dataset('maxId',data=[6])
  g=m.create_group('faces/nodes/1');g.attrs['minId']=[1];g.attrs['maxId']=[6];g.create_dataset('nnodes',data=[4]*6);g.create_dataset('nodes',data=np.array([list(reversed(f)) for f in faces]).ravel()+1)
  m.create_dataset('faces/c0/1',data=[1]*6);m.create_dataset('faces/c1/1',data=[0]*6)

class SliverScreenTests(unittest.TestCase):
 def test_rotated_thin_box_dimensions_and_volume(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'box.msh.h5';box(path);g=measure_native(path)
   np.testing.assert_allclose(g['centre'],[[30,0,100]],atol=1e-10)
   np.testing.assert_allclose(g['volume'],[.001*.2*.3],rtol=1e-9)
   np.testing.assert_allclose(g['minimum_directional_width'],[.001],rtol=1e-9)
   self.assertTrue(assess_geometry(g)['flagged'][0])

 def test_intended_wall_normal_thinness_is_not_alone_a_rejection(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'box.msh.h5';box(path,(.5,.001,.5),0);g=measure_native(path)
   self.assertFalse(assess_geometry(g)['flagged'][0])
   g['wall_adjacent'][:]=False
   self.assertTrue(assess_geometry(g)['flagged'][0])

 def test_reject_gate_writes_diagnostic_before_stopping(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'box.msh.h5';out=Path(tmp)/'risk';box(path)
   with self.assertRaisesRegex(ValueError,'gate rejected'):
    check_native_mesh(path,out,{'mode':'reject'})
   report=json.loads((out/'report.json').read_text());self.assertEqual(report['flagged_cells'],1)
   self.assertTrue((out/'flagged-cells.csv').is_file())

 def test_invalid_threshold_is_rejected(self):
  for value in (0,-1,float('nan'),True):
   with self.assertRaises(ValueError):settings({'minimum_tangential_width_mm':value})

if __name__=='__main__':unittest.main()

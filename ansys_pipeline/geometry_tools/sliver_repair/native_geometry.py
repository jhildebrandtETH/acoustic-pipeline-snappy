from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np,h5py

def load_native(source,rotor_zone='rotaryregion'):
 with h5py.File(source) as h:
  m=h['meshes/1'];ct=m['cells/zoneTopology'];names=ct['name'][0].decode().strip(';').split(';');j=names.index(rotor_zone);lo=int(ct['minId'][j]);hi=int(ct['maxId'][j]);N=int(max(ct['maxId'][()]));F=int(m.attrs['faceCount'][0]);xyz=np.concatenate([v[()] for k,v in sorted(m['nodes/coords'].items(),key=lambda x:int(x[0]))]);flags=np.zeros(N+1,dtype=bool)
  for zi,x,y in zip(ct['id'][()],ct['minId'][()],ct['maxId'][()]):
   p=f'special/boundary_layer_flags/cells/{int(zi)}/data'
   if p in h:flags[int(x):int(y)+1]=h[p][()]!=0
  blade_mask=np.zeros(F,dtype=bool);ft=m['faces/zoneTopology'];fn=ft['name'][0].decode().strip(';').split(';')
  for name,a,b in zip(fn,ft['minId'][()],ft['maxId'][()]):
   if name.startswith('propeller_blade'):blade_mask[int(a)-1:int(b)]=True
  owner=np.zeros(F,dtype=np.int64);nei=owner.copy();fc=np.zeros((F,3));sf=fc.copy();polys={};cf=defaultdict(set)
  for key,g in m['faces/nodes'].items():
   a=int(g.attrs['minId'][0])-1;o=m['faces/c0'][key][()].astype(np.int64);n=m['faces/c1'][key][()].astype(np.int64);owner[a:a+len(o)]=o;nei[a:a+len(o)]=n;sel=((o>=lo)&(o<=hi))|((n>=lo)&(n<=hi))
   if not sel.any():continue
   nn=g['nnodes'][()];off=np.r_[0,np.cumsum(nn,dtype=np.int64)];flat=g['nodes'][()].astype(np.int64)-1;ix=np.flatnonzero(sel);sizes=nn[ix];loc=np.r_[0,np.cumsum(sizes)];rep=np.repeat(np.arange(len(ix)),sizes);idx=np.arange(loc[-1])+np.repeat(off[ix]-loc[:-1],sizes);verts=flat[idx];p=xyz[verts];nxt=np.arange(len(p))+1;nxt[loc[1:]-1]=loc[:-1];q=p[nxt];ref=np.add.reduceat(p,loc[:-1])/sizes[:,None];cross=np.cross(p-ref[rep],q-ref[rep])/2;vector_area=np.add.reduceat(cross,loc[:-1]);unit_area=vector_area/np.maximum(np.linalg.norm(vector_area,axis=1)[:,None],1e-300);w=np.einsum('ij,ij->i',cross,unit_area[rep])
   faces=a+ix;fc[faces]=np.add.reduceat((p+q+ref[rep])/3*w[:,None],loc[:-1])/np.add.reduceat(w,loc[:-1])[:,None];sf[faces]=-np.add.reduceat(cross,loc[:-1])
   for k,f in zip(ix,faces):
    f=int(f);polys[f]=flat[off[k]:off[k+1]].copy()
    if lo<=o[k]<=hi:cf[int(o[k])].add(f)
    if lo<=n[k]<=hi:cf[int(n[k])].add(f)
  cells=np.arange(lo,hi+1);estimate=np.zeros((N+1,3));count=np.zeros(N+1)
  relevant=np.array(sorted(polys))
  for c in (owner[relevant],nei[relevant]):
   good=(c>=lo)&(c<=hi);np.add.at(estimate,c[good],fc[relevant[good]]);np.add.at(count,c[good],1)
  estimate[cells]/=count[cells,None];vol=np.zeros(N+1);moment=np.zeros((N+1,3))
  for c,sign in [(owner[relevant],1),(nei[relevant],-1)]:
   good=(c>=lo)&(c<=hi);cc=c[good];f=relevant[good];v=sign*np.einsum('ij,ij->i',sf[f],fc[f]-estimate[cc]);np.add.at(vol,cc,v/3);np.add.at(moment,cc,v[:,None]/3*(3*fc[f]+estimate[cc])/4)
  assert vol[cells].min()>0,'Source has nonpositive rotor cell volume'
  centre=np.zeros((N+1,3));centre[cells]=moment[cells]/vol[cells,None]
 
  wall_faces=np.array([f for f in polys if nei[f]==0],dtype=int)
  wall_cells=set(owner[wall_faces].tolist())
  fixed=set()
  for f in wall_faces:fixed.update(polys[int(f)].tolist())
  for name,a,b in zip(fn,ft['minId'][()],ft['maxId'][()]):
   if 'rotor_interface' in name:
    for f in range(int(a)-1,int(b)):
     if f in polys:fixed.update(polys[f].tolist())
  return SimpleNamespace(N=N,F=F,xyz=xyz,original_xyz=xyz.copy(),owner=owner,nei=nei,fc=fc,sf=sf,polys=polys,cf=cf,vol=vol,centre=centre,lo=lo,hi=hi,blade_mask=blade_mask,wall_cells=wall_cells,fixed=fixed,wall_faces=wall_faces)

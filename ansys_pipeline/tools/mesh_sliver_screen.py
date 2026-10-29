"""Native Fluent cell-width screening. Geometry flags are not computed CFD Co.

Projected triangle weights and signed pyramid moments match Foundation mesh
geometry. Width is 2V / sum(abs(Sf dot direction)); off-wall normal slivers
are distinguished from intentionally thin wall-normal inflation cells.
"""
from pathlib import Path
import argparse,csv,json,math,re
import numpy as np
import h5py

DEFAULTS=dict(mode='warn',rpm=4000.0,target_delta_t_s=1e-5,max_co_proxy=5.0,
 minimum_tangential_width_mm=.1,minimum_nonwall_directional_width_mm=.03)

def settings(value=None):
 value={} if value is None else value
 if not isinstance(value,dict) or set(value)-set(DEFAULTS):raise ValueError('Unknown sliver screen setting')
 result=dict(DEFAULTS,**value)
 if result['mode'] not in ('off','warn','reject'):raise ValueError('Screen mode must be off, warn or reject')
 for name in set(DEFAULTS)-{'mode'}:
  v=result[name]
  if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or (v<0 if name=='rpm' else v<=0):raise ValueError('Screen numbers must be finite and positive: '+name)
 return result

def case_settings(case,config):
 result=settings(config)
 path=Path(case)/'Parameters/rotational_parameters.cpp'
 if path.is_file():
  match=re.search(r'\bomega_val\s+([-+0-9.eE]+)\s*;',path.read_text(encoding='utf-8'))
  if not match:raise ValueError('Directional screen requires numeric omega_val in '+str(path))
  result['rpm']=abs(float(match[1]))*30/math.pi
 return settings(result)


def measure_native(source,rotor_zone='rotaryregion'):
 with h5py.File(source) as h:
  m=h['meshes/1'];N=int(m.attrs['cellCount'][0]);F=int(m.attrs['faceCount'][0]);xyz=np.concatenate([v[()] for k,v in sorted(m['nodes/coords'].items(),key=lambda x:int(x[0]))]);ct=m['cells/zoneTopology'];names=ct['name'][0].decode().strip(';').split(';');zi=names.index(rotor_zone);lo=int(ct['minId'][zi]);hi=int(ct['maxId'][zi]);owner=np.zeros(F,dtype=np.int32);nei=owner.copy();fc=np.zeros((F,3));sf=fc.copy();estimate=np.zeros((N+1,3));count=np.zeros(N+1);ft=m['faces/zoneTopology'];fn=ft['name'][0].decode().strip(';').split(';');blade=np.zeros(F,dtype=bool);hub=blade.copy()
  for name,a,b in zip(fn,ft['minId'][()],ft['maxId'][()]):
   if name.startswith('propeller_blade'):blade[int(a)-1:int(b)]=True
   if name.startswith('propeller_hub'):hub[int(a)-1:int(b)]=True
  for key,g in m['faces/nodes'].items():
   start=int(g.attrs['minId'][0])-1;o=m['faces/c0'][key][()].astype(np.int32);n=m['faces/c1'][key][()].astype(np.int32);owner[start:start+len(o)]=o;nei[start:start+len(o)]=n;nn=g['nnodes'][()].astype(np.int64);off=np.r_[0,np.cumsum(nn)];flat=g['nodes'][()].astype(np.int64)-1
   for a in range(0,len(o),30000):
    b=min(len(o),a+30000);sizes=nn[a:b];loc=np.r_[0,np.cumsum(sizes)];rep=np.repeat(np.arange(b-a),sizes);p=xyz[flat[off[a]:off[b]]];following=np.arange(len(p))+1;following[loc[1:]-1]=loc[:-1];q=p[following];ref=np.add.reduceat(p,loc[:-1])/sizes[:,None];cr=np.cross(p-ref[rep],q-ref[rep])/2;vector_area=np.add.reduceat(cr,loc[:-1]);unit_area=vector_area/np.maximum(np.linalg.norm(vector_area,axis=1)[:,None],1e-300);w=np.einsum('ij,ij->i',cr,unit_area[rep]);sf[start+a:start+b]=-np.add.reduceat(cr,loc[:-1]);fc[start+a:start+b]=np.add.reduceat((p+q+ref[rep])/3*w[:,None],loc[:-1])/np.maximum(np.add.reduceat(w,loc[:-1])[:,None],1e-30)
  for cells in [owner,nei]:
   good=cells>0;np.add.at(estimate,cells[good],fc[good]);np.add.at(count,cells[good],1)
  estimate[1:]/=count[1:,None];vol=np.zeros(N+1);moment=np.zeros((N+1,3))
  for a in range(0,F,100000):
   b=min(F,a+100000)
   for cells,sign in [(owner[a:b],1),(nei[a:b],-1)]:
    good=cells>0;c=cells[good];cc=fc[a:b][good];v=sign*np.einsum('ij,ij->i',sf[a:b][good],cc-estimate[c])/3;np.add.at(vol,c,v);np.add.at(moment,c,v[:,None]*(3*cc+estimate[c])/4)
  if not np.isfinite(vol[1:]).all() or vol[1:].min()<=0:raise ValueError('Nonpositive or nonfinite native cell volume')
  centre=moment/np.maximum(vol[:,None],1e-30)
 
  rotor=np.zeros(N+1,dtype=bool);rotor[lo:hi+1]=True
  radius=np.hypot(centre[:,0],centre[:,2]);tangent=np.c_[centre[:,2]/np.maximum(radius,1e-30),np.zeros(N+1),-centre[:,0]/np.maximum(radius,1e-30)]
  projection=np.zeros(N+1);wall=np.zeros(N+1,dtype=bool)
  for cells in (owner,nei):
   good=cells>0;c=cells[good];np.add.at(projection,c,np.abs(np.einsum('ij,ij->i',sf[good],tangent[c])))
  wall[owner[nei==0]]=True
  width=2*vol/np.maximum(projection,1e-30);minimum=np.full(N+1,np.inf)
  fs=np.flatnonzero(rotor[owner]|rotor[nei]);order=np.argsort(np.r_[owner[fs],nei[fs]],kind='stable');cells=np.r_[owner[fs],nei[fs]][order];faces=np.tile(fs,2)[order];unique,starts=np.unique(cells,return_index=True)
  for c,a,z in zip(unique,starts,np.r_[starts[1:],len(cells)]):
   if not rotor[c]:continue
   area=sf[faces[a:z]];normals=area/np.maximum(np.linalg.norm(area,axis=1)[:,None],1e-30);minimum[c]=2*vol[c]/max(float(np.abs(normals@area.T).sum(1).max()),1e-30)
  ids=np.flatnonzero(rotor)
  return dict(native_cell_1based=ids,centre=centre[ids],volume=vol[ids],tangential_width=width[ids],minimum_directional_width=minimum[ids],wall_adjacent=wall[ids])
 
def assess_geometry(geometry,config=None,scale=.001):
 cfg=settings(config)
 if not math.isfinite(scale) or scale<=0:raise ValueError('Native scale must be finite and positive')
 p=geometry['centre']*scale;wt=geometry['tangential_width']*scale;wn=geometry['minimum_directional_width']*scale
 radius=np.hypot(p[:,0],p[:,2]);proxy=cfg['rpm']*math.pi/30*radius*cfg['target_delta_t_s']/np.maximum(wt,1e-300)
 tangent_ratio=cfg['minimum_tangential_width_mm']*.001/np.maximum(wt,1e-300)
 offwall_ratio=np.where(geometry['wall_adjacent'],0,cfg['minimum_nonwall_directional_width_mm']*.001/np.maximum(wn,1e-300))
 score=np.maximum.reduce([tangent_ratio,offwall_ratio,proxy/cfg['max_co_proxy']]);flag=score>1
 return dict(flagged=flag,score=score,rotation_only_Co_proxy=proxy,centres_m=p,tangential_width_m=wt,minimum_directional_width_m=wn)

def check_native_mesh(source,output,config=None,scale=.001,rotor_zone='rotaryregion'):
 cfg=settings(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 if cfg['mode']=='off':return dict(status='disabled',flagged_cells=0)
 geometry=measure_native(source,rotor_zone);result=assess_geometry(geometry,cfg,scale);ids=np.flatnonzero(result['flagged']);rank=ids[np.argsort(result['score'][ids])[::-1]]
 def row(i):
  return dict(native_cell_1based=int(geometry['native_cell_1based'][i]),x_mm=float(result['centres_m'][i,0]*1000),y_mm=float(result['centres_m'][i,1]*1000),z_mm=float(result['centres_m'][i,2]*1000),volume_mm3=float(geometry['volume'][i]*scale**3*1e9),tangential_width_mm=float(result['tangential_width_m'][i]*1000),minimum_directional_width_mm=float(result['minimum_directional_width_m'][i]*1000),wall_adjacent=bool(geometry['wall_adjacent'][i]),rotation_only_Co_proxy=float(result['rotation_only_Co_proxy'][i]),risk_score=float(result['score'][i]))
 fields=list(row(0))
 with (output/'flagged-cells.csv').open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(row(i) for i in rank)
 report=dict(source=str(Path(source).resolve()),settings=cfg,rotor_zone=rotor_zone,rotor_cells=len(geometry['native_cell_1based']),flagged_cells=len(ids),flagged_cell_percent=100*len(ids)/len(geometry['native_cell_1based']),status='flags_found' if len(ids) else 'no_flags',worst_cells=[row(i) for i in rank[:100]],scope='Native coordinates must use the pipeline frame: rotation about +Y through origin. Rotation-only Co proxy is a geometry ranking, not a fluid-relative Co prediction or bound. Wall-normal thinness alone does not flag a wall-adjacent layer cell. Standard checkMesh remains independent.')
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(f"Sliver screen: {len(ids)} / {len(geometry['native_cell_1based'])} rotor cells flagged; {output/'report.json'}",flush=True)
 if len(ids) and cfg['mode']=='reject':raise ValueError('Directional cell-width gate rejected mesh; see '+str(output/'report.json'))
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mesh',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--scale',type=float,default=.001);parser.add_argument('--rotor-zone',default='rotaryregion');args=parser.parse_args()
 config=json.loads(args.config.read_text()) if args.config else None
 try:check_native_mesh(args.mesh,args.output,config,args.scale,args.rotor_zone)
 except ValueError as exc:parser.exit(2,str(exc)+'\n')

if __name__=='__main__':main()

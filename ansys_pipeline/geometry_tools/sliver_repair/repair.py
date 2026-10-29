"""Bounded scan/repair/rescan of internal rotor cells; never launches a flow solver."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,math,time,shutil,hashlib
import numpy as np,h5py
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from native_geometry import load_native

def topology_digest(path):
 digest=hashlib.sha256()
 with h5py.File(path) as h:
  def collect(name,obj):
   digest.update(name.encode())
   for key in sorted(obj.attrs):digest.update((key+repr(np.asarray(obj.attrs[key]).tolist())).encode())
   if isinstance(obj,h5py.Dataset):
    value=obj[()];digest.update(str(value.shape).encode());digest.update(value.dtype.str.encode())
    digest.update(repr(value.tolist()).encode() if value.dtype.kind in 'OSU' else value.tobytes())
  for name in ('faces','cells'):
   h['meshes/1/'+name].visititems(collect)
 return digest.hexdigest()

def validate_settings(cfg):
 integers=('max_passes','max_attempts_per_pass','max_cluster_cells','max_function_evaluations')
 numbers=('rpm','target_delta_t_s','max_co_screen','speed_margin','minimum_tangent_floor_mm','minimum_normal_floor_mm','maximum_normal_target_mm','max_node_motion_mm','max_layer_distance_change_fraction','max_walltime_seconds','seed_radius_min_mm')
 for name in integers:
  if type(cfg[name]) is not int or cfg[name]<=0:raise ValueError('Expected positive integer: '+name)
 for name in numbers:
  value=cfg[name]
  if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:raise ValueError('Expected finite positive setting: '+name)
 if cfg['max_layer_distance_change_fraction']>=1:raise ValueError('Layer-distance tolerance must be less than one')
 if not isinstance(cfg['rotor_zone'],str) or not cfg['rotor_zone']:raise ValueError('Specify rotor_zone')
 if any(type(c) is not int or c<1 for c in cfg.get('priority_cells',[])):raise ValueError('Priority cells use positive native IDs')
 return cfg

def face_geometry(xyz,polys):
 sizes=np.array([len(p) for p in polys]);off=np.r_[0,np.cumsum(sizes)];flat=np.concatenate(polys);rep=np.repeat(np.arange(len(sizes)),sizes);following=np.arange(len(flat))+1;following[off[1:]-1]=off[:-1]
 def calculate():
  p=xyz[flat];q=p[following];ref=np.add.reduceat(p,off[:-1])/sizes[:,None];cross=np.cross(p-ref[rep],q-ref[rep])/2;area=np.add.reduceat(cross,off[:-1]);unit=area/np.maximum(np.linalg.norm(area,axis=1)[:,None],1e-30);w=np.einsum('ij,ij->i',cross,unit[rep]);fc=np.add.reduceat((p+q+ref[rep])/3*w[:,None],off[:-1])/np.maximum(np.add.reduceat(w,off[:-1])[:,None],1e-30)
  return fc,-area
 return calculate,flat,off,rep

def widths(area,volume,centre):
 radius=max(float(np.hypot(centre[0],centre[2])),1e-30);tangent=np.array([centre[2]/radius,0,-centre[0]/radius]);normal=area/np.maximum(np.linalg.norm(area,axis=1)[:,None],1e-30)
 return np.array([2*volume/max(float(np.abs(area@tangent).sum()),1e-30),2*volume/max(float(np.abs(normal@area.T).sum(1).max()),1e-30)])

def targets(centres,cfg):
 radius=np.hypot(centres[:,0],centres[:,2]);t=np.maximum(cfg['minimum_tangent_floor_mm'],cfg['speed_margin']*cfg['rpm']*math.pi/30*radius*cfg['target_delta_t_s']/cfg['max_co_screen'])
 return np.c_[t,np.maximum(cfg['minimum_normal_floor_mm'],np.minimum(cfg['maximum_normal_target_mm'],.5*t))]

def promote_floor(previous,current,target):
 """Retain achieved width gains, capped at the target, across later repairs."""
 return np.maximum(previous,np.minimum(current,target))

def scan(mesh,cfg,baseline):
 cells=np.arange(mesh.lo,mesh.hi+1);wanted=targets(mesh.centre[cells],cfg);risk={};wt=[]
 for c,target in zip(cells,wanted):
  c=int(c)
  if c in mesh.wall_cells or np.hypot(mesh.centre[c,0],mesh.centre[c,2])<cfg['seed_radius_min_mm']:continue
  w=widths(mesh.sf[list(mesh.cf[c])],mesh.vol[c],mesh.centre[c]);baseline[c]=promote_floor(baseline.get(c,w),w,target);score=float(np.max(target/np.maximum(w,1e-30)))
  if score>1.02:risk[c]=dict(score=score,widths_mm=w.tolist(),target_mm=target.tolist(),centre_mm=mesh.centre[c].tolist())
 blade=np.flatnonzero(mesh.blade_mask);tree=cKDTree(mesh.fc[blade]);out=-mesh.sf[blade]/np.maximum(np.linalg.norm(mesh.sf[blade],axis=1)[:,None],1e-30)
 if risk:
  ids=list(risk);p=mesh.centre[ids];distance,near=tree.query(p);r=np.hypot(p[:,0],p[:,2]);t=np.c_[p[:,2]/r,np.zeros(len(r)),-p[:,0]/r];alignment=np.einsum('ij,ij->i',out[near],t);R=float(np.hypot(mesh.fc[blade,0],mesh.fc[blade,2]).max())
  for c,d,a,rr in zip(ids,distance,alignment,r):risk[c].update(leading_transition=bool(d<4 and a>.35 and rr>.6*R),wall_distance_estimate_mm=float(d),leading_alignment=float(a))
 return risk

def local_problem(mesh,central,risk,cfg,nodecells,baseline):
 adjacent=set()
 for f in mesh.cf[central]:adjacent.update((int(mesh.owner[f]),int(mesh.nei[f])))
 if cfg.get('include_vertex_clusters',False):
  for f in mesh.cf[central]:
   for n in mesh.polys[f]:adjacent.update(nodecells.get(int(n),()))
 selected=[central]+sorted((c for c in adjacent if c!=central and c in risk),key=lambda c:-risk[c]['score'])[:cfg['max_cluster_cells']-1]
 nodeset=set()
 for c in selected:
  for f in mesh.cf[c]:nodeset.update(mesh.polys[f].tolist())
 nodes=np.array(sorted(nodeset-mesh.fixed),dtype=int)
 if not len(nodes):return None,'protected_nodes'
 affected=set()
 for n in nodes:affected.update(nodecells.get(int(n),()))
 affected.discard(0)
 if not affected or any(c<mesh.lo or c>mesh.hi for c in affected):return None,'interface_neighborhood'
 checkids=sorted(set().union(*(mesh.cf[c] for c in affected)));needed=set()
 for f in checkids:needed.update((int(mesh.owner[f]),int(mesh.nei[f])))
 needed.discard(0)
 if any(c<mesh.lo or c>mesh.hi for c in needed):return None,'interface_neighborhood'
 needed=sorted(needed);ci={c:i for i,c in enumerate(needed)};ids=np.array(sorted(set().union(*(mesh.cf[c] for c in needed))),dtype=int);fi={int(f):i for i,f in enumerate(ids)};checks=np.array([fi[f] for f in checkids]);polys=[mesh.polys[int(f)] for f in ids]
 calculate,flat,off,rep=face_geometry(mesh.xyz,polys);owners=mesh.owner[ids];neis=mesh.nei[ids];own=np.array([ci.get(int(c),-1) for c in owners]);nei=np.array([ci.get(int(c),-1) for c in neis]);cell_faces={c:np.array([fi[f] for f in mesh.cf[c]]) for c in needed};affected=sorted(affected);ac=np.array([ci[c] for c in affected]);wanted=targets(mesh.centre[affected],cfg)
 original=mesh.xyz[nodes].copy();global_original=mesh.original_xyz[nodes];cellnodes={c:set(np.concatenate([mesh.polys[f] for f in mesh.cf[c]]).tolist()) for c in needed}
 def geometry(values):
  mesh.xyz[nodes]=values.reshape(-1,3);fc,sf=calculate();estimate=np.zeros((len(needed),3));count=np.zeros(len(needed))
  for c in (own,nei):
   good=c>=0;np.add.at(estimate,c[good],fc[good]);np.add.at(count,c[good],1)
  estimate/=count[:,None];v=np.zeros(len(needed));moment=np.zeros((len(needed),3))
  for c,sign in ((own,1),(nei,-1)):
   good=c>=0;k=c[good];p=sign*np.einsum('ij,ij->i',sf[good],fc[good]-estimate[k]);np.add.at(v,k,p);np.add.at(moment,k,p[:,None]*(3*fc[good]+estimate[k])/4)
  cc=moment/np.where(np.abs(v)>1e-30,v,1e-30)[:,None];volume=v/3;f=checks;oc=own[f];nc=nei[f];internal=neis[f]>0;normal=sf[f]/np.maximum(np.linalg.norm(sf[f],axis=1)[:,None],1e-30);cpf=fc[f]-cc[oc];d=normal*np.einsum('ij,ij->i',normal,cpf)[:,None];d[internal]=cc[nc[internal]]-cc[oc[internal]];den=np.einsum('ij,ij->i',sf[f],d);sv=cpf-np.einsum('ij,ij->i',sf[f],cpf)[:,None]/np.where(np.abs(den)>1e-30,den,1e-30)[:,None]*d;mag=np.linalg.norm(sv,axis=1);hat=sv/np.maximum(mag[:,None],1e-30);allhat=np.zeros_like(fc);allhat[f]=hat;fd=np.maximum(np.where(internal,.2,.4)*np.linalg.norm(d,axis=1),np.maximum.reduceat(np.abs(np.einsum('ij,ij->i',mesh.xyz[flat]-fc[rep],allhat[rep])),off[:-1])[f]);skew=mag/np.maximum(fd,1e-30);od=np.einsum('ij,ij->i',normal,cpf);nd=np.ones(len(f));nd[internal]=np.einsum('ij,ij->i',normal[internal],cc[nc[internal]]-fc[f[internal]]);orth=np.ones(len(f));orth[internal]=np.einsum('ij,ij->i',normal[internal],d[internal])/np.maximum(np.linalg.norm(d[internal],axis=1),1e-30)
  w=np.array([widths(sf[cell_faces[c]],volume[ci[c]],cc[ci[c]]) for c in affected]);return dict(fc=fc,sf=sf,cc=cc,volume=volume,skew=skew,od=od,nd=nd,orth=orth,widths=w)
 initial=geometry(original.ravel());guard=[]
 for i,c in enumerate(affected):
  baseline.setdefault(c,initial['widths'][i].copy());guard.append(np.minimum(baseline[c],wanted[i]))
 guard=np.array(guard);selected_mask=np.array([c in selected for c in affected]);goal=guard.copy();goal[selected_mask]=wanted[selected_mask];wall=np.array([mesh.nei[int(f)]==0 for f in checkids]);h0=np.array([mesh.original_wall_distance[int(f)] for f in np.array(checkids)[wall]]);hscale=np.maximum(np.abs(h0),.01);skew_goal=np.maximum(initial['skew'],3.5);v0=np.maximum(initial['volume'],1e-10);od0=np.maximum(np.abs(initial['od']),.01);nd0=np.maximum(np.abs(initial['nd']),.01)
 def residual(values):
  state=geometry(values);wd=state['od'][wall]
  return np.r_[np.maximum(state['skew']-skew_goal,0),10*np.maximum(.03-state['od']/od0,0),10*np.maximum(.03-state['nd']/nd0,0),10*np.maximum(.001-state['orth'],0),10*np.maximum(.05-state['volume']/v0,0),10*np.maximum((goal-state['widths'])/np.maximum(goal,.001),0).T.ravel(),10*np.maximum(np.abs(wd-h0)-cfg['max_layer_distance_change_fraction']*np.abs(h0),0)/hscale,.03*(values-original.ravel())/2]
 nface=len(checks);ncell=len(needed);na=len(affected);nw=int(wall.sum());nv=3*len(nodes);pattern=lil_matrix((4*nface+ncell+2*na+nw+nv,nv),dtype=int);index={int(n):i for i,n in enumerate(nodes)}
 def columns(ns):return [3*index[n]+j for n in ns if n in index for j in range(3)]
 for row,f in enumerate(checks):
  cols=columns(cellnodes.get(int(owners[f]),set())|cellnodes.get(int(neis[f]),set()))
  for block in range(4):pattern[block*nface+row,cols]=1
 for row,c in enumerate(needed):pattern[4*nface+row,columns(cellnodes[c])]=1
 for row,c in enumerate(affected):
  cols=columns(cellnodes[c]);pattern[4*nface+ncell+row,cols]=1;pattern[4*nface+ncell+na+row,cols]=1
 for row,f in enumerate(checks[wall]):pattern[4*nface+ncell+2*na+row,columns(cellnodes[int(owners[f])])]=1
 for j in range(nv):pattern[4*nface+ncell+2*na+nw+j,j]=1
 bound=cfg['max_node_motion_mm']
 try:fit=least_squares(residual,original.ravel(),bounds=((global_original-bound).ravel(),(global_original+bound).ravel()),jac_sparsity=pattern.tocsr(),max_nfev=cfg['max_function_evaluations'],ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-6)
 except Exception:
  mesh.xyz[nodes]=original
  raise
 final=geometry(fit.x);before=float(np.max(wanted[selected_mask]/initial['widths'][selected_mask]));after=float(np.max(wanted[selected_mask]/np.maximum(final['widths'][selected_mask],1e-30)));valid=(np.all(final['volume']>v0*.045) and np.all(final['od']>0) and np.all(final['nd']>0) and np.all(final['orth']>.0005) and np.all(final['skew']<=np.maximum(initial['skew']+1e-4,3.95)) and np.all(final['widths']>=guard*.995) and np.all(np.abs(final['od'][wall]-h0)<=cfg['max_layer_distance_change_fraction']*np.abs(h0)+1e-6) and after<max(1.02,before*.9))
 if not valid:mesh.xyz[nodes]=original;return None,'acceptance_guard'
 mesh.fc[ids]=final['fc'];mesh.sf[ids]=final['sf'];mesh.centre[needed]=final['cc'];mesh.vol[needed]=final['volume']
 for i,c in enumerate(affected):baseline[c]=promote_floor(baseline[c],final['widths'][i],wanted[i])
 return dict(central_cell=central,targets=sorted(selected),affected_cells=affected,moved_nodes=nodes.tolist(),maximum_local_node_motion_mm=float(np.linalg.norm(mesh.xyz[nodes]-original,axis=1).max()),before_score=before,after_score=after,initial_widths_mm=initial['widths'][selected_mask].tolist(),final_widths_mm=final['widths'][selected_mask].tolist(),max_local_skew=float(final['skew'].max()),function_evaluations=int(fit.nfev)),None

def save_mesh(mesh,source,output,progress):
 fixed=np.array(sorted(mesh.fixed),dtype=int)
 if not np.array_equal(mesh.xyz[fixed],mesh.original_xyz[fixed]):raise RuntimeError('Fixed physical/interface nodes changed')
 if not output.exists():shutil.copy2(source,output)
 changed=np.flatnonzero(np.any(mesh.xyz!=mesh.original_xyz,axis=1));np.savez_compressed(progress/'node-changes.npz',nodes=changed,original=mesh.original_xyz[changed],current=mesh.xyz[changed])
 with h5py.File(output,'r+') as h:
  group=h['meshes/1/nodes/coords'];offset=0
  for key in sorted(list(group),key=int):
   ds=group[key];count=len(ds);sel=changed[(changed>=offset)&(changed<offset+count)]
   if len(sel):
    values=ds[()];attrs=dict(ds.attrs);values[sel-offset]=mesh.xyz[sel];del group[key];new=group.create_dataset(key,data=values,chunks=(min(4096,count),3),compression='gzip',compression_opts=4)
    for name,value in attrs.items():new.attrs[name]=value
    new.attrs['chunkDim']=np.array([min(4096,count)],dtype=np.uint64)
   offset+=count
 return len(changed)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--settings',type=Path,required=True);parser.add_argument('--scan-only',action='store_true');parser.add_argument('--require-clear',action='store_true');parser.add_argument('--start-from',type=Path,help='Resume coordinates from a topology-identical checkpoint, retaining original input bounds');args=parser.parse_args();cfg=validate_settings(json.loads(args.settings.read_text()));progress=args.output.parent;progress.mkdir(parents=True,exist_ok=True)
 if args.source.resolve()==args.output.resolve():raise ValueError('Keep input mesh read-only; use a distinct output')
 if args.output.exists():raise FileExistsError(args.output)
 if args.start_from and topology_digest(args.source)!=topology_digest(args.start_from):raise ValueError('Checkpoint connectivity differs from original input')
 start=time.monotonic();mesh=load_native(args.source,cfg['rotor_zone']);baseline={};source_risk=scan(mesh,cfg,baseline)
 blade_faces=np.flatnonzero(mesh.blade_mask)
 radius=float(np.hypot(mesh.fc[blade_faces,0],mesh.fc[blade_faces,2]).max())
 if not 10<radius<1000:raise ValueError('Expected native coordinates in mm, propeller radius 10 to 1000 mm, rotation frame about Y')
 for c in mesh.wall_cells:baseline[c]=widths(mesh.sf[list(mesh.cf[c])],mesh.vol[c],mesh.centre[c])
 original_xyz=mesh.xyz;wall_distance={int(f):float(np.dot(mesh.sf[f],mesh.fc[f]-mesh.centre[mesh.owner[f]])/max(np.linalg.norm(mesh.sf[f]),1e-30)) for f in mesh.wall_faces}
 if args.start_from:
  signature=(mesh.N,mesh.F);del mesh
  mesh=load_native(args.start_from,cfg['rotor_zone'])
  if (mesh.N,mesh.F)!=signature:raise ValueError('Checkpoint topology differs from original mesh')
  mesh.original_xyz=original_xyz
  fixed=np.array(sorted(mesh.fixed),dtype=int)
  if not np.array_equal(mesh.xyz[fixed],original_xyz[fixed]):raise ValueError('Checkpoint changed fixed boundary/interface coordinates')
 mesh.original_wall_distance=wall_distance
 risk=scan(mesh,cfg,baseline);initial_risk=risk.copy();print('INITIAL SCAN',len(risk),'non-wall rotor risks;',sum(r['leading_transition'] for r in risk.values()),'in outer leading transition; original input risks',len(source_risk),flush=True);(progress/'initial-scan.json').write_text(json.dumps(risk,indent=2));(progress/'settings.resolved.json').write_text(json.dumps(cfg,indent=2));accepted=[];passes=[];blocked={};version=Counter();reason=Counter();escalated=False
 if args.scan_only:return
 for iteration in range(cfg['max_passes']):
  if time.monotonic()-start>=cfg['max_walltime_seconds'] or (progress/'STOP').exists():break
  priority=set(cfg.get('priority_cells',[]));eligible=[c for c in risk if blocked.get(c,-1)!=version[c]];ranked=sorted(eligible,key=lambda c:(c not in priority,-risk[c]['score']*(1.25 if risk[c]['leading_transition'] else 1)))[:cfg['max_attempts_per_pass']]
  if not ranked:
   retry=cfg.get('retry_cluster_cells',cfg['max_cluster_cells'])
   if not escalated and retry>cfg['max_cluster_cells']:
    cfg=dict(cfg,max_cluster_cells=retry,include_vertex_clusters=True);blocked.clear();escalated=True;print('ESCALATING TO VERTEX CLUSTERS',retry,flush=True);continue
   break
  candidate_nodes=set()
  for c in ranked:
   for f in mesh.cf[c]:candidate_nodes.update(mesh.polys[f].tolist())
   for f in mesh.cf[c]:
    for q in (int(mesh.owner[f]),int(mesh.nei[f])):
     if q in risk:
      for ff in mesh.cf[q]:candidate_nodes.update(mesh.polys[ff].tolist())
  if cfg.get('include_vertex_clusters',False) and len(risk)<=3000:
   for c in risk:
    for f in mesh.cf[c]:candidate_nodes.update(mesh.polys[f].tolist())
  nodecells={n:set() for n in candidate_nodes}
  for f,p in mesh.polys.items():
   for n in candidate_nodes.intersection(p):nodecells[n].update((int(mesh.owner[f]),int(mesh.nei[f])))
  count=0
  for c in ranked:
   if time.monotonic()-start>=cfg['max_walltime_seconds'] or (progress/'STOP').exists():break
   current=widths(mesh.sf[list(mesh.cf[c])],mesh.vol[c],mesh.centre[c]);target=targets(mesh.centre[[c]],cfg)[0]
   if np.max(target/current)<=1.02:continue
   try:result,failure=local_problem(mesh,c,risk,cfg,nodecells,baseline)
   except Exception as exc:
    save_mesh(mesh,args.source,args.output,progress)
    raise RuntimeError(f'Repair exception at cell {c}; checkpoint retained for inspection') from exc
   if result:
    accepted.append(result);count+=1
    for q in result['affected_cells']:version[q]+=1
    with (progress/'repairs.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    print('ACCEPTED',len(accepted),'cell',c,'score',round(result['before_score'],3),'->',round(result['after_score'],3),flush=True)
   else:blocked[c]=version[c];reason[failure]+=1
  changed=save_mesh(mesh,args.source,args.output,progress);risk=scan(mesh,cfg,baseline);entry=dict(pass_index=iteration+1,accepted=count,remaining_risks=len(risk),remaining_leading_transition=sum(r['leading_transition'] for r in risk.values()),changed_nodes=changed,elapsed_s=time.monotonic()-start);passes.append(entry);print('PASS',json.dumps(entry),flush=True);(progress/'latest-scan.json').write_text(json.dumps(risk,indent=2))
  if not risk:break
 changed=save_mesh(mesh,args.source,args.output,progress);report=dict(source=str(args.source.resolve()),output=str(args.output.resolve()),cells=mesh.N,initial_risks=len(initial_risk),remaining_risks=len(risk),accepted_repairs=len(accepted),changed_nodes=changed,maximum_total_node_motion_mm=float(np.linalg.norm(mesh.xyz-mesh.original_xyz,axis=1).max()),physical_boundary_nodes_fixed=True,passes=passes,blocked_reasons=dict(reason),flow_solver_run=False,independent_checkMesh_required=True,stop_reason='thresholds_met' if not risk else 'bounded_run_with_unresolved_flags',elapsed_s=time.monotonic()-start);(progress/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 if args.require_clear and risk:raise SystemExit(3)

if __name__=='__main__':main()

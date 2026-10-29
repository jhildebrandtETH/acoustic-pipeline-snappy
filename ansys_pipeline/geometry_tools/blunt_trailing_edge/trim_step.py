"""Deterministic, blade-only blunt trailing-edge trimming of two-bladed STEP solids.

All dimensions are millimetres. Retained CAD surfaces are Boolean-trimmed, never
lofted/rebuilt. Only the cutting solid is constructed from local section planes.
Run: python trim_step.py settings.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np


def frame_vectors(config):
    origin = np.asarray(config['origin_mm'], float)
    axis = np.asarray(config['rotation_axis'], float)
    span = np.asarray(config['positive_blade_axis'], float)
    if origin.shape != (3,) or axis.shape != (3,) or span.shape != (3,):
        raise ValueError('Frame vectors must have three components.')
    if not np.isfinite(np.r_[origin, axis, span]).all():
        raise ValueError('Frame vectors must be finite.')
    if np.linalg.norm(axis) < 1e-12 or np.linalg.norm(span) < 1e-12:
        raise ValueError('Frame axes must be nonzero.')
    axis /= np.linalg.norm(axis); span /= np.linalg.norm(span)
    if abs(axis @ span) > 1e-8:
        raise ValueError('Rotation and blade-span axes must be perpendicular.')
    chord = np.cross(axis, span)
    return origin, axis, span, chord


def load_settings(path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema_version') not in [1,2]:
        raise ValueError('Unsupported settings schema_version.')
    allowed = {'schema_version','input_step','output_directory','setback_mm','frame',
               'hub','station_spacing_mm','section_deflection_mm',
               'validation_tolerance_mm','overlay_radius_fractions','overwrite','station_strategy'}
    if set(cfg) - allowed:
        raise ValueError(f'Unknown settings: {sorted(set(cfg)-allowed)}')
    cfg.setdefault('station_strategy','uniform' if cfg['schema_version']==1 else 'cad_seams')
    if cfg['station_strategy'] not in ['uniform','cad_seams']:
        raise ValueError('station_strategy must be uniform or cad_seams.')
    for key in ['setback_mm','station_spacing_mm','section_deflection_mm','validation_tolerance_mm']:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    frame_vectors(cfg['frame'])
    if cfg['frame']['positive_blade_trailing_edge'] not in ['min_chord_coordinate','max_chord_coordinate']:
        raise ValueError('Set the positive-blade trailing edge to min_chord_coordinate or max_chord_coordinate.')
    if not cfg['overlay_radius_fractions'] or any(not 0 < r < 1 for r in cfg['overlay_radius_fractions']):
        raise ValueError('Overlay fractions must be between zero and one.')
    hr = cfg['hub']['radius_mm']
    if hr != 'auto' and (not isinstance(hr,(int,float)) or not math.isfinite(hr) or hr <= 0):
        raise ValueError('Hub radius must be auto or a positive number.')
    for key in ['input_step','output_directory']:
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    if Path(cfg['input_step']).suffix.lower() not in ['.step','.stp']:
        raise ValueError('Input must be a STEP file.')
    return cfg


def run(config_path):
    cfg = load_settings(config_path)
    # Optional CAD dependency is isolated from the normal CFD pipeline imports.
    from OCP import __version__ as ocp_version
    from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Pln, gp_Trsf

    def static(obj, name):
        return getattr(obj, name+'_s', None) or getattr(obj, name)

    def entities(shape, kind):
        ex = TopExp_Explorer(shape, kind)
        while ex.More():
            yield ex.Current(); ex.Next()

    def volume(shape):
        if shape.IsNull(): return 0.0
        # Touching solids can return faces/edges in a Common operation. Applying
        # volume integration to those open faces gives a spurious nonzero value.
        total=0.0
        for solid in entities(shape,TopAbs_SOLID):
            props=GProp_GProps();static(BRepGProp,'VolumeProperties')(solid,props)
            total+=abs(props.Mass())
        return total

    def boolean(cls, a, z):
        op = cls(a,z); op.SetRunParallel(False); op.Build()
        if not op.IsDone(): raise RuntimeError(f'{cls.__name__} failed.')
        return op.Shape()

    source = Path(cfg['input_step']); out = Path(cfg['output_directory'])
    if not source.is_file(): raise FileNotFoundError(source)
    if out == source.parent or source.is_relative_to(out):
        raise ValueError('Use a separate output directory, outside the input file path.')
    if out.exists() and any(out.iterdir()) and not cfg['overwrite']:
        raise FileExistsError('Output directory is not empty. Choose a new directory or explicitly set overwrite=true.')
    out.mkdir(parents=True,exist_ok=True)
    stem = source.stem+f'_blunt{cfg["setback_mm"]:.3f}mm'
    destination = out/(stem+'.step')
    if destination.resolve() == source.resolve(): raise ValueError('Input cannot be overwritten.')
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone: raise RuntimeError('STEP read failed.')
    reader.TransferRoots(); original = reader.OneShape()
    if original.IsNull() or not BRepCheck_Analyzer(original).IsValid():
        raise ValueError('Input CAD is invalid. No automatic healing or smoothing is permitted.')
    solids_before = len(list(entities(original,TopAbs_SOLID)))
    if solids_before != 1: raise ValueError('Expected one connected propeller solid; export the propeller alone.')
    origin, axis, span, chord = frame_vectors(cfg['frame'])
    def pnt(q): return gp_Pnt(*map(float,q))
    def direction(q): return gp_Dir(*map(float,q))
    transform=gp_Trsf();matrix=np.vstack([chord,axis,span]);shift=-matrix@origin
    transform.SetValues(*[float(v) for row,d in zip(matrix,shift) for v in [*row,d]])
    local_shape=BRepBuilderAPI_Transform(original,transform,True).Shape()
    box = Bnd_Box(); static(BRepBndLib,'AddOptimal')(local_shape,box,False,False)
    lo,hi = box.CornerMin(),box.CornerMax()
    limits = (lo.X(),lo.Y(),lo.Z(),hi.X(),hi.Y(),hi.Z())
    extents = {1:float(hi.Z()), -1:float(-lo.Z())}
    radius = max(extents.values())
    if min(extents.values()) <= 0: raise ValueError('Origin must lie between the two blade tips.')
    if cfg['hub']['radius_mm'] == 'auto':
        candidates = []
        for face in entities(original,TopAbs_FACE):
            sur = BRepAdaptor_Surface(static(TopoDS,'Face')(face))
            if sur.GetType() != GeomAbs_Cylinder: continue
            cy = sur.Cylinder(); d = cy.Axis().Direction(); loc = cy.Location()
            v = np.array([d.X(),d.Y(),d.Z()]); x = np.array([loc.X(),loc.Y(),loc.Z()])-origin
            if abs(v @ axis) > 1-1e-8 and np.linalg.norm(x-axis*(x@axis)) < 1e-5:
                candidates.append(cy.Radius())
        if not candidates: raise ValueError('No coaxial cylindrical hub found. Supply hub.radius_mm and correct frame.')
        hub_radius = max(candidates)
    else: hub_radius = float(cfg['hub']['radius_mm'])
    if hub_radius >= .4*min(extents.values()): raise ValueError('Hub radius is implausibly large for this frame.')
    reach = max(float(np.linalg.norm(np.array(limits[3:])-np.array(limits[:3]))),2*radius)
    # Extend beyond all cutter caps: coincident hub/cutter end planes can cause
    # Boolean ambiguity on untwisted blades.
    hub = BRepPrimAPI_MakeCylinder(gp_Ax2(pnt(origin-axis*3*reach),direction(axis)),hub_radius,6*reach).Shape()

    # Track the edge using a controlled CAD tessellation. Direct NURBS/plane
    # intersections can omit some input STEP faces despite a valid solid. This
    # tessellation defines only the cutter; original CAD faces are still Boolean
    # trimmed, not replaced with mesh or reconstructed profiles.
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    section_meshes=[]
    def triangulated(shape,label):
        mesh=BRepMesh_IncrementalMesh(shape,cfg['section_deflection_mm'],False,.10,False)
        if not mesh.IsDone():raise RuntimeError('CAD tessellation failed.')
        path=out/(label+'_analysis_mm.stl')
        stl=StlAPI_Writer();stl.ASCIIMode=False
        if not stl.Write(shape,str(path)):raise RuntimeError('Analysis tessellation export failed.')
        rd=vtk.vtkSTLReader();rd.SetFileName(str(path));rd.Update()
        p=vtk.vtkPolyData();p.DeepCopy(rd.GetOutput());return p
    original_mesh=triangulated(original,'original')
    section_meshes.append((original,original_mesh))
    def cad_edges(shape):
        seen=[]
        for edge in entities(shape,TopAbs_EDGE):
            if any(edge.IsSame(e) for e in seen):continue
            seen.append(edge)
            c=BRepAdaptor_Curve(static(TopoDS,'Edge')(edge))
            sam=GCPnts_QuasiUniformDeflection(c,cfg['section_deflection_mm'])
            if not sam.IsDone():raise RuntimeError('CAD edge sampling failed.')
            yield np.array([[sam.Value(i).X(),sam.Value(i).Y(),sam.Value(i).Z()] for i in range(1,sam.NbPoints()+1)])
    seam_positions=[]
    if cfg['station_strategy']=='cad_seams':
        for points in cad_edges(original):
            z=(points-origin)@span;x=(points-origin)@chord
            if np.ptp(z)<1e-6 and np.ptp(x)>max(.01,5*cfg['section_deflection_mm']):
                seam_positions.append(float(z.mean()))
        # Existing airfoil profile edges occur on both blade sides at the same
        # span. Deduplicate them without fitting or moving their CAD locations.
        grouped={}
        for z in seam_positions:grouped.setdefault(round(z,6),[]).append(z)
        seam_positions=sorted(float(np.mean(z)) for z in grouped.values())
    def section(shape, z):
        match=next((mesh for s,mesh in section_meshes if shape.IsSame(s)),None)
        if match is None:
            match=triangulated(shape,'trimmed');section_meshes.append((shape,match))
        plane=vtk.vtkPlane();plane.SetOrigin(*(origin+span*z));plane.SetNormal(*span)
        cut=vtk.vtkCutter();cut.SetInputData(match);cut.SetCutFunction(plane);cut.Update()
        clean=vtk.vtkCleanPolyData();clean.SetInputConnection(cut.GetOutputPort());clean.ToleranceIsAbsoluteOn();clean.SetAbsoluteTolerance(1e-6);clean.Update()
        strip=vtk.vtkStripper();strip.SetInputConnection(clean.GetOutputPort());strip.JoinContiguousSegmentsOn();strip.Update();p=strip.GetOutput()
        if not p.GetNumberOfPoints():raise RuntimeError(f'No section edges at {z:g} mm. Check the frame.')
        coords=vtk_to_numpy(p.GetPoints().GetData()).astype(float);offsets=vtk_to_numpy(p.GetLines().GetOffsetsArray());ids=vtk_to_numpy(p.GetLines().GetConnectivityArray())
        return [coords[ids[a:b]] for a,b in zip(offsets[:-1],offsets[1:]) if b-a>=2]

    def cut_frame(curves,sign):
        points = np.concatenate(curves); xs = (points-origin)@chord
        te_min = (cfg['frame']['positive_blade_trailing_edge']=='min_chord_coordinate') == (sign==1)
        te_x, le_x = (xs.min(),xs.max()) if te_min else (xs.max(),xs.min())
        te = points[np.abs(xs-te_x)<1e-6].mean(0); le = points[np.abs(xs-le_x)<1e-6].mean(0)
        e = le-te; e -= span*(e@span); length = np.linalg.norm(e)
        if length <= cfg['setback_mm']: raise ValueError(f'Setback consumes local chord {length:g} mm at span {float((te-origin)@span):g} mm.')
        e /= length; normal = np.cross(span,e)
        return te+cfg['setback_mm']*e, e, normal

    cutters=[]; frames={}; started=time.time()
    for sign in [1,-1]:
        end = extents[sign]; last = end-max(1e-4, end*1e-7)
        if cfg['station_strategy']=='cad_seams':
            native=[sign*z for z in seam_positions if 1e-6<sign*z<end-1e-6]
            if not native:print(f'Blade {sign:+d} has no interior CAD section seams; using a single ruled span.',flush=True)
            stations=np.unique(np.r_[0,native,end])
        else:
            stations = np.linspace(0,last,max(3,math.ceil(last/cfg['station_spacing_mm'])+1))
            # Preserve schema-1 construction exactly for legacy reproducibility.
            extra=[end-d for d in [.001,.002,.005,.01,.02,.05,.1,.2,.5,1.] if 0<end-d<last]
            stations=np.unique(np.r_[stations,extra])
        loft = BRepOffsetAPI_ThruSections(True,True,1e-7); loft.CheckCompatibility(False)
        blade_frames=[]
        for i,z in enumerate(stations):
            sampled_z=min(z,last)
            cut,e,n = cut_frame(section(original,sign*sampled_z),sign)
            cut+=span*(sign*(z-sampled_z))
            blade_frames.append({'span_mm':float(sign*z),'cut_point_mm':cut.tolist(),'chord_direction':e.tolist()})
            # Only cutter surfaces are lofted; original blade geometry is not rebuilt.
            for offset in ([-.01] if i==0 and cfg['station_strategy']=='uniform' else [0]):
                c = cut+span*(sign*offset)
                poly=BRepBuilderAPI_MakePolygon()
                for q in [c+n*reach,c-n*reach,c-n*reach-e*reach,c+n*reach-e*reach]: poly.Add(pnt(q))
                poly.Close(); loft.AddWire(poly.Wire())
        cut,e,n=cut_frame(section(original,sign*last),sign)
        cut += span*(sign*(end-last+1.0))
        poly=BRepBuilderAPI_MakePolygon()
        for q in [cut+n*reach,cut-n*reach,cut-n*reach-e*reach,cut+n*reach-e*reach]: poly.Add(pnt(q))
        poly.Close();loft.AddWire(poly.Wire());loft.Build()
        if not loft.IsDone() or not BRepCheck_Analyzer(loft.Shape()).IsValid(): raise RuntimeError('Cutting-solid loft failed.')
        cutters.append(boolean(BRepAlgoAPI_Cut,loft.Shape(),hub));frames[str(sign)]=blade_frames
        print(f'Constructed blade {sign:+d} cutter from {len(stations)} sections.',flush=True)
    result=original
    for i,cutter in enumerate(cutters,1):
        result=boolean(BRepAlgoAPI_Cut,result,cutter)
        print(f'CAD subtraction {i}/2 complete.',flush=True)
    if not BRepCheck_Analyzer(result).IsValid(): raise RuntimeError('Trimmed CAD failed validity check; no STEP exported.')
    solids_after=len(list(entities(result,TopAbs_SOLID)))
    if solids_after!=solids_before: raise RuntimeError('Cut changed the number of solids; no STEP exported.')
    removed=boolean(BRepAlgoAPI_Cut,original,result)
    added=boolean(BRepAlgoAPI_Cut,result,original)
    hub_removed=boolean(BRepAlgoAPI_Common,removed,hub)
    added_volume=volume(added); hub_loss=volume(hub_removed)
    if added_volume>1e-5 or hub_loss>1e-5:
        raise RuntimeError(f'Geometry invariant failed: added={added_volume:g}, hub removed={hub_loss:g} mm3.')
    alignment=None
    if cfg['station_strategy']=='cad_seams':
        reference=np.r_[seam_positions,0.,extents[1],-extents[-1]]
        offsets=[];checked=[]
        for points in cad_edges(result):
            z=(points-origin)@span
            radial=points-origin-np.outer((points-origin)@axis,axis)
            length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            if np.ptp(z)<1e-6 and 1e-5<length<max(1.,2*cfg['setback_mm']) and np.linalg.norm(radial,axis=1).min()>hub_radius+1e-5:
                value=float(z.mean());offset=float(np.min(abs(reference-value)))
                offsets.append(offset);checked.append({'span_mm':value,'nearest_source_seam_distance_mm':offset})
        alignment={'source_seam_stations_mm':seam_positions,'checked_short_cross_edges':len(offsets),'maximum_station_offset_mm':max(offsets,default=0.),'edges':checked}
        if alignment['maximum_station_offset_mm']>1e-4:
            raise RuntimeError(f'Unmatched trailing-edge cross-edge: {alignment["maximum_station_offset_mm"]:g} mm from a CAD station.')

    # Validate at station midpoints, independent of cutter construction sections.
    measured=[];overlay_data={};tol=cfg['validation_tolerance_mm']
    for sign in [1,-1]:
        validation_r=sorted(set(cfg['overlay_radius_fractions']+[.40,.60,.80,.90,.975,.99,.995,.999,.9999]))
        for frac in validation_r:
            z=sign*extents[sign]*frac
            before=section(original,z);after=section(result,z);c,e,n=cut_frame(before,sign)
            old=np.concatenate(before);new=np.concatenate(after)
            # Straight closure is the aft-most output boundary in the local chord direction.
            setback=float(np.min((new-c)@e)+cfg['setback_mm'])
            on_cut=new[np.abs((new-c)@e)<tol]
            thickness=float(np.ptp(on_cut@n)) if len(on_cut)>1 else None
            def nearest(points,curves):
                starts=np.concatenate([q[:-1] for q in curves]);ends=np.concatenate([q[1:] for q in curves]);v=ends-starts;den=np.maximum(np.sum(v*v,axis=1),1e-30);ds=[]
                for chunk in np.array_split(points,max(1,math.ceil(len(points)/128))):
                    if not len(chunk):continue
                    d=chunk[:,None,:]-starts[None,:,:];t=np.clip(np.sum(d*v,axis=2)/den,0,1)
                    ds.extend(np.sqrt(np.min(np.sum((d-t[:,:,None]*v)**2,axis=2),axis=1)))
                return max(ds,default=0.)
            old_retained=old[(old-c)@e>tol]
            new_retained=new[(new-c)@e>tol]
            retained_gap=max(nearest(old_retained,after),nearest(new_retained,before))
            if retained_gap>tol:raise RuntimeError(f'Retained contour changed at {z:g} mm: {retained_gap:g} mm.')
            row={'blade_sign':sign,'radius_fraction':frac,'setback_measured_mm':setback,'closure_thickness_mm':thickness,'retained_contour_max_gap_mm':float(retained_gap)}
            measured.append(row)
            if abs(setback-cfg['setback_mm'])>tol:
                raise RuntimeError(f'Setback validation failed at {z:g} mm: {setback:g} mm.')
            # Every aft cut endpoint should lie on a straight line; retained curves
            # are checked against original section polylines, excluding the new cap.
            def planar(curves):return [np.c_[(q-origin)@chord,(q-origin)@axis].tolist() for q in curves]
            overlay_data[f'{sign}:{frac}']={'before':planar(before),'after':planar(after)}
    writer=STEPControl_Writer();writer.Transfer(result,STEPControl_AsIs)
    if writer.Write(str(destination))!=IFSelect_RetDone:raise RuntimeError('STEP export failed.')
    reread=STEPControl_Reader();reread.ReadFile(str(destination));reread.TransferRoots()
    if not BRepCheck_Analyzer(reread.OneShape()).IsValid(): raise RuntimeError('Exported STEP failed read-back validation.')
    report={'status':'passed','tool_schema_version':2,'settings':cfg,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ocp_version':ocp_version,'output_step':str(destination),
            'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'hub_radius_mm':hub_radius,
            'original_valid':True,'result_valid':True,'export_readback_valid':True,'solids_before':solids_before,'solids_after':solids_after,
            'removed_volume_mm3':volume(removed),'added_volume_mm3':added_volume,'hub_removed_volume_mm3':hub_loss,
            'measurements':measured,'seam_alignment':alignment,'elapsed_seconds':time.time()-started,
            'definition':'Aft x-extreme of each section, toward opposite x-extreme along local section chord; two opposite blades. Piecewise ruled cutting solid; no fillet, smoothing, or reconstruction of retained surfaces.',
            'limitations':['Not a Fluent or OpenFOAM mesh quality test.','STEP named selections/assembly labels are not preserved.','Frame and trailing-edge sense must be checked for each new geometry.']}
    (out/'settings.resolved.json').write_text(json.dumps(cfg,indent=2))
    (out/'cutting-frames.json').write_text(json.dumps(frames,indent=2))
    (out/'section-overlays.json').write_text(json.dumps(overlay_data))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    plot_overlays(overlay_data,cfg,out)
    print(json.dumps({'status':'passed','output_step':str(destination),'hub_removed_volume_mm3':hub_loss,'removed_volume_mm3':volume(removed)},indent=2),flush=True)
    return report


def plot_overlays(data,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fractions=cfg['overlay_radius_fractions']
    for sign in [1,-1]:
        cols=3;fig,axs=plt.subplots(math.ceil(len(fractions)/cols),cols,figsize=(15,8),squeeze=False,layout='constrained')
        for ax,frac in zip(axs.flat,fractions):
            it=data[f'{sign}:{frac}']
            for key,color,ls,label in [('before','#252525','-','Original STEP'),('after','#dc6418','--',f'Blunt cut {cfg["setback_mm"]:.2f} mm')]:
                for i,q in enumerate(it[key]):
                    q=np.array(q);ax.plot(q[:,0],q[:,1],color=color,ls=ls,lw=1.5,label=label if i==0 else None)
            ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.set_title(f'{frac:.0%} radius');ax.set_xlabel('Chord coordinate [mm]');ax.set_ylabel('Axial coordinate [mm]')
        for ax in list(axs.flat)[len(fractions):]:ax.set_visible(False)
        axs.flat[0].legend(fontsize=8);fig.suptitle(f'Blade {sign:+d}: original coordinates; no rotation, scaling, or fitted alignment')
        fig.savefig(out/f'blade_{sign:+d}_overlays.png',dpi=180);plt.close(fig)
        zooms=[r for r in fractions if r>=.7][-3:]
        fig,axs=plt.subplots(1,len(zooms),figsize=(5*len(zooms),4.5),squeeze=False,layout='constrained')
        for ax,frac in zip(axs.flat,zooms):
            it=data[f'{sign}:{frac}'];q=np.concatenate([np.array(q) for q in it['before']]);xs=q[:,0]
            te_min=(cfg['frame']['positive_blade_trailing_edge']=='min_chord_coordinate')==(sign==1)
            a,z=(xs.min(),xs.max()) if te_min else (xs.max(),xs.min())
            te=q[abs(xs-a)<1e-6].mean(0);le=q[abs(xs-z)<1e-6].mean(0);e=(le-te)/np.linalg.norm(le-te);n=np.array([-e[1],e[0]])
            for key,color,ls,label in [('before','#252525','-','Original STEP'),('after','#dc6418','--','Blunt cut')]:
                for i,curve in enumerate(it[key]):
                    curve=np.array(curve)-te;ax.plot(curve@e,curve@n,color=color,ls=ls,lw=1.8,label=label if i==0 else None)
            ax.axvline(cfg['setback_mm'],color='#888888',lw=.8,ls=':');ax.set_xlim(-.03,3*cfg['setback_mm']);ax.set_ylim(-cfg['setback_mm'],cfg['setback_mm']);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(f'{frac:.0%} radius');ax.set_xlabel('Distance forward from original TE [mm]');ax.set_ylabel('Normal to local chord [mm]')
        axs.flat[0].legend(fontsize=9);fig.suptitle(f'Blade {sign:+d}: trailing-edge detail; local chord frame')
        fig.savefig(out/f'blade_{sign:+d}_cut_detail.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings',type=Path,help='JSON settings path; file paths inside it are relative to that JSON.')
    args=parser.parse_args()
    try:run(args.settings.resolve())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())

"""Export Fluent CFF connectivity to legacy ASCII for Foundation OpenFOAM 13.

This is a format conversion, not a mesher. Node coordinates, polygon vertex
order, face-to-cell addressing and cell/face zone ranges are preserved.
Periodic, refinement-tree and nonconformal meshes are rejected explicitly.
"""
from pathlib import Path
import re

import numpy as np


ZONE_TYPES = {2: 'interior', 3: 'wall', 4: 'pressure-inlet',
              5: 'pressure-outlet', 7: 'symmetry', 9: 'pressure-far-field',
              10: 'velocity-inlet', 12: 'periodic', 14: 'fan',
              15: 'interior', 20: 'mass-flow-inlet', 24: 'interface',
              36: 'outflow'}


def _name(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsupported Fluent zone name: {value!r}')
    return value


def _topology(group):
    fields = ('id', 'minId', 'maxId')
    arrays = {k: np.asarray(group[k][()]).reshape(-1) for k in fields}
    names = group['name'][0].decode('utf-8').split(';')
    count = len(arrays['id'])
    if len(names) != count or any(len(a) != count for a in arrays.values()):
        raise ValueError('Inconsistent Fluent zone topology')
    result = []
    for i, name in enumerate(names):
        item = {k: int(a[i]) for k, a in arrays.items()}
        item['name'] = _name(name)
        for optional in ('zoneType', 'cellType', 'dimension', 'childZoneId', 'shadowZoneId'):
            if optional in group:
                item[optional] = int(group[optional][i])
        if item['minId'] < 1 or item['maxId'] < item['minId']:
            raise ValueError('Invalid Fluent zone range')
        result.append(item)
    return result


def export_mesh(source, destination):
    """Return source counts and layer flags; raise on unsupported connectivity."""
    import h5py

    source, destination = Path(source), Path(destination)
    with h5py.File(source, 'r') as file:
        if 'meshes' not in file or list(file['meshes']) != ['1']:
            raise ValueError('Expected one Fluent mesh in meshes/1')
        mesh = file['meshes/1']
        nodes = _topology(mesh['nodes/zoneTopology'])
        cells = _topology(mesh['cells/zoneTopology'])
        faces = _topology(mesh['faces/zoneTopology'])
        if any(z.get('dimension') != 3 for z in nodes + cells + faces):
            raise ValueError('Only three-dimensional Fluent meshes are supported')
        for topology in (nodes, cells, faces):
            ordered = sorted(topology, key=lambda z: z['minId'])
            if ordered[0]['minId'] != 1 or any(
                    a['maxId'] + 1 != b['minId'] for a, b in zip(ordered, ordered[1:])):
                raise ValueError('Fluent IDs must have contiguous zone ranges')
        npoints = max(z['maxId'] for z in nodes)
        ncells = max(z['maxId'] for z in cells)
        nfaces = max(z['maxId'] for z in faces)
        for zone in faces:
            if (zone.get('childZoneId', 0) or zone.get('shadowZoneId', 0)
                    or zone['zoneType'] not in ZONE_TYPES
                    or zone['zoneType'] in (12, 24)):
                raise ValueError('Periodic, shadow, hanging-node or nonconformal faces are unsupported; export a conformal mesh')
        if any(z.get('childZoneId', 0) for z in cells):
            raise ValueError('Refinement-tree cell zones are unsupported')
        if len(cells) < 2:
            raise ValueError('Preserve separate rotor and stationary fluid cell zones in Fluent')

        def blocks(path):
            group = mesh[path]
            return sorted(group.items(), key=lambda p: int(np.asarray(p[1].attrs['minId']).item()))

        with destination.open('w', encoding='ascii', newline='\n') as out:
            out.write('(0 "Fluent HDF5 connectivity export; units unchanged")\n(2 3)\n')
            out.write(f'(10 (0 1 {npoints:x} 0 3))\n')
            point_lookup = {z['minId']: z for z in nodes}
            for _, data in blocks('nodes/coords'):
                lo = int(np.asarray(data.attrs['minId']).item())
                hi = int(np.asarray(data.attrs['maxId']).item())
                zone = point_lookup[lo]
                xyz = data[()]
                if hi != zone['maxId'] or xyz.shape != (hi-lo+1, 3) or not np.isfinite(xyz).all():
                    raise ValueError('Malformed or nonfinite node coordinates')
                out.write(f'(10 ({zone["id"]:x} {lo:x} {hi:x} {zone["zoneType"]:x} 3)(\n')
                np.savetxt(out, xyz, fmt='%.17g')
                out.write('))\n')
            # fluent3DMeshToFoam constructs polyhedra from face connectivity;
            # retain mixed element classifications for other consumers as well.
            out.write(f'(12 (0 1 {ncells:x} 0))\n')
            cell_groups = mesh['cells/ctype']
            for i, zone in enumerate(cells, 1):
                cell_type = zone.get('cellType', 0)
                if cell_type:
                    # Uniform zones store their type in zoneTopology and may
                    # omit the per-cell dataset (polyhedral zones use type 7).
                    if cell_type not in range(1, 8):
                        raise ValueError(f'Unsupported Fluent cell type: {cell_type}')
                    out.write(f'(12 ({zone["id"]:x} {zone["minId"]:x} {zone["maxId"]:x} 1 {cell_type:x}))\n')
                    out.write(f'(45 ({zone["id"]} fluid {zone["name"]})())\n')
                    continue
                if f'{i}/cell-types' not in cell_groups:
                    raise ValueError('Mixed Fluent cell zone is missing cell-types')
                types = cell_groups[f'{i}/cell-types'][()]
                if len(types) != zone['maxId']-zone['minId']+1:
                    raise ValueError('Cell classification count differs from cell zone')
                out.write(f'(12 ({zone["id"]:x} {zone["minId"]:x} {zone["maxId"]:x} 1 0)(\n')
                for start in range(0, len(types), 20000):
                    out.write(' '.join(format(int(v), 'x') for v in types[start:start+20000])+'\n')
                out.write('))\n')
                out.write(f'(45 ({zone["id"]} fluid {zone["name"]})())\n')
            out.write(f'(13 (0 1 {nfaces:x} 0))\n')
            face_lookup = {z['minId']: z for z in faces}
            for key, data in blocks('faces/nodes'):
                lo = int(np.asarray(data.attrs['minId']).item())
                hi = int(np.asarray(data.attrs['maxId']).item())
                zone = face_lookup[lo]
                sizes = data['nnodes'][()]
                vertices = data['nodes'][()]
                c0 = mesh[f'faces/c0/{key}'][()]
                c1 = mesh[f'faces/c1/{key}'][()]
                if (hi != zone['maxId'] or len(sizes) != hi-lo+1 or len(c0) != len(sizes)
                        or len(c1) != len(sizes) or np.any(sizes < 3)
                        or int(sizes.sum()) != len(vertices) or np.any(vertices < 1)
                        or np.any(vertices > npoints) or np.any(c0 > ncells)
                        or np.any(c1 > ncells) or np.any((c0 == 0) & (c1 == 0))):
                    raise ValueError('Malformed face-to-node or face-to-cell connectivity')
                typ = ZONE_TYPES[zone['zoneType']]
                if typ == 'interior' and (np.any(c0 == 0) or np.any(c1 == 0)):
                    raise ValueError('Internal Fluent zone has unpaired boundary faces')
                out.write(f'(13 ({zone["id"]:x} {lo:x} {hi:x} {zone["zoneType"]:x} 0)(\n')
                offset, lines = 0, []
                for size, a, b in zip(sizes, c0, c1):
                    size = int(size)
                    values = [size, *vertices[offset:offset+size], a, b]
                    lines.append(' '.join(format(int(v), 'x') for v in values)+'\n')
                    offset += size
                    if len(lines) >= 20000:
                        out.writelines(lines)
                        lines.clear()
                out.writelines(lines)
                out.write('))\n')
                out.write(f'(45 ({zone["id"]} {typ} {zone["name"]})())\n')
        flags = file.get('special/boundary_layer_flags/cells')
        layered = sum(int(np.count_nonzero(v['data'][()])) for v in flags.values()
                      if isinstance(v, h5py.Group) and 'data' in v) if flags else 0
    return dict(points=npoints, faces=nfaces, cells=ncells,
                boundary_layer_flagged_cells=layered, layers=bool(layered))

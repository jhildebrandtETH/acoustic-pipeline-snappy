"""Convert named OBJ curves into snappy's native feature-distance refinements."""
import hashlib
import json
import math
from pathlib import Path
import re
import warnings


def read_obj_lines(raw, scale, label='OBJ'):
    """Read explicit polylines; face triangulation never creates feature edges."""
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Feature scale must be finite and positive')
    points, edges = [], []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        fields = line.partition('#')[0].split()
        if not fields:
            continue
        try:
            if fields[0] == 'v':
                if len(fields) != 4:
                    raise ValueError('expected a three-coordinate vertex')
                point = tuple(float(v) * scale for v in fields[1:])
                if not all(math.isfinite(v) for v in point):
                    raise ValueError('coordinates must be finite')
                points.append(point)
            elif fields[0] == 'l':
                if len(fields) < 3:
                    raise ValueError('line needs at least two vertex indices')
                indices = []
                for token in fields[1:]:
                    index = int(token.split('/', 1)[0])
                    if index == 0:
                        raise ValueError('OBJ indices cannot be zero')
                    resolved = index - 1 if index > 0 else len(points) + index
                    if resolved < 0:
                        raise ValueError('vertex index out of range')
                    indices.append(resolved)
                edges.extend(zip(indices, indices[1:]))
        except ValueError as exc:
            raise ValueError(f'{label}:{number}: {exc}') from exc
    if not edges:
        raise ValueError(f"{label}: no explicit OBJ line elements ('l'); export curves as lines")
    unique = []
    seen = set()
    for a, b in edges:
        if max(a, b) >= len(points):
            raise ValueError(f'{label}: vertex index out of range')
        if points[a] == points[b]:
            raise ValueError(f'{label}: zero-length feature segment')
        key = tuple(sorted((a, b)))
        if key not in seen:
            unique.append((a, b))
            seen.add(key)
    # Unused OBJ vertices must not influence feature snapping.
    used = sorted({v for edge in unique for v in edge})
    remap = {v: i for i, v in enumerate(used)}
    return [points[v] for v in used], [(remap[a], remap[b]) for a, b in unique]


def edge_mesh_text(points, edges, name):
    text = ('FoamFile\n{\n    version 2.0;\n    format ascii;\n'
            '    class featureEdgeMesh;\n    object ' + name + ';\n}\n')
    text += str(len(points)) + '\n(\n'
    text += ''.join('(' + ' '.join(format(v, '.17g') for v in p) + ')\n' for p in points)
    text += ')\n' + str(len(edges)) + '\n(\n'
    text += ''.join(f'({a} {b})\n' for a, b in edges)
    return text + ')\n'


def load_settings(path):
    if not path.is_file():
        return dict(enabled=False, groups=[])
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or type(data.get('enabled')) is not bool:
        raise ValueError('snappyFeatures.json requires a boolean enabled entry')
    groups = data.get('groups')
    if (not isinstance(groups, list) or any(not isinstance(g, str) or
            not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', g) for g in groups)
            or len(groups) != len(set(groups))):
        raise ValueError('Feature groups must be unique names containing letters, digits or underscores')
    if data['enabled']:
        for group in groups:
            level = data.get(group + 'Level')
            distance = data.get(group + 'RefinementThickness')
            if type(level) is not int or level < 0:
                raise ValueError(f'{group}Level must be a nonnegative integer')
            if (type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0):
                raise ValueError(f'{group}RefinementThickness must be finite and nonnegative')
    return data


def prepare_features(case, source, scale):
    """Snapshot FEATURES/<STL stem>_<group>.obj and write native snappy entries."""
    from tools.snappy_pipeline import write
    case, source = Path(case), Path(source)
    settings = load_settings(case/'Parameters/snappyFeatures.json')
    entries, report = [], dict(enabled=settings['enabled'], groups={})
    if settings['enabled']:
        for group in settings['groups']:
            feature = source.parent.parent/'FEATURES'/f'{source.stem}_{group}.obj'
            level = settings[group + 'Level']
            distance = settings[group + 'RefinementThickness']
            info = dict(source=str(feature), level=level, refinement_thickness_m=distance)
            report['groups'][group] = info
            if not feature.is_file():
                info['status'] = 'missing'
                warnings.warn(f'Skipping missing feature curve: {feature}', stacklevel=2)
                continue
            raw = feature.read_bytes()
            points, edges = read_obj_lines(raw, scale, str(feature))
            name = f'custom_{group}.eMesh'
            relative = 'constant/triSurface/' + name
            write(case/relative, edge_mesh_text(points, edges, name))
            snapshot = case/'FEATURES'/feature.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(raw)
            refinement = f'levels (({distance:.17g} {level}));' if distance else f'level {level};'
            entries.append(f'{{ file "{name}"; {refinement} }}\n')
            info.update(status='active', vertices=len(points), edges=len(edges),
                        edge_file=relative, source_sha256=hashlib.sha256(raw).hexdigest(),
                        snapshot=str(snapshot.relative_to(case)), scale=scale)
    write(case/'Parameters/snappyFeatures.generated',
          '// Generated from FEATURES OBJ curves and snappyFeatures.json.\n' + ''.join(entries))
    write(case/'mesh/features.json', json.dumps(report, indent=2) + '\n')
    return report

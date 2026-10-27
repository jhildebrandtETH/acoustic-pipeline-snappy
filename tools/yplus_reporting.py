"""Area-primary y-plus reporting from exact saved patch fields and snapshot mesh."""
import json
from pathlib import Path
import re
import gzip
import numpy as np
from .plotting import pyplot as plt

def existing(path):
    return path if path.is_file() else path.with_name(path.name + '.gz')


def read(path):
    path = existing(path)
    raw = gzip.open(path, 'rb').read() if path.suffix == '.gz' else path.read_bytes()
    if re.search(rb'\bformat\s+binary\s*;', raw[:8192]):
        raise ValueError(f'ASCII input required; binary file: {path}')
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', raw.decode(), flags=re.S)


def block(text, name):
    match = re.search(r'(?<![\w.:-])(?:"' + re.escape(name) + r'"|' + re.escape(name) + r')(?![\w.:-])\s*\{', text)
    if not match:
        raise ValueError(f'Missing exact dictionary name: {name}')
    start = match.end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return text[start:i]
    raise ValueError(f'Unclosed dictionary: {name}')


def scalar_value(text, key, count):
    m = re.search(r'\b' + key + r'\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;', text, re.S)
    if m:
        values = np.fromstring(m[2], sep=' ')
        if len(values) != int(m[1]) or len(values) != count:
            raise ValueError('Declared scalar count differs from mesh/parsed count')
    else:
        m = re.search(r'\b' + key + r'\s+uniform\s+([-+\d.eE]+)\s*;', text)
        if not m:
            raise ValueError(f'Unsupported scalar entry: {key}')
        values = np.full(count, float(m[1]))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid or negative yPlus values')
    return values


def areas(mesh, start, count, mesh_files=None):
    mesh_files = mesh_files or {}
    text = read(mesh_files.get('points', mesh / 'points'))
    m = re.search(r'(?m)^\s*(\d+)\s*\(', text)
    if not m:
        raise ValueError('Unrecognized points list')
    points = np.fromstring(text[m.end():text.rfind(')')].replace('(', ' ').replace(')', ' '), sep=' ')
    if points.size != int(m[1]) * 3:
        raise ValueError('Point count mismatch')
    points = points.reshape(-1, 3)
    del text
    path = existing(mesh_files.get('faces', mesh / 'faces'))
    opener = gzip.open if path.suffix == '.gz' else open
    result = []
    with opener(path, 'rt') as stream:
        pending = stream.read(1024 * 1024)
        if re.search(r'\bformat\s+binary\s*;', pending[:8192]):
            raise ValueError('ASCII faces required')
        header = re.search(r'(?m)^\s*(\d+)\s*\(', pending)
        if not header:
            raise ValueError('Unrecognized ASCII face-list header')
        if start < 0 or start + count > int(header[1]):
            raise ValueError('Patch face range exceeds face-list length')
        pending = pending[header.end():]
        record = re.compile(r'\s*(\d+)\s*\(([^()]*)\)')
        face_i = 0
        while face_i < start + count:
            pos = 0
            while face_i < start + count:
                m = record.match(pending, pos)
                if not m:
                    break
                if start <= face_i:
                    ids = np.fromstring(m[2], sep=' ', dtype=np.int64)
                    if len(ids) != int(m[1]) or len(ids) < 3:
                        raise ValueError('Face vertex count mismatch')
                    if (ids < 0).any() or (ids >= len(points)).any():
                        raise ValueError('Face references an invalid point')
                    p = points[ids]
                    q = p - p[0]
                    result.append(float(np.linalg.norm(np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)) / 2))
                face_i += 1
                pos = m.end()
            if face_i == start + count:
                break
            pending = pending[pos:]
            more = stream.read(1024 * 1024)
            if not more or len(pending) > 1024 * 1024:
                raise ValueError(f'Unsupported or incomplete ASCII face record at face {face_i}')
            pending += more
    a = np.asarray(result)
    if len(a) != count or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Invalid or missing patch face areas')
    return a


def time_directories(case):
    result = []
    for p in Path(case).iterdir():
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        if np.isfinite(t) and t >= 0:
            result.append((t, p))
    return sorted(result, key=lambda item: item[0])


def class_masks(v):
    return [v < 5, (v >= 5) & (v <= 30), v > 30]


def histogram_masks(v):
    # Keep y+=30 in the buffer class in both chart and summary.
    return [v < 5, (v >= 5) & (v <= 30), (v > 30) & (v <= 50),
            (v > 50) & (v <= 100), (v > 100) & (v <= 200), v > 200]


def weighted_statistics(v, weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != v.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Positive finite weights are required for every face')
    total = weights.sum()
    order = np.argsort(v, kind='stable')
    median = v[order[np.searchsorted(np.cumsum(weights[order]), total / 2)]]
    return dict(mean=float(np.average(v, weights=weights)), median=float(median),
                class_percent=[float(100 * weights[m].sum() / total) for m in class_masks(v)],
                histogram_percent=[float(100 * weights[m].sum() / total) for m in histogram_masks(v)])


def create_yplus_report(case_path, report_dir, patch_name):
    case, report = Path(case_path), Path(report_dir)
    report.mkdir(parents=True, exist_ok=True)
    image = report / 'yplus_distribution.png'
    image.unlink(missing_ok=True)
    result = dict(status='unavailable', patch_name=patch_name, primary_weighting='surface_area',
                  average_yplus=None, median_yplus=None,
                  scope='One latest reconstructed snapshot; entire named patch, including hub if merged.',
                  class_intervals=['y+ < 5', '5 <= y+ <= 30', 'y+ > 30'])
    try:
        times = time_directories(case)
        if not times:
            raise ValueError('No reconstructed time directory')
        selected, directory = times[-1]
        result['time_dir'] = directory.name
        # Do not quietly use an older reconstructed field when newer data exist.
        for processor in case.glob('processor*'):
            if processor.is_dir():
                ptimes = time_directories(processor)
                if ptimes and ptimes[-1][0] > selected:
                    raise ValueError('Processor results are newer; reconstruct the latest saved yPlus first')
        boundary = block(read(directory/'yPlus'), 'boundaryField')
        value_block = block(boundary, patch_name)
        mesh_files = {}
        for name in ('boundary', 'points', 'faces'):
            candidates = [case/'constant/polyMesh'/name] + [p/'polyMesh'/name for t,p in times if t <= selected]
            found = next((p for p in reversed(candidates) if existing(p).is_file()), None)
            if found is not None:
                mesh_files[name] = found
        patch_count = None
        mesh_reason = None
        try:
            patch = block(read(mesh_files.get('boundary', case/'constant/polyMesh/boundary')), patch_name)
            patch_count = int(re.search(r'\bnFaces\s+(\d+)', patch)[1])
            start = int(re.search(r'\bstartFace\s+(\d+)', patch)[1])
        except (OSError, ValueError, TypeError) as exc:
            mesh_reason = str(exc)
        declared = re.search(r'\bvalue\s+nonuniform\s+List<scalar>\s+(\d+)', value_block)
        count = patch_count if patch_count is not None else int(declared[1]) if declared else None
        if not count:
            raise ValueError('Missing or empty patch face count; uniform fields require mesh boundary data')
        values = scalar_value(value_block, 'value', count)
        face = weighted_statistics(values, np.ones(count))
        face['median'] = float(np.median(values))
        result.update(status='face_only', n_faces=count, min_yplus=float(values.min()),
                      max_yplus=float(values.max()), face_weighted=face,
                      face_counts=[int(m.sum()) for m in histogram_masks(values)])
        try:
            if mesh_reason:
                raise ValueError(mesh_reason)
            a = areas(case/'constant/polyMesh', start, count, mesh_files)
            area = weighted_statistics(values, a)
            result.update(status='complete', area_weighted=area, patch_area_m2=float(a.sum()),
                          mesh_sources={k: str(existing(v)) for k,v in mesh_files.items()},
                          average_yplus=area['mean'], median_yplus=area['median'],
                          share_yplus_lt_5_percent=area['class_percent'][0],
                          share_yplus_5_to_30_percent=area['class_percent'][1],
                          share_yplus_gt_30_percent=area['class_percent'][2])
        except (OSError, ValueError, UnicodeError, IndexError) as exc:
            result['area_unavailable_reason'] = str(exc)
        labels = ['<5', '5-30', '>30-50', '>50-100', '>100-200', '>200']
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        try:
            if 'area_weighted' in result:
                ax.bar(x-.19, result['area_weighted']['histogram_percent'], .38,
                       color='#357d91', label='Surface area (primary)')
                ax.bar(x+.19, face['histogram_percent'], .38, color='#babfc4', label='Face count (diagnostic)')
                ax.set_ylabel('Share [%]')
            else:
                ax.bar(x, face['histogram_percent'], .65, color='#babfc4', label='Face count only; areas unavailable')
                ax.set_ylabel('Face share [%]')
            ax.set_xticks(x, labels)
            ax.set_xlabel('y+ interval (upper endpoints included)')
            ax.set_title('Wall y+ coverage - ' + patch_name)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=.25)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(image, dpi=200)
        finally:
            plt.close(fig)
    except (OSError, ValueError, UnicodeError, IndexError) as exc:
        result['reason'] = str(exc)
    (report/'yplus_distribution.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return (image if image.is_file() else None), result


def append_yplus_report(c, image, stats):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape
    if stats is None:
        return
    c.showPage()
    c.setPageSize(A4)
    _, height = A4
    c.setFont('Helvetica-Bold', 16)
    c.drawString(50, height-50, 'Wall Treatment Evaluation')
    style = ParagraphStyle('yPlus', fontName='Helvetica', fontSize=10, leading=14)
    y = height-73
    def line(text):
        nonlocal y
        p = Paragraph(text, style)
        _, h = p.wrap(495, 700)
        p.drawOn(c, 50, y-h)
        y -= h+7
    line('Patch: '+escape(stats['patch_name'])+' | Saved time: '+escape(stats.get('time_dir','unavailable'))+
         ' | Faces: '+str(stats.get('n_faces','unavailable')))
    line(escape(stats['scope']))
    area = stats.get('area_weighted')
    if area:
        line(f"<b>Area-weighted mean y+: {area['mean']:.2f} | Median: {area['median']:.2f}</b>")
        a,b,d = area['class_percent']
        line(f'<b>Surface area:</b> y+ &lt; 5: {a:.1f}% | 5 &lt;= y+ &lt;= 30: {b:.1f}% | y+ &gt; 30: {d:.1f}%')
    else:
        line('<b>Area-weighted coverage unavailable.</b> '+escape(stats.get('area_unavailable_reason',stats.get('reason',''))))
    face = stats.get('face_weighted')
    if face:
        line(f"Face-weighted mean y+: {face['mean']:.2f} | Median: {face['median']:.2f} | "
             f"Min/max: {stats['min_yplus']:.2f} / {stats['max_yplus']:.2f}")
        a,b,d = face['class_percent']
        line(f'<b>Face count (diagnostic):</b> y+ &lt; 5: {a:.1f}% | 5 &lt;= y+ &lt;= 30: {b:.1f}% | y+ &gt; 30: {d:.1f}%')
    if image is not None:
        c.drawImage(str(image),50,y-280,width=495,height=270,preserveAspectRatio=True,mask='auto')
        y-=290
    line('Data: report/yplus_distribution.json. Snapshot coverage does not establish developed-flow or revolution-averaged wall treatment.')

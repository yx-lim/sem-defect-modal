"""Flagged-crop queue + Anthropic crop verifier -> sem84 review_*.json.

queue: pick ~15-25 flagged objects from a stage annotation, render one raw+overlay panel per crop.
verify: send each crop to Claude in parallel with a fixed rubric, parse JSON verdicts.
convert: map verdicts conservatively to the review format consumed by `sem84.cli run --review`.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import pathlib
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from .cli import DARK_CLASSES, PARTICLE_CLASSES, load_gray, now
from .render import font

PANEL = 384
MIN_WIN, MAX_WIN = 192, 768
MAX_CROPS = 25
MAX_LABELS = 24
PRICES = {  # USD per million tokens (input, output); list prices, update if they change
    'claude-sonnet-4-6': (3.0, 15.0), 'claude-haiku-4-5-20251001': (1.0, 5.0), 'claude-opus-4-6': (5.0, 25.0),
}
CLS_COL = {'particle_contrast_A': (255, 200, 0), 'particle_contrast_B': (0, 220, 255), 'unknown_inclusion': (255, 0, 255),
           'particle_unclassified_contrast': (180, 180, 180), 'void_like_region': (255, 60, 60), 'fissure_candidate': (0, 255, 0),
           'interfacial_gap': (255, 140, 0), 'unresolved_dark': (150, 100, 255)}
VERDICTS = {'keep', 'remove', 'relabel', 'reclassify_dark', 'artifact_region', 'ignore', 'unsure'}


# ----------------------------------------------------------------------------- queue

def _items(ann, prev_queue=None, diff=None):
    inst, dark = ann['instances'], ann['dark_regions']
    med = float(np.median([i['area_px'] for i in inst])) if inst else 2000.0
    it = []

    def add(lst, kind, why, n):
        for o in lst[:n]:
            it.append({'kind': kind, 'id': o['id'], 'bbox': o['bbox_xyxy'], 'why': why})

    if diff is not None:  # re-verification: only objects that changed between passes
        ci = {i['id']: i for i in inst}
        cd = {d['id']: d for d in dark}
        add([ci[x['id']] for x in diff.get('instances_added', []) if x['id'] in ci], 'new_instance', 'added in this pass (re-proposal or reviewer seed)', 12)
        add([ci[x['id']] for x in diff.get('instance_class_changes', []) if x['id'] in ci], 'class_change', 'class changed between passes', 4)
        newd = [cd[k] for k in diff.get('dark_added', []) if k in cd and cd[k]['class'] != 'void_like_region']
        add(sorted(newd, key=lambda d: -d['area_px']), 'new_dark', 'dark region added in this pass', 6)
        return it
    by = lambda c: [d for d in dark if d['class'] == c]
    add(sorted(by('fissure_candidate'), key=lambda d: -d['skeleton_length_px']), 'fissure', 'fissure candidate: thin slit enclosed by one particle?', 6)
    add(sorted(by('interfacial_gap'), key=lambda d: -d['skeleton_length_px']), 'gap', 'interfacial gap candidate', 3)
    add([d for d in dark if 'interparticle_crack_candidate' in d['tags']], 'crack', 'interparticle crack candidate', 2)
    add([d for d in dark if 'particle_shaped_cavity_candidate' in d['tags']], 'cavity', 'particle-shaped cavity candidate', 2)
    add(sorted(by('unresolved_dark'), key=lambda d: -d['area_px']), 'unresolved', 'largest unresolved dark region', 2)
    add(sorted(by('void_like_region'), key=lambda d: -d['area_px']), 'void', 'largest void-like region (dark != pore)', 2)
    add(sorted([i for i in inst if i['solidity'] < 0.8], key=lambda i: i['solidity']), 'low_solidity', 'low-solidity mask (merged/leaky?)', 4)
    add(sorted([i for i in inst if i['area_px'] > 6 * med], key=lambda i: -i['area_px']), 'oversized', 'oversized mask (merged?)', 2)
    add(sorted([i for i in inst if i.get('touch_count', 0) >= 4], key=lambda i: -i['touch_count']), 'many_neighbours', 'touches many neighbours (merge/leak?)', 2)
    add(sorted(inst, key=lambda i: i['heuristic_confidence']), 'lowconf', 'lowest heuristic confidence instance', 3)
    add([i for i in inst if i['class'] == 'unknown_inclusion'], 'inclusion', 'unknown inclusion candidate', 2)
    for b in ann['qc']['directional_streaks'].get('aligned_tiles', [])[:2]:
        it.append({'kind': 'streak', 'id': f't{b[0]}_{b[1]}', 'bbox': list(b[:4]), 'why': 'aligned streak tile: curtaining / scan artefact?'})
    for r in (ann['qc'].get('scan_line_rows') or [])[:1]:
        it.append({'kind': 'scanline', 'id': f'row{r}', 'bbox': [0, max(0, r - 20), 700, r + 20], 'why': 'scan-line jump row'})
    return it


def _window(b, W, H):
    x0, y0, x1, y1 = b
    s = int(min(MAX_WIN, max(MIN_WIN, 1.6 * max(x1 - x0, y1 - y0))))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    wx0 = max(0, min(W - s, cx - s // 2))
    wy0 = max(0, min(H - s, cy - s // 2))
    return [int(wx0), int(wy0), int(wx0 + min(s, W)), int(wy0 + min(s, H))]


def _inside(b, w, frac=0.6):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    mx, my = (1 - frac) / 2 * (w[2] - w[0]), (1 - frac) / 2 * (w[3] - w[1])
    return w[0] + mx <= cx <= w[2] - mx and w[1] + my <= cy <= w[3] - my


def _isect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def build_queue(stage_dir, src, out_dir, diff=None):
    stage_dir, out_dir = pathlib.Path(stage_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = json.loads((stage_dir / 'annotation.json').read_text())
    g, _ = load_gray(src)
    H, W = g.shape
    objs = {o['id']: ('inst', o) for o in ann['instances']}
    objs.update({d['id']: ('dark', d) for d in ann['dark_regions']})
    crops = []
    for it in _items(ann, diff=diff):
        host = next((c for c in crops if _inside(it['bbox'], c['window_xyxy'])), None)
        if host:
            host['flagged'].append(it)
            continue
        if len(crops) >= MAX_CROPS:
            continue
        crops.append({'window_xyxy': _window(it['bbox'], W, H), 'flagged': [it]})
    big_dark = lambda d: d['area_px'] >= 150 or d['class'] != 'void_like_region'
    for n, c in enumerate(crops, 1):
        c['crop_id'] = f'c{n:02d}'
        w = c['window_xyxy']
        flagged = [f['id'] for f in c['flagged'] if f['id'] in objs]
        near = [i['id'] for i in ann['instances'] if _isect(i['bbox_xyxy'], w) and i['id'] not in flagged]
        neard = sorted([d for d in ann['dark_regions'] if _isect(d['bbox_xyxy'], w) and d['id'] not in flagged and big_dark(d)],
                       key=lambda d: -d['area_px'])
        c['object_ids'] = (flagged + near + [d['id'] for d in neard])[:MAX_LABELS]
        c['file'] = f"{c['crop_id']}.jpg"
        _render(g, w, [objs[k] for k in c['object_ids']], out_dir / c['file'], f"{c['crop_id']} native {w}")
        c['scale'] = PANEL / (w[2] - w[0])
        c['objects'] = [{'id': k, 'type': objs[k][0], 'class': objs[k][1]['class'],
                         'area_px': objs[k][1]['area_px'], 'solidity': round(objs[k][1]['solidity'], 2),
                         'tags': objs[k][1].get('tags', [])[:4]} for k in c['object_ids']]
    q = {'stage_dir': str(stage_dir), 'image_id': ann['image_id'], 'detector': ann['detector'], 'stage': ann['stage'],
         'generated_at': now(), 'mode': 'reverify' if diff is not None else 'full', 'crops': crops}
    (out_dir / 'queue.json').write_text(json.dumps(q, indent=1))
    return q


def _render(g, w, objs, path, title):
    x0, y0, x1, y1 = w
    s = PANEL / (x1 - x0)
    raw = Image.fromarray(g[y0:y1, x0:x1]).convert('RGB').resize((PANEL, round((y1 - y0) * s)), Image.LANCZOS)
    ann = raw.copy()
    d = ImageDraw.Draw(ann)
    f = font(11)
    for kind, o in objs:
        col = CLS_COL.get(o['class'], (255, 255, 255))
        pts = [((p[0] - x0) * s, (p[1] - y0) * s) for p in o['polygon_xy']]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=col, width=2 if kind == 'inst' else 1)
        cx, cy = (o['centroid_xy'][0] - x0) * s, (o['centroid_xy'][1] - y0) * s
        if 0 <= cx < PANEL and 0 <= cy < raw.height:
            d.text((cx - 14, cy - 6), o['id'], fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    panel = Image.new('RGB', (2 * PANEL + 8, raw.height + 22), (20, 20, 24))
    panel.paste(raw, (0, 22))
    panel.paste(ann, (PANEL + 8, 22))
    ImageDraw.Draw(panel).text((4, 4), f'{title}  zoom {s:.2f}x  left=raw right=overlay', fill=(255, 255, 255), font=font(13))
    panel.save(path, quality=90)


# ----------------------------------------------------------------------------- verify

RUBRIC = """You review one crop of a FIB-SEM cross-section of a battery electrode (detector: {det}; {detnote}).
Left panel = raw image, right panel = same region with draft pipeline outlines and object IDs. Zoom {zoom:.2f}x of native pixels; panel pixel (px,py) maps to native (x0+px/zoom, y0+py/zoom) with x0,y0={origin}.
Outline colours: particle_contrast_A yellow, particle_contrast_B cyan, unknown_inclusion magenta, particle_unclassified_contrast grey, void_like_region red, fissure_candidate green, interfacial_gap orange, unresolved_dark violet.
Flagged reason(s): {why}
Objects in this crop (id, type, class, area_px, solidity, tags):
{objs}

Rules (non-negotiable):
- Draft labelling, not ground truth. Prefer removing/ignoring over guessing. If you cannot judge an object from this crop, answer "unsure".
- Never assert chemistry (Si, SiOx, graphite, binder, carbon black, copper, contamination, oxidation). Dark != pore. Touching != electrical contact. 2D cannot prove 3D connectivity/enclosure.
- Particle masks: remove masks that merge several particles, leak over matrix/background, cover textured fine matrix rather than a discrete particle, or are not a particle. Relabel classes: particle_contrast_A, particle_contrast_B (BSE only), unknown_inclusion, particle_unclassified_contrast (mixed/intermediate contrast).
- Dark regions: keep fissure_candidate ONLY for a thin slit clearly surrounded by one particle at native resolution; a dark line along a particle boundary is interfacial_gap; a dark region the host mask leaked over, an edge shadow, or anything unclear is unresolved_dark; open dark space is void_like_region.
- Artefacts: mark curtaining (vertical streaks), charging, smearing, scratches, redeposition, scan-line jumps, seams as regions.
- Missed particles: only if a clear discrete particle has no outline; give its centre in panel pixels of the LEFT panel.

Answer with ONLY a JSON object:
{{"crop_assessable": true|false,
 "objects": [{{"id": "...", "verdict": "keep|remove|relabel|reclassify_dark|ignore|unsure", "new_class": null or class, "confidence": "high|medium|low", "reason": "<=20 words"}}],
 "regions": [{{"verdict": "artifact_region|ignore", "kind": "curtaining|charging|smearing|scratch|redeposition|scan_line|seam|exterior|other", "bbox_panel_xyxy": [x0,y0,x1,y1], "confidence": "high|medium|low", "reason": "..."}}],
 "missed_particles": [{{"xy_panel": [x,y], "confidence": "high|medium|low", "reason": "..."}}],
 "notes": "<=40 words"}}
Give one entry per listed object. Use the LEFT-panel coordinate system (0..{pw}) for all panel coordinates."""

DETNOTE = {'BSE': 'canonical composition-contrast view',
           'ETD': 'surface/relief view; contrast classes A/B are not compositional here',
           'Inlens': 'surface/relief view; contrast classes A/B are not compositional here'}


def _parse(text):
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.S)
    m = re.search(r'\{.*\}', t, re.S)
    return json.loads(m.group(0) if m else t)


def verify_one(client, model, qdir, q, c, max_retries=4):
    w = c['window_xyxy']
    objs = '\n'.join(f"- {o['id']}, {o['type']}, {o['class']}, {o['area_px']}, {o['solidity']}, {','.join(o['tags'])}" for o in c['objects'])
    prompt = RUBRIC.format(det=q['detector'], detnote=DETNOTE.get(q['detector'], 'surface view'), zoom=c['scale'], origin=w[:2],
                           why='; '.join(f"{f['id']}: {f['why']}" for f in c['flagged']), objs=objs, pw=PANEL)
    img = base64.b64encode((pathlib.Path(qdir) / c['file']).read_bytes()).decode()
    usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}
    mt, err = 2500, None
    for attempt in range(max_retries):
        try:
            r = client.messages.create(model=model, max_tokens=mt, messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': img}},
                {'type': 'text', 'text': prompt}]}])
            usage['input_tokens'] += r.usage.input_tokens
            usage['output_tokens'] += r.usage.output_tokens
            usage['calls'] += 1
            if r.stop_reason == 'max_tokens':
                mt = min(8000, mt * 2)
                err = 'max_tokens'
                continue
            v = _parse(''.join(b.text for b in r.content if b.type == 'text'))
            return {'crop_id': c['crop_id'], 'verdict': v, 'usage': usage, 'attempts': attempt + 1}
        except json.JSONDecodeError as e:
            err = f'json: {e}'
        except Exception as e:  # transient API errors
            err = f'{type(e).__name__}: {e}'
            time.sleep(min(60, 5 * 2 ** attempt))
    return {'crop_id': c['crop_id'], 'verdict': None, 'error': err, 'usage': usage, 'attempts': max_retries}


def verify(qdir, model, workers=8):
    import anthropic
    qdir = pathlib.Path(qdir)
    q = json.loads((qdir / 'queue.json').read_text())
    client = anthropic.Anthropic(max_retries=3)
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: verify_one(client, model, qdir, q, c), q['crops']))
    tin = sum(r['usage']['input_tokens'] for r in res)
    tout = sum(r['usage']['output_tokens'] for r in res)
    pin, pout = PRICES.get(model, (float('nan'), float('nan')))
    out = {'model': model, 'generated_at': now(), 'wall_s': round(time.time() - t0, 1), 'results': res,
           'usage': {'input_tokens': tin, 'output_tokens': tout, 'calls': sum(r['usage']['calls'] for r in res),
                     'usd': round(tin / 1e6 * pin + tout / 1e6 * pout, 4)}}
    (qdir / 'verdicts.json').write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- convert

OK_CONF = {'high', 'medium'}


def to_review(qdir, reviewer, out_path):
    qdir = pathlib.Path(qdir)
    q = json.loads((qdir / 'queue.json').read_text())
    vd = json.loads((qdir / 'verdicts.json').read_text())
    is_bse = q['detector'] == 'BSE'
    crops = {c['crop_id']: c for c in q['crops']}
    rv = {'reviewer': reviewer, 'notes': '', 'inspected_regions': [], 'remove_instances': [], 'relabel_instances': [],
          'reclassify_dark': [], 'add_seeds': [], 'artifact_regions': [], 'ignore_regions': [], 'verified_exterior': [],
          'verified_references': {}, 'feature_overrides': {}, 'confirm_not_assessable': []}
    stats = {'crops': len(crops), 'failed': 0, 'not_assessable': 0, 'objects': 0, 'unsure': 0, 'low_conf_dropped': 0,
             'actions': 0, 'rejected': 0, 'verdict_counts': {}}
    done = set()
    for r in vd['results']:
        c = crops[r['crop_id']]
        v = r.get('verdict')
        if not v:
            stats['failed'] += 1
            continue
        if not v.get('crop_assessable', True):
            stats['not_assessable'] += 1
            continue
        w, s = c['window_xyxy'], c['scale']
        rv['inspected_regions'].append({'bbox_xyxy': w, 'what': f"crop-verifier {c['crop_id']}: " + '; '.join(f['why'] for f in c['flagged'])[:200]})
        types = {o['id']: o['type'] for o in c['objects']}
        for o in v.get('objects', []):
            oid, verdict = o.get('id'), o.get('verdict')
            if oid not in types or verdict not in VERDICTS:
                stats['rejected'] += 1
                continue
            stats['objects'] += 1
            stats['verdict_counts'][verdict] = stats['verdict_counts'].get(verdict, 0) + 1
            if verdict == 'unsure':
                stats['unsure'] += 1
                continue
            if verdict == 'keep' or oid in done:
                continue
            if o.get('confidence') not in OK_CONF:
                stats['low_conf_dropped'] += 1
                continue
            why = f"crop-verifier {c['crop_id']}: {o.get('reason', '')}"[:240]
            nc, t = o.get('new_class'), types[oid]
            if t == 'inst' and verdict == 'remove':
                rv['remove_instances'].append({'id': oid, 'reason': why})
            elif t == 'inst' and verdict == 'ignore':
                rv['remove_instances'].append({'id': oid, 'reason': 'ambiguous, left unlabelled: ' + why})
            elif t == 'inst' and verdict == 'relabel' and nc in PARTICLE_CLASSES and (is_bse or nc in ('unknown_inclusion', 'particle_unclassified_contrast')):
                rv['relabel_instances'].append({'id': oid, 'class': nc, 'reason': why})
            elif t == 'dark' and verdict in ('reclassify_dark', 'relabel') and nc in DARK_CLASSES:
                rv['reclassify_dark'].append({'id': oid, 'class': nc, 'reason': why})
            elif t == 'dark' and verdict in ('ignore', 'remove'):
                rv['reclassify_dark'].append({'id': oid, 'class': 'unresolved_dark', 'reason': 'not a confirmed structure: ' + why})
            else:
                stats['rejected'] += 1
                continue
            done.add(oid)
            stats['actions'] += 1

        def nat(b):
            return [int(w[0] + b[0] / s), int(w[1] + b[1] / s), int(w[0] + b[2] / s), int(w[1] + b[3] / s)]
        for g in v.get('regions', []):
            b = g.get('bbox_panel_xyxy')
            if not (isinstance(b, list) and len(b) == 4) or g.get('confidence') not in OK_CONF:
                continue
            key = 'artifact_regions' if g.get('verdict') == 'artifact_region' else 'ignore_regions' if g.get('verdict') == 'ignore' else None
            if key:
                rv[key].append({'bbox_xyxy': nat(b), 'reason': f"crop-verifier {c['crop_id']} {g.get('kind', '')}: {g.get('reason', '')}"[:240]})
                stats['actions'] += 1
        for m in v.get('missed_particles', []):
            xy = m.get('xy_panel')
            if m.get('confidence') == 'high' and isinstance(xy, list) and len(xy) == 2:
                rv['add_seeds'].append({'xy': [int(w[0] + xy[0] / s), int(w[1] + xy[1] / s)],
                                        'reason': f"crop-verifier {c['crop_id']}: {m.get('reason', '')}"[:240]})
                stats['actions'] += 1
    stats['unsure_rate'] = round(stats['unsure'] / max(1, stats['objects']), 3)
    rv['notes'] = (f"Automatic crop verification ({vd['model']}) of {stats['crops']} flagged crops; unsure/low-confidence verdicts "
                   f"left unchanged (auto, unreviewed). Coverage limited to inspected crop windows. Draft labels, not ground truth. stats={json.dumps(stats)}")
    pathlib.Path(out_path).write_text(json.dumps(rv, indent=1))
    return rv, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sem84.crop_review')
    sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('review', help='queue + verify + convert')
    a.add_argument('--stage-dir', required=True)
    a.add_argument('--src', required=True)
    a.add_argument('--queue-dir', required=True)
    a.add_argument('--out', required=True, help='review json path')
    a.add_argument('--reviewer', required=True)
    a.add_argument('--diff', help='pass_diff.json: re-verify only changed objects')
    a.add_argument('--model', default='claude-sonnet-4-6')
    a.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    diff = json.loads(pathlib.Path(args.diff).read_text()) if args.diff else None
    q = build_queue(args.stage_dir, args.src, args.queue_dir, diff=diff)
    if not q['crops']:
        rv = {'reviewer': args.reviewer, 'notes': 'crop-verifier: no changed objects to re-verify', 'inspected_regions': []}
        pathlib.Path(args.out).write_text(json.dumps(rv, indent=1))
        print(json.dumps({'crops': 0}))
        return
    vd = verify(args.queue_dir, args.model, args.workers)
    _, stats = to_review(args.queue_dir, args.reviewer, args.out)
    print(json.dumps({'crops': len(q['crops']), 'wall_s': vd['wall_s'], 'usage': vd['usage'], 'stats': stats}))


if __name__ == '__main__':
    main()

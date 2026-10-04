"""Whole-image review (native tiles + one image-level policy call) combined with the flagged-crop verifier.

tiles:  every review tile (raw + annotated, native px) -> instance fixes, missed particles, artefacts, per-tile observations.
policy: one call with overview(s) + tile observations + auto feature states -> detector-wide dark-region rule, feature overrides.
crops:  sem84.crop_review flagged crops at zoom; crop verdicts win over tile verdicts for the same object.
Output: a review_*.json for `sem84.cli run --review`. Draft labels, not ground truth.
"""
import argparse
import base64
import concurrent.futures as cf
import io
import json
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw

from . import crop_review as CR
from .catalogue import CHEM_LIMITED, COATING_REF, EXTERNAL, STATES
from .cli import DARK_CLASSES, PARTICLE_CLASSES, load_gray, now
from .render import font

RELABEL_OK = PARTICLE_CLASSES - {'unknown_inclusion'}  # validate.py forbids 'sio' substrings, which 'inclusion' contains

TILE_MODEL = 'claude-sonnet-4-6'
POLICY_MODEL = 'claude-opus-4-6'
OK_CONF = {'high', 'medium'}
VOIDISH = {'void_like_region', 'fissure_candidate', 'interfacial_gap'}
OBS_KEYS = ['curtaining', 'charging', 'smearing_redeposition', 'exterior_or_cap', 'prep_debris_or_fracture',
            'interparticle_crack', 'exfoliation_like_slits', 'particle_shaped_cavity', 'unknown_inclusion', 'scan_defects_or_blur']


def _jpg(im, q=88):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=q)
    return b.getvalue()


def _call(client, model, images, prompt, mt=4000, max_retries=4):
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': base64.b64encode(b).decode()}} for b in images]
    content.append({'type': 'text', 'text': prompt})
    usage, err = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}, None
    for attempt in range(max_retries):
        try:
            r = client.messages.create(model=model, max_tokens=mt, messages=[{'role': 'user', 'content': content}])
            usage['input_tokens'] += r.usage.input_tokens
            usage['output_tokens'] += r.usage.output_tokens
            usage['calls'] += 1
            if r.stop_reason == 'max_tokens':
                mt, err = min(16000, mt * 2), 'max_tokens'
                continue
            return CR._parse(''.join(b.text for b in r.content if b.type == 'text')), usage, None
        except json.JSONDecodeError as e:
            err = f'json: {e}'
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            time.sleep(min(60, 5 * 2 ** attempt))
    return None, usage, err


def _usd(model, u):
    pin, pout = CR.PRICES.get(model, (float('nan'), float('nan')))
    return u['input_tokens'] / 1e6 * pin + u['output_tokens'] / 1e6 * pout


def dark_medians(g, dark):
    out = {}
    for d in dark:
        x0, y0, x1, y1 = [int(v) for v in d['bbox_xyxy']]
        if x1 <= x0 or y1 <= y0 or len(d['polygon_xy']) < 3:
            continue
        m = Image.new('L', (x1 - x0 + 1, y1 - y0 + 1), 0)
        ImageDraw.Draw(m).polygon([(p[0] - x0, p[1] - y0) for p in d['polygon_xy']], fill=1)
        sub = g[y0:y1 + 1, x0:x1 + 1]
        mm = np.asarray(m, bool)[:sub.shape[0], :sub.shape[1]]
        if mm.any():
            out[d['id']] = float(np.median(sub[mm]))
    return out


# ----------------------------------------------------------------------------- tiles

def _annot(g, box, inst, dark, only=None):
    x0, y0, x1, y1 = box
    im = Image.fromarray(g[y0:y1, x0:x1]).convert('RGB')
    d = ImageDraw.Draw(im)
    f = font(14)
    hit = lambda b: b[0] < x1 and x0 < b[2] and b[1] < y1 and y0 < b[3]
    labels = []
    for o in dark:
        if not hit(o['bbox_xyxy']):
            continue
        col = CR.CLS_COL.get(o['class'], (255, 255, 255))
        pts = [(p[0] - x0, p[1] - y0) for p in o['polygon_xy']]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=col, width=1)
        if o['class'] != 'void_like_region' and o['area_px'] >= 40 and (only is None or o['id'] in only):
            labels.append((o, col))
    for o in inst:
        if not hit(o['bbox_xyxy']):
            continue
        col = CR.CLS_COL.get(o['class'], (255, 255, 255))
        if only is not None and o['id'] not in only:
            col = tuple(c // 2 for c in col)
        pts = [(p[0] - x0, p[1] - y0) for p in o['polygon_xy']]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=col, width=2)
        if only is None or o['id'] in only:
            labels.append((o, col))
    for o, col in labels:
        d.text((o['centroid_xy'][0] - x0 - 24, o['centroid_xy'][1] - y0 - 8), o['id'], fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return im


TILE_RUBRIC = """You review one native-resolution tile of a FIB-SEM cross-section of a battery electrode (detector: {det}; {detnote}).
Image 1 = raw tile. Image 2 = same tile with draft pipeline outlines and IDs. Tile origin in the full image: x0={x0}, y0={y0}; size {w}x{h} px; 1 image px = 1 native px.
Outline colours: particle_contrast_A yellow, particle_contrast_B cyan, unknown_inclusion magenta, particle_unclassified_contrast grey, void_like_region red (unlabelled), fissure_candidate green, interfacial_gap orange, unresolved_dark violet.
{scope}
Pipeline intensity model: void threshold t_void={tvoid}, particle grey mode={pmode}. Dark regions are thresholded below t_void.

Rules (non-negotiable):
- Draft labelling, not ground truth. Prefer removing/ignoring over guessing; if unsure, leave an object out of your lists (unlisted = keep).
- Never assert chemistry (Si, SiOx, graphite, binder, carbon black, copper, contamination, oxidation). Dark != pore. Touching != electrical contact. 2D cannot prove 3D connectivity/enclosure.
- Particle masks: remove masks that merge several particles, leak over matrix/background, cover textured fine matrix, bright relief/fill material, flecks or slivers, or are not a discrete particle.
- Relabel: {relabel_rule}
- Missed particles: list EVERY clear, discrete, well-bounded particle that has no outline (do not cap the count); give the particle centre in tile px.
- Labelled dark regions (green/orange/violet): fissure_candidate ONLY for a thin slit clearly surrounded by one particle; along a particle boundary = interfacial_gap; host-mask leak/edge shadow/unclear = unresolved_dark; open dark space = void_like_region.
- Regions: mark artefacts (curtaining = vertical streaks, charging, smearing, redeposition, scratch, scan-line jump, seam), ignore regions (unjudgeable), exterior (cap/embedding/outside sample) as bboxes in tile px.
- observations: for each key answer "seen", "not_seen" or "unsure" for THIS tile.
- dark_regions_are: what the red/thresholded dark regions in this tile mostly are: "pores_or_gaps", "particle_faces_or_relief" (dark-grey particle bodies or shadowed relief, not empty space), "mixed" or "unsure".

Answer with ONLY a JSON object:
{{"tile_assessable": true|false,
 "instances": [{{"id": "...", "verdict": "remove|relabel", "new_class": null or class, "confidence": "high|medium|low", "reason": "<=15 words"}}],
 "dark": [{{"id": "...", "new_class": "fissure_candidate|interfacial_gap|unresolved_dark|void_like_region", "confidence": "high|medium|low", "reason": "<=15 words"}}],
 "missed_particles": [{{"xy": [x, y], "confidence": "high|medium|low", "reason": "<=10 words"}}],
 "regions": [{{"verdict": "artifact_region|ignore|exterior", "kind": "...", "bbox_xyxy": [x0, y0, x1, y1], "confidence": "high|medium|low", "reason": "..."}}],
 "observations": {{{obs}}},
 "dark_regions_are": "...",
 "instance_quality": "good|some_errors|poor",
 "notes": "<=50 words"}}"""

RELABEL = {True: 'particle_contrast_A (darker grey), particle_contrast_B (bright), particle_unclassified_contrast (intermediate/mixed contrast, e.g. lighter-grey textured phase or grey body with bright lens).',
           False: 'surface/relief view: only particle_unclassified_contrast is allowed; contrast here is not compositional.'}


def tile_review(stage_dir, src, out_dir, model=TILE_MODEL, workers=8, only=None):
    import anthropic
    stage_dir, out_dir = pathlib.Path(stage_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = json.loads((stage_dir / 'annotation.json').read_text())
    tiles = json.loads((stage_dir / 'review_tiles' / 'index.json').read_text())
    g, _ = load_gray(src)
    inst, dark, im = ann['instances'], ann['dark_regions'], ann['intensity_model']
    is_bse = ann['detector'] == 'BSE'
    if only is not None:
        objs = [o for o in inst + dark if o['id'] in only]
        tiles = [t for t in tiles if any(CR._isect(o['bbox_xyxy'], t['bbox_xyxy']) for o in objs)]
        scope = ('SCOPE: re-check only the objects shown at full brightness with ID labels (added/changed in the last pass: '
                 + ', '.join(sorted(o['id'] for o in objs if CR._isect(o['bbox_xyxy'], [0, 0, 10 ** 6, 10 ** 6]))[:80]) + '). Dimmed outlines were reviewed before.')
    else:
        scope = 'SCOPE: review every outlined object in the tile.'
    client = anthropic.Anthropic(max_retries=3)

    def one(t):
        b = t['bbox_xyxy']
        raw = Image.fromarray(g[b[1]:b[3], b[0]:b[2]]).convert('RGB')
        a = _annot(g, b, inst, dark, only)
        a.save(out_dir / f"tile_{t['tile']}_annotated.jpg", quality=85)
        p = TILE_RUBRIC.format(det=ann['detector'], detnote=CR.DETNOTE.get(ann['detector'], 'surface view'), x0=b[0], y0=b[1],
                               w=b[2] - b[0], h=b[3] - b[1], scope=scope, tvoid=round(im['t_void'], 1), pmode=im.get('particle_mode'),
                               relabel_rule=RELABEL[is_bse], obs=', '.join(f'"{k}": "..."' for k in OBS_KEYS))
        v, u, err = _call(client, model, [_jpg(raw, 92), _jpg(a)], p)
        return {'tile': t['tile'], 'bbox_xyxy': b, 'verdict': v, 'usage': u, 'error': err}

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, tiles))
    u = {k: sum(r['usage'][k] for r in res) for k in ('input_tokens', 'output_tokens', 'calls')}
    u['usd'] = round(_usd(model, u), 4)
    out = {'model': model, 'generated_at': now(), 'wall_s': round(time.time() - t0, 1), 'mode': 'diff' if only is not None else 'full',
           'results': res, 'usage': u}
    (out_dir / 'tiles.json').write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- policy

POLICY_PROMPT = """You set image-level review decisions for one FIB-SEM battery-electrode image (detector {det}; {detnote}). Draft labels, not ground truth.
Image 1: whole image, raw (downscaled {sc:.3f}x). Image 2: same with pipeline outlines (red = thresholded dark 'void_like' regions, yellow/cyan/grey = particle masks, violet = unresolved_dark).{refnote}
Intensity model: {im}
Dark-region median grey (area-weighted quantiles over thresholded dark regions now classed void_like/fissure/gap): {dq}; image grey histogram quantiles: {gq}.
Per-tile observations from native-resolution tile review (all {nt} tiles, 100% of the valid area):
{tobs}
Automatic feature states (id | name | state | rationale):
{feats}

Rules (non-negotiable):
- Never assert chemistry, 3D connectivity/enclosure/tortuosity, electrical contact or failure mechanism. Fresh, destructively prepared section.
- Dark != pore. In surface views (Inlens/ETD) dark grey particle faces/shadowed relief often fall below t_void; if most thresholded dark regions are particle bodies rather than empty space, set dark_rule.action="reclassify_grey_at_least" with grey_min = the median-grey cut above which components are particle faces (near-black gaps stay void_like). Otherwise action="none".
- If the dark rule removes most of the void mask, void/pore/crack-geometry features (V*, D03, D04, H08, etc.) derived from it are "not_assessable" here (assess in the canonical BSE view).
- not_observed_in_valid_view only if tiles report "not_seen" for that defect across all tiles; use confidence "low".
- Features currently "unreviewed" must get a state: observed | measured_2d | candidate_inference | not_observed_in_valid_view | not_assessable, each with a rationale (<=40 words, cite tiles/evidence).
- Only override where the evidence justifies it; do not touch requires_external_evidence features.

Answer ONLY a JSON object:
{{"dark_rule": {{"action": "none|reclassify_grey_at_least", "grey_min": null or number, "confidence": "high|medium|low", "reason": "..."}},
 "feature_overrides": {{"<id>": {{"state": "...", "rationale": "...", "confidence": "low|medium"}}}},
 "confirm_not_assessable": ["<id>", ...],
 "notes": "<=60 words"}}"""


def _overview(g, inst, dark, width=1568):
    s = width / g.shape[1]
    raw = Image.fromarray(g).convert('RGB')
    a = raw.copy()
    d = ImageDraw.Draw(a)
    for o in dark:
        pts = [tuple(p) for p in o['polygon_xy']]
        if len(pts) >= 3:
            d.polygon(pts, outline=CR.CLS_COL.get(o['class'], (255, 255, 255)))
    for o in inst:
        pts = [tuple(p) for p in o['polygon_xy']]
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=CR.CLS_COL.get(o['class'], (255, 255, 255)), width=5)
    sz = (width, round(g.shape[0] * s))
    return raw.resize(sz, Image.LANCZOS), a.resize(sz, Image.LANCZOS), s


def _wq(vals, wts, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not vals:
        return {}
    o = np.argsort(vals)
    v, w = np.asarray(vals)[o], np.cumsum(np.asarray(wts)[o])
    return {f'q{int(q * 100)}': float(v[np.searchsorted(w, q * w[-1])]) for q in qs}


def policy(stage_dir, src, tiles, out_dir, ref_src=None, model=POLICY_MODEL):
    import anthropic
    stage_dir, out_dir = pathlib.Path(stage_dir), pathlib.Path(out_dir)
    ann = json.loads((stage_dir / 'annotation.json').read_text())
    g, _ = load_gray(src)
    dark = ann['dark_regions']
    med = dark_medians(g, [d for d in dark if d['class'] in VOIDISH])
    area = {d['id']: d['area_px'] for d in dark}
    raw, a, s = _overview(g, ann['instances'], dark)
    imgs = [_jpg(raw), _jpg(a)]
    refnote = ''
    if ref_src and ann['detector'] != 'BSE':
        gr, _ = load_gray(ref_src)
        imgs.append(_jpg(Image.fromarray(gr).convert('RGB').resize(raw.size, Image.LANCZOS)))
        refnote = ' Image 3: co-registered canonical BSE view of the same field (raw, same scale); use it to judge whether Inlens/ETD dark regions are empty space or particle bodies.'
    tl = []
    for r in tiles['results']:
        v = r.get('verdict') or {}
        tl.append(f"- {r['tile']} {r['bbox_xyxy']}: dark_regions_are={v.get('dark_regions_are')}; instance_quality={v.get('instance_quality')}; "
                  f"obs={json.dumps(v.get('observations', {}))}; notes={v.get('notes', '')}")
    fs = [f"{f['id']} | {f['name']} | {f['state']} | {f['rationale'][:160]}" for f in ann['features'] if f['id'] not in EXTERNAL and f['id'] not in COATING_REF]
    gq = {f'q{q}': float(v) for q, v in zip((5, 25, 50, 75, 95), np.percentile(g, [5, 25, 50, 75, 95]))}
    p = POLICY_PROMPT.format(det=ann['detector'], detnote=CR.DETNOTE.get(ann['detector'], 'surface view'), sc=s, refnote=refnote,
                             im=json.dumps({k: ann['intensity_model'].get(k) for k in ('t_void', 't_multiotsu', 'particle_mode', 'particle_sigma', 't_bright')}),
                             dq=json.dumps(_wq(list(med.values()), [area[k] for k in med])), gq=json.dumps(gq), nt=len(tl), tobs='\n'.join(tl), feats='\n'.join(fs))
    client = anthropic.Anthropic(max_retries=3)
    t0 = time.time()
    v, u, err = _call(client, model, imgs, p, mt=8000)
    u['usd'] = round(_usd(model, u), 4)
    out = {'model': model, 'generated_at': now(), 'wall_s': round(time.time() - t0, 1), 'verdict': v, 'error': err, 'usage': u,
           'dark_median_grey': med}
    (out_dir / 'policy.json').write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- merge

def merge(stage_dir, tiles, pol, crop_review_path, reviewer, out_path, prev_policy=None):
    ann = json.loads((pathlib.Path(stage_dir) / 'annotation.json').read_text())
    is_bse = ann['detector'] == 'BSE'
    inst_cls = {i['id']: i['class'] for i in ann['instances']}
    dark_by = {d['id']: d for d in ann['dark_regions']}
    feats = {f['id']: f for f in ann['features']}
    crop = json.loads(pathlib.Path(crop_review_path).read_text()) if crop_review_path else {}
    rv = {'reviewer': reviewer, 'notes': '', 'inspected_regions': [], 'remove_instances': [], 'relabel_instances': [],
          'reclassify_dark': [], 'add_seeds': [], 'artifact_regions': [], 'ignore_regions': [], 'verified_exterior': [],
          'verified_references': {}, 'feature_overrides': {}, 'confirm_not_assessable': []}
    st = {'tile_failed': 0, 'tile_actions': 0, 'crop_actions': 0, 'policy_dark': 0, 'overrides': 0, 'overrides_rejected': 0}
    decided = set()
    for k in ('remove_instances', 'relabel_instances', 'reclassify_dark', 'add_seeds', 'artifact_regions', 'ignore_regions'):
        for it in crop.get(k, []):
            rv[k].append(it)
            if 'id' in it:
                decided.add(it['id'])
            st['crop_actions'] += 1
    rv['inspected_regions'] += crop.get('inspected_regions', [])
    for r in tiles['results']:
        v, b = r.get('verdict'), r['bbox_xyxy']
        if not v or not v.get('tile_assessable', True):
            st['tile_failed'] += 1
            continue
        rv['inspected_regions'].append({'bbox_xyxy': b, 'what': f"tile-verifier {r['tile']} native px ({tiles['mode']})"})
        tag = f"tile-verifier {r['tile']}: "
        for o in v.get('instances', []):
            oid, nc = o.get('id'), o.get('new_class')
            if oid not in inst_cls or oid in decided or o.get('confidence') not in OK_CONF:
                continue
            if o.get('verdict') == 'remove':
                rv['remove_instances'].append({'id': oid, 'reason': (tag + o.get('reason', ''))[:240]})
            elif o.get('verdict') == 'relabel' and nc in RELABEL_OK and nc != inst_cls[oid] and (is_bse or nc == 'particle_unclassified_contrast'):
                rv['relabel_instances'].append({'id': oid, 'class': nc, 'reason': (tag + o.get('reason', ''))[:240]})
            else:
                continue
            decided.add(oid)
            st['tile_actions'] += 1
        for o in v.get('dark', []):
            oid, nc = o.get('id'), o.get('new_class')
            if oid in dark_by and oid not in decided and nc in DARK_CLASSES and nc != dark_by[oid]['class'] and o.get('confidence') in OK_CONF:
                rv['reclassify_dark'].append({'id': oid, 'class': nc, 'reason': (tag + o.get('reason', ''))[:240]})
                decided.add(oid)
                st['tile_actions'] += 1
        for m in v.get('missed_particles', []):
            xy = m.get('xy')
            if m.get('confidence') in OK_CONF and isinstance(xy, list) and len(xy) == 2:
                gx, gy = int(b[0] + xy[0]), int(b[1] + xy[1])
                if b[0] <= gx < b[2] and b[1] <= gy < b[3] and all(abs(gx - s['xy'][0]) + abs(gy - s['xy'][1]) > 20 for s in rv['add_seeds']):
                    rv['add_seeds'].append({'xy': [gx, gy], 'reason': (tag + m.get('reason', ''))[:240]})
                    st['tile_actions'] += 1
        for gg in v.get('regions', []):
            bb = gg.get('bbox_xyxy')
            if not (isinstance(bb, list) and len(bb) == 4) or gg.get('confidence') not in OK_CONF:
                continue
            key = {'artifact_region': 'artifact_regions', 'ignore': 'ignore_regions', 'exterior': 'verified_exterior'}.get(gg.get('verdict'))
            if key == 'verified_exterior' and gg.get('confidence') != 'high':
                key = 'ignore_regions'
            if key:
                rv[key].append({'bbox_xyxy': [int(b[0] + bb[0]), int(b[1] + bb[1]), int(b[0] + bb[2]), int(b[1] + bb[3])],
                                'reason': (tag + f"{gg.get('kind', '')}: {gg.get('reason', '')}")[:240]})
                st['tile_actions'] += 1
    rule = ((pol or {}).get('verdict') or {}).get('dark_rule') or (prev_policy or {}).get('dark_rule') or {}
    if rule.get('action') == 'reclassify_grey_at_least' and isinstance(rule.get('grey_min'), (int, float)) and rule.get('confidence') in OK_CONF:
        med = (pol or {}).get('dark_median_grey') or {}
        for oid, m in med.items():
            if oid in dark_by and oid not in decided and dark_by[oid]['class'] in VOIDISH and m >= rule['grey_min']:
                rv['reclassify_dark'].append({'id': oid, 'class': 'unresolved_dark',
                                              'reason': f"policy: median grey {m:.0f} >= {rule['grey_min']} (particle face/relief, dark != pore): {rule.get('reason', '')}"[:240]})
                st['policy_dark'] += 1
    pv = (pol or {}).get('verdict') or {}
    for fid, ov in (pv.get('feature_overrides') or {}).items():
        f = feats.get(fid)
        ok = (f is not None and fid not in EXTERNAL and fid not in COATING_REF and isinstance(ov, dict) and ov.get('state') in STATES
              and ov.get('state') not in ('requires_external_evidence', 'unreviewed') and ov.get('rationale') and ov['state'] != f['state']
              and not (fid in CHEM_LIMITED and ov['state'] != 'candidate_inference'))
        if ok:
            o = {'state': ov['state'], 'rationale': 'policy-verifier: ' + ov['rationale'][:400], 'confidence': ov.get('confidence') if ov.get('confidence') in ('low', 'medium') else 'low'}
            if ov['state'] == 'not_assessable':
                o['value'] = None
            rv['feature_overrides'][fid] = o
            st['overrides'] += 1
        else:
            st['overrides_rejected'] += 1
    rv['confirm_not_assessable'] = [x for x in pv.get('confirm_not_assessable') or [] if x in feats and feats[x]['state'] == 'not_assessable']
    rv['notes'] = (f"Automatic review: native tile verifier ({tiles['model']}, {tiles['mode']}), flagged-crop verifier, image-level policy "
                   f"({(pol or {}).get('model', 'reused')}). Low-confidence/unsure verdicts left unchanged. {pv.get('notes', '')} "
                   f"Draft labels, not expert ground truth; chemistry unconfirmed. stats={json.dumps(st)}")
    pathlib.Path(out_path).write_text(json.dumps(rv, indent=1))
    return rv, st


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sem84.image_review')
    ap.add_argument('--stage-dir', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reviewer', required=True)
    ap.add_argument('--reference-src')
    ap.add_argument('--diff', help='pass_diff.json -> re-check only changed objects, reuse --prev-policy dark rule')
    ap.add_argument('--prev-policy')
    ap.add_argument('--tile-model', default=TILE_MODEL)
    ap.add_argument('--policy-model', default=POLICY_MODEL)
    ap.add_argument('--crop-model', default=TILE_MODEL)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)
    wd = pathlib.Path(a.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    diff = json.loads(pathlib.Path(a.diff).read_text()) if a.diff else None
    only = None
    if diff is not None:
        only = {x['id'] for x in diff.get('instances_added', []) + diff.get('instance_class_changes', [])}
        only |= set(diff.get('dark_added', []))
    t0 = time.time()
    q = CR.build_queue(a.stage_dir, a.src, wd / 'crops', diff=diff)

    def crops():
        if not q['crops']:
            return None
        if not (wd / 'crops' / 'verdicts.json').exists():
            CR.verify(wd / 'crops', a.crop_model, a.workers)
        CR.to_review(wd / 'crops', a.reviewer, wd / 'crop_review.json')
        return json.loads((wd / 'crops' / 'verdicts.json').read_text())['usage']

    with cf.ThreadPoolExecutor(2) as ex:
        fc = ex.submit(crops)
        ft = ex.submit(tile_review, a.stage_dir, a.src, wd / 'tiles', a.tile_model, a.workers, only)
        cu, tiles = fc.result(), ft.result()
    pol, prev = None, None
    if diff is None:
        pol = policy(a.stage_dir, a.src, tiles, wd, a.reference_src, a.policy_model)
    else:
        if a.prev_policy:
            prev = (json.loads(pathlib.Path(a.prev_policy).read_text()).get('verdict') or {})
            ann = json.loads((pathlib.Path(a.stage_dir) / 'annotation.json').read_text())
            g, _ = load_gray(a.src)
            pol = {'verdict': {'dark_rule': prev.get('dark_rule')}, 'model': 'reused-pass1-policy',
                   'dark_median_grey': dark_medians(g, [d for d in ann['dark_regions'] if d['class'] in VOIDISH]), 'usage': {'usd': 0}}
    rv, st = merge(a.stage_dir, tiles, pol, (wd / 'crop_review.json') if q['crops'] else None, a.reviewer, a.out, prev)
    usd = tiles['usage']['usd'] + (cu or {}).get('usd', 0) + ((pol or {}).get('usage') or {}).get('usd', 0)
    summ = {'wall_s': round(time.time() - t0, 1), 'crops': len(q['crops']), 'tiles': len(tiles['results']), 'usd': round(usd, 4),
            'tile_usage': tiles['usage'], 'crop_usage': cu, 'policy_usage': (pol or {}).get('usage'), 'stats': st,
            'n': {k: len(rv[k]) for k in ('remove_instances', 'relabel_instances', 'reclassify_dark', 'add_seeds', 'artifact_regions', 'ignore_regions', 'verified_exterior', 'feature_overrides')}}
    (wd / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == '__main__':
    main()

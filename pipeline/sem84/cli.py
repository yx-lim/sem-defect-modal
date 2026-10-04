"""sem84 command line: run one stage for one image, validate, package."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shutil
import stat
import sys
import tarfile
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.registration import phase_cross_correlation

from . import VERSION
from . import analysis as A
from .catalogue import ROOT, load_catalogue
from .features import apply_overrides, assess, build_stats
from .render import Renderer, charts, dashboard
from .validate import sha256, validate_stage

Image.MAX_IMAGE_PIXELS = None
STAGES = ['pass1', 'pass2', 'final']
PARTICLE_CLASSES = {'particle_contrast_A', 'particle_contrast_B', 'unknown_inclusion', 'particle_unclassified_contrast'}
DARK_CLASSES = {'void_like_region', 'fissure_candidate', 'interfacial_gap', 'unresolved_dark'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def log(*a):
    print(f'[{now()}]', *a, flush=True)


def load_gray(path):
    im = Image.open(path)
    a = np.asarray(im)
    info = {'pil_mode': im.mode, 'array_shape': list(a.shape), 'dtype': str(a.dtype),
            'compression': im.info.get('compression'), 'dpi_tag': [float(x) for x in im.info.get('dpi', [])]}
    if a.ndim == 3:
        eq = bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        info['rgb_planes_identical'] = eq
        g = a[..., 0].copy() if eq else np.round(a[..., :3].mean(2)).astype(np.uint8)
    else:
        g = a
    if g.dtype != np.uint8:
        info['rescaled_to_uint8'] = True
        g = np.round(255 * (g - g.min()) / max(1, g.max() - g.min())).astype(np.uint8)
    return g, info


def manifest_rec(image_id):
    m = json.loads((ROOT / 'manifest.json').read_text())
    for f in m['files']:
        if f['image_id'] == image_id:
            return f
    raise SystemExit(f'image {image_id} not in manifest')


def versions(models_dir):
    import onnxruntime
    import scipy
    import skimage
    code = hashlib.sha256()
    for p in sorted((ROOT / 'sem84').glob('*.py')):
        code.update(p.read_bytes())
    ms = json.loads((ROOT / 'model_sources.json').read_text())
    return {
        'pipeline': VERSION, 'pipeline_code_sha256': code.hexdigest(),
        'model': 'MobileSAM ONNX (image encoder + multimask decoder)', 'model_sources': ms,
        'model_sha256': {p.name: sha256(p) for p in sorted(pathlib.Path(models_dir).glob('*.onnx'))},
        'prompt_sha256': sha256(ROOT / 'COORDINATOR_PROMPT_BATCH3.txt'),
        'catalogue_sha256': sha256(ROOT / 'microstructure_feature_catalogue.json'),
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'scikit_image': skimage.__version__, 'onnxruntime': onnxruntime.__version__, 'pillow': Image.__version__,
    }


def check_models(models_dir):
    ms = json.loads((ROOT / 'model_sources.json').read_text())
    text = json.dumps(ms)
    for p in pathlib.Path(models_dir).glob('*.onnx'):
        if sha256(p) not in text:
            raise SystemExit(f'model hash mismatch for {p.name}')


# ----------------------------------------------------------------------------- state

def save_state(path, state):
    meta = {k: v for k, v in state.items() if k not in ('canvas', 'suppress')}
    meta['recs'] = {str(k): v for k, v in state['recs'].items()}
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, canvas=state['canvas'], suppress=state['suppress'], meta=np.array(json.dumps(meta)))
    tmp.replace(path)


def load_state(path):
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    meta['recs'] = {int(k): v for k, v in meta['recs'].items()}
    meta['canvas'] = d['canvas'].copy()
    meta['suppress'] = d['suppress'].copy() if 'suppress' in d.files else np.zeros(meta['canvas'].shape, np.uint8)
    return meta


def new_state(shape):
    return {'canvas': np.zeros(shape, np.uint16), 'suppress': np.zeros(shape, np.uint8), 'recs': {}, 'next_label': 1, 'removed': [], 'relabels': {},
            'dark_reclass': {}, 'artifact_regions': [], 'ignore_regions': [], 'verified_exterior': [],
            'verified_references': {}, 'reviews': [], 'inspected': {}, 'reject_counts': {}, 'regions_done': {}}


def remove_label(state, label, reason, stage, who):
    rec = state['recs'].pop(label, None)
    if rec is None:
        return
    if who != 'sem84':  # reviewer removals block later automatic re-proposal of the same region
        state['suppress'][state['canvas'] == label] = 1
    state['canvas'][state['canvas'] == label] = 0
    state['removed'].append({'id': rec['id'], 'label': label, 'reason': reason, 'stage': stage, 'by': who})


def apply_review(path, stage, g, sm, valid, tm, sam_factory, state, is_bse):
    rv = json.loads(pathlib.Path(path).read_text())
    rv['stage'] = stage
    rv['sha256'] = sha256(path)
    who = rv.get('reviewer', 'unknown_reviewer')
    if not rv.get('reviewer'):
        raise SystemExit('review file requires "reviewer"')
    byid = {r['id']: l for l, r in state['recs'].items()}
    warn = []
    for it in rv.get('remove_instances', []):
        if it['id'] in byid:
            remove_label(state, byid[it['id']], it.get('reason', 'reviewer_removed'), stage, who)
        else:
            warn.append(f"remove: unknown {it['id']}")
    for it in rv.get('relabel_instances', []):
        ok = it['class'] in PARTICLE_CLASSES and (is_bse or it['class'] in ('unknown_inclusion', 'particle_unclassified_contrast'))
        if ok and it['id'] in byid:
            state['relabels'][it['id']] = {'class': it['class'], 'reason': it.get('reason'), 'by': who, 'stage': stage}
        else:
            warn.append(f"relabel rejected: {it}")
    for it in rv.get('reclassify_dark', []):
        if it['class'] in DARK_CLASSES:
            state['dark_reclass'][it['id']] = {'class': it['class'], 'reason': it.get('reason'), 'by': who, 'stage': stage}
        else:
            warn.append(f"dark reclass rejected: {it}")
    seeds = rv.get('add_seeds', [])
    if seeds:
        sam = sam_factory()
        for it in seeds:
            nid, why = A.seed_instance(it['xy'], g, sm, valid, tm, sam, state, it.get('reason'), who)
            if nid is None:
                warn.append(f"seed {it['xy']} rejected: {why}")
    for k in ('artifact_regions', 'ignore_regions', 'verified_exterior'):
        for it in rv.get(k, []):
            it = dict(it, stage=stage, by=who)
            state[k].append(it)
            if k != 'artifact_regions' and it.get('bbox_xyxy'):
                x0, y0, x1, y1 = [int(v) for v in it['bbox_xyxy']]
                state['suppress'][max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
    state['verified_references'].update(rv.get('verified_references') or {})
    state['inspected'][stage] = rv.get('inspected_regions', [])
    slim = {k: rv.get(k) for k in ('reviewer', 'stage', 'sha256', 'notes', 'feature_overrides', 'confirm_not_assessable')}
    slim['warnings'] = warn
    state['reviews'].append(slim)
    return warn


def review_coverage(state, valid):
    m = np.zeros(valid.shape, bool)
    for regs in state['inspected'].values():
        for r in regs:
            x0, y0, x1, y1 = [int(v) for v in r['bbox_xyxy']]
            m[y0:y1, x0:x1] = True
    return float((m & valid).sum() / max(1, valid.sum()))


# ----------------------------------------------------------------------------- registration

def _gm(a):
    return filters.sobel(ndi.gaussian_filter(a.astype(np.float32), 1.5))


def register(ref, mov):
    H, W = min(ref.shape[0], mov.shape[0]), min(ref.shape[1], mov.shape[1])
    ref, mov = ref[:H, :W], mov[:H, :W]
    h4, w4 = H // 4, W // 4
    rs = ref[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4).mean((1, 3))
    ms = mov[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4).mean((1, 3))
    sh, err, _ = phase_cross_correlation(_gm(rs), _gm(ms), upsample_factor=4)
    gshift = sh * 4
    gr, gmv = _gm(ref), _gm(mov)
    tiles = []
    T = 768
    for cy in (H // 4, 3 * H // 4):
        for cx in (W // 6, W // 2, 5 * W // 6):
            y0, x0 = int(cy - T // 2), int(cx - T // 2)
            my0, mx0 = int(round(y0 - gshift[0])), int(round(x0 - gshift[1]))
            if min(y0, x0, my0, mx0) < 0 or max(y0, my0) + T > H or max(x0, mx0) + T > W:
                continue
            s, e, _ = phase_cross_correlation(gr[y0:y0 + T, x0:x0 + T], gmv[my0:my0 + T, mx0:mx0 + T], upsample_factor=4)
            resid = s + (np.array([my0 - y0, mx0 - x0]) + gshift)
            tiles.append({'centre_xy': [cx, cy], 'residual_dy_dx_px': [float(resid[0]), float(resid[1])]})
    res = np.array([t['residual_dy_dx_px'] for t in tiles]) if tiles else np.zeros((1, 2))
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    a = ndi.shift(gmv, gshift, order=1)
    pad = int(np.ceil(np.abs(gshift).max())) + 2
    ncc = float(np.corrcoef(gr[pad:-pad, pad:-pad].ravel()[::7], a[pad:-pad, pad:-pad].ravel()[::7])[0, 1])
    ok = bool(len(tiles) >= 4 and rms <= 2.0 and np.abs(res).max() <= 4.0)
    return {'model': 'translation', 'dy_px': float(gshift[0]), 'dx_px': float(gshift[1]),
            'convention': 'x_ref = x_this + dx; y_ref = y_this + dy (reference = canonical BSE view)',
            'tile_residuals': tiles, 'residual_rms_px': rms, 'gradient_ncc': ncc, 'validated': ok,
            'method': 'phase correlation of Sobel-magnitude images (4x downsampled global, 768px native tiles for residuals)',
            'acceptance': 'validated if >=4 tiles, residual RMS <= 2 px and max |residual| <= 4 px'}


def transfer_bright(ref_g, reg, mov_g, mov_grad, tm_mov):
    valid = A.valid_region(ref_g)
    sm, _, tm = A.intensity_model(ref_g, valid, True)
    b = ndi.binary_opening((sm > tm['t_bright']) & valid, iterations=2)
    b = morphology.remove_small_objects(b, 200)
    H, W = mov_g.shape
    full = np.zeros((H, W), bool)
    hh, ww = min(H, b.shape[0]), min(W, b.shape[1])
    full[:hh, :ww] = b[:hh, :ww]
    moved = ndi.shift(full.astype(np.uint8), (-reg['dy_px'], -reg['dx_px']), order=0).astype(bool)
    bnd = moved & ~ndi.binary_erosion(moved)
    support = float(np.median(mov_grad[bnd]) / max(tm_mov['grad_median_valid'], 1e-6)) if bnd.any() else 0.0
    return moved, {'bse_t_bright': tm['t_bright'], 'transferred_px': int(moved.sum()),
                   'local_boundary_support_ratio': support, 'locally_validated': support >= 1.2}


# ----------------------------------------------------------------------------- analysis + outputs

def analyse(g, sm, grad, valid, tm, state, is_bse):
    t0 = time.time()
    canvas = state['canvas']
    inst = A.instance_geometry(canvas, state['recs'], sm, grad, valid, tm)
    A.classify_instances(inst, tm, is_bse, state['relabels'])
    dark_lab, dark, sup, unres = A.dark_structures(sm, grad, valid, canvas, tm, inst, state['dark_reclass'])
    matrix, mref = A.matrix_candidates(sm, valid, canvas, sup, unres, tm, inst)
    rels = A.relations(canvas, inst, sup, matrix)
    clusters = A.contrast_clusters(inst) if is_bse else []
    encl = A.enclosure_candidates(canvas, inst) if is_bse else []
    qc, low, high = A.qc_maps(g, sm, valid, canvas, inst, tm)
    mx = int(canvas.max()) + 1
    lut = np.full(mx, 255, np.uint8)
    lutB = np.zeros(mx, bool)
    for i in inst:
        lut[i['label']] = A.CLASS_ID[i['class']]
        lutB[i['label']] = i['class'] == 'particle_contrast_B'
    sem = np.full(g.shape, 255, np.uint8)
    sem[matrix] = 3
    sem = np.where(canvas > 0, lut[canvas], sem)
    dl = np.full(int(dark_lab.max()) + 1, 255, np.uint8)
    for d in dark:
        if d['class'] != 'unresolved_dark':
            dl[d['label']] = A.CLASS_ID[d['class']]
    sem = np.where(sup, dl[dark_lab], sem)
    sem[unres | ~valid] = 255
    qcm = np.zeros(g.shape, np.uint8)
    qcm[low] |= A.QC_BITS['clipped_low']
    qcm[high] |= A.QC_BITS['clipped_high']
    for b in qc['blur']['low_sharpness_tiles']:
        qcm[b[1]:b[3], b[0]:b[2]] |= A.QC_BITS['blur_tile']
    for b in qc['directional_streaks']['aligned_tiles']:
        qcm[b[1]:b[3], b[0]:b[2]] |= A.QC_BITS['curtain_tile']
    for r in qc['scan_line_rows']:
        qcm[r, :] |= A.QC_BITS['scan_line']
    qcm[unres] |= A.QC_BITS['unresolved_dark']
    for it in state['verified_exterior']:
        x0, y0, x1, y1 = it['bbox_xyxy']
        sub = sem[y0:y1, x0:x1]
        sub[canvas[y0:y1, x0:x1] == 0] = 9
    for it in state['artifact_regions']:
        x0, y0, x1, y1 = it['bbox_xyxy']
        qcm[y0:y1, x0:x1] |= A.QC_BITS['reviewer_artifact']
    for it in state['ignore_regions']:
        x0, y0, x1, y1 = it['bbox_xyxy']
        qcm[y0:y1, x0:x1] |= A.QC_BITS['reviewer_ignore']
        sem[y0:y1, x0:x1] = 255
    fg = (sem == 5) | (sem == 6)
    masks = {'canvas': canvas, 'valid': valid, 'supported_dark': sup, 'unresolved_dark': unres, 'dark_lab': dark_lab,
             'skeleton': morphology.skeletonize(sup), 'matrix': matrix, 'semantic': sem, 'qc': qcm,
             'clip_low': low, 'clip_high': high, 'contrast_B': lutB[canvas], 'fissure_gap': fg, 'fissure_only': sem == 5,
             'dist_to_fissure_gap': ndi.distance_transform_edt(~fg) if (is_bse and fg.any()) else None,
             'showthrough_fraction': float((sm[sup] > 0.6 * tm['t_void']).mean()) if sup.any() else None}
    stats = build_stats(inst, dark, rels, clusters, encl, masks, qc, tm, is_bse)
    log(f'analysis {len(inst)} instances, {len(dark)} dark comps, {len(rels)} relations in {time.time() - t0:.0f}s')
    return {'inst': inst, 'dark': dark, 'rels': rels, 'clusters': clusters, 'enclosures': encl, 'qc': qc,
            'masks': masks, 'stats': stats, 'matrix_texture_ref': mref}


def dark_polygon(dark_lab, d):
    x0, y0, x1, y1 = d['bbox_xyxy']
    m = dark_lab[y0:y1, x0:x1] == d['label']
    return A._polygon(m, y0, x0, 1.0)


def write_csvs(sd, feats, ctx, stage, res_px):
    vpx = ctx['stats']['valid_area_px']
    cov = ctx['coverage']
    with open(sd / 'feature_presence.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['feature_id', 'name', 'group', 'observability', 'state', 'value_is_null', 'value_json', 'unit', 'confidence',
                    'reason', 'evidence_needed', 'review_status'])
        for f in feats:
            w.writerow([f['id'], f['name'], f['group'], f['observability'], f['state'], f['value_is_null'],
                        '' if f['value'] is None else json.dumps(f['value'])[:2000], f['unit'] or '', f['confidence'] or '',
                        f['rationale'], f['evidence_needed'] or '', f['review_status']])
    rows = []

    def add(fid, metric, oid, val, unit, method, conf, trunc=False, size=None):
        rf = 'near_resolution_limit' if size is not None and size < 3 * res_px else ''
        rows.append([len(rows) + 1, fid, metric, oid, val, unit, method, conf, f'{cov:.4f}', vpx, trunc, rf, stage])
    for i in ctx['inst']:
        e = i['equivalent_diameter_px']
        for fid, met, key, unit in (('P02', 'area', 'area_px', 'px^2'), ('P02', 'equivalent_diameter', 'equivalent_diameter_px', 'px'),
                                    ('P02', 'feret_max', 'feret_max_px', 'px'), ('P02', 'feret_min', 'feret_min_px', 'px'),
                                    ('P03', 'aspect_ratio', 'aspect_ratio', 'ratio'), ('P03', 'minor_axis', 'minor_axis_px', 'px'),
                                    ('P04', 'circularity', 'circularity', 'ratio'), ('P04', 'solidity', 'solidity', 'ratio'),
                                    ('P05', 'orientation', 'orientation_deg', 'deg'), ('P06', 'boundary_roughness', 'boundary_roughness', 'ratio'),
                                    ('I01', 'touch_count', 'touch_count', 'count'), ('V11', 'ring_void_fraction', 'ring_void_fraction', 'fraction')):
            add(fid, met, i['id'], i.get(key), unit, 'instance geometry', i['confidence_label'], i['frame_truncated'], e)
    for d in ctx['dark']:
        if d['class'] == 'unresolved_dark':
            continue
        tr = 'frame_truncated' in d['tags']
        fid = {'void_like_region': 'V02', 'fissure_candidate': 'D03', 'interfacial_gap': 'D03'}[d['class']]
        add(fid, f"{d['class']}_area", d['id'], d['area_px'], 'px^2', 'dark component', d['confidence_label'], tr, d['equivalent_diameter_px'])
        add(fid, f"{d['class']}_equivalent_diameter", d['id'], d['equivalent_diameter_px'], 'px', 'dark component', d['confidence_label'], tr, d['equivalent_diameter_px'])
        add('V03', f"{d['class']}_aspect_ratio", d['id'], d['aspect_ratio'], 'ratio', 'dark component', d['confidence_label'], tr)
        add('V03', f"{d['class']}_orientation", d['id'], d['orientation_deg'], 'deg', 'dark component', d['confidence_label'], tr)
        add('D03', f"{d['class']}_median_width", d['id'], d['width_profile_px']['median'], 'px', '2x EDT on skeleton', d['confidence_label'], tr, d['width_profile_px']['median'])
        add('D04', f"{d['class']}_skeleton_length", d['id'], d['skeleton_length_px'], 'px', 'skeleton pixel count', d['confidence_label'], tr)
    for r in ctx['rels']:
        if r['type'] == 'touches':
            add('I01', 'apparent_contact_length', f"{r['a']}|{r['b']}", r['apparent_contact_length_px'], 'px', 'boundary within 2px', 'low')
    for f in feats:
        v = f['value']
        if isinstance(v, dict):
            for k, x in v.items():
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    add(f['id'], k, 'IMAGE', x, f['unit'] or '', f['method'] or '', f['confidence'] or '')
                elif isinstance(x, dict):
                    for k2, y in x.items():
                        if isinstance(y, (int, float)) and not isinstance(y, bool):
                            add(f['id'], f'{k}.{k2}', 'IMAGE', y, f['unit'] or '', f['method'] or '', f['confidence'] or '')
    with open(sd / 'measurements.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['row', 'feature_id', 'metric', 'object_id', 'value', 'unit', 'method', 'confidence', 'coverage',
                    'valid_area_px', 'frame_truncated', 'resolution_flag', 'stage'])
        w.writerows(rows)
    return len(rows)


def qa_items(ctx):
    items = []
    cracks = [d for d in ctx['dark'] if 'interparticle_crack_candidate' in d['tags']]
    fis = sorted([d for d in ctx['dark'] if d['class'] == 'fissure_candidate'], key=lambda d: -d['skeleton_length_px'])
    gaps = sorted([d for d in ctx['dark'] if d['class'] == 'interfacial_gap'], key=lambda d: -d['skeleton_length_px'])
    cav = [d for d in ctx['dark'] if 'particle_shaped_cavity_candidate' in d['tags']]
    unk = [i for i in ctx['inst'] if i['class'] == 'unknown_inclusion']
    low = sorted(ctx['inst'], key=lambda i: i['heuristic_confidence'])
    iso = [i for i in ctx['inst'] if i['touch_count'] == 0 and not i['frame_truncated']]
    for lst, kind, why, n in ((cracks, 'crack', 'interparticle crack candidate', 4), (fis, 'fissure', 'fissure candidate (cause unknown)', 5),
                              (gaps, 'gap', 'interfacial gap candidate', 3), (cav, 'cavity', 'particle-shaped cavity candidate', 3),
                              (unk, 'inclusion', 'unknown inclusion candidate', 3), (low, 'lowconf', 'lowest heuristic confidence instance', 4),
                              (iso, 'isolated', 'isolated-looking particle', 2)):
        for o in lst[:n]:
            items.append((kind, o['id'], o['centroid_xy'], why))
    for c in ctx['clusters'][:2]:
        xy = np.mean(c['envelope_xy_visual_only'], axis=0)
        items.append(('cluster', c['id'], xy.tolist(), 'contrast_B proximity cluster (not agglomerate proof)'))
    for b in ctx['qc']['directional_streaks']['aligned_tiles'][:2]:
        items.append(('streak', f"t{b[0]}_{b[1]}", [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], 'aligned streak tile'))
    return items


def pass_diff(prev_dir, ann, state, stage):
    if prev_dir is None or not (prev_dir / 'annotation.json').exists():
        return None
    p = json.loads((prev_dir / 'annotation.json').read_text())
    pi = {i['id']: i for i in p['instances']}
    ci = {i['id']: i for i in ann['instances']}
    pd = {d['id']: d['class'] for d in p['dark_regions']}
    cd = {d['id']: d['class'] for d in ann['dark_regions']}
    pf = {f['id']: f for f in p['features']}
    return {
        'from_stage': p['stage'], 'to_stage': stage, 'generated_at': now(),
        'instances_added': [{'id': k, 'pass_added': ci[k]['pass_added'], 'class': ci[k]['class']} for k in ci if k not in pi],
        'instances_removed': [r for r in state['removed'] if r['id'] in pi and r['id'] not in ci],
        'instance_class_changes': [{'id': k, 'from': pi[k]['class'], 'to': ci[k]['class']} for k in ci if k in pi and pi[k]['class'] != ci[k]['class']],
        'dark_class_changes': [{'id': k, 'from': pd[k], 'to': cd[k]} for k in cd if k in pd and pd[k] != cd[k]],
        'dark_added': [k for k in cd if k not in pd], 'dark_removed': [k for k in pd if k not in cd],
        'feature_state_changes': [{'id': f['id'], 'from': pf[f['id']]['state'], 'to': f['state'], 'rationale': f['rationale']}
                                  for f in ann['features'] if pf[f['id']]['state'] != f['state']],
        'reviews_applied': [r for r in state['reviews'] if r['stage'] == stage],
        'counts': {'prev_instances': len(pi), 'instances': len(ci), 'prev_dark': len(pd), 'dark': len(cd)},
    }


def write_sums(sd):
    lines = []
    for p in sorted(sd.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS':
            lines.append(f'{sha256(p)}  {p.relative_to(sd)}')
    (sd / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    return sha256(sd / 'SHA256SUMS')


def make_readonly(path):
    for p in sorted(path.rglob('*'), reverse=True):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def write_checkpoint(vd, update):
    p = vd / 'checkpoint.json'
    cp = json.loads(p.read_text()) if p.exists() else {'stages': {}}
    cp.update(update)
    cp['updated_at'] = now()
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(cp, indent=1))
    tmp.replace(p)
    return cp


def run(args):
    t_start = time.time()
    rec = manifest_rec(args.image_id)
    src = pathlib.Path(args.src)
    vd = pathlib.Path(args.out)
    work = vd / 'work'
    (work / 'regions').mkdir(parents=True, exist_ok=True)
    stage = args.stage
    sd = vd / stage
    if sd.exists():
        if stage == 'final' or not args.force:
            log(f'{sd} already exists; immutable, skipping')
            return 0
        sd.rename(vd / f'{stage}.superseded-{int(time.time())}')
    src_sha = sha256(src)
    if src_sha != rec['sha256']:
        raise SystemExit(f'source hash mismatch {src_sha} != {rec["sha256"]}')
    check_models(args.models)
    write_checkpoint(vd, {'image_id': rec['image_id'], 'source_sha256': src_sha, 'current_stage': stage, 'status': f'{stage}_running'})
    g, finfo = load_gray(src)
    H, W = g.shape
    if (W, H) != (rec['width'], rec['height']):
        raise SystemExit(f'dimension mismatch {(W, H)} vs manifest {(rec["width"], rec["height"])}')
    is_bse = rec['detector'] == 'BSE'
    valid = A.valid_region(g)
    sm, grad, tm = A.intensity_model(g, valid, is_bse)
    log(f'{rec["image_id"]} {W}x{H} {rec["detector"]} thresholds {tm["t_multiotsu"]} t_bright={tm["t_bright"]}')
    _sam = {}

    def sam_factory():
        if 'm' not in _sam:
            from .sam import MobileSam
            _sam['m'] = MobileSam(pathlib.Path(args.models), args.threads)
        return _sam['m']
    review_warn = []
    stf = work / f'state_{stage}.npz'
    if stf.exists():
        state = load_state(stf)
        log(f'resumed saved {stage} state')
    else:
        if stage == 'pass1':
            state = new_state(g.shape)
        else:
            prev = work / f'state_{STAGES[STAGES.index(stage) - 1]}.npz'
            if not prev.exists():
                raise SystemExit(f'{prev} missing; run previous stage first')
            state = load_state(prev)
        if stage != 'pass1':
            if not args.review:
                raise SystemExit(f'{stage} requires --review (reviewer decisions JSON)')
            review_warn = apply_review(args.review, stage, g, sm, valid, tm, sam_factory, state, is_bse)
        if stage == 'pass2':
            inst = A.instance_geometry(state['canvas'], state['recs'], sm, grad, valid, tm)
            for i in inst:
                if i['dark_fraction'] > 0.3:
                    remove_label(state, i['label'], 'pass2_auto_screen_dark_spanning', stage, 'sem84')
        if stage in ('pass1', 'pass2'):
            A.sam_pass(stage, g, sm, valid, tm, sam_factory(), state, work / 'regions', log)
        save_state(stf, state)
        write_checkpoint(vd, {'status': f'{stage}_proposals_saved'})
    ctx = analyse(g, sm, grad, valid, tm, state, is_bse)
    registration, transfer = None, None
    if not is_bse and args.reference_src:
        ref_rec = manifest_rec(args.reference_id)
        if sha256(args.reference_src) != ref_rec['sha256']:
            raise SystemExit('reference hash mismatch')
        rg, _ = load_gray(args.reference_src)
        registration = register(rg, g)
        registration.update({'reference_image_id': ref_rec['image_id'], 'reference_sha256': ref_rec['sha256'], 'this_image_id': rec['image_id']})
        if registration['validated']:
            moved, transfer = transfer_bright(rg, registration, g, grad, tm)
            registration['bright_candidate_transfer'] = transfer
            if transfer['locally_validated']:
                ctx['masks']['transferred_bright'] = moved
        log(f"registration dx={registration['dx_px']:.1f} dy={registration['dy_px']:.1f} rms={registration['residual_rms_px']:.2f} validated={registration['validated']}")
    cov = float((ctx['masks']['semantic'] != 255)[valid].mean())
    rcov = review_coverage(state, valid)
    ctx.update({'is_bse': is_bse, 'detector': rec['detector'], 'image_id': rec['image_id'], 'coverage': cov,
                'verified_references': state['verified_references'], 'reject_counts': state['reject_counts'],
                'artifact_regions': state['artifact_regions']})
    ctx['review_coverage'] = rcov
    feats = assess(ctx)
    ov_warn = apply_overrides(feats, state['reviews'], rcov)
    for f in feats:
        f['stage'] = stage
    staging = vd / f'.{stage}.staging-{os.getpid()}'
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'masks').mkdir(parents=True)
    canvas = ctx['masks']['canvas']
    Image.fromarray(canvas.astype(np.uint16)).save(staging / 'masks/instance_mask.png')
    Image.fromarray(ctx['masks']['semantic']).save(staging / 'masks/semantic_mask.png')
    Image.fromarray(ctx['masks']['qc']).save(staging / 'masks/qc_flags.png')
    pal = Image.fromarray(ctx['masks']['semantic'], 'L').convert('P')
    palette = [0] * 768
    cols = {1: (35, 231, 213), 2: (255, 210, 63), 3: (190, 120, 255), 4: (61, 139, 255), 5: (255, 61, 210), 6: (255, 140, 0),
            7: (255, 46, 46), 8: (184, 115, 51), 9: (120, 120, 120), 10: (255, 0, 0), 11: (120, 200, 160), 255: (0, 0, 0)}
    for k, c in cols.items():
        palette[3 * k:3 * k + 3] = c
    pal = Image.fromarray(ctx['masks']['semantic'], 'P')
    pal.putpalette(palette)
    pal.save(staging / 'masks/semantic_mask_palette.png')
    (staging / 'masks/class_map.json').write_text(json.dumps({
        'semantic_mask': 'uint8, source-sized; values per class_map', 'class_map': {str(k): v for k, v in A.CLASS_MAP.items()},
        'ignore_value': 255, 'ignore_meaning': 'unknown/unreviewed/unresolved/invalid; NOT background or healthy',
        'no_background_class': True, 'palette_rgb': {str(k): v for k, v in cols.items()},
        'instance_mask': 'uint16 PNG, source-sized; value = instance label (see annotation.instances[].label); 0 = no accepted instance (NOT background)',
        'qc_flags': 'uint8 bit flags', 'qc_bits': A.QC_BITS,
        'note': 'bright-phase / contrast_B = BSE contrast candidate; chemistry unconfirmed'}, indent=1))
    dark_out = []
    for d in ctx['dark']:
        dd = dict(d)
        dd['polygon_xy'] = dark_polygon(ctx['masks']['dark_lab'], d) if d['area_px'] >= 30 else []
        dark_out.append(dd)
    coco = {'info': {'description': f"sem84 {stage} vectors for {rec['image_id']}", 'units': 'native pixels', 'created': now()},
            'images': [{'id': 1, 'file_name': rec['filename'], 'width': W, 'height': H, 'sha256': src_sha}],
            'categories': [{'id': k, 'name': v} for k, v in A.CLASS_MAP.items() if k != 255] + [{'id': 100, 'name': 'unresolved_dark'}],
            'annotations': []}
    for i in ctx['inst']:
        x0, y0, x1, y1 = i['bbox_xyxy']
        coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': 1, 'category_id': A.CLASS_ID[i['class']],
                                    'object_id': i['id'], 'segmentation': [sum(i['polygon_xy'], [])], 'bbox': [x0, y0, x1 - x0, y1 - y0],
                                    'area': i['area_px'], 'iscrowd': 0, 'attributes': {'tags': i['tags'], 'confidence': i['confidence_label']}})
    for d in dark_out:
        if d['polygon_xy']:
            x0, y0, x1, y1 = d['bbox_xyxy']
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': 1,
                                        'category_id': 100 if d['class'] == 'unresolved_dark' else A.CLASS_ID[d['class']],
                                        'object_id': d['id'], 'segmentation': [sum(d['polygon_xy'], [])], 'bbox': [x0, y0, x1 - x0, y1 - y0],
                                        'area': d['area_px'], 'iscrowd': 0, 'attributes': {'tags': d['tags'], 'host': d['host_instance']}})
    (staging / 'vectors_coco.json').write_text(json.dumps(coco))
    vers = versions(args.models)
    ann = {
        'schema_version': 'sem84.annotation.v1', 'stage': stage, 'generated_at': now(), 'image_id': rec['image_id'],
        'batch': rec['batch'], 'field_stem': rec['field_stem'], 'detector': rec['detector'],
        'canonical_composition_view': is_bse,
        'source': {'filename': rec['filename'], 'sha256': src_sha, 'width': W, 'height': H, 'drive_url': rec['source_url'],
                   'tag_estimated_nm_per_pixel': rec['tag_estimated_nm_per_pixel'], 'calibration_verified': False, 'file_info': finfo},
        'units': {'length': 'px', 'area': 'px^2', 'angle': 'deg from +x toward +y (image coords, y down)',
                  'note': 'pixel units only; TIFF ~25 nm/px tag is unverified and not instrument resolution'},
        'coordinate_frame': 'native detector pixels of this source image; origin top-left',
        'sample_context': 'fresh, destructively prepared electrode section; no cycling-damage, chemistry or performance inference',
        'intensity_model': tm, 'valid_area_px': int(valid.sum()),
        'resolution': {'effective_resolution_limit_px': ctx['qc']['effective_resolution_limit_px'], 'method': ctx['qc']['effective_resolution_method']},
        'coverage': {'labelled_non_ignore_fraction_of_valid': cov, 'reviewer_inspected_fraction_of_valid': rcov},
        'features': feats, 'instances': ctx['inst'], 'dark_regions': dark_out,
        'relations': ctx['rels'] + ctx['enclosures'], 'clusters': ctx['clusters'],
        'qc': {k: v for k, v in ctx['qc'].items()}, 'registration': registration,
        'registration_note': None if not is_bse else 'Canonical BSE frame; paired-view transforms are stored in the ETD/Inlens/SE annotations of this stem.',
        'review': {'reviews': state['reviews'], 'removed_instances': state['removed'], 'relabels': state['relabels'],
                   'dark_reclass': state['dark_reclass'], 'artifact_regions': state['artifact_regions'],
                   'ignore_regions': state['ignore_regions'], 'verified_exterior': state['verified_exterior'],
                   'inspected_regions': state['inspected'], 'warnings': review_warn + ov_warn,
                   'review_status': 'auto_pass1_unreviewed' if stage == 'pass1' else f'{stage}_reviewed_draft'},
        'proposal_history': {'reject_counts': state['reject_counts'], 'regions_done': state['regions_done'], 'pass_cfg': A.PASS_CFG},
        'versions': vers,
        'disclaimer': 'Draft labelling, not expert ground truth. Model scores are not annotation accuracy. CHEMISTRY UNCONFIRMED.',
    }
    (staging / 'annotation.json').write_text(json.dumps(ann, indent=1, default=float))
    nrows = write_csvs(staging, feats, ctx, stage, ctx['qc']['effective_resolution_limit_px'])
    t_r = time.time()
    R = Renderer(g, ctx)
    fb = {f['id']: f for f in feats}
    layer_counts, layers = R.write_all(staging, fb, f"{rec['image_id']} [{stage}]", composites=(stage == 'final' or args.full_layers))
    qa = R.qa_crops(staging, layers, qa_items(ctx))
    if stage != 'final':
        R.review_tiles(staging, layers, ctx['inst'])
    charts(staging / 'charts.png', ctx)
    dashboard(staging / 'dashboard.png', feats, f"{rec['image_id']} [{stage}]")
    log(f'render {time.time() - t_r:.0f}s')
    prev = vd / STAGES[STAGES.index(stage) - 1] if stage != 'pass1' else None
    diff = pass_diff(prev, ann, state, stage)
    if diff is not None:
        (staging / 'pass_diff.json').write_text(json.dumps(diff, indent=1))
    counts = {}
    for f in feats:
        counts[f['state']] = counts.get(f['state'], 0) + 1
    v = validate_stage(staging, check_sums=False, final_files=False)
    cls = {}
    for i in ctx['inst']:
        cls[i['class']] = cls.get(i['class'], 0) + 1
    dcls = {}
    for d in ctx['dark']:
        dcls[d['class']] = dcls.get(d['class'], 0) + 1
    qcj = {'stage': stage, 'validation': v, 'review_warnings': review_warn + ov_warn,
           'checks': {'n_features': len(feats), 'mask_shape_matches_source': True, 'ignore_value': 255,
                      'source_sha_verified': True, 'model_sha_verified': True, 'units': 'px',
                      'registration_validated': None if registration is None else registration['validated']},
           'acquisition_qc': {k: ctx['qc'][k] for k in ('clipped_low_fraction', 'clipped_high_fraction', 'effective_resolution_limit_px')},
           'low_sharpness_tiles': len(ctx['qc']['blur']['low_sharpness_tiles']),
           'aligned_streak_tiles': len(ctx['qc']['directional_streaks']['aligned_tiles']),
           'scan_line_rows': len(ctx['qc']['scan_line_rows']), 'coverage': ann['coverage'],
           'manual_review_needed': [f['id'] for f in feats if f['state'] in ('unreviewed', 'candidate_inference')]}
    (staging / 'qc.json').write_text(json.dumps(qcj, indent=1))
    result = {'image_id': rec['image_id'], 'batch': rec['batch'], 'field_stem': rec['field_stem'], 'detector': rec['detector'],
              'stage': stage, 'status': 'ok' if v['ok'] else 'validation_failed', 'generated_at': now(),
              'source_sha256': src_sha, 'instances_by_class': cls, 'dark_regions_by_class': dcls,
              'relations_by_type': {t: sum(r['type'] == t for r in ctx['rels'] + ctx['enclosures']) for t in ('touches', 'separated_by_gap', 'adjacent_to', 'inside')},
              'clusters': len(ctx['clusters']), 'layer_nonzero_pixels': layer_counts, 'qa_crops': len(qa),
              'feature_state_counts': counts, 'measurement_rows': nrows, 'coverage': ann['coverage'],
              'registration': None if registration is None else {k: registration[k] for k in ('dx_px', 'dy_px', 'residual_rms_px', 'validated')},
              'elapsed_s': round(time.time() - t_start, 1), 'versions': {k: vers[k] for k in ('pipeline', 'pipeline_code_sha256', 'prompt_sha256')}}
    (staging / 'result.json').write_text(json.dumps(result, indent=1))
    sums = write_sums(staging)
    v2 = validate_stage(staging)
    if not v2['ok']:
        log('VALIDATION FAILED', v2['errors'][:10])
        write_checkpoint(vd, {'status': f'{stage}_validation_failed', 'errors': v2['errors'][:20]})
        return 2
    staging.rename(sd)
    if stage == 'final':
        make_readonly(sd)
    cp = json.loads((vd / 'checkpoint.json').read_text())
    cp['stages'][stage] = {'completed_at': now(), 'sha256sums_sha256': sums, 'result': result['status']}
    nxt = {'pass1': 'review pass1 tiles -> write review_pass1.json -> run --stage pass2',
           'pass2': 'review pass2 diff/QA -> write review_final.json -> run --stage final',
           'final': 'done: package and upload'}[stage]
    write_checkpoint(vd, {'stages': cp['stages'], 'status': f'{stage}_saved', 'next_action': nxt,
                          'versions': {k: vers[k] for k in ('pipeline_code_sha256', 'prompt_sha256', 'catalogue_sha256', 'model_sha256')}})
    log(f'{stage} saved -> {sd} ({time.time() - t_start:.0f}s); state counts {counts}')
    return 0


def package(args):
    vd = pathlib.Path(args.version_dir)
    for s in STAGES:
        v = validate_stage(vd / s)
        if not v['ok']:
            raise SystemExit(f'{s} invalid: {v["errors"][:5]}')
    out = pathlib.Path(args.out)
    tmp = out.with_suffix('.tmp')
    with tarfile.open(tmp, 'w:gz') as t:
        t.add(vd, arcname=args.arcname or vd.name, filter=lambda ti: None if '.staging-' in ti.name or ti.name.endswith('.tmp.npz') else ti)
    tmp.replace(out)
    h = sha256(out)
    out.with_suffix(out.suffix + '.sha256').write_text(f'{h}  {out.name}\n')
    print(json.dumps({'package': str(out), 'sha256': h, 'bytes': out.stat().st_size}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sem84')
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--image-id', required=True)
    r.add_argument('--src', required=True)
    r.add_argument('--out', required=True, help='outputs/<batch>/<image_id>/v001')
    r.add_argument('--stage', choices=STAGES, required=True)
    r.add_argument('--review')
    r.add_argument('--reference-src', help='canonical BSE TIFF of the same stem (non-BSE images)')
    r.add_argument('--reference-id')
    r.add_argument('--models', default=str(ROOT / 'models'))
    r.add_argument('--threads', type=int, default=4)
    r.add_argument('--force', action='store_true')
    r.add_argument('--full-layers', action='store_true', help='also write full-size composited layer PNGs for pass1/pass2')
    v = sp.add_parser('validate')
    v.add_argument('stage_dir')
    p = sp.add_parser('package')
    p.add_argument('--version-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arcname')
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        return run(a)
    if a.cmd == 'validate':
        res = validate_stage(a.stage_dir)
        print(json.dumps(res, indent=1))
        return 0 if res['ok'] else 1
    if a.cmd == 'package':
        return package(a)


if __name__ == '__main__':
    sys.exit(main())

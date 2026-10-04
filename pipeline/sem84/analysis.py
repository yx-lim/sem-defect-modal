"""Image analysis for the 84-feature SEM pipeline. All geometry is in native source pixels."""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree
from skimage import filters, measure, morphology

CLASS_MAP = {
    1: 'particle_contrast_A', 2: 'particle_contrast_B', 3: 'unresolved_fine_matrix',
    4: 'void_like_region', 5: 'fissure_candidate', 6: 'interfacial_gap', 7: 'unknown_inclusion',
    8: 'collector_if_verified', 9: 'exterior_or_embedding', 10: 'artifact',
    11: 'particle_unclassified_contrast', 255: 'unknown_ignore',
}
CLASS_ID = {v: k for k, v in CLASS_MAP.items()}
IGNORE = 255
QC_BITS = {'clipped_low': 1, 'clipped_high': 2, 'blur_tile': 4, 'curtain_tile': 8, 'scan_line': 16,
           'unresolved_dark': 32, 'reviewer_artifact': 64, 'reviewer_ignore': 128}
S8 = np.ones((3, 3), bool)

PASS_CFG = {
    'pass1': dict(crop=1600, overlap=320, stride=150, offset=75, score_floor=0.86, stability_floor=0.88,
                  min_area=400, max_frac=0.25, edge_margin=4, max_dark_frac=0.30, min_std=1.5, max_overlap=0.5,
                  levels=[dict(crop=2400, overlap=1500, stride=200, offset=100, min_area=3000), {}]),
    'pass2': dict(crop=1000, overlap=250, stride=90, offset=40, score_floor=0.84, stability_floor=0.86,
                  min_area=120, max_frac=0.25, edge_margin=4, max_dark_frac=0.30, min_std=1.5, max_overlap=0.16),
}


def grid(n, crop, overlap):
    if n <= crop:
        return [0]
    step = crop - overlap
    return sorted(set(list(range(0, n - crop, step)) + [n - crop]))


def valid_region(g):
    valid = np.ones(g.shape, bool)
    valid[g.std(1) < 0.5, :] = False
    valid[:, g.std(0) < 0.5] = False
    return valid


def illumination_ratio(sm, valid, step=16, win=63, gate=1.15):
    """Very-low-frequency brightness trend (~1000 px median); returns (ratio map or None, info)."""
    small = sm[::step, ::step].copy()
    vs = valid[::step, ::step]
    small[~vs] = np.median(small[vs]) if vs.any() else 128.0
    bg = ndi.median_filter(small, size=(min(win, small.shape[0] | 1), min(win, small.shape[1] | 1)), mode='reflect')
    bg = ndi.gaussian_filter(bg, 4)
    p5, p95 = np.percentile(bg[vs], [5, 95]) if vs.any() else (1.0, 1.0)
    spread = float(p95 / max(p5, 1e-3))
    info = {'method': f'median filter {win}x{step}px on 1/{step} grid + gaussian; applied if p95/p5 > {gate}',
            'bg_p5': float(p5), 'bg_p95': float(p95), 'spread': spread, 'applied': spread > gate}
    if spread <= gate:
        return None, info
    r = bg / float(np.median(bg[vs]))
    ratio = ndi.zoom(r, (sm.shape[0] / r.shape[0], sm.shape[1] / r.shape[1]), order=1)
    ratio = np.clip(ratio[:sm.shape[0], :sm.shape[1]], 0.5, 2.0).astype(np.float32)
    if ratio.shape != sm.shape:
        ratio = np.pad(ratio, ((0, sm.shape[0] - ratio.shape[0]), (0, sm.shape[1] - ratio.shape[1])), mode='edge')
    return ratio, info


def intensity_model(g, valid, is_bse):
    sm = ndi.gaussian_filter(g.astype(np.float32), 1.0)
    ratio, illum = illumination_ratio(sm, valid)
    if ratio is not None:
        sm = np.clip(sm / ratio, 0, 255).astype(np.float32)
    samp = sm[valid][::7]
    t = filters.threshold_multiotsu(samp, classes=3)
    t_void = float(t[0])
    part = samp[(samp > t_void) & (samp < t[1])]
    hist, _ = np.histogram(part, bins=256, range=(0, 256))
    mode = float(np.argmax(hist)) if part.size else float(t_void + 1)
    sigma = float(np.median(np.abs(part - np.median(part))) * 1.4826) if part.size else 5.0
    t_bright = float(max(t[1], mode + 6 * sigma)) if is_bse else None
    grad = filters.sobel(sm)
    return sm, grad, {
        't_void': t_void, 't_multiotsu': [float(x) for x in t], 'particle_mode': mode,
        'particle_sigma': sigma, 't_bright': t_bright, 'grad_median_valid': float(np.median(grad[valid])),
        'illumination_correction': illum,
        'method': 'gaussian(1px), optional low-frequency illumination flattening, 3-class multi-Otsu on valid pixels; bright threshold = max(Otsu2, mode+6*MAD) (BSE only)',
    }


# ----------------------------------------------------------------------------- SAM proposals

def sam_pass(stage, g, sm, valid, tm, sam, state, ckdir, log=print):
    base = PASS_CFG[stage]
    H, W = g.shape
    canvas, recs = state['canvas'], state['recs']
    prefix = {'pass1': 'p1', 'pass2': 'p2'}[stage]
    reasons = state.setdefault('reject_counts', {}).setdefault(stage, {})
    regions = state.setdefault('regions_done', {}).setdefault(stage, [])
    jobs = []
    for li, lvl in enumerate(base.get('levels') or [{}]):
        cfg = {**base, **lvl}
        for r, y0 in enumerate(grid(H, cfg['crop'], cfg['overlap'])):
            for c, x0 in enumerate(grid(W, cfg['crop'], cfg['overlap'])):
                jobs.append((f'L{li}r{r}c{c}', cfg, y0, x0))
    for key, cfg, y0, x0 in jobs:
        if True:
            ck = ckdir / f'{stage}_{key}.npz'
            if ck.exists():
                d = np.load(ck, allow_pickle=True)
                meta = d['meta'].item()
                for a in meta['added']:
                    y_0, x_0, y_1, x_1 = a['bbox']
                    m = np.unpackbits(d[f"m{a['label']}"])[: (y_1 - y_0) * (x_1 - x_0)].reshape(y_1 - y_0, x_1 - x_0).astype(bool)
                    canvas[y_0:y_1, x_0:x_1][m] = a['label']
                    recs[a['label']] = a['rec']
                    state['next_label'] = max(state['next_label'], a['label'] + 1)
                for k, v in meta['reasons'].items():
                    reasons[k] = reasons.get(k, 0) + v
                if key not in regions:
                    regions.append(key)
                continue
            x1, y1 = min(x0 + cfg['crop'], W), min(y0 + cfg['crop'], H)
            sub, smc = g[y0:y1, x0:x1], sm[y0:y1, x0:x1]
            cv, vd = canvas[y0:y1, x0:x1], valid[y0:y1, x0:x1]
            sp = state['suppress'][y0:y1, x0:x1] if 'suppress' in state else np.zeros(cv.shape, np.uint8)
            e = sam.embed(sub)
            h, w = sub.shape
            added, local_reasons = [], {}
            E = cfg['edge_margin']
            for yy in range(cfg['offset'], h, cfg['stride']):
                for xx in range(cfg['offset'], w, cfg['stride']):
                    if not vd[yy, xx] or cv[yy, xx] or sp[yy, xx] or smc[yy, xx] < tm['t_void']:
                        continue
                    logits, scores = sam.decode(e, xx, yy)
                    reason = 'no_valid_mask'
                    for k in np.argsort(scores)[::-1]:
                        if scores[k] < cfg['score_floor']:
                            reason = 'low_model_score'
                            break
                        a = logits[k]
                        m = a > 0
                        if not m[yy, xx]:
                            continue
                        area = int(m.sum())
                        if area < cfg['min_area'] or area > cfg['max_frac'] * h * w:
                            reason = 'area_out_of_range'
                            continue
                        stab = float((a > 1).sum() / max(1, (a > -1).sum()))
                        if stab < cfg['stability_floor']:
                            reason = 'unstable_mask'
                            continue
                        lab, _ = ndi.label(m)
                        m = lab == lab[yy, xx]
                        if m.sum() < 0.94 * area:
                            reason = 'fragmented_mask'
                            continue
                        m = ndi.binary_fill_holes(m)
                        if ((x0 > 0 and m[:, :E].any()) or (x1 < W and m[:, -E:].any()) or
                                (y0 > 0 and m[:E].any()) or (y1 < H and m[-E:].any())):
                            reason = 'crop_clipped_tile_truncation'
                            break
                        vals = smc[m]
                        if (np.median(vals) < tm['t_void'] or (vals < tm['t_void']).mean() > cfg['max_dark_frac']
                                or vals.std() < cfg['min_std']):
                            reason = 'dark_spanning_or_uniform'
                            break
                        if (cv[m] > 0).mean() > cfg['max_overlap']:
                            reason = 'duplicate_or_covered'
                            break
                        if (sp[m] > 0).mean() > 0.3:
                            reason = 'reviewer_suppressed_region'
                            break
                        newm = m & (cv == 0) & vd & (sp == 0)
                        lab2, n2 = ndi.label(newm)
                        if n2 > 1:
                            newm = lab2 == (np.argmax(np.bincount(lab2.ravel())[1:]) + 1)
                        if newm.sum() < cfg['min_area']:
                            reason = 'trimmed_too_small'
                            break
                        label = state['next_label']
                        state['next_label'] += 1
                        cv[newm] = label
                        rr, cc = np.nonzero(newm)
                        bb = [int(rr.min() + y0), int(cc.min() + x0), int(rr.max() + y0 + 1), int(cc.max() + x0 + 1)]
                        rec = {'label': label, 'id': f'{prefix}-{label:05d}', 'pass_added': stage,
                               'seed': [int(x0 + xx), int(y0 + yy)], 'sam_score': float(scores[k]),
                               'sam_stability': stab, 'overlap_trimmed_px': int(m.sum() - newm.sum()),
                               'source_region': key}
                        recs[label] = rec
                        added.append({'label': label, 'bbox': bb, 'rec': rec})
                        reason = 'accepted'
                        break
                    local_reasons[reason] = local_reasons.get(reason, 0) + 1
            arrays = {}
            for a in added:
                y_0, x_0, y_1, x_1 = a['bbox']
                arrays[f"m{a['label']}"] = np.packbits(canvas[y_0:y_1, x_0:x_1] == a['label'])
            tmp = ck.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, meta=np.array({'added': added, 'reasons': local_reasons}, dtype=object), **arrays)
            tmp.replace(ck)
            for k, v in local_reasons.items():
                reasons[k] = reasons.get(k, 0) + v
            regions.append(key)
            log(f'{stage} region {key} accepted={len(added)} total={len(recs)}')
    return state


SEED_MAX_FRAC = 0.15
SEED_MIN_SOLIDITY = 0.7


def seed_instance(seed, g, sm, valid, tm, sam, state, reason, reviewer):
    """Reviewer-requested SAM proposal around a native-pixel seed."""
    H, W = g.shape
    x, y = int(seed[0]), int(seed[1])
    half = 512
    x0, y0 = max(0, min(W - 2 * half, x - half)), max(0, min(H - 2 * half, y - half))
    x1, y1 = min(W, x0 + 2 * half), min(H, y0 + 2 * half)
    e = sam.embed(g[y0:y1, x0:x1])
    logits, scores = sam.decode(e, x - x0, y - y0)
    cv = state['canvas'][y0:y1, x0:x1]
    if not (0 <= y < H and 0 <= x < W) or cv[y - y0, x - x0]:
        return None, 'seed outside image or on an existing instance'
    cap = SEED_MAX_FRAC * (y1 - y0) * (x1 - x0)
    why = 'no mask contains the seed'
    for k in np.argsort(scores)[::-1]:
        m = logits[k] > 0
        if not m[y - y0, x - x0]:
            continue
        lab, _ = ndi.label(m)
        m = ndi.binary_fill_holes(lab == lab[y - y0, x - x0]) & (cv == 0) & valid[y0:y1, x0:x1]
        lab, _ = ndi.label(m)
        if not lab[y - y0, x - x0]:
            continue
        m = lab == lab[y - y0, x - x0]
        a = int(m.sum())
        if a < 50:
            why = f'mask too small ({a}px)'
            continue
        if a > cap:
            why = f'mask too large ({a}px > {int(cap)}px; likely leak)'
            continue
        sol = float(measure.regionprops(m.astype(np.uint8))[0].solidity)
        if sol < SEED_MIN_SOLIDITY:
            why = f'mask solidity {sol:.2f} < {SEED_MIN_SOLIDITY} (likely spans several bodies)'
            continue
        label = state['next_label']
        state['next_label'] += 1
        cv[m] = label
        state['recs'][label] = {'label': label, 'id': f'rv-{label:05d}', 'pass_added': 'review_seed',
                                'seed': [x, y], 'sam_score': float(scores[k]), 'sam_stability': None,
                                'overlap_trimmed_px': 0, 'review_reason': reason, 'reviewer': reviewer,
                                'seed_mask_solidity': sol}
        return state['recs'][label]['id'], 'accepted'
    return None, why


# ----------------------------------------------------------------------------- geometry

def _angle_deg(orientation):
    # skimage orientation: angle between row axis and major axis. Report degrees from +x toward +y (y down).
    dx, dy = math.sin(orientation), math.cos(orientation)
    return float(math.degrees(math.atan2(dy, dx)) % 180.0)


def _polygon(m, off_r, off_c, tol):
    cs = measure.find_contours(np.pad(m, 1).astype(np.float32), 0.5)
    if not cs:
        return []
    c = max(cs, key=len)
    c = measure.approximate_polygon(c, tol) if tol else c
    return [[float(p[1] - 1 + off_c), float(p[0] - 1 + off_r)] for p in c]


def _ferets(pts):
    pts = np.asarray(pts)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        return 0.0, 0.0, 0.0
    ang = np.deg2rad(np.arange(180))
    proj = hull @ np.stack([np.cos(ang), np.sin(ang)])
    widths = proj.max(0) - proj.min(0)
    hp = float(np.sum(np.linalg.norm(np.roll(hull, -1, 0) - hull, axis=1)))
    return float(widths.max()), float(widths.min()), hp


def instance_geometry(canvas, recs, sm, grad, valid, tm):
    H, W = canvas.shape
    objs = ndi.find_objects(canvas)
    gmed = max(tm['grad_median_valid'], 1e-6)
    out = []
    for label in sorted(recs):
        sl = objs[label - 1] if label - 1 < len(objs) else None
        if sl is None:
            continue
        m = canvas[sl] == label
        if not m.any():
            continue
        r0, c0 = sl[0].start, sl[1].start
        rp = measure.regionprops(m.astype(np.uint8))[0]
        contour = _polygon(m, r0, c0, 0)
        fmax, fmin, hull_perim = _ferets(contour)
        perim = float(rp.perimeter) or 1.0
        facets = _polygon(m, r0, c0, 3.0)
        fl = [math.dist(facets[i], facets[i + 1]) for i in range(len(facets) - 1)] if len(facets) > 1 else []
        bnd = m & ~ndi.binary_erosion(m)
        vals = sm[sl][m]
        support = float(np.median(grad[sl][bnd]) / gmed) if bnd.any() else 0.0
        trunc = bool(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W or
                     (~valid[sl][ndi.binary_dilation(m)]).any())
        rec = recs[label]
        area = int(rp.area)
        stab = rec.get('sam_stability') or 0.9
        conf = float(np.clip(0.3 * (rec.get('sam_score', 0.85) - 0.8) / 0.2 + 0.3 * stab + 0.4 * min(1.0, support / 3.0), 0, 1))
        d = dict(rec)
        d.update({
            'bbox_xyxy': [int(c0), int(r0), int(sl[1].stop), int(sl[0].stop)],
            'centroid_xy': [float(rp.centroid[1] + c0), float(rp.centroid[0] + r0)],
            'area_px': area, 'perimeter_px': perim, 'equivalent_diameter_px': float(rp.equivalent_diameter_area),
            'major_axis_px': float(rp.axis_major_length), 'minor_axis_px': float(rp.axis_minor_length),
            'aspect_ratio': float(rp.axis_major_length / max(rp.axis_minor_length, 1e-6)),
            'feret_max_px': fmax, 'feret_min_px': fmin, 'orientation_deg': _angle_deg(rp.orientation),
            'circularity': float(min(1.0, 4 * math.pi * area / perim ** 2)), 'solidity': float(rp.solidity),
            'convexity': float(min(1.0, hull_perim / perim)) if hull_perim else None,
            'boundary_roughness': float(perim / hull_perim) if hull_perim else None,
            'facet_count': len(fl), 'median_facet_length_px': float(np.median(fl)) if fl else None,
            'median_intensity': float(np.median(vals)), 'intensity_iqr': float(np.subtract(*np.percentile(vals, [75, 25]))),
            'dark_fraction': float((vals < tm['t_void']).mean()), 'boundary_gradient_support': support,
            'frame_truncated': trunc, 'heuristic_confidence': conf,
            'polygon_xy': _polygon(m, r0, c0, 1.0),
        })
        out.append(d)
    return out


def classify_instances(inst, tm, is_bse, review_relabels):
    areas = np.array([i['area_px'] for i in inst]) if inst else np.array([0])
    fine_cut = float(np.percentile(areas, 10)) if len(inst) >= 10 else 0.0
    for i in inst:
        if is_bse:
            cls = 'particle_contrast_B' if i['median_intensity'] > tm['t_bright'] else 'particle_contrast_A'
        else:
            cls = 'particle_unclassified_contrast'
        tags = []
        if cls == 'particle_contrast_B':
            tags.append('bright_phase_candidate')
        if i['aspect_ratio'] > 8 and i['minor_axis_px'] < 15:
            tags.append('fibre_like_shape')
            cls = 'unknown_inclusion'
        if i['area_px'] <= fine_cut:
            tags.append('fine_relative_lower_decile')
        if i['frame_truncated']:
            tags.append('frame_truncated')
        if i['boundary_gradient_support'] < 1.0:
            tags.append('low_boundary_support')
        if i['id'] in review_relabels:
            cls = review_relabels[i['id']]['class']
            tags.append('reviewer_relabelled')
        i['class'] = cls
        i['tags'] = tags
        i['phase_identity_evidence'] = ('BSE contrast only; chemistry unconfirmed' if is_bse else
                                        'non-BSE detector: surface/relief contrast; composition not assessed')
        i['observed_vs_inferred'] = 'observed_2d_section_boundary'
        i['confidence_label'] = 'high' if i['heuristic_confidence'] > 0.75 else 'medium' if i['heuristic_confidence'] > 0.5 else 'low'
    return fine_cut


# ----------------------------------------------------------------------------- dark structures

def dark_structures(sm, grad, valid, canvas, tm, inst, review_dark):
    H, W = sm.shape
    dark = (sm < tm['t_void']) & valid
    dark = morphology.remove_small_objects(dark, 12)
    lab, n = ndi.label(dark, structure=S8)
    if n == 0:
        return lab, [], np.zeros_like(dark), np.zeros_like(dark)
    gmed = max(tm['grad_median_valid'], 1e-6)
    bnd = dark & ~ndi.binary_erosion(dark)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    bsupport = ndi.median(grad, labels=np.where(bnd, lab, 0), index=idx) / gmed
    minint = ndi.minimum(sm, labels=lab, index=idx)
    inside = np.bincount(lab[canvas > 0], minlength=n + 1)
    skel = morphology.skeletonize(dark)
    dt = ndi.distance_transform_edt(dark)
    skl = np.bincount(lab[skel], minlength=n + 1)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - 1
    endpoints = np.bincount(lab[skel & (nb == 1)], minlength=n + 1)
    branches = np.bincount(lab[skel & (nb >= 3)], minlength=n + 1)
    width_on_skel = 2 * dt[skel]
    lab_on_skel = lab[skel]
    order = np.argsort(lab_on_skel)
    ls, ws = lab_on_skel[order], width_on_skel[order]
    starts = np.searchsorted(ls, idx)
    ends = np.searchsorted(ls, idx, side='right')
    maxdt = ndi.maximum(dt, labels=lab, index=idx)
    # host (mode of instance label under the component) and neighbours in a 3-px ring
    both = (lab > 0) & (canvas > 0)
    pairs = np.unique(lab[both].astype(np.int64) * 70000 + canvas[both], return_counts=True)
    host = {}
    for key, cnt in zip(*pairs):
        L, I = int(key // 70000), int(key % 70000)
        if cnt > host.get(L, (0, 0))[1]:
            host[L] = (I, int(cnt))
    dil = ndi.grey_dilation(lab, size=(7, 7))
    ring = (lab == 0) & (dil > 0) & (canvas > 0)
    neigh = {}
    for key in np.unique(dil[ring].astype(np.int64) * 70000 + canvas[ring]):
        neigh.setdefault(int(key // 70000), set()).add(int(key % 70000))
    # ring around each dark component: how much of it is the host particle itself?
    ring_all = (lab == 0) & (dil > 0)
    Lr, Cr = dil[ring_all], canvas[ring_all]
    host_arr = np.zeros(n + 1, np.int64)
    for L, (I, _) in host.items():
        host_arr[L] = I
    ring_tot = np.bincount(Lr, minlength=n + 1)
    ring_host = np.bincount(Lr[(Cr == host_arr[Lr]) & (Cr > 0)], minlength=n + 1)
    rps = {rp.label: rp for rp in measure.regionprops(lab)}
    id_of = {i['label']: i['id'] for i in inst}
    host_sol = {i['label']: i.get('solidity', 0.0) for i in inst}
    host_area = {i['label']: i['area_px'] for i in inst}
    eq = np.array([i['equivalent_diameter_px'] for i in inst]) if inst else np.array([50.0])
    med_eq, med_area = float(np.median(eq)), float(np.median([i['area_px'] for i in inst])) if inst else 2000.0
    comps = []
    supported = np.zeros(n + 1, bool)
    for L in idx:
        rp = rps[L]
        a = int(area[L])
        sk = int(skl[L])
        w = ws[starts[L - 1]:ends[L - 1]]
        mean_w = a / max(sk, 1)
        elong = sk / max(rp.equivalent_diameter_area, 1)
        inside_frac = inside[L] / a
        nbs = sorted(neigh.get(L, set()))
        h = host.get(L, (0, 0))[0]
        bs = float(bsupport[L - 1])
        ring_host_frac = float(ring_host[L] / max(ring_tot[L], 1))
        host_ok = bool(h) and host_sol.get(h, 0.0) >= 0.85 and host_area.get(h, 0) <= 10 * med_area
        r0, c0, r1, c1 = rp.bbox
        touches_edge = r0 == 0 or c0 == 0 or r1 == H or c1 == W
        if bs < 1.2 and minint[L - 1] > 0.5 * tm['t_void']:
            cls = 'unresolved_dark'
        elif inside_frac >= 0.7 and host_ok and mean_w <= 8 and elong >= 3 and ring_host_frac >= 0.85:
            cls = 'fissure_candidate'
        elif inside_frac < 0.3 and len(nbs) >= 2 and mean_w <= 6 and elong >= 4:
            cls = 'interfacial_gap'
        else:
            cls = 'void_like_region'
        tags = []
        if inside_frac >= 0.7 and h and ring_host_frac < 0.85:
            tags.append('host_mask_overlaps_boundary_gap')
        if inside_frac >= 0.7 and h and not host_ok:
            tags.append('host_mask_low_solidity_or_oversized')
        if cls != 'unresolved_dark' and elong >= 6 and sk >= 3 * med_eq and len(nbs) >= 3 and mean_w <= 10 and endpoints[L] <= 8:
            tags.append('interparticle_crack_candidate')
        if cls == 'void_like_region' and a >= med_area and rp.solidity > 0.85 and inside_frac < 0.3:
            tags.append('particle_shaped_cavity_candidate')
        if minint[L - 1] <= 0:
            tags.append('contains_clipped_low_pixels')
        if touches_edge:
            tags.append('frame_truncated')
        cid = f'dk-{L:05d}'
        if cid in review_dark:
            cls = review_dark[cid]['class']
            tags.append('reviewer_reclassified')
        if inside_frac >= 0.7 and ring_host_frac >= 0.85:
            loc = 'intraparticle'
        elif len(nbs) >= 2:
            loc = 'interparticle'
        elif len(nbs) == 1:
            loc = 'particle_boundary'
        else:
            loc = 'unresolved_location'
        if cls != 'unresolved_dark':
            supported[L] = True
        comps.append({
            'label': int(L), 'id': cid, 'class': cls, 'tags': tags, 'area_px': a,
            'bbox_xyxy': [int(c0), int(r0), int(c1), int(r1)],
            'centroid_xy': [float(rp.centroid[1]), float(rp.centroid[0])],
            'equivalent_diameter_px': float(rp.equivalent_diameter_area),
            'max_inscribed_diameter_px': float(2 * maxdt[L - 1]),
            'major_axis_px': float(rp.axis_major_length), 'minor_axis_px': float(rp.axis_minor_length),
            'aspect_ratio': float(rp.axis_major_length / max(rp.axis_minor_length, 1e-6)),
            'orientation_deg': _angle_deg(rp.orientation), 'solidity': float(rp.solidity),
            'circularity': float(min(1.0, 4 * math.pi * a / max(rp.perimeter, 1) ** 2)),
            'skeleton_length_px': sk, 'mean_width_px': float(mean_w),
            'width_profile_px': {'median': float(np.median(w)) if w.size else None,
                                 'p90': float(np.percentile(w, 90)) if w.size else None,
                                 'max': float(w.max()) if w.size else None,
                                 'min_throat': float(w[w > 1].min()) if (w > 1).any() else None},
            'endpoints': int(endpoints[L]), 'branch_points': int(branches[L]),
            'boundary_support_ratio': bs, 'min_intensity': float(minint[L - 1]),
            'inside_instance_fraction': float(inside_frac), 'ring_host_fraction': ring_host_frac, 'host_instance': id_of.get(h) if h else None,
            'neighbour_instances': [id_of[x] for x in nbs if x in id_of], 'location_2d': loc,
            'shape_class': ('slit' if rp.axis_major_length / max(rp.axis_minor_length, 1e-6) > 4 and mean_w <= 6 else
                            'rounded' if rp.solidity > 0.85 and rp.axis_major_length / max(rp.axis_minor_length, 1e-6) < 2 else 'elongated_irregular'),
            'alternatives': (['preparation_fracture_candidate', 'intraparticle_slit_pore', 'lamellar_edge_shadow']
                             if cls == 'fissure_candidate' else
                             ['pore', 'shadow', 'milling_artifact', 'unresolved_matrix'] if cls == 'unresolved_dark' else
                             ['pore', 'interfacial_gap', 'preparation_pull_out'] if cls == 'void_like_region' else
                             ['interparticle_pore_slit', 'preparation_fracture_candidate']),
            'observed_vs_inferred': 'observed_dark_region_2d' if cls != 'unresolved_dark' else 'unresolved',
            'confidence_label': 'medium' if bs >= 2 else 'low',
        })
    supported_mask = supported[lab]
    unresolved_mask = dark & ~supported_mask
    return lab, comps, supported_mask, unresolved_mask


# ----------------------------------------------------------------------------- relations

def relations(canvas, inst, supported_dark, matrix):
    objs = ndi.find_objects(canvas)
    H, W = canvas.shape
    by_label = {i['label']: i for i in inst}
    rels, seen = [], set()
    for i in inst:
        L = i['label']
        sl = objs[L - 1]
        pad = 14
        r0, r1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
        c0, c1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
        sub = canvas[r0:r1, c0:c1]
        m = sub == L
        bnd = m & ~ndi.binary_erosion(m)
        d2 = ndi.binary_dilation(m, iterations=2)
        ring2 = d2 & ~m
        touch_labels = set(np.unique(sub[ring2])) - {0, L}
        d6 = ndi.binary_dilation(d2, iterations=4)
        ring6 = d6 & ~m
        sd = supported_dark[r0:r1, c0:c1]
        i['ring_void_fraction'] = float(sd[ring6].mean()) if ring6.any() else None
        i['matrix_adjacent_perimeter_fraction'] = float(matrix[r0:r1, c0:c1][ring2].mean()) if ring2.any() else None
        d12 = ndi.binary_dilation(d6, iterations=6)
        near = set(np.unique(sub[d12 & ~m])) - {0, L}
        i['touch_count'] = 0
        for N in near:
            if N not in by_label:
                continue
            key = (min(L, N), max(L, N))
            other = sub == N
            if N in touch_labels:
                cl = bnd & ndi.binary_dilation(other, iterations=2)
                length = int(cl.sum())
                if length >= 6:
                    i['touch_count'] += 1
                    if key in seen:
                        continue
                    seen.add(key)
                    rr, cc = np.nonzero(cl)
                    rels.append({'type': 'touches', 'a': i['id'], 'b': by_label[N]['id'],
                                 'apparent_contact_length_px': length,
                                 'point_xy': [float(cc.mean() + c0), float(rr.mean() + r0)],
                                 'state': 'observed_2d', 'note': 'touching contour; not proof of electrical contact'})
                    continue
            if key in seen:
                continue
            seen.add(key)
            between = d12 & ndi.binary_dilation(other, iterations=12)
            rr, cc = np.nonzero(between)
            if not rr.size:
                continue
            gap = bool((between & sd).sum() >= 10)
            rels.append({'type': 'separated_by_gap' if gap else 'adjacent_to', 'a': i['id'], 'b': by_label[N]['id'],
                         'point_xy': [float(cc.mean() + c0), float(rr.mean() + r0)],
                         'state': 'observed_2d' if gap else 'candidate', 'max_separation_px': 24})
    # touches are counted on both sides; recount from relation list for consistency
    deg = {}
    for r in rels:
        if r['type'] == 'touches':
            deg[r['a']] = deg.get(r['a'], 0) + 1
            deg[r['b']] = deg.get(r['b'], 0) + 1
    for i in inst:
        i['touch_count'] = deg.get(i['id'], 0)
    return rels


def contrast_clusters(inst, dist_factor=1.0):
    pts = [(i['centroid_xy'], i) for i in inst if i['class'] == 'particle_contrast_B']
    if len(pts) < 3:
        return []
    xy = np.array([p[0] for p in pts])
    eq = np.median([p[1]['equivalent_diameter_px'] for p in pts])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(dist_factor * eq * 2)
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for k in range(len(pts)):
        groups.setdefault(find(k), []).append(k)
    out = []
    for n, members in enumerate(g for g in groups.values() if len(g) >= 3):
        mxy = xy[members]
        try:
            hull = mxy[ConvexHull(mxy).vertices].tolist()
        except Exception:
            hull = mxy.tolist()
        out.append({'id': f'cl-{n + 1:04d}', 'members': [pts[k][1]['id'] for k in members],
                    'envelope_xy_visual_only': hull, 'linkage_distance_px': float(dist_factor * eq * 2),
                    'state': 'candidate', 'note': 'proximity cluster of contrast_B candidates; not proof of agglomerate; hull is a visual envelope, not a mask'})
    return out


def enclosure_candidates(canvas, inst):
    """2D boundary state of contrast_B candidates relative to surrounding contrast_A instances."""
    by_label = {i['label']: i for i in inst}
    objs = ndi.find_objects(canvas)
    out = []
    for i in inst:
        if i['class'] != 'particle_contrast_B':
            continue
        sl = objs[i['label'] - 1]
        r0, c0 = max(0, sl[0].start - 4), max(0, sl[1].start - 4)
        sub = canvas[r0:sl[0].stop + 4, c0:sl[1].stop + 4]
        m = sub == i['label']
        ring = ndi.binary_dilation(m, iterations=3) & ~m
        labs = sub[ring]
        a_frac = float(np.mean([by_label.get(int(x), {}).get('class') == 'particle_contrast_A' for x in labs])) if labs.size else 0.0
        state = 'closed_2d_candidate' if a_frac > 0.95 else 'partial_2d_candidate' if a_frac > 0.3 else 'not_enclosed_2d'
        out.append({'type': 'inside' if state == 'closed_2d_candidate' else 'adjacent_to', 'a': i['id'],
                    'boundary_contrast_A_fraction': a_frac, 'enclosure_2d_state': state, 'state': 'candidate',
                    'note': 'chemistry unconfirmed; 2D section cannot establish 3D enclosure'})
    return out


# ----------------------------------------------------------------------------- QC and statistics

def qc_maps(g, sm, valid, canvas, inst, tm):
    H, W = g.shape
    qc = {}
    low, high = (g <= 0) & valid, (g >= 255) & valid
    qc['clipped_low_fraction'] = float(low.sum() / valid.sum())
    qc['clipped_high_fraction'] = float(high.sum() / valid.sum())
    T = 256
    lap = ndi.laplace(sm)
    tiles = []
    for y in range(0, H, T):
        for x in range(0, W, T):
            v = valid[y:y + T, x:x + T]
            if v.mean() < 0.5:
                continue
            solid = v & (sm[y:y + T, x:x + T] >= tm['t_void'])
            if solid.mean() < 0.3:
                continue
            tiles.append((x, y, float(lap[y:y + T, x:x + T][solid].var())))
    lv = np.array([t[2] for t in tiles]) if tiles else np.array([1.0])
    med = float(np.median(lv))
    blur = [[t[0], t[1], min(t[0] + T, W), min(t[1] + T, H)] for t in tiles if t[2] < 0.25 * med]
    qc['blur'] = {'tile_px': T, 'laplacian_variance_median': med, 'low_sharpness_tiles': blur,
                  'method': 'tiles with Laplacian variance < 0.25x image median (solid pixels only)'}
    # directional streak analysis inside eroded particle interiors
    hp = sm - ndi.gaussian_filter(sm, 4)
    gy, gx = np.gradient(hp)
    interior = ndi.binary_erosion(canvas > 0, iterations=6) & valid
    jxx, jyy, jxy = (gx * gx)[interior].sum(), (gy * gy)[interior].sum(), (gx * gy)[interior].sum()
    coh = float(math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(jxx + jyy, 1e-9))
    grad_dir = 0.5 * math.degrees(math.atan2(2 * jxy, jxx - jyy))
    streak_dir = (grad_dir + 90) % 180
    per = []
    for i in inst:
        x0, y0, x1, y1 = i['bbox_xyxy']
        m = interior[y0:y1, x0:x1] & (canvas[y0:y1, x0:x1] == i['label'])
        if m.sum() < 200:
            i['texture'] = None
            continue
        a, b, c = (gx[y0:y1, x0:x1] ** 2)[m].sum(), (gy[y0:y1, x0:x1] ** 2)[m].sum(), (gx[y0:y1, x0:x1] * gy[y0:y1, x0:x1])[m].sum()
        ch = math.sqrt((a - b) ** 2 + 4 * c ** 2) / max(a + b, 1e-9)
        d = (0.5 * math.degrees(math.atan2(2 * c, a - b)) + 90) % 180
        i['texture'] = {'coherence': float(ch), 'streak_direction_deg': float(d)}
        per.append((ch, d))
    if per:
        ang = np.deg2rad(np.array([p[1] for p in per]) * 2)
        wts = np.array([p[0] for p in per])
        R = float(np.hypot((wts * np.cos(ang)).sum(), (wts * np.sin(ang)).sum()) / max(wts.sum(), 1e-9))
    else:
        R = 0.0
    curtain_tiles = []
    if coh > 0.15 and R > 0.6:
        for y in range(0, H, T):
            for x in range(0, W, T):
                m = interior[y:y + T, x:x + T]
                if m.sum() < 2000:
                    continue
                a, b, c = (gx[y:y + T, x:x + T] ** 2)[m].sum(), (gy[y:y + T, x:x + T] ** 2)[m].sum(), (gx[y:y + T, x:x + T] * gy[y:y + T, x:x + T])[m].sum()
                ch = math.sqrt((a - b) ** 2 + 4 * c ** 2) / max(a + b, 1e-9)
                d = (0.5 * math.degrees(math.atan2(2 * c, a - b)) + 90) % 180
                if ch > 0.25 and min(abs(d - streak_dir), 180 - abs(d - streak_dir)) < 15:
                    curtain_tiles.append([x, y, min(x + T, W), min(y + T, H)])
    qc['directional_streaks'] = {'global_coherence': coh, 'global_streak_direction_deg': float(streak_dir),
                                 'cross_particle_alignment_R': R, 'aligned_tiles': curtain_tiles,
                                 'method': 'structure tensor of high-pass image inside eroded particle interiors; aligned across particles => preparation/scan streak candidate'}
    rows = np.array([g[r][valid[r]].mean() if valid[r].any() else np.nan for r in range(H)])
    cols = np.array([g[:, c][valid[:, c]].mean() if valid[:, c].any() else np.nan for c in range(W)])

    def jumps(p):
        p = np.nan_to_num(p, nan=np.nanmedian(p))
        res = p - ndi.median_filter(p, 51)
        mad = np.median(np.abs(res)) * 1.4826 + 1e-6
        return set(int(k) for k in np.nonzero((np.abs(res) > 6 * mad) & (np.abs(res) > 3))[0])

    def prof(axis, sl):
        sub, vs = (g[:, sl], valid[:, sl]) if axis == 1 else (g[sl], valid[sl])
        num = (sub.astype(np.float64) * vs).sum(axis)
        return np.where(vs.sum(axis) > 0, num / np.maximum(vs.sum(axis), 1), np.nan)

    def consistent(a, b):
        return sorted(k for k in a if {k - 1, k, k + 1} & b)
    qc['scan_line_rows'] = consistent(jumps(prof(1, slice(0, W // 2))), jumps(prof(1, slice(W // 2, W))))
    qc['seam_columns'] = consistent(jumps(prof(0, slice(0, H // 2))), jumps(prof(0, slice(H // 2, H))))
    qc['line_jump_method'] = 'row/column mean residual vs 51-px median > max(6 MAD, 3 grey levels), required in both image halves'
    # effective resolution estimate from radial power spectrum of a central window
    s = min(1024, H, W)
    cy, cx = H // 2 - s // 2, W // 2 - s // 2
    win = g[cy:cy + s, cx:cx + s].astype(np.float32)
    win = (win - win.mean()) * np.outer(np.hanning(s), np.hanning(s))
    ps = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    yy, xx = np.indices(ps.shape)
    rr = np.hypot(yy - s / 2, xx - s / 2).astype(int)
    radial = np.bincount(rr.ravel(), ps.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    radial = radial[1:s // 2]
    floor = np.median(radial[int(0.8 * len(radial)):])
    above = np.nonzero(radial > 4 * floor)[0]
    fc = (above.max() + 1) / s if above.size else 0.5
    qc['effective_resolution_limit_px'] = float(round(1 / fc, 2))
    qc['effective_resolution_method'] = 'radial power spectrum of central 1024px window; cutoff where power falls to 4x high-frequency floor (approximate; TIFF pitch is not resolution)'
    # exterior/embedding candidates: large low-texture regions touching the frame
    loc_std = np.sqrt(np.maximum(ndi.uniform_filter(sm ** 2, 15) - ndi.uniform_filter(sm, 15) ** 2, 0))
    flat = (loc_std < 1.5) & valid & (canvas == 0)
    fl, nf = ndi.label(ndi.binary_opening(flat, iterations=3))
    ext = []
    if nf:
        sizes = np.bincount(fl.ravel())
        for L in np.nonzero(sizes > 0.01 * H * W)[0]:
            if L == 0:
                continue
            rr_, cc_ = np.nonzero(fl == L)
            if rr_.min() == 0 or cc_.min() == 0 or rr_.max() == H - 1 or cc_.max() == W - 1:
                ext.append({'bbox_xyxy': [int(cc_.min()), int(rr_.min()), int(cc_.max() + 1), int(rr_.max() + 1)],
                            'area_px': int(sizes[L]), 'state': 'candidate'})
    qc['exterior_candidates'] = ext
    return qc, low, high


def matrix_candidates(sm, valid, canvas, supported_dark, unresolved_dark, tm, inst):
    loc_std = np.sqrt(np.maximum(ndi.uniform_filter(sm ** 2, 5) - ndi.uniform_filter(sm, 5) ** 2, 0))
    interior = ndi.binary_erosion(canvas > 0, iterations=4)
    ref = float(np.median(loc_std[interior])) if interior.any() else 2.0
    cand = valid & (canvas == 0) & ~supported_dark & ~unresolved_dark & (loc_std > 1.5 * ref) & (sm >= tm['t_void'])
    cand = morphology.remove_small_objects(ndi.binary_opening(cand), 30)
    return cand, ref


def windowed(mask, valid, win):
    H, W = mask.shape
    vals, grid_ = [], np.full((math.ceil(H / win), math.ceil(W / win)), np.nan)
    for a, y in enumerate(range(0, H, win)):
        for b, x in enumerate(range(0, W, win)):
            v = valid[y:y + win, x:x + win]
            if v.size and v.mean() >= 0.8:
                f = float(mask[y:y + win, x:x + win][v].mean())
                grid_[a, b] = f
                vals.append(f)
    vals = np.array(vals)
    return {'window_px': win, 'n_windows': int(vals.size), 'mean': float(vals.mean()) if vals.size else None,
            'std': float(vals.std()) if vals.size else None,
            'cv': float(vals.std() / vals.mean()) if vals.size and vals.mean() > 0 else None}, grid_


def clark_evans(xy, area):
    if len(xy) < 5:
        return None
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    return {'n': len(xy), 'mean_nn_px': float(nn.mean()), 'R': float(nn.mean() / (0.5 * math.sqrt(area / len(xy)))),
            'note': 'no edge correction; R<1 clustered, R>1 dispersed in this 2D section'}


def chord_lengths(mask):
    out = {}
    for axis, name in ((1, 'x'), (0, 'y')):
        m = mask if axis == 1 else mask.T
        d = np.diff(np.pad(m.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        st, en = np.nonzero(d == 1), np.nonzero(d == -1)
        L = en[1] - st[1]
        out[name] = {'n': int(L.size), 'mean_px': float(L.mean()) if L.size else None,
                     'median_px': float(np.median(L)) if L.size else None}
    return out


def q(arr, ps=(5, 25, 50, 75, 95)):
    arr = np.asarray([a for a in arr if a is not None], float)
    if not arr.size:
        return None
    d = {f'p{p}': float(np.percentile(arr, p)) for p in ps}
    d.update({'n': int(arr.size), 'mean': float(arr.mean()), 'min': float(arr.min()), 'max': float(arr.max())})
    return d


def orientation_stats(angles, weights=None):
    if not len(angles):
        return None
    a = np.deg2rad(np.asarray(angles) * 2)
    w = np.ones(len(a)) if weights is None else np.asarray(weights)
    C, S = (w * np.cos(a)).sum() / w.sum(), (w * np.sin(a)).sum() / w.sum()
    R = math.hypot(C, S)
    t = np.deg2rad(np.asarray(angles))
    T = np.array([[np.mean(np.cos(t) ** 2), np.mean(np.cos(t) * np.sin(t))],
                  [np.mean(np.cos(t) * np.sin(t)), np.mean(np.sin(t) ** 2)]])
    return {'mean_direction_deg': float(math.degrees(math.atan2(S, C)) / 2 % 180), 'alignment_R': float(R),
            'alignment_tensor_2d': T.tolist(), 'n': int(len(a)),
            'convention': 'degrees from +x toward +y in image coordinates (y down), axial [0,180)'}

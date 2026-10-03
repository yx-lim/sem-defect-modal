"""Classical proposal generators + dedupe (SPEC §3 propose)."""

from __future__ import annotations

import hashlib

import numpy as np
from skimage import filters, morphology, measure

from .contract import Proposal

PRIORITY = ["anomaly_peak", "tophat_crack", "dark_void", "edge_band", "fft_curtain", "random"]
CAP_PER_IMAGE = 60


def proposal_id(image_id: str, bbox: tuple[int, int, int, int], source: str) -> str:
    s = f"{image_id}|{','.join(map(str, bbox))}|{source}"
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _mk(image_id, group_id, batch, detector, bbox, source, score, run_id, mask=None):
    return Proposal(
        proposal_id=proposal_id(image_id, bbox, source),
        image_id=image_id,
        group_id=group_id,
        batch=batch,
        detector=detector,
        bbox=tuple(int(v) for v in bbox),
        mask_rle=mask,
        source=source,
        score=float(score),
        run_id=run_id,
    )


def tophat_crack(img: np.ndarray, sigma: float = 1.0, radius: int = 7,
                 pct: float = 99.5, min_skel: int = 30, min_aspect: float = 4.0):
    """Black-tophat dark linear structures."""
    from scipy import ndimage as ndi
    from skimage.morphology import black_tophat, disk, skeletonize

    sm = filters.gaussian(img.astype(np.float32), sigma=sigma, preserve_range=True)
    th = black_tophat(sm, footprint=disk(radius))
    thr = np.percentile(th, pct)
    m = th > thr
    out = []
    for region in measure.regionprops(measure.label(m)):
        sk = skeletonize(region.image)
        skel_len = int(sk.sum())
        if skel_len < min_skel:
            continue
        major = region.axis_major_length or 1.0
        minor = max(region.axis_minor_length, 1.0)
        if major / minor < min_aspect:
            continue
        y0, x0, y1, x1 = region.bbox
        score = float(th[y0:y1, x0:x1][region.image].max())
        out.append(((x0, y0, x1, y1), score, sk))
    return out, th


def dark_void(img: np.ndarray, min_area: int = 400, area_pct: float = 99.0):
    """Multi-Otsu 3 classes; lowest-class components unusually large."""
    try:
        t = filters.threshold_multiotsu(img, classes=3)
    except ValueError:
        return [], None
    dark = img <= t[0]
    lab = measure.label(dark)
    comps = [r for r in measure.regionprops(lab) if r.area >= min_area]
    if not comps:
        return [], t
    areas = np.array([r.area for r in comps])
    a_thr = np.percentile(areas, area_pct)
    out = [((r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]), float(r.area), r.image)
           for r in comps if r.area > a_thr]
    return out, t


def fft_curtain(img: np.ndarray, tile: int = 512, band: int = 2, pct: float = 95.0):
    """Per-tile ratio of spectral energy in |fy|<2 band (vertical stripes), excl DC."""
    h, w = img.shape
    ratios = []
    boxes = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            t = img[y : y + tile, x : x + tile].astype(np.float32)
            F = np.fft.fftshift(np.abs(np.fft.fft2(t)))
            cy, cx = tile // 2, tile // 2
            bandmask = np.zeros_like(F, dtype=bool)
            bandmask[cy - band : cy + band + 1, :] = True
            bandmask[cy, cx] = False  # exclude DC
            total = F.sum() - F[cy, cx]
            ratio = float(F[bandmask].sum() / total) if total > 0 else 0.0
            ratios.append(ratio)
            boxes.append((x, y, x + tile, y + tile))
    if not ratios:
        return [], np.array([])
    ratios = np.array(ratios)
    thr = np.percentile(ratios, pct)
    out = [(b, r) for b, r in zip(boxes, ratios) if r > thr]
    return out, ratios


def edge_band(img: np.ndarray, band: int = 64, k: float = 3.0):
    """Outer `band`-px bands brighter than interior mean + k*MAD-sigma."""
    h, w = img.shape
    interior = img[band : h - band, band : w - band].astype(np.float32)
    med = np.median(interior)
    mad = np.median(np.abs(interior - med)) * 1.4826
    thr = interior.mean() + k * max(mad, 1e-9)
    out = []
    bands = {
        "top": (0, 0, w, band),
        "bottom": (0, h - band, w, h),
        "left": (0, band, band, h - band),
        "right": (w - band, band, w, h - band),
    }
    for name, (x0, y0, x1, y1) in bands.items():
        b = img[y0:y1, x0:x1].astype(np.float32)
        m = float(b.mean())
        if m > thr:
            out.append(((x0, y0, x1, y1), m, name))
    return out, thr


def anomaly_peaks(heatmap: np.ndarray, threshold: float, top_n: int = 20,
                  box: int = 112, ds: int = 4, img_hw: tuple[int, int] | None = None):
    """Top-N local maxima of heatmap above threshold; box is box x box at full res.
    `ds` is the heatmap downsample factor."""
    from scipy import ndimage as ndi

    m = heatmap > threshold
    if not m.any():
        return []
    mx = ndi.maximum_filter(heatmap, size=5)
    peaks = (heatmap == mx) & m
    ys, xs = np.nonzero(peaks)
    vals = heatmap[ys, xs]
    order = np.argsort(-vals)[:top_n]
    hh, hw = heatmap.shape
    if img_hw is None:
        img_hw = (hh * ds, hw * ds)
    H, W = img_hw
    out = []
    half = box // 2
    for i in order:
        cy, cx = int(ys[i] * ds), int(xs[i] * ds)
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(W, x0 + box), min(H, y0 + box)
        out.append(((x0, y0, x1, y1), float(vals[i])))
    return out


def random_boxes(img_hw: tuple[int, int], n: int = 15, box: int = 224, seed: int = 0):
    """n seeded random boxes; resampled to avoid mutual IoU>0.5 so all n survive
    dedupe (they are the 'normal' examples and must all be present)."""
    rng = np.random.default_rng(seed)
    H, W = img_hw
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        x0 = int(rng.integers(0, max(1, W - box)))
        y0 = int(rng.integers(0, max(1, H - box)))
        bb = (x0, y0, min(W, x0 + box), min(H, y0 + box))
        if all(iou(bb, o[0]) <= 0.5 for o in out):
            out.append((bb, 0.0))
    return out


def iou(a, b) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def dedupe(cands: list[tuple[tuple, float, str]], iou_thr: float = 0.5,
           priority: list[str] = PRIORITY, cap: int = CAP_PER_IMAGE):
    """cands: [(bbox, score, source)]. Keep higher-priority source on overlap."""
    rank = {s: i for i, s in enumerate(priority)}
    cands = sorted(cands, key=lambda c: (rank.get(c[2], 99), -c[1]))
    kept: list[tuple[tuple, float, str]] = []
    for c in cands:
        if all(iou(c[0], k[0]) <= iou_thr for k in kept):
            kept.append(c)
        if len(kept) >= cap:
            break
    return kept


def propose_for_image(
    img: np.ndarray,
    image_id: str,
    group_id: str,
    batch: str,
    detector: str,
    run_id: str,
    heatmap: np.ndarray | None = None,
    anomaly_threshold: float | None = None,
    seed: int = 0,
) -> list[Proposal]:
    """Run all generators + dedupe. Returns Proposal list (<= cap)."""
    cands: list[tuple[tuple, float, str]] = []
    if heatmap is not None and anomaly_threshold is not None:
        for bb, s in anomaly_peaks(heatmap, anomaly_threshold, img_hw=img.shape):
            cands.append((bb, s, "anomaly_peak"))
    for bb, s, _ in tophat_crack(img)[0]:
        cands.append((bb, s, "tophat_crack"))
    for bb, s, _ in dark_void(img)[0]:
        cands.append((bb, s, "dark_void"))
    for bb, s, _ in edge_band(img)[0]:
        cands.append((bb, s, "edge_band"))
    for bb, s in fft_curtain(img)[0]:
        cands.append((bb, s, "fft_curtain"))
    for bb, s in random_boxes(img.shape, seed=seed):
        cands.append((bb, s, "random"))
    kept = dedupe(cands)
    return [_mk(image_id, group_id, batch, detector, bb, src, s, run_id) for bb, s, src in kept]

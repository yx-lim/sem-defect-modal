"""KPIs per BSE image (SPEC §4)."""

from __future__ import annotations

import numpy as np
from skimage import filters, measure, morphology

from .contract import CLASS_INDEX, CLASSES

DEFECT_KPIS = [
    "area_frac_crack_intra",
    "area_frac_crack_inter",
    "area_frac_void",
    "area_frac_agglomerate",
    "area_frac_other_anomaly",
    "crack_length_density",
    "dark_phase_frac",
    "anomaly_prevalence",
]
# artifact / covariate KPIs: reported but EXCLUDED from the verdict family
ARTIFACT_KPIS = ["curtaining_score", "edge_bloom_score", "focus", "brightness_mean",
                 "brightness_std"]
ALL_KPIS = DEFECT_KPIS + ARTIFACT_KPIS


def _otsu(img):
    try:
        return filters.threshold_multiotsu(img, classes=3)
    except ValueError:
        return None


def fft_curtain_score(img: np.ndarray, tile: int = 512, band: int = 2) -> float:
    """Mean per-tile |fy|<2 spectral energy ratio (same def as proposals)."""
    h, w = img.shape
    ratios = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            t = img[y : y + tile, x : x + tile].astype(np.float32)
            F = np.fft.fftshift(np.abs(np.fft.fft2(t)))
            cy, cx = tile // 2, tile // 2
            bandmask = np.zeros_like(F, bool)
            bandmask[cy - band : cy + band + 1, :] = True
            bandmask[cy, cx] = False
            total = F.sum() - F[cy, cx]
            ratios.append(F[bandmask].sum() / total if total > 0 else 0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def edge_bloom_score(img: np.ndarray, band: int = 64) -> float:
    h, w = img.shape
    interior = img[band : h - band, band : w - band].astype(np.float32)
    b = np.concatenate([
        img[:band].ravel(), img[-band:].ravel(),
        img[:, :band].ravel(), img[:, -band:].ravel(),
    ]).astype(np.float32)
    return float(b.mean() - interior.mean())


def compute_kpis(
    img: np.ndarray,
    pred_mask: np.ndarray | None,
    heatmap: np.ndarray | None = None,
    heatmap_thr: float | None = None,
) -> dict:
    """All SPEC §4 KPIs for one BSE image (px units)."""
    import cv2
    from scipy import ndimage as ndi

    k: dict[str, float | None] = {}
    n_px = img.size
    if pred_mask is not None:
        for cls in ("crack_intra", "crack_inter", "void", "agglomerate", "other_anomaly"):
            k[f"area_frac_{cls}"] = float((pred_mask == CLASS_INDEX[cls]).mean())
        crack_px = np.isin(pred_mask, [CLASS_INDEX["crack_intra"], CLASS_INDEX["crack_inter"]])
        t = _otsu(img)
        solid = img > t[0] if t is not None else np.ones(img.shape, bool)
        skel_len = float(morphology.skeletonize(crack_px).sum())
        k["crack_length_density"] = skel_len / max(int(solid.sum()), 1)
    else:
        for cls in ("crack_intra", "crack_inter", "void", "agglomerate", "other_anomaly"):
            k[f"area_frac_{cls}"] = None
        k["crack_length_density"] = None
    t = _otsu(img)
    k["dark_phase_frac"] = float((img <= t[0]).mean()) if t is not None else None
    if heatmap is not None and heatmap_thr is not None:
        k["anomaly_prevalence"] = float((heatmap > heatmap_thr).mean())
    else:
        k["anomaly_prevalence"] = None
    k["curtaining_score"] = fft_curtain_score(img)
    k["edge_bloom_score"] = edge_bloom_score(img)
    k["focus"] = float(np.var(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)))
    k["brightness_mean"] = float(img.mean())
    k["brightness_std"] = float(img.std())
    return k

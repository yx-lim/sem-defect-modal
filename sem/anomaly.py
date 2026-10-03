"""PatchCore with leave-one-group-out (SPEC §3 anomaly_scan)."""

from __future__ import annotations

import numpy as np

from .tiles import TILE, STRIDE, tile_grid_shape

PATCH = 14
GRID = TILE // PATCH  # 37


def greedy_coreset(features: np.ndarray, frac: float = 0.10, seed: int = 0) -> np.ndarray:
    """Greedy k-center coreset on rows of features [N,D]."""
    rng = np.random.default_rng(seed)
    n = len(features)
    m = max(1, int(round(n * frac)))
    sel = [int(rng.integers(n))]
    d = np.linalg.norm(features - features[sel[0]], axis=1)
    for _ in range(1, m):
        idx = int(np.argmax(d))
        sel.append(idx)
        d = np.minimum(d, np.linalg.norm(features - features[idx], axis=1))
    return features[sel]


def knn_scores(bank: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """1-NN L2 distance of each query row to bank. Uses faiss if available."""
    q = queries.astype(np.float32)
    b = bank.astype(np.float32)
    try:
        import faiss

        index = faiss.IndexFlatL2(b.shape[1])
        index.add(b)
        dists, _ = index.search(q, 1)
        return np.sqrt(np.maximum(dists[:, 0], 0.0))
    except ImportError:
        # chunked fallback
        out = np.empty(len(q), dtype=np.float64)
        for i in range(0, len(q), 4096):
            d = np.linalg.norm(q[i : i + 4096, None, :] - b[None, :, :], axis=2)
            out[i : i + 4096] = d.min(axis=1)
        return out


def patch_scores_to_tilemap(patch_scores: np.ndarray) -> np.ndarray:
    """[N,1369] -> [N,37,37]"""
    return patch_scores.reshape(-1, GRID, GRID)


def stitch_heatmap(
    patch_scores: np.ndarray,
    coords: list[tuple[int, int]],
    padded_hw: tuple[int, int],
    out_hw: tuple[int, int],
    downsample: int = 4,
) -> np.ndarray:
    """Stitch per-tile 37x37 patch-score grids to a full-res heatmap.
    Max over overlaps, bilinear upsample of each 37x37 grid to the tile size.
    Output downsampled `downsample`x, cropped to out_hw/downsample."""
    import cv2

    ph, pw = padded_hw
    heat = np.zeros((ph, pw), dtype=np.float32)
    maps = patch_scores_to_tilemap(patch_scores)
    for m, (y, x) in zip(maps, coords):
        up = cv2.resize(m, (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        np.maximum(heat[y : y + TILE, x : x + TILE], up, out=heat[y : y + TILE, x : x + TILE])
    oh, ow = out_hw
    heat = heat[:oh, :ow]
    if downsample > 1:
        heat = cv2.resize(
            heat, (ow // downsample, oh // downsample), interpolation=cv2.INTER_AREA
        )
    return heat


def calibrate_threshold(ref_patch_scores: list[np.ndarray], pct: float = 99.0) -> float:
    """99th percentile of LOGO patch scores over reference images."""
    all_s = np.concatenate([s.ravel() for s in ref_patch_scores])
    return float(np.percentile(all_s, pct))

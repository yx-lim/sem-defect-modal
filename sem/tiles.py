"""Tiling: 518x518, stride 448, reflect-pad edges (SPEC §3 Embedder)."""

from __future__ import annotations

import numpy as np

TILE = 518
STRIDE = 448


def tile_coords(h: int, w: int, tile: int = TILE, stride: int = STRIDE) -> list[tuple[int, int]]:
    """(y0, x0) origins covering the padded image. Grid so that padded image
    is covered exactly; last tile flush with the padded edge."""
    ph = _padded(h, tile, stride)
    pw = _padded(w, tile, stride)
    ys = list(range(0, ph - tile + 1, stride))
    xs = list(range(0, pw - tile + 1, stride))
    return [(y, x) for y in ys for x in xs]


def _padded(n: int, tile: int, stride: int) -> int:
    if n <= tile:
        return tile
    # smallest padded size such that tiles starting at 0..p-tile step stride reach edge
    k = max(0, -(-(n - tile) // stride))
    return tile + k * stride


def pad_image(img: np.ndarray, tile: int = TILE, stride: int = STRIDE) -> tuple[np.ndarray, int, int]:
    """Reflect-pad to a multiple-of-stride cover. Returns (padded, ph, pw)."""
    h, w = img.shape[:2]
    ph, pw = _padded(h, tile, stride), _padded(w, tile, stride)
    pad_h, pad_w = ph - h, pw - w
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)) + (() if img.ndim == 2 else ((0, 0),)),
                     mode="reflect")
    return img, ph, pw


def extract_tiles(img: np.ndarray, tile: int = TILE, stride: int = STRIDE):
    """Returns (tiles [N,tile,tile], coords [(y,x)], padded_hw)."""
    padded, ph, pw = pad_image(img, tile, stride)
    coords = tile_coords(img.shape[0], img.shape[1], tile, stride)
    tiles = np.stack([padded[y : y + tile, x : x + tile] for y, x in coords])
    return tiles, coords, (ph, pw)


def tile_grid_shape(h: int, w: int, tile: int = TILE, stride: int = STRIDE) -> tuple[int, int]:
    ph, pw = _padded(h, tile, stride), _padded(w, tile, stride)
    ny = (ph - tile) // stride + 1
    nx = (pw - tile) // stride + 1
    return ny, nx

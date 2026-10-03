"""detect(): sliding-window inference + overlay (SPEC §3 detect)."""

from __future__ import annotations

import numpy as np

from .contract import CLASSES

# RGB palette per class index
PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [255, 0, 0],      # crack_intra
        [255, 128, 0],    # crack_inter
        [0, 128, 255],    # void
        [255, 0, 255],    # agglomerate
        [0, 255, 255],    # curtaining
        [255, 255, 0],    # edge_bloom
        [0, 255, 0],      # other_anomaly
    ],
    dtype=np.uint8,
)


def overlay(img_u8: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = np.stack([img_u8] * 3, axis=-1).astype(np.float32)
    colored = PALETTE[mask % len(PALETTE)].astype(np.float32)
    sel = mask > 0
    rgb[sel] = (1 - alpha) * rgb[sel] + alpha * colored[sel]
    return rgb.astype(np.uint8)


def save_class_png(mask: np.ndarray, path: str) -> None:
    import cv2

    cv2.imwrite(path, mask)


def save_overlay(img_u8: np.ndarray, mask: np.ndarray, path: str,
                 alpha: float = 0.5) -> None:
    import cv2

    ov = overlay(img_u8, mask, alpha)
    cv2.imwrite(path, ov[:, :, ::-1])

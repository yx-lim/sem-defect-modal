"""DINOv2 loader + per-tile patch embedding (SPEC §3 Embedder)."""

from __future__ import annotations

import numpy as np

from .tiles import extract_tiles

DINOV2_HUB = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vitb14"
PATCH = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dinov2(backbone: str = DINOV2_MODEL, device: str = "cpu", revision: str | None = None):
    """Load DINOv2 from torch.hub. `revision` pins the hub ref (sha/tag)."""
    import torch

    kwargs = {"source": "github"}
    if revision:
        hub_id = f"{DINOV2_HUB}:{revision}"
    else:
        hub_id = DINOV2_HUB
    model = torch.hub.load(hub_id, backbone, **kwargs)
    model.eval().to(device)
    return model


def gray_to_3ch_norm(tile_u8: np.ndarray) -> np.ndarray:
    """uint8 HxW -> float32 3xHxW, ImageNet-normalized."""
    x = tile_u8.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return x


def embed_tiles(model, tiles_u8: np.ndarray, device: str = "cpu",
                batch_size: int = 8, fp16: bool = False) -> np.ndarray:
    """tiles_u8 [N,518,518] -> patch tokens [N, 37*37, D] (cls excluded)."""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, len(tiles_u8), batch_size):
            batch = np.stack([gray_to_3ch_norm(t) for t in tiles_u8[i : i + batch_size]])
            x = torch.from_numpy(batch).to(device)
            if fp16 and device != "cpu":
                x = x.half()
                model = model.half()
            feats = model.forward_features(x)
            tok = feats["x_norm_patchtokens"]  # [B, Npatch, D]
            outs.append(tok.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def embed_image(model, img_u8: np.ndarray, device: str = "cpu", fp16: bool = False):
    """Full image -> (patch_features [Ntiles,1369,D], coords, padded_hw)."""
    tiles, coords, phw = extract_tiles(img_u8)
    feats = embed_tiles(model, tiles, device=device, fp16=fp16)
    return feats, coords, phw

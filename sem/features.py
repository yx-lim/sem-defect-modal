"""DINOv2 loader + per-tile patch embedding (SPEC §3 Embedder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tiles import extract_tiles

DINOV2_HUB = "facebookresearch/dinov2"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # main @ build time
DINOV2_MODEL = "dinov2_vitb14"
PATCH = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dinov2(backbone: str = DINOV2_MODEL, device: str = "cpu",
                revision: str | None = DINOV2_REVISION):
    """Load DINOv2 from torch.hub, pinned to DINOV2_REVISION (commit sha)."""
    import torch

    hub_id = f"{DINOV2_HUB}:{revision}" if revision else DINOV2_HUB
    model = torch.hub.load(hub_id, backbone, source="github")
    model.eval().to(device)
    return model


def hub_checkpoint_sha256(pattern: str = "dinov2_vitb14") -> str | None:
    """sha256 of the downloaded torch.hub checkpoint matching `pattern`."""
    import glob
    import hashlib
    import torch

    ckpts = glob.glob(str(Path(torch.hub.get_dir()) / "checkpoints" / f"*{pattern}*"))
    if not ckpts:
        return None
    h = hashlib.sha256()
    with open(ckpts[0], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def embed_image(model, img_u8: np.ndarray, device: str = "cpu", fp16: bool = False,
                batch_size: int = 8):
    """Full image -> (patch_features [Ntiles,1369,D], coords, padded_hw)."""
    tiles, coords, phw = extract_tiles(img_u8)
    feats = embed_tiles(model, tiles, device=device, fp16=fp16,
                        batch_size=batch_size)
    return feats, coords, phw

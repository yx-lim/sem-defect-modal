"""Supervised training: masks from labels, group split, two archs, metrics
(SPEC §3 train_supervised)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from skimage import filters, morphology, measure

from .contract import CLASS_INDEX, CLASSES, IGNORE, Label, Proposal

N_CLASSES = len(CLASSES)


# ---------- mask building ----------

def proposal_region_mask(img: np.ndarray, prop: Proposal, label: str) -> np.ndarray:
    """Region mask (uint8, same size as img) for one labelled proposal.
    mask_rle if present; else class-specific region within bbox."""
    x0, y0, x1, y1 = prop.bbox
    sub = img[y0:y1, x0:x1]
    region = np.zeros(sub.shape, dtype=bool)
    if prop.mask_rle:
        from pycocotools import mask as mask_util

        rle = dict(prop.mask_rle)
        region = mask_util.decode(rle).astype(bool)
    elif label in ("crack_intra", "crack_inter"):
        from skimage.morphology import black_tophat, disk

        sm = filters.gaussian(sub.astype(np.float32), sigma=1.0, preserve_range=True)
        th = black_tophat(sm, footprint=disk(7))
        if th.max() > 0:
            region = th > np.percentile(th, 99.5)
    elif label == "void":
        try:
            t = filters.threshold_multiotsu(sub, classes=3)
            region = sub <= t[0]
        except ValueError:
            region[:] = True
    else:
        region[:] = True
    full = np.zeros(img.shape, dtype=bool)
    full[y0:y1, x0:x1] = region
    return full


def build_mask(img: np.ndarray, props: list[Proposal], labels: list[Label]) -> np.ndarray:
    """mask uint8 full-res; IGNORE everywhere except labelled regions.
    normal -> background(0). Only accepted labels should be passed in."""
    mask = np.full(img.shape, IGNORE, dtype=np.uint8)
    by_prop = {p.proposal_id: p for p in props}
    for lab in labels:
        p = by_prop.get(lab.proposal_id)
        if p is None or lab.label == "uncertain":
            continue
        cls = lab.label if lab.label in CLASS_INDEX else "background"
        region = proposal_region_mask(img, p, cls)
        mask[region] = CLASS_INDEX[cls]
    return mask


# ---------- group split ----------

def group_split(groups: list[str], batches: dict[str, str], val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[str], list[str]]:
    """Split group_ids ~val_frac val, stratified by batch (seeded)."""
    rng = np.random.default_rng(seed)
    by_batch: dict[str, list[str]] = {}
    for g in groups:
        by_batch.setdefault(batches[g], []).append(g)
    val, train = [], []
    for b, gs in sorted(by_batch.items()):
        gs = list(gs)
        rng.shuffle(gs)
        n_val = max(1, int(round(len(gs) * val_frac))) if len(gs) > 1 else 0
        val.extend(gs[:n_val])
        train.extend(gs[n_val:])
    return train, val


def class_counts(masks: list[np.ndarray]) -> dict:
    counts = {c: 0 for c in CLASSES}
    for m in masks:
        for i, c in enumerate(CLASSES):
            counts[c] += int((m == i).sum())
    return counts


# ---------- models ----------

def build_model(arch: str, backbone=None, n_classes: int = N_CLASSES, device: str = "cpu"):
    import torch
    import torch.nn as nn

    if arch == "dinov2_head":
        from .features import load_dinov2

        bb = backbone if backbone is not None else load_dinov2(device=device)
        return DinoV2Head(bb, n_classes=n_classes)
    if arch == "micronet_unet":
        return build_micronet_unet(n_classes=n_classes)
    raise ValueError(f"unknown arch {arch}")


class DinoV2Head:  # module so it can move to .nn via import inside
    def __new__(cls, backbone, n_classes: int = N_CLASSES, in_res_scale: float = 0.5):
        import torch
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb = backbone
                for p in self.bb.parameters():
                    p.requires_grad_(False)
                self.embed_dim = getattr(self.bb, "embed_dim", 768)
                # shallow conv stem on raw image at 1/2 res
                self.stem = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                )
                self.block1 = nn.Sequential(
                    nn.Conv2d(self.embed_dim + 64, 128, 3, padding=1), nn.ReLU(),
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                )
                self.cls = nn.Conv2d(64, n_classes, 1)

            def forward(self, x):
                # x: [B,1,H,W] uint8-ish float 0..1
                B, _, H, W = x.shape
                img3 = x.repeat(1, 3, 1, 1)
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
                xn = (img3 - mean) / std
                with torch.no_grad():
                    f = self.bb.forward_features(xn)
                    tok = f["x_norm_patchtokens"]  # [B,N,D]
                N = tok.shape[1]
                g = int(N ** 0.5)
                feat = tok.transpose(1, 2).reshape(B, self.embed_dim, g, g)
                feat = nn.functional.interpolate(
                    feat, scale_factor=4, mode="bilinear", align_corners=False
                )  # 1/4 res features (patch=14 -> ~ x4 up from g)
                stem = self.stem(x)  # 1/2 res? conv stride2 -> H/2
                stem = nn.functional.interpolate(
                    stem, size=feat.shape[-2:], mode="bilinear", align_corners=False
                )
                h = self.block1(torch.cat([feat, stem], dim=1))
                h = self.block2(h)
                logits = self.cls(h)
                return nn.functional.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )

        return _Head()


def get_micronet_url(encoder: str = "resnet50", model: str = "micronet") -> str:
    """Resolve the NASA pretrained-microscopy-models weight URL. Raises if the
    package/lookup is unavailable — caller should treat that as a hard stop."""
    import pretrained_microscopy_models as pmm

    return pmm.util.get_pretrained_microscopynet_url(encoder, model)


def build_micronet_unet(n_classes: int = N_CLASSES):
    """SMP Unet, resnet50 encoder with MicroNet weights, encoder frozen."""
    import segmentation_models_pytorch as smp
    import torch

    url = get_micronet_url()
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     classes=n_classes, activation=None)
    model.encoder.load_state_dict(state)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


# ---------- training ----------

def soft_dice_loss(logits, target, n_classes: int, eps: float = 1e-6):
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    valid = target != IGNORE
    loss = 0.0
    present = 0
    for c in range(1, n_classes):  # skip background
        p = probs[:, c][valid]
        t = (target[valid] == c).float()
        if t.sum() == 0:
            continue
        loss += 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        present += 1
    return loss / max(present, 1)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    import cv2

    if rng.random() < 0.5:
        img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
    if rng.random() < 0.5:
        img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()
    a = 1.0 + rng.uniform(-0.2, 0.2)  # contrast
    b = rng.uniform(-20, 20)  # brightness
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, mask


def random_crop_pair(img, mask, size: int, rng):
    H, W = img.shape
    if H <= size or W <= size:
        return img, mask
    y = int(rng.integers(0, H - size))
    x = int(rng.integers(0, W - size))
    return img[y : y + size, x : x + size], mask[y : y + size, x : x + size]


def class_weights(masks: list[np.ndarray]) -> np.ndarray:
    counts = np.ones(len(CLASSES), dtype=np.float64)
    for m in masks:
        for c in range(len(CLASSES)):
            counts[c] += (m == c).sum()
    w = 1.0 / counts
    w[0] *= 0.25  # don't dominate with background
    return (w / w.sum() * len(CLASSES)).astype(np.float32)


def train(
    model,
    train_data: list[tuple[np.ndarray, np.ndarray]],
    val_data: list[tuple[np.ndarray, np.ndarray]],
    epochs: int = 40,
    lr_head: float = 1e-3,
    lr_enc: float = 1e-4,
    crop: int = 518,
    device: str = "cpu",
    seed: int = 0,
    progress=None,
) -> dict:
    """Returns metrics dict. train_data: list of (img_u8, mask_u8)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    w = torch.tensor(class_weights([m for _, m in train_data]), device=device)
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and ("encoder" in n or n.startswith("bb"))]
    head_params = [p for p in model.parameters() if p.requires_grad and
                   not any(p is q for q in enc_params)]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": enc_params, "lr": lr_enc},
    ])
    history = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for img, mask in train_data:
            for _ in range(2):  # 2 crops per image per epoch
                im, mk = augment(img, mask, rng)
                im, mk = random_crop_pair(im, mk, min(crop, im.shape[0], im.shape[1]), rng)
                x = torch.tensor(im[None, None].astype(np.float32) / 255.0, device=device)
                y = torch.tensor(mk.astype(np.int64)[None], device=device)
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=w, ignore_index=IGNORE)
                loss = loss + soft_dice_loss(logits, y, len(CLASSES))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss)
        history.append(ep_loss)
        if progress:
            progress(ep, ep_loss)
    return evaluate(model, val_data, device=device)


# ---------- metrics ----------

def evaluate(model, data: list[tuple[np.ndarray, np.ndarray]], device: str = "cpu",
             crop: int = 518) -> dict:
    import torch

    model.eval()
    inter = np.zeros(len(CLASSES))
    union = np.zeros(len(CLASSES))
    dice_num = np.zeros(len(CLASSES))
    dice_den = np.zeros(len(CLASSES))
    crack_tp = crack_fp = crack_fn = 0
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for img, mask in data:
            preds = predict_full(model, img, device=device, tile=crop)
            valid = mask != IGNORE
            p, t = preds[valid], mask[valid]
            for c in range(len(CLASSES)):
                pc, tc = p == c, t == c
                inter[c] += (pc & tc).sum()
                union[c] += (pc | tc).sum()
                dice_num[c] += 2 * (pc & tc).sum()
                dice_den[c] += pc.sum() + tc.sum()
            conf += np.bincount(t * len(CLASSES) + p,
                                minlength=len(CLASSES) ** 2).reshape(len(CLASSES), -1)
            # crack object-level F1 (IoU>=0.3) over crack classes combined
            for (p_lab, t_lab) in _matched_components(
                    np.isin(preds, [1, 2]), np.isin(mask, [1, 2]), iou_thr=0.3):
                crack_tp += p_lab and t_lab
    ious = {CLASSES[c]: float(inter[c] / union[c]) if union[c] else None
            for c in range(len(CLASSES))}
    dices = {CLASSES[c]: float(dice_num[c] / dice_den[c]) if dice_den[c] else None
             for c in range(len(CLASSES))}
    crack = _crack_object_f1(np.isin(preds, [1, 2]), np.isin(mask, [1, 2])) if data else None
    return {"iou": ious, "dice": dices,
            "confusion": conf.tolist(), "loss_history": None}


def _components(mask_bool):
    lab = measure.label(mask_bool)
    return lab, [r for r in measure.regionprops(lab)]


def _matched_components(pred_mask, true_mask, iou_thr=0.3):
    lab_p, regs_p = _components(pred_mask)
    lab_t, regs_t = _components(true_mask)
    matched = set()
    out = []
    for rp in regs_p:
        pmask = lab_p[rp.slice] == rp.label
        best = 0.0
        best_t = None
        for rt in regs_t:
            tmask_full = lab_t == rt.label
            tsub = tmask_full[rp.slice]
            if tsub.shape != pmask.shape:
                continue
            inter = (pmask & tsub).sum()
            uni = (pmask | tsub).sum()
            if uni and inter / uni > best:
                best = inter / uni
                best_t = rt.label
        if best >= iou_thr:
            matched.add(best_t)
            out.append((True, True))
        else:
            out.append((True, False))
    for rt in regs_t:
        if rt.label not in matched:
            out.append((False, True))
    return out


def _crack_object_f1(pred_crack, true_crack, iou_thr=0.3):
    m = _matched_components(pred_crack, true_crack, iou_thr)
    tp = sum(1 for a, b in m if a and b)
    fp = sum(1 for a, b in m if a and not b)
    fn = sum(1 for a, b in m if not a and b)
    if tp + fp + fn == 0:
        return None
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r}


def predict_full(model, img: np.ndarray, device: str = "cpu", tile: int = 518,
                 stride: int = 448) -> np.ndarray:
    """Sliding-window inference, softmax-averaged overlaps -> class index map."""
    import torch
    import torch.nn.functional as F

    from .tiles import pad_image, tile_coords

    padded, ph, pw = pad_image(img, tile, stride)
    coords = tile_coords(img.shape[0], img.shape[1], tile, stride)
    acc = torch.zeros(len(CLASSES), ph, pw)
    cnt = torch.zeros(ph, pw)
    model.eval()
    with torch.no_grad():
        for y, x in coords:
            t = padded[y : y + tile, x : x + tile]
            xin = torch.tensor(t[None, None].astype(np.float32) / 255.0, device=device)
            logits = model(xin)[0].cpu()
            probs = F.softmax(logits, dim=0)
            acc[:, y : y + tile, x : x + tile] += probs
            cnt[y : y + tile, x : x + tile] += 1
    acc = acc[:, : img.shape[0], : img.shape[1]] / cnt[: img.shape[0], : img.shape[1]].clamp(min=1)
    return acc.argmax(0).numpy().astype(np.uint8)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def weights_sha256(model) -> str:
    import io as _io
    import torch

    buf = _io.BytesIO()
    torch.save(model.state_dict(), buf)
    return sha256_bytes(buf.getvalue())

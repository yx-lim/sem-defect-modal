"""Shared orchestration. Every function takes `root: Path` which plays the
role of /vol — identical layout on Modal Volume and local disk, so the same
code paths and audit chain work in both. No Modal imports."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .contract import AuditRecord, Label, ModalInfo, ModelRef, Proposal

# Volume layout (SPEC §3)
RAW = "raw"
META_INV = "meta/inventory.json"
FEATURES = "features"
ANOMALY = "anomaly"
PROPOSALS = "proposals"
CROPS = "crops"
LABELS = "labels"
MODELS = "models"
PRED = "pred"
KPI = "kpi"
AUDIT = "audit"
EVAL = "eval"

BSE_ONLY = "BSE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(image_id: str) -> str:
    return image_id.replace("/", "_")


def load_inventory(root: Path) -> dict:
    return json.loads((root / META_INV).read_text())


def save_inventory(root: Path, inv: dict) -> None:
    p = root / META_INV
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(inv, indent=2, sort_keys=True))


def raw_path(root: Path, entry: dict) -> Path:
    return root / RAW / entry["batch"] / Path(entry["path"]).name


def _read(root: Path, entry: dict) -> np.ndarray:
    from .io import read_image_gray

    return read_image_gray(raw_path(root, entry))


def _entries(root: Path, detector: str | None = None) -> list[dict]:
    inv = load_inventory(root)
    return [e for e in inv["images"] if detector is None or e["detector"] == detector]


def _entry(root: Path, image_id: str) -> dict:
    for e in load_inventory(root)["images"]:
        if e["image_id"] == image_id:
            return e
    raise KeyError(image_id)


# ---------------- ingest ----------------

def ingest(root: Path, data_dir: Path) -> dict:
    """Copy (hardlink where possible) TIFFs into root/raw/<batch>/ and write a
    merged inventory. Re-runnable: only adds new files."""
    from .io import build_inventory, parse_filename, sha256_file

    inv_new = build_inventory(data_dir)
    inv_old = load_inventory(root) if (root / META_INV).exists() else {"images": []}
    known = {e["image_id"]: e for e in inv_old["images"]}
    added = 0
    for e in inv_new["images"]:
        dst = raw_path(root, e)
        dst.parent.mkdir(parents=True, exist_ok=True)
        changed = (e["image_id"] in known
                   and known[e["image_id"]]["sha256"] != e["sha256"])
        if changed and dst.exists():
            dst.unlink()
        if not dst.exists():
            src = Path(e["path"])
            try:
                dst.hardlink_to(src)
            except OSError:
                import shutil

                shutil.copy2(src, dst)
        if changed:
            # invalidate cached features + per-group coreset so they recompute
            for p in (feature_path(root, e["image_id"]),
                      Path(str(feature_path(root, e["image_id"]))
                           .replace(".npy", "_coords.npy")),
                      coreset_path(root, e["image_id"]),
                      Path(str(coreset_path(root, e["image_id"]))
                           .replace(".npy", "_idx.npy"))):
                if p.exists():
                    p.unlink()
        if e["image_id"] not in known or changed:
            known[e["image_id"]] = {**e, "path": str(dst)}
            added += 1
        else:
            known[e["image_id"]]["path"] = str(dst)
    merged = {"images": sorted(known.values(), key=lambda x: x["image_id"]),
              "n_images": len(known)}
    save_inventory(root, merged)
    return {"added": added, "n_images": len(known)}


# ---------------- embed ----------------

def feature_path(root: Path, image_id: str, model_name: str = "dinov2_vitb14") -> Path:
    return root / FEATURES / model_name / f"{_safe(image_id)}.npy"


def embed_image(root: Path, image_id: str, model, device: str = "cpu",
                fp16: bool = False, batch_size: int = 8) -> dict:
    """Per-tile patch features + coords. Skips if already computed."""
    from .features import embed_image as _embed

    out = feature_path(root, image_id)
    if out.exists():
        return {"image_id": image_id, "skipped": True, "path": str(out)}
    e = _entry(root, image_id)
    img = _read(root, e)
    feats, coords, phw = _embed(model, img, device=device, fp16=fp16,
                                batch_size=batch_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats.astype(np.float16))
    np.save(str(out).replace(".npy", "_coords.npy"), np.array(coords))
    return {"image_id": image_id, "skipped": False, "n_tiles": len(coords),
            "path": str(out)}


# ---------------- anomaly ----------------

def _feats_for(root: Path, e: dict) -> np.ndarray:
    f = np.load(feature_path(root, e["image_id"]))
    return f.astype(np.float32).reshape(-1, f.shape[-1])


# Per-group PatchCore coreset: 1% of the group's patches selected by greedy
# k-center in a 128-d Johnson-Lindenstrauss projection (lead design; PatchCore
# paper reports ~1% coreset retains performance — unverified on SEM). Cached
# under features/dinov2_vitb14_coreset/ so the LOGO bank is ~24k x 768.
CORESET_MODEL = "dinov2_vitb14_coreset"
CORESET_RATIO = 0.01
CORESET_PROJ_DIM = 128
CORESET_SEED = 0


def coreset_path(root: Path, image_id: str) -> Path:
    return root / FEATURES / CORESET_MODEL / f"{_safe(image_id)}.npy"


def group_coreset_cached(root: Path, e: dict,
                         ratio: float = CORESET_RATIO) -> np.ndarray:
    """Cached per-group coreset of original 768-d patch features."""
    from .anomaly import group_coreset

    out = coreset_path(root, e["image_id"])
    if out.exists():
        return np.load(out)
    feats = _feats_for(root, e)
    idx = group_coreset(feats, ratio=ratio, proj_dim=CORESET_PROJ_DIM,
                        proj_seed=CORESET_SEED, start_seed=CORESET_SEED)
    sel = feats[idx]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, sel)
    np.save(str(out).replace(".npy", "_idx.npy"), idx)
    return sel


def anomaly_scan(root: Path, reference_spec: str = "all",
                 run_id: str | None = None, coreset_ratio: float = CORESET_RATIO,
                 pct: float = 99.0, downsample: int = 4) -> dict:
    from .anomaly import (calibrate_threshold, knn_scores, logo_bank,
                          stitch_heatmap)
    from .tiles import STRIDE, TILE, _padded

    run_id = run_id or str(uuid.uuid4())
    bse = _entries(root, BSE_ONLY)
    if reference_spec.startswith("batch:"):
        bname = reference_spec.split(":", 1)[1]
        ref_groups = {e["group_id"] for e in bse if e["batch"] == bname}
    else:
        ref_groups = {e["group_id"] for e in bse}
    coresets = {e["group_id"]: group_coreset_cached(root, e, coreset_ratio)
                for e in bse if e["group_id"] in ref_groups}
    results, all_scores = {}, {}
    for e in bse:
        g = e["group_id"]
        bank = logo_bank(coresets, g)
        s = knn_scores(bank, _feats_for(root, e))
        results[e["image_id"]] = (e, s)
        all_scores[e["image_id"]] = (e, s)
    thr = calibrate_threshold(_ref_scores(all_scores, ref_groups), pct=pct)
    outdir = root / ANOMALY / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for iid, (e, s) in results.items():
        safe = _safe(iid)
        coords = np.load(str(feature_path(root, iid)).replace(".npy", "_coords.npy"))
        img = _read(root, e)
        phw = (_padded(img.shape[0], TILE, STRIDE), _padded(img.shape[1], TILE, STRIDE))
        heat = stitch_heatmap(s.reshape(len(coords), -1), [tuple(c) for c in coords],
                              phw, img.shape, downsample=downsample)
        np.save(outdir / f"{safe}_heat.npy", heat)
        png = _heat_overlay(img, heat, downsample)
        import cv2

        cv2.imwrite(str(outdir / f"{safe}_heat.png"), png)
        write_preview(outdir / f"{safe}_heat.png")
        stats[iid] = {"p99": float(np.percentile(s, 99)), "max": float(s.max()),
                      "frac_above_thr": float((heat > thr).mean())}
    (outdir / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (outdir / "stats.json").write_text(json.dumps(stats))
    return {"run_id": run_id, "threshold": thr, "stats": stats}


PREVIEW_MAX_W = 1600


def write_preview(src: Path, max_w: int = PREVIEW_MAX_W,
                  suffix: str = "_preview") -> Path:
    """Write a small JPEG copy `<stem>_preview.jpg` next to src for proxy/UI
    display. Downscale with INTER_AREA only when width > max_w; aspect kept.
    Idempotent: skip when preview exists and is newer than src."""
    import cv2

    dst = src.with_name(f"{src.stem}{suffix}.jpg")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"cannot read {src}")
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, round(h * max_w / w))),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dst


def results_for(root: Path, image_id: str) -> tuple:
    """(heat_preview, overlay_preview, kpis, status_markdown) for the Results
    tab; lazily backfills downscaled previews of existing PNGs."""
    safe = _safe(image_id)
    heat = next((root / ANOMALY).glob(f"*/{safe}_heat.png"), None)
    ovl = next((root / PRED).glob(f"*/{safe}_overlay.png"), None)
    kpis: dict = {}
    for kf in (root / "kpi").glob("*.json") if (root / "kpi").exists() else []:
        kpis = json.loads(kf.read_text())
    missing = []
    if heat is None:
        missing.append("No anomaly heatmap for this image.")
    if ovl is None:
        missing.append("No predicted mask yet — run train + detect after label review.")
    if not kpis:
        missing.append("No KPIs yet — run kpi after detect.")
    status = "All results available." if not missing else "\n\n".join(missing)
    return (str(write_preview(heat)) if heat else None,
            str(write_preview(ovl)) if ovl else None,
            kpis, status)


def _ref_scores(scored: dict, ref_groups: set) -> list:
    """Calibration input: scores of images whose group is a reference group."""
    return [s for e, s in scored.values() if e["group_id"] in ref_groups]


def _heat_overlay(img: np.ndarray, heat: np.ndarray, ds: int) -> np.ndarray:
    import cv2

    up = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.6, color, 0.4, 0)


# ---------------- propose ----------------

def propose_image(root: Path, image_id: str, run_id: str,
                  anomaly_run_id: str | None = None, seed: int = 0) -> list[Proposal]:
    from .proposals import propose_for_image

    e = _entry(root, image_id)
    img = _read(root, e)
    heat = thr = None
    if anomaly_run_id:
        hp = root / ANOMALY / anomaly_run_id / f"{_safe(image_id)}_heat.npy"
        if hp.exists():
            heat = np.load(hp)
            thr = json.loads((root / ANOMALY / anomaly_run_id / "threshold.json")
                             .read_text())["threshold"]
    return propose_for_image(img, e["image_id"], e["group_id"], e["batch"],
                             e["detector"], run_id, heat, thr, seed=seed)


def write_proposals(root: Path, run_id: str, props: list[Proposal]) -> Path:
    p = root / PROPOSALS / f"{run_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for pr in props:
            f.write(pr.model_dump_json() + "\n")
    return p


def load_proposals(root: Path, run_id: str | None = None) -> list[Proposal]:
    out = []
    paths = ([root / PROPOSALS / f"{run_id}.jsonl"] if run_id
             else sorted((root / PROPOSALS).glob("*.jsonl")))
    for p in paths:
        if p.exists():
            for line in open(p):
                if line.strip():
                    out.append(Proposal.model_validate_json(line))
    return out


# ---------------- label ----------------

def select_subset(proposals: list[Proposal], n: int, seed: int = 0,
                  min_random_frac: float = 0.2) -> list[Proposal]:
    """Stratified subset: cells (batch x source), shuffle each cell with seed,
    round-robin across cells until n, enforce >=min_random_frac 'random'."""
    import random as _r

    rng = _r.Random(seed)
    cells: dict[tuple[str, str], list[Proposal]] = {}
    for p in proposals:
        cells.setdefault((p.batch, p.source), []).append(p)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells)
    order: list[Proposal] = []
    while len(order) < n and any(cells[k] for k in keys):
        for k in keys:
            if len(order) >= n:
                break
            if cells[k]:
                order.append(cells[k].pop())
    # enforce >=min_random_frac 'random' proposals: replace the last-added
    # non-random items with leftover randoms until the floor is met
    n_rand_needed = int(np.ceil(min_random_frac * min(n, len(order))))
    rand_left = [p for k in keys for p in cells[k] if k[1] == "random"]
    li = 0
    i = len(order) - 1
    while (sum(1 for p in order if p.source == "random") < n_rand_needed
           and li < len(rand_left) and i >= 0):
        if order[i].source != "random":
            order[i] = rand_left[li]
            li += 1
        i -= 1
    return order


def latest_labels(root: Path, label_version: str) -> dict:
    """Latest Label per proposal_id (newest created_at wins). labels.jsonl is
    append-only, so reviewer/relabeled entries supersede earlier ones."""
    p = root / LABELS / label_version / "labels.jsonl"
    out: dict[str, Label] = {}
    if not p.exists():
        return out
    for line in open(p):
        lab = Label.model_validate_json(line)
        cur = out.get(lab.proposal_id)
        if cur is None or lab.created_at >= cur.created_at:
            out[lab.proposal_id] = lab
    return out


def latest_pending(root: Path) -> list[Label]:
    """Latest label per proposal_id across all versions that is still
    pending_review."""
    out: dict[str, Label] = {}
    for vf in (root / LABELS).glob("*/labels.jsonl"):
        for pid, lab in latest_labels(root, vf.parent.name).items():
            cur = out.get(pid)
            if cur is None or lab.created_at >= cur.created_at:
                out[pid] = lab
    return [l for l in out.values() if l.status == "pending_review"]


_REVIEW_LOCK = __import__("threading").Lock()


def append_review(root: Path, proposal_id: str, label: str,
                  reviewer: str, revise: bool = False) -> dict:
    """Append a reviewer label that supersedes the latest pending_review label
    for proposal_id (labels.jsonl is append-only; created_at wins).
    revise=True supersedes an earlier human decision by the SAME reviewer only.
    Returns {"written": True, "label": final_label}, {"written": False,
    "reason": "already_reviewed", ...} or {"written": False, "reason":
    "not_found"}. The read-check-append is locked against concurrent requests
    in this process."""
    from datetime import datetime, timezone
    import uuid as _uuid

    with _REVIEW_LOCK:
        found = False
        for vf in (root / LABELS).glob("*/labels.jsonl"):
            cur = latest_labels(root, vf.parent.name).get(proposal_id)
            if cur is None:
                continue
            found = True
            if cur.status == "pending_review":
                pass
            elif not (revise and cur.source == "human"
                      and cur.reviewer_id == reviewer):
                return {"written": False, "reason": "already_reviewed",
                        "reviewer_id": cur.reviewer_id, "label": cur.label,
                        "status": cur.status}
            new = cur.model_copy(update={
                "label_id": _uuid.uuid4().hex[:12],
                "status": "rejected" if label == "rejected" else "accepted_human",
                "label": cur.label if label == "rejected" else label,
                "reviewer_id": reviewer,
                "source": "human",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            with open(vf, "a") as f:
                f.write(new.model_dump_json() + "\n")
            return {"written": True, "label": new.label}
        if not found:
            return {"written": False, "reason": "not_found"}


def label(root: Path, run_id: str, label_version: str, client,
          proposal_ids: list[str] | None = None, n: int | None = None,
          seed: int = 0, vlm_model: str | None = None,
          vlm_only_first_n: int = 5) -> dict:
    """Label (a subset of) proposals; saves crops for the UI; appends labels."""
    from .label_agent import label_proposals, render_context, render_crop

    # first-five budget is global across runs for this label_version
    already_acc = sum(1 for l in latest_labels(root, label_version).values()
                      if l.status == "accepted_vlm_only")
    eff_first = max(0, vlm_only_first_n - already_acc)
    props = load_proposals(root, run_id)
    if proposal_ids:
        ids = set(proposal_ids)
        props = [p for p in props if p.proposal_id in ids]
    if n is not None:
        props = select_subset(props, n, seed=seed)
    images = {}
    for p in props:
        if p.image_id not in images:
            images[p.image_id] = _read(root, _entry(root, p.image_id))
    labels = label_proposals(props, images, client, label_version,
                             model_id=vlm_model,
                             vlm_only_first_n=eff_first, seed=seed)
    crop_dir = root / CROPS
    crop_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for p in props:
        cv2.imwrite(str(crop_dir / f"{p.proposal_id}_crop.png"),
                    render_crop(images[p.image_id], tuple(p.bbox))[:, :, ::-1])
        cv2.imwrite(str(crop_dir / f"{p.proposal_id}_context.png"),
                    render_context(images[p.image_id], tuple(p.bbox))[:, :, ::-1])
    out = root / LABELS / label_version
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "a") as f:
        for lab in labels:
            f.write(lab.model_dump_json() + "\n")
    sel_counts: dict[str, int] = {}
    for p in props:
        k = f"{p.batch}|{p.source}"
        sel_counts[k] = sel_counts.get(k, 0) + 1
    return {"n_labels": len(labels), "selection_cells": sel_counts,
            "proposal_ids": [p.proposal_id for p in props]}


# ---------------- train ----------------

def train_model(root: Path, label_version: str, arch: str = "dinov2_head",
                seed: int = 0, model_version: str | None = None,
                epochs: int = 40, device: str = "cpu",
                features_cache: bool = False, crop: int = 518) -> dict:
    import torch

    from .train import (build_mask, build_model, class_counts, group_split,
                        train, weights_sha256)

    check_name(label_version, "label_version")
    check_name(model_version, "model_version")
    labels = [l for l in latest_labels(root, label_version).values()
              if l.status in ("accepted_vlm_only", "accepted_human")]
    props = load_proposals(root)
    prop_by_id = {p.proposal_id: p for p in props}
    img_labels: dict[str, list] = {}
    for l in labels:
        p = prop_by_id.get(l.proposal_id)
        if p:
            img_labels.setdefault(p.image_id, []).append(l)
    imgs, masks, groups, batches, img_ids = [], [], [], {}, []
    for iid, labs in img_labels.items():
        img_ids.append(iid)
        e = _entry(root, iid)
        img = _read(root, e)
        iprops = [prop_by_id[l.proposal_id] for l in labs
                  if l.proposal_id in prop_by_id]
        imgs.append(img)
        masks.append(build_mask(img, iprops, labs))
        groups.append(e["group_id"])
        batches[e["group_id"]] = e["batch"]
    tr_g, va_g = group_split(list(set(groups)), batches, seed=seed)
    tr = [(i, m, iid) for (i, m), g, iid in
          zip(zip(imgs, masks), groups, img_ids) if g in tr_g]
    va = [(i, m, iid) for (i, m), g, iid in
          zip(zip(imgs, masks), groups, img_ids) if g in va_g]
    model = build_model(arch, device=device)
    provider = None
    if features_cache and arch == "dinov2_head":
        provider = _feature_provider(root, [e["image_id"] for e in
                                            _entries(root, BSE_ONLY)])
    metrics = train(model, tr, va, epochs=epochs, device=device, seed=seed,
                    crop=crop, feature_provider=provider)
    metrics["counts"] = {"train": class_counts([m for _, m, *_ in tr]),
                         "val": class_counts([m for _, m, *_ in va]),
                         "train_groups": tr_g, "val_groups": va_g}
    mv = model_version or f"{arch}_{label_version}_{uuid.uuid4().hex[:8]}"
    outdir = root / MODELS / mv
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "weights.pt")
    (outdir / "metrics.json").write_text(json.dumps(metrics, default=str))
    cfg = {"arch": arch, "label_version": label_version, "seed": seed,
           "features_cache": features_cache}
    (outdir / "config.json").write_text(json.dumps(cfg))
    return {"model_version": mv, "metrics": metrics,
            "weights_sha256": weights_sha256(model)}


def _feature_provider(root: Path, image_ids: list[str]):
    """callable(image_array_id, y, x) -> patch tokens [1369,D] for the tile at
    (y,x) of that image, from cached embed outputs. Images are keyed by id(img)
    via a caller-supplied map — see FeatureProvider below."""
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load(iid):
        if iid not in cache:
            f = feature_path(root, iid)
            if not f.exists():
                cache[iid] = None
            else:
                cache[iid] = (np.load(f), np.load(
                    str(f).replace(".npy", "_coords.npy")))
        return cache[iid]

    def get(iid: str, y: int, x: int):
        c = load(iid)
        if c is None:
            return None
        feats, coords = c
        m = np.nonzero((coords[:, 0] == y) & (coords[:, 1] == x))[0]
        if not len(m):
            return None
        return feats[int(m[0])].astype(np.float32)

    return get


# ---------------- detect ----------------

def detect_image(root: Path, image_id: str, model_version: str, model=None,
                 device: str = "cpu", out_dir: Path | None = None) -> dict:
    import torch

    from .detect import save_class_png, save_overlay
    from .train import build_model, predict_full

    check_name(model_version, "model_version")
    cfg = json.loads((root / MODELS / model_version / "config.json").read_text())
    if model is None:
        model = build_model(cfg["arch"], device=device)
        model.load_state_dict(torch.load(root / MODELS / model_version / "weights.pt",
                                         map_location=device, weights_only=True))
    e = _entry(root, image_id)
    img = _read(root, e)
    mask = predict_full(model, img, device=device)
    outdir = out_dir or root / PRED / model_version
    outdir.mkdir(parents=True, exist_ok=True)
    save_class_png(mask, str(outdir / f"{_safe(image_id)}.png"))
    save_overlay(img, mask, str(outdir / f"{_safe(image_id)}_overlay.png"))
    return {"image_id": image_id, "model_version": model_version}


# ---------------- kpi / verdict ----------------

def check_name(s: str | None, field: str) -> str | None:
    """Reject path-unsafe names. None passes through; reference_spec must be
    'all' or 'batch:<name>'."""
    import re

    if s is None:
        return s
    pat = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    if field == "reference_spec":
        if s == "all":
            return s
        if s.startswith("batch:") and ".." not in s and re.match(
                pat, s.split(":", 1)[1]):
            return s
        raise ValueError(f"bad {field}: {s!r}")
    if ".." not in s and re.match(pat, s):
        return s
    raise ValueError(f"bad {field}: {s!r}")


def kpi_verdict(root: Path, image_ids: list[str], reference_spec: str = "all",
                model_version: str | None = None,
                anomaly_run_id: str | None = None,
                n_vlm_only_labels: int = 0) -> dict:
    import cv2

    from .contract import KPIVerdict
    from .kpi import compute_kpis
    from .verdict import verdict_for_groups

    check_name(reference_spec, "reference_spec")
    check_name(model_version, "model_version")
    check_name(anomaly_run_id, "anomaly_run_id")
    bse = _entries(root, BSE_ONLY)
    if reference_spec.startswith("batch:"):
        ref = [e for e in bse if e["batch"] == reference_spec.split(":", 1)[1]]
    else:
        ref = bse
    test_ids = set(image_ids)

    if model_version:
        wanted = ([e for e in bse if e["image_id"] in test_ids] +
                  [e for e in ref if e["image_id"] not in test_ids])
        missing = [e["image_id"] for e in wanted
                   if not (root / PRED / model_version
                           / f"{_safe(e['image_id'])}.png").exists()]
        if missing:
            v = KPIVerdict(
                image_ids=image_ids,
                reference={"spec": reference_spec,
                           "image_ids": [e["image_id"] for e in ref
                                         if e["image_id"] not in test_ids],
                           "n_groups": 0},
                per_image=[{"image_id": iid, "kpis": {}} for iid in image_ids],
                per_kpi=[],
                engineering_thresholds=None,
                verdict="abstain",
                reasons=[f"missing predicted masks for image_ids: {missing} "
                         f"under pred/{model_version} — run detect first"],
                limitations=[],
            )
            out = root / KPI / f"{uuid.uuid4()}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(v.model_dump_json())
            return v.model_dump()
    thr = None
    if anomaly_run_id:
        thr = json.loads((root / ANOMALY / anomaly_run_id / "threshold.json")
                         .read_text())["threshold"]
    px_known = any(e.get("px_size_nm") for e in bse)

    def kpi_for(e):
        img = _read(root, e)
        safe = _safe(e["image_id"])
        pred = heat = None
        if model_version:
            pp = root / PRED / model_version / f"{safe}.png"
            if pp.exists():
                pred = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if anomaly_run_id:
            hp = root / ANOMALY / anomaly_run_id / f"{safe}_heat.npy"
            if hp.exists():
                heat = np.load(hp)
        return compute_kpis(img, pred, heat, thr)

    test_k = {e["group_id"]: kpi_for(e) for e in bse if e["image_id"] in test_ids}
    ref_k = {e["group_id"]: kpi_for(e)
             for e in ref if e["image_id"] not in test_ids}
    v = verdict_for_groups(test_k, ref_k, image_ids, reference_spec,
                           [e["image_id"] for e in ref
                            if e["image_id"] not in test_ids],
                           n_vlm_only_labels=n_vlm_only_labels,
                           px_calibrated=px_known)
    out = root / KPI / f"{uuid.uuid4()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(v.model_dump_json())
    return v.model_dump()


# ---------------- freeze / evaluate ----------------

def freeze(root: Path, model_version: str, anomaly_run_id: str) -> dict:
    from .audit import config_sha256, git_state
    from .io import sha256_file

    cfg = json.loads((root / MODELS / model_version / "config.json").read_text())
    w = sha256_file(root / MODELS / model_version / "weights.pt")
    thr = json.loads((root / ANOMALY / anomaly_run_id / "threshold.json").read_text())
    commit, _ = git_state()
    obj = {"config_sha256": config_sha256(cfg), "weights_sha256": w,
           "thresholds": thr, "git_commit": commit,
           "model_version": model_version, "anomaly_run_id": anomaly_run_id}
    (root / AUDIT).mkdir(parents=True, exist_ok=True)
    (root / AUDIT / "freeze.json").write_text(json.dumps(obj))
    return obj


def check_frozen(root: Path, model_version: str) -> None:
    from .audit import config_sha256
    from .io import sha256_file

    f = json.loads((root / AUDIT / "freeze.json").read_text())
    if sha256_file(root / MODELS / model_version / "weights.pt") != f["weights_sha256"]:
        raise RuntimeError("weights hash != freeze.json; holdout evaluation refused")
    cfg = json.loads((root / MODELS / model_version / "config.json").read_text())
    if config_sha256(cfg) != f["config_sha256"]:
        raise RuntimeError("config hash != freeze.json; holdout evaluation refused")


def evaluate_holdout(root: Path, image_ids: list[str], model_version: str,
                     anomaly_run_id: str, device: str = "cpu") -> dict:
    check_name(model_version, "model_version")
    check_name(anomaly_run_id, "anomaly_run_id")
    check_frozen(root, model_version)
    tag = uuid.uuid4().hex[:8]
    out = root / EVAL / tag
    out.mkdir(parents=True, exist_ok=True)
    results = [detect_image(root, iid, model_version, device=device,
                            out_dir=out / "pred")
               for iid in image_ids]
    (out / "results.json").write_text(json.dumps(results))
    return {"eval_id": tag, "results": results}


# ---------------- audit helpers ----------------

def make_audit_record(function: str, started_at: str, t0: float, config: dict,
                      inputs=None, outputs=None, gpu="cpu-local",
                      est_cost_usd: float = 0.0, metrics=None, verdict=None,
                      label_version=None, model=None) -> AuditRecord:
    from .audit import config_sha256, git_state
    from .contract import FileHash

    commit, dirty = git_state()
    wall = time.time() - t0
    return AuditRecord(
        run_id=str(uuid.uuid4()), function=function, started_at=started_at,
        ended_at=_now(), wall_s=wall, git_commit=commit, git_dirty=dirty,
        config=config, config_sha256=config_sha256(config),
        inputs=[FileHash(**i) for i in (inputs or [])],
        outputs=[FileHash(**o) for o in (outputs or [])],
        model=ModelRef(**model) if model else None,
        label_version=label_version,
        modal=ModalInfo(function_call_id=None, gpu=gpu,
                        est_cost_usd=est_cost_usd),
        metrics=metrics or {}, verdict=verdict, limitations=[],
    )


def audit_write(root: Path, rec: AuditRecord) -> None:
    from .audit import write_pending

    write_pending(root / AUDIT / "pending", rec)


def audit_flush(root: Path) -> int:
    from .audit import flush_pending

    return flush_pending(root / AUDIT / "pending", root / AUDIT / "chain.jsonl",
                         root / AUDIT / "flushed")

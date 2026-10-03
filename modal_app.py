"""Modal wrapper — thin layer over sem/ (SPEC §3). Requires modal + credentials."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("sem-defects")
vol = modal.Volume.from_name("sem-data", create_if_missing=True)
VOL = "/vol"

COST = {"L4": 0.80, "A10": 1.10, "L40S": 1.95, "cpu": 0.05}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "tifffile", "scikit-image",
        "opencv-python-headless", "numpy", "scipy", "pydantic", "faiss-cpu",
        "anthropic", "gradio", "fastapi", "segmentation-models-pytorch",
        "pillow", "pycocotools",
    )
    .add_local_python_source("sem")
)

anthropic_secret = modal.Secret.from_name("anthropic")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(function: str, started_at: str, t0: float, config: dict,
           inputs=None, outputs=None, gpu=None, metrics=None, verdict=None,
           label_version=None, model=None):
    from sem.audit import append_record, config_sha256, git_state
    from sem.contract import AuditRecord, FileHash, ModalInfo, ModelRef

    commit, dirty = git_state()
    try:
        fcid = modal.current_function_call_id()
    except Exception:
        fcid = None
    wall = time.time() - t0
    rec = AuditRecord(
        run_id=str(uuid.uuid4()),
        function=function,
        started_at=started_at,
        ended_at=_now(),
        wall_s=wall,
        git_commit=commit,
        git_dirty=dirty,
        config=config,
        config_sha256=config_sha256(config),
        inputs=[FileHash(**i) for i in (inputs or [])],
        outputs=[FileHash(**o) for o in (outputs or [])],
        model=ModelRef(**model) if model else None,
        label_version=label_version,
        modal=ModalInfo(
            function_call_id=fcid, gpu=gpu,
            est_cost_usd=wall / 3600 * COST.get(gpu or "cpu", 0.05),
        ),
        metrics=metrics or {},
        verdict=verdict,
        limitations=[],
    )
    append_record(f"{VOL}/audit/chain.jsonl", rec)
    return rec


# ---------------- upload (local entrypoint) ----------------

@app.local_entrypoint()
def upload(data_dir: str = "/home/ubuntu/data/sem"):
    from sem.io import build_inventory

    inv = build_inventory(data_dir)
    Path("/tmp/inventory.json").write_text(json.dumps(inv))
    with vol.batch_upload() as batch:
        for e in inv["images"]:
            p = Path(e["path"])
            batch.put_file(str(p), f"/vol/raw/{e['batch']}/{p.name}")
        batch.put_file("/tmp/inventory.json", "/vol/meta/inventory.json")
    print(f"uploaded {inv['n_images']} files")


def _load_inventory() -> dict:
    vol.reload()
    return json.loads(Path(f"{VOL}/meta/inventory.json").read_text())


# ---------------- Embedder ----------------

@app.cls(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1)
class Embedder:
    @modal.enter()
    def load(self):
        from sem.features import load_dinov2

        self.model = load_dinov2(device="cuda")

    @modal.method()
    def embed(self, image_id: str) -> dict:
        import numpy as np
        import tifffile

        from sem.features import embed_image
        from sem.io import image_path_for, read_image_gray

        t0, started = time.time(), _now()
        vol.reload()
        inv = _load_inventory()
        entry = next(e for e in inv["images"] if e["image_id"] == image_id)
        img = read_image_gray(f"{VOL}/raw/{entry['batch']}/{Path(entry['path']).name}")
        feats, coords, phw = embed_image(self.model, img, device="cuda", fp16=True)
        safe = image_id.replace("/", "_")
        out = f"{VOL}/features/dinov2_vitb14/{safe}.npy"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.save(out, feats.astype(np.float16))
        np.save(out.replace(".npy", "_coords.npy"), np.array(coords))
        vol.commit()
        _audit("embed", started, t0, {"image_id": image_id, "model": "dinov2_vitb14"},
               gpu="L4", model={"name": "dinov2_vitb14", "hub_id": "facebookresearch/dinov2"})
        return {"image_id": image_id, "n_tiles": len(coords), "path": out}


# ---------------- anomaly_scan ----------------

@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1)
def anomaly_scan(reference_spec: str = "all", run_id: str | None = None) -> dict:
    import numpy as np

    from sem.anomaly import calibrate_threshold, greedy_coreset, knn_scores, stitch_heatmap
    from sem.io import read_image_gray

    t0, started = time.time(), _now()
    run_id = run_id or str(uuid.uuid4())
    vol.reload()
    inv = _load_inventory()
    bse = [e for e in inv["images"] if e["detector"] == "BSE"]
    if reference_spec.startswith("batch:"):
        bname = reference_spec.split(":", 1)[1]
        ref_groups = {e["group_id"] for e in bse if e["batch"] == bname}
    else:
        ref_groups = {e["group_id"] for e in bse}

    def feats_for(e):
        safe = e["image_id"].replace("/", "_")
        return np.load(f"{VOL}/features/dinov2_vitb14/{safe}.npy").astype(np.float32)

    ref_patch_bank: dict[str, np.ndarray] = {}
    for e in bse:
        if e["group_id"] in ref_groups:
            ref_patch_bank[e["group_id"]] = feats_for(e).reshape(-1, feats_for(e).shape[-1])

    # LOGO: bank = reference groups excluding g
    thresholds_scores = []
    results = {}
    for e in bse:
        g = e["group_id"]
        bank = np.concatenate([v for gg, v in ref_patch_bank.items() if gg != g])
        bank = greedy_coreset(bank, frac=0.10)
        q = feats_for(e).reshape(-1, feats_for(e).shape[-1])
        s = knn_scores(bank, q)
        results[e["image_id"]] = (e, s)
        thresholds_scores.append(s)
    thr = calibrate_threshold(thresholds_scores, pct=99.0)

    outdir = Path(f"{VOL}/anomaly/{run_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for iid, (e, s) in results.items():
        safe = iid.replace("/", "_")
        coords = np.load(f"{VOL}/features/dinov2_vitb14/{safe}_coords.npy")
        img = read_image_gray(f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
        from sem.tiles import _padded, TILE, STRIDE
        phw = (_padded(img.shape[0], TILE, STRIDE), _padded(img.shape[1], TILE, STRIDE))
        n_tiles = len(coords)
        patch = s.reshape(n_tiles, -1)
        heat = stitch_heatmap(patch, [tuple(c) for c in coords], phw, img.shape, downsample=4)
        np.save(outdir / f"{safe}_heat.npy", heat)
        stats[iid] = {
            "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
            "frac_above_thr": float((heat > thr).mean()),
        }
    (outdir / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (outdir / "stats.json").write_text(json.dumps(stats))
    vol.commit()
    _audit("anomaly_scan", started, t0, {"reference_spec": reference_spec, "run_id": run_id},
           gpu="L4", metrics={"threshold": thr})
    return {"run_id": run_id, "threshold": thr, "stats": stats}


# ---------------- propose ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=1800, retries=1, max_containers=8)
def propose_one(image_id: str, run_id: str, anomaly_run_id: str | None = None) -> dict:
    import numpy as np

    from sem.io import read_image_gray
    from sem.proposals import propose_for_image

    t0, started = time.time(), _now()
    vol.reload()
    inv = _load_inventory()
    e = next(e for e in inv["images"] if e["image_id"] == image_id)
    img = read_image_gray(f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
    heat = thr = None
    if anomaly_run_id:
        safe = image_id.replace("/", "_")
        hp = Path(f"{VOL}/anomaly/{anomaly_run_id}/{safe}_heat.npy")
        if hp.exists():
            heat = np.load(hp)
            thr = json.loads(Path(f"{VOL}/anomaly/{anomaly_run_id}/threshold.json").read_text())["threshold"]
    props = propose_for_image(img, e["image_id"], e["group_id"], e["batch"],
                              e["detector"], run_id, heat, thr)
    with open(f"{VOL}/proposals/{run_id}.jsonl", "a") as f:
        for p in props:
            f.write(p.model_dump_json() + "\n")
    vol.commit()
    _audit("propose_one", started, t0, {"image_id": image_id, "run_id": run_id},
           metrics={"n_proposals": len(props)})
    return {"image_id": image_id, "n": len(props)}


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def propose(run_id: str | None = None, anomaly_run_id: str | None = None) -> dict:
    run_id = run_id or str(uuid.uuid4())
    Path(f"{VOL}/proposals").mkdir(parents=True, exist_ok=True)
    inv = _load_inventory()
    ids = [e["image_id"] for e in inv["images"] if e["detector"] == "BSE"]
    results = list(propose_one.map(ids, kwargs={"run_id": run_id,
                                                "anomaly_run_id": anomaly_run_id}))
    return {"run_id": run_id, "results": results}


# ---------------- label_agent ----------------

@app.function(image=image, volumes={VOL: vol}, secrets=[anthropic_secret],
              timeout=3600, retries=1, max_containers=8)
def label_agent(proposal_ids: list[str] | None, label_version: str,
                run_id: str | None = None, vlm_model: str = "claude-sonnet-4-5",
                vlm_only_first_n: int = 5) -> dict:
    from sem.contract import Proposal
    from sem.io import read_image_gray
    from sem.label_agent import label_proposals, make_anthropic_client

    t0, started = time.time(), _now()
    vol.reload()
    inv = _load_inventory()
    props = []
    if run_id:
        with open(f"{VOL}/proposals/{run_id}.jsonl") as f:
            for line in f:
                if line.strip():
                    props.append(Proposal.model_validate_json(line))
    if proposal_ids:
        ids = set(proposal_ids)
        props = [p for p in props if p.proposal_id in ids]
    images = {}
    for p in props:
        if p.image_id not in images:
            e = next(x for x in inv["images"] if x["image_id"] == p.image_id)
            images[p.image_id] = read_image_gray(
                f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
    client = make_anthropic_client(vlm_model)
    labels = label_proposals(props, images, client, label_version,
                             model_id=vlm_model, vlm_only_first_n=vlm_only_first_n)
    out = Path(f"{VOL}/labels/{label_version}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "a") as f:
        for lab in labels:
            f.write(lab.model_dump_json() + "\n")
    vol.commit()
    _audit("label_agent", started, t0,
           {"label_version": label_version, "vlm_model": vlm_model,
            "vlm_only_first_n": vlm_only_first_n, "n": len(labels)},
           label_version=label_version)
    return {"n_labels": len(labels)}


# ---------------- train / detect ----------------

@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600, retries=1)
def train_supervised(label_version: str, arch: str = "dinov2_head", seed: int = 0,
                     model_version: str | None = None) -> dict:
    import io as _io

    import numpy as np
    import torch

    from sem.contract import Label, Proposal
    from sem.io import read_image_gray
    from sem.train import (build_mask, build_model, class_counts, group_split,
                           predict_full, train, weights_sha256)

    t0, started = time.time(), _now()
    vol.reload()
    inv = _load_inventory()
    labels = [Label.model_validate_json(l) for l in
              open(f"{VOL}/labels/{label_version}/labels.jsonl")]
    labels = [l for l in labels if l.status in ("accepted_vlm_only", "accepted_human")]
    # load all proposal runs
    props = []
    for pf in Path(f"{VOL}/proposals").glob("*.jsonl"):
        props += [Proposal.model_validate_json(l) for l in open(pf)]
    imgs, masks, groups, batches = [], [], [], {}
    by_img: dict[str, list] = {}
    for l in labels:
        by_img.setdefault(l.proposal_id, []).append(l)
    prop_by_id = {p.proposal_id: p for p in props}
    img_labels: dict[str, list] = {}
    for l in labels:
        p = prop_by_id.get(l.proposal_id)
        if p:
            img_labels.setdefault(p.image_id, []).append(l)
    for iid, labs in img_labels.items():
        e = next(x for x in inv["images"] if x["image_id"] == iid)
        img = read_image_gray(f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
        iprops = [prop_by_id[l.proposal_id] for l in labs if l.proposal_id in prop_by_id]
        imgs.append(img)
        masks.append(build_mask(img, iprops, labs))
        groups.append(e["group_id"])
        batches[e["group_id"]] = e["batch"]
    tr_g, va_g = group_split(list(set(groups)), batches, seed=seed)
    tr = [(i, m) for i, m, g in zip(imgs, masks, groups) if g in tr_g]
    va = [(i, m) for i, m, g in zip(imgs, masks, groups) if g in va_g]
    model = build_model(arch, device="cuda")
    metrics = train(model, tr, va, epochs=40, device="cuda", seed=seed)
    metrics["counts"] = {"train": class_counts([m for _, m in tr]),
                         "val": class_counts([m for _, m in va]),
                         "train_groups": tr_g, "val_groups": va_g}
    mv = model_version or f"{arch}_{label_version}_{uuid.uuid4().hex[:8]}"
    outdir = Path(f"{VOL}/models/{mv}")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "weights.pt")
    (outdir / "metrics.json").write_text(json.dumps(metrics, default=str))
    cfg = {"arch": arch, "label_version": label_version, "seed": seed}
    (outdir / "config.json").write_text(json.dumps(cfg))
    vol.commit()
    _audit("train_supervised", started, t0, cfg, gpu="L4", metrics=metrics,
           model={"name": arch, "weights_sha256": weights_sha256(model)})
    return {"model_version": mv, "metrics": metrics}


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1,
              max_containers=8)
def detect(image_id: str, model_version: str) -> dict:
    import numpy as np
    import torch

    from sem.detect import save_class_png, save_overlay
    from sem.io import read_image_gray
    from sem.train import build_model, predict_full

    t0, started = time.time(), _now()
    vol.reload()
    cfg = json.loads(Path(f"{VOL}/models/{model_version}/config.json").read_text())
    model = build_model(cfg["arch"], device="cuda")
    model.load_state_dict(torch.load(f"{VOL}/models/{model_version}/weights.pt",
                                     map_location="cuda"))
    inv = _load_inventory()
    e = next(x for x in inv["images"] if x["image_id"] == image_id)
    img = read_image_gray(f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
    mask = predict_full(model, img, device="cuda")
    safe = image_id.replace("/", "_")
    outdir = Path(f"{VOL}/pred/{model_version}")
    outdir.mkdir(parents=True, exist_ok=True)
    save_class_png(mask, str(outdir / f"{safe}.png"))
    save_overlay(img, mask, str(outdir / f"{safe}_overlay.png"))
    vol.commit()
    _audit("detect", started, t0, {"image_id": image_id, "model_version": model_version},
           gpu="L4")
    return {"image_id": image_id}


# ---------------- kpi / verdict ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def kpi_verdict(image_ids: list[str], reference_spec: str = "all",
                model_version: str | None = None,
                anomaly_run_id: str | None = None) -> dict:
    import numpy as np

    from sem.io import read_image_gray
    from sem.kpi import compute_kpis
    from sem.verdict import verdict_for_groups

    t0, started = time.time(), _now()
    vol.reload()
    inv = _load_inventory()
    bse = [e for e in inv["images"] if e["detector"] == "BSE"]
    if reference_spec.startswith("batch:"):
        ref = [e for e in bse if e["batch"] == reference_spec.split(":", 1)[1]]
    else:
        ref = bse
    test_ids = set(image_ids)
    thr = None
    if anomaly_run_id:
        thr = json.loads(Path(f"{VOL}/anomaly/{anomaly_run_id}/threshold.json")
                         .read_text())["threshold"]

    def kpi_for(e):
        img = read_image_gray(f"{VOL}/raw/{e['batch']}/{Path(e['path']).name}")
        safe = e["image_id"].replace("/", "_")
        pred = heat = None
        if model_version:
            pp = Path(f"{VOL}/pred/{model_version}/{safe}.png")
            if pp.exists():
                import cv2
                pred = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if anomaly_run_id:
            hp = Path(f"{VOL}/anomaly/{anomaly_run_id}/{safe}_heat.npy")
            if hp.exists():
                heat = np.load(hp)
        return compute_kpis(img, pred, heat, thr)

    test_k = {e["group_id"]: kpi_for(e) for e in bse if e["image_id"] in test_ids}
    ref_k = {e["group_id"]: kpi_for(e)
             for e in ref if e["image_id"] not in test_ids}
    v = verdict_for_groups(test_k, ref_k, image_ids, reference_spec,
                           [e["image_id"] for e in ref if e["image_id"] not in test_ids])
    out = Path(f"{VOL}/kpi/{str(uuid.uuid4())}.json")
    out.write_text(v.model_dump_json())
    vol.commit()
    _audit("kpi_verdict", started, t0,
           {"image_ids": image_ids, "reference_spec": reference_spec,
            "model_version": model_version, "anomaly_run_id": anomaly_run_id},
           verdict=v.model_dump())
    return v.model_dump()


@app.function(image=image, volumes={VOL: vol})
@modal.fastapi_endpoint(method="POST")
def verdict_api(body: dict) -> dict:
    return kpi_verdict.remote(
        body["image_ids"], body.get("reference_spec", "all"),
        body.get("model_version"), body.get("anomaly_run_id"))


# ---------------- freeze / evaluate / verify ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=600)
def freeze(model_version: str, anomaly_run_id: str) -> dict:
    from sem.io import sha256_file

    t0, started = time.time(), _now()
    vol.reload()
    cfg = json.loads(Path(f"{VOL}/models/{model_version}/config.json").read_text())
    w = sha256_file(f"{VOL}/models/{model_version}/weights.pt")
    thr = json.loads(Path(f"{VOL}/anomaly/{anomaly_run_id}/threshold.json").read_text())
    commit, _ = __import__("sem.audit", fromlist=["git_state"]).git_state()
    freeze_obj = {"config_sha256": __import__("sem.audit", fromlist=["config_sha256"])
                  .config_sha256(cfg), "weights_sha256": w,
                  "thresholds": thr, "git_commit": commit}
    Path(f"{VOL}/audit").mkdir(parents=True, exist_ok=True)
    Path(f"{VOL}/audit/freeze.json").write_text(json.dumps(freeze_obj))
    vol.commit()
    _audit("freeze", started, t0, {"model_version": model_version,
                                   "anomaly_run_id": anomaly_run_id})
    return freeze_obj


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def evaluate_holdout(image_ids: list[str], model_version: str,
                     anomaly_run_id: str) -> dict:
    from sem.io import sha256_file

    vol.reload()
    f = json.loads(Path(f"{VOL}/audit/freeze.json").read_text())
    if sha256_file(f"{VOL}/models/{model_version}/weights.pt") != f["weights_sha256"]:
        raise RuntimeError("weights hash != freeze.json; holdout evaluation refused")
    cfg = json.loads(Path(f"{VOL}/models/{model_version}/config.json").read_text())
    from sem.audit import config_sha256
    if config_sha256(cfg) != f["config_sha256"]:
        raise RuntimeError("config hash != freeze.json; holdout evaluation refused")
    tag = uuid.uuid4().hex[:8]
    results = list(detect.map(image_ids, kwargs={"model_version": model_version}))
    out = Path(f"{VOL}/eval/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results))
    vol.commit()
    return {"eval_id": tag, "results": results}


@app.function(image=image, volumes={VOL: vol}, timeout=300)
def verify_chain() -> dict:
    from sem.audit import verify_chain as _v

    vol.reload()
    return _v(f"{VOL}/audit/chain.jsonl")


# ---------------- ui ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=3600)
@modal.asgi_app()
def ui():
    from sem.ui import build_app, mount_fastapi

    def list_pending():
        vol.reload()
        from sem.contract import Label
        out = []
        for lf in Path(f"{VOL}/labels").glob("*/labels.jsonl"):
            for line in open(lf):
                l = Label.model_validate_json(line)
                if l.status == "pending_review":
                    out.append({"proposal_id": l.proposal_id,
                                "image_id": "", "source": "",
                                "vlm_suggestion": l.vlm_suggestion.model_dump()
                                if l.vlm_suggestion else None})
        return out

    def get_crop_ctx(pid):
        c = Path(f"{VOL}/crops/{pid}_crop.png")
        x = Path(f"{VOL}/crops/{pid}_context.png")
        return (str(c) if c.exists() else None, str(x) if x.exists() else None)

    def submit_review(pid, label, reviewer):
        vol.reload()
        from datetime import datetime, timezone
        from sem.contract import Label
        for lf in Path(f"{VOL}/labels").glob("*/labels.jsonl"):
            lines = open(lf).read().splitlines()
            new = []
            changed = False
            for line in lines:
                l = Label.model_validate_json(line)
                if l.proposal_id == pid and l.status == "pending_review":
                    l.status = "rejected" if label == "rejected" else "accepted_human"
                    l.label = label if label != "rejected" else l.label
                    l.reviewer_id = reviewer
                    l.source = "human"
                    l.created_at = datetime.now(timezone.utc).isoformat()
                    changed = True
                new.append(l.model_dump_json())
            if changed:
                Path(lf).write_text("\n".join(new) + "\n")
        vol.commit()

    def list_images():
        return [e["image_id"] for e in _load_inventory()["images"]
                if e["detector"] == "BSE"]

    def get_results(image_id):
        vol.reload()
        safe = image_id.replace("/", "_")
        heat = next(Path(f"{VOL}/anomaly").glob(f"*/{safe}_heat.png"), None)
        ovl = next(Path(f"{VOL}/pred").glob(f"*/{safe}_overlay.png"), None)
        kpis = {}
        return (str(heat) if heat else None, str(ovl) if ovl else None, kpis)

    def get_audit():
        import json as _j
        p = Path(f"{VOL}/audit/chain.jsonl")
        return [_j.loads(l) for l in open(p)] if p.exists() else []

    demo = build_app(list_pending, get_crop_ctx, submit_review, list_images,
                     get_results, get_audit, verify_chain.remote)
    return mount_fastapi(demo)

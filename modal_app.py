"""Thin Modal wrapper — all orchestration lives in sem/pipeline.py so the same
code paths, Volume layout and audit chain run locally and on Modal."""

from __future__ import annotations

import uuid
from pathlib import Path

import modal

app = modal.App("sem-defects")
vol = modal.Volume.from_name("sem-data", create_if_missing=True)
VOL = "/vol"
ROOT = Path(VOL)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "torchvision", "tifffile", "scikit-image",
        "opencv-python-headless", "numpy", "scipy", "pydantic", "faiss-cpu",
        "anthropic", "gradio", "fastapi", "segmentation-models-pytorch",
        "pillow", "pycocotools", "imagecodecs",
    )
    .add_local_python_source("sem")
)

anthropic_secret = modal.Secret.from_name("anthropic")


# ---------------- upload (local entrypoint) ----------------

@app.local_entrypoint()
def upload(data_dir: str = "/home/ubuntu/data/sem"):
    with vol.batch_upload(force=True) as b:
        b.put_directory(data_dir, "/incoming")
    print(ingest_remote.remote())


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def ingest_remote() -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    r = P.ingest(ROOT, ROOT / "incoming")
    vol.commit()
    _audit(P, "ingest", started, t0, {"data_dir": "incoming"},
           metrics={"added": r["added"], "n_images": r["n_images"]})
    audit_flush.remote()
    return r


# ---------------- Embedder ----------------

@app.cls(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1)
class Embedder:
    @modal.enter()
    def load(self):
        from sem.features import load_dinov2

        self.model = load_dinov2(device="cuda")

    @modal.method()
    def embed(self, image_id: str) -> dict:
        import sem.pipeline as P
        from sem.features import DINOV2_REVISION, hub_checkpoint_sha256

        t0, started = __import__("time").time(), P._now()
        vol.reload()
        r = P.embed_image(ROOT, image_id, self.model, device="cuda", fp16=True)
        vol.commit()
        _audit(P, "embed", started, t0,
               {"image_id": image_id, "model": "dinov2_vitb14"}, gpu="L4",
               model={"name": "dinov2_vitb14", "hub_id": "facebookresearch/dinov2",
                      "revision": DINOV2_REVISION,
                      "weights_sha256": hub_checkpoint_sha256()})
        return r


# ---------------- audit ----------------

def _audit(P, function, started_at, t0, config, gpu=None, est_cost_usd=None,
           **kw):
    import time as _t

    from sem.pipeline import audit_write, make_audit_record

    wall = _t.time() - t0
    try:
        fcid = modal.current_function_call_id()
    except Exception:
        fcid = None
    rec = make_audit_record(function, started_at, t0, config, gpu=gpu or "cpu",
                            est_cost_usd=(wall / 3600 *
                                          {"L4": 0.80, "A10": 1.10,
                                           "L40S": 1.95}.get(gpu or "", 0.05)),
                            **kw)
    rec.modal.function_call_id = fcid
    audit_write(ROOT, rec)
    vol.commit()  # make the pending record visible to audit_flush
    return rec


@app.function(image=image, volumes={VOL: vol}, timeout=600, max_containers=1)
def audit_flush() -> dict:
    import sem.pipeline as P

    vol.reload()
    n = P.audit_flush(ROOT)
    vol.commit()
    return {"flushed": n}


# ---------------- anomaly_scan ----------------

@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1)
def anomaly_scan(reference_spec: str = "all", run_id: str | None = None) -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    run_id = run_id or str(uuid.uuid4())
    vol.reload()
    r = P.anomaly_scan(ROOT, reference_spec=reference_spec, run_id=run_id)
    vol.commit()
    _audit(P, "anomaly_scan", started, t0,
           {"reference_spec": reference_spec, "run_id": run_id,
            "coreset_ratio": 0.01, "coreset_scope": "per_group",
            "proj_dim": 128, "proj_seed": 0}, gpu="L4",
           metrics={"threshold": r["threshold"]})
    audit_flush.remote()
    return r


# ---------------- propose ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=1800, retries=1,
              max_containers=8)
def propose_one(image_id: str, run_id: str,
                anomaly_run_id: str | None = None) -> list[str]:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    props = P.propose_image(ROOT, image_id, run_id,
                            anomaly_run_id=anomaly_run_id)
    _audit(P, "propose_one", started, t0, {"image_id": image_id,
                                           "run_id": run_id},
           metrics={"n_proposals": len(props)})
    return [p.model_dump_json() for p in props]


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def propose(run_id: str | None = None, anomaly_run_id: str | None = None) -> dict:
    import sem.pipeline as P
    from sem.contract import Proposal

    run_id = run_id or str(uuid.uuid4())
    vol.reload()
    inv = P.load_inventory(ROOT)
    ids = [e["image_id"] for e in inv["images"] if e["detector"] == "BSE"]
    all_props = []
    for batch in propose_one.map(
            ids, kwargs={"run_id": run_id, "anomaly_run_id": anomaly_run_id}):
        all_props += [Proposal.model_validate_json(s) for s in batch]
    P.write_proposals(ROOT, run_id, all_props)
    vol.commit()
    audit_flush.remote()
    return {"run_id": run_id, "n": len(all_props)}


# ---------------- label_agent ----------------

@app.function(image=image, volumes={VOL: vol}, secrets=[anthropic_secret],
              timeout=3600, retries=1, max_containers=8)
def label_agent(proposal_ids: list[str] | None, label_version: str,
                run_id: str | None = None, vlm_model: str = "claude-opus-5-5",
                vlm_only_first_n: int = 5, n: int | None = None,
                seed: int = 0) -> dict:
    import sem.pipeline as P
    from sem.label_agent import MAX_TOKENS, make_anthropic_client

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    client = make_anthropic_client(vlm_model)  # verifies model id
    r = P.label(ROOT, run_id=run_id, label_version=label_version,
                client=client, proposal_ids=proposal_ids, n=n, seed=seed,
                vlm_model=vlm_model, vlm_only_first_n=vlm_only_first_n)
    vol.commit()
    _audit(P, "label_agent", started, t0,
           {"label_version": label_version, "vlm_model": vlm_model,
            "max_tokens": MAX_TOKENS,
            "vlm_only_first_n": vlm_only_first_n, "n": r["n_labels"],
            "sampling": "sdk-default (temperature unsupported in anthropic 1.11.0)",
            "selection_cells": r["selection_cells"],
            "proposal_ids": r["proposal_ids"]},
           label_version=label_version)
    audit_flush.remote()
    return r


# ---------------- train / detect ----------------

@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=4 * 3600,
              retries=1)
def train_supervised(label_version: str, arch: str = "dinov2_head",
                     seed: int = 0, model_version: str | None = None,
                     features_cache: bool = False) -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    r = P.train_model(ROOT, label_version, arch=arch, seed=seed,
                      model_version=model_version, device="cuda",
                      features_cache=features_cache)
    vol.commit()
    model_ref = {"name": arch, "weights_sha256": r["weights_sha256"]}
    if arch == "micronet_unet":
        from sem.train import micronet_weights_sha256

        model_ref["weights_sha256"] = micronet_weights_sha256()
    _audit(P, "train_supervised", started, t0,
           {"arch": arch, "label_version": label_version, "seed": seed},
           gpu="L4", metrics=r["metrics"], model=model_ref)
    audit_flush.remote()
    return r


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, retries=1,
              max_containers=8)
def detect(image_id: str, model_version: str) -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    r = P.detect_image(ROOT, image_id, model_version, device="cuda")
    vol.commit()
    _audit(P, "detect", started, t0, {"image_id": image_id,
                                      "model_version": model_version}, gpu="L4")
    return r


# ---------------- kpi / verdict ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def kpi_verdict(image_ids: list[str], reference_spec: str = "all",
                model_version: str | None = None,
                anomaly_run_id: str | None = None) -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    r = P.kpi_verdict(ROOT, image_ids, reference_spec=reference_spec,
                      model_version=model_version,
                      anomaly_run_id=anomaly_run_id)
    vol.commit()
    _audit(P, "kpi_verdict", started, t0,
           {"image_ids": image_ids, "reference_spec": reference_spec,
            "model_version": model_version, "anomaly_run_id": anomaly_run_id},
           verdict=r)
    audit_flush.remote()
    return r


@app.function(image=image, volumes={VOL: vol})
@modal.fastapi_endpoint(method="POST")
def verdict_api(body: dict) -> dict:
    import fastapi
    import sem.pipeline as P

    try:
        P.check_name(body.get("reference_spec", "all"), "reference_spec")
        P.check_name(body.get("model_version"), "model_version")
        P.check_name(body.get("anomaly_run_id"), "anomaly_run_id")
    except ValueError as exc:
        raise fastapi.HTTPException(400, str(exc))
    return kpi_verdict.remote(
        body["image_ids"], body.get("reference_spec", "all"),
        body.get("model_version"), body.get("anomaly_run_id"))


# ---------------- freeze / evaluate / verify ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=600)
def freeze(model_version: str, anomaly_run_id: str) -> dict:
    import sem.pipeline as P

    t0, started = __import__("time").time(), P._now()
    vol.reload()
    r = P.freeze(ROOT, model_version, anomaly_run_id)
    vol.commit()
    _audit(P, "freeze", started, t0, {"model_version": model_version,
                                      "anomaly_run_id": anomaly_run_id})
    audit_flush.remote()
    return r


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600)
def evaluate_holdout(image_ids: list[str], model_version: str,
                     anomaly_run_id: str) -> dict:
    import sem.pipeline as P

    vol.reload()
    r = P.evaluate_holdout(ROOT, image_ids, model_version, anomaly_run_id,
                           device="cuda")
    vol.commit()
    audit_flush.remote()
    return r


@app.function(image=image, volumes={VOL: vol}, timeout=300)
def verify_chain() -> dict:
    from sem.audit import verify_chain as _v

    audit_flush.remote()
    vol.reload()
    return _v(ROOT / "audit" / "chain.jsonl")


# ---------------- ui ----------------

@app.function(image=image, volumes={VOL: vol}, timeout=3600)
@modal.asgi_app()
def ui():
    import sem.pipeline as P
    from sem.contract import Label, Proposal
    from sem.ui import build_app, mount_fastapi

    prop_by_id = {p.proposal_id: p
                  for lf in (ROOT / P.PROPOSALS).glob("*.jsonl")
                  for p in [Proposal.model_validate_json(l)
                            for l in open(lf) if l.strip()]}

    def list_pending():
        vol.reload()
        return [{"proposal_id": l.proposal_id,
                 "image_id": (p.image_id if (p := prop_by_id.get(l.proposal_id))
                             else ""),
                 "source": p.source if p else "",
                 "vlm_suggestion": l.vlm_suggestion.model_dump()
                 if l.vlm_suggestion else None}
                for l in P.latest_pending(ROOT)]

    def get_crop_ctx(pid):
        c = ROOT / P.CROPS / f"{pid}_crop.png"
        x = ROOT / P.CROPS / f"{pid}_context.png"
        return (str(c) if c.exists() else None, str(x) if x.exists() else None)

    def submit_review(pid, label, reviewer):
        vol.reload()
        P.append_review(ROOT, pid, label, reviewer)
        vol.commit()

    def list_images():
        return [e["image_id"] for e in P.load_inventory(ROOT)["images"]
                if e["detector"] == "BSE"]

    def get_results(image_id):
        vol.reload()
        return P.results_for(ROOT, image_id)

    def get_audit():
        import json as _j
        p = ROOT / "audit" / "chain.jsonl"
        return [_j.loads(l) for l in open(p)] if p.exists() else []

    demo = build_app(list_pending, get_crop_ctx, submit_review, list_images,
                     get_results, get_audit, verify_chain.remote)
    return mount_fastapi(demo, allowed_paths=[f"{VOL}/{P.CROPS}",
                                              f"{VOL}/{P.ANOMALY}",
                                              f"{VOL}/{P.PRED}"])

"""Local CPU runner — same code paths, Volume layout and audit chain as Modal.

Root defaults to $SEM_ROOT or /home/ubuntu/work."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sem.pipeline import (ANOMALY, audit_flush, audit_write, ingest, load_inventory,
                          make_audit_record, _now, _safe, BSE_ONLY)

ROOT = Path(os.environ.get("SEM_ROOT", "/home/ubuntu/work"))


def _audit(fn, started, t0, config, **kw):
    audit_write(ROOT, make_audit_record(fn, started, t0, config, gpu="cpu-local",
                                        est_cost_usd=0.0, **kw))


def cmd_ingest(args):
    from sem.pipeline import ingest
    t0, started = time.time(), _now()
    r = ingest(ROOT, Path(args.data_dir))
    _audit("ingest", started, t0, {"data_dir": args.data_dir}, metrics=r)
    audit_flush(ROOT)
    print(r)


def cmd_embed(args):
    import torch

    from sem.features import DINOV2_REVISION, hub_checkpoint_sha256, load_dinov2
    from sem.pipeline import embed_image

    torch.set_num_threads(8)
    t0, started = time.time(), _now()
    model = load_dinov2(device="cpu")
    bse = [e["image_id"] for e in load_inventory(ROOT)["images"]
           if e["detector"] == BSE_ONLY]
    done = 0
    for iid in bse:
        ti = time.time()
        r = embed_image(ROOT, iid, model, device="cpu", fp16=False, batch_size=8)
        if r.get("skipped"):
            print(f"{iid}: cached")
            continue
        dt = time.time() - ti
        print(f"{iid}: {r['n_tiles']} tiles in {dt:.1f}s "
              f"({r['n_tiles']/dt:.2f} tiles/s)", flush=True)
        done += 1
    _audit("embed", started, t0, {"images": bse, "model": "dinov2_vitb14"},
           model={"name": "dinov2_vitb14", "hub_id": "facebookresearch/dinov2",
                  "revision": DINOV2_REVISION,
                  "weights_sha256": hub_checkpoint_sha256()},
           metrics={"embedded": done})
    audit_flush(ROOT)


def cmd_anomaly(args):
    from sem.pipeline import anomaly_scan
    t0, started = time.time(), _now()
    r = anomaly_scan(ROOT, reference_spec=args.reference, run_id=args.run_id)
    _audit("anomaly_scan", started, t0,
           {"reference_spec": args.reference, "run_id": r["run_id"]},
           metrics={"threshold": r["threshold"]})
    audit_flush(ROOT)
    print(r["run_id"])


def cmd_propose(args):
    from sem.pipeline import propose_image, write_proposals
    t0, started = time.time(), _now()
    import uuid as _u
    run_id = args.run_id or str(_u.uuid4())
    bse = [e["image_id"] for e in load_inventory(ROOT)["images"]
           if e["detector"] == BSE_ONLY]
    n = 0
    for iid in bse:
        props = propose_image(ROOT, iid, run_id, anomaly_run_id=args.anomaly_run_id)
        write_proposals(ROOT, run_id, props)
        n += len(props)
        print(f"{iid}: {len(props)} proposals", flush=True)
    _audit("propose", started, t0, {"run_id": run_id,
                                    "anomaly_run_id": args.anomaly_run_id},
           metrics={"n_proposals": n})
    audit_flush(ROOT)
    print(run_id)


def cmd_label(args):
    import anthropic

    from sem.label_agent import make_anthropic_client
    from sem.pipeline import label

    t0, started = time.time(), _now()
    client = make_anthropic_client(args.model)  # verifies model id
    r = label(ROOT, run_id=args.run_id, label_version=args.label_version,
              client=client, n=args.n, seed=args.seed, vlm_model=args.model,
              vlm_only_first_n=args.vlm_only_first_n)
    _audit("label_agent", started, t0,
           {"run_id": args.run_id, "label_version": args.label_version,
            "vlm_model": args.model, "n": args.n, "seed": args.seed,
            "vlm_only_first_n": args.vlm_only_first_n,
            "selection_cells": r["selection_cells"],
            "proposal_ids": r["proposal_ids"]},
           label_version=args.label_version,
           metrics={"n_labels": r["n_labels"]})
    audit_flush(ROOT)
    print(r)


def cmd_ui(args):
    from sem.ui import build_app
    from sem.pipeline import CROPS, LABELS, PROPOSALS, PRED, _entries
    from sem.contract import Label, Proposal

    prop_by_id = {p.proposal_id: p
                  for lf in (ROOT / PROPOSALS).glob("*.jsonl")
                  for p in [Proposal.model_validate_json(l)
                            for l in open(lf) if l.strip()]}

    def list_pending():
        out = []
        for lf in (ROOT / LABELS).glob("*/labels.jsonl"):
            for line in open(lf):
                l = Label.model_validate_json(line)
                if l.status == "pending_review":
                    p = prop_by_id.get(l.proposal_id)
                    out.append({"proposal_id": l.proposal_id,
                                "image_id": p.image_id if p else "",
                                "source": p.source if p else "",
                                "vlm_suggestion": l.vlm_suggestion.model_dump()
                                if l.vlm_suggestion else None})
        return out

    def get_crop_ctx(pid):
        c = ROOT / CROPS / f"{pid}_crop.png"
        x = ROOT / CROPS / f"{pid}_context.png"
        return (str(c) if c.exists() else None, str(x) if x.exists() else None)

    def submit_review(pid, label, reviewer):
        from datetime import datetime, timezone
        for lf in (ROOT / LABELS).glob("*/labels.jsonl"):
            new, changed = [], False
            for line in open(lf):
                l = Label.model_validate_json(line)
                if l.proposal_id == pid and l.status == "pending_review":
                    l.status = "rejected" if label == "rejected" else "accepted_human"
                    if label != "rejected":
                        l.label = label
                    l.reviewer_id = reviewer
                    l.source = "human"
                    l.created_at = datetime.now(timezone.utc).isoformat()
                    changed = True
                new.append(l.model_dump_json())
            if changed:
                lf.write_text("\n".join(new) + "\n")

    def list_images():
        return [e["image_id"] for e in _entries(ROOT, BSE_ONLY)]

    def get_results(image_id):
        safe = _safe(image_id)
        heat = next((ROOT / ANOMALY).glob(f"*/{safe}_heat.png"), None)
        ovl = next((ROOT / PRED).glob(f"*/{safe}_overlay.png"), None)
        kpis = {}
        for kf in (ROOT / "kpi").glob("*.json") if (ROOT / "kpi").exists() else []:
            kpis = json.loads(kf.read_text())
        return (str(heat) if heat else None, str(ovl) if ovl else None, kpis)

    def get_audit():
        p = ROOT / "audit" / "chain.jsonl"
        return [json.loads(l) for l in open(p)] if p.exists() else []

    def verify():
        from sem.audit import verify_chain
        return verify_chain(ROOT / "audit" / "chain.jsonl")

    demo = build_app(list_pending, get_crop_ctx, submit_review, list_images,
                     get_results, get_audit, verify)
    demo.launch(server_name="0.0.0.0", server_port=7860)


def cmd_train(args):
    from sem.pipeline import train_model
    t0, started = time.time(), _now()
    r = train_model(ROOT, label_version=args.label_version, arch=args.arch,
                    seed=args.seed, model_version=args.model_version,
                    epochs=args.epochs, features_cache=args.features_cache)
    _audit("train_supervised", started, t0,
           {"label_version": args.label_version, "arch": args.arch,
            "seed": args.seed, "features_cache": args.features_cache},
           metrics=r["metrics"],
           model={"name": args.arch, "weights_sha256": r["weights_sha256"]})
    audit_flush(ROOT)
    print(r["model_version"])


def cmd_detect(args):
    from sem.pipeline import detect_image
    t0, started = time.time(), _now()
    ids = args.image_ids or [e["image_id"] for e in load_inventory(ROOT)["images"]
                             if e["detector"] == BSE_ONLY]
    for iid in ids:
        detect_image(ROOT, iid, args.model_version)
        print(iid, flush=True)
    _audit("detect", started, t0, {"image_ids": ids,
                                   "model_version": args.model_version})
    audit_flush(ROOT)


def cmd_kpi(args):
    from sem.pipeline import kpi_verdict
    t0, started = time.time(), _now()
    ids = args.image_ids or []
    r = kpi_verdict(ROOT, ids, reference_spec=args.reference,
                    model_version=args.model_version,
                    anomaly_run_id=args.anomaly_run_id,
                    n_vlm_only_labels=args.n_vlm_only)
    _audit("kpi_verdict", started, t0, {"image_ids": ids,
                                        "reference_spec": args.reference,
                                        "model_version": args.model_version,
                                        "anomaly_run_id": args.anomaly_run_id},
           verdict=r)
    audit_flush(ROOT)
    print(json.dumps(r, indent=2))


def cmd_freeze(args):
    from sem.pipeline import freeze
    t0, started = time.time(), _now()
    r = freeze(ROOT, args.model_version, args.anomaly_run_id)
    _audit("freeze", started, t0, {"model_version": args.model_version,
                                   "anomaly_run_id": args.anomaly_run_id})
    audit_flush(ROOT)
    print(json.dumps(r, indent=2))


def cmd_verify(args):
    from sem.audit import verify_chain
    audit_flush(ROOT)
    print(json.dumps(verify_chain(ROOT / "audit" / "chain.jsonl")))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--data-dir", default="/home/ubuntu/data/sem")
    p.set_defaults(f=cmd_ingest)

    p = sub.add_parser("embed")
    p.set_defaults(f=cmd_embed)

    p = sub.add_parser("anomaly")
    p.add_argument("--reference", default="all")
    p.add_argument("--run-id", default=None)
    p.set_defaults(f=cmd_anomaly)

    p = sub.add_parser("propose")
    p.add_argument("--run-id", default=None)
    p.add_argument("--anomaly-run-id", default=None)
    p.set_defaults(f=cmd_propose)

    p = sub.add_parser("label")
    p.add_argument("--run-id", required=True)
    p.add_argument("--label-version", default="v1")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--vlm-only-first-n", type=int, default=5)
    p.set_defaults(f=cmd_label)

    p = sub.add_parser("ui")
    p.set_defaults(f=cmd_ui)

    p = sub.add_parser("train")
    p.add_argument("--label-version", required=True)
    p.add_argument("--arch", default="dinov2_head",
                   choices=["dinov2_head", "micronet_unet"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--model-version", default=None)
    p.add_argument("--features-cache", action="store_true")
    p.set_defaults(f=cmd_train)

    p = sub.add_parser("detect")
    p.add_argument("--model-version", required=True)
    p.add_argument("image_ids", nargs="*")
    p.set_defaults(f=cmd_detect)

    p = sub.add_parser("kpi")
    p.add_argument("image_ids", nargs="*")
    p.add_argument("--reference", default="all")
    p.add_argument("--model-version", default=None)
    p.add_argument("--anomaly-run-id", default=None)
    p.add_argument("--n-vlm-only", type=int, default=0)
    p.set_defaults(f=cmd_kpi)

    p = sub.add_parser("freeze")
    p.add_argument("--model-version", required=True)
    p.add_argument("--anomaly-run-id", required=True)
    p.set_defaults(f=cmd_freeze)

    p = sub.add_parser("verify")
    p.set_defaults(f=cmd_verify)

    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    args.f(args)


if __name__ == "__main__":
    main()

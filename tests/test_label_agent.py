import numpy as np

from sem.contract import Proposal
from sem.label_agent import (label_proposals, parse_vlm_json,
                             render_context, render_crop)
from sem.pipeline import select_subset


def mk_props(n=10, batch="Batch_1"):
    return [
        Proposal(proposal_id=f"p{i:012x}", image_id=f"{batch}/g/BSE",
                 group_id="g", batch=batch, detector="BSE",
                 bbox=(10 * i, 10 * i, 10 * i + 60, 10 * i + 60),
                 mask_rle=None, source="random" if i % 2 else "dark_void",
                 score=0.1, run_id="r")
        for i in range(n)
    ]


def img():
    return np.random.default_rng(0).integers(0, 255, (600, 800), dtype=np.uint8)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def classify(self, system, user, crop_png, context_png):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, tuple):
            return r
        return r, "end_turn"


GOOD = '{"label": "void", "confidence": 0.9, "is_artifact": false, "rationale": "dark hole"}'


def test_first5_vlm_only_rest_pending():
    client = FakeClient([GOOD] * 10)
    labels = label_proposals(mk_props(10), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1", vlm_only_first_n=5, seed=0)
    acc = [l for l in labels if l.status == "accepted_vlm_only"]
    pend = [l for l in labels if l.status == "pending_review"]
    assert len(acc) == 5 and len(pend) == 5
    assert all(l.vlm_suggestion and l.vlm_suggestion.label == "void" for l in pend)


def test_malformed_json_uncertain():
    client = FakeClient(["not json at all", "still bad"])
    labels = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1", vlm_only_first_n=5)
    assert labels[0].label == "uncertain" and labels[0].status == "pending_review"
    assert client.calls == 2  # retried once


def test_parse_strict():
    assert parse_vlm_json(GOOD).label == "void"
    assert parse_vlm_json('{"label": "defect", "confidence": 1, "is_artifact": false, "rationale": "x"}') is None
    assert parse_vlm_json('{"label": "void", "confidence": 2, "is_artifact": false, "rationale": "x"}') is None
    assert parse_vlm_json("garbage") is None


def test_fence_stripping():
    fenced = '```json\n' + GOOD + '\n```'
    assert parse_vlm_json(fenced).label == "void"
    assert parse_vlm_json('```\n' + GOOD) .label == "void"
    client = FakeClient([fenced])
    labels = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1")
    assert labels[0].label == "void"


def test_max_tokens_retry_then_truncated():
    client = FakeClient([('{"label": "void", "confidence": 0.9, "is_a',
                          "max_tokens"),
                         ('{"label": "void"', "max_tokens")])
    labels = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1", vlm_only_first_n=5)
    assert labels[0].label == "uncertain"
    assert labels[0].rationale == "truncated (max_tokens)"
    assert client.calls == 2  # retried once

    # truncated first, valid on retry -> parsed
    client2 = FakeClient([('{"label": "void"', "max_tokens"), (GOOD, "end_turn")])
    labels2 = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client2,
                              label_version="v1", vlm_only_first_n=5)
    assert labels2[0].label == "void"


def test_normal_maps_background():
    client = FakeClient(['{"label": "normal", "confidence": 0.7, "is_artifact": false, "rationale": "ok"}'])
    labels = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1")
    assert labels[0].label == "background"


def test_select_subset_balance_floor_determinism():
    # 6 sources x 3 batches, ample per cell, n=250
    sources = ["tophat_crack", "dark_void", "fft_curtain", "edge_band",
               "anomaly_peak", "random"]
    props = []
    for bi, b in enumerate(("Batch_1", "Batch_2", "Batch_3")):
        for si, s in enumerate(sources):
            for i in range(60):
                props.append(Proposal(
                    proposal_id=f"{bi}{si}{i:010d}",
                    image_id=f"{b}/g/BSE", group_id="g", batch=b,
                    detector="BSE", bbox=(0, 0, 10, 10), mask_rle=None,
                    source=s, score=0.5, run_id="r"))
    sel1 = select_subset(props, 250, seed=0)
    sel2 = select_subset(props, 250, seed=0)
    assert len(sel1) == 250
    assert len({p.proposal_id for p in sel1}) == 250  # no duplicates
    n_rand = sum(1 for p in sel1 if p.source == "random")
    assert n_rand >= 50  # >=20% floor
    assert [p.proposal_id for p in sel1] == [p.proposal_id for p in sel2]  # deterministic
    # all 3 batches still represented
    assert len({p.batch for p in sel1}) == 3


def test_render_sizes():
    im = img()
    crop = render_crop(im, (100, 100, 160, 160))
    assert crop.shape[2] == 3 and max(crop.shape[:2]) == 384
    ctx = render_context(im, (100, 100, 160, 160))
    assert max(ctx.shape[:2]) == 768

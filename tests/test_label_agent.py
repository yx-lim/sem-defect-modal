import numpy as np

from sem.contract import Proposal
from sem.label_agent import (label_proposals, parse_vlm_json,
                             render_context, render_crop)


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
        return r


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


def test_normal_maps_background():
    client = FakeClient(['{"label": "normal", "confidence": 0.7, "is_artifact": false, "rationale": "ok"}'])
    labels = label_proposals(mk_props(1), {"Batch_1/g/BSE": img()}, client,
                             label_version="v1")
    assert labels[0].label == "background"


def test_render_sizes():
    im = img()
    crop = render_crop(im, (100, 100, 160, 160))
    assert crop.shape[2] == 3 and max(crop.shape[:2]) == 384
    ctx = render_context(im, (100, 100, 160, 160))
    assert max(ctx.shape[:2]) == 768

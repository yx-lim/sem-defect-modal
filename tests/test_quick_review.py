import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sem.contract import Label, VlmSuggestion
from sem.pipeline import append_review, latest_labels, latest_pending
from sem.quick_review import quick_router, to_label


def _sug(label, conf):
    return {"label": label, "confidence": conf, "is_artifact": False,
            "rationale": "r"}


def _client(tmp_path, risk_path=None):
    pending = [
        {"proposal_id": "aa", "image_id": "B/1", "source": "random",
         "vlm_suggestion": _sug("background", 0.9)},
        {"proposal_id": "bb", "image_id": "B/1", "source": "tophat_crack",
         "vlm_suggestion": _sug("crack_inter", 0.6)},
        {"proposal_id": "cc", "image_id": "B/2", "source": "anomaly_peak",
         "vlm_suggestion": _sug("uncertain", 0.4)},
        {"proposal_id": "dd", "image_id": "B/2", "source": "anomaly_peak",
         "vlm_suggestion": _sug("uncertain", 0.6)},
    ]
    calls = []

    def submit(pid, label, reviewer, revise=False):
        calls.append((pid, label, reviewer, revise))

    cv2.imwrite(str(tmp_path / "aa_crop.png"),
                np.zeros((330, 640, 3), np.uint8))
    app = FastAPI()
    app.include_router(quick_router(lambda: pending, tmp_path, submit,
                                    risk_path))
    return TestClient(app), calls


def test_to_label_mapping():
    assert to_label("normal") == "background"
    assert to_label("reject") == "rejected"
    assert to_label("void") == "void"
    for bad in ("skip", "uncertain", "../x"):
        try:
            to_label(bad)
            assert False, bad
        except ValueError:
            pass


def test_items_page_and_thumb(tmp_path):
    c, _ = _client(tmp_path)
    assert "SEM quick review" in c.get("/quick").text
    d = c.get("/quick/api/items").json()
    assert [i["proposal_id"] for i in d["items"]] == ["cc", "dd", "bb", "aa"]
    assert [i["accept"] for i in d["items"]] == [None, None, "crack_inter",
                                                  "normal"]
    assert all(i["risk_tier"] is None for i in d["items"])
    assert "skip" not in d["choices"]
    r = c.get("/quick/thumb/aa/crop.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imread(str(tmp_path / "aa_crop_thumb.jpg")).shape[1] == 320
    assert c.get("/quick/thumb/bb/crop.jpg").status_code == 404
    assert c.get("/quick/thumb/aa/raw.jpg").status_code == 400
    assert c.get("/quick/full/aa/crop.png").status_code == 200


def test_review_and_batch_accept(tmp_path):
    c, calls = _client(tmp_path)
    assert c.post("/quick/api/review", json={
        "proposal_id": "bb", "choice": "void", "reviewer_id": " "}).status_code == 400
    assert c.post("/quick/api/review", json={
        "proposal_id": "bb", "choice": "skip", "reviewer_id": "n"}).status_code == 400
    r = c.post("/quick/api/review", json={
        "proposal_id": "bb", "choice": "normal", "reviewer_id": "n",
        "revise": True})
    assert r.json()["label"] == "background"
    assert calls[-1] == ("bb", "background", "n", True)
    calls.clear()
    r = c.post("/quick/api/accept", json={
        "proposal_ids": ["aa", "bb", "cc", "zz"], "reviewer_id": "n"}).json()
    assert r["accepted"] == {"aa": "background", "bb": "crack_inter"}
    assert sorted(r["skipped"]) == ["cc", "zz"]
    assert [x[:2] for x in calls] == [("aa", "background"), ("bb", "crack_inter")]


def test_append_review_revise(tmp_path):
    d = tmp_path / "labels" / "v1"
    d.mkdir(parents=True)
    lab = Label(label_id="l1", proposal_id="p1", label="void", confidence=0.5,
                rationale="r", is_artifact=False, source="vlm",
                prompt_version="v1",
                vlm_suggestion=VlmSuggestion(**_sug("void", 0.5)),
                status="pending_review",
                created_at="2026-01-01T00:00:00+00:00", label_version="v1")
    (d / "labels.jsonl").write_text(lab.model_dump_json() + "\n")
    append_review(tmp_path, "p1", "void", "n")
    assert latest_pending(tmp_path) == []
    append_review(tmp_path, "p1", "crack_inter", "n")  # no revise: ignored
    assert latest_labels(tmp_path, "v1")["p1"].label == "void"
    append_review(tmp_path, "p1", "crack_inter", "n", revise=True)
    cur = latest_labels(tmp_path, "v1")["p1"]
    assert (cur.label, cur.status, cur.source) == (
        "crack_inter", "accepted_human", "human")
    lines = (d / "labels.jsonl").read_text().splitlines()
    assert len(lines) == 3


def test_risk_tiers_sort_uncertain_critical_first(tmp_path):
    import json
    rp = tmp_path / "risk_tiers.json"
    rp.write_text(json.dumps({"tiers": {
        "cc": {"tier": "Low", "failure_mode": "none", "why": "noise"},
        "dd": {"tier": "Critical", "failure_mode": "F02", "why": "gap"},
        "aa": {"tier": "bogus"}}}))
    c, _ = _client(tmp_path, rp)
    d = c.get("/quick/api/items").json()
    assert [i["proposal_id"] for i in d["items"]] == ["dd", "cc", "bb", "aa"]
    by = {i["proposal_id"]: i for i in d["items"]}
    assert (by["dd"]["risk_tier"], by["dd"]["risk_mode"]) == ("Critical", "F02")
    assert by["aa"]["risk_tier"] is None
    assert d["tiers"] == ["Critical", "High", "Medium", "Low"]

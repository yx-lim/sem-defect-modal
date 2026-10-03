import json

from sem.contract import (CLASSES, IGNORE, ARTIFACT_CLASSES, AuditRecord,
                          KPIVerdict, KpiRow, Label, Proposal)


def test_proposal_roundtrip():
    p = Proposal(
        proposal_id="abc123def456", image_id="Batch_1/g1/BSE", group_id="g1",
        batch="Batch_1", detector="BSE", bbox=(1, 2, 30, 40), mask_rle=None,
        source="tophat_crack", score=0.9, run_id="r1")
    p2 = Proposal.model_validate_json(p.model_dump_json())
    assert p2 == p
    assert p2.bbox == (1, 2, 30, 40)


def test_label_roundtrip():
    l = Label(label_id="l1", proposal_id="p1", label="crack_intra", confidence=0.8,
              rationale="thin dark line", is_artifact=False, source="vlm",
              model_id="m", prompt_version="v1", reviewer_id=None,
              vlm_suggestion=None, status="accepted_vlm_only",
              created_at="2026-01-01T00:00:00+00:00", label_version="v1")
    assert Label.model_validate_json(l.model_dump_json()) == l


def test_audit_roundtrip():
    r = AuditRecord(run_id="r", function="f", started_at="a", ended_at="b",
                    wall_s=1.0, git_commit="x", git_dirty=False, config={},
                    config_sha256="z")
    d = json.loads(r.model_dump_json())
    assert AuditRecord.model_validate(d) == r


def test_kpiverdict_roundtrip():
    v = KPIVerdict(image_ids=["a"], reference={"spec": "all", "image_ids": [], "n_groups": 0},
                   per_image=[], per_kpi=[KpiRow(kpi="k", ref_median=1.0)],
                   verdict="abstain", reasons=["x"], engineering_thresholds=None,
                   limitations=["l"])
    assert KPIVerdict.model_validate_json(v.model_dump_json()) == v

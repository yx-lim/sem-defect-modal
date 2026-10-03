import json

from sem.audit import append_record, flush_pending, verify_chain, write_pending
from sem.contract import AuditRecord


def _rec(i):
    return AuditRecord(run_id=f"r{i}", function="f", started_at="a", ended_at="b",
                       wall_s=1.0, git_commit="x", git_dirty=False, config={},
                       config_sha256="z")


def test_chain_verify(tmp_path):
    p = tmp_path / "chain.jsonl"
    for i in range(3):
        append_record(p, _rec(i))
    res = verify_chain(p)
    assert res["ok"] and res["n"] == 3


def test_flush_pending_orders_and_links(tmp_path):
    pending = tmp_path / "pending"
    chain = tmp_path / "chain.jsonl"
    flushed = tmp_path / "flushed"
    # write out of order: started_at r2 < r1
    for i, ts in [(0, "2026-01-02T00:00:00"), (1, "2026-01-01T00:00:00"),
                  (2, "2026-01-03T00:00:00")]:
        r = _rec(i)
        r.started_at = ts
        write_pending(pending, r)
    n = flush_pending(pending, chain, flushed)
    assert n == 3
    assert not list(pending.glob("*.json")) and len(list(flushed.glob("*.json"))) == 3
    res = verify_chain(chain)
    assert res["ok"] and res["n"] == 3
    lines = [json.loads(l) for l in chain.read_text().splitlines()]
    assert [l["run_id"] for l in lines] == ["r1", "r0", "r2"]  # sorted by started_at


def test_chain_tamper(tmp_path):
    p = tmp_path / "chain.jsonl"
    for i in range(3):
        append_record(p, _rec(i))
    lines = p.read_text().splitlines()
    d = json.loads(lines[1])
    d["wall_s"] = 999.0
    lines[1] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n")
    res = verify_chain(p)
    assert not res["ok"] and res["broken_index"] == 1

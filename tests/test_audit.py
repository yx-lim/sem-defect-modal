import json

from sem.audit import append_record, verify_chain
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

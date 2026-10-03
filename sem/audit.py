"""Audit chain: append, hash chain, verify_chain (SPEC §2 AuditRecord, §3)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .contract import AuditRecord


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_sha256(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def record_hash(rec: AuditRecord | dict) -> str:
    d = rec.model_dump() if isinstance(rec, AuditRecord) else dict(rec)
    d.pop("record_hash", None)
    return hashlib.sha256(canonical_json(d).encode()).hexdigest()


def git_state(cwd: str | Path | None = None) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd
            ).stdout.strip()
        )
        return commit or "unknown", dirty
    except Exception:
        return "unknown", False


def _read_chain(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_record(path: str | Path, rec: AuditRecord) -> AuditRecord:
    """Set prev_hash from the last line, compute record_hash, append."""
    chain = _read_chain(path)
    rec.prev_hash = chain[-1]["record_hash"] if chain else None
    rec.record_hash = record_hash(rec)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(rec.model_dump_json() + "\n")
    return rec


def verify_chain(path: str | Path) -> dict:
    """Recompute every record_hash and prev_hash link.
    Returns {ok: bool, broken_index: int|None, n: int}."""
    chain = _read_chain(path)
    prev = None
    for i, d in enumerate(chain):
        if d.get("prev_hash") != prev:
            return {"ok": False, "broken_index": i, "n": len(chain), "why": "prev_hash link"}
        if record_hash(d) != d.get("record_hash"):
            return {"ok": False, "broken_index": i, "n": len(chain), "why": "record_hash"}
        prev = d.get("record_hash")
    return {"ok": True, "broken_index": None, "n": len(chain)}

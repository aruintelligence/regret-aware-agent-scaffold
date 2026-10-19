"""Local-only SQLite evidence gateway with strict schemas and HMAC audit chain.

Research prototype, NOT an internet-facing enterprise service. No TLS, SSO,
tenant isolation, encryption-at-rest, retention API or compliance certification.
Only non-sensitive consented research events may be ingested.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from evidence_lab import validate_label
from route_experiments import validate_decision

DECISION_KEYS = frozenset((
    "schema_version decision_id timestamp_unix task_signature logging_policy "
    "epsilon candidate_probabilities propensity action candidates runtime_kind loop_count"
).split())
CANDIDATE_KEYS = frozenset(("action", "route", "predicted_score"))
LABEL_KEYS = frozenset((
    "schema_version decision_id reward grader rubric_id source_kind graded_at_unix"
).split())
GENESIS = "0" * 64


class EvidenceConflict(ValueError):
    pass


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_event(kind, payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    if kind == "decision":
        if set(payload) - DECISION_KEYS:
            raise ValueError("unknown decision fields rejected (no raw prompts or user data)")
        for candidate in payload.get("candidates", []):
            if not isinstance(candidate, dict) or set(candidate) - CANDIDATE_KEYS:
                raise ValueError("unknown candidate fields rejected")
        validate_decision(payload)
        return payload["decision_id"]
    if kind == "label":
        if set(payload) - LABEL_KEYS:
            raise ValueError("unknown label fields rejected")
        validate_label(payload)
        return payload["decision_id"]
    raise ValueError("unsupported event kind")


def _event_hash(prev, kind, event_id, created_at, canonical, key):
    raw = "\n".join((prev, kind, event_id, created_at, canonical)).encode("utf-8")
    return hmac.new(key, raw, hashlib.sha256).hexdigest() if key else hashlib.sha256(raw).hexdigest()


@contextmanager
def _connect(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK (kind IN ('decision','label')),
            event_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL,
            UNIQUE(kind, event_id)
        )""")
        if os.name == "posix":
            os.chmod(path, 0o600)
        with connection:
            yield connection
    finally:
        connection.close()


def record_event(db_path, kind, payload, *, hmac_key=None):
    event_id = _validate_event(kind, payload)
    canonical = _canonical(payload)
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT seq,payload FROM events WHERE kind=? AND event_id=?",
            (kind, event_id),
        ).fetchone()
        if existing:
            if existing[1] != canonical:
                raise EvidenceConflict("event ID already exists with different contents")
            return {"event_id": event_id, "seq": existing[0], "duplicate": True}
        if kind == "label" and not conn.execute(
            "SELECT 1 FROM events WHERE kind='decision' AND event_id=?", (event_id,),
        ).fetchone():
            raise ValueError("label must reference an existing decision")
        last = conn.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev = last[0] if last else GENESIS
        created = datetime.now(timezone.utc).isoformat()
        digest = _event_hash(prev, kind, event_id, created, canonical, hmac_key)
        cursor = conn.execute(
            "INSERT INTO events(kind,event_id,created_at,payload,prev_hash,event_hash) "
            "VALUES(?,?,?,?,?,?)",
            (kind, event_id, created, canonical, prev, digest),
        )
        return {"event_id": event_id, "seq": cursor.lastrowid, "duplicate": False}


def verify_chain(db_path, *, hmac_key=None):
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT seq,kind,event_id,created_at,payload,prev_hash,event_hash "
            "FROM events ORDER BY seq"
        ).fetchall()
    previous = GENESIS
    errors = []
    for seq, kind, event_id, created, canonical, prev, digest in rows:
        if prev != previous:
            errors.append("broken predecessor at sequence " + str(seq))
        expected = _event_hash(prev, kind, event_id, created, canonical, hmac_key)
        if not hmac.compare_digest(digest, expected):
            errors.append("digest mismatch at sequence " + str(seq))
        try:
            payload = json.loads(canonical)
            validated_id = _validate_event(kind, payload)
            if validated_id != event_id or _canonical(payload) != canonical:
                errors.append("event payload mismatch at sequence " + str(seq))
        except (ValueError, KeyError, TypeError):
            errors.append("invalid payload at sequence " + str(seq))
        previous = digest
    return {
        "verified": not errors, "events": len(rows), "errors": errors,
        "authentication": "hmac-sha256" if hmac_key else "unsigned-sha256",
        "warning": "HMAC does not prove label truth or prevent deletion/rewrite by a key holder",
    }


def export_jsonl(db_path, decisions_path, labels_path, *, hmac_key=None):
    result = verify_chain(db_path, hmac_key=hmac_key)
    if not result["verified"]:
        raise RuntimeError("audit chain failed; export refused")
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT kind,payload FROM events ORDER BY seq").fetchall()
    targets = {"decision": Path(decisions_path), "label": Path(labels_path)}
    if targets["decision"].resolve() == targets["label"].resolve():
        raise ValueError("decision and label export paths must differ")
    if any(target.exists() for target in targets.values()):
        raise FileExistsError("export destination exists; choose new paths")
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    with targets["decision"].open("x", encoding="utf-8") as d, targets["label"].open("x", encoding="utf-8") as l:
        for kind, payload in rows:
            (d if kind == "decision" else l).write(payload + "\n")
    return {"decisions": sum(kind == "decision" for kind, _ in rows),
            "labels": sum(kind == "label" for kind, _ in rows),
            "audit_verified": True}

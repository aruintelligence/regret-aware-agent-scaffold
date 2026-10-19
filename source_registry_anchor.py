"""Separately keyed, externally retainable ĀRU source-consent checkpoints.

An ordinary local HMAC chain cannot detect deletion of the final revocation
event. An independent checkpoint can, PROVIDED the checkpoint and its
different secret are kept outside the database administrator's control.

This is not a public transparency log, trusted timestamp, or verified
human-consent service. A privileged actor controlling both keys and the
checkpoint can forge history; newer unanchored events can be deleted.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sovereign_source_registry import GENESIS, _open, verify_registry

VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _signature(record, key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("independent source anchor key must be at least 32 bytes")
    return hmac.new(key, _canonical(record), hashlib.sha256).hexdigest()


def create_source_checkpoint(db_path, checkpoint_path, *, registry_key, anchor_key):
    """Create an exclusive, non-overwriting signed external consent checkpoint."""
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError("consent checkpoint already exists")
    state = verify_registry(db_path, key=registry_key)
    if not state["verified"]:
        raise RuntimeError("cannot anchor a corrupted consent ledger")
    with _open(db_path, create=False) as db:
        last = db.execute(
            "SELECT seq,event_mac FROM consent_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        count = db.execute("SELECT COUNT(*) FROM consent_events").fetchone()[0]
    seq, head = last if last else (0, GENESIS)
    if count != seq or count != state["events"] or head != state["head_mac"]:
        raise RuntimeError("consent ledger changed during checkpoint creation")
    record = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": count,
        "last_sequence": seq,
        "head_mac": head,
        "authentication": "independent-hmac-sha256",
    }
    record["anchor_hmac_sha256"] = _signature(record, anchor_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if os.name == "posix":
        os.chmod(target, 0o600)
    return {"checkpoint": str(target), "events": count, "head_mac": head}


def verify_source_checkpoint(db_path, checkpoint_path, *, registry_key, anchor_key):
    """Verify entire ledger and independently signed historical prefix."""
    record = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("checkpoint must be an object")
    signature = record.get("anchor_hmac_sha256")
    unsigned = {key: value for key, value in record.items()
                if key != "anchor_hmac_sha256"}
    errors = []
    if (not isinstance(signature, str) or len(signature) != 64
            or not hmac.compare_digest(signature, _signature(unsigned, anchor_key))):
        errors.append("independent checkpoint authentication failed")
    if record.get("version") != VERSION or record.get("authentication") != "independent-hmac-sha256":
        errors.append("unsupported checkpoint format")
    seq = record.get("last_sequence")
    count = record.get("events")
    head = record.get("head_mac")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0 or count != seq:
        errors.append("invalid checkpoint event sequence")
    if not isinstance(head, str) or len(head) != 64:
        errors.append("invalid checkpoint head MAC")
    try:
        state = verify_registry(db_path, key=registry_key)
    except (FileNotFoundError, sqlite3.DatabaseError):
        state = {"verified": False, "events": 0, "errors": ["registry missing"]}
    if not state["verified"]:
        errors.append("source consent ledger integrity failed")
    if not errors:
        if state["events"] < seq:
            errors.append("source consent ledger truncated below anchored event count")
        else:
            if seq == 0:
                observed = GENESIS
            else:
                with _open(db_path, create=False) as db:
                    row = db.execute(
                        "SELECT event_mac FROM consent_events WHERE seq=?", (seq,)
                    ).fetchone()
                observed = row[0] if row else None
            if not isinstance(observed, str) or not hmac.compare_digest(observed, head):
                errors.append("source consent history differs from anchored prefix")
    return {
        "verified": not errors,
        "errors": errors,
        "anchored_events": count,
        "current_events": state["events"],
        "warning": (
            "A separately retained checkpoint detects truncation of its anchored prefix only. "
            "It does not verify actual consent or protect newer unanchored events."
        ),
    }

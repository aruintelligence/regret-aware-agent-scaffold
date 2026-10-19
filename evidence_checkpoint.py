"""Externally anchored, authenticated checkpoints for the local evidence chain.

An ordinary hash chain can be truncated at the tail without detection.
A checkpoint signed with a SEPARATE key and stored outside the database
binds the event count and head hash to an independently retained record.

This is not a public transparency log or trusted timestamp service.
Someone with the anchor key AND the checkpoint can forge a new anchor.
A valid historical checkpoint cannot prove no new events were omitted later.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evidence_gateway_store import GENESIS, _connect, verify_chain

ANCHOR_VERSION = 1


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_key(key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("independent anchor HMAC key must be at least 32 bytes")


def _sign(record, key):
    return hmac.new(key, _canonical(record), hashlib.sha256).hexdigest()


def create_checkpoint(db_path, output_path, *, gateway_key, anchor_key):
    """Write a non-overwriting signed checkpoint after verifying the full chain."""
    _validate_key(anchor_key)
    if not isinstance(gateway_key, bytes) or len(gateway_key) < 16:
        raise ValueError("gateway HMAC key is required")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("checkpoint already exists")
    chain = verify_chain(db_path, hmac_key=gateway_key)
    if not chain["verified"]:
        raise ValueError("cannot anchor a failed gateway chain")
    with _connect(db_path) as conn:
        last = conn.execute("SELECT seq,event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    sequence, head = last if last else (0, GENESIS)
    if count != chain["events"] or sequence != count:
        raise RuntimeError("chain changed during checkpoint creation; retry")
    record = {
        "version": ANCHOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": count,
        "last_sequence": sequence,
        "head_hash": head,
        "gateway_authentication": "hmac-sha256",
        "anchor_authentication": "hmac-sha256",
    }
    record["anchor_hmac_sha256"] = _sign(record, anchor_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents silently replacing a previously published anchor.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if os.name == "posix":
        os.chmod(target, 0o600)
    return {"checkpoint": str(target), "events": count, "head_hash": head}


def verify_checkpoint(db_path, checkpoint_path, *, gateway_key, anchor_key):
    """Require a valid signed anchor and a matching prefix in the current chain."""
    _validate_key(anchor_key)
    if not isinstance(gateway_key, bytes) or len(gateway_key) < 16:
        raise ValueError("gateway HMAC key is required")
    record = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("checkpoint must be a JSON object")
    errors = []
    if record.get("version") != ANCHOR_VERSION:
        errors.append("unsupported checkpoint version")
    if record.get("gateway_authentication") != "hmac-sha256":
        errors.append("checkpoint does not require HMAC gateway")
    if record.get("anchor_authentication") != "hmac-sha256":
        errors.append("checkpoint authentication missing")
    signed = record.get("anchor_hmac_sha256")
    if not isinstance(signed, str) or len(signed) != 64:
        errors.append("invalid checkpoint signature")
    else:
        payload = {k: v for k, v in record.items() if k != "anchor_hmac_sha256"}
        if not hmac.compare_digest(signed, _sign(payload, anchor_key)):
            errors.append("checkpoint HMAC authentication failed")
    sequence = record.get("last_sequence")
    count = record.get("events")
    head = record.get("head_hash")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        errors.append("invalid checkpoint sequence")
    if isinstance(count, bool) or not isinstance(count, int) or count != sequence:
        errors.append("invalid checkpoint event count")
    if not isinstance(head, str) or len(head) != 64:
        errors.append("invalid checkpoint head hash")
    # Refuse to describe the DB as valid if the HMAC chain itself fails.
    chain = verify_chain(db_path, hmac_key=gateway_key)
    if not chain["verified"]:
        errors.append("gateway event chain failed")
    if not errors:
        if chain["events"] < sequence:
            errors.append("database was truncated below independently anchored sequence")
        else:
            if sequence == 0:
                observed = GENESIS
            else:
                with _connect(db_path) as conn:
                    row = conn.execute("SELECT event_hash FROM events WHERE seq=?", (sequence,)).fetchone()
                observed = row[0] if row else None
            if not isinstance(observed, str) or not hmac.compare_digest(observed, head):
                errors.append("database history differs from independently anchored prefix")
    return {
        "verified": not errors,
        "errors": errors,
        "anchored_events": count,
        "current_events": chain["events"],
        "authentication": "separate-key-hmac-sha256",
        "warning": (
            "Checkpoint detects truncation/rewrite only relative to this separately retained anchor. "
            "It cannot prove no events were omitted after anchoring, validate outcome truth, "
            "or survive compromise of both the checkpoint and anchor key."
        ),
    }

"""ĀRU Sovereign Source Registry: consent-scoped, revocable document authorization.

Local research component. Stores ONLY keyed fingerprints of source text and
owner identifiers, never raw passages, questions or answers. Grants expire,
are purpose-bound, and can be revoked. Every authorization verifies the
HMAC event chain before allowing use.

CRITICAL LIMITS: A grant is an OPERATOR ATTESTATION of consent, not verified
legal consent. This HMAC chain cannot detect a deleted final suffix without
a separately retained external checkpoint. The SQLite file is not encrypted
at rest; keyed fingerprints can still reveal metadata and access patterns.
This is not a multi-tenant production authorization service.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

GENESIS = "0" * 64
MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_INIT_LOCK = threading.Lock()
_INITIALIZED = set()


def _key(key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("registry HMAC key must be at least 32 bytes")
    return key


def _field(value, name):
    if (not isinstance(value, str) or not value or len(value) > 128
            or any(ord(char) < 32 for char in value)):
        raise ValueError(name + " must be a nonempty bounded text identifier")
    return value


def _clock(now):
    if now is None:
        now = time.time()
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
        raise ValueError("invalid time")
    return float(now)


def _fingerprint(key, namespace, value):
    return hmac.new(key, (namespace + ":" + value).encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _event_mac(key, seq, kind, payload, previous):
    data = _canonical({
        "sequence": seq, "kind": kind, "payload": payload, "previous": previous,
    })
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).hexdigest()


def _initialize(path):
    identity = str(path.resolve())
    with _INIT_LOCK:
        if identity in _INITIALIZED and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path), timeout=30) as db:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS consent_events (
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('grant','revoke')),
                payload TEXT NOT NULL,
                previous_mac TEXT NOT NULL,
                event_mac TEXT NOT NULL
            )""")
        if os.name == "posix":
            os.chmod(path, 0o600)
        _INITIALIZED.add(identity)


def _open(path, *, create):
    path = Path(path)
    if not create and not path.is_file():
        raise FileNotFoundError("source registry does not exist")
    if create:
        _initialize(path)
    db = sqlite3.connect(str(path), timeout=30)
    db.execute("PRAGMA busy_timeout=30000")
    return db


def _verify_locked(db, key):
    rows = db.execute(
        "SELECT seq,kind,payload,previous_mac,event_mac FROM consent_events ORDER BY seq"
    ).fetchall()
    previous = GENESIS
    grants = {}
    revoked = set()
    errors = []
    for expected, (seq, kind, raw, linked, signature) in enumerate(rows, 1):
        if seq != expected:
            errors.append("missing or reordered consent event at sequence " + str(expected))
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            errors.append("invalid consent event JSON at " + str(seq))
            continue
        if not isinstance(payload, dict) or kind not in ("grant", "revoke"):
            errors.append("invalid consent event schema at " + str(seq))
            continue
        if linked != previous:
            errors.append("consent event chain link mismatch at " + str(seq))
        expected_mac = _event_mac(key, seq, kind, payload, linked)
        if not isinstance(signature, str) or not hmac.compare_digest(signature, expected_mac):
            errors.append("consent event authentication failed at " + str(seq))
        previous = signature
        grant_id = payload.get("grant_id")
        if not isinstance(grant_id, str) or not grant_id:
            errors.append("invalid grant identifier at " + str(seq))
            continue
        if kind == "grant":
            if grant_id in grants:
                errors.append("duplicate grant at " + str(seq))
            if not all(isinstance(payload.get(name), str) and len(payload[name]) == 64
                       for name in ("owner_ref", "source_ref")):
                errors.append("invalid keyed fingerprints at " + str(seq))
            if not isinstance(payload.get("purpose"), str):
                errors.append("invalid purpose at " + str(seq))
            if not isinstance(payload.get("issued_at"), (float, int)) or not isinstance(payload.get("expires_at"), (float, int)):
                errors.append("invalid grant times at " + str(seq))
            grants[grant_id] = payload
        else:
            if grant_id not in grants:
                errors.append("revocation without a prior grant at " + str(seq))
            if grant_id in revoked:
                errors.append("duplicate revocation at " + str(seq))
            if grant_id in grants and payload.get("owner_ref") != grants[grant_id].get("owner_ref"):
                errors.append("revocation owner mismatch at " + str(seq))
            revoked.add(grant_id)
    return {
        "verified": not errors,
        "errors": errors,
        "events": len(rows),
        "grants": grants,
        "revoked": revoked,
        "head_mac": previous,
    }


def verify_registry(db_path, *, key):
    _key(key)
    with _open(db_path, create=False) as db:
        result = _verify_locked(db, key)
    return {
        "verified": result["verified"],
        "errors": result["errors"],
        "events": result["events"],
        "head_mac": result["head_mac"],
        "warning": "An unanchored HMAC chain cannot detect deletion of its final events.",
    }


def _append(db_path, kind, payload, *, key):
    with _open(db_path, create=True) as db:
        db.execute("BEGIN IMMEDIATE")
        state = _verify_locked(db, key)
        if not state["verified"]:
            raise RuntimeError("consent registry integrity failed; no event written")
        seq = state["events"] + 1
        previous = state["head_mac"]
        signature = _event_mac(key, seq, kind, payload, previous)
        db.execute(
            "INSERT INTO consent_events(seq,kind,payload,previous_mac,event_mac) VALUES (?,?,?,?,?)",
            (seq, kind, _canonical(payload), previous, signature),
        )
    return {"sequence": seq, "head_mac": signature}


def grant_source(db_path, source, *, owner, purpose, ttl_seconds,
                 key, consent_asserted=False, now=None):
    """Operator records a purpose-limited grant; does not verify real consent."""
    _key(key)
    _field(owner, "owner")
    _field(purpose, "purpose")
    if not isinstance(source, str) or not source or len(source) > 25000:
        raise ValueError("source text must be nonempty and at most 25000 characters")
    if consent_asserted is not True:
        raise PermissionError("explicit operator consent attestation required")
    if (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int)
            or not 1 <= ttl_seconds <= MAX_TTL_SECONDS):
        raise ValueError("TTL must be between 1 second and 30 days")
    issued = _clock(now)
    grant_id = uuid.uuid4().hex
    payload = {
        "grant_id": grant_id,
        "owner_ref": _fingerprint(key, "owner", owner),
        "source_ref": _fingerprint(key, "source", source),
        "purpose": purpose,
        "issued_at": issued,
        "expires_at": issued + ttl_seconds,
    }
    _append(db_path, "grant", payload, key=key)
    return {
        "grant_id": grant_id,
        "expires_at": payload["expires_at"],
        "purpose": purpose,
        "source_text_stored": False,
        "consent_status": "operator_attested_not_independently_verified",
    }


def revoke_source(db_path, grant_id, *, owner, key, now=None):
    _key(key)
    _field(grant_id, "grant_id")
    _field(owner, "owner")
    revoked_at = _clock(now)
    with _open(db_path, create=False) as db:
        state = _verify_locked(db, key)
    if not state["verified"]:
        raise RuntimeError("consent registry integrity failed")
    grant = state["grants"].get(grant_id)
    if not grant or grant["owner_ref"] != _fingerprint(key, "owner", owner):
        raise PermissionError("grant not found for owner")
    if grant_id in state["revoked"]:
        raise ValueError("grant already revoked")
    _append(db_path, "revoke", {
        "grant_id": grant_id,
        "owner_ref": grant["owner_ref"],
        "revoked_at": revoked_at,
    }, key=key)
    return {"grant_id": grant_id, "revoked": True}


def authorize_source(db_path, source, *, owner, purpose, key, now=None):
    """Fail closed on missing registry, corruption, expiration or revocation."""
    _key(key)
    _field(owner, "owner")
    _field(purpose, "purpose")
    if not isinstance(source, str) or not source or len(source) > 25000:
        raise ValueError("invalid source")
    instant = _clock(now)
    try:
        with _open(db_path, create=False) as db:
            state = _verify_locked(db, key)
    except (FileNotFoundError, sqlite3.DatabaseError):
        return {"authorized": False, "reason": "registry_missing_or_unreadable"}
    if not state["verified"]:
        return {"authorized": False, "reason": "registry_integrity_failed"}
    owner_ref = _fingerprint(key, "owner", owner)
    source_ref = _fingerprint(key, "source", source)
    matches = [
        (grant_id, grant) for grant_id, grant in state["grants"].items()
        if grant["owner_ref"] == owner_ref
        and grant["source_ref"] == source_ref
        and grant["purpose"] == purpose
    ]
    if not matches:
        return {"authorized": False, "reason": "no_matching_consent_grant"}
    valid = [
        (grant_id, grant) for grant_id, grant in matches
        if grant_id not in state["revoked"]
        and grant["issued_at"] <= instant < grant["expires_at"]
    ]
    if not valid:
        return {"authorized": False, "reason": "grant_revoked_or_expired"}
    grant_id, grant = max(valid, key=lambda pair: pair[1]["issued_at"])
    return {
        "authorized": True,
        "reason": "active_purpose_scoped_grant",
        "grant_id": grant_id,
        "expires_at": grant["expires_at"],
        "consent_status": "operator_attested_not_independently_verified",
    }

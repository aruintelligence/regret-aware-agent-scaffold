"""Two independently pinned witnesses bind grader anchors to release evidence.

The verifier retains a monotonic head per cohort. This detects replay after it
has observed a newer witnessed statement; it does not provide global time or
prove that the witnesses actually operated independently.
"""
from __future__ import annotations

import hmac
import os
import sqlite3
import stat
from pathlib import Path

from anytime_risk_watch import _id
from public_grade_ledger import (
    _public_key, _sign, _verify, gate_with_public_grades,
    public_key_fingerprint,
)
from watch_outcome_ledger import _digest, _hex

GENESIS = "0" * 64
_DOMAIN = b"aru-grader-witness-v1\0"
_STATEMENT_FIELDS = {
    "kind", "schema_version", "cohort_id", "witness_epoch",
    "previous_statement_sha256", "grader_public_key_sha256",
    "grade_checkpoint_sha256", "ingress_hmac_sha256", "receipt_hmac_sha256",
    "ordered_outcomes_sha256", "model_revision", "rubric_id",
}
_APPROVAL_FIELDS = {
    "kind", "schema_version", "witness_id", "witness_public_key_sha256",
    "statement_sha256", "witness_signature_ed25519",
}


def _validate_statement(statement):
    if not isinstance(statement, dict) or set(statement) != _STATEMENT_FIELDS:
        raise ValueError("invalid witness statement fields")
    if (statement["kind"] != "aru_grader_witness_statement"
            or type(statement["schema_version"]) is not int
            or statement["schema_version"] != 1
            or type(statement["witness_epoch"]) is not int
            or statement["witness_epoch"] < 1):
        raise ValueError("invalid witness statement version or epoch")
    for field in ("cohort_id", "model_revision", "rubric_id"):
        _id(statement[field], field)
    for field in (
        "previous_statement_sha256", "grader_public_key_sha256",
        "grade_checkpoint_sha256", "ingress_hmac_sha256",
        "receipt_hmac_sha256", "ordered_outcomes_sha256",
    ):
        if not _hex(statement[field]):
            raise ValueError("invalid witness statement digest")
    if ((statement["witness_epoch"] == 1) !=
            (statement["previous_statement_sha256"] == GENESIS)):
        raise ValueError("genesis predecessor must occur only at epoch one")


def make_witness_statement(*, cohort_id, witness_epoch,
                           previous_statement_sha256, grader_public_key_sha256,
                           grade_checkpoint_sha256, ingress_hmac_sha256,
                           receipt_hmac_sha256, ordered_outcomes_sha256,
                           model_revision, rubric_id):
    statement = {
        "kind": "aru_grader_witness_statement", "schema_version": 1,
        "cohort_id": cohort_id, "witness_epoch": witness_epoch,
        "previous_statement_sha256": previous_statement_sha256,
        "grader_public_key_sha256": grader_public_key_sha256,
        "grade_checkpoint_sha256": grade_checkpoint_sha256,
        "ingress_hmac_sha256": ingress_hmac_sha256,
        "receipt_hmac_sha256": receipt_hmac_sha256,
        "ordered_outcomes_sha256": ordered_outcomes_sha256,
        "model_revision": model_revision, "rubric_id": rubric_id,
    }
    _validate_statement(statement)
    return statement


def statement_fingerprint(statement):
    _validate_statement(statement)
    return _digest(statement)


def sign_witness_statement(statement, *, witness_id, witness_private_key):
    """Sign a reviewed statement; witness operators must verify inputs first."""
    statement_sha256 = statement_fingerprint(statement)
    _id(witness_id, "witness_id")
    public_key = _public_key(witness_private_key)
    unsigned = {
        "kind": "aru_grader_witness_approval", "schema_version": 1,
        "witness_id": witness_id,
        "witness_public_key_sha256": public_key_fingerprint(public_key),
        "statement_sha256": statement_sha256,
    }
    return {**unsigned, "witness_signature_ed25519":
            _sign(unsigned, witness_private_key, _DOMAIN)}


def verify_witness_quorum(statement, approvals, *, trusted_witness_public_keys,
                          minimum_witnesses=2):
    """Require distinct signer keys pinned outside the supplied approvals."""
    digest = statement_fingerprint(statement)
    if (type(minimum_witnesses) is not int or minimum_witnesses < 2
            or not isinstance(trusted_witness_public_keys, dict)
            or len(trusted_witness_public_keys) < minimum_witnesses):
        raise ValueError("at least two pinned witness identities required")
    trusted_fingerprints = set()
    for witness_id, public_key in trusted_witness_public_keys.items():
        _id(witness_id, "witness_id")
        fingerprint = public_key_fingerprint(public_key)
        if fingerprint in trusted_fingerprints:
            raise ValueError("two witness identities share one signing key")
        trusted_fingerprints.add(fingerprint)
    if not isinstance(approvals, list) or len(approvals) < minimum_witnesses:
        raise ValueError("insufficient witness signatures")
    seen = set()
    for approval in approvals:
        if not isinstance(approval, dict) or set(approval) != _APPROVAL_FIELDS:
            raise ValueError("invalid witness approval fields")
        if (approval["kind"] != "aru_grader_witness_approval"
                or type(approval["schema_version"]) is not int
                or approval["schema_version"] != 1):
            raise ValueError("invalid witness approval version")
        witness_id = _id(approval["witness_id"], "witness_id")
        if witness_id in seen or witness_id not in trusted_witness_public_keys:
            raise ValueError("duplicate or untrusted witness")
        seen.add(witness_id)
        public_key = trusted_witness_public_keys[witness_id]
        fingerprint = public_key_fingerprint(public_key)
        if (not _hex(approval["statement_sha256"])
                or not hmac.compare_digest(approval["statement_sha256"], digest)
                or not _hex(approval["witness_public_key_sha256"])
                or not hmac.compare_digest(approval["witness_public_key_sha256"],
                                           fingerprint)):
            raise ValueError("witness statement or key pin mismatch")
        unsigned = {key: value for key, value in approval.items()
                    if key != "witness_signature_ed25519"}
        if not _verify(unsigned, approval["witness_signature_ed25519"],
                       public_key, _DOMAIN):
            raise ValueError("invalid witness signature")
    return {
        "status": "quorum_verified", "witness_count": len(seen),
        "statement_sha256": digest, "automatic_rollout_permitted": False,
    }


def record_witness_head(state_path, statement):
    """Advance a verifier-owned SQLite head atomically; refuse rollback/forks."""
    digest = statement_fingerprint(statement)
    target = Path(state_path)
    if str(target) == ":memory:" or target.is_symlink() or not target.is_file():
        raise ValueError("initialized on-disk witness state required")
    info = target.stat()
    if (not stat.S_ISREG(info.st_mode)
            or (os.name == "posix" and (
                (info.st_mode & 0o077) != 0 or info.st_uid != os.getuid()))):
        raise ValueError("witness state must be private to the verifier")
    with sqlite3.connect(str(target), timeout=10) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS witness_heads ("
            "cohort_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, "
            "statement_sha256 TEXT NOT NULL)"
        )
        db.execute("BEGIN IMMEDIATE")
        current = db.execute(
            "SELECT epoch, statement_sha256 FROM witness_heads WHERE cohort_id=?",
            (statement["cohort_id"],),
        ).fetchone()
        epoch = statement["witness_epoch"]
        previous = statement["previous_statement_sha256"]
        if current is None:
            if epoch != 1 or previous != GENESIS:
                raise ValueError("witness history must start at the genesis epoch")
            db.execute(
                "INSERT INTO witness_heads VALUES (?,?,?)",
                (statement["cohort_id"], epoch, digest),
            )
        elif epoch == current[0] and hmac.compare_digest(digest, current[1]):
            pass  # Repeat verification of the same accepted checkpoint.
        elif epoch == current[0] + 1 and hmac.compare_digest(previous, current[1]):
            db.execute(
                "UPDATE witness_heads SET epoch=?,statement_sha256=? WHERE cohort_id=?",
                (epoch, digest, statement["cohort_id"]),
            )
        else:
            raise ValueError("stale, forked or skipped witness statement")
        db.commit()
    return {"cohort_id": statement["cohort_id"], "witness_epoch": epoch,
            "statement_sha256": digest, "automatic_rollout_permitted": False}


def initialize_witness_state(state_path):
    """Host-only, exclusive creation of a private persistent rollback store."""
    target = Path(state_path)
    if str(target) == ":memory:" or not target.parent.is_dir():
        raise ValueError("persistent witness-state directory required")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    os.close(descriptor)
    with sqlite3.connect(str(target)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE witness_heads (cohort_id TEXT PRIMARY KEY, "
            "epoch INTEGER NOT NULL, statement_sha256 TEXT NOT NULL)"
        )
        db.commit()
    return str(target)


def gate_with_witnessed_public_grades(policy, *, statement, witness_approvals,
                                      trusted_witness_public_keys, witness_state_path,
                                      attestations, grade_checkpoint,
                                      grader_public_key, receipts, issued_rows,
                                      receipt_checkpoint, receipt_key,
                                      **gate_options):
    """Derive every current fingerprint from a quorum, then enforce rollback."""
    deny = {"accepted": False, "answer": "", "automatic_rollout_permitted": False}
    reserved = {
        "expected_grader_public_key_sha256",
        "expected_current_grade_checkpoint_sha256",
        "expected_current_receipt_hmac_sha256",
        "expected_current_checkpoint_hmac_sha256",
        "expected_outcomes_sha256",
    }
    if reserved.intersection(gate_options):
        return {**deny, "reason": "caller_supplied_anchor_not_permitted"}
    if (isinstance(trusted_witness_public_keys, dict)
            and grader_public_key in trusted_witness_public_keys.values()):
        return {**deny, "reason": "grader_key_reused_as_witness"}
    try:
        verify_witness_quorum(
            statement, witness_approvals,
            trusted_witness_public_keys=trusted_witness_public_keys,
        )
        ingress = gate_options["ingress_checkpoint"]
        if (statement["cohort_id"] != ingress["cohort_id"]
                or statement["cohort_id"] != grade_checkpoint["cohort_id"]
                or statement["cohort_id"] != receipt_checkpoint["cohort_id"]
                or statement["model_revision"] != ingress["model_revision"]
                or statement["rubric_id"] != ingress["rubric_id"]):
            return {**deny, "reason": "witness_cohort_scope_mismatch"}
        derived = {
            "expected_current_checkpoint_hmac_sha256": statement["ingress_hmac_sha256"],
            "expected_outcomes_sha256": statement["ordered_outcomes_sha256"],
        }
        result = gate_with_public_grades(
            policy, attestations=attestations, grade_checkpoint=grade_checkpoint,
            grader_public_key=grader_public_key,
            expected_grader_public_key_sha256=statement["grader_public_key_sha256"],
            expected_current_grade_checkpoint_sha256=statement["grade_checkpoint_sha256"],
            receipts=receipts, issued_rows=issued_rows,
            receipt_checkpoint=receipt_checkpoint, receipt_key=receipt_key,
            expected_current_receipt_hmac_sha256=statement["receipt_hmac_sha256"],
            **gate_options, **derived,
        )
        # Even a correctly witnessed alarm or failed policy moves the head.
        # The verifier must own this state path; a caller-controlled path can
        # erase or substitute history and defeats rollback protection.
        record_witness_head(witness_state_path, statement)
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error):
        return {**deny, "reason": "invalid_or_stale_witness_quorum"}
    return result

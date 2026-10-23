"""Publicly verifiable grade signatures without sharing the grader's private key.

The HMAC grader ledger remains a compatible research interface. This version
uses a separately pinned Ed25519 public key and a current checkpoint digest.
Neither signature establishes whether a grade is substantively correct.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

from anytime_risk_watch import _canonical, _id
from grade_attestation_ledger import _plain_grades
from receipt_cohort_ledger import gate_with_receipt_ledger
from watch_outcome_ledger import (
    _GRADE_FIELDS, _digest, _hex, _key, _validate_issues,
    verify_complete_report, verify_issued_checkpoint,
)

_GRADE_FIELDS_PUBLIC = _GRADE_FIELDS | {
    "kind", "schema_version", "grade_sequence", "cohort_id",
    "model_revision", "nonce", "grader_public_key_sha256",
    "grade_signature_ed25519",
}
_CHECKPOINT_FIELDS_PUBLIC = {
    "kind", "schema_version", "cohort_id", "grade_count", "last_sequence",
    "grade_manifest_sha256", "model_revision", "rubric_id",
    "grader_public_key_sha256", "grader_checkpoint_signature_ed25519",
}
_GRADE_DOMAIN = b"aru-public-grade-ed25519-v2\0"
_CHECKPOINT_DOMAIN = b"aru-public-grade-checkpoint-ed25519-v2\0"


def _public_key(private_key):
    if not isinstance(private_key, bytes) or len(private_key) != 32:
        raise ValueError("grader private key must be 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key).public_key().public_bytes_raw()


def public_key_fingerprint(public_key):
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("grader public key must be 32 raw bytes")
    Ed25519PublicKey.from_public_bytes(public_key)
    return hashlib.sha256(public_key).hexdigest()


def _sign(unsigned, private_key, domain):
    return Ed25519PrivateKey.from_private_bytes(private_key).sign(
        domain + _canonical(unsigned).encode("utf-8")
    ).hex()


def _verify(unsigned, signature, public_key, domain):
    if (not isinstance(signature, str) or len(signature) != 128
            or any(c not in "0123456789abcdef" for c in signature)):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature), domain + _canonical(unsigned).encode("utf-8")
        )
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _valid_unsigned_grade(row):
    if not isinstance(row, dict) or set(row) != _GRADE_FIELDS_PUBLIC - {"grade_signature_ed25519"}:
        return False
    try:
        if (row["kind"] != "aru_public_grade_attestation"
                or type(row["schema_version"]) is not int or row["schema_version"] != 2
                or type(row["grade_sequence"]) is not int or row["grade_sequence"] < 1
                or type(row["error"]) is not bool
                or not _hex(row["grader_public_key_sha256"])
                or not isinstance(row["nonce"], str) or len(row["nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in row["nonce"])):
            return False
        for field in ("request_id", "grader_id", "rubric_id", "cohort_id", "model_revision"):
            _id(row[field], field)
    except (TypeError, ValueError):
        return False
    return True


def sign_public_grade(grade_row, *, grade_sequence, cohort_id, model_revision,
                      grader_private_key):
    """Sign one binary grade; only this call needs the private key."""
    fingerprint = public_key_fingerprint(_public_key(grader_private_key))
    if not isinstance(grade_row, dict) or set(grade_row) != _GRADE_FIELDS:
        raise ValueError("grade requires exactly request_id, error, rubric_id, grader_id")
    unsigned = {
        "kind": "aru_public_grade_attestation", "schema_version": 2,
        "grade_sequence": grade_sequence, "cohort_id": cohort_id,
        "model_revision": model_revision, **grade_row,
        "nonce": secrets.token_hex(16), "grader_public_key_sha256": fingerprint,
    }
    if not _valid_unsigned_grade(unsigned):
        raise ValueError("invalid grade, scope or sequence")
    return {**unsigned, "grade_signature_ed25519":
            _sign(unsigned, grader_private_key, _GRADE_DOMAIN)}


def verify_public_grade(row, *, grader_public_key):
    try:
        fingerprint = public_key_fingerprint(grader_public_key)
    except (TypeError, ValueError):
        return False
    if not isinstance(row, dict) or set(row) != _GRADE_FIELDS_PUBLIC:
        return False
    unsigned = {k: v for k, v in row.items() if k != "grade_signature_ed25519"}
    return (_valid_unsigned_grade(unsigned)
            and hmac.compare_digest(row["grader_public_key_sha256"], fingerprint)
            and _verify(unsigned, row["grade_signature_ed25519"],
                        grader_public_key, _GRADE_DOMAIN))


def _validate_grades(rows, *, cohort_id, model_revision, rubric_id, public_key):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty graded log required")
    for field, value in (("cohort_id", cohort_id), ("model_revision", model_revision),
                         ("rubric_id", rubric_id)):
        _id(value, field)
    seen_ids, seen_nonces = set(), set()
    for sequence, row in enumerate(rows, 1):
        if not verify_public_grade(row, grader_public_key=public_key):
            raise ValueError("invalid public grade signature")
        if (row["grade_sequence"] != sequence or row["cohort_id"] != cohort_id
                or row["model_revision"] != model_revision or row["rubric_id"] != rubric_id):
            raise ValueError("grade arrival sequence or scope mismatch")
        if row["request_id"] in seen_ids or row["nonce"] in seen_nonces:
            raise ValueError("duplicate graded request ID or nonce")
        seen_ids.add(row["request_id"])
        seen_nonces.add(row["nonce"])


def close_public_grades(rows, *, cohort_id, model_revision, rubric_id,
                        grader_private_key):
    """Sign the closed grader log and return a checkpoint."""
    public_key = _public_key(grader_private_key)
    _validate_grades(rows, cohort_id=cohort_id, model_revision=model_revision,
                     rubric_id=rubric_id, public_key=public_key)
    unsigned = {
        "kind": "aru_public_grade_checkpoint", "schema_version": 2,
        "cohort_id": cohort_id, "grade_count": len(rows),
        "last_sequence": len(rows), "grade_manifest_sha256": _digest(rows),
        "model_revision": model_revision, "rubric_id": rubric_id,
        "grader_public_key_sha256": public_key_fingerprint(public_key),
    }
    return {**unsigned, "grader_checkpoint_signature_ed25519":
            _sign(unsigned, grader_private_key, _CHECKPOINT_DOMAIN)}


def verify_public_checkpoint(checkpoint, *, grader_public_key):
    try:
        fingerprint = public_key_fingerprint(grader_public_key)
    except (TypeError, ValueError):
        return False
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS_PUBLIC:
        return False
    unsigned = {k: v for k, v in checkpoint.items()
                if k != "grader_checkpoint_signature_ed25519"}
    try:
        if (checkpoint["kind"] != "aru_public_grade_checkpoint"
                or type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 2
                or type(checkpoint["grade_count"]) is not int
                or checkpoint["grade_count"] < 1
                or type(checkpoint["last_sequence"]) is not int
                or checkpoint["last_sequence"] != checkpoint["grade_count"]
                or not _hex(checkpoint["grade_manifest_sha256"])
                or not hmac.compare_digest(checkpoint["grader_public_key_sha256"], fingerprint)):
            return False
        for field in ("cohort_id", "model_revision", "rubric_id"):
            _id(checkpoint[field], field)
    except (TypeError, ValueError):
        return False
    return _verify(unsigned, checkpoint["grader_checkpoint_signature_ed25519"],
                   grader_public_key, _CHECKPOINT_DOMAIN)


def public_checkpoint_fingerprint(checkpoint):
    """Pin the exact signed checkpoint, not merely its public key."""
    return _digest(checkpoint)


def verify_public_grade_log(attestations, *, grade_checkpoint,
                            grader_public_key, expected_grader_public_key_sha256,
                            expected_current_grade_checkpoint_sha256):
    """Verify a closed grader log without any signing key or ingress access."""
    fingerprint = public_key_fingerprint(grader_public_key)
    if (not _hex(expected_grader_public_key_sha256)
            or not hmac.compare_digest(fingerprint, expected_grader_public_key_sha256)):
        raise ValueError("grader public key is not independently pinned")
    if not verify_public_checkpoint(grade_checkpoint, grader_public_key=grader_public_key):
        raise ValueError("invalid public grader checkpoint")
    current = public_checkpoint_fingerprint(grade_checkpoint)
    if (not _hex(expected_current_grade_checkpoint_sha256)
            or not hmac.compare_digest(current, expected_current_grade_checkpoint_sha256)):
        raise ValueError("missing or stale current grader checkpoint")
    _validate_grades(
        attestations, cohort_id=grade_checkpoint["cohort_id"],
        model_revision=grade_checkpoint["model_revision"],
        rubric_id=grade_checkpoint["rubric_id"], public_key=grader_public_key,
    )
    if (len(attestations) != grade_checkpoint["grade_count"]
            or not hmac.compare_digest(_digest(attestations),
                                       grade_checkpoint["grade_manifest_sha256"])):
        raise ValueError("graded log does not match signed checkpoint")
    return {
        "status": "valid_closed_grader_log",
        "grade_count": len(attestations), "grader_public_key_sha256": fingerprint,
        "grade_checkpoint_sha256": current, "automatic_rollout_permitted": False,
    }


def audit_public_grades(attestations, issued_rows, *, grade_checkpoint,
                        ingress_checkpoint, grader_public_key,
                        expected_grader_public_key_sha256,
                        expected_current_grade_checkpoint_sha256, ingress_key,
                        expected_current_ingress_hmac_sha256):
    """Read-only grade verification with a pinned public key and current log."""
    _key(ingress_key, "ingress")
    verified = verify_public_grade_log(
        attestations, grade_checkpoint=grade_checkpoint,
        grader_public_key=grader_public_key,
        expected_grader_public_key_sha256=expected_grader_public_key_sha256,
        expected_current_grade_checkpoint_sha256=expected_current_grade_checkpoint_sha256,
    )
    if not verify_issued_checkpoint(ingress_checkpoint, ingress_key=ingress_key):
        raise ValueError("invalid ingress checkpoint")
    if (not _hex(expected_current_ingress_hmac_sha256)
            or not hmac.compare_digest(expected_current_ingress_hmac_sha256,
                                       ingress_checkpoint["ingress_hmac_sha256"])):
        raise ValueError("missing or stale current ingress checkpoint")
    for field in ("cohort_id", "model_revision", "rubric_id"):
        if grade_checkpoint[field] != ingress_checkpoint[field]:
            raise ValueError("grade and ingress scope mismatch")
    _validate_issues(issued_rows, groups=ingress_checkpoint["groups"],
                     model_revision=ingress_checkpoint["model_revision"],
                     rubric_id=ingress_checkpoint["rubric_id"])
    if (len(issued_rows) != ingress_checkpoint["issued_count"]
            or not hmac.compare_digest(_digest(issued_rows),
                                       ingress_checkpoint["issued_manifest_sha256"])):
        raise ValueError("issued log does not match signed ingress checkpoint")
    issued_ids = {row["request_id"] for row in issued_rows}
    graded_ids = {row["request_id"] for row in attestations}
    missing, foreign = issued_ids - graded_ids, graded_ids - issued_ids
    return {
        "status": "complete" if not missing and not foreign else "conflict",
        "issued_count": len(issued_rows), "graded_count": len(attestations),
        "missing_count": len(missing), "foreign_count": len(foreign),
        "grader_public_key_sha256": verified["grader_public_key_sha256"],
        "grade_checkpoint_sha256": verified["grade_checkpoint_sha256"],
        "automatic_rollout_permitted": False,
    }


def gate_with_public_grades(policy, *, attestations, grade_checkpoint,
                            grader_public_key, expected_grader_public_key_sha256,
                            expected_current_grade_checkpoint_sha256,
                            receipts, issued_rows, receipt_checkpoint,
                            receipt_key, expected_current_receipt_hmac_sha256,
                            **gate_options):
    """Fail closed if any public grade, current anchor, watch or receipt differs."""
    deny = {"accepted": False, "answer": "", "automatic_rollout_permitted": False}
    # Publishing a public key that is also an HMAC secret would disclose that
    # authority's signing material, even though Ed25519 itself remains sound.
    if grader_public_key in (
        receipt_key, gate_options.get("ingress_key"),
        gate_options.get("watch_key"), gate_options.get("policy_key"),
    ):
        return {**deny, "reason": "public_key_reused_as_signing_secret"}
    try:
        result = audit_public_grades(
            attestations, issued_rows, grade_checkpoint=grade_checkpoint,
            ingress_checkpoint=gate_options["ingress_checkpoint"],
            grader_public_key=grader_public_key,
            expected_grader_public_key_sha256=expected_grader_public_key_sha256,
            expected_current_grade_checkpoint_sha256=expected_current_grade_checkpoint_sha256,
            ingress_key=gate_options["ingress_key"],
            expected_current_ingress_hmac_sha256=gate_options[
                "expected_current_checkpoint_hmac_sha256"],
        )
    except (TypeError, ValueError, KeyError):
        return {**deny, "reason": "invalid_or_stale_public_grader_ledger"}
    if result["status"] != "complete":
        return {**deny, "reason": "ungraded_or_foreign_public_grade",
                "missing_count": result["missing_count"],
                "foreign_count": result["foreign_count"]}
    if not verify_complete_report(
        gate_options.get("watch_report"), issued_rows, _plain_grades(attestations),
        checkpoint=gate_options["ingress_checkpoint"],
        ingress_key=gate_options["ingress_key"], watch_key=gate_options["watch_key"],
    ):
        return {**deny, "reason": "signed_watch_disagrees_with_public_grades"}
    return gate_with_receipt_ledger(
        policy, receipts=receipts, issued_rows=issued_rows,
        receipt_checkpoint=receipt_checkpoint, receipt_key=receipt_key,
        expected_current_receipt_hmac_sha256=expected_current_receipt_hmac_sha256,
        **gate_options,
    )

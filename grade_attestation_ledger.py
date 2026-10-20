"""Authenticate each independent grade and bind it to a closed cohort.

This proves that a holder of the grader key signed a label; it cannot prove
the label was correct or that the grader was organizationally independent.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from anytime_risk_watch import _canonical, _id
from receipt_cohort_ledger import gate_with_receipt_ledger
from watch_outcome_ledger import (
    _GRADE_FIELDS, _digest, _hex, _key, _validate_issues,
    verify_complete_report, verify_issued_checkpoint,
)

_GRADE_ATTESTATION_FIELDS = _GRADE_FIELDS | {
    "kind", "schema_version", "grade_sequence", "cohort_id",
    "model_revision", "nonce", "grade_hmac_sha256",
}
_CHECKPOINT_FIELDS = {
    "kind", "schema_version", "cohort_id", "grade_count", "last_sequence",
    "grade_manifest_sha256", "model_revision", "rubric_id",
    "grader_ledger_hmac_sha256",
}
_GRADE_DOMAIN = b"aru-independent-grade-v1\0"
_CHECKPOINT_DOMAIN = b"aru-grade-cohort-checkpoint-v1\0"


def _mac(unsigned, key, domain):
    return hmac.new(key, domain + _canonical(unsigned).encode("utf-8"), hashlib.sha256).hexdigest()


def _valid_unsigned_grade(row):
    if (not isinstance(row, dict)
            or set(row) != _GRADE_ATTESTATION_FIELDS - {"grade_hmac_sha256"}):
        return False
    try:
        if (row["kind"] != "aru_grade_attestation"
                or type(row["schema_version"]) is not int
                or row["schema_version"] != 1
                or type(row["grade_sequence"]) is not int
                or row["grade_sequence"] < 1
                or type(row["error"]) is not bool
                or not isinstance(row["nonce"], str)
                or len(row["nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in row["nonce"])):
            return False
        for field in ("cohort_id", "model_revision", "rubric_id", "grader_id", "request_id"):
            _id(row[field], field)
    except (ValueError, TypeError):
        return False
    return True


def attest_grade(grade_row, *, grade_sequence, cohort_id, model_revision, grader_key):
    """Sign one independently reviewed result in grader arrival order."""
    _key(grader_key, "grader")
    if not isinstance(grade_row, dict) or set(grade_row) != _GRADE_FIELDS:
        raise ValueError("grade requires exactly request_id, error, rubric_id, grader_id")
    unsigned = {
        "kind": "aru_grade_attestation", "schema_version": 1,
        "grade_sequence": grade_sequence, "cohort_id": cohort_id,
        "model_revision": model_revision, **grade_row,
        "nonce": secrets.token_hex(16),
    }
    if not _valid_unsigned_grade(unsigned):
        raise ValueError("invalid grade, scope or sequence")
    return {**unsigned, "grade_hmac_sha256": _mac(unsigned, grader_key, _GRADE_DOMAIN)}


def verify_grade_attestation(row, *, grader_key):
    if not isinstance(grader_key, bytes) or len(grader_key) < 32:
        return False
    if not isinstance(row, dict) or set(row) != _GRADE_ATTESTATION_FIELDS:
        return False
    signature = row["grade_hmac_sha256"]
    if not _hex(signature):
        return False
    unsigned = {k: v for k, v in row.items() if k != "grade_hmac_sha256"}
    return (_valid_unsigned_grade(unsigned)
            and hmac.compare_digest(signature, _mac(unsigned, grader_key, _GRADE_DOMAIN)))


def _validate_attestations(rows, *, cohort_id, model_revision, rubric_id, grader_key):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty graded log required")
    _id(cohort_id, "cohort_id")
    _id(model_revision, "model_revision")
    _id(rubric_id, "rubric_id")
    seen_ids, seen_nonces = set(), set()
    for sequence, row in enumerate(rows, 1):
        if not verify_grade_attestation(row, grader_key=grader_key):
            raise ValueError("invalid signed grade")
        if (row["grade_sequence"] != sequence
                or row["cohort_id"] != cohort_id
                or row["model_revision"] != model_revision
                or row["rubric_id"] != rubric_id):
            raise ValueError("grade arrival sequence or frozen scope mismatch")
        if row["request_id"] in seen_ids or row["nonce"] in seen_nonces:
            raise ValueError("duplicate graded request ID or nonce")
        seen_ids.add(row["request_id"])
        seen_nonces.add(row["nonce"])


def create_grade_checkpoint(rows, *, cohort_id, model_revision, rubric_id, grader_key):
    """Close a separately retained, arrival-ordered graded cohort."""
    _key(grader_key, "grader")
    _validate_attestations(rows, cohort_id=cohort_id,
                           model_revision=model_revision, rubric_id=rubric_id,
                           grader_key=grader_key)
    unsigned = {
        "kind": "aru_grade_cohort_checkpoint", "schema_version": 1,
        "cohort_id": cohort_id, "grade_count": len(rows),
        "last_sequence": len(rows), "grade_manifest_sha256": _digest(rows),
        "model_revision": model_revision, "rubric_id": rubric_id,
    }
    return {**unsigned, "grader_ledger_hmac_sha256":
            _mac(unsigned, grader_key, _CHECKPOINT_DOMAIN)}


def verify_grade_checkpoint(checkpoint, *, grader_key):
    if not isinstance(grader_key, bytes) or len(grader_key) < 32:
        return False
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        return False
    signature = checkpoint["grader_ledger_hmac_sha256"]
    if not _hex(signature):
        return False
    unsigned = {k: v for k, v in checkpoint.items() if k != "grader_ledger_hmac_sha256"}
    try:
        if not hmac.compare_digest(signature, _mac(unsigned, grader_key, _CHECKPOINT_DOMAIN)):
            return False
        if (checkpoint["kind"] != "aru_grade_cohort_checkpoint"
                or type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 1
                or type(checkpoint["grade_count"]) is not int
                or checkpoint["grade_count"] < 1
                or type(checkpoint["last_sequence"]) is not int
                or checkpoint["last_sequence"] != checkpoint["grade_count"]
                or not _hex(checkpoint["grade_manifest_sha256"])):
            return False
        for field in ("cohort_id", "model_revision", "rubric_id"):
            _id(checkpoint[field], field)
    except (ValueError, TypeError, KeyError):
        return False
    return True


def _plain_grades(attestations):
    return [{field: row[field] for field in _GRADE_FIELDS} for row in attestations]


def audit_grade_cohort(attestations, issued_rows, *, grade_checkpoint,
                       ingress_checkpoint, grader_key, ingress_key,
                       expected_current_grade_hmac_sha256,
                       expected_current_ingress_hmac_sha256):
    """Verify independent current logs and grade-ID coverage."""
    _key(grader_key, "grader")
    _key(ingress_key, "ingress")
    if hmac.compare_digest(grader_key, ingress_key):
        raise ValueError("grader and ingress keys must be independent")
    if not verify_grade_checkpoint(grade_checkpoint, grader_key=grader_key):
        raise ValueError("invalid grader checkpoint")
    if not verify_issued_checkpoint(ingress_checkpoint, ingress_key=ingress_key):
        raise ValueError("invalid ingress checkpoint")
    for expected, actual in (
        (expected_current_grade_hmac_sha256,
         grade_checkpoint["grader_ledger_hmac_sha256"]),
        (expected_current_ingress_hmac_sha256,
         ingress_checkpoint["ingress_hmac_sha256"]),
    ):
        if not _hex(expected) or not hmac.compare_digest(expected, actual):
            raise ValueError("missing or stale independent current checkpoint")
    for field in ("cohort_id", "model_revision", "rubric_id"):
        if grade_checkpoint[field] != ingress_checkpoint[field]:
            raise ValueError("grade and ingress cohort scope mismatch")
    _validate_attestations(
        attestations, cohort_id=grade_checkpoint["cohort_id"],
        model_revision=grade_checkpoint["model_revision"],
        rubric_id=grade_checkpoint["rubric_id"], grader_key=grader_key,
    )
    if (len(attestations) != grade_checkpoint["grade_count"]
            or not hmac.compare_digest(_digest(attestations),
                                       grade_checkpoint["grade_manifest_sha256"])):
        raise ValueError("graded log does not match its signed checkpoint")
    _validate_issues(issued_rows, groups=ingress_checkpoint["groups"],
                     model_revision=ingress_checkpoint["model_revision"],
                     rubric_id=ingress_checkpoint["rubric_id"])
    if (len(issued_rows) != ingress_checkpoint["issued_count"]
            or not hmac.compare_digest(_digest(issued_rows),
                                       ingress_checkpoint["issued_manifest_sha256"])):
        raise ValueError("issued log does not match its signed checkpoint")
    issued_ids = {row["request_id"] for row in issued_rows}
    graded_ids = {row["request_id"] for row in attestations}
    missing = issued_ids - graded_ids
    foreign = graded_ids - issued_ids
    return {
        "status": "complete" if not missing and not foreign else "conflict",
        "issued_count": len(issued_rows), "graded_count": len(attestations),
        "missing_count": len(missing), "foreign_count": len(foreign),
        "grader_ledger_hmac_sha256": grade_checkpoint["grader_ledger_hmac_sha256"],
        "ingress_hmac_sha256": ingress_checkpoint["ingress_hmac_sha256"],
        "automatic_rollout_permitted": False,
    }


def gate_with_authenticated_grades(policy, *, attestations, grade_checkpoint,
                                   grader_key, expected_current_grade_hmac_sha256,
                                   receipts, issued_rows, receipt_checkpoint,
                                   receipt_key, expected_current_receipt_hmac_sha256,
                                   **gate_options):
    """Require four distinct keys, exact independent grades and receipt gate."""
    deny = {"accepted": False, "answer": "", "automatic_rollout_permitted": False}
    try:
        _key(grader_key, "grader")
        _key(receipt_key, "receipt")
        _key(gate_options["watch_key"], "watch")
        if (grader_key == receipt_key or grader_key == gate_options["watch_key"]):
            return {**deny, "reason": "grader_key_not_independent"}
        result = audit_grade_cohort(
            attestations, issued_rows, grade_checkpoint=grade_checkpoint,
            ingress_checkpoint=gate_options["ingress_checkpoint"],
            grader_key=grader_key, ingress_key=gate_options["ingress_key"],
            expected_current_grade_hmac_sha256=expected_current_grade_hmac_sha256,
            expected_current_ingress_hmac_sha256=gate_options[
                "expected_current_checkpoint_hmac_sha256"],
        )
    except (ValueError, TypeError, KeyError):
        return {**deny, "reason": "invalid_or_stale_grader_ledger"}
    if result["status"] != "complete":
        return {**deny, "reason": "ungraded_or_foreign_attested_grade",
                "missing_count": result["missing_count"],
                "foreign_count": result["foreign_count"]}
    if not verify_complete_report(
        gate_options.get("watch_report"), issued_rows, _plain_grades(attestations),
        checkpoint=gate_options["ingress_checkpoint"],
        ingress_key=gate_options["ingress_key"],
        watch_key=gate_options["watch_key"],
    ):
        return {**deny, "reason": "signed_watch_disagrees_with_authenticated_grades"}
    return gate_with_receipt_ledger(
        policy, receipts=receipts, issued_rows=issued_rows,
        receipt_checkpoint=receipt_checkpoint, receipt_key=receipt_key,
        expected_current_receipt_hmac_sha256=expected_current_receipt_hmac_sha256,
        **gate_options,
    )

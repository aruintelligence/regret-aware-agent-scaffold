"""Independent receipt-cohort accounting for the closed-cohort release gate.

The receipt issuer retains its own ordered log and signs its own checkpoint.
Both checkpoints must be current and externally retained. The ledger does not
establish independent operation merely because distinct keys were used.
"""
from __future__ import annotations

import hashlib
import hmac

from anytime_risk_watch import _canonical, _id
from client_request_receipt import verify_request_receipt
from watch_outcome_ledger import (
    _ISSUE_FIELDS, _digest, _hex, _key, _validate_issues,
    gate_with_complete_ledger, verify_issued_checkpoint,
)

_FIELDS = {
    "kind", "schema_version", "cohort_id", "receipt_count", "last_sequence",
    "receipt_manifest_sha256", "groups", "model_revision", "rubric_id",
    "receipt_ledger_hmac_sha256",
}
_DOMAIN = b"aru-receipt-cohort-checkpoint-v1\0"


def _mac(value, key):
    return hmac.new(key, _DOMAIN + _canonical(value).encode("utf-8"), hashlib.sha256).hexdigest()


def _validate_receipts(receipts, *, cohort_id, groups, model_revision, rubric_id,
                       receipt_key):
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("nonempty receipt log required")
    if (not isinstance(groups, list) or not groups
            or any(not isinstance(g, str) for g in groups)
            or len(set(groups)) != len(groups) or "__all__" in groups):
        raise ValueError("distinct predeclared receipt groups required")
    _id(cohort_id, "cohort_id")
    _id(model_revision, "model_revision")
    _id(rubric_id, "rubric_id")
    for group in groups:
        _id(group, "group")
    seen_ids, seen_nonces = set(), set()
    for sequence, receipt in enumerate(receipts, 1):
        if not verify_request_receipt(receipt, receipt_key=receipt_key):
            raise ValueError("invalid signed receipt")
        if (receipt["sequence"] != sequence
                or receipt["cohort_id"] != cohort_id
                or receipt["group"] not in groups
                or receipt["model_revision"] != model_revision
                or receipt["rubric_id"] != rubric_id):
            raise ValueError("receipt sequence or frozen scope mismatch")
        if receipt["request_id"] in seen_ids or receipt["nonce"] in seen_nonces:
            raise ValueError("duplicate receipt request ID or nonce")
        seen_ids.add(receipt["request_id"])
        seen_nonces.add(receipt["nonce"])


def create_receipt_checkpoint(receipts, *, cohort_id, groups, model_revision,
                              rubric_id, receipt_key):
    """Close the issuer's independent receipt log before outcomes are known."""
    _key(receipt_key, "receipt")
    _validate_receipts(receipts, cohort_id=cohort_id, groups=groups,
                       model_revision=model_revision, rubric_id=rubric_id,
                       receipt_key=receipt_key)
    value = {
        "kind": "aru_receipt_cohort_checkpoint", "schema_version": 1,
        "cohort_id": cohort_id, "receipt_count": len(receipts),
        "last_sequence": len(receipts),
        "receipt_manifest_sha256": _digest(receipts),
        "groups": list(groups), "model_revision": model_revision,
        "rubric_id": rubric_id,
    }
    return {**value, "receipt_ledger_hmac_sha256": _mac(value, receipt_key)}


def verify_receipt_checkpoint(checkpoint, *, receipt_key):
    if not isinstance(receipt_key, bytes) or len(receipt_key) < 32:
        return False
    if not isinstance(checkpoint, dict) or set(checkpoint) != _FIELDS:
        return False
    signature = checkpoint["receipt_ledger_hmac_sha256"]
    if not _hex(signature):
        return False
    unsigned = {key: value for key, value in checkpoint.items()
                if key != "receipt_ledger_hmac_sha256"}
    try:
        if not hmac.compare_digest(signature, _mac(unsigned, receipt_key)):
            return False
        if (checkpoint["kind"] != "aru_receipt_cohort_checkpoint"
                or type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 1
                or type(checkpoint["receipt_count"]) is not int
                or checkpoint["receipt_count"] < 1
                or type(checkpoint["last_sequence"]) is not int
                or checkpoint["last_sequence"] != checkpoint["receipt_count"]
                or not _hex(checkpoint["receipt_manifest_sha256"])):
            return False
        _id(checkpoint["cohort_id"], "cohort_id")
        _id(checkpoint["model_revision"], "model_revision")
        _id(checkpoint["rubric_id"], "rubric_id")
        groups = checkpoint["groups"]
        if (not isinstance(groups, list) or not groups
                or any(not isinstance(g, str) for g in groups)
                or len(set(groups)) != len(groups) or "__all__" in groups):
            return False
        for group in groups:
            _id(group, "group")
    except (ValueError, TypeError, KeyError):
        return False
    return True


def reconcile_receipt_cohort(receipts, issued_rows, *, receipt_checkpoint,
                             ingress_checkpoint, receipt_key, ingress_key,
                             expected_current_receipt_hmac_sha256,
                             expected_current_ingress_hmac_sha256):
    """Require two independently current, intact logs; report any conflict."""
    _key(receipt_key, "receipt")
    _key(ingress_key, "ingress")
    if hmac.compare_digest(receipt_key, ingress_key):
        raise ValueError("receipt and ingress keys must be independent")
    if not verify_receipt_checkpoint(receipt_checkpoint, receipt_key=receipt_key):
        raise ValueError("invalid receipt checkpoint")
    if not verify_issued_checkpoint(ingress_checkpoint, ingress_key=ingress_key):
        raise ValueError("invalid ingress checkpoint")
    for expected, actual in (
        (expected_current_receipt_hmac_sha256,
         receipt_checkpoint["receipt_ledger_hmac_sha256"]),
        (expected_current_ingress_hmac_sha256,
         ingress_checkpoint["ingress_hmac_sha256"]),
    ):
        if not _hex(expected) or not hmac.compare_digest(expected, actual):
            raise ValueError("missing or stale independent current checkpoint")
    for field in ("cohort_id", "groups", "model_revision", "rubric_id"):
        if receipt_checkpoint[field] != ingress_checkpoint[field]:
            raise ValueError("receipt and ingress cohort scope mismatch")
    _validate_receipts(
        receipts, cohort_id=receipt_checkpoint["cohort_id"],
        groups=receipt_checkpoint["groups"],
        model_revision=receipt_checkpoint["model_revision"],
        rubric_id=receipt_checkpoint["rubric_id"], receipt_key=receipt_key,
    )
    if (len(receipts) != receipt_checkpoint["receipt_count"]
            or not hmac.compare_digest(_digest(receipts),
                                       receipt_checkpoint["receipt_manifest_sha256"])):
        raise ValueError("receipt log does not match its signed checkpoint")
    _validate_issues(
        issued_rows, groups=ingress_checkpoint["groups"],
        model_revision=ingress_checkpoint["model_revision"],
        rubric_id=ingress_checkpoint["rubric_id"],
    )
    if (len(issued_rows) != ingress_checkpoint["issued_count"]
            or not hmac.compare_digest(_digest(issued_rows),
                                       ingress_checkpoint["issued_manifest_sha256"])):
        raise ValueError("issued log does not match its signed checkpoint")
    conflicts = [receipt["sequence"] for receipt in receipts
                 if (receipt["sequence"] > len(issued_rows)
                     or any(receipt[field] != issued_rows[receipt["sequence"] - 1][field]
                            for field in _ISSUE_FIELDS))]
    # Extra ingress issues are also a count conflict, even when every receipt
    # matches a prefix. The signed receipt ledger never blesses unreceipted work.
    status = "complete" if not conflicts and len(receipts) == len(issued_rows) else "conflict"
    return {
        "status": status, "receipt_count": len(receipts),
        "issued_count": len(issued_rows), "conflict_sequences": conflicts,
        "receipt_ledger_hmac_sha256": receipt_checkpoint["receipt_ledger_hmac_sha256"],
        "ingress_hmac_sha256": ingress_checkpoint["ingress_hmac_sha256"],
        "automatic_rollout_permitted": False,
    }


def gate_with_receipt_ledger(policy, *, receipts, issued_rows, receipt_checkpoint,
                             receipt_key, expected_current_receipt_hmac_sha256,
                             **gate_options):
    """Require complete receipt/ingress agreement before the existing gate."""
    deny = {"accepted": False, "answer": "", "automatic_rollout_permitted": False}
    try:
        watch_key = gate_options["watch_key"]
        if (not isinstance(watch_key, bytes) or len(watch_key) < 32
                or hmac.compare_digest(watch_key, receipt_key)):
            return {**deny, "reason": "receipt_and_watch_keys_not_independent"}
        result = reconcile_receipt_cohort(
            receipts, issued_rows, receipt_checkpoint=receipt_checkpoint,
            ingress_checkpoint=gate_options["ingress_checkpoint"],
            receipt_key=receipt_key, ingress_key=gate_options["ingress_key"],
            expected_current_receipt_hmac_sha256=expected_current_receipt_hmac_sha256,
            expected_current_ingress_hmac_sha256=gate_options[
                "expected_current_checkpoint_hmac_sha256"],
        )
    except (ValueError, TypeError, KeyError):
        return {**deny, "reason": "invalid_or_stale_receipt_ledger"}
    if result["status"] != "complete":
        return {**deny, "reason": "receipt_ingress_conflict",
                "conflict_sequences": result["conflict_sequences"]}
    return gate_with_complete_ledger(policy, **gate_options)

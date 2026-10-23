"""Customer-held request receipts for challenging a closed issued cohort.

The receipt issuer must be separate from the operator who closes the ingress
manifest. A receipt proves only what its issuer signed; the challenge compares
it with the independently retained CURRENT checkpoint for the same cohort.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from anytime_risk_watch import _canonical, _id
from watch_outcome_ledger import (
    _ISSUE_FIELDS, _digest, _hex, _key, _validate_issues,
    verify_issued_checkpoint,
)

_RECEIPT_FIELDS = _ISSUE_FIELDS | {
    "kind", "schema_version", "cohort_id", "nonce", "receipt_hmac_sha256",
}
_DOMAIN = b"aru-request-receipt-v1\0"


def _valid_unsigned_receipt(value):
    if not isinstance(value, dict) or set(value) != _RECEIPT_FIELDS - {"receipt_hmac_sha256"}:
        return False
    try:
        if (value["kind"] != "aru_request_receipt"
                or type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or type(value["sequence"]) is not int
                or value["sequence"] < 1
                or not isinstance(value["nonce"], str)
                or len(value["nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in value["nonce"])):
            return False
        for field in ("cohort_id", "request_id", "group", "model_revision", "rubric_id"):
            _id(value[field], field)
        if value["group"] == "__all__":
            return False
    except (ValueError, TypeError):
        return False
    return True


def _signature(unsigned, key):
    return hmac.new(key, _DOMAIN + _canonical(unsigned).encode("utf-8"), hashlib.sha256).hexdigest()


def issue_request_receipt(issue_row, *, cohort_id, receipt_key):
    """Sign a single issued row before inference; deliver to the requester."""
    _key(receipt_key, "receipt")
    if not isinstance(issue_row, dict) or set(issue_row) != _ISSUE_FIELDS:
        raise ValueError("receipt requires exactly one issued manifest row")
    unsigned = {
        "kind": "aru_request_receipt", "schema_version": 1,
        "cohort_id": cohort_id, **issue_row, "nonce": secrets.token_hex(16),
    }
    if not _valid_unsigned_receipt(unsigned):
        raise ValueError("invalid receipt issue row or cohort")
    return {**unsigned, "receipt_hmac_sha256": _signature(unsigned, receipt_key)}


def verify_request_receipt(receipt, *, receipt_key):
    """Check exact fields and the domain-separated, constant-time MAC."""
    if not isinstance(receipt_key, bytes) or len(receipt_key) < 32:
        return False
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        return False
    signature = receipt["receipt_hmac_sha256"]
    if not _hex(signature):
        return False
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_hmac_sha256"}
    return (_valid_unsigned_receipt(unsigned)
            and hmac.compare_digest(signature, _signature(unsigned, receipt_key)))


def challenge_request_receipt(receipt, issued_rows, checkpoint, *, receipt_key,
                              ingress_key, expected_current_checkpoint_hmac_sha256):
    """Compare a held receipt with the current authenticated closed manifest.

    Raises ValueError for unverifiable or stale evidence. Returns a definite
    inclusion or omission finding only for a verified, current closed cohort.
    The expected HMAC MUST come from an independent current-checkpoint store.
    """
    _key(receipt_key, "receipt")
    _key(ingress_key, "ingress")
    if hmac.compare_digest(receipt_key, ingress_key):
        raise ValueError("receipt and ingress keys must be independent")
    if not verify_request_receipt(receipt, receipt_key=receipt_key):
        raise ValueError("invalid customer request receipt")
    if not verify_issued_checkpoint(checkpoint, ingress_key=ingress_key):
        raise ValueError("invalid closed ingress checkpoint")
    if (not _hex(expected_current_checkpoint_hmac_sha256)
            or not hmac.compare_digest(expected_current_checkpoint_hmac_sha256,
                                       checkpoint["ingress_hmac_sha256"])):
        raise ValueError("missing or stale current ingress checkpoint")
    _validate_issues(
        issued_rows, groups=checkpoint["groups"],
        model_revision=checkpoint["model_revision"], rubric_id=checkpoint["rubric_id"],
    )
    if (len(issued_rows) != checkpoint["issued_count"]
            or not hmac.compare_digest(_digest(issued_rows), checkpoint["issued_manifest_sha256"])):
        raise ValueError("issued manifest does not match closed checkpoint")
    if (receipt["cohort_id"] != checkpoint["cohort_id"]
            or receipt["model_revision"] != checkpoint["model_revision"]
            or receipt["rubric_id"] != checkpoint["rubric_id"]
            or receipt["group"] not in checkpoint["groups"]):
        raise ValueError("receipt belongs to another cohort or scope")

    position = receipt["sequence"] - 1
    included = (position < len(issued_rows)
                and all(receipt[field] == issued_rows[position][field]
                        for field in _ISSUE_FIELDS))
    return {
        "cohort_id": receipt["cohort_id"], "sequence": receipt["sequence"],
        "receipt_sha256": _digest(receipt),
        "checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
        "included": included,
        "reason": "included" if included else "authenticated_receipt_missing_or_rewritten",
        "automatic_rollout_permitted": False,
    }

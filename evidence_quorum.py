"""Deterministic multi-source evidence quorum with conflict quarantine.

Accepted answers must be verified contiguous source spans in at least k
DISTINCT source fingerprints with no conflicting verified candidate.
Matching documents can copy one another or be wrong: this is not proof
of factual truth, entailment, epistemic independence, or source trust.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

from grounding_contract import ground_answer, verify_grounding

MAX_CANDIDATES = 16


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key(answer):
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def evidence_quorum(candidates, *, min_distinct_sources=2,
                    approved_source_sha256=None, source_approval_kind="operator_allowlist"):
    """Issue a deterministic source-backed quorum receipt or abstain.

    Each candidate must have exactly {source_id, context, answer}.
    The model never supplies trusted proofs: source spans are verified here.
    An optional OPERATOR-controlled SHA-256 allowlist restricts source input.
    """
    if (isinstance(min_distinct_sources, bool)
            or not isinstance(min_distinct_sources, int)
            or not 1 <= min_distinct_sources <= MAX_CANDIDATES):
        raise ValueError("min_distinct_sources must be an integer in [1,16]")
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("candidates must be a nonempty list of at most 16 items")
    if source_approval_kind not in ("operator_allowlist", "purpose_scoped_registry"):
        raise ValueError("unsupported source approval kind")
    if approved_source_sha256 is not None:
        if not isinstance(approved_source_sha256, (set, frozenset, list, tuple)) or any(
            not isinstance(h, str) or len(h) != 64
            or any(ch not in "0123456789abcdef" for ch in h)
            for h in approved_source_sha256
        ):
            raise ValueError("approved_source_sha256 must contain lowercase SHA-256 digests")
        approved_source_sha256 = frozenset(approved_source_sha256)

    groups = {}
    source_ids = {}
    evidence = []
    unsupported = 0
    first_answer = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"source_id", "context", "answer"}:
            raise ValueError("candidate must contain exactly source_id, context, answer")
        source_id, context, answer = (
            candidate[field] for field in ("source_id", "context", "answer")
        )
        if not isinstance(source_id, str) or not 1 <= len(source_id) <= 128 or "\n" in source_id:
            raise ValueError("source_id must be short, nonempty text")
        if not isinstance(context, str) or not isinstance(answer, str):
            raise ValueError("context and answer must be strings")
        digest = _hash(context)
        if source_id in source_ids and source_ids[source_id] != digest:
            raise ValueError("source ID reused with different source contents")
        source_ids[source_id] = digest
        if approved_source_sha256 is not None and digest not in approved_source_sha256:
            raise ValueError("source fingerprint not in operator-approved allowlist")
        proof = ground_answer(answer, context)
        if not proof["accepted"]:
            unsupported += 1
            continue
        if not verify_grounding(proof, context):
            raise RuntimeError("source proof failed independent verification")
        key = _key(proof["answer"])
        groups.setdefault(key, set()).add(digest)
        first_answer.setdefault(key, proof["answer"])
        evidence.append({
            "source_id_sha256": _hash(source_id),
            "source_sha256": digest,
            "answer_sha256": proof["answer_sha256"],
            "span_start": proof["span_start"],
            "span_end": proof["span_end"],
            "normalized_answer_sha256": _hash(key),
        })

    conflict = len(groups) > 1
    agreement = max((len(hashes) for hashes in groups.values()), default=0)
    accepted = not conflict and agreement >= min_distinct_sources
    answer = first_answer[next(iter(groups))] if accepted else ""
    reason = (
        "conflicting_verified_source_answers" if conflict else
        "verified_distinct_source_quorum" if accepted else
        "no_supported_source_answers" if not groups else
        "insufficient_distinct_source_quorum"
    )
    receipt = {
        "schema_version": 1,
        "accepted": accepted,
        "answer": answer,
        "reason": reason,
        "minimum_distinct_source_fingerprints": min_distinct_sources,
        "verified_candidate_count": len(evidence),
        "unsupported_candidate_count": unsupported,
        "distinct_verified_source_fingerprints": len({
            item["source_sha256"] for item in evidence
        }),
        "conflicting_answer_groups": len(groups),
        "winning_distinct_source_fingerprints": agreement if accepted else 0,
        "evidence": evidence,
        "source_approval": source_approval_kind if approved_source_sha256 is not None else "unverified",
        "automatic_rollout_permitted": False,
        "warning": "Textual agreement is not truth, entailment, source trust or epistemic independence.",
    }
    receipt["receipt_sha256"] = _hash(_canonical(receipt))
    return receipt


def verify_quorum(receipt, candidates, *, approved_source_sha256=None,
                  source_approval_kind="operator_allowlist"):
    """Independently recompute the entire receipt; a hash is NOT a signature."""
    if not isinstance(receipt, dict) or "receipt_sha256" not in receipt:
        return False
    try:
        expected = evidence_quorum(
            candidates,
            min_distinct_sources=receipt["minimum_distinct_source_fingerprints"],
            approved_source_sha256=approved_source_sha256,
            source_approval_kind=source_approval_kind,
        )
    except (ValueError, RuntimeError, KeyError, TypeError):
        return False
    return _canonical(expected) == _canonical(receipt)

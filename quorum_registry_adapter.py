"""Consent-scoped source registry adapter for multi-document QA quorum.

Authorization is checked before AND after inference to reduce revocation
race exposure. This cannot make revocation atomic with an HTTP response;
the operator-attested registry is not verified legal consent.
"""
from __future__ import annotations

import hashlib

from evidence_quorum import evidence_quorum, verify_quorum
from grounding_contract import verify_grounding
from sovereign_source_registry import authorize_source
from source_registry_anchor import verify_source_checkpoint


def evaluate_registered_quorum(engine, *, registry_db, registry_key,
                               owner, question, sources,
                               min_distinct_sources=2, max_answer_tokens=30,
                               registry_checkpoint=None, registry_anchor_key=None):
    """Fail closed unless all documents have active purpose-scoped grants.

    The returned receipt contains no raw passages and never authorizes
    production rollout. Contradictory verified answers force abstention.
    """
    if registry_db is None or registry_key is None:
        raise PermissionError("consent-scoped source registry required")
    if registry_checkpoint is not None and (
        not isinstance(registry_anchor_key, bytes)
        or len(registry_anchor_key) < 32
    ):
        raise PermissionError("separate registry anchor key required")
    if not isinstance(owner, str) or not 0 < len(owner) <= 128:
        raise ValueError("valid owner identifier required")
    if not isinstance(question, str) or not 0 < len(question) <= 2000:
        raise ValueError("question must be bounded nonempty text")
    if not isinstance(sources, list) or not 2 <= len(sources) <= 4:
        raise ValueError("two to four sources required")
    if (isinstance(min_distinct_sources, bool)
            or not isinstance(min_distinct_sources, int)
            or not 2 <= min_distinct_sources <= len(sources)):
        raise ValueError("invalid quorum threshold")
    if (isinstance(max_answer_tokens, bool)
            or not isinstance(max_answer_tokens, int)
            or not 1 <= max_answer_tokens <= 64):
        raise ValueError("invalid answer token limit")
    seen_ids = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"source_id", "context"}:
            raise ValueError("source must contain exactly source_id and context")
        source_id, context = source["source_id"], source["context"]
        if (not isinstance(source_id, str) or not 0 < len(source_id) <= 128
                or any(ord(c) < 32 for c in source_id) or source_id in seen_ids):
            raise ValueError("invalid or duplicate source ID")
        seen_ids.add(source_id)
        if not isinstance(context, str) or not 0 < len(context) <= 5000:
            raise ValueError("source text must be nonempty and at most 5000 characters")

    def authorize_all():
        if registry_checkpoint is not None:
            try:
                anchored = verify_source_checkpoint(
                    registry_db, registry_checkpoint,
                    registry_key=registry_key, anchor_key=registry_anchor_key,
                )
            except (OSError, ValueError) as error:
                raise PermissionError("source checkpoint unavailable") from error
            if not anchored["verified"]:
                raise PermissionError("source checkpoint failed verification")
        for source in sources:
            result = authorize_source(
                registry_db, source["context"], owner=owner,
                purpose="qa_source", key=registry_key,
            )
            if not result["authorized"]:
                raise PermissionError("source_not_authorized: " + result["reason"])

    # Check ALL grants before processing ANY document.
    authorize_all()
    candidates = []
    for source in sources:
        answer = engine.answer(
            source["context"], question,
            max_answer_tokens=max_answer_tokens,
        )
        if answer["accepted"] and not verify_grounding(answer["proof"], source["context"]):
            raise RuntimeError("invalid model source certificate")
        candidates.append({
            "source_id": source["source_id"],
            "context": source["context"],
            "answer": answer["answer"] if answer["accepted"] else "",
        })
    # Recheck grants after inference; revocation or expiry blocks display.
    authorize_all()
    approved = {
        hashlib.sha256(source["context"].encode("utf-8")).hexdigest()
        for source in sources
    }
    receipt = evidence_quorum(
        candidates, min_distinct_sources=min_distinct_sources,
        approved_source_sha256=approved,
        source_approval_kind="purpose_scoped_registry",
    )
    if not verify_quorum(
        receipt, candidates, approved_source_sha256=approved,
        source_approval_kind="purpose_scoped_registry",
    ):
        raise RuntimeError("independent quorum verification failed")
    return receipt

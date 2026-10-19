"""Source-grounded answer contract: every accepted answer carries a verifiable span.

This is a deterministic provenance guard for short extractive answers.
It does NOT prove an answer is relevant, correct, safe, or entailed by the
source. It only proves the displayed answer occurs as a contiguous span
in the supplied source text (with optional case-insensitive matching).

Raw source passages are never stored in the returned proof. The consumer
must retain the source separately to independently verify it.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata


MAX_ANSWER_CHARS = 200
MAX_CONTEXT_CHARS = 25000


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_answer(model_text):
    if not isinstance(model_text, str):
        raise ValueError("model answer must be text")
    value = model_text.strip()
    if not value or len(value) > MAX_ANSWER_CHARS:
        return ""
    if "\n" in value or "\r" in value or "\x00" in value:
        return ""
    # Remove only a small, explicit list of answer-format wrappers.
    value = re.sub(r"^(?:answer|final answer)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip("\"'“”‘’").strip()
    value = value.rstrip(".!?").strip()
    return value if 0 < len(value) <= MAX_ANSWER_CHARS else ""


def _is_word_character(character):
    return bool(character) and (
        character.isalnum() or unicodedata.category(character).startswith("M")
    )


def _word_bounded(context, start, end):
    # Prevent accepting "on" in "London" as an independent answer.
    if start > 0 and _is_word_character(context[start - 1]) and _is_word_character(context[start]):
        return False
    if end < len(context) and _is_word_character(context[end - 1]) and _is_word_character(context[end]):
        return False
    return True


def ground_answer(model_text, context, *, model_revision=None):
    """Return a proof-carrying answer or a transparent abstention.

    The proof contains only hashes and character offsets, not the passage.
    It deliberately does not use the human-gold answer during grounding.
    """
    if not isinstance(context, str) or not context or len(context) > MAX_CONTEXT_CHARS:
        raise ValueError("context must be nonempty text within 25000 characters")
    answer = _candidate_answer(model_text)
    base = {
        "schema_version": 1,
        "source_sha256": _sha256(context),
        "model_revision": model_revision,
        "verification_scope": "contiguous_source_substring_only",
    }
    if not answer:
        return {
            **base, "accepted": False, "answer": "",
            "reason": "empty_oversized_or_unstructured_answer",
            "span_start": None, "span_end": None,
        }
    # Prefer an exact, word-bounded match; then allow case-insensitive
    # matching while returning the source's actual original spelling.
    matches = list(re.finditer(re.escape(answer), context))
    matches += list(re.finditer(re.escape(answer), context, flags=re.IGNORECASE))
    for match in matches:
        start, end = match.span()
        if not _word_bounded(context, start, end):
            continue
        extracted = context[start:end]
        return {
            **base,
            "accepted": True,
            "answer": extracted,
            "answer_sha256": _sha256(extracted),
            "span_start": start,
            "span_end": end,
            "reason": "exact_source_span" if extracted == answer else "case_insensitive_source_span",
        }
    return {
        **base, "accepted": False, "answer": "",
        "reason": "model_answer_not_found_as_source_span",
        "span_start": None, "span_end": None,
    }


def verify_grounding(proof, context):
    """Recheck a certificate against independently supplied source text."""
    if not isinstance(proof, dict) or not isinstance(context, str):
        return False
    if proof.get("schema_version") != 1 or not proof.get("accepted"):
        return False
    if proof.get("verification_scope") != "contiguous_source_substring_only":
        return False
    start, end = proof.get("span_start"), proof.get("span_end")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end > len(context) or start >= end):
        return False
    extracted = context[start:end]
    return (
        proof.get("source_sha256") == _sha256(context)
        and proof.get("answer") == extracted
        and proof.get("answer_sha256") == _sha256(extracted)
        and _word_bounded(context, start, end)
    )

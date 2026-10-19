"""Real extractive QA model with cryptographically checkable source spans.

Uses pretrained question-answering weights on CPU, not a scripted
retriever or fake model. The model proposes a span using start/end logits;
a separate deterministic verifier checks its original source offsets and
hash. A valid span proves textual provenance, NOT factual truth or
relevance. Optional heavyweight dependencies load only on explicit use.
"""
from __future__ import annotations

import math
import threading

from grounding_contract import MAX_CONTEXT_CHARS, ground_answer, verify_grounding

DEFAULT_QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


def best_context_span(context, offsets, sequence_ids, start_logits, end_logits,
                      *, max_answer_tokens=30):
    """Choose the highest-scoring legal context span from one token window.

    Context token offsets refer to original source characters. Question,
    padding and special tokens are never eligible. Scores are raw model
    logits, NOT calibrated probabilities.
    """
    if not isinstance(context, str) or not context:
        raise ValueError("nonempty context required")
    n = len(offsets)
    if n != len(sequence_ids) or n != len(start_logits) or n != len(end_logits):
        raise ValueError("token arrays must have equal lengths")
    if isinstance(max_answer_tokens, bool) or not isinstance(max_answer_tokens, int) or not 1 <= max_answer_tokens <= 64:
        raise ValueError("max_answer_tokens must be in [1,64]")
    valid = []
    for i, (span, role) in enumerate(zip(offsets, sequence_ids)):
        if role != 1 or not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        a, b = span
        if (isinstance(a, bool) or isinstance(b, bool)
                or not isinstance(a, int) or not isinstance(b, int)
                or not 0 <= a < b <= len(context)):
            continue
        if not math.isfinite(float(start_logits[i])) or not math.isfinite(float(end_logits[i])):
            continue
        valid.append(i)
    best = None
    for i in valid:
        start = offsets[i][0]
        for j in range(i, min(i + max_answer_tokens, n)):
            if j not in valid:
                continue
            end = offsets[j][1]
            if end <= start or end - start > 200:
                continue
            score = float(start_logits[i]) + float(end_logits[j])
            # Prefer shorter spans for exactly tied scores.
            candidate = (score, -(end - start), -start, start, end)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    return {"start": best[3], "end": best[4], "logit_score": best[0]}


class ExtractiveQAEngine:
    def __init__(self, model_id=DEFAULT_QA_MODEL, *, revision=None,
                 local_files_only=False, threads=2):
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model ID required")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 32:
            raise ValueError("threads must be in [1,32]")
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        torch.set_num_threads(threads)
        kwargs = {"trust_remote_code": False, "local_files_only": local_files_only}
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)
        if not self.tokenizer.is_fast:
            raise ValueError("fast tokenizer required for original source offsets")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, use_safetensors=True, **kwargs,
        )
        self.model.eval()
        self.torch = torch
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.lock = threading.Lock()

    def answer(self, context, question, *, max_length=384,
               stride=96, max_answer_tokens=30):
        if not isinstance(context, str) or not 0 < len(context) <= MAX_CONTEXT_CHARS:
            raise ValueError("context length invalid")
        if not isinstance(question, str) or not 0 < len(question) <= 2000:
            raise ValueError("question length invalid")
        if not 128 <= max_length <= 512 or not 0 <= stride < max_length // 2:
            raise ValueError("invalid token window configuration")
        with self.lock, self.torch.inference_mode():
            encoded = self.tokenizer(
                question, context,
                truncation="only_second",
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            if len(encoded["input_ids"]) > 32:
                raise ValueError("too many token windows; source must be shortened")
            best = None
            for window in range(len(encoded["input_ids"])):
                inputs = {
                    key: self.torch.tensor([encoded[key][window]])
                    for key in ("input_ids", "attention_mask", "token_type_ids")
                    if key in encoded
                }
                result = self.model(**inputs)
                span = best_context_span(
                    context, encoded["offset_mapping"][window],
                    encoded.sequence_ids(window),
                    result.start_logits[0].tolist(),
                    result.end_logits[0].tolist(),
                    max_answer_tokens=max_answer_tokens,
                )
                if span is not None and (
                    best is None or span["logit_score"] > best["logit_score"]
                ):
                    best = span
        if best is None:
            return {
                "accepted": False, "answer": "", "proof": None,
                "model": self.model_id, "model_revision": self.revision,
                "reason": "no_legal_context_span",
            }
        raw_answer = context[best["start"]:best["end"]]
        proof = ground_answer(raw_answer, context, model_revision=self.revision)
        # Requiring exactly the predicted offsets avoids a repeated-phrase
        # ambiguity where a different occurrence would otherwise be certified.
        if not proof["accepted"] or proof["span_start"] != best["start"]:
            # Build a direct, independently verifiable certificate for the
            # model-selected occurrence; do not silently pick a different one.
            from grounding_contract import _sha256, _word_bounded
            if not _word_bounded(context, best["start"], best["end"]):
                return {
                    "accepted": False, "answer": "", "proof": None,
                    "model": self.model_id, "model_revision": self.revision,
                    "reason": "model_span_not_word_bounded",
                }
            proof = {
                "schema_version": 1,
                "source_sha256": _sha256(context),
                "model_revision": self.revision,
                "verification_scope": "contiguous_source_substring_only",
                "accepted": True,
                "answer": raw_answer,
                "answer_sha256": _sha256(raw_answer),
                "span_start": best["start"],
                "span_end": best["end"],
                "reason": "model_selected_exact_source_span",
            }
        if not verify_grounding(proof, context):
            raise RuntimeError("independent source proof verification failed")
        return {
            "accepted": True, "answer": raw_answer, "proof": proof,
            "model": self.model_id, "model_revision": self.revision,
            "reason": "model_selected_verified_source_span",
            "uncalibrated_span_logit_score": best["logit_score"],
            "warning": "Source span does not prove answer correctness or entailment.",
        }

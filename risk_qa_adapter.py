"""Bind real extractive QA model outputs to a signed selective-risk gate.

The QA engine supplies an uncalibrated start/end logit score, original
source certificate and immutable model revision. The gate checks a
separately calibrated and signed risk policy, score threshold, trusted
task group, model/rubric identity and source certificate before releasing
any answer. Missing evidence always causes abstention.

The group MUST be assigned by trusted application routing metadata, not
chosen by the model or end user. Research-only, no automatic rollout.
"""
from __future__ import annotations

from selective_risk_gate import apply_selective_risk_gate


def answer_with_calibrated_risk(engine, context, question, *, policy, key,
                                trusted_group, rubric_id):
    """Return source-backed answer only if every independent gate passes."""
    base = {
        "accepted": False, "answer": "", "automatic_rollout_permitted": False,
        "warning": "Statistical risk bounds require IID representative calibration; source spans do not prove truth.",
    }
    if not isinstance(context, str) or not isinstance(question, str):
        return {**base, "reason": "invalid_task"}
    if not isinstance(trusted_group, str) or not trusted_group:
        return {**base, "reason": "missing_trusted_group"}
    # Reject invalid or unsigned policy BEFORE running the model.
    from selective_risk_gate import verify_risk_policy
    if not verify_risk_policy(policy, key=key):
        return {**base, "reason": "unsigned_or_modified_policy"}
    if (not policy.get("conditional_risk_screen_passed")
            or policy.get("model_revision") != getattr(engine, "revision", None)
            or policy.get("rubric_id") != rubric_id):
        return {**base, "reason": "policy_or_model_not_qualified"}
    if trusted_group not in policy.get("groups", []):
        return {**base, "reason": "unrecognized_group"}
    try:
        result = engine.answer(context, question)
    except (ValueError, RuntimeError):
        return {**base, "reason": "qa_inference_failed"}
    if not isinstance(result, dict) or not result.get("accepted"):
        return {**base, "reason": "qa_model_abstained"}
    if result.get("model_revision") != engine.revision:
        return {**base, "reason": "qa_revision_mismatch"}
    return apply_selective_risk_gate(
        policy, key=key,
        score=result.get("uncalibrated_span_logit_score"),
        group=trusted_group,
        source_proof=result.get("proof"),
        source_text=context,
        model_revision=engine.revision,
        rubric_id=rubric_id,
    )

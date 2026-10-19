"""ĀRU sovereign risk pipeline: purpose-scoped source consent + risk + proof.

The local source registry contains operator attestations, NOT verified legal
consent. A separately retained registry checkpoint can protect an anchored
prefix against deletion of revocation events. The independent signed risk
policy screens an extractive QA answer before release. All gates fail closed.

No public deployment, verified human consent or guarantee of factual truth.
"""
from __future__ import annotations

from risk_qa_adapter import answer_with_calibrated_risk
from sovereign_source_registry import authorize_source
from source_registry_anchor import verify_source_checkpoint


def answer_with_sovereign_risk(
    engine, context, question, *,
    owner, purpose, registry_db, registry_key,
    policy, risk_key, trusted_group, rubric_id,
    registry_checkpoint=None, registry_anchor_key=None,
    expected_calibration_set_id=None,
):
    """Refuse to run inference until source authorization is established.

    Group and owner MUST be derived from trusted application metadata, not
    the model or untrusted request. Checkpoints are optional for research,
    but necessary for stronger detection of previously anchored revocations.
    """
    base = {
        "accepted": False, "answer": "",
        "automatic_rollout_permitted": False,
        "consent_status": "operator_attested_not_independently_verified",
        "warning": (
            "Consent is operator-attested; source provenance and statistical "
            "risk bounds do not prove factual truth or legal consent."
        ),
    }
    if not isinstance(context, str) or not isinstance(question, str):
        return {**base, "reason": "invalid_task"}
    if not isinstance(owner, str) or not owner or not isinstance(purpose, str) or not purpose:
        return {**base, "reason": "missing_trusted_owner_or_purpose"}
    if registry_checkpoint is not None:
        if registry_anchor_key is None:
            return {**base, "reason": "missing_independent_checkpoint_key"}
        try:
            anchored = verify_source_checkpoint(
                registry_db, registry_checkpoint,
                registry_key=registry_key, anchor_key=registry_anchor_key,
            )
        except (ValueError, TypeError, OSError, RuntimeError):
            return {**base, "reason": "registry_checkpoint_verification_failed"}
        if not anchored["verified"]:
            return {**base, "reason": "registry_checkpoint_verification_failed"}
    try:
        authorization = authorize_source(
            registry_db, context, owner=owner, purpose=purpose,
            key=registry_key,
        )
    except (ValueError, TypeError, OSError, RuntimeError):
        return {**base, "reason": "source_authorization_unavailable"}
    if not authorization["authorized"]:
        return {**base, "reason": authorization["reason"]}
    if (expected_calibration_set_id is not None
            and policy.get("calibration_set_id") != expected_calibration_set_id):
        return {**base, "reason": "calibration_policy_replay_or_mismatch"}
    result = answer_with_calibrated_risk(
        engine, context, question,
        policy=policy, key=risk_key,
        trusted_group=trusted_group, rubric_id=rubric_id,
    )
    return {
        **result,
        "source_authorization": "active_purpose_scoped_operator_attestation",
        "registry_checkpoint_verified": registry_checkpoint is not None,
        "automatic_rollout_permitted": False,
        "consent_status": "operator_attested_not_independently_verified",
    }

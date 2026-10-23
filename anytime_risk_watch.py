"""ĀRU Anytime Risk Watch: prospective group-aware degradation detection.

For prospective binary error E_t in {0,1} and target risk r in (0,1),
under the conditional null H0: E[E_t | past] <= r, each fixed lambda
in [0, 1/r] defines a nonnegative supermartingale

    M_t(lambda) = product_i (1 + lambda*(E_i-r)).

A predeclared convex mixture is also a supermartingale. Ville's inequality
bounds the chance of ever crossing (number_of_monitored_slices / alpha)
under the joint conditional null by alpha, via a union bound over groups
and the aggregate. A crossing is evidence of deterioration relative to
the target, NOT proof that the AI is safe before a crossing.

This technique is established sequential testing, not a new theorem.
It requires a prospectively fixed stream with COMPLETE independently
graded outcomes and trusted group labels; selective/delayed feedback,
distribution changes and clustered or manipulated labels can invalidate
the interpretation. A monitoring report is not a deployment approval.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math

FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def _id(value, field):
    if not isinstance(value, str) or not value or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError(field + " must be a nonempty bounded string")
    return value


def _logsumexp(values):
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(x - top) for x in values))


def _monitor_slice(errors, *, target_error, log_threshold, bet_fractions):
    bets = [fraction / target_error for fraction in bet_fractions]
    wealth = [0.0] * len(bets)
    crossing = None
    peak = 0.0
    observed_errors = 0
    for index, error in enumerate(errors, 1):
        observed_errors += int(error)
        for j, bet in enumerate(bets):
            if wealth[j] == -math.inf:
                continue
            factor = 1 + bet * (int(error) - target_error)
            wealth[j] = wealth[j] + math.log(factor) if factor > 0 else -math.inf
        current = _logsumexp(wealth) - math.log(len(wealth))
        peak = max(peak, current)
        if crossing is None and current >= log_threshold:
            crossing = index
    last = _logsumexp(wealth) - math.log(len(wealth))
    return {
        "n_observed": len(errors),
        "errors": observed_errors,
        "observed_error_rate": observed_errors / len(errors) if errors else None,
        "final_log_e": last,
        "max_log_e": peak,
        "first_crossing_observation": crossing,
        "degradation_alarm": crossing is not None,
    }


def watch_risk_stream(rows, *, groups, model_revision, rubric_id,
                      target_error=0.1, alpha=0.05,
                      bet_fractions=FRACTIONS):
    """Compute anytime evidence on a predeclared, ordered, complete stream.

    rows are chronological dicts: event_id, group, error (strict bool).
    This function intentionally refuses timestamps or raw question text:
    the caller is responsible for retaining trusted event ordering.
    """
    _id(model_revision, "model_revision")
    _id(rubric_id, "rubric_id")
    if not isinstance(groups, (tuple, list)) or not groups:
        raise ValueError("predeclared nonempty groups required")
    for group in groups:
        _id(group, "group")
    if len(set(groups)) != len(groups) or "__all__" in groups:
        raise ValueError("duplicate or reserved group")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty prospective outcomes required")
    if (isinstance(target_error, bool) or not isinstance(target_error, (float, int))
            or not math.isfinite(target_error) or not 0 < target_error < 1):
        raise ValueError("target_error must be in (0,1)")
    if (isinstance(alpha, bool) or not isinstance(alpha, (float, int))
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if (not isinstance(bet_fractions, (list, tuple)) or not bet_fractions
            or len(bet_fractions) > 64
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) or not 0 < v < 1 for v in bet_fractions)):
        raise ValueError("fixed betting fractions must be in (0,1)")
    if len(set(bet_fractions)) != len(bet_fractions):
        raise ValueError("duplicate betting fractions")
    seen = set()
    ordered = {"__all__": []}
    ordered.update({group: [] for group in groups})
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"event_id", "group", "error"}:
            raise ValueError("each event requires exactly event_id, group, error")
        eid = _id(row["event_id"], "event_id")
        if eid in seen:
            raise ValueError("duplicate event id")
        seen.add(eid)
        if row["group"] not in groups:
            raise ValueError("unrecognized group")
        if type(row["error"]) is not bool:
            raise ValueError("independent error label must be boolean")
        ordered["__all__"].append(row["error"])
        ordered[row["group"]].append(row["error"])
    slices = len(groups) + 1
    threshold = math.log(slices / alpha)
    evidence = {
        group: _monitor_slice(
            events, target_error=target_error,
            log_threshold=threshold, bet_fractions=bet_fractions,
        )
        for group, events in ordered.items()
    }
    alarms = [group for group, result in evidence.items() if result["degradation_alarm"]]
    digest = hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "kind": "aru_anytime_risk_watch",
        "model_revision": model_revision,
        "rubric_id": rubric_id,
        "target_error": target_error,
        "familywise_alpha": alpha,
        "groups": list(groups),
        "bet_fractions": list(bet_fractions),
        "slice_count": slices,
        "log_alarm_threshold": threshold,
        "outcome_count": len(rows),
        "ordered_outcomes_sha256": digest,
        "slices": evidence,
        "alarm_groups": alarms,
        "degradation_alarm": bool(alarms),
        "automatic_rollout_permitted": False,
        "monitoring_state": "alarm" if alarms else "no_alarm_observed",
        "limitations": [
            "A lack of alarm is NOT evidence that the model is safe or below the target error rate.",
            "Anytime-valid error control requires the prospective conditional null, fixed betting mixture and complete outcomes.",
            "Selective, delayed, missing or manipulated labels can invalidate the risk alarm interpretation.",
            "Trusted groups and model/rubric identities must be assigned outside the AI model.",
            "Monitoring does not itself authorize rollout; human review remains required.",
        ],
    }


def sign_watch_report(report, *, key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("independent monitoring signing key must be at least 32 bytes")
    if not isinstance(report, dict) or report.get("kind") != "aru_anytime_risk_watch" or "watch_hmac_sha256" in report:
        raise ValueError("unsigned watch report required")
    signed = dict(report)
    signed["watch_hmac_sha256"] = hmac.new(
        key, _canonical(report).encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    return signed


def verify_watch_report(report, *, key):
    if not isinstance(report, dict) or not isinstance(key, bytes) or len(key) < 32:
        return False
    digest = report.get("watch_hmac_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    unsigned = {k: v for k, v in report.items() if k != "watch_hmac_sha256"}
    try:
        expected = hmac.new(key, _canonical(unsigned).encode("utf-8"), hashlib.sha256).hexdigest()
    except (ValueError, TypeError):
        return False
    return (hmac.compare_digest(expected, digest)
            and report.get("schema_version") == 1
            and report.get("kind") == "aru_anytime_risk_watch"
            and report.get("automatic_rollout_permitted") is False)


def gate_with_risk_watch(policy, *, policy_key, watch_report, watch_key,
                         score, group, source_proof, source_text,
                         model_revision, rubric_id,
                         expected_outcomes_sha256=None, min_group_outcomes=1):
    """Require independently signed prospective monitoring before risk release.

    A report with no alarm is not evidence of safety; the calibrated
    static risk screen must still pass. Caller must verify the report is
    current, complete, and from an authorized monitoring pipeline.
    """
    deny = {
        "accepted": False, "answer": "", "automatic_rollout_permitted": False,
        "warning": "A missing alarm is not proof of safety; complete representative monitoring is required.",
    }
    if not verify_watch_report(watch_report, key=watch_key):
        return {**deny, "reason": "missing_or_invalid_signed_watch"}
    if (watch_report.get("model_revision") != model_revision
            or watch_report.get("rubric_id") != rubric_id):
        return {**deny, "reason": "watch_model_or_rubric_mismatch"}
    # A valid monitoring signature does not make a different risk target
    # interchangeable with the one certified by static calibration.
    if (not isinstance(policy, dict)
            or watch_report.get("target_error") != policy.get("target_error")
            or watch_report.get("groups") != policy.get("groups")):
        return {**deny, "reason": "watch_calibration_scope_mismatch"}
    if group not in watch_report.get("groups", []):
        return {**deny, "reason": "unmonitored_group"}
    # Require an independently expected digest of the COMPLETE current
    # outcome stream. A valid HMAC alone cannot detect replay of an older
    # alarm-free report. The caller must obtain this digest from a trusted
    # separately controlled monitoring ledger, not from the report itself.
    if (not isinstance(expected_outcomes_sha256, str)
            or len(expected_outcomes_sha256) != 64
            or not hmac.compare_digest(
                expected_outcomes_sha256,
                str(watch_report.get("ordered_outcomes_sha256", "")),
            )):
        return {**deny, "reason": "missing_or_stale_watch_checkpoint"}
    if (isinstance(min_group_outcomes, bool) or not isinstance(min_group_outcomes, int)
            or min_group_outcomes < 1):
        return {**deny, "reason": "invalid_minimum_monitoring_support"}
    group_evidence = watch_report.get("slices", {}).get(group)
    if (not isinstance(group_evidence, dict)
            or group_evidence.get("n_observed", 0) < min_group_outcomes):
        return {**deny, "reason": "insufficient_recent_group_outcomes"}
    if watch_report.get("degradation_alarm") or any(
        slice_data.get("degradation_alarm")
        for slice_data in watch_report.get("slices", {}).values()
    ):
        return {**deny, "reason": "prospective_degradation_alarm"}
    from selective_risk_gate import apply_selective_risk_gate
    result = apply_selective_risk_gate(
        policy, key=policy_key, score=score, group=group,
        source_proof=source_proof, source_text=source_text,
        model_revision=model_revision, rubric_id=rubric_id,
    )
    if result["accepted"]:
        result["watch_hmac_sha256"] = watch_report["watch_hmac_sha256"]
        result["watch_outcome_count"] = watch_report["outcome_count"]
        result["warning"] = (
            "Calibrated static risk screen and prospective monitor had no alarm. "
            "This is not proof of safety or production authorization."
        )
    return result

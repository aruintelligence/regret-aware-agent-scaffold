"""ĀRU Selective Risk Gate: evidence-calibrated, fail-closed answer release.

A fixed threshold on IID calibration (score, correctness, group) pairs
has an exact one-sided binomial confidence upper bound for conditional
error risk. Bonferroni correction simultaneously covers all predeclared
thresholds and groups, including after choosing a threshold on the labels.

These are established statistical methods. Guarantees require representative
IID calibration and future tasks, fixed models, correct independent labels,
trusted groups and no selection bias. No automatic rollout is permitted.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math

from grounding_contract import verify_grounding


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _valid_number(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError(name + " must be a nonempty bounded identifier")
    return value


def _log_binomial_cdf(k, n, p):
    """log P[Binomial(n,p) <= k], stable for extreme tails."""
    if k >= n or p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * log_p + (n - j) * log_q
        for j in range(k + 1)
    ]
    largest = max(terms)
    return largest + math.log(sum(math.exp(t - largest) for t in terms))


def exact_binomial_error_upper(errors, accepted, *, alpha):
    """Exact one-sided Clopper-Pearson bound for a Bernoulli error rate."""
    if (isinstance(errors, bool) or isinstance(accepted, bool)
            or not isinstance(errors, int) or not isinstance(accepted, int)
            or not 0 <= errors <= accepted):
        raise ValueError("errors and accepted must be valid integer counts")
    if not _valid_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if accepted == 0 or errors == accepted:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / accepted)
    low, high = errors / accepted, 1.0
    log_alpha = math.log(alpha)
    for _ in range(55):
        mid = (low + high) / 2
        if _log_binomial_cdf(errors, accepted, mid) > log_alpha:
            low = mid
        else:
            high = mid
    return high


def _validate_rows(rows, groups):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty calibration list required")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"sample_id", "score", "correct", "group"}:
            raise ValueError("calibration row must have exactly sample_id, score, correct, group")
        sample_id = _identifier(row["sample_id"], "sample_id")
        if sample_id in seen:
            raise ValueError("duplicate calibration sample ID")
        seen.add(sample_id)
        if not _valid_number(row["score"]):
            raise ValueError("calibration score must be finite")
        if type(row["correct"]) is not bool:
            raise ValueError("correct must be a boolean from an independent grader")
        if row["group"] not in groups:
            raise ValueError("unknown group; groups must be predeclared")


def fit_selective_risk_policy(rows, *, thresholds, groups,
                              target_error=0.1, alpha=0.05,
                              min_group_support=30,
                              model_revision, rubric_id, calibration_set_id):
    """Choose maximum coverage among simultaneously risk-bounded thresholds."""
    for name, value in (("model_revision", model_revision),
                        ("rubric_id", rubric_id),
                        ("calibration_set_id", calibration_set_id)):
        _identifier(value, name)
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("predeclared groups required")
    for group in groups:
        _identifier(group, "group")
    if "__all__" in groups or len(set(groups)) != len(groups):
        raise ValueError("duplicate or reserved group identifier")
    if not isinstance(thresholds, (list, tuple)) or not thresholds or len(thresholds) > 64:
        raise ValueError("predeclared threshold grid of 1 to 64 values required")
    if any(not _valid_number(t) for t in thresholds) or any(
        a >= b for a, b in zip(thresholds, thresholds[1:])
    ):
        raise ValueError("thresholds must be strictly increasing finite numbers")
    if not _valid_number(target_error) or not 0 < target_error < 1:
        raise ValueError("target_error must be in (0,1)")
    if not _valid_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if (isinstance(min_group_support, bool) or not isinstance(min_group_support, int)
            or min_group_support < 1):
        raise ValueError("min_group_support must be a positive integer")
    _validate_rows(rows, set(groups))
    tests = len(thresholds) * (len(groups) + 1)
    per_test_alpha = alpha / tests
    candidates = []
    for threshold in thresholds:
        accepted = [r for r in rows if r["score"] >= threshold]
        slices = {"__all__": accepted}
        slices.update({group: [r for r in accepted if r["group"] == group] for group in groups})
        diagnostics, eligible = {}, True
        for name, subset in slices.items():
            n = len(subset)
            errors = sum(not r["correct"] for r in subset)
            upper = exact_binomial_error_upper(errors, n, alpha=per_test_alpha)
            diagnostics[name] = {
                "accepted": n, "errors": errors,
                "observed_error": errors / n if n else None,
                "simultaneous_error_upper": upper,
                "sufficient_support": n >= min_group_support,
                "meets_target": n >= min_group_support and upper <= target_error,
            }
            if not diagnostics[name]["meets_target"]:
                eligible = False
        candidates.append({
            "threshold": threshold, "eligible": eligible,
            "accepted": len(accepted), "coverage": len(accepted) / len(rows),
            "slices": diagnostics,
        })
    eligible = [c for c in candidates if c["eligible"]]
    best = sorted(eligible, key=lambda c: (-c["accepted"], c["threshold"]))[0] if eligible else None
    digest = hashlib.sha256(_canonical(sorted(rows, key=lambda r: r["sample_id"])).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1, "kind": "aru_selective_risk_calibration",
        "model_revision": model_revision, "rubric_id": rubric_id,
        "calibration_set_id": calibration_set_id, "calibration_sha256": digest,
        "calibration_rows": len(rows), "groups": list(groups),
        "threshold_grid": list(thresholds), "target_error": target_error,
        "familywise_alpha": alpha, "per_test_alpha": per_test_alpha,
        "simultaneous_tests": tests, "min_group_support": min_group_support,
        "candidates": candidates,
        "selected_threshold": best["threshold"] if best else None,
        "selected_coverage": best["coverage"] if best else 0.0,
        "conditional_risk_screen_passed": best is not None,
        "automatic_rollout_permitted": False,
        "scope": "exact one-sided binomial confidence with Bonferroni across fixed thresholds and groups",
        "assumptions": [
            "Independent identically distributed calibration and future tasks within each group.",
            "Predeclared thresholds and groups; fixed model, score, independent rubric and group assignment.",
            "Correct independent labels; no adaptive selection, missingness, clustered tasks or distribution shift.",
            "Statistical screening never authorizes deployment or proves answer truth.",
        ],
    }


def sign_risk_policy(policy, *, key):
    """HMAC-sign the exact calibrated policy with a separately controlled key."""
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("risk signing key must be at least 32 bytes")
    if not isinstance(policy, dict) or policy.get("schema_version") != 1 or "policy_hmac_sha256" in policy:
        raise ValueError("unsigned version-1 policy required")
    signed = dict(policy)
    signed["policy_hmac_sha256"] = hmac.new(
        key, _canonical(policy).encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    return signed


def verify_risk_policy(policy, *, key):
    if not isinstance(key, bytes) or len(key) < 32 or not isinstance(policy, dict):
        return False
    digest = policy.get("policy_hmac_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    unsigned = {k: v for k, v in policy.items() if k != "policy_hmac_sha256"}
    try:
        actual = hmac.new(key, _canonical(unsigned).encode("utf-8"), hashlib.sha256).hexdigest()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, actual) and policy.get("schema_version") == 1


def apply_selective_risk_gate(policy, *, key, score, group,
                              source_proof, source_text,
                              model_revision, rubric_id):
    """Release only if signed policy, score, group and source proof all pass.

    Callers must derive group from trusted routing metadata, not an
    untrusted model response or user-controlled request field.
    """
    base = {
        "accepted": False, "answer": "",
        "reason": "denied", "automatic_rollout_permitted": False,
        "warning": "A source span is not factual truth. Statistical risk bounds require IID calibration and future data.",
    }
    if not verify_risk_policy(policy, key=key):
        return {**base, "reason": "unsigned_or_modified_policy"}
    if policy.get("kind") != "aru_selective_risk_calibration" or policy.get("automatic_rollout_permitted") is not False:
        return {**base, "reason": "unsupported_policy"}
    if not policy.get("conditional_risk_screen_passed") or policy.get("selected_threshold") is None:
        return {**base, "reason": "no_risk_qualified_threshold"}
    if not _valid_number(policy["selected_threshold"]) or not _valid_number(policy.get("target_error")):
        return {**base, "reason": "invalid_policy_values"}
    if model_revision != policy.get("model_revision") or rubric_id != policy.get("rubric_id"):
        return {**base, "reason": "model_or_rubric_mismatch"}
    if group not in policy.get("groups", []):
        return {**base, "reason": "unrecognized_group"}
    if not _valid_number(score):
        return {**base, "reason": "invalid_score"}
    if score < policy["selected_threshold"]:
        return {**base, "reason": "score_below_threshold"}
    if not verify_grounding(source_proof, source_text):
        return {**base, "reason": "invalid_source_certificate"}
    if source_proof.get("model_revision") != model_revision:
        return {**base, "reason": "source_proof_model_mismatch"}
    # Check signed per-group risk bounds at release time too, not merely
    # a top-level boolean that might have been set incorrectly at fit time.
    matches = [c for c in policy.get("candidates", [])
               if c.get("threshold") == policy["selected_threshold"] and c.get("eligible") is True]
    if len(matches) != 1:
        return {**base, "reason": "inconsistent_policy_threshold"}
    candidate = matches[0]
    group_slice = candidate.get("slices", {}).get(group)
    aggregate = candidate.get("slices", {}).get("__all__")
    if not isinstance(group_slice, dict) or not isinstance(aggregate, dict):
        return {**base, "reason": "missing_group_evidence"}
    for evidence in (group_slice, aggregate):
        if (not evidence.get("meets_target")
                or evidence.get("accepted", 0) < policy.get("min_group_support", 1)
                or not _valid_number(evidence.get("simultaneous_error_upper"))
                or evidence["simultaneous_error_upper"] > policy["target_error"]):
            return {**base, "reason": "insufficient_group_risk_evidence"}
    return {
        **base, "accepted": True, "answer": source_proof["answer"],
        "reason": "signed_risk_screen_and_source_span_verified",
        "calibration_set_id": policy["calibration_set_id"],
        "group": group,
        "risk_upper_bound": max(
            group_slice["simultaneous_error_upper"],
            aggregate["simultaneous_error_upper"],
        ),
        "target_error": policy["target_error"],
        "policy_hmac_sha256": policy["policy_hmac_sha256"],
    }

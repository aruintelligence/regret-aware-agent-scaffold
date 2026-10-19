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

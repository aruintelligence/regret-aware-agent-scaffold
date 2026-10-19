"""Research-only calibrated restart monitor for binary outcome streams.

This has two separate sources of uncertainty. A one-sided Hoeffding bound
uses a frozen independent calibration stream (IID Bernoulli errors per slice)
and spends delta over the aggregate and predeclared groups. A fixed schedule
of restarted e-processes spends monitoring_alpha over starts and slices.
Under stationary calibration-to-monitoring probabilities, a predictable
group schedule, complete outcomes and the joint conditional monitoring null,
the union bound controls the probability of ANY alarm at delta +
monitoring_alpha. Neither IID sampling nor stationarity follows from the code.
This unsigned report cannot be submitted to the signed release gate.
"""
from __future__ import annotations

import math

from anytime_risk_watch import FRACTIONS, _monitor_slice, watch_risk_stream


def calibrated_restart_risk_stream(
        calibration_rows, monitoring_rows, *, groups, model_revision, rubric_id,
        alpha_total=0.05, calibration_delta=0.025, min_slice_calibration=20,
        horizon=120, starts=(1, 21, 41, 61, 81, 101),
        bet_fractions=FRACTIONS):
    """Recompute calibration and evaluate a separate complete stream.

    Global starts are committed in advance. The method never silently drops
    a slice with inadequate calibration; it reports an inconclusive study.
    """
    if (isinstance(alpha_total, bool) or not isinstance(alpha_total, (int, float))
            or not math.isfinite(alpha_total) or not 0 < alpha_total < 1
            or isinstance(calibration_delta, bool)
            or not isinstance(calibration_delta, (int, float))
            or not math.isfinite(calibration_delta) or not 0 < calibration_delta < alpha_total):
        raise ValueError("finite positive calibration and monitoring budgets required")
    if (isinstance(min_slice_calibration, bool) or not isinstance(min_slice_calibration, int)
            or min_slice_calibration < 1):
        raise ValueError("positive minimum calibration support required")
    if (isinstance(horizon, bool) or not isinstance(horizon, int)
            or not 1 <= horizon <= 100000
            or not isinstance(monitoring_rows, list) or len(monitoring_rows) > horizon):
        raise ValueError("finite horizon must cover monitored prefix")
    if (not isinstance(starts, (tuple, list)) or not starts or len(starts) > 64
            or starts[0] != 1 or list(starts) != sorted(set(starts))
            or any(type(s) is not int or not 1 <= s <= horizon for s in starts)):
        raise ValueError("precommitted sorted unique starts beginning at one required")

    # Existing strict stream validator rejects duplicate IDs, invalid groups,
    # non-boolean labels, extra text, invalid bets and model/rubric identities.
    calibration = watch_risk_stream(
        calibration_rows, groups=groups, model_revision=model_revision,
        rubric_id=rubric_id, target_error=0.5, alpha=0.05,
        bet_fractions=bet_fractions,
    )
    monitored = watch_risk_stream(
        monitoring_rows, groups=groups, model_revision=model_revision,
        rubric_id=rubric_id, target_error=0.5, alpha=0.05,
        bet_fractions=bet_fractions,
    )
    names = ["__all__", *groups]
    n_slices = len(names)
    targets = {}
    invalid = []
    for name in names:
        row = calibration["slices"][name]
        n = row["n_observed"]
        if n < min_slice_calibration:
            invalid.append(name + ":insufficient_calibration")
            upper = None
        else:
            upper = min(1.0, row["errors"] / n
                        + math.sqrt(math.log(n_slices / calibration_delta) / (2 * n)))
            if upper >= 1.0:
                invalid.append(name + ":unmonitorable_bound")
        targets[name] = {"n_calibration": n, "errors": row["errors"],
                         "observed_error_fraction": row["observed_error_rate"],
                         "one_sided_upper_error_bound": upper}

    monitor_alpha = alpha_total - calibration_delta
    threshold = math.log(len(starts) * n_slices / monitor_alpha)
    windows, alarms = [], []
    if not invalid:
        for start in starts:
            suffix = list(enumerate(monitoring_rows[start - 1:], start))
            components = {}
            for name in names:
                selected = [(i, row["error"]) for i, row in suffix
                            if name == "__all__" or row["group"] == name]
                evidence = _monitor_slice(
                    [error for _, error in selected],
                    target_error=targets[name]["one_sided_upper_error_bound"],
                    log_threshold=threshold, bet_fractions=bet_fractions,
                )
                crossing = evidence["first_crossing_observation"]
                global_crossing = selected[crossing - 1][0] if crossing is not None else None
                components[name] = {**evidence,
                                    "first_crossing_global_observation": global_crossing}
                if global_crossing is not None:
                    alarms.append({"start_global_observation": start, "slice": name,
                                   "first_crossing_global_observation": global_crossing})
            windows.append({"start_global_observation": start, "slices": components})
    first = min((a["first_crossing_global_observation"] for a in alarms), default=None)
    return {
        "schema_version": 1, "kind": "aru_research_calibrated_restart_watch",
        "model_revision": model_revision, "rubric_id": rubric_id,
        "groups": list(groups), "bet_fractions": list(bet_fractions),
        "alpha_total": alpha_total, "calibration_delta": calibration_delta,
        "monitoring_alpha": monitor_alpha,
        "calibration_bound": "one_sided_Hoeffding_IID_Bernoulli_union_over_slices",
        "min_slice_calibration": min_slice_calibration,
        "calibration_targets": targets,
        "calibration_outcome_count": len(calibration_rows),
        "calibration_outcomes_sha256": calibration["ordered_outcomes_sha256"],
        "monitoring_outcome_count": len(monitoring_rows),
        "monitoring_outcomes_sha256": monitored["ordered_outcomes_sha256"],
        "horizon": horizon, "starts": list(starts),
        "component_count": len(starts) * n_slices,
        "log_alarm_threshold": threshold,
        "calibration_ready": not invalid, "calibration_failures": invalid,
        "windows": windows, "alarm_components": alarms,
        "first_alarm_global_observation": first,
        "degradation_alarm": first is not None if not invalid else None,
        "monitoring_state": ("inconclusive_calibration" if invalid else
                             "alarm" if first is not None else "no_alarm_observed"),
        "automatic_rollout_permitted": False,
        "limitations": [
            "The combined alpha bound requires IID per-slice calibration errors, stable probabilities, a complete independent monitoring stream and the conditional monitoring null.",
            "Group assignment must precede each outcome; changing group mix can break the aggregate null.",
            "Shared articles and repository grading may violate sampling assumptions.",
            "A no-alarm path is not proof of safety or an error-rate ceiling.",
            "This unsigned research monitor is not a signed production release certificate.",
        ],
    }

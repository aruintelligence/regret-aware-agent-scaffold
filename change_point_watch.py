"""Research-only finite-horizon restart monitor for abrupt risk deterioration.

For every precommitted (global start, slice), the existing fixed-bet mixture
starts with wealth one at that position. Under the *conditional* error null
and predictable group assignment it is a nonnegative supermartingale. Ville's
inequality plus a union bound over K starts and S slices gives a familywise
anytime alarm probability at most alpha when each crossing uses K*S/alpha.
The guarantee does not describe detection power or safety on a no-alarm path.
"""
from __future__ import annotations

import math

from anytime_risk_watch import FRACTIONS, _monitor_slice, watch_risk_stream


def restart_risk_stream(rows, *, groups, model_revision, rubric_id,
                        target_error=0.1, alpha=0.05,
                        bet_fractions=FRACTIONS, horizon=120,
                        starts=(1, 21, 41, 61, 81, 101)):
    """Watch a complete prefix of a prospectively ordered finite stream.

    Starts refer to GLOBAL one-based observation positions, even for groups.
    The whole schedule and horizon must be committed before observing labels.
    This unsigned research report is never accepted by the release gate.
    """
    if (isinstance(horizon, bool) or not isinstance(horizon, int)
            or not 1 <= horizon <= 100000 or len(rows) > horizon):
        raise ValueError("finite positive horizon must cover the observed prefix")
    if (not isinstance(starts, (list, tuple)) or not starts or len(starts) > 64
            or starts[0] != 1
            or any(isinstance(s, bool) or not isinstance(s, int)
                   or not 1 <= s <= horizon for s in starts)
            or list(starts) != sorted(set(starts))):
        raise ValueError("predeclared starts must be sorted, unique, begin at one and lie inside horizon")
    # Delegate strict event, target, group, bet, and identity validation to
    # the existing watch. Its outcome digest is the common evidence anchor.
    from_start = watch_risk_stream(
        rows, groups=groups, model_revision=model_revision, rubric_id=rubric_id,
        target_error=target_error, alpha=alpha, bet_fractions=bet_fractions,
    )
    group_names = ["__all__", *groups]
    component_count = len(starts) * len(group_names)
    threshold = math.log(component_count / alpha)
    windows = []
    alarms = []
    for start in starts:
        components = {}
        # A start outside the observed prefix is reserved for future data;
        # it still consumes its precommitted share of the error budget.
        suffix = list(enumerate(rows[start - 1:], start))
        for name in group_names:
            selected = [(index, row["error"]) for index, row in suffix
                        if name == "__all__" or row["group"] == name]
            result = _monitor_slice(
                [error for _, error in selected], target_error=target_error,
                log_threshold=threshold, bet_fractions=bet_fractions,
            )
            crossing = result["first_crossing_observation"]
            global_crossing = selected[crossing - 1][0] if crossing is not None else None
            components[name] = {
                **result, "first_crossing_global_observation": global_crossing,
            }
            if global_crossing is not None:
                alarms.append({"start_global_observation": start, "slice": name,
                               "first_crossing_global_observation": global_crossing})
        windows.append({"start_global_observation": start, "slices": components})
    first = min((a["first_crossing_global_observation"] for a in alarms), default=None)
    return {
        "schema_version": 1, "kind": "aru_research_change_point_watch",
        "model_revision": model_revision, "rubric_id": rubric_id,
        "target_error": target_error, "familywise_alpha": alpha,
        "groups": list(groups), "bet_fractions": list(bet_fractions),
        "horizon": horizon, "starts": list(starts),
        "component_count": component_count, "log_alarm_threshold": threshold,
        "outcome_count": len(rows),
        "ordered_outcomes_sha256": from_start["ordered_outcomes_sha256"],
        "windows": windows, "alarm_components": alarms,
        "first_alarm_global_observation": first,
        "degradation_alarm": first is not None,
        "automatic_rollout_permitted": False,
        "limitations": [
            "A no-alarm path is not evidence of safety or an error-rate ceiling.",
            "Anytime false-alarm control requires a prospectively fixed schedule and complete, independently graded outcomes under the conditional null.",
            "Group assignment must be known before its outcome; missing or manipulated labels invalidate the interpretation.",
            "A finite schedule spends the error budget on all components, including starts beyond the current prefix.",
            "This unsigned research report is not accepted by the production release gate.",
        ],
    }

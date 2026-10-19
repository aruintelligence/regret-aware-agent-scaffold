"""One-sided, time-uniform evidence for a fixed pair of routing policies.

The bound covers the average *conditional* policy-value difference on the
logged contexts. It permits adaptive contexts and logging probabilities, but
requires valid action randomization and complete, bounded outcome labels.
It does not make a post-hoc choice of target policies valid.
"""
from __future__ import annotations

import math
from typing import List

from route_experiments import (
    choose_policy_action, logging_action_probabilities, validate_decision,
)


def anytime_ips_lower_bound(rows: List[dict], baseline_policy: str,
                            candidate_policy: str, *, alpha: float = 0.05) -> dict:
    """Time-uniform lower bound for the logged-context average uplift.

    For a pre-action candidate set, q(a) comes from the declared epsilon-greedy
    or categorical logging distribution. With distinct target actions b,c,
    Z = reward * (1[A=c]/q(c) - 1[A=b]/q(b)) has conditional mean equal to
    the candidate-minus-baseline value, and lies in [-1/q(b), 1/q(c)].

    Conditional Hoeffding gives a nonnegative supermartingale for each fixed
    positive lambda: exp(lambda * sum(E[Z|past]-Z) - lambda²*sum(width²)/8).
    Ville's inequality, then a union bound over a countable lambda grid with
    alpha_j = alpha/((j+1)(j+2)), makes the best grid bound valid at EVERY
    sample size. This code uses the first 40 grid values (and thus spends
    less than alpha). The statement concerns conditional mean on these
    contexts, not a future population or the realized unobserved outcomes.
    """
    if not rows:
        raise ValueError("need at least one completely labeled decision")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0,1)")

    contributions = []
    range_squares = []
    for row in rows:
        decision, label = row["decision"], row["label"]
        validate_decision(decision)
        reward = label["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("reward must be in [0,1]")
        baseline = choose_policy_action(decision, baseline_policy)
        candidate = choose_policy_action(decision, candidate_policy)
        if baseline == candidate:
            contributions.append(0.0)
            range_squares.append(0.0)
            continue
        probabilities = logging_action_probabilities(decision)
        if len(probabilities) < 2 or baseline not in probabilities or candidate not in probabilities:
            raise ValueError("target action missing from candidate set")
        q_base = probabilities[baseline]
        q_cand = probabilities[candidate]
        contributions.append(reward * (
            (1 / q_cand if decision["action"] == candidate else 0)
            - (1 / q_base if decision["action"] == baseline else 0)
        ))
        range_squares.append((1 / q_cand + 1 / q_base) ** 2)

    n = len(rows)
    estimate = math.fsum(contributions) / n
    v = math.fsum(range_squares)
    if v == 0:
        radius = 0.0  # Identical actions on every logged context: Z is exactly zero.
        grid_index = None
    else:
        radii = []
        for j in range(40):
            lam = 2.0 ** -j
            spent = alpha / ((j + 1) * (j + 2))
            radii.append(((-math.log(spent)) / lam + lam * v / 8) / n)
        radius = min(radii)
        grid_index = radii.index(radius)
    return {
        "method": "conditional_hoeffding_ville_grid_v1",
        "n": n,
        "confidence": 1 - alpha,
        "ips_difference_unclipped": estimate,
        "radius": radius,
        "lower_bound": max(-1.0, estimate - radius),
        "grid_index": grid_index,
        "range_square_sum": v,
        "scope": "time-uniform average conditional value difference on logged contexts",
    }

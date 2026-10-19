"""Adversarial task-domain mixture stress test under a total-variation budget.

This solves a finite, discrete reweighting problem on OBSERVED domain values.
It is not a statistical confidence interval and does not cover unseen domains
or changes to outcome rates within each domain.
"""
from __future__ import annotations

import math


def worst_case_topic_mix(values, weights, *, tv_budget=0.10):
    """Minimize weighted policy uplift over a total-variation ball.

    With fixed observed domain values, the linear objective is minimized by
    moving available probability mass from highest-value domains to
    lowest-value domains. The greedy transport is optimal for this finite
    unconstrained discrete simplex reweighting problem.
    """
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must have equal nonzero lengths")
    if not isinstance(tv_budget, (float, int)) or not math.isfinite(tv_budget) or not 0 <= tv_budget <= 1:
        raise ValueError("tv_budget must be finite and in [0,1]")
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not -1 <= v <= 1
           for v in values):
        raise ValueError("domain uplifts must be finite in [-1,1]")
    if any(not isinstance(w, (float, int)) or not math.isfinite(w) or w < 0
           for w in weights):
        raise ValueError("weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-10):
        raise ValueError("weights must sum to one")
    n = len(values)
    q = [float(w) for w in weights]
    baseline = sum(v * w for v, w in zip(values, weights))
    order = sorted(range(n), key=lambda i: (values[i], i))
    left, right = 0, n - 1
    remaining = float(tv_budget)
    while remaining > 1e-12 and left < right:
        low, high = order[left], order[right]
        if values[high] <= values[low]:
            break
        move = min(remaining, q[high], 1.0 - q[low])
        if move <= 1e-12:
            if q[high] <= 1e-12:
                right -= 1
            if 1 - q[low] <= 1e-12:
                left += 1
            continue
        q[high] -= move
        q[low] += move
        remaining -= move
        if q[high] <= 1e-12:
            right -= 1
        if 1 - q[low] <= 1e-12:
            left += 1
    realized_tv = sum(abs(a - b) for a, b in zip(q, weights)) / 2
    worst = sum(v * w for v, w in zip(values, q))
    if worst > baseline + 1e-10:
        raise AssertionError("worst-case optimization increased value")
    return {
        "observed_uplift": baseline,
        "worst_case_uplift": worst,
        "requested_tv_budget": float(tv_budget),
        "realized_tv": realized_tv,
        "domain_count": n,
        "warning": "Observed-domain reweighting only; not a confidence bound or guarantee under new tasks",
    }


def topic_mix_stress_table(article_diagnostics, budgets=(0.05, 0.10, 0.20)):
    if not article_diagnostics:
        raise ValueError("article diagnostics required")
    total = sum(row["n"] for row in article_diagnostics)
    if total <= 0:
        raise ValueError("positive question counts required")
    values = [row["difference"] for row in article_diagnostics]
    weights = [row["n"] / total for row in article_diagnostics]
    return [
        worst_case_topic_mix(values, weights, tv_budget=budget)
        for budget in budgets
    ]

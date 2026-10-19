"""Preregistered pilot planning for paired binary agent outcomes.

Approximate sample size for detecting a fixed difference in paired binary
success probabilities. Uses a normal approximation and a simple cluster
design effect. Not a replacement for simulation-based power analysis under
adaptive policies, missing outcomes, unequal clusters, or non-binary rewards.
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist


def paired_binary_pilot_plan(*, target_uplift=0.05,
                             expected_discordance=0.25,
                             alpha=0.05, power=0.80,
                             mean_cluster_size=1,
                             intracluster_correlation=0.0,
                             missing_outcome_fraction=0.0,
                             grade_cost_per_task=0.0,
                             execution_cost_per_task=0.0):
    """Return transparent approximate task count and grading/compute budget.

    Let D in {-1,0,+1} be candidate minus baseline binary success.
    E[D]=delta, E[D^2]=q (discordance), Var[D]=q-delta^2.
    Paired normal approximation:
      n ≈ (z_(1-alpha/2)*sqrt(q) +
           z_power*sqrt(q-delta^2))^2 / delta^2
    The design effect 1+(m-1)*ICC is a rough adjustment for clustering.
    """
    numeric = (target_uplift, expected_discordance, alpha, power,
               intracluster_correlation, missing_outcome_fraction,
               grade_cost_per_task, execution_cost_per_task)
    if any(isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) for v in numeric):
        raise ValueError("all numeric inputs must be finite real numbers")
    if not 0 < target_uplift <= expected_discordance <= 1:
        raise ValueError("target uplift must be positive and no greater than discordance")
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("alpha must be in (0,1), power in (0.5,1)")
    if isinstance(mean_cluster_size, bool) or not isinstance(mean_cluster_size, int) or mean_cluster_size < 1:
        raise ValueError("mean_cluster_size must be a positive integer")
    if not 0 <= intracluster_correlation <= 1:
        raise ValueError("ICC must be in [0,1]")
    if not 0 <= missing_outcome_fraction < 1:
        raise ValueError("missing fraction must be in [0,1)")
    if grade_cost_per_task < 0 or execution_cost_per_task < 0:
        raise ValueError("task costs must be nonnegative")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    q = expected_discordance
    delta = target_uplift
    variance = max(0.0, q - delta ** 2)
    independent_n = math.ceil(((z_alpha * math.sqrt(q)
                                + z_power * math.sqrt(variance)) / delta) ** 2)
    design_effect = 1 + (mean_cluster_size - 1) * intracluster_correlation
    required_labeled = math.ceil(independent_n * design_effect)
    required_requested = math.ceil(required_labeled / (1 - missing_outcome_fraction))
    clusters = math.ceil(required_requested / mean_cluster_size)
    requested_rounded = clusters * mean_cluster_size
    cost_per_task = grade_cost_per_task + execution_cost_per_task
    return {
        "planning_method": "paired_binary_normal_approximation_with_simple_cluster_design_effect",
        "assumed_uplift": delta,
        "assumed_discordance": q,
        "alpha_two_sided": alpha,
        "target_power": power,
        "mean_cluster_size": mean_cluster_size,
        "assumed_intracluster_correlation": intracluster_correlation,
        "assumed_missing_outcome_fraction": missing_outcome_fraction,
        "independent_pairs_required": independent_n,
        "cluster_design_effect": design_effect,
        "labeled_pairs_required": required_labeled,
        "requested_tasks_rounded_to_clusters": requested_rounded,
        "clusters_required": clusters,
        "estimated_task_execution_and_grading_cost": round(requested_rounded * cost_per_task, 2),
        "cost_currency": "operator_supplied",
        "warnings": [
            "Planning estimates are sensitive to the assumed uplift and discordance; no real-world effect is guaranteed.",
            "The cluster design effect is approximate and may understate variance for unequal or adaptive clusters.",
            "Do not use this formula for SNIPS/DR estimators, continuous rewards, sequential peeking, or selective missing labels.",
            "A preregistered external pilot needs independent outcomes and appropriate cluster-aware inference.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uplift", type=float, default=0.05)
    parser.add_argument("--discordance", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--cluster-size", type=int, default=1)
    parser.add_argument("--icc", type=float, default=0.0)
    parser.add_argument("--missing", type=float, default=0.0)
    parser.add_argument("--grade-cost", type=float, default=0.0)
    parser.add_argument("--execution-cost", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(paired_binary_pilot_plan(
        target_uplift=args.uplift,
        expected_discordance=args.discordance,
        alpha=args.alpha,
        power=args.power,
        mean_cluster_size=args.cluster_size,
        intracluster_correlation=args.icc,
        missing_outcome_fraction=args.missing,
        grade_cost_per_task=args.grade_cost,
        execution_cost_per_task=args.execution_cost,
    ), indent=2))


if __name__ == "__main__":
    main()

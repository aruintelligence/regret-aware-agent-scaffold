"""Falsifiable synthetic validation against a known analytic ground truth.

This tests estimator behavior, not the effectiveness of any real AI system.
The environment uses a fixed, explicit Bernoulli reward model with three
contexts and three routes. Both the data generator and truth are inspectable.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from statistics import mean

from crossfit_diagnostics import crossfit_dr
from evidence_lab import paired_bootstrap_difference
from off_policy_evaluation import Decision, estimate
from route_experiments import action_id, choose_policy_action

CONTEXTS = ("design", "research", "mixed")
REWARD_PROBABILITIES = {
    "design": (0.65, 0.55, 0.80),
    "research": (0.60, 0.88, 0.70),
    "mixed": (0.65, 0.72, 0.78),
}
ROUTES = (
    ["systems_design_node", "simulation_modeling_node", "coherence_node", "compression_node"],
    ["research_node", "truth_evidence_node", "meaning_synthesis_node", "coherence_node"],
    ["systems_design_node", "coherence_node"],
)


def analytic_truth() -> dict:
    baseline = mean(REWARD_PROBABILITIES[c][0] for c in CONTEXTS)
    candidate = mean(REWARD_PROBABILITIES[c][2] for c in CONTEXTS)
    return {"baseline": baseline, "candidate": candidate, "uplift": candidate - baseline}


def simulate_trial(n: int, seed: int, epsilon: float = 0.4) -> list:
    if n < 10 or not 0 < epsilon <= 1:
        raise ValueError("n >= 10 and epsilon in (0,1] required")
    rng = random.Random(seed)
    k = len(ROUTES)
    result = []
    for i in range(n):
        context = CONTEXTS[i % len(CONTEXTS)]
        selected = rng.randrange(k) if rng.random() < epsilon else 0
        propensity = epsilon / k + (1 - epsilon if selected == 0 else 0)
        reward = float(rng.random() < REWARD_PROBABILITIES[context][selected])
        decision = {
            "decision_id": f"trial-{seed}-{i}",
            "task_signature": context,
            "action": action_id(ROUTES[selected]),
            "propensity": propensity,
            "candidates": [
                {"action": action_id(route), "route": route,
                 "predicted_score": 0.9 - 0.1 * j}
                for j, route in enumerate(ROUTES)
            ],
        }
        result.append({"decision": decision, "label": {"reward": reward}})
    return result


def _estimates(rows: list, *, seed: int, bootstrap_replicates: int) -> dict:
    observations = [
        Decision(row["decision"]["decision_id"], row["decision"]["action"],
                 row["decision"]["propensity"], row["label"]["reward"])
        for row in rows
    ]
    policies = {}
    for name in ("greedy", "shortest"):
        targets = {row["decision"]["decision_id"]:
                   choose_policy_action(row["decision"], name) for row in rows}
        policies[name] = estimate(observations, lambda context: targets[context])
    dr = crossfit_dr(rows, "greedy", "shortest", seed=seed)
    interval = paired_bootstrap_difference(
        rows, "greedy", "shortest", seed=seed, replicates=bootstrap_replicates,
    )
    return {
        "ips": policies["shortest"]["ips"] - policies["greedy"]["ips"],
        "snips": (policies["shortest"]["snips"] - policies["greedy"]["snips"]
                  if policies["shortest"]["snips"] is not None and policies["greedy"]["snips"] is not None else None),
        "crossfit_dr": dr["difference"],
        "interval": interval,
    }


def run_validation(*, trials: int = 40, n: int = 900,
                   seed: int = 7, bootstrap_replicates: int = 150) -> dict:
    if trials < 2 or n < 30 or bootstrap_replicates < 100:
        raise ValueError("require >=2 trials, >=30 tasks per trial, >=100 bootstrap replicates")
    truth = analytic_truth()
    estimates = []
    for i in range(trials):
        rows = simulate_trial(n, seed + i)
        estimates.append(_estimates(rows, seed=seed + i,
                                    bootstrap_replicates=bootstrap_replicates))
    summaries = {}
    for name in ("ips", "snips", "crossfit_dr"):
        values = [row[name] for row in estimates if row[name] is not None]
        errors = [value - truth["uplift"] for value in values]
        summaries[name] = {
            "trials_estimated": len(values),
            "mean_uplift": mean(values) if values else None,
            "bias": mean(errors) if errors else None,
            "mean_absolute_error": mean(abs(e) for e in errors) if errors else None,
            "root_mean_squared_error": math.sqrt(mean(e * e for e in errors)) if errors else None,
        }
    covered = [
        row["interval"]["lower_95"] <= truth["uplift"] <= row["interval"]["upper_95"]
        for row in estimates if row["interval"]["lower_95"] is not None
    ]
    return {
        "experiment": "synthetic_contextual_bandit_with_known_ground_truth",
        "real_world_validation": False,
        "trials": trials,
        "tasks_per_trial": n,
        "seed": seed,
        "epsilon": 0.4,
        "analytic_truth": truth,
        "estimator_summary": summaries,
        "bootstrap_interval_coverage": {
            "covered": sum(covered),
            "estimable_trials": len(covered),
            "empirical_coverage": mean(covered) if covered else None,
            "nominal_coverage": 0.95,
            "note": "Empirical coverage is a simulation diagnostic, not a guarantee.",
        },
        "limitations": [
            "The reward distribution is authored by the simulator, not discovered from external outcomes.",
            "No actual agent task is solved; this validates statistical code on a toy environment.",
            "The simulated contexts are balanced and independent; real-world tasks may not be.",
            "Monte Carlo estimates fluctuate by seed and sample size.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-replicates", type=int, default=150)
    parser.add_argument("--json", help="Optional output JSON path")
    args = parser.parse_args()
    report = run_validation(
        trials=args.trials, n=args.n, seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        from pathlib import Path
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)

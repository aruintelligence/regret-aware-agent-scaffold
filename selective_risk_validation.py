"""Seeded falsification of the selective-risk gate on known synthetic error rates.

This is a numerical sanity check, not a mathematical proof or a real-model
evaluation. The exact binomial/Bonferroni argument is the conditional
statistical justification under IID sampling assumptions.
"""
from __future__ import annotations

import argparse
import json
import random

from selective_risk_gate import fit_selective_risk_policy


def run_risk_falsification(*, trials=120, rows_per_group=150,
                           seed=20261018, alpha=0.05,
                           target_error=0.1):
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 20:
        raise ValueError("at least 20 trials required")
    if isinstance(rows_per_group, bool) or not isinstance(rows_per_group, int) or rows_per_group < 30:
        raise ValueError("at least 30 observations per group required")
    rng = random.Random(seed)
    results = {}
    for condition, true_error in (("bad_model", 0.2), ("good_model", 0.005)):
        passed = 0
        for _ in range(trials):
            rows = [
                {
                    "sample_id": f"{group}-{i}", "group": group,
                    "score": 0.9, "correct": rng.random() >= true_error,
                }
                for group in ("technical", "general")
                for i in range(rows_per_group)
            ]
            policy = fit_selective_risk_policy(
                rows, thresholds=[0.5], groups=["technical", "general"],
                target_error=target_error, alpha=alpha,
                min_group_support=30,
                model_revision="fixed-synthetic-model",
                rubric_id="known-truth-simulator",
                calibration_set_id="synthetic-only",
            )
            passed += int(policy["conditional_risk_screen_passed"])
        results[condition] = {
            "true_error": true_error,
            "passed_trials": passed,
            "pass_fraction": passed / trials,
        }
    return {
        "kind": "synthetic_selective_risk_falsification",
        "real_world_evidence": False,
        "seed": seed, "trials_per_condition": trials,
        "rows_per_group": rows_per_group,
        "groups": ["technical", "general"],
        "alpha": alpha, "target_error": target_error,
        "conditions": results,
        "automatic_rollout_permitted": False,
        "warning": "Finite Monte Carlo is not a guarantee; risk bounds require independent representative calibration and future tasks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--rows-per-group", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_risk_falsification(
        trials=args.trials, rows_per_group=args.rows_per_group, seed=args.seed,
    )
    result = json.dumps(report, sort_keys=True, indent=2)
    if args.output:
        from pathlib import Path
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()

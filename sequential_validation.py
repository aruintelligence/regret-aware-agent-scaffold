"""Reproducible Monte Carlo falsification of prospective e-process behavior.

This tests a declared synthetic conditional null with fair discordant
directions, plus a declared positive alternative. Empirical false-alarm
rates fluctuate; this is a numerical sanity check, not a proof. The
supermartingale argument in MATHEMATICAL_FOUNDATIONS.md is the proof
under the conditional null assumptions.
"""
from __future__ import annotations

import argparse
import json
import random

from anytime_evidence import paired_e_process


def run_sequential_validation(*, trials=400, tasks=250, alpha=0.05, seed=20261018):
    if trials < 20 or tasks < 10:
        raise ValueError("at least 20 trials and 10 tasks required")
    rng = random.Random(seed)
    settings = {
        "null": (0.20, 0.20),  # baseline-only, candidate-only
        "alternative": (0.10, 0.30),
    }
    result = {}
    for name, (baseline_only_p, candidate_only_p) in settings.items():
        alarms = 0
        crossing_times = []
        for _ in range(trials):
            pairs = []
            for _ in range(tasks):
                draw = rng.random()
                if draw < baseline_only_p:
                    pairs.append((1, 0))
                elif draw < baseline_only_p + candidate_only_p:
                    pairs.append((0, 1))
                else:
                    pairs.append((1, 1))
            evidence = paired_e_process(pairs, alpha=alpha)
            alarms += int(evidence["ever_crossed_threshold"])
            if evidence["first_crossing_pair"] is not None:
                crossing_times.append(evidence["first_crossing_pair"])
        result[name] = {
            "crossed_trials": alarms,
            "crossing_fraction": alarms / trials,
            "mean_first_crossing_pair": sum(crossing_times) / len(crossing_times)
            if crossing_times else None,
            "baseline_only_probability": baseline_only_p,
            "candidate_only_probability": candidate_only_p,
        }
    return {
        "kind": "synthetic_prospective_e_process_validation",
        "real_world_evidence": False,
        "trials_per_condition": trials,
        "tasks_per_trial": tasks,
        "alpha": alpha,
        "seed": seed,
        "conditions": result,
        "interpretation": (
            "Under the stated synthetic conditional null, empirical anytime false alarms "
            "should be near or below alpha on average; finite Monte Carlo results vary. "
            "The positive alternative illustrates detection power, not a guarantee."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_sequential_validation(
        trials=args.trials, tasks=args.tasks, alpha=args.alpha, seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

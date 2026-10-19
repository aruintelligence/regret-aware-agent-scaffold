"""Cross-fitted doubly robust policy-value diagnostics.

Uses held-out folds and shrinkage reward estimates by task signature/route.
This is a diagnostic, NOT an independently validated reward model. With
correct randomized propensities, DR can reduce variance, but small data,
clipping, nonstationarity, and dependent tasks can invalidate inference.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import List

from route_experiments import choose_policy_action


def _fold(decision_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:{decision_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def crossfit_dr(rows: List[dict], baseline_policy: str, candidate_policy: str,
                *, folds: int = 5, seed: int = 7,
                prior_strength: float = 8.0, max_weight: float = 20.0) -> dict:
    """Evaluate deterministic target policies with out-of-fold reward models.

    The nuisance reward model is fit ONLY on other folds. Context is the
    coarse pre-decision task signature, not the original prompt text.
    """
    if folds < 2 or len(rows) < 2 * folds:
        return {"baseline_dr": None, "candidate_dr": None, "difference": None,
                "folds": folds, "warning": "Insufficient tasks for cross-fitting"}
    if prior_strength <= 0 or max_weight < 1:
        raise ValueError("prior_strength must be positive; max_weight >= 1")
    assignments = [
        _fold(row["decision"]["decision_id"], seed, folds) for row in rows
    ]
    contributions = {baseline_policy: [], candidate_policy: []}
    for held_out in range(folds):
        train = [row for row, f in zip(rows, assignments) if f != held_out]
        test = [row for row, f in zip(rows, assignments) if f == held_out]
        if not test:
            continue
        if not train:
            return {"baseline_dr": None, "candidate_dr": None,
                    "difference": None, "folds": folds,
                    "warning": "Empty training fold"}
        global_mean = sum(row["label"]["reward"] for row in train) / len(train)
        action_totals = defaultdict(lambda: [0.0, 0])
        context_action_totals = defaultdict(lambda: [0.0, 0])
        for row in train:
            decision = row["decision"]
            reward = row["label"]["reward"]
            action_totals[decision["action"]][0] += reward
            action_totals[decision["action"]][1] += 1
            key = (decision["task_signature"], decision["action"])
            context_action_totals[key][0] += reward
            context_action_totals[key][1] += 1

        def predict(signature, action):
            action_sum, action_count = action_totals[action]
            action_prior = ((action_sum + prior_strength * global_mean) /
                            (action_count + prior_strength))
            total, count = context_action_totals[(signature, action)]
            return (total + prior_strength * action_prior) / (count + prior_strength)

        for row in test:
            decision = row["decision"]
            for policy in contributions:
                target = choose_policy_action(decision, policy)
                mu = predict(decision["task_signature"], target)
                matched = decision["action"] == target
                weight = min(1 / decision["propensity"], max_weight) if matched else 0.0
                contributions[policy].append(mu + weight * (row["label"]["reward"] - mu))
    if not all(len(values) == len(rows) for values in contributions.values()):
        raise RuntimeError("cross-fit fold coverage error")
    baseline = sum(contributions[baseline_policy]) / len(rows)
    candidate = sum(contributions[candidate_policy]) / len(rows)
    return {
        "baseline_dr": baseline,
        "candidate_dr": candidate,
        "difference": candidate - baseline,
        "folds": folds,
        "prior_strength": prior_strength,
        "warning": "DR may be biased by clipping or invalid logging propensities"
                   if any(1 / row["decision"]["propensity"] > max_weight
                          for row in rows) else None,
    }


def segment_diagnostics(rows: List[dict], baseline_policy: str,
                        candidate_policy: str, *,
                        max_weight: float = 20.0) -> List[dict]:
    """Inspect domain-level negative outcomes without inventing subgroup labels."""
    from off_policy_evaluation import Decision, estimate

    groups = defaultdict(list)
    for row in rows:
        groups[row["decision"]["task_signature"]].append(row)
    result = []
    for signature, group in groups.items():
        observations = [
            Decision(context=row["decision"]["decision_id"],
                     action=row["decision"]["action"],
                     propensity=row["decision"]["propensity"],
                     reward=row["label"]["reward"])
            for row in group
        ]
        summaries = {}
        for policy in (baseline_policy, candidate_policy):
            actions = {row["decision"]["decision_id"]:
                       choose_policy_action(row["decision"], policy)
                       for row in group}
            summaries[policy] = estimate(
                observations, lambda context: actions[context], max_weight=max_weight,
            )
        b = summaries[baseline_policy]
        c = summaries[candidate_policy]
        delta = c["snips"] - b["snips"] if b["snips"] is not None and c["snips"] is not None else None
        material_drop = (len(group) >= 30 and b["effective_sample_size"] >= 10
                         and c["effective_sample_size"] >= 10
                         and delta is not None and delta < -0.05)
        result.append({
            "task_signature": signature,
            "n_labeled": len(group),
            "baseline_snips": b["snips"],
            "candidate_snips": c["snips"],
            "difference": delta,
            "baseline_ess": b["effective_sample_size"],
            "candidate_ess": c["effective_sample_size"],
            "material_negative_segment": material_drop,
        })
    return sorted(result, key=lambda row: (-row["n_labeled"], row["task_signature"]))

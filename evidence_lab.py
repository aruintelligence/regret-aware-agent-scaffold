"""Join independent outcomes to route decisions and compare policies.

No automatic deployment is authorized. Bootstrap intervals are exploratory
and assume independent tasks; adaptive, clustered, or delayed outcomes need
a preregistered evaluation protocol and cluster-aware inference.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Dict, List, Optional

from off_policy_evaluation import Decision, estimate
from route_experiments import (
    append_jsonl, choose_policy_action, load_decisions, read_jsonl,
)

LABEL_SCHEMA_VERSION = 1
SOURCE_KINDS = frozenset({"external", "synthetic"})


def validate_label(row: dict) -> None:
    if row.get("schema_version") != LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported label schema")
    for key in ("decision_id", "grader", "rubric_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} is required")
    if row.get("source_kind") not in SOURCE_KINDS:
        raise ValueError("source_kind must be external or synthetic")
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("reward must be finite and in [0,1]")


def load_labels(path: str) -> Dict[str, dict]:
    labels = {}
    for row in read_jsonl(path):
        validate_label(row)
        decision_id = row["decision_id"]
        if decision_id in labels:
            raise ValueError(f"duplicate label: {decision_id}")
        labels[decision_id] = row
    return labels


def add_label(decisions_path: str, labels_path: str, decision_id: str,
              reward: float, grader: str, rubric_id: str,
              source_kind: str) -> dict:
    decisions = load_decisions(decisions_path)
    ids = {row["decision_id"] for row in decisions}
    if decision_id not in ids:
        raise ValueError("decision_id not found in decision log")
    try:
        labels = load_labels(labels_path)
    except FileNotFoundError:
        labels = {}
    if decision_id in labels:
        raise ValueError("decision already labeled; no silent overwrite")
    row = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "decision_id": decision_id,
        "reward": reward,
        "grader": grader,
        "rubric_id": rubric_id,
        "source_kind": source_kind,
        "graded_at_unix": time.time(),
    }
    validate_label(row)
    append_jsonl(labels_path, row)
    return row


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def paired_bootstrap_difference(rows: List[dict], baseline_policy: str,
                                candidate_policy: str, *, replicates: int = 1000,
                                seed: int = 7, max_weight: float = 20.0) -> dict:
    """Paired SNIPS difference, with both policies evaluated on the same draws."""
    if replicates < 100 or not rows:
        raise ValueError("need nonempty rows and at least 100 replicates")
    n = len(rows)
    baseline_w = []
    candidate_w = []
    rewards = []
    for row in rows:
        decision = row["decision"]
        rewards.append(row["label"]["reward"])
        baseline_w.append(min(1 / decision["propensity"], max_weight)
                          if choose_policy_action(decision, baseline_policy) == decision["action"] else 0)
        candidate_w.append(min(1 / decision["propensity"], max_weight)
                           if choose_policy_action(decision, candidate_policy) == decision["action"] else 0)
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        sampled = [rng.randrange(n) for _ in range(n)]
        bw = sum(baseline_w[i] for i in sampled)
        cw = sum(candidate_w[i] for i in sampled)
        if bw == 0 or cw == 0:
            continue
        b = sum(baseline_w[i] * rewards[i] for i in sampled) / bw
        c = sum(candidate_w[i] * rewards[i] for i in sampled) / cw
        differences.append(c - b)
    if len(differences) < 0.95 * replicates:
        return {"lower_95": None, "upper_95": None,
                "valid_replicates": len(differences),
                "warning": "Insufficient bootstrap support for paired interval"}
    return {
        "lower_95": _percentile(differences, 0.025),
        "upper_95": _percentile(differences, 0.975),
        "valid_replicates": len(differences),
        "warning": None,
    }


def evaluate_experiment(decisions_path: str, labels_path: str, *,
                        candidate_policy: str = "shortest",
                        baseline_policy: str = "greedy",
                        replicates: int = 1000, seed: int = 7,
                        max_weight: float = 20.0,
                        min_labeled: int = 200, min_ess: float = 30.0,
                        min_uplift: float = 0.02) -> dict:
    if candidate_policy not in ("greedy", "shortest", "evidence_first"):
        raise ValueError("unsupported candidate policy")
    if baseline_policy not in ("greedy", "shortest", "evidence_first"):
        raise ValueError("unsupported baseline policy")
    if not isinstance(min_labeled, int) or min_labeled < 1 or not math.isfinite(min_ess) or min_ess < 1:
        raise ValueError("invalid review thresholds")
    if not math.isfinite(min_uplift) or not -1 <= min_uplift <= 1:
        raise ValueError("invalid min_uplift")
    decisions = load_decisions(decisions_path)
    if not decisions:
        raise ValueError("empty decision log")
    labels = load_labels(labels_path)
    ids = {row["decision_id"] for row in decisions}
    unknown = set(labels) - ids
    if unknown:
        raise ValueError("labels refer to unknown decision IDs")
    rows = [{"decision": row, "label": labels[row["decision_id"]]}
            for row in decisions if row["decision_id"] in labels]
    if not rows:
        raise ValueError("no labeled decisions")
    source_counts = dict(Counter(row["label"]["source_kind"] for row in rows))
    grader_count = len({row["label"]["grader"] for row in rows})
    policies = {}
    for name in (baseline_policy, candidate_policy):
        actions = {row["decision"]["decision_id"]:
                   choose_policy_action(row["decision"], name) for row in rows}
        observations = [
            Decision(context=row["decision"]["decision_id"],
                     action=row["decision"]["action"],
                     propensity=row["decision"]["propensity"],
                     reward=row["label"]["reward"])
            for row in rows
        ]
        policies[name] = estimate(observations, lambda context: actions[context],
                                  max_weight=max_weight)
    baseline = policies[baseline_policy]
    candidate = policies[candidate_policy]
    paired = paired_bootstrap_difference(
        rows, baseline_policy, candidate_policy,
        replicates=replicates, seed=seed, max_weight=max_weight,
    )
    difference = (candidate["snips"] - baseline["snips"]
                  if candidate["snips"] is not None and baseline["snips"] is not None
                  else None)
    paired["snips_difference"] = difference

    # Advisory review gate only; never silently change production policy.
    reasons = []
    if len(rows) < min_labeled:
        reasons.append(f"Only {len(rows)} labeled tasks; require {min_labeled}")
    if len(rows) != len(decisions):
        reasons.append("Incomplete labels; selective missingness may bias estimates")
    if source_counts.get("synthetic", 0):
        reasons.append("Synthetic labels cannot justify real-world deployment")
    if baseline["effective_sample_size"] < min_ess or candidate["effective_sample_size"] < min_ess:
        reasons.append(f"Effective sample size below {min_ess:g}")
    if baseline["clipped"] or candidate["clipped"]:
        reasons.append("Clipped importance weights introduce bias")
    if baseline["match_rate"] < 0.05 or candidate["match_rate"] < 0.05:
        reasons.append("Insufficient target-policy overlap")
    if paired["lower_95"] is None or paired["lower_95"] <= min_uplift:
        reasons.append(f"Exploratory lower confidence bound does not exceed {min_uplift:g} uplift")
    if candidate_policy == baseline_policy:
        reasons.append("Candidate and baseline policies are identical")
    if not reasons:
        reasons.append("Statistical screen passed; independent audit and prospective trial still required")
    candidate_for_review = len(reasons) == 1 and reasons[0].startswith("Statistical screen passed")
    return {
        "report_schema_version": 1,
        "experiment_kind": "logged_randomized_route_policy_comparison",
        "runtime_kind": sorted({row["runtime_kind"] for row in decisions}),
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "n_logged": len(decisions),
        "n_labeled": len(rows),
        "n_unlabeled": len(decisions) - len(rows),
        "label_source_counts": source_counts,
        "distinct_graders": grader_count,
        "observed_action_counts": dict(Counter(row["action"] for row in decisions)),
        "baseline": baseline,
        "candidate": candidate,
        "paired_difference": paired,
        "review_gate": {
            "candidate_for_human_review": candidate_for_review,
            "automatic_rollout_permitted": False,
            "reasons": reasons,
            "minimum_labeled": min_labeled,
            "minimum_effective_sample_size": min_ess,
            "minimum_uplift": min_uplift,
        },
        "limitations": [
            "The existing specialists are scripted, not LLMs; internal quality is not an external reward.",
            "External labels are self-attested by the data provider, not independently authenticated.",
            "SNIPS is generally biased at finite samples; clipped IPS is also biased.",
            "The percentile bootstrap assumes independent tasks and may be invalid for adaptive or clustered logs.",
            "Evaluation uses only observed contexts and cannot establish performance under distribution shift.",
            "A passed screen is not proof of a causal gain and never authorizes automatic deployment.",
        ],
    }

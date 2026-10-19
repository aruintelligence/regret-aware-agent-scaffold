"""Opt-in randomized route logging for the scripted ĀRU routing scaffold.

This is instrumentation, NOT evidence of real-world agent performance.
No raw prompts, user identities, or internal proxy scores are stored as rewards.
The experimental runtime executes one selected route without retries so a
single independently graded outcome can be attributed to that decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from regret_aware_scaffold import (
    MasterAgentRuntime, PlannerNode, RouteMemory, task_signature_from_state,
)

SCHEMA_VERSION = 1


def action_id(route: List[str]) -> str:
    """Stable identifier for an ordered specialist route."""
    if not isinstance(route, list) or not route or not all(
        isinstance(node, str) and node for node in route
    ):
        raise ValueError("route must be a nonempty list of node names")
    canonical = json.dumps(route, ensure_ascii=True, separators=(",", ":"))
    return "route_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def append_jsonl(path: str, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Single JSONL record per write; single-process collection is supported.
    # For multi-process production ingestion, use a transactional database.
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def validate_decision(row: dict) -> None:
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported decision schema")
    if not isinstance(row.get("decision_id"), str) or not row["decision_id"]:
        raise ValueError("decision_id is required")
    if not isinstance(row.get("task_signature"), str) or not row["task_signature"]:
        raise ValueError("task_signature is required")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate routes are required")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid candidate")
        identifier = action_id(candidate.get("route"))
        if candidate.get("action") != identifier:
            raise ValueError("candidate action does not match route")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate routes")
    if row.get("action") not in ids:
        raise ValueError("selected action missing from candidates")
    policy = row.get("logging_policy")
    propensity = row.get("propensity")
    if not isinstance(propensity, (int, float)) or isinstance(propensity, bool) or not math.isfinite(propensity) or not 0 < propensity <= 1:
        raise ValueError("propensity must be finite and in (0,1]")
    if policy == "epsilon_greedy":
        epsilon = row.get("epsilon")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("epsilon must be in (0,1]")
        expected = epsilon / len(ids) + (1 - epsilon if row["action"] == ids[0] else 0)
    elif policy == "categorical":
        probabilities = row.get("candidate_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(ids):
            raise ValueError("categorical policy requires one probability per candidate")
        if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or p <= 0 for p in probabilities):
            raise ValueError("categorical probabilities must be finite and positive")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("categorical probabilities must sum to 1")
        expected = probabilities[ids.index(row["action"])]
    else:
        raise ValueError("unsupported logging policy")
    if not math.isclose(propensity, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("propensity inconsistent with logged policy")


def load_decisions(path: str) -> List[dict]:
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        validate_decision(row)
        if row["decision_id"] in seen:
            raise ValueError("duplicate decision_id")
        seen.add(row["decision_id"])
    return rows


def choose_policy_action(row: dict, policy: str) -> str:
    """Deterministic policies defined entirely from pre-decision candidates."""
    candidates = row["candidates"]
    if policy == "greedy":
        return candidates[0]["action"]
    if policy == "shortest":
        winner = min(candidates, key=lambda c: (
            len(c["route"]), -float(c.get("predicted_score", 0)),
            c["action"],
        ))
        return winner["action"]
    if policy == "evidence_first":
        for candidate in candidates:
            if "truth_evidence_node" in candidate["route"]:
                return candidate["action"]
        return candidates[0]["action"]
    raise ValueError(f"unknown policy: {policy}")


class ExperimentPlannerNode(PlannerNode):
    """Epsilon-greedy selection with known probabilities and private log."""

    def __init__(self, memory: RouteMemory, decisions_path: str,
                 epsilon: float = 0.25, seed: Optional[int] = None):
        super().__init__(memory)
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("epsilon must be in (0,1]")
        if not decisions_path:
            raise ValueError("decisions_path is required")
        self.decisions_path = decisions_path
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)

    def process(self, state):
        outputs, delta, recommendations, confidence, complexity = super().process(state)
        candidates = outputs["candidate_routes"]
        if not candidates:
            raise RuntimeError("planner returned no routes")
        k = len(candidates)
        index = self.rng.randrange(k) if self.rng.random() < self.epsilon else 0
        chosen = candidates[index]
        selected_action = action_id(chosen["route"])
        propensity = self.epsilon / k + (1 - self.epsilon if index == 0 else 0)
        decision_id = uuid.uuid4().hex
        record = {
            "schema_version": SCHEMA_VERSION,
            "decision_id": decision_id,
            "timestamp_unix": time.time(),
            "task_signature": task_signature_from_state(state),
            "logging_policy": "epsilon_greedy",
            "epsilon": self.epsilon,
            "propensity": propensity,
            "action": selected_action,
            "candidates": [
                {
                    "action": action_id(candidate["route"]),
                    "route": list(candidate["route"]),
                    "predicted_score": candidate["predicted_score"],
                }
                for candidate in candidates
            ],
            "runtime_kind": "scripted_scaffold",
            "loop_count": state.route.loop_count,
        }
        validate_decision(record)
        # Fail closed: a route is not executed without a logged propensity.
        append_jsonl(self.decisions_path, record)
        outputs["selected_route"] = list(chosen["route"])
        outputs["planner_confidence"] = chosen["predicted_score"]
        outputs["selected_prediction"] = {
            "predicted_quality": chosen["predicted_quality"],
            "predicted_cost": chosen["predicted_cost"],
            "predicted_score": chosen["predicted_score"],
        }
        outputs["experiment"] = {
            "decision_id": decision_id,
            "action": selected_action,
            "propensity": propensity,
            "epsilon": self.epsilon,
        }
        recommendations.reason = "Logged randomized single-route experiment"
        return outputs, delta, recommendations, confidence, complexity


class ExperimentRuntime(MasterAgentRuntime):
    """One-route experimental runtime; regular MasterAgentRuntime is unchanged."""

    def __init__(self, decisions_path: str, epsilon: float = 0.25,
                 seed: Optional[int] = None, memory: Optional[RouteMemory] = None):
        super().__init__(memory=memory)
        self.nodes["planner_node"] = ExperimentPlannerNode(
            self.memory, decisions_path, epsilon=epsilon, seed=seed,
        )

    def new_state(self, user_input: str, session_id: str = "default"):
        state = super().new_state(user_input, session_id=session_id)
        # No downstream retries: each external outcome refers to one route.
        state.route.max_loops = 0
        return state

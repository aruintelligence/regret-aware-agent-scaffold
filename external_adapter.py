"""Vendor-neutral route-decision instrumentation adapter.

Call immediately when an external controller makes a *prospective randomized*
choice. Do not reconstruct probabilities after seeing the outcome. The
caller is responsible for truthful propensity logging, user consent, and
separate independently graded outcomes. No raw prompts are stored here.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import List, Optional

from route_experiments import (
    SCHEMA_VERSION, action_id, append_jsonl, validate_decision,
)


def log_instrumented_decision(
    decisions_path: str,
    *,
    task_signature: str,
    candidate_routes: List[List[str]],
    selected_route: List[str],
    selection_probabilities: List[float],
    predicted_scores: Optional[List[float]] = None,
) -> dict:
    """Append a verified-format event with the *actual* selection probability.

    The logging mechanism itself cannot be authenticated by this adapter.
    Probability values must come from the deployed randomized controller.
    """
    if not isinstance(task_signature, str) or not task_signature.strip():
        raise ValueError("nonempty task_signature required")
    if not isinstance(candidate_routes, list) or not candidate_routes:
        raise ValueError("candidate_routes must be nonempty")
    if not isinstance(selection_probabilities, list) or len(selection_probabilities) != len(candidate_routes):
        raise ValueError("one probability per candidate is required")
    if predicted_scores is None:
        predicted_scores = [0.0] * len(candidate_routes)
    if not isinstance(predicted_scores, list) or len(predicted_scores) != len(candidate_routes):
        raise ValueError("one predicted score per candidate is required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in predicted_scores):
        raise ValueError("predicted scores must be finite")
    action = action_id(selected_route)
    candidates = [
        {"route": list(route), "action": action_id(route), "predicted_score": float(score)}
        for route, score in zip(candidate_routes, predicted_scores)
    ]
    ids = [candidate["action"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate routes")
    if action not in ids:
        raise ValueError("selected route must be one of the candidates")
    propensity = selection_probabilities[ids.index(action)]
    record = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": uuid.uuid4().hex,
        "timestamp_unix": time.time(),
        "task_signature": task_signature,
        "logging_policy": "categorical",
        "candidate_probabilities": list(selection_probabilities),
        "propensity": propensity,
        "action": action,
        "candidates": candidates,
        "runtime_kind": "external_instrumented",
        "loop_count": 0,
    }
    validate_decision(record)
    append_jsonl(decisions_path, record)
    return {"decision_id": record["decision_id"], "action": action,
            "propensity": propensity, "task_signature": task_signature}

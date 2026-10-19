"""Canonical, decision-bound analysis plan for an opt-in routing experiment.

The digest detects inconsistencies between a protocol and a decision log.
Neither a local timestamp nor a hash proves the protocol existed before data;
independent publication or a trusted witness is needed for that claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


POLICIES = ("greedy", "shortest", "evidence_first", "first_candidate", "second_candidate")
ANALYSIS_KEYS = frozenset({"replicates", "seed", "max_weight", "min_labeled", "min_ess", "min_uplift"})
PROTOCOL_KEYS = frozenset({"schema_version", "study_id", "baseline_policy", "candidate_policy", "rubric_id", "analysis"})


def make_protocol(study_id: str, baseline_policy: str, candidate_policy: str,
                  rubric_id: str, *, replicates: int = 1000, seed: int = 7,
                  max_weight: float = 20.0, min_labeled: int = 200,
                  min_ess: float = 30.0, min_uplift: float = 0.02) -> dict:
    protocol = {
        "schema_version": 1,
        "study_id": study_id,
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "rubric_id": rubric_id,
        "analysis": {
            "replicates": replicates, "seed": seed, "max_weight": max_weight,
            "min_labeled": min_labeled, "min_ess": min_ess,
            "min_uplift": min_uplift,
        },
    }
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if (not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS or
            type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1):
        raise ValueError("unsupported or incomplete experiment protocol")
    for key in ("study_id", "rubric_id"):
        if not isinstance(protocol[key], str) or not protocol[key].strip() or len(protocol[key]) > 128:
            raise ValueError(f"{key} must be a nonempty short string")
    for key in ("baseline_policy", "candidate_policy"):
        if protocol[key] not in POLICIES:
            raise ValueError(f"unsupported {key}")
    if protocol["baseline_policy"] == protocol["candidate_policy"]:
        raise ValueError("baseline and candidate policies must differ")
    analysis = protocol["analysis"]
    if not isinstance(analysis, dict) or set(analysis) != ANALYSIS_KEYS:
        raise ValueError("incomplete analysis plan")
    for key, minimum in (("replicates", 100), ("min_labeled", 1)):
        value = analysis[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if isinstance(analysis["seed"], bool) or not isinstance(analysis["seed"], int):
        raise ValueError("seed must be an integer")
    for key, minimum, maximum in (("max_weight", 1, math.inf),
                                   ("min_ess", 1, math.inf),
                                   ("min_uplift", -1, 1)):
        value = analysis[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"invalid {key}")


def protocol_digest(protocol: dict) -> str:
    validate_protocol(protocol)
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_protocol(protocol: dict, path: str) -> None:
    validate_protocol(protocol)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate protocol key: {key}")
        value[key] = item
    return value


def load_protocol(path: str) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_pairs)
    validate_protocol(protocol)
    return protocol


def check_protocol_binding(protocol: dict, decisions: list, labels: dict, *,
                           baseline_policy: str, candidate_policy: str,
                           replicates: int, seed: int, max_weight: float,
                           min_labeled: int, min_ess: float,
                           min_uplift: float) -> dict:
    digest = protocol_digest(protocol)
    if any(row.get("protocol_sha256") != digest for row in decisions):
        raise ValueError("a decision is missing or mismatches the experiment protocol")
    if any(label["rubric_id"] != protocol["rubric_id"] for label in labels.values()):
        raise ValueError("an outcome label does not match the fixed rubric")
    expected = protocol["analysis"]
    supplied = {"replicates": replicates, "seed": seed,
                "max_weight": max_weight, "min_labeled": min_labeled,
                "min_ess": min_ess, "min_uplift": min_uplift}
    if baseline_policy != protocol["baseline_policy"] or candidate_policy != protocol["candidate_policy"] or supplied != expected:
        raise ValueError("analysis differs from the decision-bound protocol")
    return {
        "status": "decision_bound_self_attested",
        "sha256": digest,
        "study_id": protocol["study_id"],
        "rubric_id": protocol["rubric_id"],
        "warning": "Digest consistency does not prove the protocol predates outcomes; obtain an independent timestamp or witness.",
    }

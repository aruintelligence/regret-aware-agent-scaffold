"""Paired cold/warm comparison of the scaffold's internal proxy behavior.

This is a software regression benchmark, not an evaluation of agent intelligence
or success on a real-world task. Both arms use the same scripted nodes.
"""

import argparse
import json
from statistics import mean

from regret_aware_scaffold import MasterAgentRuntime


PROMPTS = {
    "design": "Design a Python agent scaffold with persistent route memory.",
    "research": "Research evidence for assessing multi-agent route quality.",
    "mixed": "Design and research a Python master agent scaffold with route memory and evaluation.",
}


def measure(state):
    feedback = state.agent_outputs["feedback_aggregator_node"]
    return {
        "executed_nodes": len(state.route.selected_path),
        "reroutes": state.route.loop_count,
        "internal_quality": round(state.evaluation.overall_quality, 6),
        "internal_gate_passed": state.evaluation.passed,
        "absolute_prediction_error": round(abs(feedback["prediction_error"]), 6),
        "route_hints": len(state.memory_hints.get("best_known_paths", [])),
    }


def compare(runs=5):
    if runs < 2:
        raise ValueError("runs must be at least 2")
    result = {}
    for name, prompt in PROMPTS.items():
        # Each cold run starts with an empty memory. The warm arm reuses only
        # observations from the same prompt; no training/test data are hidden.
        cold = [measure(MasterAgentRuntime().run(prompt)) for _ in range(runs)]
        warm_runtime = MasterAgentRuntime()
        warm = [measure(warm_runtime.run(prompt)) for _ in range(runs)]
        result[name] = {
            "cold": cold,
            "warm": warm,
            "comparison": {
                key: {
                    "cold_mean": round(mean(row[key] for row in cold[1:]), 6),
                    "warm_mean": round(mean(row[key] for row in warm[1:]), 6),
                }
                for key in ("executed_nodes", "reroutes", "internal_quality", "absolute_prediction_error")
            },
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs), indent=2))

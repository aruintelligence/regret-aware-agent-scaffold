import math
import unittest

from crossfit_diagnostics import crossfit_dr, segment_diagnostics
from route_experiments import action_id


def fixtures():
    long_route = ["systems_design_node", "coherence_node", "compression_node"]
    short_route = ["coherence_node"]
    baseline_action = action_id(long_route)
    candidate_action = action_id(short_route)
    rows = []
    for context in ("strong", "weak"):
        for i in range(100):
            chosen = baseline_action if i % 2 == 0 else candidate_action
            if context == "strong":
                reward = 0.6 if chosen == baseline_action else 0.9
            else:
                reward = 0.9 if chosen == baseline_action else 0.1
            decision = {
                "decision_id": f"{context}-{i}",
                "task_signature": context,
                "action": chosen,
                "propensity": 0.5,
                "candidates": [
                    {"route": long_route, "action": baseline_action, "predicted_score": 0.9},
                    {"route": short_route, "action": candidate_action, "predicted_score": 0.5},
                ],
            }
            rows.append({"decision": decision, "label": {"reward": reward}})
    return rows


class CrossfitDiagnosticsTests(unittest.TestCase):
    def test_crossfit_reproducible_and_finite(self):
        rows = fixtures()
        a = crossfit_dr(rows, "greedy", "shortest", folds=5, seed=42)
        b = crossfit_dr(rows, "greedy", "shortest", folds=5, seed=42)
        self.assertEqual(a, b)
        self.assertTrue(math.isfinite(a["baseline_dr"]))
        self.assertTrue(math.isfinite(a["candidate_dr"]))
        self.assertLess(a["difference"], 0)

    def test_segments_identify_material_negative_cohort(self):
        segments = segment_diagnostics(fixtures(), "greedy", "shortest")
        by_name = {segment["task_signature"]: segment for segment in segments}
        self.assertFalse(by_name["strong"]["material_negative_segment"])
        self.assertTrue(by_name["weak"]["material_negative_segment"])
        self.assertGreater(by_name["strong"]["difference"], 0)
        self.assertLess(by_name["weak"]["difference"], 0)

    def test_small_samples_are_not_reported_as_dr_proof(self):
        result = crossfit_dr(fixtures()[:4], "greedy", "shortest", folds=5)
        self.assertIsNone(result["difference"])
        self.assertIn("Insufficient", result["warning"])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from external_adapter import log_instrumented_decision
from route_experiments import load_decisions


class ExternalAdapterTests(unittest.TestCase):
    def test_logs_real_categorical_propensity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "external.jsonl")
            result = log_instrumented_decision(
                path, task_signature="research|public-data",
                candidate_routes=[["research_node", "truth_evidence_node"], ["coherence_node"]],
                selected_route=["coherence_node"],
                selection_probabilities=[0.8, 0.2],
                predicted_scores=[0.8, 0.5],
            )
            row = load_decisions(path)[0]
            self.assertEqual(row["logging_policy"], "categorical")
            self.assertEqual(row["runtime_kind"], "external_instrumented")
            self.assertAlmostEqual(row["propensity"], 0.2)
            self.assertEqual(result["decision_id"], row["decision_id"])
            self.assertNotIn("prompt", row)

    def test_greedy_target_is_highest_scoring_even_when_external_routes_unsorted(self):
        from route_experiments import choose_policy_action
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "external.jsonl")
            log_instrumented_decision(
                path, task_signature="design",
                candidate_routes=[["short"], ["long", "route"]],
                selected_route=["short"],
                selection_probabilities=[0.5, 0.5],
                predicted_scores=[0.2, 0.9],
            )
            row = load_decisions(path)[0]
            self.assertEqual(choose_policy_action(row, "greedy"), row["candidates"][1]["action"])

    def test_rejects_invalid_probability_and_never_writes_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "external.jsonl"
            with self.assertRaises(ValueError):
                log_instrumented_decision(
                    str(path), task_signature="design",
                    candidate_routes=[["a"], ["b"]], selected_route=["a"],
                    selection_probabilities=[0.8, 0.8],
                )
            self.assertFalse(path.exists())

    def test_rejects_unsupported_selected_route(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "external.jsonl")
            with self.assertRaises(ValueError):
                log_instrumented_decision(
                    path, task_signature="design",
                    candidate_routes=[["a"], ["b"]], selected_route=["c"],
                    selection_probabilities=[0.5, 0.5],
                )

    def test_rejects_duplicate_candidate_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "external.jsonl")
            with self.assertRaises(ValueError):
                log_instrumented_decision(
                    path, task_signature="design",
                    candidate_routes=[["a"], ["a"]], selected_route=["a"],
                    selection_probabilities=[0.5, 0.5],
                )


if __name__ == "__main__":
    unittest.main()

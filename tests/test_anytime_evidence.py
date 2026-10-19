"""Boundary and integration checks for time-uniform policy evidence."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from anytime_evidence import anytime_ips_lower_bound
from evidence_dashboard import render_dashboard
from evidence_lab import add_label, evaluate_experiment
from external_adapter import log_instrumented_decision
from experiment_cli import demo
from route_experiments import action_id, load_decisions


ROUTES = [["research_node", "truth_evidence_node"], ["coherence_node"]]
ACTIONS = [action_id(route) for route in ROUTES]


def labeled_row(index, epsilon, selected):
    q = epsilon / 2 + (1 - epsilon if selected == 0 else 0)
    return {
        "decision": {
            "schema_version": 1, "decision_id": f"d-{index}",
            "task_signature": "test", "candidates": [
                {"action": action, "route": route, "predicted_score": 1 - j}
                for j, (action, route) in enumerate(zip(ACTIONS, ROUTES))
            ],
            "action": ACTIONS[selected], "logging_policy": "epsilon_greedy",
            "epsilon": epsilon, "propensity": q,
        },
        "label": {"reward": float(selected == 1)},
    }


class AnytimeEvidenceTests(unittest.TestCase):
    def test_adaptive_probabilities_use_the_pre_action_target_probabilities(self):
        rows = [labeled_row(0, 0.5, 1), labeled_row(1, 1.0, 0)]
        result = anytime_ips_lower_bound(rows, "greedy", "shortest")
        # The first candidate route is selected at probability 1/4, giving
        # an unclipped IPS contribution of 4 for a reward of 1.
        self.assertAlmostEqual(result["ips_difference_unclipped"], 2)
        self.assertAlmostEqual(result["range_square_sum"], (4 + 4 / 3) ** 2 + 4 ** 2)
        self.assertLessEqual(result["lower_bound"], 1)
        self.assertGreater(result["radius"], 0)

    def test_identical_policies_have_exact_zero_difference(self):
        rows = [labeled_row(i, 0.25 + i / 100, i % 2) for i in range(20)]
        result = anytime_ips_lower_bound(rows, "greedy", "greedy")
        self.assertEqual(result["lower_bound"], 0)
        self.assertEqual(result["radius"], 0)

    def test_vendor_categorical_logging_uses_both_target_probabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            decisions = str(Path(folder) / "decisions.jsonl")
            labels = str(Path(folder) / "labels.jsonl")
            logged = log_instrumented_decision(
                decisions, task_signature="design|test",
                candidate_routes=[["short"], ["long", "route"]],
                selected_route=["short"],
                selection_probabilities=[0.8, 0.2],
                predicted_scores=[0.2, 0.9],
            )
            label = add_label(decisions, labels, logged["decision_id"], 1.0,
                              "synthetic-test", "fixture-v1", "synthetic")
            row = {"decision": load_decisions(decisions)[0], "label": label}
            result = anytime_ips_lower_bound([row], "greedy", "shortest")
            self.assertAlmostEqual(result["ips_difference_unclipped"], 1 / 0.8)
            self.assertAlmostEqual(result["range_square_sum"], (1 / 0.8 + 1 / 0.2) ** 2)
            report = evaluate_experiment(decisions, labels, replicates=100, min_labeled=1)
            self.assertEqual(report["anytime_ips_difference"], result)
            self.assertFalse(report["review_gate"]["candidate_for_human_review"])

    def test_forged_categorical_propensity_is_rejected(self):
        row = labeled_row(0, 0.5, 1)
        row["decision"].update(logging_policy="categorical",
                               candidate_probabilities=[0.9, 0.1])
        with self.assertRaises(ValueError):
            anytime_ips_lower_bound([row], "greedy", "shortest")

    def test_strong_synthetic_effect_eventually_has_positive_lower_bound(self):
        # This exercises a nonvacuous bound; synthetic fixtures are never
        # evidence about the real scaffold or a rollout decision.
        rows = [labeled_row(i, 1.0, i % 2) for i in range(2000)]
        result = anytime_ips_lower_bound(rows, "greedy", "shortest")
        self.assertEqual(result["ips_difference_unclipped"], 1.0)
        self.assertGreater(result["lower_bound"], 0.5)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            anytime_ips_lower_bound([], "greedy", "shortest")
        with self.assertRaises(ValueError):
            anytime_ips_lower_bound([labeled_row(0, 0.5, 1)], "greedy", "shortest", alpha=0)
        row = labeled_row(0, 0.5, 1)
        row["label"]["reward"] = float("nan")
        with self.assertRaises(ValueError):
            anytime_ips_lower_bound([row], "greedy", "shortest")

    def test_missing_label_disables_bound_and_review_screen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "demo"
            demo(argparse.Namespace(out_dir=str(root), n=30, seed=13))
            decisions = root / "decisions.synthetic.jsonl"
            labels = root / "labels.synthetic.jsonl"
            labels.write_text(labels.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
            report = evaluate_experiment(str(decisions), str(labels), replicates=100, min_labeled=1)
            self.assertIsNone(report["anytime_ips_difference"]["lower_bound"])
            self.assertFalse(report["review_gate"]["candidate_for_human_review"])
            self.assertIn("Time-uniform lower bound", " ".join(report["review_gate"]["reasons"]))
            self.assertIn("Time-uniform 95% lower bound", render_dashboard(report))
            self.assertIn("every logged decision", render_dashboard(report))

    def test_complete_demo_reports_bound_without_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            demo(argparse.Namespace(out_dir=str(Path(folder) / "demo"), n=50, seed=17))
            report = json.loads((Path(folder) / "demo" / "report.json").read_text(encoding="utf-8"))
            bound = report["anytime_ips_difference"]
            self.assertEqual(bound["n"], 50)
            self.assertEqual(bound["method"], "conditional_hoeffding_ville_grid_v1")
            self.assertTrue(-1 <= bound["lower_bound"] <= 1)
            self.assertEqual(report["label_source_counts"]["synthetic"], 50)
            self.assertFalse(report["review_gate"]["automatic_rollout_permitted"])


if __name__ == "__main__":
    unittest.main()

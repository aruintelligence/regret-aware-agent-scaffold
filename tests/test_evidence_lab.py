"""Contract, integration, and safety tests for the Decision Evidence Lab."""
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

from evidence_dashboard import render_dashboard
from evidence_lab import add_label, evaluate_experiment, validate_label
from experiment_cli import demo
from regret_aware_scaffold import MasterAgentRuntime
from route_experiments import (
    ExperimentRuntime, action_id, choose_policy_action,
    load_decisions, validate_decision,
)


class ExperimentTests(unittest.TestCase):
    def test_stable_route_identity_is_order_sensitive(self):
        self.assertEqual(action_id(["a", "b"]), action_id(["a", "b"]))
        self.assertNotEqual(action_id(["a", "b"]), action_id(["b", "a"]))

    def test_real_runtime_logs_randomized_route_and_stops_after_one_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "decisions.jsonl")
            runtime = ExperimentRuntime(path, epsilon=1.0, seed=12)
            state = runtime.run("Design a Python agent scaffold with route memory")
            records = load_decisions(path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(state.route.loop_count, 0)
            self.assertEqual(len(state.route.loop_paths), 1)
            self.assertEqual(record["action"], action_id(state.route.planned_path))
            self.assertAlmostEqual(record["propensity"], 1 / len(record["candidates"]))
            self.assertEqual(state.agent_outputs["planner_node"]["experiment"]["decision_id"],
                             record["decision_id"])
            self.assertNotIn("user_input", record)
            self.assertNotIn("reward", record)

    def test_original_runtime_remains_opt_in(self):
        state = MasterAgentRuntime().run("Research evidence for agent route quality")
        self.assertNotIn("experiment", state.agent_outputs["planner_node"])

    def test_propensity_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "decisions.jsonl")
            ExperimentRuntime(path, epsilon=0.5, seed=3).run("Design a Python scaffold")
            record = load_decisions(path)[0]
            record["propensity"] = 0.99
            with self.assertRaises(ValueError):
                validate_decision(record)

    def test_label_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            decisions = str(Path(folder) / "decisions.jsonl")
            labels = str(Path(folder) / "labels.jsonl")
            ExperimentRuntime(decisions, epsilon=0.5, seed=3).run("Design a Python scaffold")
            decision_id = load_decisions(decisions)[0]["decision_id"]
            row = add_label(decisions, labels, decision_id, 0.7, "grader-a", "rubric-1", "external")
            self.assertEqual(row["source_kind"], "external")
            with self.assertRaises(ValueError):
                add_label(decisions, labels, decision_id, 0.8, "grader-a", "rubric-1", "external")
            with self.assertRaises(ValueError):
                add_label(decisions, labels, "missing", 0.8, "grader-a", "rubric-1", "external")

    def test_invalid_rewards_rejected(self):
        base = {"schema_version": 1, "decision_id": "d", "grader": "g",
                "rubric_id": "r", "source_kind": "external"}
        for reward in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(reward=reward):
                with self.assertRaises(ValueError):
                    validate_label({**base, "reward": reward})

    def test_policy_uses_only_predecision_candidates(self):
        candidates = [
            {"route": ["a", "b", "c"], "action": action_id(["a", "b", "c"]), "predicted_score": .9},
            {"route": ["a"], "action": action_id(["a"]), "predicted_score": .4},
        ]
        row = {"candidates": candidates}
        self.assertEqual(choose_policy_action(row, "greedy"), candidates[0]["action"])
        self.assertEqual(choose_policy_action(row, "shortest"), candidates[1]["action"])
        self.assertEqual(choose_policy_action(row, "evidence_first"), candidates[0]["action"])

    def test_synthetic_demo_is_reproducible_and_never_promotable(self):
        with tempfile.TemporaryDirectory() as folder:
            demo(argparse.Namespace(out_dir=str(Path(folder) / "a"), n=120, seed=17))
            demo(argparse.Namespace(out_dir=str(Path(folder) / "b"), n=120, seed=17))
            a = json.loads((Path(folder) / "a" / "report.json").read_text())
            b = json.loads((Path(folder) / "b" / "report.json").read_text())
            self.assertEqual(a, b)
            self.assertEqual(a["n_logged"], 120)
            self.assertEqual(a["label_source_counts"]["synthetic"], 120)
            self.assertFalse(a["review_gate"]["candidate_for_human_review"])
            self.assertFalse(a["review_gate"]["automatic_rollout_permitted"])
            html = (Path(folder) / "a" / "index.html").read_text()
            self.assertIn("SYNTHETIC DEMONSTRATION", html)
            self.assertIn("Automatic rollout: prohibited", html)

    def test_incomplete_labeling_blocks_review(self):
        with tempfile.TemporaryDirectory() as folder:
            demo(argparse.Namespace(out_dir=str(Path(folder) / "demo"), n=30, seed=1))
            labels = Path(folder) / "demo" / "labels.synthetic.jsonl"
            labels.write_text(labels.read_text().splitlines()[0] + "\n")
            report = evaluate_experiment(
                str(Path(folder) / "demo" / "decisions.synthetic.jsonl"),
                str(labels), replicates=100, min_labeled=1,
            )
            self.assertEqual(report["n_unlabeled"], 29)
            self.assertTrue(any("Incomplete labels" in r
                                for r in report["review_gate"]["reasons"]))

    def test_dashboard_escapes_policy_names_from_report(self):
        with tempfile.TemporaryDirectory() as folder:
            demo(argparse.Namespace(out_dir=str(Path(folder) / "demo"), n=20, seed=2))
            report = json.loads((Path(folder) / "demo" / "report.json").read_text())
            report["baseline_policy"] = "<script>alert(1)</script>"
            html = render_dashboard(report)
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()

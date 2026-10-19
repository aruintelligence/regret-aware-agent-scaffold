"""Prospective-plan consistency checks across collection, grading and audit."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evidence_audit import build_manifest, save_manifest, verify_manifest
from evidence_lab import add_label, evaluate_experiment
from experiment_cli import build_parser
from experiment_protocol import (
    load_protocol, make_protocol, protocol_digest, save_protocol, validate_protocol,
)
from external_adapter import log_instrumented_decision
from route_experiments import ExperimentRuntime, load_decisions


def protocol_fixture(folder):
    path = Path(folder) / "protocol.json"
    plan = make_protocol("pilot-001", "greedy", "shortest", "rubric-v1",
                         replicates=100, min_labeled=1, min_ess=1)
    save_protocol(plan, str(path))
    return path, plan


class ProtocolTests(unittest.TestCase):
    def test_invalid_analysis_plan_cannot_be_locked(self):
        with self.assertRaises(ValueError):
            make_protocol("study", "greedy", "greedy", "rubric")
        with self.assertRaises(ValueError):
            make_protocol("study", "greedy", "shortest", "rubric", min_labeled=True)
        with self.assertRaises(ValueError):
            make_protocol("study", "greedy", "shortest", "rubric", max_weight=float("nan"))
        plan = make_protocol("study", "greedy", "shortest", "rubric")
        with self.assertRaises(ValueError):
            validate_protocol({**plan, "schema_version": True})
        with self.assertRaises(ValueError):
            validate_protocol({**plan, "unrecognized_gate": True})

    def test_canonical_digest_and_non_overwriting_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = protocol_fixture(folder)
            expected = protocol_digest(plan)
            self.assertEqual(protocol_digest(load_protocol(str(path))), expected)
            path.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
            self.assertEqual(protocol_digest(load_protocol(str(path))), expected)
            with self.assertRaises(FileExistsError):
                save_protocol(plan, str(path))
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(str(path))

    def test_builtin_decisions_rubric_and_settings_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = protocol_fixture(folder)
            decisions, labels = str(Path(folder) / "decisions.jsonl"), str(Path(folder) / "labels.jsonl")
            ExperimentRuntime(decisions, epsilon=0.5, seed=3,
                              protocol_path=str(path)).run("Design a Python scaffold")
            rows = load_decisions(decisions)
            self.assertEqual(rows[0]["protocol_sha256"], protocol_digest(plan))
            add_label(decisions, labels, rows[0]["decision_id"], 0.8,
                      "grader-a", "rubric-v1", "external")
            kwargs = dict(replicates=100, min_labeled=1, min_ess=1, protocol_path=str(path))
            report = evaluate_experiment(decisions, labels, **kwargs)
            self.assertEqual(report["protocol_binding"]["sha256"], protocol_digest(plan))
            self.assertIn("decision_bound", report["protocol_binding"]["status"])
            with self.assertRaisesRegex(ValueError, "analysis differs"):
                evaluate_experiment(decisions, labels, candidate_policy="evidence_first", **kwargs)
            with self.assertRaisesRegex(ValueError, "analysis differs"):
                evaluate_experiment(decisions, labels, min_uplift=0, **kwargs)
            with self.assertRaisesRegex(ValueError, "rubric"):
                Path(labels).write_text(Path(labels).read_text().replace("rubric-v1", "rubric-v2"))
                evaluate_experiment(decisions, labels, **kwargs)

    def test_changed_protocol_and_unbound_log_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = protocol_fixture(folder)
            decisions, labels = str(Path(folder) / "decisions.jsonl"), str(Path(folder) / "labels.jsonl")
            ExperimentRuntime(decisions, protocol_path=str(path)).run("Research evidence for routing")
            add_label(decisions, labels, load_decisions(decisions)[0]["decision_id"],
                      0.5, "g", "rubric-v1", "external")
            altered = {**plan, "study_id": "pilot-002"}
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or mismatches"):
                evaluate_experiment(decisions, labels, replicates=100,
                                    min_labeled=1, min_ess=1, protocol_path=str(path))
            with self.assertRaisesRegex(ValueError, "different experiment protocol"):
                ExperimentRuntime(decisions, protocol_path=str(path))
            self.assertEqual(len(load_decisions(decisions)), 1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_manifest(decisions, labels, protocol_path=str(path))
            exploratory = evaluate_experiment(decisions, labels, replicates=100,
                                               min_labeled=1, min_ess=1)
            self.assertEqual(exploratory["protocol_binding"]["status"], "unbound_exploratory")
            self.assertFalse(exploratory["review_gate"]["candidate_for_human_review"])

    def test_external_adapter_binds_protocol_and_audit_includes_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path, plan = protocol_fixture(folder)
            decisions, labels = str(Path(folder) / "decisions.jsonl"), str(Path(folder) / "labels.jsonl")
            logged = log_instrumented_decision(
                decisions, task_signature="pilot", candidate_routes=[["long", "route"], ["short"]],
                selected_route=["short"], selection_probabilities=[0.5, 0.5],
                predicted_scores=[0.9, 0.2], protocol_path=str(path),
            )
            self.assertEqual(load_decisions(decisions)[0]["protocol_sha256"], protocol_digest(plan))
            add_label(decisions, labels, logged["decision_id"], 0.6, "g", "rubric-v1", "synthetic")
            manifest = Path(folder) / "manifest.json"
            save_manifest(build_manifest(decisions, labels, protocol_path=str(path)), str(manifest))
            self.assertTrue(verify_manifest(str(manifest), decisions, labels,
                                            protocol_path=str(path))["verified"])
            self.assertFalse(verify_manifest(str(manifest), decisions, labels)["verified"])
            path.write_text(json.dumps({**plan, "study_id": "altered"}), encoding="utf-8")
            self.assertFalse(verify_manifest(str(manifest), decisions, labels,
                                             protocol_path=str(path))["verified"])

    def test_external_adapter_rejects_mixed_bound_and_unbound_log(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = protocol_fixture(folder)
            decisions = str(Path(folder) / "decisions.jsonl")
            kwargs = dict(task_signature="pilot", candidate_routes=[["a"], ["b"]],
                          selected_route=["a"], selection_probabilities=[0.5, 0.5])
            log_instrumented_decision(decisions, protocol_path=str(path), **kwargs)
            with self.assertRaisesRegex(ValueError, "different experiment protocol"):
                log_instrumented_decision(decisions, **kwargs)
            self.assertEqual(len(load_decisions(decisions)), 1)

    def test_cli_lock_collect_label_evaluate_and_mismatched_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, prompts = root / "protocol.json", root / "prompts.txt"
            decisions, labels, report = root / "decisions.jsonl", root / "labels.jsonl", root / "report.json"
            prompts.write_text("Design a Python scaffold\n", encoding="utf-8")
            parser = build_parser()

            def invoke(*arguments):
                parsed = parser.parse_args(arguments)
                with contextlib.redirect_stdout(io.StringIO()):
                    parsed.handler(parsed)

            invoke("lock-protocol", "--protocol", str(protocol), "--study-id", "pilot-001",
                   "--rubric-id", "rubric-v1", "--replicates", "100",
                   "--min-labeled", "1", "--min-ess", "1")
            invoke("collect", "--protocol", str(protocol), "--prompts", str(prompts),
                   "--decisions", str(decisions))
            decision_id = load_decisions(str(decisions))[0]["decision_id"]
            invoke("label", "--protocol", str(protocol), "--decisions", str(decisions),
                   "--labels", str(labels), "--decision-id", decision_id,
                   "--reward", "0.7", "--grader", "g", "--rubric-id", "rubric-v1",
                   "--source-kind", "external")
            invoke("evaluate", "--protocol", str(protocol), "--decisions", str(decisions),
                   "--labels", str(labels), "--report-json", str(report))
            self.assertEqual(json.loads(report.read_text())["protocol_binding"]["status"],
                             "decision_bound_self_attested")
            with self.assertRaisesRegex(ValueError, "differs"):
                invoke("evaluate", "--protocol", str(protocol), "--decisions", str(decisions),
                       "--labels", str(labels), "--candidate-policy", "evidence_first")
            with self.assertRaisesRegex(ValueError, "rubric"):
                invoke("label", "--protocol", str(protocol), "--decisions", str(decisions),
                       "--labels", str(labels), "--decision-id", decision_id,
                       "--reward", "0.7", "--grader", "g", "--rubric-id", "wrong",
                       "--source-kind", "external")


if __name__ == "__main__":
    unittest.main()

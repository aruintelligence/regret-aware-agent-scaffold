"""Complete outcome accounting, independent anchor, replay and gate tests."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy
from watch_outcome_ledger import (
    create_issued_checkpoint, gate_with_complete_ledger, reconcile_issued_grades,
    verify_complete_report, verify_issued_checkpoint,
)

INGRESS_KEY = b"research-ingress-test-secret-32-bytes"
WATCH_KEY = b"independent-watch-test-secret-32-bytes"
POLICY_KEY = b"independent-static-risk-key-32-bytes"


def fixture(n=40):
    issued = [
        {"sequence": i + 1, "request_id": f"request-{i + 1}",
         "group": "general", "model_revision": "fixed-qa-revision",
         "rubric_id": "rubric-v1"}
        for i in range(n)
    ]
    grades = [
        {"request_id": row["request_id"], "error": False,
         "rubric_id": "rubric-v1", "grader_id": "reviewer-1"}
        for row in issued
    ]
    checkpoint = create_issued_checkpoint(
        issued, cohort_id="pilot-closed-001", groups=["general"],
        model_revision="fixed-qa-revision", rubric_id="rubric-v1",
        ingress_key=INGRESS_KEY,
    )
    return issued, grades, checkpoint


def reconcile(issued, grades, checkpoint, **options):
    return reconcile_issued_grades(
        issued, grades, checkpoint=checkpoint,
        ingress_key=options.get("ingress_key", INGRESS_KEY),
        watch_key=options.get("watch_key", WATCH_KEY),
    )


def policy():
    rows = [
        {"sample_id": str(i), "score": 0.9 if i < 150 else 0.1,
         "correct": i < 150, "group": "general"}
        for i in range(200)
    ]
    return sign_risk_policy(fit_selective_risk_policy(
        rows, thresholds=[0.05, 0.5, 0.95], groups=["general"],
        target_error=0.1, alpha=0.05, min_group_support=30,
        model_revision="fixed-qa-revision", rubric_id="rubric-v1",
        calibration_set_id="frozen-calibration",
    ), key=POLICY_KEY)


class WatchOutcomeLedgerTests(unittest.TestCase):
    def test_cli_closed_cohort_pending_then_complete_and_exact_verify(self):
        issued, grades, _ = fixture(5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            issue_path = root / "issued.jsonl"
            grades_path = root / "grades.jsonl"
            checkpoint_path = root / "checkpoint.json"
            report_path = root / "watch.json"
            issue_path.write_text("".join(json.dumps(row) + "\n" for row in issued))
            grades_path.write_text("".join(json.dumps(row) + "\n" for row in grades[:-1]))
            env = {**os.environ, "ARU_ISSUED_SIGNING_KEY": INGRESS_KEY.decode(),
                   "ARU_WATCH_SIGNING_KEY": WATCH_KEY.decode()}
            cli = str(Path(__file__).resolve().parents[1] / "watch_outcome_ledger_cli.py")

            def run(*args):
                return subprocess.run(
                    [sys.executable, cli, *args], env=env, cwd=root,
                    capture_output=True, text=True, check=False,
                )

            anchored = run(
                "anchor", "--issued-jsonl", str(issue_path), "--cohort-id", "pilot-closed-001",
                "--groups", "general", "--model-revision", "fixed-qa-revision",
                "--rubric-id", "rubric-v1", "--out", str(checkpoint_path),
            )
            self.assertEqual(anchored.returncode, 0, anchored.stderr)
            checkpoint = json.loads(checkpoint_path.read_text())
            cmd = (
                "reconcile", "--issued-jsonl", str(issue_path),
                "--grades-jsonl", str(grades_path), "--checkpoint", str(checkpoint_path),
                "--out", str(report_path),
            )
            pending = run(*cmd)
            self.assertEqual(pending.returncode, 3, pending.stderr)
            self.assertEqual(json.loads(pending.stdout)["missing_count"], 1)
            self.assertFalse(report_path.exists())
            grades_path.write_text("".join(json.dumps(row) + "\n" for row in grades))
            complete = run(*cmd)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            self.assertTrue(report_path.exists())
            verified = run(
                "verify", "--issued-jsonl", str(issue_path),
                "--grades-jsonl", str(grades_path), "--checkpoint", str(checkpoint_path),
                "--report", str(report_path), "--expected-checkpoint-hmac",
                checkpoint["ingress_hmac_sha256"],
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            rejected = run(
                "verify", "--issued-jsonl", str(issue_path),
                "--grades-jsonl", str(grades_path), "--checkpoint", str(checkpoint_path),
                "--report", str(report_path), "--expected-checkpoint-hmac", "0" * 64,
            )
            self.assertEqual(rejected.returncode, 1, rejected.stderr)
            grades_path.write_text("".join(json.dumps({**row, "error": True}) + "\n" for row in grades))
            alarm_path = root / "alarmed.json"
            alarm = run(*(cmd[:-1] + (str(alarm_path),)))
            self.assertEqual(alarm.returncode, 10, alarm.stderr)
            self.assertTrue(json.loads(alarm_path.read_text())["degradation_alarm"])

    def test_complete_permuted_grade_arrival_uses_issued_order(self):
        issued, grades, checkpoint = fixture()
        result = reconcile(issued, list(reversed(grades)), checkpoint)
        self.assertEqual(result["status"], "complete")
        report = result["report"]
        self.assertEqual(report["outcome_count"], 40)
        self.assertEqual(report["graded_count"], report["issued_count"])
        self.assertTrue(verify_issued_checkpoint(checkpoint, ingress_key=INGRESS_KEY))
        self.assertTrue(verify_complete_report(
            report, issued, grades, checkpoint=checkpoint,
            ingress_key=INGRESS_KEY, watch_key=WATCH_KEY,
        ))
        self.assertEqual(report, reconcile(issued, grades, checkpoint)["report"])

    def test_delayed_and_selectively_missing_grades_never_create_a_watch(self):
        issued, grades, checkpoint = fixture()
        for missing in (grades[-1:], grades[10:15]):
            with self.subTest(missing=missing):
                result = reconcile(
                    issued, [g for g in grades if g not in missing], checkpoint,
                )
                self.assertEqual(result["status"], "pending_grades")
                self.assertEqual(result["missing_count"], len(missing))
                self.assertNotIn("report", result)
                self.assertFalse(result["automatic_rollout_permitted"])
        self.assertEqual(reconcile(issued, [], checkpoint)["missing_count"], 40)

    def test_duplicate_unknown_malformed_and_rubric_mismatch_fail(self):
        issued, grades, checkpoint = fixture()
        cases = (
            grades + grades[:1],
            grades + [{**grades[0], "request_id": "unknown"}],
            [{**grades[0], "error": 1}] + grades[1:],
            [{**grades[0], "rubric_id": "wrong"}] + grades[1:],
            [{**grades[0], "prompt": "private data"}] + grades[1:],
            [{**grades[0], "grader_id": ""}] + grades[1:],
        )
        for rows in cases:
            with self.subTest(rows=rows[:1]):
                with self.assertRaises(ValueError):
                    reconcile(issued, rows, checkpoint)

    def test_tampered_issued_manifest_and_checkpoint_fail(self):
        issued, grades, checkpoint = fixture()
        variants = (
            [{**issued[0], "group": "technical"}] + issued[1:],
            [{**issued[0], "request_id": "other"}] + issued[1:],
            [issued[1], issued[0]] + issued[2:],
            issued[:-1],
            issued + [{**issued[-1], "sequence": 41}],
        )
        for rows in variants:
            with self.subTest(rows=rows[:1]):
                with self.assertRaises(ValueError):
                    reconcile(rows, grades, checkpoint)
        changed = {**checkpoint, "issued_count": checkpoint["issued_count"] - 1}
        self.assertFalse(verify_issued_checkpoint(changed, ingress_key=INGRESS_KEY))
        with self.assertRaises(ValueError):
            reconcile(issued, grades, changed)
        with self.assertRaises(ValueError):
            reconcile(issued, grades, checkpoint, watch_key=INGRESS_KEY)

    def test_exact_recompute_detects_changed_label_and_signed_report_tampering(self):
        issued, grades, checkpoint = fixture()
        report = reconcile(issued, grades, checkpoint)["report"]
        changed_grade = [{**grades[0], "error": True}] + grades[1:]
        self.assertFalse(verify_complete_report(
            report, issued, changed_grade, checkpoint=checkpoint,
            ingress_key=INGRESS_KEY, watch_key=WATCH_KEY,
        ))
        changed_report = copy.deepcopy(report)
        changed_report["complete_outcomes_verified"] = False
        self.assertFalse(verify_complete_report(
            changed_report, issued, grades, checkpoint=checkpoint,
            ingress_key=INGRESS_KEY, watch_key=WATCH_KEY,
        ))

    def test_gate_requires_current_independent_anchor_and_digests(self):
        issued, grades, checkpoint = fixture()
        report = reconcile(issued, grades, checkpoint)["report"]
        source = "The telescope was in Oregon."
        kwargs = {
            "policy": policy(), "policy_key": POLICY_KEY,
            "watch_report": report, "watch_key": WATCH_KEY,
            "ingress_checkpoint": checkpoint, "ingress_key": INGRESS_KEY,
            "expected_current_checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
            "expected_outcomes_sha256": report["ordered_outcomes_sha256"],
            "score": 0.9, "group": "general",
            "source_proof": ground_answer("Oregon", source, model_revision="fixed-qa-revision"),
            "source_text": source, "model_revision": "fixed-qa-revision",
            "rubric_id": "rubric-v1",
        }
        accepted = gate_with_complete_ledger(**kwargs)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["answer"], "Oregon")
        self.assertFalse(accepted["automatic_rollout_permitted"])
        for update, reason in (
            ({"expected_current_checkpoint_hmac_sha256": "0" * 64}, "missing_or_stale_ingress_checkpoint"),
            ({"expected_current_checkpoint_hmac_sha256": None}, "missing_or_stale_ingress_checkpoint"),
            ({"ingress_key": b"wrong-ingress-key-with-32-byte-minimum"}, "missing_or_stale_ingress_checkpoint"),
            ({"watch_key": INGRESS_KEY}, "monitor_and_ingress_keys_not_independent"),
            ({"expected_outcomes_sha256": "0" * 64}, "missing_or_stale_watch_checkpoint"),
            ({"expected_outcomes_sha256": None}, "missing_or_stale_watch_checkpoint"),
            ({"watch_report": None}, "missing_or_invalid_signed_watch"),
            ({"min_group_outcomes": 41}, "insufficient_recent_group_outcomes"),
        ):
            with self.subTest(reason=reason):
                denied = gate_with_complete_ledger(**{**kwargs, **update})
                self.assertFalse(denied["accepted"])
                self.assertEqual(denied["reason"], reason)
        old_checkpoint = create_issued_checkpoint(
            issued[:20], cohort_id="pilot-old", groups=["general"],
            model_revision="fixed-qa-revision", rubric_id="rubric-v1",
            ingress_key=INGRESS_KEY,
        )
        stale = gate_with_complete_ledger(**{
            **kwargs, "ingress_checkpoint": old_checkpoint,
        })
        self.assertEqual(stale["reason"], "missing_or_stale_ingress_checkpoint")

    def test_gate_rejects_signed_but_unbound_or_scope_mismatched_reports(self):
        from anytime_risk_watch import sign_watch_report, watch_risk_stream
        issued, grades, checkpoint = fixture()
        report = reconcile(issued, grades, checkpoint)["report"]
        source = "Oregon"
        kwargs = {
            "policy": policy(), "policy_key": POLICY_KEY,
            "watch_report": report, "watch_key": WATCH_KEY,
            "ingress_checkpoint": checkpoint, "ingress_key": INGRESS_KEY,
            "expected_current_checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
            "expected_outcomes_sha256": report["ordered_outcomes_sha256"],
            "score": 0.9, "group": "general",
            "source_proof": ground_answer("Oregon", source, model_revision="fixed-qa-revision"),
            "source_text": source, "model_revision": "fixed-qa-revision", "rubric_id": "rubric-v1",
        }
        unsigned = {k: v for k, v in report.items() if k != "watch_hmac_sha256"}
        for edited, reason in (
            ({**unsigned, "issued_count": 39, "graded_count": 39, "outcome_count": 39}, "incomplete_or_unbound_outcome_ledger"),
            ({**unsigned, "complete_outcomes_verified": False}, "incomplete_or_unbound_outcome_ledger"),
            ({**unsigned, "bet_fractions": [0.03]}, "ledger_watch_scope_mismatch"),
        ):
            with self.subTest(reason=reason):
                modified = sign_watch_report(edited, key=WATCH_KEY)
                result = gate_with_complete_ledger(**{**kwargs, "watch_report": modified})
                self.assertEqual(result["reason"], reason)
        unbound = sign_watch_report(watch_risk_stream(
            [{"event_id": r["request_id"], "group": "general", "error": False}
             for r in issued], groups=["general"], model_revision="fixed-qa-revision",
            rubric_id="rubric-v1",
        ), key=WATCH_KEY)
        denied = gate_with_complete_ledger(**{**kwargs, "watch_report": unbound})
        self.assertEqual(denied["reason"], "incomplete_or_unbound_outcome_ledger")

    def test_archived_real_model_trace_reconstruction_is_only_a_replay(self):
        path = Path(__file__).resolve().parents[1] / "evidence/calibrated_change_20261019/intervention_outcomes.jsonl"
        if not path.exists():
            self.skipTest("archived trace unavailable")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 120)
        issued = [
            {"sequence": i, "request_id": r["event_id"], "group": r["group"],
             "model_revision": "archived-replay", "rubric_id": "archived-label-replay"}
            for i, r in enumerate(rows, 1)
        ]
        grades = [
            {"request_id": r["event_id"], "error": r["error"],
             "rubric_id": "archived-label-replay", "grader_id": "archived-replay"}
            for r in rows
        ]
        anchor = create_issued_checkpoint(
            issued, cohort_id="retrospective-demo", groups=["short_source", "long_source"],
            model_revision="archived-replay", rubric_id="archived-label-replay",
            ingress_key=INGRESS_KEY,
        )
        report = reconcile(issued, grades, anchor)["report"]
        self.assertEqual(report["slices"]["__all__"]["errors"], 49)
        self.assertEqual(report["issued_count"], 120)
        self.assertTrue(report["degradation_alarm"])
        # Exhaust all 120 ways a single difficult request could be omitted.
        for index in range(120):
            with self.subTest(omitted=index):
                partial = grades[:index] + grades[index + 1:]
                result = reconcile(issued, partial, anchor)
                self.assertEqual(result["status"], "pending_grades")
                self.assertEqual(result["missing_count"], 1)
                self.assertNotIn("report", result)


if __name__ == "__main__":
    unittest.main()

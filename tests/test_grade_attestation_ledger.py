"""Independent grade claims, one-to-one coverage and monitor forgery tests."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from grade_attestation_ledger import (
    attest_grade, audit_grade_cohort, create_grade_checkpoint,
    gate_with_authenticated_grades, verify_grade_attestation,
    verify_grade_checkpoint,
)
from grounding_contract import ground_answer
from receipt_cohort_ledger import gate_with_receipt_ledger
from tests.test_receipt_cohort_ledger import RECEIPT_KEY, receipt_fixture
from tests.test_watch_outcome_ledger import INGRESS_KEY, POLICY_KEY, WATCH_KEY, policy
from watch_outcome_ledger import reconcile_issued_grades

GRADER_KEY = b"distinct-independent-grader-secret-32-byte-key"


def setup(n=40):
    issued, grades, ingress, receipts, receipt_checkpoint = receipt_fixture(n)
    # Grade arrival order need not match issuance order.
    attested = [attest_grade(grade, grade_sequence=index, cohort_id=ingress["cohort_id"],
                             model_revision=ingress["model_revision"], grader_key=GRADER_KEY)
                for index, grade in enumerate(reversed(grades), 1)]
    checkpoint = create_grade_checkpoint(
        attested, cohort_id=ingress["cohort_id"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
        grader_key=GRADER_KEY,
    )
    return issued, grades, ingress, receipts, receipt_checkpoint, attested, checkpoint


def audit(issued, ingress, attested, checkpoint, **updates):
    kwargs = {
        "grade_checkpoint": checkpoint, "ingress_checkpoint": ingress,
        "grader_key": GRADER_KEY, "ingress_key": INGRESS_KEY,
        "expected_current_grade_hmac_sha256": checkpoint["grader_ledger_hmac_sha256"],
        "expected_current_ingress_hmac_sha256": ingress["ingress_hmac_sha256"],
    }
    return audit_grade_cohort(attested, issued, **{**kwargs, **updates})


class GradeAttestationLedgerTests(unittest.TestCase):
    def test_arrival_order_grades_bind_exactly_to_issued_ids(self):
        issued, _, ingress, _, _, attested, checkpoint = setup()
        self.assertTrue(verify_grade_attestation(attested[0], grader_key=GRADER_KEY))
        self.assertTrue(verify_grade_checkpoint(checkpoint, grader_key=GRADER_KEY))
        result = audit(issued, ingress, attested, checkpoint)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["graded_count"], 40)
        self.assertFalse(result["automatic_rollout_permitted"])
        for change in (
            {"error": True}, {"request_id": "other"},
            {"grade_sequence": 5}, {"nonce": "0" * 32},
            {"rubric_id": "wrong"},
        ):
            with self.subTest(change=change):
                self.assertFalse(verify_grade_attestation(
                    {**attested[0], **change}, grader_key=GRADER_KEY))
        self.assertFalse(verify_grade_attestation(attested[0], grader_key=WATCH_KEY))
        self.assertFalse(verify_grade_checkpoint(
            {**checkpoint, "grade_count": 39}, grader_key=GRADER_KEY))
        with self.assertRaises(ValueError):
            attest_grade({"request_id": "r", "error": 1, "rubric_id": "v1",
                          "grader_id": "reviewer"}, grade_sequence=1,
                         cohort_id="c", model_revision="m", grader_key=GRADER_KEY)

    def test_missing_foreign_stale_duplicate_and_key_reuse_fail_closed(self):
        issued, grades, ingress, _, _, attested, checkpoint = setup(8)
        partial = attested[:-1]
        partial_checkpoint = create_grade_checkpoint(
            partial, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
            grader_key=GRADER_KEY,
        )
        self.assertEqual(audit(issued, ingress, partial, partial_checkpoint)["missing_count"], 1)
        foreign = [*attested, attest_grade(
            {**grades[0], "request_id": "unissued"}, grade_sequence=9,
            cohort_id=ingress["cohort_id"], model_revision=ingress["model_revision"],
            grader_key=GRADER_KEY)]
        foreign_checkpoint = create_grade_checkpoint(
            foreign, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
            grader_key=GRADER_KEY)
        self.assertEqual(audit(issued, ingress, foreign, foreign_checkpoint)["foreign_count"], 1)
        for changed in (
            {"expected_current_grade_hmac_sha256": "0" * 64},
            {"expected_current_ingress_hmac_sha256": "0" * 64},
            {"grader_key": INGRESS_KEY},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    audit(issued, ingress, attested, checkpoint, **changed)
        with self.assertRaises(ValueError):
            audit(issued, ingress, partial, checkpoint)
        with self.assertRaises(ValueError):
            create_grade_checkpoint(
                attested[:-1] + attested[:1], cohort_id=ingress["cohort_id"],
                model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
                grader_key=GRADER_KEY)

    def test_resigned_monitor_label_is_rejected_by_authenticated_grade_gate(self):
        issued, grades, ingress, receipts, receipt_checkpoint, attested, checkpoint = setup()
        source = "The telescope was in Oregon."
        static = policy()

        def kwargs(labels):
            watch = reconcile_issued_grades(
                issued, labels, checkpoint=ingress,
                ingress_key=INGRESS_KEY, watch_key=WATCH_KEY)["report"]
            return {
                "policy_key": POLICY_KEY, "watch_report": watch,
                "watch_key": WATCH_KEY, "ingress_checkpoint": ingress,
                "ingress_key": INGRESS_KEY,
                "expected_current_checkpoint_hmac_sha256": ingress["ingress_hmac_sha256"],
                "expected_outcomes_sha256": watch["ordered_outcomes_sha256"],
                "score": 0.9, "group": "general",
                "source_proof": ground_answer("Oregon", source,
                                              model_revision="fixed-qa-revision"),
                "source_text": source, "model_revision": "fixed-qa-revision",
                "rubric_id": "rubric-v1",
            }

        held = {
            "receipts": receipts, "issued_rows": issued,
            "receipt_checkpoint": receipt_checkpoint, "receipt_key": RECEIPT_KEY,
            "expected_current_receipt_hmac_sha256": receipt_checkpoint[
                "receipt_ledger_hmac_sha256"],
        }
        graded = {
            "attestations": attested, "grade_checkpoint": checkpoint,
            "grader_key": GRADER_KEY,
            "expected_current_grade_hmac_sha256": checkpoint["grader_ledger_hmac_sha256"],
        }
        self.assertTrue(gate_with_authenticated_grades(
            static, **graded, **held, **kwargs(grades))["accepted"])
        # The watch signer alone can sign a new internally consistent report
        # after altering any label, but cannot sign the grader's log.
        for index in range(40):
            with self.subTest(label=index):
                changed = [*grades]
                changed[index] = {**grades[index], "error": True}
                candidate = kwargs(changed)
                self.assertTrue(gate_with_receipt_ledger(
                    static, **held, **candidate)["accepted"])
                denial = gate_with_authenticated_grades(
                    static, **graded, **held, **candidate)
                self.assertEqual(denial["reason"],
                                 "signed_watch_disagrees_with_authenticated_grades")
                self.assertFalse(denial["automatic_rollout_permitted"])
        # More dangerous direction: hide the one real error. Test every
        # position because the watch processes the stream in issuance order.
        for index in range(40):
            with self.subTest(hidden_error=index):
                observed = [*grades]
                observed[index] = {**grades[index], "error": True}
                observed_attestations = [
                    attest_grade(grade, grade_sequence=sequence,
                                 cohort_id=ingress["cohort_id"],
                                 model_revision=ingress["model_revision"],
                                 grader_key=GRADER_KEY)
                    for sequence, grade in enumerate(reversed(observed), 1)
                ]
                observed_checkpoint = create_grade_checkpoint(
                    observed_attestations, cohort_id=ingress["cohort_id"],
                    model_revision=ingress["model_revision"],
                    rubric_id=ingress["rubric_id"], grader_key=GRADER_KEY,
                )
                observed_grader = {
                    **graded, "attestations": observed_attestations,
                    "grade_checkpoint": observed_checkpoint,
                    "expected_current_grade_hmac_sha256": observed_checkpoint[
                        "grader_ledger_hmac_sha256"],
                }
                self.assertTrue(gate_with_authenticated_grades(
                    static, **observed_grader, **held, **kwargs(observed))["accepted"])
                self.assertTrue(gate_with_receipt_ledger(
                    static, **held, **kwargs(grades))["accepted"])
                hidden = gate_with_authenticated_grades(
                    static, **observed_grader, **held, **kwargs(grades))
                self.assertEqual(hidden["reason"],
                                 "signed_watch_disagrees_with_authenticated_grades")
        stale = gate_with_authenticated_grades(
            static, **{**graded, "expected_current_grade_hmac_sha256": "0" * 64},
            **held, **kwargs(grades))
        self.assertEqual(stale["reason"], "invalid_or_stale_grader_ledger")
        shared = gate_with_authenticated_grades(
            static, **{**graded, "grader_key": WATCH_KEY},
            **held, **kwargs(grades))
        self.assertEqual(shared["reason"], "grader_key_not_independent")

    def test_cli_grade_anchor_reconcile_pending_stale_and_complete(self):
        issued, grades, ingress, _, _, attested, checkpoint = setup(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grade_json = root / "grade.json"
            single = root / "single-attestation.json"
            log = root / "grades.jsonl"
            issued_path = root / "issued.jsonl"
            ingress_path = root / "ingress.json"
            checkpoint_path = root / "grader-checkpoint.json"
            report_path = root / "signed-watch.json"
            grade_json.write_text(json.dumps(grades[0]))
            log.write_text("".join(json.dumps(row) + "\n" for row in attested))
            issued_path.write_text("".join(json.dumps(row) + "\n" for row in issued))
            ingress_path.write_text(json.dumps(ingress))
            env = {**os.environ, "ARU_GRADER_SIGNING_KEY": GRADER_KEY.decode(),
                   "ARU_ISSUED_SIGNING_KEY": INGRESS_KEY.decode(),
                   "ARU_WATCH_SIGNING_KEY": WATCH_KEY.decode()}
            cli = str(Path(__file__).resolve().parents[1] / "watch_outcome_ledger_cli.py")

            def run(*args):
                return subprocess.run([sys.executable, cli, *args], cwd=root,
                                      env=env, capture_output=True, text=True, check=False)

            one = run("grade", "--grade-json", str(grade_json), "--grade-sequence", "1",
                      "--cohort-id", ingress["cohort_id"], "--model-revision",
                      ingress["model_revision"], "--out", str(single))
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertTrue(verify_grade_attestation(json.loads(single.read_text()),
                                                      grader_key=GRADER_KEY))
            self.assertEqual(single.stat().st_mode & 0o777, 0o600)
            anchored = run("grade-anchor", "--attestations-jsonl", str(log),
                           "--cohort-id", ingress["cohort_id"],
                           "--model-revision", ingress["model_revision"],
                           "--rubric-id", ingress["rubric_id"], "--out", str(checkpoint_path))
            self.assertEqual(anchored.returncode, 0, anchored.stderr)
            self.assertEqual(json.loads(checkpoint_path.read_text()), checkpoint)

            def cmd(grade_hmac):
                return ("grade-reconcile", "--attestations-jsonl", str(log),
                        "--issued-jsonl", str(issued_path),
                        "--grade-checkpoint", str(checkpoint_path),
                        "--ingress-checkpoint", str(ingress_path),
                        "--expected-grade-hmac", grade_hmac,
                        "--expected-checkpoint-hmac", ingress["ingress_hmac_sha256"],
                        "--out", str(report_path))

            self.assertEqual(run(*cmd("0" * 64)).returncode, 4)
            complete = run(*cmd(checkpoint["grader_ledger_hmac_sha256"]))
            self.assertEqual(complete.returncode, 0, complete.stderr)
            self.assertTrue(report_path.exists())
            report_path.unlink()
            partial = attested[:-1]
            log.write_text("".join(json.dumps(row) + "\n" for row in partial))
            partial_cp = create_grade_checkpoint(
                partial, cohort_id=ingress["cohort_id"],
                model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
                grader_key=GRADER_KEY)
            checkpoint_path.write_text(json.dumps(partial_cp))
            pending = run(*cmd(partial_cp["grader_ledger_hmac_sha256"]))
            self.assertEqual(pending.returncode, 3, pending.stderr)
            self.assertEqual(json.loads(pending.stdout)["missing_count"], 1)
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()

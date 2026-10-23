"""Independent receipt/ingress reconciliation and release-gate falsification."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from client_request_receipt import issue_request_receipt
from grounding_contract import ground_answer
from receipt_cohort_ledger import (
    create_receipt_checkpoint, gate_with_receipt_ledger,
    reconcile_receipt_cohort, verify_receipt_checkpoint,
)
from tests.test_watch_outcome_ledger import (
    INGRESS_KEY, POLICY_KEY, WATCH_KEY, fixture, policy,
)
from watch_outcome_ledger import (
    create_issued_checkpoint, gate_with_complete_ledger, reconcile_issued_grades,
)

RECEIPT_KEY = b"distinct-receipt-cohort-test-secret-32-bytes"


def receipt_fixture(n=40):
    issued, grades, ingress = fixture(n)
    receipts = [issue_request_receipt(row, cohort_id=ingress["cohort_id"],
                                      receipt_key=RECEIPT_KEY) for row in issued]
    checkpoint = create_receipt_checkpoint(
        receipts, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
        receipt_key=RECEIPT_KEY,
    )
    return issued, grades, ingress, receipts, checkpoint


def reconcile(issued, receipts, ingress, checkpoint, **options):
    return reconcile_receipt_cohort(
        receipts, issued, ingress_checkpoint=ingress, receipt_checkpoint=checkpoint,
        ingress_key=options.get("ingress_key", INGRESS_KEY),
        receipt_key=options.get("receipt_key", RECEIPT_KEY),
        expected_current_ingress_hmac_sha256=options.get(
            "current_ingress", ingress["ingress_hmac_sha256"]),
        expected_current_receipt_hmac_sha256=options.get(
            "current_receipt", checkpoint["receipt_ledger_hmac_sha256"]),
    )


class ReceiptCohortLedgerTests(unittest.TestCase):
    def test_all_40_signed_manifest_rewrites_conflict_with_separate_receipt_log(self):
        issued, _, ingress, receipts, checkpoint = receipt_fixture()
        self.assertTrue(verify_receipt_checkpoint(checkpoint, receipt_key=RECEIPT_KEY))
        self.assertEqual(reconcile(issued, receipts, ingress, checkpoint)["status"], "complete")
        for index in range(40):
            with self.subTest(sequence=index + 1):
                rewritten = [*issued]
                rewritten[index] = {**issued[index], "request_id": f"hidden-{index}"}
                signed = create_issued_checkpoint(
                    rewritten, cohort_id=ingress["cohort_id"],
                    groups=ingress["groups"], model_revision=ingress["model_revision"],
                    rubric_id=ingress["rubric_id"], ingress_key=INGRESS_KEY,
                )
                conflict = reconcile(rewritten, receipts, signed, checkpoint)
                self.assertEqual(conflict["status"], "conflict")
                self.assertEqual(conflict["conflict_sequences"], [index + 1])
                self.assertFalse(conflict["automatic_rollout_permitted"])

    def test_missing_extra_duplicate_and_stale_records_fail_closed(self):
        issued, _, ingress, receipts, checkpoint = receipt_fixture(8)
        truncated = issued[:-1]
        truncated_anchor = create_issued_checkpoint(
            truncated, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
            model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
            ingress_key=INGRESS_KEY,
        )
        result = reconcile(truncated, receipts, truncated_anchor, checkpoint)
        self.assertEqual(result["conflict_sequences"], [8])
        extra = issued + [{**issued[-1], "sequence": 9, "request_id": "unreceipted"}]
        extra_anchor = create_issued_checkpoint(
            extra, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
            model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
            ingress_key=INGRESS_KEY,
        )
        self.assertEqual(reconcile(extra, receipts, extra_anchor, checkpoint)["status"], "conflict")
        for changes in (
            {"current_receipt": "0" * 64},
            {"current_ingress": "0" * 64},
            {"receipt_key": INGRESS_KEY},
            {"ingress_key": RECEIPT_KEY},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    reconcile(issued, receipts, ingress, checkpoint, **changes)
        altered = {**checkpoint, "receipt_count": 7}
        self.assertFalse(verify_receipt_checkpoint(altered, receipt_key=RECEIPT_KEY))
        with self.assertRaises(ValueError):
            reconcile(issued, receipts[:-1], ingress, checkpoint)
        with self.assertRaises(ValueError):
            create_receipt_checkpoint(
                receipts[:-1] + receipts[:1], cohort_id=ingress["cohort_id"],
                groups=ingress["groups"], model_revision=ingress["model_revision"],
                rubric_id=ingress["rubric_id"], receipt_key=RECEIPT_KEY,
            )
        with self.assertRaises(ValueError):
            create_receipt_checkpoint(
                [{**receipts[0], "request_id": "forged"}] + receipts[1:],
                cohort_id=ingress["cohort_id"], groups=ingress["groups"],
                model_revision=ingress["model_revision"],
                rubric_id=ingress["rubric_id"], receipt_key=RECEIPT_KEY,
            )

    def test_receipt_gate_blocks_validly_resigned_omission_that_old_gate_accepts(self):
        issued, grades, ingress, receipts, checkpoint = receipt_fixture()
        source = "The telescope was in Oregon."

        def options(rows, labels, signed_ingress):
            watch = reconcile_issued_grades(
                rows, labels, checkpoint=signed_ingress,
                ingress_key=INGRESS_KEY, watch_key=WATCH_KEY,
            )["report"]
            return {
                "policy_key": POLICY_KEY, "watch_report": watch,
                "watch_key": WATCH_KEY, "ingress_checkpoint": signed_ingress,
                "ingress_key": INGRESS_KEY,
                "expected_current_checkpoint_hmac_sha256": signed_ingress["ingress_hmac_sha256"],
                "expected_outcomes_sha256": watch["ordered_outcomes_sha256"],
                "score": 0.9, "group": "general",
                "source_proof": ground_answer("Oregon", source,
                                              model_revision="fixed-qa-revision"),
                "source_text": source, "model_revision": "fixed-qa-revision",
                "rubric_id": "rubric-v1",
            }

        static = policy()
        correct = options(issued, grades, ingress)
        held = {
            "receipts": receipts, "issued_rows": issued,
            "receipt_checkpoint": checkpoint, "receipt_key": RECEIPT_KEY,
            "expected_current_receipt_hmac_sha256": checkpoint["receipt_ledger_hmac_sha256"],
        }
        self.assertTrue(gate_with_receipt_ledger(static, **held, **correct)["accepted"])
        changed = [*issued]
        changed[17] = {**issued[17], "request_id": "replacement"}
        replacement_grades = [*grades]
        replacement_grades[17] = {**grades[17], "request_id": "replacement"}
        signed = create_issued_checkpoint(
            changed, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
            model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
            ingress_key=INGRESS_KEY,
        )
        rewritten = options(changed, replacement_grades, signed)
        self.assertTrue(gate_with_complete_ledger(static, **rewritten)["accepted"])
        blocked = gate_with_receipt_ledger(static, **{**held, "issued_rows": changed},
                                          **rewritten)
        self.assertFalse(blocked["accepted"])
        self.assertEqual(blocked["reason"], "receipt_ingress_conflict")
        self.assertEqual(blocked["conflict_sequences"], [18])
        self.assertFalse(blocked["automatic_rollout_permitted"])
        stale = gate_with_receipt_ledger(static, **{**held,
            "expected_current_receipt_hmac_sha256": "0" * 64}, **correct)
        self.assertEqual(stale["reason"], "invalid_or_stale_receipt_ledger")
        same_key = gate_with_receipt_ledger(static, **{**held, "receipt_key": WATCH_KEY},
                                            **correct)
        self.assertEqual(same_key["reason"], "receipt_and_watch_keys_not_independent")

    def test_cli_anchor_and_audit_statuses(self):
        issued, _, ingress, receipts, checkpoint = receipt_fixture(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_log = root / "receipts.jsonl"
            issued_log = root / "issued.jsonl"
            receipt_anchor = root / "receipt-anchor.json"
            ingress_anchor = root / "ingress-anchor.json"
            receipt_log.write_text("".join(json.dumps(row) + "\n" for row in receipts))
            issued_log.write_text("".join(json.dumps(row) + "\n" for row in issued))
            ingress_anchor.write_text(json.dumps(ingress))
            env = {**os.environ, "ARU_RECEIPT_SIGNING_KEY": RECEIPT_KEY.decode(),
                   "ARU_ISSUED_SIGNING_KEY": INGRESS_KEY.decode()}
            cli = str(Path(__file__).resolve().parents[1] / "watch_outcome_ledger_cli.py")

            def run(*args):
                return subprocess.run([sys.executable, cli, *args], cwd=root,
                                      env=env, capture_output=True, text=True, check=False)

            anchor = run("receipt-anchor", "--receipts-jsonl", str(receipt_log),
                         "--cohort-id", ingress["cohort_id"], "--groups", "general",
                         "--model-revision", ingress["model_revision"],
                         "--rubric-id", ingress["rubric_id"], "--out", str(receipt_anchor))
            self.assertEqual(anchor.returncode, 0, anchor.stderr)
            self.assertEqual(json.loads(receipt_anchor.read_text()), checkpoint)
            self.assertEqual(receipt_anchor.stat().st_mode & 0o777, 0o600)

            def audit(current_ingress, current_receipt):
                return run("receipt-audit", "--receipts-jsonl", str(receipt_log),
                           "--issued-jsonl", str(issued_log),
                           "--receipt-checkpoint", str(receipt_anchor),
                           "--ingress-checkpoint", str(ingress_anchor),
                           "--expected-receipt-hmac", current_receipt,
                           "--expected-checkpoint-hmac", current_ingress)

            self.assertEqual(audit(ingress["ingress_hmac_sha256"],
                                   checkpoint["receipt_ledger_hmac_sha256"]).returncode, 0)
            missing = issued[:-1]
            new_ingress = create_issued_checkpoint(
                missing, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
                model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
                ingress_key=INGRESS_KEY,
            )
            issued_log.write_text("".join(json.dumps(row) + "\n" for row in missing))
            ingress_anchor.write_text(json.dumps(new_ingress))
            self.assertEqual(audit(ingress["ingress_hmac_sha256"],
                                   checkpoint["receipt_ledger_hmac_sha256"]).returncode, 4)
            conflict = audit(new_ingress["ingress_hmac_sha256"],
                             checkpoint["receipt_ledger_hmac_sha256"])
            self.assertEqual(conflict.returncode, 5, conflict.stderr)
            self.assertEqual(json.loads(conflict.stdout)["conflict_sequences"], [3])


if __name__ == "__main__":
    unittest.main()

"""Bounded falsification of customer-held receipts against closed cohorts."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from client_request_receipt import (
    challenge_request_receipt, issue_request_receipt, verify_request_receipt,
)
from watch_outcome_ledger import create_issued_checkpoint

INGRESS = b"independent-ingress-secret-32-byte-key"
RECEIPT = b"independent-receipt-secret-32-byte-key"
COHORT = "receipt-pilot-001"


def rows(n=120):
    return [
        dict(sequence=i, request_id=f"opaque-{i}", group="general",
             model_revision="fixed-model", rubric_id="fixed-rubric")
        for i in range(1, n + 1)
    ]


def anchor(issued):
    return create_issued_checkpoint(
        issued, cohort_id=COHORT, groups=["general"],
        model_revision="fixed-model", rubric_id="fixed-rubric", ingress_key=INGRESS,
    )


def challenge(receipt, issued, checkpoint, **options):
    return challenge_request_receipt(
        receipt, issued, checkpoint,
        receipt_key=options.get("receipt_key", RECEIPT),
        ingress_key=options.get("ingress_key", INGRESS),
        expected_current_checkpoint_hmac_sha256=options.get(
            "current", checkpoint["ingress_hmac_sha256"]),
    )


class ClientRequestReceiptTests(unittest.TestCase):
    def test_issued_receipt_is_opaque_and_verifies_only_with_its_own_key(self):
        issue = rows(1)[0]
        receipt = issue_request_receipt(issue, cohort_id=COHORT, receipt_key=RECEIPT)
        self.assertTrue(verify_request_receipt(receipt, receipt_key=RECEIPT))
        self.assertFalse(verify_request_receipt(receipt, receipt_key=INGRESS))
        self.assertEqual(set(receipt) - set(issue), {
            "kind", "schema_version", "cohort_id", "nonce", "receipt_hmac_sha256",
        })
        self.assertNotIn("prompt", receipt)
        self.assertNotEqual(receipt["nonce"], issue_request_receipt(
            issue, cohort_id=COHORT, receipt_key=RECEIPT)["nonce"])
        for field, value in (
            ("request_id", "altered"), ("sequence", 2),
            ("group", "other"), ("cohort_id", "other"),
            ("nonce", "0" * 32), ("receipt_hmac_sha256", "0" * 64),
            ("schema_version", True),
        ):
            with self.subTest(field=field):
                self.assertFalse(verify_request_receipt(
                    {**receipt, field: value}, receipt_key=RECEIPT))
        self.assertFalse(verify_request_receipt({**receipt, "prompt": "sensitive"}, receipt_key=RECEIPT))
        with self.assertRaises(ValueError):
            issue_request_receipt({**issue, "prompt": "sensitive"},
                                  cohort_id=COHORT, receipt_key=RECEIPT)
        with self.assertRaises(ValueError):
            issue_request_receipt({**issue, "sequence": True},
                                  cohort_id=COHORT, receipt_key=RECEIPT)

    def test_all_120_single_position_rewrites_are_exposed_by_held_receipts(self):
        issued = rows()
        checkpoint = anchor(issued)
        for index, issue in enumerate(issued):
            with self.subTest(sequence=index + 1):
                receipt = issue_request_receipt(issue, cohort_id=COHORT, receipt_key=RECEIPT)
                included = challenge(receipt, issued, checkpoint)
                self.assertTrue(included["included"])
                self.assertFalse(included["automatic_rollout_permitted"])
                # Simulate an operator closing a new, internally valid signed
                # manifest with this particular issued ID silently replaced.
                changed = [*issued]
                changed[index] = {**issue, "request_id": f"rewritten-{index}"}
                closed = anchor(changed)
                finding = challenge(receipt, changed, closed)
                self.assertFalse(finding["included"])
                self.assertEqual(finding["reason"], "authenticated_receipt_missing_or_rewritten")

    def test_truncation_and_stale_checkpoint_are_distinct(self):
        issued = rows(8)
        receipt = issue_request_receipt(issued[-1], cohort_id=COHORT, receipt_key=RECEIPT)
        truncated = issued[:-1]
        old = anchor(truncated)
        self.assertFalse(challenge(receipt, truncated, old)["included"])
        current = anchor(issued)
        with self.assertRaisesRegex(ValueError, "stale"):
            challenge(receipt, truncated, old, current=current["ingress_hmac_sha256"])
        with self.assertRaisesRegex(ValueError, "manifest"):
            challenge(receipt, truncated, current)
        with self.assertRaises(ValueError):
            challenge(receipt, list(reversed(issued)), current)

    def test_cross_scope_invalid_authentication_and_key_reuse_fail_closed(self):
        issued = rows(2)
        checkpoint = anchor(issued)
        receipt = issue_request_receipt(issued[0], cohort_id=COHORT, receipt_key=RECEIPT)
        variants = [
            ({**receipt, "request_id": "forged"}, {}),
            (receipt, {"receipt_key": b"wrong-receipt-key-with-32-bytes!!!"}),
            (receipt, {"ingress_key": RECEIPT}),
            (receipt, {"current": "0" * 64}),
            (receipt, {"current": None}),
        ]
        for altered, kwargs in variants:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    challenge(altered, issued, checkpoint, **kwargs)
        foreign = issue_request_receipt(issued[0], cohort_id="another-cohort", receipt_key=RECEIPT)
        with self.assertRaisesRegex(ValueError, "another cohort"):
            challenge(foreign, issued, checkpoint)

    def test_cli_receipt_before_anchor_and_challenge_exit_status(self):
        issued = rows(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            issue_json = root / "issue.json"
            issued_jsonl = root / "issued.jsonl"
            receipt_json = root / "receipt.json"
            checkpoint_json = root / "checkpoint.json"
            issue_json.write_text(json.dumps(issued[1]))
            issued_jsonl.write_text("".join(json.dumps(row) + "\n" for row in issued))
            env = {**os.environ, "ARU_RECEIPT_SIGNING_KEY": RECEIPT.decode(),
                   "ARU_ISSUED_SIGNING_KEY": INGRESS.decode()}
            cli = str(Path(__file__).resolve().parents[1] / "watch_outcome_ledger_cli.py")

            def run(*arguments):
                return subprocess.run([sys.executable, cli, *arguments], cwd=root,
                                      env=env, capture_output=True, text=True, check=False)

            receipt_result = run("receipt", "--issue-json", str(issue_json),
                                 "--cohort-id", COHORT, "--out", str(receipt_json))
            self.assertEqual(receipt_result.returncode, 0, receipt_result.stderr)
            self.assertTrue(verify_request_receipt(json.loads(receipt_json.read_text()),
                                                   receipt_key=RECEIPT))
            self.assertEqual(receipt_json.stat().st_mode & 0o777, 0o600)

            def close(manifest):
                issued_jsonl.write_text("".join(json.dumps(row) + "\n" for row in manifest))
                checkpoint = anchor(manifest)
                checkpoint_json.write_text(json.dumps(checkpoint))
                return checkpoint["ingress_hmac_sha256"]

            current = close(issued)

            def cmd(expected):
                return ("challenge", "--receipt", str(receipt_json),
                        "--issued-jsonl", str(issued_jsonl),
                        "--checkpoint", str(checkpoint_json),
                        "--expected-checkpoint-hmac", expected)

            present = run(*cmd(current))
            self.assertEqual(present.returncode, 0, present.stderr)
            self.assertTrue(json.loads(present.stdout)["included"])
            rewritten = [*issued]
            rewritten[1] = {**issued[1], "request_id": "hidden-request"}
            previous = current
            current = close(rewritten)
            stale = run(*cmd(previous))
            self.assertEqual(stale.returncode, 4, stale.stderr)
            missing = run(*cmd(current))
            self.assertEqual(missing.returncode, 5, missing.stderr)
            self.assertFalse(json.loads(missing.stdout)["included"])
            self.assertFalse(json.loads(missing.stdout)["automatic_rollout_permitted"])


if __name__ == "__main__":
    unittest.main()

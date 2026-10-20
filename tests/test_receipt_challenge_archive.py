"""Guard the corrected, immutable public-benchmark receipt evidence bundle."""
import hashlib
import json
import unittest
from pathlib import Path

from anytime_risk_watch import _canonical

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/receipt_challenge_20261020"
PRIOR = ROOT / "evidence/prospective_ledger_20261019"


def read_json(path):
    return json.loads((EVIDENCE / path).read_text())


def read_rows(path):
    return [json.loads(line) for line in (EVIDENCE / path).read_text().splitlines()]


def digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class ArchivedReceiptEvidenceTests(unittest.TestCase):
    def test_protocol_hashes_rows_and_reports_recompute_without_secret_keys(self):
        for name, protocol in (
            ("protocol.sha256", "PROSPECTIVE_LEDGER_PROTOCOL.md"),
            ("receipt-protocol.sha256", "RECEIPT_CHALLENGE_PROTOCOL.md"),
        ):
            self.assertEqual((EVIDENCE / name).read_text().split()[0],
                             hashlib.sha256((ROOT / protocol).read_bytes()).hexdigest())
        issued = read_rows("prepared/issued.jsonl")
        receipts = read_rows("prepared/receipts.jsonl")
        grades = read_rows("run/grades.jsonl")
        outcomes = read_rows("run/actual_model/outcomes.jsonl")
        checkpoint = read_json("prepared/checkpoint.json")
        plan = read_json("prepared/plan.json")
        watch = read_json("run/signed_watch.json")
        report = read_json("run/accounting_report.json")
        self.assertEqual((len(issued), len(receipts), len(grades), len(outcomes)),
                         (120, 120, 120, 120))
        self.assertEqual([r["sequence"] for r in receipts], list(range(1, 121)))
        self.assertEqual(len({r["nonce"] for r in receipts}), 120)
        for claim, row, grade in zip(receipts, issued, grades):
            self.assertEqual({k: claim[k] for k in row}, row)
            self.assertEqual(grade["request_id"], row["request_id"])
        self.assertEqual(digest(issued), checkpoint["issued_manifest_sha256"])
        self.assertEqual(digest(issued), plan["issued_manifest_sha256"])
        self.assertEqual(digest(issued), watch["issued_manifest_sha256"])
        self.assertEqual(digest(issued), report["issued_manifest_sha256"])
        self.assertEqual(digest(receipts), plan["receipt_manifest_sha256"])
        self.assertEqual(digest(receipts), report["receipt_manifest_sha256"])
        self.assertEqual(sum(grade["error"] for grade in grades), report["errors"])
        self.assertEqual(report["errors"], 15)
        self.assertEqual(report["receipt_count"], 120)
        self.assertIs(report["independent_customer_receipt_delivery"], False)
        self.assertIs(report["automatic_rollout_permitted"], False)
        for path in ("prepared/issued.jsonl", "run/actual_model/outcomes.jsonl",
                     "run/grades.jsonl"):
            self.assertEqual((EVIDENCE / path).read_bytes(), (PRIOR / path).read_bytes())


if __name__ == "__main__":
    unittest.main()

"""Public hash-only recomputation of the retained prospective CI artifact.

Ephemeral HMAC keys were discarded. This checks immutable public evidence
relationships; it cannot reauthenticate the historical signatures.
"""
import hashlib
import json
import unittest
from pathlib import Path

from anytime_risk_watch import _canonical, watch_risk_stream

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/prospective_ledger_20261019"


class ProspectiveArchivedEvidenceTests(unittest.TestCase):
    def test_exact_manifest_grades_and_model_trace_recompute(self):
        if not EVIDENCE.exists():
            self.skipTest("measured prospective artifact not yet archived")

        def read(path):
            return json.loads((EVIDENCE / path).read_text())

        def rows(path):
            return [json.loads(line) for line in (EVIDENCE / path).read_text().splitlines()]

        checkpoint = read("prepared/checkpoint.json")
        plan = read("prepared/plan.json")
        issued = rows("prepared/issued.jsonl")
        grades = rows("run/grades.jsonl")
        observed = rows("run/actual_model/outcomes.jsonl")
        signed = read("run/signed_watch.json")
        report = read("run/accounting_report.json")
        original = read("run/actual_model/report.json")
        recorded_protocol_hash = (EVIDENCE / "protocol.sha256").read_text().split()[0]
        actual_protocol_hash = hashlib.sha256((ROOT / "PROSPECTIVE_LEDGER_PROTOCOL.md").read_bytes()).hexdigest()
        self.assertEqual(recorded_protocol_hash, actual_protocol_hash)
        self.assertEqual(len(issued), len(grades))
        self.assertEqual(len(grades), len(observed))
        self.assertEqual(len(observed), 120)
        self.assertEqual(plan["issued_manifest_sha256"], checkpoint["issued_manifest_sha256"])
        self.assertEqual(checkpoint["issued_manifest_sha256"],
                         hashlib.sha256(_canonical(issued).encode()).hexdigest())
        self.assertEqual(signed["ordered_grades_sha256"],
                         hashlib.sha256(_canonical(grades).encode()).hexdigest())
        reconstructed = []
        for index, (issue, grade, outcome) in enumerate(zip(issued, grades, observed), 1):
            self.assertEqual(issue["sequence"], index)
            self.assertEqual(issue["request_id"], grade["request_id"])
            self.assertEqual(outcome["event_id"], "observed-%04d" % index)
            self.assertEqual(issue["group"], outcome["group"])
            self.assertEqual(grade["error"], outcome["error"])
            reconstructed.append({"event_id": issue["request_id"],
                                  "group": issue["group"], "error": grade["error"]})
        calculated = watch_risk_stream(
            reconstructed, groups=checkpoint["groups"],
            model_revision=checkpoint["model_revision"],
            rubric_id=checkpoint["rubric_id"],
            target_error=checkpoint["target_error"],
            alpha=checkpoint["familywise_alpha"],
            bet_fractions=checkpoint["bet_fractions"],
        )
        self.assertEqual(calculated,
                         {k: v for k, v in signed.items()
                          if k in calculated})
        self.assertEqual(sum(row["error"] for row in grades), report["errors"])
        self.assertEqual(report["errors"], 15)
        self.assertEqual(report["model_call_failures_counted_as_errors"], 0)
        self.assertEqual(original["observed_model_watch"]["degradation_alarm"],
                         signed["degradation_alarm"])
        self.assertFalse(report["automatic_rollout_permitted"])


if __name__ == "__main__":
    unittest.main()

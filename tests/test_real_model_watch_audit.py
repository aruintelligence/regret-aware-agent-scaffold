"""Contract tests for real-model handoff and clearly synthetic drift replay."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from real_model_watch_audit import (
    GROUPS, RUBRIC, analyze_outcomes, first_alarm_index, run_audit,
)
from anytime_risk_watch import watch_risk_stream


class FakeQA:
    model_id = "fake-public-qa"
    revision = "fake-fixed-revision"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def answer(self, context, question):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("model execution failed")
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {"accepted": proof["accepted"], "answer": proof["answer"],
                "proof": proof, "model_revision": self.revision}


def dataset_fixture(path):
    short = "The telescope was in Oregon."
    long = short + " It was used for astronomy." * 30
    data = {"version": "1.1", "data": [{"title": "public-test", "paragraphs": [
        {"context": context, "qas": [{
            "id": "test-%02d" % index, "question": "Where was the telescope?",
            "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}],
        }]}
        for index, context in enumerate([short, long] * 8)
    ]}]}
    raw = json.dumps(data).encode("utf-8")
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class RealModelWatchAuditTests(unittest.TestCase):
    def test_real_grading_handoff_keeps_failed_calls_in_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            digest = dataset_fixture(dataset)
            output = Path(folder) / "audit"
            report = run_audit(dataset, output, expected_dataset_sha256=digest,
                               qa_model=FakeQA.model_id, engine=FakeQA(fail_at=4),
                               acknowledge_model_calls=True, limit=12, offset=2,
                               onsets=(4, 8), null_trials=20)
            self.assertEqual(report["observed"], 12)
            self.assertEqual(report["model_call_failures_counted_as_errors"], 1)
            self.assertEqual(report["verified_source_certificates"], 11)
            self.assertEqual(report["observed_model_watch"]["outcome_count"], 12)
            self.assertFalse(report["automatic_rollout_permitted"])
            rows = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["error"] for row in rows), 1)
            self.assertEqual(set(row["group"] for row in rows), set(GROUPS))
            self.assertNotIn("telescope", (output / "report.json").read_text())
            self.assertEqual(watch_risk_stream(
                rows, groups=GROUPS, model_revision=FakeQA.revision,
                rubric_id=RUBRIC, target_error=0.5,
            )["ordered_outcomes_sha256"],
                report["observed_model_watch"]["ordered_outcomes_sha256"])
            with self.assertRaises(FileExistsError):
                run_audit(dataset, output, expected_dataset_sha256=digest,
                          qa_model=FakeQA.model_id, engine=FakeQA(),
                          acknowledge_model_calls=True, limit=12, offset=2,
                          onsets=(4, 8), null_trials=20)

    def test_pinned_dataset_and_explicit_call_consent(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            digest = dataset_fixture(dataset)
            opts = dict(expected_dataset_sha256=digest, qa_model=FakeQA.model_id,
                        engine=FakeQA(), limit=12, offset=2, onsets=(4,), null_trials=2)
            with self.assertRaises(ValueError):
                run_audit(dataset, Path(folder) / "unapproved", **opts)
            with self.assertRaises(ValueError):
                run_audit(dataset, Path(folder) / "wrong-hash",
                          **{**opts, "expected_dataset_sha256": "0" * 64},
                          acknowledge_model_calls=True)

    def test_detected_drift_is_explicitly_synthetic(self):
        rows = [{"event_id": str(i), "group": GROUPS[i % 2], "error": False}
                for i in range(120)]
        result = analyze_outcomes(rows, model_revision="fixed", onsets=(30, 60, 90), null_trials=60)
        self.assertIsNone(result["observed_baseline_first_alarm_global_observation"])
        strong = next(s for s in result["injected_deterioration"]
                      if s["onset_after_observations"] == 30
                      and s["originally_correct_flip_probability"] == 1.0)
        self.assertTrue(strong["synthetic_injection"])
        self.assertTrue(strong["detected_after_onset"])
        self.assertGreater(strong["detection_delay_observations"], 0)
        self.assertEqual(strong["flipped_outcomes"], 90)
        self.assertTrue(result["synthetic_iid_null"]["does_not_estimate_real_model_false_alarm_rate"])

    def test_subgroup_crossing_maps_to_global_observation(self):
        rows = [{"event_id": str(i), "group": GROUPS[i % 2],
                 "error": i % 2 == 1} for i in range(40)]
        watch = watch_risk_stream(rows, groups=GROUPS, model_revision="fixed",
                                  rubric_id=RUBRIC, target_error=0.5)
        alarm = first_alarm_index(rows, watch)
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm % 2, 0)
        self.assertIn("long_source", watch["alarm_groups"])


if __name__ == "__main__":
    unittest.main()

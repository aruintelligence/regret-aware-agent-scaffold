"""Calibration coverage arithmetic, alarm budget and real-call handoff checks."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from calibrated_change_audit import run_calibrated_audit
from calibrated_change_watch import calibrated_restart_risk_stream
from grounding_contract import ground_answer


GROUPS = ("short_source", "long_source")


def stream(n, *, errors=(), prefix="x"):
    return [{"event_id": f"{prefix}-{i}", "group": GROUPS[i % 2], "error": i in errors}
            for i in range(n)]


def watch(calibration, evaluation, **kwargs):
    opts = {"groups": GROUPS, "model_revision": "fixed-revision",
            "rubric_id": "human-gold-v1", "starts": (1, 21, 41, 61, 81, 101),
            "horizon": 120}
    return calibrated_restart_risk_stream(calibration, evaluation, **{**opts, **kwargs})


class FakeQA:
    model_id = "test-extractive"
    revision = "fixed-revision"

    def __init__(self):
        self.calls = 0

    def answer(self, context, question):
        self.calls += 1
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {"accepted": proof["accepted"], "answer": proof["answer"],
                "proof": proof, "model_revision": self.revision}


class CalibratedWatchTests(unittest.TestCase):
    def test_bound_and_familywise_budget_are_exact(self):
        calibration = stream(120, errors=set(range(0, 120, 10)))
        result = watch(calibration, stream(120, prefix="e"))
        target = result["calibration_targets"]["__all__"]
        self.assertEqual(target["n_calibration"], 120)
        self.assertEqual(target["errors"], 12)
        self.assertAlmostEqual(target["one_sided_upper_error_bound"],
                               .1 + math.sqrt(math.log(3 / .025) / 240))
        self.assertTrue(result["calibration_ready"])
        self.assertAlmostEqual(result["log_alarm_threshold"], math.log(18 / .025))
        self.assertEqual(result["component_count"], 18)
        self.assertFalse(result["degradation_alarm"])
        self.assertFalse(result["automatic_rollout_permitted"])

    def test_late_shift_is_detected_with_separate_calibration(self):
        calibration = stream(120)
        baseline = stream(120, prefix="e")
        changed = stream(120, errors=set(range(55, 120)), prefix="e")
        self.assertIsNone(watch(calibration, baseline)["first_alarm_global_observation"])
        result = watch(calibration, changed)
        self.assertTrue(result["degradation_alarm"])
        self.assertGreater(result["first_alarm_global_observation"], 55)
        self.assertLessEqual(result["first_alarm_global_observation"], 120)
        self.assertEqual(result["monitoring_outcome_count"], 120)

    def test_inadequate_calibration_is_explicitly_inconclusive(self):
        calibration = [{"event_id": f"c-{i}", "group": "short_source", "error": False}
                       for i in range(120)]
        result = watch(calibration, stream(120, prefix="e"))
        self.assertFalse(result["calibration_ready"])
        self.assertEqual(result["monitoring_state"], "inconclusive_calibration")
        self.assertIsNone(result["degradation_alarm"])
        self.assertEqual(result["windows"], [])
        self.assertIn("long_source:insufficient_calibration", result["calibration_failures"])

    def test_invalid_design_and_labels_fail(self):
        calibration, evaluation = stream(120), stream(120, prefix="e")
        for opts in ({"calibration_delta": 0.05}, {"alpha_total": True},
                     {"min_slice_calibration": 0}, {"horizon": 119}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                watch(calibration, evaluation, **opts)
        with self.assertRaises(ValueError):
            watch([{**calibration[0], "error": 1}], evaluation)

    def test_distinct_real_call_streams_and_disjoint_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            short = "Telescope records were stored in a museum. " * 4 + "Oregon was home."
            long = "Telescope records were stored in a museum. " * 20 + "Oregon was home."
            data = {"version": "1.1", "data": [{"title": "test", "paragraphs": [
                {"context": context, "qas": [{
                    "id": f"q-{i:03d}", "question": "Where was home?",
                    "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}],
                }]}
                for i, context in enumerate([short, long] * 26)
            ]}]}
            raw = json.dumps(data).encode()
            dataset = Path(folder) / "dev.json"
            dataset.write_bytes(raw)
            qa = FakeQA()
            out = Path(folder) / "audit"
            report = run_calibrated_audit(
                dataset, out, expected_dataset_sha256=hashlib.sha256(raw).hexdigest(),
                qa_model=qa.model_id, engine=qa, acknowledge_model_calls=True,
                calibration_offset=0, evaluation_offset=24,
                calibration_limit=24, evaluation_limit=24,
                starts=(1, 9, 17), onsets=(6, 12, 18),
                intervention_onset=12, context_cap=128,
            )
            self.assertEqual(qa.calls, 60)  # 24 calibration + 24 baseline + 12 intervention.
            self.assertEqual(report["calibration_model_calls"], 24)
            self.assertEqual(report["baseline_model_calls"], 24)
            self.assertEqual(report["intervention_additional_model_calls"], 12)
            self.assertEqual(report["calibration_errors"], 0)
            self.assertEqual(report["baseline_errors"], 0)
            self.assertEqual(report["intervention_errors"], 12)
            self.assertFalse(report["baseline_calibrated_watch"]["calibration_ready"])
            self.assertIsNone(report["intervention_calibrated_watch"]["degradation_alarm"])
            self.assertFalse(report["automatic_rollout_permitted"])
            self.assertEqual(len((out / "calibration/outcomes.jsonl").read_text().splitlines()), 24)
            self.assertEqual(len((out / "evaluation/intervention_outcomes.jsonl").read_text().splitlines()), 24)
            with self.assertRaises(ValueError):
                run_calibrated_audit(dataset, Path(folder) / "overlap",
                                     expected_dataset_sha256=hashlib.sha256(raw).hexdigest(),
                                     qa_model=qa.model_id, engine=qa,
                                     acknowledge_model_calls=True, calibration_offset=0,
                                     evaluation_offset=20, calibration_limit=24,
                                     evaluation_limit=24)


if __name__ == "__main__":
    unittest.main()

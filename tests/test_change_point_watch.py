"""Contract and falsification checks for the finite restart research monitor."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from change_point_audit import analyze_restarts, run_change_point_audit
from change_point_watch import restart_risk_stream
from grounding_contract import ground_answer


def rows(n, *, onset=None, groups=("short_source", "long_source")):
    return [{"event_id": str(i), "group": groups[i % len(groups)],
             "error": onset is not None and i >= onset} for i in range(n)]


class FakeQA:
    model_id = "test-extractive-model"
    revision = "fixed-test-revision"

    def __init__(self):
        self.calls = 0

    def answer(self, context, question):
        self.calls += 1
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {"accepted": proof["accepted"], "answer": proof["answer"],
                "proof": proof, "model_revision": self.revision}


class RestartMonitorTests(unittest.TestCase):
    def test_late_shift_alarms_after_start_and_pays_entire_budget(self):
        stream = rows(120, onset=60)
        result = restart_risk_stream(
            stream, groups=["short_source", "long_source"],
            model_revision="m", rubric_id="r", target_error=0.5,
            starts=(1, 21, 41, 61, 81, 101), horizon=120,
        )
        self.assertEqual(result["component_count"], 18)
        self.assertAlmostEqual(result["log_alarm_threshold"], math.log(18 / 0.05))
        self.assertTrue(result["degradation_alarm"])
        self.assertGreater(result["first_alarm_global_observation"], 60)
        self.assertFalse(result["automatic_rollout_permitted"])
        self.assertEqual(len(result["ordered_outcomes_sha256"]), 64)
        # A future start is counted in the multiplicity penalty on an early prefix.
        early = restart_risk_stream(
            stream[:30], groups=["short_source", "long_source"],
            model_revision="m", rubric_id="r", target_error=0.5,
            starts=(1, 21, 41, 61, 81, 101), horizon=120,
        )
        self.assertEqual(early["component_count"], 18)
        self.assertEqual(early["windows"][-1]["slices"]["__all__"]["n_observed"], 0)
        self.assertFalse(early["degradation_alarm"])

    def test_group_crossing_is_global_request_position(self):
        stream = [{"event_id": str(i), "group": "short_source" if i % 2 == 0 else "long_source",
                   "error": bool(i % 2)} for i in range(60)]
        result = restart_risk_stream(
            stream, groups=["short_source", "long_source"],
            model_revision="m", rubric_id="r", target_error=0.1,
            starts=(1, 21), horizon=60,
        )
        group_crossings = [a for a in result["alarm_components"] if a["slice"] == "long_source"]
        self.assertTrue(group_crossings)
        self.assertTrue(all(a["first_crossing_global_observation"] % 2 == 0
                            for a in group_crossings))

    def test_uncommitted_or_malformed_design_is_rejected(self):
        stream = rows(6)
        opts = dict(groups=["short_source", "long_source"], model_revision="m",
                    rubric_id="r", target_error=0.5, horizon=6)
        for schedule in ((2, 3), (1, 1), (1, 7), (1, 0), (1, True), (1, 3, 2), ()):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                restart_risk_stream(stream, starts=schedule, **opts)
        with self.assertRaises(ValueError):
            restart_risk_stream(stream, starts=(1, 3), **{**opts, "horizon": 5})
        with self.assertRaises(ValueError):
            restart_risk_stream([{**stream[0], "error": 1}], starts=(1,), **opts)

    def test_one_real_trace_and_distinctly_marked_synthetic_stress(self):
        stream = rows(120)
        result = analyze_restarts(stream, model_revision="m", starts=(1, 21, 41, 61, 81, 101),
                                  onsets=(25, 55, 85), null_trials=15)
        self.assertEqual(len(result["injected_label_flip_scenarios"]), 6)
        self.assertTrue(all(s["kind"].startswith("synthetic_")
                            for s in result["injected_label_flip_scenarios"]))
        self.assertTrue(result["synthetic_iid_conditional_null"]["does_not_estimate_real_world_false_alarm_rate"])

    def test_actual_context_intervention_is_scored_and_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            context = "The telescope was placed in a museum. " * 5 + "Oregon was its home."
            data = {"version": "1.1", "data": [{"title": "test", "paragraphs": [{
                "context": context, "qas": [{"id": "q-%02d" % i, "question": "Where was it?",
                                      "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}]}
                                     for i in range(14)]}]}]}
            path = Path(folder) / "dev.json"
            raw = json.dumps(data).encode()
            path.write_bytes(raw)
            qa = FakeQA()
            dest = Path(folder) / "new-audit"
            report = run_change_point_audit(
                path, dest, expected_dataset_sha256=hashlib.sha256(raw).hexdigest(),
                qa_model=qa.model_id, engine=qa, acknowledge_model_calls=True,
                offset=0, limit=12, starts=(1, 5, 9), onsets=(3, 7),
                intervention_onset=6, context_cap=20, null_trials=3,
            )
            self.assertEqual(qa.calls, 18)
            self.assertEqual(report["baseline_errors"], 0)
            self.assertEqual(report["intervention_errors"], 6)
            self.assertEqual(report["intervention_additional_model_calls"], 6)
            self.assertEqual(report["baseline_verified_source_certificates"], 12)
            self.assertEqual(report["intervention_verified_source_certificates"], 0)
            self.assertFalse(report["intervention_is_natural_live_drift"])
            self.assertFalse(report["automatic_rollout_permitted"])
            observed = [json.loads(line) for line in (dest / "intervention_outcomes.jsonl").read_text().splitlines()]
            self.assertEqual(len(observed), 12)
            self.assertEqual(sum(row["error"] for row in observed[:6]), 0)
            self.assertEqual(sum(row["error"] for row in observed[6:]), 6)
            self.assertEqual(report["observed_intervention_restart_watch"]["outcome_count"], 12)


if __name__ == "__main__":
    unittest.main()

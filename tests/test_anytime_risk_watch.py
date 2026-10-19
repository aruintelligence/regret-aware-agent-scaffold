"""Prospective degradation e-process mathematical and tamper tests."""
import copy
import math
import random
import unittest

from anytime_risk_watch import (
    gate_with_risk_watch, sign_watch_report, verify_watch_report,
    watch_risk_stream,
)
from grounding_contract import ground_answer
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy

WATCH_KEY = b"independent-prospective-watch-secret-32"
POLICY_KEY = b"independent-calibration-policy-key-32"


def outcomes(n=120, *, group="general", error=False):
    return [
        {"event_id": f"event-{group}-{i}", "group": group, "error": error}
        for i in range(n)
    ]


def watch(rows=None):
    return watch_risk_stream(
        rows if rows is not None else outcomes(),
        groups=["general"], model_revision="fixed-qa-sha",
        rubric_id="independent-grader-v1", target_error=0.1, alpha=0.05,
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
        model_revision="fixed-qa-sha", rubric_id="independent-grader-v1",
        calibration_set_id="frozen-calibration",
    ), key=POLICY_KEY)


class AnytimeRiskWatchTests(unittest.TestCase):
    def test_no_errors_does_not_trigger_alarm(self):
        report = watch()
        self.assertFalse(report["degradation_alarm"])
        self.assertEqual(report["slices"]["general"]["errors"], 0)
        self.assertEqual(report["slices"]["general"]["n_observed"], 120)
        self.assertEqual(report["monitoring_state"], "no_alarm_observed")
        self.assertFalse(report["automatic_rollout_permitted"])

    def test_sustained_errors_trigger_anytime_alarm(self):
        report = watch(outcomes(40, error=True))
        self.assertTrue(report["degradation_alarm"])
        self.assertIn("general", report["alarm_groups"])
        self.assertIn("__all__", report["alarm_groups"])
        self.assertIsNotNone(report["slices"]["general"]["first_crossing_observation"])

    def test_no_group_data_is_not_a_pass_certificate(self):
        report = watch(outcomes(20, group="general"))
        self.assertEqual(report["slices"]["general"]["n_observed"], 20)
        self.assertFalse(report["degradation_alarm"])
        self.assertIn("A lack of alarm is NOT evidence", report["limitations"][0])

    def test_detects_one_group_deterioration(self):
        rows = outcomes(80, group="general", error=False)
        rows += outcomes(40, group="technical", error=True)
        report = watch_risk_stream(
            rows, groups=["general", "technical"],
            model_revision="fixed-qa-sha", rubric_id="independent-grader-v1",
            target_error=0.1,
        )
        self.assertIn("technical", report["alarm_groups"])
        self.assertNotIn("general", report["alarm_groups"])
        self.assertEqual(report["slice_count"], 3)

    def test_fixed_single_bet_product_matches_exact_formula(self):
        report = watch_risk_stream(
            outcomes(3, error=True), groups=["general"],
            model_revision="fixed-qa-sha", rubric_id="independent-grader-v1",
            target_error=0.1, bet_fractions=[0.5],
        )
        # lambda=0.5/0.1=5, factor=1+5*(1-0.1)=5.5
        self.assertAlmostEqual(
            report["slices"]["general"]["final_log_e"],
            3 * math.log(5.5),
        )

    def test_bad_labels_groups_and_duplicates_fail_closed(self):
        for rows in (
            outcomes(3) + [outcomes(3)[0]],
            [{"event_id": "x", "group": "unknown", "error": False}],
            [{"event_id": "x", "group": "general", "error": 0}],
            [{"event_id": "x", "group": "general", "error": False, "prompt": "private"}],
        ):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    watch(rows)
        with self.assertRaises(ValueError):
            watch_risk_stream(
                outcomes(3), groups=["general"], model_revision="m", rubric_id="r",
                bet_fractions=[1.0],
            )

    def test_sign_verify_and_tamper_detection(self):
        signed = sign_watch_report(watch(), key=WATCH_KEY)
        self.assertTrue(verify_watch_report(signed, key=WATCH_KEY))
        self.assertFalse(verify_watch_report(signed, key=b"wrong-separate-monitor-key-32-bytes"))
        changed = copy.deepcopy(signed)
        changed["degradation_alarm"] = True
        self.assertFalse(verify_watch_report(changed, key=WATCH_KEY))

    def test_dual_gate_requires_both_signatures_and_source_proof(self):
        signed_watch = sign_watch_report(watch(), key=WATCH_KEY)
        signed_policy = policy()
        source = "The telescope was in Oregon."
        proof = ground_answer("Oregon", source, model_revision="fixed-qa-sha")
        kwargs = {
            "policy": signed_policy, "policy_key": POLICY_KEY,
            "watch_report": signed_watch, "watch_key": WATCH_KEY,
            "expected_outcomes_sha256": signed_watch["ordered_outcomes_sha256"],
            "score": 0.9, "group": "general",
            "source_proof": proof, "source_text": source,
            "model_revision": "fixed-qa-sha",
            "rubric_id": "independent-grader-v1",
        }
        accepted = gate_with_risk_watch(**kwargs)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["answer"], "Oregon")
        self.assertFalse(accepted["automatic_rollout_permitted"])
        self.assertEqual(accepted["watch_outcome_count"], 120)
        for update, reason in (
            ({"watch_report": None}, "missing_or_invalid_signed_watch"),
            ({"watch_key": b"wrong-watch-secret-key-32-bytes"}, "missing_or_invalid_signed_watch"),
            ({"group": "unknown"}, "unmonitored_group"),
            ({"model_revision": "new-model"}, "watch_model_or_rubric_mismatch"),
            ({"source_text": source + " altered"}, "invalid_source_certificate"),
            ({"expected_outcomes_sha256": "0" * 64}, "missing_or_stale_watch_checkpoint"),
            ({"expected_outcomes_sha256": None}, "missing_or_stale_watch_checkpoint"),
            ({"min_group_outcomes": 121}, "insufficient_recent_group_outcomes"),
        ):
            with self.subTest(update=update):
                result = gate_with_risk_watch(**{**kwargs, **update})
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], reason)
        alarm = sign_watch_report(watch(outcomes(40, error=True)), key=WATCH_KEY)
        blocked = gate_with_risk_watch(**{**kwargs, "watch_report": alarm})
        self.assertFalse(blocked["accepted"])
        self.assertEqual(blocked["reason"], "prospective_degradation_alarm")

    def test_synthetic_conditional_null_false_alarms_sanity(self):
        rng = random.Random(20261019)
        alarms = 0
        for trial in range(80):
            rows = [
                {"event_id": f"{trial}-{i}", "group": "general", "error": rng.random() < 0.1}
                for i in range(80)
            ]
            alarms += watch(rows)["degradation_alarm"]
        self.assertLessEqual(alarms, 10)


if __name__ == "__main__":
    unittest.main()

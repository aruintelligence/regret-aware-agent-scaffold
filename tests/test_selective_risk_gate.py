"""Mathematical and security regression tests for simultaneous risk screening."""
import copy
import math
import unittest

from grounding_contract import ground_answer
from selective_risk_gate import (
    _log_binomial_cdf, apply_selective_risk_gate,
    exact_binomial_error_upper, fit_selective_risk_policy,
    sign_risk_policy, verify_risk_policy,
)

KEY = b"independent-risk-screen-test-key-32-bytes"


def calibration():
    rows = []
    for group in ("technical", "general"):
        for i in range(300):
            rows.append({
                "sample_id": f"{group}-{i}",
                "group": group,
                "score": 0.9 if i < 180 else 0.2,
                "correct": i < 180,
            })
    return rows


def policy(rows=None, *, groups=("technical", "general"), target_error=0.05):
    return fit_selective_risk_policy(
        calibration() if rows is None else rows,
        thresholds=[0.1, 0.5, 0.95], groups=list(groups),
        target_error=target_error, alpha=0.05,
        min_group_support=30, model_revision="model-sha-123",
        rubric_id="independent-human-grader-v1",
        calibration_set_id="predeclared-holdout-001",
    )


class ExactBinomialTests(unittest.TestCase):
    def test_zero_error_bound_is_exact(self):
        n, alpha = 100, 0.05
        upper = exact_binomial_error_upper(0, n, alpha=alpha)
        self.assertAlmostEqual(upper, 1 - alpha ** (1 / n), places=12)

    def test_one_sided_cdf_inversion(self):
        n, k, alpha = 100, 5, 0.025
        upper = exact_binomial_error_upper(k, n, alpha=alpha)
        self.assertAlmostEqual(_log_binomial_cdf(k, n, upper), math.log(alpha), places=9)
        self.assertGreater(upper, k / n)

    def test_bound_monotone_in_errors(self):
        values = [exact_binomial_error_upper(k, 100, alpha=0.05) for k in range(10)]
        self.assertEqual(values, sorted(values))

    def test_no_evidence_cannot_certify(self):
        self.assertEqual(exact_binomial_error_upper(0, 0, alpha=0.05), 1)
        self.assertEqual(exact_binomial_error_upper(10, 10, alpha=0.05), 1)

    def test_invalid_counts_and_alpha(self):
        for k, n, alpha in ((-1, 10, .05), (11, 10, .05), (True, 10, .05),
                            (1, 2.5, .05), (0, 5, 0), (0, 5, float("nan"))):
            with self.subTest(k=k, n=n, alpha=alpha):
                with self.assertRaises(ValueError):
                    exact_binomial_error_upper(k, n, alpha=alpha)


class SelectiveRiskTests(unittest.TestCase):
    def test_chooses_widest_eligible_coverage(self):
        result = policy()
        self.assertTrue(result["conditional_risk_screen_passed"])
        self.assertEqual(result["selected_threshold"], 0.5)
        self.assertAlmostEqual(result["selected_coverage"], 0.6)
        self.assertFalse(result["candidates"][0]["eligible"])
        self.assertTrue(result["candidates"][1]["eligible"])
        self.assertFalse(result["candidates"][2]["eligible"])
        self.assertEqual(result["simultaneous_tests"], 9)
        self.assertFalse(result["automatic_rollout_permitted"])

    def test_all_groups_must_pass_even_if_aggregate_looks_good(self):
        rows = []
        for group, count in (("technical", 400), ("general", 100)):
            for i in range(count):
                rows.append({
                    "sample_id": f"{group}-{i}", "score": 0.9,
                    "group": group,
                    "correct": not (group == "general" and i < 25),
                })
        result = policy(rows, target_error=0.12)
        self.assertFalse(result["conditional_risk_screen_passed"])
        self.assertIsNone(result["selected_threshold"])
        aggregate = result["candidates"][1]["slices"]["__all__"]
        group = result["candidates"][1]["slices"]["general"]
        self.assertTrue(aggregate["meets_target"])
        self.assertFalse(group["meets_target"])

    def test_missing_predeclared_group_fails_closed(self):
        rows = [r for r in calibration() if r["group"] == "technical"]
        result = policy(rows)
        self.assertFalse(result["conditional_risk_screen_passed"])
        self.assertEqual(result["selected_coverage"], 0)

    def test_sample_reordering_does_not_change_digest_or_threshold(self):
        rows = calibration()
        a = policy(rows)
        b = policy(list(reversed(rows)))
        self.assertEqual(a["calibration_sha256"], b["calibration_sha256"])
        self.assertEqual(a["selected_threshold"], b["selected_threshold"])

    def test_duplicate_sample_ids_rejected(self):
        rows = calibration()
        with self.assertRaises(ValueError):
            policy(rows + [rows[0]])

    def test_posthoc_unknown_group_and_thresholds_rejected(self):
        rows = calibration()
        with self.assertRaises(ValueError):
            policy(rows, groups=("technical",))
        with self.assertRaises(ValueError):
            fit_selective_risk_policy(
                rows, thresholds=[0.5, 0.5], groups=["technical", "general"],
                model_revision="sha", rubric_id="rubric", calibration_set_id="id",
            )

    def test_nan_and_incorrect_label_types_rejected(self):
        rows = calibration()
        bad = copy.deepcopy(rows)
        bad[0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            policy(bad)
        bad = copy.deepcopy(rows)
        bad[0]["correct"] = 1
        with self.assertRaises(ValueError):
            policy(bad)

    def test_signature_and_tamper_detection(self):
        signed = sign_risk_policy(policy(), key=KEY)
        self.assertTrue(verify_risk_policy(signed, key=KEY))
        self.assertFalse(verify_risk_policy(signed, key=b"different-independent-key-32-bytes"))
        altered = copy.deepcopy(signed)
        altered["selected_threshold"] = 0.1
        self.assertFalse(verify_risk_policy(altered, key=KEY))
        altered = copy.deepcopy(signed)
        altered["candidates"][1]["slices"]["general"]["errors"] = 0.1
        self.assertFalse(verify_risk_policy(altered, key=KEY))

    def test_verified_source_proof_and_risk_screen_release(self):
        signed = sign_risk_policy(policy(), key=KEY)
        source = "The observatory is in Oregon."
        proof = ground_answer("Oregon", source, model_revision="model-sha-123")
        kwargs = dict(
            key=KEY, score=0.9, group="technical", source_proof=proof,
            source_text=source, model_revision="model-sha-123",
            rubric_id="independent-human-grader-v1",
        )
        accepted = apply_selective_risk_gate(signed, **kwargs)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["answer"], "Oregon")
        self.assertLessEqual(accepted["risk_upper_bound"], 0.05)
        self.assertFalse(accepted["automatic_rollout_permitted"])
        for change, reason in (
            ({"score": 0.1}, "score_below_threshold"),
            ({"group": "unknown"}, "unrecognized_group"),
            ({"model_revision": "new-model"}, "model_or_rubric_mismatch"),
            ({"source_text": source + " changed"}, "invalid_source_certificate"),
            ({"source_proof": None}, "invalid_source_certificate"),
            ({"source_proof": ground_answer("Oregon", source, model_revision="other-model")},
             "source_proof_model_mismatch"),
        ):
            with self.subTest(change=change):
                outcome = apply_selective_risk_gate(signed, **{**kwargs, **change})
                self.assertFalse(outcome["accepted"])
                self.assertEqual(outcome["answer"], "")
                self.assertEqual(outcome["reason"], reason)

    def test_unsigned_and_failed_calibration_always_abstain(self):
        source = "The observatory is in Oregon."
        proof = ground_answer("Oregon", source)
        kwargs = dict(
            key=KEY, score=0.9, group="technical", source_proof=proof,
            source_text=source, model_revision="model-sha-123",
            rubric_id="independent-human-grader-v1",
        )
        self.assertEqual(
            apply_selective_risk_gate(policy(), **kwargs)["reason"],
            "unsigned_or_modified_policy",
        )
        failed = sign_risk_policy(policy(calibration()[:10]), key=KEY)
        self.assertEqual(
            apply_selective_risk_gate(failed, **kwargs)["reason"],
            "no_risk_qualified_threshold",
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from grounding_contract import ground_answer
from risk_qa_adapter import answer_with_calibrated_risk
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy

KEY = b"risk-adapter-independent-key-32-bytes"


class FakeQA:
    model_id = "test-qa"
    revision = "immutable-qa-model-sha"

    def __init__(self, score=0.9, answer="Oregon"):
        self.score = score
        self.text = answer
        self.calls = 0

    def answer(self, context, question):
        self.calls += 1
        proof = ground_answer(self.text, context, model_revision=self.revision)
        return {
            "accepted": proof["accepted"], "answer": proof["answer"],
            "proof": proof if proof["accepted"] else None,
            "model_revision": self.revision,
            "uncalibrated_span_logit_score": self.score,
        }


def signed_policy():
    rows = [
        {"sample_id": str(i), "group": "public_qa",
         "score": 0.9 if i < 150 else 0.1, "correct": i < 150}
        for i in range(200)
    ]
    policy = fit_selective_risk_policy(
        rows, thresholds=[0.05, 0.5, 0.95], groups=["public_qa"],
        target_error=0.05, alpha=0.05, min_group_support=30,
        model_revision="immutable-qa-model-sha", rubric_id="independent-exact-match-v1",
        calibration_set_id="predeclared-calibration",
    )
    return sign_risk_policy(policy, key=KEY)


class RiskQAAdapterTests(unittest.TestCase):
    def test_qualified_model_with_source_proof_releases(self):
        engine = FakeQA()
        result = answer_with_calibrated_risk(
            engine, "The telescope was in Oregon.", "Where?",
            policy=signed_policy(), key=KEY,
            trusted_group="public_qa", rubric_id="independent-exact-match-v1",
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["answer"], "Oregon")
        self.assertFalse(result["automatic_rollout_permitted"])
        self.assertEqual(engine.calls, 1)

    def test_low_score_abstains(self):
        engine = FakeQA(score=0.1)
        result = answer_with_calibrated_risk(
            engine, "The telescope was in Oregon.", "Where?",
            policy=signed_policy(), key=KEY,
            trusted_group="public_qa", rubric_id="independent-exact-match-v1",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "score_below_threshold")

    def test_unsupported_model_answer_abstains(self):
        engine = FakeQA(answer="Washington")
        result = answer_with_calibrated_risk(
            engine, "The telescope was in Oregon.", "Where?",
            policy=signed_policy(), key=KEY,
            trusted_group="public_qa", rubric_id="independent-exact-match-v1",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "qa_model_abstained")

    def test_invalid_signature_blocks_before_model_call(self):
        engine = FakeQA()
        policy = signed_policy()
        policy["target_error"] = 0.9
        result = answer_with_calibrated_risk(
            engine, "The telescope was in Oregon.", "Where?",
            policy=policy, key=KEY,
            trusted_group="public_qa", rubric_id="independent-exact-match-v1",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(engine.calls, 0)

    def test_wrong_rubric_and_untrusted_group_fail_before_model(self):
        engine = FakeQA()
        for group, rubric in (("unknown", "independent-exact-match-v1"),
                              ("public_qa", "different-rubric")):
            result = answer_with_calibrated_risk(
                engine, "The telescope was in Oregon.", "Where?",
                policy=signed_policy(), key=KEY,
                trusted_group=group, rubric_id=rubric,
            )
            self.assertFalse(result["accepted"])
        self.assertEqual(engine.calls, 0)


if __name__ == "__main__":
    unittest.main()

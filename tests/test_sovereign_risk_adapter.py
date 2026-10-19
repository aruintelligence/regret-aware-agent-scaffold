"""End-to-end sovereign source authorization + calibrated risk + provenance tests."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy
from sovereign_risk_adapter import answer_with_sovereign_risk
from sovereign_source_registry import grant_source, revoke_source
from source_registry_anchor import create_source_checkpoint

REGISTRY_KEY = b"independent-sovereign-registry-key-32"
ANCHOR_KEY = b"independent-sovereign-anchor-key-32"
RISK_KEY = b"independent-sovereign-risk-key-32"


class FakeQA:
    model_id = "qa-model"
    revision = "immutable-model-revision"

    def __init__(self):
        self.calls = 0

    def answer(self, context, question):
        self.calls += 1
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {
            "accepted": proof["accepted"], "answer": proof["answer"],
            "proof": proof, "model_revision": self.revision,
            "uncalibrated_span_logit_score": 0.9,
        }


def policy():
    rows = [
        {"sample_id": str(i), "group": "public_qa",
         "score": 0.9 if i < 150 else 0.1, "correct": i < 150}
        for i in range(200)
    ]
    return sign_risk_policy(fit_selective_risk_policy(
        rows, thresholds=[0.05, 0.5, 0.95], groups=["public_qa"],
        target_error=0.05, alpha=0.05, min_group_support=30,
        model_revision="immutable-model-revision", rubric_id="independent-rubric",
        calibration_set_id="approved-calibration-001",
    ), key=RISK_KEY)


class SovereignRiskTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = str(Path(self.folder.name) / "registry.sqlite")
        self.source = "The telescope was in Oregon."
        self.qa = FakeQA()
        self.policy = policy()
        self.kwargs = {
            "owner": "authorized-operator",
            "purpose": "approved-research",
            "registry_db": self.db,
            "registry_key": REGISTRY_KEY,
            "policy": self.policy,
            "risk_key": RISK_KEY,
            "trusted_group": "public_qa",
            "rubric_id": "independent-rubric",
            "expected_calibration_set_id": "approved-calibration-001",
        }

    def tearDown(self):
        self.folder.cleanup()

    def ask(self, **changes):
        return answer_with_sovereign_risk(
            self.qa, self.source, "Where was the telescope?",
            **{**self.kwargs, **changes},
        )

    def grant(self):
        return grant_source(
            self.db, self.source, owner="authorized-operator",
            purpose="approved-research", ttl_seconds=3600,
            key=REGISTRY_KEY, consent_asserted=True,
        )

    def test_missing_grant_denies_before_model_call(self):
        result = self.ask()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(self.qa.calls, 0)

    def test_active_grant_and_signed_policy_release_verified_span(self):
        self.grant()
        result = self.ask()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["answer"], "Oregon")
        self.assertEqual(self.qa.calls, 1)
        self.assertFalse(result["automatic_rollout_permitted"])
        self.assertEqual(result["consent_status"], "operator_attested_not_independently_verified")

    def test_revocation_blocks_before_inference(self):
        grant = self.grant()
        revoke_source(self.db, grant["grant_id"], owner="authorized-operator", key=REGISTRY_KEY)
        result = self.ask()
        self.assertFalse(result["accepted"])
        self.assertEqual(self.qa.calls, 0)

    def test_wrong_owner_and_purpose_fail_closed(self):
        self.grant()
        self.assertFalse(self.ask(owner="other-operator")["accepted"])
        self.assertFalse(self.ask(purpose="different-purpose")["accepted"])
        self.assertEqual(self.qa.calls, 0)

    def test_pinned_calibration_prevents_old_policy_replay(self):
        self.grant()
        result = self.ask(expected_calibration_set_id="future-calibration-002")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "calibration_policy_replay_or_mismatch")
        self.assertEqual(self.qa.calls, 0)

    def test_external_checkpoint_detects_deleted_revocation(self):
        grant = self.grant()
        revoke_source(self.db, grant["grant_id"], owner="authorized-operator", key=REGISTRY_KEY)
        checkpoint = str(Path(self.folder.name) / "external-anchor.json")
        create_source_checkpoint(
            self.db, checkpoint, registry_key=REGISTRY_KEY,
            anchor_key=ANCHOR_KEY,
        )
        result = self.ask(registry_checkpoint=checkpoint, registry_anchor_key=ANCHOR_KEY)
        self.assertFalse(result["accepted"])
        self.assertEqual(self.qa.calls, 0)
        # Simulate a privileged actor deleting the last revocation. The
        # shortened local HMAC chain may verify; the retained anchor rejects it.
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM consent_events WHERE seq=(SELECT MAX(seq) FROM consent_events)")
        result = self.ask(registry_checkpoint=checkpoint, registry_anchor_key=ANCHOR_KEY)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "registry_checkpoint_verification_failed")
        self.assertEqual(self.qa.calls, 0)

    def test_missing_checkpoint_key_denies_before_inference(self):
        self.grant()
        result = self.ask(registry_checkpoint="unavailable.json")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "missing_independent_checkpoint_key")
        self.assertEqual(self.qa.calls, 0)


if __name__ == "__main__":
    unittest.main()

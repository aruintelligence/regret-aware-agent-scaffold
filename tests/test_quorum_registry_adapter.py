"""Consent-scoped multi-source quorum tests, including mid-inference revocation."""
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from quorum_registry_adapter import evaluate_registered_quorum
from sovereign_source_registry import grant_source, revoke_source


KEY = b"independent-purpose-scoped-registry-secret-32"
OWNER = "opaque-research-owner"


class FakeQA:
    model_id = "fake-extractive-qa"
    revision = "fake-test-revision"

    def __init__(self):
        self.calls = 0
        self.context_mode = False
        self.on_call = None

    def answer(self, context, question, *, max_answer_tokens=30):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        text = ("Washington" if "Washington" in context else "Oregon") if self.context_mode else "Oregon"
        proof = ground_answer(text, context, model_revision=self.revision)
        return {
            "accepted": proof["accepted"],
            "answer": proof["answer"],
            "proof": proof if proof["accepted"] else None,
        }


class RegistryQuorumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "consent.db")
        self.sources = [
            {"source_id": "a", "context": "The telescope is in Oregon."},
            {"source_id": "b", "context": "A second document says Oregon."},
            {"source_id": "c", "context": "The telescope is in Washington."},
        ]
        self.grants = [
            grant_source(
                self.db, source["context"], owner=OWNER, purpose="qa_source",
                ttl_seconds=300, key=KEY, consent_asserted=True,
            )
            for source in self.sources
        ]
        self.engine = FakeQA()

    def tearDown(self):
        self.temp.cleanup()

    def run_quorum(self, *, sources=None, owner=OWNER):
        return evaluate_registered_quorum(
            self.engine, registry_db=self.db, registry_key=KEY,
            owner=owner, question="Where was the telescope?",
            sources=sources or self.sources[:2],
            min_distinct_sources=2,
        )

    def test_authorized_agreement_produces_recomputable_receipt(self):
        receipt = self.run_quorum()
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["answer"], "Oregon")
        self.assertEqual(receipt["source_approval"], "purpose_scoped_registry")
        self.assertEqual(self.engine.calls, 2)
        self.assertFalse(receipt["automatic_rollout_permitted"])

    def test_wrong_owner_denied_before_model_calls(self):
        with self.assertRaises(PermissionError):
            self.run_quorum(owner="other-owner")
        self.assertEqual(self.engine.calls, 0)

    def test_conflicting_verified_sources_abstain(self):
        self.engine.context_mode = True
        receipt = self.run_quorum(sources=self.sources)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "conflicting_verified_source_answers")
        self.assertEqual(receipt["answer"], "")

    def test_revocation_before_inference_blocks_all_calls(self):
        revoke_source(self.db, self.grants[1]["grant_id"], owner=OWNER, key=KEY)
        with self.assertRaises(PermissionError):
            self.run_quorum()
        self.assertEqual(self.engine.calls, 0)

    def test_revocation_during_inference_blocks_receipt(self):
        def revoke_on_first_call(number):
            if number == 1:
                revoke_source(
                    self.db, self.grants[1]["grant_id"], owner=OWNER, key=KEY,
                )
        self.engine.on_call = revoke_on_first_call
        with self.assertRaises(PermissionError):
            self.run_quorum()
        self.assertEqual(self.engine.calls, 2)

    def test_duplicate_ids_and_unconfigured_registry_fail_closed(self):
        with self.assertRaises(ValueError):
            self.run_quorum(sources=[self.sources[0], self.sources[0]])
        self.assertEqual(self.engine.calls, 0)
        with self.assertRaises(PermissionError):
            evaluate_registered_quorum(
                self.engine, registry_db=None, registry_key=None,
                owner=OWNER, question="Where?", sources=self.sources[:2],
            )


if __name__ == "__main__":
    unittest.main()

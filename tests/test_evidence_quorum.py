"""Adversarial evidence-quorum tests; no model weights or network required."""
import copy
import hashlib
import unittest

from evidence_quorum import evidence_quorum, verify_quorum


class EvidenceQuorumTests(unittest.TestCase):
    def test_distinct_sources_agree_and_receipt_reverifies(self):
        candidates = [
            {"source_id": "doc1", "context": "The telescope is in Oregon.", "answer": "Oregon"},
            {"source_id": "doc2", "context": "The observatory was in OREGON in 1999.", "answer": "oregon"},
        ]
        receipt = evidence_quorum(candidates)
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["answer"], "Oregon")
        self.assertEqual(receipt["winning_distinct_source_fingerprints"], 2)
        self.assertTrue(verify_quorum(receipt, candidates))
        self.assertNotIn("telescope is", str(receipt))

    def test_identical_source_cannot_count_twice(self):
        candidates = [
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Oregon"},
            {"source_id": "b", "context": "The telescope is in Oregon.", "answer": "Oregon"},
        ]
        receipt = evidence_quorum(candidates)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["distinct_verified_source_fingerprints"], 1)

    def test_conflict_blocks_even_if_two_other_sources_agree(self):
        candidates = [
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Oregon"},
            {"source_id": "b", "context": "Another report places it in Oregon.", "answer": "Oregon"},
            {"source_id": "c", "context": "The telescope is in Washington.", "answer": "Washington"},
        ]
        receipt = evidence_quorum(candidates)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["answer"], "")
        self.assertEqual(receipt["reason"], "conflicting_verified_source_answers")

    def test_unsupported_answer_abstains(self):
        receipt = evidence_quorum([
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Washington"},
        ], min_distinct_sources=1)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["unsupported_candidate_count"], 1)

    def test_source_and_receipt_tampering_detected(self):
        candidates = [
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Oregon"},
        ]
        receipt = evidence_quorum(candidates, min_distinct_sources=1)
        altered = copy.deepcopy(receipt)
        altered["answer"] = "Washington"
        self.assertFalse(verify_quorum(altered, candidates))
        candidates[0]["context"] = "The telescope is in Washington."
        self.assertFalse(verify_quorum(receipt, candidates))

    def test_operator_source_allowlist(self):
        candidates = [
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Oregon"},
        ]
        digest = hashlib.sha256(candidates[0]["context"].encode("utf-8")).hexdigest()
        receipt = evidence_quorum(
            candidates, min_distinct_sources=1, approved_source_sha256={digest},
        )
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["source_approval"], "operator_allowlist")
        self.assertTrue(verify_quorum(receipt, candidates, approved_source_sha256={digest}))
        self.assertFalse(verify_quorum(receipt, candidates))
        with self.assertRaises(ValueError):
            evidence_quorum(
                candidates, min_distinct_sources=1,
                approved_source_sha256={"0" * 64},
            )

    def test_source_id_reuse_with_different_text_rejected(self):
        candidates = [
            {"source_id": "a", "context": "The telescope is in Oregon.", "answer": "Oregon"},
            {"source_id": "a", "context": "Another report says Oregon.", "answer": "Oregon"},
        ]
        with self.assertRaises(ValueError):
            evidence_quorum(candidates)

    def test_schema_limits(self):
        with self.assertRaises(ValueError):
            evidence_quorum([])
        with self.assertRaises(ValueError):
            evidence_quorum([{"context": "a", "answer": "a"}])
        with self.assertRaises(ValueError):
            evidence_quorum([
                {"source_id": "a", "context": "a", "answer": "a"},
            ], min_distinct_sources=0)

    def test_wrong_source_proof_not_promoted_to_truth(self):
        # Even if a source itself contains false claims, quorum only proves
        # textual agreement. It cannot establish independent factual truth.
        candidates = [
            {"source_id": "one", "context": "The Moon is made of cheese.", "answer": "cheese"},
            {"source_id": "two", "context": "This fictional tale says the Moon is cheese.", "answer": "cheese"},
        ]
        receipt = evidence_quorum(candidates)
        self.assertTrue(receipt["accepted"])
        self.assertIn("not truth", receipt["warning"])


if __name__ == "__main__":
    unittest.main()

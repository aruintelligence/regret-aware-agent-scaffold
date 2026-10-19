import copy
import unittest

from grounding_contract import ground_answer, verify_grounding


class GroundingContractTests(unittest.TestCase):
    def test_exact_span_certificate_and_verification(self):
        source = "The telescope was in Oregon in 1999."
        proof = ground_answer("Oregon", source, model_revision="fixed-revision")
        self.assertTrue(proof["accepted"])
        self.assertEqual(proof["answer"], "Oregon")
        self.assertEqual(source[proof["span_start"]:proof["span_end"]], "Oregon")
        self.assertTrue(verify_grounding(proof, source))
        self.assertEqual(proof["model_revision"], "fixed-revision")
        self.assertNotIn(source, str(proof))

    def test_case_insensitive_recovery_preserves_source_spelling(self):
        source = "The observatory is in Oregon."
        proof = ground_answer("Answer: oregon.", source)
        self.assertTrue(proof["accepted"])
        self.assertEqual(proof["answer"], "Oregon")
        self.assertEqual(proof["reason"], "case_insensitive_source_span")
        self.assertTrue(verify_grounding(proof, source))

    def test_hallucinated_answer_abstains(self):
        proof = ground_answer("Washington", "The observatory is in Oregon.")
        self.assertFalse(proof["accepted"])
        self.assertEqual(proof["answer"], "")
        self.assertFalse(verify_grounding(proof, "The observatory is in Oregon."))

    def test_subword_match_is_not_evidence(self):
        proof = ground_answer("on", "London is a city.")
        self.assertFalse(proof["accepted"])
        self.assertFalse(verify_grounding(proof, "London is a city."))

    def test_quote_wrappers_can_be_removed_but_multiline_refused(self):
        source = "The team found a comet in 1999."
        self.assertTrue(ground_answer('"1999."', source)["accepted"])
        self.assertFalse(ground_answer("1999\nI think so", source)["accepted"])

    def test_repeated_span_uses_first_valid_source_occurrence(self):
        source = "Oregon is west. Oregon is beautiful."
        proof = ground_answer("Oregon", source)
        self.assertEqual(proof["span_start"], 0)
        self.assertTrue(verify_grounding(proof, source))

    def test_unicode_source_character_offsets(self):
        source = "The team studied São Paulo in 2012."
        proof = ground_answer("São Paulo", source)
        self.assertTrue(verify_grounding(proof, source))
        self.assertEqual(source[proof["span_start"]:proof["span_end"]], "São Paulo")

    def test_certificate_detects_source_and_offset_tampering(self):
        source = "The observatory is in Oregon."
        proof = ground_answer("Oregon", source)
        self.assertFalse(verify_grounding(proof, "The observatory is in Oregon!"))
        modified = copy.deepcopy(proof)
        modified["span_start"] += 1
        self.assertFalse(verify_grounding(modified, source))
        modified = copy.deepcopy(proof)
        modified["answer"] = "Washington"
        self.assertFalse(verify_grounding(modified, source))
        modified = copy.deepcopy(proof)
        modified["source_sha256"] = "0" * 64
        self.assertFalse(verify_grounding(modified, source))

    def test_refuses_unstructured_oversized_and_empty_answers(self):
        source = "The telescope was in Oregon."
        for answer in ("", "   ", "Oregon\nWashington", "A" * 201):
            self.assertFalse(ground_answer(answer, source)["accepted"])
        with self.assertRaises(ValueError):
            ground_answer("Oregon", "")
        with self.assertRaises(ValueError):
            ground_answer("Oregon", "A" * 25001)


if __name__ == "__main__":
    unittest.main()

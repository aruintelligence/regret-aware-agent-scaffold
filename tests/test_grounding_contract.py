import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_local_cli_certifies_and_verifies_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            context = root / "context.txt"
            answer = root / "answer.txt"
            proof = root / "proof.json"
            context.write_text("The observatory is in Oregon.", encoding="utf-8")
            answer.write_text("Oregon", encoding="utf-8")
            script = str(Path(__file__).resolve().parents[1] / "grounding_contract.py")
            result = subprocess.run(
                [sys.executable, script, "certify", "--context-file", str(context),
                 "--answer-file", str(answer), "--out", str(proof)],
                check=True, capture_output=True, text=True,
            )
            self.assertTrue(json.loads(result.stdout)["accepted"])
            self.assertTrue(proof.is_file())
            check = subprocess.run(
                [sys.executable, script, "verify", "--context-file", str(context),
                 "--proof-file", str(proof)],
                capture_output=True, text=True,
            )
            self.assertEqual(check.returncode, 0)
            self.assertTrue(json.loads(check.stdout)["verified"])
            context.write_text("The observatory is in Washington.", encoding="utf-8")
            tampered = subprocess.run(
                [sys.executable, script, "verify", "--context-file", str(context),
                 "--proof-file", str(proof)],
                capture_output=True, text=True,
            )
            self.assertEqual(tampered.returncode, 1)
            self.assertFalse(json.loads(tampered.stdout)["verified"])

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

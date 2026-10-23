"""Adversarial public-key grader verification and secret-free gate checks."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from grade_attestation_ledger import attest_grade
from grounding_contract import ground_answer
from public_grade_ledger import (
    audit_public_grades, close_public_grades, gate_with_public_grades,
    public_checkpoint_fingerprint, public_key_fingerprint, sign_public_grade,
    verify_public_checkpoint, verify_public_grade, verify_public_grade_log,
)
from receipt_cohort_ledger import gate_with_receipt_ledger
from tests.test_receipt_cohort_ledger import RECEIPT_KEY, receipt_fixture
from tests.test_watch_outcome_ledger import INGRESS_KEY, POLICY_KEY, WATCH_KEY, policy
from watch_outcome_ledger import reconcile_issued_grades

PRIVATE = bytes(range(1, 33))
PUBLIC = Ed25519PrivateKey.from_private_bytes(PRIVATE).public_key().public_bytes_raw()
ATTACKER_PRIVATE = bytes(range(33, 65))
ATTACKER_PUBLIC = Ed25519PrivateKey.from_private_bytes(
    ATTACKER_PRIVATE).public_key().public_bytes_raw()


def signed_fixture(n=40, error_index=17):
    issued, grades, ingress, receipts, receipt_checkpoint = receipt_fixture(n)
    grades[error_index] = {**grades[error_index], "error": True}
    attested = [
        sign_public_grade(grade, grade_sequence=sequence,
                          cohort_id=ingress["cohort_id"],
                          model_revision=ingress["model_revision"],
                          grader_private_key=PRIVATE)
        for sequence, grade in enumerate(reversed(grades), 1)
    ]
    checkpoint = close_public_grades(
        attested, cohort_id=ingress["cohort_id"],
        model_revision=ingress["model_revision"],
        rubric_id=ingress["rubric_id"], grader_private_key=PRIVATE,
    )
    return issued, grades, ingress, receipts, receipt_checkpoint, attested, checkpoint


def audit(issued, ingress, attested, checkpoint, **changes):
    options = {
        "grade_checkpoint": checkpoint, "ingress_checkpoint": ingress,
        "grader_public_key": PUBLIC,
        "expected_grader_public_key_sha256": public_key_fingerprint(PUBLIC),
        "expected_current_grade_checkpoint_sha256": public_checkpoint_fingerprint(checkpoint),
        "ingress_key": INGRESS_KEY,
        "expected_current_ingress_hmac_sha256": ingress["ingress_hmac_sha256"],
    }
    return audit_public_grades(attested, issued, **{**options, **changes})


class PublicGradeLedgerTests(unittest.TestCase):
    def test_verifier_has_public_key_only_and_rejects_key_substitution(self):
        issued, grades, ingress, _, _, attested, checkpoint = signed_fixture()
        self.assertTrue(verify_public_grade(attested[0], grader_public_key=PUBLIC))
        self.assertTrue(verify_public_checkpoint(checkpoint, grader_public_key=PUBLIC))
        self.assertEqual(audit(issued, ingress, attested, checkpoint)["status"], "complete")
        self.assertEqual(verify_public_grade_log(
            attested, grade_checkpoint=checkpoint, grader_public_key=PUBLIC,
            expected_grader_public_key_sha256=public_key_fingerprint(PUBLIC),
            expected_current_grade_checkpoint_sha256=public_checkpoint_fingerprint(
                checkpoint))["grade_count"], 40)
        self.assertFalse(verify_public_grade(attested[0], grader_public_key=ATTACKER_PUBLIC))
        self.assertFalse(verify_public_checkpoint(
            checkpoint, grader_public_key=ATTACKER_PUBLIC))
        self.assertFalse(verify_public_grade(
            {**attested[0], "error": not attested[0]["error"]}, grader_public_key=PUBLIC))
        self.assertFalse(verify_public_grade(
            {**attested[0], "rubric_id": "different"}, grader_public_key=PUBLIC))
        self.assertFalse(verify_public_checkpoint(
            {**checkpoint, "grade_count": 2}, grader_public_key=PUBLIC))
        legacy = attest_grade(grades[0], grade_sequence=1,
                              cohort_id=ingress["cohort_id"],
                              model_revision=ingress["model_revision"],
                              grader_key=b"a-legacy-hmac-key-longer-than-32-bytes")
        self.assertFalse(verify_public_grade(legacy, grader_public_key=PUBLIC))
        attacker_log = [
            sign_public_grade(
                {**grade, "error": False}, grade_sequence=sequence,
                cohort_id=ingress["cohort_id"], model_revision=ingress["model_revision"],
                grader_private_key=ATTACKER_PRIVATE)
            for sequence, grade in enumerate(reversed(grades), 1)
        ]
        attacker_checkpoint = close_public_grades(
            attacker_log, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"],
            rubric_id=ingress["rubric_id"], grader_private_key=ATTACKER_PRIVATE,
        )
        with self.assertRaisesRegex(ValueError, "not independently pinned"):
            audit(issued, ingress, attacker_log, attacker_checkpoint,
                  grader_public_key=ATTACKER_PUBLIC,
                  expected_current_grade_checkpoint_sha256=public_checkpoint_fingerprint(
                      attacker_checkpoint))
        with self.assertRaisesRegex(ValueError, "stale"):
            audit(issued, ingress, attested, checkpoint,
                  expected_current_grade_checkpoint_sha256="0" * 64)
        with self.assertRaises(ValueError):
            audit(issued, ingress, attested[:-1], checkpoint)
        with self.assertRaises(ValueError):
            sign_public_grade({**grades[0], "error": 0}, grade_sequence=1,
                              cohort_id=ingress["cohort_id"],
                              model_revision=ingress["model_revision"],
                              grader_private_key=PRIVATE)

    def test_monitor_hides_real_error_with_valid_watch_signature_but_gate_denies(self):
        issued, grades, ingress, receipts, receipt_checkpoint, attested, checkpoint = signed_fixture()
        source = "The telescope was in Oregon."
        hidden = [dict(row, error=False) for row in grades]

        def gate_args(labels):
            watch = reconcile_issued_grades(issued, labels, checkpoint=ingress,
                ingress_key=INGRESS_KEY, watch_key=WATCH_KEY)["report"]
            return {
                "policy_key": POLICY_KEY, "watch_report": watch,
                "watch_key": WATCH_KEY, "ingress_checkpoint": ingress,
                "ingress_key": INGRESS_KEY,
                "expected_current_checkpoint_hmac_sha256": ingress["ingress_hmac_sha256"],
                "expected_outcomes_sha256": watch["ordered_outcomes_sha256"],
                "score": 0.9, "group": "general",
                "source_proof": ground_answer(
                    "Oregon", source, model_revision="fixed-qa-revision"),
                "source_text": source, "model_revision": "fixed-qa-revision",
                "rubric_id": "rubric-v1",
            }

        held = {
            "receipts": receipts, "issued_rows": issued,
            "receipt_checkpoint": receipt_checkpoint, "receipt_key": RECEIPT_KEY,
            "expected_current_receipt_hmac_sha256":
                receipt_checkpoint["receipt_ledger_hmac_sha256"],
        }
        grader = {
            "attestations": attested, "grade_checkpoint": checkpoint,
            "grader_public_key": PUBLIC,
            "expected_grader_public_key_sha256": public_key_fingerprint(PUBLIC),
            "expected_current_grade_checkpoint_sha256":
                public_checkpoint_fingerprint(checkpoint),
        }
        self.assertTrue(gate_with_public_grades(
            policy(), **grader, **held, **gate_args(grades))["accepted"])
        self.assertTrue(gate_with_receipt_ledger(
            policy(), **held, **gate_args(hidden))["accepted"])
        result = gate_with_public_grades(
            policy(), **grader, **held, **gate_args(hidden))
        self.assertEqual(result["reason"], "signed_watch_disagrees_with_public_grades")
        self.assertFalse(result["automatic_rollout_permitted"])
        wrong_key = gate_with_public_grades(
            policy(), **{**grader, "grader_public_key": ATTACKER_PUBLIC},
            **held, **gate_args(hidden))
        self.assertEqual(wrong_key["reason"], "invalid_or_stale_public_grader_ledger")
        reused = gate_with_public_grades(
            policy(), **grader, **{**held, "receipt_key": PUBLIC},
            **gate_args(grades))
        self.assertEqual(reused["reason"], "public_key_reused_as_signing_secret")

    def test_missing_foreign_replay_and_sequence_fail_closed(self):
        issued, grades, ingress, _, _, attested, checkpoint = signed_fixture(8, 3)
        partial = attested[:-1]
        partial_checkpoint = close_public_grades(
            partial, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"],
            rubric_id=ingress["rubric_id"], grader_private_key=PRIVATE)
        self.assertEqual(audit(
            issued, ingress, partial, partial_checkpoint,
            expected_current_grade_checkpoint_sha256=public_checkpoint_fingerprint(
                partial_checkpoint))["missing_count"], 1)
        foreign = [*attested, sign_public_grade(
            {**grades[0], "request_id": "unissued"}, grade_sequence=9,
            cohort_id=ingress["cohort_id"], model_revision=ingress["model_revision"],
            grader_private_key=PRIVATE)]
        foreign_checkpoint = close_public_grades(
            foreign, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"],
            rubric_id=ingress["rubric_id"], grader_private_key=PRIVATE)
        self.assertEqual(audit(
            issued, ingress, foreign, foreign_checkpoint,
            expected_current_grade_checkpoint_sha256=public_checkpoint_fingerprint(
                foreign_checkpoint))["foreign_count"], 1)
        with self.assertRaises(ValueError):
            close_public_grades(attested[::-1], cohort_id=ingress["cohort_id"],
                model_revision=ingress["model_revision"],
                rubric_id=ingress["rubric_id"], grader_private_key=PRIVATE)
        with self.assertRaises(ValueError):
            close_public_grades(attested[:-1] + attested[:1],
                cohort_id=ingress["cohort_id"], model_revision=ingress["model_revision"],
                rubric_id=ingress["rubric_id"], grader_private_key=PRIVATE)

    def test_cli_reconcile_has_no_grader_secret_and_rejects_stale_pin(self):
        issued, grades, ingress, _, _, attested, checkpoint = signed_fixture(3, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = str(Path(__file__).resolve().parents[1] / "public_grade_cli.py")
            private_file, public_file = root / "private.json", root / "public.json"
            env = {**os.environ, "ARU_ISSUED_SIGNING_KEY": INGRESS_KEY.decode(),
                   "ARU_WATCH_SIGNING_KEY": WATCH_KEY.decode()}
            env.pop("ARU_GRADER_SIGNING_KEY", None)

            def run(*args):
                return subprocess.run([sys.executable, cli, *args], cwd=root,
                                      env=env, capture_output=True, text=True, check=False)

            keygen = run("keygen", "--private-out", str(private_file),
                         "--public-out", str(public_file))
            self.assertEqual(keygen.returncode, 0, keygen.stderr)
            self.assertEqual(private_file.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(
                json.loads(private_file.read_text())["private_key_hex"], keygen.stdout)
            self.assertEqual(run("keygen", "--private-out", str(private_file),
                                 "--public-out", str(public_file)).returncode, 4)
            self.assertEqual(run("keygen", "--private-out", str(root / "extra-private.json"),
                                 "--public-out", str(public_file)).returncode, 4)
            self.assertFalse((root / "extra-private.json").exists())
            grade_path = root / "grade.json"
            signed_path = root / "signed.json"
            grade_path.write_text(json.dumps(grades[0]))
            signing = run("grade", "--private-key", str(private_file),
                          "--grade-json", str(grade_path), "--grade-sequence", "1",
                          "--cohort-id", ingress["cohort_id"],
                          "--model-revision", ingress["model_revision"],
                          "--out", str(signed_path))
            self.assertEqual(signing.returncode, 0, signing.stderr)
            public_bytes = bytes.fromhex(json.loads(public_file.read_text())["public_key_hex"])
            self.assertTrue(verify_public_grade(
                json.loads(signed_path.read_text()), grader_public_key=public_bytes))
            private_bytes = bytes.fromhex(
                json.loads(private_file.read_text())["private_key_hex"])
            local_log = [
                sign_public_grade(grade, grade_sequence=seq,
                    cohort_id=ingress["cohort_id"],
                    model_revision=ingress["model_revision"],
                    grader_private_key=private_bytes)
                for seq, grade in enumerate(reversed(grades), 1)
            ]
            log_path = root / "attestations.jsonl"
            issued_path = root / "issued.jsonl"
            ingress_path = root / "ingress.json"
            checkpoint_path = root / "checkpoint.json"
            watch_path = root / "watch.json"
            log_path.write_text("".join(json.dumps(row) + "\n" for row in local_log))
            issued_path.write_text("".join(json.dumps(row) + "\n" for row in issued))
            ingress_path.write_text(json.dumps(ingress))
            close = run("close", "--private-key", str(private_file),
                        "--attestations-jsonl", str(log_path),
                        "--cohort-id", ingress["cohort_id"],
                        "--model-revision", ingress["model_revision"],
                        "--rubric-id", ingress["rubric_id"],
                        "--out", str(checkpoint_path))
            self.assertEqual(close.returncode, 0, close.stderr)
            checkpoint = json.loads(checkpoint_path.read_text())
            private_file.unlink()  # verifier cannot read the grader signing key
            public_options = ("--public-key", str(public_file),
                              "--expected-public-key-sha256",
                              public_key_fingerprint(public_bytes),
                              "--expected-current-grade-sha256",
                              public_checkpoint_fingerprint(checkpoint),
                              "--attestations-jsonl", str(log_path),
                              "--grade-checkpoint", str(checkpoint_path))
            clean_env = {key: value for key, value in os.environ.items()
                         if not key.startswith("ARU_")}
            independent = subprocess.run(
                [sys.executable, cli, "verify", *public_options],
                cwd=root, env=clean_env, capture_output=True, text=True, check=False)
            self.assertEqual(independent.returncode, 0, independent.stderr)
            self.assertEqual(json.loads(independent.stdout)["status"],
                             "valid_closed_grader_log")
            rejected = subprocess.run(
                [sys.executable, cli, "verify", *public_options[:3],
                 "0" * 64, *public_options[4:]],
                cwd=root, env=clean_env, capture_output=True, text=True, check=False)
            self.assertEqual(rejected.returncode, 4)

            def cmd(pin):
                return ("reconcile", "--public-key", str(public_file),
                        "--expected-public-key-sha256", pin,
                        "--expected-current-grade-sha256",
                        public_checkpoint_fingerprint(checkpoint),
                        "--expected-current-ingress-hmac", ingress["ingress_hmac_sha256"],
                        "--attestations-jsonl", str(log_path),
                        "--issued-jsonl", str(issued_path),
                        "--grade-checkpoint", str(checkpoint_path),
                        "--ingress-checkpoint", str(ingress_path),
                        "--out", str(watch_path))

            self.assertEqual(run(*cmd("0" * 64)).returncode, 4)
            self.assertFalse(watch_path.exists())
            verified = run(*cmd(public_key_fingerprint(public_bytes)))
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertTrue(watch_path.exists())
            self.assertEqual(watch_path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()

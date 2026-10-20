"""Falsify self-fed grader pins, single-witness claims and checkpoint rollback."""
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from grader_witness_quorum import (
    GENESIS, gate_with_witnessed_public_grades, initialize_witness_state,
    make_witness_statement, sign_witness_statement, statement_fingerprint,
    verify_witness_quorum,
)
from grounding_contract import ground_answer
from public_grade_ledger import (
    close_public_grades, gate_with_public_grades,
    public_checkpoint_fingerprint, public_key_fingerprint, sign_public_grade,
)
from tests.test_public_grade_ledger import (
    ATTACKER_PRIVATE, ATTACKER_PUBLIC, PUBLIC, signed_fixture,
)
from tests.test_receipt_cohort_ledger import RECEIPT_KEY
from tests.test_watch_outcome_ledger import INGRESS_KEY, POLICY_KEY, WATCH_KEY, policy
from watch_outcome_ledger import reconcile_issued_grades

WITNESS_A = bytes(range(66, 98))
WITNESS_B = bytes(range(99, 131))
PUBLIC_A = Ed25519PrivateKey.from_private_bytes(WITNESS_A).public_key().public_bytes_raw()
PUBLIC_B = Ed25519PrivateKey.from_private_bytes(WITNESS_B).public_key().public_bytes_raw()
PINNED = {"review-a": PUBLIC_A, "review-b": PUBLIC_B}


def fixture():
    issued, grades, ingress, receipts, receipt_checkpoint, attested, checkpoint = (
        signed_fixture()
    )
    source = "The telescope was in Oregon."
    watch = reconcile_issued_grades(
        issued, grades, checkpoint=ingress, ingress_key=INGRESS_KEY,
        watch_key=WATCH_KEY,
    )["report"]
    statement = make_witness_statement(
        cohort_id=ingress["cohort_id"], witness_epoch=1,
        previous_statement_sha256=GENESIS,
        grader_public_key_sha256=public_key_fingerprint(PUBLIC),
        grade_checkpoint_sha256=public_checkpoint_fingerprint(checkpoint),
        ingress_hmac_sha256=ingress["ingress_hmac_sha256"],
        receipt_hmac_sha256=receipt_checkpoint["receipt_ledger_hmac_sha256"],
        ordered_outcomes_sha256=watch["ordered_outcomes_sha256"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
    )
    approvals = [
        sign_witness_statement(statement, witness_id="review-a",
                               witness_private_key=WITNESS_A),
        sign_witness_statement(statement, witness_id="review-b",
                               witness_private_key=WITNESS_B),
    ]
    held = {
        "attestations": attested, "grade_checkpoint": checkpoint,
        "grader_public_key": PUBLIC, "receipts": receipts,
        "issued_rows": issued, "receipt_checkpoint": receipt_checkpoint,
        "receipt_key": RECEIPT_KEY,
    }
    decision = {
        "policy_key": POLICY_KEY, "watch_report": watch,
        "watch_key": WATCH_KEY, "ingress_checkpoint": ingress,
        "ingress_key": INGRESS_KEY, "score": 0.9, "group": "general",
        "source_proof": ground_answer(
            "Oregon", source, model_revision="fixed-qa-revision"),
        "source_text": source, "model_revision": "fixed-qa-revision",
        "rubric_id": "rubric-v1",
    }
    return issued, grades, ingress, statement, approvals, held, decision


class WitnessQuorumTests(unittest.TestCase):
    def test_two_pinned_witnesses_accept_and_one_or_mixed_signature_fails(self):
        issued, grades, ingress, statement, approvals, held, decision = fixture()
        verified = verify_witness_quorum(
            statement, approvals, trusted_witness_public_keys=PINNED)
        self.assertEqual(verified["witness_count"], 2)
        with self.assertRaises(ValueError):
            verify_witness_quorum(
                statement, approvals[:1], trusted_witness_public_keys=PINNED)
        with self.assertRaises(ValueError):
            verify_witness_quorum(
                statement, [approvals[0], approvals[0]],
                trusted_witness_public_keys=PINNED)
        with self.assertRaises(ValueError):
            verify_witness_quorum(
                statement, approvals,
                trusted_witness_public_keys={"review-a": PUBLIC_A,
                                             "review-b": PUBLIC_A})
        changed_statement = {**statement, "grade_checkpoint_sha256": "f" * 64}
        with self.assertRaises(ValueError):
            verify_witness_quorum(
                changed_statement, approvals, trusted_witness_public_keys=PINNED)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "witness.sqlite3"
            self.assertEqual(gate_with_witnessed_public_grades(
                policy(), statement=statement, witness_approvals=approvals,
                trusted_witness_public_keys=PINNED,
                witness_state_path=state_path, **held, **decision)["reason"],
                "invalid_or_stale_witness_quorum")
            initialize_witness_state(state_path)
            self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)
            params = {
                "statement": statement, "witness_approvals": approvals,
                "trusted_witness_public_keys": PINNED,
                "witness_state_path": state_path, **held, **decision,
            }
            self.assertTrue(gate_with_witnessed_public_grades(
                policy(), **params)["accepted"])
            self.assertTrue(gate_with_witnessed_public_grades(
                policy(), **params)["accepted"])  # idempotent same epoch
            self.assertEqual(gate_with_witnessed_public_grades(
                policy(), **{**params, "witness_approvals": approvals[:1]})["reason"],
                "invalid_or_stale_witness_quorum")
            self.assertEqual(gate_with_witnessed_public_grades(
                policy(), **{**params,
                             "expected_current_grade_checkpoint_sha256":
                                 statement["grade_checkpoint_sha256"]})["reason"],
                "caller_supplied_anchor_not_permitted")
            self.assertEqual(gate_with_witnessed_public_grades(
                policy(), **{**params,
                             "trusted_witness_public_keys":
                                 {"review-a": PUBLIC, "review-b": PUBLIC_B}})["reason"],
                "grader_key_reused_as_witness")

    def test_self_fed_attacker_key_passes_unwitnessed_gate_but_quorum_blocks_it(self):
        issued, grades, ingress, statement, approvals, held, decision = fixture()
        false_grades = [{**grade, "error": False} for grade in grades]
        forged_watch = reconcile_issued_grades(
            issued, false_grades, checkpoint=ingress,
            ingress_key=INGRESS_KEY, watch_key=WATCH_KEY)["report"]
        attacker_log = [
            sign_public_grade(grade, grade_sequence=seq,
                cohort_id=ingress["cohort_id"],
                model_revision=ingress["model_revision"],
                grader_private_key=ATTACKER_PRIVATE)
            for seq, grade in enumerate(reversed(false_grades), 1)
        ]
        attacker_checkpoint = close_public_grades(
            attacker_log, cohort_id=ingress["cohort_id"],
            model_revision=ingress["model_revision"],
            rubric_id=ingress["rubric_id"], grader_private_key=ATTACKER_PRIVATE)
        attacker_held = {
            **held, "attestations": attacker_log,
            "grade_checkpoint": attacker_checkpoint,
            "grader_public_key": ATTACKER_PUBLIC,
        }
        forged_decision = {**decision, "watch_report": forged_watch}
        # The plain public-key gate is safe only if its caller really pins
        # the key and current digest independently; copied attacker values pass.
        self.assertTrue(gate_with_public_grades(
            policy(), **attacker_held, **forged_decision,
            expected_grader_public_key_sha256=public_key_fingerprint(ATTACKER_PUBLIC),
            expected_current_grade_checkpoint_sha256=
                public_checkpoint_fingerprint(attacker_checkpoint),
            expected_current_receipt_hmac_sha256=
                held["receipt_checkpoint"]["receipt_ledger_hmac_sha256"],
            expected_current_checkpoint_hmac_sha256=ingress["ingress_hmac_sha256"],
            expected_outcomes_sha256=forged_watch["ordered_outcomes_sha256"])["accepted"])
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.sqlite3"
            initialize_witness_state(state_path)
            denied = gate_with_witnessed_public_grades(
                policy(), statement=statement, witness_approvals=approvals,
                trusted_witness_public_keys=PINNED,
                witness_state_path=state_path,
                **attacker_held, **forged_decision)
            self.assertFalse(denied["accepted"])
            self.assertEqual(denied["reason"], "invalid_or_stale_public_grader_ledger")

    def test_monotonic_state_rejects_old_epoch_fork_and_skipped_successor(self):
        issued, grades, ingress, first, approvals, held, decision = fixture()
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "witness.sqlite3"
            initialize_witness_state(state_path)

            def gate(statement, approvals):
                return gate_with_witnessed_public_grades(
                    policy(), statement=statement, witness_approvals=approvals,
                    trusted_witness_public_keys=PINNED, witness_state_path=state_path,
                    **held, **decision)

            self.assertTrue(gate(first, approvals)["accepted"])
            second = make_witness_statement(
                cohort_id=first["cohort_id"], witness_epoch=2,
                previous_statement_sha256=statement_fingerprint(first),
                grader_public_key_sha256=first["grader_public_key_sha256"],
                grade_checkpoint_sha256=first["grade_checkpoint_sha256"],
                ingress_hmac_sha256=first["ingress_hmac_sha256"],
                receipt_hmac_sha256=first["receipt_hmac_sha256"],
                ordered_outcomes_sha256=first["ordered_outcomes_sha256"],
                model_revision=first["model_revision"], rubric_id=first["rubric_id"])
            second_approvals = [
                sign_witness_statement(second, witness_id="review-a",
                                       witness_private_key=WITNESS_A),
                sign_witness_statement(second, witness_id="review-b",
                                       witness_private_key=WITNESS_B),
            ]
            self.assertTrue(gate(second, second_approvals)["accepted"])
            self.assertEqual(gate(first, approvals)["reason"],
                             "invalid_or_stale_witness_quorum")
            fork = {**second, "ordered_outcomes_sha256": "f" * 64}
            fork_approvals = [
                sign_witness_statement(fork, witness_id="review-a",
                                       witness_private_key=WITNESS_A),
                sign_witness_statement(fork, witness_id="review-b",
                                       witness_private_key=WITNESS_B),
            ]
            self.assertEqual(gate(fork, fork_approvals)["reason"],
                             "invalid_or_stale_witness_quorum")
            skipped = {**second, "witness_epoch": 4}
            skipped_approvals = [
                sign_witness_statement(skipped, witness_id="review-a",
                                       witness_private_key=WITNESS_A),
                sign_witness_statement(skipped, witness_id="review-b",
                                       witness_private_key=WITNESS_B),
            ]
            self.assertEqual(gate(skipped, skipped_approvals)["reason"],
                             "invalid_or_stale_witness_quorum")


if __name__ == "__main__":
    unittest.main()

"""Public synthetic inputs for the offline witnessed gate demo.

Every key here is intentionally published. This fixture makes no model calls
and cannot establish that independent parties operated the witnesses.
"""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from client_request_receipt import issue_request_receipt
from grader_witness_quorum import GENESIS, make_witness_statement
from grounding_contract import ground_answer
from public_grade_ledger import (
    close_public_grades, public_checkpoint_fingerprint,
    public_key_fingerprint, sign_public_grade,
)
from receipt_cohort_ledger import create_receipt_checkpoint
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy
from watch_outcome_ledger import create_issued_checkpoint, reconcile_issued_grades

INGRESS_KEY = b"research-ingress-test-secret-32-bytes"
WATCH_KEY = b"independent-watch-test-secret-32-bytes"
POLICY_KEY = b"independent-static-risk-key-32-bytes"
RECEIPT_KEY = b"distinct-receipt-cohort-test-secret-32-bytes"
GRADER_PRIVATE = bytes(range(1, 33))
ATTACKER_PRIVATE = bytes(range(33, 65))
WITNESS_A = bytes(range(66, 98))
WITNESS_B = bytes(range(99, 131))


def _public(private):
    return Ed25519PrivateKey.from_private_bytes(private).public_key().public_bytes_raw()


PUBLIC = _public(GRADER_PRIVATE)
ATTACKER_PUBLIC = _public(ATTACKER_PRIVATE)
PUBLIC_A = _public(WITNESS_A)
PUBLIC_B = _public(WITNESS_B)
PINNED = {"review-a": PUBLIC_A, "review-b": PUBLIC_B}


def policy():
    rows = [
        {"sample_id": str(i), "score": 0.9 if i < 150 else 0.1,
         "correct": i < 150, "group": "general"}
        for i in range(200)
    ]
    fitted = fit_selective_risk_policy(
        rows, thresholds=[0.05, 0.5, 0.95], groups=["general"],
        target_error=0.1, alpha=0.05, min_group_support=30,
        model_revision="fixed-qa-revision", rubric_id="rubric-v1",
        calibration_set_id="frozen-calibration",
    )
    return sign_risk_policy(fitted, key=POLICY_KEY)


def fixture():
    issued = [
        {"sequence": i + 1, "request_id": f"request-{i + 1}",
         "group": "general", "model_revision": "fixed-qa-revision",
         "rubric_id": "rubric-v1"}
        for i in range(40)
    ]
    grades = [
        {"request_id": row["request_id"], "error": i == 17,
         "rubric_id": "rubric-v1", "grader_id": "reviewer-1"}
        for i, row in enumerate(issued)
    ]
    ingress = create_issued_checkpoint(
        issued, cohort_id="pilot-closed-001", groups=["general"],
        model_revision="fixed-qa-revision", rubric_id="rubric-v1",
        ingress_key=INGRESS_KEY,
    )
    receipts = [issue_request_receipt(
        row, cohort_id=ingress["cohort_id"], receipt_key=RECEIPT_KEY,
    ) for row in issued]
    receipt_checkpoint = create_receipt_checkpoint(
        receipts, cohort_id=ingress["cohort_id"], groups=ingress["groups"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
        receipt_key=RECEIPT_KEY,
    )
    attested = [sign_public_grade(
        grade, grade_sequence=sequence, cohort_id=ingress["cohort_id"],
        model_revision=ingress["model_revision"], grader_private_key=GRADER_PRIVATE,
    ) for sequence, grade in enumerate(reversed(grades), 1)]
    grade_checkpoint = close_public_grades(
        attested, cohort_id=ingress["cohort_id"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
        grader_private_key=GRADER_PRIVATE,
    )
    watch = reconcile_issued_grades(
        issued, grades, checkpoint=ingress, ingress_key=INGRESS_KEY,
        watch_key=WATCH_KEY,
    )["report"]
    statement = make_witness_statement(
        cohort_id=ingress["cohort_id"], witness_epoch=1,
        previous_statement_sha256=GENESIS,
        grader_public_key_sha256=public_key_fingerprint(PUBLIC),
        grade_checkpoint_sha256=public_checkpoint_fingerprint(grade_checkpoint),
        ingress_hmac_sha256=ingress["ingress_hmac_sha256"],
        receipt_hmac_sha256=receipt_checkpoint["receipt_ledger_hmac_sha256"],
        ordered_outcomes_sha256=watch["ordered_outcomes_sha256"],
        model_revision=ingress["model_revision"], rubric_id=ingress["rubric_id"],
    )
    held = {
        "attestations": attested, "grade_checkpoint": grade_checkpoint,
        "grader_public_key": PUBLIC, "receipts": receipts,
        "issued_rows": issued, "receipt_checkpoint": receipt_checkpoint,
        "receipt_key": RECEIPT_KEY,
    }
    source = "The telescope was in Oregon."
    decision = {
        "policy_key": POLICY_KEY, "watch_report": watch,
        "watch_key": WATCH_KEY, "ingress_checkpoint": ingress,
        "ingress_key": INGRESS_KEY, "score": 0.9, "group": "general",
        "source_proof": ground_answer(
            "Oregon", source, model_revision="fixed-qa-revision"),
        "source_text": source, "model_revision": "fixed-qa-revision",
        "rubric_id": "rubric-v1",
    }
    return issued, ingress, statement, held, decision

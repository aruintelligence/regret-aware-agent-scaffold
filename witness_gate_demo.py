"""Offline falsification demo using only published synthetic inputs.

Run after installing the public-grader extra: aru-witness-demo
No model is called, no customer evidence is used, and demo keys are public.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from grader_witness_quorum import (
    gate_with_witnessed_public_grades, initialize_witness_state,
    make_witness_statement, review_and_sign_witness_statement,
    sign_witness_statement, statement_fingerprint, verify_witness_quorum,
)
from public_grade_ledger import public_key_fingerprint
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy
from witness_demo_fixture import (
    PINNED, PUBLIC_A, PUBLIC_B, WITNESS_A, WITNESS_B, fixture,
    ATTACKER_PRIVATE, ATTACKER_PUBLIC, PUBLIC, RECEIPT_KEY,
    INGRESS_KEY, WATCH_KEY, POLICY_KEY, policy,
)


def run_demo():
    issued, ingress, first, held, decision = fixture()
    expected = public_key_fingerprint(PUBLIC)
    with tempfile.TemporaryDirectory(prefix="aru-witness-demo-") as directory:
        root = Path(directory)
        signer_states = [root / "review-a.sqlite3", root / "review-b.sqlite3"]
        verifier_state = root / "verifier.sqlite3"
        for path, witness_id, public_key in (
            (signer_states[0], "review-a", PUBLIC_A),
            (signer_states[1], "review-b", PUBLIC_B),
        ):
            initialize_witness_state(
                path, witness_id=witness_id, witness_public_key=public_key)
        initialize_witness_state(
            verifier_state, trusted_witness_public_keys=PINNED,
            risk_policy=policy())

        evidence = {
            "grader_public_key": PUBLIC, "attestations": held["attestations"],
            "grade_checkpoint": held["grade_checkpoint"],
            "receipts": held["receipts"],
            "receipt_checkpoint": held["receipt_checkpoint"],
            "issued_rows": issued, "ingress_checkpoint": ingress,
            "watch_report": decision["watch_report"],
            "receipt_key": RECEIPT_KEY, "ingress_key": INGRESS_KEY,
            "watch_key": WATCH_KEY,
        }

        def reviewed(statement):
            return [review_and_sign_witness_statement(
                statement, witness_id=witness_id, witness_private_key=private_key,
                witness_state_path=state,
                authorized_grader_public_key_sha256=expected, **evidence)
                for witness_id, private_key, state in (
                    ("review-a", WITNESS_A, signer_states[0]),
                    ("review-b", WITNESS_B, signer_states[1]),
                )]

        def gate(statement, approvals, *, keys=PINNED, held_change=None,
                 decision_change=None, calibration=None):
            return gate_with_witnessed_public_grades(
                policy() if calibration is None else calibration,
                statement=statement, witness_approvals=approvals,
                trusted_witness_public_keys=keys,
                witness_state_path=verifier_state,
                **{**held, **(held_change or {})},
                **{**decision, **(decision_change or {})})

        first_approvals = reviewed(first)
        replacement = {"review-a": ATTACKER_PUBLIC, "review-b": PUBLIC_B}
        substituted = [
            sign_witness_statement(first, witness_id="review-a",
                                   witness_private_key=ATTACKER_PRIVATE),
            first_approvals[1],
        ]
        lower_level = verify_witness_quorum(
            first, substituted, trusted_witness_public_keys=replacement)
        permissive = sign_risk_policy(fit_selective_risk_policy(
            [{"sample_id": str(i), "score": 0.9 if i < 150 else 0.1,
              "correct": True, "group": "general"} for i in range(200)],
            thresholds=[0.05, 0.5, 0.95], groups=["general"],
            target_error=0.1, alpha=0.05, min_group_support=30,
            model_revision="fixed-qa-revision", rubric_id="rubric-v1",
            calibration_set_id="alternate-calibration",
        ), key=POLICY_KEY)
        if permissive["selected_threshold"] >= policy()["selected_threshold"]:
            raise AssertionError("replacement did not lower the risk threshold")
        cases = [
            ("self_fed_witness_map", gate(first, substituted, keys=replacement),
             "invalid_or_stale_witness_quorum"),
            ("missing_grade", gate(first, first_approvals,
                                   held_change={"attestations":
                                                held["attestations"][:-1]}),
             "invalid_or_incomplete_witness_evidence"),
            ("missing_receipt", gate(first, first_approvals,
                                     held_change={"receipts": held["receipts"][:-1]}),
             "invalid_or_incomplete_witness_evidence"),
            ("altered_watch", gate(first, first_approvals,
                                   decision_change={"watch_report": {
                                       **decision["watch_report"],
                                       "ordered_outcomes_sha256": "f" * 64}}),
             "invalid_or_incomplete_witness_evidence"),
            ("signed_policy_replacement",
             gate(first, first_approvals, calibration=permissive,
                  decision_change={"score": 0.1}),
             "untrusted_or_stale_risk_policy"),
        ]
        for name, result, expected_reason in cases:
            if result.get("accepted") or result.get("reason") != expected_reason:
                raise AssertionError(f"{name}: unexpected gate result {result!r}")
        first_result = gate(first, first_approvals)
        if not first_result.get("accepted"):
            raise AssertionError(f"honest cohort denied: {first_result!r}")

        second = make_witness_statement(
            **{**{key: value for key, value in first.items()
                  if key not in ("kind", "schema_version")},
               "witness_epoch": 2,
               "previous_statement_sha256": statement_fingerprint(first)})
        second_result = gate(second, reviewed(second))
        if not second_result.get("accepted"):
            raise AssertionError(f"valid successor denied: {second_result!r}")
        replay = gate(first, first_approvals)
        if replay.get("accepted") or replay.get("reason") != "invalid_or_stale_witness_quorum":
            raise AssertionError(f"rollback replay accepted: {replay!r}")

    return {
        "kind": "aru_synthetic_witness_gate_demo", "schema_version": 1,
        "status": "all_expected_boundaries_observed",
        "synthetic_fixture": True, "model_calls": 0,
        "issued_count": len(issued),
        "low_level_self_fed_quorum": lower_level["status"],
        "attacks": [{"name": name, "accepted": result["accepted"],
                     "reason": result["reason"]}
                    for name, result, _ in cases] + [
                        {"name": "rollback_replay", "accepted": replay["accepted"],
                         "reason": replay["reason"]}],
        "honest_epoch_1_accepted": first_result["accepted"],
        "honest_epoch_2_accepted": second_result["accepted"],
        "automatic_rollout_permitted": False,
    }


def main():
    try:
        result = run_demo()
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "unexpected_result", "error": str(exc)}),
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

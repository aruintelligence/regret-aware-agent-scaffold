"""Research pilot: reconcile every issued request with one graded outcome.

An independently retained ingress checkpoint binds a CLOSED cohort. This
establishes completeness only relative to that checkpoint, never to requests
that the ingress operator did not record or to later, unanchored requests.
"""
from __future__ import annotations

import hashlib
import hmac

from anytime_risk_watch import (
    FRACTIONS, _canonical, _id, gate_with_risk_watch, sign_watch_report,
    verify_watch_report, watch_risk_stream,
)

_ISSUE_FIELDS = {"sequence", "request_id", "group", "model_revision", "rubric_id"}
_GRADE_FIELDS = {"request_id", "error", "rubric_id", "grader_id"}
_CHECKPOINT_FIELDS = {
    "kind", "schema_version", "cohort_id", "issued_count", "last_sequence",
    "issued_manifest_sha256", "groups", "model_revision", "rubric_id",
    "target_error", "familywise_alpha", "bet_fractions", "ingress_hmac_sha256",
}


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _key(key, role):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError(role + " key must be at least 32 bytes")


def _hex(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _validate_issues(rows, *, groups, model_revision, rubric_id):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty issued manifest required")
    if (not isinstance(groups, list) or not groups
            or any(not isinstance(group, str) for group in groups)
            or len(set(groups)) != len(groups)):
        raise ValueError("distinct predeclared groups required")
    for group in groups:
        _id(group, "group")
    if "__all__" in groups:
        raise ValueError("reserved group")
    _id(model_revision, "model_revision")
    _id(rubric_id, "rubric_id")
    seen = set()
    for sequence, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != _ISSUE_FIELDS:
            raise ValueError("issued rows require exactly sequence, request_id, group, model_revision, rubric_id")
        if type(row["sequence"]) is not int or row["sequence"] != sequence:
            raise ValueError("issued sequence must be contiguous from one in arrival order")
        request_id = _id(row["request_id"], "request_id")
        if request_id in seen:
            raise ValueError("duplicate issued request_id")
        seen.add(request_id)
        if row["group"] not in groups:
            raise ValueError("unrecognized issued group")
        if row["model_revision"] != model_revision or row["rubric_id"] != rubric_id:
            raise ValueError("issued model or rubric differs from frozen cohort")


def create_issued_checkpoint(issued_rows, *, cohort_id, groups, model_revision,
                             rubric_id, ingress_key, target_error=0.1, alpha=0.05,
                             bet_fractions=FRACTIONS):
    """Call from a separately controlled ingress process at cohort closure."""
    _key(ingress_key, "ingress")
    _id(cohort_id, "cohort_id")
    _validate_issues(issued_rows, groups=groups, model_revision=model_revision,
                     rubric_id=rubric_id)
    # Reuse the watch's parameter checks before freezing the testing design.
    check = watch_risk_stream(
        [{"event_id": issued_rows[0]["request_id"],
          "group": issued_rows[0]["group"], "error": False}],
        groups=groups, model_revision=model_revision, rubric_id=rubric_id,
        target_error=target_error, alpha=alpha, bet_fractions=bet_fractions,
    )
    checkpoint = {
        "kind": "aru_issued_cohort_checkpoint", "schema_version": 1,
        "cohort_id": cohort_id, "issued_count": len(issued_rows),
        "last_sequence": len(issued_rows),
        "issued_manifest_sha256": _digest(issued_rows),
        "groups": list(groups), "model_revision": model_revision,
        "rubric_id": rubric_id, "target_error": check["target_error"],
        "familywise_alpha": check["familywise_alpha"],
        "bet_fractions": check["bet_fractions"],
    }
    checkpoint["ingress_hmac_sha256"] = hmac.new(
        ingress_key, _canonical(checkpoint).encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    return checkpoint


def verify_issued_checkpoint(checkpoint, *, ingress_key):
    if not isinstance(ingress_key, bytes) or len(ingress_key) < 32:
        return False
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        return False
    signature = checkpoint["ingress_hmac_sha256"]
    if not _hex(signature):
        return False
    unsigned = {k: v for k, v in checkpoint.items() if k != "ingress_hmac_sha256"}
    try:
        digest = hmac.new(
            ingress_key, _canonical(unsigned).encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, digest):
            return False
        if (checkpoint["kind"] != "aru_issued_cohort_checkpoint"
                or type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != 1
                or type(checkpoint["issued_count"]) is not int
                or checkpoint["issued_count"] < 1
                or type(checkpoint["last_sequence"]) is not int
                or checkpoint["last_sequence"] != checkpoint["issued_count"]
                or not _hex(checkpoint["issued_manifest_sha256"])):
            return False
        _id(checkpoint["cohort_id"], "cohort_id")
        # Validate all signed scope parameters against the watch's own rules.
        watch_risk_stream(
            [{"event_id": "checkpoint-validation", "group": checkpoint["groups"][0],
              "error": False}],
            groups=checkpoint["groups"], model_revision=checkpoint["model_revision"],
            rubric_id=checkpoint["rubric_id"],
            target_error=checkpoint["target_error"],
            alpha=checkpoint["familywise_alpha"],
            bet_fractions=checkpoint["bet_fractions"],
        )
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False
    return True


def reconcile_issued_grades(issued_rows, grade_rows, *, checkpoint, ingress_key,
                             watch_key):
    """Return a signed watch only when every anchored issue has one grade.

    Missing grades return a pending status without an alarm-free report.
    Malformed, duplicated, foreign or inconsistent records raise ValueError.
    """
    _key(watch_key, "watch")
    if ingress_key == watch_key:
        raise ValueError("ingress and watch keys must be independent")
    if not verify_issued_checkpoint(checkpoint, ingress_key=ingress_key):
        raise ValueError("invalid ingress checkpoint")
    _validate_issues(
        issued_rows, groups=checkpoint["groups"],
        model_revision=checkpoint["model_revision"], rubric_id=checkpoint["rubric_id"],
    )
    if (len(issued_rows) != checkpoint["issued_count"]
            or not hmac.compare_digest(_digest(issued_rows), checkpoint["issued_manifest_sha256"])):
        raise ValueError("issued manifest does not match signed ingress checkpoint")
    if not isinstance(grade_rows, list):
        raise ValueError("grade rows must be a list")
    issued = {row["request_id"] for row in issued_rows}
    grades = {}
    for row in grade_rows:
        if not isinstance(row, dict) or set(row) != _GRADE_FIELDS:
            raise ValueError("grade rows require exactly request_id, error, rubric_id, grader_id")
        request_id = _id(row["request_id"], "request_id")
        if request_id not in issued:
            raise ValueError("grade for an unissued request")
        if request_id in grades:
            raise ValueError("duplicate grade for an issued request")
        if type(row["error"]) is not bool:
            raise ValueError("graded error must be boolean")
        if row["rubric_id"] != checkpoint["rubric_id"]:
            raise ValueError("graded rubric differs from issued rubric")
        _id(row["grader_id"], "grader_id")
        grades[request_id] = row
    missing = [row["request_id"] for row in issued_rows if row["request_id"] not in grades]
    if missing:
        return {
            "status": "pending_grades", "issued_count": len(issued_rows),
            "graded_count": len(grades), "missing_count": len(missing),
            "missing_request_ids": missing, "automatic_rollout_permitted": False,
        }
    ordered_grades = [grades[row["request_id"]] for row in issued_rows]
    outcomes = [
        {"event_id": issue["request_id"], "group": issue["group"],
         "error": grade["error"]}
        for issue, grade in zip(issued_rows, ordered_grades)
    ]
    report = watch_risk_stream(
        outcomes, groups=checkpoint["groups"],
        model_revision=checkpoint["model_revision"], rubric_id=checkpoint["rubric_id"],
        target_error=checkpoint["target_error"],
        alpha=checkpoint["familywise_alpha"],
        bet_fractions=checkpoint["bet_fractions"],
    )
    report.update({
        "cohort_id": checkpoint["cohort_id"],
        "complete_outcomes_verified": True,
        "issued_count": len(issued_rows), "graded_count": len(ordered_grades),
        "issued_manifest_sha256": checkpoint["issued_manifest_sha256"],
        "ordered_grades_sha256": _digest(ordered_grades),
        "ingress_checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
    })
    return {"status": "complete", "report": sign_watch_report(report, key=watch_key)}


def verify_complete_report(report, issued_rows, grade_rows, *, checkpoint,
                           ingress_key, watch_key):
    """Independently recompute the report from the supplied exact source rows."""
    if not verify_watch_report(report, key=watch_key):
        return False
    try:
        result = reconcile_issued_grades(
            issued_rows, grade_rows, checkpoint=checkpoint,
            ingress_key=ingress_key, watch_key=watch_key,
        )
    except (ValueError, TypeError):
        return False
    return result["status"] == "complete" and result["report"] == report


def gate_with_complete_ledger(policy, *, policy_key, watch_report, watch_key,
                              ingress_checkpoint, ingress_key,
                              expected_current_checkpoint_hmac_sha256,
                              expected_outcomes_sha256, score, group, source_proof,
                              source_text, model_revision, rubric_id,
                              min_group_outcomes=1):
    """Require a separately supplied CURRENT closed-cohort ingress anchor.

    The caller must independently obtain the current checkpoint fingerprint
    and outcome digest; copying either from the report defeats replay checks.
    """
    deny = {"accepted": False, "answer": "", "automatic_rollout_permitted": False,
            "warning": "A complete closed cohort is not proof of safety or current live coverage."}
    if ingress_key == watch_key:
        return {**deny, "reason": "monitor_and_ingress_keys_not_independent"}
    if (not verify_issued_checkpoint(ingress_checkpoint, ingress_key=ingress_key)
            or not _hex(expected_current_checkpoint_hmac_sha256)
            or not hmac.compare_digest(
                expected_current_checkpoint_hmac_sha256,
                ingress_checkpoint["ingress_hmac_sha256"],
            )):
        return {**deny, "reason": "missing_or_stale_ingress_checkpoint"}
    if not verify_watch_report(watch_report, key=watch_key):
        return {**deny, "reason": "missing_or_invalid_signed_watch"}
    if (watch_report.get("complete_outcomes_verified") is not True
            or type(watch_report.get("issued_count")) is not int
            or watch_report["issued_count"] < 1
            or type(watch_report.get("graded_count")) is not int
            or watch_report["graded_count"] != watch_report["issued_count"]
            or watch_report["issued_count"] != ingress_checkpoint["issued_count"]
            or type(watch_report.get("outcome_count")) is not int
            or watch_report.get("outcome_count") != watch_report["issued_count"]
            or not _hex(watch_report.get("ordered_grades_sha256"))
            or watch_report.get("issued_manifest_sha256") != ingress_checkpoint["issued_manifest_sha256"]
            or watch_report.get("ingress_checkpoint_hmac_sha256") != ingress_checkpoint["ingress_hmac_sha256"]
            or watch_report.get("cohort_id") != ingress_checkpoint["cohort_id"]):
        return {**deny, "reason": "incomplete_or_unbound_outcome_ledger"}
    if (watch_report.get("model_revision") != ingress_checkpoint["model_revision"]
            or watch_report.get("rubric_id") != ingress_checkpoint["rubric_id"]
            or watch_report.get("groups") != ingress_checkpoint["groups"]
            or watch_report.get("target_error") != ingress_checkpoint["target_error"]
            or watch_report.get("familywise_alpha") != ingress_checkpoint["familywise_alpha"]
            or watch_report.get("bet_fractions") != ingress_checkpoint["bet_fractions"]):
        return {**deny, "reason": "ledger_watch_scope_mismatch"}
    return gate_with_risk_watch(
        policy, policy_key=policy_key, watch_report=watch_report,
        watch_key=watch_key, score=score, group=group,
        source_proof=source_proof, source_text=source_text,
        model_revision=model_revision, rubric_id=rubric_id,
        expected_outcomes_sha256=expected_outcomes_sha256,
        min_group_outcomes=min_group_outcomes,
    )

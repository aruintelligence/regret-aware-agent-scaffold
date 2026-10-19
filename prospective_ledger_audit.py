"""Two-stage public QA audit: freeze issued requests before model inference.

This research experiment uses one CI authority and repository-authored grading.
It demonstrates prospective accounting, not independent customer ingress or
externally audited correctness labels.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

from anytime_risk_watch import FRACTIONS, watch_risk_stream
from extractive_qa_engine import DEFAULT_QA_MODEL
from real_model_watch_audit import (
    ALPHA, GROUPS, OFFICIAL_DATASET_SHA256, RUBRIC, SOURCE_CUTOFF,
    TARGET_ERROR, run_audit,
)
from squad_sentence_retrieval import load_questions
from watch_outcome_ledger import (
    create_issued_checkpoint, reconcile_issued_grades,
    verify_complete_report, verify_issued_checkpoint,
)

OFFSET = 680
LIMIT = 120
MODEL_REVISION = "564e9b582944a57a3e586bbb98fd6f0a4118db7f"
COHORT_ID = "squad-680-799-prospective-ledger-v1"
GRADER_ID = "repository_squad_human_gold_grader_v1"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_json(path, value):
    target = Path(path)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if os.name == "posix":
        os.chmod(target, 0o600)


def _write_jsonl(path, rows):
    target = Path(path)
    with target.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    if os.name == "posix":
        os.chmod(target, 0o600)


def frozen_manifest(dataset_path, *, offset=OFFSET, limit=LIMIT,
                    model_revision=MODEL_REVISION,
                    expected_dataset_sha256=OFFICIAL_DATASET_SHA256):
    """Bind each request ID to its public question AND source content."""
    questions, digest, _ = load_questions(dataset_path, limit, offset)
    if digest != expected_dataset_sha256 or len(questions) != limit:
        raise ValueError("dataset digest mismatch or missing reserved questions")
    rows = []
    for index, (qid, _title, context, question, _gold) in enumerate(questions, 1):
        token = json.dumps([digest, qid, context, question], ensure_ascii=True,
                           separators=(",", ":")).encode("utf-8")
        rows.append({
            "sequence": index, "request_id": "squad-" + hashlib.sha256(token).hexdigest(),
            "group": "short_source" if len(context) <= SOURCE_CUTOFF else "long_source",
            "model_revision": model_revision, "rubric_id": RUBRIC,
        })
    return rows, digest


def prepare_cohort(dataset_path, out_dir, *, ingress_key,
                   expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
                   offset=OFFSET, limit=LIMIT, model_revision=MODEL_REVISION):
    """Run and retain this stage before making any calls to the model."""
    target = Path(out_dir)
    if target.exists():
        raise FileExistsError("choose a fresh preparation directory")
    issued, digest = frozen_manifest(
        dataset_path, offset=offset, limit=limit, model_revision=model_revision,
        expected_dataset_sha256=expected_dataset_sha256,
    )
    checkpoint = create_issued_checkpoint(
        issued, cohort_id=COHORT_ID, groups=list(GROUPS),
        model_revision=model_revision, rubric_id=RUBRIC,
        ingress_key=ingress_key, target_error=TARGET_ERROR, alpha=ALPHA,
    )
    target.mkdir(parents=True)
    _write_jsonl(target / "issued.jsonl", issued)
    _write_json(target / "checkpoint.json", checkpoint)
    _write_json(target / "plan.json", {
        "kind": "aru_prospective_preinference_public_cohort",
        "dataset_sha256": digest, "offset": offset, "limit": limit,
        "model_revision": model_revision, "model_id": DEFAULT_QA_MODEL,
        "rubric_id": RUBRIC, "groups": list(GROUPS),
        "cohort_id": COHORT_ID,
        "issued_count": len(issued),
        "issued_manifest_sha256": checkpoint["issued_manifest_sha256"],
        "ingress_checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
        "graded": False, "automatic_rollout_permitted": False,
    })
    return checkpoint


def _validate_prepared(dataset_path, prepared_dir, *, ingress_key,
                       expected_checkpoint_hmac, expected_dataset_sha256,
                       offset, limit, model_revision):
    folder = Path(prepared_dir)
    checkpoint = _read_json(folder / "checkpoint.json")
    planned = _read_json(folder / "plan.json")
    issued = _read_jsonl(folder / "issued.jsonl")
    expected, digest = frozen_manifest(
        dataset_path, offset=offset, limit=limit, model_revision=model_revision,
        expected_dataset_sha256=expected_dataset_sha256,
    )
    expected_checkpoint = create_issued_checkpoint(
        expected, cohort_id=COHORT_ID, groups=list(GROUPS),
        model_revision=model_revision, rubric_id=RUBRIC,
        ingress_key=ingress_key, target_error=TARGET_ERROR, alpha=ALPHA,
        bet_fractions=FRACTIONS,
    )
    if (not verify_issued_checkpoint(checkpoint, ingress_key=ingress_key)
            or not isinstance(expected_checkpoint_hmac, str)
            or not hmac.compare_digest(expected_checkpoint_hmac,
                                       checkpoint.get("ingress_hmac_sha256", ""))
            or issued != expected
            or checkpoint != expected_checkpoint
            or checkpoint["cohort_id"] != COHORT_ID
            or checkpoint["model_revision"] != model_revision
            or checkpoint["rubric_id"] != RUBRIC
            or checkpoint["groups"] != list(GROUPS)
            or checkpoint["target_error"] != TARGET_ERROR
            or checkpoint["familywise_alpha"] != ALPHA
            or checkpoint["issued_count"] != limit
            or planned != {
                "kind": "aru_prospective_preinference_public_cohort",
                "dataset_sha256": digest, "offset": offset, "limit": limit,
                "model_revision": model_revision, "model_id": DEFAULT_QA_MODEL,
                "rubric_id": RUBRIC, "groups": list(GROUPS),
                "cohort_id": COHORT_ID, "issued_count": len(issued),
                "issued_manifest_sha256": checkpoint["issued_manifest_sha256"],
                "ingress_checkpoint_hmac_sha256": checkpoint["ingress_hmac_sha256"],
                "graded": False, "automatic_rollout_permitted": False,
            }):
        raise ValueError("invalid, stale or changed preinference issuance plan")
    return issued, checkpoint, digest


def _grades_from_observed(issued, observed):
    if len(issued) != len(observed):
        raise ValueError("missing actual model outcomes: refuse completed report")
    grades = []
    for index, (issue, outcome) in enumerate(zip(issued, observed), 1):
        if (not isinstance(outcome, dict)
                or set(outcome) != {"event_id", "group", "error"}
                or outcome["event_id"] != "observed-%04d" % index
                or outcome["group"] != issue["group"]
                or type(outcome["error"]) is not bool):
            raise ValueError("model outcome disagrees with frozen issuance order or group")
        grades.append({
            "request_id": issue["request_id"], "error": outcome["error"],
            "rubric_id": RUBRIC, "grader_id": GRADER_ID,
        })
    return grades


def _accounting_report(issued, grades, watch, observed_report, *, digest,
                       qa_model, offset, model_revision):
    return {
        "kind": "aru_prospective_public_real_model_outcome_accounting",
        "dataset_sha256": digest, "cohort_id": COHORT_ID,
        "model_revision": model_revision, "model_id": qa_model,
        "offset": offset, "requested": len(issued), "graded": len(grades),
        "errors": sum(row["error"] for row in grades),
        "model_call_failures_counted_as_errors": observed_report["model_call_failures_counted_as_errors"],
        "verified_source_certificates": observed_report["verified_source_certificates"],
        "issued_manifest_sha256": watch["issued_manifest_sha256"],
        "ingress_checkpoint_hmac_sha256": watch["ingress_checkpoint_hmac_sha256"],
        "ordered_grades_sha256": watch["ordered_grades_sha256"],
        "ordered_outcomes_sha256": watch["ordered_outcomes_sha256"],
        "degradation_alarm": watch["degradation_alarm"],
        "preinference_manifest_created_before_model_calls": True,
        "independent_ingress_operator": False,
        "independently_audited_grader": False,
        "automatic_rollout_permitted": False,
    }


def run_cohort(dataset_path, prepared_dir, out_dir, *, ingress_key, watch_key,
               expected_checkpoint_hmac, acknowledge_model_calls=False,
               expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
               offset=OFFSET, limit=LIMIT, model_revision=MODEL_REVISION,
               qa_model=DEFAULT_QA_MODEL, engine=None):
    if not acknowledge_model_calls:
        raise ValueError("explicit acknowledgement of model calls required")
    if qa_model != DEFAULT_QA_MODEL and engine is None:
        raise ValueError("prospective public study requires its pinned model ID")
    # All independent-plan checks precede model initialization and inference.
    issued, checkpoint, digest = _validate_prepared(
        dataset_path, prepared_dir, ingress_key=ingress_key,
        expected_checkpoint_hmac=expected_checkpoint_hmac,
        expected_dataset_sha256=expected_dataset_sha256,
        offset=offset, limit=limit, model_revision=model_revision,
    )
    target = Path(out_dir)
    if target.exists():
        raise FileExistsError("choose a fresh run directory")
    target.mkdir(parents=True)
    observed_report = run_audit(
        dataset_path, target / "actual_model", acknowledge_model_calls=True,
        expected_dataset_sha256=digest, qa_model=qa_model,
        qa_revision=model_revision, engine=engine, offset=offset, limit=limit,
        onsets=(30, 60, 90) if limit > 90 else (1,), null_trials=1,
    )
    if (observed_report["model_revision"] != model_revision
            or observed_report["model"] != qa_model
            or observed_report["requested"] != limit
            or observed_report["model_call_failures_counted_as_errors"] > limit):
        raise ValueError("observed model audit differs from precommitted design")
    observed = _read_jsonl(target / "actual_model/outcomes.jsonl")
    grades = _grades_from_observed(issued, observed)
    signed = reconcile_issued_grades(
        issued, grades, checkpoint=checkpoint,
        ingress_key=ingress_key, watch_key=watch_key,
    )
    if signed["status"] != "complete":
        raise ValueError("ungraded issued requests: no signed watch")
    watch = signed["report"]
    if (watch["degradation_alarm"] != observed_report["observed_model_watch"]["degradation_alarm"]
            or watch["outcome_count"] != limit):
        raise ValueError("signed watch differs from observed model report")
    _write_jsonl(target / "grades.jsonl", grades)
    _write_json(target / "signed_watch.json", watch)
    report = _accounting_report(
        issued, grades, watch, observed_report, digest=digest,
        qa_model=qa_model, offset=offset, model_revision=model_revision,
    )
    _write_json(target / "accounting_report.json", report)
    return report


def verify_cohort(dataset_path, prepared_dir, out_dir, *, ingress_key, watch_key,
                  expected_checkpoint_hmac,
                  expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
                  offset=OFFSET, limit=LIMIT, model_revision=MODEL_REVISION,
                  qa_model=DEFAULT_QA_MODEL):
    issued, checkpoint, digest = _validate_prepared(
        dataset_path, prepared_dir, ingress_key=ingress_key,
        expected_checkpoint_hmac=expected_checkpoint_hmac,
        expected_dataset_sha256=expected_dataset_sha256,
        offset=offset, limit=limit, model_revision=model_revision,
    )
    root = Path(out_dir)
    grades = _read_jsonl(root / "grades.jsonl")
    signed = _read_json(root / "signed_watch.json")
    report = _read_json(root / "accounting_report.json")
    observed = _read_jsonl(root / "actual_model/outcomes.jsonl")
    actual_report = _read_json(root / "actual_model/report.json")
    recomputed_actual_watch = watch_risk_stream(
        observed, groups=GROUPS, model_revision=model_revision,
        rubric_id=RUBRIC, target_error=TARGET_ERROR, alpha=ALPHA,
    )
    if (not verify_complete_report(signed, issued, grades, checkpoint=checkpoint,
                                   ingress_key=ingress_key, watch_key=watch_key)
            or grades != _grades_from_observed(issued, observed)
            or actual_report.get("observed_model_watch") != recomputed_actual_watch
            or actual_report.get("dataset_sha256") != digest
            or actual_report.get("model_revision") != model_revision
            or actual_report.get("model") != qa_model
            or actual_report.get("rubric_id") != RUBRIC
            or actual_report.get("offset") != offset
            or actual_report.get("requested") != limit
            or actual_report.get("observed") != len(observed)
            or actual_report.get("target_error") != TARGET_ERROR
            or actual_report.get("familywise_alpha") != ALPHA
            or actual_report.get("source_group_cutoff_characters") != SOURCE_CUTOFF
            or actual_report.get("automatic_rollout_permitted") is not False
            or signed["degradation_alarm"] != recomputed_actual_watch["degradation_alarm"]
            or report != _accounting_report(
                issued, grades, signed, actual_report, digest=digest,
                qa_model=qa_model, offset=offset,
                model_revision=model_revision,
            )):
        raise ValueError("prospective issued/graded evidence did not verify")
    return report


def _key(name):
    value = os.environ.get(name, "").encode("utf-8")
    if len(value) < 32:
        raise ValueError(name + " requires at least 32 bytes")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", required=True)
        if name == "prepare":
            sub.add_argument("--prepared-dir", required=True)
        else:
            sub.add_argument("--prepared-dir", required=True)
            sub.add_argument("--out-dir", required=True)
            sub.add_argument("--expected-checkpoint-hmac", required=True)
        if name == "run":
            sub.add_argument("--acknowledge-model-calls", action="store_true")
    args = parser.parse_args()
    try:
        ingress_key = _key("ARU_ISSUED_SIGNING_KEY")
        if args.command == "prepare":
            result = prepare_cohort(args.dataset, args.prepared_dir, ingress_key=ingress_key)
            print(json.dumps({"prepared": True, "issued": result["issued_count"],
                              "checkpoint_hmac": result["ingress_hmac_sha256"]}))
            return
        watch_key = _key("ARU_WATCH_SIGNING_KEY")
        if args.command == "run":
            report = run_cohort(
                args.dataset, args.prepared_dir, args.out_dir,
                ingress_key=ingress_key, watch_key=watch_key,
                expected_checkpoint_hmac=args.expected_checkpoint_hmac,
                acknowledge_model_calls=args.acknowledge_model_calls,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            if report["degradation_alarm"]:
                raise SystemExit(10)
        else:
            report = verify_cohort(
                args.dataset, args.prepared_dir, args.out_dir,
                ingress_key=ingress_key, watch_key=watch_key,
                expected_checkpoint_hmac=args.expected_checkpoint_hmac,
            )
            print(json.dumps({"verified": True, "requested": report["requested"],
                              "graded": report["graded"],
                              "automatic_rollout_permitted": False}))
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(4, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()

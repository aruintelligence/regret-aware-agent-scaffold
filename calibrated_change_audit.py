"""Frozen two-stage human-gold audit of the calibrated restart research watch."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from calibrated_change_watch import calibrated_restart_risk_stream
from change_point_audit import (
    CONTEXT_CAP, INTERVENTION_ONSET, LIMIT, ONSETS, SEVERITIES, STARTS,
    run_change_point_audit,
)
from extractive_qa_engine import DEFAULT_QA_MODEL, ExtractiveQAEngine
from real_model_watch_audit import (
    GROUPS, OFFICIAL_DATASET_SHA256, RUBRIC, run_audit,
)
from squad_sentence_retrieval import load_questions

CALIBRATION_OFFSET = 440
EVALUATION_OFFSET = 560
CALIBRATION_LIMIT = 120
TOTAL_ALPHA = 0.05
CALIBRATION_DELTA = 0.025
MIN_SLICE_CALIBRATION = 20
STRESS_SEED = 20261021


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run_calibrated_audit(dataset_path, out_dir, *, acknowledge_model_calls=False,
                         expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
                         qa_model=DEFAULT_QA_MODEL, qa_revision=None, engine=None,
                         calibration_offset=CALIBRATION_OFFSET,
                         evaluation_offset=EVALUATION_OFFSET,
                         calibration_limit=CALIBRATION_LIMIT, evaluation_limit=LIMIT,
                         starts=STARTS, onsets=ONSETS,
                         intervention_onset=INTERVENTION_ONSET,
                         context_cap=CONTEXT_CAP):
    if not acknowledge_model_calls:
        raise ValueError("explicit acknowledgement of real-model calls required")
    if (any(type(v) is not int or v < 0 for v in
            (calibration_offset, evaluation_offset, calibration_limit, evaluation_limit))
            or calibration_limit < 1 or evaluation_limit < 1
            or not (calibration_offset + calibration_limit <= evaluation_offset
                    or evaluation_offset + evaluation_limit <= calibration_offset)):
        raise ValueError("disjoint nonempty calibration and evaluation offsets required")
    dest = Path(out_dir)
    if dest.exists():
        raise FileExistsError("choose a fresh audit directory")
    cal_questions, digest, _ = load_questions(dataset_path, calibration_limit, calibration_offset)
    eval_questions, eval_digest, _ = load_questions(dataset_path, evaluation_limit, evaluation_offset)
    if (digest != expected_dataset_sha256 or eval_digest != digest
            or len(cal_questions) != calibration_limit or len(eval_questions) != evaluation_limit):
        raise ValueError("dataset hash mismatch or missing reserved questions")

    qa = engine if engine is not None else ExtractiveQAEngine(qa_model, revision=qa_revision, threads=2)
    cal_report = run_audit(
        dataset_path, dest / "calibration", acknowledge_model_calls=True,
        expected_dataset_sha256=digest, qa_model=qa_model,
        qa_revision=qa_revision, engine=qa, offset=calibration_offset,
        limit=calibration_limit, onsets=onsets, null_trials=1,
    )
    eval_report = run_change_point_audit(
        dataset_path, dest / "evaluation", acknowledge_model_calls=True,
        expected_dataset_sha256=digest, qa_model=qa_model,
        qa_revision=qa_revision, engine=qa, offset=evaluation_offset,
        limit=evaluation_limit, starts=starts, onsets=onsets,
        intervention_onset=intervention_onset, context_cap=context_cap,
        null_trials=1,
    )
    calibration = _read_rows(dest / "calibration/outcomes.jsonl")
    baseline = _read_rows(dest / "evaluation/baseline/outcomes.jsonl")
    intervention = _read_rows(dest / "evaluation/intervention_outcomes.jsonl")

    def monitor(rows):
        return calibrated_restart_risk_stream(
            calibration, rows, groups=GROUPS, model_revision=qa.revision,
            rubric_id=RUBRIC, alpha_total=TOTAL_ALPHA,
            calibration_delta=CALIBRATION_DELTA,
            min_slice_calibration=MIN_SLICE_CALIBRATION,
            horizon=evaluation_limit, starts=starts,
        )

    observed_baseline = monitor(baseline)
    observed_intervention = monitor(intervention)
    scenarios = []
    for onset in onsets:
        for severity in SEVERITIES:
            rng = random.Random(STRESS_SEED + onset * 10 + int(severity * 100))
            modified = [dict(row) for row in baseline]
            flipped = 0
            for row in modified[onset:]:
                if not row["error"] and rng.random() < severity:
                    row["error"] = True
                    flipped += 1
            result = monitor(modified)
            first = result["first_alarm_global_observation"]
            scenarios.append({
                "kind": "synthetic_correct_to_error_label_flips",
                "onset_after_observations": onset, "flip_probability": severity,
                "flipped": flipped, "alarm": first,
                "pre_onset_alarm": first is not None and first <= onset,
                "detection_delay": first - onset if first is not None and first > onset else None,
                "calibration_ready": result["calibration_ready"],
            })
    first = observed_intervention["first_alarm_global_observation"]
    report = {
        "kind": "aru_independently_calibrated_restart_real_model_research_audit",
        "dataset": "Stanford SQuAD v1.1 public development split",
        "dataset_sha256": digest, "model": qa.model_id,
        "model_revision": qa.revision, "rubric_id": RUBRIC,
        "calibration_offset": calibration_offset, "calibration_requested": calibration_limit,
        "evaluation_offset": evaluation_offset, "evaluation_requested": evaluation_limit,
        "calibration_model_calls": cal_report["observed"],
        "calibration_errors": sum(row["error"] for row in calibration),
        "calibration_verified_source_certificates": cal_report["verified_source_certificates"],
        "calibration_execution_failures": cal_report["model_call_failures_counted_as_errors"],
        "baseline_model_calls": eval_report["baseline_model_calls"],
        "baseline_errors": sum(row["error"] for row in baseline),
        "baseline_verified_source_certificates": eval_report["baseline_verified_source_certificates"],
        "baseline_execution_failures": eval_report["baseline_model_call_failures"],
        "intervention_additional_model_calls": eval_report["intervention_additional_model_calls"],
        "intervention_errors": sum(row["error"] for row in intervention),
        "intervention_verified_source_certificates": eval_report["intervention_verified_source_certificates"],
        "intervention_execution_failures": eval_report["intervention_model_call_failures"],
        "intervention_onset_after_observations": intervention_onset,
        "intervention_context_cap_characters": context_cap,
        "intervention_is_natural_live_drift": False,
        "baseline_calibrated_watch": observed_baseline,
        "intervention_calibrated_watch": observed_intervention,
        "intervention_pre_onset_alarm": first is not None and first <= intervention_onset,
        "intervention_detection_delay": (
            first - intervention_onset if first is not None and first > intervention_onset else None
        ),
        "synthetic_label_flip_scenarios": scenarios,
        "same_questions_previous_50_percent_watch": {
            "baseline_first_alarm": eval_report["baseline_restart_watch"]["first_alarm_global_observation"],
            "intervention_first_alarm": eval_report["observed_intervention_restart_watch"]["first_alarm_global_observation"],
        },
        "automatic_rollout_permitted": False,
        "limitations": [
            "The calibration upper bounds rely on IID Bernoulli per-slice errors and stationarity, neither established on this public dataset.",
            "Disjoint question IDs may share articles, so cluster dependence can invalidate the nominal bound.",
            "The independent gold answers are public, while this repository's grader is not externally audited.",
            "The 128-character retrieval failure is a deliberate intervention, not spontaneous model drift.",
            "Synthetic label flips are not real model outcomes; a no-alarm path is not proof of safety.",
            "This unsigned research audit is not a deployment authorization or customer assurance certificate.",
        ],
    }
    (dest / "calibrated_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--qa-revision")
    parser.add_argument("--acknowledge-model-calls", action="store_true")
    args = parser.parse_args()
    report = run_calibrated_audit(
        args.dataset, args.out_dir, qa_model=args.qa_model,
        qa_revision=args.qa_revision, acknowledge_model_calls=args.acknowledge_model_calls,
    )
    print(json.dumps({
        "calibration_errors": report["calibration_errors"],
        "calibration_targets": report["baseline_calibrated_watch"]["calibration_targets"],
        "baseline_errors": report["baseline_errors"],
        "baseline_alarm": report["baseline_calibrated_watch"]["first_alarm_global_observation"],
        "intervention_errors": report["intervention_errors"],
        "intervention_alarm": report["intervention_calibrated_watch"]["first_alarm_global_observation"],
        "intervention_detection_delay": report["intervention_detection_delay"],
        "automatic_rollout_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

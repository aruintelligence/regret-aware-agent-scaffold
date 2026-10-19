"""Preregistered research audit: human-gold model errors and injected drift.

Only the baseline stream comes from actual model calls. Post-onset flips and
null trials are synthetic stress tests, never represented as model outcomes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from anytime_risk_watch import watch_risk_stream
from extractive_qa_engine import DEFAULT_QA_MODEL, ExtractiveQAEngine
from grounding_contract import verify_grounding
from llm_squad_benchmark import score_response
from squad_sentence_retrieval import load_questions

OFFICIAL_DATASET_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
OFFSET = 200
LIMIT = 120
GROUPS = ("short_source", "long_source")
SOURCE_CUTOFF = 600  # Source characters; assigned before inference.
RUBRIC = "squad_v1.1_human_gold_normalized_f1_at_least_0.5_v1"
TARGET_ERROR = 0.5  # A stress-test target, not an enterprise release target.
ALPHA = 0.05
ONSETS = (30, 60, 90)  # Number of observed events before perturbation.
SEVERITIES = (0.5, 1.0)  # Chance of flipping an originally correct outcome.
NULL_TRIALS = 200
SEED = 20261019


def first_alarm_index(rows, watch):
    """Convert a subgroup-local first crossing to its global stream index."""
    positions = {group: [] for group in GROUPS}
    for index, row in enumerate(rows, 1):
        positions[row["group"]].append(index)
    crossings = []
    for group, evidence in watch["slices"].items():
        crossing = evidence["first_crossing_observation"]
        if crossing is not None:
            crossings.append(crossing if group == "__all__" else positions[group][crossing - 1])
    return min(crossings) if crossings else None


def analyze_outcomes(rows, *, model_revision, onsets=ONSETS, null_trials=NULL_TRIALS):
    """Analyze the real baseline, then distinctly labeled synthetic scenarios."""
    if not rows or any(isinstance(k, bool) or not isinstance(k, int)
                       or not 0 <= k < len(rows) for k in onsets):
        raise ValueError("predeclared onsets must be inside the observed stream")
    if isinstance(null_trials, bool) or not isinstance(null_trials, int) or null_trials < 1:
        raise ValueError("positive null trial count required")

    def monitor(events):
        return watch_risk_stream(
            events, groups=GROUPS, model_revision=model_revision,
            rubric_id=RUBRIC, target_error=TARGET_ERROR, alpha=ALPHA,
        )

    baseline = monitor(rows)
    scenarios = []
    for onset in onsets:
        for severity in SEVERITIES:
            rng = random.Random(SEED + onset * 10 + int(severity * 100))
            altered = [dict(row) for row in rows]
            flipped = 0
            for row in altered[onset:]:
                if not row["error"] and rng.random() < severity:
                    row["error"] = True
                    flipped += 1
            watch = monitor(altered)
            alarm = first_alarm_index(altered, watch)
            scenarios.append({
                "synthetic_injection": True,
                "onset_after_observations": onset,
                "originally_correct_flip_probability": severity,
                "flipped_outcomes": flipped,
                "first_alarm_global_observation": alarm,
                "pre_onset_alarm": alarm is not None and alarm <= onset,
                "detected_after_onset": alarm is not None and alarm > onset,
                "detection_delay_observations": alarm - onset if alarm is not None and alarm > onset else None,
                "missed_by_horizon": alarm is None,
                "alarm_groups": watch["alarm_groups"],
            })

    rng = random.Random(SEED)
    null_alarms = 0
    for trial in range(null_trials):
        simulated = [
            {"event_id": row["event_id"], "group": row["group"],
             "error": rng.random() < TARGET_ERROR}
            for row in rows
        ]
        null_alarms += monitor(simulated)["degradation_alarm"]
    return {
        "observed_model_watch": baseline,
        "observed_baseline_first_alarm_global_observation": first_alarm_index(rows, baseline),
        "injected_deterioration": scenarios,
        "synthetic_iid_null": {
            "trials": null_trials, "alarms": null_alarms,
            "alarm_fraction": null_alarms / null_trials,
            "generating_error_probability": TARGET_ERROR,
            "does_not_estimate_real_model_false_alarm_rate": True,
        },
    }


def run_audit(dataset_path, out_dir, *, acknowledge_model_calls=False,
              expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
              qa_model=DEFAULT_QA_MODEL, qa_revision=None,
              engine=None, offset=OFFSET, limit=LIMIT,
              onsets=ONSETS, null_trials=NULL_TRIALS):
    if not acknowledge_model_calls:
        raise ValueError("explicit model-call acknowledgement required")
    if not isinstance(expected_dataset_sha256, str) or len(expected_dataset_sha256) != 64:
        raise ValueError("a pinned dataset SHA-256 is required")
    questions, digest, total = load_questions(dataset_path, limit, offset)
    if digest != expected_dataset_sha256 or len(questions) != limit:
        raise ValueError("dataset hash mismatch or insufficient reserved questions")
    dest = Path(out_dir)
    if dest.exists():
        raise FileExistsError("choose a fresh audit directory")
    qa = engine if engine is not None else ExtractiveQAEngine(qa_model, revision=qa_revision, threads=2)
    if qa.model_id != qa_model or not qa.revision:
        raise ValueError("a named model and immutable resolved revision are required")
    rows, failures, verified, f1_total, exact_total = [], 0, 0, 0.0, 0.0
    for index, (_qid, _article, context, question, gold) in enumerate(questions, 1):
        group = "short_source" if len(context) <= SOURCE_CUTOFF else "long_source"
        try:
            answer = qa.answer(context, question)
            if answer.get("model_revision") != qa.revision:
                raise RuntimeError("per-answer model revision mismatch")
            if answer["accepted"]:
                if not verify_grounding(answer["proof"], context):
                    raise RuntimeError("independent source certificate failed")
                verified += 1
            elif answer["answer"]:
                raise RuntimeError("abstention included an unsupported answer")
            score = score_response(answer["answer"], gold)
            f1_total += score["f1"]
            exact_total += score["exact_match"]
            error = not (answer["accepted"] and score["f1"] >= 0.5)
        except (ValueError, RuntimeError):
            # Requested calls remain in the denominator; failures cannot be
            # quietly deleted to improve the observed error stream.
            failures += 1
            error = True
        rows.append({"event_id": "observed-%04d" % index, "group": group, "error": error})
    analysis = analyze_outcomes(rows, model_revision=qa.revision,
                                onsets=onsets, null_trials=null_trials)
    report = {
        "kind": "aru_real_model_anytime_watch_public_human_gold_audit",
        "dataset": "Stanford SQuAD v1.1 public development split",
        "dataset_sha256": digest, "total_answerable_questions": total,
        "selection": "SHA256-sorted question IDs, no outcome-dependent exclusions",
        "offset": offset, "requested": limit, "observed": len(rows),
        "model": qa.model_id, "model_revision": qa.revision,
        "rubric_id": RUBRIC, "groups": list(GROUPS),
        "source_group_cutoff_characters": SOURCE_CUTOFF,
        "target_error": TARGET_ERROR, "familywise_alpha": ALPHA,
        "model_call_failures_counted_as_errors": failures,
        "verified_source_certificates": verified,
        "mean_normalized_token_f1": f1_total / limit,
        "exact_match_fraction": exact_total / limit,
        "observed_outcomes_kind": "actual_model_calls_scored_after_inference_against_published_human_answers",
        "stress_outcomes_kind": "synthetic_label_flips_and_iid_null_only",
        "automatic_rollout_permitted": False,
        "limitations": [
            "Public development questions are not a hidden holdout or customer workflow.",
            "Human gold answers are external, but this repository's grading code is not independently audited.",
            "The F1 cutoff and high error target are research stress-test choices, not safety thresholds.",
            "Questions within articles may be correlated; the sequential conditional null is not established.",
            "The injected drift changes graded labels, not a real deployed model; detection delays are stress results only.",
            "One observed stream cannot estimate a real-world false-alarm rate or power.",
            "This unsigned audit is not an independently operated monitoring ledger or signed release certificate.",
        ],
        **analysis,
    }
    dest.mkdir(parents=True)
    (dest / "outcomes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8",
    )
    (dest / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--qa-revision")
    parser.add_argument("--acknowledge-model-calls", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.dataset, args.out_dir, qa_model=args.qa_model,
                       qa_revision=args.qa_revision,
                       acknowledge_model_calls=args.acknowledge_model_calls)
    print(json.dumps({
        "model_revision": report["model_revision"],
        "requested": report["requested"], "observed": report["observed"],
        "observed_alarm": report["observed_model_watch"]["degradation_alarm"],
        "synthetic_injected_scenarios": report["injected_deterioration"],
        "automatic_rollout_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

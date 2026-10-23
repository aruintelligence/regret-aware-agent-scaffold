"""Precommitted disjoint human-gold audit for the research restart monitor.

Baseline and the context-limited intervention are *actual model calls*.
The label-flip and Bernoulli streams are separately marked synthetic.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from anytime_risk_watch import watch_risk_stream
from change_point_watch import restart_risk_stream
from extractive_qa_engine import DEFAULT_QA_MODEL, ExtractiveQAEngine
from grounding_contract import verify_grounding
from llm_squad_benchmark import score_response
from real_model_watch_audit import (
    ALPHA, GROUPS, OFFICIAL_DATASET_SHA256, RUBRIC, SOURCE_CUTOFF,
    TARGET_ERROR, run_audit,
)
from squad_sentence_retrieval import load_questions

OFFSET = 320
LIMIT = 120
STARTS = (1, 21, 41, 61, 81, 101)
ONSETS = (25, 55, 85)
SEVERITIES = (0.5, 1.0)
INTERVENTION_ONSET = 55
CONTEXT_CAP = 128
NULL_TRIALS = 400
SEED = 20261020


def analyze_restarts(rows, *, model_revision, starts=STARTS, onsets=ONSETS,
                     intervention_rows=None, null_trials=NULL_TRIALS, seed=SEED):
    """Evaluate identical frozen schedules on a new real trace and stress arms."""
    if (not isinstance(rows, list) or not rows
            or any(type(onset) is not int or not 0 <= onset < len(rows) for onset in onsets)
            or null_trials < 1 or isinstance(null_trials, bool)):
        raise ValueError("invalid stream, onset, or number of null trials")

    def restart(events):
        return restart_risk_stream(
            events, groups=GROUPS, model_revision=model_revision, rubric_id=RUBRIC,
            target_error=TARGET_ERROR, alpha=ALPHA, horizon=len(rows), starts=starts,
        )

    def original(events):
        return watch_risk_stream(
            events, groups=GROUPS, model_revision=model_revision, rubric_id=RUBRIC,
            target_error=TARGET_ERROR, alpha=ALPHA,
        )

    def summary(events):
        new, old = restart(events), original(events)
        # The existing watch indexes subgroup crossings locally; map to the
        # full request order for an apples-to-apples delay comparison.
        positions = {group: [] for group in GROUPS}
        for i, row in enumerate(events, 1):
            positions[row["group"]].append(i)
        old_crossings = [
            crossing if group == "__all__" else positions[group][crossing - 1]
            for group, data in old["slices"].items()
            if (crossing := data["first_crossing_observation"]) is not None
        ]
        return new, min(old_crossings) if old_crossings else None

    baseline, baseline_old = summary(rows)
    scenarios = []
    for onset in onsets:
        for severity in SEVERITIES:
            rng = random.Random(seed + onset * 10 + int(severity * 100))
            changed = [dict(row) for row in rows]
            flips = 0
            for row in changed[onset:]:
                if not row["error"] and rng.random() < severity:
                    row["error"] = True
                    flips += 1
            new, old = summary(changed)
            alarm = new["first_alarm_global_observation"]
            scenarios.append({
                "kind": "synthetic_correct_to_error_label_flips",
                "onset_after_observations": onset, "flip_probability": severity,
                "flipped": flips, "restart_alarm": alarm, "from_start_alarm": old,
                "restart_pre_onset_alarm": alarm is not None and alarm <= onset,
                "restart_delay": alarm - onset if alarm is not None and alarm > onset else None,
                "from_start_delay": old - onset if old is not None and old > onset else None,
            })

    intervention = None
    if intervention_rows is not None:
        if (len(intervention_rows) != len(rows)
                or any(a["event_id"] != b["event_id"] or a["group"] != b["group"]
                       for a, b in zip(rows, intervention_rows))):
            raise ValueError("intervention must preserve every ID and pre-outcome group")
        intervention, intervention_old = summary(intervention_rows)
    else:
        intervention_old = None

    rng = random.Random(seed)
    new_alarms = old_alarms = 0
    for trial in range(null_trials):
        sim = [{"event_id": row["event_id"], "group": row["group"],
                "error": rng.random() < TARGET_ERROR} for row in rows]
        new, old = summary(sim)
        new_alarms += new["degradation_alarm"]
        old_alarms += old is not None
    return {
        "baseline_restart_watch": baseline,
        "baseline_from_start_alarm": baseline_old,
        "observed_intervention_restart_watch": intervention,
        "observed_intervention_from_start_alarm": intervention_old,
        "injected_label_flip_scenarios": scenarios,
        "synthetic_iid_conditional_null": {
            "trials": null_trials, "restart_alarms": new_alarms,
            "from_start_alarms": old_alarms,
            "generating_error_probability": TARGET_ERROR,
            "does_not_estimate_real_world_false_alarm_rate": True,
        },
    }


def run_change_point_audit(dataset_path, out_dir, *, acknowledge_model_calls=False,
                           expected_dataset_sha256=OFFICIAL_DATASET_SHA256,
                           qa_model=DEFAULT_QA_MODEL, qa_revision=None, engine=None,
                           offset=OFFSET, limit=LIMIT, starts=STARTS, onsets=ONSETS,
                           intervention_onset=INTERVENTION_ONSET,
                           context_cap=CONTEXT_CAP, null_trials=NULL_TRIALS):
    if not acknowledge_model_calls:
        raise ValueError("explicit model-call acknowledgement required")
    if (type(intervention_onset) is not int or not 0 < intervention_onset < limit
            or type(context_cap) is not int or context_cap < 1):
        raise ValueError("invalid preregistered intervention")
    dest = Path(out_dir)
    if dest.exists():
        raise FileExistsError("choose a fresh audit directory")
    # Validate pin before loading model. Gold is used only AFTER each call.
    questions, digest, _ = load_questions(dataset_path, limit, offset)
    if digest != expected_dataset_sha256 or len(questions) != limit:
        raise ValueError("dataset hash mismatch or insufficient reserved questions")
    qa = engine if engine is not None else ExtractiveQAEngine(qa_model, revision=qa_revision, threads=2)
    baseline_report = run_audit(
        dataset_path, dest / "baseline", acknowledge_model_calls=True,
        expected_dataset_sha256=expected_dataset_sha256, qa_model=qa_model,
        qa_revision=qa_revision, engine=qa, offset=offset, limit=limit,
        onsets=onsets, null_trials=1,
    )
    rows = [json.loads(line) for line in (dest / "baseline/outcomes.jsonl").read_text().splitlines()]
    changed = [dict(row) for row in rows]
    failures = verified = 0
    for index in range(intervention_onset, limit):
        _qid, _article, context, question, gold = questions[index]
        restricted_context = context[:context_cap]  # deterministic, gold-independent
        try:
            answer = qa.answer(restricted_context, question)
            if answer.get("model_revision") != qa.revision:
                raise RuntimeError("intervention model revision mismatch")
            if answer["accepted"]:
                if not verify_grounding(answer["proof"], restricted_context):
                    raise RuntimeError("intervention source certificate failed")
                verified += 1
            elif answer["answer"]:
                raise RuntimeError("unsupported answer on abstention")
            score = score_response(answer["answer"], gold)
            error = not (answer["accepted"] and score["f1"] >= 0.5)
        except (ValueError, RuntimeError):
            failures += 1
            error = True
        changed[index]["error"] = error
    analysis = analyze_restarts(
        rows, model_revision=qa.revision, starts=starts, onsets=onsets,
        intervention_rows=changed, null_trials=null_trials,
    )
    report = {
        "kind": "aru_disjoint_real_model_restart_research_audit",
        "dataset_sha256": digest, "offset": offset, "requested": limit,
        "model": qa.model_id, "model_revision": qa.revision,
        "rubric_id": RUBRIC, "target_error": TARGET_ERROR, "familywise_alpha": ALPHA,
        "groups": list(GROUPS), "source_group_cutoff_characters": SOURCE_CUTOFF,
        "starts": list(starts), "horizon": limit, "onsets": list(onsets),
        "baseline_model_calls": baseline_report["observed"],
        "baseline_errors": sum(row["error"] for row in rows),
        "baseline_verified_source_certificates": baseline_report["verified_source_certificates"],
        "baseline_model_call_failures": baseline_report["model_call_failures_counted_as_errors"],
        "intervention_kind": "actual_same_model_calls_with_context_truncated_before_inference",
        "intervention_onset_after_observations": intervention_onset,
        "intervention_context_cap_characters": context_cap,
        "intervention_additional_model_calls": limit - intervention_onset,
        "intervention_errors": sum(row["error"] for row in changed),
        "intervention_verified_source_certificates": verified,
        "intervention_model_call_failures": failures,
        "intervention_is_natural_live_drift": False,
        "intervention_restart_pre_onset_alarm": (
            analysis["observed_intervention_restart_watch"]["first_alarm_global_observation"] is not None
            and analysis["observed_intervention_restart_watch"]["first_alarm_global_observation"] <= intervention_onset
        ),
        "intervention_restart_detection_delay": (
            analysis["observed_intervention_restart_watch"]["first_alarm_global_observation"] - intervention_onset
            if analysis["observed_intervention_restart_watch"]["first_alarm_global_observation"] is not None
            and analysis["observed_intervention_restart_watch"]["first_alarm_global_observation"] > intervention_onset
            else None
        ),
        "intervention_from_start_detection_delay": (
            analysis["observed_intervention_from_start_alarm"] - intervention_onset
            if analysis["observed_intervention_from_start_alarm"] is not None
            and analysis["observed_intervention_from_start_alarm"] > intervention_onset
            else None
        ),
        "label_flip_scenarios_are_synthetic": True,
        "automatic_rollout_permitted": False,
        "limitations": [
            "Public development questions and the repository grader are not an independent customer trial.",
            "The context-limited model calls represent a deliberate retrieval intervention, not spontaneous live drift.",
            "Label-flip stress and Bernoulli null streams are synthetic, not model outcomes.",
            "Shared articles and incomplete or delayed feedback may violate conditional-null assumptions.",
            "The 50% risk target is a severe-degradation stress target, not a commercial quality promise.",
            "Neither a no-alarm result nor this unsigned research report authorizes deployment.",
        ],
        **analysis,
    }
    (dest / "intervention_outcomes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in changed), encoding="utf-8",
    )
    (dest / "change_point_report.json").write_text(
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
    report = run_change_point_audit(
        args.dataset, args.out_dir, qa_model=args.qa_model,
        qa_revision=args.qa_revision, acknowledge_model_calls=args.acknowledge_model_calls,
    )
    print(json.dumps({
        "baseline_errors": report["baseline_errors"],
        "baseline_restart_alarm": report["baseline_restart_watch"]["first_alarm_global_observation"],
        "intervention_errors": report["intervention_errors"],
        "intervention_restart_alarm": report["observed_intervention_restart_watch"]["first_alarm_global_observation"],
        "synthetic_stress": report["injected_label_flip_scenarios"],
        "automatic_rollout_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

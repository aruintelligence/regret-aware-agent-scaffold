"""Run a real human-annotated SQuAD sentence-retrieval comparison.

Gold answers are withheld from both algorithms. Two routes run on each
question for a full-information audit; randomized logging reveals only the
chosen route's reward to the off-policy evaluator. This is not an LLM test.

Run:
  python squad_benchmark.py --dataset dev-v1.1.json --out-dir squad-results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from evidence_dashboard import save_dashboard
from evidence_lab import evaluate_experiment
from route_experiments import action_id, append_jsonl, validate_decision
from squad_sentence_retrieval import (
    SOURCE_URL, gold_span_contained, idf_proximity_sentence,
    load_questions, overlap_sentence,
)

BASELINE_ROUTE = ["token_overlap_sentence_retriever"]
CANDIDATE_ROUTE = ["idf_proximity_sentence_retriever"]


def paired_oracle_interval(pairs, *, replicates=1000, seed=13):
    """Exploratory paired bootstrap with both route outcomes fully observed."""
    if replicates < 100 or not pairs:
        raise ValueError("need at least 100 replicates and nonempty pairs")
    rng = random.Random(seed)
    n = len(pairs)
    deltas = []
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(pairs[i][1] - pairs[i][0] for i in indices) / n)
    deltas.sort()
    def percentile(p):
        index = p * (len(deltas) - 1)
        lo = int(index)
        hi = min(lo + 1, len(deltas) - 1)
        return deltas[lo] + (deltas[hi] - deltas[lo]) * (index - lo)
    return {"lower_95": percentile(.025), "upper_95": percentile(.975),
            "replicates": replicates, "method": "paired_question_bootstrap_exploratory"}


def cluster_oracle_interval(groups, *, replicates=1000, seed=13):
    """Resample whole articles/paragraphs, preserving paired outcomes and within-cluster dependence."""
    if replicates < 100 or not groups:
        raise ValueError("need nonempty clusters and >=100 resamples")
    keys = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        total = sum(len(groups[key]) for key in sampled)
        numerator = sum(sum(b - a for a, b in groups[key]) for key in sampled)
        deltas.append(numerator / total)
    deltas.sort()
    def percentile(p):
        index = p * (len(deltas) - 1)
        lo = int(index)
        hi = min(lo + 1, len(deltas) - 1)
        return deltas[lo] + (deltas[hi] - deltas[lo]) * (index - lo)
    return {"lower_95": percentile(.025), "upper_95": percentile(.975),
            "clusters": len(keys), "replicates": replicates,
            "method": "paired_cluster_bootstrap_exploratory"}


def run_benchmark(dataset_path, output_dir, *, limit=2000,
                  seed=20261018, replicates=500):
    questions, dataset_sha256, total = load_questions(dataset_path, limit)
    if not questions:
        raise ValueError("dataset contains no answerable questions")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "decisions.benchmark.jsonl"
    labels_path = output / "labels.benchmark.jsonl"
    report_path = output / "report.json"
    oracle_path = output / "oracle.json"
    dashboard_path = output / "index.html"
    if any(p.exists() for p in (decisions_path, labels_path, report_path,
                                oracle_path, dashboard_path)):
        raise FileExistsError("output already exists; choose a fresh directory")
    rng = random.Random(seed)
    baseline_action = action_id(BASELINE_ROUTE)
    candidate_action = action_id(CANDIDATE_ROUTE)
    paired = []
    by_article = defaultdict(list)
    by_paragraph = defaultdict(list)
    route_time = [0.0, 0.0]
    action_counts = [0, 0]
    examples = []
    for question_id, title, context, question, answers in questions:
        started = time.perf_counter()
        baseline_span = overlap_sentence(question, context)
        route_time[0] += time.perf_counter() - started
        started = time.perf_counter()
        candidate_span = idf_proximity_sentence(question, context)
        route_time[1] += time.perf_counter() - started
        outcomes = (
            int(gold_span_contained(baseline_span, answers)),
            int(gold_span_contained(candidate_span, answers)),
        )
        paired.append(outcomes)
        by_article[title].append(outcomes)
        by_paragraph[hashlib.sha256(context.encode("utf-8")).hexdigest()].append(outcomes)
        # Equal-probability logging: the evaluator sees only the chosen reward.
        chosen = rng.randrange(2)
        action_counts[chosen] += 1
        decision_id = "squad-" + hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:24]
        decision = {
            "schema_version": 1,
            "decision_id": decision_id,
            "timestamp_unix": 0,
            "task_signature": "squad1.1|" + title,
            "logging_policy": "categorical",
            "candidate_probabilities": [0.5, 0.5],
            "propensity": 0.5,
            "action": (baseline_action, candidate_action)[chosen],
            "candidates": [
                {"route": BASELINE_ROUTE, "action": baseline_action, "predicted_score": 0.5},
                {"route": CANDIDATE_ROUTE, "action": candidate_action, "predicted_score": 0.5},
            ],
            "runtime_kind": "squad1.1_human_annotated_sentence_retrieval",
            "loop_count": 0,
        }
        validate_decision(decision)
        append_jsonl(str(decisions_path), decision)
        append_jsonl(str(labels_path), {
            "schema_version": 1,
            "decision_id": decision_id,
            "reward": float(outcomes[chosen]),
            "grader": "stanford_squad1.1_gold_answer_offset",
            "rubric_id": "selected_sentence_contains_human_answer_span_v1",
            "source_kind": "external",
            "graded_at_unix": 0,
        })
        if len(examples) < 10:
            examples.append({"question_id": question_id, "article": title,
                             "baseline_correct": bool(outcomes[0]),
                             "candidate_correct": bool(outcomes[1])})
    n = len(paired)
    baseline_rate = sum(a for a, _ in paired) / n
    candidate_rate = sum(b for _, b in paired) / n
    both_correct = sum(a and b for a, b in paired)
    baseline_only = sum(a and not b for a, b in paired)
    candidate_only = sum(b and not a for a, b in paired)
    both_wrong = n - both_correct - baseline_only - candidate_only
    article_diagnostics = [
        {"article": title, "n": len(group),
         "baseline_accuracy": sum(a for a, _ in group) / len(group),
         "candidate_accuracy": sum(b for _, b in group) / len(group),
         "difference": sum(b - a for a, b in group) / len(group)}
        for title, group in sorted(by_article.items())
    ]
    negative_articles = [row for row in article_diagnostics
                         if row["n"] >= 20 and row["difference"] < -0.05]
    direct = {
        "dataset": "Stanford SQuAD v1.1 development set",
        "dataset_source": SOURCE_URL,
        "dataset_sha256": dataset_sha256,
        "dataset_total_answerable_questions": total,
        "n_evaluated": n,
        "selection_method": "ascending SHA-256 of official question ID",
        "random_logging_seed": seed,
        "baseline_algorithm": "unique_question_token_overlap_normalized_by_sentence_length",
        "candidate_algorithm": "idf_bm25_plus_coverage_and_proximity",
        "task_metric": "selected_sentence_fully_contains_any_human_annotated_answer_span",
        "baseline_accuracy": baseline_rate,
        "candidate_accuracy": candidate_rate,
        "difference": candidate_rate - baseline_rate,
        "both_correct": both_correct,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": both_wrong,
        "logged_action_counts": {"baseline": action_counts[0], "candidate": action_counts[1]},
        "mean_latency_ms": {"baseline": route_time[0] * 1000 / n,
                            "candidate": route_time[1] * 1000 / n},
        "paired_question_interval": paired_oracle_interval(paired, replicates=replicates, seed=seed + 1),
        "paired_paragraph_cluster_interval": cluster_oracle_interval(by_paragraph, replicates=replicates, seed=seed + 2),
        "paired_article_cluster_interval": cluster_oracle_interval(by_article, replicates=replicates, seed=seed + 3),
        "article_diagnostics": article_diagnostics,
        "negative_article_count": len(negative_articles),
        "audit_examples": examples,
        "scope": "human-annotated sentence retrieval, not LLM question answering",
    }
    report = evaluate_experiment(
        str(decisions_path), str(labels_path),
        baseline_policy="first_candidate", candidate_policy="second_candidate",
        replicates=replicates, seed=seed,
    )
    report["external_benchmark"] = direct
    if negative_articles:
        report["review_gate"]["candidate_for_human_review"] = False
        report["review_gate"]["reasons"].insert(
            0, "Full-information article-level regressions observed in "
            + str(len(negative_articles)) + " topic(s); investigate before review"
        )
    report["full_information_crosscheck"] = {
        "direct_difference": direct["difference"],
        "logged_snips_difference": report["paired_difference"]["snips_difference"],
        "absolute_estimator_error": abs(direct["difference"] -
                                        report["paired_difference"]["snips_difference"])
        if report["paired_difference"]["snips_difference"] is not None else None,
    }
    report["limitations"].append(
        "SQuAD is a human-annotated benchmark but this test measures sentence retrieval only, not full answer extraction or LLM ability."
    )
    report["limitations"].append(
        "Article/paragraph cluster bootstrap intervals account for within-cluster correlation only approximately; they are exploratory, not confirmatory."
    )
    report["limitations"].append(
        "Full-information outcomes are used solely for the offline audit; only the randomized chosen reward enters the off-policy estimator."
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    oracle_path.write_text(json.dumps(direct, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_dashboard(report, str(dashboard_path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to official SQuAD v1.1 dev JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    report = run_benchmark(args.dataset, args.out_dir, limit=args.limit,
                           seed=args.seed, replicates=args.replicates)
    print(json.dumps({
        "dataset_sha256": report["external_benchmark"]["dataset_sha256"],
        "n_evaluated": report["n_labeled"],
        "direct_baseline_accuracy": report["external_benchmark"]["baseline_accuracy"],
        "direct_candidate_accuracy": report["external_benchmark"]["candidate_accuracy"],
        "direct_difference": report["external_benchmark"]["difference"],
        "logged_snips_difference": report["paired_difference"]["snips_difference"],
        "review_gate": report["review_gate"],
        "dashboard": str(Path(args.out_dir) / "index.html"),
    }, indent=2))


if __name__ == "__main__":
    main()

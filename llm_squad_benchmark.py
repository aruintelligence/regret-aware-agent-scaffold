"""Opt-in evaluation of a REAL local/OpenAI-compatible LLM on SQuAD.

This requires an operator-supplied model server and explicit acknowledgement.
The model sees only passage + question, never human gold answers. The official
gold answers grade the selected route after execution. No automatic rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

from evidence_dashboard import save_dashboard
from evidence_lab import evaluate_experiment
from llm_executor import EndpointError, run_chat, validate_endpoint
from route_experiments import action_id, append_jsonl, validate_decision
from squad_sentence_retrieval import load_questions

DIRECT_ROUTE = ["llm_direct_question_answer"]
EXTRACTIVE_ROUTE = ["llm_grounded_extract_answer"]


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def squad_f1(prediction, gold):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_response(prediction, answers):
    golds = [a["text"] for a in answers]
    return {
        "f1": max(squad_f1(prediction, gold) for gold in golds),
        "exact_match": float(any(normalize_answer(prediction) == normalize_answer(gold)
                                 for gold in golds)),
    }


def build_messages(context, question, route_index):
    if route_index == 0:
        system = (
            "Answer the user's question using only the supplied passage. "
            "Output only the short answer, without explanations or extra text. "
            "If the passage lacks an answer, output UNKNOWN."
        )
    elif route_index == 1:
        system = (
            "You are an extractive question-answering engine. "
            "Find the shortest exact text span from the passage that answers "
            "the question. Return only that span verbatim. No explanations, "
            "no invented facts. If absent, return UNKNOWN."
        )
    else:
        raise ValueError("unsupported route index")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Passage:\n" + context + "\n\nQuestion:\n" + question},
    ]


def run_llm_benchmark(dataset_path, output_dir, *, model,
                      endpoint="http://127.0.0.1:1234/v1/chat/completions",
                      allow_remote=False, acknowledge_model_calls=False,
                      limit=20, seed=20261018, timeout_s=60,
                      max_tokens=96, replicates=300):
    if not acknowledge_model_calls:
        raise ValueError("explicit acknowledgement required before model calls")
    if not 1 <= limit <= 500:
        raise ValueError("research limit must be between 1 and 500 requests")
    validate_endpoint(endpoint, allow_remote=allow_remote)
    questions, dataset_sha, total = load_questions(dataset_path, limit)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "decisions.llm.jsonl"
    labels_path = output / "labels.llm.jsonl"
    failures_path = output / "failures.llm.jsonl"
    report_path = output / "report.json"
    dashboard_path = output / "index.html"
    if any(p.exists() for p in (decisions_path, labels_path, failures_path,
                                report_path, dashboard_path)):
        raise FileExistsError("output files already exist; choose fresh directory")
    rng = random.Random(seed)
    routes = (DIRECT_ROUTE, EXTRACTIVE_ROUTE)
    ids = [action_id(route) for route in routes]
    observed = {0: {"requests": 0, "labeled": 0, "f1_sum": 0.0, "em_sum": 0.0,
                    "latency_s_sum": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
                1: {"requests": 0, "labeled": 0, "f1_sum": 0.0, "em_sum": 0.0,
                    "latency_s_sum": 0.0, "prompt_tokens": 0, "completion_tokens": 0}}
    for qid, title, context, question, answers in questions:
        chosen = rng.randrange(2)
        decision_id = "llm-squad-" + hashlib.sha256(qid.encode("utf-8")).hexdigest()[:24]
        decision = {
            "schema_version": 1,
            "decision_id": decision_id,
            "timestamp_unix": time.time(),
            "task_signature": "squad1.1|" + title,
            "logging_policy": "categorical",
            "candidate_probabilities": [0.5, 0.5],
            "propensity": 0.5,
            "action": ids[chosen],
            "candidates": [
                {"route": route, "action": action_id(route), "predicted_score": 0.5}
                for route in routes
            ],
            "runtime_kind": "opt_in_openai_compatible_llm_squad_qa",
            "loop_count": 0,
        }
        validate_decision(decision)
        # Record the decision BEFORE the model sees the task.
        append_jsonl(str(decisions_path), decision)
        observed[chosen]["requests"] += 1
        started = time.perf_counter()
        try:
            result = run_chat(
                build_messages(context, question, chosen),
                model=model, endpoint=endpoint, allow_remote=allow_remote,
                timeout_s=timeout_s, max_tokens=max_tokens,
            )
        except EndpointError as exc:
            append_jsonl(str(failures_path), {
                "decision_id": decision_id,
                "error_type": type(exc).__name__,
                "no_outcome_label": True,
            })
            continue
        elapsed = time.perf_counter() - started
        scores = score_response(result["text"], answers)
        append_jsonl(str(labels_path), {
            "schema_version": 1,
            "decision_id": decision_id,
            "reward": scores["f1"],
            "grader": "squad1.1_human_gold_token_f1",
            "rubric_id": "squad_v1.1_normalized_token_f1",
            "source_kind": "external",
            "graded_at_unix": time.time(),
        })
        arm = observed[chosen]
        arm["labeled"] += 1
        arm["f1_sum"] += scores["f1"]
        arm["em_sum"] += scores["exact_match"]
        arm["latency_s_sum"] += elapsed
        for field in ("prompt_tokens", "completion_tokens"):
            arm[field] += result[field] or 0
    if not sum(arm["labeled"] for arm in observed.values()):
        raise RuntimeError("no completed model calls; decision logs retained without fabricated labels")
    report = evaluate_experiment(
        str(decisions_path), str(labels_path),
        baseline_policy="first_candidate", candidate_policy="second_candidate",
        replicates=replicates, seed=seed,
    )
    report["llm_benchmark"] = {
        "model": model,
        "endpoint_location": "remote_https" if allow_remote else "loopback",
        "dataset_sha256": dataset_sha,
        "dataset_total_answerable_questions": total,
        "task": "SQuAD v1.1 extractive question answering",
        "n_requested": len(questions),
        "n_completed": sum(arm["labeled"] for arm in observed.values()),
        "n_failed": sum(arm["requests"] - arm["labeled"] for arm in observed.values()),
        "observed_arms": {
            name: {
                "requests": arm["requests"],
                "completed": arm["labeled"],
                "observed_mean_f1": arm["f1_sum"] / arm["labeled"] if arm["labeled"] else None,
                "observed_exact_match": arm["em_sum"] / arm["labeled"] if arm["labeled"] else None,
                "mean_latency_s": arm["latency_s_sum"] / arm["labeled"] if arm["labeled"] else None,
                "reported_prompt_tokens": arm["prompt_tokens"],
                "reported_completion_tokens": arm["completion_tokens"],
            }
            for name, arm in (("direct", observed[0]), ("grounded_extract", observed[1]))
        },
        "warning": "Real LLM benchmark if a genuine model endpoint was supplied; mocked endpoints do not validate model ability.",
    }
    report["limitations"].append(
        "This optional LLM benchmark is a public development-set experiment; it is not a hidden leaderboard test."
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_dashboard(report, str(dashboard_path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--acknowledge-model-calls", action="store_true",
                        help="Explicitly authorize sending SQuAD tasks to the supplied endpoint")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    report = run_llm_benchmark(
        args.dataset, args.out_dir, model=args.model,
        endpoint=args.endpoint, allow_remote=args.allow_remote,
        acknowledge_model_calls=args.acknowledge_model_calls,
        limit=args.limit, seed=args.seed,
        timeout_s=args.timeout_s, max_tokens=args.max_tokens,
    )
    print(json.dumps({
        "n_completed": report["llm_benchmark"]["n_completed"],
        "n_failed": report["llm_benchmark"]["n_failed"],
        "baseline_snips": report["baseline"]["snips"],
        "candidate_snips": report["candidate"]["snips"],
        "human_review_only": report["review_gate"]["automatic_rollout_permitted"] is False,
    }, indent=2))


if __name__ == "__main__":
    main()

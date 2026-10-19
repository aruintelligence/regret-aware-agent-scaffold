"""Evaluate a genuine pretrained extractive QA model and local LLM on identical public SQuAD questions.

Gold answers are withheld until after both model calls. Reports contain only
aggregates and provenance. This is exploratory public-dev-set evidence, not
a hidden holdout or proof of general intelligence.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from html import escape
from pathlib import Path

from extractive_qa_engine import DEFAULT_QA_MODEL, ExtractiveQAEngine
from grounding_contract import ground_answer, verify_grounding
from llm_executor import EndpointError, run_chat, validate_endpoint
from llm_squad_benchmark import build_messages, score_response
from squad_benchmark import cluster_oracle_interval, paired_oracle_interval
from squad_sentence_retrieval import load_questions


def evaluate_qa_models(dataset_path, out_dir, *, qa_model=DEFAULT_QA_MODEL,
                       qa_revision=None, llm_model,
                       llm_endpoint="http://127.0.0.1:1234/v1/chat/completions",
                       acknowledge_model_calls=False, limit=80, offset=120,
                       seed=20261018, timeout_s=90, max_tokens=48,
                       replicates=400, engine=None):
    if not acknowledge_model_calls:
        raise ValueError("explicit model-call acknowledgement required")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be in [1,500]")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be nonnegative")
    validate_endpoint(llm_endpoint)
    questions, dataset_hash, total = load_questions(dataset_path, limit, offset)
    if len(questions) != limit:
        raise ValueError("not enough questions in reserved subset")
    out = Path(out_dir)
    if (out / "report.json").exists() or (out / "index.html").exists():
        raise FileExistsError("choose a fresh report directory")
    qa = engine if engine is not None else ExtractiveQAEngine(
        qa_model, revision=qa_revision, threads=2,
    )
    if qa.model_id != qa_model or not qa.revision:
        raise ValueError("QA model identifier and immutable revision required")
    arms = {name: {"requested": 0, "completed": 0, "accepted": 0,
                   "f1_sum": 0.0, "em_sum": 0.0, "latency_sum": 0.0}
            for name in ("qa", "llm_raw", "llm_gated")}
    pairs, gated_pairs = [], []
    article_pairs = defaultdict(list)
    failures = {"qa": 0, "llm": 0}
    for _qid, article, context, question, gold in questions:
        observed = {}
        arms["qa"]["requested"] += 1
        start = time.perf_counter()
        try:
            answer = qa.answer(context, question)
        except (ValueError, RuntimeError):
            failures["qa"] += 1
        else:
            elapsed = time.perf_counter() - start
            if answer["accepted"] and not verify_grounding(answer["proof"], context):
                raise RuntimeError("invalid independent QA source certificate")
            score = score_response(answer["answer"], gold)
            observed["qa"] = score
            arm = arms["qa"]
            arm["completed"] += 1
            arm["accepted"] += int(answer["accepted"])
            arm["f1_sum"] += score["f1"]
            arm["em_sum"] += score["exact_match"]
            arm["latency_sum"] += elapsed
        arms["llm_raw"]["requested"] += 1
        arms["llm_gated"]["requested"] += 1
        start = time.perf_counter()
        try:
            answer = run_chat(
                build_messages(context, question, 0), model=llm_model,
                endpoint=llm_endpoint, timeout_s=timeout_s, max_tokens=max_tokens,
            )
        except EndpointError:
            failures["llm"] += 1
        else:
            elapsed = time.perf_counter() - start
            raw = score_response(answer["text"], gold)
            proof = ground_answer(answer["text"], context)
            if proof["accepted"] and not verify_grounding(proof, context):
                raise RuntimeError("invalid independent LLM source certificate")
            gated = score_response(proof["answer"], gold)
            observed["llm_raw"] = raw
            observed["llm_gated"] = gated
            for key, result, accepted in (
                ("llm_raw", raw, True),
                ("llm_gated", gated, proof["accepted"]),
            ):
                arm = arms[key]
                arm["completed"] += 1
                arm["accepted"] += int(accepted)
                arm["f1_sum"] += result["f1"]
                arm["em_sum"] += result["exact_match"]
                arm["latency_sum"] += elapsed
        if len(observed) == 3:
            pair = (observed["llm_raw"]["f1"], observed["qa"]["f1"])
            gated_pair = (observed["llm_gated"]["f1"], observed["qa"]["f1"])
            pairs.append(pair)
            gated_pairs.append(gated_pair)
            article_pairs[article].append(pair)
    if len(pairs) < 2:
        raise RuntimeError("fewer than two complete matched model outcomes")
    stats = {
        name: {
            "requested": a["requested"], "completed": a["completed"],
            "accepted": a["accepted"],
            "accepted_fraction": a["accepted"] / a["completed"] if a["completed"] else None,
            "mean_f1": a["f1_sum"] / a["completed"] if a["completed"] else None,
            "exact_match": a["em_sum"] / a["completed"] if a["completed"] else None,
            "mean_latency_s": a["latency_sum"] / a["completed"] if a["completed"] else None,
        }
        for name, a in arms.items()
    }
    report = {
        "kind": "real_pretrained_extractive_qa_vs_small_llm_public_dev",
        "dataset": "Stanford SQuAD v1.1 public development split",
        "dataset_sha256": dataset_hash,
        "dataset_total_answerable_questions": total,
        "selection": "SHA256 sorted question IDs, reserved offset and limit",
        "offset": offset, "limit": limit, "seed": seed,
        "qa_model": qa.model_id, "qa_revision": qa.revision,
        "llm_model": llm_model,
        "qa_model_training": "pretrained and fine-tuned for extractive question answering",
        "gold_answers_visible_to_models": False,
        "n_complete_pairs": len(pairs),
        "failures": failures,
        "arms": stats,
        "direct_qa_minus_raw_llm_f1": sum(b - a for a, b in pairs) / len(pairs),
        "direct_qa_minus_gated_llm_f1": sum(b - a for a, b in gated_pairs) / len(pairs),
        "paired_question_interval": paired_oracle_interval(pairs, replicates=replicates, seed=seed + 1),
        "article_cluster_interval": cluster_oracle_interval(article_pairs, replicates=replicates, seed=seed + 2),
        "automatic_rollout_permitted": False,
        "limitations": [
            "Public development set, not a hidden independent holdout.",
            "Specialized QA model is trained for this task; tiny generative model is not.",
            "A source certificate proves a contiguous substring, not truth, relevance or entailment.",
            "Missing model outcomes may bias complete-pair comparisons.",
            "Cluster bootstrap intervals are exploratory and may be unstable.",
            "No general intelligence, customer ROI or enterprise security claim.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render_dashboard(report), encoding="utf-8")
    return report

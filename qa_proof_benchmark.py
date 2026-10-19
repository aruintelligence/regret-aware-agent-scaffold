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


def render_dashboard(report):
    def pct(v):
        return f"{100 * v:.1f}%" if v is not None else "N/A"
    qa, llm, gate = (report["arms"][key] for key in ("qa", "llm_raw", "llm_gated"))
    ci = report["article_cluster_interval"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ĀRU · Real Extractive QA Evidence</title>
<style>:root{{color-scheme:dark;--bg:#071321;--panel:#11283c;--line:#2e4d5e;--cyan:#75e6e3;--gold:#efc87f;--ink:#f1f8fb;--muted:#b6cad5}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 10% 0%,#17394c,transparent 65%),var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:60px 25px}}h1{{font-size:clamp(38px,6vw,70px);line-height:1.08;letter-spacing:-.055em}}
h2{{font-size:27px}}.eyebrow{{font-size:12px;color:var(--cyan);letter-spacing:.16em;font-weight:800;text-transform:uppercase}}
.lede,.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:30px 0}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:25px}}.card strong{{display:block;font-size:42px;letter-spacing:-.05em}}
.card small{{color:var(--muted)}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--muted)}}.warning{{border-left:4px solid var(--gold);padding:18px 24px;background:#30291f;margin-top:25px}}
code{{overflow-wrap:anywhere}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}main{{padding:30px 16px}}}}</style></head>
<body><main><div class="eyebrow">ĀRU Intelligence · Real pretrained models · Human-gold evaluation</div>
<h1>Answers with<br>checkable evidence.</h1>
<p class="lede">A specialized pretrained extractive QA model and a small generative language model answered the same {report["n_complete_pairs"]} public questions. Human answers were used only for scoring after generation.</p>
<div class="grid">
<div class="card"><small>Specialized QA mean F1</small><strong>{pct(qa["mean_f1"])}</strong></div>
<div class="card"><small>Small LLM mean F1</small><strong>{pct(llm["mean_f1"])}</strong></div>
<div class="card"><small>Verified QA source coverage</small><strong>{pct(qa["accepted_fraction"])}</strong></div>
</div>
<section class="panel"><h2>Accuracy and provenance tradeoffs</h2>
<table><tr><th>Measure</th><th>Specialized QA</th><th>Small LLM</th><th>LLM + source gate</th></tr>
<tr><td>Mean token F1</td><td>{pct(qa["mean_f1"])}</td><td>{pct(llm["mean_f1"])}</td><td>{pct(gate["mean_f1"])}</td></tr>
<tr><td>Exact match</td><td>{pct(qa["exact_match"])}</td><td>{pct(llm["exact_match"])}</td><td>{pct(gate["exact_match"])}</td></tr>
<tr><td>Source-backed coverage</td><td>{pct(qa["accepted_fraction"])}</td><td>Not enforced</td><td>{pct(gate["accepted_fraction"])}</td></tr>
<tr><td>Mean inference latency</td><td>{qa["mean_latency_s"]:.2f}s</td><td>{llm["mean_latency_s"]:.2f}s</td><td>{gate["mean_latency_s"]:.2f}s</td></tr></table>
<p class="muted">Direct paired QA minus raw LLM F1: <strong>{100 * report["direct_qa_minus_raw_llm_f1"]:+.2f} percentage points</strong>.</p>
<p class="muted">Exploratory article-cluster interval: {100 * ci["lower_95"]:+.2f} to {100 * ci["upper_95"]:+.2f} points ({ci["clusters"]} articles).</p></section>
<section class="panel"><h2>What the certificate proves</h2>
<p>Accepted QA answers carry independently verifiable original source offsets and SHA-256 fingerprints. A source can still be false or irrelevant. Provenance is not factual entailment.</p>
<p class="muted">Model: <code>{escape(report["qa_model"])}</code> · revision: <code>{escape(str(report["qa_revision"]))}</code>.</p>
<p class="muted">Dataset SHA-256: <code>{escape(report["dataset_sha256"])}</code> · offset {report["offset"]}.</p></section>
<div class="warning"><strong>Exploratory research only.</strong> The specialized QA model was fine-tuned for this task; the comparator is a tiny general-purpose LLM. No general reasoning, customer productivity or enterprise deployment claims are supported. Automatic rollout is disabled.</div>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--qa-revision")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--acknowledge-model-calls", action="store_true")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--offset", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--timeout-s", type=float, default=90)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    report = evaluate_qa_models(
        args.dataset, args.out_dir, qa_model=args.qa_model,
        qa_revision=args.qa_revision, llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
        acknowledge_model_calls=args.acknowledge_model_calls,
        limit=args.limit, offset=args.offset, seed=args.seed,
        timeout_s=args.timeout_s, max_tokens=args.max_tokens,
    )
    print(json.dumps({
        "qa_model": report["qa_model"],
        "qa_revision": report["qa_revision"],
        "n_complete_pairs": report["n_complete_pairs"],
        "arms": report["arms"],
        "qa_minus_raw_llm_f1": report["direct_qa_minus_raw_llm_f1"],
        "article_cluster_interval": report["article_cluster_interval"],
        "automatic_rollout_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

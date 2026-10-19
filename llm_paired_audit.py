"""Follow-up paired audit of two real LLM prompts on the SAME public questions.

This is an exploratory full-information evaluation: each question is sent
once to each frozen prompt strategy, in seeded randomized order. The human
SQuAD answers are withheld until AFTER both model outputs are generated.
Results are aggregate-only; no raw passages, answers, or model text logged.

A 40-question public development subset is not a statistically powered
holdout and cannot prove one prompt or model is generally better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from html import escape
from pathlib import Path

from llm_executor import EndpointError, run_chat, validate_endpoint
from llm_squad_benchmark import build_messages, score_response
from squad_benchmark import cluster_oracle_interval, paired_oracle_interval
from squad_sentence_retrieval import load_questions


def render_paired_dashboard(report):
    baseline = report["direct"]
    ci = report["article_cluster_interval"]
    def pct(v):
        return f"{100*v:.1f}%"
    def points(v):
        return f"{100*v:+.2f} pp"
    model = escape(report["model"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ĀRU · Real Model Paired Audit</title><style>
:root{{color-scheme:dark;--bg:#081421;--card:#122638;--line:#2e4a5d;--ink:#f3f9fb;--muted:#b4c8d5;--cyan:#72e4e1;--gold:#f3ce89}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 15% 0%,#183a4b,transparent 60%),var(--bg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}}
main{{max-width:1020px;margin:auto;padding:55px 24px}}.eyebrow{{color:var(--cyan);font-size:12px;letter-spacing:.2em;font-weight:800;text-transform:uppercase}}
h1{{font-size:clamp(36px,6vw,70px);letter-spacing:-.055em;line-height:1.1;margin:18px 0}}h2{{font-size:25px;letter-spacing:-.03em}}
.lead,.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:32px 0}}
.card{{background:var(--card);border:1px solid var(--line);padding:25px;border-radius:18px}}.card strong{{font-size:clamp(28px,4vw,44px);display:block;letter-spacing:-.04em}}.card small{{color:var(--muted)}}
.panel{{background:var(--card);border:1px solid var(--line);padding:28px;border-radius:18px;margin:22px 0}}
.notice{{border-left:4px solid var(--gold);background:#30291e;padding:18px 24px;border-radius:4px;color:#f6e7cc}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:13px 9px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}code{{overflow-wrap:anywhere}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}main{{padding:30px 18px}}}}
</style></head><body><main>
<div class="eyebrow">ĀRU Intelligence · Independent human-gold follow-up</div>
<h1>Two prompts.<br>Exactly the same questions.</h1>
<p class="lead">Actual open-weight model execution. Both frozen prompt strategies answered the same {report["n_complete_pairs"]} public questions; gold answers were used only for scoring afterward.</p>
<div class="grid">
<div class="card"><small>Direct prompt · mean F1</small><strong>{pct(baseline["baseline_mean_f1"])}</strong></div>
<div class="card"><small>Grounded extractive · mean F1</small><strong>{pct(baseline["candidate_mean_f1"])}</strong></div>
<div class="card"><small>Observed paired difference</small><strong style="color:var(--cyan)">{points(baseline["f1_difference"])}</strong></div>
</div>
<section class="panel"><h2>Matched evaluation</h2>
<table><tr><th>Metric</th><th>Direct</th><th>Grounded extractive</th></tr>
<tr><td>Exact match</td><td>{pct(baseline["baseline_exact_match"])}</td><td>{pct(baseline["candidate_exact_match"])}</td></tr>
<tr><td>Mean token F1</td><td>{pct(baseline["baseline_mean_f1"])}</td><td>{pct(baseline["candidate_mean_f1"])}</td></tr>
<tr><td>Mean inference latency</td><td>{report["arms"]["direct"]["mean_latency_s"]:.2f}s</td><td>{report["arms"]["grounded_extract"]["mean_latency_s"]:.2f}s</td></tr></table>
<p class="muted">Exploratory article-cluster bootstrap 95% interval: <strong>{points(ci["lower_95"])} to {points(ci["upper_95"])}</strong> ({ci["clusters"]} article clusters).</p>
<p class="muted">Model: <code>{model}</code> · SHA-sorted public question offset: {report["offset"]} · Dataset SHA-256: <code>{escape(report["dataset_sha256"])}</code>.</p></section>
<div class="notice"><strong>Research limitations:</strong> {report["n_complete_pairs"]} questions is a small, public development-set sample, not an independent private benchmark or statistically powered trial. Cluster intervals are exploratory. No automatic rollout or enterprise performance claim is justified.</div>
<p class="muted">Total calls: {report["n_requested_calls"]}; completed: {report["n_completed_calls"]}; failed: {report["n_failed_calls"]}. No raw model responses or question text are included in this report.</p>
</main></body></html>"""


def run_paired_audit(dataset_path, out_dir, *, model,
                     endpoint="http://127.0.0.1:1234/v1/chat/completions",
                     allow_remote=False, acknowledge_model_calls=False,
                     limit=40, offset=40, seed=20261018,
                     timeout_s=90, max_tokens=48, replicates=400):
    if not acknowledge_model_calls:
        raise ValueError("explicit model-call acknowledgement required")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be in [1,500]")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be nonnegative")
    validate_endpoint(endpoint, allow_remote=allow_remote)
    questions, dataset_sha, total = load_questions(dataset_path, limit, offset)
    if len(questions) != limit:
        raise ValueError("not enough questions in requested predeclared subset")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "paired-report.json"
    dashboard_path = output / "paired-index.html"
    if report_path.exists() or dashboard_path.exists():
        raise FileExistsError("paired report already exists; choose fresh directory")
    rng = random.Random(seed)
    scores = []
    by_article = defaultdict(list)
    by_paragraph = defaultdict(list)
    arms = {
        0: {"requests": 0, "completed": 0, "f1_sum": 0.0,
            "exact_match_sum": 0.0, "latency_sum": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0},
        1: {"requests": 0, "completed": 0, "f1_sum": 0.0,
            "exact_match_sum": 0.0, "latency_sum": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0},
    }
    failed = 0
    for _qid, title, context, question, answers in questions:
        first = rng.randrange(2)
        order = (first, 1 - first)
        outcome = {}
        for route in order:
            arms[route]["requests"] += 1
            started = time.perf_counter()
            try:
                response = run_chat(
                    build_messages(context, question, route),
                    model=model, endpoint=endpoint, allow_remote=allow_remote,
                    timeout_s=timeout_s, max_tokens=max_tokens,
                )
            except EndpointError:
                failed += 1
                continue
            elapsed = time.perf_counter() - started
            result = score_response(response["text"], answers)
            outcome[route] = result
            arm = arms[route]
            arm["completed"] += 1
            arm["f1_sum"] += result["f1"]
            arm["exact_match_sum"] += result["exact_match"]
            arm["latency_sum"] += elapsed
            for field in ("prompt_tokens", "completion_tokens"):
                arm[field] += response[field] or 0
        if len(outcome) == 2:
            pair = (outcome[0]["f1"], outcome[1]["f1"])
            scores.append((pair, (outcome[0]["exact_match"], outcome[1]["exact_match"])))
            by_article[title].append(pair)
            by_paragraph[hashlib.sha256(context.encode("utf-8")).hexdigest()].append(pair)
    if len(scores) < 2:
        raise RuntimeError("fewer than two complete pairs; no valid paired audit")
    pairs = [row[0] for row in scores]
    exact = [row[1] for row in scores]
    n = len(pairs)
    direct = {
        "baseline_mean_f1": sum(a for a, _ in pairs) / n,
        "candidate_mean_f1": sum(b for _, b in pairs) / n,
        "f1_difference": sum(b - a for a, b in pairs) / n,
        "baseline_exact_match": sum(a for a, _ in exact) / n,
        "candidate_exact_match": sum(b for _, b in exact) / n,
        "exact_match_difference": sum(b - a for a, b in exact) / n,
    }
    arm_reports = {
        name: {
            "requests": arm["requests"],
            "completed": arm["completed"],
            "mean_f1_on_completed_calls": arm["f1_sum"] / arm["completed"] if arm["completed"] else None,
            "mean_latency_s": arm["latency_sum"] / arm["completed"] if arm["completed"] else None,
            "prompt_tokens": arm["prompt_tokens"],
            "completion_tokens": arm["completion_tokens"],
        }
        for name, arm in (("direct", arms[0]), ("grounded_extract", arms[1]))
    }
    report = {
        "kind": "actual_open_weight_model_paired_public_human_gold_audit",
        "model": model,
        "dataset": "Stanford SQuAD v1.1 public development split",
        "dataset_sha256": dataset_sha,
        "dataset_total_answerable_questions": total,
        "selection_method": "SHA256 sorted question IDs, fixed offset and limit",
        "offset": offset,
        "limit": limit,
        "seed": seed,
        "question_order": "seeded randomized prompt order per question",
        "scoring": "SQuAD normalized token F1 and exact match against human gold",
        "gold_answers_visible_to_model": False,
        "n_requested_calls": 2 * len(questions),
        "n_completed_calls": sum(arm["completed"] for arm in arms.values()),
        "n_failed_calls": failed,
        "n_complete_pairs": n,
        "direct": direct,
        "arms": arm_reports,
        "paired_question_interval": paired_oracle_interval(pairs, replicates=replicates, seed=seed + 1),
        "article_cluster_interval": cluster_oracle_interval(by_article, replicates=replicates, seed=seed + 2),
        "paragraph_cluster_interval": cluster_oracle_interval(by_paragraph, replicates=replicates, seed=seed + 3),
        "article_clusters": len(by_article),
        "paragraph_clusters": len(by_paragraph),
        "automatic_rollout_permitted": False,
        "limitations": [
            "Exploratory small public development-set follow-up; no hidden holdout or superiority claim.",
            "Only complete pairs enter the matched estimate; failures may create selection bias.",
            "Cluster percentile intervals are exploratory and may be unstable with few clusters.",
            "Model endpoint provenance must be independently recorded; mock endpoints do not validate a model.",
            "No raw questions, answers or generated text are included in the report.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dashboard_path.write_text(render_paired_dashboard(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1/chat/completions")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--acknowledge-model-calls", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--timeout-s", type=float, default=90)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    report = run_paired_audit(
        args.dataset, args.out_dir, model=args.model,
        endpoint=args.endpoint, allow_remote=args.allow_remote,
        acknowledge_model_calls=args.acknowledge_model_calls,
        limit=args.limit, offset=args.offset, seed=args.seed,
        timeout_s=args.timeout_s, max_tokens=args.max_tokens,
    )
    print(json.dumps({
        "model": report["model"],
        "n_complete_pairs": report["n_complete_pairs"],
        "n_failed_calls": report["n_failed_calls"],
        "direct": report["direct"],
        "article_cluster_interval": report["article_cluster_interval"],
        "automatic_rollout_permitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()

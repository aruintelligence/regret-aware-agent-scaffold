"""Self-contained, accessible HTML evidence dashboard. No remote assets or JS."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path


def _number(value, digits=3):
    return "Not estimable" if value is None else f"{value:,.{digits}f}"


def _percent(value):
    return "Not estimable" if value is None else f"{100 * value:.1f}%"


def _bar(value, label, accent):
    width = 0 if value is None else max(0, min(100, 100 * value))
    return (
        f'<div class="bar" role="img" aria-label="{escape(label)}: {_percent(value)}">'
        f'<span class="fill {accent}" style="width:{width:.2f}%"></span></div>'
    )


def render_dashboard(report: dict) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    paired = report["paired_difference"]
    gate = report["review_gate"]
    dr = report.get("crossfit_doubly_robust", {})
    segments = report.get("segment_diagnostics", [])
    segment_rows = "".join(
        "<tr><th scope=\"row\">" + escape(row["task_signature"]) + "</th>"
        + "<td>" + str(row["n_labeled"]) + "</td>"
        + "<td>" + _percent(row["baseline_snips"]) + "</td>"
        + "<td>" + _percent(row["candidate_snips"]) + "</td>"
        + "<td>" + _percent(row["difference"]) + (" ⚠" if row["material_negative_segment"] else "") + "</td></tr>"
        for row in segments[:20]
    )
    is_synthetic = report["label_source_counts"].get("synthetic", 0) > 0
    is_human_gold = any("squad1.1_human_annotated" in kind for kind in report.get("runtime_kind", []))
    is_llm_gold = any("opt_in_openai_compatible_llm" in kind for kind in report.get("runtime_kind", []))
    badge = ("SYNTHETIC DEMONSTRATION" if is_synthetic else
             "PUBLIC HUMAN-ANNOTATED RETRIEVAL BENCHMARK" if is_human_gold else
             "OPERATOR-SUPPLIED LLM · HUMAN-GOLD BENCHMARK" if is_llm_gold else
             "SELF-ATTESTED EXTERNAL LABELS")
    status = "REVIEW SCREEN PASSED" if gate["candidate_for_human_review"] else "NOT CLEARED FOR REVIEW"
    benchmark_notice = (
        '<p class="notice"><strong>Benchmark scope:</strong> SQuAD human-annotated sentence retrieval only. This does not measure LLM reasoning, complete question answering, or enterprise agent performance.</p>'
        if is_human_gold else
        '<p class="notice"><strong>Real model execution is operator supplied.</strong> Public SQuAD dev-set scores do not establish general reasoning, hidden-test performance, or commercial readiness.</p>'
        if is_llm_gold else ""
    )
    direct = report.get("external_benchmark")
    direct_panel = ""
    if is_human_gold and direct:
        interval = direct["paired_article_cluster_interval"]
        lower = interval["lower_95"] * 100
        upper = interval["upper_95"] * 100
        improvement = direct["difference"] * 100
        exact = direct.get("exact_mcnemar_descriptive")
        exact_note = (
            '<p class="sub">Discordant questions: baseline-only '
            + str(exact["baseline_only"]) + ', challenger-only '
            + str(exact["candidate_only"]) + '; exact paired p = '
            + f'{exact["exact_two_sided_p"]:.4g}'
            + '. Descriptive only: questions share paragraphs and articles.</p>'
            if exact else ""
        )
        direct_panel = (
            '<section class="panel"><div class="eyebrow">Direct full-information audit · independently annotated</div>'
            '<h2>Human-gold sentence retrieval</h2>'
            '<div class="compare"><div class="card"><div class="label">Baseline · lexical overlap</div>'
            '<div class="value">' + _percent(direct["baseline_accuracy"]) + '</div></div>'
            '<div class="card"><div class="label">Challenger · IDF + proximity</div>'
            '<div class="value">' + _percent(direct["candidate_accuracy"]) + '</div></div></div>'
            '<p><strong>Direct observed difference: ' + f'{improvement:+.2f}' + ' percentage points.</strong> '
            + f'{direct["n_evaluated"]:,}' + ' public human-annotated questions; '
            + f'{interval["clusters"]:,}' + ' article clusters.</p>'
            '<p>Exploratory article-cluster bootstrap 95% interval: <strong>'
            + f'{lower:+.2f} to {upper:+.2f}' + ' percentage points</strong>.</p>'
            '<p class="sub">Negative article-level domains flagged: '
            + str(direct["negative_article_count"]) + '. This is sentence retrieval, not an LLM or general-agent test.</p>'
            + exact_note + '</section>'
        )
        stress_rows = "".join(
            "<tr><td>" + f'{item["requested_tv_budget"]*100:.0f}%' + "</td>"
            + "<td>" + f'{item["worst_case_uplift"]*100:+.2f} pp' + "</td>"
            + "<td>" + f'{item["realized_tv"]*100:.1f}%' + "</td></tr>"
            for item in direct.get("topic_mix_stress", [])
        )
        if stress_rows:
            radius = direct.get("break_even_topic_shift_tv")
            radius_text = f'{radius*100:.1f}%' if radius is not None else "Not erased by observed-topic reweighting"
            direct_panel += (
                '<section class="panel"><div class="eyebrow">Adversarial robustness · observed topics only</div>'
                '<h2>What if tomorrow’s tasks look different?</h2>'
                '<p><strong>Observed-topic break-even shift radius: ' + radius_text + '</strong></p>'
                '<p class="sub">Worst-case measured uplift when probability mass is moved from stronger to weaker observed articles within a total-variation budget.</p>'
                '<table><thead><tr><th>Topic-mix shift budget</th><th>Worst-case uplift</th><th>Shift used</th></tr></thead>'
                '<tbody>' + stress_rows + '</tbody></table>'
                '<p class="sub">This is a finite reweighting stress test, not a confidence interval or guarantee on unseen domains.</p></section>'
            )
    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"])
    limitations = "".join(f"<li>{escape(reason)}</li>" for reason in report["limitations"])
    source = ", ".join(f"{escape(str(k))}: {int(v)}" for k, v in report["label_source_counts"].items())
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    low, high = paired["lower_95"], paired["upper_95"]
    interval = f"{_percent(low)} to {_percent(high)}" if low is not None and high is not None else "Not estimable"
    delta = paired["snips_difference"]
    delta_display = ("+" if delta is not None and delta > 0 else "") + _percent(delta)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><meta name="description" content="ĀRU Decision Evidence Lab — experimental policy evaluation with transparent uncertainty.">
<title>ĀRU · Decision Evidence Lab</title>
<style>
:root{{--bg:#07101e;--surface:#101e31;--surface2:#152740;--line:#28405a;--text:#eef5fc;--muted:#b5c7d9;--cyan:#65d9ec;--gold:#e9c27a;--red:#ffadab;--green:#95e0ba}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 15% 0%,#16314b 0%,var(--bg) 48%);color:var(--text);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}}
.wrap{{max-width:1100px;margin:auto;padding:36px 24px 70px}}.eyebrow{{color:var(--cyan);letter-spacing:.2em;font-size:12px;font-weight:750;text-transform:uppercase}}
header{{border-bottom:1px solid var(--line);padding-bottom:26px;display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap}}
.brand{{font-size:27px;font-weight:800;letter-spacing:.15em}}h1{{font-size:clamp(34px,6vw,66px);letter-spacing:-.04em;line-height:1.08;margin:30px 0 16px}}
.lead{{max-width:720px;color:var(--muted);font-size:18px}}.pill{{display:inline-block;padding:8px 13px;border:1px solid var(--line);border-radius:999px;color:var(--gold);font-size:12px;font-weight:800;letter-spacing:.06em}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:34px 0}}.card{{background:linear-gradient(145deg,var(--surface),#0c182a);border:1px solid var(--line);border-radius:18px;padding:23px;min-width:0}}
.label{{color:var(--muted);font-size:13px;letter-spacing:.05em;text-transform:uppercase}}.value{{font-size:clamp(28px,4vw,44px);font-weight:760;line-height:1.3;letter-spacing:-.04em}}.sub{{font-size:13px;color:var(--muted)}}
h2{{font-size:25px;letter-spacing:-.02em;margin:32px 0 14px}}.compare{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.bar{{height:13px;background:#25384d;border-radius:99px;overflow:hidden;margin:16px 0 10px}}.fill{{display:block;height:100%;border-radius:99px}}.cyan{{background:linear-gradient(90deg,#3689b2,var(--cyan))}}.gold{{background:linear-gradient(90deg,#aa7e3b,var(--gold))}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:12px 9px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-weight:650}}td{{overflow-wrap:anywhere}}.panel{{margin:20px 0;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:25px}}
.notice{{border-left:4px solid var(--gold);padding:13px 18px;background:#302718;border-radius:4px;color:#fff0cf}}.status{{font-weight:800;letter-spacing:.08em;color:var(--gold)}}
.flow{{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}}.flow span{{border:1px solid var(--line);background:var(--surface2);border-radius:10px;padding:10px 14px;font-size:13px}}
ul{{padding-left:22px;color:var(--muted)}}li{{margin:7px 0}}footer{{color:var(--muted);border-top:1px solid var(--line);padding-top:18px;margin-top:38px;font-size:13px}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}.compare{{grid-template-columns:1fr}}.wrap{{padding:22px 16px 50px}}}}
</style></head><body><main class="wrap">
<header><div class="brand">ĀRU <span style="color:var(--cyan)">/</span> INTELLIGENCE</div><span class="pill">DECISION EVIDENCE LAB · v1</span></header>
<div class="eyebrow" style="margin-top:34px">Research instrument · no deployment authorization</div>
<h1>Measure decisions.<br><span style="color:var(--cyan)">Not mythology.</span></h1>
<p class="lead">An auditable comparison of two routing policies using logged selection probabilities and independently supplied outcome scores. Statistical estimates are not proof of general intelligence.</p>
{benchmark_notice}
<span class="pill">{escape(badge)}</span>
<div class="flow"><span>01 · Randomized route</span><span>→</span><span>02 · Logged probability</span><span>→</span><span>03 · Outcome label</span><span>→</span><span>04 · Counterfactual estimate</span><span>→</span><span>05 · Human review</span></div>
<div class="grid">
<section class="card"><div class="label">Logged decisions</div><div class="value">{report["n_logged"]:,}</div><div class="sub">Randomized policy decisions</div></section>
<section class="card"><div class="label">Labeled outcomes</div><div class="value">{report["n_labeled"]:,}</div><div class="sub">{report["n_unlabeled"]:,} still unlabeled</div></section>
<section class="card"><div class="label">Candidate uplift (SNIPS)</div><div class="value">{delta_display}</div><div class="sub">Candidate minus baseline · exploratory</div></section>
</div>
{direct_panel}
<h2>Policy comparison · propensity-weighted estimate</h2>
<div class="compare">
<section class="card"><div class="label">Baseline · {escape(report["baseline_policy"])}</div><div class="value">{_percent(baseline["snips"])}</div>{_bar(baseline["snips"], "Baseline self-normalized estimate", "cyan")}<div class="sub">Self-normalized inverse propensity estimate</div></section>
<section class="card"><div class="label">Candidate · {escape(report["candidate_policy"])}</div><div class="value">{_percent(candidate["snips"])}</div>{_bar(candidate["snips"], "Candidate self-normalized estimate", "gold")}<div class="sub">Self-normalized inverse propensity estimate</div></section>
</div>
<section class="panel"><h2 style="margin-top:0">Diagnostics & uncertainty</h2>
<table><caption style="text-align:left;color:var(--muted)">Evaluation diagnostics — higher sample support improves stability, not necessarily validity</caption>
<thead><tr><th scope="col">Metric</th><th scope="col">Baseline</th><th scope="col">Candidate</th></tr></thead>
<tbody><tr><th scope="row">IPS estimate</th><td>{_number(baseline["ips"])}</td><td>{_number(candidate["ips"])}</td></tr>
<tr><th scope="row">Effective sample size</th><td>{_number(baseline["effective_sample_size"],1)}</td><td>{_number(candidate["effective_sample_size"],1)}</td></tr>
<tr><th scope="row">Observed policy match</th><td>{_percent(baseline["match_rate"])}</td><td>{_percent(candidate["match_rate"])}</td></tr>
<tr><th scope="row">Clipped weights</th><td>{baseline["clipped"]}</td><td>{candidate["clipped"]}</td></tr></tbody></table>
<p><strong>Paired bootstrap 95% interval for candidate − baseline:</strong> {interval}</p>
<p class="sub">Valid resamples: {paired["valid_replicates"]:,}. Assumes independent tasks; repeated or adaptive logs need a different inference design.</p></section>
<section class="panel"><div class="eyebrow">Conservative review gate</div><h2 class="status">{escape(status)}</h2>
<ul>{reasons}</ul><p class="notice"><strong>Automatic rollout: prohibited.</strong> A passed statistical screen only permits human evaluation, prospective testing, and independent review.</p></section>
<section class="panel"><h2 style="margin-top:0">Second estimator · cross-fitted doubly robust</h2>
<p>Baseline DR: <strong>{_percent(dr.get("baseline_dr"))}</strong> · Candidate DR: <strong>{_percent(dr.get("candidate_dr"))}</strong> · Difference: <strong>{_percent(dr.get("difference"))}</strong></p>
<p class="sub">Out-of-fold shrinkage reward model; this is a separate diagnostic, not an independent replication or validated reward predictor.</p>
<h2>Domain-level safety signals</h2>
<div style="overflow-x:auto"><table><thead><tr><th scope="col">Task signature</th><th scope="col">N</th><th scope="col">Baseline</th><th scope="col">Candidate</th><th scope="col">Difference</th></tr></thead>
<tbody>{segment_rows}</tbody></table></div><p class="sub">A ⚠ flags an observed material negative segment with minimum support. It is an investigation trigger, not proof of subgroup harm.</p></section>
<section class="panel"><h2 style="margin-top:0">Evidence provenance</h2>
<p>Labels: {source or "None"} · distinct graders: {report["distinct_graders"]} · runtime: {escape(", ".join(report["runtime_kind"]))}</p>
<h2>Interpretation limits</h2><ul>{limitations}</ul></section>
<footer>ĀRU Intelligence · Decision Evidence Lab · Generated {generated}. This is a research report, not a certification, sales performance guarantee, or deployment approval.</footer>
</main></body></html>"""


def save_dashboard(report: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard(report), encoding="utf-8")

"""ĀRU Selective Risk Gate — standalone signed-evidence research dashboard.

No raw calibration examples, source passages or model outputs are rendered.
The dashboard can only be produced from an HMAC-verified policy.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from selective_risk_gate import verify_risk_policy


def render_risk_dashboard(policy, *, key):
    if not verify_risk_policy(policy, key=key):
        raise ValueError("risk policy HMAC invalid; refusing to render")
    if policy.get("kind") != "aru_selective_risk_calibration":
        raise ValueError("unsupported policy kind")
    def fmt(value):
        return f"{100 * value:.1f}%"
    passed = policy["conditional_risk_screen_passed"]
    status = "Risk screen passed · human review required" if passed else "Insufficient evidence · abstain"
    selected = policy["selected_threshold"]
    chosen = next((c for c in policy["candidates"] if c["threshold"] == selected), None) if passed else None
    summary = "Not selected" if selected is None else f"{selected:.4g}"
    rows = []
    for candidate in policy["candidates"]:
        label = "QUALIFIED" if candidate["eligible"] else "BLOCKED"
        group_cells = []
        for group, item in candidate["slices"].items():
            state = "pass" if item["meets_target"] else "fail"
            observed = fmt(item["observed_error"]) if item["observed_error"] is not None else "N/A"
            group_cells.append(
                f'<div class="slice {state}"><strong>{escape(group)}</strong>'
                f'<span>n={item["accepted"]} · errors={item["errors"]}</span>'
                f'<span>Observed {observed} · upper {fmt(item["simultaneous_error_upper"])}</span></div>'
            )
        rows.append(
            f'<article class="candidate"><div class="candidate-top">'
            f'<div><span class="eyebrow">Threshold {candidate["threshold"]:.4g}</span>'
            f'<h3>{label}</h3></div><strong>{fmt(candidate["coverage"])} coverage</strong></div>'
            f'<div class="track"><div class="fill" style="width:{100 * candidate["coverage"]:.1f}%"></div></div>'
            f'<div class="slices">{"".join(group_cells)}</div></article>'
        )
    selected_groups = ""
    if chosen:
        selected_groups = "".join(
            f'<span class="pill">{escape(group)} · upper {fmt(item["simultaneous_error_upper"])}</span>'
            for group, item in chosen["slices"].items()
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#071321"><title>ĀRU · Signed Selective Risk Evidence</title>
<style>
:root{{color-scheme:dark;--bg:#071321;--panel:#12283b;--line:#2c4b60;--ink:#f1f8fb;--muted:#b9ccd7;--cyan:#78e5e2;--gold:#efcd86}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse 850px 600px at 10% 0%,#193e4e,transparent),var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:auto;padding:70px 28px}}h1{{font-size:clamp(40px,7vw,78px);line-height:1.07;letter-spacing:-.06em;margin:16px 0}}h2{{font-size:30px;letter-spacing:-.04em}}
.lede,.muted{{color:var(--muted)}}.eyebrow{{font-size:12px;letter-spacing:.16em;color:var(--cyan);text-transform:uppercase;font-weight:800}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:32px 0}}.card,.candidate{{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:26px}}
.card strong{{display:block;font-size:clamp(27px,3.8vw,47px);letter-spacing:-.05em}}.card small{{color:var(--muted)}}
.status{{display:inline-block;border:1px solid #4a807b;background:#174045;color:#a7f6eb;border-radius:100px;padding:8px 15px;font-size:12px;font-weight:800}}
.status.blocked{{background:#3a2b24;color:#f2d3a1;border-color:#775c40}}
.candidate{{margin:17px 0}}.candidate-top{{display:flex;justify-content:space-between;gap:12px;align-items:center}}.candidate h3{{font-size:21px;margin:6px 0 0}}.candidate-top>strong{{font-size:20px;color:var(--cyan)}}
.track{{height:12px;background:#2d4858;border-radius:99px;overflow:hidden;margin:22px 0}}.fill{{height:100%;background:linear-gradient(90deg,#3d8f9d,var(--cyan));border-radius:99px}}
.slices{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}}.slice{{padding:16px;border:1px solid var(--line);border-radius:12px}}.slice strong,.slice span{{display:block}}.slice span{{color:var(--muted);font-size:13px;margin-top:4px}}.slice.pass{{border-color:#377f7b}}.slice.fail{{border-color:#79613f}}
.pill{{display:inline-block;padding:9px 13px;border:1px solid #396a71;border-radius:100px;color:var(--cyan);font-size:12px;margin:5px}}
.notice{{margin:35px 0;padding:24px;border-left:4px solid var(--gold);background:#31291f;border-radius:5px;color:#f3e5c9}}
footer{{border-top:1px solid var(--line);padding:25px 0;color:var(--muted);font-size:13px;overflow-wrap:anywhere}}
@media(max-width:700px){{main{{padding:35px 18px}}.grid{{grid-template-columns:1fr}}.candidate-top{{align-items:start;flex-direction:column}}}}
</style></head><body><main>
<div class="eyebrow">ĀRU Intelligence · independently signed calibration evidence</div>
<h1>Trust the evidence.<br>Not the confidence score.</h1>
<p class="lede">Exact binomial error bounds across predeclared thresholds and trusted task groups. The HMAC policy signature was verified before this report was generated.</p>
<span class="status {"qualified" if passed else "blocked"}">{status}</span>
<div class="grid">
<div class="card"><small>Selected score threshold</small><strong>{summary}</strong></div>
<div class="card"><small>Observed calibrated coverage</small><strong>{fmt(policy["selected_coverage"])}</strong></div>
<div class="card"><small>Maximum targeted error</small><strong>{fmt(policy["target_error"])}</strong></div>
</div>
<section><div class="eyebrow">01 / Group-aware risk evidence</div><h2>Every group must qualify.</h2>
<p class="muted">The confidence family uses alpha={policy["familywise_alpha"]:.3g}, {policy["simultaneous_tests"]} simultaneous tests, and at least {policy["min_group_support"]} accepted calibration examples in every group and the aggregate.</p>
{selected_groups}
{"".join(rows)}</section>
<div class="notice"><strong>Research limitations.</strong> The statistical interpretation requires independent representative calibration and future tasks, fixed model/rubric/score, trusted groups and accurate labels. It does not survive arbitrary distribution shift or establish factual truth. A source-span proof only confirms textual provenance. <strong>Automatic deployment remains disabled.</strong></div>
<footer>Model revision: {escape(policy["model_revision"])} · Rubric: {escape(policy["rubric_id"])} · Calibration: {escape(policy["calibration_set_id"])}<br>
Calibration fingerprint: {escape(policy["calibration_sha256"])} · No raw prompts, source text or per-row labels included.</footer>
</main></body></html>"""


def save_risk_dashboard(policy, *, key, output_path):
    html = render_risk_dashboard(policy, key=key)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(html)
    return str(target)

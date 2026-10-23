# Disjoint change-point audit: measured result and misses

**Source run:** [PR #7 regression workflow 39321106145](https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39321106145), real change-point job, on protocol-and-code commit `5ff4ba6ca11b546c56c194ff0d783e4eb774e603`, 2026-10-19. The [protocol](CHANGE_POINT_PROTOCOL.md) and executable settings were committed **before** these new model calls. This note was written after the run. The uploaded artifact ZIP SHA-256 was `51c11fde60d2516397c03136861b32fb60ac2aeac153190ca7cd1f39eaa578e0` (GitHub artifact `12219099407`). CI and a separate local recheck recomputed both restart reports exactly from the archived ordered JSONL traces.

The complete [aggregate report](evidence/change_point_20261019/change_point_report.json), [baseline trace](evidence/change_point_20261019/baseline_outcomes.jsonl), [intervention trace](evidence/change_point_20261019/intervention_outcomes.jsonl), [baseline report](evidence/change_point_20261019/baseline_report.json), and [protocol hash](evidence/change_point_20261019/protocol.sha256) are retained because the CI artifact expires. These contain no question, passage, answer or model weight. The baseline outcome fingerprint is `135d9fc00eeb39263384d603415bed1309a5c7c9291f8e385a025bdb9cae96d2`; the intervention fingerprint is `94efb369f3e04b438cdf8ef1eae2deadc18bf24036dfe284ff4056d475a222ba`.

## Actual model-call results

Both arms used `distilbert/distilbert-base-cased-distilled-squad` at resolved revision `564e9b582944a57a3e586bbb98fd6f0a4118db7f`. The same published human answers, repository grader, source-offset verification, 50% error target and familywise alpha 0.05 were used for both. Question IDs at SHA-sorted offsets **320–439** do not overlap the preceding audit's 200–319; article overlap is possible.

| Measure | Baseline | Deliberate context-limit intervention |
|---|---:|---:|
| Total ordered requests | 120 | 120, including 55 copied baseline-prefix outcomes |
| Additional model calls | 120 | 65 after request 55 |
| Total rubric errors | 18/120 | 58/120 |
| Suffix errors, requests 56–120 | 8/65 | 48/65 |
| Verified source certificates on actual calls | 120 | 65 additional |
| Execution failures | 0 | 0 |
| Finite restart first alarm | None | **None by 120** |
| Original from-start first alarm | None | **None by 120** |

The intervention truncated every suffix source context to its first 128 characters **before** another genuine model call. This is an intentional retrieval failure, not spontaneous deployed-model deterioration. Its observed suffix error fraction was 73.85%, yet the predeclared restart monitor **missed it** within the 120-request horizon. The most favorable monitored restart on the intervention had peak log evidence **5.141**, below its predeclared log threshold **5.886** across 18 components. This identifies a sensitivity limit in the tested setting; it does not justify retroactively lowering the threshold.

## Separately synthetic stress on the real baseline trace

All numbers here are post-onset **label flips**, not new model calls or actual model drift. Neither monitor alarmed on the unmodified baseline or before any injection. A dash denotes no alarm by request 120; delay is measured in requests after onset.

| Onset after request | Correct-to-error flip probability | Labels flipped | Restart first alarm (delay) | Original first alarm (delay) |
|---:|---:|---:|---:|---:|
| 25 | 50% | 45 | — | — |
| 25 | 100% | 82 | 52 (27) | 68 (43) |
| 55 | 50% | 28 | — | — |
| 55 | 100% | 57 | 73 (18) | — |
| 85 | 50% | 13 | — | — |
| 85 | 100% | 29 | 112 (27) | — |

Thus the restart watch detected **3/6 synthetic scenarios** versus **1/6** for the original watch on this disjoint real baseline trace. It missed all three 50% flip scenarios, and both monitors missed the actual context-limit intervention. These six correlated stress scenarios are a diagnostic, **not an estimated detection-power probability**.

In 400 seeded **synthetic IID Bernoulli(0.5)** streams, the restart watch alarmed once and the original watch five times. These fractions are software diagnostics under a synthetic conditional null. They do not estimate the real model's false-alarm rate; article clustering, grading error and selection can violate the necessary conditional-null interpretation.

## Decision

The preregistered result supports only the narrow conclusion that scheduled restarts helped with the extreme injected suffix changes tested here, while **the genuine model-call retrieval intervention was missed**. The high 50% target and 18-component multiplicity are material to that miss. A next study should freeze a commercially meaningful error target **with independent calibration**, a detection-horizon requirement, and a new disjoint evaluation set before running the model. No tuning on offsets 320–439 can turn this audit into an independent validation. The report remains unsigned research evidence, outside the signed release gate; no automatic rollout or customer assurance claim follows.

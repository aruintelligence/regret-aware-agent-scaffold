# Separate calibration caught a controlled real-model retrieval failure

**Run:** [PR #7 Python regression workflow 39329178175](https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39329178175), “Separate calibration and real-model retrieval intervention” job, on commit `2b26d604f60fc49a215a6bdcb5484163d3238910`, 2026-10-19. The [protocol](CALIBRATED_CHANGE_PROTOCOL.md), code, model revision, question offsets, budget and success rule were committed **before** these model calls. This results note was written afterward. The GitHub artifact (`12222714884`) ZIP has SHA-256 `0a17721bba284495e307211a7b00055edbd0e6a0648c6321d9e11a13cc6d4ef4`.

The [aggregate report](evidence/calibrated_change_20261019/calibrated_report.json), [calibration trace](evidence/calibrated_change_20261019/calibration_outcomes.jsonl), [ordinary baseline trace](evidence/calibrated_change_20261019/baseline_outcomes.jsonl), [intervention trace](evidence/calibrated_change_20261019/intervention_outcomes.jsonl), [protocol hash](evidence/calibrated_change_20261019/protocol.sha256), [calibration report](evidence/calibrated_change_20261019/calibration_report.json), [baseline report](evidence/calibrated_change_20261019/baseline_report.json) and [fixed-50% comparator report](evidence/calibrated_change_20261019/fixed_50_percent_report.json) are retained because the CI artifact expires. They contain only binary outcomes, group labels and aggregate statistics, without question, passage, answer text or model weights. CI and a separate local check recomputed both calibrated watches exactly from the three archived outcome streams.

## Real-model observations

All **305 actual model calls** used `distilbert/distilbert-base-cased-distilled-squad` at pinned revision `564e9b582944a57a3e586bbb98fd6f0a4118db7f`, published human gold answers and the frozen source-offset verification / F1 rubric. There were **zero execution failures** and **305 source certificates** for accepted outputs.

| Measure | Observed result |
|---|---:|
| Separate calibration, offsets 440–559 | 12 errors / 120 calls |
| Calibrated aggregate one-sided upper error target | 24.12% |
| Calibrated long-source / short-source upper targets | 29.54% / 29.90% |
| Ordinary evaluation, offsets 560–679 | 9 errors / 120 calls; **no alarm** |
| Additional model calls after intervention onset at request 55 | 65 |
| Ordinary versus context-limited errors in requests 56–120 | 4/65 versus **44/65** |
| Full intervention stream, including the ordinary 55-request prefix | 49 errors / 120 requests |
| **Calibrated restart first alarm** | **Request 80; delay 25** |
| Previous fixed-50% restart first alarm on the *same* intervention stream | Request 111; delay 56 |

The earliest calibrated crossing came from the aggregate restart beginning at request **61**. The calibrated watch flagged this deliberately damaged retrieval pipeline **31 requests earlier** than the prior fixed-50% watch on the same ordered outcomes. The ordinary evaluation baseline had no alarm. The three stored SHA-256 outcome fingerprints are calibration `4c1f1d6befa351c3c13bf8b921b1f74e9526367d814bd56ceb2acf14e2f90f7a`, baseline `4e96bff8029558bb2717d8bc309d6e7305bf53492804b6a81dfe6fd821535525`, and intervention `cc1675f66e3f96efbf8ccf13aeb21f579ab351516856d0d3e11d8457ab013b8a`. The protocol checksum is `a11501a3cc1c91e799214c85f38a625d6110667e71fbbb8732e311e76326fba6`.

## Clearly synthetic secondary stress

The six **label-flip** scenarios below alter graded labels on the ordinary real-model trace; they are *not* additional model predictions. Delay is in global requests after onset. An empty alarm means no crossing by 120.

| Onset after request | Correct-to-error flip probability | Labels flipped | Calibrated alarm | Delay |
|---:|---:|---:|---:|---:|
| 25 | 50% | 43 | 63 | 38 |
| 25 | 100% | 88 | 38 | 13 |
| 55 | 50% | 27 | 103 | 48 |
| 55 | 100% | 61 | 67 | 12 |
| 85 | 50% | 16 | — | — |
| 85 | 100% | 33 | 98 | 13 |

The calibrated detector caught **5/6** injected scenarios, missing the late 50% case. These six correlated stresses do not estimate a general detection-power probability.

## Limits of the result

This is a **successful controlled intervention on real model outputs**, not evidence of naturally occurring live drift. The source contexts were deliberately cut to 128 characters for suffix calls, while grading against published human answers. Calibration and evaluation have disjoint question IDs but may share articles; the repository grader has not been externally audited. The nominal 0.05 combined alarm budget requires IID per-slice calibration, stable error probabilities, predictable groups, complete independent labels and the conditional monitoring null. We have **not established those assumptions** for the public dataset or a customer environment. A single baseline with no alarm does not estimate a real-world false-alarm rate or prove safety. This unsigned research evidence cannot replace the signed, independently anchored release gate and authorizes no automatic rollout.

The next commercial validation is a prospectively graded, article-separated or customer-generated task stream with a prespecified service target, independent label review, and measured detection delay and false alarms. Preserve this benchmark result as one reproducible engineering milestone, not a product guarantee.

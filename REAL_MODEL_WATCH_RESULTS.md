# First public real-model risk-watch audit — observed result

**Run:** [PR #7 GitHub Actions 39318796848](https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39318796848) on source commit `5b062b9748255dbb15c9e0a053ef9a1a15198a5d`, 2026-10-19. The [protocol](REAL_MODEL_WATCH_PROTOCOL.md) and code were committed before the model run. This results note was written after examining the artifact. The downloaded artifact ZIP had SHA-256 `cf7d0d1fc9120905566247ce2db983c6b17fbfb761667c4edf9579d35a612d01`.

The exact aggregate [report](evidence/real_watch_20261019/report.json), [ordered public-benchmark binary outcomes](evidence/real_watch_20261019/outcomes.jsonl) and [protocol checksum](evidence/real_watch_20261019/protocol.sha256) are preserved here because the Actions artifact expires. They contain no passage, question, answer text, or model weights. The outcome stream fingerprint in the report is `5d013b729c863ab5420d4153b094725da281461850f91b8c497561683c1559a8`. Recompute it with `watch_risk_stream` using the recorded model revision, rubric, groups, target and alpha; the CI job already did this independently from the archived JSONL.

## Actual model observations

The pretrained extractive QA model `distilbert/distilbert-base-cased-distilled-squad` resolved to revision `564e9b582944a57a3e586bbb98fd6f0a4118db7f`. It answered all **120/120** fixed public SQuAD questions with **zero execution failures** and **120 independently checkable source-offset certificates**. After inference, its outputs were scored against published human answers by this repository's normalized token-F1 code.

| Measure | Observed result |
|---|---:|
| F1 below 0.5 or unsupported answer | 12 / 120 (10.0%) |
| Mean normalized token F1 | 88.75% |
| Exact match | 97 / 120 (80.83%) |
| Short-source group errors | 8 / 53 (15.09%) |
| Long-source group errors | 4 / 67 (5.97%) |
| Baseline degradation alarm against predeclared 50% error target | None |

The difference between the two source-length groups is descriptive and not a validated subgroup risk comparison. The target of 50% is a **severe-deterioration stress threshold**, not a useful customer quality target. No alarm does not certify an error rate below that target.

## Synthetic stress replay on that real baseline

Every post-onset change below is an **injected label flip**, not a second model execution or observed real-world drift. Delay counts requests after the specified onset. A pre-onset alarm would not count as detecting the injection; none occurred here.

| Onset after request | Correct-to-error flip chance | Labels flipped | First alarm | Result by request 120 |
|---:|---:|---:|---:|---|
| 30 | 50% | 36 | None | Missed |
| 30 | 100% | 81 | Request 85, delay 55 | Detected |
| 60 | 50% | 34 | None | Missed |
| 60 | 100% | 53 | None | Missed |
| 90 | 50% | 9 | None | Missed |
| 90 | 100% | 28 | None | Missed |

The 200 seeded **synthetic IID-null** streams produced 1 alarm (0.5%). That fraction is a software diagnostic, not the false-alarm rate of this model or a customer deployment. The actual 120-question baseline produced no alarm, which is only one observation path.

This run exposes a practical limitation: evidence built up during a good prefix can make a simple from-start e-process slow to react to a later abrupt change. We should evaluate a predeclared restart or change-point mixture on **new, disjoint observations** and report its extra multiplicity cost, missed detections, and false alarms without retuning this recorded protocol. The original result stays fixed as a baseline.

**Claim boundary:** public development data, repository-owned grading code, correlated questions and synthetically modified suffix labels cannot demonstrate production reliability, independent customer grading, live drift detection, safety, or commercial value. The artifact is unsigned; it is not the separately anchored current ledger required by the dual-evidence release gate. Human review and separate calibration remain necessary.

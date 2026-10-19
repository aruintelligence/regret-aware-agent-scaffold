# ĀRU Decision Evidence Lab — Independent Human-Gold Benchmark Record

**Status:** executed, reproducible exploratory research result. Not a model leaderboard submission, enterprise validation, or certification.

## Result at a glance

The ĀRU benchmark compared **two actual dependency-free sentence-retrieval algorithms** against human-annotated answer locations in the official Stanford SQuAD v1.1 development dataset. The algorithms received only each question and passage, never the annotated answers.

| Metric | Baseline: normalized token overlap | Challenger: BM25/IDF + coverage/proximity |
| --- | ---: | ---: |
| Selected sentence contains a human answer span | **78.50%** | **82.10%** |
| Errors per 2,000 questions | 430 | 358 |

**Observed difference:** **+3.60 percentage points** in sentence-selection accuracy, or 72 additional correct selections out of 2,000 evaluated questions. Relative error reduction versus this baseline: **16.7%**. This is **not** a 16.7% improvement in general AI intelligence.

### Uncertainty and dependence checks

| Paired resampling unit | Exploratory 95% interval for challenger minus baseline |
| --- | --- |
| Individual question (2,000 observations) | **+2.75 to +4.55 percentage points** |
| Paragraph (1,301 clusters) | **+2.55 to +4.67 percentage points** |
| Article/topic (48 clusters) | **+2.47 to +4.61 percentage points** |

Each interval used 400 seeded bootstrap replicates. These are exploratory percentile intervals, **not confirmatory inference**, and they do not correct for all benchmark-selection, annotation, or modeling biases.

**Domain-level caution:** the full-information audit found **one article with a material negative challenger result** under the preregistered-in-code screen (at least 20 examples and more than 5 percentage points worse). The affected article is **Victoria_(Australia)** (25 questions): baseline **72%**, challenger **64%**, a decline of **8 percentage points**. The system blocks advancement to the human-review screen until that regression is investigated. An aggregate win is not universal superiority. This small subgroup difference is exploratory and may be noisy.

### Adversarial topic-mix robustness

The new **observed-domain total-variation stress test** reweights the same 48 measured article domains to make the challenger look as unfavorable as possible, while holding each article's observed accuracy difference fixed. It does **not** model unseen topics or changes within a topic.

| Adversarial change in observed topic mix | Worst-case empirical challenger uplift |
| --- | ---: |
| 5% total-variation shift | **+2.51 percentage points** |
| 10% shift | **+1.59 percentage points** |
| 20% shift | **+0.013 percentage points** (nearly erased) |

The computed **break-even topic-mix radius** is approximately **20.09%**: at that adversarial reweighting budget, the observed aggregate gain can be eliminated. This is a descriptive fragility threshold under a constrained reweighting model, **not** a confidence interval, causal proof, or guarantee about real future customers.

### Exact paired-discordance cross-check

Across the same 2,000 questions, **93** questions were correct only for the challenger, while **21** were correct only for the baseline. These **114 discordant pairs** account for the net **72** additional correct selections.

The exact two-sided McNemar conditional calculation gives `p = 5.3124 × 10⁻¹²` **if the question pairs are independent**. They are not fully independent: questions share paragraphs and articles. Therefore this p-value is **descriptive, not confirmatory**, and it should not be used as the primary statistical claim. The article-cluster exploratory interval and negative-domain audit above are essential context. No multiple-testing or post-hoc model-selection correction is claimed.

### Off-policy estimator check

Only the randomized selected route's outcome was supplied to the off-policy estimator; both routes were executed offline solely to establish the directly observable audit benchmark.

- **Direct full-information difference:** +3.600 percentage points.
- **Logged-policy SNIPS estimate:** +5.2615 percentage points.
- **Absolute estimation error against the full-information audit:** **1.6615 percentage points**.

The difference between the SNIPS estimate and the direct audit is a reminder that propensity-weighted evaluation has sampling uncertainty and must not be presented as an exact counterfactual oracle.

## Reproducibility record

- **Dataset:** Stanford Question Answering Dataset, v1.1 development split.
- **Official source:** https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
- **Dataset SHA-256:** `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`
- **Selection:** first 2,000 question IDs sorted by SHA-256 digest (independent of answer correctness).
- **Randomization:** 50/50 independent logged route choice, seed `20261018`.
- **Rubric:** 1 if the selected sentence fully contains any human-annotated answer span; 0 otherwise.
- **CI workflow:** `Decision Evidence Lab CI`, human-gold job, Python 3.12.
- **Successful initial run:** https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39195975743
- **Reproduction:** `python3 squad_benchmark.py --dataset dev-v1.1.json --out-dir benchmark-report --limit 2000 --seed 20261018 --replicates 400`
- **Published artifacts:** HTML evidence dashboard, JSON report and direct oracle summary. The raw dataset is not redistributed in the artifacts.

The SQuAD dataset is distributed under **CC BY-SA 4.0**. Attribution belongs to the original dataset creators. The benchmark uses the public development set and does not constitute a held-out private leaderboard submission.

## Interpretation

**What was actually demonstrated:** on this fixed, publicly annotated retrieval task, the challenger selected the sentence containing a human answer span more often than the baseline. The evaluation pipeline can log randomized decisions, grade outcomes independently of the route selector, compute propensity-aware estimates, and compare them to a direct full-information audit.

**What was not demonstrated:** improvement in the original Divine Whisper scripted specialist system, general reasoning, autonomous real-world agent success, safe policy rollout, market leadership, novel mathematical theorems, or superiority to commercial LLMs. Neither retrieval algorithm is an LLM. No independent security audit or commercial customer trial has been completed.

## Next experiment (prospective)

1. Freeze the algorithms and evaluation rubric; preserve this record without tuning on the same evaluation examples.
2. Connect a real local LLM using `llm_squad_benchmark.py`, requiring explicit operator consent and a running model endpoint.
3. Use a separately sourced task set or blinded held-out tasks and an independently specified rubric.
4. Measure exact-match/F1, task success, cost per success, latency, failure rate, and domain-level regressions.
5. Predefine decision thresholds, account for clustering and repeated tasks, and require human review before any rollout.

This evidence record is deliberately limited to claims that the observed data can support.

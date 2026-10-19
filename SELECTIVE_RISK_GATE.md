# ĀRU Selective Risk Gate — From Source Provenance to Measured Answer Reliability

**Status:** research alpha. This module does not improve a model's reasoning, guarantee factual truth, establish production security, or authorize automatic deployment.

## Why this is the next engineering step

ĀRU's real-model experiments exposed a tradeoff: the tiny generative model often produced answers that could not be recovered as exact source spans, and a strict source gate withheld nearly all of them. A specialist extractive model can produce source spans by construction, but **a source-backed answer can still be incorrect or irrelevant**.

The new `selective_risk_gate.py` addresses the next, distinct question:

> Of the source-backed answers a *fixed* model would release above a given score threshold, how many were independently graded wrong on a representative calibration set?

Rather than interpreting raw model logits as probabilities, ĀRU treats the score as an arbitrary ranking variable and measures **empirical error on independently graded examples**. It chooses a threshold only when a simultaneous, conservative statistical upper bound meets a predeclared target in the aggregate **and every predeclared group**.

## Exact finite-sample statistical construction

Let `(S_i,Y_i,G_i)` be independent identically distributed calibration observations, where `S_i` is a fixed model score, `Y_i=1` means an independently graded correct answer, and `G_i` is a trusted task group. Fix thresholds `T={t_1,...,t_K}` and groups `G={g_1,...,g_J}` **before inspecting the correctness labels**.

For threshold `t` and group `g`, count `n(t,g)` accepted examples and `k(t,g)` errors among `S_i >= t`. Let `p(t,g)` denote the unknown future error rate conditional on this threshold and group, assuming the same data-generating distribution.

For each predeclared threshold/group slice, compute the **exact one-sided Clopper–Pearson binomial upper bound** `U(t,g)` by solving:

```text
P_{X ~ Binomial(n, U)}[X <= k] = alpha / M,
M = K * (J + 1)
```

The `+1` includes the aggregate slice. If `n=0` or `k=n`, set `U=1`. For zero observed errors the closed form is:

```text
U(0,n) = 1 - (alpha/M)^(1/n).
```

For each *fixed* slice, the one-sided binomial procedure covers the true conditional error rate with probability at least `1-alpha/M`. **By the union bound**, all `M` bounds hold simultaneously with probability at least `1-alpha`. This permits choosing the maximum-coverage threshold **after inspecting calibration outcomes** without treating the selected threshold as a fresh uncorrected test. It does not justify inventing new thresholds/groups after seeing the results.

A threshold is eligible only if every predeclared group **and** the aggregate has at least the configured minimum accepted support and `U <= target_error`. Otherwise the policy abstains. The output includes all candidate thresholds, counts, error rates, bounds, missing support, the chosen threshold or `null`, and a SHA-256 fingerprint of the calibration rows. Raw sample IDs and labels are not written to the report.

### Why 80 questions may not be enough

With `alpha=0.05`, three candidate thresholds and two groups, `M=9`. Even with **zero observed errors** among 40 accepted answers in a group, the exact upper bound is:

```text
1 - (0.05/9)^(1/40) ≈ 12.2%,
```

which **cannot** certify a 5% conditional error target. The system must abstain instead of exaggerating certainty. The required sample size for a low error target can be substantial.

## Signed, fail-closed release

A calibrated policy is HMAC-SHA256 signed with a separate 32+ byte secret. The `risk_qa_adapter.py` binds it to:

- The exact model revision and independent grading rubric.
- A **trusted, predeclared group** supplied by application routing (never by the model).
- A finite QA confidence/ranking score meeting the selected threshold.
- A separately verified source SHA-256/character-offset certificate whose model revision matches.
- A passing per-group and aggregate error bound, with sufficient calibration support.

If any condition fails, ĀRU returns an empty answer and a specific rejection reason. An invalid signature blocks inference **before the model is run**. HMAC prevents undetected edits by parties without the key, but a key holder can forge policies; key management and independent review are still required.

## Operator workflow

Use an independent, consented calibration dataset in JSONL format, with no raw source passages or prompts:

```json
{"sample_id":"heldout-001","score":0.91,"correct":true,"group":"technical"}
{"sample_id":"heldout-002","score":0.22,"correct":false,"group":"technical"}
{"sample_id":"heldout-003","score":0.83,"correct":true,"group":"general"}
```

**Important:** these three rows are illustrative schema examples, **not sufficient calibration data**.

```bash
# Keep the signing key out of command history and Git.
export ARU_RISK_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Freeze thresholds, groups, target, model and rubric BEFORE inspecting labels.
aru-risk fit --calibration-jsonl independent-calibration.jsonl \
  --thresholds 0.1,0.5,0.9 \
  --groups technical,general \
  --target-error 0.05 --alpha 0.05 --min-group-support 30 \
  --model-revision exact-immutable-model-commit \
  --rubric-id independent-human-gold-v1 \
  --calibration-set-id preregistered-calibration-001 \
  --out signed-risk-policy.json

# Verify HMAC, then check a specific source-certified model answer.
aru-risk verify --policy signed-risk-policy.json
aru-risk check --policy signed-risk-policy.json \
  --score 0.91 --group technical \
  --source-file approved-source.txt --proof-file answer-proof.json \
  --model-revision exact-immutable-model-commit \
  --rubric-id independent-human-gold-v1
```

The CLI exits nonzero on a rejected answer or invalid policy. **Do not use it as a substitute for verified consent, human review, tenant isolation or access control.**

## Reproducible falsification

`selective_risk_validation.py` simulates independently generated outcomes under two *known* error rates: a poor model (20% error) and a strong model (0.5% error), across two groups. It records how often the predeclared 10% risk screen passes. This is a seeded implementation sanity check, **not real-world proof**.

```bash
python selective_risk_validation.py --trials 120 --rows-per-group 150 \
  --seed 20261018 --output synthetic-risk-validation.json
python -m unittest discover -s tests -v
```

## Failure conditions and scope boundaries

The bound can become invalid if the calibration sample is nonrepresentative, correlated by article, selected after seeing labels, graded with a biased or changed rubric, missing difficult outcomes, or drawn from a different model/domain than future requests. A fixed threshold may fail on unseen populations. The chosen score need not be a calibrated probability and may not correlate with correctness. A valid source span is **not proof of truth**. A predeclared group can still conceal unmeasured disparities.

A policy passing this statistical screen is a **research recommendation for human evaluation**, not a license to deploy it. A production system additionally needs independently audited model and data provenance, secure authentication, privacy/consent, live drift monitoring, rollback, and appropriate statistical review.

**Commercial hypothesis:** enterprise teams may value a measurable, source-backed answer-release policy that is conservative enough to abstain when evidence is insufficient. Demand, pricing, model quality, customer ROI and enterprise security are not yet established.

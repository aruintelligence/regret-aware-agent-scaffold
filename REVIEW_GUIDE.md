# ĀRU Decision Evidence Lab — Engineering Review Guide

**Purpose:** make this large research PR reviewable, falsifiable, and safe to merge into a repository whose default runtime is a scripted simulator.

## What is and is not changed

The original `MasterAgentRuntime`, CLI behavior and internal cold/warm benchmark remain opt-in compatible. The new decision-evidence modules are separate entry points. No external model, network endpoint, or public web server starts automatically. The new local gateway binds to loopback only and requires explicitly provisioned secrets.

## Review map

| Layer | Key files | Verify |
| --- | --- | --- |
| Randomized decision logging | `route_experiments.py`, `external_adapter.py` | Action identity, exact propensities, candidate support |
| External outcome evaluation | `off_policy_evaluation.py`, `evidence_lab.py` | IPS/SNIPS/ESS, missingness, clipping, conservative human-review gate |
| Robustness and diagnostics | `crossfit_diagnostics.py`, `distribution_robustness.py` | Cross-fitting, domain regressions, adversarial topic-mix constraints |
| Real human-gold benchmark | `squad_sentence_retrieval.py`, `squad_benchmark.py` | No gold-answer leakage, dataset hash, direct-vs-logged comparisons |
| Opt-in real model path | `llm_executor.py`, `hf_local_server.py`, `llm_squad_benchmark.py`, `llm_paired_audit.py` | Actual CPU open-weight inference, disjoint public-data matched trials, consent, provenance and human-gold scoring |
| Source-grounding contract | `grounding_contract.py` | Exact source offsets and hashes, independent verification, abstention, source-trust limitations |
| Evidence provenance | `evidence_audit.py`, `evidence_checkpoint.py` | HMAC verification, downgrade rejection, independent key checkpoint, tail-truncation detection |
| Paired inference | `paired_inference.py` | Exact McNemar discordance arithmetic and dependence caveat |
| Local ingestion service | `evidence_gateway_store.py`, `evidence_gateway.py` | SQLite transactions, idempotency, schema allowlists, consent header, concurrency |
| CLI and reporting | `experiment_cli.py`, `evidence_dashboard.py` | Reproducible reports, accessible HTML, no automatic rollout |
| Public-facing research page | `docs/index.html`, `docs/styles.css` | Accurate claims, mobile readability, attribution, no deployed claims |
| Packaging and CI | `pyproject.toml`, `.github/workflows/*` | Python 3.9/3.12, installable CLIs, external benchmark |

## Required verification

```bash
python3 -m unittest discover -s tests -v
python3 benchmark.py --runs 5
python3 experiment_cli.py demo --out-dir demo-report --n 900 --seed 7
python3 scientific_validation.py --trials 40 --n 900 --seed 7
python3 -m pip install .
aru-evidence --help
aru-gateway --help
aru-squad --help
aru-llm-squad --help
```

For the human-annotated benchmark, download the official SQuAD v1.1 dev JSON and verify its SHA-256 matches the value in `BENCHMARK_EVIDENCE.md`. Execute `squad_benchmark.py` on 2,000 fixed hash-selected questions and inspect the article-level regression. No raw SQuAD dataset is committed to the repository.

## Explicit risk register

1. **No independent customer success evidence:** the measured 3.6 percentage-point gain applies to a simple sentence-retrieval task, not to real agent productivity or LLM reasoning.
2. **Public development set:** not an untouched private holdout. Do not tune the algorithm on this data and continue describing it as a fresh confirmatory test.
3. **Incomplete causal guarantees:** inverse-propensity estimators require honest prospective propensities, support and appropriate dependence assumptions. Exploratory bootstrap is not automatically valid for adaptive data.
4. **Gateway is local-only research infrastructure:** no TLS, SSO/RBAC, tenant isolation, encryption at rest, backup/retention/deletion API, or security audit.
5. **HMAC limits:** an unanchored HMAC chain silently accepts a deleted tail. A separately keyed checkpoint now detects truncation or rewrite **relative to the last retained anchor**. A key holder or privileged administrator can still rewrite history if the independent checkpoint/key are also compromised; external retention remains necessary.
6. **Consent limits:** the research opt-in header is self-attested and does not replace verified human consent or privacy compliance.
7. **Real model has now been executed:** public SmolLM2-135M-Instruct completed a 40-call randomized pilot and an 80-call matched follow-up. The matched grounded prompt did **not** improve F1. Source-gating on an additional disjoint subset remains an exploratory test. None of this proves enterprise agent quality.
10. **Source grounding is not factual verification:** an exact substring certificate proves textual provenance only; incorrect, irrelevant, or malicious source text can pass. Strict abstention may discard useful paraphrases.
8. **Public MIT license:** commercial reuse by third parties is permitted; verify contributor rights before asserting exclusivity or negotiating proprietary licensing.
9. **No automatic deployment:** even a positive review screen cannot switch production policies.

## Suggested merge gate

- Both CI workflows green on the current head.
- Human reviewer checks the gold-answer isolation boundary and reproducible dataset hash.
- Security reviewer examines API key, HMAC key, SQLite and consent handling.
- Statistical reviewer inspects OPE assumptions, domain regressions and topic-mix sensitivity.
- Confirm no raw private prompts or credentials were committed.
- Approve research-only merge; **do not deploy the gateway publicly**.

## Reversal

The new research components are separate modules and entry points. If a problem is discovered, revert the PR as a unit and retain the original `MasterAgentRuntime` implementation and baseline tests. Research datasets, HMAC keys and local SQLite files should remain outside Git history.

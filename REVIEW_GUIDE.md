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
| Sovereign source registry | `sovereign_source_registry.py`, `source_registry_anchor.py`, `source_registry_cli.py`, `qa_local_server.py` | Exact owner/source/purpose authorization, expiry, revocation, no raw source storage, HMAC tamper detection, anchored revocation rollback denial |
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

1. **No independent customer success evidence:** the measured 3.6 percentage-point gain applies to sentence retrieval, not to customer productivity or general AI reasoning.
2. **Public development set:** SQuAD dev is not a private hidden holdout; public-data experiments and follow-ups are exploratory.
3. **Incomplete causal guarantees:** off-policy estimators require honest prospective propensities, support, valid outcome grading and dependence assumptions.
4. **Gateway is local-only research infrastructure:** no TLS, SSO/RBAC, tenant isolation, encrypted metadata storage, verified consent, retention/deletion API or independent security audit.
5. **HMAC limits:** an unanchored hash chain silently accepts a deleted tail. Independently keyed checkpoints detect rollback only for the anchored prefix and only when retained outside the same trust boundary.
6. **Consent and identity are not independently verified:** operator-attested source grants do not establish legal consent. The owner header is caller-supplied and is not bound to an authenticated user identity.
7. **Revocation rollback requires prompt anchoring:** a privileged actor can delete newer unanchored revocations. Anchored QA mode detects rollback of previously anchored revocations; it does not protect unanchored changes.
8. **Real models have been executed but not validated commercially:** a small general-purpose LLM completed public-data tests; the restrictive prompt did not improve matched F1. The specialized QA experiment remains task-specific and subject to independent evaluation.
9. **Source grounding is not factual verification:** exact source spans prove textual provenance, not truth, relevance, entailment or immunity to malicious documents. Strict abstention can reject useful paraphrases.
10. **Public MIT license:** commercial reuse by third parties is permitted; verify contributor rights before asserting exclusivity or negotiating proprietary licensing.
11. **No automatic deployment:** even a positive review screen cannot switch production policies.

## Suggested merge gate

- Both CI workflows green on the current head.
- Human reviewer checks the gold-answer isolation boundary and reproducible dataset hash.
- Security reviewer examines API key, HMAC key, SQLite and consent handling.
- Statistical reviewer inspects OPE assumptions, domain regressions and topic-mix sensitivity.
- Confirm no raw private prompts or credentials were committed.
- Approve research-only merge; **do not deploy the gateway publicly**.

## Reversal

The new research components are separate modules and entry points. If a problem is discovered, revert the PR as a unit and retain the original `MasterAgentRuntime` implementation and baseline tests. Research datasets, HMAC keys and local SQLite files should remain outside Git history.

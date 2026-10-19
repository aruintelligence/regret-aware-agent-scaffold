# ĀRU Proof-Carrying Answer Contract — Prospective Public-Data Stress Protocol

**Status at creation:** source-grounding guard implemented and unit-tested; no actual model outcomes on the reserved offset-80 sample observed. This is a predeclared exploratory engineering test, not a private holdout or commercial validation.

## Product question

Can an AI assistant provide a verifiable citation to an exact source span, and abstain when its answer is not present, while transparently quantifying the lost coverage and task accuracy?

The contract deliberately proves a narrow, testable statement: **every accepted answer is a contiguous substring of the supplied passage and is accompanied by independently verifiable source SHA-256 and character offsets.** It does NOT prove the answer is relevant, correct, entailed, safe, or complete. Incorrect statements can appear verbatim in a source.

## Frozen experiment

- Dataset: official Stanford SQuAD v1.1 development split; SHA-256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
- Sample: **next 40 SHA-sorted questions, offset 80 through 119**. These are disjoint from prior 0–39 and 40–79 experiments.
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision recorded by the runtime.
- Two frozen prompts: direct concise answer and grounded-extractive answer.
- Both prompts run on each question, with randomized execution order (seed `20261018`), 48 output tokens maximum.
- A deterministic, predeclared grounding gate strips only limited wrappers, verifies an exact or case-insensitive word-bounded contiguous source span, returns the original source substring, or abstains.
- The gate never sees human-gold answers. Gold annotations are used only for grading after the model responds.
- **Primary engineering outcome:** among accepted answers, 100% pass independent source-offset and SHA-256 verification; zero unsupported displayed answers by construction.
- **Tradeoff metrics:** accepted coverage, rejected count, raw F1, F1 after abstention-as-zero, selective F1 on accepted answers, exact match, latency, and failures, reported separately by prompt.
- **Integration pass:** 80 attempted model calls, at least 36 complete question pairs, no more than four failures, no automatic rollout. **No required F1 improvement** or minimum acceptance rate.
- Reports contain aggregate metrics and hashes, not raw passages, model outputs or gold answer text.
- The test is public dev-set exploratory analysis. Do not claim general hallucination elimination: unsupported *source-span* outputs are withheld, but supported source text may still be false or irrelevant.

## Failure modes and follow-up

A strict gate may reject useful paraphrases and multi-line answers, may accept irrelevant source snippets, and may be misled by an untrusted or malicious source. In real deployments, source trust, relevance, entailment, user consent, prompt-injection defense, secure retention, and independent customer evaluation are separate unsolved requirements.

**No autonomous policy rollout.** Human reviewers decide whether this narrow provenance guarantee justifies the coverage/accuracy tradeoff.

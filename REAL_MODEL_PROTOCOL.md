# ĀRU Real Open-Weight Model — Preregistered Engineering Smoke Experiment

**Status at protocol creation:** no real-model CI outcome observed. This protocol fixes the evaluation configuration before inspecting results. It is a *software integration smoke test*, not a statistically powered scientific trial.

## Purpose

Demonstrate that a genuine, publicly downloadable, pretrained language model can run on a local CPU endpoint, receive public questions through ĀRU's existing decision-routing adapter, return real model-generated answers, and be graded against separately stored human-annotated answer text without fabricating results.

## Predeclared design

| Element | Fixed choice |
| --- | --- |
| Dataset | Stanford SQuAD v1.1 public development JSON |
| Dataset SHA-256 | `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` |
| Sample | First **40** answerable question IDs ordered by SHA-256 digest |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Weights | Public Hugging Face safetensors, remote custom code disabled |
| Revision | Resolve from model host at runtime; record immutable commit SHA in artifact |
| Runtime | Python 3.12, PyTorch 2.5.1 CPU, Transformers 4.48.3 |
| Decoding | Greedy (`temperature=0`), maximum 48 new tokens, two CPU threads |
| Routing | Uniform random 50/50 direct concise answer vs grounded extractive answer |
| Random seed | `20261018` |
| Main outcome | Standard normalized token-F1 vs any official human-gold answer |
| Secondary outcomes | Exact match, latency, reported tokens, model failure count |
| Labels | Recorded only for completed model requests, after execution |
| Data | Public questions only; do not archive raw passages, model outputs, or weights |
| Success criterion for integration | 40 requested, at least 36 completed, recorded model commit, no fabricated labels, all rollout controls disabled |
| Model quality success threshold | **None preregistered.** Report both arms, even if performance is zero. |
| Statistical significance claim | **None.** This sample is too small for credible superiority claims. |
| Automatic deployment | **Always forbidden.** |

## Data isolation

The selected model route receives the question and source passage but **never** the official answer list or answer offsets. The grader sees the response and the official answers *after* the model call. The decision log stores route ID, propensity and coarse public article signature; it does not store raw questions.

## Validity and failure rules

- A failed model request has no fabricated reward. The number and allocation of failures must be reported.
- If model weights cannot be downloaded, the job fails as **infrastructure unavailable**; no model-performance claim is made.
- If fewer than 36 requests complete, the integration job fails. Any partial reports must still be interpreted as incomplete.
- If one prompt arm has too few completed tasks, comparison is descriptive only.
- Public dev-set outcomes are **not** independent hidden holdout evidence and must not be used to tune the same prompts while claiming a fresh test.
- No benchmark score establishes general intelligence, real customer productivity, or security certification.

## Reproduction

```bash
python3 -m pip install '.[real-model]'
aru-local-model --model HuggingFaceTB/SmolLM2-135M-Instruct --threads 2 --port 1234
aru-llm-squad --dataset dev-v1.1.json --out-dir real-model-report \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --endpoint http://127.0.0.1:1234/v1/chat/completions \
  --limit 40 --seed 20261018 --max-tokens 48 --timeout-s 90 \
  --acknowledge-model-calls
```

For stricter reproducibility, rerun with the exact model commit hash recorded by the initial execution (`--revision` on `aru-local-model`). A model host may remove old revisions; preserve permitted model artifacts in an approved internal model registry if this becomes a production dependency.

## Commercial relevance

This test validates one important *integration* prerequisite for a prospective enterprise pilot. It does not validate market demand, ROI, production security, or the effectiveness of an AI routing policy on customer tasks. Those require a separately planned, consented, blinded and appropriately powered customer evaluation.

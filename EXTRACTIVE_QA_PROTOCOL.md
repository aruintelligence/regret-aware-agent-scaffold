# ĀRU Specialized Extractive QA — Predeclared Public-Data Experiment

**Protocol frozen before observing any model scores on the reserved subset.**

## Research hypothesis

The earlier small generative LLM produced few answers that could be verified as exact source spans (direct 2/40; extractive prompt 1/40; zero gated F1 on the third 40-question public sample). A pretrained **extractive question-answering model** may provide higher source-backed answer coverage because its architecture predicts start/end positions directly in the supplied passage.

This is a narrow engineering hypothesis, **not a claim that ĀRU invented a new model architecture or improved general intelligence**.

## Fixed design

- Official public Stanford SQuAD v1.1 development set, SHA-256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
- **80 new questions, SHA-sorted IDs offset 120–199**, disjoint from all earlier 0–119 evaluations.
- Specialized pretrained model: `distilbert/distilbert-base-cased-distilled-squad`. Its exact commit revision must be recorded. Weights must load as safetensors with remote custom code disabled.
- Small generative comparator: `HuggingFaceTB/SmolLM2-135M-Instruct`, direct concise answer prompt, greedy decoding, 48 output tokens.
- Each question and source passage is sent to both models; the human-gold answers are withheld until both responses exist.
- Extractive model uses tokenized context windows (384 tokens, 96 overlap), maximum 30 answer tokens, highest valid start/end logit sum, and an independently verified SHA-256 source-span certificate.
- Generative model is scored raw and after the strict existing source gate. Source-gated unsupported answers count as abstentions, not fabricated labels.
- **Primary engineering measure:** verified source-backed answer coverage of the specialized model, alongside normalized human-gold F1 and exact match.
- **Secondary:** matched QA minus raw-LLM F1, exploratory article-cluster bootstrap interval (400 replicates), per-model latency and failures.
- Integration pass: 80 questions requested, at least 72 complete pairs, at least 72 verified QA answers, immutable QA model revision recorded, zero accepted answers with invalid source certificates, no automatic rollout. **No minimum F1 gain is required.**
- Aggregate JSON/HTML only; no raw passages, model outputs, gold answers or model weights uploaded.
- This is a public dev-set comparison. The specialized QA model is explicitly fine-tuned for SQuAD-like tasks, while the 135M model is a tiny general-purpose LLM. Any observed improvement is **task-specific**, not a fair general-model leaderboard comparison.

## Falsification and failure

The hypothesis is weakened if the QA model frequently abstains, emits invalid spans, or achieves poor F1 despite high source coverage. A source-backed span can be incorrect or irrelevant. The experiment must report that honestly. Download failures or incompatible weights invalidate the run; do not replace real results with mocks. Missing model calls must be disclosed.

**No real customer, security certification, AGI, or production deployment claim.**

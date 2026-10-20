# ĀRU Decision Evidence Lab — evaluation brief

**Status:** open-source research alpha. This is a review package for a design partner, not a deployed enterprise service or a claim that an AI model became more accurate.

## The decision it helps a buyer make

When a team changes an AI assistant, it needs to know whether the observed work was actually issued, received, graded and monitored before approving the new version. A signed result by itself cannot answer that: a missing hard request, a replacement grader key, a stale checkpoint or an omitted adverse outcome can make an internally consistent report misleading.

ĀRU's narrow product hypothesis is an **evidence gate for one AI workflow**. It checks source-backed answers and a static risk policy, accounts for a closed set of requests and receipts, verifies every grade and its authorized key, recomputes a prospective group risk watch, and requires two pinned witness approvals over the same evidence. A failed check withholds the answer. A passed check is eligible for **human review**, never automatic rollout.

| Buyer question | Inspectable mechanism | Boundary |
| --- | --- | --- |
| Did every planned request enter the measured cohort? | Signed issuance and preinference receipt claims; manifest reconciliation | The archived receipts were created by CI, not independently delivered by customers |
| Did every request receive an authorized grade? | Ed25519 grade log, signed checkpoint, exact ID coverage | A valid signature cannot establish whether the grader's judgment is correct |
| Did the monitor include the exact ordered outcomes? | Recomputed signed watch report and expected outcome digest | A 50% research alarm target does not certify customer quality |
| Can a request supply its own trusted keys or replay old evidence? | Two setup-pinned witness keys, private high-water state, exact record review | Local state can be lost or compromised; there is no global freshness service |
| Can the final answer be traced to the claimed passage? | Source digest, offsets and exact extracted text | Textual provenance is not factual truth or entailment |

## What has actually been observed

- A [prospective public QA receipt challenge](RECEIPT_CHALLENGE_RESULTS.md) issued **120** requests before inference; CI produced **120** receipt claims, completed **120** real-model calls, independently recomputed **120** source-span certificates, and recorded **15** rubric errors with **zero execution failures**. There was **no alarm at the predeclared 50% research target**. One CI runner controlled issuance, inference, receipts and grading; this is an integration result, not customer independence.
- The [first real-model watch audit](REAL_MODEL_WATCH_RESULTS.md) recorded **12/120** errors on a separate fixed public-question run. In six *synthetic label-flip replays* against that stream, the original from-start monitor detected only one severe injected change and missed five. That negative result motivated later predeclared restart studies; it remains visible rather than being rewritten as success.
- The [witness falsification tests](GRADER_WITNESS_QUORUM.md) show that a fully signed, self-pinned replacement grade log can pass a low-level check but is denied by the pinned release gate. Tests also cover missing grades and receipts, altered watch outcomes, substituted witness identities, rollback, and unsafe local state permissions. These are local adversarial software tests, not an external security audit.
- A distinct [2,000-question sentence retrieval benchmark](BENCHMARK_EVIDENCE.md) measured **82.1%** answer-containing sentence selection versus **78.5%** for its baseline, with one topic regression that blocked advancement. It measures retrieval, not better LLM reasoning. The [real-model matched answer experiment](COMMERCIALIZATION.md) reported a negative F1 difference for its restrictive prompt.

## Reproduce the bounded checks

From this repository with Python 3.9 or newer:

```bash
python3 -m pip install '.[public-grader]'
python3 witness_gate_demo.py
python3 -m unittest -v tests.test_grader_witness_quorum
python3 -m unittest -v tests.test_receipt_cohort_ledger tests.test_public_grade_ledger
python3 -m unittest discover -s tests -q
```

The demo prints JSON showing an honest 40-request **synthetic fixture** accepted at epochs 1 and 2, five rejected attack attempts, and a low-level self-fed quorum that the pinned gate refuses. It exits nonzero on any unexpected acceptance. It imports repository test fixtures and uses public test keys; it makes **zero model calls** and is intended to run from the checkout. The archived real-model files, frozen protocols, limitations and run provenance are linked above. Unit tests use constructed evidence; they do not rerun model inference. GitHub Actions runs the regression suite on supported Python versions and the archived model workflows; inspect the current PR's checks for their actual status.

## A bounded design-partner offer

The existing [six-week pilot draft](DESIGN_PARTNER_PILOT.md) proposes **one non-consequential workflow**, a frozen baseline and candidate, customer-approved minimized tasks, a preregistered grading rubric, an independent evaluator, and an auditable go/no-go package. Its **$20,000 fixed-fee figure is a pricing hypothesis**, subject to discovery and agreement; it is not a validated market price or a current sale. Completion means reproducible evidence and a human decision, including an inconclusive or negative result.

Before a paid deployment, a customer and ĀRU would need to agree on verified request delivery, independently retained current checkpoints, separate witness and grader operations, key management, privacy and access controls, retention, failure recovery, statistical design and an external security review. The research implementation uses HMAC verification secrets at witness review time. No customer pilot, independent witness service, certification, or enterprise SLA is demonstrated here.

The code is [MIT licensed](LICENSE), so an exclusive license to the published implementation should not be promised. Confirm contributor rights and negotiate any customer-specific IP separately.

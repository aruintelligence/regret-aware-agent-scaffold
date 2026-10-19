# ĀRU Evidence Quorum — Multi-Source Agreement and Conflict Quarantine

**Status:** implemented research prototype; deterministic guarantees are tested, but no customer reliability or factual-truth claim has been established.

## What it does

Single-document source certificates establish that a displayed answer is present in a supplied passage. They do **not** establish that the passage is correct or that a second source agrees. The new `evidence_quorum.py` extends that narrow contract:

1. An AI model proposes a short answer for each source.
2. Every proposed answer is independently checked as a word-bounded contiguous substring of that source.
3. Duplicate source texts cannot increase the agreement count.
4. The operator must configure the purpose-scoped, revocable Sovereign Source Registry; unauthorized documents are rejected **before** inference.
5. If two verified sources give incompatible normalized answers, the system **abstains**, even if another answer already meets the minimum count.
6. If there is no conflict and at least `k` distinct source fingerprints agree, the system returns the original source wording and a reproducible evidence receipt.
7. An independent verifier recomputes the receipt from the original sources. The receipt contains fingerprints, character offsets and counts, not full passages.

### What it does not do

**Agreement is not truth.** Two documents can copy the same error; SHA-256-different documents are not necessarily epistemically independent. A malicious or incorrect source can contain a plausible answer. The system does not prove entailment, source authenticity, relevance, absence of prompt injection, or compliance certification.

The receipt SHA-256 is a checksum, **not** an authenticated signature. Independent verification requires access to the original source text and the operator allowlist. For stronger provenance, combine this with the separately keyed HMAC checkpoint system.

## Deterministic invariants

For candidate answer `a_i`, source `s_i`, source digest `h_i` and conservative normalized answer `n_i`:

- `verified(i)` means the displayed answer is a word-bounded substring of `s_i`, with independently checkable character offsets and `SHA256(s_i)`.
- `D(v) = |{h_i : verified(i) and n_i = v}|` counts distinct source fingerprints supporting answer `v`.
- `C = |{n_i : verified(i)}|` counts distinct verified answer groups.
- **Accept if and only if** `C == 1` and `D(v) >= k`.
- **Abstain** if `C > 1`, if no answer is verified, or if the distinct-source count is below `k`.
- Reject every source without an active, owner-matched, purpose-scoped grant; authorization is rechecked after inference.

This is a deterministic policy invariant, not a probabilistic statement about factual accuracy.

## Consent-scoped authenticated API

The local research QA server supports `POST /v1/qa/quorum` in addition to `POST /v1/qa/proof`. The quorum endpoint **fails closed unless a purpose-scoped Sovereign Source Registry is configured**. Every source must have an active, nonexpired, nonrevoked `qa_source` grant for the same operator-specified owner. The server checks every grant **before and after model inference**, reducing—but not eliminating—revocation race exposure.

```bash
# Generate and protect two separate secrets; never commit them.
export ARU_QA_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ARU_SOURCE_REGISTRY_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Each document is granted individually after real-world consent is checked.
# Owner identity is read from a local file, not placed in shell history.
aru-source-registry grant --db /secure/sources.db \
  --source-file /secure/document-a.txt --owner-file /secure/owner.txt \
  --purpose qa_source --ttl-seconds 3600 --consent-asserted
aru-source-registry grant --db /secure/sources.db \
  --source-file /secure/document-b.txt --owner-file /secure/owner.txt \
  --purpose qa_source --ttl-seconds 3600 --consent-asserted

aru-qa-serve --model distilbert/distilbert-base-cased-distilled-squad \
  --port 1235 --source-registry-db /secure/sources.db
```

Authenticated JSON request to `http://127.0.0.1:1235/v1/qa/quorum`:

```json
{
  "model": "distilbert/distilbert-base-cased-distilled-squad",
  "question": "Where was the telescope built?",
  "sources": [
    {"source_id": "document-a", "context": "The telescope was built in Oregon."},
    {"source_id": "document-b", "context": "A separate report places the telescope in Oregon."}
  ],
  "min_distinct_sources": 2
}
```

Headers: `Authorization: Bearer <ARU_QA_API_KEY>`, `X-ARU-Consent: research-opt-in`, `X-ARU-Owner: <operator-attested-owner>`, `Content-Type: application/json`.

The server allows **2–4 documents**, at most 5,000 characters per document, and refuses unknown fields. The operator-scoped registry checks each exact source text, expiry and revocation before inference and again before the receipt is returned. The response contains source fingerprints and offsets but no full passages. A revoked, expired, missing or mismatched grant blocks the request.

When the server is started with an independently retained source-registry checkpoint and separate anchor key, the quorum adapter verifies that checkpoint **before and after inference** as well. This blocks the known attack where an administrator deletes the latest revocation event from an otherwise internally consistent HMAC chain. An external checkpoint does not establish factual truth or independently verified legal consent.

**Consent caveat:** the registry stores operator attestations, not independently verified legal consent. Its HMAC event chain does not detect a deleted tail without a separately retained source checkpoint. Hash-distinct documents are not epistemically independent. The server remains IPv4-loopback-only research infrastructure without TLS, SSO/RBAC, multi-tenancy or external certification.

## Adversarial tests

The tests cover:

- Two distinct matching sources → accepted, independently reproducible receipt.
- Identical source text under two IDs → cannot count twice.
- Two agreeing sources plus one conflicting verified source → quarantine, no answer.
- Model answer absent from source → abstain.
- Modified receipt or modified source → independent verification fails.
- Purpose-scoped source grant → permitted; expired or revoked grant → rejected before any model call.
- Reused source ID with changed contents → rejected.
- Untrusted or factually false but matching sources → can still pass the textual agreement check, explicitly demonstrating the limit.

## Commercial research hypothesis

A customer may value a system that **does not quietly choose between contradictory documents**, particularly in contract review, compliance research, engineering records, or internal knowledge retrieval. That value must be established through real customer studies measuring accepted-answer coverage, conflict sensitivity, false agreement, latency, cost, and human-review workload. Do not sell this as a factual-truth oracle.

**Research principle:** a transparent abstention is often preferable to a confident unsupported claim, but the cost of abstention must also be measured.

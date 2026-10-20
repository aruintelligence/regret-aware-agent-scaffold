# Customer-held request receipt challenge

**Status:** opt-in research control. The receipt is an evidence claim about issuance, not proof of execution, grading quality, customer safety, or a deployed service.

The [closed-cohort ledger](COMPLETE_OUTCOME_LEDGER.md) catches missing grades **after** a request appears in the issued manifest. A customer who holds a separately signed receipt can also challenge a request **omitted or rewritten before** that manifest was signed. The receipt key and ingress checkpoint key must be distinct, separately controlled secrets of at least 32 bytes. The customer retains the receipt outside the manifest operator's control.

## Issuance, closure, challenge

At the trusted request boundary, assign a unique opaque request ID and contiguous one-based sequence for a predeclared cohort. Deliver a receipt **before inference or grading**, at the same time as writing that exact issue row to the ingress log. The receipt service signs the five manifest fields, cohort ID, format version and a fresh 128-bit random nonce using a domain-separated HMAC-SHA256. The signed receipt contains no prompt or answer. The CLI takes a JSON object with exactly `sequence`, `request_id`, `group`, `model_revision`, and `rubric_id`.

```bash
# Provision distinct secrets in separately controlled services. Retain the
# verifier keys safely; never put them in a customer receipt or repository.
export ARU_RECEIPT_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export ARU_ISSUED_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

aru-watch-ledger receipt --issue-json one-issued-row.json \
  --cohort-id cohort-001 --out customer-receipt.json
```

The receipt output is created exclusively with mode 0600 on POSIX. Deliver it to the requester and let the requester retain it. The ingress operator later closes and signs the **complete** manifest. The verifier obtains the current closed checkpoint HMAC from a separately controlled store; copying it from the checkpoint under challenge would defeat the freshness check.

```bash
aru-watch-ledger anchor --issued-jsonl issued.jsonl \
  --cohort-id cohort-001 --groups general \
  --model-revision fixed-model --rubric-id fixed-rubric \
  --out ingress-checkpoint.json

aru-watch-ledger challenge --receipt customer-receipt.json \
  --issued-jsonl issued.jsonl --checkpoint ingress-checkpoint.json \
  --expected-checkpoint-hmac CURRENT_HMAC_FROM_SEPARATE_STORE
```

The challenge authenticates the receipt, authenticates the current closed checkpoint, verifies the **entire** ordered manifest against its signed digest, matches the cohort/model/rubric/group scope, then compares the receipt's exact issue row at its claimed position. Its JSON result contains `included`, a reason, cohort ID, sequence, receipt digest, checkpoint HMAC, and `automatic_rollout_permitted: false`. Exit `0` means included; exit `5` means a valid held receipt is missing or rewritten in the current signed manifest; exit `4` means the evidence is invalid, mismatched or stale. CLI usage errors exit `2`. A finding should trigger investigation; it does not by itself prove which party acted incorrectly.

| Evidence | What the verifier can conclude |
|---|---|
| Valid receipt and exact row in current signed manifest | This claimed issuance is represented in that closed cohort. |
| Valid receipt and row missing/rewritten in current signed manifest | The receipt issuer's claim conflicts with the ingress operator's signed cohort. |
| Invalid receipt, bad manifest signature/digest, wrong scope, or stale checkpoint | No inclusion or omission conclusion; fail closed. |

The test suite issues a receipt for each of 120 synthetic sequence positions, independently signs a valid manifest with that position rewritten, and detects **120/120** such conflicts. It also tests truncation, stale checkpoints, malformed receipts, key reuse and the CLI statuses. These are software falsification cases, **not** 120 customer requests or a statistical performance estimate. The earlier [120-call real-model cohort](PROSPECTIVE_LEDGER_RESULTS.md) had no customer-held receipts at issuance; this feature does not retroactively authenticate that run.

The [frozen follow-on protocol](RECEIPT_CHALLENGE_PROTOCOL.md) added CI-prepared receipt claims before another real-model run. The [measured results and exact evidence](RECEIPT_CHALLENGE_RESULTS.md) account for 120/120 claims, model calls and grades. The original audit protocol and its recorded hash remain intact. CI preparation did not deliver a receipt to an independently controlled customer.

An [independent receipt-cohort ledger](RECEIPT_COHORT_LEDGER.md) can also close the issuer's *entire* receipt log and require one-to-one agreement with the ingress manifest before the release gate. The customer's held copy remains useful if the issuer's own log omits their receipt.

## Limits and deployment requirements

The HMAC only authenticates a claim made by the receipt issuer. The CLI cannot prove the receipt was handed to the customer at request time, that the ingress row was durably recorded, that a request was executed, or that a grade is true. The issuer and ingress operator must be independent for a conflict to be meaningful; a compromised or colluding receipt issuer can sign after the fact. A request that never receives a receipt remains invisible to this control. The current schema cannot distinguish two receipts for the **same** five-field issue row; unique ID and sequence assignment needs an audited atomic ingress implementation. The nonce gives receipts distinct bytes but is not bound into the manifest.

This prototype has no external timestamp, transactional storage, public key signature, customer identity or live rolling cohort protocol. HMAC verification requires a trusted verifier with the secret; losing the keys prevents later cryptographic reauthentication. For a customer pilot, operate receipt issuance and ingress closure under separate controls, deliver receipts to customers before inference, retain current checkpoints and keys durably outside the monitoring operator, enforce atomic ID/sequence uniqueness, and have an independent grader validate every issued request. Preserve this separation in a prospective representative audit before making commercial or safety claims.

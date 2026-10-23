# Independent receipt-cohort accounting

**Status:** opt-in research control. This is a proposed trust boundary for a customer pilot, not a deployed independently operated service or a safety certificate.

The [individual receipt challenge](CLIENT_RECEIPT_CHALLENGE.md) lets one requester expose a conflict. `receipt_cohort_ledger.py` lets a release gate compare **the issuer's full closed receipt log** with the ingress operator's full closed manifest. Each party signs a separate checkpoint using a distinct secret. The verifier obtains both *current* checkpoint fingerprints from stores outside the report and receives both exact ordered logs. It verifies every receipt signature, unique request ID and nonce, contiguous sequence, scope, both signed log digests, then compares each receipt to the corresponding ingress row. An unreceipted extra ingress row is also a conflict.

This independent receipt log must actually be operated outside the ingress operator's control, written at the trusted request boundary before inference, with unique IDs and sequences assigned atomically. The code enforces the properties of the *supplied logs*; separate HMAC keys alone do not prove separate operations or timely customer delivery.

## Close the two logs

Give each customer their own receipt immediately with `aru-watch-ledger receipt`; append exactly that signed JSON object to the issuer's `receipts.jsonl`, outside the ingress operator's control. The ingress operator separately appends the five-field issue row to `issued.jsonl`. After the cohort closes, sign each complete log and retain the checkpoints outside the monitoring store:

```bash
# Provision three distinct secrets in separate services; the watch key is a
# fourth distinct secret. Do not commit or print any of them.
export ARU_RECEIPT_SIGNING_KEY="...at least 32 bytes..."
export ARU_ISSUED_SIGNING_KEY="...a different 32-byte secret..."

aru-watch-ledger receipt-anchor --receipts-jsonl receipts.jsonl \
  --cohort-id cohort-001 --groups general \
  --model-revision fixed-model --rubric-id fixed-rubric \
  --out receipt-checkpoint.json

aru-watch-ledger anchor --issued-jsonl issued.jsonl \
  --cohort-id cohort-001 --groups general \
  --model-revision fixed-model --rubric-id fixed-rubric \
  --out ingress-checkpoint.json
```

Obtain **both current HMAC fingerprints** from separately controlled checkpoint stores. Copying a fingerprint from the candidate checkpoint itself defeats the freshness test. Compare the exact logs:

```bash
aru-watch-ledger receipt-audit \
  --receipts-jsonl receipts.jsonl --issued-jsonl issued.jsonl \
  --receipt-checkpoint receipt-checkpoint.json \
  --ingress-checkpoint ingress-checkpoint.json \
  --expected-receipt-hmac CURRENT_RECEIPT_HMAC_FROM_ISSUER_STORE \
  --expected-checkpoint-hmac CURRENT_INGRESS_HMAC_FROM_INGRESS_STORE
```

Exit `0` reports complete one-to-one alignment; exit `5` reports an authenticated mismatch with conflict sequence numbers and both counts; exit `4` means at least one log/checkpoint/key/current anchor is invalid or stale, so no conclusion is drawn. CLI usage errors exit `2`. Every result explicitly sets `automatic_rollout_permitted=false`. The Python `gate_with_receipt_ledger(...)` runs this reconciliation **before** the existing static policy, signed watch, current outcome digest and source-proof gate. It refuses a conflict or unverifiable receipt ledger, even when the earlier gate alone would accept the re-signed ingress/watch.

## Falsification and limits

The tests independently re-sign an otherwise valid ingress manifest after rewriting each of 40 positions; all 40 conflicts are detected. A separate gate test builds a valid signed watch for a rewritten request: the previous gate accepts its self-consistent evidence, while the receipt-gated path blocks it. Tests also cover truncation, unreceipted extra rows, duplicate receipts, forged data, stale fingerprints, key reuse and the complete CLI path. These are bounded software tests, **not** evidence that a customer received a receipt or that independent operators maintained the logs.

The issuer could omit a receipt from its own log after delivery, or refuse to issue one at all; a customer-held [individual challenge](CLIENT_RECEIPT_CHALLENGE.md) remains necessary. Colluding or compromised operators can manufacture matching logs. HMAC verification requires trusted key custody; a leaked key destroys secrecy-based provenance. Checkpoint signatures provide no external timestamp or transactional write guarantee. The archived [120-call public QA run](RECEIPT_CHALLENGE_RESULTS.md) prepared receipt claims within one CI runner but did **not** create an independently operated receipt ledger. The new gate has not been validated on customer traffic, and the watch's statistical assumptions and grader accuracy remain unverified. It never permits automatic rollout.

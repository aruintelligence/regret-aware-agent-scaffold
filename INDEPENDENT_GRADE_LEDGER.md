# Independent grade attestation and exact watch verification

**Status:** opt-in research interface. A cryptographic grade claim establishes what the holder of a grader key signed. It does not establish that a human reviewed the result, that the reviewer was independent, or that the binary label is correct.

The [issued-request ledger](COMPLETE_OUTCOME_LEDGER.md) accounts for every request and grade, and the [receipt-cohort gate](RECEIPT_COHORT_LEDGER.md) detects discrepancies between the receipt issuer and ingress. Its ordinary grade rows still contain only a self-asserted `grader_id`. `grade_attestation_ledger.py` adds a fourth separately controlled key: the grader signs each binary outcome when ready and retains an arrival-ordered graded log. The verifier requires the grader's *current* signed checkpoint, the ingress operator's *current* signed checkpoint, both exact logs and one-to-one request-ID coverage. It then recomputes the signed watch from the authenticated labels before calling the receipt and existing risk gates.

## Independent grader workflow

The grader receives an opaque request ID, a frozen cohort/model/rubric, and enough task context to evaluate outside this metadata log. The grade JSON contains exactly `request_id`, strict Boolean `error`, `rubric_id` and `grader_id`. `grade_sequence` records **grade arrival order**, which may differ from issuance order. A random nonce and all scope fields are covered by a domain-separated HMAC-SHA256. Raw prompts and answers are excluded from the attestation.

```bash
# Separate from ingress, receipt issuer and monitor keys. Keep it in the
# independently operated grader service, never in the repository or logs.
export ARU_GRADER_SIGNING_KEY="...at least 32 bytes..."

aru-watch-ledger grade --grade-json one-grade.json \
  --grade-sequence 1 --cohort-id cohort-001 \
  --model-revision fixed-model --out signed-grade.json

# Append each signed grade to the grader-owned grades.jsonl in arrival order.
aru-watch-ledger grade-anchor --attestations-jsonl grades.jsonl \
  --cohort-id cohort-001 --model-revision fixed-model \
  --rubric-id fixed-rubric --out grader-checkpoint.json
```

Obtain the two **current** checkpoint fingerprints from independently controlled stores. The `grade-reconcile` command authenticates every grade signature and log digest, checks unique IDs/nonces, exact cohort scope and one-to-one issuance coverage, restores issued order, then signs the watch with the separate monitoring key:

```bash
aru-watch-ledger grade-reconcile \
  --attestations-jsonl grades.jsonl --issued-jsonl issued.jsonl \
  --grade-checkpoint grader-checkpoint.json \
  --ingress-checkpoint ingress-checkpoint.json \
  --expected-grade-hmac CURRENT_HMAC_FROM_GRADER_STORE \
  --expected-checkpoint-hmac CURRENT_HMAC_FROM_INGRESS_STORE \
  --out signed-watch.json
```

Exit `0` writes a complete watch with no alarm; `10` writes a complete watch with an alarm; `3` means an authenticated grade is missing or foreign and **no watch is written**; `4` means the signatures, scopes, logs, keys or current fingerprints are invalid or stale. CLI usage errors exit `2`. The Python `gate_with_authenticated_grades(...)` requires this grader evidence, exact watch recomputation, the complete receipt/ingress reconciliation, static calibration, current outcome digest and source proof. No path permits automatic rollout.

## Falsification and limits

In the tests, the monitoring operator re-signs an otherwise valid watch after changing one binary grade. The receipt-only gate accepts the internally consistent replacement. The grader-gated path rejects **all 40 positions in both directions**, including hiding the one genuine error in a 40-request cohort. It compares the watch to a separately signed grader log, so the monitor's new signature cannot make the false label authentic. Tests also cover grade arrival out of issue order, missing and foreign grades, duplicate IDs, malformed Booleans, changed rubric/scope, stale checkpoints, wrong keys, and CLI pending/complete states. These are software falsifications, not a measured grader error rate.

The previous [public real-model audit](PROSPECTIVE_LEDGER_RESULTS.md) used a repository-authored grader under one CI runner. It had **no independently signed grade attestations at scoring time**; new signatures cannot retroactively establish independence or correct labels. The grader may sign an incorrect label or collude with monitoring. The issuer and grader logs need durable atomic writes, external current anchors, independent key custody, task access controls, documented rubric review and representative prospective customer evaluation. HMACs are symmetric secrets, so a leaked verifier key destroys provenance. The code neither establishes an external timestamp nor validates the statistical conditional-null assumption of the watch.

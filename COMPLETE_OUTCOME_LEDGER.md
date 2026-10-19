# Closed-cohort outcome accounting pilot

**Status:** opt-in research module. This is an accounting and release-gate prototype, not a live service or a safety certificate.

An alarm-free watch can be misleading if difficult requests disappear before grading. `watch_outcome_ledger.py` joins two separately handled logs for a **closed cohort**: an ingress manifest of requests and a grader log of binary outcomes. A separate ingress key signs the manifest checkpoint; the watch key signs a report only after every issued request has **exactly one** grade. The gate requires a currently expected checkpoint fingerprint supplied outside the report. It also requires the current outcome digest, signed static calibration, source proof, scope alignment, group support, and absence of a watch alarm.

## Two sources, one ordered stream

Record issuance at the trusted request boundary, before knowing the outcome. A manifest row has exactly these five fields; `sequence` starts at 1 with no gaps. Keep raw prompts and answers outside this log.

```json
{"sequence":1,"request_id":"opaque-1","group":"general","model_revision":"model-immutable-sha","rubric_id":"qa-rubric-v1"}
{"sequence":2,"request_id":"opaque-2","group":"general","model_revision":"model-immutable-sha","rubric_id":"qa-rubric-v1"}
```

Once the ingress operator closes a fixed cohort, sign the complete issued manifest and retain the checkpoint in an independently controlled store. Fix the ordered groups, model, rubric, error target, alpha, and betting grid **before examining grades**. This CLI fixes the library's default betting grid; use the Python API for another grid. A signed checkpoint alone cannot prove when this step occurred.

```bash
# Provision separately controlled, secret, distinct keys; do not commit them.
export ARU_ISSUED_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export ARU_WATCH_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

aru-watch-ledger anchor --issued-jsonl issued.jsonl \
  --cohort-id cohort-2026-10-19-001 --groups general \
  --model-revision model-immutable-sha --rubric-id qa-rubric-v1 \
  --target-error 0.1 --alpha 0.05 --out ingress-checkpoint.json
```

An independently operated grader supplies rows with exactly `request_id`, strict Boolean `error`, `rubric_id`, and `grader_id`. The identifier identifies the purported grader; it is **not** cryptographic proof of who graded or whether the label is correct.

```json
{"request_id":"opaque-2","error":true,"rubric_id":"qa-rubric-v1","grader_id":"reviewer-1"}
{"request_id":"opaque-1","error":false,"rubric_id":"qa-rubric-v1","grader_id":"reviewer-1"}
```

Grade arrival may differ from request order. Reconciliation restores the frozen issuance order, rejects duplicate, foreign, malformed, non-Boolean or wrong-rubric grades, and verifies the signed manifest exactly. The output binds the issued-manifest digest, ordered-grade digest, both counts, cohort ID and ingress HMAC to the signed watch.

```bash
aru-watch-ledger reconcile --issued-jsonl issued.jsonl \
  --grades-jsonl grades.jsonl --checkpoint ingress-checkpoint.json \
  --out complete-watch.json
```

Exit code `0` means a complete report without an alarm; `10` means a complete report **with an alarm**; `3` means at least one grade is pending, and **no report is written**. Invalid input exits `4` (CLI usage errors exit `2`). For ongoing monitoring, close smaller cohorts and obtain complete labels for each; unresolved grades must remain visible and cannot be silently omitted. The current prototype does not maintain a rolling live window.

For an independent recomputation, the verifier needs the exact issued and graded files, both verifier keys, and the **current** checkpoint HMAC supplied by the ingress system rather than copied from the report:

```bash
aru-watch-ledger verify --issued-jsonl issued.jsonl \
  --grades-jsonl grades.jsonl --checkpoint ingress-checkpoint.json \
  --report complete-watch.json \
  --expected-checkpoint-hmac CURRENT_INGRESS_HMAC_FROM_SEPARATE_STORE
```

The Python `gate_with_complete_ledger(...)` additionally requires an independently obtained `expected_outcomes_sha256`; it refuses a stale, missing or inconsistent ingress checkpoint, an unsigned or incomplete report, a mismatched model/rubric/watch design, or any failure of the existing static-calibration and source-proof gate. The gate never permits automatic rollout. **Supplying the HMAC or outcome digest from the report itself defeats the freshness check.** A checkpoint for an older closed cohort cannot certify coverage of requests issued afterward.

## Tested falsification and trust boundary

The tests remove one or more grades, duplicate a grade, add an unissued grade, change a rubric or label, reorder and truncate issuance, alter signed fields, replay an old checkpoint, and submit an ordinary signed watch without a ledger. Missing grades produce only `pending_grades`, and the gate rejects the altered or unbound evidence. A CLI end-to-end test covers pending, completed, verified and stale states.

We also **retrospectively reconstructed** a manifest from the archived 120-call real-model intervention trace. The replay joined all 120 outcomes (49 errors). We removed **each grade in turn, all 120 possible single omissions**, and every replay returned `pending_grades` without a signed report. This proves a bounded software behavior on archived outcomes; the manifest was reconstructed after the experiment and **is not independent ingress evidence**. No customer-grade completeness or live alarm latency is claimed.

The [preinference audit protocol](PROSPECTIVE_LEDGER_PROTOCOL.md) now reserves fresh public questions 680–799 and freezes a signed manifest **before** new real-model inference in CI. It will report the observed accounting result separately. Both stages run under one CI authority, so the new sequence must not be represented as independent customer ingress.

Completeness holds only relative to requests captured by a separately controlled ingress process at the **current closed checkpoint**. The code cannot detect a request never entered at ingress, prove that an independent grader actually did the work or labeled correctly, prevent collusion or compromised keys, establish an external timestamp, resolve delayed feedback, or make local JSONL atomic or tamper-proof. For a customer pilot, instrument ingress before outcomes, store the checkpoint outside the monitoring store, independently authenticate the grader, publish current checkpoint and grade-log digests to a separate verifier, and audit a prospective representative stream. This accounting control does not validate the watch's conditional-null or static calibration assumptions.

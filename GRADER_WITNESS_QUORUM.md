# Witnessed grader anchors and rollback state

**Status:** opt-in research interface. The [public-key grader ledger](PUBLIC_GRADE_LEDGER.md) verifies the grader's signatures, but its low-level gate accepts a public-key fingerprint and a claimed current checkpoint digest from its caller. Copying those values from an attacker's replacement log defeats that freshness assumption. `grader_witness_quorum.py` binds those claims to a **two-signature witness statement** and remembers the last accepted statement per cohort in verifier-owned SQLite state.

## What two witnesses sign

The exact canonical statement includes: cohort ID, monotonically increasing epoch, previous statement SHA-256, grader public-key SHA-256, signed grader-checkpoint SHA-256, ingress-checkpoint HMAC, receipt-checkpoint HMAC, ordered outcome SHA-256, model revision and rubric. Each witness signs the statement digest and its own identity/fingerprint with a domain-separated Ed25519 signature. `verify_witness_quorum` requires at least two distinct identities and two distinct public keys from a **host-pinned map**, then verifies every supplied signature against the same statement. A witness key must differ from the grader key.

The release host controls the trusted witness key map and the state path. Neither may come from a request or the evidence bundle. A witness should use `review_and_sign_witness_statement(...)` with its own authorized grader key fingerprint, private signing key, HMAC verification keys and private state file. The function validates the complete signed grader log, receipt and ingress logs, and recomputed watch report against the statement **before** returning an approval. It records the witness's own monotonic head, so that witness will refuse to approve a rollback after it has seen a successor. `sign_witness_statement(...)` remains a low-level primitive: it signs arbitrary valid statements without evidence review. Use separately operated key custodians if actual organizational independence is required.

## Verifier setup and gate

Create a persistent, private state database once from the release host:

```python
from grader_witness_quorum import initialize_witness_state
initialize_witness_state("/private/aru/witness-heads.sqlite3")
```

The path must exist at gate time, be a regular file owned by the verifier, and have no group or other permissions on POSIX. The host pins at least two witness public keys out of band. Each witness can generate a separate Ed25519 key pair using `aru-grade-public keygen`; each should initialize its **own** witness state file and review the evidence before signing:

```python
from grader_witness_quorum import (
    initialize_witness_state, review_and_sign_witness_statement,
)

initialize_witness_state("/private/review-a/witness.sqlite3")
approval = review_and_sign_witness_statement(
    statement, witness_id="review-a", witness_private_key=witness_private_key,
    witness_state_path="/private/review-a/witness.sqlite3",
    authorized_grader_public_key_sha256=witness_held_grader_pin,
    grader_public_key=grader_public_key, attestations=attestations,
    grade_checkpoint=grade_checkpoint, receipts=receipts,
    receipt_checkpoint=receipt_checkpoint, issued_rows=issued_rows,
    ingress_checkpoint=ingress_checkpoint, watch_report=watch_report,
    receipt_key=receipt_key, ingress_key=ingress_key, watch_key=watch_key,
)
```

The fingerprint and keys must come from witness-controlled configuration, not the request. Another witness runs the same review with its own state and key; a coordinator gathers both approvals. It cannot change the statement without invalidating both signatures.

`gate_with_witnessed_public_grades(...)` takes the statement, approvals, the host's pinned witness key map and state path, plus the full grade, receipt, ingress, policy, watch and source evidence. It **derives** all five expected key/checkpoint/outcome fingerprints from the witnessed statement and refuses caller-supplied substitutes. It invokes the exact [public-grader gate](PUBLIC_GRADE_LEDGER.md), then advances the verifier's witness head even if a valid witnessed alarm or policy denial blocked release. The existing no-automatic-rollout behavior remains.

The SQLite transaction accepts the first genesis epoch, repeats the same statement idempotently, or accepts exactly one successor referencing the stored statement hash. It rejects an earlier epoch, a competing statement at the same epoch, and a skipped or unlinked successor. A verifier that has recorded epoch 2 therefore rejects a replay of epoch 1. Concurrent writers serialize through SQLite's immediate transaction.

## Falsification

```bash
python -m unittest -v tests.test_grader_witness_quorum
```

The counterexample constructs a **fully signed attacker grader log and checkpoint** that hide a real error. If the low-level public-grader gate is handed the attacker's own key pin, checkpoint digest and forged watch outcome digest, it accepts the internally consistent evidence. The witnessed gate, using two different host-pinned witness keys and the honest statement, rejects that substitution. Reviewed signing tests refuse a wrong grader pin, missing grade or receipt, tampered watch, changed checkpoint, shared signing keys and stale or forked witness epochs; valid independent signatures verify as a quorum. Gate tests also cover a single witness, duplicate identity or signing key, caller-supplied anchor overrides, an uninitialized state file and skipped epoch.

This is a local consistency boundary, **not a trusted global clock or public transparency service**. An old statement can still pass on a verifier that has never observed a newer one; deletion, replacement or rollback of any state file defeats that high-water mark. Witness review currently needs the receipt, ingress and watch HMAC verification secrets, so the witness service must protect them and actual role separation requires independent deployment and key custody. The host must back up and protect state, pin keys through its own change control, and handle key compromise and rotation. Two signatures cannot prove two independent human reviewers, grading accuracy, honest request delivery or statistically valid monitoring. The archived real-model cohort was not witnessed at scoring time and gains no retroactive assurance.

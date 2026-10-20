# Witnessed grader anchors and rollback state

**Status:** opt-in research interface. The [public-key grader ledger](PUBLIC_GRADE_LEDGER.md) verifies the grader's signatures, but its low-level gate accepts a public-key fingerprint and a claimed current checkpoint digest from its caller. Copying those values from an attacker's replacement log defeats that freshness assumption. `grader_witness_quorum.py` binds those claims to a **two-signature witness statement** and remembers the last accepted statement per cohort in verifier-owned SQLite state.

## What two witnesses sign

The exact canonical statement includes: cohort ID, monotonically increasing epoch, previous statement SHA-256, grader public-key SHA-256, signed grader-checkpoint SHA-256, ingress-checkpoint HMAC, receipt-checkpoint HMAC, ordered outcome SHA-256, model revision and rubric. Each witness signs the statement digest and its own identity/fingerprint with a domain-separated Ed25519 signature. `verify_witness_quorum` requires at least two distinct identities and two distinct public keys from a **host-pinned map**, then verifies every supplied signature against the same statement. A witness key must differ from the grader key.

The release host controls the trusted witness key map and the state path. Neither may come from a request or the evidence bundle. Witness operators must independently inspect the exact checkpoint and stream claims *before* signing; the Python signing function cannot determine whether they did so. Use separately operated key custodians if actual organizational independence is required.

## Verifier setup and gate

Create a persistent, private state database once from the release host:

```python
from grader_witness_quorum import initialize_witness_state
initialize_witness_state("/private/aru/witness-heads.sqlite3")
```

The path must exist at gate time, be a regular file owned by the verifier, and have no group or other permissions on POSIX. The host pins at least two witness public keys out of band. Each witness can generate a separate Ed25519 key pair using `aru-grade-public keygen`, then sign a reviewed statement with `sign_witness_statement(...)`. A coordinator gathers their independent approvals; it cannot change the statement without invalidating both signatures.

`gate_with_witnessed_public_grades(...)` takes the statement, approvals, the host's pinned witness key map and state path, plus the full grade, receipt, ingress, policy, watch and source evidence. It **derives** all five expected key/checkpoint/outcome fingerprints from the witnessed statement and refuses caller-supplied substitutes. It invokes the exact [public-grader gate](PUBLIC_GRADE_LEDGER.md), then advances the verifier's witness head even if a valid witnessed alarm or policy denial blocked release. The existing no-automatic-rollout behavior remains.

The SQLite transaction accepts the first genesis epoch, repeats the same statement idempotently, or accepts exactly one successor referencing the stored statement hash. It rejects an earlier epoch, a competing statement at the same epoch, and a skipped or unlinked successor. A verifier that has recorded epoch 2 therefore rejects a replay of epoch 1. Concurrent writers serialize through SQLite's immediate transaction.

## Falsification

```bash
python -m unittest -v tests.test_grader_witness_quorum
```

The counterexample constructs a **fully signed attacker grader log and checkpoint** that hide a real error. If the low-level public-grader gate is handed the attacker's own key pin, checkpoint digest and forged watch outcome digest, it accepts the internally consistent evidence. The witnessed gate, using two different host-pinned witness keys and the honest statement, rejects that substitution. The tests also cover a single witness, duplicate identity or signing key, a changed statement, caller-supplied anchor overrides, an uninitialized state file, idempotent replay, rollback after a successor, same-epoch fork and skipped epoch.

This is a local consistency boundary, **not a trusted global clock or public transparency service**. An old statement can still pass on a verifier that has never observed a newer one; deletion, replacement or rollback of the verifier's state defeats the high-water mark. The host must back up and protect the state, pin witness keys through its own change control, and handle key compromise and rotation. Two signatures cannot prove two independent human reviewers, grading accuracy, honest request delivery or statistically valid monitoring. The archived real-model cohort was not witnessed at scoring time and gains no retroactive assurance.

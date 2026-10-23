# Public-key grader attestations

**Status:** opt-in research interface. The Ed25519 path removes the grader's *signing secret* from the watch verifier. It does not prove that a human reviewed the result, that the grader was organizationally independent, or that the binary label is correct.

The [HMAC grade ledger](INDEPENDENT_GRADE_LEDGER.md) requires its verifier to know the same secret that signs grades. This alternative uses a grader-only private key to sign each result and the closed log; the verifier needs only the public key. The existing ingress, receipt, watch, policy and source gates retain their own separate keys and checks.

## Grader and verifier workflow

Install with `pip install '.[public-grader]'`. Generate the key pair **inside the grader's independently controlled environment**:

```bash
aru-grade-public keygen --private-out grader-private.json --public-out grader-public.json
```

The CLI creates both files exclusively with mode `0600` on POSIX and prints only the public-key SHA-256 fingerprint, never the private seed. The private file contains an unencrypted 32-byte Ed25519 seed in hex: store it outside the repository in a protected secret store. The public file can be distributed, but its fingerprint must be pinned from a trusted channel independent of the artifacts being verified. A public key copied from the grade checkpoint itself is **not** a trust anchor.

The grader receives the opaque request ID, frozen cohort/model/rubric and the material needed to score the result. A grade JSON has exactly `request_id`, a strict Boolean `error`, `rubric_id` and `grader_id`. Grade arrival order can differ from request order. Each signature covers the grade, scope, arrival sequence, random nonce and public-key fingerprint with an Ed25519 domain distinct from the checkpoint domain:

```bash
aru-grade-public grade --private-key grader-private.json \
  --grade-json one-grade.json --grade-sequence 1 \
  --cohort-id cohort-001 --model-revision fixed-model \
  --out signed-grade.json

# Append signed grades in grade arrival order to grader-owned grades.jsonl.
aru-grade-public close --private-key grader-private.json \
  --attestations-jsonl grades.jsonl --cohort-id cohort-001 \
  --model-revision fixed-model --rubric-id fixed-rubric \
  --out public-grade-checkpoint.json
```

The grader checkpoint signs the exact ordered grade-log SHA-256, count, frozen scope and public-key fingerprint. Obtain its **current signed-checkpoint SHA-256** from an independently controlled append-only or versioned store, and obtain the ingress checkpoint fingerprint from its own store. The verifier invokes:

```bash
# Read-only public audit; requires no ARU_* signing keys or ingress access.
aru-grade-public verify --public-key grader-public.json \
  --expected-public-key-sha256 PINNED_PUBLIC_KEY_SHA256 \
  --expected-current-grade-sha256 CURRENT_SIGNED_CHECKPOINT_SHA256 \
  --attestations-jsonl grades.jsonl \
  --grade-checkpoint public-grade-checkpoint.json
```

That command authenticates the exact closed grader log but cannot determine whether those request IDs were actually issued. The release verifier additionally reconciles the grader log against the independently anchored ingress log and signs a watch:

```bash
aru-grade-public reconcile --public-key grader-public.json \
  --expected-public-key-sha256 PINNED_PUBLIC_KEY_SHA256 \
  --expected-current-grade-sha256 CURRENT_SIGNED_CHECKPOINT_SHA256 \
  --expected-current-ingress-hmac CURRENT_INGRESS_HMAC \
  --attestations-jsonl grades.jsonl --issued-jsonl issued.jsonl \
  --grade-checkpoint public-grade-checkpoint.json \
  --ingress-checkpoint ingress-checkpoint.json --out signed-watch.json
```

The reconcile command reads `ARU_ISSUED_SIGNING_KEY` and `ARU_WATCH_SIGNING_KEY` from the verifier's environment. It does **not** read `grader-private.json` or an `ARU_GRADER_SIGNING_KEY`. Exit `0` writes a complete alarm-free watch, `10` writes a complete watch with an alarm, `3` means authenticated grade coverage is missing or foreign and writes no watch, and `4` reports invalid, stale or malformed evidence and writes no watch. Read-only verify exits `0` for an exact valid grader log or `4` for invalid evidence. A completed signed watch is still research evidence; no alarm does not authorize rollout.

For a release decision, call `gate_with_public_grades(...)` with the public key, independently pinned key fingerprint, independently current checkpoint digest, full grader log and the receipt/ingress evidence. It recomputes the watch from the authenticated grades before calling the existing receipt, calibrated risk and source-proof gates. A different signing key can create internally valid grades, but cannot replace the externally pinned public key. A stale signed checkpoint cannot match the externally supplied current digest.

## Reproduce the challenge

```bash
python -m unittest -v tests.test_public_grade_ledger
python -m unittest discover -s tests -q
```

The tests show that a monitor can re-sign an alarm-free watch after changing one independently signed error to a success, and the receipt-only gate accepts it. The public-grader gate rejects the same watch. They also test public-key substitution with a fully signed attacker log, checkpoint replay, changed labels and scope, duplicate/missing/foreign grades, out-of-order grade arrival, strict Boolean validation, and read-only public verification **after deleting the private key file and removing all ARU signing keys from the environment**. No private key is needed at the grader-log verification boundary.

Key registration, rotation and revocation need an external governance process: pin a key to the intended grader and cohort before accepting grades, and replace the pin only through an authorized change. A compromised grader private key, a dishonest grader, a colluding ingress/receipt authority, a forged external current-anchor service or selectively missing requests can defeat claims beyond this software boundary. The archived [120-call public real-model run](PROSPECTIVE_LEDGER_RESULTS.md) was graded on a single CI runner before these signatures existed and cannot be retroactively treated as independent evidence. There is no external timestamp, real customer pilot or validation of the watch's statistical conditional-null assumption here.

For a host that requires two separately pinned attestations of the current grader key, checkpoint and ordered outcome digest, use the opt-in [witness quorum and verifier-owned high-water state](GRADER_WITNESS_QUORUM.md). It rejects a fully re-signed, self-pinned grader replacement and local rollback after a newer witnessed epoch.

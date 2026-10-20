# Preinference receipt claims on the pinned public QA cohort

**Corrected run:** [PR #7 Python regression workflow 39364196045](https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39364196045), job “Frozen issuance before real-model calls · complete outcome audit,” on precommitted receipt/code commit `dcd9c857db967fa23369230e334cc12e85483cdb`, 2026-10-20. The [frozen receipt extension](RECEIPT_CHALLENGE_PROTOCOL.md) SHA-256 is `20e67ca8514d06201a49d469cc3fae434426b42a81dc5108d1521f9936dcdab4`. The original cohort protocol still hashes to `dde659f6dfb28a46c7bb6fe5c4b9c91d9a5fec5333d21f9ef4bba8f744bffd69`; its [earlier results and erratum](PROSPECTIVE_LEDGER_RESULTS.md) remain separate.

The CI job completed `prepare` **before** real-model `run`, then `verify`. All three steps passed. Artifact `12238407226` was downloaded and matched its published ZIP SHA-256 `984d56570ea8ec22083a25769776211ec4df4bf2633d4f43ea78ef129834c2c5` (expires 2026-11-03). The [exact 12-file evidence bundle](evidence/receipt_challenge_20261020/) is preserved on this branch beyond artifact expiry, without raw questions, passages, answers, weights or signing keys.

| Measure | Observed |
|---|---:|
| Frozen issued requests | **120** |
| Receipt claims generated before inference | **120 / 120** |
| Receipt challenges included in signed manifest | **120 / 120** |
| Actual calls represented and graded | **120 / 120** |
| Human-gold F1 rubric errors | **15 / 120** |
| Execution failures | **0** |
| Verified source-span certificates | **120 / 120** |
| Alarm at predeclared 50% research target | **None** |
| Independent customer delivery or operator | **No** |

A separate local recomputation matched the archived issue-manifest SHA-256 against the checkpoint, plan, watch and accounting report. The ordered receipt digest `41eccbb6f79c24ef6db34760feb28aeafb30fe9db01171ec04c89e93657fd105` matched plan and report. All 120 receipt rows had distinct nonces, contiguous sequences and issue fields equal to the corresponding manifest row; the 120 grades had 15 errors. The manifest, observed outcomes and grades were **byte-identical** to the [prior run](PROSPECTIVE_LEDGER_RESULTS.md), which used the same public questions and model. Unit tests independently falsified each of 120 single-position signed-manifest rewrites and fail closed on stale or malformed evidence.

**Credential correction:** the first receipt CI execution on commit `365c910b4bc76786c4eb6109cd43dca4306fc513` logged its ephemeral keys in runner environment output. The earlier no-receipt run did too. Their signatures have no secrecy-based provenance value. The corrected workflow masks each newly generated key before subsequent steps; an inspection of the corrected job log found all three key names masked in every step environment occurrence. Keys are absent from the artifact. This does not create an independent signer, a customer-held receipt or a durable external timestamp. Because the corrected keys were not retained for public verification, the preserved bundle supports exact public-hash recomputation and the CI record of verification, **not** later independent HMAC authentication.

These were **CI-prepared claims**, not requests made by customers. One runner controlled issuance, receipt creation, inference and grading. The receipt nonce is not in the issued manifest; atomic uniqueness of customer-facing requests is still unproven. Public SQuAD questions can share articles, the grader is not externally audited, and an alarm-free stream at a deliberately high research target does not establish safety or customer performance. **Automatic rollout remains disabled.** The next meaningful milestone is actual independently delivered receipts, externally retained current checkpoints and independent grade review on representative customer tasks.

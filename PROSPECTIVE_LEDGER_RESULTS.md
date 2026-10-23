# Fresh model calls accounted against a frozen preinference cohort

**Run:** [PR #7 Python regression workflow 39337122344](https://github.com/aruintelligence/regret-aware-agent-scaffold/actions/runs/39337122344), job “Frozen issuance before real-model calls · complete outcome audit,” on **precommitted** protocol/code commit `50d2bf724a8d03e6e4955bf2275a3bb6c0c8f6a4`, 2026-10-19. The protocol SHA-256 was `dde659f6dfb28a46c7bb6fe5c4b9c91d9a5fec5333d21f9ef4bba8f744bffd69`. The job executed `prepare` **before** `run`, then `verify`; all three stages passed. The CI artifact `12227820162` has ZIP SHA-256 `d203f69642b56f97b6e26e21e5e1b9c876de24aa7bcb7a825ce6b1eda758468a` and expires after 14 days.

The [frozen protocol](PROSPECTIVE_LEDGER_PROTOCOL.md) reserved public SQuAD development questions at SHA-sorted offsets **680–799**, a pinned QA model and revision, a human-gold F1 rubric, trusted source-length groups, and the fixed 50% research watch target **before** the new model calls. The committed [exact CI evidence bundle](evidence/prospective_ledger_20261019/) preserves the manifest, signed ingress checkpoint, planned scope, actual ordered binary outcomes, grades, signed watch, reports and protocol hash after the CI artifact expires. There are no raw questions, passages, answers, secret keys or model weights in this bundle.

## Observed result

| Measure | Observation |
|---|---:|
| Preissued requests before inference | **120** |
| Actual model calls represented and graded | **120 / 120** |
| Execution failures counted as errors | **0** |
| Verified source-span certificates | **120 / 120** |
| Human-gold rubric errors, aggregate | **15 / 120 (12.5%)** |
| Short-source errors | **6 / 33** |
| Long-source errors | **9 / 87** |
| Predeclared 50% research watch alarm | **None observed** |

The ingress manifest fingerprint is `405a239cc6a0d875e6fad9282ea47628c800e3adef0d94f9c1ea41cecd97b6cd`; ordered grade fingerprint `680039baece3015ffa011a8380f691e72fc5ac90c97aefbc36e1a1af5ea023ae`; and ordered watch-outcome fingerprint `1209d48101c97545b10477b9cfb1cc3977661e3bbd668bed3b828cea46dfbd25`. The model revision was `564e9b582944a57a3e586bbb98fd6f0a4118db7f`, and the official dataset SHA-256 matched `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.

CI verified the exact cohort and signatures while the ephemeral keys existed. Separately, a local recomputation from the downloaded ZIP matched the manifest SHA-256, ordered grades SHA-256, all 120 one-to-one group and error joins, every signed-watch slice and the ordered outcome SHA-256. Unit tests also block inference on an altered manifest, stale checkpoint, changed dataset or validly re-signed looser risk target; they reject missing or rewritten grades after inference. The entire local suite had **223 tests passing** before the real calls.

## What the result does and does not prove

This is prospective **within the CI workflow**: the manifest was recorded in a distinct completed step before the pinned model was initialized and the calls began. It is stronger accounting evidence than the prior manifest reconstructed from an old outcome trace. A single CI runner still controlled both steps and used a repository-authored grader; there was **no independently operated customer ingress or independently authenticated grading**. The keys were excluded from the ZIP, but a later log review found that the runner printed them in step environment output. **Treat these historical HMAC keys as exposed and the signatures as unsuitable for a secrecy-based provenance claim.** The preserved hashes, ordered rows and stage sequence remain auditable; the GitHub job record, rather than an external timestamp authority, establishes stage order. The frozen protocol's statement that the keys could not be recovered is superseded by this erratum without altering its recorded protocol hash.

Public SQuAD development questions can share articles and are not a hidden customer holdout. Source-span verification proves the text was in the provided context, not that the answer was correct. The grader's F1 implementation is not externally audited. A 50% research target and one stream with no alarm cannot demonstrate product safety, calibrated customer error risk, real-world false-alarm frequency or a sales guarantee. The watch's conditional-null assumptions remain unverified. **No automatic rollout is authorized.** The next evidence milestone is a separately controlled ingress and grader with durable externally retained checkpoints, representative customer tasks and independent review.

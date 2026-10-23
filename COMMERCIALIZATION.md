# ĀRU Decision Evidence Lab — Commercialization Blueprint

**Status:** experimental software research; not a production service, independently validated AI, or security-certified product.

## Verified research progress (October 2026)

A reproducible, human-annotated **sentence-retrieval** experiment on 2,000 Stanford SQuAD v1.1 questions measured **78.5%** baseline versus **82.1%** challenger answer-containing-sentence selection (**+3.6 percentage points**). The article-cluster exploratory interval was **+2.47 to +4.61 percentage points**. One article regressed, so the human-review gate blocked advancement. Full methodology, dataset SHA-256 and CI provenance are in [BENCHMARK_EVIDENCE.md](BENCHMARK_EVIDENCE.md).

The system also includes an opt-in local/OpenAI-compatible LLM benchmark adapter tested against a mock server, and a separate pinned open-weight model pipeline exercised on public QA questions. The [receipt challenge](RECEIPT_CHALLENGE_RESULTS.md) records 120 real-model calls and 15 rubric errors with no independent customer operator. A loopback-only consent-gated SQLite/HMAC ingestion gateway exports a dashboard evidence passport. A Python alpha distribution exposes command-line tools, and a responsive commercial research page is included under `docs/` but **has not been publicly deployed**. The human-gold benchmark additionally stress-tests adversarial changes to the observed topic mixture; these are empirical sensitivity checks, not guarantees about unseen customers. These are tangible integration steps, **not evidence of enterprise production readiness**.

## The product in one sentence

A vendor-neutral evidence layer that helps teams **measure and audit AI agent routing decisions before changing a production policy**.

The product is not "an AI that remembers everything." The concrete customer value is knowing **which agent workflow decisions worked, how certain the comparison is, what could go wrong in particular task domains, and whether the evidence is sufficient for a human rollout decision**.

## Additional enterprise differentiators (research-only)

The latest engineering iteration adds three commercially relevant but **not yet enterprise-validated** capabilities:

1. **Independently anchored audit history.** A separately keyed checkpoint can detect a deleted or rewritten SQLite audit-log prefix, addressing a specific weakness of an ordinary HMAC chain. The checkpoint must be retained in separately controlled storage; neither a local HMAC nor a timestamp alone proves real-world label truth.
2. **Pilot evidence design before spending.** A paired-outcome sample-size and grading/compute budget planner makes assumptions explicit, including expected uplift, discordance, clustering, missingness and per-task costs. Its normal approximation is not a substitute for design-specific statistical review.
3. **Prospective anytime monitoring.** A fixed-mixture e-process has a self-contained nonnegative-supermartingale derivation for optional stopping under a stated conditional null. This is an established statistical method, not a new discovery. It cannot retroactively validate the SQuAD development benchmark or selective off-policy outcomes.

A customer-facing proposition should emphasize **independently reproducible evidence, defensible experimental design, and human-controlled rollout**, not unverified claims of general AI intelligence or immunity to tampering.

## New product wedge: proof-carrying source answers (experimental)

A promising customer-facing feature is **source provenance with explicit abstention**. A short answer is displayed only when its exact text can be recovered from an identified passage, accompanied by a source digest and independently checkable character offsets. This is a verifiable **textual provenance** guarantee, not a claim that the answer is true or entailed.

A genuine open-weight 135M-parameter model has now completed a public human-gold 40-question randomized integration test (40/40 calls) and a disjoint matched 40-question follow-up (80/80 calls). **The restrictive prompt did not improve matched F1**: direct 23.53% vs grounded 21.50%; paired difference −2.03 percentage points, exploratory article-cluster interval −5.75 to +1.75. This negative evidence reinforces the need to sell **auditability and transparent controls**, not an unproven quality uplift.

The new source-grounding gate is being evaluated on a further disjoint public question subset. Its success criteria are correct certificate verification and honest reporting of coverage, rejection, and F1 tradeoffs, not higher model intelligence. Before selling it for enterprise knowledge management, add trusted-source controls, citation relevance and entailment checks, privacy safeguards, prompt-injection defenses, and independent customer tests.

## ĀRU consent-scoped source memory (new research component)

The [Sovereign Source Registry](SOVEREIGN_SOURCE_REGISTRY.md) extends source provenance with a narrow **permission-at-use** check: an exact source is authorized for a specific opaque owner and purpose until its grant expires or is revoked. A keyed ledger stores no raw source text. An optional independently keyed external checkpoint can detect rollback of previously anchored revocations, and the local QA server can deny inference when that checkpoint fails.

This is a stronger *research control* than a generic consent banner, but it is **not** verified legal consent, real user authentication, enterprise access control, encrypted data storage, or complete protection against deletion of unanchored revocations. Production commercialization would require customer-controlled identity, consent evidence, independently retained audit anchors, encryption, privacy and deletion policy, and a security assessment.

**Potential buyer use case:** internal knowledge assistants whose source documents have explicit owner, purpose, expiry and revocation rules. The measurable pilot outcome should be the rate of unauthorized-source requests blocked before inference, the latency overhead, the false-denial rate on valid grants, and independently reviewed privacy controls—not claims of human-like memory or perfect data sovereignty.

## Target buyer and wedge

Initial buyer hypothesis: head of AI platform, engineering director for agentic workflows, or an enterprise AI assurance lead at an organization deploying multiple AI agents.

Initial use cases:
- Compare a new agent routing policy with an incumbent using logged decisions and independent task outcomes.
- Detect domains where an apparent global improvement masks a decline.
- Produce a portable evidence package for a technical review committee.
- Track how much of an improvement claim is supported by observed overlap and effective sample size.

Avoid first deploying into autonomous medical, financial eligibility, policing, or other consequential decision systems. Those need additional domain-specific safeguards, qualified review, and legal/regulatory assessment.

## Current prototype — what exists

| Capability | Current status |
| --- | --- |
| Scripted route simulator with memory | Existing scaffold |
| Opt-in single-route randomized logging | Implemented |
| Vendor-neutral categorical decision adapter | Implemented; caller's probabilities are not independently verified |
| Separate outcome labels with grader/rubric provenance | Implemented; self-attested |
| IPS / SNIPS / overlap / effective sample size | Implemented |
| Paired bootstrap uncertainty | Implemented; assumes independent tasks |
| Cross-fitted doubly robust diagnostics | Implemented; simple, unvalidated reward model |
| Task-signature negative-segment screen | Implemented; exploratory |
| Portable, accessible offline HTML report | Implemented |
| Optional SHA-256 / HMAC evidence attestation | Implemented; not immutable storage |
| Analytic-ground-truth Monte Carlo falsification | Implemented on synthetic data |
| Python 3.9 / 3.12 continuous tests | Implemented |
| Real human-gold sentence-retrieval benchmark | Implemented and executed on 2,000 public SQuAD questions; **not an LLM** |
| Optional OpenAI-compatible LLM executor | Adapter tested against local mock; separate pinned open-weight public-QA model pipeline executed 120/120 calls in the receipt challenge, without customer operation |
| Independent customer grader / customer outcomes | **Not implemented** |
| Loopback-only SQLite/HMAC research ingestion gateway | Implemented; **not production certified** |
| Production API, multi-tenancy, SSO/RBAC, encryption, retention controls | **Not implemented** |
| Prospective external performance study | **Not performed** |
| Security certification, compliance, enterprise SLA | **Not established** |

## Customer demo story (do not overclaim)

1. A team has a baseline agent policy and a proposed routing policy.
2. Its own controller randomly explores allowed routes and logs exact selection probabilities.
3. Independent graders assign outcomes using a predeclared rubric.
4. The evidence engine estimates policy values, shows uncertainty and coverage, and flags weak support or domain-level declines.
5. The system produces a portable report and an optional authenticated checksum manifest.
6. A human reviewer decides whether to run a prospective experiment. **No automatic rollout occurs.**

Use synthetic examples for the initial product demonstration, labeled prominently. Do not describe those examples as real-world gains.

## Differentiation to validate, not assert

- **Evidence-first architecture:** external labels remain separate from internally generated agent quality scores.
- **Decision provenance:** explicit logging propensities and candidate sets instead of untraceable "AI improved itself" narratives.
- **Safety visibility:** a global win can still be blocked by a negative domain-level signal.
- **Portable audit:** dependency-free local reports and optional HMAC authentication.
- **Vendor-neutral instrumentation:** the same event schema can wrap external orchestrators.

These are product hypotheses, not proof of superiority to existing commercial AI observability or evaluation tools. Conduct a competitor comparison before sales claims.

## Pilot offer — pricing hypotheses, not verified market rates

**Decision Assurance Design Partner Pilot**
- 6–8 weeks; proposed fixed pilot fee: **$15,000–$35,000**, to be tested in discovery.
- Scope: one workflow, one baseline and one challenger, instrumentation review, independent grading rubric, policy evidence report, risk register, and rollout recommendation.
- Customer supplies task data, safe experimentation approval, and a qualified evaluator.
- Acceptance criteria: valid propensity logs, complete grading, reproducible report, clear cost-per-success and failure metrics, and a documented go/no-go decision.

**Possible annual platform offering (future):** a hosted subscription with SSO, tenant isolation, signed evidence bundles, audit exports, API integrations, alerts, and SLA. Pricing must be derived from paid pilots, support burden, security costs, and demonstrated ROI; no annual price is established.

## Evidence and revenue milestones

### Days 1–30 — engineering validation
- Complete schema review and threat model.
- Integrate one external agent with real randomized logging.
- Pre-register two candidate policies and a blinded grading rubric.
- Measure latency overhead, dropped events, logging consistency, and missing labels.
- Set up reproducible tests and independent code review.

### Days 31–60 — prospective pilot
- Recruit one design partner with non-consequential tasks.
- Collect sufficient independently graded decisions to support the preregistered precision target; do not promise a fixed sample size without a power analysis.
- Compare policy value, task completion, failure rate, cost per successful task, and domain-level regressions.
- Reassess bootstrap assumptions; use cluster-aware or sequential methods when appropriate.
- Audit data provenance and potential evaluator bias.

### Days 61–90 — commercial decision
- Deliver a signed pilot evidence pack and customer-facing dashboard.
- Obtain explicit customer feedback and evidence of willingness to pay.
- Decide whether to invest in hardened ingestion, authenticated tenancy, RBAC/SSO, retention controls, and operational monitoring.
- Do not claim an enterprise-ready product until security, privacy, reliability, and support criteria are met.

## Decision gates

A credible launch requires:
1. **Measurement validity:** logged probabilities reflect the actual randomized controller.
2. **Outcome validity:** an independent rubric and graders are documented; missingness and leakage are audited.
3. **Statistical validity:** a preregistered comparison, appropriate dependence-aware inference, and sufficient overlap.
4. **Operational validity:** stable ingestion, bounded overhead, rollback, incident response, and support.
5. **Security/privacy validity:** tenant isolation, authentication, encryption, data minimization, deletion and consent mechanisms.
6. **Commercial validity:** customer willingness to pay for a demonstrable decision or risk reduction.

Fail any gate: keep the system in research or pilot mode.

## Intellectual property and ownership

The repository currently uses the **MIT License**, which permits reuse and commercial sale by others. A future commercial moat would need to come from deployment quality, hosted infrastructure, trusted integrations, brand, customer relationships, proprietary service features, or properly protected IP—not exclusive control over publicly MIT-licensed code. The repository README attributes the original work to two creators. Confirm contributor rights and ownership before assigning IP, negotiating licenses, or claiming exclusivity. Trademark availability, patentability, and protection status have **not** been established.

## Enterprise engineering roadmap

A credible production architecture could separate:
- **Ingestion service:** authenticated, idempotent event acceptance; exact propensity and candidate-set validation.
- **Evidence store:** encrypted tenant-isolated decisions and grader outcomes; transactional writes; retention policy.
- **Evaluation worker:** deterministic versioned estimators, cross-fitting, sensitivity analysis, power and overlap checks.
- **Review service:** human approval, policy versioning, audit logs, rollback and change control.
- **Dashboard:** reproducible reports with provenance, segment warnings, accessible design and signed exports.

The existing Python modules are a research reference implementation of part of this architecture. They should not be presented as a production deployment.

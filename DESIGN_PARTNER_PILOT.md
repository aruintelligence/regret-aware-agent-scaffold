# ĀRU Decision Evidence Lab — Design Partner Pilot Statement of Work (Draft)

**Commercial status:** nonbinding proposal template. The software is a research alpha, not a security-certified hosted service. All terms, intellectual-property ownership, privacy obligations, pricing, insurance and warranties require counsel and customer negotiation before signature.

## The customer problem

Organizations experimenting with AI agents often cannot establish whether a new prompt, tool route, retrieval policy, or agent version genuinely improves task outcomes. Offline demonstrations can mask weak evidence, unsupported claims, domain regressions, and uncontrolled deployment risk.

ĀRU proposes a bounded, **human-governed decision assurance pilot** that helps a customer measure one AI workflow using independently graded outcomes, a preregistered baseline, auditable routing logs and reproducible reports.

## Pilot offer (commercial hypothesis, not a market-validated price)

| Item | Proposed scope |
| --- | --- |
| Duration | 6 weeks from kickoff |
| Fixed professional-services fee | **$20,000** proposed; subject to discovery and contract |
| Workflow | One non-consequential AI-assisted business workflow |
| Comparisons | One baseline and one candidate prompt/route policy |
| Data | Customer-approved, minimized, deidentified or public tasks |
| Deployment | Isolated research environment; no autonomous production policy changes |
| Deliverables | Protocol, logging adapter, independent grading rubric, baseline/challenger evidence, risk register, reproducible dashboard, human go/no-go review |
| Customer obligations | Provide approved tasks, a responsible owner, independent evaluator, experiment permissions, and timely feedback |
| Acceptance | Delivery of auditable evidence and review package; **not** a promise that the candidate improves |

**Price is an initial positioning hypothesis.** It is not evidence of market demand, established margin, or a guaranteed quote.

## Six-week delivery plan

### Week 1 — Define truth before experiments
- Identify a narrow workflow with measurable binary or bounded outcomes.
- Freeze baseline/challenger versions, grading rubric, safety exclusions, data-handling rules and decision criteria.
- Review consent, privacy, evaluation conflicts and known business risks.
- Use `aru-power` to estimate sample requirements from explicitly agreed assumptions; engage a qualified statistician where needed.

### Week 2 — Instrument without changing production behavior
- Add prospective action/propensity logging to an approved sandbox controller.
- Keep raw prompts and sensitive data outside ĀRU's research event schema.
- Establish an independent grading pipeline and missing-outcome tracking.
- Verify event schema, idempotency, and replay safety on test data.

### Week 3 — Dry run and falsification
- Run a bounded synthetic dry run and manual audit.
- Confirm real propensities reflect the actual controller; refuse retrospective invented probabilities.
- Audit whether labels can be traced to approved graders and whether both policies have support.
- Halt if task safety, consent, logging or grading integrity cannot be established.

### Week 4 — Prospective controlled evaluation
- Run the preregistered comparison on approved tasks.
- Measure task success, failures, latency, cost per successful task and label completeness.
- Flag adverse domain-level outcomes and incomplete overlap.
- Use dependence-aware or sequential inference **only** when the design assumptions hold.

### Week 5 — Reproduce and challenge
- Independently reproduce the evaluation from exported event/label files.
- Produce a source/provenance report with HMAC evidence and optional separately retained checkpoint.
- Stress-test reasonable changes in task distribution and evaluator disagreement.
- Document what the evidence does not establish.

### Week 6 — Human review and decision
- Deliver an executive report, technical appendix, signed-off risk register and evidence dashboard.
- Facilitate a go/no-go/extend decision meeting; never auto-deploy the candidate.
- Deliver prioritized engineering backlog and estimates for production hardening.
- Collect feedback on willingness to pay for ongoing evidence assurance.

## Objective acceptance criteria

The pilot is complete when:
1. Both parties have approved a written protocol, baseline, candidate and rubric.
2. Decision logs include real action support and prospective propensities or are explicitly classified as noncausal.
3. Outcome provenance and missingness are disclosed.
4. A third party can rerun the supplied analysis on authorized data.
5. Reports show point estimates, uncertainty, sample/overlap limitations and adverse segments.
6. A responsible customer decision-maker receives a human-review recommendation.

**Improvement in the customer's underlying AI performance is not guaranteed.** Negative or inconclusive results are valid, useful pilot outcomes.

## Scope exclusions and stop conditions

- No medical diagnosis, lending eligibility, law enforcement, child safety decisions or other high-stakes autonomous decisioning.
- No public hosting of the current local research gateway or model server.
- No transfer of secrets, raw private customer data, or personally identifiable prompts into public CI or GitHub artifacts.
- No claims of immutable storage, SOC 2 certification, third-party audit or enterprise encryption where none exists.
- No automated production deployment, continuous surveillance, unapproved experimentation, or outcome manipulation.
- Pause or terminate research if consent, safety, access controls, provenance, or data minimization fails.

## Security and privacy handoff

The current prototype uses local-only research services and self-attested consent headers. Before any customer deployment, specify verified consent, access control, encryption at rest and in transit, key management, retention/deletion, backup, tenant isolation, audit logging, incident response, vendor/subprocessor terms, and data-processing agreements as applicable. Customer data stays in a customer-controlled environment unless a separately reviewed agreement permits otherwise.

## Intellectual property and licensing

The public repository uses the MIT License, and contributor rights must be confirmed. The pilot proposal does not assign exclusive ownership of already open-source code. New customer-specific deliverables, confidential data, feedback, and any proprietary extensions require an explicit negotiated IP schedule. ĀRU brand and trademark status must not be represented as legally registered unless independently confirmed.

## Indicative pilot economics (assumptions, not financial projections)

A proposed $20,000 fixed-fee pilot could be viable only if staff hours, independent grading, computing, travel, security work, insurance, and support fit within an agreed budget. For illustration, 80 engineering/analysis hours at a fully loaded $125/hour cost = $10,000; $2,000 in approved compute/grading; $1,000 administrative and contingency = $13,000 modeled delivery cost and **$7,000 illustrative contribution before overhead, taxes, customer acquisition, and unmodeled expenses**. Actual costs and margins are unverified.

## Suggested executive positioning

**ĀRU helps teams replace AI performance anecdotes with independently graded, reproducible decision evidence.** We instrument one workflow, compare a frozen baseline against a candidate, surface regressions and uncertainty, and deliver a human-governed decision package. The result may be improvement, no improvement, or evidence that the experiment needs redesign; in all cases the organization gains a clearer basis for its next decision.

## Approval checklist

- [ ] Customer problem and pilot owner identified.
- [ ] Scope, grading, sample size, pricing and acceptance negotiated.
- [ ] Legal counsel reviews IP, privacy, warranties and liability.
- [ ] Security lead approves the sandbox and data classification.
- [ ] Independent evaluator and statistical reviewer assigned.
- [ ] Both parties sign a final agreement before work or payment.

*Draft for discussion; not a binding contract or legal advice.*

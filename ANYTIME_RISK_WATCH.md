# ĀRU Anytime Risk Watch — Detecting Degradation After Calibration

**Status:** opt-in research alpha. No scheduled monitoring, customer deployment, automatic rollout, or external alerting is enabled. The mathematics uses established nonnegative-supermartingale methods.

## Problem addressed

An offline risk certificate can become outdated after a model, task population, source distribution, or grading process changes. A static calibrated threshold cannot by itself detect subsequent deterioration.

**ĀRU Anytime Risk Watch** processes prospectively collected, independently graded binary error outcomes and raises a group-specific alarm when sequential evidence against a predeclared conditional error-rate target becomes sufficiently strong.

## Mathematical derivation

Let `E_t ∈ {0,1}` denote whether the model's answer was independently graded incorrect at time `t`. Fix target error rate `r ∈ (0,1)`. The conditional null is:

```text
H0: E[E_t | F_(t-1)] <= r for every t.
```

For any **fixed, predeclared** bet `lambda ∈ [0,1/r]`:

```text
M_0(lambda) = 1
M_t(lambda) = M_(t-1)(lambda) * (1 + lambda * (E_t-r)).
```

Because `E_t >= 0`, each factor is nonnegative; and under H0,

```text
E[M_t | F_(t-1)] <= M_(t-1).
```

Thus the process is a nonnegative supermartingale. A predeclared convex mixture over a fixed bet grid is also a supermartingale. By Ville's inequality, its probability of ever crossing `1/alpha_slice` under the conditional null is at most `alpha_slice`.

For `J` predeclared groups, monitor each group and the aggregate with `alpha_slice = alpha / (J+1)`. The union bound controls the chance of **any false degradation alarm** under the joint conditional null by `alpha`.

**Important:** a lack of an alarm is *not* proof that error is below target. It merely means this particular monitor has not accumulated enough evidence to reject the conditional null.

## Security controls

- Signed monitoring report with an independently managed HMAC-SHA256 key.
- Model revision, grading rubric, predeclared groups, target, bet grid, event count, and SHA-256 digest of the ordered prospective outcomes are bound into the signature.
- `gate_with_risk_watch` refuses missing or invalid signatures, model/rubric mismatches, unmonitored groups, a degradation alarm, or insufficient group observations.
- **Replay protection:** the caller must supply `expected_outcomes_sha256` from a separately controlled, current and complete monitoring ledger. An older signed report with a different digest is rejected.
- The existing signed static selective-risk calibration and independently verified source-span certificate must still pass. **No alarm is not a deployment approval.**
- Watch and calibration must declare the same error target and ordered group scope; a looser signed watch cannot stand in for a stricter signed calibration.

The caller must obtain the current digest from a trusted source; supplying the digest copied from the report defeats replay protection. A compromised signing key, incomplete labels, selective reporting, or a malicious monitoring pipeline can still undermine the system.

## Operator example

Provide a prospective JSONL stream with **exactly** `event_id`, `group`, and `error` (strict boolean), ordered by the trusted observation sequence:

```json
{"event_id":"observed-0001","group":"general","error":false}
{"event_id":"observed-0002","group":"technical","error":true}
{"event_id":"observed-0003","group":"general","error":false}
```

```bash
export ARU_WATCH_SIGNING_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

aru-watch observe --outcomes-jsonl prospective-outcomes.jsonl \
  --groups general,technical \
  --model-revision immutable-model-sha \
  --rubric-id independent-grader-v1 \
  --target-error 0.10 --alpha 0.05 \
  --out signed-watch.json

# Returns exit code 2 if a degradation alarm was raised.
aru-watch verify --report signed-watch.json
```

This CLI is an operator-triggered analysis. It **does not** watch files or events automatically, connect to production systems, or deliver alerts.

## Falsification and limitations

Unit tests cover all-correct streams, sustained failures, subgroup-only deterioration, the exact single-bet wealth product, seeded conditional-null simulations, invalid labels, signatures, and stale-report replay.

The sequential interpretation requires a prospective, predeclared inclusion rule; complete independently graded outcomes; trusted group assignment; and the conditional null. It is invalidated by missing difficult outcomes, selective feedback, retrospective cherry-picking, inconsistent group assignment, or assumptions broken by clustered data. When feedback is delayed, record and handle that delay explicitly rather than silently excluding unresolved outcomes.

**Research goal:** combine static risk calibration, real-time degradation evidence, and source verification into a defensible human-reviewed AI assurance workflow. Production readiness, commercial value, safety, and customer performance remain unproven.

The [preregistered public real-model audit](REAL_MODEL_WATCH_PROTOCOL.md) connects this monitor to actual QA calls scored against published human answers. Its injected post-onset errors and simulated null streams are explicitly synthetic. The unsigned audit is distinct from the signed, independently anchored evidence required for the release gate.

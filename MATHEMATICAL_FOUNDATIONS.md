# ĀRU Decision Evidence Lab — Mathematical Foundations and Falsification

**Status:** reproducible engineering notes, not a new mathematical theorem or a claim of AGI. The techniques are established statistical methods implemented and combined for a concrete AI-decision-evaluation workflow.

## 1. Counterfactual policy value from logged randomized decisions

Let `X_i` be a task context, `A_i` the logged action, `R_i in [0,1]` the independently graded outcome, and `p_i = P(A_i | X_i, history)` the *actual* logging propensity. Let `pi(X_i)` be a deterministic candidate policy chosen independently of the outcome data.

For each observation, the inverse-propensity weighted contribution is

```text
W_i = 1{A_i = pi(X_i)} / p_i
IPS(pi) = (1/n) sum_i W_i R_i
SNIPS(pi) = (sum_i W_i R_i) / (sum_i W_i), if denominator > 0
ESS(pi) = (sum_i W_i)^2 / sum_i W_i^2
```

**Derivation of IPS under valid logging:** conditional on the context and all pre-action history,

```text
E[1{A_i=pi(X_i)} R_i / p_i | X_i, history]
  = sum_a P(A_i=a | X_i,history)
          * 1{a=pi(X_i)}
          * E[R_i(a) | X_i,history] / p_i
  = E[R_i(pi(X_i)) | X_i,history].
```

This requires truthful prospective propensities, consistency of observed and potential outcomes, full support for the target action, and valid conditional randomization. Clipping weights reduces variance but introduces bias. SNIPS is generally biased in finite samples. A small ESS means that the apparent evidence is supported by few effectively weighted observations. No estimator can identify unsupported actions.

**Falsification:** `scientific_validation.py` generates a disclosed toy reward model with analytic policy values, then measures bias, RMSE and empirical bootstrap coverage over seeded Monte Carlo trials. This verifies behavior on that simulator, not real agent performance.

## 2. Doubly robust diagnostic

With an out-of-fold reward predictor `m(X,a)`, the implementation evaluates

```text
DR(pi) = (1/n) sum_i [
  m(X_i,pi(X_i)) +
  1{A_i=pi(X_i)} / p_i * (R_i - m(X_i,A_i))
]
```

With correct propensities and sufficient support, the residual term corrects reward-model errors in expectation. Cross-fitting avoids using a sample's own outcome to train its nuisance prediction. This does not make an unvalidated reward model accurate or repair incorrect propensities. Clipping introduces bias. The implementation's shrinkage model is a diagnostic rather than an independent replication.

## 3. Exact paired discordance (McNemar)

When both algorithms are evaluated on the same binary-outcome questions, define `b` as baseline-only correct and `c` as candidate-only correct. Condition on `m=b+c` discordant questions. Under equal marginal success probability and independent paired observations,

```text
C ~ Binomial(m, 1/2)
p_two_sided = min(1, 2 * sum_{k=0}^{min(b,c)} choose(m,k) / 2^m).
```

The implementation computes the exact integer binomial tail. **SQuAD questions share paragraphs and articles, violating the independent-question assumption.** Therefore this p-value is descriptive. Separate paired paragraph- and article-cluster bootstrap intervals provide a dependence sensitivity check, but those intervals are exploratory and not a complete multiple-comparison correction.

## 4. Adversarial topic-mixture fragility

For observed domains `j`, let `w_j` be their empirical frequency and `d_j` the observed candidate-minus-baseline accuracy difference. Under a total-variation shift budget `epsilon`, solve

```text
min_q  sum_j q_j d_j
subject to: q_j >= 0, sum_j q_j = 1,
            0.5 * sum_j |q_j - w_j| <= epsilon.
```

This is a finite linear optimization. To minimize a linear objective under a simplex total-variation budget, transfer mass from higher-`d_j` domains to lower-`d_j` domains until the budget is exhausted or further transfers cannot improve the objective. Each unit of transferred mass changes the objective by `d_low - d_high <= 0`; therefore an exchange argument shows that any transfer involving a smaller gap can be replaced by an equal feasible transfer involving a larger gap, improving or preserving the objective. The greedy transport is optimal for this *observed-domain* problem.

The break-even radius is the smallest shift budget for which the worst-case observed-domain uplift is nonpositive. It is **not** a guarantee under unseen domains or changes to conditional outcomes within domains.

## 5. Anytime-valid prospective evidence (e-process)

For complete, prospectively defined paired binary tasks, let

```text
D_t = candidate_success_t - baseline_success_t in {-1,0,+1}.
```

Consider the conditional null `H0: E[D_t | F_(t-1)] <= 0` for every time `t`. For a **fixed predeclared** `lambda in [0,1]`, define

```text
M_0(lambda) = 1
M_t(lambda) = M_(t-1)(lambda) * (1 + lambda * D_t).
```

**Proof:** `1+lambda*D_t >= 0` since `D_t >= -1`. Moreover,

```text
E[M_t(lambda) | F_(t-1)]
 = M_(t-1)(lambda) * (1 + lambda*E[D_t | F_(t-1)])
 <= M_(t-1)(lambda).
```

Thus `M_t(lambda)` is a nonnegative supermartingale with initial value 1. A fixed convex mixture `M_t = sum_k w_k M_t(lambda_k)` is also a nonnegative supermartingale. Ville's inequality gives

```text
P_H0(there exists t with M_t >= 1/alpha) <= alpha.
```

This supports repeated monitoring *only when the conditional null, prospective inclusion rule, and complete paired observations are valid*. It does **not** make post-hoc public-benchmark results anytime-valid. The implementation uses a fixed bet grid and log-space arithmetic; it never authorizes rollout.

**Falsification:** deterministic tests check ties (`M=1`), losses (wealth falls), all wins (wealth grows and crosses), exact single-bet products, invalid data rejection and long-stream numerical stability. `sequential_validation.py` also runs seeded Monte Carlo trials under a known fair conditional null (baseline-only and candidate-only probability 0.20 each) and a known positive alternative (0.10 and 0.30). It reports empirical ever-crossing frequency and alternative detection, clearly labeled as synthetic evidence. The theorem, not finite Monte Carlo, supplies the mathematical guarantee under its assumptions.

## 6. Prospective pilot power and budget

For a paired binary difference `D in {-1,0,+1}`, assume an expected uplift `delta=E[D]>0` and discordance rate `q=E[D^2]`. Then `Var[D]=q-delta^2`. A simple normal-approximation sample size is

```text
n ≈ [z_(1-alpha/2)*sqrt(q) +
     z_power*sqrt(q-delta^2)]^2 / delta^2.
```

For mean cluster size `m` and assumed intracluster correlation `rho`, the rough design effect is `1+(m-1)*rho`. Missing outcomes inflate the requested count by `1/(1-missing_fraction)`. All assumptions are operator-supplied. This is a planning approximation, not a guarantee of adequate power for a real adaptive agent experiment.

## 7. Evidence integrity and limits

Each local event is chained by `H_i = HMAC(K_gateway, H_(i-1) || event_i)`. The chain detects edits and internal deletions when the key remains protected. **It does not detect deletion of the final event(s)**: a shorter valid prefix is still valid.

The independent checkpoint signs `(event_count, last_sequence, head_hash, timestamp, version)` using `K_anchor != K_gateway`. If the checkpoint is retained outside the gateway's trust boundary, verification detects a truncated prefix or rewritten anchored history. This is a precise, testable improvement over the unanchored chain.

Neither a local HMAC nor a separately retained checkpoint proves the truth of a grader's label, prevents an attacker controlling both keys from forging evidence, or certifies production security. External trusted storage and independent audits are still needed.

## 8. Claim boundaries

- **Measured:** 2,000 SQuAD v1.1 human-annotated questions; baseline 78.5%, challenger 82.1% sentence containing an answer span, with exploratory clustered intervals.
- **Tested:** software/statistical behavior, local gateway, integrity verification, mock LLM HTTP contract, installable CLIs.
- **Not tested:** actual real LLM reasoning performance, private holdout benchmarks, independent customer outcomes, enterprise security, deployment ROI, general intelligence.
- **Not claimed:** original discovery of IPS, DR, McNemar, e-processes, or total-variation robust optimization.

**Research principle:** a falsifiable limited result is more valuable than an extraordinary unsupported claim.

# Time-uniform route evidence

This is a mathematical statement about the opt-in **randomized routing log**, not a proof that the scripted scaffold improves real-world outcomes. The report continues to prohibit automatic rollout.

## Estimand and assumptions

Fix the baseline and candidate policy before inspecting outcome labels. For decision `t`, the candidate routes and logging probabilities `q_t(a)` are determined before the random action `A_t` is drawn. The label `R_t` lies in `[0,1]`. Assume the logging draw is conditionally independent of the potential reward for each available route given everything known before that draw. The rubric must be consistent, every decision must have a label, and actions must have no unmodeled interference that breaks this conditional statement. Logs may contain adaptive route scores, repeated contexts, and history-dependent outcome distributions.

The target is the running mean of the **conditional** candidate-minus-baseline values for these logged decision contexts. It is not a population effect for future tasks. Reward provenance is self-attested, and the repository cannot verify the conditional independence assumption or the fact that the policies were chosen in advance.

## Derivation

For fixed target actions `c_t` and `b_t`, set

```
Z_t = R_t [1{A_t=c_t}/q_t(c_t) - 1{A_t=b_t}/q_t(b_t)].
```

Conditional on the pre-action history, `E[Z_t]` equals the difference between the two policies' conditional expected rewards. If the actions coincide, `Z_t=0`. Otherwise, since `0≤R_t≤1`, `Z_t` lies in `[-1/q_t(b_t), 1/q_t(c_t)]`; its predictable range width is `w_t=1/q_t(b_t)+1/q_t(c_t)`. The code obtains *both* target probabilities from the validated full distribution: either epsilon-greedy or the vendor-neutral categorical probabilities. The selected action's propensity is insufficient to compute the width when a target action was not selected.

For each fixed `λ>0`, conditional Hoeffding's lemma makes

```
M_n(λ) = exp(λ Σ_{t≤n}(E[Z_t|past] - Z_t) - λ² Σ_{t≤n}w_t²/8)
```

a nonnegative supermartingale with initial value one. Ville's inequality bounds the probability that it ever exceeds `1/α_j` by `α_j`. The implementation fixes `λ_j=2^{-j}` for `j=0,…,39` and spends `α_j=α/((j+1)(j+2))`. Since `Σ α_j < α`, a union bound covers all grid choices **and all sample sizes**. Thus, with probability at least `1-α`, simultaneously at every `n`:

```
(1/n) Σ E[Z_t|past] ≥ (1/n) Σ Z_t
    - min_j {log(1/α_j)/λ_j + λ_j Σw_t²/8}/n.
```

The displayed lower bound is also truncated below at `-1`, since a difference of two `[0,1]` conditional means cannot be smaller. It uses **unclipped** inverse propensities. The existing SNIPS, bootstrap, and cross-fitted DR values remain separate exploratory diagnostics; their assumptions do not become true merely because this lower bound exists.

## Interpretation

The review screen now requires this one-sided 95% lower bound to exceed the requested uplift, in addition to its existing checks. Small experiments or low exploration probabilities may produce a bound of `-1`; that is an honest lack of evidence. Incomplete labeling suppresses the bound rather than dropping decisions and treating the rest as representative. Running many candidate comparisons selected after looking at rewards is outside this guarantee; a preregistered policy pair or a separate multiplicity budget is required. Neither an HMAC manifest nor this bound authenticates external graders or proves a causal improvement on future customers.

Reproduce the synthetic pipeline with `python3 experiment_cli.py demo --out-dir demo-report --n 500 --seed 7` and inspect `anytime_ips_difference` in `demo-report/report.json`. Synthetic labels exercise the calculation and can never clear the review screen.

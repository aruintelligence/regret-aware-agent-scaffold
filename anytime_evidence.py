"""Anytime-valid *conditional* evidence for prospectively observed paired outcomes.

For D_t = candidate_success - baseline_success in {-1,0,+1}, assume
H0: E[D_t | past] <= 0 for every t. For fixed predictable lambda in [0,1],
M_t(lambda) = product_{i<=t}(1 + lambda*D_i) is a nonnegative
supermartingale with E[M_0]=1. A mixture over fixed lambdas is also an
e-process. By Ville's inequality:
  P_H0(sup_t M_t >= 1/alpha) <= alpha.

These guarantees REQUIRE the conditional null and valid prospective data.
They do NOT apply automatically to post-hoc public benchmarks, clustered
data, selectively missing outcomes, adaptive task inclusion, or OPE rewards.
This is an established e-process technique, not a novel theorem.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_BETS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)


def _logsumexp(values):
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def paired_e_process(pairs, *, alpha=0.05, bets=DEFAULT_BETS):
    """Compute mixture e-values without premature stopping or cherry-picking.

    The test never chooses bets using current/future outcomes. The fixed
    mixture is declared before observing the stream.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not bets or any(isinstance(b, bool) or not isinstance(b, (float, int))
                       or not math.isfinite(b) or not 0 <= b <= 1 for b in bets):
        raise ValueError("bets must be a nonempty sequence in [0,1]")
    log_wealth = [0.0] * len(bets)
    log_threshold = -math.log(alpha)
    first_crossing = None
    peak_log_e = 0.0
    n = 0
    candidate_only = 0
    baseline_only = 0
    for row in pairs:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each pair must be [baseline_binary, candidate_binary]")
        baseline, candidate = row
        if baseline not in (0, 1) or candidate not in (0, 1) or isinstance(baseline, bool) or isinstance(candidate, bool):
            raise ValueError("paired outcomes must be integer 0 or 1")
        difference = candidate - baseline
        candidate_only += int(difference == 1)
        baseline_only += int(difference == -1)
        for index, bet in enumerate(bets):
            if log_wealth[index] == -math.inf:
                continue
            factor = 1 + bet * difference
            log_wealth[index] = log_wealth[index] + math.log(factor) if factor > 0 else -math.inf
        n += 1
        log_e = _logsumexp(log_wealth) - math.log(len(log_wealth))
        peak_log_e = max(peak_log_e, log_e)
        if first_crossing is None and log_e >= log_threshold:
            first_crossing = n
    if n == 0:
        raise ValueError("no pairs")
    log_e = _logsumexp(log_wealth) - math.log(len(log_wealth))
    return {
        "method": "fixed_bet_mixture_e_process",
        "n_pairs": n,
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "observed_paired_difference": (candidate_only - baseline_only) / n,
        "alpha": alpha,
        "bet_grid": list(bets),
        "final_log_e_value": log_e,
        "max_log_e_value": peak_log_e,
        "final_e_value": math.exp(log_e) if log_e < 700 else math.inf,
        "ever_crossed_threshold": first_crossing is not None,
        "first_crossing_pair": first_crossing,
        "log_threshold": log_threshold,
        "automatic_rollout_permitted": False,
        "assumption": "prospectively observed complete pairs with conditional null E[D_t | past] <= 0",
        "warning": (
            "Ville's inequality is conditional on the stated null and a predeclared fixed betting mixture. "
            "Post-hoc selection, dependence violating the conditional null, missing outcomes, "
            "or changing the stream after inspection invalidate anytime-valid claims."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", required=True,
                        help='JSONL records with exactly {"baseline":0|1,"candidate":0|1}')
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--acknowledge-design-assumptions", action="store_true",
                        help="Confirm the stream was prospectively defined and fully observed")
    args = parser.parse_args()
    if not args.acknowledge_design_assumptions:
        parser.error("explicit --acknowledge-design-assumptions required")
    pairs = []
    with Path(args.pairs_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"baseline", "candidate"}:
                raise ValueError(f"line {line_number}: only baseline and candidate fields allowed")
            pairs.append((row["baseline"], row["candidate"]))
    result = paired_e_process(pairs, alpha=args.alpha)
    # JSON output is standard-compliant even when a huge e-value overflows.
    if not math.isfinite(result["final_e_value"]):
        result["final_e_value"] = None
        result["warning"] += " Final e-value overflowed; use final_log_e_value."
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

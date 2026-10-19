"""Conservative off-policy evaluation for logged route decisions.

Each observation records the probability of the route actually selected by
the logging policy. Rewards MUST be external, independently graded [0, 1].
This module does not establish agent intelligence or causal validity.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from random import Random
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Decision:
    context: str
    action: str
    propensity: float
    reward: float

    def __post_init__(self):
        if not self.context or not self.action:
            raise ValueError("context and action are required")
        if not isfinite(self.propensity) or not 0 < self.propensity <= 1:
            raise ValueError("propensity must be in (0,1]")
        if not isfinite(self.reward) or not 0 <= self.reward <= 1:
            raise ValueError("externally graded reward must be in [0,1]")


def estimate(log: Iterable[Decision], policy: Callable[[str], str],
             *, max_weight: float = 20.0) -> dict:
    """Report IPS, self-normalized IPS, effective sample size, and support.

    IPS is unbiased only with valid randomized propensities, full target
    support, and no unmeasured selection; clipping introduces bias.
    """
    rows = list(log)
    if not rows:
        raise ValueError("empty log")
    if not isfinite(max_weight) or max_weight < 1:
        raise ValueError("max_weight must be finite and >= 1")
    weights = []
    weighted_rewards = []
    matches = 0
    clipped = 0
    for row in rows:
        selected = policy(row.context)
        if not isinstance(selected, str) or not selected:
            raise ValueError("policy must return a nonempty action")
        match = selected == row.action
        matches += int(match)
        raw = 1.0 / row.propensity if match else 0.0
        clipped += int(raw > max_weight)
        w = min(raw, max_weight)
        weights.append(w)
        weighted_rewards.append(w * row.reward)
    total = sum(weights)
    n = len(rows)
    return {
        "n": n,
        "matched": matches,
        "match_rate": matches / n,
        "ips": sum(weighted_rewards) / n,
        "snips": sum(weighted_rewards) / total if total else None,
        "effective_sample_size": total ** 2 / sum(w*w for w in weights) if total else 0.0,
        "clipped": clipped,
        "max_weight": max_weight,
        "warning": "No matching actions; target policy cannot be evaluated" if not matches else
                   ("Some weights clipped; IPS is biased" if clipped else None),
    }


def bootstrap_snips(log: Sequence[Decision], policy: Callable[[str], str],
                    *, replicates: int = 1000, seed: int = 0,
                    max_weight: float = 20.0) -> dict:
    """Percentile interval; exploratory only, not a guarantee of coverage.

    Independent observations are assumed. For repeated users/tasks,
    resample at the user/task cluster level instead.
    """
    rows = list(log)
    if not rows or replicates < 100:
        raise ValueError("nonempty log and at least 100 replicates required")
    rng = Random(seed)
    values = []
    for _ in range(replicates):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = estimate(sample, policy, max_weight=max_weight)["snips"]
        if value is not None:
            values.append(value)
    if len(values) < replicates * 0.9:
        raise ValueError("insufficient overlap in bootstrap resamples")
    values.sort()
    return {"snips": estimate(rows, policy, max_weight=max_weight)["snips"],
            "lower_95": values[int(0.025 * (len(values)-1))],
            "upper_95": values[int(0.975 * (len(values)-1))],
            "valid_replicates": len(values)}

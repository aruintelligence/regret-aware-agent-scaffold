"""Exact paired-discordance test for two binary systems on the SAME questions.

McNemar's exact test conditions on discordant outcomes. It assumes
independent question pairs. SQuAD questions share paragraphs and articles,
so the exact p-value is DESCRIPTIVE and may be anti-conservative; use the
article-cluster bootstrap as a separate dependence sensitivity check.

No significance test proves causality, general intelligence, or deployment
readiness, and no post-hoc selection correction is performed here.
"""
from __future__ import annotations

import math


def exact_mcnemar(baseline_only: int, candidate_only: int) -> dict:
    """Return exact conditional two-sided McNemar p, using integer arithmetic.

    Under the null of equal marginal success probability for two paired
    binary classifiers, candidate-only wins among discordant pairs have a
    Binomial(discordant, 0.5) distribution. The two-sided p-value is
    2 * P[X <= min(b, c)], capped at 1.
    """
    b, c = baseline_only, candidate_only
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (b, c)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
        p = min(1.0, (2 * tail) / (1 << n))
    return {
        "baseline_only": b,
        "candidate_only": c,
        "discordant_pairs": n,
        "candidate_win_fraction_among_discordant": c / n if n else None,
        "exact_two_sided_p": p,
        "null": "equal marginal success probability for paired binary outcomes",
        "assumption": "independent question pairs; violated by within-article/paragraph clustering",
        "interpretation": "exploratory descriptive test, not confirmatory evidence",
    }

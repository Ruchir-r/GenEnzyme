"""Small statistics helpers used for the reported recovery rates.

wilson, mcnemar_exact : numerically identical to the project code they mirror
                        (scripts/rerank_free.py:wilson, scripts/mcnemar.py:binom_two_sided).
cluster_bootstrap_ci  : NOT mirrored from any project script (none was found when this release was
                        assembled), so no paper number is claimed to come from it. Treat as a
                        convenience implementation and check it against the paper's procedure.
"""
import math
import random


def wilson(h, n, z=1.96):
    """95% Wilson score interval for h hits of n, returned in PERCENT (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - m), 100 * (c + m)


def mcnemar_exact(n10, n01):
    """Exact two-sided McNemar p-value from the discordant counts (binomial test, p = 0.5)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    tot = 2.0 ** n
    pk = math.comb(n, n10) / tot
    s = sum(math.comb(n, i) / tot for i in range(n + 1) if math.comb(n, i) / tot <= pk + 1e-15)
    return min(1.0, s)


def cluster_bootstrap_ci(hits, clusters, n_boot=10000, seed=0, alpha=0.05):
    """Percentile CI for the mean of `hits` (0/1 per row), resampling whole clusters (e.g. enzymes)."""
    by = {}
    for h, c in zip(hits, clusters):
        by.setdefault(c, []).append(h)
    groups = list(by.values())
    rng = random.Random(seed)
    k = len(groups)
    stats = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(k):
            g = groups[rng.randrange(k)]
            tot += sum(g)
            cnt += len(g)
        stats.append(tot / cnt)
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return 100 * lo, 100 * hi

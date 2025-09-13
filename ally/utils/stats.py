import math
from typing import Sequence, List

def _phi(z: float) -> float:
    # CDF of standard normal using erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def p_from_t_two_sided(t: float) -> float:
    z = abs(t)
    # two-sided p using normal approx (df large); CI uses synthetic OOS where this is fine
    return 2.0 * (1.0 - _phi(z))

def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0]*m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        q_i = pvals[i] * m / k
        prev = min(prev, q_i)
        q[i] = min(prev, 1.0)
    return q
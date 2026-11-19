"""Power calculations for the held-out test-set size (normal approximation, stdlib only).

Everything here takes ASSUMED inputs (discordance rate, seed noise). The pilot cannot measure
them yet; the outputs are planning numbers, not results.

Model. We compare two arms on the same test items. Let delta = true accuracy gain of arm A over
arm B, psi = share of items on which the two arms disagree (one right, one wrong; "discordant"),
sigma_seed = sd of one arm's accuracy across training seeds, k = seeds per arm.
  per-item variance of the paired difference  = psi - delta^2
  SE of the mean difference = sqrt((psi - delta^2)/n + 2*sigma_seed^2/k)
Seed noise does not shrink with more test items, so it puts a floor under the detectable gain.
"""
from math import ceil, sqrt
from statistics import NormalDist

_N = NormalDist()


def _z(alpha, power):
    return _N.inv_cdf(1 - alpha / 2), _N.inv_cdf(power)


def n_independent(p1, p2, alpha=0.05, power=0.8):
    """Items per arm for two independent proportions (reference only)."""
    za, zb = _z(alpha, power)
    pbar = (p1 + p2) / 2
    return ceil(((za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2)


def n_paired(delta, psi, alpha=0.05, power=0.8):
    """Items for a paired (McNemar-type) comparison, ignoring seed noise."""
    za, zb = _z(alpha, power)
    return ceil(((za * sqrt(psi) + zb * sqrt(psi - delta ** 2)) / delta) ** 2)


def n_with_seed_noise(delta, psi, sigma_seed, k, alpha=0.05, power=0.8):
    """Items needed when seed noise is included, or None if no test size can reach the power
    (seed noise alone already makes the SE too large)."""
    za, zb = _z(alpha, power)
    rhs = (delta / (za + zb)) ** 2 - 2 * sigma_seed ** 2 / k
    if rhs <= 0:
        return None
    return ceil((psi - delta ** 2) / rhs)


def power_at(n, delta, psi, sigma_seed, k, alpha=0.05):
    se = sqrt((psi - delta ** 2) / n + 2 * sigma_seed ** 2 / k)
    return _N.cdf(delta / se - _N.inv_cdf(1 - alpha / 2))


def min_detectable_delta(n, psi, sigma_seed, k, alpha=0.05, power=0.8):
    """Smallest gain detectable with the given power (bisection); None if not reachable.
    A gain cannot exceed the discordance rate psi (every gained item is a discordant item)."""
    lo, hi = 1e-4, psi * 0.999
    if power_at(n, hi, psi, sigma_seed, k, alpha) < power:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_at(n, mid, psi, sigma_seed, k, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def max_sigma_seed(delta, k, alpha=0.05, power=0.8):
    """Largest seed sd for which ANY test size can reach the power (beyond it, only more seeds
    or a bigger margin help)."""
    za, zb = _z(alpha, power)
    return delta / (za + zb) * sqrt(k / 2)

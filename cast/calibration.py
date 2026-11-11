"""Calibration by fixed-sequence exact binomial tests.

Paper: Section V-A and Theorem 3 (fixed-sequence calibration). Candidates are ordered from conservative to aggressive
and fixed before CAL is read; each is tested for H0: risk >= eps at level delta on its realised failures; the last
candidate before the first non-rejection is returned.
"""
import numpy as np
from scipy.stats import binom


def realised_failures(S, F, idx, U):
    """Number of failures on the queries idx when query i stops at the first checkpoint j with F[i, j] >= U[i]."""
    Fc = np.maximum.accumulate(np.clip(F, 0, 1), axis=1); Fc[:, -1] = 1.0; jst = (Fc >= U[:, None]).argmax(1)
    return int((S.X[idx] > S.c[idx, jst]).sum())


def fixed_sequence(S, thetas, makeF, eps, idx=None, U=None, delta=0.05):
    """Fixed-sequence test over the candidates `thetas` (conservative first).

    makeF(theta, idx) returns the stopping matrix of the candidate on the queries idx, or None when the candidate is
    infeasible (decided without reading idx); such candidates are skipped. Returns (last passing candidate or None, trace).
    The default sample is CAL with its stopping draws.
    """
    idx = S.CAL if idx is None else idx; U = S.u_cal if U is None else U; prev, trace = None, []
    for th in thetas:
        F = makeF(th, idx); thr = th if np.isscalar(th) else str(th)
        if F is None: trace.append((thr, None, 'infeasible')); continue
        k = realised_failures(S, F, idx, U); ok = bool(binom.cdf(k, len(idx), eps) <= delta); trace.append((thr, k, ok))
        if not ok: break
        prev = th
    return prev, trace


def fixed_sequence_deterministic(candidates, cost_fail, rows, eps, level):
    """Fixed-sequence test for rules given by per-query outcomes: cost_fail(candidate, rows) -> (cost, failure flags)."""
    n = len(rows); prev, trace = None, []
    for c in candidates:
        _, f = cost_fail(c, rows); k = int(f.sum()); ok = bool(binom.cdf(k, n, eps) <= level); trace.append((c, k, ok))
        if not ok: break
        prev = c
    return prev, trace


def binomial_critical(n, eps, alpha):
    """Smallest count f with P(Bin(n, eps) >= f) <= alpha: a one-sided exact test of H0: p <= eps rejects from f failures."""
    f = int(n * eps)
    while binom.sf(f - 1, n, eps) > alpha: f += 1
    return f

"""The nested chain of staircase rules: exact Lagrangian minimizers by dynamic programming.

Paper: Section IV-B (Lagrangian chain), Lemma 3 (duality), Eq. (4), Theorem 2 (nested chain) and Algorithm 1
(exact chain element for one multiplier, O(J (B + 1)^2) time).
"""
import numpy as np

from .monotone_class import cached_coefs, frontier_slope
from .rules import staircase_table


def chain_element(S, idx, lam_int):
    """Algorithm 1: the maximal minimizer of the empirical Lagrangian cost + lam * risk over the class, on the queries idx.

    Minimises the sum over the stopping cells of theta = -(cc - lam rc) (Eq. (4)) over staircases beta_j(a) in {-1..NB-1}
    with beta_j(0) <= beta_j(1) and beta nondecreasing in j. Integer arithmetic; ties go to the larger stopping region,
    which makes the chain nested (Theorem 2). Returns (beta of shape (J-1, 2), Lagrangian value of the rule, rule table).
    """
    J, NB, MP = S.J, S.NB, S.tie_weight
    cc, rc, bc, br, n = cached_coefs(S, idx)
    th = (np.rint(-cc[:-1]).astype(np.int64) + np.int64(lam_int) * np.rint(rc[:-1]).astype(np.int64)) * MP - 1
    th = th.reshape(J - 1, NB, 2); Bp = NB + 1
    P = np.zeros((J - 1, 2, Bp), np.int64); P[:, :, 1:] = np.cumsum(th.transpose(0, 2, 1), axis=2)
    INF = np.int64(1) << 62; tri = np.arange(Bp)[:, None] <= np.arange(Bp)[None, :]
    V = np.empty((J - 1, Bp, Bp), np.int64); V[0] = np.where(tri, P[0, 0][:, None] + P[0, 1][None, :], INF)
    for j in range(1, J - 1):
        Mn = np.minimum.accumulate(np.minimum.accumulate(V[j - 1], axis=0), axis=1)
        V[j] = np.where(tri, P[j, 0][:, None] + P[j, 1][None, :] + Mn, INF)
    k = int(V[J - 2].argmin()); b0, b1 = divmod(k, Bp); beta = np.zeros((J - 1, 2), np.int64); beta[J - 2] = (b0, b1)
    for j in range(J - 2, 0, -1):
        sub = V[j - 1, :b0 + 1, :b1 + 1]; k = int(sub.argmin()); b0, b1 = divmod(k, b1 + 1); beta[j - 1] = (b0, b1)
    beta = beta - 1; f = staircase_table(S, beta)
    val = (bc + lam_int * br - (cc - lam_int * rc).ravel() @ f.ravel()) / n
    return beta, float(val), f


def chain_multipliers(slopes=()):
    """Multipliers of the chain: 800 integers on a geometric grid from 1e2 to 1e7.

    The grid is extended outward at the same ratio until it covers every frontier slope in `slopes`.
    Returns (sorted unique integer multipliers, description of the extension).
    """
    v = [float(x) for x in slopes if x]; r = (1e7 / 1e2) ** (1 / 799); nlo = nhi = 0
    while v and 1e2 * r ** (-nlo) > min(v): nlo += 1
    while v and 1e7 * r ** nhi < max(v): nhi += 1
    if nlo == 0 and nhi == 0:
        return (np.unique(np.rint(np.geomspace(1e2, 1e7, 800)).astype(np.int64)),
                dict(extended=False, slope_min=min(v) if v else None, slope_max=max(v) if v else None))
    lo, hi = 1e2 * r ** (-nlo), 1e7 * r ** nhi
    return (np.unique(np.rint(np.geomspace(lo, hi, 800 + nlo + nhi)).astype(np.int64)),
            dict(extended=True, n_below=nlo, n_above=nhi, lo=lo, hi=hi, slope_min=min(v), slope_max=max(v)))


def bound_multipliers(S, levels):
    """Multipliers lambda_k of the lower bound: the negated slopes of the SEL frontier at the risk levels, rounded.

    Returns (sorted distinct multipliers, slope per level, rounded slope per level).
    """
    slopes = {e: frontier_slope(S, S.SEL, e) for e in levels}
    by_level = {e: int(round(v)) for e, v in slopes.items() if v}
    return sorted(set(by_level.values())), slopes, by_level

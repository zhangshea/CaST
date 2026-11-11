"""Stopping rules as matrices of cumulative stopping probabilities, and their cost and risk on logged trajectories.

Paper: Definition 2 (a rule stops at the first checkpoint T with f_T(z_T) >= xi), Eq. (2) (cost C = E c_T and
risk R = P(c_T < X)), Theorem 1(b) (staircase rules) and the threshold rules of Section VI-A (Level, Level+cap, Fixed).

A rule evaluated on the queries idx is a matrix F of shape (len(idx), J): F[i, j] is the probability that query i has
stopped by checkpoint j. Deterministic rules have 0/1 entries.
"""
import numpy as np


def evaluate_rule(S, F, idx):
    """Per-query expected cost c_T and failure probability P(c_T < X) of a rule on the queries idx (Eq. (2))."""
    c = S.c; F = np.maximum.accumulate(np.clip(F, 0, 1), axis=1); F[:, -1] = 1.0
    dF = np.diff(np.concatenate([np.zeros((len(idx), 1)), F], 1), axis=1)
    return (dF * c[idx]).sum(1), (dF * (S.X[idx, None] > c[idx])).sum(1)


def terminal_F(S, idx):
    """The rule that waits until the last checkpoint."""
    J = S.J
    return np.where(np.arange(J)[None, :] >= J - 1, 1.0, 0.0) * np.ones((len(idx), 1))


def fixed_budget_F(S, j, idx):
    """Fixed: stop at checkpoint j for every query."""
    return np.where(np.arange(S.J)[None, :] >= j, 1.0, 0.0) * np.ones((len(idx), 1))


def level_F(S, lam, j_cap, idx):
    """Level+cap: stop once the running minimum of the score falls below lam, or at checkpoint j_cap (Level: j_cap = J - 1)."""
    F = (np.minimum.accumulate(S.s[idx], axis=1) < lam).astype(float); F[:, j_cap:] = 1.0
    return F


def staircase_table(S, beta):
    """J x ns 0/1 table of the staircase with thresholds beta[j, a]: f_j(b, a) = 1{b <= beta_j(a)} (Theorem 1(b))."""
    J, NB = S.J, S.NB
    f = np.ones((J, S.cls.ns))
    for a_ in (0, 1): f[:J - 1, a_::2] = (np.arange(NB)[None, :] <= beta[:, a_][:, None]).astype(float)
    return f


def staircase_F(S, beta, idx):
    """Stopping matrix of a staircase rule on the queries idx: stop at checkpoint j iff b_j <= beta_j(a_j)."""
    J, NB = S.J, S.NB
    bf = np.vstack([beta, [NB - 1, NB - 1]])
    return (S.b[idx] <= bf[np.arange(J)[None, :], S.a[idx]]).astype(float)


def bootstrap_means(x, B=2000, seed=0):
    """Means of B bootstrap resamples of x."""
    rng = np.random.default_rng(seed); n = len(x); w = rng.multinomial(n, np.ones(n) / n, size=B) / n
    return w @ x

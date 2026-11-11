"""Rigorous versions of the bounds U and L, without floating-point allowances.

Paper: Theorem 4 and Corollary 1. The reported ratios U / L are computed here and rounded up.

Differences from cast/bounds.py:
  (1) the coefficients of W and W^2 are accumulated in exact integers;
  (2) every cut keeps its floating-point slopes (a, b) and gets the constant
          c(a, b) = min over W in [0, R] of [(1 - a) W - b W^2] - kappa_up^2 / (4 |b|),   R = H + lambda,  b < 0,
      which makes the cut an under-estimator of G(W, Q) = W - kappa sqrt(Q - W^2) on {0 <= W <= R, Q >= W^2};
  (3) each master value is bounded below by Lagrangian relaxation in exact rationals, with the solver's multipliers
      clipped at zero and the cut multipliers rescaled to sum to one;
  (4) logarithms and square roots are rounded outward (60 decimal digits plus one unit in the last place), and the levels
      alpha_L / K and alpha_U / M are exact rationals.
"""
import math
from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from .bounds import lagrangian_coefs, lagrangian_minimum, polytope_box

PREC = 60


def ulp(d): return Decimal(1).scaleb(d.adjusted() - PREC + 1)


def ln_up(q):
    """Upper bound of ln(q) for an integer q > 1."""
    assert isinstance(q, int) and q > 1
    with localcontext() as ctx:
        ctx.prec = PREC; v = Decimal(q).ln(); return Fr(v + ulp(v))


def frac_up_decimal(x):
    """A Decimal >= the positive Fraction x."""
    with localcontext() as ctx:
        ctx.prec = PREC; ctx.rounding = ROUND_CEILING; return Decimal(x.numerator) / Decimal(x.denominator)


def sqrt_up(x):
    """Upper bound of sqrt(x) for a Fraction x >= 0."""
    if x <= 0: return Fr(0)
    with localcontext() as ctx:
        ctx.prec = PREC; d = frac_up_decimal(x); r = d.sqrt(); return Fr(r + ulp(r))


def down6(x): return math.floor(x * 10 ** 6) / 10 ** 6


def up6(x): return math.ceil(x * 10 ** 6) / 10 ** 6


def lagrangian_coefs_int(S, idx, lam):
    """lagrangian_coefs in exact int64 arithmetic."""
    J, cls = S.J, S.cls
    c = S.c[idx].astype(np.int64); w = c + int(lam) * (S.X[idx, None] > S.c[idx]).astype(np.int64)
    assert int(w.max()) ** 2 * len(idx) < 2 ** 62
    dw = np.diff(w, axis=1); d2 = np.diff(w * w, axis=1); Zi = S.Z[idx]
    cw = np.zeros((J, cls.ns), np.int64); cq = np.zeros((J, cls.ns), np.int64)
    for j in range(J - 1): np.add.at(cw[j], Zi[:, j], dw[:, j]); np.add.at(cq[j], Zi[:, j], d2[:, j])
    return cw.ravel(), int(w[:, -1].sum()), cq.ravel(), int((w[:, -1] * w[:, -1]).sum()), len(idx)


def cut_constant(a, b, R, kap2_up):
    """Constant of the cut with slopes (a, b), b < 0, that keeps it below G on the whole domain."""
    nb = -b; assert nb > 0
    ws = (a - 1) / (2 * nb); ws = min(max(ws, Fr(0)), Fr(R))
    return (1 - a) * ws + nb * ws * ws - kap2_up / (4 * nb)


class ExactBounds:
    """Exact bounds for one setting, with levels alpha_L / K (per multiplier) and alpha_U / M (per risk level)."""

    def __init__(self, S, alpha_L, alpha_U, K, M):
        self.S = S
        self.alpha_k = Fr(str(alpha_L)) / K; self.delta_u = Fr(str(alpha_U)) / M
        assert (2 / self.alpha_k).denominator == 1 and (2 / self.delta_u).denominator == 1
        self.ln_L = ln_up(int(2 / self.alpha_k)); self.ln_U = ln_up(int(2 / self.delta_u))
        A0 = S.cls.A_mono.tocoo()
        self.rows, self.cols, self.vals = A0.row.tolist(), A0.col.tolist(), [int(v) for v in A0.data]
        assert all(v in (-1, 1) for v in self.vals)
        LO, HI, _ = polytope_box(S)
        self.lo = [Fr(int(v)) for v in LO]; self.hi = [Fr(int(v)) for v in HI]

    def master_bound(self, ym, yc, cuts, cw, bw, cq, bq, n):
        """Lower bound on the master problem by Lagrangian relaxation, in exact rationals."""
        cls = self.S.cls
        s_ = sum((Fr(float(v)) for v in yc), Fr(0))
        if s_ <= 0: return None
        ycn = [Fr(float(v)) / s_ for v in yc]
        A_ = sum((y * ct['a'] for y, ct in zip(ycn, cuts)), Fr(0)); B_ = sum((y * ct['b'] for y, ct in zip(ycn, cuts)), Fr(0))
        rx = [Fr(0)] * cls.nv
        ymf = [Fr(float(v)) for v in ym]
        for r_, c_, v_ in zip(self.rows, self.cols, self.vals):
            if ymf[r_]: rx[c_] += v_ * ymf[r_]
        tot = Fr(0)
        for v in range(cls.nv):
            r_v = rx[v] - (A_ * int(cw[v]) + B_ * int(cq[v])) / n
            tot += self.lo[v] * r_v if r_v >= 0 else self.hi[v] * r_v
        tot += sum((y * ((ct['a'] * bw + ct['b'] * bq) / n + ct['c']) for y, ct in zip(ycn, cuts)), Fr(0))
        return tot

    def lagrangian_lower_bound(self, idx, lam, iters=10):
        """Exact ell(lam) on the queries idx: the same cutting-plane loop as the float version, with rigorous constants."""
        S = self.S; cls, H = S.cls, S.H; LO, HI, B0 = polytope_box(S)
        n = len(idx); R = H + lam
        kap = float(np.sqrt(2 * np.log(float(2 / self.alpha_k)) / (n - 1)))        # the float kappa only steers the cut points
        kap2_up = 2 * self.ln_L / (n - 1); b_up = Fr(7) * R * self.ln_L / (3 * (n - 1))
        cw, bw, cq, bq, nn = lagrangian_coefs_int(S, idx, lam); cwf = cw.astype(np.float64); cqf = cq.astype(np.float64)
        cwfl, bwfl, cqfl, bqfl, _ = lagrangian_coefs(S, idx, lam)                 # float coefficients, compared below
        check = dict(cw_maxrel=float(np.abs(cwfl.ravel() - cwf).max() / max(1.0, np.abs(cwf).max())),
                     cq_maxrel=float(np.abs(cqfl.ravel() - cqf).max() / max(1.0, np.abs(cqf).max())),
                     bw_rel=abs(bwfl - bw) / max(1, abs(bw)), bq_rel=abs(bqfl - bq) / max(1, abs(bq)))
        Wf = lambda x: (bw - cwf @ x) / nn; Qf = lambda x: (bq - cqf @ x) / nn
        v0 = lagrangian_minimum(S, idx, lam); x = v0['x']; nv = cls.nv; T = 1e7
        A0t = csr_matrix((cls.A_mono.data, cls.A_mono.indices, cls.A_mono.indptr), shape=(cls.n_mono, nv + 1))
        lo = np.append(LO, -T); hi = np.append(HI, T); obj = np.zeros(nv + 1); obj[-1] = 1.0
        cuts, rows, rhs, best, tr = [], [], [], None, []
        for it in range(iters):
            W0 = Wf(x); sd = np.sqrt(max(Qf(x) - W0 ** 2, 1.0)); wc = 1.0 + kap * W0 / sd; wq = -kap / (2 * sd); assert wq < 0
            ct = dict(a=Fr(wc), b=Fr(wq)); ct['c'] = cut_constant(ct['a'], ct['b'], R, kap2_up); cuts.append(ct)
            rows.append(np.append(-(wc * cwf + wq * cqf) / nn, -1.0)); rhs.append(-(wc * bw + wq * bq) / nn - float(ct['c']))
            A = vstack([A0t, csr_matrix(np.array(rows))]).tocsr(); bb = np.append(B0, rhs)
            res = linprog(obj, A_ub=A, b_ub=bb, bounds=list(zip(lo, hi)), method='highs')
            if res.status != 0: break
            y = np.maximum(-np.asarray(res.ineqlin.marginals), 0.0)
            bound = self.master_bound(y[:cls.n_mono], y[cls.n_mono:], cuts, cw, bw, cq, bq, nn)
            x = res.x[:nv]
            true_obj = Wf(x) - kap * np.sqrt(max(Qf(x) - Wf(x) ** 2, 0.0))
            if bound is not None:
                ell_it = bound - b_up; best = ell_it if best is None or ell_it > best else best
            tr.append(dict(t=float(res.fun), bound=float(bound) if bound is not None else None, cut_sum=float(np.sum(y[cls.n_mono:])),
                           true_obj_at_x=float(true_obj)))
            if true_obj - res.fun <= 1e-3 * max(abs(res.fun), 1.0): break
        assert best is not None, 'no master problem was solved'
        return dict(lam=lam, ell=best, ell_down6=down6(best), coef_check=check, kappa2_up=float(kap2_up), b_up=float(b_up), trace=tr)

    def cost_upper_bound(self, cost):
        """Exact empirical-Bernstein upper bound U on the mean of the integer per-query costs; returns (U, mean)."""
        ci = np.rint(cost).astype(np.int64); assert np.array_equal(ci.astype(np.float64), cost)
        n = len(ci); S1 = int(ci.sum()); S2 = int(sum(int(v) * int(v) for v in ci.tolist()))
        mean = Fr(S1, n); var = (Fr(S2) - Fr(S1) ** 2 / n) / (n - 1)
        return mean + sqrt_up(2 * var * self.ln_U / n) + Fr(7) * self.S.H * self.ln_U / (3 * (n - 1)), mean

    @staticmethod
    def frontier(ells, eps):
        """Exact L(eps) = max(0, max_k [ell(lam_k) - lam_k eps])."""
        return max([Fr(0)] + [d_['ell'] - d_['lam'] * Fr(str(eps)) for d_ in ells])

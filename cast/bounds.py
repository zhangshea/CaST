"""Upper bound U on the cost of a calibrated rule and lower bounds on the class optimum.

Paper: Section V-B, Theorem 4 (simultaneous lower bound, Eq. (5)) and Corollary 1 (optimality-gap bound U / L);
the lower bound valid at one risk level only is the comparison of the paragraph "Interface and lower bound" (Section VI-B).

Floating-point versions. pipeline/exact_bounds.py recomputes U and L with outward rounding for the reported ratios.
"""
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, vstack

from .monotone_class import dual_lower_bound, empirical_optimum, lp_parts

# Relative allowance for floating-point error applied to the float bounds.
REL = 1e-6


def cost_upper_bound(x, delta, R):
    """Empirical-Bernstein upper bound (Maurer and Pontil) on the mean of variables in [0, R], at level delta."""
    n = len(x)
    return float(x.mean() + np.sqrt(2 * x.var(ddof=1) * np.log(2 / delta) / n) + 7 * R * np.log(2 / delta) / (3 * (n - 1)))


def kl_upper(eps, n, alpha):
    """The q >= eps with n * kl(q || eps) = log(1 / alpha)."""
    t = np.log(1 / alpha) / n; f = lambda q: q * np.log(q / eps) + (1 - q) * np.log((1 - q) / (1 - eps)) - t
    return 1.0 if f(1 - 1e-12) <= 0 else float(brentq(f, eps, 1 - 1e-12))


# ------------------------------------------------------------------ simultaneous lower bound (Theorem 4)
def lagrangian_coefs(S, idx, lam):
    """Cell sums of the Lagrangian loss W = c_T + lam 1{c_T < X} and of its square on the queries idx.

    W_i(f) = w_{J-1} - sum_j f_j(z_j) (w_{j+1} - w_j) and Q_i(f) = w_{J-1}^2 - sum_j f_j(z_j) (w_{j+1}^2 - w_j^2)
    with w_j = c_j + lam 1{c_j < X}. Returns (cw, bw, cq, bq, n): cell sums and constants of W and of W^2.
    """
    J, cls = S.J, S.cls
    c = S.c[idx].astype(np.float64); w = c + lam * (S.X[idx, None] > S.c[idx]); dw = np.diff(w, axis=1); d2 = np.diff(w ** 2, axis=1)
    Zi = S.Z[idx]; cw = np.zeros((J, cls.ns)); cq = np.zeros((J, cls.ns))
    for j in range(J - 1): np.add.at(cw[j], Zi[:, j], dw[:, j]); np.add.at(cq[j], Zi[:, j], d2[:, j])
    return cw, float(w[:, -1].sum()), cq, float((w[:, -1] ** 2).sum()), len(idx)


def polytope_box(S):
    """Bounds of the order polytope without the risk row: (lo, hi, zero right-hand side of the difference constraints)."""
    cls = S.cls
    lo = np.zeros(cls.nv); hi = np.ones(cls.nv); lo[(S.J - 1) * cls.ns:] = 1.0
    return lo, hi, np.zeros(cls.n_mono)


def lagrangian_minimum(S, idx, lam):
    """Minimum of the empirical Lagrangian over the order polytope: primal value, weak-duality lower bound, minimizer."""
    cls = S.cls; LO, HI, B0 = polytope_box(S)
    cw, bw, _, _, n = lagrangian_coefs(S, idx, lam); obj = -cw.ravel() / n
    res = linprog(obj, A_ub=cls.A_mono, b_ub=B0, bounds=list(zip(LO, HI)), method='highs'); assert res.status == 0
    return dict(value=bw / n + float(res.fun), dual=bw / n + dual_lower_bound(res, obj, cls.A_mono, B0, LO, HI), x=res.x)


def lagrangian_lower_bound(S, idx, lam, alpha_k, iters=10):
    """ell(lam): a lower bound, at level alpha_k, on the population minimum of cost + lam * risk over the class (Theorem 4).

    ell(lam) = min over the polytope of [What - kappa sqrt(Qhat - What^2)] - b, the empirical-Bernstein bound for the fixed
    population minimizer, whose loss has range H + lam. The concave term is handled by Kelley cutting planes: every
    tangent plane under-estimates the objective, and each master value is replaced by its weak-duality lower bound.
    """
    cls, H = S.cls, S.H; LO, HI, B0 = polytope_box(S)
    n = len(idx); lc = np.log(2 / alpha_k); kap = np.sqrt(2 * lc / (n - 1)); b = 7 * (H + lam) * lc / (3 * (n - 1))
    cw, bw, cq, bq, nn = lagrangian_coefs(S, idx, lam); cw = cw.ravel(); cq = cq.ravel()
    Wf = lambda x: (bw - cw @ x) / nn; Qf = lambda x: (bq - cq @ x) / nn
    v0 = lagrangian_minimum(S, idx, lam); x = v0['x']; nv = cls.nv; T = 1e7
    A0t = csr_matrix((cls.A_mono.data, cls.A_mono.indices, cls.A_mono.indptr), shape=(cls.n_mono, nv + 1))
    lo = np.append(LO, -T); hi = np.append(HI, T); obj = np.zeros(nv + 1); obj[-1] = 1.0
    cut_rows, cut_b, best, tr = [], [], -np.inf, []
    for it in range(iters):
        W0 = Wf(x); sd = np.sqrt(max(Qf(x) - W0 ** 2, 1.0)); wc = 1.0 + kap * W0 / sd; wq = -kap / (2 * sd); const = -kap * (W0 ** 2 + sd ** 2) / (2 * sd)
        cut_rows.append(np.append(-(wc * cw + wq * cq) / nn, -1.0)); cut_b.append(-(wc * bw + wq * bq) / nn - const)
        A = vstack([A0t, csr_matrix(np.array(cut_rows))]).tocsr(); bb = np.append(B0, cut_b)
        res = linprog(obj, A_ub=A, b_ub=bb, bounds=list(zip(lo, hi)), method='highs')
        if res.status != 0: break
        t_dual = dual_lower_bound(res, obj, A, bb, lo, hi); x = res.x[:nv]
        true_obj = Wf(x) - kap * np.sqrt(max(Qf(x) - Wf(x) ** 2, 0.0)); best = max(best, t_dual - b)
        tr.append(dict(t=float(res.fun), t_dual=float(t_dual), true_obj_at_x=float(true_obj), gap=float(true_obj - res.fun)))
        if true_obj - res.fun <= 1e-3 * max(abs(res.fun), 1.0): break
    return dict(ell=float(best), lam=float(lam), alpha=float(alpha_k), kappa=float(kap), b=float(b), g_hat=v0['value'], g_hat_dual=v0['dual'],
                sd_W_at_min=float(np.sqrt(max(Qf(v0['x']) - Wf(v0['x']) ** 2, 0.0))), trace=tr)


def frontier_lower_bound(ells, eps, rel=REL):
    """L(eps) = max(0, max_k [ell(lam_k) - lam_k eps]) (Eq. (5)), valid for all risk levels at once."""
    return max(0.0, max(d['ell'] - rel * max(1.0, abs(d['ell'])) - d['lam'] * eps for d in ells))


# ------------------------------------------------------------------ lower bound valid at one risk level
def second_moment_coefs(S, idx):
    """Cell sums of the squared cost: Q_i(f) = c_{J-1}^2 - sum_j f_j(z_j) (c_{j+1}^2 - c_j^2)."""
    J, cls = S.J, S.cls
    c = S.c[idx].astype(np.float64); d2 = np.diff(c ** 2, axis=1); Zi = S.Z[idx]; cq = np.zeros((J, cls.ns))
    for j in range(J - 1): np.add.at(cq[j], Zi[:, j], d2[:, j])
    return cq, float((c[:, -1] ** 2).sum())


def single_level_lower_bound(S, idx, eps, alpha, iters=8):
    """Lower bound on the class optimum at the single risk level eps, at level alpha.

    A KL relaxation of the risk constraint (alpha / 2) and the empirical-Bernstein cost bound (alpha / 2) for the fixed
    population optimizer. The objective C - kappa sqrt(Q - C^2) is treated by Kelley cutting planes as in
    lagrangian_lower_bound.
    """
    cls, H = S.cls, S.H
    n = len(idx); qr = kl_upper(eps, n, alpha / 2); lc = np.log(2 / (alpha / 2)); kap = np.sqrt(2 * lc / (n - 1)); b = 7 * H * lc / (3 * (n - 1))
    P_ = lp_parts(S, idx, qr)
    if P_ is None: return dict(L=None)
    cq, bq = second_moment_coefs(S, idx); cq = cq.ravel(); nn = P_['n']
    Cf = lambda x: (P_['bc'] - P_['cc'] @ x) / nn; Qf = lambda x: (bq - cq @ x) / nn
    v0 = empirical_optimum(S, idx, qr); x = v0['x']; nv = cls.nv
    A0 = P_['A0']; A0t = csr_matrix((A0.data, A0.indices, A0.indptr), shape=(A0.shape[0], nv + 1))
    lo = np.append(P_['lo'], -1e5); hi = np.append(P_['hi'], 1e5); obj = np.zeros(nv + 1); obj[-1] = 1.0
    cut_rows, cut_b, best, tr = [], [], -np.inf, []
    for it in range(iters):
        C0 = Cf(x); sd = np.sqrt(max(Qf(x) - C0 ** 2, 1.0)); wc = 1.0 + kap * C0 / sd; wq = -kap / (2 * sd); const = -kap * (C0 ** 2 + sd ** 2) / (2 * sd)
        cut_rows.append(np.append(-(wc * P_['cc'] + wq * cq) / nn, -1.0)); cut_b.append(-(wc * P_['bc'] + wq * bq) / nn - const)
        A = vstack([A0t, csr_matrix(np.array(cut_rows))]).tocsr(); bb = np.append(P_['b0'], cut_b)
        res = linprog(obj, A_ub=A, b_ub=bb, bounds=list(zip(lo, hi)), method='highs')
        if res.status != 0: break
        t_dual = dual_lower_bound(res, obj, A, bb, lo, hi); x = res.x[:nv]
        true_obj = Cf(x) - kap * np.sqrt(max(Qf(x) - Cf(x) ** 2, 0.0))
        lbv = t_dual - b; best = max(best, lbv)
        tr.append(dict(t=float(res.fun), t_dual=float(t_dual), true_obj_at_x=float(true_obj), gap=float(true_obj - res.fun), lb=float(lbv)))
        if true_obj - res.fun <= 1e-3 * max(abs(res.fun), 1.0): break
    return dict(L=float(max(best, 0.0)), q_r=qr, kappa=float(kap), b=float(b), V_qr=v0['value'], V_qr_dual=v0['dual'], trace=tr)

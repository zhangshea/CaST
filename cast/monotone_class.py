"""The class of monotone stopping rules as a linear program over an order polytope.

Paper: Definition 2 (monotone stopping rules, Eq. (1)), Proposition 1 (affine representation of cost and risk),
Theorem 1 (order polytope) and problem (3).

A rule is a table f[j, z] in [0, 1] over checkpoints j and interface states z = (bucket b, flag a). Cell (j, z) has
the flat index j * ns + z with z = 2 * b + a, and f[J-1, .] = 1.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


def state_index(b, a):
    """Flat index of the interface state z = (bucket b, flag a)."""
    return b * 2 + a


class MonotoneClass:
    """Difference constraints of Eq. (1) for J checkpoints and nb buckets.

    Each row of A_mono encodes x_u - x_v <= 0 for a covering pair u <= v of the product order:
    a lower bucket, a set flag and a later checkpoint never lower the stopping probability.
    """

    def __init__(self, J, nb):
        self.J = J
        self.nb = nb
        self.ns = nb * 2
        self.nv = J * self.ns
        rows, cols, vals = [], [], []
        r = 0
        v = lambda j, s: j * self.ns + s

        def le(x, y):
            nonlocal r
            rows.extend([r, r]); cols.extend([x, y]); vals.extend([1.0, -1.0]); r += 1

        for j in range(J):
            for b in range(nb):
                for a in range(2):
                    s = state_index(b, a)
                    if b + 1 < nb: le(v(j, state_index(b + 1, a)), v(j, s))
                    if a == 0: le(v(j, s), v(j, state_index(b, 1)))
                    if j + 1 < J: le(v(j, s), v(j + 1, s))
        self.A_mono = coo_matrix((vals, (rows, cols)), shape=(r, self.nv)).tocsr()
        self.n_mono = r


def class_coefs(S, idx):
    """Empirical coefficients of Proposition 1 on the queries idx.

    Returns (cc, rc, bc, br, n): cc[j, z] sums the saved visits c_{j+1} - c_j and rc[j, z] counts the queries that
    complete between checkpoints j and j + 1, over the queries in state z at checkpoint j; bc is the total cost of
    never stopping early and br the number of queries that do not complete by the last checkpoint.
    """
    J, cls = S.J, S.cls
    c, dC, X = S.c, S.dC, S.X
    Zi = S.Z[idx]; n = len(idx)
    cc = np.zeros((J, cls.ns)); rc = np.zeros((J, cls.ns))
    hit = (X[idx, None] > c[idx, :-1]) & (X[idx, None] <= c[idx, 1:])
    for j in range(J - 1):
        np.add.at(cc[j], Zi[:, j], dC[idx, j]); np.add.at(rc[j], Zi[:, j], hit[:, j])
    return cc, rc, c[idx, -1].sum(), (X[idx] > c[idx, -1]).sum(), n


def cached_coefs(S, idx):
    """class_coefs of a sample, computed once per sample."""
    key = (len(idx), int(idx[0]), int(idx[-1]), int(np.asarray(idx, np.int64).sum()))
    if key not in S.coef_cache:
        S.coef_cache[key] = class_coefs(S, idx)
    return S.coef_cache[key]


def solve_class_lp(S, idx, eps, pen=0.0):
    """Empirical version of problem (3) on the queries idx: least mean cost over the class at risk at most eps.

    `pen` is a small per-cell penalty that breaks ties toward stopping later. Returns None when infeasible.
    """
    J, cls = S.J, S.cls
    cc, rc, bc, br, n = class_coefs(S, idx); budget = eps * n - br
    if budget < 0: return None
    A_ub = vstack([cls.A_mono.copy(), csr_matrix(rc.ravel()[None, :])]); b_ub = np.append(np.zeros(cls.n_mono), budget)
    bounds = [(0.0, 1.0)] * cls.nv
    for s in range(cls.ns): bounds[(J - 1) * cls.ns + s] = (1.0, 1.0)
    obj = -cc.ravel() + (pen if pen else 0.0)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.status != 0: return None
    f = res.x.reshape(J, cls.ns); val = (bc - cc.ravel() @ res.x) / n; rsk = (br + rc.ravel() @ res.x) / n
    return dict(f=f, value=float(val), risk=float(rsk), viol=float(max(0.0, (A_ub @ res.x - b_ub).max())),
                empty=float((cc[:-1] == 0).mean()))


def table_F(S, f, idx):
    """Cumulative stopping probabilities f_j(z_j) of a rule table along the trajectories of the queries idx."""
    return f[np.arange(S.J)[None, :], S.Z[idx]]


def dual_lower_bound(res, obj, A_ub, b_ub, lo, hi):
    """Lower bound on min obj.x s.t. A_ub x <= b_ub, lo <= x <= hi by weak duality with the solver's multipliers.

    Any multiplier vector y >= 0 gives a valid bound when all boxes are finite.
    """
    y = np.maximum(-np.asarray(res.ineqlin.marginals), 0.0); r = obj + A_ub.T @ y
    return float(np.minimum(lo * r, hi * r).sum() - b_ub @ y)


def lp_parts(S, idx, risk_budget):
    """Matrices of the empirical problem (3) at the given risk budget, or None when infeasible."""
    J, cls = S.J, S.cls
    cc, rc, bc, br, n = class_coefs(S, idx); budget = risk_budget * n - br
    if budget < 0: return None
    lo = np.zeros(cls.nv); hi = np.ones(cls.nv); lo[(J - 1) * cls.ns:] = 1.0
    return dict(cc=cc.ravel(), bc=bc, n=n, A0=vstack([cls.A_mono, csr_matrix(rc.ravel()[None, :])]).tocsr(),
                b0=np.append(np.zeros(cls.n_mono), budget), lo=lo, hi=hi)


def empirical_optimum(S, idx, risk_budget):
    """Empirical class optimum at a risk budget: primal value and a weak-duality lower bound on it."""
    P_ = lp_parts(S, idx, risk_budget)
    if P_ is None: return None
    obj = -P_['cc'] / P_['n']
    res = linprog(obj, A_ub=P_['A0'], b_ub=P_['b0'], bounds=list(zip(P_['lo'], P_['hi'])), method='highs')
    if res.status != 0: return None
    const = P_['bc'] / P_['n']
    return dict(value=const + float(res.fun), dual=const + dual_lower_bound(res, obj, P_['A0'], P_['b0'], P_['lo'], P_['hi']), x=res.x)


def frontier_slope(S, idx, eps):
    """Slope -dV/d(eps) of the empirical cost-risk frontier at eps: the multiplier of the risk row, in visits per unit risk."""
    P_ = lp_parts(S, idx, eps)
    if P_ is None: return None
    res = linprog(-P_['cc'] / P_['n'], A_ub=P_['A0'], b_ub=P_['b0'], bounds=list(zip(P_['lo'], P_['hi'])), method='highs')
    return None if res.status != 0 else float(max(-res.ineqlin.marginals[-1], 0.0) * P_['n'])

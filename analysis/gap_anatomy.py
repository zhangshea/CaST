"""Exact split of the gap U - L of a calibrated rule into five terms.

Paper: Section VI-B, paragraph "Near-optimality" ("The gap U - L of the chain rule splits exactly into five terms"):
    U - L = (U - C) + (C - V(r)) + (V(r) - V(eps)) + (V(eps) - E(eps)) + (E(eps) - L(eps)),
the cost bound, the rule against the empirical frontier, the displacement of the risk budget, the multiplier grid and
the lower bound. C is the EVAL cost of the rule and r its EVAL failure rate; V is the empirical frontier of the class on
EVAL, the lower convex hull of the (risk, cost) points of the chain elements computed on EVAL (Algorithm 1) over a dense
grid of multipliers; E(eps) = max(0, max_k (g_k - lambda_k eps)) is the envelope of the empirical Lagrangian minima g_k
at the multipliers of the lower bound (Lemma 3, Theorem 4). Both candidate sequences of the frontier run are split.

Usage:
    python analysis/gap_anatomy.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TERMS = ('cost_bound', 'rule_vs_frontier', 'displacement', 'multiplier_grid', 'lower_bound')


def eval_frontier(S, multipliers, log=lambda *x: None):
    """Lower convex hull of the (risk, cost) points on EVAL of the chain elements computed on EVAL."""
    from cast.chain import chain_element
    from cast.monotone_class import cached_coefs
    cc, rc, bc, br, n = cached_coefs(S, S.EVAL)
    pts = []
    for i, lam in enumerate(multipliers):
        beta, val, f = chain_element(S, S.EVAL, int(lam))
        pts.append((float((br + rc.ravel() @ f.ravel()) / n), float((bc - cc.ravel() @ f.ravel()) / n), int(lam)))
        if i % 200 == 0: log('frontier', i, pts[-1])
    return convex_hull(pts)


def convex_hull(pts):
    pts = sorted(set(pts))
    hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    return hull


def gap_terms(C, U, L, Vr, Vh, ell, eps):
    """The five terms, their shares and the residual of the identity."""
    E = max(0.0, max(x['g_hat'] - x['lam'] * eps for x in ell))
    t = dict(cost_bound=U - C, rule_vs_frontier=C - Vr, displacement=Vr - Vh, multiplier_grid=Vh - E, lower_bound=E - L)
    tot = sum(t.values())
    return dict(envelope=E, terms=t, share={a: x / tot for a, x in t.items()}, U_minus_L=U - L, residual=tot - (U - L))


def anatomy(S, FR, n_multipliers=1600, log=lambda *x: None):
    grid = np.unique(np.rint(np.geomspace(1e2, 1e7, n_multipliers)).astype(np.int64))
    hull = eval_frontier(S, grid, log)
    xs = np.array([h[0] for h in hull]); ys = np.array([h[1] for h in hull])
    V = lambda r: float(np.interp(r, xs, ys))
    ell = FR['ell']
    res = dict(ds=FR.get('ds'), graph=FR.get('graph'), n_eval=int(len(S.EVAL)), hull_points=len(hull), levels={})
    for fam in ('chain', 'mixture'):
        for e_s, r in FR['rules'][fam].items():
            e = float(e_s)
            if not r.get('calibrated'): continue
            C, U, rhat = r['mean_visits'], r['U'], 1 - r['coverage']
            b = FR['bounds'][e_s]; Lv = b['L']
            Vr, Vh = V(rhat), V(e)
            res['levels'][f'{fam}_{e_s}'] = dict(sequence=fam, eps=e, mean_visits=C, U=U, L=Lv, fail_rate=rhat, frontier_at_fail_rate=Vr, frontier_at_eps=Vh,
                                                 empirical_optimum=b.get('empirical_optimum'), ratio=U / Lv if Lv > 0 else None, **gap_terms(C, U, Lv, Vr, Vh, ell, e))
            d = res['levels'][f'{fam}_{e_s}']
            log(fam, e_s, {k: round(v, 1) for k, v in d['terms'].items()}, 'U-L', round(U - Lv, 1), 'residual', round(d['residual'], 6))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the same setting')
    ap.add_argument('--out', required=True)
    ap.add_argument('--multipliers', type=int, default=1600, help='size of the geometric multiplier grid from 1e2 to 1e7 of the EVAL frontier')
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from replay import setting_of_frontier
    FR = json.load(open(a.frontier))
    S = setting_of_frontier(a.red, a.ds, a.graph, FR, seeds=a.seeds)
    res = anatomy(S, FR, a.multipliers, log)
    json.dump(res, open(a.out, 'w'), indent=1); log('done')


if __name__ == '__main__':
    main()

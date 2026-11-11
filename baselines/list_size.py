"""List size: native searches with a calibrated list of ef entries, on the grid of the logger and on a finer grid.

The rule runs the traversal with a list of ef entries and stops when that list is exhausted. In the logged traversal it
stops at the visit count T_ef and fails iff T_ef < X; pipeline/common_target.py calibrates it from the log. This script
runs the same calibration with native searches of that list size (engine method 'list'), compares them with the log
query by query, and refines the grid.

Paper: Section VI-A, "Compared methods" (third group: on the 160,000 CAL and EVAL queries of each main setting, native
runs at the selected list size differ from the log in the visits of at most two queries and the exactness of two; 46 list
sizes from 8,192 down to 10) and "Implementation details" (ef = 662 and 319 on the grid of the logger, 627 and 314 on
the finer grid); the List size rows of Table III are timed at these sizes by timing/.

Steps:
  native  CAL searches for the list sizes of the logger grid from 8,192 downward, each tested at eps = delta = 0.05 on its
          native failure count, first rejection stops; then EVAL searches at the selected size
  fine    list sizes 10 * 1.02^j added to the grid. A search with a list of L entries is a prefix of the search with a
          longer list, so the CAL failure count is nonincreasing in L and the fixed sequence from 8,192 downward is
          decided between the accepted size and the rejected size of the coarser grid: CAL searches for the fine sizes
          in (rejected, accepted] in descending order until the first rejection, and for the two fine sizes just above
          the accepted one (they must pass); then EVAL searches at the selected size
  report  EVAL visits and failures of a native run and the visit ratio to our rule of the common target, with the
          paired bootstrap resamples of pipeline/common_target.py (same seed)

Usage:
  python baselines/list_size.py native --data DATA --work WORK --red RED_DIR --ds amazon --graph A --out NATIVE.npz
  python baselines/list_size.py fine --data DATA --work WORK --red RED_DIR --ds amazon --graph A --accepted 662 --rejected 552 --out FINE.npz
  python baselines/list_size.py report --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --common-target COMMON.json --run NATIVE.npz --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

EPS, DELTA = 0.05, 0.05


def setup(a):
    """Engine context, CAL and EVAL queries with exact answers, and the logged stop times T_L and completion times X."""
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a)
    rows, _ = C.pool_rows(over); rows_cal, rows_ev = rows['CAL'], rows['EVAL']
    TL, _, X, sizes, _ = C.load_stops(a.red, a.ds, a.graph, 'pool', 'X_exact'); grid = [int(x) for x in sizes]
    cal = C.take_pool(a.data, a.ds, a.workload, rows_cal); ev = C.take_pool(a.data, a.ds, a.workload, rows_ev)
    return ctx, grid, TL, X, rows_cal, rows_ev, cal, ev


def native_run(ctx, Q, S, L, threads):
    """Visits and failure flags (returned answer differs from S as a set) of native searches with a list of L entries."""
    r = ctx.run(dict(method='list', k=ctx.k), Q, L=L, threads=threads)
    nv = np.asarray(r['visits']).astype(np.int64); fail = np.array([set(x.tolist()) != set(y.tolist()) for x, y in zip(r['ids'], S)])
    return nv, fail


def native(a):
    from scipy.stats import binom
    ctx, grid, TL, X, rows_cal, rows_ev, (Qc, Sc), (Qe, Se) = setup(a)
    seq, cal_visits, cal_fail, sel = [], {}, {}, None
    for L in sorted(grid, reverse=True):
        nv, fail = native_run(ctx, Qc, Sc, L, a.threads)
        k = int(fail.sum()); ok = bool(binom.cdf(k, len(fail), EPS) <= DELTA); j = grid.index(L)
        # (list size, CAL failures, test passed, queries whose visits differ from T_L, queries whose failure differs from T_L < X)
        seq.append((L, k, int(ok), int((nv != TL[rows_cal, j]).sum()), int((fail != (TL[rows_cal, j] < X[rows_cal])).sum())))
        cal_visits[L] = nv; cal_fail[L] = fail
        print(a.ds, a.graph, 'CAL', seq[-1], flush=True)
        if not ok: break
        sel = L
    assert sel is not None
    nvE, failE = native_run(ctx, Qe, Se, sel, a.threads); j = grid.index(sel)
    print(a.ds, a.graph, 'EVAL L', sel, 'visits', nvE.mean(), 'fail', failE.mean(), 'visit differences to T_L', int((nvE != TL[rows_ev, j]).sum()),
          'failure differences to the log', int((failE != (TL[rows_ev, j] < X[rows_ev])).sum()), flush=True)
    np.savez(a.out, seq=np.array(seq, np.int64), L_sel=sel, eval_visits=nvE, eval_fail=failE, eval_list_stop=TL[rows_ev, j], eval_X=X[rows_ev],
             **{f'cal_visits_{L}': v_ for L, v_ in cal_visits.items()}, **{f'cal_fail_{L}': v_ for L, v_ in cal_fail.items()})


def fine(a):
    from scipy.stats import binom
    ctx, grid, TL, X, rows_cal, rows_ev, (Qc, Sc), (Qe, Se) = setup(a); LACC, LREJ = a.accepted, a.rejected
    fine_grid = sorted({int(round(10 * 1.02 ** j)) for j in range(400) if 10 * 1.02 ** j <= 8192} | set(grid))
    assert LACC in grid and LREJ in grid and LREJ < LACC
    above = [L for L in fine_grid if L > LACC][:2]; between = sorted([L for L in fine_grid if LREJ < L <= LACC], reverse=True)
    nmax = int(binom.ppf(0.05, len(rows_cal), 0.05))
    cal_visits, cal_fail, sel = {}, {}, None
    def cal(L):
        nv, fail = native_run(ctx, Qc, Sc, L, a.threads)
        k = int(fail.sum()); ok = bool(binom.cdf(k, len(fail), EPS) <= DELTA)
        j = grid.index(L) if L in grid else None
        rec = (L, k, int(ok), int((nv != TL[rows_cal, j]).sum()) if j is not None else -1, int((fail != (TL[rows_cal, j] < X[rows_cal])).sum()) if j is not None else -1)
        cal_visits[L] = nv; cal_fail[L] = fail; print(a.ds, a.graph, 'CAL', rec, flush=True); return rec
    mono = [cal(L) for L in above]                                   # the two fine sizes just above the accepted size
    seq = []
    for L in between:
        rec = cal(L); seq.append(rec)
        if not rec[2]: break
        sel = L
    assert sel is not None
    nvE, failE = native_run(ctx, Qe, Se, sel, a.threads)
    print(a.ds, a.graph, 'EVAL L', sel, 'visits', nvE.mean(), 'fail', failE.mean(), flush=True)
    np.savez(a.out, seq=np.array(seq, np.int64), above=np.array(mono, np.int64), L_sel=sel, eval_visits=nvE, eval_fail=failE,
             eval_list_stop=-np.ones(len(nvE), np.int64), eval_X=X[rows_ev], fine_grid=np.array(fine_grid), accepted=LACC, rejected=LREJ, nmax=nmax,
             **{f'cal_visits_{L}': v_ for L, v_ in cal_visits.items()}, **{f'cal_fail_{L}': v_ for L, v_ in cal_fail.items()})


def report(a):
    _, over = C.read_seeds(a.seeds); FR = json.load(open(a.frontier)); CT = json.load(open(a.common_target))
    S = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over); EVAL = S.EVAL
    # Our rule of the common target, the same rule as in pipeline/common_target.py.
    rec, c_ch, f_ch = C.common_target_staircase(S, FR, EPS, DELTA)
    assert str(rec['theta']) == str(CT['table']['cast']['theta']), (rec['theta'], CT['table']['cast']['theta'])
    assert abs(float(c_ch.mean()) - CT['table']['cast']['mean_visits']) < 1e-9
    Rn = np.load(a.run); nvE, fE = Rn['eval_visits'], Rn['eval_fail']
    logged = bool((Rn['eval_list_stop'] >= 0).all())                   # a fine-grid run has no logged stop times to compare with
    assert np.array_equal(Rn['eval_X'], S.X[EVAL])                     # same EVAL order as the reduced arrays
    W = C.bootstrap_weights(len(EVAL), int(CT['bootstrap_seed']))
    ra = (W @ nvE.astype(np.float64)) / (W @ c_ch.astype(np.float64))
    OUT = dict(ds=a.ds, graph=a.graph, eps=EPS, delta=DELTA,
               cal_sequence=[dict(L=int(s[0]), fail=int(s[1]), passed=bool(s[2]), visits_differ_from_log=int(s[3]), failure_differs_from_log=int(s[4])) for s in Rn['seq']],
               list_size=int(Rn['L_sel']), list_size_from_log=CT['table']['list_size'].get('param'),
               eval=dict(mean_visits=float(nvE.mean()), fail=int(fE.sum()), fail_rate=float(fE.mean()), n=int(len(nvE)),
                         visits_differ_from_log=int((nvE != Rn['eval_list_stop']).sum()) if logged else None,
                         failure_differs_from_log=int((fE != (Rn['eval_list_stop'] < Rn['eval_X'])).sum()) if logged else None),
               cast=dict(theta=rec['theta'], mean_visits=float(c_ch.mean()), fail_rate=float(f_ch.mean())),
               visit_ratio=dict(ratio=float(nvE.mean() / c_ch.mean()), lower_bound_one_sided=float(np.quantile(ra, 0.025))), bootstrap_seed=int(CT['bootstrap_seed']))
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); print(json.dumps(OUT)[:1200], flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    for name in ('native', 'fine'):
        p = sub.add_parser(name); C.add_setting_args(p)
        p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload (with the *_stops.npz files)')
        p.add_argument('--seeds', default=None); p.add_argument('--out', required=True, help='output npz')
        if name == 'fine':
            p.add_argument('--accepted', type=int, required=True, help='list size selected by the native step')
            p.add_argument('--rejected', type=int, required=True, help='next smaller size of the logger grid (rejected by the native step)')
    p = sub.add_parser('report')
    p.add_argument('--red', required=True); p.add_argument('--ds', required=True); p.add_argument('--graph', default='A')
    p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting')
    p.add_argument('--common-target', required=True, help='output of pipeline/common_target.py for the setting (bootstrap seed, our rule)')
    p.add_argument('--run', required=True, help='output npz of the native or fine step'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(native=native, fine=fine, report=report)[a.step](a)

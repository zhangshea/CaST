"""Distance-adaptive termination on a fine grid of gamma: stop times from a second logging pass, calibration by the
common fixed-sequence test, and the visit ratio to our rule.

Method: adapted from Y. Al-Jazzazi, H. Diwan, J. Gou, C. Musco, C. Musco and T. Suel, "Distance Adaptive Beam Search for
Provably Accurate Graph-Based Nearest Neighbor Search", Advances in Neural Information Processing Systems 38.
The rule stops the logged traversal at the first pop at which the constrained answer is full and the popped candidate is
farther than (1 + gamma) times the k-th distance of the answer (float64 comparison of the logged float32 distances);
otherwise it ends with the list of 8,192 entries. A query fails iff it stops before its completion time X. A larger
gamma stops later, so the candidates run from the largest gamma down.

Paper: Section VI-A, "Compared methods" (third group; gamma from 2 down to 0), the column "Dist.-adaptive" of Table II
(every row, recall-target rows included), the distance-adaptive curve of Fig. 4 (nine risk levels at delta / (2 M)) and
the equal-failure-rate comparison of Section VI-B.

Gamma grid: the 22 values stored with the reduced arrays plus 0 to --gamma-max in steps of 0.0025: 129 values for the
default 0.3, and 246 values for --gamma-max 0.6, which YFCC-10M uses (Section VI-A, Compared methods).

Steps:
  relog      search the CAL and EVAL queries again with the logger (HOLD queries are never searched) and store, per
             query, the stop of the rule for every gamma; counts the queries whose stops differ from the stored ones
  calibrate  common target eps = delta = 0.05 (EVAL visits, failures, visit ratio to our rule with paired bootstrap;
             output key table.distance_adaptive, in the layout of pipeline/common_target.py) and the nine risk levels at
             delta / (2 M) (EVAL visits over those of the frontier rule of the level; output key levels)
  summary    one line per calibrate output

Usage:
  python baselines/distance_adaptive.py relog --data DATA --work WORK --red RED_DIR --ds amazon --graph A --out-dir FINE_DIR
  python baselines/distance_adaptive.py calibrate --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --fine FINE_DIR --seed SEED --out OUT.json
  python baselines/distance_adaptive.py summary --out SUMMARY.json name=OUT.json [name=OUT.json ...]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

from pipeline.log_trajectories import LIST_SIZES                              # the 46 list sizes of the logger
from pipeline.reduce_logs import GAMMAS as COARSE_GAMMAS, adaptive_stops    # the 22 gammas stored with the reduced arrays
from pipeline.reduce_logs import fine_gammas as gamma_grid                  # these plus 0 to gamma_max in steps of 0.0025, descending

LMAX = 8192
assert len(LIST_SIZES) == 46 and int(LIST_SIZES[-1]) == LMAX


def relog(a):
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); os.makedirs(a.out_dir, exist_ok=True)
    from cast import datasets
    GAMMAS = gamma_grid(a.gamma_max); i22 = [int(np.flatnonzero(GAMMAS == g_)[0]) for g_ in COARSE_GAMMAS]
    P = dict(method='answer', k=ctx.k, L=LMAX); rep = dict(layout=a.layout, gammas=GAMMAS.tolist(), chunks={})
    jobs = []                                # (name, rows, group of queries, rows within the group, stops file)
    if a.layout == 'pool':
        rows, perm = C.pool_rows(over); want = np.zeros(len(perm), bool); want[rows['CAL']] = True; want[rows['EVAL']] = True
        for k in range(20):
            loc = np.flatnonzero(want[k * C.CHUNK:(k + 1) * C.CHUNK])
            jobs.append((f'{a.ds}_{a.graph}_chunk{k}', loc + k * C.CHUNK, f'chunk{k}', loc, f'{a.red}/{a.ds}_{a.graph}_chunk{k}_stops.npz'))
    else:
        for part in ('cal', 'eval'):
            n_ = len(datasets.queries(a.data, a.ds, a.group_prefix + part))
            for s in range(0, n_, a.chunk):
                loc = np.arange(s, min(s + a.chunk, n_))
                jobs.append((f'{a.ds}_{a.graph}_{part}_{s // a.chunk:02d}', loc, a.group_prefix + part, loc, f'{a.red}/{a.ds}_{a.graph}_{part}_stops.npz'))
    for name, rows_, group, loc, exf in jobs:
        fo = os.path.join(a.out_dir, f'adaptive_{name}.npz')
        if os.path.exists(fo): continue
        t = time.time(); Q = np.ascontiguousarray(datasets.queries(a.data, a.ds, group)[loc], np.float32)
        S = np.ascontiguousarray(datasets.exact_answers(a.data, a.ds, group, a.workload)[loc], np.int64)
        r = eng.log_search(ctx.G, ctx.C, Q, S, P, LMAX, LIST_SIZES, threads=a.threads); n = len(loc)
        ex = np.load(exf); d = adaptive_stops(r, ctx.k, GAMMAS)
        # Agreement with the reduced arrays: the list stops T_L, and the stops on the 22 coarse gammas.
        tl_mis = int((r['list_stop'] != ex['list_stop'][loc]).any(axis=1).sum()); dat_mis = int((d[:, i22] != ex['adaptive_stop'][loc]).any(axis=1).sum())
        np.savez(fo, rows=rows_, adaptive_stop=d, gammas=GAMMAS, final_visits=r['final_visits'])
        rep['chunks'][name] = dict(n=n, list_stop_mismatch=tl_mis, coarse_gamma_mismatch=dat_mis, secs=round(time.time() - t, 1)); print(name, rep['chunks'][name], flush=True)
        json.dump(rep, open(os.path.join(a.out_dir, f'report_{a.ds}_{a.graph}.json'), 'w'), indent=1); del r
    print('relog done', flush=True)


def load_fine(fine_dir, red_dir, ds, graph, layout, n_rows):
    """Fine-grid stops in the row order of the setting's files; -1 where a query was not searched (FIT, SEL, HOLD)."""
    from cast.setting import PARTS
    if layout == 'pool':
        files = [(os.path.join(fine_dir, f'adaptive_{ds}_{graph}_chunk{k}.npz'), 0) for k in range(20)]
    else:
        offs, o = {}, 0
        for p_ in PARTS: offs[p_] = o; o += len(np.load(f'{red_dir}/{ds}_{graph}_{p_}_stops.npz')['list_stop'])
        assert o == n_rows
        files = [(f, offs[os.path.basename(f)[:-4].split('_')[-2]]) for f in sorted(glob.glob(os.path.join(fine_dir, f'adaptive_{ds}_{graph}_*.npz')))]
    GAMf, DATf, seen = None, None, np.zeros(n_rows, bool)
    for f, off in files:
        z = np.load(f)
        if GAMf is None: GAMf = z['gammas']; DATf = np.full((n_rows, len(GAMf)), -1, np.int64)
        assert np.array_equal(z['gammas'], GAMf); r = off + z['rows']; DATf[r] = z['adaptive_stop']; seen[r] = True
    return GAMf, DATf, seen


def calibrate(a):
    _, over = C.read_seeds(a.seeds); FR = json.load(open(a.frontier)); layout, target = FR['layout'], FR['target']
    S = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over); eps, lvl = a.eps, a.delta
    rec, c_ch, f_ch = C.common_target_staircase(S, FR, eps, lvl)
    _, DATc, Xf, _, GAMc = C.load_stops(a.red, a.ds, a.graph, layout, target); PART = C.sample_rows(S); ev = PART['EVAL']
    for k, idx in (('SEL', S.SEL), ('CAL', S.CAL), ('EVAL', S.EVAL)): assert np.array_equal(Xf[PART[k]], S.X[idx]), k    # the two row orders agree
    W = C.bootstrap_weights(len(ev), a.seed)
    OUT = dict(ds=a.ds, graph=a.graph, layout=layout, target=target, eps=eps, delta=lvl, bootstrap_seed=a.seed, table=dict(cast=dict(theta=rec['theta'], **C.summarize(c_ch, f_ch))))
    GAMf, DATf, seen = load_fine(a.fine, a.red, a.ds, a.graph, layout, len(Xf))
    assert seen[PART['CAL']].all() and seen[PART['EVAL']].all() and np.all(np.diff(GAMf) < 0)
    i22 = [int(np.flatnonzero(GAMf == x)[0]) for x in GAMc]
    assert all(np.array_equal(DATf[PART[p], i], DATc[PART[p], t]) for p in ('CAL', 'EVAL') for t, i in enumerate(i22))   # equal on the coarse gammas
    from cast.calibration import fixed_sequence_deterministic
    dat_cf = lambda t, rows: (DATf[rows, t], DATf[rows, t] < Xf[rows])
    SEQ = list(range(len(GAMf))); OUT['gammas'] = GAMf.tolist()
    # ---- common target: eps at the single level delta
    t2, tr = fixed_sequence_deterministic(SEQ, dat_cf, PART['CAL'], eps, lvl)
    if t2 is None: rule = dict(calibrated=False)
    else:
        ce, fe = dat_cf(t2, ev)
        rule = dict(calibrated=True, param=float(GAMf[t2]), **C.summarize(ce, fe), visit_ratio=C.visit_ratio(W, ce, c_ch, with_se=True),
                    trace_tail=[(float(GAMf[t]), k_, ok) for t, k_, ok in tr[-3:]])
    OUT['table']['distance_adaptive'] = rule                                   # the layout of pipeline/common_target.py
    # ---- nine risk levels at delta / (2 M), relative to the frontier rule of the level
    E = FR['config']['levels']; M = len(E); OUT['levels'] = {}
    for e in E:
        ch = FR['rules']['chain'].get(str(e)); row = dict(frontier_mean_visits=(ch or {}).get('mean_visits'))
        p, _ = fixed_sequence_deterministic(SEQ, dat_cf, PART['CAL'], e, lvl / (2 * M))
        if p is None: row.update(calibrated=False); OUT['levels'][str(e)] = row; continue
        c, f_ = dat_cf(p, ev); row.update(calibrated=True, param=float(GAMf[p]), **C.summarize(c, f_))
        if row['frontier_mean_visits']: row['visits_over_frontier'] = float(c.mean() / row['frontier_mean_visits'])
        OUT['levels'][str(e)] = row
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in OUT.items() if k not in ('levels', 'gammas')}, default=str)[:2500])
    print({e: (r.get('param'), round(r.get('visits_over_frontier', 0), 4)) for e, r in OUT['levels'].items()})


def summary(a):
    res = {}
    for arg in a.inputs:
        name, path = arg.split('=', 1); d = json.load(open(path)); f, st = d['table']['distance_adaptive'], d['table']['cast']
        r = dict(gamma=f['param'], mean_visits=f['mean_visits'], fail_rate=f['fail_rate'], ratio=f['visit_ratio']['ratio'], lower_bound_one_sided=f['visit_ratio']['lower_bound_one_sided'],
                 cast_mean_visits=st['mean_visits'], levels={e: v.get('visits_over_frontier') for e, v in d.get('levels', {}).items()})
        res[name] = r
        print(f"{name:24s} gamma {r['gamma']:.4f} visits {r['mean_visits']:9.1f} fail {100 * r['fail_rate']:.2f}% ratio {r['ratio']:.3f} (lower bound {r['lower_bound_one_sided']:.3f})")
    json.dump(res, open(a.out, 'w'), indent=1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('relog'); C.add_setting_args(p)
    p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload (with the *_stops.npz files)')
    p.add_argument('--layout', default='pool', choices=('pool', 'parts')); p.add_argument('--chunk', type=int, default=10000, help="layout 'parts': queries per output file")
    p.add_argument('--group-prefix', default='', help="layout 'parts': prefix of the query groups in DATA (for example title_ or official_)")
    p.add_argument('--gamma-max', type=float, default=0.3, help='fine steps of 0.0025 up to this gamma (0.6 for YFCC-10M)')
    p.add_argument('--seeds', default=None); p.add_argument('--out-dir', required=True)
    p = sub.add_parser('calibrate')
    p.add_argument('--red', required=True); p.add_argument('--ds', required=True); p.add_argument('--graph', default='A')
    p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting (layout, target, interface, chain grid, frontier rules)')
    p.add_argument('--fine', required=True, help='output directory of the relog step')
    p.add_argument('--eps', type=float, default=0.05); p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    p = sub.add_parser('summary'); p.add_argument('--out', required=True); p.add_argument('inputs', nargs='+', help='name=OUT.json of the calibrate step')
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(relog=relog, calibrate=calibrate, summary=summary)[a.step](a)

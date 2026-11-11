"""Patience: stop once the answer is full and unchanged for a calibrated number of visits.

Method: T. Teofili and J. Lin, "Patience in Proximity: A Simple Early Termination Strategy for HNSW Graph Traversal in
Approximate k-Nearest Neighbor Search", Advances in Information Retrieval (ECIR), LNCS 15574. Applied per expansion of the
logged traversal: the rule with parameter Delta stops at the first expansion e, before the horizon expansion, at which
the answer holds k items and v_e - (visit count of the last answer change) >= Delta, else at the horizon expansion (the
first with at least 32,768 visits, or the last one). Its cost is the visit count v_e, and it fails iff some member of the
exact answer S has not been visited by then. Candidates: the 80 values of Delta on a geometric grid of ratio 1.1 from 16
that do not exceed 32,768, the largest first.

Paper: Section VI-A, "Compared methods" (third group), the Patience rows of Table III (the rule timed by timing/ at the
calibrated Delta), and Section VI-B (Patience on all EVAL queries at the common target; Patience swept over its
parameter at equal failure rates).

Steps:
  relog      search the CAL and EVAL queries again with the logger (HOLD queries are never searched) and store, per
             query and Delta, the stop and the failure flag
  calibrate  Delta from the largest value downward on CAL at eps = delta = 0.05, EVAL visits and failures, visit ratio to our rule
  points     (EVAL failure rate, mean EVAL visits) for each of the 80 values of Delta, no calibration

Usage:
  python baselines/patience.py relog --data DATA --work WORK --red RED_DIR --ds amazon --graph A --out-dir PATIENCE_DIR
  python baselines/patience.py calibrate --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --stops PATIENCE_DIR --seed SEED --out OUT.json
  python baselines/patience.py points --stops PATIENCE_DIR --out POINTS.json [--settings amazon:A amazon:B arxiv:A arxiv:B]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

LMAX, HVIS = 8192, 32768
DELTAS = np.unique(np.round(16 * 1.1 ** np.arange(0, 120)).astype(np.int64)); DELTAS = DELTAS[DELTAS <= HVIS]
N_POOL, N_CHUNKS = 200000, 20


def relog(a):
    from baselines.distance_adaptive import LIST_SIZES
    from cast import datasets
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); os.makedirs(a.out_dir, exist_ok=True); K = ctx.k
    rows, perm = C.pool_rows(over); want = np.zeros(N_POOL, bool); want[rows['CAL']] = True; want[rows['EVAL']] = True
    P = dict(method='answer', k=K, L=LMAX); rep = dict(ds=a.ds, graph=a.graph, deltas=DELTAS.tolist(), chunks={})
    for k_ in range(N_CHUNKS):
        fo = f'{a.out_dir}/patience_{a.ds}_{a.graph}_chunk{k_}.npz'
        if os.path.exists(fo): continue
        t = time.time(); loc = np.flatnonzero(want[k_ * C.CHUNK:(k_ + 1) * C.CHUNK]); rows_ = loc + k_ * C.CHUNK
        Q = np.ascontiguousarray(datasets.queries(a.data, a.ds, f'chunk{k_}')[loc]); S = np.ascontiguousarray(datasets.exact_answers(a.data, a.ds, f'chunk{k_}', a.workload)[loc].astype(np.int64))
        r = eng.log_search(ctx.G, ctx.C, Q, S, P, LMAX, LIST_SIZES, threads=a.threads)
        ex = np.load(f'{a.red}/{a.ds}_{a.graph}_chunk{k_}_stops.npz'); assert [int(x) for x in ex['list_sizes']] == LIST_SIZES.tolist()
        tl_mis = int((r['list_stop'] != ex['list_stop'][loc]).any(axis=1).sum())             # agreement with the reduced arrays
        n = len(loc); pv = np.zeros((n, len(DELTAS)), np.int32); pf = np.zeros((n, len(DELTAS)), bool)
        rs = r['visit_of_member']; valid = S >= 0
        for i in range(n):
            s_, b = r['ptr'][i], r['ptr'][i + 1]; nv = r['visits'][s_:b].astype(np.int64); chg = r['chg'][s_:b].astype(np.int64); full = r['answer_size'][s_:b] >= K
            eh = int(np.searchsorted(nv, HVIS, side='left')); eh = min(eh, len(nv) - 1)            # horizon expansion
            w = np.where(full[:eh + 1], nv[:eh + 1] - chg[:eh + 1], -1); M = np.maximum.accumulate(w)
            e = np.searchsorted(M, DELTAS, side='left'); e = np.where(e > eh, eh, e)
            pv[i] = nv[e]
            ri = rs[i][valid[i]]; X = np.iinfo(np.int32).max if (ri == 0).any() else int(ri.max())   # visit at which the last member of S arrives
            pf[i] = pv[i] < X
        np.savez(fo, rows=rows_, deltas=DELTAS, final_visits=r['final_visits'], patience_stop=pv, patience_fail=pf)
        rep['chunks'][k_] = dict(n=n, list_stop_mismatch=tl_mis, secs=round(time.time() - t, 1))
        print(a.ds, a.graph, k_, rep['chunks'][k_], flush=True); del r
        json.dump(rep, open(f'{a.out_dir}/patience_{a.ds}_{a.graph}.json', 'w'), indent=1)
    print('relog done', a.ds, a.graph, flush=True)


def load_stops(stops_dir, ds, graph):
    PV = np.zeros((N_POOL, len(DELTAS)), np.int64); PF = np.zeros((N_POOL, len(DELTAS)), bool); seen = np.zeros(N_POOL, bool)
    for k in range(N_CHUNKS):
        z = np.load(os.path.join(stops_dir, f'patience_{ds}_{graph}_chunk{k}.npz')); PV[z['rows']] = z['patience_stop']; PF[z['rows']] = z['patience_fail']; seen[z['rows']] = True
    return PV, PF, seen


def calibrate(a):
    from cast.calibration import fixed_sequence_deterministic
    _, over = C.read_seeds(a.seeds); FR = json.load(open(a.frontier)); assert FR['layout'] == 'pool'
    S = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over); eps, lvl = a.eps, a.delta
    rec, c_ch, f_ch = C.common_target_staircase(S, FR, eps, lvl); PART = C.sample_rows(S); ev = PART['EVAL']
    W = C.bootstrap_weights(len(ev), a.seed)
    OUT = dict(ds=a.ds, graph=a.graph, eps=eps, delta=lvl, cast=dict(theta=rec['theta'], **C.summarize(c_ch, f_ch)), bootstrap_seed=a.seed)
    PV, PF, seen = load_stops(a.stops, a.ds, a.graph); assert seen[PART['CAL']].all() and seen[PART['EVAL']].all()
    di = {int(d): i for i, d in enumerate(DELTAS)}; seq = [int(x) for x in DELTAS[::-1]]
    pat_cf = lambda dl, rows: (PV[rows, di[dl]], PF[rows, di[dl]])
    p2, tr2 = fixed_sequence_deterministic(seq, pat_cf, PART['CAL'], eps, lvl)
    if p2 is None: OUT['patience'] = dict(calibrated=False)
    else:
        c, f = pat_cf(p2, ev); OUT['patience'] = dict(calibrated=True, param=p2, **C.summarize(c, f), visit_ratio=C.visit_ratio(W, c, c_ch, with_se=True), trace_tail=tr2[-3:])
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'trace_tail'}) for k, v in OUT.items()}, default=str)[:3000])


def points(a):
    rows, _ = C.pool_rows(C.read_seeds(a.seeds)[1]); is_eval = np.zeros(N_POOL, bool); is_eval[rows['EVAL']] = True
    res = dict(deltas=DELTAS.tolist())
    for s in a.settings:
        ds, g = s.split(':')
        sv = np.zeros(len(DELTAS)); sf = np.zeros(len(DELTAS)); n = 0; seen = np.zeros(N_POOL, bool)
        for k in range(N_CHUNKS):
            z = np.load(os.path.join(a.stops, f'patience_{ds}_{g}_chunk{k}.npz')); r = z['rows']; assert z['patience_stop'].shape[1] == len(DELTAS)
            assert not seen[r].any(); seen[r] = True; m = is_eval[r]
            sv += z['patience_stop'][m].astype(np.float64).sum(0); sf += z['patience_fail'][m].sum(0); n += int(m.sum())
        assert n == int(is_eval.sum()) and seen[is_eval].all(), (ds, g, n)
        res[f'{ds}_{g}'] = dict(n=n, mean_visits=(sv / n).tolist(), fail_rate=(sf / n).tolist())
        print(ds, g, 'n', n, 'failure rates from', float((sf / n).min()), 'to', float((sf / n).max()), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('relog'); C.add_setting_args(p)
    p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload (with the *_stops.npz files)')
    p.add_argument('--seeds', default=None); p.add_argument('--out-dir', required=True)
    p = sub.add_parser('calibrate')
    p.add_argument('--red', required=True); p.add_argument('--ds', required=True); p.add_argument('--graph', default='A')
    p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting')
    p.add_argument('--stops', required=True, help='output directory of the relog step')
    p.add_argument('--eps', type=float, default=0.05); p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    p = sub.add_parser('points')
    p.add_argument('--stops', required=True, help='output directory of the relog step')
    p.add_argument('--settings', nargs='+', default=['amazon:A', 'amazon:B', 'arxiv:A', 'arxiv:B'], help='dataset:graph of each setting')
    p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(relog=relog, calibrate=calibrate, points=points)[a.step](a)

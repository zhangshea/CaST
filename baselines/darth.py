"""DARTH in our engine: training log, recall regressor, adaptive prediction intervals, and the calibration of its recall
target by the common fixed-sequence test.

Method: M. Chatzakis, Y. Papakonstantinou and T. Palpanas, "DARTH: Declarative Recall Through Early Termination for
Approximate Nearest Neighbor Search", Proceedings of the ACM on Management of Data 3(4). The rule is implemented in
the engine of this package (rule types 6 and 7 of cast_engine.stop_search):
  features   eleven features of the current answer: expansions, distance computations, accepted insertions, distance of
             the entry point of the bottom layer, nearest and furthest answer distance, their mean and variance, and the
             25th, 50th and 75th percentile of the answer distances;
  model      a gradient-boosted regressor of the current recall (100 trees, learning rate 0.1, 31 leaves, at least 20
             rows per leaf, no early stopping), trained on rows logged every 16 distance computations once the answer is full;
  intervals  a prediction after pi distance computations, pi = mpi + (ipi - mpi) (Rt - Rp) with ipi = d_Rt / 2 and
             mpi = d_Rt / 10, d_Rt the mean number of distance computations at which a training query first reaches recall
             Rt; the predicted recall Rp is clipped to [0, 1] and the search stops, inside an expansion, once Rp >= Rt.
The answer is the constrained answer of the traversal and recall is its overlap with the exact answer S. A query fails
when the returned answer differs from S, and the recall target Rt is the parameter calibrated on CAL.

Paper: Section VI-A, "Compared methods" (second group; Rt on a grid of step 1e-4, calibrated to 0.9942 and 0.9939),
the DARTH rows of Table III (the rule timed by timing/), Fig. 4 (DARTH at the nine risk levels, each calibrated at
delta / (2 M)) and the equal-failure-rate comparison of Section VI-B.

Steps:
  train      log the first 10,000 SEL queries, fit the regressor, write the flattened forest and the interval schedule
  calibrate  Rt = 1.0000, 0.9999, ... on the 40,000 CAL queries at (eps, delta), first rejection stops
  levels     one CAL sweep over Rt read by the test of every risk level at delta / (2 M), then EVAL runs at every level
  report     pair the EVAL runs with the frontier rule of each level (visit ratio with a paired bootstrap)

Usage:
  python baselines/darth.py train --data DATA --work WORK --ds amazon
  python baselines/darth.py calibrate --data DATA --work WORK --ds amazon --out WORK/results/darth_amazon_A.json
  python baselines/darth.py levels --data DATA --work WORK --ds amazon --frontier FRONTIER.json --out WORK/results/darth_levels_amazon_A
  python baselines/darth.py report --red RED_DIR --ds amazon --frontier FRONTIER.json --run WORK/results/darth_levels_amazon_A --seed SEED --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

TYPE_DARTH, TYPE_DARTH_LOG = 6, 7           # rule types of cast_engine.stop_search
N_TRAIN = 10000                             # training queries: the first 10,000 SEL queries
LOG_INTERVAL = 16                           # one training row every 16 distance computations once the answer is full
N_FEATURES = 11
# Recall targets of the interval schedule: 1.000 down to 0.850 by 0.001.
SCHEDULE_TARGETS = np.round(np.arange(1.0, 0.8495, -0.001), 3)
# Candidates of the calibration at the common target, conservative first: 1.0000 down to 0.9900 by 0.0001.
CALIBRATION_TARGETS = np.round(np.arange(1.0, 0.98995, -0.0001), 4)

t0 = time.time()
def log(*a): print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


def search_params(ctx):
    """Parameters of the traversal that maintains the constrained answer, with the list of the logged traversal."""
    from cast.grid import LIST_SIZE
    return dict(method='answer', k=ctx.k, L=LIST_SIZE), LIST_SIZE


def base_rule(rule_type, **kw):
    """Rule record of the engine. DARTH reads neither the checkpoints nor the bucket edges; the search ends with its list."""
    from cast.grid import GRID
    return dict(type=rule_type, grid=GRID.astype(np.int32), edges=np.zeros(1), **kw)


def darth_rule(Rt, sc):
    return base_rule(TYPE_DARTH, Rt=float(Rt), ipi=int(sc['ipi']), mpi=int(sc['mpi']))


def schedule_entry(schedule, Rt):
    """Intervals of the target Rt: recall is a multiple of 1 / k, so d_Rt is that of ceil(Rt k) / k (1.0 where it is not tabulated)."""
    key = f'{np.ceil(round(Rt * 10, 6)) / 10:.1f}'
    return schedule[key] if key in schedule else schedule['1.0']


def first_reach_schedule(X, y, ptr, targets, key=str):
    """For each recall target: d_Rt, the mean distance computations (feature 1) at which a training query first reaches it."""
    first = {}
    for Rt in targets:
        vals = []
        for i in range(len(ptr) - 1):
            r = y[ptr[i]:ptr[i + 1]]; w = np.nonzero(r >= Rt - 1e-12)[0]
            if len(w): vals.append(X[ptr[i] + w[0], 1])
        dRt = float(np.mean(vals)); first[key(Rt)] = dict(dists_Rt=dRt, ipi=int(dRt / 2), mpi=int(dRt / 10), reached=len(vals))
    return first


def model_prefix(a):
    return a.model or os.path.join(a.work, 'models', f'darth_{a.ds}_{a.graph}')


def load_model(prefix):
    """(flattened regressor forest, interval schedule) written by the train step."""
    Fz = dict(np.load(prefix + '.npz')); Mj = json.load(open(prefix + '.json'))
    F = {f: Fz[f'reg_{f}'] for f in C.FOREST_KEYS}; F['base'] = Mj['base']
    return F, Mj['schedule']


def train(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from cast.forest import export_forest
    seeds, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); P, Lmax = search_params(ctx)
    rows, _ = C.pool_rows(over); Qt, St = C.take_pool(a.data, a.ds, a.workload, rows['SEL'][:N_TRAIN])
    o = eng.stop_search(ctx.G, ctx.C, Qt, P, Lmax, base_rule(TYPE_DARTH_LOG, li=LOG_INTERVAL), None, None, a.threads, False, False, St)
    drows, ptr = o['drows'], o['dptr']; X, y = drows[:, :N_FEATURES], drows[:, N_FEATURES]
    log('log rows', drows.shape, 'queries with rows', int((np.diff(ptr) > 0).sum()))
    mdl = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False,
                                        random_state=seeds['darth_regressor']).fit(X, y)
    F = export_forest(mdl)
    # The engine evaluates the flattened forest as scikit-learn evaluates the model (largest difference on up to 200,000 logged rows).
    sub = np.random.default_rng(seeds['darth_forest_check']).choice(len(X), min(200000, len(X)), replace=False); Xs = np.ascontiguousarray(X[sub])
    diff = float(np.abs(eng.forest_raw(F, Xs) - mdl.predict(Xs)).max()); log('forest, engine against scikit-learn: max abs difference', diff)
    first = first_reach_schedule(X, y, ptr, SCHEDULE_TARGETS)
    prefix = model_prefix(a); os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.savez(prefix + '.npz', **{f'reg_{f}': v for f, v in F.items() if f != 'base'})
    json.dump(dict(ds=a.ds, graph=a.graph, base=F['base'], forest_max_abs_diff=diff, n_rows=int(len(X)), n_queries=int(len(ptr) - 1), log_interval=LOG_INTERVAL,
                   fit=dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20), mean_recall_rows=float(y.mean()), schedule=first),
              open(prefix + '.json', 'w'), indent=1)
    log('d_Rt at 1.0 / 0.99 / 0.95:', first['1.0'], first['0.99'], first['0.95']); log('done')


def calibrate(a):
    from scipy.stats import binom
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); P, Lmax = search_params(ctx); F, sch = load_model(model_prefix(a))
    rows, _ = C.pool_rows(over); Qc, Sc = C.take_pool(a.data, a.ds, a.workload, rows['CAL']); n = len(Qc); eps, delta = a.eps, a.delta
    assert CALIBRATION_TARGETS.min() > 0.9                             # d_Rt is the same for every Rt in (0.9, 1]
    trace, prev = [], None
    for Rt in CALIBRATION_TARGETS:
        o = eng.stop_search(ctx.G, ctx.C, Qc, P, Lmax, darth_rule(float(Rt), sch['1.0']), None, F, a.threads, False, False)
        fail = int((~C.same_answer(o['ids'], Sc)).sum()); ok = bool(binom.cdf(fail, n, eps) <= delta)
        trace.append(dict(Rt=float(Rt), fail=fail, ok=ok, cal_mean_visits=float(o['visits'].mean()))); log('Rt', Rt, trace[-1])
        if not ok: break
        prev = float(Rt)
    json.dump(dict(ds=a.ds, graph=a.graph, eps=eps, delta=delta, n=n, Rt_calibrated=prev, trace=trace), open(a.out, 'w'), indent=1); log('done')


def levels(a):
    """One sweep of CAL searches over Rt = 1.0000, 0.9999, ...: the rule is deterministic, so the failure trace does not
    depend on eps, and the fixed-sequence test of every level reads the same trace with its own threshold and stops at
    its first rejection. The sweep ends when the tests of the largest level and of the common target have rejected, or
    at Rt = 0.9 (the schedule used holds on (0.9, 1] only). Then EVAL searches at the Rt of every level."""
    from scipy.stats import binom
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); P, Lmax = search_params(ctx); F, sch = load_model(model_prefix(a)); sch = sch['1.0']
    rule = lambda Rt: darth_rule(Rt, sch)
    rows, _ = C.pool_rows(over); Qc, Sc = C.take_pool(a.data, a.ds, a.workload, rows['CAL']); n = len(Qc)
    E = [float(e) for e in json.load(open(a.frontier))['config']['levels']]; M = len(E); LEV = a.delta / (2 * M)
    OUT = dict(ds=a.ds, graph=a.graph, n_cal=n, levels=E, M=M, level=LEV, schedule=sch)
    act = {e: True for e in E}; last = {e: None for e in E}; act_d, last_d = True, None; trace = []; reached_09 = False
    for i in range(10001):
        Rt = round(1.0 - 1e-4 * i, 4)
        if Rt <= 0.9: reached_09 = True; break
        o = eng.stop_search(ctx.G, ctx.C, Qc, P, Lmax, rule(Rt), None, F, a.threads, False, False)
        fail = int((~C.same_answer(o['ids'], Sc)).sum()); row = dict(Rt=Rt, fail=fail, cal_mean_visits=float(o['visits'].mean()))
        for e in E:
            if act[e]:
                if binom.cdf(fail, n, e) <= LEV: last[e] = Rt
                else: act[e] = False
        if act_d:                                                    # the sequence of the common target (eps = 0.05 at level delta)
            if binom.cdf(fail, n, 0.05) <= a.delta: last_d = Rt
            else: act_d = False
        trace.append(row); log('Rt', Rt, row, 'active', [e for e in E if act[e]])
        if not act[max(E)] and not act_d: break
    OUT.update(trace=trace, reached_0_9=reached_09, Rt=dict((str(e), last[e]) for e in E), Rt_common_target=last_d,
               cal_fail_at_Rt={str(e): (next(t['fail'] for t in trace if t['Rt'] == last[e]) if last[e] is not None else None) for e in E})
    json.dump(OUT, open(a.out + '.json', 'w'), indent=1)
    Qe, Se = C.take_pool(a.data, a.ds, a.workload, rows['EVAL'])
    EV = {}; OUT['eval'] = {}
    for e in E:
        if last[e] is None: OUT['eval'][str(e)] = 'no rule'; continue
        o = eng.stop_search(ctx.G, ctx.C, Qe, P, Lmax, rule(last[e]), None, F, a.threads, False, False)
        vis = o['visits'].astype(np.int64); fl = ~C.same_answer(o['ids'], Se); EV[f'visits_{e}'] = vis; EV[f'fail_{e}'] = fl
        OUT['eval'][str(e)] = dict(Rt=last[e], mean_visits=float(vis.mean()), fail=int(fl.sum()), fail_rate=float(fl.mean()))
        if e == 0.05:                                                # the same run twice gives the same visits
            o2 = eng.stop_search(ctx.G, ctx.C, Qe, P, Lmax, rule(last[e]), None, F, a.threads, False, False); OUT['determinism'] = float((o2['visits'] == o['visits']).mean())
        log('EVAL', e, OUT['eval'][str(e)]); json.dump(OUT, open(a.out + '.json', 'w'), indent=1)
    np.savez(a.out + '_eval.npz', **EV); json.dump(OUT, open(a.out + '.json', 'w'), indent=1); log('done')


def report(a):
    _, over = C.read_seeds(a.seeds)
    C.levels_report(a.red, a.ds, a.graph, json.load(open(a.frontier)), a.run, a.out, a.seed, seeds=over, log=log); log('done')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    for name in ('train', 'calibrate', 'levels'):
        p = sub.add_parser(name); C.add_setting_args(p)
        p.add_argument('--model', default=None, help='prefix of the model files (default WORK/models/darth_<ds>_<graph>)')
        p.add_argument('--seeds', default=None, help='JSON file overriding recorded seeds')
        if name == 'calibrate':
            p.add_argument('--eps', type=float, default=0.05); p.add_argument('--delta', type=float, default=0.05)
            p.add_argument('--out', required=True, help='output JSON')
        if name == 'levels':
            p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py (its risk levels are used)')
            p.add_argument('--delta', type=float, default=0.05); p.add_argument('--out', required=True, help='output prefix (.json and _eval.npz)')
    p = sub.add_parser('report')
    p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload'); p.add_argument('--ds', required=True)
    p.add_argument('--graph', default='A'); p.add_argument('--frontier', required=True); p.add_argument('--run', required=True, help='output prefix of the levels step')
    p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(train=train, calibrate=calibrate, levels=levels, report=report)[a.step](a)

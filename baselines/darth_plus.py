"""DARTH+ in our engine, implemented from its paper: recall regressor and lower-quantile regressor on the answer features
and eight query statistics, the choice of the quantile level, and the calibration of the recall target by the common
fixed-sequence test.

Method: M. Chatzakis, Y. Papakonstantinou and T. Palpanas, "DARTH+: Approximate Nearest Neighbor Search with Declarative
Recall and Quality Guarantees", HAL preprint hal-05566027. As described there, the rule adds to the eleven answer features of DARTH eight
statistics of the query vector (mean, median, std, minimum, maximum, range, L1 and L2 norm of its
coordinates, computed once per query), fits a recall regressor and a lower-quantile regressor on the same 19 features
(the settings of DARTH; quantile loss for the second), keeps the adaptive prediction interval of DARTH, and alternates:
the regressor is used until its prediction first reaches the target Rt, then the quantile model only, and the search
stops once this lower bound reaches Rt (rule type 8 of cast_engine.stop_search; the interval after the switch follows
the quantile prediction). No training acceleration of the paper is used. Training rows: the log of the DARTH port
(first 10,000 SEL queries, every 16 distance computations once the answer is full) with the eight statistics appended.

Paper: Section VI-A, "Compared methods" (second group: quantile level chosen among 0.05, 0.1 and 0.2 on the SEL queries
not used in training; the engine evaluates the fitted forests as scikit-learn does to within 1e-9), the DARTH+ rows of
Table III (the rule timed by timing/), Fig. 4 (DARTH+ at the nine risk levels), and the sentences of Section VI-B on
DARTH+ at equal failure rates and on all EVAL queries (visit ratio to our rule at eps = delta = 0.05).

Steps:
  train    log the first 10,000 SEL queries with the query statistics, fit the regressor and the three quantile models
  select   on SEL queries 10,000 to 19,999: per quantile level, Rt calibrated at (0.05, delta); fewest visits wins (ties: 0.10)
  levels   one CAL sweep over Rt read by the test of every risk level at delta / (2 M) and by the common target
           (0.05, delta); EVAL runs at every level and at the common target
  report   pair the EVAL runs with the frontier rule of each level (as baselines/darth.py report)
  compare  our rule and DARTH online on all EVAL queries at the common target, paired with DARTH+ (run it with the numpy
           version that reduced the logs: it builds the logarithm tables of the online features)

Usage:
  python baselines/darth_plus.py train --data DATA --work WORK --ds amazon
  python baselines/darth_plus.py select --data DATA --work WORK --ds amazon
  python baselines/darth_plus.py levels --data DATA --work WORK --ds amazon --frontier FRONTIER.json --out WORK/results/darthplus_levels_amazon_A
  python baselines/darth_plus.py report --red RED_DIR --ds amazon --frontier FRONTIER.json --run WORK/results/darthplus_levels_amazon_A --seed SEED --out OUT.json
  python baselines/darth_plus.py compare --data DATA --work WORK --red RED_DIR --ds amazon --frontier FRONTIER.json \
      --darth WORK/results/darth_amazon_A.json --run WORK/results/darthplus_levels_amazon_A --seed SEED --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C
from baselines import darth as D

TYPE_DARTH_PLUS = 8                         # rule type of cast_engine.stop_search
QUANTILE_LEVELS = (0.05, 0.10, 0.20)
N_FEATURES = 19
DELTA = 0.05
# Candidates of the recall target, conservative first: (0.9, 1] by 0.0001, (0.8, 0.9] by 0.001, [0.5, 0.8] by 0.01.
RT_SEQ = [round(1.0 - 1e-4 * i, 4) for i in range(1000)] + [round(0.9 - 1e-3 * i, 3) for i in range(100)] + [round(0.8 - 1e-2 * i, 2) for i in range(31)]
assert RT_SEQ[999] == 0.9001 and RT_SEQ[1000] == 0.9 and RT_SEQ[1099] == 0.801 and RT_SEQ[1100] == 0.8 and RT_SEQ[-1] == 0.5 and len(set(RT_SEQ)) == len(RT_SEQ)

log = D.log


def query_statistics(Q):
    """The eight query statistics in float64 (reference for the values the engine computes)."""
    Q = Q.astype(np.float64); mn, mx = Q.min(1), Q.max(1)
    return np.stack([Q.mean(1), np.median(Q, 1), Q.std(1), mn, mx, mx - mn, np.abs(Q).sum(1), np.sqrt((Q * Q).sum(1))], 1)


def schedule_entry(first, Rt):
    """Intervals of the target Rt: recall is a multiple of 1 / k, so d_Rt is that of ceil(Rt k) / k."""
    key = f'{np.ceil(round(Rt * 10, 6)) / 10:.1f}'
    return first[key]


def darth_plus_rule(Rt, sc, qF):
    return D.base_rule(TYPE_DARTH_PLUS, Rt=float(Rt), ipi=int(sc['ipi']), mpi=int(sc['mpi']), qforest=qF)


def model_prefix(a):
    return a.model or os.path.join(a.work, 'models', f'darthplus_{a.ds}_{a.graph}')


def load_models(prefix):
    """(training record with the interval schedule, regressor forest, {quantile level: quantile forest})."""
    TJ = json.load(open(prefix + '.json'))
    return TJ, C.load_forest(prefix + '_reg.npz'), {lq: C.load_forest(f'{prefix}_q{lq}.npz') for lq in QUANTILE_LEVELS}


def train(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from cast.forest import export_forest
    seeds, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); P, Lmax = D.search_params(ctx)
    rows, _ = C.pool_rows(over); Qt, St = C.take_pool(a.data, a.ds, a.workload, rows['SEL'][:D.N_TRAIN])
    o = eng.stop_search(ctx.G, ctx.C, Qt, P, Lmax, D.base_rule(D.TYPE_DARTH_LOG, li=D.LOG_INTERVAL, qstats=1), None, None, a.threads, False, False, St)
    drows, ptr = o['drows'], o['dptr']; X, y = drows[:, :N_FEATURES], drows[:, N_FEATURES]; log('log rows', drows.shape)
    # Self-test of the log: its first eleven columns and its recall equal the log without query statistics, and the eight
    # statistics equal their numpy values and are constant within a query.
    o0 = eng.stop_search(ctx.G, ctx.C, Qt[:500], P, Lmax, D.base_rule(D.TYPE_DARTH_LOG, li=D.LOG_INTERVAL), None, None, a.threads, False, False, St[:500])
    m0 = int(ptr[500]); qs = query_statistics(Qt); rq = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    LOGCHK = dict(rows_500=int(m0), first11_equal=bool(o0['drows'].shape[0] == m0 and (o0['drows'][:, :11] == drows[:m0, :11]).all() and (o0['drows'][:, 11] == drows[:m0, 19]).all()),
                  statistics_max_abs_diff_vs_numpy=float(np.abs(drows[:, 11:19] - qs[rq]).max()), statistics_constant_within_query=bool((drows[:, 11:19] == drows[ptr[rq], 11:19]).all()))
    LOGCHK['pass'] = bool(LOGCHK['first11_equal'] and LOGCHK['statistics_max_abs_diff_vs_numpy'] <= 1e-12 and LOGCHK['statistics_constant_within_query']); log('log self-test', LOGCHK)
    mk = lambda **kw: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False,
                                                    random_state=seeds['darthplus_regressors'], **kw)
    nqt = len(ptr) - 1; prefix = model_prefix(a); os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    # The engine evaluates each flattened forest as scikit-learn evaluates the model (largest difference on up to 200,000 logged rows).
    sub = np.random.default_rng(seeds['darthplus_forest_check']).choice(len(X), min(200000, len(X)), replace=False); Xs = np.ascontiguousarray(X[sub]); DIFF = {}
    reg = mk().fit(X, y); Fr = export_forest(reg); DIFF['reg'] = float(np.abs(eng.forest_raw(Fr, Xs) - reg.predict(Xs)).max()); C.save_forest(prefix + '_reg.npz', Fr)
    for lq in QUANTILE_LEVELS:
        mqq = mk(loss='quantile', quantile=lq).fit(X, y); Fq = export_forest(mqq); DIFF[f'q{lq}'] = float(np.abs(eng.forest_raw(Fq, Xs) - mqq.predict(Xs)).max())
        C.save_forest(f'{prefix}_q{lq}.npz', Fq)
    log('forests, engine against scikit-learn: max abs difference', DIFF)
    first = D.first_reach_schedule(X, y, ptr, np.round(np.arange(1.0, 0.05, -0.1), 1), key=lambda r: f'{r:.1f}')
    json.dump(dict(ds=a.ds, graph=a.graph, log_self_test=LOGCHK, forest_max_abs_diff=DIFF, forest_within_1e_9=bool(max(DIFF.values()) <= 1e-9), schedule=first,
                   n_rows=int(len(X)), n_queries=nqt, log_interval=D.LOG_INTERVAL, quantile_levels=list(QUANTILE_LEVELS),
                   fit=dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20)), open(prefix + '.json', 'w'), indent=1)
    log('schedule', first['1.0'], first['0.9']); log('done')


def sweep_point(eng, ctx, Q, S, sch, F, qF, threads, Rt):
    P, Lmax = D.search_params(ctx)
    o = eng.stop_search(ctx.G, ctx.C, Q, P, Lmax, darth_plus_rule(Rt, schedule_entry(sch, Rt), qF), None, F, threads, False, False)
    return dict(Rt=Rt, fail=int((~C.same_answer(o['ids'], S)).sum()), mean_visits=float(o['visits'].mean()), quantile_calls=float(o['qcalls'].mean()))


def sweep(eng, ctx, Q, S, sch, F, qF, threads, tests, stop_when, tag):
    """Rt along RT_SEQ; tests: name -> (eps, level); each test keeps its last pass and stops at its first rejection."""
    from scipy.stats import binom
    n = len(Q); act = {t: True for t in tests}; last = {t: None for t in tests}; trace = []
    for Rt in RT_SEQ:
        trace.append(sweep_point(eng, ctx, Q, S, sch, F, qF, threads, Rt))
        fail = trace[-1]['fail']
        for t, (e, lev) in tests.items():
            if act[t]:
                if binom.cdf(fail, n, e) <= lev: last[t] = Rt
                else: act[t] = False
        log(tag, 'Rt', Rt, trace[-1], 'active', [t for t in tests if act[t]])
        if not any(act[t] for t in stop_when): break
    return trace, last


def select(a):
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); prefix = model_prefix(a); TJ, F, QF = load_models(prefix)
    rows, _ = C.pool_rows(over); Q, S = C.take_pool(a.data, a.ds, a.workload, rows['SEL'][D.N_TRAIN:])   # the 10,000 SEL queries not used in training
    assert (S >= 0).all() and len(Q) == len(rows['SEL']) - D.N_TRAIN; assert TJ['log_self_test']['pass'] and TJ['forest_within_1e_9']
    OUT = dict(ds=a.ds, graph=a.graph, n=len(Q), quantile_level={})
    for lq in QUANTILE_LEVELS:
        trace, last = sweep(eng, ctx, Q, S, TJ['schedule'], F, QF[lq], a.threads, {'d': (0.05, DELTA)}, ('d',), f'sel lq={lq}')
        OUT['quantile_level'][str(lq)] = dict(Rt=last['d'], mean_visits=next((t['mean_visits'] for t in trace if t['Rt'] == last['d']), None), trace=trace)
    ok = {lq: OUT['quantile_level'][str(lq)]['mean_visits'] for lq in QUANTILE_LEVELS if OUT['quantile_level'][str(lq)]['Rt'] is not None}
    best = min(ok.values()) if ok else None
    OUT['selected'] = 0.10 if (not ok or ok.get(0.10) == best) else min(ok, key=ok.get)
    json.dump(OUT, open(prefix + '_select.json', 'w'), indent=1); log('selected quantile level', OUT['selected'], ok)


def levels(a):
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); prefix = model_prefix(a); TJ, F, QF = load_models(prefix); sch = TJ['schedule']
    lqs = json.load(open(prefix + '_select.json'))['selected']
    rows, _ = C.pool_rows(over); Qc, Sc = C.take_pool(a.data, a.ds, a.workload, rows['CAL']); n = len(Qc)
    E = [float(e) for e in json.load(open(a.frontier))['config']['levels']]; M = len(E); LEV = DELTA / (2 * M)
    tests = {str(e): (e, LEV) for e in E}; tests['common_target'] = (0.05, DELTA)
    trace, last = sweep(eng, ctx, Qc, Sc, sch, F, QF[lqs], a.threads, tests, (str(max(E)), 'common_target'), f'cal lq={lqs}')
    OUT = dict(ds=a.ds, graph=a.graph, quantile_level=lqs, n_cal=n, levels=E, M=M, level=LEV, trace=trace, Rt={str(e): last[str(e)] for e in E}, Rt_common_target=last['common_target'],
               cal_fail_at_Rt={str(e): (next(t['fail'] for t in trace if t['Rt'] == last[str(e)]) if last[str(e)] is not None else None) for e in E})
    json.dump(OUT, open(a.out + '.json', 'w'), indent=1)
    Qe, Se = C.take_pool(a.data, a.ds, a.workload, rows['EVAL']); EV = {}; OUT['eval'] = {}; P, Lmax = D.search_params(ctx)
    def ev(Rt, qF):
        o = eng.stop_search(ctx.G, ctx.C, Qe, P, Lmax, darth_plus_rule(Rt, schedule_entry(sch, Rt), qF), None, F, a.threads, False, False)
        return o['visits'].astype(np.int64), ~C.same_answer(o['ids'], Se), o
    for e in E:
        if last[str(e)] is None: OUT['eval'][str(e)] = 'no rule'; continue
        vis, fl, o = ev(last[str(e)], QF[lqs]); EV[f'visits_{e}'] = vis; EV[f'fail_{e}'] = fl
        OUT['eval'][str(e)] = dict(Rt=last[str(e)], mean_visits=float(vis.mean()), fail=int(fl.sum()), fail_rate=float(fl.mean()), quantile_calls=float(o['qcalls'].mean()))
        if e == 0.05: vis2, _, _ = ev(last[str(e)], QF[lqs]); OUT['determinism'] = float((vis2 == vis).mean())
        log('EVAL', e, OUT['eval'][str(e)]); json.dump(OUT, open(a.out + '.json', 'w'), indent=1)
    if last['common_target'] is not None:
        vis, fl, _ = ev(last['common_target'], QF[lqs]); EV['visits_common_target'] = vis; EV['fail_common_target'] = fl
        OUT['eval_common_target'] = dict(Rt=last['common_target'], mean_visits=float(vis.mean()), fail=int(fl.sum()), fail_rate=float(fl.mean()))
    np.savez(a.out + '_eval.npz', **EV); json.dump(OUT, open(a.out + '.json', 'w'), indent=1); log('done')


def report(a):
    _, over = C.read_seeds(a.seeds)
    C.levels_report(a.red, a.ds, a.graph, json.load(open(a.frontier)), a.run, a.out, a.seed, seeds=over, log=log); log('done')


def compare(a):
    """All EVAL queries at eps = 0.05, level delta: our rule (the chain calibrated on CAL, run online) and DARTH (its
    calibrated Rt, run online), paired with the EVAL run of DARTH+ at the common target."""
    from cast.forest import export_forest
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); P, Lmax = D.search_params(ctx); FR = json.load(open(a.frontier))
    S_ = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over); rec, _, _ = C.common_target_staircase(S_, FR, 0.05, DELTA)
    LOGT = np.log(np.arange(40001, dtype=np.float64)); LOG1PT = np.log1p(np.arange(40001, dtype=np.float64))
    Fc = export_forest(S_.model)
    rc = dict(type=1, grid=S_.grid.astype(np.int32), edges=S_.edges.astype(np.float64), log_tab=LOGT, log1p_tab=LOG1PT, beta=np.array(rec['beta'], np.int32))
    Fd, schd = D.load_model(a.darth_model or os.path.join(a.work, 'models', f'darth_{a.ds}_{a.graph}')); dz = schd['1.0']; Rd = json.load(open(a.darth))['Rt_calibrated']
    rows, _ = C.pool_rows(over); Qe, Se = C.take_pool(a.data, a.ds, a.workload, rows['EVAL'])
    EV = dict(np.load(a.run + '_eval.npz')); RJ = json.load(open(a.run + '.json'))
    oc = eng.stop_search(ctx.G, ctx.C, Qe, P, Lmax, rc, Fc, None, a.threads, False, False)
    od = eng.stop_search(ctx.G, ctx.C, Qe, P, Lmax, D.darth_rule(float(Rd), dz), None, Fd, a.threads, False, False)
    vc, vd, vp = oc['visits'].astype(np.int64), od['visits'].astype(np.int64), EV['visits_common_target']
    fc, fd, fp = ~C.same_answer(oc['ids'], Se), ~C.same_answer(od['ids'], Se), EV['fail_common_target']
    Wb = C.bootstrap_weights(len(vc), a.seed)
    def br(x, y):
        ra = (Wb @ x.astype(np.float64)) / (Wb @ y.astype(np.float64))
        return dict(ratio=float(x.mean() / y.mean()), lower_bound_one_sided=float(np.quantile(ra, 0.025)), upper_bound_one_sided=float(np.quantile(ra, 0.975)))
    OUT = dict(ds=a.ds, graph=a.graph, n=int(len(vc)), bootstrap_seed=a.seed, quantile_level=RJ['quantile_level'], Rt_darth_plus=RJ['Rt_common_target'], Rt_darth=Rd,
               cast=dict(theta=rec['theta'], mean_visits=float(vc.mean()), fail_rate=float(fc.mean())), darth=dict(mean_visits=float(vd.mean()), fail_rate=float(fd.mean())),
               darth_plus=dict(mean_visits=float(vp.mean()), fail_rate=float(fp.mean())),
               darth_plus_over_cast=br(vp, vc), darth_over_cast=br(vd, vc), darth_plus_over_darth=br(vp, vd))
    json.dump(OUT, open(a.out, 'w'), indent=1); log(json.dumps(OUT))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    for name in ('train', 'select', 'levels', 'compare'):
        p = sub.add_parser(name); C.add_setting_args(p)
        p.add_argument('--model', default=None, help='prefix of the model files (default WORK/models/darthplus_<ds>_<graph>)')
        p.add_argument('--seeds', default=None, help='JSON file overriding recorded seeds')
        if name in ('levels', 'compare'): p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting')
        if name == 'levels': p.add_argument('--out', required=True, help='output prefix (.json and _eval.npz)')
        if name == 'compare':
            p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload')
            p.add_argument('--darth', required=True, help='output JSON of baselines/darth.py calibrate')
            p.add_argument('--darth-model', default=None, help='prefix of the DARTH model files (default WORK/models/darth_<ds>_<graph>)')
            p.add_argument('--run', required=True, help='output prefix of the levels step')
            p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--out', required=True)
    p = sub.add_parser('report')
    p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload'); p.add_argument('--ds', required=True)
    p.add_argument('--graph', default='A'); p.add_argument('--frontier', required=True); p.add_argument('--run', required=True, help='output prefix of the levels step')
    p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(train=train, select=select, levels=levels, report=report, compare=compare)[a.step](a)

"""Timing bundle of one setting: everything the timed online search needs, and the offline replay it is checked against.

Paper: Table III and the implementation details of Section VI-A (offline calibration from logged trajectories, online
agreement checks). The script fits the interface on FIT, calibrates every rule that reads it at the common target
(eps = delta = 0.05 on CAL) and writes, under the prefix --out:

    OUT.npz   Q, S            vectors and exact answers of the timing queries
              rows            their rows in the query pool (layout 'pool') or in the eval part (layout 'parts')
              grid, edges     checkpoint grid and bucket edges of the interface
              c, X            logged visits at the checkpoints and completion times of the timing queries
              features_head, score_head, full_head     logged features, scores and flags of the first 1,000 of them
              clf_<field>, reg_<field>                 flattened score model and LAET-style regressor (cast.forest)
              log_tab, log1p_tab       log(n) and log1p(n) for integer n, computed here so that the online features
                                       equal the reduced ones bit for bit (run this script in the environment of
                                       pipeline/reduce_logs.py)
              replay_<rule>_{cost,checkpoint,fail}     offline replay of every rule on the timing queries
    OUT.json  the calibrated rule records (cast.compared), eps, delta, k, list size, J, the number of buckets

Timing queries: --queries hold takes the 20,000 HOLD queries of a setting in layout 'pool' (Table III); --queries eval
takes the first 10,000 EVAL queries (agreement checks of Section VI-A; latency on YFCC-10M, "Workload breadth").
No model is shipped: the forests are fitted here from the reduced arrays.

Usage:
    python timing/export_bundle.py --red RED_DIR --data DATA_DIR --ds amazon --graph A --frontier FRONTIER.json \\
        --queries hold --out WORK/timing/amazon_A_hold
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_EVAL_QUERIES = 10000                 # timing queries of --queries eval
FRONTIER_LEVELS = ('0.03', '0.05', '0.1')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays of the workload')
    ap.add_argument('--data', required=True, help='prepared data directory (query vectors and exact answers)')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output JSON of pipeline/frontier.py for this setting')
    ap.add_argument('--out', required=True, help='prefix of the bundle (OUT.npz and OUT.json)')
    ap.add_argument('--queries', default='hold', choices=('hold', 'eval'), help='timing queries')
    ap.add_argument('--layout', default='pool', choices=('pool', 'parts'))
    ap.add_argument('--workload', default='k10_kappa1', help='workload name of the exact answers')
    ap.add_argument('--eps', type=float, default=0.05, help='risk level of the common target')
    ap.add_argument('--delta', type=float, default=0.05, help='level of the calibration test on CAL')
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    return ap.parse_args(argv)


def _log_tables():
    """log(n) and log1p(n) for the integers n up to 40,000: the values that entered two of the reduced features."""
    import numpy as np
    n = np.arange(40001, dtype=np.float64)
    with np.errstate(divide='ignore'): return np.log(n), np.log1p(n)


def main(argv=None):
    a = parse_args(argv); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    import numpy as np
    from cast import datasets
    from cast.chain import chain_multipliers
    from cast.compared import calibrate_common_target, rule_F
    from cast.forest import export_forest, forest_raw
    from cast.grid import LIST_SIZE
    from cast.rules import evaluate_rule
    from cast.setting import N_CHUNKS, N_POOL, fit_interface, load_setting
    from engine_rules import HEAD

    assert a.queries == 'eval' or a.layout == 'pool', "HOLD exists in layout 'pool' only"
    seeds = json.load(open(a.seeds)) if a.seeds else None
    FJ = json.load(open(a.frontier)); front = FJ['rules']['chain']; itf = FJ['interface']; assert FJ['layout'] == a.layout
    S = load_setting(a.red, a.ds, a.graph, layout=a.layout, target=FJ['target'], keep_hold=(a.queries == 'hold'), fit_block=itf['fit_block'],
                     stride=itf['stride'], seeds=seeds, resplit_seed=itf.get('resplit_seed'))
    fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])
    J, NB = S.J, S.NB; EVAL, HOLD = S.EVAL, S.HOLD
    log(a.ds, a.graph, 'loaded; rows', S.n, 'HOLD', len(HOLD))

    # ---- timing queries (rows of the setting) and the queries whose logged features are kept
    TQ = HOLD if a.queries == 'hold' else EVAL[:N_EVAL_QUERIES]; HD = TQ[:HEAD]

    # ---- the flattened score model against scikit-learn
    Fclf = export_forest(S.model)
    Xs = S.features[HD].reshape(-1, 7)
    check_clf = float(np.abs((1.0 / (1.0 + np.exp(forest_raw(Fclf, Xs)))) - (1 - S.model.predict_proba(Xs)[:, 1])).max())
    log('flattened classifier against scikit-learn, max |difference of the score|', check_clf)

    # ---- calibrations at the common target (Level+cap, Level, LAET-style, Fixed, our staircase), and the frontier rules
    chain_grid = np.array(FJ['chain_grid'], np.int64)          # multipliers of the chain, as recorded by the frontier run
    cal = calibrate_common_target(S, a.eps, a.delta, chain_grid, frontier_rules={e: front[e] for e in FRONTIER_LEVELS if e in front}, log=log)
    RULE, Freg = cal['RULE'], cal['laet_forest']
    check_reg = None
    if 'laet_style' in RULE:
        j0 = RULE['laet_style']['j0']
        Xr = np.column_stack([S.features[HD, j0], S.full[HD, j0].astype(np.float64)])
        check_reg = float(np.abs(forest_raw(Freg, Xr) - cal['laet_reg'].predict(Xr)).max()); log('flattened regressor against scikit-learn', check_reg)

    # ---- offline replay of every rule on the timing queries: stop checkpoint, cost, failure
    SIM = {}
    for nm in RULE:
        F_ = np.maximum.accumulate(np.clip(rule_F(S, cal, nm, TQ), 0, 1), axis=1); F_[:, -1] = 1.0
        cst, rsk = evaluate_rule(S, F_, TQ)
        SIM[nm] = dict(cost=cst.astype(np.int64), checkpoint=F_.argmax(1).astype(np.int32), fail=rsk.astype(np.int8))
        RULE[nm]['replay_mean_visits'] = float(cst.mean())
        if a.queries == 'hold': f_ = int(round(rsk.sum())); RULE[nm]['replay_coverage'] = 1 - f_ / len(HOLD)
        else: RULE[nm]['replay_coverage'] = float(1 - rsk.mean())
    log('replay on the timing queries', {k: (round(v['replay_mean_visits'], 1), round(v['replay_coverage'], 4)) for k, v in RULE.items()})

    # ---- timing queries: vectors and exact answers
    if a.layout == 'pool':
        qrow = S.pool_rows['HOLD'] if a.queries == 'hold' else S.pool_rows['EVAL'][:N_EVAL_QUERIES]; per = N_POOL // N_CHUNKS
        Q, Sx = None, None
        for kch in np.unique(qrow // per):
            sel_ = np.where(qrow // per == kch)[0]; offs = qrow[sel_] % per
            Qk = datasets.queries(a.data, a.ds, f'chunk{kch}'); Sk = datasets.exact_answers(a.data, a.ds, f'chunk{kch}', a.workload)
            if Q is None: Q = np.zeros((len(qrow), Qk.shape[1]), np.float32); Sx = np.zeros((len(qrow), Sk.shape[1]), np.int64)
            Q[sel_] = Qk[offs]; Sx[sel_] = Sk[offs]
        described = 'all HOLD queries, in the order of the pool permutation' if a.queries == 'hold' else 'first 10,000 EVAL queries, in the order of the pool permutation'
    else:
        qrow = np.arange(N_EVAL_QUERIES)
        Q = np.ascontiguousarray(datasets.queries(a.data, a.ds, 'eval')[:N_EVAL_QUERIES], dtype=np.float32)
        Sx = datasets.exact_answers(a.data, a.ds, 'eval', a.workload)[:N_EVAL_QUERIES].astype(np.int64)
        assert len(TQ) == len(Q) == len(Sx) == N_EVAL_QUERIES and (TQ == EVAL[:N_EVAL_QUERIES]).all()
        described = 'first 10,000 queries of the eval part'
    k = int(Sx.shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(f'{a.out}.npz', Q=Q, S=Sx, rows=qrow, grid=S.grid.astype(np.int32), edges=S.edges, c=S.c[TQ], X=S.X[TQ],
             features_head=S.features[HD], score_head=S.s[HD], full_head=S.full[HD],
             log_tab=_log_tables()[0], log1p_tab=_log_tables()[1],
             **{f'clf_{f}': v for f, v in Fclf.items() if f != 'base'},
             **({f'reg_{f}': v for f, v in Freg.items() if f != 'base'} if Freg is not None else {}),
             **{f'replay_{nm}_{f}': v for nm, d in SIM.items() for f, v in d.items()})
    json.dump(dict(ds=a.ds, graph=a.graph, layout=a.layout, workload=a.workload, queries=a.queries, eps=a.eps, delta=a.delta,
                   list_size=LIST_SIZE, k=k, J=int(J), n_buckets=int(NB), clf_base=Fclf['base'], reg_base=None if Freg is None else Freg['base'],
                   classifier_check=check_clf, regressor_check=check_reg, rules=RULE,
                   laet_features='features[:, j0] + full[:, j0]', timing_queries=described),
              open(f'{a.out}.json', 'w'), indent=1, default=str)
    log('bundle written', a.out)


if __name__ == '__main__':
    main()

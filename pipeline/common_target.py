"""Visits of every calibrated rule at one common risk target (eps = delta = 0.05) on the EVAL queries of one setting.

Paper: Table II (the ratio columns Fixed, LAET-style, Level, Level+cap, List size, Dist.-adaptive and the column
"CaST visits") and the implementation details of Section VI-A: every rule, ours included, is calibrated by the same
fixed-sequence test on CAL. Ratios to CaST come with 2,000 paired bootstrap resamples of the EVAL queries.

The distance-adaptive rule is calibrated here on the gamma grid stored with the reduced arrays;
baselines/distance_adaptive.py recalibrates it on the finer grid.

Usage:
    python pipeline/common_target.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --seed SEED --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.calibration import fixed_sequence_deterministic
from cast.compared import calibrate_common_target, rule_F
from cast.rules import evaluate_rule
from cast.setting import N_CHUNKS, PARTS, fit_interface, load_setting


def load_stops(red_dir, ds, graph, layout, target):
    """Per-query stop times of the list-size and distance-adaptive rules, in the row order of the setting's files."""
    names = [f'chunk{k}' for k in range(N_CHUNKS)] if layout == 'pool' else list(PARTS)
    EX = [np.load(f'{red_dir}/{ds}_{graph}_{nm}_stops.npz') for nm in names]
    list_stop = np.concatenate([e['list_stop'] for e in EX]).astype(np.int64)
    adaptive_stop = np.concatenate([e['adaptive_stop'] for e in EX])
    X = np.concatenate([np.load(f'{red_dir}/{ds}_{graph}_{nm}.npz')[target] for nm in names])
    return list_stop, adaptive_stop, X, EX[0]['list_sizes'].astype(np.int64), EX[0]['gammas']


def summarize(cost, fail):
    return dict(mean_visits=float(cost.mean()), fail=int(fail.sum()), fail_rate=float(fail.mean()), n=int(len(cost)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True); ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the same setting (chain grid, interface, frontier rules)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap (recorded values: configs/seeds.json)')
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--seeds', default=None)
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)

    FR = json.load(open(a.frontier)); itf = FR['interface']; layout, target = FR['layout'], FR['target']
    seeds = json.load(open(a.seeds)) if a.seeds else None
    S = load_setting(a.red, a.ds, a.graph, layout=layout, target=target, fit_block=itf['fit_block'], stride=itf['stride'], seeds=seeds,
                     resplit_seed=itf.get('resplit_seed'))
    fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])
    eps, delta, EVAL = a.eps, a.delta, S.EVAL
    front = {e: FR['rules']['chain'][e] for e in ('0.03', '0.05', '0.1') if FR['rules']['chain'].get(e, {}).get('calibrated')}
    cal = calibrate_common_target(S, eps, delta, np.array(FR['chain_grid'], np.int64), frontier_rules=front, log=log)

    # ---- list size and distance-adaptive rule: per-query stop times from the logged traversal; a query fails iff it stops before X
    list_stop, adaptive_stop, Xf, sizes, gammas = load_stops(a.red, a.ds, a.graph, layout, target)
    if layout == 'pool':                 # rows of each sample among the 200,000 pool queries
        PART = {k: S.pool_rows[k] for k in ('SEL', 'CAL', 'EVAL')}
    else:                                # rows of the concatenated part files
        PART = dict(SEL=S.SEL, CAL=S.CAL, EVAL=S.EVAL)
    for k, idx in (('SEL', S.SEL), ('CAL', S.CAL), ('EVAL', S.EVAL)): assert np.array_equal(Xf[PART[k]], S.X[idx]), k
    gi = {int(L): i for i, L in enumerate(sizes)}
    list_cf = lambda L, rows: (list_stop[rows, gi[L]], list_stop[rows, gi[L]] < Xf[rows])
    adaptive_cf = lambda t, rows: (adaptive_stop[rows, t], adaptive_stop[rows, t] < Xf[rows])
    LIST_SEQ = [int(L) for L in sizes[::-1]]; GAMMA_SEQ = list(range(len(gammas)))       # conservative first
    ev = PART['EVAL']
    F_c = np.maximum.accumulate(np.clip(rule_F(S, cal, 'cast', EVAL), 0, 1), axis=1); F_c[:, -1] = 1.0
    c_cast, r_cast = evaluate_rule(S, F_c, EVAL); c_cast = c_cast.astype(np.int64); f_cast = r_cast.round().astype(bool)
    L2, tr_list = fixed_sequence_deterministic(LIST_SEQ, list_cf, PART['CAL'], eps, delta)
    t2, tr_adapt = fixed_sequence_deterministic(GAMMA_SEQ, adaptive_cf, PART['CAL'], eps, delta)

    # ---- paired bootstrap of the visit ratio to CaST over the EVAL queries
    rng = np.random.default_rng(a.seed)
    W = rng.multinomial(len(ev), np.full(len(ev), 1.0 / len(ev)), size=a.resamples).astype(np.float64)
    def visit_ratio(x, y):
        ra = (W @ x.astype(np.float64)) / (W @ y.astype(np.float64))
        return dict(ratio=float(x.mean() / y.mean()), lower_bound_one_sided=float(np.quantile(ra, 0.025)), se=float(ra.std(ddof=1)))

    TAB = dict(cast=dict(theta=cal['RULE']['cast']['theta'], **summarize(c_cast, f_cast)))
    for nm, sel_, trace, cf in (('list_size', L2, tr_list, list_cf), ('distance_adaptive', t2, tr_adapt, adaptive_cf)):
        if sel_ is None: TAB[nm] = dict(calibrated=False, trace=trace); continue
        ce, fe = cf(sel_, ev)
        TAB[nm] = dict(calibrated=True, param=(sel_ if nm == 'list_size' else float(gammas[sel_])), **summarize(ce, fe),
                       visit_ratio=visit_ratio(ce, c_cast), trace=trace)
    for nm in ('level_cap', 'level', 'laet_style', 'fixed'):
        if nm not in cal['RULE']: TAB[nm] = dict(calibrated=False); continue
        cst, rsk = evaluate_rule(S, rule_F(S, cal, nm, EVAL), EVAL); cst = cst.astype(np.int64); fl = rsk.round().astype(bool)
        TAB[nm] = dict(calibrated=True, rule=cal['RULE'][nm], **summarize(cst, fl), visit_ratio=visit_ratio(cst, c_cast))
    OUT = dict(ds=a.ds, graph=a.graph, layout=layout, target=target, eps=eps, delta=delta, n_eval=int(len(ev)), n_cal=int(len(PART['CAL'])),
               list_sizes=sizes.tolist(), gammas=gammas.tolist(), bootstrap_seed=a.seed, table=TAB, rules=cal['RULE'])
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    log('table', {k: (v.get('mean_visits'), v.get('fail_rate'), (v.get('visit_ratio') or {}).get('ratio')) for k, v in TAB.items()})
    log('done')


if __name__ == '__main__':
    main()

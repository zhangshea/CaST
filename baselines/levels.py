"""The compared rules at the nine risk levels, each calibrated on CAL at level delta / (2 M) like the frontier rule of
the same level, and their EVAL visits against that rule.

First group (Level+cap, Level, LAET-style, Fixed): the calibrations of cast.compared with the level of the CAL test set
to delta / (2 M); the tests run on SEL to choose the cap of Level+cap and the checkpoint of LAET-style keep their levels,
and no candidate grid is extended. A rule without a calibrated parameter at a level is reported as such. Each rule comes
with its visit ratio to the frontier rule (2,000 paired bootstrap resamples, one-sided 2.5% lower bound) and a flag that
tells whether the selected parameter is the last candidate of its grid.
List size and distance-adaptive rule: calibrated from the stop times of the logged traversal on the grids stored with
the reduced arrays (baselines/distance_adaptive.py repeats the distance-adaptive rule on the fine grid of gamma).

Paper: Fig. 4 (visits of each compared rule over ours per risk level; graph B in the text of Section VI-B, paragraph
"Cost and latency at a common risk target") and the inputs of the equal-failure-rate comparison (baselines/equal_failure_rate.py).

Usage:
  python baselines/levels.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --seed SEED --out LEVELS.json [--levels 0.01,0.02,0.03]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

NAMES = ('level_cap', 'level', 'laet_style', 'fixed')
LABEL = dict(level_cap='Level+cap', level='Level', laet_style='LAET-style', fixed='Fixed')


def calibrate_first_group(S, e, lev):
    """The four rules of the first group at risk e with CAL level lev: ({name: rule record or None}, LAET predictions)."""
    from cast.compared import calibrate_fixed, calibrate_laet, calibrate_level, calibrate_level_cap
    res = dict(level_cap=calibrate_level_cap(S, e, lev), level=calibrate_level(S, e, lev))
    res['laet_style'], pred, _ = calibrate_laet(S, e, lev)
    res['fixed'] = calibrate_fixed(S, e, lev)
    return res, pred


def saturated(S, nm, R_):
    """The selected parameter is the last candidate of its grid."""
    from cast.compared import LAET_MULTIPLIERS, level_candidates
    if nm in ('level_cap', 'level'): return bool(R_['lam'] == float(level_candidates(S)[-1]))
    if nm == 'laet_style': return bool(R_['mult'] == float(LAET_MULTIPLIERS[-1]))
    return bool(R_['jfix'] == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays of the workload'); ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting')
    ap.add_argument('--levels', default=None, help='comma-separated risk levels to run (default: all levels of the frontier run)')
    ap.add_argument('--delta', type=float, default=0.05, help='error level of the calibration; each level is tested at delta / (2 M)')
    ap.add_argument('--no-stops', action='store_true', help='skip the list size and the distance-adaptive rule (no *_stops.npz files needed)')
    ap.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); ap.add_argument('--seeds', default=None); ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from cast.calibration import fixed_sequence_deterministic
    from cast.compared import rule_F
    _, over = C.read_seeds(a.seeds); FR = json.load(open(a.frontier)); S = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over)
    EVAL, CAL = S.EVAL, S.CAL
    E = FR['config']['levels']; M = len(E); LEV = a.delta / (2 * M)
    RUN = E if a.levels is None else [e for e in E if any(abs(e - float(x)) < 1e-12 for x in a.levels.split(','))]; assert RUN, a.levels
    OUT = dict(ds=a.ds, graph=a.graph, n_eval=int(len(EVAL)), n_cal=int(len(CAL)), bootstrap_seed=a.seed, level=LEV, M=M)
    W = C.bootstrap_weights(len(EVAL), a.seed)
    if not a.no_stops:                   # stop times of the list-size and distance-adaptive rules; a query fails iff it stops before X
        list_stop, adaptive_stop, Xf, sizes, gammas = C.load_stops(a.red, a.ds, a.graph, FR['layout'], FR['target']); PART = C.sample_rows(S); ev = PART['EVAL']
        for k, idx in (('SEL', S.SEL), ('CAL', S.CAL), ('EVAL', S.EVAL)): assert np.array_equal(Xf[PART[k]], S.X[idx]), k
        gi = {int(L): i for i, L in enumerate(sizes)}
        list_cf = lambda L, rows: (list_stop[rows, gi[L]], list_stop[rows, gi[L]] < Xf[rows])
        adaptive_cf = lambda t, rows: (adaptive_stop[rows, t], adaptive_stop[rows, t] < Xf[rows])
        LIST_SEQ = [int(L) for L in sizes[::-1]]; GAMMA_SEQ = list(range(len(gammas)))
    OUT['levels'] = {}
    for e in RUN:
        r_ = FR['rules']['chain'][str(e)]; row = {}
        if not a.no_stops:
            for nm, seq, cf in (('list_size', LIST_SEQ, list_cf), ('distance_adaptive', GAMMA_SEQ, adaptive_cf)):
                p, _ = fixed_sequence_deterministic(seq, cf, PART['CAL'], e, LEV)
                row[nm] = dict(calibrated=False) if p is None else dict(calibrated=True, param=(p if nm == 'list_size' else float(gammas[p])), **C.summarize(*cf(p, ev)))
        if not r_.get('calibrated', True) or r_.get('theta') is None:
            row['frontier'] = 'no rule'; OUT['levels'][str(e)] = row; continue
        th, Fch = C.frontier_staircase(S, r_, EVAL); c_ch, f_ch = C.staircase_outcomes(S, Fch, EVAL)
        row['frontier'] = dict(theta=th, **C.summarize(c_ch, f_ch), recorded_mean_visits=r_.get('mean_visits'),
                                equals_recorded=bool(abs(float(c_ch.mean()) - float(r_.get('mean_visits'))) < 1e-9))
        res, pred = calibrate_first_group(S, float(e), LEV)
        for nm in NAMES:
            R_ = res[nm]
            if R_ is None: row[nm] = dict(label=LABEL[nm], calibrated=False); continue
            cst, fl = C.staircase_outcomes(S, rule_F(S, dict(RULE={nm: R_}, laet_pred=pred), nm, EVAL), EVAL)
            row[nm] = dict(label=LABEL[nm], calibrated=True, rule=R_, saturated=saturated(S, nm, R_), **C.summarize(cst, fl), visit_ratio=C.visit_ratio(W, cst, c_ch))
        OUT['levels'][str(e)] = row
        log('level', e, {k: (v.get('mean_visits'), v.get('fail_rate'), (v.get('visit_ratio') or {}).get('lower_bound_one_sided')) if isinstance(v, dict) else v for k, v in row.items()})
        json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    log('done')


if __name__ == '__main__':
    main()

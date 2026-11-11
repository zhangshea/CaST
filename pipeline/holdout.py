"""Final check on the held-out queries (HOLD) of one setting of the main workload.

Paper: paragraph "Held-out check" of Section VI-B. HOLD is read by no fitting or calibration step (Section III-D).
For the frontier rule of every risk level: failures on HOLD against the critical count of an exact binomial test at
level 0.05 / 4, the Maurer-Pontil lower bound on the mean HOLD cost against U, and the bounds U and L recomputed on HOLD
alone. For the rules of the common target (Table III): visits and failures on HOLD.

Usage:
    python pipeline/holdout.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.bounds import REL, cost_upper_bound, frontier_lower_bound, lagrangian_lower_bound
from cast.calibration import binomial_critical
from cast.chain import chain_element
from cast.compared import calibrate_common_target, rule_F
from cast.rules import evaluate_rule, staircase_F, terminal_F
from cast.setting import SPLIT, fit_interface, load_setting

N_SETTINGS = 4          # settings of the main workload that share the level 0.05 of the held-out tests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True); ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help="output of pipeline/frontier.py for the same setting (layout 'pool')")
    ap.add_argument('--out', required=True)
    ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--seeds', default=None)
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)

    FR = json.load(open(a.frontier)); cfg = FR['config']; itf = FR['interface']; assert FR['layout'] == 'pool'
    seeds = json.load(open(a.seeds)) if a.seeds else None
    S = load_setting(a.red, a.ds, a.graph, layout='pool', target=FR['target'], keep_hold=True, fit_block=itf['fit_block'], stride=itf['stride'], seeds=seeds,
                     resplit_seed=itf.get('resplit_seed'))
    fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])
    H, SEL, EVAL, HOLD = S.H, S.SEL, S.EVAL, S.HOLD; nH = len(HOLD)
    alpha_L, alpha_U = cfg['alpha_L'], cfg['alpha_U']
    E = cfg['levels']; M = len(E); LAMS = FR['multipliers']; K = len(LAMS)
    log(a.ds, a.graph, 'rows', S.n, 'HOLD', nH)
    OUT = dict(ds=a.ds, graph=a.graph, n_hold=nH, split=SPLIT)

    # ---- rules of the common target on HOLD
    front = {e: FR['rules']['chain'][e] for e in ('0.03', '0.05', '0.1') if FR['rules']['chain'].get(e, {}).get('calibrated')}
    cal = calibrate_common_target(S, a.eps, a.delta, np.array(FR['chain_grid'], np.int64), frontier_rules=front, log=log)
    CT = {}
    for nm in cal['RULE']:
        F_ = np.maximum.accumulate(np.clip(rule_F(S, cal, nm, HOLD), 0, 1), axis=1); F_[:, -1] = 1.0
        cst, rsk = evaluate_rule(S, F_, HOLD); f_ = int(round(rsk.sum()))
        CT[nm] = dict(mean_visits=float(cst.mean()), fail=f_, fail_rate=f_ / nH)
    for nm in CT: CT[nm]['visit_ratio_vs_cast'] = CT[nm]['mean_visits'] / CT['cast']['mean_visits']
    OUT['common_target'] = CT; OUT['rules'] = cal['RULE']

    # ---- frontier rules on EVAL and on HOLD
    FRT = {}
    for e in E:
        r_ = FR['rules']['chain'][str(e)]
        if not r_.get('calibrated'): FRT[str(e)] = dict(calibrated=False); continue
        th = r_['theta']; th = th if th == 'terminal' else int(th)
        mk = (lambda idx: terminal_F(S, idx)) if th == 'terminal' else (lambda idx, b_=chain_element(S, SEL, th)[0]: staircase_F(S, b_, idx))
        cE, _ = evaluate_rule(S, mk(EVAL), EVAL)
        cH, rH = evaluate_rule(S, mk(HOLD), HOLD); f_ = int(round(rH.sum()))
        dlt = float(cH.mean() - cE.mean()); se = float(np.sqrt(cH.var(ddof=1) / nH + cE.var(ddof=1) / len(EVAL)))
        crit = binomial_critical(nH, e, 0.05 / N_SETTINGS)
        FRT[str(e)] = dict(calibrated=True, theta=th, mean_visits_eval=float(cE.mean()), eval_mean_matches=bool(abs(cE.mean() - r_['mean_visits']) < 1e-6),
                           U_eval=r_['U'], mean_visits_hold=float(cH.mean()), fail=f_, fail_rate=f_ / nH,
                           critical=crit, reject=bool(f_ >= crit),
                           cost_lower_bound_hold=2 * float(cH.mean()) - cost_upper_bound(cH, 0.05 / N_SETTINGS, H),
                           U_hold=cost_upper_bound(cH, alpha_U / M, H), hold_minus_eval=[dlt, dlt - 1.96 * se, dlt + 1.96 * se])
        FRT[str(e)]['cost_lower_bound_below_U'] = bool(FRT[str(e)]['cost_lower_bound_hold'] <= r_['U'])
    OUT['frontier'] = FRT

    # ---- lower bound and ratio recomputed on HOLD alone (same multipliers and error levels)
    ELL = [lagrangian_lower_bound(S, HOLD, float(l_), alpha_L / K) for l_ in LAMS]
    Lh = lambda e: frontier_lower_bound(ELL, e)
    OUT['hold_bounds'] = {str(e): dict(L_hold=Lh(e), U_hold=FRT[str(e)].get('U_hold'),
                                       ratio_hold=(FRT[str(e)]['U_hold'] * (1 + REL) / Lh(e)) if FRT[str(e)].get('calibrated') and Lh(e) > 0 else None,
                                       L_eval=FR['bounds'][str(e)]['L'], ratio_eval=FR['bounds'][str(e)].get('ratio_chain')) for e in E}
    OUT['hold_ell'] = [{k_: v for k_, v in d.items() if k_ != 'trace'} for d in ELL]
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

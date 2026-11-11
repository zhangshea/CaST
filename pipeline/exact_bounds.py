"""Reported optimality-gap bounds U / L of the frontier rules of one setting, with outward rounding.

Paper: Corollary 1; the U / L columns of Table II and Fig. 3(a) (ratios rounded up).

Reads the output of pipeline/frontier.py for the multipliers lambda_k, the error levels and the calibrated chain rules.

Usage:
    python pipeline/exact_bounds.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --out OUT.json
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.bounds import REL
from cast.chain import chain_element
from cast.exact import ExactBounds, down6, up6
from cast.rules import evaluate_rule, staircase_F, terminal_F
from cast.setting import fit_interface, load_setting


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True); ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the same setting and interface')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None)
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)

    FR = json.load(open(a.frontier)); cfg = FR['config']; itf = FR['interface']
    seeds = json.load(open(a.seeds)) if a.seeds else None
    S = load_setting(a.red, a.ds, a.graph, layout=FR['layout'], target=FR['target'], fit_block=itf['fit_block'],
                     stride=itf['stride'], seeds=seeds, resplit_seed=itf.get('resplit_seed'))
    fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])
    E = cfg['levels']; M = len(E); LAMS = [int(l_) for l_ in FR['multipliers']]; K = len(LAMS)
    EB = ExactBounds(S, cfg['alpha_L'], cfg['alpha_U'], K, M)
    OUT = dict(ds=a.ds, graph=a.graph, alpha_k=str(EB.alpha_k), delta_U=str(EB.delta_u), ln_up_L=float(EB.ln_L), ln_up_U=float(EB.ln_U), multipliers=LAMS)
    idx = S.EVAL; log('EVAL', 'start', len(idx))
    ELL = [EB.lagrangian_lower_bound(idx, l_) for l_ in LAMS]
    for d_, o_ in zip(ELL, FR['ell']): assert int(o_['lam']) == d_['lam']; d_['ell_float'] = float(o_['ell'])
    res_ = {}
    for e in E:
        r_ = FR['rules']['chain'][str(e)]
        if not r_.get('calibrated'): res_[str(e)] = dict(calibrated=False); continue
        th = r_['theta']; th = th if th == 'terminal' else int(th)
        F_ = terminal_F(S, idx) if th == 'terminal' else staircase_F(S, chain_element(S, S.SEL, th)[0], idx)
        cost, _ = evaluate_rule(S, F_, idx); U_, mean_ = EB.cost_upper_bound(cost); L_ = EB.frontier(ELL, e)
        res_[str(e)] = dict(calibrated=True, theta=th, mean=float(mean_), U_up6=up6(U_), L_down6=down6(L_), ratio_up6=(up6(U_ / L_) if L_ > 0 else None))
        # the exact mean must equal the float mean of pipeline/frontier.py
        res_[str(e)]['mean_check'] = bool(abs(float(mean_) - r_['mean_visits']) < 1e-6)
        res_[str(e)]['ratio_float'] = r_['U'] * (1 + REL) / FR['bounds'][str(e)]['L']
        log('EVAL', e, {k_: v_ for k_, v_ in res_[str(e)].items() if k_ != 'theta'})
    curve = [(float(e), down6(EB.frontier(ELL, e))) for e in np.round(np.arange(0.005, 0.1501, 0.005), 3)]
    OUT['EVAL'] = dict(n=len(idx), ell=[{k_: (float(v_) if isinstance(v_, Fr) else v_) for k_, v_ in d_.items() if k_ != 'ell'} | dict(ell_down6=d_['ell_down6'])
                                       for d_ in ELL], rules=res_, L_curve=curve)
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    log('done'); json.dump(OUT, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()

"""Effect of the sizes of CAL and EVAL on the frontier rule at eps = 0.05 and on its bound.

Paper: Section VI-B, paragraph "Sample size and labeling cost" ("With 10,000 instead of 40,000 CAL queries, the rule at
eps = 0.05 spends 2.3% to 5.6% more visits, and with 30,000 instead of 120,000 EVAL queries its ratio grows by 0.036 to
0.044").
    CAL size   the chain is calibrated at level delta / (2 M) on the first n queries of CAL, n in {5k, 10k, 20k, 40k},
               with the stopping draws of those queries; mean visits, failure rate and U on all of EVAL.
    EVAL size  for the frontier rule at eps, U at level alpha_U / M and L(eps) from the multipliers of the frontier run
               at level alpha_L / K each, both on the first n queries of EVAL, n in {10k, 30k, 60k, 120k}; ratio U / L.
The chain uses the multipliers recorded by the frontier run.

Usage:
    python analysis/sample_size.py run --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --out OUT.json
    python analysis/sample_size.py summary amazon_A=OUT.json amazon_B=... arxiv_A=... arxiv_B=...
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import labelled_paths

CAL_SIZES = (5000, 10000, 20000, 40000); EVAL_SIZES = (10000, 30000, 60000, 120000)


def sample_size(S, FR, eps=0.05, cal_sizes=CAL_SIZES, eval_sizes=EVAL_SIZES, log=print):
    import numpy as np
    from cast.bounds import cost_upper_bound, lagrangian_lower_bound
    from cast.calibration import fixed_sequence
    from cast.rules import evaluate_rule
    from replay import sel_chain
    cfg = FR['config']; M = len(cfg['levels']); delta, alpha_L, alpha_U = cfg['delta'], cfg['alpha_L'], cfg['alpha_U']; H = S.H
    CAL, EVAL = S.CAL, S.EVAL
    LAMS = [float(l_) for l_ in FR['multipliers']]; K = len(LAMS)
    grid = np.array(FR['chain_grid'], np.int64); CH, mk = sel_chain(S, grid)
    OUT = dict(eps=eps, n_cal={}, n_eval={})
    for nc in cal_sizes:
        idx = CAL[:nc]; th, _ = fixed_sequence(S, ['terminal'] + [int(l) for l in grid[::-1]], mk, eps, idx=idx, U=S.u_cal[:nc], delta=delta / (2 * M))
        if th is None: OUT['n_cal'][nc] = dict(calibrated=False); continue
        cE, rE = evaluate_rule(S, mk(th, EVAL), EVAL)
        OUT['n_cal'][nc] = dict(calibrated=True, theta=th, mean_visits_eval=float(cE.mean()), U=cost_upper_bound(cE, alpha_U / M, H), fail_rate_eval=float(rE.mean()))
        log('CAL size', nc, OUT['n_cal'][nc])
    th05 = FR['rules']['chain'][str(eps)]['theta']; th05 = th05 if th05 == 'terminal' else int(th05)
    c05, _ = evaluate_rule(S, mk(th05, EVAL), EVAL)
    for ne in eval_sizes:
        idx = EVAL[:ne]; ell = [lagrangian_lower_bound(S, idx, l_, alpha_L / K) for l_ in LAMS]; Lv = max(0.0, max(d_['ell'] - d_['lam'] * eps for d_ in ell))
        U_ = cost_upper_bound(c05[:ne], alpha_U / M, H)
        OUT['n_eval'][ne] = dict(mean_visits=float(c05[:ne].mean()), U=U_, L=Lv, ratio=(U_ / Lv) if Lv > 0 else None)
        log('EVAL size', ne, OUT['n_eval'][ne])
    return OUT


def summary(settings):
    """Extra visits with a quarter of CAL and growth of the ratio with a quarter of EVAL, per setting and their ranges."""
    res = {}
    for lab, d in settings:
        c, e = d['n_cal'], d['n_eval']
        res[lab] = dict(extra_visits_cal_10000_vs_40000=c['10000']['mean_visits_eval'] / c['40000']['mean_visits_eval'] - 1,
                        ratio_growth_eval_30000_vs_120000=e['30000']['ratio'] - e['120000']['ratio'])
    rng = lambda k: [min(r[k] for r in res.values()), max(r[k] for r in res.values())]
    return dict(settings=res, ranges={k: rng(k) for k in ('extra_visits_cal_10000_vs_40000', 'ratio_growth_eval_30000_vs_120000')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='readings of one setting')
    r.add_argument('--red', required=True); r.add_argument('--ds', required=True); r.add_argument('--graph', required=True)
    r.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the same setting')
    r.add_argument('--eps', type=float, default=0.05); r.add_argument('--out', required=True)
    r.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    s = sub.add_parser('summary', help='reported differences over settings')
    s.add_argument('settings', nargs='+', help='label=OUT.json of the run command'); s.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.cmd == 'summary':
        res = summary([(lab, json.load(open(p))) for lab, (p,) in labelled_paths(a.settings, 1)])
        if a.out: json.dump(res, open(a.out, 'w'), indent=1)
        print(json.dumps(res, indent=1)); return
    t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from replay import setting_of_frontier
    FR = json.load(open(a.frontier)); S = setting_of_frontier(a.red, a.ds, a.graph, FR, keep_hold=True, seeds=a.seeds)
    OUT = dict(ds=a.ds, graph=a.graph, **sample_size(S, FR, a.eps, log=log))
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

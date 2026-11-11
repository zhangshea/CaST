"""Frontier results of one setting: the chain on SEL, the frontier rules calibrated on CAL, and the bounds on EVAL.

Paper: Algorithm 1 and Theorem 2 (chain), Theorem 3 (calibration at level delta / (2 M) for each of the M risk levels and
each of the two candidate sequences), Theorem 4 (lower bound L), Corollary 1 (ratio U / L). It produces the float values
behind Fig. 3 and the U / L columns of Table II; pipeline/exact_bounds.py recomputes the reported ratios with outward rounding.

Usage:
    python pipeline/frontier.py --red RED_DIR --ds amazon --graph A --config configs/frontier.json --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.bounds import REL, cost_upper_bound, frontier_lower_bound, lagrangian_lower_bound, lagrangian_minimum, single_level_lower_bound
from cast.calibration import fixed_sequence
from cast.chain import bound_multipliers, chain_element, chain_multipliers
from cast.monotone_class import solve_class_lp, table_F
from cast.rules import evaluate_rule, staircase_F, terminal_F
from cast.setting import SPLIT, fit_interface, load_setting


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--config', required=True, help='risk levels, error levels and candidate sequence (configs/frontier*.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layout', default='pool', choices=('pool', 'parts'))
    ap.add_argument('--target', default='X_exact', choices=('X_exact', 'X_recall09'))
    ap.add_argument('--buckets', type=int, default=100, help='B: bucket edges at the quantiles k / B of the FIT scores, merged with the multiples of 0.05 (100 buckets for B = 100, 40 for B = 25, 60 for B = 50)')
    ap.add_argument('--exact-levels', action='store_true', help='bucket edges at exactly the quantiles k / B')
    ap.add_argument('--stride', type=int, default=1, help='keep every stride-th checkpoint')
    ap.add_argument('--model', default='gbdt', choices=('gbdt', 'logreg'), help='score model')
    ap.add_argument('--fit-block', type=int, default=0, help="layout 'pool': which block of 2,500 queries is FIT")
    ap.add_argument('--resplit-seed', type=int, default=None, help="layout 'pool': seed of a further random split of SEL, CAL and EVAL")
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    return ap.parse_args()


def main():
    a = parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    cfg = json.load(open(a.config)); seeds = json.load(open(a.seeds)) if a.seeds else None
    S = load_setting(a.red, a.ds, a.graph, layout=a.layout, target=a.target, fit_block=a.fit_block, stride=a.stride, seeds=seeds,
                     resplit_seed=a.resplit_seed)
    fit_interface(S, n_buckets=a.buckets, model=a.model, exact_levels=a.exact_levels)
    J, H = S.J, S.H; SEL, CAL, EVAL = S.SEL, S.CAL, S.EVAL
    delta, alpha_L, alpha_U = cfg['delta'], cfg['alpha_L'], cfg['alpha_U']
    E = cfg['levels']; M = len(E); EP = cfg['report_levels']
    log(a.ds, a.graph, 'rows', S.n, 'FIT', len(S.FIT), 'SEL', len(SEL), 'CAL', len(CAL), 'EVAL', len(EVAL), 'config', cfg)
    OUT = dict(ds=a.ds, graph=a.graph, layout=a.layout, target=a.target, H=H, J=J, n_buckets=int(S.NB), config=cfg,
               interface=dict(buckets=a.buckets, exact_levels=a.exact_levels, stride=a.stride, model=a.model, fit_block=a.fit_block, resplit_seed=a.resplit_seed),
               n=dict(FIT=len(S.FIT), SEL=len(SEL), CAL=len(CAL), EVAL=len(EVAL)), max_final_visits=int(S.c[:, -1].max()),
               split=SPLIT if a.layout == 'pool' else None)

    # ---- multipliers of the lower bound: negated slopes of the SEL frontier at the risk levels
    LAMS, slopes, by_level = bound_multipliers(S, E)
    K = len(LAMS); OUT['sel_slopes'] = slopes; OUT['multipliers'] = LAMS; OUT['K'] = K; log('multipliers', K, LAMS)

    # ---- chain on SEL (Algorithm 1 for every multiplier of the chain grid)
    LAMG, OUT['chain_grid_extension'] = chain_multipliers(list(by_level.values())); OUT['chain_grid'] = LAMG.tolist()
    log('chain grid', len(LAMG), int(LAMG[0]), int(LAMG[-1]), OUT['chain_grid_extension'])
    chain = {}
    for lam_ in LAMG:
        beta, val, f = chain_element(S, SEL, int(lam_)); chain[int(lam_)] = dict(beta=beta, g=val)
    ks = sorted(chain)
    # Checks: the chain is nested (Theorem 2), and the dynamic program attains the linear-programming minimum (Lemma 3).
    OUT['chain_nested'] = bool(all((chain[ks[i + 1]]['beta'] <= chain[ks[i]]['beta']).all() for i in range(len(ks) - 1)))
    chk = []
    for lam_ in (by_level.get(0.05), by_level.get(0.01)):
        if lam_:
            lv = lagrangian_minimum(S, SEL, float(lam_))
            chk.append(dict(lam=lam_, dp=chain_element(S, SEL, lam_)[1], lp=lv['value'], lp_dual=lv['dual']))
    OUT['dp_equals_lp'] = chk; log('dp vs lp', chk, 'nested', OUT['chain_nested'])

    # ---- frontier rules: both candidate sequences at every risk level, calibrated on CAL at delta / (2 M); U on EVAL at alpha_U / M
    mkC = lambda th, idx: terminal_F(S, idx) if th == 'terminal' else staircase_F(S, chain[th]['beta'], idx)
    sols = {}
    def mkL(th, idx):                      # mixtures of chain elements: solutions of problem (3) on SEL at fixed multiples of eps
        if th == 'terminal': return terminal_F(S, idx)
        if th not in sols: sols[th] = solve_class_lp(S, SEL, float(th), 1e-9)
        return None if sols[th] is None else table_F(S, sols[th]['f'], idx)
    RULES = {'chain': {}, 'mixture': {}}
    for e in E:
        seqs = dict(chain=['terminal'] + [int(l) for l in LAMG[::-1]],
                    mixture=['terminal'] + [round(r_ * e, 6) for r_ in (0.1, 0.2, 0.3, 0.4)] + [round(r_ * e, 6) for r_ in np.round(np.arange(0.50, 1.2001, 0.02), 2)])
        for fam, mk in (('chain', mkC), ('mixture', mkL)):
            th, tr = fixed_sequence(S, seqs[fam], mk, e, idx=CAL, U=S.u_cal, delta=delta / (2 * M))
            if th is None: RULES[fam][e] = dict(calibrated=False); continue
            cost, risk = evaluate_rule(S, mk(th, EVAL), EVAL)
            RULES[fam][e] = dict(calibrated=True, beta=(np.asarray(chain[th]['beta']).tolist() if fam == 'chain' and th != 'terminal' else None), theta=th,
                                 mean_visits=float(cost.mean()), coverage=float(1 - risk.mean()), U=cost_upper_bound(cost, alpha_U / M, H), trace_tail=tr[-3:])
            log('rule', fam, e, {k: v for k, v in RULES[fam][e].items() if k not in ('trace_tail', 'beta')})
    OUT['rules'] = RULES; json.dump(OUT, open(a.out, 'w'), indent=1, default=str)

    # ---- lower bounds on EVAL: ell(lambda_k) at alpha_L / K each, then L(eps) for every eps (Theorem 4)
    ELL = [lagrangian_lower_bound(S, EVAL, float(l_), alpha_L / K) for l_ in LAMS]; OUT['ell'] = [dict(d) for d in ELL]
    BOUNDS = {}; fam = cfg['sequence']
    for e in E:
        r = dict(L=frontier_lower_bound(ELL, e), L_single_level=single_level_lower_bound(S, EVAL, e, alpha_L / M, iters=10).get('L'))
        sol = solve_class_lp(S, EVAL, e); r['empirical_optimum'] = None if sol is None else sol['value']
        for f_, RR in RULES.items():
            ru = RR.get(e, {})
            if ru.get('calibrated'):
                if r['L']: r[f'ratio_{f_}'] = ru['U'] * (1 + REL) / r['L']
                if r['L_single_level']: r[f'ratio_{f_}_single_level'] = ru['U'] * (1 + REL) / (r['L_single_level'] * (1 - REL))
        BOUNDS[str(e)] = r; log('bounds', e, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        OUT['bounds'] = BOUNDS; json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    main_ratios = {str(e): BOUNDS[str(e)].get(f'ratio_{fam}') for e in EP}
    OUT['summary'] = dict(sequence=fam, report_levels=EP, ratios=main_ratios,
                          max_ratio=None if any(v is None for v in main_ratios.values()) else max(main_ratios.values()),
                          all_calibrated=all(RULES[fam].get(e, {}).get('calibrated') for e in EP), joint_confidence=1 - alpha_L - alpha_U - delta,
                          coverage_ok=all(RULES[fam][e]['coverage'] >= 1 - e for e in EP if RULES[fam].get(e, {}).get('calibrated')))
    OUT['terminal_fail_eval'] = float((S.X[EVAL] > S.c[EVAL, -1]).mean())
    OUT['L_curve'] = [(float(e), frontier_lower_bound(ELL, e)) for e in np.round(np.arange(0.005, 0.1501, 0.005), 3)]
    log('summary', OUT['summary']); json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

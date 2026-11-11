"""Readouts of the frontier results of several settings.

Paper: Section VI-B, paragraph "Near-optimality" (range of the maximum of U / L over the report levels, the ratios at
0.02 and 0.01, the EVAL failure rate as a share of the target, the largest saving 1 - L / U left within the class at
0.05), the ablation "Interface and lower bound" (lower bound valid at one risk level against the Lagrangian bound L),
Section VI-A (share of EVAL queries that do not complete by the last checkpoint) and Section IV-B / V-B (the dynamic
program against the linear program at the compared multipliers; cutting-plane iterations of the lower bound and the
gap between its objective and the master value).

Each setting is given as label=FRONTIER.json,EXACT.json: the outputs of pipeline/frontier.py and
pipeline/exact_bounds.py. Ratios are rounded up to three decimals.

Usage:
    python analysis/frontier_readouts.py amazon_A=F.json,X.json amazon_B=... arxiv_A=... arxiv_B=... --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import labelled_paths, up3


def setting_readout(FR, EX):
    cfg = FR['config']; EP = [str(e) for e in cfg['report_levels']]
    X = {e: v for e, v in EX['EVAL']['rules'].items() if v.get('calibrated')}
    ratio = {e: up3(v['ratio_up6']) for e, v in X.items() if v.get('ratio_up6') is not None}
    ru = {e: v for e, v in FR['rules']['chain'].items() if v.get('calibrated')}
    r = dict(ratio=ratio, max_ratio=max(ratio[e] for e in EP if e in ratio), ratio_005=ratio.get('0.05'),
             fail_rate_over_target={e: (1 - v['coverage']) / float(e) for e, v in ru.items()},
             terminal_fail_eval=FR.get('terminal_fail_eval'))
    if '0.05' in X: r['saving_bound_005'] = 1 - X['0.05']['L_down6'] / X['0.05']['U_up6']
    b = FR['bounds']
    r['single_level_over_L'] = {e: b[e]['L_single_level'] / b[e]['L'] - 1 for e in EP if b.get(e, {}).get('L') and b[e].get('L_single_level')}
    r['dp_minus_lp'] = [abs(x['dp'] - x['lp']) for x in FR.get('dp_equals_lp', [])]
    r['lower_bound_iterations'] = [len(d['trace']) for d in FR['ell']]; r['lower_bound_gap'] = [t['gap'] for d in FR['ell'] for t in d['trace']]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('settings', nargs='+', help='label=FRONTIER.json,EXACT.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); res = {}
    for lab, (pf, px) in labelled_paths(a.settings, 2):
        res[lab] = setting_readout(json.load(open(pf)), json.load(open(px)))
        r = res[lab]; print(lab, 'U/L by level', r['ratio'], 'max over the report levels', r['max_ratio'], 'at 0.05', r['ratio_005'])
    rng = lambda xs: [min(xs), max(xs)]
    agg = dict(max_ratio=rng([r['max_ratio'] for r in res.values()]),
               fail_rate_over_target=rng([x for r in res.values() for x in r['fail_rate_over_target'].values()]),
               single_level_over_L=rng([x for r in res.values() for x in r['single_level_over_L'].values()]),
               lower_bound_iterations=rng([x for r in res.values() for x in r['lower_bound_iterations']]),
               lower_bound_gap=rng([x for r in res.values() for x in r['lower_bound_gap']]))
    v = [x for r in res.values() for x in r['dp_minus_lp']]
    if v: agg['dp_minus_lp'] = rng(v)
    for e in ('0.02', '0.01'):
        v = [r['ratio'][e] for r in res.values() if e in r['ratio']]
        if v: agg[f'ratio_{e}'] = rng(v)
    v = [r['saving_bound_005'] for r in res.values() if 'saving_bound_005' in r]
    if v: agg['saving_bound_005'] = rng(v)
    v = [r['terminal_fail_eval'] for r in res.values() if r['terminal_fail_eval'] is not None]
    if v: agg['terminal_fail_eval'] = rng(v)
    json.dump(dict(settings=res, ranges=agg), open(a.out, 'w'), indent=1)
    for k, v in agg.items(): print(k, [float(f'{x:.5g}') for x in v])


if __name__ == '__main__':
    main()

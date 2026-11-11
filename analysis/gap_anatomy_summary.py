"""Ranges of the five terms of U - L over settings and risk levels.

Paper: Section VI-B, paragraph "Near-optimality" (shares of the displacement, the cost bound and the lower bound, of the
rule against the empirical frontier and of the multiplier grid over the 28 pairs of setting and level from 0.03 to
0.10; the ratio of the mixture sequence at 0.01 against the chain).

Usage:
    python analysis/gap_anatomy_summary.py amazon_A=ANATOMY.json amazon_B=... arxiv_A=... arxiv_B=... --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import labelled_paths
from gap_anatomy import TERMS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('settings', nargs='+', help='label=ANATOMY.json (output of analysis/gap_anatomy.py)')
    ap.add_argument('--min-level', type=float, default=0.03, help='smallest risk level of the reported ranges')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = {lab: json.load(open(p))['levels'] for lab, (p,) in labelled_paths(a.settings, 1)}
    c = [v for d in D.values() for v in d.values() if v['sequence'] == 'chain' and v['eps'] >= a.min_level]
    res = dict(pairs=len(c), share={k: [min(x['share'][k] for x in c), max(x['share'][k] for x in c)] for k in TERMS},
               visits={k: [min(x['terms'][k] for x in c), max(x['terms'][k] for x in c)] for k in TERMS})
    st = [x['share']['cost_bound'] + x['share']['displacement'] + x['share']['lower_bound'] for x in c]
    res['share_displacement_cost_bound_lower_bound'] = [min(st), max(st)]; res['max_residual'] = max(abs(x['residual']) for x in c)
    res['ratio_001'] = {lab: dict(chain=d.get('chain_0.01', {}).get('ratio'), mixture=d.get('mixture_0.01', {}).get('ratio')) for lab, d in D.items()}
    json.dump(res, open(a.out, 'w'), indent=1)
    for k in TERMS:
        print(k, 'share %', round(100 * res['share'][k][0], 2), round(100 * res['share'][k][1], 2), 'visits', round(res['visits'][k][0], 2), round(res['visits'][k][1], 2))
    print('displacement + cost bound + lower bound, share %', [round(100 * s, 2) for s in res['share_displacement_cost_bound_lower_bound']],
          'over', res['pairs'], 'pairs; max residual', res['max_residual'])
    for lab, r in res['ratio_001'].items(): print(lab, 'U/L at 0.01: chain', r['chain'], 'mixture', r['mixture'])


if __name__ == '__main__':
    main()

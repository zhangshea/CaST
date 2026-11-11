"""Joint test of the risk target on HOLD over all settings and risk levels, and the bound recomputed on HOLD alone.

Paper: Section VI-B, paragraph "Held-out check" (joint test of the 36 pairs of setting and level, each a one-sided exact
binomial test at level 0.05 / 36; largest HOLD-only bound U / L from 0.03 to 0.10 and at 0.01, rounded up).

Each setting is given as label=HOLDOUT.json, the output of pipeline/holdout.py: failures of the frontier rule of every
level on HOLD and the ratio U / L recomputed on HOLD. No query is read here.

Usage:
    python analysis/holdout_joint_test.py amazon_A=H.json amazon_B=H.json arxiv_A=H.json arxiv_B=H.json --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cast.calibration import binomial_critical
from common import labelled_paths, up3


def joint_test(settings, alpha=0.05):
    """settings: [(label, holdout output)]. Each pair is tested at level alpha / number of pairs."""
    pairs = [(name, e) for name, H_ in settings for e in H_['frontier'] if 'fail' in H_['frontier'][e]]
    A = alpha / len(pairs)
    res = dict(level_each=A, pairs=[], reject_any=False)
    mx_hi, mx_01 = (0, None), (0, None)
    for name, H_ in settings:
        n = int(H_['n_hold'])
        for e, r in H_['frontier'].items():
            if 'fail' not in r: continue
            f = int(r['fail']); c = binomial_critical(n, float(e), A); rej = f >= c
            rh = H_['hold_bounds'][e].get('ratio_hold'); rh = None if rh is None else float(rh)
            res['pairs'].append(dict(setting=name, eps=e, n=n, fail=f, critical=c, reject=rej, ratio_hold=rh))
            res['reject_any'] |= rej
            if rh is not None and float(e) >= 0.03 and rh > mx_hi[0]: mx_hi = (rh, f'{name} {e}')
            if rh is not None and e == '0.01' and rh > mx_01[0]: mx_01 = (rh, name)
    res['max_ratio_hold_003_010'] = dict(value=mx_hi[0], at=mx_hi[1], rounded_up=up3(mx_hi[0]))
    res['max_ratio_hold_001'] = dict(value=mx_01[0], at=mx_01[1], rounded_up=up3(mx_01[0]))
    res['min_slack'] = min(p['critical'] - p['fail'] for p in res['pairs'])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('settings', nargs='+', help='label=HOLDOUT.json')
    ap.add_argument('--alpha', type=float, default=0.05, help='joint level, split evenly over the pairs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = joint_test([(lab, json.load(open(ph))) for lab, (ph,) in labelled_paths(a.settings, 1)], a.alpha)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('pairs', len(res['pairs']), 'level each', res['level_each'], '| reject any:', res['reject_any'], '| min critical - fail:', res['min_slack'])
    print('max HOLD-only ratio 0.03-0.10:', res['max_ratio_hold_003_010'], '| at 0.01:', res['max_ratio_hold_001'])
    for p in res['pairs']:
        if p['eps'] in ('0.01', '0.02'): print(p['setting'], p['eps'], p['fail'], '<', p['critical'])


if __name__ == '__main__':
    main()

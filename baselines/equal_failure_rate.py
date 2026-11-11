"""Visits of the compared rules over ours at equal EVAL failure rates.

Each rule gives a set of points (EVAL failure rate, mean EVAL visits): its calibrated rules at the nine risk levels, or,
for Patience, one point per value of its parameter. The logarithm of the visits is interpolated linearly in the failure
rate between neighbouring points, never outside the measured range of a rule, and the ratio to the same interpolation of
the frontier rules of our method is read at the failure rates 0.03 to 0.08.

Paper: Section VI-B, paragraph "Cost and latency at a common risk target": interpolated to equal EVAL failure rates
from 0.03 to 0.08, the first group spends 1.04 to 1.68 times our visits, the distance-adaptive rule 1.08 to 1.44,
Patience 1.17 to 1.40, and DARTH and DARTH+ on graph A 1.28 to 1.63.

Inputs, by file name inside the given directories (<s> = <ds>_<graph>):
  --levels-dir     levels_<s>*.json                  outputs of baselines/levels.py (one or several files per setting)
  --adaptive-dir   distance_adaptive_<s>.json        outputs of baselines/distance_adaptive.py calibrate
  --patience       one file                          output of baselines/patience.py points
  --recall-dir     darth_report_<s>.json, darthplus_report_<s>.json   outputs of the report step of baselines/darth.py and darth_plus.py

Usage:
  python baselines/equal_failure_rate.py --levels-dir DIR [--adaptive-dir DIR] [--patience POINTS.json] [--recall-dir DIR] --out OUT.json
"""
import argparse
import glob
import json
import math
import os

FIRST_GROUP = ['level_cap', 'level', 'laet_style', 'fixed']
RATES = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]


def curve(pts):
    """Log-linear interpolation of the visits in the failure rate through the points (rate, visits); None outside their range."""
    pts = sorted(pts)
    def f(r):
        if r < pts[0][0] or r > pts[-1][0]:
            return None
        for (a, ca), (b, cb) in zip(pts, pts[1:]):
            if a <= r <= b:
                t = 0 if b == a else (r - a) / (b - a)
                return math.exp((1 - t) * math.log(ca) + t * math.log(cb))
        return None
    return f


def ratios(c, cs):
    return {str(r): (None if c(r) is None or cs(r) is None else c(r) / cs(r)) for r in RATES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--levels-dir', required=True); ap.add_argument('--adaptive-dir', default=None); ap.add_argument('--patience', default=None)
    ap.add_argument('--recall-dir', default=None); ap.add_argument('--settings', nargs='+', default=['amazon_A', 'amazon_B', 'arxiv_A', 'arxiv_B'])
    ap.add_argument('--recall-graph', default='A', help='graph on which DARTH and DARTH+ were run'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {}
    for s in a.settings:
        # ---- first group against the frontier rules of our method
        pts = {k: [] for k in ['frontier'] + FIRST_GROUP}
        same = {k: {} for k in FIRST_GROUP}
        for p in glob.glob(os.path.join(a.levels_dir, f'levels_{s}*.json')):
            d = json.load(open(p))
            for e, row in d['levels'].items():
                if not isinstance(row['frontier'], dict): continue          # no frontier rule at this level
                pts['frontier'].append((row['frontier']['fail_rate'], row['frontier']['mean_visits']))
                for k in FIRST_GROUP:
                    x = row[k]
                    if x.get('calibrated') and not x.get('saturated') and 'mean_visits' in x:
                        pts[k].append((x['fail_rate'], x['mean_visits']))
                        same[k][e] = x['mean_visits'] / row['frontier']['mean_visits']
        stair = pts['frontier']; cs = curve(stair)
        c = {k: curve(v) for k, v in pts.items()}
        res[s] = {}
        for k in FIRST_GROUP:
            row = {}
            for r in RATES:
                x, y = c[k](r), c['frontier'](r)
                row[str(r)] = None if x is None or y is None else x / y
            res[s][k] = dict(same_failure=row, same_target=same[k], n_levels=len(pts[k]))
        # ---- distance-adaptive rule on the fine grid of gamma
        if a.adaptive_dir:
            bf = json.load(open(os.path.join(a.adaptive_dir, f'distance_adaptive_{s}.json')))['levels']
            pd = [(v['fail_rate'], v['mean_visits']) for v in bf.values() if v.get('calibrated')]
            res[s]['distance_adaptive'] = dict(n_levels=len(pd), same_failure=ratios(curve(pd), cs), same_target={k: v.get('visits_over_frontier') for k, v in bf.items()})
        # ---- Patience, swept over its parameter
        if a.patience:
            assert len(stair) == 9, (s, len(stair))
            P = json.load(open(a.patience)); pp = list(zip(P[s]['fail_rate'], P[s]['mean_visits']))
            res[s]['patience'] = dict(n_points=len(pp), same_failure=ratios(curve(pp), cs))
        # ---- DARTH and DARTH+ at the nine risk levels
        if a.recall_dir and s.endswith('_' + a.recall_graph):
            assert len(stair) == 9, (s, len(stair))
            for nm, path in (('darth', os.path.join(a.recall_dir, f'darth_report_{s}.json')), ('darth_plus', os.path.join(a.recall_dir, f'darthplus_report_{s}.json'))):
                lv = [v for v in json.load(open(path))['levels'].values() if isinstance(v['frontier'], dict)]
                rp = [(v['rule']['fail_rate'], v['rule']['mean_visits']) for v in lv if v['rule'].get('calibrated')]
                st9 = sorted((v['frontier']['fail_rate'], v['frontier']['mean_visits']) for v in lv)
                assert all(abs(x[0] - y[0]) <= 1e-12 and abs(x[1] - y[1]) <= 1e-6 * y[1] for x, y in zip(st9, sorted(stair))), (nm, s)   # the same frontier rules
                res[s][nm] = dict(n_points=len(rp), same_failure=ratios(curve(rp), cs))
    fam = dict(first_group=FIRST_GROUP, distance_adaptive=['distance_adaptive'], patience=['patience'], darth=['darth'], darth_plus=['darth_plus'])
    allv = {nm: [v for s in res for k, x in res[s].items() if k in ks for v in x['same_failure'].values() if v is not None] for nm, ks in fam.items()}
    res['summary'] = {nm: dict(n=len(v), min=min(v) if v else None, max=max(v) if v else None) for nm, v in allv.items()}
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res['summary'], indent=1))
    for s in a.settings:
        for k, v in res[s].items():
            print(s, k, 'same failure', [None if x is None else round(x, 3) for x in v['same_failure'].values()],
                  *(('same target 0.05', round(v['same_target'].get('0.05') or float('nan'), 3)) if 'same_target' in v else ()))


if __name__ == '__main__':
    main()

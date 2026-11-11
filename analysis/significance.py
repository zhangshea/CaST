"""Significance of the visit ratios of Table II under a Bonferroni correction over the whole table.

Paper: caption of Table II ("every ratio above one is significant at level 0.025 / 102": 17 rows below the four main
rows, six compared rules each). With z = Phi^{-1}(1 - level / family), a rule is costlier than ours when
ratio - z se > 1 and cheaper when ratio + z se < 1, se being the standard deviation of the paired bootstrap ratios.

Each row of the table is given as label=COMMON_TARGET.json[,DISTANCE_ADAPTIVE.json]: the output of
pipeline/common_target.py for the setting (table[rule].visit_ratio) and, when given, the output of
baselines/distance_adaptive.py (table.distance_adaptive.visit_ratio), whose fine-grid rule replaces the distance-adaptive rule of
the first file, as in Table II.

Usage:
    python analysis/significance.py "Amazon, kappa 2=CT.json,DA.json" ... --out OUT.json
"""
import argparse
import json
from statistics import NormalDist

RULES = ('fixed', 'laet_style', 'level', 'level_cap', 'list_size', 'distance_adaptive')


def decide(vr, z):
    lo, hi = vr['ratio'] - z * vr['se'], vr['ratio'] + z * vr['se']
    return dict(ratio=vr['ratio'], se=vr['se'], lower=lo, upper=hi, costlier=lo > 1, cheaper=hi < 1)


def row(T, fine, z):
    """Decisions of one row: T the table of pipeline/common_target.py, fine the fine-grid distance-adaptive record or None."""
    res = {}
    for k in RULES:
        v = fine if (k == 'distance_adaptive' and fine is not None) else T.get(k)
        if v is None or not v.get('calibrated', True): res[k] = dict(calibrated=False); continue
        res[k] = decide(v['visit_ratio'], z)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rows', nargs='+', help='label=COMMON_TARGET.json[,DISTANCE_ADAPTIVE.json]')
    ap.add_argument('--level', type=float, default=0.025, help='one-sided level before the correction')
    ap.add_argument('--family', type=int, default=102, help='number of ratios the correction covers')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = NormalDist().inv_cdf(1 - a.level / a.family)
    out = dict(z=z, level=a.level, family=a.family, rows={})
    for arg in a.rows:
        lab, paths = arg.split('=', 1); paths = paths.split(',')
        T = json.load(open(paths[0]))['table']; fine = json.load(open(paths[1]))['table']['distance_adaptive'] if len(paths) > 1 else None
        out['rows'][lab] = row(T, fine, z)
    cells = [(lab, k, v) for lab, r in out['rows'].items() for k, v in r.items() if 'ratio' in v]
    above = [c for c in cells if c[2]['ratio'] > 1]
    out['summary'] = dict(ratios=len(cells), above_one=len(above), above_one_significant=sum(c[2]['costlier'] for c in above),
                          below_one=[(lab, k, v['ratio'], v['cheaper']) for lab, k, v in cells if v['ratio'] <= 1],
                          smallest_margin_in_se=min(((v['ratio'] - 1) / v['se'], lab, k) for lab, k, v in above) if above else None)
    json.dump(out, open(a.out, 'w'), indent=1)
    for lab, r in out['rows'].items():
        print(lab, {k: (round(v['ratio'], 3), round(v['lower'], 3), 'costlier' if v['costlier'] else ('cheaper' if v['cheaper'] else 'undecided')) for k, v in r.items() if 'ratio' in v})
    print('z', round(z, 4), out['summary'])


if __name__ == '__main__':
    main()

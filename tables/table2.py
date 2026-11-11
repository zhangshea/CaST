"""Rows of Table II: visits and optimality-gap bound on EVAL, one row per workload.

Paper: Table II. For each setting directory holding the outputs of pipeline/exact_bounds.py (exact.json) and
pipeline/common_target.py (common_target.json): the largest U / L over the evaluated risk levels from 0.03 to 0.10 and the
value at 0.05 (both rounded up to three decimals), the mean EVAL visits of CaST at eps = delta = 0.05, and the visit ratio
of every other rule to CaST, in the column order of the table. The JSON summary adds the one-sided 2.5% bootstrap lower
bound and the standard error of every ratio, which the significance statement of the caption uses.
A file distance_adaptive.json (output of baselines/distance_adaptive.py, finer gamma grid) replaces the
distance-adaptive cell when present.

Usage:
    python tables/table2.py "Amazon, A"=DIR "arXiv, A"=DIR ...        (prints LaTeX rows, then a JSON summary)
"""
import json
import math
import os
import sys

ORDER = [('fixed', 'Fixed'), ('laet_style', 'LAET-style'), ('level', 'Level'), ('level_cap', 'Level+cap'), ('list_size', 'List size'),
         ('distance_adaptive', 'Dist.-adaptive')]


def up3(x): return math.ceil(x * 1000 - 1e-9) / 1000


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__); return
    rows, summ = [], {}
    for arg in sys.argv[1:]:
        lab, d = arg.split('=', 1)
        X = json.load(open(f'{d}/exact.json'))['EVAL']['rules']; T = json.load(open(f'{d}/common_target.json'))['table']
        fine = f'{d}/distance_adaptive.json'
        if os.path.exists(fine): T = dict(T, distance_adaptive=json.load(open(fine))['table']['distance_adaptive'])
        lv = {float(e): v for e, v in X.items() if v.get('calibrated')}
        inr = [v['ratio_up6'] for e, v in lv.items() if 0.03 - 1e-9 <= e <= 0.10 + 1e-9]
        mx = up3(max(inr)); r05 = up3(lv[0.05]['ratio_up6']) if 0.05 in lv else None
        st = T['cast']; cells = []
        s = dict(max_ratio=mx, levels=sorted(lv), ratio_005=r05, cast_visits=st['mean_visits'], cast_fail_rate=st['fail_rate'])
        for k, nm in ORDER:
            v = T.get(k)
            if v is None or not v.get('calibrated', True): cells.append('--'); continue
            r = v['visit_ratio']; lb = r['lower_bound_one_sided']
            cells.append(f"{r['ratio']:.3f}")
            s[k] = dict(ratio=r['ratio'], lower_bound=lb, se=r.get('se'), visits=v['mean_visits'], fail_rate=v['fail_rate'])
        rows.append(f"{lab} & " + ' & '.join(cells) + ' & ' + f"{st['mean_visits']:,.0f}".replace(',', '{,}') + f" & {mx:.3f} & " + ('--' if r05 is None else f"{r05:.3f}") + "\\\\")
        summ[lab] = s
    print('\n'.join(rows)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()

"""Aggregates of the richer learned policies over the settings, from the outputs of baselines/learned_policies.py.

Paper: Section VI-B, "Ablation study", paragraph "Richer learned policies". Over the settings and the seven risk
levels from 0.03 to 0.10, with ratios of mean visits on the second half of CAL:
  fully fitted policy over our rule: smallest, largest and median ratio;
  cross-fitted and declared-state policies over our rule: the median per setting, and its range over the settings;
  learning curves at eps = 0.05: the change of the out-of-sample Lagrangian value of the class when SEL shrinks from
  20,000 to 5,000 queries, and the value of the cross-fitted policy over that of the class at both sizes.

Usage:
  python baselines/learned_policies_summary.py POLICIES_amazon_A.json POLICIES_amazon_B.json POLICIES_arxiv_A.json POLICIES_arxiv_B.json [--out SUMMARY.json]
"""
import argparse
import json
import statistics as st

REPORT_LEVELS = ['0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.1']
FAMILIES = ('fully_fitted', 'cross_fitted', 'declared_state')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='outputs of baselines/learned_policies.py, one per setting'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    over, lc = {f: {} for f in FAMILIES}, {}
    for p in a.inputs:
        o = json.load(open(p)); tag = f"{o['ds']}_{o['graph']}"
        for f in FAMILIES:           # visits of the policy over ours = 1 / (ours over the policy)
            over[f][tag] = [1.0 / o['compare'][e][f'cast_over_{f}'][0] for e in REPORT_LEVELS if f'cast_over_{f}' in o['compare'][e]]
        c = o['learning_curve_005']; lc[tag] = dict(class_change=abs(c['cast']['5000'] / c['cast']['20000'] - 1),
                                                    gap_20000=c['cross_fitted']['20000'] / c['cast']['20000'], gap_5000=c['cross_fitted']['5000'] / c['cast']['5000'])
    allv = [v for t in over['fully_fitted'].values() for v in t]
    res = dict(fully_fitted=dict(n=len(allv), min=min(allv), max=max(allv), median=st.median(allv)),
               median_per_setting={f: {t: st.median(v) for t, v in over[f].items()} for f in FAMILIES}, learning_curve_005=lc)
    for f in ('cross_fitted', 'declared_state'):
        m = res['median_per_setting'][f].values(); res[f] = dict(min_of_medians=min(m), max_of_medians=max(m))
    res['class_value_change_max'] = max(v['class_change'] for v in lc.values())
    res['cross_fitted_over_class_20000'] = [min(v['gap_20000'] for v in lc.values()), max(v['gap_20000'] for v in lc.values())]
    res['cross_fitted_over_class_5000'] = [min(v['gap_5000'] for v in lc.values()), max(v['gap_5000'] for v in lc.values())]
    print('fully fitted over ours:', res['fully_fitted'])
    for f in ('cross_fitted', 'declared_state'): print(f, 'over ours, median per setting:', res[f])
    print('class value, change from 20,000 to 5,000 SEL queries (largest):', res['class_value_change_max'])
    print('cross-fitted over class, 20,000 SEL queries:', res['cross_fitted_over_class_20000'], '5,000:', res['cross_fitted_over_class_5000'])
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

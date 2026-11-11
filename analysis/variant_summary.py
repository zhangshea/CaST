"""Summary over frontier runs of the main settings under other error levels, FIT samples and interfaces.

Paper: Section VI-B, paragraph "Near-optimality" (joint level 95%; three further FIT samples) and the ablation
"Interface and lower bound" (40, 60 or 200 buckets, J = 50 or 25, logistic-regression score, checkpoint and flag alone).

Reads every <variant>_<ds>_<graph>.json of a directory written by analysis/run_variants.py (outputs of
pipeline/frontier.py). Per run: the float ratio U / L of the chain rule at every level, its maximum over the report
levels, the value at 0.05, the EVAL mean visits of the rule calibrated at 0.05, L(0.05) and the empirical class optimum
at 0.05; each variant is also compared with the run `main` of the same setting.

Usage:
    python analysis/variant_summary.py RUN_DIR --out SUMMARY.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import up3

INTERFACE_VARIANTS = ('buckets40', 'buckets60', 'buckets200', 'J50', 'J25', 'logreg')


def run_record(d):
    bounds = d['bounds']; ru = d['rules']['chain']
    ratios = {e: bounds[e].get('ratio_chain') for e in bounds}
    return dict(interface=d.get('interface'), J=d.get('J'), config={k: d['config'][k] for k in ('delta', 'alpha_U', 'alpha_L')},
                ratios=ratios, max_ratio=d['summary']['max_ratio'], ratio_005=ratios.get('0.05'),
                mean_visits_005=ru.get('0.05', {}).get('mean_visits'), coverage_005=ru.get('0.05', {}).get('coverage'),
                L_005=bounds.get('0.05', {}).get('L'), empirical_optimum_005=bounds.get('0.05', {}).get('empirical_optimum'),
                fail_rates={e: round(1 - v['coverage'], 4) for e, v in ru.items() if v.get('calibrated')})


def summarize(res):
    """Compare every run with the run `main` of its setting, and collect the reported ranges."""
    for n, r in res.items():
        if n.startswith('main_') or n.startswith('level95_'): continue
        c = res.get('main_' + n.split('_', 1)[1])
        if c: r['vs_main'] = dict(mean_visits_005=r['mean_visits_005'] / c['mean_visits_005'], L_005=r['L_005'] / c['L_005'], max_ratio_main=c['max_ratio'],
                                  empirical_optimum_005=(r['empirical_optimum_005'] / c['empirical_optimum_005']) if r['empirical_optimum_005'] and c['empirical_optimum_005'] else None)
    for n, r in res.items():
        if n.startswith('level95_'):
            c = res.get('main_' + n.split('_', 1)[1])
            if c: r['vs_main'] = dict(max_increase=r['max_ratio'] - c['max_ratio'], max_increase_rounded_up=round(up3(r['max_ratio']) - up3(c['max_ratio']), 3),
                                      mean_visits_005=r['mean_visits_005'] / c['mean_visits_005'],
                                      per_level_increase={e: (r['ratios'][e] - c['ratios'][e]) if r['ratios'][e] and c['ratios'][e] else None for e in r['ratios']})
    of = lambda *vs: {n: r for n, r in res.items() if n.split('_', 1)[0] in vs}
    agg = {}
    m = of('main')
    if m: agg['main'] = dict(max_ratio=[min(r['max_ratio'] for r in m.values()), max(r['max_ratio'] for r in m.values())])
    c95 = {n: r for n, r in of('level95').items() if 'vs_main' in r}
    if c95: agg['level95'] = dict(max_ratio=max(r['max_ratio'] for r in c95.values()), max_increase=max(r['vs_main']['max_increase'] for r in c95.values()),
                                  max_increase_rounded_up=max(r['vs_main']['max_increase_rounded_up'] for r in c95.values()),
                                  mean_visits_005_unchanged=all(r['vs_main']['mean_visits_005'] == 1.0 for r in c95.values()))
    f = of('fit1', 'fit2', 'fit3')
    if f: agg['fit_samples'] = dict(max_ratio=[min(r['max_ratio'] for r in f.values()), max(r['max_ratio'] for r in f.values())])
    v = {n: r for n, r in of(*INTERFACE_VARIANTS).items() if 'vs_main' in r}
    if v: agg['interface'] = dict(mean_visits_005_vs_main=[min(r['vs_main']['mean_visits_005'] for r in v.values()), max(r['vs_main']['mean_visits_005'] for r in v.values())],
                                  max_ratio=max(r['max_ratio'] for r in v.values()))
    fo = {n: r for n, r in of('flagonly').items() if 'vs_main' in r}
    if fo: agg['flag_only'] = {n: dict(empirical_optimum_005_vs_main=r['vs_main']['empirical_optimum_005'], ratio_005=r['ratio_005']) for n, r in fo.items()}
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); res = {}
    for f in sorted(glob.glob(os.path.join(a.run_dir, '*.json'))):
        if os.path.abspath(f) == os.path.abspath(a.out) or os.path.basename(f).startswith('resplit') or '_' not in os.path.basename(f): continue
        d = json.load(open(f))
        if 'summary' not in d or 'bounds' not in d: continue
        res[os.path.basename(f)[:-5]] = run_record(d)
    agg = summarize(res)
    json.dump(dict(runs=res, reported=agg), open(a.out, 'w'), indent=1)
    for n, r in sorted(res.items()):
        print(n, r['J'], round(r['max_ratio'], 4) if r['max_ratio'] else None, round(r['ratio_005'], 4) if r['ratio_005'] else None,
              round(r['mean_visits_005'], 1) if r['mean_visits_005'] else None,
              {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.get('vs_main', {}).items() if k != 'per_level_increase'})
    if 'main' in agg: print('main: maximum of U/L over the report levels, by setting', [up3(x) for x in agg['main']['max_ratio']])
    if 'level95' in agg: print('joint level 95%: maximum', up3(agg['level95']['max_ratio']), 'largest increase', round(agg['level95']['max_increase'], 4), 'of the rounded maxima', agg['level95']['max_increase_rounded_up'],
                               'mean visits of the rule at 0.05 unchanged', agg['level95']['mean_visits_005_unchanged'])
    if 'fit_samples' in agg: print('further FIT samples: maximum of U/L', [up3(x) for x in agg['fit_samples']['max_ratio']])
    if 'interface' in agg: print('interface variants: visits at 0.05 over main', [round(x, 4) for x in agg['interface']['mean_visits_005_vs_main']],
                                 'maximum of U/L', up3(agg['interface']['max_ratio']))
    for n, r in agg.get('flag_only', {}).items(): print(n, 'empirical optimum at 0.05 over main', round(r['empirical_optimum_005_vs_main'], 4), 'U/L at 0.05', up3(r['ratio_005']))


if __name__ == '__main__':
    main()

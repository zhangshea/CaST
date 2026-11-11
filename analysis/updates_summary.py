"""Index updates: the recalibrated rule against a full refit on samples logged on the updated index.

Paper: Section VI-B, paragraph "Query shift and index updates" ("After updates, the recalibrated rule spends 0.92 to 1.03
times the visits of a full refit on FIT, SEL and CAL samples logged on the updated index"). Each update setting is given as
label=SHIFT.json,REFIT.json: the output of analysis/shift_updates.py and the output of pipeline/frontier.py run on the
reduced arrays logged on the updated index (score, chain and calibration all refitted there).

Usage:
    python analysis/updates_summary.py amazon_insert1=SHIFT.json,REFIT.json ... --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import labelled_paths


def setting_summary(S, F):
    r = dict(summary=S['summary'],
             refit_max_ratio_003_010=max(v for e, v in F['summary']['ratios'].items() if 0.03 <= float(e) <= 0.10),
             refit_all_calibrated=F['summary']['all_calibrated'], levels={})
    for e in S['levels']:
        k = str(e); s = S['before_recalibration'][k]; rr = S['recalibrated'][k]; rr = rr if rr.get('calibrated') else None
        fr = F['rules']['chain'].get(k, {}); a = S['reference'][k]
        r['levels'][k] = dict(reference_mean_visits=a['mean_visits'], reference_fail_rate=a['fail_rate'], before_mean_visits=s['mean_visits'], before_fail_rate=s['fail_rate'],
                              before_p=s['p_one_sided'], before_reject=s['reject'],
                              recalibrated_mean_visits=rr['mean_visits'] if rr else None, recalibrated_fail_rate=rr['fail_rate'] if rr else None,
                              recalibrated_reject=rr['reject'] if rr else None, recalibrated_over_before=rr['cost_ratio_vs_before'] if rr else None,
                              refit_mean_visits=fr.get('mean_visits'),
                              recalibrated_over_refit=(rr['mean_visits'] / fr['mean_visits']) if rr and fr.get('mean_visits') else None,
                              refit_ratio=F['summary']['ratios'].get(k))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('settings', nargs='+', help='label=SHIFT.json,REFIT.json'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); res = {}
    for lab, (ps, pf) in labelled_paths(a.settings, 2):
        r = res[lab] = setting_summary(json.load(open(ps)), json.load(open(pf))); L = r['levels']
        print(lab, 'rejections before', r['summary']['rejections_before'], 'after recalibration', r['summary']['rejections_recalibrated'],
              'refit max U/L', round(r['refit_max_ratio_003_010'], 3),
              'recalibrated / before', [round(v['recalibrated_over_before'], 3) for v in L.values() if v['recalibrated_over_before']],
              'recalibrated / refit', [round(v['recalibrated_over_refit'], 3) for v in L.values() if v['recalibrated_over_refit']])
    x = [v['recalibrated_over_refit'] for r in res.values() for v in r['levels'].values() if v['recalibrated_over_refit']]
    out = dict(settings=res, recalibrated_over_refit=[min(x), max(x)] if x else None)
    json.dump(out, open(a.out, 'w'), indent=1); print('recalibrated / refit over all settings and levels', out['recalibrated_over_refit'])


if __name__ == '__main__':
    main()

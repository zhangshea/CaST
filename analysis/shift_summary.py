"""Counts over the readouts of the query shifts and index updates.

Paper: Section VI-B, paragraph "Query shift and index updates" (levels at which the risk test rejects the target before
recalibration, per change and in total; levels without a recalibrated rule; rejections after recalibration; extra cost
before recalibration where no level is rejected).

Usage:
    python analysis/shift_summary.py title_amazon=OUT.json replace5_amazon=OUT.json ... [--out SUMMARY.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import labelled_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('readouts', nargs='+', help='label=OUT.json of analysis/shift_title.py, shift_updates.py or shift_official.py')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(); res = {}
    for lab, (p,) in labelled_paths(a.readouts, 1):
        d = json.load(open(p)); b = {k: v for k, v in d['before_recalibration'].items() if v.get('calibrated')}; r = d['recalibrated']
        res[lab] = dict(levels=len(b), rejections_before=sum(v['reject'] for v in b.values()),
                        fail_rate_over_target_before={k: v['fail_rate'] / float(k) for k, v in b.items()},
                        no_recalibrated_rule=[k for k in b if not r[k].get('calibrated')],
                        rejections_recalibrated=sum(v['reject'] for k, v in r.items() if v.get('calibrated')),
                        fail_rate_over_target_recalibrated={k: v['fail_rate'] / float(k) for k, v in r.items() if v.get('calibrated')},
                        extra_cost_before=[min(v['cost_ratio_vs_reference'] for v in b.values()) - 1, max(v['cost_ratio_vs_reference'] for v in b.values()) - 1])
        print(lab, 'rejected before recalibration', res[lab]['rejections_before'], 'of', res[lab]['levels'], '| no recalibrated rule at', res[lab]['no_recalibrated_rule'],
              '| rejected after', res[lab]['rejections_recalibrated'], '| extra cost before', [round(x, 4) for x in res[lab]['extra_cost_before']])
    rej = {lab: r for lab, r in res.items() if r['rejections_before']}
    tot = dict(changes_with_rejections=sorted(rej), rejections_before=sum(r['rejections_before'] for r in rej.values()), levels=sum(r['levels'] for r in rej.values()),
               rejections_recalibrated=sum(r['rejections_recalibrated'] for r in res.values()))
    fam = {lab: r for lab, r in res.items() if lab.startswith(('title', 'replace'))}      # family fixed by name, not by outcome
    tot['title_and_replacements'] = dict(changes=sorted(fam), rejections_before=sum(r['rejections_before'] for r in fam.values()),
                                         levels=sum(r['levels'] for r in fam.values()))
    print(tot)
    if a.out: json.dump(dict(readouts=res, total=tot), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

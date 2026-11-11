"""Optimality-gap bound over random re-splits of SEL, CAL and EVAL.

Paper: Section VI-B, paragraph "Near-optimality" (ten further random splits on graph A, FIT and HOLD fixed: range and
median of the maximum of U / L over the report levels; failure rate below the target at all levels on every split).

Reads resplit<seed>_<ds>_<graph>.json (outputs of pipeline/frontier.py --resplit-seed <seed>) from a directory.

Usage:
    python analysis/resplit_summary.py RUN_DIR --out SUMMARY.json [--ds amazon arxiv] [--graph A] [--seeds 1 2 ... 10]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import up3


def summarize(runs):
    """runs: the frontier outputs of the re-splits of one setting."""
    ul = [d['summary'] for d in runs]; E = list(ul[0]['ratios'])
    a = [u['ratios']['0.05'] for u in ul] if '0.05' in E else None
    m = [max(u['ratios'][e] for e in E) for u in ul]
    # EVAL failure rate over the target, for every calibrated rule of the chain at every risk level of every split
    rel = [(1 - r['coverage']) / float(e) for d in runs for e, r in d['rules']['chain'].items() if r.get('calibrated')]
    return dict(at_005=a, max_over_report_levels=m, median_005=st.median(a) if a else None, median_max=st.median(m),
                by_level={e: dict(median=st.median([u['ratios'][e] for u in ul]), max=max(u['ratios'][e] for u in ul)) for e in E},
                all_calibrated=all(u['all_calibrated'] for u in ul),
                calibrated_at_every_level=all(r.get('calibrated') for d in runs for r in d['rules']['chain'].values()),
                fail_rate_over_target=[min(rel), max(rel)], all_fail_rates_within_target=bool(max(rel) <= 1.0), rules=len(rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('--out', required=True)
    ap.add_argument('--ds', nargs='+', default=['amazon', 'arxiv']); ap.add_argument('--graph', default='A')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    a = ap.parse_args(); res = {}
    for ds in a.ds:
        res[ds] = summarize([json.load(open(os.path.join(a.run_dir, f'resplit{s}_{ds}_{a.graph}.json'))) for s in a.seeds])
    json.dump(res, open(a.out, 'w'), indent=1)
    for ds, r in res.items():
        print(ds, 'U/L (rounded up) at 0.05: min, median, max', up3(min(r['at_005'])), up3(r['median_005']), up3(max(r['at_005'])),
              '| maximum over the report levels: min, median, max', up3(min(r['max_over_report_levels'])), up3(r['median_max']), up3(max(r['max_over_report_levels'])),
              'failure rate over target', [round(x, 3) for x in r['fail_rate_over_target']], 'rules', r['rules'])


if __name__ == '__main__':
    main()

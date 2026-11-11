"""A single visit budget for all queries against the median completion time.

Paper: Section I ("a budget calibrated to risk 0.05 must be 6,781 and 5,978 visits, 7.1 and 5.8 times the median
completion time on 10,000 evaluation queries"). The budget is that of the rule Fixed, calibrated on CAL at eps = 0.05
and delta = 0.05 by the common fixed-sequence test (Section VI-A); the median completion time is taken over the first
10,000 EVAL queries, a query that never completes counting as infinite.

Usage:
    python analysis/intro_budget.py --red RED_DIR --ds amazon arxiv --graph A --out OUT.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def budget_and_median(S, eps=0.05, delta=0.05, n_queries=10000):
    from cast.compared import calibrate_fixed
    X = S.X[S.EVAL[:n_queries]].astype(float); X[X >= 2 ** 31 - 1] = np.inf
    med = float(np.median(X)); rec = calibrate_fixed(S, eps, delta); budget = int(rec['budget'])
    return dict(n=int(len(X)), median_X=med, budget=budget, budget_over_median=budget / med, not_complete=int(np.isinf(X).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays'); ap.add_argument('--ds', nargs='+', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--queries', type=int, default=10000, help='number of EVAL queries of the median')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    a = ap.parse_args()
    from cast.setting import load_setting
    seeds = json.load(open(a.seeds)) if a.seeds else None
    res = {ds: budget_and_median(load_setting(a.red, ds, a.graph, seeds=seeds), a.eps, a.delta, a.queries) for ds in a.ds}
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

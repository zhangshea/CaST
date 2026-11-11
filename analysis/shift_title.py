"""Title queries on graph A: the frontier rules calibrated on item queries, without and with recalibration.

Paper: Section VI-B, paragraph "Query shift and index updates" and Fig. 5 ("we query graph A with only the titles of the
CAL items and of 40,000 EVAL items"). The title queries are given as groups of reduced arrays: the CAL groups hold the
titles of the CAL items in the order of CAL, the EVAL groups the titles of the first EVAL items in the order of EVAL
(groups titleshift_cal and titleshift_eval of data/prepare_title_queries.py).
The chain uses the multipliers recorded by the frontier run.

Usage:
    python analysis/shift_title.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --shift-red TITLE_RED_DIR --out OUT.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='reduced arrays of the item queries')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the item queries')
    ap.add_argument('--shift-red', required=True, help='reduced arrays of the title queries')
    ap.add_argument('--cal-groups', nargs='+', default=['titleshift_cal']); ap.add_argument('--eval-groups', nargs='+', default=['titleshift_eval'])
    ap.add_argument('--alpha', type=float, default=0.05, help='level of the risk tests, split over the risk levels')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    return ap


def load(a, FR):
    """Setting before the change, multipliers of the chain, the title CAL and EVAL queries and the reference EVAL rows."""
    import numpy as np
    from replay import setting_of_frontier
    from shift import load_groups
    S = setting_of_frontier(a.red, a.ds, a.graph, FR, keep_hold=True, seeds=a.seeds)
    cal = load_groups(a.shift_red, a.ds, a.graph, a.cal_groups, FR['target']); ev = load_groups(a.shift_red, a.ds, a.graph, a.eval_groups, FR['target'])
    return S, np.array(FR['chain_grid'], np.int64), cal, ev, S.EVAL[:len(ev['c'])]


def main():
    a = parse_args().parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from shift import shift_readout
    FR = json.load(open(a.frontier)); S, mult, cal, ev, ref = load(a, FR)
    OUT = dict(ds=a.ds, graph=a.graph, change='title queries', **shift_readout(S, FR, mult, cal, ev, ref, a.alpha, log))
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

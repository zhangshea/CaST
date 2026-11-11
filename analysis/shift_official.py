"""Official queries of YFCC-10M: the frontier rules calibrated on held-out items, without and with recalibration.

Paper: Section VI-B, paragraph "Query shift and index updates" ("Rules calibrated on held-out queries keep the target at
all five levels on the official EVAL queries at 2.0% to 2.8% higher cost"). The setting of the held-out queries (layout
'parts') supplies the model and the chain, on the multiplier grid stored in its frontier output; the official CAL and
EVAL queries are two groups of their own directory of reduced arrays (`cal` and `eval` when the logs of the groups
official_cal and official_eval are reduced with --strip-prefix official_).

Usage:
    python analysis/shift_official.py --red RED_DIR --ds yfcc10m --graph A --frontier FRONTIER.json --shift-red OFFICIAL_RED_DIR --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='reduced arrays of the held-out queries')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the held-out queries')
    ap.add_argument('--shift-red', required=True, help='reduced arrays of the official queries')
    ap.add_argument('--cal-group', default='cal'); ap.add_argument('--eval-group', default='eval')
    ap.add_argument('--alpha', type=float, default=0.05, help='level of the risk tests, split over the risk levels')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from replay import setting_of_frontier
    from shift import load_groups, shift_readout
    FR = json.load(open(a.frontier)); S = setting_of_frontier(a.red, a.ds, a.graph, FR, seeds=a.seeds)
    cal = load_groups(a.shift_red, a.ds, a.graph, [a.cal_group], FR['target']); ev = load_groups(a.shift_red, a.ds, a.graph, [a.eval_group], FR['target'])
    OUT = dict(ds=a.ds, graph=a.graph, change='official queries',
               **shift_readout(S, FR, np.array(FR['chain_grid'], np.int64), cal, ev, S.EVAL, a.alpha, log))
    extra = [v['cost_ratio_vs_reference'] - 1 for v in OUT['before_recalibration'].values() if v.get('calibrated')]
    OUT['summary']['extra_cost_before'] = [min(extra), max(extra)]; log('extra cost on the official EVAL queries', OUT['summary']['extra_cost_before'])
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

"""Index updates on graph A: the frontier rules calibrated before the update, without and with recalibration.

Paper: Section VI-B, paragraph "Query shift and index updates" and Fig. 5 (1% or 5% new items inserted, or as many base
items replaced in place). The same 200,000 pool queries are logged on the updated index (reduced arrays
<updated ds>_<graph>_chunk<k>.npz); their CAL and EVAL rows are those of the pool permutation.
The chain uses the multipliers recorded by the frontier run.

Usage:
    python analysis/shift_updates.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json \
        --shift-red UPDATED_RED_DIR --shift-ds amazon_replace5 --out OUT.json
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
    ap.add_argument('--red', required=True, help='reduced arrays logged on the index before the update')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py before the update')
    ap.add_argument('--shift-red', required=True, help='reduced arrays logged on the updated index')
    ap.add_argument('--shift-ds', required=True, help='name of the updated dataset in the file names, for example amazon_replace5')
    ap.add_argument('--shift-graph', default=None, help='name of the updated graph in the file names (default: --graph)')
    ap.add_argument('--alpha', type=float, default=0.05, help='level of the risk tests, split over the risk levels')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    return ap


def load(a, FR):
    """Setting before the update, multipliers of the chain, the CAL and EVAL queries on the updated index, reference rows."""
    import numpy as np
    from cast.setting import N_CHUNKS
    from replay import setting_of_frontier
    from shift import load_groups, take
    S = setting_of_frontier(a.red, a.ds, a.graph, FR, keep_hold=True, seeds=a.seeds)
    U = load_groups(a.shift_red, a.shift_ds, a.shift_graph or a.graph, [f'chunk{k}' for k in range(N_CHUNKS)], FR['target'])
    return S, np.array(FR['chain_grid'], np.int64), take(U, S.pool_rows['CAL']), take(U, S.pool_rows['EVAL']), S.EVAL


def main():
    a = parse_args().parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from shift import shift_readout
    FR = json.load(open(a.frontier)); S, mult, cal, ev, ref = load(a, FR)
    OUT = dict(ds=a.ds, graph=a.graph, change=a.shift_ds, **shift_readout(S, FR, mult, cal, ev, ref, a.alpha, log))
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); log('done')


if __name__ == '__main__':
    main()

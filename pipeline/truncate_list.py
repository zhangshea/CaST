"""Reduced arrays of a search with a shorter candidate list, derived from the logged traversal with 8,192 entries.

Paper: Section VI-B, paragraph "Workload breadth" (the list of 2,048 entries on YFCC-10M and on the main workload).

A list of L' entries expands the same candidates in the same order until it ends after T_{L'} visits, and checkpoints
after its end are placed at the end. So, row by row,
    c'_j = min(c_j, T_{L'}),   X' = X if X <= T_{L'} else NEVER,   final_visits' = min(final_visits, T_{L'}),
and the same for the recall target 0.9. Features and flags are kept as logged: after the end, c'_{j+1} - c'_j = 0 and the
interval (c'_j, c'_{j+1}] is empty, so no cost, risk, rule or bound reads them. The FIT group is copied unchanged, so the
score model and the bucket edges are those of the full list and only the search changes. T_{L'} is the column of L' in
`list_stop` of the stop-time file written by pipeline/reduce_logs.py.

Output  WORK/reduced/<out-name>/ with the same file names, and <ds>_<graph>_list<L'>.json (shares of truncated queries).

Usage:
    python pipeline/truncate_list.py --work WORK --name k10_kappa1 --out-name k10_kappa1_list2048 --ds yfcc10m --graph A \
        --layout parts --list-size 2048
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.grid import NEVER


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--name', required=True, help='subdirectory of WORK/reduced with the arrays of the full list')
    ap.add_argument('--out-name', required=True, help='subdirectory of WORK/reduced for the truncated arrays')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--layout', default='pool', choices=('pool', 'parts'))
    ap.add_argument('--list-size', type=int, default=2048, help="L', one of the logged list sizes")
    return ap.parse_args()


def main():
    a = parse_args(); ds, g, LP = a.ds, a.graph, a.list_size
    RIN = os.path.join(a.work, 'reduced', a.name); ROUT = os.path.join(a.work, 'reduced', a.out_name); os.makedirs(ROUT, exist_ok=True)
    if a.layout == 'pool':
        keep = [f'{ds}_{g}_fitpool']; trunc = [f'{ds}_{g}_chunk{k}' for k in range(20)]
    else:
        keep = [f'{ds}_{g}_fit']; trunc = [f'{ds}_{g}_{p}' for p in ('sel', 'cal', 'eval')]
    rep = dict(layout=a.layout, ds=ds, graph=g, list_size=LP, files={})
    for f in keep:
        shutil.copyfile(os.path.join(RIN, f + '.npz'), os.path.join(ROUT, f + '.npz'))
    for f in trunc:
        R = dict(np.load(os.path.join(RIN, f + '.npz'))); E = np.load(os.path.join(RIN, f + '_stops.npz'))
        sizes = [int(x) for x in E['list_sizes']]; T_list = E['list_stop'][:, sizes.index(LP)].astype(np.int64)
        # alignment of the two files: the same completion times row by row
        xe, xr = E['X_exact'].astype(np.int64), R['X_exact'].astype(np.int64)
        assert np.array_equal(xe == NEVER, xr == NEVER) and np.array_equal(xe[xr < NEVER], xr[xr < NEVER]), f
        c0 = R['c']; assert int(c0.max()) < NEVER; R['c'] = np.minimum(c0, T_list[:, None]).astype(c0.dtype)
        for t in ('X_exact', 'X_recall09'):
            if t in R: R[t] = np.where(R[t].astype(np.int64) <= T_list, R[t], NEVER).astype(R[t].dtype)
        R['final_visits'] = np.minimum(R['final_visits'].astype(np.int64), T_list).astype(R['final_visits'].dtype)
        np.savez(os.path.join(ROUT, f + '.npz'), **R)
        rep['files'][f] = dict(n=len(T_list), share_truncated=float((c0[:, -1] > T_list).mean()),
                               share_X_lost=float(((E['X_exact'] <= c0[:, -1]) & (E['X_exact'] > T_list)).mean()), mean_list_stop=float(T_list.mean()),
                               mean_c_last_before=float(c0[:, -1].mean()), mean_c_last_after=float(R['c'][:, -1].mean()))
    json.dump(rep, open(os.path.join(ROUT, f'{ds}_{g}_list{LP}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'files'}), {f: round(v['share_X_lost'], 4) for f, v in rep['files'].items()})


if __name__ == '__main__':
    main()

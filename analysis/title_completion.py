"""Completion times of the title queries next to those of the item queries of the same items.

Paper: Section VI-B, paragraph "Query shift and index updates" ("Title queries complete later"). No stopping rule is
applied: per dataset, for the CAL items and for the first EVAL items, quantiles of the completion time X, the share of
queries that never complete within the logged search and the share with X beyond the last checkpoint, for both kinds of
query, and how often the title query completes later than the item query.

Usage:
    python analysis/title_completion.py --red RED_DIR --shift-red TITLE_RED_DIR --ds amazon arxiv --graph A --out OUT.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.grid import NEVER
from cast.setting import DEFAULT_SEEDS, N_CHUNKS, N_POOL, SPLIT


def stats(c, X):
    inf = X >= NEVER; Xf = np.where(inf, np.inf, X.astype(np.float64))
    return dict(n=int(len(X)), X_p50=float(np.median(Xf)), X_p90=float(np.quantile(Xf, .9)), X_p99=float(np.quantile(Xf, .99)),
                share_never_complete=float(inf.mean()), share_incomplete=float((X > c[:, -1]).mean()), mean_visits_last_checkpoint=float(c[:, -1].mean()))


def compare(ic, ix, Tc, TX):
    """Item queries (ic, ix) and title queries (Tc, TX) of the same items, row by row."""
    both = (ix < NEVER) & (TX < NEVER)
    return dict(item=stats(ic, ix), title=stats(Tc, TX),
                share_title_later=float((TX > ix).mean()), share_title_earlier=float((TX < ix).mean()),
                median_ratio_title_over_item_both_finite=float(np.median(TX[both] / np.maximum(ix[both], 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='reduced arrays of the item queries'); ap.add_argument('--shift-red', required=True, help='reduced arrays of the title queries')
    ap.add_argument('--ds', nargs='+', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--cal-groups', nargs='+', default=['titleshift_cal']); ap.add_argument('--eval-groups', nargs='+', default=['titleshift_eval'])
    ap.add_argument('--pool-seed', type=lambda x: int(x, 0), default=DEFAULT_SEEDS['pool_permutation'], help='seed of the permutation that cuts the pool into samples')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    Pf = np.random.default_rng(a.pool_seed).permutation(N_POOL); res = {}
    for ds in a.ds:
        I = [np.load(f'{a.red}/{ds}_{a.graph}_chunk{k}.npz') for k in range(N_CHUNKS)]
        Ic = np.concatenate([d['c'] for d in I]); IX = np.concatenate([d['X_exact'] for d in I]); assert len(IX) == N_POOL
        res[ds] = {}
        for part, groups in (('cal', a.cal_groups), ('eval', a.eval_groups)):
            T = [np.load(f'{a.shift_red}/{ds}_{a.graph}_{p}.npz') for p in groups]
            Tc = np.concatenate([d['c'] for d in T]); TX = np.concatenate([d['X_exact'] for d in T])
            a0, b0 = SPLIT[part.upper()]; r = Pf[a0:b0][:len(TX)] if part == 'eval' else Pf[a0:b0]; assert len(TX) == len(r)
            res[ds][part] = compare(Ic[r], IX[r], Tc, TX)
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

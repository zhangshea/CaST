"""Cost of a clairvoyant stopping rule on the EVAL queries of one setting, against the empirical class optimum.

Paper: Section VI-B, paragraph "Near-optimality" ("Outside the class, a clairvoyant rule stops at the first checkpoint
after completion and spends the risk budget on the queries of largest avoidable cost"). The rule knows the completion
time X: it stops at the first checkpoint whose visit count reaches X; at risk eps it gives up a share eps of the queries
at the first checkpoint, first those that do not complete by the last checkpoint and then those whose completion costs
most beyond the first checkpoint. No stopping rule on these checkpoints spends fewer mean visits on this sample at
empirical risk at most eps. The empirical class optimum on EVAL is read from the output of pipeline/frontier.py.

Usage:
    python analysis/clairvoyant.py --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --out OUT.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.setting import DEFAULT_SEEDS, N_CHUNKS, N_POOL, SPLIT


def clairvoyant_costs(c, X, levels):
    """Mean visits of the clairvoyant rule at each risk level; c (n, J) visits at the checkpoints, X (n,) completion times."""
    n = len(X); comp = X <= c[:, -1]; jstar = np.argmax(c >= X[:, None], axis=1)
    stop = np.where(comp, c[np.arange(n), jstar], c[:, 0]); save = np.where(comp, stop - c[:, 0], -np.inf)
    res = {}
    for e in levels:
        k = int(np.floor(float(e) * n)) - int((~comp).sum())
        if k < 0: res[e] = None; continue
        cost = stop.copy(); idx = np.argsort(-save)[:k]; cost[idx] = c[idx, 0]
        res[e] = float(cost.mean())
    return res, int((~comp).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the same setting')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool-seed', type=lambda x: int(x, 0), default=DEFAULT_SEEDS['pool_permutation'], help="layout 'pool': seed of the permutation that cuts the pool into samples")
    a = ap.parse_args()
    FR = json.load(open(a.frontier)); target = FR['target']
    if FR['layout'] == 'pool':
        Z = [np.load(f'{a.red}/{a.ds}_{a.graph}_chunk{k}.npz') for k in range(N_CHUNKS)]
        c = np.concatenate([z['c'] for z in Z]).astype(np.float64); X = np.concatenate([z[target] for z in Z]).astype(np.float64)
        ev = np.random.default_rng(a.pool_seed).permutation(N_POOL)[SPLIT['EVAL'][0]:SPLIT['EVAL'][1]]; c, X = c[ev], X[ev]
    else:
        z = np.load(f'{a.red}/{a.ds}_{a.graph}_eval.npz'); c = z['c'].astype(np.float64); X = z[target].astype(np.float64)
    costs, never = clairvoyant_costs(c, X, list(FR['bounds']))
    res = dict(ds=a.ds, graph=a.graph, n=len(X), never_complete=never, levels={})
    for e, b in FR['bounds'].items():
        if costs[e] is None: res['levels'][e] = dict(feasible=False); continue
        opt = b.get('empirical_optimum')
        res['levels'][e] = dict(clairvoyant_mean_visits=costs[e], empirical_optimum=opt, optimum_over_clairvoyant=(float(opt) / costs[e]) if opt else None)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(a.ds, a.graph, {e: (round(v['clairvoyant_mean_visits'], 1), round(v['optimum_over_clairvoyant'], 3)) if v.get('optimum_over_clairvoyant') else None
                          for e, v in res['levels'].items()})


if __name__ == '__main__':
    main()

"""Dataset statistics of Table I and the share of queries whose constrained answer differs from the unconstrained one.

Paper: Section VI-A, Datasets; Table I (base size, dimension, number of categories with at least one base item, maximum
and mean number of categories per base item).

    table   one row per dataset from DATA/<ds>/base.npy and categories_main.npz.
    share   (layout 'pool': amazon, arxiv) over the first 10,000 EVAL queries of the main workload: the share whose exact
            constrained answer differs from the 10 items of largest inner product, found by a full scan in float32.
            For the other datasets this share is the field share_differs_from_unconstrained of the summary of
            pipeline/exact_answers.py.

Usage:
    python data/dataset_stats.py table --data DATA --ds amazon arxiv yfcc10m youtube8m
    python data/dataset_stats.py share --data DATA --ds amazon arxiv --out OUT.json
"""
import argparse
import json
import os
import time

import numpy as np


def table(a):
    out = {}
    for ds in a.ds:
        d = os.path.join(a.data, ds)
        X = np.load(os.path.join(d, 'base.npy'), mmap_mode='r'); z = np.load(os.path.join(d, 'categories_main.npz'))
        ptr, idx, m = z['ptr'], z['idx'], int(z['n_categories'])
        lens = np.diff(ptr); sizes = np.bincount(idx, minlength=m)
        out[ds] = dict(n_base=int(X.shape[0]), dim=int(X.shape[1]), categories=int((sizes > 0).sum()), categories_per_item_max=int(lens.max()),
                       categories_per_item_mean=float(lens.mean()))
        st = out[ds]
        print(f"{ds:10s} {st['n_base']:>10,} {st['dim']:>5} {st['categories']:>6,} {st['categories_per_item_max']:>4} {st['categories_per_item_mean']:.2f}", flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


def share(a):
    out = {}
    Pf = np.random.default_rng(a.pool_seed).permutation(200000); rows = Pf[60000:180000][:a.n]       # the first EVAL queries
    for ds in a.ds:
        t = time.time(); d = os.path.join(a.data, ds)
        Qc = [np.load(os.path.join(d, 'queries', f'chunk{k}.npy'), mmap_mode='r') for k in range(20)]
        Sc = [np.load(os.path.join(d, 'answers', f'chunk{k}_{a.workload}.npz'))['S'] for k in range(20)]
        Q = np.ascontiguousarray(np.stack([Qc[r // 10000][r % 10000] for r in rows]), dtype=np.float32)
        S = np.stack([Sc[r // 10000][r % 10000] for r in rows]).astype(np.int64); n = len(Q)
        X = np.ascontiguousarray(np.load(os.path.join(d, 'base.npy'), mmap_mode='r'), dtype=np.float32); N = X.shape[0]
        top = np.empty((n, 10), np.int64); bs = 500
        for i in range(0, n, bs):
            Dm = Q[i:i+bs] @ X.T
            p = np.argpartition(-Dm, 10, axis=1)[:, :10]
            top[i:i+bs] = p
        same = (np.sort(top, 1) == np.sort(S[:, :10], 1)).all(1)
        out[ds] = dict(n=int(n), N=int(N), share_differs=float(1 - same.mean()), seconds=round(time.time() - t, 1))
        print(ds, out[ds], flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('table'); p.add_argument('--data', required=True); p.add_argument('--ds', nargs='+', required=True); p.add_argument('--out', default=None)
    p = sub.add_parser('share'); p.add_argument('--data', required=True); p.add_argument('--ds', nargs='+', required=True); p.add_argument('--out', default=None)
    p.add_argument('--workload', default='k10_kappa1'); p.add_argument('--n', type=int, default=10000, help='number of EVAL queries')
    p.add_argument('--pool-seed', type=lambda x: int(x, 0), default=0x13528ef, help='seed of the permutation of the 200,000 queries (cast.setting)')
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(table=table, share=share)[a.cmd](a)

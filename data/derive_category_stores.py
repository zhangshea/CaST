"""Amazon and arXiv: category stores of the workloads with capacity 2 and without constraint.

Paper: Table II (rows "kappa = 2" and "unconstrained") and Section VI-B, paragraph "Workload breadth".

Both stores are used with unit capacities, so that exact answers, logging and the engine run as in the main workload.
    kappa2          every category c becomes two unit-capacity slots 2c and 2c + 1. A set of items can be assigned to
                    distinct slots iff it has an assignment with at most two items per category, so the feasible answers
                    (Definition 1) are those of capacity 2 on the categories of the main workload.
    unconstrained   every item carries the same k unit-capacity slots 0..k-1. Every set of at most k items is feasible, so
                    the greedy answer is the k nearest items.

Output  DATA/<ds>/categories_kappa2.npz, DATA/<ds>/categories_unconstrained.npz.

Usage:
    python data/derive_category_stores.py --data DATA --ds amazon
"""
import argparse
import json
import os

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--ds', required=True)
    ap.add_argument('--k', type=int, default=10, help='answer size of the unconstrained workload')
    return ap.parse_args()


def main():
    a = parse_args(); d = os.path.join(a.data, a.ds); K = a.k
    z = np.load(os.path.join(d, 'categories_main.npz')); ptr = z['ptr'].astype(np.int64); idx = z['idx'].astype(np.int32); m = int(z['n_categories']); n = len(ptr) - 1
    deg = np.diff(ptr); assert (deg >= 1).all()
    c2 = np.empty(2 * len(idx), np.int32); c2[0::2] = 2 * idx; c2[1::2] = 2 * idx + 1
    np.savez(os.path.join(d, 'categories_kappa2.npz'), ptr=2 * ptr, idx=c2, n_categories=np.int64(2 * m))
    co = np.tile(np.arange(K, dtype=np.int32), n)
    np.savez(os.path.join(d, 'categories_unconstrained.npz'), ptr=K * np.arange(n + 1, dtype=np.int64), idx=co, n_categories=np.int64(K))
    print(json.dumps(dict(ds=a.ds, items=n, n_categories=m, per_item_mean=float(deg.mean()), per_item_max=int(deg.max()),
                          kappa2=dict(n_categories=2 * m, nnz=int(len(c2))), unconstrained=dict(n_categories=K, nnz=int(len(co))))), flush=True)


if __name__ == '__main__':
    main()

"""What a failed answer looks like, and where failures fall, on the timing queries of a bundle.

Paper: Section VI-B, "Held-out check" (when the rule of Table III fails, most of its answers miss exactly one of the ten
members, at a small increase of the total distance).

For each rule, among the queries whose answer differs from the exact answer S: the recall of the returned answer, and
the ratio of its total distance to that of S (both are feasible sets of k items, so the ratio is at least one up to the
search order). The failure rate is also given by quartile of the rarity of a query: the frequency in the base of the
rarest category among the categories of the items of S.

Usage:
    python timing/failed_answers.py --bundle BUNDLE --data DATA --work WORK --out OUT.json [parameters and files of the rules]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    from engine_rules import THIRD_WINDOW, add_engine_args, add_rule_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True, help='prefix of the timing bundle (timing/export_bundle.py)')
    ap.add_argument('--out', required=True, help='output JSON')
    ap.add_argument('--rules', nargs='+', default=THIRD_WINDOW)
    ap.add_argument('--threads', type=int, default=16)
    add_engine_args(ap); add_rule_args(ap)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    import numpy as np
    import engine_rules as ER
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); R, info = ER.build_rules(a.rules, B, M, a)
    from cast import datasets
    ds = M['ds']
    Q = np.ascontiguousarray(B['Q'], dtype=np.float32)
    S = B['S'].astype(np.int64)
    n, k = S.shape
    X = datasets.base(a.data, ds)
    cp, ci, m = datasets.category_csr(a.data, ds, a.variant)
    freq = np.bincount(ci, minlength=int(m)).astype(np.float64)
    rare_item = np.full(cp.shape[0] - 1, np.inf)
    for v in np.unique(S):
        p0, p1 = cp[v], cp[v + 1]
        if p1 > p0:
            rare_item[v] = freq[ci[p0:p1]].min()
    rare_q = rare_item[S].min(1)
    edges = np.quantile(rare_q, [0.25, 0.5, 0.75])
    grp = np.searchsorted(edges, rare_q)
    def dist(ids):
        out = np.empty(ids.shape, np.float64)
        for i in range(ids.shape[0]):
            out[i] = 1.0 - X[ids[i]].astype(np.float32) @ Q[i]
        return out
    dS = dist(S).sum(1)
    res = dict(ds=ds, n=int(n), k=int(k), quartile_edges=[float(e) for e in edges], rules={}, **info)
    for nm in a.rules:
        o = ER.search(eng, ctx, Q, k, R[nm], a.threads)
        ids = o['ids'].astype(np.int64)
        fail = ~ER.same_answer(ids, S)
        rec = np.array([len(set(ids[i].tolist()) & set(S[i].tolist())) / k for i in np.flatnonzero(fail)])
        idx = np.flatnonzero(fail)
        full = (ids[idx] >= 0).all(1)
        dR = np.full(len(idx), np.nan)
        if full.any():
            sub = idx[full]
            dd = np.empty((len(sub), k), np.float64)
            for t, i in enumerate(sub):
                dd[t] = 1.0 - X[ids[i]].astype(np.float32) @ Q[i]
            dR[full] = dd.sum(1) / dS[sub]
        res['rules'][nm] = dict(fail=int(fail.sum()), fail_rate=float(fail.mean()), mean_recall_failed=float(rec.mean()), share_missing_one=float((rec >= (k - 1) / k - 1e-9).mean()),
                                share_recall_below_08=float((rec < 0.8 - 1e-9).mean()), not_full=int((~full).sum()),
                                dist_ratio_mean=float(np.nanmean(dR)), dist_ratio_p50=float(np.nanpercentile(dR, 50)), dist_ratio_p99=float(np.nanpercentile(dR, 99)),
                                fail_by_rarity_quartile=[float(fail[grp == g].mean()) for g in range(4)])
        log(nm, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in res['rules'][nm].items() if x != 'fail_by_rarity_quartile'},
            [round(x, 4) for x in res['rules'][nm]['fail_by_rarity_quartile']])
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
    log('done')


if __name__ == '__main__':
    main()

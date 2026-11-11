"""Throughput and latency percentiles of the calibrated rules under concurrency, on the timing queries of a bundle.

Paper: paragraph "Cost and latency at a common risk target" of Section VI-B (at 16 threads, our throughput over that of
each rule of the third timing window of Table III).

For each thread count, every rule runs `--passes` times in rotated order over all timing queries; the throughput is the
number of queries divided by the median wall time of the batch. The per-query latencies of the last pass give the mean
and the percentiles under that load. The visits of every rule are compared with those of the first thread count.

Usage:
    python timing/throughput.py --bundle BUNDLE --data DATA --work WORK --out OUT.json [parameters and files of the rules]
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
    ap.add_argument('--rules', nargs='+', default=THIRD_WINDOW, help='rules in the order of the first pass')
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 4, 8, 16, 32]); ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=2000, help='queries of the warm-up of every rule at every thread count')
    add_engine_args(ap); add_rule_args(ap)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv); t0_ = time.time()
    def log(*x): print(f'[{time.time() - t0_:6.0f}s]', *x, flush=True)
    import numpy as np
    import engine_rules as ER
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); names = a.rules; R, info = ER.build_rules(names, B, M, a); assert 'cast' in names
    Q = np.ascontiguousarray(B['Q'], dtype=np.float32)
    n = len(Q); k = M['k']
    run = lambda nm, Q_, th: ER.search(eng, ctx, Q_, k, R[nm], th)
    res = dict(ds=M['ds'], n=n, passes=a.passes, threads=a.threads, engine_build=eng.build_flags(), per_thread={}, **info)
    ref_vis = {}
    for th in a.threads:
        for nm in names:
            run(nm, Q[:a.warmup], th)                               # warm-up
        wall = {nm: [] for nm in names}
        last = {}
        for rep in range(a.passes):
            for nm in names[rep % len(names):] + names[:rep % len(names)]:
                t0 = time.perf_counter()
                o = run(nm, Q, th)
                wall[nm].append(time.perf_counter() - t0)
                last[nm] = o
        row = {}
        for nm in names:
            o = last[nm]
            ns = o['ns'].astype(np.float64) / 1e3
            if nm not in ref_vis:
                ref_vis[nm] = o['visits'].copy()
            med = float(np.median(wall[nm]))
            row[nm] = dict(wall_s=wall[nm], qps=n / med, mean_us=float(ns.mean()), p50_us=float(np.percentile(ns, 50)), p95_us=float(np.percentile(ns, 95)),
                           p99_us=float(np.percentile(ns, 99)), visits_equal_first_thread_count=float((o['visits'] == ref_vis[nm]).mean()))
        for nm in names:
            row[nm]['qps_cast_over_rule'] = row['cast']['qps'] / row[nm]['qps']
        res['per_thread'][str(th)] = row
        log('threads', th, {nm: round(row[nm]['qps']) for nm in names})
        json.dump(res, open(a.out, 'w'), indent=1)
    log('done')


if __name__ == '__main__':
    main()

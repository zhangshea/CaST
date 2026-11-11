"""A drift signal that needs no labels: the distribution of the stopping checkpoint of the rule in use.

Paper: Section VI-B, paragraph "Query shift and index updates" ("the Kolmogorov-Smirnov distance between the stopping
checkpoints of 1,000 queries and those on EVAL, thresholded for 5% false alarms on EVAL batches, flags 54% to 100% of
the title and replacement batches"). For the frontier rule at eps = 0.05, the statistic of a batch of unlabeled queries
is the Kolmogorov-Smirnov distance between the distribution of its stopping checkpoint in the batch and on the EVAL
queries before the change. The alarm threshold is the 95th percentile of the statistic over batches drawn from EVAL
itself. Batches are drawn with replacement by one generator, the EVAL batches first and the batches after the change
second.

Usage (the arguments of analysis/shift_title.py or analysis/shift_updates.py select the queries after the change):
    python analysis/drift_signal.py --kind title --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json \
        --shift-red TITLE_RED_DIR --out OUT.json
    python analysis/drift_signal.py --kind update --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json \
        --shift-red UPDATED_RED_DIR --shift-ds amazon_replace5 --out OUT.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def drift_signal(S, mk, th, ref_rows, shift_rows, n=1000, batches=4000, seed=0x13528f3):
    """Share of batches flagged before (false alarms) and after the change, for the rule `th`."""
    J = S.J; rng = np.random.default_rng(seed)
    def stop(idx):
        F = mk(th, idx); j = np.argmax(F > 0, axis=1); c = S.c[idx, j].astype(np.float64)
        return j, c, c < S.X[idx]
    jr, cr, fr = stop(ref_rows); js, cs, fs = stop(shift_rows)
    ref = np.bincount(jr, minlength=J).cumsum() / len(jr)
    def stats(j):
        KS = np.empty(batches)
        for b in range(batches):
            i = rng.integers(0, len(j), n)
            KS[b] = np.abs(np.bincount(j[i], minlength=J).cumsum() / n - ref).max()
        return KS
    KSr = stats(jr); KSs = stats(js)
    tk = float(np.quantile(KSr, 0.95))
    return dict(n=n, batches=batches, seed=seed, ref_fail_rate=float(fr.mean()), shift_fail_rate=float(fs.mean()), ref_visits=float(cr.mean()), shift_visits=float(cs.mean()),
                ks_threshold=tk, ks_ref_alarm=float((KSr > tk).mean()), ks_detect=float((KSs > tk).mean()),
                last_share_ref=float((jr == J - 1).mean()), last_share_shift=float((js == J - 1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', required=True, choices=('title', 'update'))
    ap.add_argument('--red', required=True); ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--frontier', required=True, help='output of pipeline/frontier.py before the change')
    ap.add_argument('--shift-red', required=True, help='reduced arrays of the queries after the change')
    ap.add_argument('--shift-ds', default=None, help='kind update: name of the updated dataset in the file names')
    ap.add_argument('--shift-graph', default=None, help='kind update: name of the updated graph in the file names (default: --graph)')
    ap.add_argument('--cal-groups', nargs='+', default=['titleshift_cal'], help='kind title')
    ap.add_argument('--eval-groups', nargs='+', default=['titleshift_eval'], help='kind title')
    ap.add_argument('--eps', type=float, default=0.05, help='risk level of the frontier rule in use')
    ap.add_argument('--batch', type=int, default=1000, help='queries per batch')
    ap.add_argument('--batches', type=int, default=4000, help='batches drawn before and after the change')
    ap.add_argument('--seed', type=lambda x: int(x, 0), default=0x13528f3, help='seed of the batch draws')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    a = ap.parse_args(); t0 = time.time()
    if a.kind == 'update' and not a.shift_ds: ap.error('--kind update needs --shift-ds')
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    import shift_title
    import shift_updates
    from shift import prepare
    FR = json.load(open(a.frontier))
    S, mult, cal, ev, _ = (shift_title if a.kind == 'title' else shift_updates).load(a, FR)
    mk, TH, _, _, EVAL2 = prepare(S, FR, mult, cal, ev, log)
    res = dict(kind=a.kind, change=('title queries' if a.kind == 'title' else a.shift_ds), ds=a.ds, eps=a.eps,
               **drift_signal(S, mk, TH[a.eps], S.EVAL, EVAL2, a.batch, a.batches, a.seed))
    json.dump(res, open(a.out, 'w'), indent=1); log(res)


if __name__ == '__main__':
    main()

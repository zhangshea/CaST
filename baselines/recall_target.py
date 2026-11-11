"""DARTH and DARTH+ under the recall target rho = 0.9 on one setting.

A query fails when the returned answer holds fewer than ceil(rho k) members of the exact answer S, the failure event of
the recall rows of Table II. The forests of baselines/darth.py and baselines/darth_plus.py are used as trained: they
predict the current recall, which does not depend on the target. Each rule is calibrated on CAL by the common
fixed-sequence exact binomial test at eps = delta = 0.05 with Rt from 1.0000 downward in steps of 0.0001, then run on all
EVAL queries; DARTH+ at each of its three quantile levels. The rules are also run uncalibrated at the literal targets
Rt = 0.90 and Rt = 0.95.

Paper: Section VI-A, "Compared methods" (at the literal targets 0.90 and 0.95 the DARTH port reaches mean recalls of
0.908 and 0.956 to 0.959) and Section VI-B, paragraph "Recall target" (DARTH and DARTH+ spend 1.23 to 1.60 times our
visits under the same test; with --common-target the ratios to our rule calibrated for the same target are printed).

Usage:
  python baselines/recall_target.py --data DATA --work WORK --ds amazon --out WORK/results/recall_target_amazon_A.json \
      [--common-target COMMON_TARGET_RECALL.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C
from baselines import darth as D
from baselines import darth_plus as DP

RHO, EPS, DELTA = 0.9, 0.05, 0.05


def hits(ids, S):
    """Members of the exact answer S in each returned answer."""
    return (ids[:, :, None] == S[:, None, :]).any(2).sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_setting_args(ap)
    ap.add_argument('--darth-model', default=None, help='prefix of the DARTH model files (default WORK/models/darth_<ds>_<graph>)')
    ap.add_argument('--darth-plus-model', default=None, help='prefix of the DARTH+ model files (default WORK/models/darthplus_<ds>_<graph>)')
    ap.add_argument('--common-target', default=None, help='output of pipeline/common_target.py for the recall target of this setting')
    ap.add_argument('--seeds', default=None); ap.add_argument('--out', required=True, help='output JSON; per-rule EVAL visits go to <out>_<rule>_visits.npy')
    a = ap.parse_args()
    from scipy.stats import binom
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); Par, Lmax = D.search_params(ctx); k = ctx.k
    need = int(np.ceil(RHO * k))
    rows, _ = C.pool_rows(over); Qc, Sc = C.take_pool(a.data, a.ds, a.workload, rows['CAL']); Qe, Se = C.take_pool(a.data, a.ds, a.workload, rows['EVAL']); n = len(Qc)
    Fd, schd = D.load_model(a.darth_model or os.path.join(a.work, 'models', f'darth_{a.ds}_{a.graph}'))
    TJ, Fr, QF = DP.load_models(a.darth_plus_model or os.path.join(a.work, 'models', f'darthplus_{a.ds}_{a.graph}')); sch = TJ['schedule']
    RULES = {'darth': (lambda Rt: D.darth_rule(Rt, D.schedule_entry(schd, Rt)), Fd)}
    for lq in DP.QUANTILE_LEVELS:
        RULES[f'darth_plus_{lq}'] = ((lambda Rt, q=QF[lq]: DP.darth_plus_rule(Rt, DP.schedule_entry(sch, Rt), q)), Fr)
    def run(nm, Rt, Q):
        mk, F = RULES[nm]
        return eng.stop_search(ctx.G, ctx.C, Q, Par, Lmax, mk(Rt), None, F, a.threads, False, False)
    res = dict(ds=a.ds, graph=a.graph, rho=RHO, eps=EPS, delta=DELTA, n_cal=n, n_eval=len(Qe), rules={})
    for nm in RULES:
        trace, last = [], None
        for Rt in np.round(np.arange(1.0, 0.6999, -0.0001), 4):
            o = run(nm, float(Rt), Qc); fail = int((hits(o['ids'].astype(np.int64), Sc) < need).sum()); trace.append((float(Rt), fail, float(o['visits'].mean())))
            if binom.cdf(fail, n, EPS) <= DELTA: last = float(Rt)
            else: break
        r = dict(Rt=last, cal_trace_tail=trace[-3:], points=len(trace))
        if last is not None:
            o = run(nm, last, Qe); h = hits(o['ids'].astype(np.int64), Se)
            r.update(eval_mean_visits=float(o['visits'].mean()), eval_fail=int((h < need).sum()), eval_fail_rate=float((h < need).mean()), eval_exact=float((h == k).mean()),
                     eval_mean_recall=float(h.mean() / k))
            np.save(a.out.replace('.json', f'_{nm}_visits.npy'), o['visits'].astype(np.int32))
        for Rt0 in (0.9, 0.95):
            o = run(nm, Rt0, Qe); h = hits(o['ids'].astype(np.int64), Se)
            r[f'uncalibrated_{Rt0}'] = dict(mean_visits=float(o['visits'].mean()), fail_rate=float((h < need).mean()), exact_share=float((h == k).mean()),
                                             exact_share_first_10000=float((h[:10000] == k).mean()),
                                             mean_recall=float(h.mean() / k))
        res['rules'][nm] = r; D.log(nm, {x: y for x, y in r.items() if x != 'cal_trace_tail'})
        json.dump(res, open(a.out, 'w'), indent=1)
    if a.common_target:                  # visits over those of our rule calibrated for the same recall target
        ours = json.load(open(a.common_target))['table']['cast']['mean_visits']
        res['visits_over_cast'] = {nm: r['eval_mean_visits'] / ours for nm, r in res['rules'].items() if 'eval_mean_visits' in r}
        json.dump(res, open(a.out, 'w'), indent=1); D.log('visits over ours', res['visits_over_cast'])
    D.log('done')


if __name__ == '__main__':
    main()

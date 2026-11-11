"""A parametric threshold curve of the class: fit on SEL, calibration by the common test, comparison with our rule
over random splits, and where our rule saves visits over the curve.

Curve rule: beta_j(a) = clip(round(c + s max(0, j - j0) + u (a - 1)), -1, B - 1) for checkpoints j before the cap and
B - 1 from the cap on, made nondecreasing in j and in the flag a: an offset c, a break point j0, a slope s, a gap u
between the two flags, and a cap. It is a staircase of the monotone class (Theorem 1(b)). The parameters (s, j0, u, cap)
are chosen on SEL as the cap of Level+cap is (fixed-sequence test on SEL at level 0.01 over c in steps of 3, least SEL
cost), then c is calibrated on CAL in steps of 0.5 by the common test at eps = delta = 0.05, and EVAL is read once.
HOLD is not read.

Paper: Section VI-B, paragraph "Cost and latency at a common risk target": the curve spends 0.976 to 1.074 times our
visits over ten random splits per setting, at failure rates within 0.3 points; its median per setting is 1.012 to 1.053
and it recovers a median of 41% to 85% of the gap to Level+cap; on the reported split, 100% and 91% of the visits the
chain saves over the curve fall after 4,096 visits with a full answer. The curve of the reported split is the rule
"Curve" timed by timing/. The thresholds of our calibrated rule per checkpoint (the staircase profile described in the
same paragraph) are the table `table_cast` written by the attribution step.

Steps:
  fit          one split: --split-seed permutes the SEL, CAL and EVAL queries among themselves (FIT and HOLD stay); without it, the reported split
  summary      smallest, median and largest ratio per setting over the split seeds 1..10
  attribution  reported split: visits our rule saves over the curve, by the rule that stops first, the flag and the visit range at that stop

Usage:
  python baselines/threshold_curve.py fit --red RED_DIR --ds amazon --graph A --out CURVE_amazon_A.json
  python baselines/threshold_curve.py fit --red RED_DIR --ds amazon --graph A --split-seed 1 --out SPLITS/amazon_A_1.json
  python baselines/threshold_curve.py summary --splits SPLITS --out SUMMARY.json
  python baselines/threshold_curve.py attribution --red RED_DIR --ds amazon --graph A --curve CURVE_amazon_A.json --out OUT.json
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

EPS, DELTA = 0.05, 0.05
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
J0_GRID = [0, 16, 32, 48, 64]
CAP_VISITS = (2000, 4000, 8000, 16000)

t0 = time.time()
def log(*a): print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


def table(c, s_, u, cap, j0, J, NB):
    """(J-1) x 2 thresholds beta_j(a) of the curve rule."""
    j = np.arange(J - 1, dtype=np.float64)[:, None]; a = np.array([0.0, 1.0])[None, :]
    b = np.rint(c + s_ * np.maximum(j - j0, 0.0) + u * (a - 1.0)); b = np.clip(b, -1, NB - 1); b = np.maximum.accumulate(b, axis=0); b[cap:, :] = NB - 1
    b[:, 0] = np.minimum(b[:, 0], b[:, 1])
    return b.astype(np.int32)


def load(a, over):
    """The setting with HOLD kept apart, its interface, and the rules of the common target (ours, Level+cap, Level)."""
    from cast.chain import chain_multipliers
    from cast.compared import calibrate_level, calibrate_level_cap, calibrate_staircase
    from cast.setting import SPLIT, fit_interface, load_setting
    S = load_setting(a.red, a.ds, a.graph, layout='pool', target='X_exact', keep_hold=True, seeds=over)
    seed = getattr(a, 'split_seed', None)
    if seed is not None:                 # permute the SEL, CAL and EVAL queries among themselves; FIT and HOLD stay
        off = len(S.FIT); Pf = np.concatenate([S.pool_rows[k] for k in ('SEL', 'CAL', 'EVAL', 'HOLD')]); n3 = SPLIT['EVAL'][1]
        Pf[:n3] = np.random.default_rng(int(seed)).permutation(Pf[:n3]); sl = {k: Pf[x:y] for k, (x, y) in SPLIT.items()}
        assert S.n == off + len(Pf) and np.array_equal(S.HOLD, off + sl['HOLD'])
        S.SEL, S.CAL, S.EVAL = (off + sl[k] for k in ('SEL', 'CAL', 'EVAL')); S.pool_rows = sl
    fit_interface(S)
    RULE = dict(level_cap=calibrate_level_cap(S, EPS, DELTA), level=calibrate_level(S, EPS, DELTA))
    RULE['cast'], _ = calibrate_staircase(S, EPS, DELTA, chain_multipliers()[0])
    assert all(v is not None for v in RULE.values()) and RULE['cast']['type'] == 1
    return S, RULE


def fit(a):
    from cast.calibration import fixed_sequence
    from cast.compared import rule_F
    from cast.rules import evaluate_rule, staircase_F
    _, over = C.read_seeds(a.seeds); S, RULE = load(a, over); J, NB, SEL, EVAL = S.J, S.NB, S.SEL, S.EVAL; eps = EPS
    U_GRID = [0, 5, 10, 20, 40, NB]
    CAP_J = sorted({J - 1} | {int(np.searchsorted(S.grid, v)) for v in CAP_VISITS if int(np.searchsorted(S.grid, v)) < J - 1})
    def cgrid(s_, step, j0): return list(np.arange(-1.0 - s_ * max(J - 2 - j0, 0), NB - 1.0 + step, step))
    tab = lambda c, s_, u, cap, j0: table(c, s_, u, cap, j0, J, NB)
    res = dict(ds=a.ds, graph=a.graph, S=S_GRID, U=U_GRID, J0=J0_GRID, CAP=[int(x) for x in CAP_J], search=[])
    best = None
    for s_ in S_GRID:
        for j0 in (J0_GRID if s_ > 0 else [0]):
            for u in U_GRID:
                for cap in CAP_J:
                    if j0 >= cap: continue
                    cs, _ = fixed_sequence(S, cgrid(s_, 3.0, j0), lambda c, idx, s_=s_, u=u, cap=cap, j0=j0: staircase_F(S, tab(c, s_, u, cap, j0), idx), eps, idx=SEL, U=S.u_sel, delta=0.01)
                    if cs is None: continue
                    ca, _ = evaluate_rule(S, staircase_F(S, tab(cs, s_, u, cap, j0), SEL), SEL)
                    r = dict(s=s_, u=u, cap=int(cap), j0=int(j0), c=float(cs), sel_mean_visits=float(ca.mean())); res['search'].append(r)
                    if best is None or r['sel_mean_visits'] < best['sel_mean_visits']: best = r
        log('slope done', s_, len(res['search']), best)
    log('best on SEL', best)
    cc, tr = fixed_sequence(S, cgrid(best['s'], 0.5, best['j0']), lambda c, idx: staircase_F(S, tab(c, best['s'], best['u'], best['cap'], best['j0']), idx), eps, delta=DELTA)
    res['cal_trace_tail'] = [[str(y) for y in x] for x in tr[-3:]]
    res['selected'] = best
    res['c_cal'] = None if cc is None else float(cc)
    rows = {}; cal = dict(RULE=RULE)
    for nm in ('cast', 'level_cap', 'level'):
        F_ = np.maximum.accumulate(np.clip(rule_F(S, cal, nm, EVAL), 0, 1), axis=1); F_[:, -1] = 1.0
        c_, r_ = evaluate_rule(S, F_, EVAL); rows[nm] = dict(mean_visits=float(c_.mean()), fail_rate=float(r_.mean()))
    if cc is not None:
        F_ = staircase_F(S, tab(cc, best['s'], best['u'], best['cap'], best['j0']), EVAL); F_ = np.maximum.accumulate(np.clip(F_, 0, 1), axis=1); F_[:, -1] = 1.0
        c_, r_ = evaluate_rule(S, F_, EVAL); rows['curve'] = dict(mean_visits=float(c_.mean()), fail_rate=float(r_.mean()))
    else:
        rows['curve'] = dict(calibrated=False)
    res['rules'] = rows
    for nm in rows:
        if 'mean_visits' in rows[nm]: rows[nm]['over_cast'] = rows[nm]['mean_visits'] / rows['cast']['mean_visits']
    if rows['curve'].get('mean_visits'):
        lc_, ch_, cv_ = rows['level_cap']['mean_visits'], rows['cast']['mean_visits'], rows['curve']['mean_visits']
        res['recovered'] = (lc_ - cv_) / (lc_ - ch_); res['fail_rate_gap_points'] = 100 * (rows['curve']['fail_rate'] - rows['cast']['fail_rate'])
    res['split_seed'] = a.split_seed
    json.dump(res, open(a.out, 'w'), indent=1); log('done', rows)


def summary(a):
    res = {}
    for s in a.settings:
        ds, g = s.split(':'); name = f'{ds} {g}'
        Js = [json.load(open(os.path.join(a.splits, f'{ds}_{g}_{k}.json'))) for k in range(1, 11)]
        assert [str(j['split_seed']) for j in Js] == [str(k) for k in range(1, 11)] and all(j['ds'] == ds and str(j['graph']) == g for j in Js)
        ratio = [j['rules']['curve']['mean_visits'] / j['rules']['cast']['mean_visits'] for j in Js]
        lc = [j['rules']['level_cap']['mean_visits'] / j['rules']['cast']['mean_visits'] for j in Js]
        rec = [j['recovered'] for j in Js]
        gap = [j['fail_rate_gap_points'] for j in Js]
        r = dict(ratio=ratio, recovered=rec, fail_rate_gap_points=gap, level_cap_over_cast=lc,
                 ratio_min=min(ratio), ratio_median=st.median(ratio), ratio_max=max(ratio),
                 recovered_min=min(rec), recovered_median=st.median(rec), recovered_max=max(rec),
                 level_cap_min=min(lc), level_cap_median=st.median(lc), level_cap_max=max(lc),
                 below_one=sum(v < 1 for v in ratio), within_1pct=sum(v < 1.01 for v in ratio), recovered_at_least_half=sum(v >= 0.5 for v in rec),
                 gap_abs_max=max(abs(v) for v in gap), gap_over_03=sum(abs(v) > 0.3 for v in gap),
                 curve_fail_rate=[j['rules']['curve']['fail_rate'] for j in Js], cast_fail_rate=[j['rules']['cast']['fail_rate'] for j in Js],
                 selected=[j['selected'] for j in Js], c_cal=[j['c_cal'] for j in Js])
        res[name] = r
        print(name, 'curve over ours', *(round(r[k], 3) for k in ('ratio_min', 'ratio_median', 'ratio_max')), 'below one', r['below_one'], 'below 1.01', r['within_1pct'],
              '| recovered', *(round(r[k], 3) for k in ('recovered_min', 'recovered_median', 'recovered_max')), 'at least half', r['recovered_at_least_half'],
              '| Level+cap over ours', *(round(r[k], 3) for k in ('level_cap_min', 'level_cap_median', 'level_cap_max')),
              '| failure-rate gap', round(r['gap_abs_max'], 3), 'above 0.3', r['gap_over_03'])
        print('   ratio', [round(v, 3) for v in ratio])
        print('   gap  ', [round(v, 3) for v in gap])
        print('   curve', [(x['s'], x['j0'], x['u'], x['cap']) for x in r['selected']])
    allr = [v for r in res.values() for v in r['ratio']]
    print('all', round(min(allr), 3), round(st.median(allr), 3), round(max(allr), 3), 'below one', sum(v < 1 for v in allr), 'of', len(allr))
    json.dump(res, open(a.out, 'w'), indent=1)


def attribution(a):
    """Per-query differences (curve visits - our visits) on EVAL, attributed to the earlier stop j* of the two rules:
    which rule stops first x the flag at j* x the range of g_{j*} ([64, 1024), [1024, 4096), [4096, H]); per cell the count,
    the sum of the differences, the failures of the earlier stopper alone and the failures of both; shares over the
    cells in which our rule stops first, and their margins by flag and by range. Also the two threshold tables and where
    one lies above the other."""
    from cast.compared import rule_F
    from cast.rules import evaluate_rule, staircase_F
    _, over = C.read_seeds(a.seeds); S, RULE = load(a, over); J, NB, EVAL = S.J, S.NB, S.EVAL
    fit_out = json.load(open(a.curve)); sel_p = fit_out['selected']; c_cal = fit_out['c_cal']
    res = dict(ds=a.ds, graph=a.graph, curve=dict(selected=sel_p, c_cal=c_cal), checks={})
    Tcur = table(c_cal, sel_p['s'], sel_p['u'], sel_p['cap'], sel_p['j0'], J, NB)
    Tch = np.array(RULE['cast']['beta']).reshape(J - 1, 2).astype(np.int32)
    Fc = np.clip(rule_F(S, dict(RULE=RULE), 'cast', EVAL), 0, 1); Fv = np.clip(staircase_F(S, Tcur, EVAL), 0, 1)
    DET = dict(cast=bool(np.isin(Fc, (0.0, 1.0)).all()), curve=bool(np.isin(Fv, (0.0, 1.0)).all())); res['checks']['deterministic'] = DET
    assert DET['cast'] and DET['curve'], DET
    Fc = np.maximum.accumulate(Fc, axis=1); Fc[:, -1] = 1.0; Fv = np.maximum.accumulate(Fv, axis=1); Fv[:, -1] = 1.0
    cc, rc = evaluate_rule(S, Fc, EVAL); cv, rv = evaluate_rule(S, Fv, EVAL)
    cc = cc.astype(np.float64); cv = cv.astype(np.float64); fc = rc.round().astype(bool); fv = rv.round().astype(bool)
    fit_rules = fit_out['rules']               # the two rules are those of the fit step
    P0 = dict(cast_mean_visits=float(cc.mean()), cast_mean_visits_fit=fit_rules['cast']['mean_visits'], cast_fail_rate=float(fc.mean()), cast_fail_rate_fit=fit_rules['cast']['fail_rate'],
              curve_mean_visits=float(cv.mean()), curve_mean_visits_fit=fit_rules['curve']['mean_visits'], curve_fail_rate=float(fv.mean()), curve_fail_rate_fit=fit_rules['curve']['fail_rate'])
    P0['pass'] = bool(abs(P0['cast_mean_visits'] - P0['cast_mean_visits_fit']) <= 1e-9 * P0['cast_mean_visits_fit'] and abs(P0['curve_mean_visits'] - P0['curve_mean_visits_fit']) <= 1e-9 * P0['curve_mean_visits_fit']
                      and abs(P0['cast_fail_rate'] - P0['cast_fail_rate_fit']) <= 1e-12 and abs(P0['curve_fail_rate'] - P0['curve_fail_rate_fit']) <= 1e-12)
    res['checks']['same_rules_as_fit'] = P0; log('same rules as the fit step', P0); assert P0['pass'], P0
    jc = Fc.argmax(1); jv = Fv.argmax(1); jstar = np.minimum(jc, jv)
    gj = np.asarray(S.grid)[: J]
    def rng_of(j):
        v = gj[np.minimum(j, len(gj) - 1)]
        return np.where(v < 1024, 0, np.where(v < 4096, 1, 2))
    RN = ('[64,1024)', '[1024,4096)', '[4096,H]')
    EV = np.asarray(EVAL); flag = np.asarray(S.full)[EV, :][np.arange(len(EV)), np.minimum(jstar, S.full.shape[1] - 1)].astype(int)
    rg = rng_of(jstar); dlt = cv - cc
    who = np.where(jc < jv, 'cast_first', np.where(jv < jc, 'curve_first', 'same'))
    cells = {}
    for w in ('cast_first', 'curve_first'):
        for a_ in (0, 1):
            for r_ in range(3):
                m = (who == w) & (flag == a_) & (rg == r_)
                early_fail = fc if w == 'cast_first' else fv; late_fail = fv if w == 'cast_first' else fc
                cells[f'{w}|a={a_}|{RN[r_]}'] = dict(n=int(m.sum()), sum_delta=float(dlt[m].sum()), early_fails_only=int((m & early_fail & ~late_fail).sum()),
                                                     both_fail=int((m & early_fail & late_fail).sum()))
    same = who == 'same'; res['same_stop'] = dict(n=int(same.sum()), sum_delta=float(dlt[same].sum()))
    tot = float(dlt.sum()); P1 = dict(total=tot, cells=float(sum(v['sum_delta'] for v in cells.values()) + res['same_stop']['sum_delta']))
    P1['pass'] = bool(abs(P1['total'] - P1['cells']) <= 1e-6 * max(1.0, abs(P1['total']))); res['checks']['cells_add_up'] = P1; assert P1['pass'], P1
    den = sum(v['sum_delta'] for k, v in cells.items() if k.startswith('cast_first'))
    for k, v in cells.items():
        if k.startswith('cast_first'): v['share_of_cast_first'] = v['sum_delta'] / den if den else None
    marg = dict(by_flag={}, by_range={})
    for a_ in (0, 1): marg['by_flag'][f'a={a_}'] = sum(cells[f'cast_first|a={a_}|{RN[r_]}']['sum_delta'] for r_ in range(3)) / den if den else None
    for r_ in range(3): marg['by_range'][RN[r_]] = sum(cells[f'cast_first|a={a_}|{RN[r_]}']['sum_delta'] for a_ in (0, 1)) / den if den else None
    res['cells'] = cells; res['margins_cast_first'] = marg; res['cast_first_total'] = den; res['net_total'] = tot; res['n_eval'] = int(len(EV))
    res['net_mean_visits'] = tot / len(EV); res['curve_over_cast'] = float(cv.mean() / cc.mean())
    dif = Tch.astype(np.int64) - Tcur.astype(np.int64); prof = {}
    jr = rng_of(np.arange(J - 1))
    for a_ in (0, 1):
        for r_ in range(3):
            m = jr == r_; d_ = dif[m, a_]
            prof[f'a={a_}|{RN[r_]}'] = dict(cells=int(m.sum()), cast_higher=int((d_ > 0).sum()), curve_higher=int((d_ < 0).sum()), mean_diff=float(d_.mean()) if len(d_) else None)
    res['profile'] = prof; res['table_cast'] = Tch.tolist(); res['table_curve'] = Tcur.tolist(); res['grid'] = [int(x) for x in gj]
    json.dump(res, open(a.out, 'w'), indent=1); log('done', dict(curve_over_cast=res['curve_over_cast'], margins=marg))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    for name in ('fit', 'attribution'):
        p = sub.add_parser(name)
        p.add_argument('--red', required=True, help="directory of the reduced arrays of the workload (layout 'pool')"); p.add_argument('--ds', required=True)
        p.add_argument('--graph', required=True); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
        if name == 'fit': p.add_argument('--split-seed', type=int, default=None, help='seed of a random split of SEL, CAL and EVAL (1..10 in the paper); omitted: the reported split')
        else: p.add_argument('--curve', required=True, help='output of the fit step for the reported split')
    p = sub.add_parser('summary')
    p.add_argument('--splits', required=True, help='directory with the fit outputs <ds>_<graph>_<seed>.json for the seeds 1..10')
    p.add_argument('--settings', nargs='+', default=['amazon:A', 'amazon:B', 'arxiv:A', 'arxiv:B']); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(fit=fit, summary=summary, attribution=attribution)[a.step](a)

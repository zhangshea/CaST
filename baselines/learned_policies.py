"""Richer learned stopping policies by backward induction, calibrated like our rule and compared with it.

Longstaff-Schwartz style backward induction (F. A. Longstaff and E. S. Schwartz, "Valuing American Options by
Simulation: A Simple Least-Squares Approach", Review of Financial Studies 14(1)) at a multiplier lam, the slope
of the SEL frontier at the target risk level: with T' the stop of the policy already fitted for the later checkpoints,
models at each checkpoint j estimate the further cost of continuing E[c_T' - c_j] and the chance that continuing turns a
failure into an exact answer P(c_j < X <= c_T'); the policy stops at the first checkpoint where the former is at least
lam' times the latter (floored at 1e-3). The decision multiplier lam' is the calibrated parameter (600 geometric values
from 1e4 lam down to 0.05 lam, after the rule that never stops early).
  fully_fitted     gradient-boosted models on the seven features, the flag, the score and its running minimum, fitted
                   and decided on all of SEL
  cross_fitted     the same with two halves of SEL: each half decides with the models fitted on the other half and
                   updates its own continuation; deployment averages the predictions of the two halves
  declared_state   exact per-state means on the interface state z_j = (b_j, a_j) only; states with fewer than 30 SEL
                   queries at a checkpoint never stop
Our rule ('cast') and these policies fit the same SEL queries, are calibrated on one half of CAL by the fixed-sequence
test at level delta / M and are compared on the other half of CAL. No EVAL or HOLD outcome is used.
Learning curves at eps = 0.05, on the first 5,000, 10,000 and 20,000 queries of one permutation of SEL: the
out-of-sample Lagrangian value (mean of visits + lam * failure on the second half of CAL, at lam' = lam, before any
calibration) of the cross-fitted policy and of the staircase of the class fitted at the same multiplier.

Paper: Section VI-A, "Compared methods" (fourth group) and Section VI-B, "Ablation study", paragraph "Richer learned
policies"; baselines/learned_policies_summary.py prints the reported aggregates.

Usage:
  python baselines/learned_policies.py --red RED_DIR --ds amazon --graph A --out POLICIES_amazon_A.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

DELTA_CAL = 0.05
E = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]; M = len(E)
ETA = 1e-3                                # floor of the estimated chance of turning a failure into an exact answer
FAMILIES = ('cross_fitted', 'fully_fitted', 'declared_state')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help="directory of the reduced arrays of the workload (layout 'pool')"); ap.add_argument('--ds', required=True)
    ap.add_argument('--graph', required=True); ap.add_argument('--seeds', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--subsamples', type=int, nargs='+', default=[5000, 10000, 20000], help='SEL sizes of the learning curves')
    a = ap.parse_args(); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from cast.calibration import fixed_sequence
    from cast.chain import chain_element, chain_multipliers
    from cast.monotone_class import frontier_slope
    from cast.rules import bootstrap_means, evaluate_rule, staircase_F, terminal_F
    from cast.setting import fit_interface, load_setting
    seeds, over = C.read_seeds(a.seeds)
    S = fit_interface(load_setting(a.red, a.ds, a.graph, layout='pool', target='X_exact', seeds=over))
    J, X, c_all, Z, ns = S.J, S.X, S.c, S.Z, S.cls.ns; FIT, SEL, CAL = S.FIT, S.SEL, S.CAL
    PC = np.random.default_rng(seeds['policy_cal_halves']).permutation(len(CAL)); half = len(CAL) // 2     # 20,000 queries each
    CALa, CALb = CAL[PC[:half]], CAL[PC[half:2 * half]]                  # CALa calibrates, CALb compares
    Ucal = np.random.default_rng(S.seeds['cal_uniforms']).random(len(CALa))
    log(a.ds, a.graph, 'rows', S.n, 'FIT', len(FIT), 'SEL', len(SEL), 'CAL-a', len(CALa), 'CAL-b', len(CALb))

    # ---------------------------------------------------------------- policies by backward induction
    def feats_R(idx, j):
        return np.column_stack([S.features[idx, j, :], S.full[idx, j].astype(float), S.s[idx, j], S.m[idx, j]])
    def _gbr(): return HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, random_state=seeds['policy_models'])
    def _gbc(): return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, random_state=seeds['policy_models'])
    class Const:                            # a classification target with one class only
        def __init__(self, v): self.v = float(v)
        def predict(self, P): return np.full(len(P), self.v)
    def _fit_prob(P, y):
        if y.min() == y.max(): return Const(y[0])
        m_ = _gbc().fit(P, y); m_.predict = (lambda P, m_=m_: m_.predict_proba(P)[:, 1]); return m_
    def lsm_fit(idx, lam):
        """Fully fitted policy: backward induction on idx, each query decided by the models fitted on all of idx."""
        n = len(idx); c = c_all[idx].astype(np.float64); fl = (X[idx, None] > c_all[idx]); Tc = np.full(n, J - 1); out = [None] * (J - 1); ar = np.arange(n)
        for j in range(J - 2, -1, -1):
            inc, rs = c[ar, Tc] - c[:, j], (fl[:, j] & ~fl[ar, Tc]).astype(float); Phi = feats_R(idx, j)
            mi = _gbr().fit(Phi, inc); mr = _fit_prob(Phi, rs); out[j] = (mi, mr)
            G, Dp = np.maximum(mi.predict(Phi), 0.0), np.maximum(mr.predict(Phi), ETA); Tc = np.where(G >= lam * Dp, j, Tc)
        return [out]
    def lsm_fit_cf(idx, lam, iA, iB):
        """Cross-fitted policy: each half decides with the models fitted on the other half and updates its own continuation."""
        n = len(idx); c = c_all[idx].astype(np.float64); fl = (X[idx, None] > c_all[idx]); Tc = np.full(n, J - 1); ar = np.arange(n)
        oA, oB = [None] * (J - 1), [None] * (J - 1)
        for j in range(J - 2, -1, -1):
            inc, rs = c[ar, Tc] - c[:, j], (fl[:, j] & ~fl[ar, Tc]).astype(float); Phi = feats_R(idx, j)
            oA[j] = (_gbr().fit(Phi[iA], inc[iA]), _fit_prob(Phi[iA], rs[iA])); oB[j] = (_gbr().fit(Phi[iB], inc[iB]), _fit_prob(Phi[iB], rs[iB]))
            for here_, (mi, mr) in ((iA, oB[j]), (iB, oA[j])):
                G, Dp = np.maximum(mi.predict(Phi[here_]), 0.0), np.maximum(mr.predict(Phi[here_]), ETA); Tc[here_] = np.where(G >= lam * Dp, j, Tc[here_])
        return [oA, oB]
    def lsm_preds(outs, idx):
        """(further cost, rescue chance) for every checkpoint before the last, averaged over the fitted versions."""
        G = np.zeros((len(idx), J - 1)); Dp = np.zeros((len(idx), J - 1)); k = len(outs)
        for j in range(J - 1):
            Phi = feats_R(idx, j)
            G[:, j] = np.maximum(sum(o[j][0].predict(Phi) for o in outs) / k, 0.0); Dp[:, j] = np.maximum(sum(o[j][1].predict(Phi) for o in outs) / k, ETA)
        return G, Dp
    def tab_terms(o, zj): return np.where(o['ok'][zj], o['inc'][zj], -np.inf), np.maximum(o['rs'][zj], ETA)
    def tab_fit(idx, lam):
        """Declared-state policy: exact per-state means of the further cost and of the rescue indicator on z_j."""
        n = len(idx); c = c_all[idx].astype(np.float64); fl = (X[idx, None] > c_all[idx]); Tc = np.full(n, J - 1); out = [None] * (J - 1); ar = np.arange(n)
        for j in range(J - 2, -1, -1):
            inc, rs = c[ar, Tc] - c[:, j], (fl[:, j] & ~fl[ar, Tc]).astype(float); zj = Z[idx, j]
            ct = np.bincount(zj, minlength=ns); dv = np.maximum(ct, 1)
            out[j] = dict(ok=ct >= 30, inc=np.bincount(zj, inc, ns) / dv, rs=np.bincount(zj, rs, ns) / dv)
            G, Dp = tab_terms(out[j], zj); Tc = np.where(G >= lam * Dp, j, Tc)
        return out
    def tab_preds(out, idx):
        G = np.zeros((len(idx), J - 1)); Dp = np.zeros((len(idx), J - 1))
        for j in range(J - 1): G[:, j], Dp[:, j] = tab_terms(out[j], Z[idx, j])
        return G, Dp
    def stop_at(G, Dp, lam2):
        dec = G >= lam2 * Dp; return np.where(dec.any(1), dec.argmax(1), J - 1)
    def F_from_stop(jst): return (np.arange(J)[None, :] >= jst[:, None]).astype(float)

    OUT = dict(ds=a.ds, graph=a.graph, levels=E, n=dict(FIT=len(FIT), SEL=len(SEL), CALa=len(CALa), CALb=len(CALb)), delta=DELTA_CAL)
    LAMK = {e: int(round(v)) for e, v in ((e, frontier_slope(S, SEL, e)) for e in E) if v}; OUT['multipliers'] = LAMK; log('multipliers', LAMK)
    # ---- our rule: the chain on SEL, calibrated on CAL-a at delta / M, read on CAL-b
    LAMG = chain_multipliers()[0]; CH = {int(l): chain_element(S, SEL, int(l))[0] for l in LAMG}
    SEQ = ['terminal'] + [int(l) for l in LAMG[::-1]]
    mkC = lambda th, idx: terminal_F(S, idx) if th == 'terminal' else staircase_F(S, CH[th], idx)
    RES, COST = {}, {}
    for e in E:
        th, tr = fixed_sequence(S, SEQ, mkC, e, idx=CALa, U=Ucal, delta=DELTA_CAL / M)
        if th is None: RES[('cast', e)] = dict(calibrated=False); continue
        cost, risk = evaluate_rule(S, mkC(th, CALb), CALb); COST[('cast', e)] = cost
        RES[('cast', e)] = dict(calibrated=True, theta=th, mean_visits=float(cost.mean()), coverage=float(1 - risk.mean()))
    log('our rule done', {e: round(RES[('cast', e)].get('mean_visits', -1), 1) for e in E})
    # ---- the three policies: fitted once per target eps at the SEL slope lam; calibrated over the decision multiplier lam'
    pcf = np.random.default_rng(seeds['policy_crossfit']).permutation(len(SEL)); iA, iB = np.sort(pcf[:len(SEL) // 2]), np.sort(pcf[len(SEL) // 2:])
    def fit_family(fam, lam):
        if fam == 'cross_fitted': return ('gb', lsm_fit_cf(SEL, lam, iA, iB))
        if fam == 'fully_fitted': return ('gb', lsm_fit(SEL, lam))
        return ('tab', tab_fit(SEL, lam))
    def preds_family(fit, idx): return lsm_preds(fit[1], idx) if fit[0] == 'gb' else tab_preds(fit[1], idx)
    def dump(): OUT['rules'] = {f'{k[0]}_{k[1]}': v for k, v in RES.items()}; json.dump(OUT, open(a.out, 'w'), indent=1, default=str)
    for fam in FAMILIES:
        for e in E:
            lam = float(LAMK.get(e, 0.0))
            if not lam: RES[(fam, e)] = dict(calibrated=False, note='no slope'); continue
            t1 = time.time(); fit = fit_family(fam, lam)
            P = {nm: preds_family(fit, idx) for nm, idx in (('CALa', CALa), ('CALb', CALb))}
            grid = list(np.geomspace(1e4 * lam, 0.05 * lam, 600))
            mk = lambda th, idx, P=P: terminal_F(S, idx) if th == 'terminal' else F_from_stop(stop_at(*P['CALa' if idx is CALa else 'CALb'], th))
            th, tr = fixed_sequence(S, ['terminal'] + grid, mk, e, idx=CALa, U=Ucal, delta=DELTA_CAL / M)
            r = dict(lam=lam, fit_secs=round(time.time() - t1, 1))
            if th is None: r.update(calibrated=False)
            else:
                cost, risk = evaluate_rule(S, mk(th, CALb), CALb); COST[(fam, e)] = cost
                r.update(calibrated=True, terminal=(th == 'terminal'), lam2=None if th == 'terminal' else float(th), lam2_over_lam=None if th == 'terminal' else float(th) / lam,
                         mean_visits=float(cost.mean()), coverage=float(1 - risk.mean()), trace_tail=[(str(a_), b_, c_) for a_, b_, c_ in tr[-2:]])
            RES[(fam, e)] = r; log(fam, e, {k: v for k, v in r.items() if k != 'trace_tail'}); dump()
    # ---- learning curves at eps = 0.05 on SEL subsamples: out-of-sample Lagrangian value on CAL-b at lam' = lam
    LC, LCC = {}, {}
    lam05 = float(LAMK.get(0.05, 0.0)); psub = np.random.default_rng(seeds['policy_subsamples']).permutation(len(SEL))
    for nsub in a.subsamples:
        if not lam05: break
        sub = SEL[np.sort(psub[:nsub])]; q = np.random.default_rng(seeds['policy_subsample_crossfit']).permutation(nsub); jA, jB = np.sort(q[:nsub // 2]), np.sort(q[nsub // 2:])
        fit = ('gb', lsm_fit_cf(sub, lam05, jA, jB)); G, Dp = lsm_preds(fit[1], CALb)
        cf_, rf_ = evaluate_rule(S, F_from_stop(stop_at(G, Dp, lam05)), CALb); LC[str(nsub)] = float((cf_ + lam05 * rf_).mean())
        beta = chain_element(S, sub, int(round(lam05)))[0]
        cc_, rc_ = evaluate_rule(S, staircase_F(S, beta, CALb), CALb); LCC[str(nsub)] = float((cc_ + lam05 * rc_).mean())
        log('learning curve', nsub, 'cross-fitted', LC[str(nsub)], 'class', LCC[str(nsub)])
    OUT['lam_005'] = lam05; OUT['learning_curve_005'] = dict(cross_fitted=LC, cast=LCC)
    # ---- paired comparisons on CAL-b: visits of our rule over those of each policy, with a bootstrap interval
    CMP = {}; bs = seeds['policy_bootstrap']
    for e in E:
        r = {}
        for fam in FAMILIES:
            if ('cast', e) in COST and (fam, e) in COST:
                a_, b_ = COST[('cast', e)], COST[(fam, e)]; rat = bootstrap_means(a_, seed=bs) / bootstrap_means(b_, seed=bs)
                r[f'cast_over_{fam}'] = [float(a_.mean() / b_.mean()), float(np.quantile(rat, .025)), float(np.quantile(rat, .975))]
        CMP[str(e)] = r; log('compare', e, r)
    OUT['compare'] = CMP; dump(); log('done')


if __name__ == '__main__':
    main()

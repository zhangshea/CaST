"""The staircase on a shorter candidate list: calibration and latency.

Paper: Section VI-B, "Workload breadth" (latency on 10,000 EVAL queries of YFCC-10M; the list of 2,048 entries and its
time against the calibrated list size; the same change on the HOLD queries of the main workload).

A search with a list of L' entries expands the same candidates in the same order as the logged search with 8,192 entries
until it ends after T_{L'} visits. So the staircase run on a list of L' stops at the earlier of its own stop on the
logged trajectory and T_{L'}, and it fails when that stop is below the completion time X.

Commands:
    calibrate   for every L' of 8192, 4096, 2048, 1024 (and for the full list), calibrate the composite staircase on CAL
                by the fixed-sequence test at eps = delta = 0.05, read it on SEL and EVAL, and optionally export its
                table and offline replay on the timing queries of a bundle
    time        single-thread latency of the exported staircases on their lists, of our rule on the list of 8,192, of
                the calibrated list size and of rules of the bundle (protocol of timing/time_rules.py)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LISTS = [8192, 4096, 2048, 1024]
N_EVAL_QUERIES = 10000                 # timing queries of a bundle of --queries eval


def parse_args(argv=None):
    from engine_rules import add_engine_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('calibrate', help='calibrate the staircase on every shorter list')
    p.add_argument('--red', required=True, help='directory of the reduced arrays'); p.add_argument('--stops', default=None, help='directory of the <group>_stops.npz files (default: --red)')
    p.add_argument('--ds', required=True); p.add_argument('--graph', required=True); p.add_argument('--layout', default='pool', choices=('pool', 'parts'))
    p.add_argument('--frontier', default=None, help="output JSON of pipeline/frontier.py (layout 'parts': its chain multipliers are used)")
    p.add_argument('--out', required=True, help='output prefix (OUT.json, OUT_export.npz, OUT_hold_export.npz)')
    p.add_argument('--eps', type=float, default=0.05); p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--seed', type=int, required=True, help='bootstrap seed of the visit ratios')
    p.add_argument('--export-eval', action='store_true', help='export tables and replay on the first 10,000 EVAL queries')
    p.add_argument('--export-hold', action='store_true', help="export tables and replay on the HOLD queries (layout 'pool')")
    p.add_argument('--seeds', default=None, help='JSON file overriding the recorded seeds of cast/setting.py')
    p = sub.add_parser('time', help='latency of the staircase on shorter lists')
    p.add_argument('--bundle', required=True, help='prefix of the timing bundle (timing/export_bundle.py)')
    p.add_argument('--export', required=True, help='export npz of the command calibrate for the timing queries of the bundle')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--list-size', type=int, required=True, help='calibrated list size of the rule list_size')
    p.add_argument('--lists', type=int, nargs='+', default=[2048, 4096], help='shorter lists to time; the first is the reference of the ratios')
    p.add_argument('--bundle-rules', nargs='*', default=['level_cap', 'level', 'laet_style', 'fixed'], help='rules of the bundle timed in the same window')
    p.add_argument('--seed', type=int, required=True, help='bootstrap seed')
    p.add_argument('--passes', type=int, default=5); p.add_argument('--warmup', type=int, default=1000); p.add_argument('--check-threads', type=int, default=8)
    p.add_argument('--resamples', type=int, default=2000)
    p.add_argument('--max-differ', type=int, default=None, help='stop without timing when the online visits of a rule differ from its replay on more queries')
    p.add_argument('--require-same-staircase', action='store_true', help='require that the exported full-list staircase and its replay equal those of the bundle')
    add_engine_args(p)
    return ap.parse_args(argv)


def calibrate(a, log):
    import numpy as np
    from scipy.stats import binom as _binom
    from cast.chain import chain_multipliers
    from cast.compared import calibrate_staircase
    from cast.rules import evaluate_rule, staircase_F, terminal_F
    from cast.setting import N_CHUNKS, fit_interface, load_setting
    seeds = json.load(open(a.seeds)) if a.seeds else None; ds, g = a.ds, a.graph; eps = a.eps; stops = a.stops or a.red
    S = load_setting(a.red, ds, g, layout=a.layout, keep_hold=(a.layout == 'pool'), seeds=seeds); fit_interface(S)
    J, NB, X = S.J, S.NB, S.X; SEL, CAL, EVAL, HOLD = S.SEL, S.CAL, S.EVAL, S.HOLD

    # ---- our rule at the common target: the chain on SEL, calibrated on CAL (the rule of Table III)
    LAMG = np.array(json.load(open(a.frontier))['chain_grid'], np.int64) if a.frontier else chain_multipliers(())[0]
    rec, CH = calibrate_staircase(S, eps, a.delta, LAMG); assert rec is not None, 'no candidate passes the test on CAL'; thc = rec['theta']
    mkC = lambda th, idx: terminal_F(S, idx) if th == 'terminal' else staircase_F(S, CH[th], idx)
    log('chain', len(LAMG), 'multipliers; calibrated theta', thc)

    # ---- T_L per query, aligned with the rows of the setting
    if a.layout == 'parts':
        EX = {p_: np.load(f'{stops}/{ds}_{g}_{p_}_stops.npz') for p_ in ('fit', 'sel', 'cal', 'eval')}
        TLf = np.concatenate([EX[p_]['list_stop'] for p_ in EX]).astype(np.int64); TLG = EX['fit']['list_sizes'].astype(np.int64)
        Xf = np.concatenate([np.load(f'{a.red}/{ds}_{g}_{p_}.npz')['X_exact'] for p_ in EX]); PART = dict(SEL=SEL, CAL=CAL, EVAL=EVAL)
    else:
        EX = [np.load(f'{stops}/{ds}_{g}_chunk{k}_stops.npz') for k in range(N_CHUNKS)]
        TLf = np.concatenate([e['list_stop'] for e in EX]).astype(np.int64); TLG = EX[0]['list_sizes'].astype(np.int64)
        Xf = np.concatenate([np.load(f'{a.red}/{ds}_{g}_chunk{k}.npz')['X_exact'] for k in range(N_CHUNKS)]); PART = {k_: S.pool_rows[k_] for k_ in ('SEL', 'CAL', 'EVAL')}
    DIDX = dict(SEL=SEL, CAL=CAL, EVAL=EVAL)
    for k_ in PART: assert np.array_equal(Xf[PART[k_]], X[DIDX[k_]]), k_
    gi = {int(L): i for i, L in enumerate(TLG)}; assert all(L in gi for L in LISTS)
    TLp = {k_: TLf[PART[k_]] for k_ in PART}

    def stop_visits(th, part):                                                   # staircase stop on the logs (deterministic 0/1 matrix)
        idx = DIDX[part]; F = np.maximum.accumulate(np.clip(mkC(th, idx), 0, 1), axis=1); F[:, -1] = 1.0
        return S.c[idx, (F >= 0.5).argmax(1)].astype(np.int64)
    def comp(sv, part, L):
        cost = sv if L is None else np.minimum(sv, TLp[part][:, gi[L]]); return cost, X[DIDX[part]] > cost

    # ---- check: the 0/1 stop equals the cost of the rule evaluation for our rule on EVAL
    F_tab = np.maximum.accumulate(np.clip(mkC(thc, EVAL), 0, 1), axis=1); F_tab[:, -1] = 1.0; c_tab, r_tab = evaluate_rule(S, F_tab, EVAL)
    sv_tab = stop_visits(thc, 'EVAL'); CHK = dict(stop_equals_rule_evaluation=bool(np.array_equal(sv_tab, c_tab.astype(np.int64))), cast_theta=thc, cast_mean_visits=float(c_tab.mean()))
    log('check', CHK)

    # ---- one sweep over the chain on CAL; every L' tests the same sequence with its own failures
    cands = ['terminal'] + [int(l) for l in LAMG[::-1]]; nC = len(CAL)
    keys = [None] + LISTS; act = {L: True for L in keys}; last = {L: None for L in keys}; trace = {str(L): [] for L in keys}
    for th in cands:
        sv = stop_visits(th, 'CAL')
        for L in keys:
            if not act[L]: continue
            _, f = comp(sv, 'CAL', L); k = int(f.sum()); ok = bool(_binom.cdf(k, nC, eps) <= a.delta); trace[str(L)].append((th if th == 'terminal' else int(th), k, ok))
            if ok: last[L] = th
            else: act[L] = False
        if not any(act.values()): break
    CHK['theta_equal_full_list'] = bool(last[None] == thc); CHK['theta_equal_list8192'] = bool(last[8192] == thc); log('calibrated', {str(L): last[L] for L in keys}, CHK)

    seed = a.seed; rng = np.random.default_rng(seed); nE = len(EVAL)
    Wb = rng.multinomial(nE, np.full(nE, 1.0 / nE), size=2000).astype(np.float64)
    def boot_ratio(x, y):
        ra = (Wb @ x.astype(np.float64)) / (Wb @ y.astype(np.float64)); return dict(ratio=float(x.mean() / y.mean()), lower_bound_one_sided=float(np.quantile(ra, 0.025)))
    def summ(cost, fail): return dict(mean_visits=float(cost.mean()), fail=int(fail.sum()), fail_rate=float(fail.mean()), n=int(len(cost)))
    name = lambda L: 'cast' if L is None else f'list{L}'
    OUT = dict(ds=ds, graph=g, layout=a.layout, eps=eps, delta=a.delta, lists=LISTS, bootstrap_seed=seed, checks=CHK, rules={}, trace=trace)
    sv_ref = stop_visits(thc, 'EVAL'); c_ref, f_ref = comp(sv_ref, 'EVAL', None)
    EXP = {}
    for L in keys:
        th = last[L]; nm = name(L)
        if th is None: OUT['rules'][nm] = dict(L=L, calibrated=False, reason='the first candidate (terminal) fails the test on CAL'); continue
        row = dict(L=L, calibrated=True, theta=th if th == 'terminal' else int(th))
        for part in ('CAL', 'SEL', 'EVAL'):
            c_, f_ = comp(stop_visits(th, part), part, L); row[part] = summ(c_, f_)
            if part == 'EVAL':
                row['EVAL']['visit_ratio'] = boot_ratio(c_, c_ref); row['EVAL']['share_list_binds'] = float((TLp['EVAL'][:, gi[L]] < stop_visits(th, 'EVAL')).mean()) if L else 0.0
                k_ = int(f_.sum()); row['EVAL']['p_one_sided_vs_eps'] = float(_binom.sf(k_ - 1, nE, eps))   # P(K >= k | p = eps): small means significantly above the target
        OUT['rules'][nm] = row; log(nm, row['theta'], row['EVAL'])
        if a.export_eval:
            beta = np.full((J - 1, 2), -1) if th == 'terminal' else CH[th]; nq = N_EVAL_QUERIES
            svq = stop_visits(th, 'EVAL')[:nq]; cq = svq if L is None else np.minimum(svq, TLp['EVAL'][:nq, gi[L]]); fq = X[EVAL[:nq]] > cq
            EXP[f'beta_{nm}'] = np.asarray(beta, np.int32).ravel(); EXP[f'replay_cost_{nm}'] = cq.astype(np.int64); EXP[f'replay_fail_{nm}'] = fq.astype(np.int8)
    json.dump(OUT, open(a.out + '.json', 'w'), indent=1, default=str)
    if EXP: np.savez(a.out + '_export.npz', **EXP)
    if a.export_hold:                                                             # offline stops on HOLD, in the row order of the HOLD bundle
        assert a.layout == 'pool' and len(HOLD) == 20000
        PART['HOLD'] = S.pool_rows['HOLD']; DIDX['HOLD'] = HOLD; assert np.array_equal(Xf[PART['HOLD']], X[HOLD]); TLp['HOLD'] = TLf[PART['HOLD']]
        EH = {}
        for L in keys:
            th = last[L]; nm = name(L)
            if th is None: continue
            svh = stop_visits(th, 'HOLD'); ch = svh if L is None else np.minimum(svh, TLp['HOLD'][:, gi[L]])
            EH[f'beta_{nm}'] = np.asarray(np.full((J - 1, 2), -1) if th == 'terminal' else CH[th], np.int32).ravel()
            EH[f'replay_cost_{nm}'] = ch.astype(np.int64); EH[f'replay_fail_{nm}'] = (X[HOLD] > ch).astype(np.int8)
            log('HOLD', nm, 'theta', th, 'mean', float(ch.mean()), 'fail', int((X[HOLD] > ch).sum()))
        np.savez(a.out + '_hold_export.npz', **EH)


def time_lists(a, log):
    import numpy as np
    import engine_rules as ER
    import protocol as PR
    os.makedirs(a.out, exist_ok=True)
    B, M = ER.load_bundle(a.bundle); EXd = dict(np.load(a.export)); k = M['k']; Lmax = M['list_size']; L_ef = a.list_size; assert Lmax == 8192
    eng, ctx = ER.load_engine(a, M); base = ER.interface_fields(B); Fm = ER.bundle_forests(B, M)
    RULES = {}
    for L in a.lists:
        if f'beta_list{L}' in EXd: RULES[f'list{L}'] = ER.staircase_entry(EXd[f'beta_list{L}'], base, Fm, L)
    ref = f'list{a.lists[0]}'; assert ref in RULES, 'no calibrated rule on the reference list'
    RULES['cast'] = ER.bundle_entry(M, 'cast', base, Fm); RULES['list_size'] = (ER.natural_rule(), None, None, L_ef)
    for nm in a.bundle_rules: RULES[nm] = ER.bundle_entry(M, nm, base, Fm)
    Q = np.ascontiguousarray(B['Q'], dtype=np.float32); S = B['S']; n = len(Q)
    run = lambda nm, Q_, th, instr=False: ER.search(eng, ctx, Q_, k, RULES[nm], th, instr)
    OUT = dict(ds=M['ds'], graph=M['graph'], engine_build=eng.build_flags(), list_size=L_ef, n=n, reference=ref,
               staircase_equals_bundle=bool(np.array_equal(EXd['beta_cast'].astype(np.int32), np.array(M['rules']['cast']['beta'], np.int32))),
               replay_equals_bundle=float((EXd['replay_cost_cast'] == B['replay_cast_cost']).mean()))
    if a.require_same_staircase: assert OUT['staircase_equals_bundle'] and OUT['replay_equals_bundle'] == 1.0, (OUT['staircase_equals_bundle'], OUT['replay_equals_bundle'])

    # ---- before timing: online visits against the offline stop (for a list L': min(stop, T_L')), and the failure flags
    AGR = {}
    for nm in RULES:
        if nm == 'list_size': continue
        o = run(nm, Q, a.check_threads)
        if nm in M['rules']:
            AGR[nm] = dict(visits_equal=float((o['visits'] == B[f'replay_{nm}_cost']).mean()), visits_differ=int((o['visits'] != B[f'replay_{nm}_cost']).sum()))
        else:
            on_fail = ~ER.same_answer(o['ids'], S); off_fail = EXd[f'replay_fail_{nm}'].astype(bool)
            AGR[nm] = dict(visits_equal=float((o['visits'] == EXd[f'replay_cost_{nm}']).mean()), visits_differ=int((o['visits'] != EXd[f'replay_cost_{nm}']).sum()),
                          fail_online=int(on_fail.sum()), fail_offline=int(off_fail.sum()), fail_flag_mismatch=int((on_fail != off_fail).sum()))
    OUT['online_against_replay'] = AGR; log('online against replay', AGR, 'same staircase', OUT['staircase_equals_bundle'])
    if a.max_differ is not None and any(AGR[nm]['visits_differ'] > a.max_differ for nm in AGR):
        OUT['stopped'] = f'online visits differ from the offline stop on more than {a.max_differ} queries'
        json.dump(OUT, open(os.path.join(a.out, 'time.json'), 'w'), indent=1, default=str); log('stopped before timing'); return

    names = list(RULES)
    NS, VIS, COV = PR.timed_passes(run, names, Q, S, a.passes, a.warmup, log)
    SHARE = {}
    for nm in (ref, 'cast'):                                             # share of the stopping check in the latency
        oi = run(nm, Q, 1, instr=True); SHARE[nm] = float(oi['hook_ns'].sum() / max(oi['ns'].sum(), 1))
    seed = a.seed; OUT['bootstrap_seed'] = seed; OUT['check_share'] = SHARE
    bidx = np.random.default_rng(seed).integers(0, n, (a.resamples, n)); MED = PR.median_latency(NS)
    def ratio(x, y):
        rat = MED[x][bidx].mean(1) / MED[y][bidx].mean(1); return [float(MED[x].mean() / MED[y].mean()), float(np.quantile(rat, .025)), float(np.quantile(rat, .975))]
    TIM = {}
    for nm in names:
        lat = MED[nm]; np.save(os.path.join(a.out, f'lat_{nm}.npy'), lat); f_ = int((~COV[nm]).sum())
        TIM[nm] = dict(mean_us=float(lat.mean() / 1e3), p50_us=float(np.percentile(lat, 50) / 1e3), p95_us=float(np.percentile(lat, 95) / 1e3), p99_us=float(np.percentile(lat, 99) / 1e3),
                       mean_visits=float(VIS[nm].mean()), fail=f_, fail_rate=f_ / n, us_per_visit=float(lat.mean() / 1e3 / VIS[nm].mean()),
                       pass_means_us=[float(x.mean() / 1e3) for x in NS[nm]])
        TIM[nm][f'ratio_vs_{ref}'] = ratio(nm, ref); TIM[nm]['ratio_vs_cast'] = ratio(nm, 'cast')
    OUT[f'pass_ratio_cast_over_{ref}'] = [float(NS['cast'][p].mean() / NS[ref][p].mean()) for p in range(a.passes)]
    OUT['summary'] = {f'list_size_over_{ref}': TIM['list_size'][f'ratio_vs_{ref}'], f'cast_over_{ref}': TIM['cast'][f'ratio_vs_{ref}'],
                      f'{ref}_over_cast': TIM[ref]['ratio_vs_cast'], 'list_size_over_cast': TIM['list_size']['ratio_vs_cast']}
    OUT['timing'] = TIM; json.dump(OUT, open(os.path.join(a.out, 'time.json'), 'w'), indent=1, default=str); log('written', OUT['summary'])


def main(argv=None):
    a = parse_args(argv); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    {'calibrate': calibrate, 'time': time_lists}[a.command](a, log)
    log('done')


if __name__ == '__main__':
    main()

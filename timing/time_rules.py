"""Single-thread latency of the calibrated rules on the timing queries of a bundle, and the online agreement checks.

Paper: Table III (latency on HOLD, four timing windows), the paragraph "Cost and latency at a common risk target" of
Section VI-B (ratios, medians, 99th percentiles, share of the stopping check) and the implementation details of
Section VI-A (the online stopping point against its offline replay; "significantly slower").

Commands:
    window         one timing window: agreement check of the bundle rules, warm-up, passes in rotated order, per-query
                   median, ratio of the mean latency to that of our rule, paired bootstrap
    share          one pass of a rule with the timer of the stopping check: its share of the query time
    check-replay   bundle of --queries eval: online features against the logged ones (first 1,000 queries), the engine's
                   forests against the bundle's scores, and the online stopping point and exactness of every bundle rule
                   against the offline replay (all timing queries)
    check-visits   bundle of --queries eval: Patience and the distance-adaptive rule stop online where their
                   definitions stop on the logged trajectory (first 1,000 queries)
    check-curve    the threshold curve stops online where the offline staircase evaluation stops it (first 1,000 queries)

Windows of Table III (rules in the order of the first pass; bootstrap resamples, index type, Bonferroni family):
    1  cast level_cap level laet_style fixed frontier_0.05 darth              2,000   int64   seed 0
    2  cast level_cap list_size natural                                      2,000   int64   seed 0
    3  cast level_cap list_size darth darth_plus patience distance_adaptive  20,000  int32   family 12
    4  cast level_cap curve                                                  20,000  int32   family 4

Run it on one pinned core with its hyperthread sibling idle (see timing/README.md), for example
    OMP_NUM_THREADS=1 taskset -c CORE python timing/time_rules.py window --window 3 --bundle BUNDLE --out OUT_DIR ...
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Per window: rules in the order of the first pass, bundle rules whose online visits are compared with the replay
# before timing, a rule run twice to confirm that it is deterministic, rules whose visits are stored before timing,
# and the bootstrap settings.
WINDOWS = {
    1: dict(rules=['cast', 'level_cap', 'level', 'laet_style', 'fixed', 'frontier_0.05', 'darth'],
            check=['cast', 'level_cap', 'level', 'laet_style', 'fixed', 'frontier_0.05'], twice='darth', before=[],
            resamples=2000, index_dtype='int64', seed=0, family=None),
    2: dict(rules=['cast', 'level_cap', 'list_size', 'natural'], check=['cast', 'level_cap'], twice=None, before=['list_size', 'natural'],
            resamples=2000, index_dtype='int64', seed=0, family=None),
    3: dict(rules=['cast', 'level_cap', 'list_size', 'darth', 'darth_plus', 'patience', 'distance_adaptive'], check=['cast'], twice='darth_plus', before=[],
            resamples=20000, index_dtype='int32', seed=None, family=12),
    4: dict(rules=['cast', 'level_cap', 'curve'], check=['cast'], twice=None, before=[],
            resamples=20000, index_dtype='int32', seed=None, family=4),
}

# List sizes of the trajectory log: 46 sizes from 10 to 8,192.
LOG_LIST_SIZES = sorted({int(round(10 * 1.2 ** j)) for j in range(38)} | {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
TIE_TOL = 1e-9


def parse_args(argv=None):
    from engine_rules import add_engine_args, add_rule_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    def common(p, rules=True):
        p.add_argument('--bundle', required=True, help='prefix of the timing bundle (timing/export_bundle.py)')
        add_engine_args(p)
        if rules: add_rule_args(p)
    p = sub.add_parser('window', help='one timing window'); common(p)
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--window', type=int, required=True, choices=sorted(WINDOWS))
    p.add_argument('--rules', nargs='+', default=None, help='rules in the order of the first pass (default: those of the window)')
    p.add_argument('--passes', type=int, default=5); p.add_argument('--warmup', type=int, default=1000, help='queries of the warm-up of every rule')
    p.add_argument('--check-threads', type=int, default=8, help='threads of the agreement checks before timing')
    p.add_argument('--seed', type=int, default=None, help='bootstrap seed (windows 1 and 2: 0; windows 3 and 4: required)')
    p.add_argument('--resamples', type=int, default=None); p.add_argument('--family', type=int, default=None, help='number of ratios of the Bonferroni family')
    p = sub.add_parser('share', help='share of the stopping check in the query time'); common(p)
    p.add_argument('--out', required=True, help='output JSON'); p.add_argument('--rule', default='cast')
    p = sub.add_parser('check-replay', help='online features, forests and stopping points against the bundle'); common(p, rules=False)
    p.add_argument('--out', required=True, help='output JSON'); p.add_argument('--threads', type=int, default=8)
    p.add_argument('--log-seed', type=int, default=1, help='seed of the random arguments of the float32 logarithm check')
    p = sub.add_parser('check-visits', help='Patience and distance-adaptive against the logged trajectory'); common(p)
    p.add_argument('--out', required=True, help='output JSON'); p.add_argument('--threads', type=int, default=8)
    p = sub.add_parser('check-curve', help='threshold curve against the offline staircase evaluation'); common(p)
    p.add_argument('--out', required=True, help='output JSON'); p.add_argument('--threads', type=int, default=8)
    return ap.parse_args(argv)


def window(a, log):
    import numpy as np
    import engine_rules as ER
    import protocol as PR
    W = WINDOWS[a.window]; names = a.rules or W['rules']; assert 'cast' in names
    seed = W['seed'] if a.seed is None else a.seed
    if seed is None: raise SystemExit('windows 3 and 4 need --seed (the seeds of the reported runs are in configs/seeds.json)')
    resamples = a.resamples or W['resamples']; family = a.family or W['family']
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); R, info = ER.build_rules(names, B, M, a); os.makedirs(a.out, exist_ok=True)
    ds, k = M['ds'], M['k']; Q = np.ascontiguousarray(B['Q'], dtype=np.float32); S = B['S']; n = len(Q)
    run = lambda nm, Q_, th: ER.search(eng, ctx, Q_, k, R[nm], th)
    OUT = dict(ds=ds, graph=M['graph'], window=a.window, rules=names, engine_build=eng.build_flags(), passes=a.passes, n=n, **info)

    # ---- before timing: the online visits of the bundle rules equal the offline replay; determinism of the DARTH port
    eq = {}
    for nm in [x for x in W['check'] if x in names]:
        o = run(nm, Q, a.check_threads); eq[nm] = float((o['visits'] == B[f'replay_{nm}_cost']).mean())
    OUT['visits_equal_replay'] = eq
    if W['twice'] in names:
        a1 = run(W['twice'], Q, a.check_threads); a2 = run(W['twice'], Q, a.check_threads)
        OUT['deterministic'] = {W['twice']: float((a1['visits'] == a2['visits']).mean())}
    for nm in [x for x in W['before'] if x in names]:
        o = run(nm, Q, a.check_threads); np.save(os.path.join(a.out, f'{ds}_visits_{nm}_before.npy'), o['visits'])
    log('visits equal replay', eq, 'deterministic', OUT.get('deterministic'))

    # ---- the passes
    NS, VIS, COV = PR.timed_passes(run, names, Q, S, a.passes, a.warmup, log)

    # ---- statistics: per-query median, ratio of means against our rule, paired bootstrap
    OUT['bootstrap'] = dict(seed=seed, resamples=resamples, index_dtype=W['index_dtype'], family=family)
    bidx = PR.bootstrap_indices(n, seed, resamples, W['index_dtype'])
    MED = PR.median_latency(NS)
    for nm in names:
        np.save(os.path.join(a.out, f'{ds}_lat_{nm}.npy'), MED[nm]); np.save(os.path.join(a.out, f'{ds}_visits_{nm}.npy'), VIS[nm])
    TIM = PR.summarize(names, MED, VIS, {nm: int((~COV[nm]).sum()) for nm in names}, bidx, 'cast', family=family,
                       pass_means={nm: [float(x.mean() / 1e3) for x in NS[nm]] for nm in names})
    OUT['timing'] = TIM; OUT['fail_count_rejecting_target'] = PR.rejection_count(n, M['eps'], 0.05)
    if 'curve' in names and 'level_cap' in names:
        TL = {nm: TIM[nm]['mean_us'] for nm in names}
        OUT['curve'] = dict(time_minus_visit_ratio=TIM['curve']['ratio'] - TIM['curve']['visit_ratio'],
                            time_recovered_share=(TL['level_cap'] - TL['curve']) / (TL['level_cap'] - TL['cast']),
                            fail_rate_minus_cast_points=100 * (TIM['curve']['fail_rate'] - TIM['cast']['fail_rate']),
                            pass_ratio_curve=[x / y for x, y in zip(TIM['curve']['pass_means_us'], TIM['cast']['pass_means_us'])],
                            pass_ratio_level_cap=[x / y for x, y in zip(TIM['level_cap']['pass_means_us'], TIM['cast']['pass_means_us'])])
    json.dump(OUT, open(os.path.join(a.out, f'{ds}_time.json'), 'w'), indent=1, default=str); log('written', os.path.join(a.out, f'{ds}_time.json'))


def share(a, log):
    import numpy as np
    import engine_rules as ER
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); R, info = ER.build_rules([a.rule], B, M, a)
    Q = np.ascontiguousarray(B['Q'], dtype=np.float32)
    o = ER.search(eng, ctx, Q, M['k'], R[a.rule], 1, instr=True)
    res = dict(ds=M['ds'], rule=a.rule, engine_build=eng.build_flags(), check_share=float(o['hook_ns'].sum() / o['ns'].sum()), mean_us=float(o['ns'].mean() / 1e3),
               mean_check_us=float(o['hook_ns'].mean() / 1e3), mean_visits=float(o['visits'].mean()))
    json.dump(res, open(a.out, 'w'), indent=1); log(res)


def check_replay(a, log):
    import numpy as np
    import engine_rules as ER
    from cast.forest import forest_raw
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); k = M['k']; L = M['list_size']
    base = ER.interface_fields(B); F = ER.bundle_forests(B, M); grid = base['grid']
    P = dict(method='answer', k=k, L=L)
    def run(nm, Q_, dump=False):
        r = M['rules'][nm]; clf = F['clf'] if (dump or r['type'] in (1, 2)) else None; reg = F['reg'] if r['type'] == 4 else None
        return eng.stop_search(ctx.G, ctx.C, Q_, P, L, ER.bundle_rule(M, nm, base) if not dump else dict(type=0, **base), clf, reg, a.threads, dump, False)
    OUT = dict(ds=M['ds'], graph=M['graph'], engine_build=eng.build_flags(), n=int(len(B['Q'])), rules=list(M['rules']))
    Q = np.ascontiguousarray(B['Q'], dtype=np.float32); S = B['S']; H_ = len(B['features_head'])

    # ---- the engine's forest evaluation against the bundle's scores (classifier) and the reference evaluator (regressor)
    raw = eng.forest_raw(F['clf'], np.ascontiguousarray(B['features_head'].reshape(-1, 7), dtype=np.float64))
    OUT['classifier_max_abs_score'] = float(np.abs(1.0 / (1.0 + np.exp(raw)) - B['score_head'].ravel()).max())
    if 'laet_style' in M['rules']:
        j0 = int(M['rules']['laet_style']['j0'])
        Xr = np.ascontiguousarray(np.column_stack([B['features_head'][:, j0], B['full_head'][:, j0].astype(np.float64)]))
        OUT['regressor_max_abs_raw'] = float(np.abs(eng.forest_raw(F['reg'], Xr) - forest_raw(F['reg'], Xr)).max())
    log('forests', OUT['classifier_max_abs_score'], OUT.get('regressor_max_abs_raw'))

    # ---- online features at every reached checkpoint of the first 1,000 timing queries against the logged ones
    d1 = run('none', Q[:H_], dump=True); c1 = B['c'][:H_]; reached = c1 >= grid[None, :]
    fo, fl = d1['features'], B['features_head']; same = np.isclose(fo, fl, rtol=0, atol=0, equal_nan=True).all(2)
    OUT['features'] = dict(rows=int(reached.sum()), exact_rows=float(same[reached].mean()), max_abs=float(np.nanmax(np.abs(np.where(reached[:, :, None], fo - fl, 0.0)))),
                           score_max_abs=float(np.abs(np.where(reached, d1['score'] - B['score_head'], 0.0)).max()),
                           full_equal=float((d1['full'].astype(bool) == B['full_head'])[reached].mean()),
                           per_feature_exact=[float(np.isclose(fo[:, :, f_], fl[:, :, f_], rtol=0, atol=0, equal_nan=True)[reached].mean()) for f_ in range(7)])
    xr = (np.random.default_rng(a.log_seed).random(200000) * 50).astype(np.float32) + np.float32(1e-6)
    OUT['features']['float32_log_equal_numpy'] = float((eng.npy_logf_vec(xr) == np.log(xr)).mean())
    log('features', OUT['features'])

    # ---- every bundle rule on all timing queries: stopping point and exactness against the offline replay
    ST = {}
    for nm in M['rules']:
        o = run(nm, Q); sim_cost = B[f'replay_{nm}_cost']; sim_fail = B[f'replay_{nm}_fail'].astype(bool)
        ok = np.array([set(x.tolist()) == set(y.tolist()) for x, y in zip(o['ids'], S)])
        ST[nm] = dict(visits_equal=float((o['visits'] == sim_cost).mean()), online_mean_visits=float(o['visits'].mean()), replay_mean_visits=float(sim_cost.mean()),
                      online_coverage=float(ok.mean()), replay_coverage=float(1 - sim_fail.mean()), exactness_equal=float((ok == ~sim_fail).mean()))
        log('stopping point', nm, ST[nm])
    OUT['stopping_point'] = ST; json.dump(OUT, open(a.out, 'w'), indent=1, default=str)


def check_visits(a, log):
    import numpy as np
    import engine_rules as ER
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); k = M['k']; L = M['list_size']; n = ER.HEAD; HVIS = ER.HORIZON_VISITS
    R, info = ER.build_rules(['patience', 'distance_adaptive'], B, M, a)
    Q = np.ascontiguousarray(B['Q'][:n], dtype=np.float32); S = np.ascontiguousarray(B['S'][:n].astype(np.int64))
    sizes = np.array(LOG_LIST_SIZES, np.int32); sizes = sizes[sizes <= L]
    r = eng.log_search(ctx.G, ctx.C, Q, S, dict(method='answer', k=k, L=L), L, sizes, threads=a.threads)
    pv = np.zeros(n, np.int64); dv = np.zeros(n, np.int64)
    for i in range(n):
        # Patience: first visit count at which the answer is full and unchanged for `patience` visits, at most the horizon.
        # Distance-adaptive: first expansion with a full answer whose candidate is farther than (1 + gamma) times the k-th distance.
        p0, p1 = r['ptr'][i], r['ptr'][i + 1]; nv = r['visits'][p0:p1].astype(np.int64); chg = r['chg'][p0:p1].astype(np.int64); full = r['answer_size'][p0:p1] >= k
        eh = int(np.searchsorted(nv, HVIS, side='left')); eh = min(eh, len(nv) - 1)
        w = np.where(full[:eh + 1], nv[:eh + 1] - chg[:eh + 1], -1); Mx = np.maximum.accumulate(w)
        e = int(np.searchsorted(Mx, a.patience, side='left')); e = min(e, eh); pv[i] = nv[e]
        dx = r['dx'][p0:p1].astype(np.float64); dbk = r['dbk'][p0:p1].astype(np.float64)
        hit = np.flatnonzero(full & (dx > (1.0 + a.gamma) * dbk)); dv[i] = nv[hit[0]] if len(hit) else int(r['final_visits'][i])
    op = ER.search(eng, ctx, Q, k, R['patience'], a.threads); od = ER.search(eng, ctx, Q, k, R['distance_adaptive'], a.threads)
    res = dict(ds=M['ds'], n=n, engine_build=eng.build_flags(), **info,
               patience_equal=float((op['visits'] == pv).mean()), distance_adaptive_equal=float((od['visits'] == dv).mean()),
               patience_mismatch_rows=np.flatnonzero(op['visits'] != pv).tolist()[:20], distance_adaptive_mismatch_rows=np.flatnonzero(od['visits'] != dv).tolist()[:20],
               patience_mean_visits=float(pv.mean()), distance_adaptive_mean_visits=float(dv.mean()),
               patience_fail=int((~ER.same_answer(op['ids'], S)).sum()), distance_adaptive_fail=int((~ER.same_answer(od['ids'], S)).sum()))
    json.dump(res, open(a.out, 'w'), indent=1); log(res)


def offline_cost(B, tab, NB):
    """Stop of a staircase table on the logged scores of the first queries of a bundle: running minimum, bucket, first stop."""
    import numpy as np
    s = B['score_head'].astype(np.float64); m = np.minimum.accumulate(s, axis=1); bq = np.searchsorted(B['edges'].astype(np.float64), m, side='right')
    aq = B['full_head'].astype(int); J = s.shape[1]; bf = np.vstack([tab, [NB - 1, NB - 1]])
    stop = bq <= bf[np.arange(J)[None, :], aq]; j = stop.argmax(1); assert stop.any(1).all()
    return B['c'][:len(s)][np.arange(len(s)), j].astype(np.int64), j, m


def check_curve(a, log):
    import numpy as np
    import engine_rules as ER
    B, M = ER.load_bundle(a.bundle); eng, ctx = ER.load_engine(a, M); k = M['k']; J, NB = int(M['J']), int(M['n_buckets'])
    R, info = ER.build_rules(['cast', 'curve'], B, M, a); tab = np.array(info['curve_table'], np.int32)
    bch = np.array(M['rules']['cast']['beta'], np.int32); assert bch.size == 2 * (J - 1); bch = bch.reshape(J - 1, 2)
    n = len(B['score_head']); Q = np.ascontiguousarray(B['Q'][:n], dtype=np.float32)
    off_cast, _, _ = offline_cost(B, bch, NB); off_curve, jc, mrun = offline_cost(B, tab, NB)
    oc = ER.search(eng, ctx, Q, k, R['cast'], a.threads); ov = ER.search(eng, ctx, Q, k, R['curve'], a.threads)
    edges = B['edges'].astype(np.float64); cq = B['c'][:n].astype(np.int64); bad = np.flatnonzero(ov['visits'].astype(np.int64) != off_curve); ties = []
    for i in bad:                                                     # online stop: first checkpoint whose visits equal the engine's
        jo = np.flatnonzero(cq[i] == int(ov['visits'][i])); js = [int(jc[i])] + ([int(jo[0])] if len(jo) else [])
        dist = min(float(np.abs(edges - mrun[i, j]).min()) for j in js); ties.append(dict(row=int(i), stops=js, min_edge_dist=dist, tie=dist <= TIE_TOL))
    res = dict(ds=M['ds'], n=n, engine_build=eng.build_flags(), curve=info['curve_params'],
               offline_cast_equals_replay=float((off_cast == B['replay_cast_cost'][:n]).mean()),
               engine_curve_equals_offline=float((ov['visits'].astype(np.int64) == off_curve).mean()),
               engine_cast_equals_offline=float((oc['visits'].astype(np.int64) == off_cast).mean()),
               curve_mismatch_rows=np.flatnonzero(ov['visits'].astype(np.int64) != off_curve).tolist()[:20],
               cast_mismatch_rows=np.flatnonzero(oc['visits'].astype(np.int64) != off_cast).tolist()[:20],
               table_rows_distinct=int(len({tuple(r) for r in tab.tolist()})), curve_mismatches=ties, tie_tol=TIE_TOL)
    # Agreement holds when the offline evaluation reproduces the replay and every online mismatch of the curve is a tie:
    # at the offline or the online stop, the running minimum of the score lies within tie_tol of a bucket edge.
    res['agree'] = bool(res['offline_cast_equals_replay'] == 1.0 and all(t['tie'] for t in ties))
    json.dump(res, open(a.out, 'w'), indent=1); log(res)


def main(argv=None):
    a = parse_args(argv); t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    {'window': window, 'share': share, 'check-replay': check_replay, 'check-visits': check_visits, 'check-curve': check_curve}[a.command](a, log)
    log('done')


if __name__ == '__main__':
    main()

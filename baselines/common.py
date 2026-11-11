"""Helpers shared by the scripts of the compared rules: engine and data access by sample, our rule as the reference of
the visit ratios, the paired bootstrap of these ratios, and the per-level report of the recall predictors.

Paper: Section VI-A, "Compared methods" and "Implementation details" (every rule is calibrated by the same
fixed-sequence exact binomial test on CAL; ratios to our rule are paired over the EVAL queries).
"""
import json
import os
import sys

import numpy as np

CHUNK = 10000               # queries per chunk of the pool (layout 'pool')
FOREST_KEYS = ('root', 'left', 'right', 'feat', 'thr', 'val', 'leaf', 'miss_left')

# Recorded seeds of the random draws of the compared rules. A JSON file given as --seeds overrides them; the same file
# may carry the seeds of cast.setting.
DEFAULT_SEEDS = dict(darth_regressor=0, darth_forest_check=0, darthplus_regressors=0, darthplus_forest_check=0,
                     policy_models=0, policy_cal_halves=0x13528f0, policy_crossfit=0x13528f1, policy_subsamples=0x13528f2,
                     policy_subsample_crossfit=0x13528f3, policy_bootstrap=1)


def read_seeds(path):
    """(seeds of this module with the overrides of the JSON file `path`, the overrides alone for cast.setting)."""
    over = json.load(open(path)) if path else None
    if over: over = {k: int(v, 0) if isinstance(v, str) else v for k, v in over.items()}       # seeds may be written in hexadecimal
    return dict(DEFAULT_SEEDS, **(over or {})), over


# ------------------------------------------------------------------ arguments, engine, data
def add_setting_args(ap, engine=True, threads=16):
    """Arguments naming one setting (dataset + graph) of one workload and where its data lives."""
    ap.add_argument('--data', required=True, help='directory of the prepared data (DATA/<ds>/...)')
    ap.add_argument('--work', required=True, help='work directory (graphs, logs, reduced arrays, trained models, results)')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', default='A')
    ap.add_argument('--workload', default='k10_kappa1', help='workload: answer size, categories and capacity (cast.datasets.workload_spec)')
    if engine:
        ap.add_argument('--engine', default='engine/build', help='build directory of the engine module cast_engine')
        ap.add_argument('--threads', type=int, default=threads)


def load_engine(engine_dir):
    """Import the search engine from its build directory."""
    sys.path.insert(0, os.path.abspath(engine_dir))
    import cast_engine
    return cast_engine


def open_context(a):
    """The engine module and the search context (graph G, categories C, answer size k) of the setting named by the arguments."""
    eng = load_engine(a.engine)
    from cast import datasets
    return eng, datasets.Context.for_workload(a.data, a.work, a.ds, a.graph, a.workload)


def pool_rows(seeds=None):
    """Rows of SEL, CAL, EVAL and HOLD among the 200,000 pool queries (the permutation of cast.setting), and the permutation."""
    from cast.setting import DEFAULT_SEEDS, N_POOL, SPLIT
    perm = np.random.default_rng(dict(DEFAULT_SEEDS, **(seeds or {}))['pool_permutation']).permutation(N_POOL)
    return {k: perm[a:b] for k, (a, b) in SPLIT.items()}, perm


def take_pool(data_dir, ds, workload, rows):
    """Query vectors and exact answers S of the pool queries `rows`; pool row r is query r % 10,000 of chunk r // 10,000."""
    from cast import datasets
    Q = S = None
    for kch in np.unique(rows // CHUNK):
        sel = np.where(rows // CHUNK == kch)[0]; off = rows[sel] % CHUNK
        q = datasets.queries(data_dir, ds, f'chunk{kch}'); s = datasets.exact_answers(data_dir, ds, f'chunk{kch}', workload)
        if Q is None: Q = np.zeros((len(rows), q.shape[1]), np.float32); S = np.zeros((len(rows), s.shape[1]), np.int64)
        Q[sel] = q[off]; S[sel] = s[off]
    return np.ascontiguousarray(Q, dtype=np.float32), S.astype(np.int64)


def same_answer(ids, S):
    """Per query: the returned answer equals the exact answer S as a set."""
    return (np.sort(ids, 1) == np.sort(S, 1)).all(1)


def save_forest(path, F):
    np.savez(path, **{f: F[f] for f in FOREST_KEYS}, base=np.array(F['base']))


def load_forest(path):
    z = dict(np.load(path)); F = {f: z[f] for f in FOREST_KEYS}; F['base'] = float(z['base'])
    return F


def summarize(cost, fail):
    return dict(mean_visits=float(cost.mean()), fail=int(fail.sum()), fail_rate=float(fail.mean()), n=int(len(cost)))


# ------------------------------------------------------------------ reduced arrays and our rule as reference
def load_frontier_setting(red_dir, ds, graph, FR, seeds=None, keep_hold=False):
    """Load a setting and fit its interface with the layout, target and interface recorded by pipeline/frontier.py."""
    from cast.setting import fit_interface, load_setting
    itf = FR['interface']
    S = load_setting(red_dir, ds, graph, layout=FR['layout'], target=FR['target'], keep_hold=keep_hold,
                     fit_block=itf['fit_block'], stride=itf['stride'], seeds=seeds, resplit_seed=itf.get('resplit_seed'))
    return fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])


def staircase_outcomes(S, F, idx):
    """Per-query visits (int64) and failure flags of a deterministic rule with stopping matrix F on the queries idx."""
    from cast.rules import evaluate_rule
    F = np.maximum.accumulate(np.clip(F, 0, 1), axis=1); F[:, -1] = 1.0
    c, r = evaluate_rule(S, F, idx)
    return c.astype(np.int64), r.round().astype(bool)


def common_target_staircase(S, FR, eps, delta):
    """Our rule at the common target: the chain calibrated on CAL at (eps, delta). Returns (rule record, EVAL visits, EVAL failures)."""
    from cast.compared import calibrate_staircase, rule_F
    rec, _ = calibrate_staircase(S, eps, delta, np.array(FR['chain_grid'], np.int64))
    assert rec is not None and rec['type'] == 1
    c, f = staircase_outcomes(S, rule_F(S, dict(RULE=dict(cast=rec)), 'cast', S.EVAL), S.EVAL)
    return rec, c, f


def frontier_staircase(S, record, idx):
    """Stopping matrix on idx of the frontier rule `record` of pipeline/frontier.py (its chain element is recomputed on SEL)."""
    from cast.chain import chain_element
    from cast.rules import staircase_F, terminal_F
    th = record['theta']; th = th if th == 'terminal' else int(th)
    return th, (terminal_F(S, idx) if th == 'terminal' else staircase_F(S, chain_element(S, S.SEL, th)[0], idx))


def load_stops(red_dir, ds, graph, layout, target):
    """Stop times of the list-size and distance-adaptive rules of the logged traversal, in the row order of the setting's files."""
    from cast.setting import N_CHUNKS, PARTS
    names = [f'chunk{k}' for k in range(N_CHUNKS)] if layout == 'pool' else list(PARTS)
    EX = [np.load(f'{red_dir}/{ds}_{graph}_{nm}_stops.npz') for nm in names]
    list_stop = np.concatenate([e['list_stop'] for e in EX]).astype(np.int64)
    adaptive_stop = np.concatenate([e['adaptive_stop'] for e in EX])
    X = np.concatenate([np.load(f'{red_dir}/{ds}_{graph}_{nm}.npz')[target] for nm in names])
    return list_stop, adaptive_stop, X, EX[0]['list_sizes'].astype(np.int64), EX[0]['gammas']


def sample_rows(S):
    """Rows of SEL, CAL and EVAL in the order of the setting's files (pool rows for layout 'pool')."""
    if S.pool_rows is not None: return {k: S.pool_rows[k] for k in ('SEL', 'CAL', 'EVAL')}
    return dict(SEL=S.SEL, CAL=S.CAL, EVAL=S.EVAL)


# ------------------------------------------------------------------ paired bootstrap of visit ratios
def bootstrap_weights(n, seed, resamples=2000):
    """Resampling weights of `resamples` paired bootstrap resamples of n queries."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)


def visit_ratio(W, x, y, with_se=False):
    """Ratio of the mean visits x / y and its one-sided 2.5% bootstrap lower bound."""
    ra = (W @ x.astype(np.float64)) / (W @ y.astype(np.float64))
    out = dict(ratio=float(x.mean() / y.mean()), lower_bound_one_sided=float(np.quantile(ra, 0.025)))
    if with_se: out['se'] = float(ra.std(ddof=1))
    return out


# ------------------------------------------------------------------ recall predictors at the nine risk levels
def levels_report(red_dir, ds, graph, FR, run_prefix, out_path, seed, seeds=None, log=print):
    """Pair the EVAL runs of a recall predictor at every risk level with the frontier rule of the same level (Fig. 4).

    run_prefix names the output of the `levels` step of baselines/darth.py or baselines/darth_plus.py: RUN.json and
    RUN_eval.npz with the per-query EVAL visits and failures in the order of the EVAL sample.
    """
    S = load_frontier_setting(red_dir, ds, graph, FR, seeds); EVAL = S.EVAL
    RUN = json.load(open(run_prefix + '.json')); EV = dict(np.load(run_prefix + '_eval.npz')); assert RUN['ds'] == ds
    E = [float(e) for e in FR['config']['levels']]; assert E == RUN['levels']
    W = bootstrap_weights(len(EVAL), seed)
    OUT = dict(ds=ds, graph=graph, n_eval=int(len(EVAL)), level=RUN['level'], bootstrap_seed=seed, Rt=RUN['Rt'], determinism=RUN.get('determinism'), levels={})
    for e in E:
        r_ = FR['rules']['chain'][str(e)]; row = {}
        if not r_.get('calibrated'): row['frontier'] = 'no rule'; OUT['levels'][str(e)] = row; continue
        th, Fch = frontier_staircase(S, r_, EVAL); c_ch, f_ch = staircase_outcomes(S, Fch, EVAL)
        row['frontier'] = dict(theta=th, mean_visits=float(c_ch.mean()), fail=int(f_ch.sum()), fail_rate=float(f_ch.mean()), recorded_mean_visits=r_.get('mean_visits'),
                                equals_recorded=bool(abs(float(c_ch.mean()) - float(r_.get('mean_visits'))) < 1e-9))
        if f'visits_{e}' not in EV: row['rule'] = dict(calibrated=False); OUT['levels'][str(e)] = row; continue
        vis = EV[f'visits_{e}']; fl = EV[f'fail_{e}']; assert len(vis) == len(c_ch)
        row['rule'] = dict(calibrated=True, Rt=RUN['Rt'][str(e)], cal_fail=RUN['cal_fail_at_Rt'][str(e)], mean_visits=float(vis.mean()), fail=int(fl.sum()), fail_rate=float(fl.mean()),
                           visit_ratio=visit_ratio(W, vis, c_ch))
        OUT['levels'][str(e)] = row; log('level', e, row['frontier']['mean_visits'], row['rule'].get('mean_visits'), row['rule'].get('visit_ratio'))
    json.dump(OUT, open(out_path, 'w'), indent=1, default=str)
    return OUT

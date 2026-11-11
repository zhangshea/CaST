"""One setting (a dataset searched on one graph): checkpoint arrays of its labeled queries, the five samples and the
declared stopping interface.

Paper: Section III-B (checkpoints), Section III-C (declared interface), Section III-D (samples FIT, SEL, CAL, EVAL, HOLD)
and the implementation details of Section VI-A (score model, features, bucket edges).

Reduced arrays (written by pipeline/reduce_logs.py), one file per group of queries:
    c         (n, J) int64    visits c_j at the checkpoints
    features  (n, J, 7) f64   the seven trajectory features at the checkpoints
    full      (n, J) bool     flag a_j: the answer holds k items
    X_exact   (n,)   int64    completion time X of the exact answer
    X_recall09 (n,)  int64    completion time of the recall target 0.9

Two layouts of the files of a setting:
    'pool'   <ds>_<graph>_fitpool.npz (queries from which FIT is drawn) and <ds>_<graph>_chunk<k>.npz, k = 0..19
             (200,000 queries, permuted once and cut into SEL, CAL, EVAL, HOLD);
    'parts'  <ds>_<graph>_{fit,sel,cal,eval}.npz, each sample drawn at data preparation (no HOLD).
"""
import numpy as np

from . import grid as G
from .monotone_class import MonotoneClass, state_index

N_POOL, N_CHUNKS = 200000, 20
N_FIT = 2500
SPLIT = dict(SEL=(0, 20000), CAL=(20000, 60000), EVAL=(60000, 180000), HOLD=(180000, 200000))
PARTS = ('fit', 'sel', 'cal', 'eval')
PART_SIZES = (2500, 20000, 40000, 120000)

# Recorded seeds of the random draws (see docs/RANDOM_DRAWS.md). They are parameters of load_setting.
DEFAULT_SEEDS = dict(fit_permutation=0x13528ee, pool_permutation=0x13528ef, cal_uniforms=7, sel_uniforms=8, score_model=0, laet_regressor=0)


class Setting:
    """Arrays of all loaded queries (rows), the row indices of the samples, and the interface once it is fitted."""

    def __init__(self):
        self.grid, self.J, self.H = G.GRID, G.J, G.H
        self.coef_cache = {}


def _load(files, target):
    Ds = [np.load(f) for f in files]
    return dict(c=np.concatenate([d['c'] for d in Ds]), features=np.concatenate([d['features'] for d in Ds]),
                full=np.concatenate([d['full'] for d in Ds]), X=np.concatenate([d[target] for d in Ds]))


def load_setting(red_dir, ds, graph, layout='pool', target='X_exact', keep_hold=False, fit_block=0, stride=1, seeds=None, resplit_seed=None):
    """Load the reduced arrays of a setting and draw its samples.

    target      column used as the completion time X ('X_exact', or 'X_recall09' for the recall target 0.9);
    keep_hold   layout 'pool' only: keep the HOLD rows (they are dropped on load otherwise);
    fit_block   layout 'pool' only: FIT is block `fit_block` of 2,500 queries of the FIT permutation (0 in the main results);
    stride      keep every stride-th checkpoint, counted back from the last one (1 in the main results);
    resplit_seed  layout 'pool' only: draw SEL, CAL and EVAL again by permuting the first 180,000 entries of the pool
                permutation with this seed (FIT and HOLD keep their queries; None in the main results).
    """
    seeds = dict(DEFAULT_SEEDS, **{k: int(v, 0) if isinstance(v, str) else v for k, v in (seeds or {}).items()})
    S = Setting(); S.ds, S.graph, S.layout, S.target, S.seeds = ds, graph, layout, target, seeds
    if layout == 'pool':
        fitpool = _load([f'{red_dir}/{ds}_{graph}_fitpool.npz'], target)
        perm_fit = np.random.default_rng(seeds['fit_permutation']).permutation(len(fitpool['c']))
        fit_rows = perm_fit[N_FIT * fit_block:N_FIT * (fit_block + 1)]
        pool = _load([f'{red_dir}/{ds}_{graph}_chunk{k}.npz' for k in range(N_CHUNKS)], target); assert len(pool['c']) == N_POOL
        perm = np.random.default_rng(seeds['pool_permutation']).permutation(N_POOL)
        if resplit_seed is not None:
            perm[:SPLIT['EVAL'][1]] = np.random.default_rng(resplit_seed).permutation(perm[:SPLIT['EVAL'][1]])
        sl = {k: perm[a:b] for k, (a, b) in SPLIT.items()}
        used = ('SEL', 'CAL', 'EVAL', 'HOLD') if keep_hold else ('SEL', 'CAL', 'EVAL')
        assert not np.isin(sl['HOLD'], np.concatenate([sl['SEL'], sl['CAL'], sl['EVAL']])).any()
        keep = np.sort(np.concatenate([sl[k] for k in used]))
        pos = -np.ones(N_POOL, np.int64); pos[keep] = np.arange(len(keep))
        D = {k: np.concatenate([fitpool[k][fit_rows], pool[k][keep]]) for k in fitpool}; off = len(fit_rows); del pool
        S.FIT = np.arange(off)
        S.SEL, S.CAL, S.EVAL = (off + pos[sl[k]] for k in ('SEL', 'CAL', 'EVAL'))
        S.HOLD = off + pos[sl['HOLD']] if keep_hold else np.zeros(0, np.int64)
        S.pool_rows = sl                                    # rows of each sample in the order of the 200,000 pool queries
    elif layout == 'parts':
        R = {p: _load([f'{red_dir}/{ds}_{graph}_{p}.npz'], target) for p in PARTS}
        D = {k: np.concatenate([R[p][k] for p in PARTS]) for k in R['fit']}
        o = np.cumsum([0] + [len(R[p]['c']) for p in PARTS]); del R
        S.FIT, S.SEL, S.CAL, S.EVAL = (np.arange(o[i], o[i + 1]) for i in range(4)); S.HOLD = np.zeros(0, np.int64)
        assert (len(S.FIT), len(S.SEL), len(S.CAL), len(S.EVAL)) == PART_SIZES
        S.pool_rows = None
    else:
        raise ValueError(layout)
    if stride != 1:
        keep_j = np.arange(S.J - 1, -1, -stride)[::-1]
        D['c'] = D['c'][:, keep_j]; D['features'] = D['features'][:, keep_j]; D['full'] = D['full'][:, keep_j]
        S.grid = S.grid[keep_j]; S.J = len(keep_j)
    S.c, S.features, S.full, S.X = D['c'], D['features'], D['full'], D['X']
    S.dC = np.diff(S.c, axis=1); S.n = len(S.c)
    # One uniform per CAL query and per SEL query, shared by all candidate rules (the stopping draw xi of Definition 2).
    S.u_cal = np.random.default_rng(seeds['cal_uniforms']).random(len(S.CAL))
    S.u_sel = np.random.default_rng(seeds['sel_uniforms']).random(len(S.SEL))
    return S


def quantile_levels(n_buckets):
    """Quantile levels k / n_buckets of the bucket edges.

    The multiples of 0.05 take the values of numpy.linspace(0.05, 0.95, 19); levels that agree to two decimals are merged,
    so this function is meant for n_buckets <= 100 (use exact_levels in fit_interface otherwise).
    """
    base = np.linspace(0.05, 0.95, 19)
    d = {round(float(x), 2): float(x) for x in np.arange(1, n_buckets) / n_buckets}
    d.update({round(float(x), 2): float(x) for x in base})
    return np.array([d[k] for k in sorted(d)])


def make_score_model(kind='gbdt', seed=0):
    """Score model of the interface: a gradient-boosted classifier (main results) or a standardized logistic regression."""
    if kind == 'gbdt':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    if kind == 'logreg':
        # A missing feature value (for example before the answer is full) is replaced by the FIT median, with an indicator.
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(max_iter=5000))
    raise ValueError(kind)


def fit_interface(S, n_buckets=100, model='gbdt', exact_levels=False):
    """Fit the score model on FIT and declare the interface z_j = (b_j, a_j) (Section III-C).

    The score s_j = 1 - P(answer complete at checkpoint j); m_j is its running minimum; the bucket b_j counts the edges
    e_i <= m_j, with edges at quantiles of the FIT scores pooled over all checkpoints; a_j is the flag `full`.
    """
    J, nf = S.J, S.features.shape[2]
    S.model = make_score_model(model, S.seeds['score_model'])
    if model == 'logreg': assert not np.isinf(S.features).any()
    y = (S.c[S.FIT] >= S.X[S.FIT, None]).astype(int).ravel()
    S.model.fit(S.features[S.FIT].reshape(-1, nf), y)
    S.s = 1 - S.model.predict_proba(S.features.reshape(-1, nf))[:, 1].reshape(S.n, J)
    S.m = np.minimum.accumulate(S.s, axis=1)
    levels = np.arange(1, n_buckets) / n_buckets if exact_levels else quantile_levels(n_buckets)
    S.edges = np.quantile(S.s[S.FIT].ravel(), levels) if len(levels) else np.zeros(0)
    S.NB = len(S.edges) + 1
    S.b = np.searchsorted(S.edges, S.m, side='right'); S.a = S.full.astype(int)
    S.Z = state_index(S.b, S.a); S.cls = MonotoneClass(J, S.NB)
    # Tie-break weight of the chain dynamic program: any weight above the number of free cells selects the same rule.
    S.tie_weight = (1 << 17) if n_buckets * 2 * (J - 1) >= 2 ** 15 else (1 << 15)
    assert (J - 1) * S.cls.ns < S.tie_weight
    S.coef_cache = {}
    return S

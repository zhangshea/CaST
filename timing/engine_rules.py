"""Rule records of the timed online search, built from a timing bundle and from the files of the compared rules.

Paper: Table III (latency on HOLD) and the paragraph "Cost and latency at a common risk target" of Section VI-B. Every
timed rule is one entry (rule, classifier forest, regressor forest, list size) passed to the engine's stop_search.

Rule types of the engine:
    0  no stopping rule: the search ends at the last checkpoint of the rule's grid, or by itself when the closest
       unexpanded candidate is farther than all entries of the list (a single checkpoint at 2^30 visits leaves only this)
    1  staircase table beta of shape (J - 1, 2) on the interface (score bucket, flag)
    2  threshold lam on the running minimum of the score, with a cap at checkpoint jcap (Level, Level+cap)
    3  fixed checkpoint jfix (Fixed)
    4  LAET-style: from checkpoint j0, stop once the visits reach mult times the predicted completion time
    6  DARTH (recall predictor on eleven answer features, adaptive prediction intervals)
    8  DARTH+ (adds eight query statistics and a lower-quantile predictor)
    9  Patience (answer full and unchanged for pdelta visits, or hvis visits reached)
    10 distance-adaptive (answer full and the next candidate farther than (1 + gamma) times the k-th answer distance)
"""
import json
import os
import sys

import numpy as np

FOREST_FIELDS = ('root', 'left', 'right', 'feat', 'thr', 'val', 'leaf', 'miss_left')
HORIZON_VISITS = 32768                 # last checkpoint: Patience never runs past it
HEAD = 1000                            # queries of a bundle whose logged features and scores are kept

# Rules of the third timing window of Table III, in the order in which the first pass runs them.
THIRD_WINDOW = ['cast', 'level_cap', 'list_size', 'darth', 'darth_plus', 'patience', 'distance_adaptive']


def add_engine_args(ap):
    ap.add_argument('--data', required=True, help='prepared data directory (DATA of the package layout)')
    ap.add_argument('--work', required=True, help='work directory holding graphs/ (WORK of the package layout)')
    ap.add_argument('--engine', default='engine/build', help='build directory of the engine module cast_engine')
    ap.add_argument('--variant', default='main', help='category definition of the dataset')
    ap.add_argument('--kappa', type=int, default=1, help='capacity of every category')


def add_rule_args(ap):
    g = ap.add_argument_group('parameters and files of the compared rules (outputs of baselines/)')
    g.add_argument('--list-size', type=int, default=None, help='calibrated list size of the rule list_size')
    g.add_argument('--patience', type=int, default=None, help='calibrated number of visits of the rule patience')
    g.add_argument('--gamma', type=float, default=None, help='calibrated gamma of the rule distance_adaptive')
    g.add_argument('--darth-model', default=None, help='prefix of the DARTH model files (baselines/darth.py train): PREFIX.npz, PREFIX.json')
    g.add_argument('--darth-calibration', default=None, help='output JSON of baselines/darth.py calibrate (recall target at the common target)')
    g.add_argument('--darth-plus-model', default=None, help='prefix of the DARTH+ model files (baselines/darth_plus.py train): PREFIX.json, PREFIX_reg.npz, PREFIX_q<level>.npz')
    g.add_argument('--darth-plus-levels', default=None, help='output JSON of baselines/darth_plus.py levels (quantile level, recall target at the common target)')
    g.add_argument('--curve', default=None, help='output JSON of baselines/threshold_curve.py fit (parameters selected on SEL, offset calibrated on CAL)')


def load_engine(a, M):
    """Import the engine and load the graph and the categories of the setting of the bundle description M."""
    sys.path.insert(0, os.path.abspath(a.engine))
    import cast_engine as eng
    from cast.datasets import Context
    return eng, Context(a.data, a.work, M['ds'], M['graph'], M['k'], variant=a.variant, kappa=a.kappa)


def search(eng, ctx, Q, k, entry, threads, instr=False):
    """Run one rule entry on the queries Q; per query: ids, ns (time inside the engine), visits, and hook_ns with instr."""
    rule, clf, reg, L = entry
    return eng.stop_search(ctx.G, ctx.C, Q, dict(method='answer', k=k, L=int(L)), int(L), rule, clf, reg, threads, False, instr)


def same_answer(ids, S):
    """Exactness: the returned set equals the exact answer."""
    return (np.sort(ids, 1) == np.sort(S, 1)).all(1)


def load_bundle(prefix):
    """Arrays and description of a timing bundle written by timing/export_bundle.py."""
    return dict(np.load(prefix + '.npz')), json.load(open(prefix + '.json'))


def interface_fields(B):
    """Fields shared by the rules that read the interface: checkpoint grid, bucket edges, and the values of
    log(n) and log1p(n) for integer n stored in the bundle (two of the seven features), so that the online features equal the
    logged ones."""
    grid = B['grid'].astype(np.int32); edges = B['edges'].astype(np.float64)
    return dict(grid=grid, edges=edges, log_tab=B['log_tab'].astype(np.float64), log1p_tab=B['log1p_tab'].astype(np.float64))


def bundle_forests(B, M):
    """Flattened score model ('clf') and LAET-style regressor ('reg') of the bundle."""
    return {nm: dict(**{f: B[f'{nm}_{f}'] for f in FOREST_FIELDS}, base=M[f'{nm}_base']) for nm in ('clf', 'reg') if f'{nm}_root' in B}


def bundle_rule(M, name, base):
    """Engine rule of a calibrated rule record of the bundle."""
    r = M['rules'][name]; d = dict(type=int(r['type']), **base)
    if r['type'] == 1: d['beta'] = np.array(r['beta'], np.int32)
    if r['type'] == 2: d.update(lam=float(r['lam']), jcap=int(r['jcap']))
    if r['type'] == 3: d['jfix'] = int(r['jfix'])
    if r['type'] == 4: d.update(j0=int(r['j0']), mult=float(r['mult']))
    return d


def bundle_entry(M, name, base, forests):
    t = M['rules'][name]['type']
    return (bundle_rule(M, name, base), forests['clf'] if t in (1, 2) else None, forests['reg'] if t == 4 else None, M['list_size'])


def staircase_entry(beta, base, forests, L):
    """A staircase table that is not in the bundle (another calibration, or the threshold curve), run on a list of L entries."""
    return (dict(type=1, beta=np.asarray(beta).ravel().astype(np.int32), **base), forests['clf'], None, L)


def natural_rule():
    """No checkpoint is ever reached, so only the end of the list search applies."""
    return dict(type=0, grid=np.array([1 << 30], np.int32))


def load_forest(path):
    z = dict(np.load(path)); F = {f: z[f] for f in FOREST_FIELDS}; F['base'] = float(z['base']); return F


def schedule_of(first, Rt):
    """Prediction intervals of a recall target: recall is a multiple of 1 / k, so they are those of ceil(Rt k) / k."""
    key = f'{np.ceil(round(Rt * 10, 6)) / 10:.1f}'; return first[key]


def load_darth(model_prefix, calibration_json):
    """DARTH regressor, its intervals for targets in (0.9, 1], and the recall target calibrated on CAL."""
    Fz = dict(np.load(model_prefix + '.npz')); Mj = json.load(open(model_prefix + '.json'))
    F = {f: Fz[f'reg_{f}'] for f in FOREST_FIELDS}; F['base'] = Mj['base']
    Rt = json.load(open(calibration_json))['Rt_calibrated']; assert Rt is not None and Rt > 0.9
    return F, Mj['schedule']['1.0'], Rt


def darth_rule(base, Rt, sc):
    return dict(type=6, grid=base['grid'], edges=base['edges'], Rt=float(Rt), ipi=int(sc['ipi']), mpi=int(sc['mpi']))


def load_darth_plus(model_prefix, levels_json):
    """DARTH+ regressors, the selected quantile level, the recall target calibrated on CAL and its intervals."""
    TJ = json.load(open(model_prefix + '.json')); RJ = json.load(open(levels_json))
    lq = RJ['quantile_level']; Rp = RJ['Rt_common_target']; assert Rp is not None, 'no DARTH+ rule at the common target'
    Fr = load_forest(model_prefix + '_reg.npz'); qF = load_forest(f'{model_prefix}_q{lq}.npz')
    return Fr, qF, schedule_of(TJ['schedule'], Rp), Rp, lq


def darth_plus_rule(base, Rt, sc, qF):
    return dict(type=8, grid=base['grid'], edges=base['edges'], Rt=float(Rt), ipi=int(sc['ipi']), mpi=int(sc['mpi']), qforest=qF)


def curve_table(c, s_, u, cap, j0, J, NB):
    """Threshold curve beta_j(a) = clip(round(c + s max(0, j - j0) + u (a - 1)), -1, B - 1), nondecreasing in j,
    equal to B - 1 from checkpoint `cap` on, with beta_j(0) <= beta_j(1): a staircase of the class with an offset, a slope, a break point, a gap between the flags
    and a cap."""
    j = np.arange(J - 1, dtype=np.float64)[:, None]; a = np.array([0.0, 1.0])[None, :]
    b = np.rint(c + s_ * np.maximum(j - j0, 0.0) + u * (a - 1.0)); b = np.clip(b, -1, NB - 1); b = np.maximum.accumulate(b, axis=0); b[cap:, :] = NB - 1
    b[:, 0] = np.minimum(b[:, 0], b[:, 1])
    return b.astype(np.int32)


def load_curve(path):
    """Parameters of the threshold curve: (s, j0, u, cap) selected on SEL and the offset c calibrated on CAL."""
    d = json.load(open(path)); sel = d['selected']
    return dict(s=float(sel['s']), j0=int(sel['j0']), u=int(sel['u']), cap=int(sel['cap']), c=float(d['c_cal']))


def _need(a, *names):
    for n in names:
        if getattr(a, n) is None: raise SystemExit(f"the selected rules need --{n.replace('_', '-')}")


def build_rules(names, B, M, a):
    """Entries {name: (rule, clf, reg, list size)} of the rules `names`, and the parameters read for them.

    Names found in the bundle (cast, level_cap, level, laet_style, fixed, none, frontier_<eps>) use its records. The
    others are: list_size and natural (list search without a stopping rule, with the calibrated list size or the full
    list), darth, darth_plus, patience, distance_adaptive, and curve.
    """
    base = interface_fields(B); forests = bundle_forests(B, M); L = M['list_size']; one = np.array([1 << 30], np.int32)
    R, info = {}, {}
    for nm in names:
        if nm in M['rules']:
            R[nm] = bundle_entry(M, nm, base, forests)
        elif nm == 'list_size':
            _need(a, 'list_size'); assert 1 <= a.list_size <= L
            R[nm] = (natural_rule(), None, None, a.list_size); info['list_size'] = a.list_size
        elif nm == 'natural':
            R[nm] = (natural_rule(), None, None, L)
        elif nm == 'darth':
            _need(a, 'darth_model', 'darth_calibration'); Fd, dz, Rd = load_darth(a.darth_model, a.darth_calibration)
            R[nm] = (darth_rule(base, Rd, dz), None, Fd, L); info['darth_target'] = Rd
        elif nm == 'darth_plus':
            _need(a, 'darth_plus_model', 'darth_plus_levels'); Fr, qF, sc, Rp, lq = load_darth_plus(a.darth_plus_model, a.darth_plus_levels)
            R[nm] = (darth_plus_rule(base, Rp, sc, qF), None, Fr, L); info.update(darth_plus_target=Rp, darth_plus_quantile=lq)
        elif nm == 'patience':
            _need(a, 'patience')
            R[nm] = (dict(type=9, grid=one, pdelta=a.patience, hvis=HORIZON_VISITS), None, None, L); info['patience'] = a.patience
        elif nm == 'distance_adaptive':
            _need(a, 'gamma')
            R[nm] = (dict(type=10, grid=one, gamma=a.gamma), None, None, L); info['gamma'] = a.gamma
        elif nm == 'curve':
            _need(a, 'curve'); p = load_curve(a.curve); J, NB = int(M['J']), int(M['n_buckets'])
            assert len(base['grid']) == J and len(base['edges']) == NB - 1
            tab = curve_table(p['c'], p['s'], p['u'], p['cap'], p['j0'], J, NB)
            R[nm] = staircase_entry(tab, base, forests, L); info.update(curve_params=p, curve_table=tab.tolist())
        else:
            raise SystemExit(f'unknown rule {nm}')
    return R, info

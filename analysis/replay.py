"""Helpers of the analyses that replay calibrated rules on logged queries (Section VI-B): loading a setting with the
interface of a frontier run, appending queries logged after a change, rebuilding the chain."""
import json
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.chain import chain_element
from cast.monotone_class import state_index
from cast.rules import evaluate_rule, staircase_F, terminal_F
from cast.setting import N_POOL, SPLIT, fit_interface, load_setting


def setting_of_frontier(red, ds, graph, FR, keep_hold=False, seeds=None):
    """Load a setting and fit the interface with which the frontier output FR (pipeline/frontier.py) was computed.

    seeds: path of a JSON file overriding the recorded seeds of cast/setting.py, or None.
    """
    itf = FR['interface']; seeds = json.load(open(seeds)) if seeds else None
    S = load_setting(red, ds, graph, layout=FR['layout'], target=FR['target'], keep_hold=keep_hold, fit_block=itf['fit_block'],
                     stride=itf['stride'], seeds=seeds, resplit_seed=itf.get('resplit_seed'))
    return fit_interface(S, n_buckets=itf['buckets'], model=itf['model'], exact_levels=itf['exact_levels'])


def append_rows(S, c, features, full, X):
    """Append queries to a setting whose interface is fitted: they are scored by the same model and bucketed by the same
    edges. Returns the row indices of the appended queries."""
    J = S.J; n0 = S.n; nT = len(c); assert c.shape[1] == J
    sT = 1 - S.model.predict_proba(features.reshape(-1, features.shape[2]))[:, 1].reshape(nT, J); mT = np.minimum.accumulate(sT, axis=1)
    S.c = np.concatenate([S.c, c.astype(S.c.dtype)]); S.features = np.concatenate([S.features, features.astype(S.features.dtype)])
    S.full = np.concatenate([S.full, full.astype(S.full.dtype)]); S.X = np.concatenate([S.X, X.astype(S.X.dtype)])
    S.dC = np.diff(S.c, axis=1); S.n = len(S.c)
    S.s = np.concatenate([S.s, sT]); S.m = np.concatenate([S.m, mT])
    S.b = np.concatenate([S.b, np.searchsorted(S.edges, mT, side='right')]); S.a = np.concatenate([S.a, full.astype(int)])
    S.Z = state_index(S.b, S.a)
    return n0 + np.arange(nT)


def sel_chain(S, multipliers):
    """Chain elements on SEL for the given multipliers, and the map (candidate, queries) -> stopping matrix."""
    CH = {int(l_): chain_element(S, S.SEL, int(l_))[0] for l_ in multipliers}
    mk = lambda th, idx: terminal_F(S, idx) if th == 'terminal' else staircase_F(S, CH[th], idx)
    return CH, mk


def frontier_candidates(FR):
    """Calibrated chain candidate of every risk level of a frontier output: {level: 'terminal' | multiplier | None}."""
    TH = {}
    for e in FR['config']['levels']:
        r_ = FR['rules']['chain'][str(e)]; th = r_.get('theta')
        TH[e] = None if th is None or not r_.get('calibrated', True) else (th if th == 'terminal' else int(th))
    return TH


def check_frontier_reproduced(S, FR, mk, TH):
    """The frontier rules rebuilt from the arrays must have the EVAL mean cost stored in the frontier output."""
    chk = {}
    for e, th in TH.items():
        if th is None: chk[str(e)] = None; continue
        ce, _ = evaluate_rule(S, mk(th, S.EVAL), S.EVAL)
        chk[str(e)] = dict(mean_visits=float(ce.mean()), mean_visits_frontier=FR['rules']['chain'][str(e)].get('mean_visits'))
    ok = bool(all(v is None or abs(v['mean_visits'] - v['mean_visits_frontier']) <= 1e-6 * max(1.0, abs(v['mean_visits_frontier']))
                  for v in chk.values()))
    assert ok, 'the frontier output does not belong to these arrays'
    return chk


def risk_test(k, n, eps, level):
    """One-sided exact binomial test of H0: risk <= eps from k failures in n queries."""
    p = float(binom.sf(k - 1, n, eps)); return dict(p_one_sided=p, reject=bool(p <= level))


def reading(cost, fail, eps, level):
    """Mean visits, failures and the risk test of a rule on one sample."""
    k = int(np.asarray(fail).sum()); n = len(cost)
    return dict(mean_visits=float(np.mean(cost)), fail=k, fail_rate=k / n, n=n, **risk_test(k, n, eps, level))

"""Calibration of our rule and of the rules that read the same features, at one common risk target.

Paper: Section VI-A, "Compared methods" (first group: Level, Level+cap, LAET-style, Fixed) and "Implementation details"
(every rule, ours included, is calibrated by the same fixed-sequence test on CAL at eps = 0.05 and delta = 0.05).
These are the rules of Table II (ratio columns) and Table III.

Rule records carry the integer type the search engine uses online:
    0 no early stop, 1 staircase, 2 threshold on the running minimum of the score with a cap, 3 fixed checkpoint,
    4 LAET-style (stop when the visits reach a multiple of a predicted completion time).
"""
import numpy as np

from . import grid as G
from .calibration import fixed_sequence
from .chain import chain_element
from .forest import export_forest
from .rules import evaluate_rule, fixed_budget_F, level_F, staircase_F, terminal_F

TYPE_NONE, TYPE_STAIRCASE, TYPE_LEVEL, TYPE_FIXED, TYPE_LAET = 0, 1, 2, 3, 4


def laet_F(S, pred, j0, mult, idx):
    """LAET-style: from checkpoint j0 on, stop once the visits reach mult times the predicted completion time."""
    return np.where(np.arange(S.J)[None, :] < j0, 0.0, (S.c[idx] >= mult * pred[idx, None]).astype(float))


# Levels of the tests run on SEL to choose the cap of Level+cap and the checkpoint of LAET-style (they never read CAL).
CAP_PROXY_LEVEL, LAET_PROXY_LEVEL = 0.01, 0.05


def level_candidates(S):
    """Thresholds of Level and Level+cap: 1,801 quantiles of the FIT scores, conservative first."""
    return np.quantile(S.s[S.FIT].ravel(), np.linspace(0.0, 0.9, 1801))


LAET_MULTIPLIERS = np.geomspace(64.0, 0.25, 1601)


def calibrate_level_cap(S, eps, delta):
    """Level+cap: the cap is chosen on SEL by the same test run on SEL, then the threshold is calibrated on CAL."""
    LAM = level_candidates(S); capJ = lambda C: G.cap_checkpoint(C, S.grid); SEL = S.SEL
    caps = []
    for C in G.CAPS:
        ls_, _ = fixed_sequence(S, LAM, lambda lam, idx, Jc_=capJ(C): level_F(S, lam, Jc_, idx), eps, idx=SEL, U=S.u_sel, delta=CAP_PROXY_LEVEL)
        if ls_ is None: continue
        ca, _ = evaluate_rule(S, level_F(S, ls_, capJ(C), SEL), SEL); caps.append(dict(C=C, Jc=capJ(C), sel_visits=float(ca.mean())))
    if not caps: return None
    bp = min(caps, key=lambda x: x['sel_visits']); Jc = bp['Jc']
    lam, _ = fixed_sequence(S, LAM, lambda lam, idx: level_F(S, lam, Jc, idx), eps, delta=delta)
    return None if lam is None else dict(type=TYPE_LEVEL, lam=float(lam), jcap=int(Jc), cap=bp['C'])


def calibrate_level(S, eps, delta):
    """Level: one threshold on the running minimum of the score for the whole search."""
    J = S.J
    lam, _ = fixed_sequence(S, level_candidates(S), lambda lam, idx: level_F(S, lam, J - 1, idx), eps, delta=delta)
    return None if lam is None else dict(type=TYPE_LEVEL, lam=float(lam), jcap=int(J - 1))


def calibrate_laet(S, eps, delta):
    """LAET-style: regress log X on the seven features and the flag at one checkpoint j0 (chosen on SEL), stop at a multiple
    of the prediction.

    Returns (rule record or None, predictions of all rows, fitted regressor).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    H, grid, FIT, SEL = S.H, S.grid, S.FIT, S.SEL
    Xc = np.log(np.minimum(S.X, H).astype(np.float64)); MULT = LAET_MULTIPLIERS
    best = None
    for g0 in (64, 128, 256, 512, 1024):
        j0 = int(np.searchsorted(grid, g0, side='left'))
        rg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=S.seeds['laet_regressor'])
        rg.fit(np.column_stack([S.features[FIT, j0], S.full[FIT, j0]]), Xc[FIT])
        pred = np.exp(rg.predict(np.column_stack([S.features[:, j0], S.full[:, j0]])))
        mk = lambda mult, idx, pred=pred, j0=j0: laet_F(S, pred, j0, mult, idx)
        m_s, _ = fixed_sequence(S, MULT, mk, eps, idx=SEL, U=S.u_sel, delta=LAET_PROXY_LEVEL)
        if m_s is None: continue
        cs, _ = evaluate_rule(S, mk(m_s, SEL), SEL)
        if best is None or cs.mean() < best[0]: best = (float(cs.mean()), g0, mk, j0, rg, pred)
    if best is None: return None, None, None
    _, g0, mk_laet, j0, reg, pred = best
    mult_p, _ = fixed_sequence(S, MULT, mk_laet, eps, delta=delta)
    return (None if mult_p is None else dict(type=TYPE_LAET, j0=int(j0), mult=float(mult_p), g0=int(g0))), pred, reg


def calibrate_fixed(S, eps, delta):
    """Fixed: one checkpoint for all queries, last checkpoint first."""
    J = S.J
    jb, _ = fixed_sequence(S, list(range(J - 1, -1, -1)), lambda j, idx: fixed_budget_F(S, j, idx), eps, delta=delta)
    return None if jb is None else dict(type=TYPE_FIXED, jfix=int(jb), budget=int(S.grid[jb]))


def calibrate_staircase(S, eps, delta, chain_grid, chain=None):
    """Our rule: the chain on SEL, terminal rule first, then decreasing multipliers (Theorem 3).

    Returns (rule record or None, chain elements {multiplier: beta}).
    """
    J = S.J
    CH = chain if chain is not None else {int(l): chain_element(S, S.SEL, int(l))[0] for l in chain_grid}
    mk_chain = lambda th, idx: terminal_F(S, idx) if th == 'terminal' else staircase_F(S, CH[th], idx)
    thc, _ = fixed_sequence(S, ['terminal'] + [int(l) for l in chain_grid[::-1]], mk_chain, eps, delta=delta)
    if thc is None: return None, CH
    beta = np.full((J - 1, 2), -1) if thc == 'terminal' else CH[thc]              # terminal: no bucket stops before the last checkpoint
    return dict(type=TYPE_STAIRCASE, theta=thc, beta=beta.astype(np.int32).ravel().tolist()), CH


def frontier_rule_record(S, r):
    """Rule record of a frontier rule of pipeline/frontier.py (its chain element is recomputed on SEL)."""
    th = r['theta']; th = th if th == 'terminal' else int(th)
    beta = np.full((S.J - 1, 2), -1) if th == 'terminal' else chain_element(S, S.SEL, th)[0]
    return dict(type=TYPE_STAIRCASE, theta=th, beta=beta.astype(np.int32).ravel().tolist(), frontier_mean_visits=r['mean_visits'])


def calibrate_common_target(S, eps, delta, chain_grid, frontier_rules=None, log=print):
    """Calibrate Level+cap, Level, LAET-style, Fixed and the staircase chain on CAL at (eps, delta).

    chain_grid       multipliers of the chain (those of the frontier run of the setting);
    frontier_rules   optional {level: record of pipeline/frontier.py}: the frontier rules are added under 'frontier_<level>'.
    A rule without a calibrated parameter is left out of RULE. Returns the rule records RULE with what rule_F needs.
    """
    RULE = {}
    for name, rec in (('level_cap', calibrate_level_cap(S, eps, delta)), ('level', calibrate_level(S, eps, delta))):
        if rec is not None: RULE[name] = rec
    rec, pred, reg = calibrate_laet(S, eps, delta)
    if rec is not None: RULE['laet_style'] = rec
    rec = calibrate_fixed(S, eps, delta)
    if rec is not None: RULE['fixed'] = rec
    RULE['none'] = dict(type=TYPE_NONE)
    rec, CH = calibrate_staircase(S, eps, delta, chain_grid)
    if rec is not None: RULE['cast'] = rec
    for e, r in (frontier_rules or {}).items(): RULE[f'frontier_{e}'] = frontier_rule_record(S, r)
    log('calibrated', {k: {kk: vv for kk, vv in v.items() if kk != 'beta'} for k, v in RULE.items()})
    return dict(RULE=RULE, laet_pred=pred, laet_reg=reg, laet_forest=None if reg is None else export_forest(reg), chain=CH)


def rule_F(S, cal, name, idx):
    """Stopping matrix of a calibrated rule record on the queries idx."""
    R_ = cal['RULE'][name]; J = S.J
    if R_['type'] == TYPE_NONE: return terminal_F(S, idx)
    if R_['type'] == TYPE_STAIRCASE:
        return terminal_F(S, idx) if R_.get('theta') == 'terminal' else staircase_F(S, np.array(R_['beta']).reshape(J - 1, 2), idx)
    if R_['type'] == TYPE_LEVEL: return level_F(S, R_['lam'], R_['jcap'], idx)
    if R_['type'] == TYPE_FIXED: return fixed_budget_F(S, R_['jfix'], idx)
    return laet_F(S, cal['laet_pred'], R_['j0'], R_['mult'], idx)

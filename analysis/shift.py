"""Replay of the frontier rules on queries logged after a change of the queries or of the index, without and with
recalibration.

Paper: Section VI-B, paragraph "Query shift and index updates" and Fig. 5. The setting before the change supplies the
score model, the bucket edges and the chain on SEL. The CAL and EVAL queries logged after the change are scored by the
same model and bucketed by the same edges. Two readings per risk level:
    before recalibration   the frontier rule calibrated before the change, on the EVAL queries after the change;
    recalibrated           the same chain, calibrated again on the CAL queries after the change by the same
                           fixed-sequence test at level delta / (2 M) with the same stopping draws (Theorem 3).
Each reading carries a one-sided exact binomial test of the risk target at level alpha / M.
"""
import numpy as np

from replay import append_rows, check_frontier_reproduced, frontier_candidates, reading, sel_chain
from cast.calibration import fixed_sequence
from cast.rules import evaluate_rule


def load_groups(red, ds, graph, groups, target):
    """Reduced arrays of several groups of queries, concatenated."""
    Z = [np.load(f'{red}/{ds}_{graph}_{p}.npz') for p in groups]
    return dict(c=np.concatenate([z['c'] for z in Z]), features=np.concatenate([z['features'] for z in Z]),
                full=np.concatenate([z['full'] for z in Z]), X=np.concatenate([z[target] for z in Z]), grid=Z[0]['grid'])


def take(D, rows):
    return {k: (v if k == 'grid' else v[rows]) for k, v in D.items()}


def prepare(S, FR, multipliers, after_cal, after_eval, log=print):
    """Rebuild the chain and the frontier rules, then append the CAL and EVAL queries logged after the change.

    Returns (candidate -> stopping matrix, calibrated candidate per level, check of the frontier output, rows of the
    appended CAL queries, rows of the appended EVAL queries).
    """
    assert np.array_equal(after_cal['grid'], S.grid) and np.array_equal(after_eval['grid'], S.grid)
    CH, mk = sel_chain(S, multipliers); TH = frontier_candidates(FR)
    chk = check_frontier_reproduced(S, FR, mk, TH); log('frontier rules reproduced', chk)
    n_cal = len(after_cal['c'])
    rows = append_rows(S, *(np.concatenate([after_cal[k], after_eval[k]]) for k in ('c', 'features', 'full', 'X')))
    assert n_cal == len(S.u_cal)                    # the stopping draws of CAL are reused query by query
    return mk, TH, chk, rows[:n_cal], rows[n_cal:]


def shift_readout(S, FR, multipliers, after_cal, after_eval, ref_rows, alpha=0.05, log=print):
    """Readings before recalibration and after it, at every risk level of the frontier run.

    ref_rows: EVAL queries before the change on which the frontier rule is read for reference.
    """
    cfg = FR['config']; E = [float(e) for e in cfg['levels']]; M = len(E); calib_level = cfg['delta'] / (2 * M); test_level = alpha / M
    mk, TH, chk, CAL2, EVAL2 = prepare(S, FR, multipliers, after_cal, after_eval, log)
    OUT = dict(levels=E, test_level=test_level, calibration_level=calib_level, n_cal=len(CAL2), n_eval=len(EVAL2), frontier_check=chk,
               share_incomplete_after=float(np.mean(S.c[EVAL2, -1] < S.X[EVAL2])), share_incomplete_reference=float(np.mean(S.c[ref_rows, -1] < S.X[ref_rows])),
               reference={}, before_recalibration={}, recalibrated={})
    for e in E:
        ke = str(e)
        if TH[e] is None: OUT['before_recalibration'][ke] = dict(calibrated=False); continue
        ci, ri = evaluate_rule(S, mk(TH[e], ref_rows), ref_rows); OUT['reference'][ke] = reading(ci, ri.round(), e, test_level)
        cs, rs = evaluate_rule(S, mk(TH[e], EVAL2), EVAL2)
        OUT['before_recalibration'][ke] = dict(calibrated=True, theta=TH[e], **reading(cs, rs.round(), e, test_level), cost_ratio_vs_reference=float(cs.mean() / ci.mean()))
        th2, _ = fixed_sequence(S, ['terminal'] + [int(l_) for l_ in multipliers[::-1]], mk, e, idx=CAL2, U=S.u_cal, delta=calib_level)
        if th2 is None: OUT['recalibrated'][ke] = dict(calibrated=False)
        else:
            c2, r2 = evaluate_rule(S, mk(th2, EVAL2), EVAL2)
            OUT['recalibrated'][ke] = dict(calibrated=True, theta=th2, **reading(c2, r2.round(), e, test_level),
                                           cost_ratio_vs_before=float(c2.mean() / cs.mean()), cost_ratio_vs_reference=float(c2.mean() / ci.mean()))
        log('eps', e, 'before', OUT['before_recalibration'][ke]['fail_rate'], OUT['before_recalibration'][ke]['reject'],
            'recalibrated', OUT['recalibrated'][ke].get('fail_rate'))
    lev = [str(e) for e in E if TH[e] is not None]; rec = [k for k in lev if OUT['recalibrated'][k]['calibrated']]
    OUT['summary'] = dict(levels=len(lev), rejections_before=sum(OUT['before_recalibration'][k]['reject'] for k in lev),
                          levels_recalibrated=len(rec), rejections_recalibrated=sum(OUT['recalibrated'][k]['reject'] for k in rec))
    log('summary', OUT['summary'])
    return OUT

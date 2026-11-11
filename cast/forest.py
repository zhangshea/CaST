"""Flattened gradient-boosted forests, the form in which the search engine evaluates the score model online.

Paper: implementation details of Section VI-A (the engine evaluates the fitted forests as scikit-learn does).
"""
import numpy as np


def export_forest(model):
    """Flatten a fitted HistGradientBoosting classifier or regressor into node arrays.

    raw(x) = base + sum over trees of the leaf value; a non-leaf node sends x to the left child iff
    x[feat] <= thr, and a missing value goes left iff miss_left.
    """
    roots, cols = [], {k: [] for k in ('feat', 'thr', 'left', 'right', 'leaf', 'miss_left', 'val')}; o = 0
    for it in model._predictors:
        assert len(it) == 1; nd = it[0].nodes; assert not nd['is_categorical'].any()
        roots.append(o); cols['feat'].append(nd['feature_idx'].astype(np.int16)); cols['thr'].append(nd['num_threshold'].astype(np.float64))
        cols['left'].append(nd['left'].astype(np.int32) + o); cols['right'].append(nd['right'].astype(np.int32) + o)
        cols['leaf'].append(nd['is_leaf'].astype(np.uint8)); cols['miss_left'].append(nd['missing_go_to_left'].astype(np.uint8))
        cols['val'].append(nd['value'].astype(np.float64))
        o += len(nd)
    F = {k: np.concatenate(v) for k, v in cols.items()}; F['root'] = np.array(roots, np.int32)
    F['base'] = float(np.ravel(model._baseline_prediction)[0])
    return F


def forest_raw(F, Xr):
    """Raw output of a flattened forest on the rows Xr (reference implementation of the engine's evaluation)."""
    out = np.full(len(Xr), F['base']); ar = np.arange(len(Xr))
    for r in F['root']:
        n = np.full(len(Xr), r)
        while True:
            lf = F['leaf'][n].astype(bool)
            if lf.all(): break
            v = Xr[ar, F['feat'][n]]; gl = np.where(np.isnan(v), F['miss_left'][n] == 1, v <= F['thr'][n])
            n = np.where(lf, n, np.where(gl, F['left'][n], F['right'][n]))
        out += F['val'][n]
    return out

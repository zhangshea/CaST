"""Reduce raw trajectory logs to the checkpoint arrays of the offline pipeline, and to the per-query stop times of the
list-size and distance-adaptive rules.

Paper: Section III-B (checkpoints c_j, completion time X), Section III-C and the implementation details of Section VI-A
(the seven features, the flag a_j), Section VI-A Metrics (recall target 0.9), compared methods "List size" and
"Dist.-adaptive" of Table II.

Input   WORK/logs/<workload>/<ds>_<graph>_<group>.npz, or its pieces <group>.part00.npz, <group>.part01.npz, ...
        (pipeline/log_trajectories.py). A group logged in pieces is reduced piece by piece and joined in piece order.
Output  WORK/reduced/<name>/<ds>_<graph>_<group>.npz
            c (n, J) int64, features (n, J, 7) float64, full (n, J) bool, X_exact (n,) int64, X_recall09 (n,) int64,
            final_visits (n,), grid (J,)
        WORK/reduced/<name>/<ds>_<graph>_<group>_stops.npz
            list_stop (n, G) int32      visits at which a search with a list of each logged size ends
            list_sizes (G,) int32
            adaptive_stop (n, n_gamma) int64   visits at which the distance-adaptive rule stops: the first expansion at which
                                        the answer holds k items and the expanded candidate is farther than (1 + gamma) times
                                        the k-th answer distance (float64 on the logged float32 values); final_visits otherwise
            gammas, X_exact, final_visits (int64)
<name> is the workload unless --out-name is given; --strip-prefix removes a prefix of the group names in the output file
names (for example title_fit -> fit), which gives the 'parts' layout of cast.setting for a named query set.

Requirement: numpy 1.26.x. The float64 logarithm of numpy 2.x differs from that of numpy 1.26 by one unit in the last place
on some arguments, which changes the features log(visits) and log1p(visits since the answer changed) in a small share of
cells. The reported arrays were reduced under numpy 1.26.4.

Usage:
    python pipeline/reduce_logs.py --work WORK --workload k10_kappa1 --ds amazon --graph A --groups fitpool chunk0 chunk1
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.grid import GRID, J, NEVER

# Thresholds gamma of the distance-adaptive rule.
GAMMAS = np.array([2, 1.5, 1.2, 1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.0])
GAMMA_STEP = 0.0025

# Fields of a raw log (engine log_search output plus S and list_sizes): per expansion `visits`, `dx`, `dbk`, `dpk`,
# `answer_size`, `chg`; per query `ptr`, `visit_of_member`, `list_stop`, `final_visits`.
RED_KEYS = ('c', 'features', 'full', 'X_exact', 'X_recall09', 'final_visits')
STOP_KEYS = ('list_stop', 'X_exact', 'adaptive_stop', 'final_visits')


def fine_gammas(gamma_max):
    """The thresholds of GAMMAS together with 0, 0.0025, ..., gamma_max, in decreasing order."""
    return np.array(sorted(set(GAMMAS.tolist()) | set(np.round(np.arange(0, gamma_max + 1e-5, GAMMA_STEP), 4).tolist()), reverse=True))


def checkpoint_arrays(A, k):
    """Checkpoint arrays of one raw log A for the answer size k.

    At checkpoint j the search has made c_j visits: the visit count after the first expansion that reaches GRID[j] visits,
    or the final visit count when the search ends earlier. The features are read at that expansion:
    log visits, log of the distance dx of the expanded candidate, log(dbk / dx) and log(dpk / dx) for the k-th distance of
    the answer and of the unconstrained nearest items, the answer size, log1p of the visits since the answer last changed,
    and those visits as a fraction of all visits.
    """
    n = len(A['final_visits'])
    ptr, visits, dx, dbk, dpk, answer_size, chg = (A[f] for f in ('ptr', 'visits', 'dx', 'dbk', 'dpk', 'answer_size', 'chg'))
    fin = A['final_visits'].astype(np.int64)
    cp = np.zeros((n, J), np.int64); c = np.zeros((n, J), np.int64)
    for i in range(n):
        seg = visits[ptr[i]:ptr[i + 1]]; idx = np.searchsorted(seg, GRID, side='left'); beyond = idx >= len(seg)
        idx = np.minimum(idx, len(seg) - 1); cp[i] = ptr[i] + idx; c[i] = np.where(beyond, fin[i], seg[idx])
    c = np.maximum.accumulate(c, axis=1); cnv = visits[cp].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        features = np.stack([np.log(cnv), np.log(np.maximum(dx[cp], 1e-9)), np.nan_to_num(np.log(dbk[cp] / dx[cp]), posinf=np.nan),
                             np.nan_to_num(np.log(dpk[cp] / dx[cp]), posinf=np.nan), answer_size[cp].astype(np.float64),
                             np.log1p(cnv - chg[cp]), (cnv - chg[cp]) / cnv], axis=2)
    return c, features, answer_size[cp] >= k


def completion_time(A, tau):
    """Completion time X of the recall target tau: the visit count at which ceil(tau |S|) members of S have been visited
    (NEVER when the search ends before that). tau = 1 is the exact answer."""
    S, rs = A['S'], A['visit_of_member']; valid = S >= 0
    need = np.ceil(tau * valid.sum(1) - 1e-9).astype(int); r = np.sort(np.where(valid & (rs > 0), rs, NEVER), axis=1)
    return r[np.arange(len(r)), need - 1].astype(np.int64)


def adaptive_stops(A, k, gammas=GAMMAS):
    """Visits at which the distance-adaptive rule stops, per query and threshold gamma."""
    ptr, visits, dx, dbk, answer_size, fin = (A[f] for f in ('ptr', 'visits', 'dx', 'dbk', 'answer_size', 'final_visits'))
    n = len(fin)
    dx64, dbk64, full = dx.astype(np.float64), dbk.astype(np.float64), answer_size >= k
    out = np.empty((n, len(gammas)), np.int64); st, en = ptr[:-1], ptr[1:]
    for t, gm in enumerate(gammas):
        hit = np.flatnonzero(full & (dx64 > (1.0 + gm) * dbk64))     # expansions at which the rule would stop
        pos = np.searchsorted(hit, st); ok = pos < len(hit)
        first = np.where(ok, hit[np.minimum(pos, len(hit) - 1)] if len(hit) else 0, -1); ok &= first < en
        out[:, t] = np.where(ok, visits[np.where(ok, first, 0)], fin)
    return out


def reduce_log(A, k, gammas=GAMMAS):
    """Reduced arrays and stop times of one raw log (a dict of arrays)."""
    if A['S'].shape[1] != k:
        raise ValueError(f"the log holds answers of size {A['S'].shape[1]}, not k = {k}")
    c, features, full = checkpoint_arrays(A, k)
    red = dict(c=c, features=features, full=full, X_exact=completion_time(A, 1.0), X_recall09=completion_time(A, 0.9),
               final_visits=A['final_visits'], grid=GRID)
    stops = dict(list_stop=A['list_stop'].astype(np.int32), list_sizes=A['list_sizes'].astype(np.int32), X_exact=red['X_exact'],
                 adaptive_stop=adaptive_stops(A, k, gammas), gammas=gammas, final_visits=A['final_visits'].astype(np.int64))
    # Self-check: X_exact is the largest visit count over the members of S (NEVER if one of them was not visited).
    S, rs = A['S'], A['visit_of_member']; valid = S >= 0
    direct = np.where(valid & (rs == 0), NEVER, np.where(valid, rs, 0)).max(1)
    some = valid.any(1)
    if not np.array_equal(direct[some], red['X_exact'][some]):
        raise RuntimeError('completion time differs from its direct recomputation')
    return red, stops


def log_files(log_dir, ds, graph, group):
    """The file of a group, or its pieces in order."""
    one = os.path.join(log_dir, f'{ds}_{graph}_{group}.npz')
    if os.path.exists(one):
        return [one]
    return sorted(glob.glob(os.path.join(log_dir, f'{ds}_{graph}_{group}.part[0-9][0-9].npz')))


def join(parts, keys, fixed):
    out = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    for key in fixed:
        out[key] = parts[0][key]
    return out


def completion_summary(red):
    """Shares of queries that do not complete, and quantiles of the completion time X (never completed: infinite)."""
    X = red['X_exact']; cJ = red['c'][:, -1]
    Xf = np.where(X >= NEVER, np.inf, X).astype(np.float64)
    q = {str(p): (lambda v: 'inf' if np.isinf(v) else float(v))(np.percentile(Xf, p, method='higher')) for p in (50, 75, 90, 95, 99)}
    return dict(n=int(len(X)), share_incomplete_at_last_checkpoint=float(np.mean(cJ < X)), share_X_above_32768=float(np.mean(X > 32768)),
                share_never_complete=float(np.mean(X >= NEVER)), mean_last_checkpoint_visits=float(cJ.mean()),
                mean_final_visits=float(red['final_visits'].mean()), X_quantiles=q)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True, help='work directory (logs/ and reduced/)')
    ap.add_argument('--workload', required=True, help='workload of the logs, for example k10_kappa1')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--groups', nargs='+', required=True, help='query groups to reduce')
    ap.add_argument('--k', type=int, default=10, help='answer size of the workload')
    ap.add_argument('--out-name', default=None, help='subdirectory of WORK/reduced (default: the workload)')
    ap.add_argument('--strip-prefix', default='', help='prefix removed from the group names in the output file names')
    ap.add_argument('--gamma-max', type=float, default=None,
                    help='add the thresholds 0, 0.0025, ..., GAMMA_MAX of the distance-adaptive rule to the 22 default ones')
    ap.add_argument('--summary', default=None, help='JSON file for the shares of queries that do not complete, per group and overall')
    return ap.parse_args()


def main():
    a = parse_args()
    if not np.__version__.startswith('1.26.'):
        print(f'warning: numpy {np.__version__}; the reported features are reproduced bit for bit under numpy 1.26.x only', file=sys.stderr)
    log_dir = os.path.join(a.work, 'logs', a.workload)
    out_dir = os.path.join(a.work, 'reduced', a.out_name or a.workload); os.makedirs(out_dir, exist_ok=True)
    gammas = GAMMAS if a.gamma_max is None else fine_gammas(a.gamma_max)
    summary = dict(ds=a.ds, graph=a.graph, workload=a.workload, k=a.k, groups={}); allX, allc = [], []
    for group in a.groups:
        files = log_files(log_dir, a.ds, a.graph, group)
        if not files:
            raise SystemExit(f'no log of group {group} in {log_dir}')
        reds, stops = [], []
        for f in files:
            r, s = reduce_log(dict(np.load(f)), a.k, gammas); reds.append(r); stops.append(s)
        red = join(reds, RED_KEYS, ('grid',)); stop = join(stops, STOP_KEYS, ('list_sizes', 'gammas'))
        name = group[len(a.strip_prefix):] if a.strip_prefix and group.startswith(a.strip_prefix) else group
        np.savez(os.path.join(out_dir, f'{a.ds}_{a.graph}_{name}.npz'), **red)
        np.savez(os.path.join(out_dir, f'{a.ds}_{a.graph}_{name}_stops.npz'), **stop)
        summary['groups'][name] = dict(pieces=len(files), **completion_summary(red))
        allX.append(red['X_exact']); allc.append(red['c'][:, -1])
        print('reduced', a.ds, a.graph, group, len(red['c']), 'queries; k', a.k, '; mean c_J', float(red['c'][:, -1].mean()),
              '; incomplete at the last checkpoint', summary['groups'][name]['share_incomplete_at_last_checkpoint'], flush=True)
    X = np.concatenate(allX); cJ = np.concatenate(allc)
    summary['all'] = dict(n=int(len(X)), share_incomplete_at_last_checkpoint=float(np.mean(cJ < X)),
                          share_X_above_32768=float(np.mean(X > 32768)), share_never_complete=float(np.mean(X >= NEVER)))
    if a.summary:
        json.dump(summary, open(a.summary, 'w'), indent=1)
    print(json.dumps(summary['all']), flush=True)


if __name__ == '__main__':
    main()

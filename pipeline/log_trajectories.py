"""Log the search trajectory of every labeled query of a setting (never timed).

Paper: Section III-A (graph search), Section III-B (checkpoints) and the implementation details of Section VI-A: a
best-first traversal with a list of 8,192 entries that maintains the greedy constrained answer of the visited items and
ends when the closest unexpanded candidate is farther than all kept entries.

The engine's log_search returns, per expansion, the visit count, the distance of the expanded candidate, the k-th distances
of the answer and of the unconstrained nearest visited items, the answer size and the visit count at the last change of
the answer; per query, the visit count at which each member of the exact answer S was visited and the visit count at which
a search with a list of each size of LIST_SIZES would end. pipeline/reduce_logs.py turns a log into checkpoint arrays.

LIST_SIZES is {round(10 * 1.2^j), j < 38} together with the powers of two from 16 to 8,192: 46 sizes, adjacent sizes at
most a factor 1.24 apart (the candidates of the list-size rule of Table II).

Output  WORK/logs/<workload>/<ds>_<graph>_<group>.npz (engine fields, S, list_sizes); a group with more than --piece
        queries is written in pieces <group>.part00.npz, <group>.part01.npz, ... Existing files are kept.
        WORK/logs/<workload>/<ds>_<graph>_log.json: per file, the number of expansions, mean visits and the share of
        queries whose exact answer was visited in full.

--check-list L1 L2 ... compares, for each of these list sizes, the share of queries whose logged trajectory has visited
all of S when a list of that size would end with the share of exact answers of a real search with that list size.

Usage:
    python pipeline/log_trajectories.py --data DATA --work WORK --ds amazon --graph A --groups fitpool chunk0 --threads 24
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from cast import datasets as D
from cast.grid import LIST_SIZE, NEVER

LIST_SIZES = sorted({int(round(10 * 1.2 ** j)) for j in range(38)} | {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192})
LIST_SIZES = np.array([g for g in LIST_SIZES if g <= LIST_SIZE], np.int32)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='prepared data directory')
    ap.add_argument('--work', required=True, help='work directory (graphs/ and logs/)')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--workload', default='k10_kappa1')
    ap.add_argument('--groups', nargs='+', required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', '1')),
                    help='search threads (the logged values do not depend on it)')
    ap.add_argument('--piece', type=int, default=10000, help='queries per log file')
    ap.add_argument('--check-list', type=int, nargs='*', default=[], help='list sizes of the comparison with real searches')
    ap.add_argument('--engine', default=os.path.join(HERE, '..', 'engine', 'build'), help='directory of the built cast_engine module')
    return ap.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, os.path.abspath(a.engine))
    import cast_engine as engine
    spec = D.workload_spec(a.ds, a.workload); K = spec['k']
    out = os.path.join(a.work, 'logs', a.workload); os.makedirs(out, exist_ok=True)
    print('list sizes', len(LIST_SIZES), LIST_SIZES[:6], LIST_SIZES[-3:], 'k', K, 'groups', a.groups, flush=True)
    ctx = D.Context(a.data, a.work, a.ds, a.graph, K, variant=spec['variant'], kappa=spec['kappa'])
    MJ = os.path.join(out, f'{a.ds}_{a.graph}_log.json'); meta = json.load(open(MJ)) if os.path.exists(MJ) else {}
    for group in a.groups:
        Qa = D.queries(a.data, a.ds, group); Sa = D.exact_answers(a.data, a.ds, group, a.workload)
        assert Sa.shape == (len(Qa), K), (group, Sa.shape)
        pieces = list(range(0, len(Qa), a.piece))
        for c, s in enumerate(pieces):
            tag = group if len(pieces) == 1 else f'{group}.part{c:02d}'; fn = os.path.join(out, f'{a.ds}_{a.graph}_{tag}.npz')
            if os.path.exists(fn):
                continue
            Q = np.ascontiguousarray(Qa[s:s + a.piece]); S = np.ascontiguousarray(Sa[s:s + a.piece]); t = time.time()
            r = engine.log_search(ctx.G, ctx.C, Q, S, dict(method='answer', k=K, L=LIST_SIZE), LIST_SIZE, LIST_SIZES, threads=a.threads)
            np.savez(fn + '.tmp.npz', list_sizes=LIST_SIZES, S=S, **r); os.replace(fn + '.tmp.npz', fn)
            rs = r['visit_of_member']; Ts = np.where((S >= 0) & (rs == 0), NEVER, rs).max(1)
            meta[tag] = dict(queries=len(Q), first_row=s, expansions=int(len(r['visits'])), mean_visits=float(r['final_visits'].mean()),
                             mean_descent_distances=float(r['ndesc'].mean()),
                             share_covered=float(np.mean(Ts < NEVER)), secs=time.time() - t)
            if a.check_list:
                chk = {}
                for L in a.check_list:
                    j = int(np.searchsorted(LIST_SIZES, L)); assert LIST_SIZES[j] == L, f'{L} is not a logged list size'
                    logged = float(np.mean(r['list_stop'][:, j] >= Ts))
                    q = ctx.run(dict(method='list', k=K), Q, S=S, L=L, threads=a.threads); real = float(np.mean(q['exact']))
                    chk[str(L)] = dict(logged=logged, real=real, mean_list_stop=float(np.mean(r['list_stop'][:, j])))
                meta[tag]['check_list'] = chk
            print(tag, json.dumps(meta[tag]), flush=True)
            json.dump(meta, open(MJ, 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()

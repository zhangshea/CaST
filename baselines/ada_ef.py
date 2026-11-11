"""Ada-ef under the common calibration: query statistics from a Gaussian model of the query's distances, a table of
list sizes per score group, calibration of its target on CAL, and the visit ratio to our rule.

Method: C. Zhang and R. J. Miller, "Distribution-Aware Exploration for Adaptive HNSW Search", Proceedings of the ACM on
Management of Data 4(1). As described there (cosine distance): with the mean and covariance of the unit base
vectors, the distances of query q to the data are modelled as Gaussian with mu = 1 - q . E[x] and sigma^2 = q Sigma q^T;
the distances D of the first l visited nodes are counted in the bins with edges theta_i = mu + sigma Phi^{-1}(delta i),
i = 1..m (delta = 1e-3, m = 5); the score is s = sum_i w_i c_i / |D| with w_i = 100 exp(-(i - 1)); queries are grouped
by round(s), and a table built on sample queries gives each group the smallest list size ef whose group reaches the
target r, with the weighted average WAE(r) as floor: ef(q, r) = max(table(group(q), r), WAE(r)).
Here success means the exact answer: a query fails when the returned answer differs from S. The rule has already made
l visits with an unbounded list when it chooses ef, so its cost is max(T_ef, l) and, when the list-ef search would end
before l visits, its answer is the greedy answer of the first l visits (exact iff X <= l).

Paper: Section VI-A, "Compared methods" (third group) and Section VI-B (Ada-ef passes the test only with its table
built on all of SEL, at 1.49 to 1.81 times our visits). --build-queries 200 gives the table size of the method itself.

Steps:
  stats     per query, the score for l in (1,025, 4,097), from a traversal of the graph file that makes the same first
            visits as the engine (descent of the upper layers, then best-first search of the bottom layer with an unbounded list)
  evaluate  table on the build queries (the first --build-queries SEL queries) with native searches at every candidate
            list size; target r = 1.000, 0.999, ..., 0.800 calibrated on CAL at eps = delta = 0.05; EVAL searches at ef(q, r)
  compare   EVAL visits, failures and the visit ratio to our rule of the common target (paired bootstrap)

Usage:
  python baselines/ada_ef.py stats --data DATA --work WORK --ds amazon --graph A --out STATS.npz --procs 16
  python baselines/ada_ef.py evaluate --data DATA --work WORK --red RED_DIR --ds amazon --graph A --stats STATS.npz --out ADAEF.npz
  python baselines/ada_ef.py compare --red RED_DIR --ds amazon --graph A --frontier FRONTIER.json --run ADAEF.npz --seed SEED --out OUT.json
"""
import argparse
import heapq
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from baselines import common as C

LS = (1025, 4097); BIN_DELTA, MBINS = 1e-3, 5; BIN_WEIGHTS = 100.0 * np.exp(-np.arange(MBINS))
N_POOL, N_CHUNKS = 200000, 20
# Candidate list sizes: 10, 15 and a geometric grid of ratio 1.02 up to 5,000.
EFC = sorted({10, 15} | {int(round(10 * 1.02 ** j)) for j in range(400) if 10 * 1.02 ** j <= 5000} | {5000})


# ------------------------------------------------------------------ stats: traversal of the HNSW graph file
def parse_graph(path, dim):
    """Bottom-layer lists, vectors and upper-layer lists of an HNSW index file (the layout the engine loads)."""
    f = open(path, 'rb'); h = f.read(96)
    (off0, maxel, n, spe, loff, offd), = [struct.unpack('<6Q', h[:48])]
    maxlevel, entry = struct.unpack('<iI', h[48:56]); maxM, maxM0, M = struct.unpack('<3Q', h[56:80])
    assert off0 == 0 and offd == 4 * (maxM0 + 1)
    blob = np.frombuffer(f.read(n * spe), np.uint8).reshape(n, spe)
    L0 = blob[:, :offd].copy().view(np.uint32); cnt0 = (L0[:, 0] & 0xFFFF).astype(np.int64); nb0 = L0[:, 1:].astype(np.int64)
    X = blob[:, offd:offd + 4 * dim].copy().view(np.float32)
    rest = f.read(); per = (maxM + 1) * 4; up = {}; p = 0
    for i in range(n):
        sz = struct.unpack_from('<I', rest, p)[0]; p += 4
        if sz:
            a = np.frombuffer(rest, np.uint32, sz // 4, p).reshape(sz // per, maxM + 1); up[i] = a; p += sz
    assert p == len(rest)
    return dict(n=n, X=X, cnt0=cnt0, nb0=nb0, up=up, entry=entry, maxlevel=maxlevel)


def dist(G, q, ids):
    """1 - <q, x> in float32 in the summation order of the 16-lane inner-product kernel of hnswlib, which the engine uses
    when built with 512-bit vector instructions: each lane accumulates its coordinates chunk by chunk with fused
    multiply-adds, then the lanes are reduced by halving (8 + 8, 4 + 4, 2 + 2, 1 + 1)."""
    X = G['X'][ids]; d = X.shape[1]
    Xr = X.reshape(len(ids), d // 16, 16).astype(np.float64); qr = q.reshape(d // 16, 16).astype(np.float64); acc = np.zeros((len(ids), 16), np.float32)
    for c in range(d // 16): acc = (acc.astype(np.float64) + Xr[:, c] * qr[c]).astype(np.float32)
    while acc.shape[1] > 1:
        h = acc.shape[1] // 2; acc = acc[:, :h] + acc[:, h:]
    return (np.float32(1.0) - acc[:, 0]).astype(np.float32)


def traverse(G, q, lmax):
    """Distances of the first lmax visited nodes: descent of the upper layers (first-improvement scan of the current node's
    list, repeated until no change), then the entry of the bottom layer is visit 1 and every pop of the closest
    unexpanded candidate (order (distance, id)) visits its unvisited neighbours in list order."""
    cur = G['entry']; dcur = float(dist(G, q, np.array([cur]))[0])
    for lvl in range(G['maxlevel'], 0, -1):
        changed = True
        while changed:
            changed = False; a = G['up'][cur][lvl - 1]; c = int(a[0] & 0xFFFF); ids = a[1:1 + c].astype(np.int64)
            if c == 0: continue
            ds = dist(G, q, ids)
            for j in range(c):                         # first-improvement scan of the list of the node current at loop start
                if ds[j] < dcur: dcur = float(ds[j]); cur = int(ids[j]); changed = True
    D = [np.float32(dcur)]; vis = {cur}; Cq = [(np.float32(dcur), cur)]
    while Cq and len(D) < lmax:
        d0, u = heapq.heappop(Cq)
        ids = G['nb0'][u, :G['cnt0'][u]]
        new = [int(x) for x in ids if int(x) not in vis]
        if not new: continue
        ds = dist(G, q, np.array(new, np.int64))
        for x, dx in zip(new, ds):
            vis.add(x); D.append(dx); heapq.heappush(Cq, (dx, x))
            if len(D) >= lmax: break
    return np.array(D[:lmax], np.float32)


_G = _Q = _TH = None
def _init(G, Q, TH):
    global _G, _Q, _TH
    _G, _Q, _TH = G, Q, TH


def _work(args):
    lo, hi = args; out = np.zeros((hi - lo, len(LS), MBINS), np.int32); nd = np.zeros((hi - lo, len(LS)), np.int32)
    for t, i in enumerate(range(lo, hi)):
        q = _Q[i]; Dq = traverse(_G, q, max(LS))
        for a, l in enumerate(LS):
            d = Dq[:l]; nd[t, a] = len(d)
            out[t, a] = np.histogram(d, bins=np.concatenate([[-np.inf], _TH[i]]))[0]
    return lo, out, nd


def stats(a):
    from multiprocessing import Pool
    from scipy.stats import norm
    from cast import datasets
    _, over = C.read_seeds(a.seeds); rows_of, _ = C.pool_rows(over); rows = np.concatenate([rows_of[s] for s in a.samples])
    G = parse_graph(datasets.graph_path(a.work, a.ds, a.graph), int(datasets.base(a.data, a.ds).shape[1]))   # the base vectors are read from the graph file
    Xb = G['X'].astype(np.float64); Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    mean = Xb.mean(0); Sig = np.cov(Xb, rowvar=False); del Xb
    Qall = np.concatenate([datasets.queries(a.data, a.ds, f'chunk{k}') for k in range(N_CHUNKS)]); Q = np.ascontiguousarray(Qall[rows], np.float32); del Qall
    mu = 1.0 - Q.astype(np.float64) @ mean; sd = np.sqrt(np.einsum('ij,jk,ik->i', Q.astype(np.float64), Sig, Q.astype(np.float64)))
    TH = mu[:, None] + sd[:, None] * norm.ppf(BIN_DELTA * np.arange(1, MBINS + 1))[None, :]
    step = 200; jobs = [(s, min(s + step, len(Q))) for s in range(0, len(Q), step)]
    cnt = np.zeros((len(Q), len(LS), MBINS), np.int32); nds = np.zeros((len(Q), len(LS)), np.int32)
    with Pool(a.procs, initializer=_init, initargs=(G, Q, TH)) as pool:
        for lo, o, nd in pool.imap_unordered(_work, jobs): cnt[lo:lo + len(o)] = o; nds[lo:lo + len(o)] = nd
    score = (cnt * BIN_WEIGHTS[None, None, :]).sum(2) / np.maximum(nds, 1)
    np.savez(a.out, rows=rows, LS=np.array(LS), counts=cnt, nD=nds, score=score, mu=mu, sigma=sd, theta=TH)
    print('stats done', len(Q), {l: (float(score[:, i].mean()), float((np.rint(score[:, i]) == 0).mean())) for i, l in enumerate(LS)}, flush=True)


# ------------------------------------------------------------------ evaluate: table, calibration, EVAL
def evaluate(a):
    from scipy.stats import binom
    from cast import datasets
    t0 = time.time()
    def log(*x): print(f'[{time.time() - t0:6.0f}s]', *x, flush=True)
    _, over = C.read_seeds(a.seeds); eng, ctx = C.open_context(a); K = ctx.k
    rows_of, _ = C.pool_rows(over); SEL, CAL, EVAL = rows_of['SEL'], rows_of['CAL'], rows_of['EVAL']
    BUILD = SEL[:a.build_queries]
    Qall = np.concatenate([datasets.queries(a.data, a.ds, f'chunk{k}') for k in range(N_CHUNKS)])
    Sall = np.concatenate([datasets.exact_answers(a.data, a.ds, f'chunk{k}', a.workload) for k in range(N_CHUNKS)]).astype(np.int64)
    Xall = np.concatenate([np.load(f'{a.red}/{a.ds}_{a.graph}_chunk{k}.npz')['X_exact'] for k in range(N_CHUNKS)]).astype(np.int64)
    _sts = [np.load(f) for f in a.stats]                                    # one or more stats files and their rows
    ST = dict(LS=_sts[0]['LS'], score=np.concatenate([z['score'] for z in _sts])); srow = np.concatenate([z['rows'] for z in _sts])
    assert len(np.unique(srow)) == len(srow) == len(ST['score'])
    pos = -np.ones(N_POOL, np.int64); pos[srow] = np.arange(len(srow)); LSv = [int(x) for x in ST['LS']]
    assert (pos[CAL] >= 0).all() and (pos[EVAL] >= 0).all() and (pos[BUILD] >= 0).all()
    cache = {}                                                              # (ef, row) -> (T_ef, returned answer equals S)
    def run(ef, rows):
        rows = np.asarray([r for r in rows if (ef, r) not in cache], np.int64)
        if len(rows) == 0: return
        res = eng.search(ctx.G, ctx.C, np.ascontiguousarray(Qall[rows]), dict(method='list', k=K, L=int(ef)), threads=a.threads)
        for i, r in enumerate(rows):
            ids = set(int(x) for x in res['ids'][i] if x >= 0); s = set(int(x) for x in Sall[r] if x >= 0)
            cache[(ef, int(r))] = (int(res['visits'][i]), ids == s)
    def outcome(ef, r, l):
        """(cost, failure) of the rule on query r with list size ef after its first l visits."""
        T, ok = cache[(ef, int(r))]
        if T >= l: return T, (not ok)
        return l, bool(Xall[r] > l)
    OUT = dict(ds=a.ds, graph=a.graph, build_queries=a.build_queries, EFC=EFC, LS=LSv, res={})
    for ef in EFC: run(ef, BUILD)
    log('build searches done', len(EFC))
    for i_l, l in enumerate(LSv):
        sc = ST['score'][:, i_l]; grp = np.rint(sc).astype(np.int64)
        bg = grp[pos[BUILD]]; groups = np.unique(bg)
        succ = {gg: np.array([[1.0 - outcome(ef, r, l)[1] for ef in EFC] for r in BUILD[bg == gg]]).mean(0) for gg in groups}
        nb = {gg: int((bg == gg).sum()) for gg in groups}
        def table(tab, rr):
            efg = {}
            for gg in groups:
                ok = np.flatnonzero(tab[gg] >= rr - 1e-12); efg[gg] = EFC[ok[0]] if len(ok) else EFC[-1]
            wae = sum(nb[gg] * efg[gg] for gg in groups) / sum(nb.values()); wae = next(x for x in EFC if x >= wae - 1e-9)
            return efg, wae
        def nearest(gq):                         # a query whose group is empty takes the nearest group (ties: the larger score)
            d = np.abs(groups - gq); m = d.min(); return int(groups[d == m].max())
        gmap = {int(gq): (int(gq) if gq in nb else nearest(gq)) for gq in np.unique(grp)}
        def efs(rows, tab, rr):
            efg, wae = table(tab, rr); return np.array([max(efg[gmap[int(grp[pos[r]])]], wae) for r in rows], np.int64), wae
        empty_share = float(np.mean([gmap[int(grp[pos[r]])] != int(grp[pos[r]]) for r in EVAL]))
        R = dict(l=l, n_groups=int(len(groups)), group_sizes={int(k_): v_ for k_, v_ in nb.items()}, empty_group_share_eval=empty_share)
        # calibration on CAL over the target r
        RG = [round(1.0 - 0.001 * i, 3) for i in range(201)]; need = {}
        E_cal = {}
        for rr in RG:
            e, _ = efs(CAL, succ, rr); E_cal[rr] = e
            for ef in np.unique(e): need.setdefault(int(ef), set()).update(CAL[e == ef].tolist())
        for ef, rs_ in sorted(need.items()): run(ef, sorted(rs_))
        log('CAL searches done', l, len(need))
        trace, sel = [], None
        for rr in RG:
            f = np.array([outcome(int(e), r, l)[1] for e, r in zip(E_cal[rr], CAL)]); kf = int(f.sum()); ok = bool(binom.cdf(kf, len(CAL), 0.05) <= 0.05)
            trace.append((rr, kf, ok))
            if not ok: break
            sel = rr
        R['cal_trace_tail'] = trace[-4:]; R['r_sel'] = sel
        if sel is not None:
            e_ev, wae = efs(EVAL, succ, sel)
            for ef in np.unique(e_ev): run(int(ef), EVAL[e_ev == ef])
            oc = [outcome(int(e), r, l) for e, r in zip(e_ev, EVAL)]
            cost = np.array([o[0] for o in oc], np.int64); fail = np.array([o[1] for o in oc], bool)
            R.update(wae=int(wae), mean_ef=float(e_ev.mean()), mean_visits=float(cost.mean()), fail_rate=float(fail.mean()), share_cost_is_l=float((cost == l).mean()))
            OUT[f'eval_cost_l{l}'] = cost; OUT[f'eval_fail_l{l}'] = fail
        OUT['res'][l] = R; log('l', l, {k_: v_ for k_, v_ in R.items() if k_ != 'group_sizes'})
    np.savez(a.out, eval_rows=EVAL, meta=json.dumps(OUT['res']), EFC=np.array(EFC), **{k_: v_ for k_, v_ in OUT.items() if k_.startswith('eval_')})
    print('evaluate done', json.dumps(OUT['res'])[:3000], flush=True)


def compare(a):
    _, over = C.read_seeds(a.seeds); FR = json.load(open(a.frontier)); assert FR['layout'] == 'pool'
    S = C.load_frontier_setting(a.red, a.ds, a.graph, FR, over); rec, c_ch, f_ch = C.common_target_staircase(S, FR, 0.05, 0.05)
    ev = C.sample_rows(S)['EVAL']; W = C.bootstrap_weights(len(ev), a.seed)
    OUT = dict(ds=a.ds, graph=a.graph, cast=dict(theta=rec['theta'], **C.summarize(c_ch, f_ch)), bootstrap_seed=a.seed)
    Z = np.load(a.run); assert np.array_equal(Z['eval_rows'], ev); OUT['ada_ef_meta'] = json.loads(str(Z['meta']))
    for key in Z.files:
        if key.startswith('eval_cost_l'):
            l = key[len('eval_cost_l'):]; c = Z[key]; f = Z[f'eval_fail_l{l}']
            OUT[f'ada_ef_l{l}'] = dict(**C.summarize(c, f), visit_ratio=C.visit_ratio(W, c, c_ch, with_se=True))
    json.dump(OUT, open(a.out, 'w'), indent=1, default=str); print(json.dumps(OUT, default=str)[:3000])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('stats'); C.add_setting_args(p, engine=False)
    p.add_argument('--samples', nargs='+', default=['SEL', 'CAL', 'EVAL'], choices=('SEL', 'CAL', 'EVAL'), help='samples of the pool whose queries get statistics')
    p.add_argument('--procs', type=int, default=16); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True, help='output npz')
    p = sub.add_parser('evaluate'); C.add_setting_args(p)
    p.add_argument('--red', required=True, help='directory of the reduced arrays of the workload (completion times X)')
    p.add_argument('--stats', nargs='+', required=True, help='one or more outputs of the stats step')
    p.add_argument('--build-queries', type=int, default=20000, help='SEL queries on which the table is built (20,000: all of SEL; 200: the size of the method)')
    p.add_argument('--seeds', default=None); p.add_argument('--out', required=True, help='output npz')
    p = sub.add_parser('compare')
    p.add_argument('--red', required=True); p.add_argument('--ds', required=True); p.add_argument('--graph', default='A')
    p.add_argument('--frontier', required=True, help='output of pipeline/frontier.py for the setting')
    p.add_argument('--run', required=True, help='output npz of the evaluate step')
    p.add_argument('--seed', type=int, required=True, help='seed of the paired bootstrap'); p.add_argument('--seeds', default=None); p.add_argument('--out', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(stats=stats, evaluate=evaluate, compare=compare)[a.step](a)

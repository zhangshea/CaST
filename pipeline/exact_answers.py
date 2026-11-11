"""Exact constrained answers S of the queries of a workload, by a full scan of the base.

Paper: Section III-A (Definition 1, feasible answer; the greedy that scans the items by increasing distance and keeps an
item when the kept set stays feasible) and Section VI-A ("Exact constrained answers are computed by a full scan").

For each query: inner products with every base item in float32 give the T nearest candidates (T = 2,048 at first); they are
rescored in float64 and ordered by (distance 1 - <q, x>, item id); the greedy runs over this order and stops at k items.
If it does not finish within the first T / 2 positions, T is multiplied by 4 (up to the whole base) and the query is redone,
so S does not depend on the candidate cut. Feasibility is the existence of an assignment of the kept items to distinct
category slots (capacity kappa per category), maintained incrementally by augmenting paths.

Checks: every S is verified to be feasible by an independent bipartite matching (unit capacities), and the answers of a
random subset of queries are recomputed by a naive greedy that tests each prefix with that matching routine.

Workloads (cast.datasets.workload_spec): k10_kappa1, k10_kappa2, k50_kappa1, k10_unconstrained. For a dataset whose
unconstrained workload uses one shared category of capacity k, S is the stored exact unconstrained top k of the workload
given by --plain-from (its answers must exist).

Output  DATA/<ds>/answers/<group>_<workload>.npz: S (n, k) int32 item ids (-1 padded), depth (position in the distance
        order of the last kept item), size, plain_ids and plain_d (the exact unconstrained nearest items and distances).

Defaults of --batch and --float64-base follow the reported runs: 256 queries per batch with a float64 copy of the base on
the device for the base of amazon and arxiv; 64 queries per batch with the float64 cast per batch otherwise.

Usage:
    python pipeline/exact_answers.py --data DATA --ds amazon --workload k10_kappa1 --groups fitpool chunk0 chunk1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast import datasets as D


def make_greedy():
    """Compile the greedy of Section III-A. State per query: load[m] items assigned to each category, head[m] / sel_next
    linked lists of the kept items assigned to a category, sel_item / sel_cat the kept items and their categories."""
    from numba import njit, prange

    @njit(inline='always')
    def _list_remove(head, sel_next, cat, idx):
        cur = head[cat]
        if cur == idx:
            head[cat] = sel_next[idx]
            return
        while cur != -1:
            nx = sel_next[cur]
            if nx == idx:
                sel_next[cur] = sel_next[idx]
                return
            cur = nx

    @njit(inline='always')
    def _list_add(head, sel_next, cat, idx):
        sel_next[idx] = head[cat]
        head[cat] = idx

    @njit
    def offer(p, nsel, ptr, idx, kappa, load, head, ctag, gen, par_cat, par_sel, bfsq, sel_item, sel_cat, sel_next):
        """Try to add item p to the kept set (size nsel). Returns 1 if the set stays feasible (state updated), else 0.
        gen is a mark of this call, larger than every previous one."""
        a = ptr[p]; b = ptr[p + 1]
        # a category of p with spare capacity
        for t in range(a, b):
            g = idx[t]
            if load[g] < kappa[g]:
                sel_item[nsel] = p; sel_cat[nsel] = g; _list_add(head, sel_next, g, nsel)
                load[g] += 1
                return 1
        # breadth-first search for an augmenting path over categories
        qh = 0; qt = 0
        for t in range(a, b):
            g = idx[t]
            if ctag[g] != gen:
                ctag[g] = gen; par_cat[g] = -1; par_sel[g] = -1
                bfsq[qt] = g; qt += 1
        end = -1
        while qh < qt and end < 0:
            g = bfsq[qh]; qh += 1
            y = head[g]
            while y != -1 and end < 0:
                it = sel_item[y]
                for t in range(ptr[it], ptr[it + 1]):
                    g2 = idx[t]
                    if ctag[g2] != gen:
                        ctag[g2] = gen; par_cat[g2] = g; par_sel[g2] = y
                        if load[g2] < kappa[g2]:
                            end = g2
                            break
                        bfsq[qt] = g2; qt += 1
                y = sel_next[y]
        if end < 0:
            return 0
        # augment: walk back from the category with spare capacity
        cur = end
        load[end] += 1
        while par_sel[cur] != -1:
            y = par_sel[cur]; frm = par_cat[cur]
            _list_remove(head, sel_next, frm, y)
            sel_cat[y] = cur
            _list_add(head, sel_next, cur, y)
            cur = frm
        # cur is a category of p whose slot was freed by the first move
        sel_item[nsel] = p; sel_cat[nsel] = cur; _list_add(head, sel_next, cur, nsel)
        return 1

    @njit
    def reset_state(nsel, ptr, idx, sel_item, load, head):
        for i in range(nsel):
            it = sel_item[i]
            for t in range(ptr[it], ptr[it + 1]):
                g = idx[t]
                load[g] = 0; head[g] = -1

    @njit
    def greedy_order(order, cnt, k, ptr, idx, kappa, load, head, ctag, gen0, par_cat, par_sel, bfsq, sel_item, sel_cat, sel_next, out):
        """Greedy over order[0:cnt] (sorted by distance), stopped at k items. Writes the kept item ids to out[0:nsel];
        returns (nsel, 1-based position of the last kept item, gen)."""
        nsel = 0; last = 0; gen = gen0
        for j in range(cnt):
            gen += 1
            if offer(order[j], nsel, ptr, idx, kappa, load, head, ctag, gen, par_cat, par_sel, bfsq, sel_item, sel_cat, sel_next) == 1:
                nsel += 1; last = j + 1
                if nsel == k:
                    break
        for i in range(nsel):
            out[i] = sel_item[i]
        reset_state(nsel, ptr, idx, sel_item, load, head)
        return nsel, last, gen

    @njit(parallel=True)
    def greedy_batch(orders, cnts, k, ptr, idx, kappa, m, out_S, out_L, out_nsel):
        nq = orders.shape[0]
        for qi in prange(nq):
            load = np.zeros(m, np.int32); head = -np.ones(m, np.int32); ctag = np.zeros(m, np.int64)
            par_cat = np.zeros(m, np.int32); par_sel = np.zeros(m, np.int32); bfsq = np.zeros(m, np.int32)
            sel_item = np.zeros(k, np.int32); sel_cat = np.zeros(k, np.int32); sel_next = np.zeros(k, np.int32)
            out = np.empty(k, np.int32)
            nsel, last, g = greedy_order(orders[qi], cnts[qi], k, ptr, idx, kappa, load, head, ctag,
                                         np.int64(0), par_cat, par_sel, bfsq, sel_item, sel_cat, sel_next, out)
            for j in range(k):
                out_S[qi, j] = out[j] if j < nsel else -1
            out_L[qi] = last; out_nsel[qi] = nsel

    return greedy_batch


def make_exact_order(Xg, float64_base):
    """Return f(Qg, T) -> (ids int32 [b, T], d float64 [b, T]): the T nearest base items of each query, ordered by
    (1 - <q, x> in float64, item id)."""
    import torch
    n = Xg.shape[0]
    Xg64 = Xg.double() if float64_base else None

    def exact_order(Qg, T):
        S = Qg @ Xg.T
        T = min(T, n)
        _, idx = torch.topk(S, T, dim=1, largest=True, sorted=False)
        del S
        idx, _ = torch.sort(idx, dim=1)  # ascending ids, so the stable sort below breaks ties by id
        out_i = torch.empty_like(idx); out_d = torch.empty(idx.shape, dtype=torch.float64, device=idx.device)
        sb = max(1, int(2e8 // (T * Xg.shape[1] * 8)))
        for s in range(0, idx.shape[0], sb):
            ii = idx[s:s + sb]
            q64 = Qg[s:s + sb].double()
            sc = torch.einsum('btd,bd->bt', Xg64[ii] if float64_base else Xg[ii].double(), q64)
            dd = 1.0 - sc
            o = torch.sort(dd, dim=1, stable=True).indices
            out_i[s:s + sb] = torch.gather(ii, 1, o); out_d[s:s + sb] = torch.gather(dd, 1, o)
        return out_i.cpu().numpy().astype(np.int32), out_d.cpu().numpy()

    return exact_order


def _matching_size(items, ptr, idx, m):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    indptr = [0]; ind = []
    for it in items:
        ind.extend(idx[ptr[it]:ptr[it + 1]].tolist()); indptr.append(len(ind))
    A = csr_matrix((np.ones(len(ind)), ind, indptr), shape=(len(items), m))
    return int((maximum_bipartite_matching(A, perm_type='column') >= 0).sum())


def count_infeasible(S, ptr, idx, m):
    """Independent feasibility check (unit capacities): the items of S are matched to distinct categories."""
    bad = 0
    for row in S:
        row = row[row >= 0]
        bad += int(_matching_size(row, ptr, idx, m) != len(row))
    return bad


def naive_greedy(order, k, ptr, idx, m):
    """Slow independent greedy: keep p iff a maximum matching of the kept items and p matches all of them."""
    S = []; last = 0
    for j, p in enumerate(order):
        cand = S + [int(p)]
        if _matching_size(cand, ptr, idx, m) == len(cand):
            S = cand; last = j + 1
            if len(S) == k:
                break
    return S, last


def answers_of_group(Q, n, k, plain_k, exact_order, greedy_batch, to_device, cat, T0, batch, diff_ids, log=print, tag=''):
    """Exact answers of the queries Q. Returns S, depth, size, plain ids and distances, T used, and the candidate orders
    of the queries in diff_ids (for the naive-greedy check)."""
    ptr, idx, m, kappa = cat
    nq = Q.shape[0]
    plain = np.zeros((nq, plain_k), np.int32); plain_d = np.zeros((nq, plain_k), np.float64)
    S_ = np.full((nq, k), -1, np.int32); L_ = np.zeros(nq, np.int64); N_ = np.zeros(nq, np.int64)
    Tused = np.zeros(nq, np.int64)
    keep_orders = {}
    pending = np.arange(nq); T = T0
    t0 = time.time()
    while len(pending) > 0:
        nxt = []
        bs = batch if T <= 8192 else max(1, int(batch * 8192 // T))
        for s in range(0, len(pending), bs):
            qi = pending[s:s + bs]
            ids, dd = exact_order(to_device(Q[qi]), T)
            Tq = ids.shape[1]
            plain[qi] = ids[:, :plain_k]; plain_d[qi] = dd[:, :plain_k]
            cnts = np.full(len(qi), Tq, np.int64)
            Sb = np.empty((len(qi), k), np.int32); Lb = np.empty(len(qi), np.int64); Nb = np.empty(len(qi), np.int64)
            greedy_batch(ids, cnts, k, ptr, idx, kappa, m, Sb, Lb, Nb)
            ok = ((Nb == k) & (Lb <= Tq // 2)) | (Tq == n)
            S_[qi] = Sb; L_[qi] = Lb; N_[qi] = Nb
            Tused[qi] = Tq
            for j, q in enumerate(qi):
                if not ok[j]:
                    nxt.append(q)
                elif int(q) in diff_ids:
                    keep_orders[int(q)] = ids[j].copy()
        log(tag, f'T={T} done {len(pending) - len(nxt)}/{len(pending)} {time.time() - t0:.0f}s', flush=True)
        pending = np.array(nxt, np.int64); T *= 4
    return S_, L_, N_, plain, plain_d, Tused, keep_orders


def group_summary(S_, L_, N_, plain, k, cat, keep_orders):
    ptr, idx, m, kappa = cat
    nq = len(S_)
    mism = 0
    for q, order in keep_orders.items():
        Sn, Ln = naive_greedy(order, k, ptr, idx, m)
        mism += int(sorted(Sn) != sorted(x for x in S_[q].tolist() if x >= 0) or Ln != L_[q])
    r = L_ / k
    return dict(n=nq, infeasible=count_infeasible(S_, ptr, idx, m), incomplete=int((N_ < k).sum()),
                share_differs_from_unconstrained=float(np.mean([set(S_[i].tolist()) != set(plain[i, :k].tolist()) for i in range(nq)])),
                depth_over_k_mean=float(r.mean()), depth_over_k_p99=float(np.percentile(r, 99)), depth_over_k_max=float(r.max()),
                naive_checked=len(keep_orders), naive_mismatch=mism)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='prepared data directory')
    ap.add_argument('--ds', required=True)
    ap.add_argument('--workload', default='k10_kappa1')
    ap.add_argument('--groups', nargs='+', required=True, help='query groups (files DATA/<ds>/queries/<group>.npy)')
    ap.add_argument('--plain-k', type=int, default=50, help='number of exact unconstrained nearest items stored (at least k)')
    ap.add_argument('--plain-from', default='k10_kappa1',
                    help='workload whose stored unconstrained top k gives S when the category store is one shared category')
    ap.add_argument('--t0', type=int, default=2048, help='first number of candidates T')
    ap.add_argument('--batch', type=int, default=None, help='queries per batch (default: 256 for amazon and arxiv, 64 otherwise)')
    ap.add_argument('--float64-base', choices=('device', 'batch'), default=None,
                    help="keep a float64 copy of the base on the device, or cast the gathered candidates per batch "
                         "(default: 'device' for amazon and arxiv, 'batch' otherwise; the values are the same)")
    ap.add_argument('--n-check', type=int, default=200, help='queries per group recomputed by the naive greedy')
    ap.add_argument('--check-seed', type=int, default=1, help='seed of the draw of these queries')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4, help='threads of the greedy')
    ap.add_argument('--summary', default=None, help='JSON file for the per-group summary')
    return ap.parse_args()


def main():
    a = parse_args()
    spec = D.workload_spec(a.ds, a.workload); k = spec['k']
    out_dir = os.path.join(a.data, a.ds, 'answers'); os.makedirs(out_dir, exist_ok=True)
    summary = {}; failed = []
    if spec['variant'] == 'shared':
        # one category of capacity k shared by all items: the exact answer is the exact unconstrained top k
        for group in a.groups:
            z = np.load(D.answers_path(a.data, a.ds, group, a.plain_from))
            S = z['plain_ids'][:, :k]
            np.savez(D.answers_path(a.data, a.ds, group, a.workload), S=S, depth=np.full(len(S), k, np.int64),
                     size=np.full(len(S), k, np.int64), plain_ids=z['plain_ids'], plain_d=z['plain_d'])
            summary[group] = dict(n=int(len(S)), source=a.plain_from); print(group, json.dumps(summary[group]), flush=True)
    else:
        import numba
        import torch
        numba.set_num_threads(min(a.threads, numba.config.NUMBA_NUM_THREADS))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        text_base = a.ds in D.SLOT_STORE_DATASETS
        batch = a.batch if a.batch is not None else (256 if text_base else 64)
        f64 = (a.float64_base or ('device' if text_base else 'batch')) == 'device'
        plain_k = max(a.plain_k, k)
        X = np.load(os.path.join(a.data, a.ds, 'base.npy')); n = X.shape[0]
        Xg = torch.from_numpy(X).to(a.device); del X
        exact_order = make_exact_order(Xg, f64); greedy_batch = make_greedy()
        ptr, idx, m = D.category_csr(a.data, a.ds, spec['variant'])
        cat = (ptr, idx, m, np.full(m, spec['kappa'], np.int32))
        to_device = lambda q: torch.from_numpy(q).to(a.device)
        for group in a.groups:
            Q = D.queries(a.data, a.ds, group); nq = Q.shape[0]
            diff_ids = set(np.random.default_rng(a.check_seed).choice(nq, min(a.n_check, nq), replace=False).tolist())
            S_, L_, N_, plain, plain_d, Tused, keep = answers_of_group(Q, n, k, plain_k, exact_order, greedy_batch, to_device, cat,
                                                                      a.t0, batch, diff_ids, tag=group)
            np.savez(D.answers_path(a.data, a.ds, group, a.workload), S=S_, depth=L_, size=N_, plain_ids=plain, plain_d=plain_d, T_used=Tused)
            st = group_summary(S_, L_, N_, plain, k, cat, keep)
            if spec['variant'] == 'unconstrained':
                # every set of at most k items is feasible, so S is the exact unconstrained top k
                st['equals_unconstrained_top_k'] = bool(np.array_equal(S_, plain[:, :k]) and (L_ == k).all() and (N_ == k).all())
            summary[group] = st; print(group, json.dumps(st), flush=True)
            if st['infeasible'] or st['naive_mismatch'] or not st.get('equals_unconstrained_top_k', True):
                failed.append(group)
    if a.summary:
        json.dump(dict(ds=a.ds, workload=a.workload, k=k, groups=summary), open(a.summary, 'w'), indent=1)
    if failed:
        raise SystemExit(f'check failed for groups {failed}')
    print('done')


if __name__ == '__main__':
    main()

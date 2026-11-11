"""YFCC-10M (filtered-search data set of the NeurIPS'23 Big-ANN competition): base, categories, item queries and the
official queries.

Paper: Section VI-A, Datasets and Table I; Table II (rows YFCC-10M); Section VI-B, "Query shift and index updates"
(official queries).

Vectors: uint8 -> float32, scaled to unit L2 norm; distance 1 - <q, x>. Categories: the 1,000 most frequent tags (ids =
frequency ranks); items without such a tag are removed.

    split     RAW: base.10M.u8bin, base.metadata.10M.spmat. One random permutation of the kept items (seed in
              configs/seeds.json); the first 202,500 are held out as fit 2,500 | sel 20,000 | cal 40,000 | eval 120,000 |
              spare 20,000 (in this order), the rest is the base in source row order. Writes DATA/yfcc10m/queries/<group>.npy
              and, in PREP/yfcc10m, base_pre.npy, base_rows_pre.npy, categories_pre.npz, rows_<group>.npy, split.json.
    renumber  after the multi-threaded graph build on PREP/yfcc10m/base_pre.npy (pipeline/build_graphs.py hnsw --base ...
              --threads 32): renumber the base so that item id == node id; writes DATA/yfcc10m/base.npy and
              categories_main.npz.
    official  RAW: query.public.100K.u8bin, query.private.2727415019.100K.u8bin (vectors only; the tag predicates of the
              official queries are not used). Each set is permuted separately (two seeds in configs/seeds.json) and every
              sample takes half from each set: fit 1,250 + 1,250 | sel 10,000 + 10,000 | cal 20,000 + 20,000 |
              eval 60,000 + 60,000. Writes DATA/yfcc10m/queries/official_<part>.npy and PREP/yfcc10m/official_rows_<part>.npy
              (columns: set 0 public / 1 private, row in that set).

Usage:
    python data/prepare_yfcc10m.py split --raw RAW/yfcc --prep PREP --data DATA
    python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds yfcc10m --graph A --seed 94926219 --threads 32 \
        --base PREP/yfcc10m/base_pre.npy
    python data/prepare_yfcc10m.py renumber --prep PREP --data DATA --graph-file WORK/graphs/yfcc10m_A.hnsw
    python data/prepare_yfcc10m.py official --raw RAW/yfcc --prep PREP --data DATA
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS = 'yfcc10m'
PARTS = [('fit', 2500), ('sel', 20000), ('cal', 40000), ('eval', 120000), ('spare', 20000)]
OFFICIAL_PARTS = [('fit', 1250), ('sel', 10000), ('cal', 20000), ('eval', 60000)]
VOCAB = 1000
SPLIT_SEED = 2090873961
OFFICIAL_SEEDS = (1550168123, 414748336)


def unit_f32(X8, rows, chunk=1 << 20):
    out = np.empty((len(rows), X8.shape[1]), np.float32)
    for s in range(0, len(rows), chunk):
        x = X8[rows[s:s + chunk]].astype(np.float32); nr = np.linalg.norm(x, axis=1, keepdims=True)
        assert (nr > 0).all(); out[s:s + chunk] = x / nr
    return out


def csr_sorted(rowlists_ptr, idx):
    """sort the category ids inside every row"""
    n = len(rowlists_ptr) - 1; rid = np.repeat(np.arange(n), np.diff(rowlists_ptr))
    o = np.lexsort((idx, rid)); return idx[o]


def u8bin(p):
    n, d = (int(x) for x in np.fromfile(p, np.int32, 2)); return np.fromfile(p, np.uint8, offset=8).reshape(n, d)


def split(a):
    RAW = a.raw; prep = os.path.join(a.prep, DS); qd = os.path.join(a.data, DS, 'queries')
    os.makedirs(prep, exist_ok=True); os.makedirs(qd, exist_ok=True); t0 = time.time()
    n_all, d = np.fromfile(os.path.join(RAW, 'base.10M.u8bin'), np.int32, 2); n_all, d = int(n_all), int(d)
    X8 = np.fromfile(os.path.join(RAW, 'base.10M.u8bin'), np.uint8, offset=8).reshape(n_all, d)
    f = open(os.path.join(RAW, 'base.metadata.10M.spmat'), 'rb'); nr, nc, nnz = (int(x) for x in np.fromfile(f, np.int64, 3))
    indptr = np.fromfile(f, np.int64, nr + 1); indices = np.fromfile(f, np.int32, nnz); f.close()
    assert nr == n_all
    freq = np.bincount(indices, minlength=nc); order_t = np.argsort(-freq, kind='stable')
    remap = -np.ones(nc, np.int64); remap[order_t[:VOCAB]] = np.arange(VOCAB)
    rid = np.repeat(np.arange(n_all), np.diff(indptr)); r = remap[indices]; m = r >= 0
    nk = np.bincount(rid[m], minlength=n_all)                  # tags of the vocabulary per item
    kept = np.nonzero(nk > 0)[0]
    seed = a.seed; perm = np.random.default_rng(seed).permutation(kept)
    nh = sum(c for _, c in PARTS); held = perm[:nh]; base_rows = np.sort(perm[nh:])
    info = dict(seed=seed, n_raw=n_all, dim=d, vocab_raw=nc, vocab=VOCAB, n_no_tag=int((np.diff(indptr) == 0).sum()),
                n_kept=int(len(kept)), n_heldout=nh, n_base=int(len(base_rows)), parts={}, tags_per_item_mean=float(nk[kept].mean()),
                tags_per_item_max=int(nk[kept].max()))
    lo = 0
    for p, c in PARTS:
        rows = held[lo:lo + c]; lo += c
        np.save(os.path.join(qd, f'{p}.npy'), unit_f32(X8, rows)); np.save(os.path.join(prep, f'rows_{p}.npy'), rows)
        info['parts'][p] = c
    np.save(os.path.join(prep, 'base_pre.npy'), unit_f32(X8, base_rows)); np.save(os.path.join(prep, 'base_rows_pre.npy'), base_rows)
    # categories of the base rows (order before renumbering), ids = frequency rank of the tag
    bm = np.zeros(n_all, bool); bm[base_rows] = True; sel = m & bm[rid]
    cnt = np.bincount(rid[sel], minlength=n_all)[base_rows]
    cptr = np.zeros(len(base_rows) + 1, np.int64); cptr[1:] = np.cumsum(cnt)
    cidx = csr_sorted(cptr, r[sel].astype(np.int32))           # rid[sel] is ascending, so rows follow base_rows order
    np.savez(os.path.join(prep, 'categories_pre.npz'), ptr=cptr, idx=cidx, n_categories=np.int64(VOCAB))
    info['secs'] = time.time() - t0
    json.dump(info, open(os.path.join(prep, 'split.json'), 'w'), indent=1); print(json.dumps(info))


def renumber_base(a):
    from renumber import renumber_dataset
    t0 = time.time()
    perm = renumber_dataset(os.path.join(a.prep, DS), os.path.join(a.data, DS), a.graph_file)
    print('renumbered', len(perm), 'moved', int((perm != np.arange(len(perm))).sum()), f'{time.time()-t0:.0f}s')


def official(a):
    RO = a.raw; prep = os.path.join(a.prep, DS); qd = os.path.join(a.data, DS, 'queries'); os.makedirs(qd, exist_ok=True)
    Q = [u8bin(os.path.join(RO, 'query.public.100K.u8bin')), u8bin(os.path.join(RO, 'query.private.2727415019.100K.u8bin'))]
    dim = int(np.load(os.path.join(a.data, DS, 'base.npy'), mmap_mode='r').shape[1])
    assert all(q.shape == (100000, dim) for q in Q)
    info = dict(sets=['public', 'private'], n=[len(q) for q in Q], dim=dim)
    seeds = list(a.seeds); perms = [np.random.default_rng(s).permutation(100000) for s in seeds]; info['seeds'] = seeds
    lo = 0; info['parts'] = {}
    for p, c in OFFICIAL_PARTS:
        rows = np.concatenate([np.stack([np.full(c, si), perms[si][lo:lo + c]], 1) for si in (0, 1)])
        emb = np.concatenate([unit_f32(Q[si], perms[si][lo:lo + c]) for si in (0, 1)])
        np.save(os.path.join(qd, f'official_{p}.npy'), emb); np.save(os.path.join(prep, f'official_rows_{p}.npy'), rows)
        info['parts'][p] = dict(n=int(len(rows)), public=int((rows[:, 0] == 0).sum()), private=int((rows[:, 0] == 1).sum())); lo += c
    info['unread_per_set'] = 100000 - lo
    json.dump(info, open(os.path.join(prep, 'official.json'), 'w'), indent=1); print(json.dumps(info, indent=1))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('split'); p.add_argument('--raw', required=True); p.add_argument('--prep', required=True); p.add_argument('--data', required=True)
    p.add_argument('--seed', type=int, default=SPLIT_SEED, help='seed of the permutation of the kept items')
    p = sub.add_parser('renumber'); p.add_argument('--prep', required=True); p.add_argument('--data', required=True)
    p.add_argument('--graph-file', required=True, help='HNSW file of the multi-threaded build (rewritten in place)')
    p = sub.add_parser('official'); p.add_argument('--raw', required=True); p.add_argument('--prep', required=True); p.add_argument('--data', required=True)
    p.add_argument('--seeds', type=int, nargs=2, default=OFFICIAL_SEEDS, help='seeds of the permutations of the public and the private set')
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(split=split, renumber=renumber_base, official=official)[a.cmd](a)

"""Amazon and arXiv: texts of the base items and of the item queries, and the category sets of the base.

Paper: Section VI-A, Datasets and Table I ("one random permutation of the pool selects the base, 2,500 FIT queries and
200,000 further queries").

    texts    parse the raw files (data/text_pool.py), draw the pool permutation and write
             PREP/<ds>/texts_items.parquet      10,000 fitpool queries | 1,000 unused items | 1,000,000 base items
             PREP/<ds>/texts_queries_a.parquet  queries 0 .. 99,999 of the 200,000 further queries
             PREP/<ds>/texts_queries_b.parquet  queries 100,000 .. 199,999
             PREP/<ds>/titles_queries.parquet   titles of the 200,000 queries (data/prepare_title_queries.py)
             PREP/<ds>/category_names.json      category name of every category id
             DATA/<ds>/categories_main.npz      category sets of the base items (CSR)
             Category ids are frequency ranks over the 1,011,000 rows of texts_items.parquet.
    install  after data/embed_texts.py has embedded the three text files (one call per file), write
             DATA/<ds>/base.npy, DATA/<ds>/queries/fitpool.npy and DATA/<ds>/queries/chunk0.npy .. chunk19.npy.

Usage:
    python data/prepare_items.py texts --ds amazon --raw RAW/amazon --prep PREP --data DATA
    python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/texts_items.parquet \
        PREP/amazon/texts_queries_a.parquet PREP/amazon/texts_queries_b.parquet
    python data/prepare_items.py install --ds amazon --prep PREP --data DATA
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_pool as TP


def category_csr(catsets):
    """CSR of the category sets, with category ids ranked by frequency (ties by first appearance), and the names by id."""
    freq = collections.Counter(c for s in catsets for c in s)
    names = [c for c, _ in freq.most_common()]
    cid = {c: j for j, c in enumerate(names)}
    lens = np.array([len(s) for s in catsets], np.int64)
    ptr = np.zeros(len(catsets) + 1, np.int64); ptr[1:] = np.cumsum(lens)
    idx = np.array([cid[c] for s in catsets for c in s], np.int32)
    return ptr, idx, names


def texts(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    prep = os.path.join(a.prep, a.ds); data = os.path.join(a.data, a.ds); os.makedirs(prep, exist_ok=True); os.makedirs(data, exist_ok=True)
    src_ids, txt, titles, catsets, stats = TP.parse(a.ds, a.raw)
    N = len(txt); stats["kept"] = N
    if N < TP.N_ITEMS + TP.N_QUERIES:
        raise SystemExit(f"the pool has only {N} items")
    perm = TP.pool_permutation(N, a.seed)
    items = perm[:TP.N_ITEMS]
    pq.write_table(pa.table({"row": np.arange(len(items), dtype=np.int64), "src_id": [src_ids[i] for i in items],
                             "text": [txt[i] for i in items]}), os.path.join(prep, "texts_items.parquet"))
    queries = perm[TP.N_ITEMS:TP.N_ITEMS + TP.N_QUERIES]
    half = TP.N_QUERIES // 2
    for name, lo in (("a", 0), ("b", half)):
        q = queries[lo:lo + half]
        pq.write_table(pa.table({"row": np.arange(lo, lo + half, dtype=np.int64), "pool_idx": q.astype(np.int64),
                                 "src_id": [src_ids[i] for i in q], "text": [txt[i] for i in q]}),
                       os.path.join(prep, f"texts_queries_{name}.parquet"))
    pq.write_table(pa.table({"row": np.arange(TP.N_QUERIES, dtype=np.int64), "pool_idx": queries.astype(np.int64),
                             "src_id": [src_ids[i] for i in queries], "text": [titles[i] for i in queries]}),
                   os.path.join(prep, "titles_queries.parquet"))
    # category ids over the 1,011,000 rows; the stored CSR holds the base rows only
    ptr, idx, names = category_csr([catsets[i] for i in items])
    lo = TP.N_FITPOOL + TP.N_SPARE
    np.savez(os.path.join(data, "categories_main.npz"), ptr=ptr[lo:] - ptr[lo], idx=idx[ptr[lo]:], n_categories=np.int64(len(names)))
    json.dump(names, open(os.path.join(prep, "category_names.json"), "w"))
    lens = np.diff(ptr)[lo:]
    info = dict(ds=a.ds, seed=a.seed, n_pool=N, n_fitpool=TP.N_FITPOOL, n_spare=TP.N_SPARE, n_base=TP.N_BASE, n_queries=TP.N_QUERIES,
                stats=dict(stats), n_categories=len(names), categories_per_base_item_max=int(lens.max()),
                categories_per_base_item_mean=float(lens.mean()))
    json.dump(info, open(os.path.join(prep, "pool.json"), "w"), indent=1); print(json.dumps(info), flush=True)


def install(a):
    prep = os.path.join(a.prep, a.ds); data = os.path.join(a.data, a.ds); os.makedirs(os.path.join(data, "queries"), exist_ok=True)
    E = np.load(os.path.join(prep, "emb_items.npy")); assert E.shape[0] == TP.N_ITEMS and E.dtype == np.float32
    np.save(os.path.join(data, "queries", "fitpool.npy"), E[:TP.N_FITPOOL])
    np.save(os.path.join(data, "base.npy"), E[TP.N_FITPOOL + TP.N_SPARE:])
    for name, first in (("a", 0), ("b", 10)):
        E = np.load(os.path.join(prep, f"emb_queries_{name}.npy")); assert E.shape[0] == TP.N_QUERIES // 2
        for k in range(10):
            np.save(os.path.join(data, "queries", f"chunk{first + k}.npy"), E[k * 10000:(k + 1) * 10000])
    print("installed", a.ds, flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("texts"); p.add_argument("--ds", required=True, choices=("amazon", "arxiv"))
    p.add_argument("--raw", required=True, help="directory of the raw files of this dataset")
    p.add_argument("--prep", required=True, help="directory for intermediate files"); p.add_argument("--data", required=True)
    p.add_argument("--seed", type=int, default=TP.POOL_SEED, help="seed of the pool permutation")
    p = sub.add_parser("install"); p.add_argument("--ds", required=True, choices=("amazon", "arxiv"))
    p.add_argument("--prep", required=True); p.add_argument("--data", required=True)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    dict(texts=texts, install=install)[a.cmd](a)

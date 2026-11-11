"""Amazon and arXiv: the items that the index updates insert, drawn from the part of the pool that no other sample uses.

Paper: Section VI-B, paragraph "Query shift and index updates" (inserts and in-place replacements of 1% and 5% of the base).

The pool is parsed and permuted as in data/prepare_items.py; the rows after the base and the 200,000 queries are unused.
One random permutation of them (seed per dataset in configs/seeds.json) gives the update items; the first 50,000 are
kept. Texts follow the item recipe. Categories are mapped through PREP/<ds>/category_names.json; a category name that no
sampled item carries gets the next free id, in order of first appearance.
Check: the same mapping reproduces the stored category sets of 1,000 random base items.

Output  PREP/<ds>/texts_update_items.parquet (row, pool_idx, src_id, text), PREP/<ds>/update_items_categories.npz
        (ptr, idx over the 50,000 items, n_categories_base, n_categories), PREP/<ds>/update_items.json.
Embed the texts with data/embed_texts.py, then run data/prepare_updated_bases.py.

Usage:
    python data/prepare_update_items.py --ds amazon --raw RAW/amazon --prep PREP --data DATA
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_pool as TP

N_UPDATE = 50000
ITEM_SEEDS = dict(amazon=968572019, arxiv=2079861226)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', required=True, choices=('amazon', 'arxiv'))
    ap.add_argument('--raw', required=True, help='directory of the raw files of this dataset')
    ap.add_argument('--prep', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--pool-seed', type=int, default=TP.POOL_SEED, help='seed of the pool permutation')
    ap.add_argument('--seed', type=int, default=None, help='seed of the permutation of the unused items (default: the recorded seed)')
    ap.add_argument('--check-seed', type=int, default=1, help='seed of the draw of the checked base items')
    return ap.parse_args()


def main():
    a = parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    prep = os.path.join(a.prep, a.ds); seed = a.seed if a.seed is not None else ITEM_SEEDS[a.ds]
    src_ids, texts, titles, catsets, stats = TP.parse(a.ds, a.raw)
    N = len(texts); perm = TP.pool_permutation(N, a.pool_seed); n_s = TP.N_ITEMS
    unused = perm[n_s + TP.N_QUERIES:]; upd = np.random.default_rng(seed).permutation(unused)[:N_UPDATE]
    names = json.load(open(os.path.join(prep, 'category_names.json'))); cid = {c: j for j, c in enumerate(names)}; m_base = len(names)
    z = np.load(os.path.join(a.data, a.ds, 'categories_main.npz')); ptr0, idx0 = z['ptr'], z['idx']; assert int(z['n_categories']) == m_base

    def ids_of(cs, extra):                                             # category ids of one item; unseen names get the next free ids in order of appearance
        out = []
        for c in cs:
            if c not in cid: cid[c] = m_base + len(extra); extra.append(c)
            out.append(cid[c])
        return out

    # check on base items: base item b is the pool position n_fitpool + n_spare + b
    off = TP.N_FITPOOL + TP.N_SPARE
    chk = np.random.default_rng(a.check_seed).choice(TP.N_BASE, 1000, replace=False); bad = 0; probe = []
    for b in chk:
        got = ids_of(catsets[perm[off + b]], probe); bad += int(got != idx0[ptr0[b]:ptr0[b + 1]].tolist())
    check = dict(rows=int(len(chk)), mismatch=int(bad), added_ids_in_check=len(probe)); print('check', check, flush=True)
    assert bad == 0 and not probe
    extra = []; rows = [ids_of(catsets[i], extra) for i in upd]
    lens = np.array([len(r) for r in rows], np.int64); ptr = np.zeros(N_UPDATE + 1, np.int64); ptr[1:] = np.cumsum(lens)
    idx = np.array([c for r in rows for c in r], np.int32)
    ntxt = [texts[i] for i in upd]
    pq.write_table(pa.table({'row': np.arange(N_UPDATE, dtype=np.int64), 'pool_idx': upd.astype(np.int64), 'src_id': [src_ids[i] for i in upd],
                             'text': ntxt}), os.path.join(prep, 'texts_update_items.parquet'))
    np.savez(os.path.join(prep, 'update_items_categories.npz'), ptr=ptr, idx=idx, n_categories_base=np.int64(m_base),
             n_categories=np.int64(m_base + len(extra)))
    info = dict(ds=a.ds, check=check, n_unused=int(len(unused)), n_items=N_UPDATE, seed=seed, n_categories_base=m_base, added_category_ids=len(extra),
                items_with_added_category=int(sum(any(c >= m_base for c in r) for r in rows)), categories_per_item_mean=float(lens.mean()))
    json.dump(info, open(os.path.join(prep, 'update_items.json'), 'w'), indent=1); print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()

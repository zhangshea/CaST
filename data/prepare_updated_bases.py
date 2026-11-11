"""Amazon and arXiv: the four updated bases of a dataset.

Paper: Section VI-B, paragraph "Query shift and index updates" and Fig. 5.

    <ds>_insert1, <ds>_insert5     the first 10,000 / 50,000 update items are appended to the base (ids n_base, n_base + 1, ...);
    <ds>_replace1, <ds>_replace5   the same items replace 10,000 / 50,000 base items, in the slots given by the first
                                   entries of one random permutation of the base ids (seed per dataset in configs/seeds.json).

Per updated base: DATA/<name>/base.npy, categories_main.npz, queries/ (links to the query groups of <ds>),
update_vectors.npy, update_slots.npy (replacements) and update.json. pipeline/build_graphs.py update applies the same
update to graph A; pipeline/exact_answers.py recomputes the exact answers on the updated base.

Usage:
    python data/prepare_updated_bases.py --ds amazon --prep PREP --data DATA
"""
import argparse
import glob
import json
import os

import numpy as np

SLOT_SEEDS = dict(amazon=164228088, arxiv=1826210509)
SETTINGS = (('insert1', 'insert', 10000), ('insert5', 'insert', 50000), ('replace1', 'replace', 10000), ('replace5', 'replace', 50000))


def csr(rs):
    ptr = np.zeros(len(rs) + 1, np.int64); ptr[1:] = np.cumsum([len(r) for r in rs]); return ptr, np.concatenate(rs).astype(np.int32)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', required=True, choices=('amazon', 'arxiv'))
    ap.add_argument('--prep', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--seed', type=int, default=None, help='seed of the permutation of the base slots (default: the recorded seed)')
    return ap.parse_args()


def main():
    a = parse_args(); ds = a.ds; prep = os.path.join(a.prep, ds); seed_slot = a.seed if a.seed is not None else SLOT_SEEDS[ds]
    X = np.load(os.path.join(a.data, ds, 'base.npy')); nb = X.shape[0]; assert X.dtype == np.float32
    z = np.load(os.path.join(a.data, ds, 'categories_main.npz')); cptr, cidx, m_base = z['ptr'], z['idx'], int(z['n_categories'])
    E = np.load(os.path.join(prep, 'emb_update_items.npy')); zi = np.load(os.path.join(prep, 'update_items_categories.npz'))
    icptr, icidx, m_all = zi['ptr'], zi['idx'], int(zi['n_categories']); assert int(zi['n_categories_base']) == m_base and E.shape == (50000, X.shape[1])
    rows = lambda cp, ci, lo, hi: [ci[cp[r]:cp[r + 1]] for r in range(lo, hi)]
    base_rows = rows(cptr, cidx, 0, nb); item_rows = rows(icptr, icidx, 0, 50000)
    slots = np.random.default_rng(seed_slot).permutation(nb)[:50000].astype(np.int64)
    for tag, mode, m in SETTINGS:
        S = os.path.join(a.data, f'{ds}_{tag}'); os.makedirs(os.path.join(S, 'queries'), exist_ok=True)
        if mode == 'insert':
            Xb = np.concatenate([X, E[:m]]); allr = base_rows + item_rows[:m]
        else:
            Xb = X.copy(); Xb[slots[:m]] = E[:m]; allr = list(base_rows)
            for j, s in enumerate(slots[:m]): allr[int(s)] = item_rows[j]
        np.save(os.path.join(S, 'base.npy'), Xb)
        bp, bi = csr(allr); np.savez(os.path.join(S, 'categories_main.npz'), ptr=bp, idx=bi, n_categories=np.int64(m_all))
        for f in sorted(glob.glob(os.path.join(a.data, ds, 'queries', 'fitpool.npy')) + glob.glob(os.path.join(a.data, ds, 'queries', 'chunk*.npy'))):
            dst = os.path.join(S, 'queries', os.path.basename(f))
            if not os.path.lexists(dst): os.symlink(os.path.relpath(f, os.path.dirname(dst)), dst)
        np.save(os.path.join(S, 'update_vectors.npy'), np.ascontiguousarray(E[:m]))
        if mode == 'replace': np.save(os.path.join(S, 'update_slots.npy'), slots[:m])
        info = dict(ds=ds, name=f'{ds}_{tag}', mode=mode, m=m, n_base=int(len(Xb)), seed_slots=seed_slot, n_categories=m_all)
        json.dump(info, open(os.path.join(S, 'update.json'), 'w'), indent=1); print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()

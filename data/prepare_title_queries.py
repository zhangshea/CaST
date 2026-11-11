"""Amazon and arXiv: title-only queries (the title of an item instead of its full text).

Paper: Table II (rows "title queries": a complete workload with its own FIT, SEL, CAL and EVAL samples) and Section VI-B,
paragraph "Query shift and index updates" (the titles of the CAL items and of 40,000 EVAL items, Fig. 5).

Both sets take the titles of the 200,000 item queries written by data/prepare_items.py (PREP/<ds>/titles_queries.parquet).
    --set refit   one permutation of the 200,000 rows (seed in configs/seeds.json): FIT = the first 2,500, SEL the next
                  20,000, CAL the next 40,000, EVAL the next 120,000; 17,500 rows stay unused. Texts are written in files
                  of at most 10,000 rows, PREP/<ds>/title_chunks/texts_<part>_<nn>.parquet, each embedded in its own call.
                  Groups: title_fit, title_sel, title_cal, title_eval.
    --set shift   the rows of CAL and the first 40,000 rows of EVAL of the main workload, in the order of these samples
                  (permutation of the 200,000 queries of cast.setting): PREP/<ds>/texts_titleshift.parquet, 80,000 rows
                  embedded in one call. Groups: titleshift_cal, titleshift_eval.
Check: every full text starts with its title followed by ". " (texts are cut at 1,500 characters).

Usage:
    python data/prepare_title_queries.py texts --set refit --ds amazon --prep PREP
    python data/embed_texts.py --model MODEL --texts PREP/amazon/title_chunks/texts_*.parquet
    python data/prepare_title_queries.py install --set refit --ds amazon --prep PREP --data DATA
"""
import argparse
import glob
import json
import os

import numpy as np

N = 200000
REFIT_SEED = 1900014855          # permutation of the title queries into FIT, SEL, CAL, EVAL
POOL_SEED = 0x13528ef            # permutation of the 200,000 queries of the main workload (cast.setting, 'pool_permutation')
REFIT_PARTS = (('fit', 0, 2500), ('sel', 2500, 22500), ('cal', 22500, 62500), ('eval', 62500, 182500))


def texts(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    prep = os.path.join(a.prep, a.ds)
    T = pq.read_table(os.path.join(prep, 'titles_queries.parquet'), columns=['row', 'pool_idx', 'src_id', 'text']).to_pydict()
    assert T['row'] == list(range(N))
    titles = T['text']
    full = sum((pq.read_table(os.path.join(prep, f'texts_queries_{h}.parquet'), columns=['text']).column('text').to_pylist() for h in 'ab'), [])
    TT = dict(prefix_mismatch=int(sum(not f.startswith((t + '. ')[:1500]) for t, f in zip(titles, full))), empty_titles=int(sum(not t for t in titles)))
    print('title check', json.dumps(TT), flush=True); assert TT['prefix_mismatch'] == 0 and TT['empty_titles'] == 0
    lt = np.array([len(t) for t in titles]); lf = np.array([len(f) for f in full])
    info = dict(ds=a.ds, set=a.set, check=TT, title_len_mean=float(lt.mean()), title_len_median=float(np.median(lt)), full_len_mean=float(lf.mean()))
    if a.set == 'refit':
        P = np.random.default_rng(a.seed if a.seed is not None else REFIT_SEED).permutation(N)
        parts = [(nm, P[lo:hi]) for nm, lo, hi in REFIT_PARTS]
        CD = os.path.join(prep, 'title_chunks'); os.makedirs(CD, exist_ok=True); chunks = {}
        for nm, rows in parts:
            for c in range(0, len(rows), 10000):
                r_ = rows[c:c + 10000]; cn = f'{nm}_{c // 10000:02d}'; chunks[cn] = int(len(r_))
                pq.write_table(pa.table({'row': r_.astype(np.int64), 'text': [titles[i] for i in r_]}), os.path.join(CD, f'texts_{cn}.parquet'))
        np.savez(os.path.join(prep, 'title_rows.npz'), **{nm: rows for nm, rows in parts})
        info.update(seed=int(a.seed if a.seed is not None else REFIT_SEED), sizes={nm: int(len(r)) for nm, r in parts}, chunks=chunks)
    else:
        Pf = np.random.default_rng(a.seed if a.seed is not None else POOL_SEED).permutation(N); CAL, EVAL = Pf[20000:60000], Pf[60000:180000]
        rows = np.concatenate([CAL, EVAL[:40000]]); part = ['cal'] * 40000 + ['eval'] * 40000
        pq.write_table(pa.table({'part': part, 'row': rows.astype(np.int64), 'pool_idx': [T['pool_idx'][i] for i in rows],
                                 'src_id': [T['src_id'][i] for i in rows], 'text': [titles[i] for i in rows]}),
                       os.path.join(prep, 'texts_titleshift.parquet'))
        info.update(seed=int(a.seed if a.seed is not None else POOL_SEED), n=int(len(rows)))
    json.dump(info, open(os.path.join(prep, f'title_{a.set}.json'), 'w'), indent=1); print(json.dumps(info), flush=True)


def install(a):
    prep = os.path.join(a.prep, a.ds); qd = os.path.join(a.data, a.ds, 'queries'); os.makedirs(qd, exist_ok=True)
    if a.set == 'refit':
        for nm, lo, hi in REFIT_PARTS:
            fs = sorted(glob.glob(os.path.join(prep, 'title_chunks', f'emb_{nm}_[0-9][0-9].npy')))
            assert len(fs) == (hi - lo + 9999) // 10000, (nm, len(fs))
            E = np.concatenate([np.load(f) for f in fs]); assert len(E) == hi - lo
            np.save(os.path.join(qd, f'title_{nm}.npy'), E)
    else:
        E = np.load(os.path.join(prep, 'emb_titleshift.npy')); assert E.shape[0] == 80000
        np.save(os.path.join(qd, 'titleshift_cal.npy'), E[:40000]); np.save(os.path.join(qd, 'titleshift_eval.npy'), E[40000:])
    print('installed', a.ds, a.set, flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('texts', 'install'):
        p = sub.add_parser(cmd); p.add_argument('--set', required=True, choices=('refit', 'shift'))
        p.add_argument('--ds', required=True, choices=('amazon', 'arxiv')); p.add_argument('--prep', required=True)
        if cmd == 'texts':
            p.add_argument('--seed', type=int, default=None, help='seed of the permutation (default: the recorded seed of the set)')
        else:
            p.add_argument('--data', required=True)
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(texts=texts, install=install)[a.cmd](a)

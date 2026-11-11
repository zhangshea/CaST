"""YouTube-8M video-level features: download of a random set of training shards, base, categories and queries.

Paper: Section VI-A, Datasets and Table I; Table II (rows YouTube-8M).

Vectors: the 1,024-dimensional mean visual feature (mean_rgb) as float32, scaled to unit L2 norm; distance 1 - <q, x>.
The audio features are not used. Categories: the machine-generated entity labels (3,862 entities, original ids).

    fetch     draw training shards at random from the official download plan (a JSON with the md5 of every shard) and
              fetch them with md5 checks. The shard order is one random permutation of the plan entries (seed in
              configs/seeds.json), recorded in RAW/manifest.json; the first N are fetched, and a later call with a larger N
              continues the same order.
    split     parse every fetched shard (TFRecord files of tf.train.Example records, read by the parser below); video ids
              must be distinct and labels below 3,862; videos without labels are dropped. One random permutation of the
              rest (seed in configs/seeds.json): the first 202,500 are held out as fit 2,500 | sel 20,000 | cal 40,000 |
              eval 120,000 | spare 20,000 (in this order), the next 1,000,000 are the base, the rest is unused. Writes
              DATA/youtube8m/queries/<group>.npy and, in PREP/youtube8m, base_pre.npy, base_rows_pre.npy,
              categories_pre.npz, rows_<group>.npy, video_ids.npy, split.json.
    renumber  after the multi-threaded graph build on PREP/youtube8m/base_pre.npy: renumber the base so that item id ==
              node id; writes DATA/youtube8m/base.npy and categories_main.npz.

Usage:
    python data/prepare_youtube8m.py fetch --plan RAW/video_train.json --raw RAW/youtube8m --n 1230
    python data/prepare_youtube8m.py split --raw RAW/youtube8m --prep PREP --data DATA
    python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds youtube8m --graph A --seed 1236488918 --threads 32 \
        --base PREP/youtube8m/base_pre.npy
    python data/prepare_youtube8m.py renumber --prep PREP --data DATA --graph-file WORK/graphs/youtube8m_A.hnsw
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS = 'youtube8m'
PARTS = [('fit', 2500), ('sel', 20000), ('cal', 40000), ('eval', 120000), ('spare', 20000)]
NBASE, VOCAB = 1000000, 3862
SHARD_SEED = 8161741155825784349
SPLIT_SEED = 1859506485
SHARD_URL = 'http://us.data.yt8m.org/2/video/train/%s'


# ---- TFRecord / tf.train.Example parser
def records(path):
    with open(path, 'rb') as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            n = struct.unpack('<Q', h[:8])[0]
            data = f.read(n); f.read(4)
            yield data


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7f) << s; s += 7
        if c < 0x80:
            return r, i


def fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i); yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        elif wt == 2:
            L, i = varint(b, i); yield fn, wt, b[i:i + L]; i += L
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        else:
            raise ValueError('wire type %d' % wt)


def parse_example(b):
    out = {}
    for fn, _, feats in fields(b):                       # Example.features = 1
        if fn != 1:
            continue
        for fn2, _, entry in fields(feats):              # Features.feature (map entries) = 1
            key = feat = None
            for fn3, _, x in fields(entry):
                if fn3 == 1: key = x.decode()
                elif fn3 == 2: feat = x
            for fn4, _, lst in fields(feat):             # bytes_list = 1, float_list = 2, int64_list = 3
                if fn4 == 1:
                    out[key] = [x for f5, _, x in fields(lst) if f5 == 1]
                elif fn4 == 2:
                    out[key] = np.concatenate([np.frombuffer(x, '<f4') for f5, w5, x in fields(lst) if f5 == 1] or [np.zeros(0, '<f4')])
                elif fn4 == 3:
                    vals = []
                    for f5, w5, x in fields(lst):
                        if f5 != 1: continue
                        if w5 == 0: vals.append(x)
                        else:
                            i = 0
                            while i < len(x):
                                v, i = varint(x, i); vals.append(v)
                    out[key] = vals
    return out


def parse_shard(path):
    ids, labs, rgb = [], [], []
    for rec in records(path):
        e = parse_example(rec)
        ids.append(e['id'][0].decode()); labs.append(sorted(set(int(c) for c in e['labels']))); rgb.append(e['mean_rgb'])
    return ids, labs, np.stack(rgb).astype(np.float32)


def unit(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True); assert (nr > 0).all(); return (X / nr).astype(np.float32)


# ---- commands
def fetch(a):
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    out, N = a.raw, a.n; os.makedirs(out, exist_ok=True)
    files = json.load(open(a.plan))['files']; names = sorted(files)
    MF = os.path.join(out, 'manifest.json')
    if os.path.exists(MF):
        man = json.load(open(MF))
    else:
        seed = a.seed
        man = dict(seed=seed, n_plan=len(names), order=[names[i] for i in np.random.default_rng(seed).permutation(len(names))], fetched={})
        json.dump(man, open(MF, 'w'))

    def get(f):
        dst = os.path.join(out, f)
        for attempt in range(5):
            try:
                if not (os.path.exists(dst) and hashlib.md5(open(dst, 'rb').read()).hexdigest() == files[f]):
                    urllib.request.urlretrieve(a.url % f, dst + '.part'); os.replace(dst + '.part', dst)
                md5 = hashlib.md5(open(dst, 'rb').read()).hexdigest()
                if md5 == files[f]:
                    return f, os.path.getsize(dst), True
            except Exception:
                time.sleep(5 * (attempt + 1))
        return f, -1, False

    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        for f, sz, ok in ex.map(get, [f for f in man['order'][:N] if f not in man['fetched']]):
            if ok:
                man['fetched'][f] = sz
            else:
                print('FAILED', f, flush=True)
    json.dump(man, open(MF, 'w'))
    print('fetched', len(man['fetched']), 'of', N, 'bytes', sum(man['fetched'].values()), f'{time.time()-t:.0f}s', 'seed', man['seed'])


def split(a):
    from multiprocessing import Pool
    RAW = a.raw; prep = os.path.join(a.prep, DS); qd = os.path.join(a.data, DS, 'queries')
    os.makedirs(prep, exist_ok=True); os.makedirs(qd, exist_ok=True); t0 = time.time()
    man = json.load(open(os.path.join(RAW, 'manifest.json')))
    shards = [f for f in man['order'] if f in man['fetched']]
    with Pool(a.workers) as pool:
        parsed = pool.map(parse_shard, [os.path.join(RAW, f) for f in shards])
    ids = [i for p in parsed for i in p[0]]; labs = [l for p in parsed for l in p[1]]; X = np.concatenate([p[2] for p in parsed])
    del parsed
    assert len(set(ids)) == len(ids), 'duplicate video ids'
    assert max(max(l) for l in labs if l) < VOCAB, 'label id out of range'
    keep = np.array([i for i in range(len(labs)) if labs[i]], np.int64)
    n_pool = len(keep); need = sum(c for _, c in PARTS) + NBASE
    info = dict(shards=len(shards), seed_shards=man['seed'], n_parsed=len(ids), n_unlabelled=len(ids) - n_pool, n_pool=n_pool, dim=int(X.shape[1]))
    if n_pool < need:
        print(json.dumps(info)); raise SystemExit(f'the pool is too small: {need} labelled videos are needed; fetch more shards')
    seed = a.seed; perm = keep[np.random.default_rng(seed).permutation(n_pool)]
    nh = sum(c for _, c in PARTS); held = perm[:nh]; base_rows = perm[nh:nh + NBASE]
    info.update(seed=seed, n_heldout=nh, n_base=NBASE, n_unused=int(n_pool - need), parts={})
    lo = 0
    for p, c in PARTS:
        rows = held[lo:lo + c]; lo += c
        np.save(os.path.join(qd, f'{p}.npy'), unit(X[rows])); np.save(os.path.join(prep, f'rows_{p}.npy'), rows)
        info['parts'][p] = c
    XB = unit(X[base_rows]); np.save(os.path.join(prep, 'base_pre.npy'), XB); np.save(os.path.join(prep, 'base_rows_pre.npy'), base_rows)
    np.save(os.path.join(prep, 'video_ids.npy'), np.array(ids))
    cnt = np.array([len(labs[r]) for r in base_rows], np.int64)
    cptr = np.zeros(NBASE + 1, np.int64); cptr[1:] = np.cumsum(cnt)
    cidx = np.array([c for r in base_rows for c in labs[r]], np.int32)
    np.savez(os.path.join(prep, 'categories_pre.npz'), ptr=cptr, idx=cidx, n_categories=np.int64(VOCAB))
    info.update(labels_per_item_mean=float(cnt.mean()), labels_per_item_max=int(cnt.max()),
                entities_in_base=int(len(np.unique(cidx))), vocab=VOCAB, secs=time.time() - t0)
    json.dump(info, open(os.path.join(prep, 'split.json'), 'w'), indent=1); print(json.dumps(info))


def renumber_base(a):
    from renumber import renumber_dataset
    t0 = time.time()
    perm = renumber_dataset(os.path.join(a.prep, DS), os.path.join(a.data, DS), a.graph_file)
    print('renumbered', len(perm), 'moved', int((perm != np.arange(len(perm))).sum()), f'{time.time()-t0:.0f}s')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('fetch'); p.add_argument('--plan', required=True, help='official download plan of the training shards (JSON)')
    p.add_argument('--raw', required=True, help='directory for the shards'); p.add_argument('--n', type=int, default=1230, help='number of shards')
    p.add_argument('--seed', type=int, default=SHARD_SEED, help='seed of the shard order'); p.add_argument('--url', default=SHARD_URL)
    p = sub.add_parser('split'); p.add_argument('--raw', required=True); p.add_argument('--prep', required=True); p.add_argument('--data', required=True)
    p.add_argument('--seed', type=int, default=SPLIT_SEED, help='seed of the permutation of the labelled videos')
    p.add_argument('--workers', type=int, default=16, help='processes that parse the shards')
    p = sub.add_parser('renumber'); p.add_argument('--prep', required=True); p.add_argument('--data', required=True)
    p.add_argument('--graph-file', required=True, help='HNSW file of the multi-threaded build (rewritten in place)')
    return ap.parse_args()


if __name__ == '__main__':
    a = parse_args()
    dict(fetch=fetch, split=split, renumber=renumber_base)[a.cmd](a)

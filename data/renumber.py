"""Renumber the base of a dataset after a multi-threaded HNSW build, so that item id == graph node id.

Paper: implementation details of Section VI-A (the graphs of YFCC-10M and YouTube-8M are built with several threads).

A multi-threaded hnswlib build assigns node ids in arrival order. The graph is kept as it is and the items are renumbered:
item c of the renumbered base is the row perm[c] = label of node c, so links, levels and the entry point are unchanged, and
only the label field of each bottom-layer block is rewritten to c. Base vectors, category sets and row maps follow perm.
"""
import os

import numpy as np

HDR = 96   # header of an hnswlib index file: 6 size_t, int, uint32, 3 size_t, double, size_t


def header(path):
    h = np.fromfile(path, dtype=np.uint8, count=HDR)
    u = lambda a, b: int(np.frombuffer(h[a:b].tobytes(), np.uint64)[0])
    return dict(offsetLevel0=u(0, 8), cur_count=u(16, 24), size_per_elem=u(24, 32), label_offset=u(32, 40), offset_data=u(40, 48))


def relabel(path):
    """Rewrite the labels of the index file in place to the node ids; return perm (previous label of every node)."""
    h = header(path)
    assert h['offsetLevel0'] == 0
    n, s, lo = h['cur_count'], h['size_per_elem'], h['label_offset']
    blk = np.memmap(path, dtype=np.uint8, mode='r+', offset=HDR, shape=(n, s))
    perm = np.ascontiguousarray(blk[:, lo:lo + 8]).view(np.uint64).ravel().astype(np.int64)
    assert np.array_equal(np.sort(perm), np.arange(n)), 'labels are not a permutation of 0..n-1'
    blk[:, lo:lo + 8] = np.arange(n, dtype=np.uint64).view(np.uint8).reshape(n, 8)
    blk.flush(); del blk
    return perm


def renumber_dataset(prep, data, graph_file):
    """Apply the renumbering of graph_file to PREP/base_pre.npy, base_rows_pre.npy and categories_pre.npz; write
    DATA/base.npy, DATA/categories_main.npz and PREP/base_rows.npy (source row of every base item)."""
    if os.path.exists(os.path.join(prep, 'renumbering.npy')):
        raise SystemExit('this data set is already renumbered (PREP/renumbering.npy exists)')
    perm = relabel(graph_file); np.save(os.path.join(prep, 'renumbering.npy'), perm)
    X = np.load(os.path.join(prep, 'base_pre.npy'), mmap_mode='r'); np.save(os.path.join(data, 'base.npy'), np.ascontiguousarray(X[perm]))
    br = np.load(os.path.join(prep, 'base_rows_pre.npy')); np.save(os.path.join(prep, 'base_rows.npy'), br[perm])
    z = np.load(os.path.join(prep, 'categories_pre.npz')); cp, ci = z['ptr'], z['idx']
    deg = np.diff(cp)[perm]; cp2 = np.zeros(len(perm) + 1, np.int64); cp2[1:] = np.cumsum(deg)
    src = np.repeat(cp[perm], deg) + (np.arange(cp2[-1]) - np.repeat(cp2[:-1], deg))
    np.savez(os.path.join(data, 'categories_main.npz'), ptr=cp2, idx=ci[src].astype(np.int32), n_categories=z['n_categories'])
    return perm

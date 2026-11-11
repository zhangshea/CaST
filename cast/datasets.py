"""Access to the prepared data of a setting: base vectors, category sets, queries, exact answers and graphs.

Paper: Section VI-A (Datasets, Table I; Implementation details). The files are written by the scripts of data/ and by
pipeline/exact_answers.py and pipeline/build_graphs.py.

Layout:
    DATA/<ds>/base.npy                       (n_base, dim) float32 unit vectors; row = item id = graph node id
    DATA/<ds>/categories_<variant>.npz       category sets of the base items, CSR: ptr (n_base + 1,) int64, idx int32, n_categories
    DATA/<ds>/queries/<group>.npy            (n, dim) float32 query vectors of one group
    DATA/<ds>/answers/<group>_<workload>.npz S (n, k) int64 exact answers (item ids, -1 padded)
    WORK/graphs/<ds>_<graph>.hnsw            HNSW graph; WORK/graphs/<ds>_vamana/ann for the Vamana graph

Category variants:
    'main'           the categories of Section VI-A, capacity kappa per category;
    'kappa2'         two unit-capacity slots 2c and 2c + 1 for every category c of 'main' (capacity 2 per category);
    'unconstrained'  the same k unit-capacity slots on every item (every set of at most k items is feasible);
    'shared'         one category shared by every item, built in memory; used with capacity k it is unconstrained as well.
"""
import os
import struct

import numpy as np

# Datasets whose kappa = 2 and unconstrained workloads use the stores written by data/derive_category_stores.py.
SLOT_STORE_DATASETS = ('amazon', 'arxiv')


def _engine():
    import cast_engine
    return cast_engine


def queries(data_dir, ds, group):
    """Query vectors of one group, float32 (n, dim)."""
    return np.ascontiguousarray(np.load(os.path.join(data_dir, ds, 'queries', f'{group}.npy')), dtype=np.float32)


def base(data_dir, ds, mmap=True):
    """Base vectors, float32 (n_base, dim)."""
    return np.load(os.path.join(data_dir, ds, 'base.npy'), mmap_mode='r' if mmap else None)


def category_csr(data_dir, ds, variant='main'):
    """Category sets of the base items as (ptr, idx, n_categories); item i carries the categories idx[ptr[i]:ptr[i + 1]]."""
    if variant == 'shared':
        nb = base(data_dir, ds).shape[0]
        return np.arange(nb + 1, dtype=np.int64), np.zeros(nb, np.int32), 1
    z = np.load(os.path.join(data_dir, ds, f'categories_{variant}.npz'))
    ptr = np.ascontiguousarray(z['ptr'].astype(np.int64)); idx = np.ascontiguousarray(z['idx'].astype(np.int32))
    return ptr, idx, int(z['n_categories'])


def categories(data_dir, ds, variant='main', kappa=1):
    """Category store of the base items for the engine, with capacity kappa per category (a scalar or one value per category)."""
    ptr, idx, m = category_csr(data_dir, ds, variant)
    kap = np.full(m, kappa, np.int32) if np.isscalar(kappa) else np.asarray(kappa, np.int32)
    return _engine().Categories(ptr, idx, m, kap)


def workload_spec(ds, name):
    """Answer size k, category variant and capacity of a workload name.

    'k<k>_kappa1'        k items, categories 'main' with capacity 1 (the main workload is 'k10_kappa1');
    'k<k>_kappa2'        capacity 2 per category, as two unit-capacity slots per category (variant 'kappa2');
    'k<k>_unconstrained' no constraint: k unit slots shared by all items (variant 'unconstrained') for the text datasets,
                         one shared category of capacity k (variant 'shared') for the others.
    """
    ks, what = name.split('_', 1)
    if not ks.startswith('k') or what not in ('kappa1', 'kappa2', 'unconstrained'):
        raise ValueError(f'unknown workload {name}')
    k = int(ks[1:])
    slots = ds.split('_')[0] in SLOT_STORE_DATASETS
    if what == 'kappa1':
        return dict(k=k, variant='main', kappa=1)
    if what == 'kappa2':
        if not slots:
            raise ValueError(f'workload {name} is prepared for {SLOT_STORE_DATASETS} only')
        return dict(k=k, variant='kappa2', kappa=1)
    return dict(k=k, variant='unconstrained', kappa=1) if slots else dict(k=k, variant='shared', kappa=k)


def answers_path(data_dir, ds, group, workload_name):
    return os.path.join(data_dir, ds, 'answers', f'{group}_{workload_name}.npz')


def exact_answers(data_dir, ds, group, workload):
    """Exact answers S of one group under one workload, int64 (n, k); -1 pads answers with fewer than k items."""
    return np.load(answers_path(data_dir, ds, group, workload))['S'].astype(np.int64)


def graph_path(work_dir, ds, graph):
    """File of an HNSW graph, or the directory of the Vamana graph when graph == 'vamana'."""
    g = str(graph)
    if g == 'vamana':
        return os.path.join(work_dir, 'graphs', f'{ds}_vamana')
    return os.path.join(work_dir, 'graphs', f'{ds}_{g}.hnsw')


def load_vamana(data_dir, work_dir, ds):
    """A Vamana graph written by the DiskANN reference implementation, as a graph with a bottom layer only.

    File <dir>/ann: header u64 file size, u32 maximum degree, u32 start node, u64 frozen points; then per node u32 degree
    and its neighbor ids. The search enters at the start node of the file.
    """
    b = np.fromfile(os.path.join(graph_path(work_dir, ds, 'vamana'), 'ann'), dtype=np.uint8)
    size, maxdeg, start, frozen = (int(v) for v in struct.unpack('<QIIQ', b[:24].tobytes()))
    if size != b.size or frozen != 0 or (b.size - 24) % 4:
        raise ValueError(f'unexpected Vamana graph file for {ds}')
    w = b[24:].view(np.uint32)
    X = np.ascontiguousarray(base(data_dir, ds, mmap=False), dtype=np.float32)
    n = X.shape[0]
    nbr0 = -np.ones((n, maxdeg), np.int32)
    pos = 0
    for i in range(n):
        k = int(w[pos]); nbr0[i, :k] = w[pos + 1:pos + 1 + k]; pos += 1 + k
    if pos != w.size:
        raise ValueError(f'the Vamana graph of {ds} does not have {n} nodes')
    return _engine().graph_from_arrays(X, nbr0, np.zeros(n, np.int32), -np.ones(n, np.int64), np.zeros(0, np.uint32), 0, start, 0)


def load_graph(data_dir, work_dir, ds, graph):
    """Engine graph of a setting: an HNSW file, or the Vamana directory when graph == 'vamana'."""
    if str(graph) == 'vamana':
        return load_vamana(data_dir, work_dir, ds)
    dim = int(base(data_dir, ds).shape[1])
    return _engine().load_hnswlib(graph_path(work_dir, ds, graph), dim)


def answers_equal(ids, S):
    """Per query: the returned ids and the exact answer S hold the same items."""
    return np.array([set(x.tolist()) == set(y.tolist()) for x, y in zip(ids, S)])


class Context:
    """Graph G and category store C of one setting, with the answer size k."""

    def __init__(self, data_dir, work_dir, ds, graph, k, variant='main', kappa=1):
        self.data_dir, self.work_dir, self.ds, self.graph, self.k = data_dir, work_dir, ds, str(graph), k
        self.variant, self.kappa = variant, kappa
        self.G = load_graph(data_dir, work_dir, ds, graph)
        self.C = categories(data_dir, ds, variant, kappa)

    @classmethod
    def for_workload(cls, data_dir, work_dir, ds, graph, workload_name):
        w = workload_spec(ds, workload_name)
        return cls(data_dir, work_dir, ds, graph, w['k'], variant=w['variant'], kappa=w['kappa'])

    def run(self, params, Q, S=None, L=None, threads=1):
        """Run the engine's search() with `params` (list size L if given) and return its dict.

        When the exact answers S are given, the key 'exact' is added: per query, the returned answer equals S.
        """
        p = dict(params)
        if L is not None:
            p['L'] = int(L)
        r = _engine().search(self.G, self.C, Q, p, threads=threads)
        if S is not None:
            r = dict(r); r['exact'] = answers_equal(r['ids'], S)
        return r

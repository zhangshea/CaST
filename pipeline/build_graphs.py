"""Build the graphs of the settings.

Paper: implementation details of Section VI-A (two HNSW graphs A and B with degree parameter 32 and construction list 200,
built with two independent draws of the insertion levels; a Vamana graph with out-degree 64, construction list 200 and
alpha = 1.2) and Section VI-B, paragraph "Query shift and index updates" (inserts and in-place replacements).

    hnsw     HNSW graph through the engine's build tool (hnswlib, inner-product space): M = 32, construction list 200,
             level generator seeded with --seed. With one thread the items are inserted in id order. A build with several
             threads assigns node ids in arrival order; the tool then reports label mismatches, and the base must be
             renumbered afterwards (data/prepare_yfcc10m.py renumber, data/prepare_youtube8m.py renumber).
    vamana   Vamana graph with the DiskANN reference implementation (diskannpy build_memory_index, l2 on unit vectors).
             The build is multi-threaded and not deterministic.
    update   Insert items into, or replace items of, an existing HNSW graph through the engine's update tool, which uses
             hnswlib's public calls: addPoint with an unused label (insert), addPoint with an existing label (in-place
             replacement that repairs the neighborhood). Vectors and slots come from data/prepare_updated_bases.py.
    degree   Mean out-degree of the bottom layer of a graph (the degree change reported for the replacements).

Recorded build seeds are listed in configs/seeds.json (graph A, graph B, the YFCC-10M and YouTube-8M graphs, the level
seeds of the updates).

Usage:
    python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds amazon --graph A --seed 866159836
    python pipeline/build_graphs.py vamana --data DATA --work WORK --ds amazon --threads 32
    python pipeline/build_graphs.py update --data DATA --work WORK --ds amazon --from-graph A --updated amazon_replace1 --seed SEED
    python pipeline/build_graphs.py degree --data DATA --work WORK --ds amazon --graph A
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from cast import datasets as D

ENGINE = os.path.join(HERE, '..', 'engine', 'build')


def build_hnsw(a):
    out = D.graph_path(a.work, a.ds, a.graph); os.makedirs(os.path.dirname(out), exist_ok=True)
    base = a.base or os.path.join(a.data, a.ds, 'base.npy')
    cmd = [a.tool or os.path.join(a.engine, 'bin', 'build_hnsw'), '--base', base, '--out', out, '--info', out[:-5] + '.json',
           '--M', str(a.M), '--efc', str(a.efc), '--seed', str(a.seed), '--threads', str(a.threads)]
    print(' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    # a multi-threaded build reports label mismatches (node id != item id) and exits with status 1; the renumbering step resolves it
    if r.returncode != 0 and not (a.threads > 1 and 'label mismatches' in r.stderr):
        raise SystemExit(r.returncode)


def build_vamana(a):
    import resource
    import diskannpy as dk
    out = D.graph_path(a.work, a.ds, 'vamana'); os.makedirs(out, exist_ok=True)
    X = np.ascontiguousarray(np.load(os.path.join(a.data, a.ds, 'base.npy')), dtype=np.float32)
    t = time.time()
    dk.build_memory_index(X, distance_metric='l2', index_directory=out, complexity=a.complexity, graph_degree=a.degree,
                          num_threads=a.threads, alpha=a.alpha, index_prefix='ann')
    sec = time.time() - t
    b = open(os.path.join(out, 'ann'), 'rb').read(); exp, maxdeg, start, nfz = struct.unpack('<QIIQ', b[:24])
    deg = []; off = 24
    while off < len(b):
        k = struct.unpack('<I', b[off:off + 4])[0]; deg.append(k); off += 4 + 4 * k
    deg = np.array(deg)
    rep = dict(n=int(X.shape[0]), dim=int(X.shape[1]), build_secs=sec, threads=a.threads,
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
               header=dict(expected_size=exp, file_size=len(b), max_observed_degree=maxdeg, start=start, frozen=nfz), nodes=int(len(deg)),
               deg_mean=float(deg.mean()), deg_min=int(deg.min()), deg_max=int(deg.max()), deg_p10=float(np.percentile(deg, 10)),
               sha256=hashlib.sha256(b).hexdigest(), diskannpy=getattr(dk, '__version__', None),
               params=dict(R=a.degree, L=a.complexity, alpha=a.alpha, metric='l2'))
    json.dump(rep, open(os.path.join(out, 'build_report.json'), 'w'), indent=1); print(json.dumps(rep), flush=True)


def update_hnsw(a):
    src = D.graph_path(a.work, a.ds, a.from_graph); out = D.graph_path(a.work, a.updated, a.graph or a.from_graph)
    d = os.path.join(a.data, a.updated); info = json.load(open(os.path.join(d, 'update.json'))); mode = info['mode']
    cmd = [a.tool or os.path.join(a.engine, 'bin', 'update_hnsw'), '--in', src, '--out', out, '--vec', os.path.join(d, 'update_vectors.npy'),
           '--mode', mode, '--seed', str(a.seed), '--info', out[:-5] + '.json']
    if mode == 'replace':
        cmd += ['--slots', os.path.join(d, 'update_slots.npy')]
    print(' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def mean_degree(a):
    sys.path.insert(0, os.path.abspath(a.engine))
    G = D.load_graph(a.data, a.work, a.ds, a.graph)
    print(json.dumps(dict(ds=a.ds, graph=a.graph, mean_degree_bottom_layer=float(G.mean_degree0()))), flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    def common(p):
        p.add_argument('--data', required=True, help='prepared data directory'); p.add_argument('--work', required=True, help='work directory')
        p.add_argument('--ds', required=True)
        p.add_argument('--engine', default=ENGINE, help='build directory of the engine (module cast_engine, tools in bin/)')
    p = sub.add_parser('hnsw', help='HNSW graph (hnswlib through the engine build tool)'); common(p)
    p.add_argument('--graph', required=True, help='graph name, for example A or B')
    p.add_argument('--seed', type=int, required=True, help='seed of the level generator')
    p.add_argument('--M', type=int, default=32); p.add_argument('--efc', type=int, default=200, help='construction list size')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--base', default=None, help='base vectors (default DATA/<ds>/base.npy; the vectors before renumbering for a multi-threaded build)')
    p.add_argument('--tool', default=None, help='build tool (default <engine>/bin/build_hnsw)')
    p = sub.add_parser('vamana', help='Vamana graph (diskannpy)'); common(p)
    p.add_argument('--degree', type=int, default=64); p.add_argument('--complexity', type=int, default=200, help='construction list size')
    p.add_argument('--alpha', type=float, default=1.2); p.add_argument('--threads', type=int, default=32)
    p = sub.add_parser('update', help='insert into or replace in an HNSW graph'); common(p)
    p.add_argument('--from-graph', required=True, help='graph of --ds that is updated (a copy is written)')
    p.add_argument('--updated', required=True, help='name of the updated dataset directory, for example amazon_replace1')
    p.add_argument('--graph', default=None, help='name of the updated graph (default: the name of --from-graph)')
    p.add_argument('--seed', type=int, required=True, help='seed of the level generator for the inserted items')
    p.add_argument('--tool', default=None, help='update tool (default <engine>/bin/update_hnsw)')
    p = sub.add_parser('degree', help='mean bottom-layer degree of a graph'); common(p)
    p.add_argument('--graph', required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    dict(hnsw=build_hnsw, vamana=build_vamana, update=update_hnsw, degree=mean_degree)[a.cmd](a)


if __name__ == '__main__':
    main()

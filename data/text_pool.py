"""Item pool of the two text datasets: item texts, titles and category sets parsed from the raw files, and the random
permutation that selects the base and the queries.

Paper: Section VI-A, Datasets (Amazon Reviews 2023, Sports and Outdoors; arXiv metadata snapshot).

Kept items have a non-empty title and at least one category; of items with the same lower-cased title the first is kept.
    arxiv   text = title + ". " + abstract; categories = the subject classes of the `categories` field.
    amazon  text = title + ". " + the first two features and the first description paragraph, cut at 1,500 characters;
            categories = the second-level node and the leaf of the category path, each node named by its full prefix.
Person-related fields are never read.

One permutation of the kept items (seed in configs/seeds.json) gives, in this order: 10,000 queries from which FIT is
drawn (group fitpool), 1,000 further items that the experiments do not use, the 1,000,000 base items, the 200,000 queries
that are split into SEL, CAL, EVAL and HOLD, and the remaining items, from which the items of the index updates are drawn.
"""
import collections
import glob
import os
import re

import numpy as np

N_FITPOOL, N_SPARE, N_BASE = 10_000, 1_000, 1_000_000
N_ITEMS = N_FITPOOL + N_SPARE + N_BASE          # rows embedded together: fitpool | spare | base
N_QUERIES = 200_000
POOL_SEED = 0

WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", s or "").strip()


def parse_arxiv(raw):
    """raw: directory with the parquet shards train-*.parquet of the snapshot."""
    import pyarrow.parquet as pq
    src_ids, texts, titles, catsets = [], [], [], []
    stats = collections.Counter()
    seen = set()
    for f in sorted(glob.glob(os.path.join(raw, "train-*.parquet"))):
        t = pq.read_table(f, columns=["id", "title", "abstract", "categories"])
        for i, ti, ab, ca in zip(*[t.column(c).to_pylist() for c in ["id", "title", "abstract", "categories"]]):
            stats["raw"] += 1
            ti = norm(ti); ab = norm(ab)
            cats = sorted(set((ca or "").split()))
            if not ti or not cats:
                stats["drop_empty"] += 1; continue
            key = ti.lower()
            if key in seen:
                stats["drop_dup_title"] += 1; continue
            seen.add(key)
            src_ids.append(i); texts.append(ti + ". " + ab); titles.append(ti)
            catsets.append(cats)
    return src_ids, texts, titles, catsets, stats


def parse_amazon(raw):
    """raw: directory with meta_Sports_and_Outdoors.jsonl."""
    import orjson
    src_ids, texts, titles, catsets = [], [], [], []
    stats = collections.Counter()
    seen = set()
    with open(os.path.join(raw, "meta_Sports_and_Outdoors.jsonl"), "rb") as fh:
        for line in fh:
            stats["raw"] += 1
            d = orjson.loads(line)
            ti = norm(d.get("title")); path = [norm(x) for x in (d.get("categories") or []) if norm(x)]
            if not ti or not path:
                stats["drop_empty"] += 1; continue
            key = ti.lower()
            if key in seen:
                stats["drop_dup_title"] += 1; continue
            seen.add(key)
            feats = [norm(x) for x in (d.get("features") or [])][:2]
            desc = [norm(x) for x in (d.get("description") or [])][:1]
            txt = ti + ". " + " ".join(feats + desc)
            src_ids.append(d.get("parent_asin") or ""); texts.append(txt[:1500]); titles.append(ti)
            pref = [" > ".join(path[:j + 1]) for j in range(len(path))]
            leaf = pref[-1]
            lvl2 = pref[1] if len(pref) >= 2 else pref[0]
            catsets.append(sorted({lvl2, leaf}))
    return src_ids, texts, titles, catsets, stats


def parse(ds, raw):
    """Kept items of a dataset in file order: source ids, texts, titles, category-name sets, and counts."""
    if ds == "arxiv":
        return parse_arxiv(raw)
    if ds == "amazon":
        return parse_amazon(raw)
    raise SystemExit("unknown dataset " + ds)


def pool_permutation(n, seed=POOL_SEED):
    """perm[r] = index (in file order of the kept items) of the item at position r of the pool order."""
    return np.random.default_rng(seed).permutation(n)

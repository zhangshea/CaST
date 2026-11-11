# Data

This package ships no data, no vectors, no graphs and no model weights. Every data set and the embedding model are public;
this directory holds the scripts that turn the original files into the layout the rest of the package reads
(Section VI-A of the paper, Datasets, Table I).

`RAW` is the directory of the downloaded files, `PREP` a directory for intermediate files (texts, embeddings before they are
split, row maps), `DATA` the prepared data, `WORK` the directory of graphs, logs and results.

## Sources

| Data set / model | Publisher and official location | What to download | Citation in the paper |
|---|---|---|---|
| Amazon Reviews 2023, category Sports and Outdoors | McAuley Lab, `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023` | item metadata `raw/meta_categories/meta_Sports_and_Outdoors.jsonl` (4,132,263,769 bytes; revision `2b6d039e`) | Hou et al., "Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders", ACL 2026 |
| arXiv metadata snapshot | `https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot` (CC0) | the 11 shards `data/train-000NN-of-00011.parquet` (2.7 GB; revision `c428226b`) | arXiv Metadata Snapshot (the URL above) |
| all-MiniLM-L6-v2 | Sentence-Transformers, `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`, `model.safetensors`, `modules.json`, `sentence_bert_config.json`, `1_Pooling/config.json` (revision `1110a243`) | Reimers and Gurevych, "Sentence-BERT", EMNLP-IJCNLP 2019; the model page |
| YFCC-10M, filtered-search data set of the NeurIPS'23 Big-ANN competition | `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/` | `base.10M.u8bin` (10,000,000 x 192 uint8), `base.metadata.10M.spmat` (tags), `query.public.100K.u8bin`, `query.private.2727415019.100K.u8bin` | Simhadri et al., "Results of the Big ANN: NeurIPS'23 Competition", NeurIPS 2025 Datasets and Benchmarks; Thomee et al., "YFCC100M", CACM 2016; Radford et al., CLIP, ICML 2021 |
| YouTube-8M, video-level features (training partition) | download plan `http://data.yt8m.org/2/download_plans/video_train.json` (3,844 shards with md5), shards at `http://us.data.yt8m.org/2/video/train/<file>` | 1,230 shards drawn at random by `prepare_youtube8m.py fetch` (5,864,109,495 bytes) | Abu-El-Haija et al., "YouTube-8M: A Large-Scale Video Classification Benchmark", arXiv:1609.08675 |

`download.sh RAW` fetches everything except the YouTube-8M shards with `curl` into
`RAW/amazon/`, `RAW/arxiv/`, `RAW/models/all-MiniLM-L6-v2/`, `RAW/yfcc/` and `RAW/video_train.json`.
Set `HF_REV_AMAZON`, `HF_REV_ARXIV`, `HF_REV_MODEL` to fetch a fixed revision of the Hugging Face files (default `main`).
The licenses and terms of the original publishers apply to all files.

Libraries: numpy, pyarrow, orjson (Amazon), torch and transformers (embedding), numba, scipy and torch (exact answers).

## Prepared layout

```
DATA/<ds>/base.npy                        (n_base, dim) float32 unit vectors; row = item id = graph node id
DATA/<ds>/categories_<variant>.npz        category sets of the base items, CSR: ptr (n_base + 1,) int64, idx int32, n_categories
DATA/<ds>/queries/<group>.npy             (n, dim) float32 query vectors of one group
DATA/<ds>/answers/<group>_<workload>.npz  exact answers (pipeline/exact_answers.py)
```
`ds` is `amazon`, `arxiv`, `yfcc10m`, `youtube8m`, or an updated base `<ds>_insert1`, `<ds>_insert5`, `<ds>_replace1`,
`<ds>_replace5` (Amazon and arXiv). The distance is `1 - <q, x>` on unit vectors. `cast/datasets.py` reads this layout.

| Data set | Query groups | Sizes |
|---|---|---|
| amazon, arxiv | `fitpool` (FIT is drawn from it), `chunk0` .. `chunk19` (the 200,000 queries split into SEL, CAL, EVAL, HOLD by `cast/setting.py`) | 10,000 each |
| amazon, arxiv | `title_fit`, `title_sel`, `title_cal`, `title_eval` (title queries as a workload of their own) | 2,500 / 20,000 / 40,000 / 120,000 |
| amazon, arxiv | `titleshift_cal`, `titleshift_eval` (titles of the CAL items and of the first 40,000 EVAL items, in sample order) | 40,000 each |
| yfcc10m, youtube8m | `fit`, `sel`, `cal`, `eval`, `spare` | 2,500 / 20,000 / 40,000 / 120,000 / 20,000 |
| yfcc10m | `official_fit`, `official_sel`, `official_cal`, `official_eval` (official query vectors, half public and half private) | 2,500 / 20,000 / 40,000 / 120,000 |

Category variants: `main` (Table I), and for Amazon and arXiv `kappa2` and `unconstrained` (`derive_category_stores.py`).
Workloads: `k10_kappa1` (main), `k10_kappa2`, `k10_unconstrained`, `k50_kappa1`.

## Amazon and arXiv

Kept items have a non-empty title and at least one category; of items with the same lower-cased title the first is kept
(Amazon: 1,587,421 raw items, 1,269,302 kept; arXiv: 3,224,635 and 3,218,786). Texts and categories are described in
`text_pool.py`. One permutation of the kept items selects, in this order, the 10,000 `fitpool` queries, 1,000 items that
are not used, the 1,000,000 base items and the 200,000 further queries; the items after them supply the index updates.

```
python data/prepare_items.py texts --ds amazon --raw RAW/amazon --prep PREP --data DATA
python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/texts_items.parquet \
    PREP/amazon/texts_queries_a.parquet PREP/amazon/texts_queries_b.parquet
python data/prepare_items.py install --ds amazon --prep PREP --data DATA
python data/derive_category_stores.py --data DATA --ds amazon            # kappa = 2 and unconstrained workloads
```
Title queries (Table II, and the query shift of Fig. 5):
```
python data/prepare_title_queries.py texts --set refit --ds amazon --prep PREP
python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/title_chunks/texts_*.parquet
python data/prepare_title_queries.py install --set refit --ds amazon --prep PREP --data DATA
python data/prepare_title_queries.py texts --set shift --ds amazon --prep PREP
python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/texts_titleshift.parquet
python data/prepare_title_queries.py install --set shift --ds amazon --prep PREP --data DATA
```
Index updates (Fig. 5): 50,000 items from the unused part of the pool, inserted into or replacing items of the base.
```
python data/prepare_update_items.py --ds amazon --raw RAW/amazon --prep PREP --data DATA
python data/embed_texts.py --model RAW/models/all-MiniLM-L6-v2 --texts PREP/amazon/texts_update_items.parquet
python data/prepare_updated_bases.py --ds amazon --prep PREP --data DATA
python pipeline/build_graphs.py update --data DATA --work WORK --ds amazon --from-graph A --updated amazon_replace1 --seed <level seed>
```
The same commands with `--ds arxiv --raw RAW/arxiv` prepare arXiv.

### Reproducibility of the embeddings

`embed_texts.py` runs the model with float16 autocast on a GPU and forms its batches after sorting the texts of one file by
length. The vector of a text therefore depends on the file it is embedded with (which is why the base, the two halves of
the 200,000 queries, each title file and the update items are separate calls) and on the device. On one device and
software stack the output is deterministic. Across GPU models it is not bit-identical: embedded on a second GPU model, the
arXiv query texts gave the same bits, while about 4% of the Amazon query vectors differed, by at most about 2e-4 in a
coordinate, which changed the exact answer of about 0.1% of the queries. A reproduction that embeds the texts again thus
works on vectors that agree with ours to this precision, and its statistics agree with the reported ones up to the
corresponding small differences rather than digit for digit. All later steps (exact answers, graphs, logs, calibration) are computed from the prepared vectors.

## YFCC-10M

```
python data/prepare_yfcc10m.py split --raw RAW/yfcc --prep PREP --data DATA
python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds yfcc10m --graph A --seed 94926219 --threads 32 \
    --base PREP/yfcc10m/base_pre.npy
python data/prepare_yfcc10m.py renumber --prep PREP --data DATA --graph-file WORK/graphs/yfcc10m_A.hnsw
python data/prepare_yfcc10m.py official --raw RAW/yfcc --prep PREP --data DATA
```
The 1,000 most frequent tags are the categories; the 1,072 images without any tag and those without one of these tags are
removed (9,993,795 kept), 202,500 are held out as queries and 9,791,295 form the base. Duplicate vectors are kept.
The graph is built with 32 threads, which assigns node ids in arrival order; `renumber` then renumbers the base so that
item id equals node id. A multi-threaded build is not deterministic, so a rebuilt graph, its renumbering and the logs
taken on it differ from ours in detail.

## YouTube-8M

```
python data/prepare_youtube8m.py fetch --plan RAW/video_train.json --raw RAW/youtube8m --n 1230
python data/prepare_youtube8m.py split --raw RAW/youtube8m --prep PREP --data DATA
python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds youtube8m --graph A --seed 1236488918 --threads 32 \
    --base PREP/youtube8m/base_pre.npy
python data/prepare_youtube8m.py renumber --prep PREP --data DATA --graph-file WORK/graphs/youtube8m_A.hnsw
```
`fetch` draws the shards by one random permutation of the plan entries and verifies every md5. The 1,230 shards hold
1,244,749 videos; the unit-normalized mean visual feature (1,024 dimensions) is the vector and the entity labels are the
categories. One permutation of the labelled videos gives 202,500 queries, 1,000,000 base items and leaves the rest unused.
The TFRecord files are read by the small parser inside `prepare_youtube8m.py`.

## Statistics of Table I

```
python data/dataset_stats.py table --data DATA --ds amazon arxiv yfcc10m youtube8m
python data/dataset_stats.py share --data DATA --ds amazon arxiv        # constrained answer differs from the unconstrained top 10
```
`table` prints the base size, the dimension, the number of categories that occur in the base, and the maximum and mean
number of categories per base item. For YFCC-10M and YouTube-8M the share of queries whose constrained answer differs
from the unconstrained one is the field `share_differs_from_unconstrained` in the summary of `pipeline/exact_answers.py`.

## After the preparation

```
python pipeline/build_graphs.py hnsw --data DATA --work WORK --ds amazon --graph A --seed 866159836       # graph B: --seed 744972238
python pipeline/exact_answers.py --data DATA --ds amazon --workload k10_kappa1 --groups fitpool chunk0 ... chunk19
python pipeline/log_trajectories.py --data DATA --work WORK --ds amazon --graph A --groups fitpool chunk0 ... chunk19 --threads 24
python pipeline/reduce_logs.py --work WORK --workload k10_kappa1 --ds amazon --graph A --groups fitpool chunk0 ... chunk19
```
Seeds are listed in `configs/seeds.json`. A single-threaded HNSW build is a deterministic function of the vectors, the
parameters, the seed and the compiled distance kernel; builds and logs taken with a different instruction set
(for example AVX2 instead of AVX-512) can differ in the last bits of float32 distances and so in a few trajectories.

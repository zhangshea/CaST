# How the random draws were made

Every random draw of the paper is made by a seeded generator. The code takes each seed as a parameter; the values
used for the reported results are the defaults below and in `configs/seeds.json`. All generators are
`numpy.random.default_rng(seed)` unless stated otherwise. Seeds written with the prefix `0x` are hexadecimal integers.

## Samples (paper Section III-D and Section VI-A, Datasets)

| Draw | Seed | Where |
|---|---|---|
| Permutation of the item pool of Amazon and of arXiv. It selects, in this order, the FIT pool of 10,000 queries, 1,000 items that the experiments do not use, the 1,000,000 base items and the 200,000 further queries | 0 | `data/text_pool.py` |
| FIT: the first 2,500 queries of a permutation of the FIT pool. The three further FIT samples of the paragraph "Near-optimality" are the next three blocks of 2,500 | 0x13528ee | `cast/setting.py` (`fit_permutation`, `--fit-block`) |
| SEL, CAL, EVAL, HOLD: one permutation of the 200,000 further queries, cut at 20,000, 60,000 and 180,000 | 0x13528ef | `cast/setting.py` (`pool_permutation`) |
| Ten further random splits of SEL, CAL and EVAL (FIT and HOLD fixed) | 1 to 10 | `pipeline/frontier.py --resplit-seed` |

The other workloads draw FIT, SEL, CAL and EVAL at data preparation ("parts" layout); their seeds are listed in
`configs/seeds.json` under the script that uses them.

## Stopping draws (paper Definition 2)

A randomized rule draws one uniform number per query. Calibration uses one fixed draw per CAL query, shared by all
candidate rules, and one per SEL query for the tests run on SEL.

| Draw | Seed | Where |
|---|---|---|
| One uniform per CAL query | 7 | `cast/setting.py` (`cal_uniforms`) |
| One uniform per SEL query | 8 | `cast/setting.py` (`sel_uniforms`) |

## Models

| Draw | Seed | Where |
|---|---|---|
| Score model (gradient-boosted classifier) | 0 | `cast/setting.py` (`score_model`) |
| Regressor of the LAET-style rule | 0 | `cast/compared.py` (`laet_regressor`) |

## Graphs (paper Section VI-A, Implementation details)

| Draw | Seed | Where |
|---|---|---|
| Insertion levels of HNSW graph A | 866159836 | `pipeline/build_graphs.py` |
| Insertion levels of HNSW graph B | 744972238 | `pipeline/build_graphs.py` |
| Insertion levels of the YFCC-10M and YouTube-8M graphs | 94926219, 1236488918 | `pipeline/build_graphs.py` |

Graphs A and B are built single-threaded, so the insertion order is the item order and the seed fixes the graph.
The YFCC-10M and YouTube-8M graphs are built with 32 threads, so their seed does not fix the graph.

## Bootstrap resamples

The paired bootstrap of a visit or latency ratio takes its seed as the argument `--seed`. For the reported runs the seed
was drawn once from the operating system's entropy source and stored with the result; the stored values are in
`configs/seeds.json`.

## Further draws

| Draw | Seed | Where |
|---|---|---|
| Title queries of the refit rows of Table II: split into FIT, SEL, CAL, EVAL | 1900014855 | `data/prepare_title_queries.py` |
| Index updates: new items, replaced slots, insertion levels | per data set, in `configs/seeds.json` | `data/prepare_update_items.py`, `data/prepare_updated_bases.py`, `pipeline/build_graphs.py update` |
| YFCC-10M and YouTube-8M: samples of the item queries, split of the official queries, choice of shards | in `configs/seeds.json` | `data/prepare_yfcc10m.py`, `data/prepare_youtube8m.py` |
| Richer learned policies: halves of CAL, cross-fitting folds, subsamples of SEL | 0x13528f0 to 0x13528f3 | `baselines/learned_policies.py` |
| Batches of the label-free drift signal | 0x13528f3 | `analysis/drift_signal.py` |
| Regressors of the DARTH and DARTH+ implementations, models of the learned policies | 0 | `baselines/` |
| Ten splits of the threshold-curve comparison | 1 to 10 | `baselines/threshold_curve.py --split-seed` |

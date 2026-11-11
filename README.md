# CaST: Calibrated Early Termination with Bounded Optimality Gaps for Constrained Vector Search

Code of the paper "Calibrated Early Termination with Bounded Optimality Gaps for Constrained Vector Search".

A graph index answers a constrained k-nearest-neighbor query by visiting items until it stops. CaST is a stopping rule
with two guarantees. Its answer is exact with probability at least 1 - eps, and its cost is within a computed factor
U / L of the best rule of a declared class of monotone rules. The package contains

- the search engine (C++), which maintains the exact constrained answer of the visited items and evaluates stopping rules online;
- the offline pipeline (Python): the declared interface, the chain of staircase rules computed by dynamic programming,
  its calibration, and the upper and lower bounds;
- our implementations of the compared stopping rules;
- the scripts that prepare the public data sets, and the scripts behind every table and figure of the paper.

No data set, model, graph or result file is included. `data/README.md` lists every data set with its original source
and how to obtain it.

## Layout

| Directory | Content | Paper |
|---|---|---|
| `cast/` | Offline library: checkpoint grid, samples and interface, monotone class as a linear program, chain dynamic program, fixed-sequence calibration, bounds U and L (float and exact) | Sections III to V |
| `engine/` | C++ search engine with Python bindings, graph build and update tools | Sections III-A, III-B, VI-A |
| `data/` | Preparation of the four public corpora and of the query sets; sources and download instructions | Section VI-A, Table I |
| `pipeline/` | Stages of the main pipeline: exact answers, graphs, trajectory logs, reduction, frontier rules and bounds, common-target comparison, held-out check | Sections VI-A and VI-B, Table II, Fig. 3 |
| `baselines/` | Compared stopping rules and their calibration | Section VI-A (Compared methods), Tables II and III, Fig. 4 |
| `timing/` | Online deployment bundle, latency and throughput measurements | Table III |
| `analysis/` | The further analyses of Section VI-B, one script each | Section VI-B |
| `figures/` | Scripts that draw Figs. 3 to 5 from the outputs above | Figs. 3 to 5 |
| `tables/table2.py`, `data/dataset_stats.py`, `timing/make_table3.py` | Rows of Tables II, I and III | Tables I to III |
| `reproduce/` | One shell script per table and figure, and the pipeline of one setting | |
| `configs/` | Risk levels and error levels of the frontier runs, recorded seeds | Section VI-A |
| `docs/` | `PAPER_MAP.md` (every paper element and the script that produces it), `RANDOM_DRAWS.md` (how the random draws were made), `techreport.pdf` (proofs and further experimental details) | |

## From the paper to the code

| Paper | Code |
|---|---|
| Definition 1, Lemma 1, Lemma 2: feasible answers, the greedy answer of the visited items, completion time | `engine/include/answer_queue.hpp`, `engine/include/search.hpp`, `pipeline/exact_answers.py` |
| Section III-B: checkpoints, cost, horizon | `cast/grid.py`, `pipeline/reduce_logs.py` |
| Section III-C: score, running minimum, buckets, flag; Definition 2 and Eq. (1): monotone rules | `cast/setting.py`, `cast/monotone_class.py` |
| Eq. (2) cost and risk; problem (3); Proposition 1 | `cast/rules.py` (`evaluate_rule`), `cast/monotone_class.py` (`class_coefs`, `solve_class_lp`) |
| Theorem 1: staircase rules | `cast/rules.py` (`staircase_table`, `staircase_F`) |
| Lemma 3, Eq. (4), Theorem 2, Algorithm 1: the nested chain | `cast/chain.py` (`chain_element`) |
| Theorem 3: fixed-sequence calibration | `cast/calibration.py` (`fixed_sequence`) |
| Theorem 4, Eq. (5): simultaneous lower bound | `cast/bounds.py` (`lagrangian_lower_bound`, `frontier_lower_bound`), `cast/exact.py` |
| Corollary 1: optimality-gap bound U / L | `pipeline/frontier.py`, `pipeline/exact_bounds.py` |

`docs/PAPER_MAP.md` continues this table for every table, figure and numbered statement of Section VI.

## Requirements

Three Python environments were used. Versions are pinned where the reported numbers depend on them.

| Purpose | Packages |
|---|---|
| Offline pipeline (`cast/`, `pipeline/frontier.py` and later stages, `baselines/`, `analysis/`), and `pipeline/reduce_logs.py` | `requirements/analysis.txt` |
| Engine, trajectory logging and timing | `requirements/engine.txt`, a C++17 compiler, the hnswlib headers of release v0.8.0 (not included; see `engine/README.md`) |
| Data preparation (text embedding, exact answers on a GPU, Vamana graph) | `requirements/data.txt` |

Two version effects are known. The seven features must be reduced with numpy 1.26, because the float64 logarithm of
numpy 2 differs in the last bit on some arguments. For the same reason `timing/export_bundle.py` and
`baselines/darth_plus.py compare`, which build the logarithm tables of the online features, run with numpy 1.26 (the
engine module works with both versions). The multipliers of the lower bound come from dual values of a
degenerate linear program, and scipy 1.17.1 returns a different value than 1.17.0 for one multiplier of one setting.

## Main pipeline for one setting

A setting is a data set searched on one graph. `DATA` holds the prepared data and `WORK` the graphs, logs and results.

```sh
# 1. engine, data, exact answers, graphs (details in engine/README.md and data/README.md)
bash engine/build.sh /path/to/hnswlib
python pipeline/exact_answers.py --help
python pipeline/build_graphs.py --help

# 2. logged traversals of all labeled queries, reduced to checkpoint arrays
python pipeline/log_trajectories.py --help
python pipeline/reduce_logs.py --help

# 3. interface, chain, frontier rules and bounds (Algorithm 1, Theorems 3 and 4, Corollary 1)
python pipeline/frontier.py --red $WORK/reduced/k10_kappa1 --ds arxiv --graph A --config configs/frontier.json --out $WORK/results/k10_kappa1/arxiv_A/frontier.json
python pipeline/exact_bounds.py --red $WORK/reduced/k10_kappa1 --ds arxiv --graph A --frontier $WORK/results/k10_kappa1/arxiv_A/frontier.json --out $WORK/results/k10_kappa1/arxiv_A/exact.json

# 4. every rule at the common risk target eps = delta = 0.05 (Table II)
python pipeline/common_target.py --red $WORK/reduced/k10_kappa1 --ds arxiv --graph A --frontier $WORK/results/k10_kappa1/arxiv_A/frontier.json --seed 832258306 --out $WORK/results/k10_kappa1/arxiv_A/common_target.json

# 5. held-out check (main workload)
python pipeline/holdout.py --red $WORK/reduced/k10_kappa1 --ds arxiv --graph A --frontier $WORK/results/k10_kappa1/arxiv_A/frontier.json --out $WORK/results/k10_kappa1/arxiv_A/holdout.json

# 6. table rows
python tables/table2.py "arXiv, A"=$WORK/results/k10_kappa1/arxiv_A
```

`reproduce/` holds one shell script per table and figure with the full sequence of commands.

## Reproducibility notes

- The probability statements of the paper hold for the samples drawn as in `docs/RANDOM_DRAWS.md`. With the same reduced
  arrays and the pinned versions, the offline pipeline is deterministic.
- The mixture sequence consists of optimal solutions of linear programs, which need not be unique, so its rules can
  differ slightly with another solver version. The chain, which the paper reports, does not depend on the solver.
- The logged traversals depend on the last bits of the computed distances, hence on the instruction set the distance
  kernel is compiled for. A different instruction set changes individual trajectories slightly.
- The text embeddings are computed on a GPU in half precision and are not bit-identical across devices. On newly computed
  Amazon embeddings a few percent of the query vectors differ slightly, which moves the bounds at the two smallest risk
  levels in the third decimal. `data/README.md` gives the details.
- Graphs built with several threads (YFCC-10M, YouTube-8M, Vamana) depend on thread scheduling. Graphs A and B of
  Amazon and arXiv are built single-threaded from recorded seeds.
- Latency depends on the machine. Ratios are formed between rules timed in the same session (`timing/README.md`).

## License

The code is released under the MIT License (`LICENSE`). `THIRD_PARTY_NOTICES.md` lists the one function derived from
third-party source code and the libraries, data sets and models that the package uses but does not include.

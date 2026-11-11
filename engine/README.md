# Search engine (`cast_engine`)

The C++ search engine of the paper, with Python bindings. It runs a best-first traversal of a proximity graph
with a candidate list of `L` entries and maintains, exactly and at every visit, the greedy constrained answer
`Greedy(A_t)` of the visited items (Sec. III-A). On top of this traversal it provides

- `log_search`: the trajectory log from which completion times, checkpoints and features are reduced offline;
- `stop_search`: the timed online search with a stopping rule (ours and every compared rule of Sec. VI-A);
- `search`: the search with a list of `L` entries followed by `Greedy` over all visited items (the rule
  "List size"), and the plain top-k;
- the category store, graph loaders, an independent feasibility check, and two command-line tools that build
  and update the HNSW graphs.

Items, categories, capacities `kappa`, answers and distances are those of Sec. III-A: the distance is
`d(q, x) = 1 - <q, x>` on unit vectors, ties are broken by item id.

## Layout

| file | content |
|---|---|
| `include/answer_queue.hpp` | answer queue: `Greedy` of the offered items, maintained for offers in any order |
| `include/category_store.hpp` | category lists `c(v)` in compressed sparse row form and the capacities |
| `include/graph.hpp` | graph storage; loaders for an hnswlib index file and for exported arrays (Vamana) |
| `include/search.hpp` | the traversal: `log_run`, `stop_run` (stopping rules, features, forest evaluation), list search |
| `include/verify.hpp` | independent feasibility check of an answer (Definition 1) |
| `include/npy.hpp` | `.npy` reader of the command-line tools |
| `bindings.cpp` | pybind11 module `cast_engine` |
| `tools/build_hnsw.cpp` | HNSW graph construction with seeded insertion levels |
| `tools/update_hnsw.cpp` | inserts and in-place replacements on an existing index |
| `tests/test_answer_queue.cpp` | differential test of the answer queue against `Greedy` computed from scratch |
| `tests/test_properties.py` | Lemma 1, Lemma 2 and the feasibility check on small random instances |
| `build.sh` | builds everything |

## Build

Requirements: a C++17 compiler, Python 3 with `numpy` and `pybind11`, and the headers of
[hnswlib](https://github.com/nmslib/hnswlib) release **v0.8.0** (commit `3f34296`). hnswlib is not part of this
package. Its headers are used unmodified, for the distance kernel, the index file format and the two tools:

```sh
git clone --depth 1 --branch v0.8.0 https://github.com/nmslib/hnswlib.git third_party/hnswlib
./build.sh third_party/hnswlib            # or: HNSWLIB_DIR=third_party/hnswlib ./build.sh
```

`build.sh` accepts the directory that holds `hnswlib.h` or the root of the checkout. It compiles every binary
with the same flags, by default `-std=c++17 -O3 -march=native -DNDEBUG -pthread` (no fast-math), and writes
`bin/build_hnsw`, `bin/update_hnsw`, `bin/test_answer_queue` and the module `cast_engine<suffix>.so` into
`build/`, or into `$BUILD_DIR`. Variables: `PYTHON` (default `python3`), `CXX` (default `g++`), `FLAGS`,
`BUILD_DIR`. Add the build directory to `sys.path` to import the module.

Tests:

```sh
build/bin/test_answer_queue [traces] [seed]     # default 1000000 traces; exit code 0 iff no mismatch
python tests/test_properties.py [--quick] [--seed S] [--module-dir DIR]
```

## Command-line tools

```
build_hnsw  --base emb_base.npy --out index.hnsw [--info info.json] [--M 32] [--efc 200] [--seed S] [--threads 1]
update_hnsw --in A.hnsw --out B.hnsw --vec items.npy --mode insert|replace [--slots slots.npy] --seed S [--info info.json]
```

`build_hnsw` builds an HNSW graph in the inner-product space over the rows of a float32 array `[n, d]`
(Sec. VI-A: degree parameter `M = 32`, construction list `efc = 200`; `seed` seeds the random insertion levels,
and the two seeds of graphs A and B are recorded in `configs/seeds.json`). With one thread the rows are inserted
in id order, so node id = row id and the graph is a deterministic function of the data, `M`, `efc` and `seed`.
With several threads the node ids depend on the schedule: the tool prints the number of label mismatches, exits
with code 1, and the caller renumbers the items by node id before loading the graph. The info file holds
`n_nodes`, `n_base`, `dim`, `M`, `efc`, `seed`, `threads`, `load_s`, `build_s`, `build_cpu_s`, `peak_rss_kb`,
`maxlevel`, `entry`, `size_data_per_element`, `level0_bytes`, `upper_bytes`, `n_upper_nodes`, `label_mismatch`
and `mean_degree0`.

`update_hnsw` applies the index updates of "Query shift and index updates" (Sec. VI-B) with hnswlib's public
calls, on one thread. `insert` appends the rows of `items.npy` with labels `n, n + 1, ...` after seeding the
level generator with `S`. `replace` writes row `i` of `items.npy` into the existing slot `slots[i]` (int64),
which hnswlib performs as an in-place update that repairs the neighborhood. The info file holds `mode`,
`n_before`, `m`, `n_after`, `dim`, `seed`, `secs`, `label_mismatch`, `marked_deleted`, `mean_degree0`,
`maxlevel` and `entry`.

## Python API

Arrays are NumPy arrays; `nq` is the number of queries, `k` the answer size, `J` the number of checkpoints.
With `threads = t`, query `i` is handled by thread `i mod t`, and every thread owns its searcher. Timed runs
use one thread.

### Graph and categories

- `load_hnswlib(path, dim) -> Graph`: parses an hnswlib v0.8.0 index file. Node id must equal the label.
- `graph_from_arrays(X, nbr0, level, upper_ptr, upper, maxM, entry, maxlevel) -> Graph` (positional): `X`
  float32 `[n, dim]`; `nbr0` int32 `[n, maxM0]` bottom-layer neighbors, `-1` padded; `level` int32 `[n]` top
  layer of each node; `upper_ptr` int64 `[n]` offset of the node's upper lists in `upper`, or `-1`; `upper`
  uint32, per node and layer `[count, maxM ids]`; `entry` the entry node; `maxlevel` the top layer. A Vamana
  graph is loaded with `level = 0`, `upper_ptr = -1`, an empty `upper`, `maxM = 0`, `maxlevel = 0` and its
  start node as `entry`.
- `Graph`: attributes `n`, `dim`, `maxM0`, `maxM`, `maxlevel`, `entry`; method `mean_degree0()` (mean
  out-degree on the bottom layer).
- `Categories(ptr, idx, n_categories, kappa)`: `ptr` int64 `[n + 1]` and `idx` int32, the category lists
  `c(v) = idx[ptr[v]:ptr[v + 1]]`; `kappa` int32 `[n_categories]`, the capacities. Attributes `n`,
  `n_categories`. Every item must carry at least one category. Unconstrained k-NN is one shared category
  with capacity `k`.
- `rank_capped(categories, cap) -> int`: the size of the largest feasible set of items, truncated at `cap`
  (Sec. III-A assumes that `k` does not exceed it).
- `verify(ids, categories) -> int8 [nq]`: independent feasibility check of answers `ids` int64 `[nq, k]`
  (negative = empty slot): `1` feasible with `k` items, `2` feasible with fewer, `0` infeasible or duplicate.
- `build_flags() -> str`: compiler version and the instruction-set and fast-math macros of the build.

### `log_search(graph, categories, Q, S, params, Lmax, grid, threads=1) -> dict`

The trajectory log of the traversal with a list of `Lmax` entries (never timed). `Q` float32 `[nq, dim]`;
`S` int64 `[nq, k]`, the exact answers; `params = dict(method='answer', k=k, L=Lmax)`; `grid` int32 `[G]`,
list sizes. One row is recorded per expansion, before the expansion happens. The rows of query `i` are
`ptr[i]:ptr[i + 1]`.

| key | type | content |
|---|---|---|
| `ptr` | int64 `[nq + 1]` | row offsets |
| `visits` | int32 per row | number of visits so far |
| `dx` | float32 per row | distance of the candidate to expand (the closest unexpanded candidate) |
| `dbk` | float32 per row | distance of the last answer item, `inf` while the answer holds fewer than `k` items |
| `dpk` | float32 per row | `k`-th distance of the unconstrained top `k` of the visited items, `inf` before `k` visits |
| `answer_size` | int32 per row | number of answer items |
| `chg` | int32 per row | number of visits at the last change of the answer |
| `visit_of_member` | int32 `[nq, k]` | the visit at which each member of `S` was visited, `0` if it was not |
| `list_stop` | int32 `[nq, G]` | number of visits at which a search with a list of `grid[g]` entries ends |
| `final_visits` | int32 `[nq]` | number of visits at the end of the search |
| `ndesc` | int32 `[nq]` | distance computations of the descent through the upper layers, entry point included |

### `stop_search(graph, categories, Q, params, Lmax, rule, clf=None, reg=None, threads=1, dump=False, instr=False, S=None) -> dict`

The timed online search: the traversal of `log_search` with the stopping rule `rule`.
`params = dict(method='answer', k=k, L=Lmax)`. `clf` is the flattened score model, `reg` the flattened
regressor of the rule (LAET-style, DARTH, DARTH+). `dump=True`, with a rule of type 0 to 5, never stops and records the
features, the score and the flag at every checkpoint. `instr=True` also measures the time spent evaluating the rule. `S` (exact
answers) is needed by rule type 7 only.

| key | type | content |
|---|---|---|
| `ids` | int64 `[nq, k]` | the returned answer in (distance, id) order, `-1` padded |
| `ns` | int64 `[nq]` | time of the query in nanoseconds (`steady_clock`, measured inside the engine) |
| `hook_ns` | int64 `[nq]` | time spent evaluating the rule, `0` unless `instr` |
| `visits` | int32 `[nq]` | number of visits when the search stopped (the cost `c_T` of Eq. (2)) |
| `stop_j` | int32 `[nq]` | checkpoint at which the rule stopped; `0` for a stop of types 6 to 10; `-1` if the search ended by itself |
| `calls` | int32 `[nq]` | DARTH and DARTH+: prediction points reached |
| `qcalls` | int32 `[nq]` | type 8 only: evaluations of the lower-quantile model |
| `drows`, `dptr` | float64 `[rows, 12 or 20]`, int64 `[nq + 1]` | type 7 only: the DARTH features (11, or 19 with `qstats`) and the recall of the answer, per logged point |
| `features` | float64 `[nq, J, 7]` | `dump` only: the features at every checkpoint the search reached (others stay `0`) |
| `score` | float64 `[nq, J]` | `dump` only: the score `s_j` |
| `full` | uint8 `[nq, J]` | `dump` only: the flag `a_j` |

`rule` is a dict with `type`, `grid` (int32 `[J]`, the checkpoint grid `g_0 < ... < g_{J-1}`) and the keys of
its type. Types 0 to 4 act at the checkpoints and stop at the last checkpoint at the latest; types 6 to 10
act between checkpoints.

| type | rule | keys |
|---|---|---|
| 0 | no early stop. With `grid = [2**30]` the search runs to its own end, which is the rule "List size" at list size `Lmax` | |
| 1 | staircase, our rule: stop iff `bucket(m_j) <= beta[j][a_j]` | `edges` (float64 `[B - 1]`), `beta` (int32 `[(J - 1) * 2]`, row `j`, column `a`), `log_tab`, `log1p_tab` |
| 2 | Level and Level+cap: stop iff `m_j < lam` or `j >= jcap` | `lam`, `jcap`, `log_tab`, `log1p_tab` |
| 3 | Fixed: stop at checkpoint `jfix` | `jfix` |
| 4 | LAET-style: at checkpoint `j0` predict `X^ = exp(reg(features, flag))`, stop at the first checkpoint `j >= j0` with `visits >= mult * X^` | `j0`, `mult`, `log_tab`, `log1p_tab` |
| 6 | DARTH: every `dpi` distance computations predict the recall `r = clip(reg(features), 0, 1)` once the answer is full; stop if `r >= Rt`, else `dpi = mpi + (ipi - mpi)(Rt - r)` | `Rt`, `ipi`, `mpi` |
| 7 | DARTH training log: never stop; every `li`-th distance computation with a full answer, record the features and the recall against `S` | `li`, optional `qstats = 1` (adds the eight query statistics) |
| 8 | DARTH+: as type 6 with 19 features until `r` first reaches `Rt`, then only the lower-quantile model `qforest`; stop when its prediction reaches `Rt` | `Rt`, `ipi`, `mpi`, `qforest` |
| 9 | Patience: stop at the first expansion at which the answer is full and unchanged for `pdelta` visits, or with `visits >= hvis` | `pdelta`, `hvis` |
| 10 | Distance-adaptive: stop at the first expansion at which the answer is full and `dx > (1 + gamma) * dbk` | `gamma` |

`log_tab[n]` and `log1p_tab[n]` are NumPy's float64 `log(n)` and `log1p(n)` for the integers `n = 0, 1, ...`;
arguments beyond the tables use `std::log` and `std::log1p`. Rule types 6 to 8 need `k <= 64`.

A forest (`clf`, `reg`, `qforest`, and the argument of `forest_raw`) is a dict of node arrays: `root` (int32,
root node of each tree), `left`, `right` (int32 child nodes), `feat` (int16), `thr` (float64), `val` (float64),
`leaf` (uint8), `miss_left` (uint8) and `base` (float). Its raw output is `base` plus the leaf values; a node
sends `x` to the left iff `x[feat] <= thr`, and a missing value (NaN) to the left iff `miss_left`.

- `forest_raw(forest, X) -> float64 [n]`: raw outputs on the rows of `X` (float64).
- `npy_logf_vec(x) -> float32`: the float32 logarithm used by the online features (see "Score model and interface").

### `search(graph, categories, Q, params, Ls=None, threads=1) -> dict`

`params = dict(method='list', k=k, L=L)`: search with a list of `L` entries, then `Greedy` over all visited
items (the native run of the rule "List size", used for its calibrated list size).
`params = dict(method='plain', k=k, L=L)`: the top `k` of the list, without the categories. `Ls` (int32
`[nq]`) gives one list size per query. Returns `ids` (int64 `[nq, k]`, `-1` padded, in (distance, id) order) and
`visits` (int64 `[nq]`). Method `'answer'` runs through `log_search` and `stop_search`.

## Paper elements

**Answer maintenance (Sec. III-A, Definition 1).** `AnswerQueue::offer` keeps the answer together with one
assignment `phi` of answer items to category slots. An item whose category has a free slot enters directly.
Otherwise an alternating-path search between answer items and category slots decides it: if a path reaches a
category with a free slot, the items along the path are reassigned and the item enters; if not, the item enters
only if it precedes the last answer item on such paths, which it then replaces. After every offer the queue
equals `Greedy` of the offered items, which `tests/test_answer_queue.cpp` checks against `Greedy` computed from
scratch. Its cost is inside every timed query.

**Lemma 1.** If the visited set contains the exact answer `S`, the returned answer is `S`. A stop after `t`
visits is therefore exact iff `t >= X`, the completion time. `log_search` records `visit_of_member`, and
`X = max_j visit_of_member[j]`, infinite if a member was not visited.

**Lemma 2.** A visited member of `S` is never displaced, so the recall of the returned answer is the fraction
of `S` visited. The completion time of a recall target `rho` is the `ceil(rho k)`-th smallest
`visit_of_member`, and rule type 7 logs the recall of the current answer.

**Checkpoints and cost (Sec. III-B).** An expansion removes the closest unexpanded candidate and visits its
unvisited neighbors. Checkpoint `j` is the first expansion at which the number of visits reaches `grid[j]`,
and `c_j` is the number of visits at that moment. `stop_search` evaluates the rule there, before the expansion;
several checkpoints reached at one expansion are evaluated in order on the same state. Offline, `c_j` is the
first logged `visits >= grid[j]` of the query, or `final_visits` if the search ended before reaching it. The search ends by
itself when the closest unexpanded candidate is farther than all `Lmax` kept entries.

**Score model and interface (Sec. III-C, Sec. VI-A).** The seven features at a checkpoint, in order: `log(visits)`,
`log(max(dx, 1e-9))`, `log(dbk / dx)`, `log(dpk / dx)`, the answer size, `log1p(visits - chg)` and
`(visits - chg) / visits`, with `dx`, `dbk`, `dpk` and `chg` as in the log. The second to fourth are
float32 logarithms, and an infinite ratio (the answer or the top `k` is not full) is a missing value. The engine
computes them with a scalar form of the float32 logarithm of NumPy 1.26 on x86-64 (its AVX512F and FMA3 code
paths), so that the online features equal, bit for bit, the features reduced offline from the logs with that
NumPy build. With another NumPy version or processor, `npy_logf_vec(x) == numpy.log(x)` measures the agreement. The score is
`s_j = 1 / (1 + exp(raw_j))` with `raw_j` the raw output of the classifier, `m_j` is its running minimum, the
bucket is `b_j = |{i : edges[i] <= m_j}|`, and the flag `a_j` says whether the answer holds `k` items. The
LAET-style regressor reads the seven features and the flag.

**Compared rules (Sec. VI-A).** Level is type 2 with `jcap = J - 1`, Level+cap type 2, LAET-style type 4, Fixed
type 3, DARTH type 6 (trained from the log of type 7), DARTH+ type 8, List size type 0 with a list of the
calibrated size, Distance-adaptive type 10 and Patience type 9.

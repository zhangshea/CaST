# Latency experiments

Scripts behind Table III and the latency sentences of Section VI-B. They run the calibrated rules online in the search
engine (`cast_engine`) and compare each run with its offline replay.

| Script | Content | Paper |
|---|---|---|
| `export_bundle.py` | Timing bundle of one setting: flattened score model and LAET-style regressor, bucket edges, checkpoint grid, calibrated rule records, timing queries with exact answers, offline replay of every rule | Table III; Section VI-A, Implementation details |
| `time_rules.py` | One timing window (`window`), share of the stopping check (`share`), online agreement checks (`check-replay`, `check-visits`, `check-curve`) | Table III; Sections VI-A and VI-B |
| `make_table3.py` | Rows of Table III and the readings quoted with it | Table III |
| `throughput.py` | Throughput and percentiles at 1 to 32 threads | Section VI-B, "Cost and latency at a common risk target" |
| `short_list.py` | The staircase on a shorter candidate list: calibration (`calibrate`) and latency (`time`) | Section VI-B, "Workload breadth" |
| `failed_answers.py` | Recall and total distance of the answers that differ from the exact one | Section VI-B, "Held-out check" |
| `engine_rules.py`, `protocol.py` | Rule records passed to the engine; passes, medians and bootstrap statistics | |

## Measurement protocol

- One thread on one pinned core of the machine, with the hyperthread sibling of that core idle and no other load on
  the machine. The scripts do not pin themselves; start them under the pinning tool of the system, for example
  `OMP_NUM_THREADS=1 taskset -c CORE python timing/time_rules.py window ...`.
- The time of a query is taken inside the engine, around the whole search of that query: graph traversal, answer
  maintenance, the seven features, the score model and the stopping check are included.
- Before timing, every rule of the bundle named by the window is run with 8 threads and its visits are compared with
  the offline replay. Then every rule is warmed up on the first 1,000 queries.
- Five passes over all timing queries. The order of the rules rotates by one position per pass.
- The latency of a query under a rule is the median of its five times. A window reports the mean of these medians.
- Ratios are formed only between rules of one window (one session of one process), because the mean latency of a rule
  moves between sessions.
- A rule is significantly slower than ours when the 2.5% quantile of the paired bootstrap of the ratio of means
  (query indices resampled, the same indices for all rules) exceeds one. `--family` adds the one-sided bound at level
  0.025 divided by the number of ratios of the family (Bonferroni).

## Bundles

```sh
# HOLD queries of a main setting (Table III)
python timing/export_bundle.py --red $WORK/reduced/k10_kappa1 --data $DATA --ds amazon --graph A \
    --frontier $WORK/results/k10_kappa1/amazon_A/frontier.json --queries hold --out $WORK/timing/amazon_A_hold
# first 10,000 EVAL queries (agreement checks; latency on YFCC-10M with --layout parts)
python timing/export_bundle.py --red $WORK/reduced/k10_kappa1 --data $DATA --ds amazon --graph A \
    --frontier $WORK/results/k10_kappa1/amazon_A/frontier.json --queries eval --out $WORK/timing/amazon_A_eval
```

## Timing windows of Table III

Each window is one run per dataset on the HOLD bundle. The rules of the compared methods take their calibrated
parameters from the outputs of `baselines/`: the list size (`--list-size`), the number of visits of Patience
(`--patience`), gamma of the distance-adaptive rule (`--gamma`), the DARTH and DARTH+ models and recall targets, and the
parameters of the threshold curve (`--curve`, the output of `baselines/threshold_curve.py fit`).

```sh
B=$WORK/timing/amazon_A_hold; E="--bundle $B --data $DATA --work $WORK"
python timing/time_rules.py window --window 1 $E --out $WORK/timing/window1 \
    --darth-model $WORK/models/darth_amazon_A --darth-calibration $WORK/results/darth_amazon_A.json
python timing/time_rules.py window --window 2 $E --out $WORK/timing/window2 --list-size LIST_SIZE
python timing/time_rules.py window --window 3 $E --out $WORK/timing/window3 --seed SEED \
    --list-size LIST_SIZE --patience VISITS --gamma GAMMA \
    --darth-model $WORK/models/darth_amazon_A --darth-calibration $WORK/results/darth_amazon_A.json \
    --darth-plus-model $WORK/models/darthplus_amazon_A --darth-plus-levels $WORK/results/darthplus_levels_amazon_A.json
python timing/time_rules.py window --window 4 $E --out $WORK/timing/window4 --seed SEED --curve CURVE.json
python timing/time_rules.py share --rule cast $E --out $WORK/timing/window3/amazon_share_cast.json
python timing/make_table3.py --window1 $WORK/timing/window1 --window2 $WORK/timing/window2 \
    --window3 $WORK/timing/window3 --window4 $WORK/timing/window4
```

Windows 1 and 2 use the bootstrap seed 0 with 2,000 resamples. Windows 3 and 4 use 20,000 resamples and take the seed
as `--seed`; the seeds of the reported runs are in `configs/seeds.json`. Window 3 tests twelve ratios (six rules, two
datasets) and window 4 four.

## Agreement checks

```sh
python timing/time_rules.py check-replay --bundle $WORK/timing/amazon_A_eval --data $DATA --work $WORK --out check_replay.json
python timing/time_rules.py check-visits --bundle $WORK/timing/amazon_A_eval --data $DATA --work $WORK \
    --patience VISITS --gamma GAMMA --out check_visits.json
python timing/time_rules.py check-curve --bundle $WORK/timing/amazon_A_hold --data $DATA --work $WORK --curve CURVE.json --out check_curve.json
```

`check-replay` compares the online features with the logged ones at every reached checkpoint of the first 1,000 timing
queries, and the stopping point and exactness of every rule of the bundle with the offline replay on all of them.

## Throughput, failed answers, shorter lists

```sh
python timing/throughput.py $E --out throughput.json [rule parameters as for window 3]
python timing/failed_answers.py $E --out failed_answers.json [rule parameters as for window 3]

# YFCC-10M: calibrate the staircase on shorter lists, export for the timing queries, time
python timing/short_list.py calibrate --red $WORK/reduced/k10_kappa1 --ds yfcc10m --graph A --layout parts \
    --frontier FRONTIER.json --seed SEED --export-eval --out $WORK/timing/yfcc10m_lists
python timing/short_list.py time --bundle $WORK/timing/yfcc10m_A_eval --export $WORK/timing/yfcc10m_lists_export.npz \
    --list-size LIST_SIZE --seed SEED --data $DATA --work $WORK --out $WORK/timing/yfcc10m_short
# main workload, HOLD queries
python timing/short_list.py calibrate --red $WORK/reduced/k10_kappa1 --ds amazon --graph A --seed SEED --export-hold \
    --out $WORK/timing/amazon_lists
python timing/short_list.py time --bundle $B --export $WORK/timing/amazon_lists_hold_export.npz --lists 2048 1024 \
    --bundle-rules level_cap --list-size LIST_SIZE --seed SEED --max-differ 5 --require-same-staircase \
    --data $DATA --work $WORK --out $WORK/timing/amazon_short
```

The staircase of the search with a list of 2,048 entries (chain fitted on that search) uses the reduced arrays written
by `pipeline/truncate_list.py`: pass them to `calibrate` as `--red`, the directory of the full-list arrays (which holds
the `_stops.npz` files) as `--stops`, and as `--frontier` the frontier output computed on the truncated arrays. Its
full-list staircase then differs from that of the bundle, so `time` runs without `--require-same-staircase`.

## Parameters of the reported runs

Calibrated on CAL at eps = delta = 0.05 by the scripts of `baselines/` (Section VI-A).

| Parameter | Amazon | arXiv | Used by |
|---|---|---|---|
| List size, logged grid of 46 sizes | 662 | 319 | window 2; `short_list.py time` on HOLD |
| List size, finer grid | 627 | 314 | window 3, `throughput.py`, `failed_answers.py` |
| Patience, visits | 4,429 | 3,660 | window 3 |
| Distance-adaptive, gamma | 0.1475 | 0.1075 | window 3 |
| DARTH, recall target | 0.9942 | 0.9939 | windows 1 and 3 |
| DARTH+, quantile level and recall target | 0.05, 0.9462 | 0.2, 0.9937 | window 3 |
| Threshold curve (s, j0, u, cap, c) | 0.25, 32, 0, none, 28.0 | 0.25, 64, 0, none, 36.5 | window 4 |

On YFCC-10M the calibrated list size is 1,145 (`short_list.py time`).

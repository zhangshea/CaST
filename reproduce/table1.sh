#!/bin/sh
# Table I (dataset statistics: base items, dimension, categories with at least one base item, maximum and mean number
# of categories per base item) and the share of main-workload queries whose constrained answer differs from the
# unconstrained top 10 (Section VI-A, Datasets).
# Before: the scripts of data/ for the four datasets (data/README.md); for the share, the exact answers of the main
# workload (reproduce/main_setting.sh, its exact-answer step).
set -eu

DATA=${DATA:-data}
WORK=${WORK:-work}
PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
POOL_SEED=0x13528ef               # recorded value: configs/seeds.json
mkdir -p "$WORK/results"

$PY "$ROOT/data/dataset_stats.py" table --data "$DATA" --ds amazon arxiv yfcc10m youtube8m --out "$WORK/results/table1.json"
$PY "$ROOT/data/dataset_stats.py" share --data "$DATA" --ds amazon arxiv --workload k10_kappa1 --n 10000 \
    --pool-seed "$POOL_SEED" --out "$WORK/results/table1_share.json"

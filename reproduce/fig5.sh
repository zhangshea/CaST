#!/bin/sh
# Fig. 5 (failure rate over the target on EVAL of graph A after a query shift or an index update, before and after
# recalibration on CAL queries drawn after the change); then the counts of the paragraph "Query shift and index
# updates" that come from the same readouts (levels rejected before recalibration, levels without a recalibrated rule).
# Before: reproduce/main_setting.sh for amazon and arxiv, graph A; the scripts of data/ for the title queries
# (data/prepare_title_queries.py --set shift) and the updated bases (data/prepare_update_items.py,
# data/prepare_updated_bases.py).
set -eu

DATA=${DATA:-data}
WORK=${WORK:-work}
THREADS=${THREADS:-24}
PY=${PY:-python3}                 # environment of requirements/analysis.txt
PY_ENGINE=${PY_ENGINE:-python3}   # environment of requirements/engine.txt
PY_DATA=${PY_DATA:-python3}       # environment of requirements/data.txt
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engine/build}
LEVEL_SEED_AMAZON=1075205510      # recorded value: configs/seeds.json (insertion levels of the updates)
LEVEL_SEED_ARXIV=2049732898       # recorded value: configs/seeds.json
WL=k10_kappa1
RED=$WORK/reduced/$WL
OUT=$WORK/results/shift
POOL="fitpool chunk0 chunk1 chunk2 chunk3 chunk4 chunk5 chunk6 chunk7 chunk8 chunk9 chunk10 chunk11 chunk12 chunk13 chunk14 chunk15 chunk16 chunk17 chunk18 chunk19"
TITLE="titleshift_cal titleshift_eval"
mkdir -p "$OUT"

for ds in amazon arxiv; do
    FR=$WORK/results/$WL/${ds}_A/frontier.json
    if [ $ds = amazon ]; then LSEED=$LEVEL_SEED_AMAZON; else LSEED=$LEVEL_SEED_ARXIV; fi

    # title queries of the CAL items and of the first 40,000 EVAL items, searched on graph A
    $PY_DATA "$ROOT/pipeline/exact_answers.py" --data "$DATA" --ds $ds --workload $WL --groups $TITLE
    $PY_ENGINE "$ROOT/pipeline/log_trajectories.py" --data "$DATA" --work "$WORK" --ds $ds --graph A --workload $WL \
        --groups $TITLE --threads "$THREADS" --engine "$ENGINE"
    $PY "$ROOT/pipeline/reduce_logs.py" --work "$WORK" --workload $WL --ds $ds --graph A --groups $TITLE
    $PY "$ROOT/analysis/shift_title.py" --red "$RED" --ds $ds --graph A --frontier "$FR" --shift-red "$RED" \
        --out "$OUT/title_$ds.json"

    # 1% or 5% new items inserted, or as many base items replaced in place, in a copy of graph A
    for u in insert1 insert5 replace1 replace5; do
        $PY_ENGINE "$ROOT/pipeline/build_graphs.py" update --data "$DATA" --work "$WORK" --ds $ds --from-graph A \
            --updated ${ds}_$u --seed "$LSEED" --engine "$ENGINE"
        $PY_DATA "$ROOT/pipeline/exact_answers.py" --data "$DATA" --ds ${ds}_$u --workload $WL --groups $POOL
        $PY_ENGINE "$ROOT/pipeline/log_trajectories.py" --data "$DATA" --work "$WORK" --ds ${ds}_$u --graph A --workload $WL \
            --groups $POOL --threads "$THREADS" --engine "$ENGINE"
        $PY "$ROOT/pipeline/reduce_logs.py" --work "$WORK" --workload $WL --ds ${ds}_$u --graph A --groups $POOL
        $PY "$ROOT/analysis/shift_updates.py" --red "$RED" --ds $ds --graph A --frontier "$FR" --shift-red "$RED" \
            --shift-ds ${ds}_$u --out "$OUT/${u}_$ds.json"
    done
done

$PY "$ROOT/figures/fig5_shift.py" \
    --title-amazon "$OUT/title_amazon.json" --replace5-amazon "$OUT/replace5_amazon.json" \
    --replace1-amazon "$OUT/replace1_amazon.json" --insert5-amazon "$OUT/insert5_amazon.json" \
    --insert1-amazon "$OUT/insert1_amazon.json" \
    --title-arxiv "$OUT/title_arxiv.json" --replace5-arxiv "$OUT/replace5_arxiv.json" \
    --replace1-arxiv "$OUT/replace1_arxiv.json" --insert5-arxiv "$OUT/insert5_arxiv.json" \
    --insert1-arxiv "$OUT/insert1_arxiv.json" \
    --out "$WORK/results/shift.pdf"

ARGS=""
for ds in amazon arxiv; do
    for c in title insert1 insert5 replace1 replace5; do ARGS="$ARGS ${c}_$ds=$OUT/${c}_$ds.json"; done
done
$PY "$ROOT/analysis/shift_summary.py" $ARGS --out "$OUT/summary.json"

#!/bin/sh
# One setting of the main workload (k = 10, capacity 1) from the prepared data to the frontier rules and their bounds
# (Algorithm 1, Theorems 3 and 4, Corollary 1), the exact bounds U / L, the common-target table (eps = delta = 0.05)
# and the held-out check. Serves Table II (rows 1 to 4), Fig. 3 and the paragraph "Held-out check" of Section VI-B.
# Before: the scripts of data/ for DS (data/README.md) and engine/build.sh.
# Run it four times: DS in {amazon, arxiv}, GRAPH in {A, B}, with GRAPH_SEED and CT_SEED of that setting.
set -eu

DATA=${DATA:-data}                # prepared data (DATA/<ds>/base.npy, categories_main.npz, queries/)
WORK=${WORK:-work}                # graphs, logs, reduced arrays, results
DS=${DS:-arxiv}
GRAPH=${GRAPH:-A}
GRAPH_SEED=${GRAPH_SEED:-866159836}   # recorded value: configs/seeds.json
CT_SEED=${CT_SEED:-832258306}         # recorded value: configs/seeds.json
THREADS=${THREADS:-24}
PY=${PY:-python3}                 # environment of requirements/analysis.txt (numpy 1.26 for the reduction)
PY_ENGINE=${PY_ENGINE:-python3}   # environment of requirements/engine.txt
PY_DATA=${PY_DATA:-python3}       # environment of requirements/data.txt
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engine/build}

WL=k10_kappa1
QGROUPS="fitpool chunk0 chunk1 chunk2 chunk3 chunk4 chunk5 chunk6 chunk7 chunk8 chunk9 chunk10 chunk11 chunk12 chunk13 chunk14 chunk15 chunk16 chunk17 chunk18 chunk19"
RED=$WORK/reduced/$WL
OUT=$WORK/results/$WL/${DS}_$GRAPH
mkdir -p "$OUT"

# exact constrained answers by a full scan (one run per dataset; graphs A and B share them)
$PY_DATA "$ROOT/pipeline/exact_answers.py" --data "$DATA" --ds "$DS" --workload $WL --groups $QGROUPS \
    --summary "$WORK/results/$WL/${DS}_answers.json"

# HNSW graph, single-threaded from the recorded level seed
$PY_ENGINE "$ROOT/pipeline/build_graphs.py" hnsw --data "$DATA" --work "$WORK" --ds "$DS" --graph "$GRAPH" \
    --seed "$GRAPH_SEED" --engine "$ENGINE"

# logged traversals of all labeled queries, reduced to checkpoint arrays and stop times
$PY_ENGINE "$ROOT/pipeline/log_trajectories.py" --data "$DATA" --work "$WORK" --ds "$DS" --graph "$GRAPH" \
    --workload $WL --groups $QGROUPS --threads "$THREADS" --engine "$ENGINE"
$PY "$ROOT/pipeline/reduce_logs.py" --work "$WORK" --workload $WL --ds "$DS" --graph "$GRAPH" --groups $QGROUPS \
    --summary "$OUT/completion.json"

# chain on SEL, frontier rules calibrated on CAL at delta / (2 M), bounds on EVAL
$PY "$ROOT/pipeline/frontier.py" --red "$RED" --ds "$DS" --graph "$GRAPH" --config "$ROOT/configs/frontier.json" \
    --out "$OUT/frontier.json"
$PY "$ROOT/pipeline/exact_bounds.py" --red "$RED" --ds "$DS" --graph "$GRAPH" --frontier "$OUT/frontier.json" \
    --out "$OUT/exact.json"

# every rule calibrated at eps = delta = 0.05 on CAL, visits on EVAL, paired bootstrap of the visit ratios
$PY "$ROOT/pipeline/common_target.py" --red "$RED" --ds "$DS" --graph "$GRAPH" --frontier "$OUT/frontier.json" \
    --seed "$CT_SEED" --out "$OUT/common_target.json"

# held-out check on HOLD
$PY "$ROOT/pipeline/holdout.py" --red "$RED" --ds "$DS" --graph "$GRAPH" --frontier "$OUT/frontier.json" \
    --out "$OUT/holdout.json"

#!/bin/sh
# Table III (mean latency, visits and failure rate on the 20,000 HOLD queries of graph A, every rule calibrated at
# eps = delta = 0.05, four timing windows) and the readings quoted with it: share of the stopping check, throughput at
# 1 to 32 threads, the failed answers of our rule.
# Before: reproduce/main_setting.sh for amazon and arxiv, graph A; reproduce/table2.sh (fine-grid distance-adaptive
# outputs of graph A); reproduce/fig4.sh (DARTH and DARTH+ models and CAL sweeps, Patience stop times).
# Timing runs one thread on one pinned core whose hyperthread sibling is idle, on an otherwise idle machine
# (timing/README.md); calibrated parameters are read from the outputs of the compared rules.
set -eu

DATA=${DATA:-data}
WORK=${WORK:-work}
THREADS=${THREADS:-16}
CORE=${CORE:-2}                   # pinned core of the timing runs
PIN=${PIN:-taskset -c $CORE}
PY=${PY:-python3}                 # environment of requirements/analysis.txt
PY_ENGINE=${PY_ENGINE:-python3}   # environment of requirements/engine.txt
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engine/build}
W3_AMAZON=5392337270486950629     # recorded value: configs/seeds.json (timing window 3)
W3_ARXIV=2044015271982866979      # recorded value: configs/seeds.json
W4_AMAZON=2323460136693915647     # recorded value: configs/seeds.json (timing window 4)
W4_ARXIV=537168968883106647       # recorded value: configs/seeds.json
PATIENCE_SEED=0                   # no recorded value in configs/seeds.json; changes only the bootstrap bound of the visit ratio
WL=k10_kappa1
RED=$WORK/reduced/$WL
M=$WORK/results/$WL
T=$WORK/timing
mkdir -p "$T"

jget() {      # FILE KEY...: one field of a JSON file
    $PY -c 'import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d = d[k]
print(d)' "$@"
}
rejected() {  # report of baselines/list_size.py: the first list size of the CAL sequence that fails the test
    $PY -c 'import json, sys
print(next(s["L"] for s in json.load(open(sys.argv[1]))["cal_sequence"] if not s["passed"]))' "$1"
}

for ds in amazon arxiv; do
    if [ $ds = amazon ]; then S3=$W3_AMAZON; S4=$W4_AMAZON; else S3=$W3_ARXIV; S4=$W4_ARXIV; fi
    FR=$M/${ds}_A/frontier.json; CT=$M/${ds}_A/common_target.json
    E="--data $DATA --work $WORK --ds $ds --graph A --workload $WL --engine $ENGINE --threads $THREADS"

    # calibrated parameters of the compared rules (eps = delta = 0.05 on CAL)
    $PY_ENGINE "$ROOT/baselines/darth.py" calibrate $E --out "$WORK/results/darth_${ds}_A.json"
    $PY_ENGINE "$ROOT/baselines/list_size.py" native $E --red "$RED" --out "$WORK/results/list_size_native_${ds}_A.npz"
    $PY "$ROOT/baselines/list_size.py" report --red "$RED" --ds $ds --graph A --frontier "$FR" --common-target "$CT" \
        --run "$WORK/results/list_size_native_${ds}_A.npz" --out "$WORK/results/list_size_native_${ds}_A.json"
    L_NATIVE=$(jget "$WORK/results/list_size_native_${ds}_A.json" list_size)
    L_REJECTED=$(rejected "$WORK/results/list_size_native_${ds}_A.json")
    $PY_ENGINE "$ROOT/baselines/list_size.py" fine $E --red "$RED" --accepted "$L_NATIVE" --rejected "$L_REJECTED" \
        --out "$WORK/results/list_size_fine_${ds}_A.npz"
    $PY "$ROOT/baselines/list_size.py" report --red "$RED" --ds $ds --graph A --frontier "$FR" --common-target "$CT" \
        --run "$WORK/results/list_size_fine_${ds}_A.npz" --out "$WORK/results/list_size_fine_${ds}_A.json"
    L_FINE=$(jget "$WORK/results/list_size_fine_${ds}_A.json" list_size)
    $PY "$ROOT/baselines/patience.py" calibrate --red "$RED" --ds $ds --graph A --frontier "$FR" --stops "$WORK/patience" \
        --seed "$PATIENCE_SEED" --out "$WORK/results/patience_${ds}_A.json"
    PATIENCE=$(jget "$WORK/results/patience_${ds}_A.json" patience param)
    GAMMA=$(jget "$M/${ds}_A/distance_adaptive.json" table distance_adaptive param)
    $PY "$ROOT/baselines/threshold_curve.py" fit --red "$RED" --ds $ds --graph A --out "$WORK/results/curve_${ds}_A.json"

    # timing bundle of the HOLD queries, then the four timing windows
    $PY "$ROOT/timing/export_bundle.py" --red "$RED" --data "$DATA" --ds $ds --graph A --frontier "$FR" --queries hold \
        --out "$T/${ds}_A_hold"
    B="--bundle $T/${ds}_A_hold --data $DATA --work $WORK --engine $ENGINE"
    DARTH="--darth-model $WORK/models/darth_${ds}_A --darth-calibration $WORK/results/darth_${ds}_A.json"
    DPLUS="--darth-plus-model $WORK/models/darthplus_${ds}_A --darth-plus-levels $WORK/results/darthplus_levels_${ds}_A.json"
    W3="--list-size $L_FINE --patience $PATIENCE --gamma $GAMMA $DARTH $DPLUS"
    OMP_NUM_THREADS=1 $PIN $PY_ENGINE "$ROOT/timing/time_rules.py" window --window 1 $B $DARTH --out "$T/window1"
    OMP_NUM_THREADS=1 $PIN $PY_ENGINE "$ROOT/timing/time_rules.py" window --window 2 $B --list-size "$L_NATIVE" --out "$T/window2"
    OMP_NUM_THREADS=1 $PIN $PY_ENGINE "$ROOT/timing/time_rules.py" window --window 3 $B $W3 --seed "$S3" --out "$T/window3"
    OMP_NUM_THREADS=1 $PIN $PY_ENGINE "$ROOT/timing/time_rules.py" window --window 4 $B --curve "$WORK/results/curve_${ds}_A.json" \
        --seed "$S4" --out "$T/window4"

    # share of the stopping check in the query time, throughput under concurrency, failed answers
    OMP_NUM_THREADS=1 $PIN $PY_ENGINE "$ROOT/timing/time_rules.py" share $B --rule cast --out "$T/window3/${ds}_share_cast.json"
    $PY_ENGINE "$ROOT/timing/throughput.py" $B $W3 --out "$T/${ds}_throughput.json"
    $PY_ENGINE "$ROOT/timing/failed_answers.py" $B $W3 --out "$T/${ds}_failed_answers.json"
done

$PY "$ROOT/timing/make_table3.py" --window1 "$T/window1" --window2 "$T/window2" --window3 "$T/window3" --window4 "$T/window4" \
    > "$WORK/results/table3.txt"

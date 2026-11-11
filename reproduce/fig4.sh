#!/bin/sh
# Fig. 4 (visits of each compared rule over ours on the EVAL queries of graph A, every rule calibrated at each risk
# level at delta / (2 M) like the frontier rule); the same readings on graph B; and the comparison at equal EVAL
# failure rates of the paragraph "Cost and latency at a common risk target".
# Before: reproduce/main_setting.sh for amazon and arxiv, graphs A and B; reproduce/table2.sh (its fine-grid
# distance-adaptive outputs of the main workload, files distance_adaptive.json).
set -eu

DATA=${DATA:-data}
WORK=${WORK:-work}
THREADS=${THREADS:-16}
PY=${PY:-python3}                 # environment of requirements/analysis.txt
PY_ENGINE=${PY_ENGINE:-python3}   # environment of requirements/engine.txt
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engine/build}
# bootstrap seeds of baselines/levels.py --seed, one per group of levels (0.01-0.03, 0.04-0.06, 0.07-0.10)
LV_AMAZON_A="350267489 1072392063 1999518660"    # recorded value: configs/seeds.json
LV_AMAZON_B="495734224 46749071 1002935347"      # recorded value: configs/seeds.json
LV_ARXIV_A="1556097536 989341589 1457279540"     # recorded value: configs/seeds.json
LV_ARXIV_B="122394615 592980840 299595782"       # recorded value: configs/seeds.json
# bootstrap seeds of the report steps of baselines/darth.py and baselines/darth_plus.py
DARTH_AMAZON=1726052678           # recorded value: configs/seeds.json
DARTH_ARXIV=769134099             # recorded value: configs/seeds.json
DARTHPLUS_AMAZON=1207146141       # recorded value: configs/seeds.json
DARTHPLUS_ARXIV=394627324         # recorded value: configs/seeds.json
WL=k10_kappa1
RED=$WORK/reduced/$WL
M=$WORK/results/$WL
OUT=$WORK/results/levels
mkdir -p "$OUT"

# first group, list size and distance-adaptive rule (logged grid) at the nine levels, in three groups of levels
for s in amazon:A amazon:B arxiv:A arxiv:B; do
    ds=${s%:*}; g=${s#*:}
    case $s in
        amazon:A) seeds=$LV_AMAZON_A ;; amazon:B) seeds=$LV_AMAZON_B ;;
        arxiv:A) seeds=$LV_ARXIV_A ;; arxiv:B) seeds=$LV_ARXIV_B ;;
    esac
    i=1
    for lv in 0.01,0.02,0.03 0.04,0.05,0.06 0.07,0.08,0.1; do
        seed=$(echo $seeds | cut -d' ' -f$i)
        $PY "$ROOT/baselines/levels.py" --red "$RED" --ds $ds --graph $g --frontier "$M/${ds}_$g/frontier.json" \
            --levels $lv --seed "$seed" --out "$OUT/levels_${ds}_${g}_$i.json"
        i=$((i + 1))
    done
    # distance-adaptive rule on the fine grid of gamma (levels block of the output of reproduce/table2.sh)
    cp "$M/${ds}_$g/distance_adaptive.json" "$OUT/distance_adaptive_${ds}_$g.json"
done

# DARTH and DARTH+ on graph A: training on 10,000 SEL queries, one CAL sweep read at every level, EVAL runs, pairing
for ds in amazon arxiv; do
    if [ $ds = amazon ]; then SD=$DARTH_AMAZON; SP=$DARTHPLUS_AMAZON; else SD=$DARTH_ARXIV; SP=$DARTHPLUS_ARXIV; fi
    FR=$M/${ds}_A/frontier.json
    E="--data $DATA --work $WORK --ds $ds --graph A --workload $WL --engine $ENGINE --threads $THREADS"
    $PY_ENGINE "$ROOT/baselines/darth.py" train $E
    $PY_ENGINE "$ROOT/baselines/darth.py" levels $E --frontier "$FR" --out "$WORK/results/darth_levels_${ds}_A"
    $PY "$ROOT/baselines/darth.py" report --red "$RED" --ds $ds --graph A --frontier "$FR" \
        --run "$WORK/results/darth_levels_${ds}_A" --seed "$SD" --out "$OUT/darth_report_${ds}_A.json"
    $PY_ENGINE "$ROOT/baselines/darth_plus.py" train $E
    $PY_ENGINE "$ROOT/baselines/darth_plus.py" select $E
    $PY_ENGINE "$ROOT/baselines/darth_plus.py" levels $E --frontier "$FR" --out "$WORK/results/darthplus_levels_${ds}_A"
    $PY "$ROOT/baselines/darth_plus.py" report --red "$RED" --ds $ds --graph A --frontier "$FR" \
        --run "$WORK/results/darthplus_levels_${ds}_A" --seed "$SP" --out "$OUT/darthplus_report_${ds}_A.json"
done

$PY "$ROOT/figures/fig4_visit_ratios.py" \
    --levels-amazon "$OUT/levels_amazon_A_1.json" "$OUT/levels_amazon_A_2.json" "$OUT/levels_amazon_A_3.json" \
    --distance-adaptive-amazon "$OUT/distance_adaptive_amazon_A.json" \
    --darth-amazon "$OUT/darth_report_amazon_A.json" --darth-plus-amazon "$OUT/darthplus_report_amazon_A.json" \
    --levels-arxiv "$OUT/levels_arxiv_A_1.json" "$OUT/levels_arxiv_A_2.json" "$OUT/levels_arxiv_A_3.json" \
    --distance-adaptive-arxiv "$OUT/distance_adaptive_arxiv_A.json" \
    --darth-arxiv "$OUT/darth_report_arxiv_A.json" --darth-plus-arxiv "$OUT/darthplus_report_arxiv_A.json" \
    --out "$WORK/results/levels.pdf"

# equal EVAL failure rates: Patience swept over its parameter, then the interpolated ratios
for s in amazon:A amazon:B arxiv:A arxiv:B; do
    $PY_ENGINE "$ROOT/baselines/patience.py" relog --data "$DATA" --work "$WORK" --ds ${s%:*} --graph ${s#*:} --workload $WL \
        --engine "$ENGINE" --threads "$THREADS" --red "$RED" --out-dir "$WORK/patience"
done
$PY "$ROOT/baselines/patience.py" points --stops "$WORK/patience" --settings amazon:A amazon:B arxiv:A arxiv:B \
    --out "$OUT/patience_points.json"
$PY "$ROOT/baselines/equal_failure_rate.py" --levels-dir "$OUT" --adaptive-dir "$OUT" --patience "$OUT/patience_points.json" \
    --recall-dir "$OUT" --settings amazon_A amazon_B arxiv_A arxiv_B --recall-graph A --out "$OUT/equal_failure_rate.json"

#!/bin/sh
# Table II: visits of every rule calibrated at eps = delta = 0.05 over those of CaST on EVAL, CaST visits, and the
# optimality-gap bound U / L of the frontier rule (maximum over the levels from 0.03 to 0.10, and at 0.05), for the
# main workload, the thirteen further settings and the recall target rho = 0.9; then the significance statement of the
# caption (level 0.025 / 102).
# Before: reproduce/main_setting.sh for amazon and arxiv, graphs A and B; the scripts of data/ for every dataset and
# query set below (data/README.md: category stores kappa2 and unconstrained, title queries --set refit, YFCC-10M with
# its official queries and its graph, YouTube-8M with its graph).
set -eu

DATA=${DATA:-data}
WORK=${WORK:-work}
THREADS=${THREADS:-24}
PY=${PY:-python3}                 # environment of requirements/analysis.txt
PY_ENGINE=${PY_ENGINE:-python3}   # environment of requirements/engine.txt
PY_DATA=${PY_DATA:-python3}       # environment of requirements/data.txt
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engine/build}
RES=$WORK/results
POOL="fitpool chunk0 chunk1 chunk2 chunk3 chunk4 chunk5 chunk6 chunk7 chunk8 chunk9 chunk10 chunk11 chunk12 chunk13 chunk14 chunk15 chunk16 chunk17 chunk18 chunk19"
PARTS="fit sel cal eval"

# bootstrap seeds of pipeline/common_target.py --seed
CT_KAPPA2_AMAZON=1522294183       # recorded value: configs/seeds.json
CT_KAPPA2_ARXIV=989305394         # recorded value: configs/seeds.json
CT_UNC_AMAZON=673637328           # recorded value: configs/seeds.json
CT_UNC_ARXIV=13466739             # recorded value: configs/seeds.json
CT_VAMANA_AMAZON=175307109        # recorded value: configs/seeds.json
CT_VAMANA_ARXIV=361482005         # recorded value: configs/seeds.json
CT_TITLE_AMAZON=19487254          # recorded value: configs/seeds.json
CT_TITLE_ARXIV=939934268          # recorded value: configs/seeds.json
CT_YFCC_TAGS=293633634            # recorded value: configs/seeds.json
CT_YFCC_UNC=1852160409            # recorded value: configs/seeds.json
CT_YFCC_OFFICIAL=484976561        # recorded value: configs/seeds.json
CT_YT8M_ENT=1624280872            # recorded value: configs/seeds.json
CT_YT8M_UNC=1256281484            # recorded value: configs/seeds.json
CT_RECALL_AMAZON_A=1282111919     # recorded value: configs/seeds.json
CT_RECALL_AMAZON_B=1495248145     # recorded value: configs/seeds.json
CT_RECALL_ARXIV_A=1647710080      # recorded value: configs/seeds.json
CT_RECALL_ARXIV_B=1858666467      # recorded value: configs/seeds.json
# bootstrap seeds of baselines/distance_adaptive.py calibrate --seed
DA_AMAZON_A=396436122             # recorded value: configs/seeds.json
DA_AMAZON_B=617700024             # recorded value: configs/seeds.json
DA_ARXIV_A=82055672               # recorded value: configs/seeds.json
DA_ARXIV_B=1992764085             # recorded value: configs/seeds.json
DA_KAPPA2_AMAZON=1492228236       # recorded value: configs/seeds.json
DA_KAPPA2_ARXIV=710617975         # recorded value: configs/seeds.json
DA_UNC_AMAZON=1503096895          # recorded value: configs/seeds.json
DA_UNC_ARXIV=736544522            # recorded value: configs/seeds.json
DA_VAMANA_AMAZON=1274654702       # recorded value: configs/seeds.json
DA_VAMANA_ARXIV=66139246          # recorded value: configs/seeds.json
DA_YFCC_TAGS=690378728            # recorded value: configs/seeds.json
DA_YFCC_UNC=148134221             # recorded value: configs/seeds.json
DA_YFCC_OFFICIAL=867939300        # recorded value: configs/seeds.json
DA_TITLE_AMAZON=409210656         # recorded value: configs/seeds.json
DA_TITLE_ARXIV=1306051833         # recorded value: configs/seeds.json
DA_YT8M_ENT=2086885997            # recorded value: configs/seeds.json
DA_YT8M_UNC=1980319086            # recorded value: configs/seeds.json
DA_RECALL_AMAZON_A=1818310444     # recorded value: configs/seeds.json
DA_RECALL_AMAZON_B=1746464820     # recorded value: configs/seeds.json
DA_RECALL_ARXIV_A=1547844387      # recorded value: configs/seeds.json
DA_RECALL_ARXIV_B=1863741118      # recorded value: configs/seeds.json

answers() {   # WORKLOAD DS SUMMARY_NAME [exact_answers.py options] -- GROUP...
    wl=$1; ds=$2; nm=$3; shift 3; opts=""
    while [ "$1" != "--" ]; do opts="$opts $1"; shift; done; shift
    mkdir -p "$RES/$wl"
    $PY_DATA "$ROOT/pipeline/exact_answers.py" --data "$DATA" --ds "$ds" --workload "$wl" $opts --groups "$@" \
        --summary "$RES/$wl/${nm}_answers.json"
}

logs() {      # WORKLOAD DS GRAPH OUT_NAME PREFIX GROUP...   (PREFIX "-": none)
    wl=$1; ds=$2; g=$3; nm=$4; pre=$5; shift 5
    $PY_ENGINE "$ROOT/pipeline/log_trajectories.py" --data "$DATA" --work "$WORK" --ds "$ds" --graph "$g" --workload "$wl" \
        --groups "$@" --threads "$THREADS" --engine "$ENGINE"
    if [ "$pre" = "-" ]; then
        $PY "$ROOT/pipeline/reduce_logs.py" --work "$WORK" --workload "$wl" --ds "$ds" --graph "$g" --groups "$@" --out-name "$nm"
    else
        $PY "$ROOT/pipeline/reduce_logs.py" --work "$WORK" --workload "$wl" --ds "$ds" --graph "$g" --groups "$@" --out-name "$nm" \
            --strip-prefix "$pre"
    fi
}

offline() {   # RED_NAME OUT_NAME DS GRAPH LAYOUT LEVEL_RULE TARGET CT_SEED   (LEVEL_RULE "-": configs/frontier.json as is)
    red=$WORK/reduced/$1; out=$RES/$2/$3_$4; mkdir -p "$out"
    if [ "$6" = "-" ]; then
        cfg=$ROOT/configs/frontier.json
    else
        cfg=$out/config.json
        $PY "$ROOT/pipeline/completion_share.py" --red "$red" --ds "$3" --graph "$4" --layout "$5" \
            --config "$ROOT/configs/frontier.json" --out-config "$cfg" --rule "$6" --out "$out/completion.json"
    fi
    $PY "$ROOT/pipeline/frontier.py" --red "$red" --ds "$3" --graph "$4" --layout "$5" --target "$7" --config "$cfg" \
        --out "$out/frontier.json"
    $PY "$ROOT/pipeline/exact_bounds.py" --red "$red" --ds "$3" --graph "$4" --frontier "$out/frontier.json" --out "$out/exact.json"
    $PY "$ROOT/pipeline/common_target.py" --red "$red" --ds "$3" --graph "$4" --frontier "$out/frontier.json" --seed "$8" \
        --out "$out/common_target.json"
}

relog() {     # WORKLOAD RED_NAME DS GRAPH LAYOUT GAMMA_MAX [GROUP_PREFIX]: stops of the distance-adaptive rule on the fine grid of gamma
    $PY_ENGINE "$ROOT/baselines/distance_adaptive.py" relog --data "$DATA" --work "$WORK" --ds "$3" --graph "$4" --workload "$1" \
        --engine "$ENGINE" --threads "$THREADS" --red "$WORK/reduced/$2" --layout "$5" --gamma-max "$6" --group-prefix "${7:-}" \
        --out-dir "$WORK/fine/$2"
}

adaptive() {  # RED_NAME OUT_NAME DS GRAPH DA_SEED: the Dist.-adaptive cell on the fine grid (read by tables/table2.py)
    out=$RES/$2/$3_$4
    $PY "$ROOT/baselines/distance_adaptive.py" calibrate --red "$WORK/reduced/$1" --ds "$3" --graph "$4" \
        --frontier "$out/frontier.json" --fine "$WORK/fine/$1" --seed "$5" --out "$out/distance_adaptive.json"
}

# ---- main workload, graphs A and B (frontier, exact bounds and common target from reproduce/main_setting.sh)
for s in amazon:A amazon:B arxiv:A arxiv:B; do
    relog k10_kappa1 k10_kappa1 "${s%:*}" "${s#*:}" pool 0.3
done
adaptive k10_kappa1 k10_kappa1 amazon A "$DA_AMAZON_A"
adaptive k10_kappa1 k10_kappa1 amazon B "$DA_AMAZON_B"
adaptive k10_kappa1 k10_kappa1 arxiv A "$DA_ARXIV_A"
adaptive k10_kappa1 k10_kappa1 arxiv B "$DA_ARXIV_B"

# ---- capacity 2 and unconstrained, graph A
for wl in k10_kappa2 k10_unconstrained; do
    for ds in amazon arxiv; do
        answers $wl $ds $ds -- $POOL
        logs $wl $ds A $wl - $POOL
        relog $wl $wl $ds A pool 0.3
    done
done
offline k10_kappa2 k10_kappa2 amazon A pool above X_exact "$CT_KAPPA2_AMAZON"
offline k10_kappa2 k10_kappa2 arxiv A pool above X_exact "$CT_KAPPA2_ARXIV"
offline k10_unconstrained k10_unconstrained amazon A pool above X_exact "$CT_UNC_AMAZON"
offline k10_unconstrained k10_unconstrained arxiv A pool above X_exact "$CT_UNC_ARXIV"
adaptive k10_kappa2 k10_kappa2 amazon A "$DA_KAPPA2_AMAZON"
adaptive k10_kappa2 k10_kappa2 arxiv A "$DA_KAPPA2_ARXIV"
adaptive k10_unconstrained k10_unconstrained amazon A "$DA_UNC_AMAZON"
adaptive k10_unconstrained k10_unconstrained arxiv A "$DA_UNC_ARXIV"

# ---- Vamana graphs (main workload; exact answers of reproduce/main_setting.sh)
for ds in amazon arxiv; do
    $PY_DATA "$ROOT/pipeline/build_graphs.py" vamana --data "$DATA" --work "$WORK" --ds $ds --engine "$ENGINE"
    logs k10_kappa1 $ds vamana k10_kappa1 - $POOL
    relog k10_kappa1 k10_kappa1 $ds vamana pool 0.3
done
offline k10_kappa1 k10_kappa1 amazon vamana pool above X_exact "$CT_VAMANA_AMAZON"
offline k10_kappa1 k10_kappa1 arxiv vamana pool above X_exact "$CT_VAMANA_ARXIV"
adaptive k10_kappa1 k10_kappa1 amazon vamana "$DA_VAMANA_AMAZON"
adaptive k10_kappa1 k10_kappa1 arxiv vamana "$DA_VAMANA_ARXIV"

# ---- title queries with their own FIT, SEL, CAL and EVAL (graph A)
for ds in amazon arxiv; do
    answers k10_kappa1 $ds ${ds}_title -- title_fit title_sel title_cal title_eval
    logs k10_kappa1 $ds A k10_kappa1_title title_ title_fit title_sel title_cal title_eval
    relog k10_kappa1 k10_kappa1_title $ds A parts 0.3 title_
done
offline k10_kappa1_title k10_kappa1_title amazon A parts above X_exact "$CT_TITLE_AMAZON"
offline k10_kappa1_title k10_kappa1_title arxiv A parts above X_exact "$CT_TITLE_ARXIV"
adaptive k10_kappa1_title k10_kappa1_title amazon A "$DA_TITLE_AMAZON"
adaptive k10_kappa1_title k10_kappa1_title arxiv A "$DA_TITLE_ARXIV"

# ---- YFCC-10M (graph A built at data preparation): tags, unconstrained, official query vectors
answers k10_kappa1 yfcc10m yfcc10m -- $PARTS
answers k10_unconstrained yfcc10m yfcc10m --plain-from k10_kappa1 -- $PARTS
answers k10_kappa1 yfcc10m yfcc10m_official -- official_fit official_sel official_cal official_eval
logs k10_kappa1 yfcc10m A k10_kappa1 - $PARTS
logs k10_unconstrained yfcc10m A k10_unconstrained - $PARTS
logs k10_kappa1 yfcc10m A k10_kappa1_official official_ official_fit official_sel official_cal official_eval
relog k10_kappa1 k10_kappa1 yfcc10m A parts 0.6
relog k10_unconstrained k10_unconstrained yfcc10m A parts 0.6
relog k10_kappa1 k10_kappa1_official yfcc10m A parts 0.6 official_
offline k10_kappa1 k10_kappa1 yfcc10m A parts margin X_exact "$CT_YFCC_TAGS"
offline k10_unconstrained k10_unconstrained yfcc10m A parts margin X_exact "$CT_YFCC_UNC"
offline k10_kappa1_official k10_kappa1_official yfcc10m A parts margin X_exact "$CT_YFCC_OFFICIAL"
adaptive k10_kappa1 k10_kappa1 yfcc10m A "$DA_YFCC_TAGS"
adaptive k10_unconstrained k10_unconstrained yfcc10m A "$DA_YFCC_UNC"
adaptive k10_kappa1_official k10_kappa1_official yfcc10m A "$DA_YFCC_OFFICIAL"

# ---- YouTube-8M (graph A built at data preparation): entities and unconstrained
answers k10_kappa1 youtube8m youtube8m -- $PARTS
answers k10_unconstrained youtube8m youtube8m --plain-from k10_kappa1 -- $PARTS
for wl in k10_kappa1 k10_unconstrained; do
    logs $wl youtube8m A $wl - $PARTS
    relog $wl $wl youtube8m A parts 0.3
done
offline k10_kappa1 k10_kappa1 youtube8m A parts above X_exact "$CT_YT8M_ENT"
offline k10_unconstrained k10_unconstrained youtube8m A parts above X_exact "$CT_YT8M_UNC"
adaptive k10_kappa1 k10_kappa1 youtube8m A "$DA_YT8M_ENT"
adaptive k10_unconstrained k10_unconstrained youtube8m A "$DA_YT8M_UNC"

# ---- recall target rho = 0.9 on the main workload: completion time X_recall09, score refitted on that label
offline k10_kappa1 k10_kappa1_recall09 amazon A pool - X_recall09 "$CT_RECALL_AMAZON_A"
offline k10_kappa1 k10_kappa1_recall09 amazon B pool - X_recall09 "$CT_RECALL_AMAZON_B"
offline k10_kappa1 k10_kappa1_recall09 arxiv A pool - X_recall09 "$CT_RECALL_ARXIV_A"
offline k10_kappa1 k10_kappa1_recall09 arxiv B pool - X_recall09 "$CT_RECALL_ARXIV_B"
adaptive k10_kappa1 k10_kappa1_recall09 amazon A "$DA_RECALL_AMAZON_A"
adaptive k10_kappa1 k10_kappa1_recall09 amazon B "$DA_RECALL_AMAZON_B"
adaptive k10_kappa1 k10_kappa1_recall09 arxiv A "$DA_RECALL_ARXIV_A"
adaptive k10_kappa1 k10_kappa1_recall09 arxiv B "$DA_RECALL_ARXIV_B"

# ---- rows of Table II (LaTeX rows, then a JSON summary with the bootstrap bounds and standard errors)
M=$RES/k10_kappa1; R=$RES/k10_kappa1_recall09
$PY "$ROOT/tables/table2.py" \
    "Amazon, A=$M/amazon_A" "Amazon, B=$M/amazon_B" "arXiv, A=$M/arxiv_A" "arXiv, B=$M/arxiv_B" \
    "Amazon, kappa 2=$RES/k10_kappa2/amazon_A" "Amazon, unconstrained=$RES/k10_unconstrained/amazon_A" \
    "arXiv, kappa 2=$RES/k10_kappa2/arxiv_A" "arXiv, unconstrained=$RES/k10_unconstrained/arxiv_A" \
    "Amazon, Vamana=$M/amazon_vamana" "arXiv, Vamana=$M/arxiv_vamana" \
    "Amazon, title queries=$RES/k10_kappa1_title/amazon_A" "arXiv, title queries=$RES/k10_kappa1_title/arxiv_A" \
    "YFCC-10M, tags=$M/yfcc10m_A" "YFCC-10M, unconstrained=$RES/k10_unconstrained/yfcc10m_A" \
    "YFCC-10M, official vectors=$RES/k10_kappa1_official/yfcc10m_A" \
    "YouTube-8M, entities=$M/youtube8m_A" "YouTube-8M, unconstrained=$RES/k10_unconstrained/youtube8m_A" \
    "Recall, Amazon, A=$R/amazon_A" "Recall, Amazon, B=$R/amazon_B" "Recall, arXiv, A=$R/arxiv_A" "Recall, arXiv, B=$R/arxiv_B" \
    > "$RES/table2.txt"

# ---- caption: every ratio above one of the 17 rows below the main rows is significant at level 0.025 / 102
ct() { echo "$RES/$1/common_target.json,$RES/$1/distance_adaptive.json"; }
$PY "$ROOT/analysis/significance.py" \
    "Amazon, kappa 2=$(ct k10_kappa2/amazon_A)" "Amazon, unconstrained=$(ct k10_unconstrained/amazon_A)" \
    "arXiv, kappa 2=$(ct k10_kappa2/arxiv_A)" "arXiv, unconstrained=$(ct k10_unconstrained/arxiv_A)" \
    "Amazon, Vamana=$(ct k10_kappa1/amazon_vamana)" "arXiv, Vamana=$(ct k10_kappa1/arxiv_vamana)" \
    "Amazon, title queries=$(ct k10_kappa1_title/amazon_A)" \
    "arXiv, title queries=$(ct k10_kappa1_title/arxiv_A)" \
    "YFCC-10M, tags=$(ct k10_kappa1/yfcc10m_A)" "YFCC-10M, unconstrained=$(ct k10_unconstrained/yfcc10m_A)" \
    "YFCC-10M, official vectors=$(ct k10_kappa1_official/yfcc10m_A)" \
    "YouTube-8M, entities=$(ct k10_kappa1/youtube8m_A)" "YouTube-8M, unconstrained=$(ct k10_unconstrained/youtube8m_A)" \
    "Recall, Amazon, A=$(ct k10_kappa1_recall09/amazon_A)" "Recall, Amazon, B=$(ct k10_kappa1_recall09/amazon_B)" \
    "Recall, arXiv, A=$(ct k10_kappa1_recall09/arxiv_A)" "Recall, arXiv, B=$(ct k10_kappa1_recall09/arxiv_B)" \
    --level 0.025 --family 102 --out "$RES/table2_significance.json"

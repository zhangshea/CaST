#!/bin/sh
# Fig. 3 (near-optimality on EVAL of the frontier rule): (a) U / L rounded up at the nine risk levels for the four
# settings of the main workload, (b) cost-risk frontier of graph A; then the readouts of the paragraph
# "Near-optimality" that come from the same files (ranges of the maximum U / L, U / L at 0.02 and 0.01, EVAL failure
# rate over the target, 1 - L / U at 0.05).
# Before: reproduce/main_setting.sh for amazon and arxiv, graphs A and B.
set -eu

WORK=${WORK:-work}
PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=$WORK/results/k10_kappa1

$PY "$ROOT/figures/fig3_near_optimality.py" \
    --exact-bounds "$M/amazon_A/exact.json" "$M/amazon_B/exact.json" "$M/arxiv_A/exact.json" "$M/arxiv_B/exact.json" \
    --frontier "$M/amazon_A/frontier.json" "$M/arxiv_A/frontier.json" \
    --out "$WORK/results/nearopt.pdf"

$PY "$ROOT/analysis/frontier_readouts.py" \
    "amazon_A=$M/amazon_A/frontier.json,$M/amazon_A/exact.json" "amazon_B=$M/amazon_B/frontier.json,$M/amazon_B/exact.json" \
    "arxiv_A=$M/arxiv_A/frontier.json,$M/arxiv_A/exact.json" "arxiv_B=$M/arxiv_B/frontier.json,$M/arxiv_B/exact.json" \
    --out "$WORK/results/frontier_readouts.json"

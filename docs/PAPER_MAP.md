# From the paper to the code

Every table, figure and number of Section VI (Experiments), and the numbers of the abstract, the introduction and the
contribution list, with the command that produces it and the output file and field it is read from. The order follows
the paper. Numbers that no script of the package produces are listed at the end.

## Conventions

- `DATA` holds the prepared data (`data/README.md`), `WORK` the graphs, logs, reduced arrays and results.
- `RED = WORK/reduced/k10_kappa1` (reduced arrays of the main workload), `RES = WORK/results`,
  `M = RES/k10_kappa1`. A setting directory is `RES/<name>/<ds>_<graph>/`, for example `M/amazon_A/`.
- The full command lines are in `reproduce/` (one script per table and figure); this file names the script and the
  step. A command written here in full is one that no reproduction script runs.
- `python` is the interpreter of the environment the script needs (`README.md`, Requirements).
- "Rounded up" is `analysis/common.py:up3` (and `tables/table2.py:up3`): ceiling to three decimals. Ratios U / L in
  Table II and Fig. 3(a) are the exact values of `pipeline/exact_bounds.py` (`ratio_up6`) rounded up; the float ratios
  of `pipeline/frontier.py` (`bounds[eps].ratio_chain`) are used where a row says so.
- "Range over fields" means that the script writes the per-setting values and the stated range is their minimum and
  maximum, read from the output.

## The main pipeline of one setting (`reproduce/main_setting.sh`)

| Step | Command (main workload, one setting) | Output |
|---|---|---|
| exact answers | `pipeline/exact_answers.py --data DATA --ds DS --workload k10_kappa1 --groups fitpool chunk0 ... chunk19 --summary ...` | `DATA/<ds>/answers/<group>_k10_kappa1.npz` (`S`) |
| graph | `pipeline/build_graphs.py hnsw --data DATA --work WORK --ds DS --graph G --seed SEED` | `WORK/graphs/<ds>_<G>.hnsw` |
| logs | `pipeline/log_trajectories.py --data DATA --work WORK --ds DS --graph G --workload k10_kappa1 --groups ...` | `WORK/logs/k10_kappa1/` |
| reduction | `pipeline/reduce_logs.py --work WORK --workload k10_kappa1 --ds DS --graph G --groups ... --summary M/<s>/completion.json` | `RED/<s>_<group>.npz`, `RED/<s>_<group>_stops.npz` |
| frontier | `pipeline/frontier.py --red RED --ds DS --graph G --config configs/frontier.json --out M/<s>/frontier.json` | chain, frontier rules, U, L, float U / L |
| exact bounds | `pipeline/exact_bounds.py --red RED --ds DS --graph G --frontier M/<s>/frontier.json --out M/<s>/exact.json` | `EVAL.rules[eps].ratio_up6`, `U_up6`, `L_down6`, `EVAL.L_curve` |
| common target | `pipeline/common_target.py --red RED --ds DS --graph G --frontier M/<s>/frontier.json --seed SEED --out M/<s>/common_target.json` | `table[rule].{mean_visits, fail_rate, visit_ratio}` |
| held-out check | `pipeline/holdout.py --red RED --ds DS --graph G --frontier M/<s>/frontier.json --out M/<s>/holdout.json` | `frontier[eps]`, `hold_bounds[eps]`, `common_target[rule]` |

## VI-A Experiments Setup

### Datasets and Table I

| Paper | Script and command | Output and field |
|---|---|---|
| Table I, all cells (base items, dimension, categories, max and mean categories per item) | `reproduce/table1.sh`: `data/dataset_stats.py table --data DATA --ds amazon arxiv yfcc10m youtube8m --out RES/table1.json` | `RES/table1.json`: `<ds>.{n_base, dim, categories, categories_per_item_max, categories_per_item_mean}` |
| Datasets: Amazon Sports and Outdoors, second-level node and leaf; arXiv subject classes; one million base items; all-MiniLM-L6-v2, 384 dimensions | `data/text_pool.py` (parsing, categories), `data/prepare_items.py texts` / `install`, `data/embed_texts.py` | `DATA/<ds>/base.npy`, `categories_main.npz`; dimension in Table I |
| YFCC-10M: unit vectors, the 1,000 most frequent tags, images without such a tag removed, 9,791,295 base items, 200,000 official queries | `data/prepare_yfcc10m.py split`, `renumber`, `official` | `DATA/yfcc10m/`; base size in Table I |
| YFCC-10M: the constrained answer differs from the unconstrained top 10 for "4.4%" of queries | `pipeline/exact_answers.py --data DATA --ds yfcc10m --workload k10_kappa1 --groups fit sel cal eval --summary RES/k10_kappa1/yfcc10m_answers.json` (`reproduce/table2.sh`) | `RES/k10_kappa1/yfcc10m_answers.json`: `groups.eval.share_differs_from_unconstrained` |
| YouTube-8M: 1,024-dimensional mean visual features of videos, one million base items and the queries drawn by one random permutation, 3,862 entities | `data/prepare_youtube8m.py fetch`, `split`, `renumber` | `PREP/youtube8m/split.json`; Table I row |
| YouTube-8M: the constrained answer differs from the unconstrained top 10 for "about 70%" of queries | `pipeline/exact_answers.py --data DATA --ds youtube8m --workload k10_kappa1 --groups fit sel cal eval --summary RES/k10_kappa1/youtube8m_answers.json` (`reproduce/table2.sh`) | `share_differs_from_unconstrained` of the groups |
| Main workload: "differs from the unconstrained top 10 for 98.6% and 98.2% of queries" | `reproduce/table1.sh`: `data/dataset_stats.py share --data DATA --ds amazon arxiv --workload k10_kappa1 --n 10000 --out RES/table1_share.json` | `RES/table1_share.json`: `<ds>.share_differs` (first 10,000 EVAL queries) |
| Samples: base, 2,500 FIT queries and 200,000 further queries from one permutation; SEL, CAL, EVAL, HOLD at random; caption of Table I: FIT 2,500, SEL 20,000, CAL 40,000, EVAL 120,000, HOLD 20,000 | `data/text_pool.py` (pool permutation), `cast/setting.py` (`load_setting`, seeds `fit_permutation`, `pool_permutation`); `reproduce/main_setting.sh` | `M/<s>/frontier.json`: `n.{FIT, SEL, CAL, EVAL}`; `M/<s>/holdout.json`: `n_hold` |
| "Exact constrained answers are computed by a full scan" | `pipeline/exact_answers.py` | `DATA/<ds>/answers/` |
| "Between 0.10% and 0.58% of EVAL queries of the main workload do not complete within 32,768 visits" | `reproduce/fig3.sh`: `analysis/frontier_readouts.py` on the four settings | `RES/frontier_readouts.json`: `ranges.terminal_fail_eval` (from `frontier.json`: `terminal_fail_eval`) |
| "Thirteen further settings": capacity 2, unconstrained, Vamana, title queries, YFCC-10M (tags, unconstrained, official), YouTube-8M (entities, unconstrained) | `reproduce/table2.sh`; stores of capacity 2 and without constraint: `data/derive_category_stores.py`; title queries: `data/prepare_title_queries.py --set refit`; workloads: `cast/datasets.py:workload_spec` | one setting directory per row of Table II |

### Metrics

| Paper | Script and command | Output and field |
|---|---|---|
| Cost: mean bottom-layer visits; risk: fraction of answers that differ from S, or hold fewer than ceil(rho k) members | `cast/rules.py:evaluate_rule`; `pipeline/reduce_logs.py` (completion times `X_exact`, `X_recall09`) | `mean_visits`, `fail_rate` fields of every output |
| "The upper-layer descent adds about 170 distances per query" | `pipeline/log_trajectories.py` | `WORK/logs/<workload>/<ds>_<graph>_log.json`: `<group>.mean_descent_distances` |
| Latency: mean over queries of the median of five passes, features and models included | `timing/protocol.py`, `timing/time_rules.py window` | `WORK/timing/window<k>/<ds>_time.json`: `timing[rule].mean_us` |
| Optimality-gap bound U_m / L(eps_m) | `pipeline/frontier.py` (float), `pipeline/exact_bounds.py` (exact) | `bounds[eps].ratio_chain`; `EVAL.rules[eps].ratio_up6` |

### Compared methods

| Paper | Script and command | Output and field |
|---|---|---|
| First group (Level, Level+cap, LAET-style, Fixed), calibrated by the same test on CAL | `cast/compared.py` (`calibrate_level`, `calibrate_level_cap`, `calibrate_laet`, `calibrate_fixed`, `calibrate_common_target`), run by `pipeline/common_target.py` | `M/<s>/common_target.json`: `rules` |
| Level and Level+cap: "Both read the score at a finer resolution than z_j" (1,801 quantiles of the FIT scores as thresholds) | `cast/compared.py:level_candidates` | constant of the code |
| DARTH: eleven answer features, regressor settings, adaptive intervals, 10,000 SEL queries | `baselines/darth.py train` (`reproduce/fig4.sh`) | `WORK/models/darth_<ds>_A.json` |
| DARTH recall target, calibrated to 0.9942 (Amazon) and 0.9939 (arXiv) | `baselines/darth.py calibrate` (`reproduce/table3.sh`) | `RES/darth_<ds>_A.json`: `Rt_calibrated` |
| "At the literal targets 0.90 and 0.95, the port reaches mean recalls of 0.908 and 0.956 to 0.959" | `python baselines/recall_target.py --data DATA --work WORK --ds <ds> --graph A --out RES/recall_target_<ds>_A.json` | `rules.darth.uncalibrated_0.9.mean_recall`, `rules.darth.uncalibrated_0.95.mean_recall` (both datasets) |
| DARTH+: eight query statistics, lower quantile of the recall, "with its level chosen on unused SEL queries" | `baselines/darth_plus.py train`, `select` (`reproduce/fig4.sh`) | `WORK/models/darthplus_<ds>_A_select.json`: `selected` |
| "We check only that the engine evaluates the fitted forests as scikit-learn does" | `baselines/darth_plus.py train` | `WORK/models/darthplus_<ds>_A.json`: `forest_max_abs_diff`, `forest_within_1e_9` |
| List size: fails iff T_ef < X | `pipeline/log_trajectories.py` (`list_stop` per list size), `pipeline/common_target.py` | `table.list_size` |
| "On the 160,000 CAL and EVAL queries of each main setting, native runs at the selected list size differ from the replay in the visits of at most two queries and the exactness of two" | `baselines/list_size.py native` then `report` (`reproduce/table3.sh` for graph A; the same two commands with `--graph B` for graph B) | `RES/list_size_native_<s>.json`: `cal_sequence[]` entry of the selected size `.{visits_differ_from_log, failure_differs_from_log}` plus `eval.{visits_differ_from_log, failure_differs_from_log}` |
| Post-filtering is exact only when List size is (Lemma 1) | no number | |
| Distance-adaptive: stop once the answer is full and the next candidate is farther than (1 + gamma) times the k-th distance | `pipeline/reduce_logs.py` (`adaptive_stop`, 22 values of gamma), `baselines/distance_adaptive.py relog` / `calibrate` (fine grid) | `distance_adaptive.json`: `table.distance_adaptive` |
| "46 list sizes from 8,192 down to 10" | `pipeline/log_trajectories.py:LIST_SIZES` | constant of the code; `common_target.json`: `list_sizes` |
| "over 129 values of gamma from 2 down to 0, 246 on YFCC-10M" | `baselines/distance_adaptive.py relog` (default `--gamma-max 0.3`; `--gamma-max 0.6` for YFCC-10M) | `distance_adaptive.json`: `gammas` |
| Ada-ef: list size from a Gaussian model of the distances, success meaning the exact answer | `baselines/ada_ef.py stats`, `evaluate`, `compare` | see VI-B, "Cost and latency" |
| Patience: answer full and unchanged for a calibrated number of visits | `baselines/patience.py relog`, `calibrate` | `RES/patience_<s>.json`: `patience` |
| Fourth group: Longstaff-Schwartz backward induction; fully fitted, cross-fitted and declared-state policies | `baselines/learned_policies.py` | see Ablation study |

### Implementation details

| Paper | Script and command | Output and field |
|---|---|---|
| Two HNSW graphs per dataset (degree parameter 32, construction list 200), two independent draws of the insertion levels | `pipeline/build_graphs.py hnsw` (defaults `--M 32 --efc 200`; seeds in `configs/seeds.json`) | `WORK/graphs/<ds>_<graph>.json` (build information) |
| Vamana graph: out-degree 64, construction list 200, alpha = 1.2 | `pipeline/build_graphs.py vamana` (defaults `--degree 64 --complexity 200 --alpha 1.2`; `reproduce/table2.sh`) | `WORK/graphs/<ds>_vamana/build_report.json`: `params` |
| Best-first traversal with a list of 8,192 entries maintaining the greedy constrained answer, HNSW stopping condition | `engine/include/search.hpp`, `pipeline/log_trajectories.py` | |
| J = 100 checkpoints on a geometric grid from 64 to 32,768 visits, H = 32,832 | `cast/grid.py` (`GRID`, `J`, `H`) | `frontier.json`: `J`, `H` |
| Score model: gradient-boosted classifier (at most 200 iterations, 31 leaves) on FIT; seven features; B = 100 buckets at the 99 percentiles | `cast/setting.py:fit_interface`; features in `pipeline/reduce_logs.py` | `frontier.json`: `n_buckets`, `interface` |
| M = 9 risk levels 0.01 to 0.08 and 0.10, delta = 0.05, F = 2 sequences, calibration at delta / (2 M) | `configs/frontier.json`; `pipeline/frontier.py` | `frontier.json`: `config`, `rules.chain`, `rules.mixture` |
| Mixtures at 40 fixed multiples of eps | `pipeline/frontier.py` (sequence `mixture`: 4 + 36 multiples) | `rules.mixture[eps]` |
| "The multipliers are the negated slopes of the SEL frontier at the nine levels"; the chain is computed on a fixed grid of integer multipliers | `cast/chain.py` (`chain_multipliers`, `bound_multipliers`) | `frontier.json`: `chain_grid`, `multipliers`, `sel_slopes` |
| alpha_U = 0.025 split over the M cost bounds, alpha_L = 0.05 split over K = 9 multipliers, joint level 87.5% | `configs/frontier.json`; `pipeline/frontier.py` | `frontier.json`: `K`, `summary.joint_confidence` |
| Comparison rule: every rule calibrated by the same test on CAL at eps = delta = 0.05 | `pipeline/common_target.py` (`cast/compared.py:calibrate_common_target`) | `common_target.json`: `table`, `rules` |
| Latency on one pinned core, passes in rotated order; ratios within a timing window | `timing/time_rules.py window`, `timing/protocol.py`, `timing/README.md` | `<ds>_time.json` |
| "the staircase moves by up to 3.9% between the first three windows and by 9% in the fourth" | `reproduce/table3.sh` (windows 1 to 4) | ratios of `timing.cast.mean_us` between the files of the four windows; see "Not covered by a script" |
| "The list size is calibrated to ef = 662 and 319 ... and to 627 and 314 on a finer grid in the third window" | `baselines/list_size.py native` / `fine` and `report` (`reproduce/table3.sh`); 662 and 319 also by `pipeline/common_target.py` from the log | `RES/list_size_native_<ds>_A.json`: `list_size`; `RES/list_size_fine_<ds>_A.json`: `list_size`; `M/<s>/common_target.json`: `table.list_size.param` |
| "On 10,000 EVAL queries of Amazon and arXiv, the online stopping point of each rule checked equals the offline one on all but at most five queries, and exactness always agrees" | `timing/export_bundle.py --queries eval`, then `python timing/time_rules.py check-replay --bundle WORK/timing/<ds>_A_eval --data DATA --work WORK --out WORK/timing/<ds>_check_replay.json` (rules of the bundle, all timing queries): `stopping_point[rule].{visits_equal, exactness_equal}` (the shares of equal stopping points and equal exactness); Patience and the distance-adaptive rule: `timing/time_rules.py check-visits` on the first 1,000 queries | `patience_equal`, `distance_adaptive_equal` (share of queries with the same visit count as the definition) |
| "the stopping point differs on 1 of the 30,000 EVAL and HOLD queries of Amazon and on none of arXiv" | `check-replay` (10,000 EVAL) and `timing/time_rules.py window` on the HOLD bundle (20,000 HOLD) | `stopping_point[cast and frontier_0.05].visits_equal`; window file: `visits_equal_replay` |
| 0.016% bound on the probability of a difference; "raises the ratio by at most 0.004" | `python analysis/deployment_margin.py --exact RES/k10_kappa1/<ds>_<graph>/exact.json --differ 1 --checked 30000` | printed `p`, `increase` |
| "Significantly slower": one-sided paired bootstrap at 2.5%; in the third and fourth windows, Bonferroni over the ratios of the window | `timing/time_rules.py window` (`WINDOWS`: family 12 in window 3, 4 in window 4); `timing/protocol.py` | `timing[rule].ci95`, `timing[rule].lower_bound_one_sided_bonferroni` (windows 3 and 4) |
| "the dynamic programs on SEL, the calibration on CAL and the bounds on EVAL take 12 to 13 minutes per setting" | `pipeline/frontier.py` | elapsed seconds printed at the start of every log line; see "Not covered by a script" |

## VI-B Experimental Results and Analysis

### Near-optimality

| Paper | Script and command | Output and field |
|---|---|---|
| "the maximum bound U/L of each setting ranges from 1.082 to 1.128 at joint level 87.5%" (Fig. 3(a)) | `reproduce/fig3.sh`: `analysis/frontier_readouts.py` on the four settings | `RES/frontier_readouts.json`: `ranges.max_ratio` (per setting `settings.<s>.max_ratio`, from `exact.json` rounded up) |
| "At 95%, with delta = 0.02, alpha_U = 0.01 and alpha_L = 0.02, these maxima rise by at most 0.004, to 1.132, and the rule calibrated at 0.05 stays the same" | `python analysis/run_variants.py --red RED --out-dir RES/variants --groups main level95`; `python analysis/variant_summary.py RES/variants --out RES/variants_summary.json` (runs `pipeline/frontier.py --config configs/frontier_95.json`) | `RES/variants_summary.json`: `reported.level95.{max_ratio, max_increase_rounded_up, mean_visits_005_unchanged}` (float ratios `bounds[eps].ratio_chain`) |
| "These bounds are for the rule replayed on the logged searches" | no number | |
| "With three further disjoint FIT samples of 2,500 queries on graph A ... it is 1.085 to 1.142" | `analysis/run_variants.py ... --groups main fit` (`pipeline/frontier.py --fit-block 1..3`); `analysis/variant_summary.py` | `reported.fit_samples.max_ratio` (rounded up) |
| "Over ten further random splits of SEL, CAL and EVAL on graph A ... 1.094 to 1.223 on Amazon and 1.070 to 1.165 on arXiv. Its medians are 1.145 and 1.100" | `analysis/run_variants.py ... --groups resplit` (`pipeline/frontier.py --resplit-seed 1..10`); `python analysis/resplit_summary.py RES/variants --out RES/resplit_summary.json` | `<ds>.max_over_report_levels` (min, max), `<ds>.median_max` (printed rounded up) |
| "against 1.112 and 1.082 on the reported split" | `reproduce/fig3.sh` | `RES/frontier_readouts.json`: `settings.amazon_A.max_ratio`, `settings.arxiv_A.max_ratio` |
| "In every split the failure rate stays below the target at all nine levels" | `analysis/resplit_summary.py` | `<ds>.fail_rate_over_target` (maximum below 1), `<ds>.all_calibrated` |
| "The failure rate on EVAL stays at 75% to 97% of the target in every setting" | `analysis/frontier_readouts.py` | `ranges.fail_rate_over_target` |
| Fig. 3(b), "the rule stays close to both the empirical optimum and the lower bound from 0.03 upward" | `reproduce/fig3.sh` | see Fig. 3 below |
| "At 0.02 the ratio is 1.076 to 1.161 ... at 0.01 it is 1.124 to 1.572, the largest on Amazon graph B" | `analysis/frontier_readouts.py` | `ranges.ratio_0.02`, `ranges.ratio_0.01`; per setting `settings.<s>.ratio` |
| "no rule of the class with risk at most 0.05 saves more than a fraction 1 - L/U ... 6.0% to 8.0% by setting" | `analysis/frontier_readouts.py` | `ranges.saving_bound_005` (from `exact.json`: `EVAL.rules.0.05.{U_up6, L_down6}`) |
| Split of the gap U - L: "Over the 28 pairs of setting and level from 0.03 to 0.10, the cost bound, the lower bound and the displacement of the risk budget take 72.5% to 94.2% of the gap ... The rule against the empirical frontier takes 4.7% to 24.2% and the multiplier grid at most 4.5%" | per setting `python analysis/gap_anatomy.py --red RED --ds <ds> --graph <g> --frontier M/<s>/frontier.json --out RES/anatomy_<s>.json`; then `python analysis/gap_anatomy_summary.py amazon_A=RES/anatomy_amazon_A.json amazon_B=... arxiv_A=... arxiv_B=... --out RES/anatomy_summary.json` | `share_displacement_cost_bound_lower_bound`, `share.rule_vs_frontier`, `share.multiplier_grid`, `pairs` (28) |
| "At 0.01 on Amazon graph B, the mixture sequence lowers the ratio to 1.32 and moves the other three settings by at most 0.03" | `analysis/gap_anatomy_summary.py` | `ratio_001.<s>.{chain, mixture}` |
| "At 0.05 it [a clairvoyant rule] is 2.8 to 3.3 times cheaper than the empirical class optimum" | per setting `python analysis/clairvoyant.py --red RED --ds <ds> --graph <g> --frontier M/<s>/frontier.json --out RES/clairvoyant_<s>.json` | `levels.0.05.optimum_over_clairvoyant`, range over the four settings |

### Workload breadth

| Paper | Script and command | Output and field |
|---|---|---|
| "Outside YFCC-10M, the ratio in Table II stays at most 1.181 at every level from 0.03 to 0.10, and at most 1.110 on the two Vamana graphs" | `reproduce/table2.sh` (`tables/table2.py`) | `RES/table2.txt`: JSON summary `<row>.max_ratio`; maximum over the rows outside YFCC-10M, and over the two Vamana rows |
| "Title queries need 1.28 and 1.94 times the visits of item queries at 0.05" | `reproduce/table2.sh` | `<row>.cast_visits` of the rows "title queries" over those of the rows "Amazon, A" and "arXiv, A" |
| "yet refit on them the ratio stays at most 1.114" | `reproduce/table2.sh` | `max_ratio` of the two title rows |
| "our rule is the cheapest in eleven of the thirteen settings" | `reproduce/table2.sh` | number of the 13 rows below the main rows in which every `<row>.<rule>.ratio` exceeds 1 (JSON summary of `tables/table2.py`) |
| "the distance-adaptive rule ... is 2.7% and 0.2% cheaper at a similar failure rate" (unconstrained Amazon and arXiv) | `reproduce/table2.sh` (`baselines/distance_adaptive.py calibrate`) | `RES/k10_unconstrained/<ds>_A/distance_adaptive.json`: `table.distance_adaptive.{visit_ratio.ratio, fail_rate}` against `table.cast.fail_rate` |
| "On YFCC-10M, 3.9% of the queries do not complete within 32,768 visits, so we evaluate the levels from 0.05 to 0.10 only" | `reproduce/table2.sh`: `pipeline/completion_share.py --layout parts --rule margin --config configs/frontier.json --out-config ...` | `RES/k10_kappa1/yfcc10m_A/completion.json`: `share_incomplete`, `levels`; `config.json`: `levels` |
| "There the ratio is 1.364 and 1.372 at 0.05, ... and it falls to 1.124 and 1.125 at 0.10" | `python analysis/frontier_readouts.py yfcc_tags=M/yfcc10m_A/frontier.json,M/yfcc10m_A/exact.json yfcc_unc=RES/k10_unconstrained/yfcc10m_A/frontier.json,RES/k10_unconstrained/yfcc10m_A/exact.json --out RES/frontier_readouts_yfcc.json` | `settings.<label>.ratio.0.05`, `settings.<label>.ratio.0.1` |
| "On 10,000 EVAL queries of YFCC-10M with tags, the first group takes 1.2% to 22.7% more time than our rule" | `python timing/export_bundle.py --red RED --data DATA --ds yfcc10m --graph A --layout parts --frontier M/yfcc10m_A/frontier.json --queries eval --out WORK/timing/yfcc10m_A_eval`; `python timing/short_list.py calibrate --red RED --ds yfcc10m --graph A --layout parts --frontier M/yfcc10m_A/frontier.json --seed 2029968942 --export-eval --out WORK/timing/yfcc10m_lists`; `python timing/short_list.py time --bundle WORK/timing/yfcc10m_A_eval --export WORK/timing/yfcc10m_lists_export.npz --list-size 1145 --seed 1942499709 --data DATA --work WORK --out WORK/timing/yfcc10m_short` | `WORK/timing/yfcc10m_short/time.json`: `timing.{level_cap, level, laet_style, fixed}.ratio_vs_cast` |
| "The list size takes 12.1% less, at 0.18 us per visit against our 0.24 us on the list of 8,192" | same `short_list.py time` | `timing.list_size.ratio_vs_cast`, `timing.list_size.us_per_visit`, `timing.cast.us_per_visit`; list size 1,145: `M/yfcc10m_A/common_target.json`: `table.list_size.param` |
| "With a list of 2,048, chosen among four sizes without reading CAL and with both bounds split over the four ... Its bound is 1.402 at 0.05 and 1.129 at 0.10" | `python pipeline/truncate_list.py --work WORK --name k10_kappa1 --out-name k10_kappa1_list2048 --ds yfcc10m --graph A --layout parts --list-size 2048`, then `pipeline/frontier.py --layout parts` on `WORK/reduced/k10_kappa1_list2048` | `--config configs/frontier_yfcc10m_list2048.json` (alpha_U and alpha_L split over the four list sizes), then `pipeline/exact_bounds.py`: `EVAL.rules[eps].ratio_up6` |
| "... our staircase is within 1% of the list size in time" | `timing/short_list.py calibrate --red WORK/reduced/k10_kappa1_list2048 --stops RED --ds yfcc10m --graph A --frontier <frontier of the truncated arrays> --layout parts --export-eval --seed SEED --out OUT` (the seed of the reported run is not recorded), then `short_list.py time ... --seed 640375004` without `--require-same-staircase` (`timing/README.md`) | `time.json`: `summary.list_size_over_list2048` |
| "On the main workload, the same change makes our rule 1.2% and 2.2% slower on HOLD" | `python timing/short_list.py calibrate --red RED --ds <ds> --graph A --seed <1124647655 Amazon, 2048718251 arXiv> --export-hold --out WORK/timing/<ds>_lists`; `python timing/short_list.py time --bundle WORK/timing/<ds>_A_hold --export WORK/timing/<ds>_lists_hold_export.npz --lists 2048 1024 --bundle-rules level_cap --list-size <662, 319> --seed <413869110, 1066157808> --max-differ 5 --require-same-staircase --data DATA --work WORK --out WORK/timing/<ds>_short` | `time.json`: `summary.list2048_over_cast` |
| "With k = 50, 15.8% and 30.2% of the queries of Amazon and arXiv do not complete within the horizon" | `pipeline/exact_answers.py --workload k50_kappa1`, `pipeline/log_trajectories.py --workload k50_kappa1`, `pipeline/reduce_logs.py --workload k50_kappa1 --k 50` (groups `fitpool chunk0 ... chunk19`, graph A); `python pipeline/completion_share.py --red WORK/reduced/k50_kappa1 --ds <ds> --graph A --out RES/k50_<ds>.json` | `share_incomplete_eval` |

### Recall target

| Paper | Script and command | Output and field |
|---|---|---|
| "our rule is again the cheapest, and every other rule of Table II is significantly costlier at eps = 0.05" | `reproduce/table2.sh` (recall rows: `pipeline/frontier.py --target X_recall09`, `common_target.py`, `distance_adaptive.py calibrate`; `analysis/significance.py`) | `RES/table2_significance.json`: `rows.Recall, ...[rule].costlier`; `RES/table2.txt`: recall rows |
| "the ratio stays at most 1.106 at every level from 0.03 to 0.10" | `reproduce/table2.sh` | `max_ratio` of the four recall rows |
| "at 0.05 the staircase needs less than half of its exact-answer visits" | `reproduce/table2.sh` | `cast_visits` of each recall row over that of the main row of the same setting |
| "On graph A, DARTH and DARTH+ ... spend 1.23 to 1.60 times our visits under the same test" | `python baselines/recall_target.py --data DATA --work WORK --ds <ds> --graph A --common-target RES/k10_kappa1_recall09/<ds>_A/common_target.json --out RES/recall_target_<ds>_A.json` (models of `reproduce/fig4.sh`) | `visits_over_cast` (DARTH and DARTH+ at its three quantile levels), range over both datasets |

### Cost and latency at a common risk target

| Paper | Script and command | Output and field |
|---|---|---|
| "our rule has the lowest mean latency in every timing window"; "every other calibrated rule takes significantly more mean time than ours, by 4.8% to 45.5%" (also abstract and contributions) | `reproduce/table3.sh` (`timing/time_rules.py window` 1 to 4, `timing/make_table3.py`) | `RES/table3.txt`: last line "ratios of the other rules over all windows read"; per rule `timing[rule].{ratio, ci95}` |
| "the stopping check of the staircase, score model included, takes 7.5% and 7.1% of its query time" | `reproduce/table3.sh`: `timing/time_rules.py share --rule cast` | `WORK/timing/window3/<ds>_share_cast.json`: `check_share` |
| "At 16 threads, our throughput over that of each rule of the third window is 1.07 to 1.66, within 0.06 of the visit ratio" | `reproduce/table3.sh`: `timing/throughput.py` | `WORK/timing/<ds>_throughput.json`: `per_thread.16[rule].qps_cast_over_rule`; compared with `timing[rule].visit_ratio` of window 3 |
| Medians: the distance-adaptive rule and, on Amazon, Level+cap have a lower median; 99th percentiles 3,423 and 3,417 us, 2,670 and 1,811 (list size), 3,033 and 2,682 (Patience), 2,449 (Level+cap, arXiv) | `reproduce/table3.sh` (window 3) | `WORK/timing/window3/<ds>_time.json`: `timing[rule].{p50_us, p99_us}` (printed by `timing/make_table3.py`) |
| "In visits, our rule is the cheapest at every level on graph A" (Fig. 4) | `reproduce/fig4.sh` | values printed by `figures/fig4_visit_ratios.py` |
| "On graph B, the six rules of Table II calibrated the same way spend at least 2.5% more visits at every level from 0.02 upward" | `reproduce/fig4.sh` (`baselines/levels.py` and the fine-grid distance-adaptive file on graph B) | `RES/levels/levels_<ds>_B_<i>.json`: `levels[eps].<rule>.mean_visits` over `levels[eps].frontier.mean_visits`; `RES/levels/distance_adaptive_<ds>_B.json`: `levels[eps].visits_over_frontier` |
| "Cheaper rules appear only at 0.01 on Amazon graph B, where the distance-adaptive rule spends 0.80 times our visits and Level and Level+cap 0.84 times" | same | same fields at `0.01` |
| "On graph A, the gap to Fixed is largest at 0.01" | `reproduce/fig4.sh` | `levels[eps].fixed.visit_ratio.ratio` of graph A |
| "Interpolated to equal EVAL failure rates from 0.03 to 0.08, the first group still spends 1.04 to 1.68 times our visits and the distance-adaptive rule 1.08 to 1.44 times. Patience, swept over its parameter, spends 1.17 to 1.40 times, and DARTH and DARTH+ on graph A 1.28 to 1.63 times" | `reproduce/fig4.sh`: `baselines/patience.py points`, `baselines/equal_failure_rate.py` | `RES/levels/equal_failure_rate.json`: `summary.{first_group, distance_adaptive, patience, darth, darth_plus}.{min, max}` (DARTH and DARTH+ together) |
| "On all EVAL queries, Patience and DARTH+ spend 1.32 to 1.52 times our visits wherever we run them, each significantly more" | Patience, four settings: `python baselines/patience.py calibrate --red RED --ds <ds> --graph <g> --frontier M/<s>/frontier.json --stops WORK/patience --seed SEED --out RES/patience_<s>.json` (`configs/seeds.json` has no entry for this seed; it changes only the bootstrap bound); DARTH+, graph A: `python baselines/darth_plus.py compare --data DATA --work WORK --ds <ds> --graph A --red RED --frontier M/<ds>_A/frontier.json --darth RES/darth_<ds>_A.json --run RES/darthplus_levels_<ds>_A --seed <303682605 Amazon, 118077128 arXiv> --out RES/darthplus_compare_<ds>_A.json` | `patience.visit_ratio.{ratio, lower_bound_one_sided}`; `darth_plus_over_cast.{ratio, lower_bound_one_sided}` |
| "Ada-ef passes the test only with its table built on all of SEL, at 1.49 to 1.81 times our visits" | per dataset, graph A: `python baselines/ada_ef.py stats --data DATA --work WORK --ds <ds> --graph A --out RES/adaef_stats_<ds>.npz`; `python baselines/ada_ef.py evaluate --data DATA --work WORK --red RED --ds <ds> --graph A --stats RES/adaef_stats_<ds>.npz --out RES/adaef_<ds>.npz` (default `--build-queries 20000`: all of SEL; `--build-queries 200` for the table size of the method); `python baselines/ada_ef.py compare --red RED --ds <ds> --graph A --frontier M/<ds>_A/frontier.json --run RES/adaef_<ds>.npz --seed SEED --out RES/adaef_compare_<ds>.json` (`configs/seeds.json` has no entry for this seed; it changes only the bootstrap bound) | `ada_ef_l1025.visit_ratio.ratio`, `ada_ef_l4097.visit_ratio.ratio`; the calibration in `ada_ef_meta` |
| "At eps = 0.05 on Amazon, the calibrated staircase stops only in buckets up to 20 until 702 visits ... On arXiv it never stops before 1,318 visits" | `python baselines/threshold_curve.py attribution --red RED --ds <ds> --graph A --curve RES/curve_<ds>_A.json --out RES/curve_attribution_<ds>_A.json` (curve from `reproduce/table3.sh`) | `table_cast` (thresholds per checkpoint and flag) with `grid`; the two visit counts are read from this table |
| "Fit on SEL and calibrated by the same test, it spends 0.976 to 1.074 times our visits over ten random splits per setting, at failure rates within 0.3 points. Its median per setting is 1.012 to 1.053, and it recovers a median of 41% to 85% of the gap to Level+cap" | per setting and seed 1 to 10: `python baselines/threshold_curve.py fit --red RED --ds <ds> --graph <g> --split-seed <k> --out RES/splits/<ds>_<g>_<k>.json`; `python baselines/threshold_curve.py summary --splits RES/splits --out RES/curve_splits.json` | `<ds> <g>.{ratio_min, ratio_max, ratio_median, recovered_median, gap_abs_max}` |
| "The rest of this gap comes from long queries, where the thresholds of the chain rise faster than the single slope of the curve" | `baselines/threshold_curve.py attribution` | `cells["cast_first|a=1|[4096,H]"].share_of_cast_first` (full answer, after 4,096 visits) |
| "On HOLD, the curve of the reported split takes 5.8% and 6.1% more mean time than ours, within 0.004 of its visit ratio" | `reproduce/table3.sh` (window 4) | `WORK/timing/window4/<ds>_time.json`: `timing.curve.ratio`, `curve.time_minus_visit_ratio` |
| "the reported split favors our rule only on Amazon graph A, where its EVAL visit ratio of 1.059 exceeds all ten" | `baselines/threshold_curve.py fit` without `--split-seed` (`reproduce/table3.sh`), and `summary` | `RES/curve_<s>.json`: `rules.curve.over_cast` against `ratio_max` of the same setting in `RES/curve_splits.json` |

### Held-out check

| Paper | Script and command | Output and field |
|---|---|---|
| "no test rejects the risk target of the frontier rule in any setting or at any level" | `reproduce/main_setting.sh` (`pipeline/holdout.py`), `analysis/holdout_joint_test.py` | `M/<s>/holdout.json`: `frontier[eps].reject` |
| "the rule at eps = 0.05, calibrated at level delta / (2M), fails on at most 995 of the 20,000 HOLD queries. An exact binomial test at level 0.05/4 would need 1,071 failures" | `pipeline/holdout.py` | `frontier.0.05.{fail, critical}`, `n_hold` |
| "a joint test of the 36 pairs of setting and level at level 0.05/36 each rejects none. Recomputed on HOLD alone ... the bound is at most 1.217 from 0.03 to 0.10 and 1.897 at 0.01" | `python analysis/holdout_joint_test.py amazon_A=M/amazon_A/holdout.json amazon_B=M/amazon_B/holdout.json arxiv_A=M/arxiv_A/holdout.json arxiv_B=M/arxiv_B/holdout.json --out RES/holdout_joint.json` | `reject_any`, `level_each`, `max_ratio_hold_003_010.rounded_up`, `max_ratio_hold_001.rounded_up` (from `holdout.json`: `hold_bounds[eps].ratio_hold`) |
| "the Maurer-Pontil lower bound on the mean HOLD cost at level 0.05/4 stays below U" | `pipeline/holdout.py` | `frontier[eps].{cost_lower_bound_hold, cost_lower_bound_below_U}` |
| "When the rule of Table III fails, most of its answers miss exactly one of the ten members, at 0.7% and 0.4% more total distance on average" | `reproduce/table3.sh`: `timing/failed_answers.py` | `WORK/timing/<ds>_failed_answers.json`: `rules.cast.{share_missing_one, dist_ratio_mean}` |

### Query shift and index updates

| Paper | Script and command | Output and field |
|---|---|---|
| Fig. 5 and "rejects the target at 43 of the 54 levels of the title queries and the two replacements. Inserts leave the rules within the target" | `reproduce/fig5.sh` (`analysis/shift_title.py`, `analysis/shift_updates.py`, `analysis/shift_summary.py`) | `RES/shift/summary.json`: `total.rejections_before`, `total.levels`, `readouts.<change>_<ds>.rejections_before` |
| "Title queries complete later" | `python analysis/title_completion.py --red RED --shift-red RED --ds amazon arxiv --graph A --out RES/shift/title_completion.json` | `<ds>.<part>.{item, title}.X_p50`, `share_title_later` |
| "replacement prunes the bottom layer, whose mean degree falls by 10% and 25%" | `python pipeline/build_graphs.py degree --data DATA --work WORK --ds <ds> --graph A` and the same with `--ds <ds>_replace1`, `--ds <ds>_replace5` | printed `mean_degree_bottom_layer`; the drop is their ratio |
| "Recalibration uses 40,000 CAL queries labeled and logged after the change, and the chain returns no rule only for title queries at 0.01" | `reproduce/fig5.sh` | `RES/shift/summary.json`: `readouts.<change>_<ds>.no_recalibrated_rule`, `total.rejections_recalibrated`; `n_cal` of each readout |
| "a Kolmogorov-Smirnov distance on the stopping checkpoints of 1,000 queries, thresholded for 5% false alarms, flags 54% to 100% of the title and replacement batches" | `python analysis/drift_signal.py --kind title --red RED --ds <ds> --graph A --frontier M/<ds>_A/frontier.json --shift-red RED --seed 0x13528f3 --out RES/shift/drift_title_<ds>.json`; `--kind update --shift-red RED --shift-ds <ds>_<replace1, replace5>` likewise | `ks_detect`, `ks_ref_alarm`, `ks_threshold` |
| "After updates, the recalibrated rule spends 0.92 to 1.03 times the visits of a full refit on FIT, SEL and CAL samples logged on the updated index" | per update: `python pipeline/frontier.py --red RED --ds <ds>_<u> --graph A --config configs/frontier.json --out RES/shift/refit_<u>_<ds>.json`; `python analysis/updates_summary.py <u>_<ds>=RES/shift/<u>_<ds>.json,RES/shift/refit_<u>_<ds>.json ... --out RES/shift/updates_summary.json` | `recalibrated_over_refit` |
| "The official queries of YFCC-10M give a second shift, where the constraint changes the answer of 1.6% of the queries" | `pipeline/exact_answers.py --summary` (`reproduce/table2.sh`) | `RES/k10_kappa1/yfcc10m_official_answers.json`: `groups.official_eval.share_differs_from_unconstrained` |
| "Rules calibrated on held-out queries keep the target at all five levels on the official EVAL queries at 2.0% to 2.8% higher cost" | `python analysis/shift_official.py --red RED --ds yfcc10m --graph A --frontier M/yfcc10m_A/frontier.json --shift-red WORK/reduced/k10_kappa1_official --out RES/shift/official.json` | `summary.rejections_before`, `summary.extra_cost_before` |
| "With all samples from the official queries, the ratio is 1.177 at 0.05 and 1.084 at 0.10" (1.372 and 1.125 with held-out queries, Workload breadth) | `reproduce/table2.sh`, then `python analysis/frontier_readouts.py official=RES/k10_kappa1_official/yfcc10m_A/frontier.json,RES/k10_kappa1_official/yfcc10m_A/exact.json tags=M/yfcc10m_A/frontier.json,M/yfcc10m_A/exact.json --out RES/frontier_readouts_official.json` | `settings.<label>.ratio.{0.05, 0.1}` |

### Sample size and labeling cost

| Paper | Script and command | Output and field |
|---|---|---|
| "With 10,000 instead of 40,000 CAL queries, the rule at eps = 0.05 spends 2.3% to 5.6% more visits" (and "a quarter of the CAL queries costs at most 5.6% more visits") | per setting `python analysis/sample_size.py run --red RED --ds <ds> --graph <g> --frontier M/<s>/frontier.json --out RES/sample_size_<s>.json`; `python analysis/sample_size.py summary amazon_A=RES/sample_size_amazon_A.json amazon_B=... arxiv_A=... arxiv_B=... --out RES/sample_size.json` | `ranges.extra_visits_cal_10000_vs_40000` |
| "with 30,000 instead of 120,000 EVAL queries its ratio grows by 0.036 to 0.044" | same | `ranges.ratio_growth_eval_30000_vs_120000` |
| "Per 10,000 sampled queries over one million items, the exact answers and logged trajectories take under 5 seconds on one GPU and 50 to 90 seconds on one 32-core server" | `pipeline/exact_answers.py`, `pipeline/log_trajectories.py` | see "Not covered by a script" |
| "A deployment that needs no bound uses only FIT, SEL and CAL, 62,500 queries" | arithmetic on the sample sizes (`frontier.json`: `n`) | |

### Ablation study: Richer learned policies

| Paper | Script and command | Output and field |
|---|---|---|
| "The strongest richer policy, the fully fitted one, is at most 0.5% cheaper than our rule and at most 5.4% costlier, with a median of 1.6% more over four settings and seven risk levels" | per setting `python baselines/learned_policies.py --red RED --ds <ds> --graph <g> --out RES/policies_<s>.json`; `python baselines/learned_policies_summary.py RES/policies_amazon_A.json RES/policies_amazon_B.json RES/policies_arxiv_A.json RES/policies_arxiv_B.json --out RES/policies_summary.json` | `fully_fitted.{min, max, median}` |
| "calibrated on one half of CAL at level delta/M and are compared on the other half" | `baselines/learned_policies.py` (design; seeds `cal_halves` in `configs/seeds.json`) | |
| "the cross-fitted and declared-state policies cost a median of 1.6% to 3.5% and 1.2% to 2.8% more" | same summary | `cross_fitted.{min_of_medians, max_of_medians}`, `declared_state.{min_of_medians, max_of_medians}` |
| "When SEL shrinks from 20,000 to 5,000 queries, the out-of-sample Lagrangian value of the class changes by at most 1%" | same summary | `class_value_change_max` |
| "the gap of the cross-fitted policy to the class ... grows from between 5.6% and 7.2% to between 18.6% and 26.7%" | same summary | `cross_fitted_over_class_20000`, `cross_fitted_over_class_5000` |

### Ablation study: Interface and lower bound

| Paper | Script and command | Output and field |
|---|---|---|
| "with 40, 60 or 200 buckets, J = 50 or 25, or a logistic-regression score ... costs between 0.8% less and 2.5% more than with the main interface, and the ratio stays at most 1.134 at every level from 0.03 to 0.10" | `python analysis/run_variants.py --red RED --out-dir RES/variants --groups main interface` (`pipeline/frontier.py` with `--buckets 25`, `--buckets 50`, `--buckets 200 --exact-levels`, `--stride 2`, `--stride 4`, `--model logreg`); `python analysis/variant_summary.py RES/variants --out RES/variants_summary.json` | `reported.interface.{mean_visits_005_vs_main, max_ratio}` (float ratios, rounded up) |
| "With the checkpoint and the flag alone, the empirical class optimum at 0.05 is 58% and 51% higher on Amazon and arXiv, yet its bound of 1.098 and 1.099" | `analysis/run_variants.py ... --groups main flagonly` (`--buckets 1 --exact-levels`); `analysis/variant_summary.py` | `reported.flag_only.flagonly_<ds>_A.{empirical_optimum_005_vs_main, ratio_005}` |
| "From 0.03 to 0.10, a lower bound valid at one risk level only lies within 0.4% of the Lagrangian bound" | `reproduce/fig3.sh`: `analysis/frontier_readouts.py` | `ranges.single_level_over_L` (from `frontier.json`: `bounds[eps].{L_single_level, L}`) |

## Tables

| Element | Script | Output |
|---|---|---|
| Table I | `reproduce/table1.sh` (`data/dataset_stats.py table`) | `RES/table1.json`; caption sample sizes from `frontier.json`: `n` and `holdout.json`: `n_hold` |
| Table II, columns Fixed, LAET-style, Level, Level+cap, List size | `reproduce/main_setting.sh`, `reproduce/table2.sh` (`pipeline/common_target.py`, `tables/table2.py`) | `<setting>/common_target.json`: `table[rule].visit_ratio.ratio`; rows in `RES/table2.txt` |
| Table II, column Dist.-adaptive | `reproduce/table2.sh` (`baselines/distance_adaptive.py relog`, `calibrate`; read by `tables/table2.py` from `<setting>/distance_adaptive.json`) | `table.distance_adaptive.visit_ratio.ratio` |
| Table II, column CaST visits | `pipeline/common_target.py` | `table.cast.mean_visits` |
| Table II, columns U/L max and U/L at 0.05 | `pipeline/exact_bounds.py`, rounded up by `tables/table2.py` | `exact.json`: `EVAL.rules[eps].ratio_up6`, maximum over the calibrated levels from 0.03 to 0.10 |
| Table II, bold and underline | `RES/table2.txt` | smallest and second smallest of the ratio cells and 1 (CaST) per row |
| Table II, caption "every ratio above one is significant at level 0.025/102" | `reproduce/table2.sh`: `analysis/significance.py --level 0.025 --family 102` | `RES/table2_significance.json`: `summary.{above_one, above_one_significant}` |
| Table III, all cells (us, visits, fail %, ratio per window) | `reproduce/table3.sh` (`timing/time_rules.py window --window 1..4`, `timing/make_table3.py`) | `WORK/timing/window<k>/<ds>_time.json`: `timing[rule].{mean_us, mean_visits, fail_rate, ratio}`; rows in `RES/table3.txt` |
| Table III, caption "Fail ... stays below 5.26, where an exact binomial test at level 0.05 rejects the target" | `timing/time_rules.py window` | `fail_count_rejecting_target` (over `n` = 20,000) |

## Figures

| Element | Script | Inputs and fields |
|---|---|---|
| Fig. 3(a) | `reproduce/fig3.sh`: `figures/fig3_near_optimality.py --exact-bounds <four exact.json> --frontier <two frontier.json of graph A> --out RES/nearopt.pdf` | `exact.json`: `EVAL.rules[eps].ratio_up6` (rounded up) |
| Fig. 3(b) | same | `exact.json`: `EVAL.L_curve`; `frontier.json`: `rules.chain[eps].mean_visits`, `bounds[eps].empirical_optimum` |
| Fig. 4 | `reproduce/fig4.sh`: `figures/fig4_visit_ratios.py` with the outputs of `baselines/levels.py`, `baselines/distance_adaptive.py calibrate`, `baselines/darth.py report`, `baselines/darth_plus.py report` (graph A) | `levels[eps].<rule>.mean_visits` over `levels[eps].frontier.mean_visits`; `levels[eps].rule.mean_visits` |
| Fig. 5 | `reproduce/fig5.sh`: `figures/fig5_shift.py` with the outputs of `analysis/shift_title.py` and `analysis/shift_updates.py` | `before_recalibration[eps].{fail_rate, reject}`, `recalibrated[eps].{calibrated, fail_rate}` |

## Abstract, introduction and contributions

| Paper | Script | Output and field |
|---|---|---|
| Abstract: "costs at most 1.132 times the class optimum at risk levels 0.03 to 0.08 and 0.10, at joint confidence 95% per setting" (also contributions, "costs at most 1.132 times the class optimum at risk levels from 0.03 to 0.10, at confidence 95%") | `analysis/run_variants.py --groups main level95`, `analysis/variant_summary.py` | `reported.level95.max_ratio` (rounded up) |
| Abstract and contributions: "every other calibrated rule we time takes 4.8% to 45.5% more mean time" | `reproduce/table3.sh` | `RES/table3.txt`, last line |
| Introduction: "the budget calibrated to risk eps = 0.05 is 6,781 (5,978) visits, i.e., 7.1x (5.8x) the median completion time over 10,000 evaluation queries" (Amazon first, graph A) | `python analysis/intro_budget.py --red RED --ds amazon arxiv --graph A --out RES/intro_budget.json` | `<ds>.budget`, `<ds>.budget_over_median` |
| Introduction: "DARTH, configured for a target average recall of 0.95, returns the exact constrained answer for only 71% (68%) of queries" | `baselines/recall_target.py` (DARTH run at the literal target 0.95 on the EVAL queries of graph A) | `rules.darth.uncalibrated_0.95.exact_share_first_10000` (first 10,000 EVAL queries) |
| Introduction: "under the cap of 2^15 = 32,768 visits, 0.10% and 0.58% of queries, depending on the dataset and graph, never complete" (the range over the four settings of the main workload, EVAL) | `reproduce/fig3.sh` (`analysis/frontier_readouts.py`) | `ranges.terminal_fail_eval` |
| Introduction: "approximately e^266 deterministic rules" | `python analysis/class_size.py --J 100 --B 100` | printed `ln N_Z` |
| Introduction: "a clairvoyant rule that knows each query's completion time is 2.8x to 3.3x cheaper than the empirical class optimum on the evaluation queries" (risk 0.05) | `analysis/clairvoyant.py` (four settings) | `levels.0.05.optimum_over_clairvoyant` |
| Introduction: "eight other rules calibrated to the same target spend 6.2% to 70.6% more visits than our calibrated staircase, which lies within 1.087x of the lower bound with 87.5% confidence" (risk 0.05, graph A) | `reproduce/fig4.sh`; `reproduce/table2.sh` | ratios at 0.05 printed by `figures/fig4_visit_ratios.py` (first group, list size, distance-adaptive rule, DARTH, DARTH+; both datasets); 1.087: Table II, column U/L at 0.05, row Amazon, A (arXiv, A: 1.064) |
| Contributions: "CaST costs at most 1.132 times the class optimum at risk levels from 0.03 to 0.10, at confidence 95%" | as the abstract | |
| Contributions: "this near-optimality carries over to most further sample splits, workloads and indexes" (Section VI-B: at most 1.223 over ten further random splits, at most 1.181 on the ten further settings outside YFCC-10M) | `analysis/resplit_summary.py`; `reproduce/table2.sh` | `<ds>.max_over_report_levels` (largest, rounded up), `<ds>.fail_rate_over_target`; `RES/table2.txt`: `max_ratio` of the ten rows outside the main workload and YFCC-10M |
| Contributions: "on held-out queries" (the 20,000 HOLD queries) | `timing/export_bundle.py --queries hold` | bundle JSON / `holdout.json`: `n_hold` |

## Numbers of Sections IV and V

| Paper | Script | Output field |
|---|---|---|
| Section IV-A: "ln N_Z is about 266" for J = 100 and B = 100 | `python analysis/class_size.py` | printed value |
| Section IV-B: the dynamic program and the linear program give the same value "to within 10^-4 visits" | `pipeline/frontier.py`; `analysis/frontier_readouts.py` | `dp_equals_lp[].{dp, lp, lp_dual}` |
| Section V-B: one cutting-plane iteration leaves a gap of "at most 0.07 visits" | `pipeline/frontier.py`; `analysis/frontier_readouts.py` | `ell[].trace[].gap` |
| Section V-D: "about 0.003 below eps" and "about 0.0003" | none | see "Not covered by a script" |

## Not covered by a script

Figs. 1 and 2 of the paper are schematic drawings and are not produced by code.

The numbers below are not produced by any script of the package, or only after a step the package does not contain.

1. **"the staircase moves by up to 3.9% between the first three windows and by 9% in the fourth"** (Implementation
   details). Arithmetic on `timing.cast.mean_us` of the window files (maximum over minimum of windows 1 to 3; window 4
   over window 3 on arXiv); no script forms these ratios.
2. **"take 12 to 13 minutes per setting on one server"** (Implementation details). Read from the elapsed time printed
   by `pipeline/frontier.py`; not stored in its output, and machine dependent.
3. **"under 5 seconds on one GPU and 50 to 90 seconds on one 32-core server"** (Sample size and labeling cost).
   Wall-clock times: `pipeline/log_trajectories.py` stores the seconds of each log file (`secs`), `pipeline/exact_answers.py`
   does not store its running time; no script aggregates them.
4. **The margins "about 0.003" and "about 0.0003"** of Section V-D: arithmetic on the sample sizes and error levels; no
   script computes them.
5. **Bootstrap seeds of the Patience and Ada-ef comparisons and of the calibration on the list of 2,048 entries**: the
   values of the reported runs are not recorded, so their bootstrap bounds reproduce only up to resampling.
6. **Ranges and counts read from several outputs**, each of whose values a script writes but whose aggregate no
    script prints: the maxima 1.181 and 1.110 and "eleven of the thirteen settings" (Workload breadth, from the JSON
    summary of `tables/table2.py`); the visit factors 1.28 and 1.94 and "less than half" (ratios of `cast_visits` of two
    rows); the graph-B readings and the factors 0.80 and 0.84 (Cost and latency, from `baselines/levels.py` and the
    distance-adaptive levels); "6.2% to 70.6%" (from the values printed by `figures/fig4_visit_ratios.py`); the range
    1.32 to 1.52 for Patience and DARTH+ (two scripts); "within 0.06 of the visit ratio" (throughput against window 3);
    "1.059 exceeds all ten" (fit output against the split summary); the degree drop of 10% and 25% (ratio of two
    printed degrees); the visit counts 702 and 1,318 (read from the table `table_cast`); the clairvoyant factor 2.8 to
    3.3 and Ada-ef 1.49 to 1.81 (ranges over per-setting files); 62,500 (sum of sample sizes).

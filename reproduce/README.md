# Reproduction scripts

One POSIX shell script per table and figure of Section VI. Each sets `DATA` (prepared data, `data/README.md`) and `WORK`
(graphs, logs, reduced arrays, results) at the top, together with the Python interpreters of the three environments
(`PY`, `PY_ENGINE`, `PY_DATA`), the engine build directory and the recorded seeds; all can be overridden from the
environment. Results go to `WORK/results/`. `docs/PAPER_MAP.md` lists every reported number with its command and field.

| Order | Script | Reproduces | Needs before it |
|---|---|---|---|
| 0 | `engine/build.sh`, scripts of `data/` | engine module and tools; prepared data sets | |
| 1 | `main_setting.sh` | one setting of the main workload: exact answers, graph, logs, reduced arrays, frontier rules, exact bounds, common-target table, held-out check. Run it for amazon and arxiv on graphs A and B (`DS`, `GRAPH`, `GRAPH_SEED`, `CT_SEED`) | 0 |
| 2 | `table1.sh` | Table I and the share of constrained answers that differ from the top 10 | 0; 1 for the share |
| 3 | `fig3.sh` | Fig. 3 and the readouts of the paragraph "Near-optimality" taken from the same files | 1 (four settings) |
| 4 | `table2.sh` | Table II: the thirteen further settings, the recall target, the fine-grid distance-adaptive cells, the rows and the significance test of the caption | 1 (four settings) |
| 5 | `fig4.sh` | Fig. 4, the same readings on graph B, the comparison at equal failure rates | 1, 4 |
| 6 | `table3.sh` | Table III (four timing windows), share of the stopping check, throughput, failed answers | 1, 4, 5 |
| 7 | `fig5.sh` | Fig. 5 and the counts of the paragraph "Query shift and index updates" | 1 (graph A) |

Timing (`table3.sh`) must run on one pinned core with its hyperthread sibling idle (`timing/README.md`); set `CORE`.

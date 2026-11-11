"""Launch pipeline/frontier.py on the main settings under other error levels, FIT samples, interfaces and sample splits.

Paper: Section VI-B, paragraph "Near-optimality" (joint level 95%; three further FIT samples; ten random re-splits) and
the ablation "Interface and lower bound". Every run is pipeline/frontier.py with the flags below (the re-splits go
through its option --resplit-seed); outputs are written to OUT_DIR/<run>_<ds>_<graph>.json and summarized by
analysis/variant_summary.py and analysis/resplit_summary.py.

    run          settings          flags
    main         graphs A and B    --config configs/frontier.json                  (joint level 87.5%)
    level95      graphs A and B    --config configs/frontier_95.json               (delta 0.02, alpha_U 0.01, alpha_L 0.02)
    fit1..fit3   graph A           --fit-block 1..3                                 (three further disjoint FIT samples)
    buckets40    graph A           --buckets 25      (edges at the quantiles k/25 and at the multiples of 0.05: 40 buckets)
    buckets60    graph A           --buckets 50      (edges at the quantiles k/50 and at the multiples of 0.05: 60 buckets)
    buckets200   graph A           --buckets 200 --exact-levels
    J50, J25     graph A           --stride 2, --stride 4                           (every second or fourth checkpoint)
    logreg       graph A           --model logreg                                   (logistic-regression score)
    flagonly     graph A           --buckets 1 --exact-levels                       (the rule reads the checkpoint and the flag only)
    resplit<s>   graph A           --resplit-seed s, s = 1..10

Runs with an existing complete output are skipped. Each run uses the threads the numerical libraries are given
(OMP_NUM_THREADS); the reported runs used four threads per run.

Usage:
    python analysis/run_variants.py --red RED_DIR --out-dir OUT_DIR [--groups main level95 fit interface flagonly resplit] [--jobs 6]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
FRONTIER = os.path.join(ROOT, 'pipeline', 'frontier.py')
CFG = os.path.join(ROOT, 'configs', 'frontier.json'); CFG95 = os.path.join(ROOT, 'configs', 'frontier_95.json')
RESPLIT_SEEDS = list(range(1, 11))
INTERFACE = [('buckets40', ['--buckets', '25']), ('buckets60', ['--buckets', '50']), ('buckets200', ['--buckets', '200', '--exact-levels']),
             ('J50', ['--stride', '2']), ('J25', ['--stride', '4']), ('logreg', ['--model', 'logreg'])]


def job_list(datasets, groups, resplit_seeds):
    """[(run name, graph, script, extra flags)] per dataset."""
    jobs = []
    for ds in datasets:
        for g in ('A', 'B'):
            if 'main' in groups: jobs.append((f'main_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG]))
            if 'level95' in groups: jobs.append((f'level95_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG95]))
        g = 'A'
        if 'fit' in groups: jobs += [(f'fit{b}_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG, '--fit-block', str(b)]) for b in (1, 2, 3)]
        if 'interface' in groups: jobs += [(f'{n}_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG] + fl) for n, fl in INTERFACE]
        if 'flagonly' in groups: jobs.append((f'flagonly_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG, '--buckets', '1', '--exact-levels']))
        if 'resplit' in groups: jobs += [(f'resplit{s}_{ds}_{g}', ds, g, FRONTIER, ['--config', CFG, '--resplit-seed', str(s)]) for s in resplit_seeds]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays of the main workload')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ds', nargs='+', default=['amazon', 'arxiv'])
    ap.add_argument('--groups', nargs='+', default=['main', 'level95', 'fit', 'interface', 'flagonly', 'resplit'],
                    choices=('main', 'level95', 'fit', 'interface', 'flagonly', 'resplit'))
    ap.add_argument('--resplit-seeds', nargs='+', type=int, default=RESPLIT_SEEDS)
    ap.add_argument('--jobs', type=int, default=1, help='runs in parallel')
    ap.add_argument('--print-only', action='store_true', help='print the commands without running them')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    def run(job):
        name, ds, g, script, flags = job; out = os.path.join(a.out_dir, name + '.json')
        cmd = [sys.executable, script, '--red', a.red, '--ds', ds, '--graph', g, '--out', out] + flags
        if a.print_only: print(' '.join(cmd)); return name, 0
        try:
            if 'summary' in json.load(open(out)): return name, 0
        except (OSError, ValueError): pass
        with open(os.path.join(a.out_dir, name + '.log'), 'w') as lg: rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
        print(name, 'done' if rc == 0 else f'failed ({rc})', flush=True); return name, rc

    with ThreadPoolExecutor(max_workers=a.jobs) as ex: res = list(ex.map(run, job_list(a.ds, a.groups, a.resplit_seeds)))
    sys.exit(1 if any(rc for _, rc in res) else 0)


if __name__ == '__main__':
    main()

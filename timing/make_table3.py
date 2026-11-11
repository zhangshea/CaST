"""Rows of Table III (latency on HOLD) and the readings quoted with it, from the outputs of timing/time_rules.py.

Paper: Table III; implementation details of Section VI-A ("significantly slower", Bonferroni correction within a timing
window); paragraph "Cost and latency at a common risk target" of Section VI-B (range of the ratios, medians, 99th
percentiles, the threshold curve against its visit ratio).

Each --windowN directory holds <ds>_time.json of both datasets, written by `time_rules.py window --window N`.

Usage:
    python timing/make_table3.py --window1 DIR1 --window2 DIR2 --window3 DIR3 --window4 DIR4
"""
import argparse
import json
import os

# Rows of each timing window, top to bottom: (rule of time_rules.py, label of the table).
ROWS = {
    1: [('fixed', 'Fixed'), ('laet_style', 'LAET-style~\\cite{li2020laet}'), ('level', 'Level'), ('level_cap', 'Level+cap'),
        ('darth', 'DARTH~\\cite{chatzakis2025darth}'), ('cast', '\\textbf{CaST (ours)}')],
    2: [('list_size', 'List size~\\cite{malkov2020}'), ('level_cap', 'Level+cap'), ('cast', '\\textbf{CaST (ours)}')],
    3: [('darth', 'DARTH~\\cite{chatzakis2025darth}'), ('darth_plus', 'DARTH+~\\cite{chatzakis2026darthplus}'), ('list_size', 'List size~\\cite{malkov2020}'),
        ('patience', 'Patience~\\cite{teofili2025patience}'), ('distance_adaptive', 'Dist.-adaptive~\\cite{aljazzazi2025adaptive}'),
        ('level_cap', 'Level+cap'), ('cast', '\\textbf{CaST (ours)}')],
    4: [('curve', 'Curve (in class)'), ('level_cap', 'Level+cap'), ('cast', '\\textbf{CaST (ours)}')],
}


def fmt(x): return f'{x:,.0f}'.replace(',', '{,}')


def table_rows(D, rows, datasets):
    """LaTeX rows of one window: per dataset the mean latency (microseconds), mean visits, failure rate (%) and ratio."""
    out = []
    for nm, label in rows:
        cells = []
        for ds in datasets:
            t = D[ds]['timing'][nm]
            us, vis, fr, ra = fmt(t['mean_us']), fmt(t['mean_visits']), f"{100 * t['fail_rate']:.2f}", f"{t['ratio']:.3f}"
            if nm == 'cast': us, vis, ra = f'\\textbf{{{us}}}', f'\\textbf{{{vis}}}', '\\textbf{1.000}'
            cells += [us, vis, fr, ra]
        out.append(label + ' & ' + ' & '.join(cells) + '\\\\')
    return out


def readings(D, rows, datasets):
    """Per dataset: ratio with its bootstrap interval and Bonferroni bound, visit ratio, failures, percentiles."""
    out = []
    for ds in datasets:
        o = D[ds]; T = o['timing']
        out.append(f"{ds}  visits equal replay {o.get('visits_equal_replay')}  fail count rejecting the target {o.get('fail_count_rejecting_target')}  bootstrap {o.get('bootstrap')}")
        for nm, _ in rows:
            t = T[nm]
            if nm == 'cast':
                out.append(f"  {nm:18s} fail {t['fail']} us/visit {t['us_per_visit']:.4f} p50 {t['p50_us']:.0f} p99 {t['p99_us']:.0f}"); continue
            bonf = f" bonferroni lb {t['lower_bound_one_sided_bonferroni']:.4f} ub {t['upper_bound_one_sided_bonferroni']:.4f}" if 'lower_bound_one_sided_bonferroni' in t else ''
            out.append(f"  {nm:18s} time {t['ratio']:.4f} ci95 [{t['ci95'][0]:.4f}, {t['ci95'][1]:.4f}]{bonf} significantly slower {t['ci95'][0] > 1.0}"
                       f" visits {t['visit_ratio']:.4f} fail {t['fail']} rate {100 * t['fail_rate']:.2f}% us/visit {t['us_per_visit']:.4f} p50 {t['p50_us']:.0f} p99 {t['p99_us']:.0f}")
        if 'curve' in o:
            out.append('  curve ' + str({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in o['curve'].items()}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for w in sorted(ROWS): ap.add_argument(f'--window{w}', default=None, help=f'output directory of timing window {w}')
    ap.add_argument('--datasets', nargs='+', default=['amazon', 'arxiv'])
    a = ap.parse_args(argv)
    lo, hi = None, None
    for w in sorted(ROWS):
        d = getattr(a, f'window{w}')
        if d is None: continue
        D = {ds: json.load(open(os.path.join(d, f'{ds}_time.json'))) for ds in a.datasets}
        print(f'% timing window {w}')
        for r in table_rows(D, ROWS[w], a.datasets): print(r)
        print()
        for r in readings(D, ROWS[w], a.datasets): print(r)
        print()
        for ds in a.datasets:
            for nm, _ in ROWS[w]:
                if nm == 'cast': continue
                x = D[ds]['timing'][nm]['ratio']; lo = x if lo is None else min(lo, x); hi = x if hi is None else max(hi, x)
    if lo is not None: print(f'ratios of the other rules over all windows read: {lo:.4f} to {hi:.4f}')


if __name__ == '__main__':
    main()

"""Visits of each compared rule over those of our rule at each of the nine risk levels, on the EVAL queries of graph A
of the main workload (one panel per dataset). Every rule is calibrated on CAL at level delta / (2 M) for each level, like
the frontier rule of that level; a rule without a calibrated parameter at a level has no marker there.

Paper: Fig. 4.

Inputs, per dataset (graph A):
  --levels-<ds>             output(s) of baselines/levels.py (several files when it ran on subsets of the levels);
                            keys levels[eps].frontier.mean_visits (our rule, the reference of every ratio) and
                            levels[eps].{fixed, laet_style, level, level_cap, list_size}.{calibrated, mean_visits}
  --distance-adaptive-<ds>  output of baselines/distance_adaptive.py calibrate (fine grid of gamma);
                            keys levels[eps].{calibrated, mean_visits, frontier_mean_visits}
  --darth-<ds>              output of baselines/darth.py report; keys levels[eps].rule.{calibrated, mean_visits}, levels[eps].frontier.mean_visits
  --darth-plus-<ds>         output of baselines/darth_plus.py report; same keys
Each source's mean visits of the frontier rule must equal those of baselines/levels.py (relative tolerance 1e-6).

Usage:
    python figures/fig4_visit_ratios.py --levels-amazon LEVELS.json [LEVELS2.json ...] --distance-adaptive-amazon DA.json \
        --darth-amazon DARTH.json --darth-plus-amazon DARTHPLUS.json  (the same four for arxiv)  --out levels.pdf
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.1']   # the nine risk levels, as keys of the inputs
EF = [float(e) for e in E]
SETS = [('amazon', 'Amazon, graph A'), ('arxiv', 'arXiv, graph A')]
FIRST_GROUP = (('fixed', 'Fixed'), ('laet_style', 'LAET-style'), ('level', 'Level'), ('level_cap', 'Level+cap'))
ORDER = ['Fixed', 'LAET-style', 'Level', 'Level+cap', 'List size', 'Dist.-adaptive', 'DARTH', 'DARTH+']


def close(a, b, what):
    assert abs(a - b) <= 1e-6 * max(1.0, abs(a)), (what, a, b)


def merged_levels(paths):
    out = {}
    for p in paths:
        out.update(json.load(open(p))['levels'])
    return out


def ratio(rec, ref):
    """Mean visits of a rule record over those of our rule, None if the rule has no calibrated parameter."""
    return rec['mean_visits'] / ref if rec.get('calibrated') else None


def collect(ds, a):
    L = merged_levels(getattr(a, f'levels_{ds}'))
    assert sorted(L, key=float) == E, sorted(L)
    for e in E:
        assert isinstance(L[e].get('frontier'), dict), f'{ds}: no frontier rule at {e}'
    stair = {e: L[e]['frontier']['mean_visits'] for e in E}
    rows = {}
    for key, name in FIRST_GROUP + (('list_size', 'List size'),):
        rows[name] = [ratio(L[e][key], stair[e]) for e in E]
    d = json.load(open(getattr(a, f'distance_adaptive_{ds}')))['levels']
    for e in E:
        close(d[e]['frontier_mean_visits'], stair[e], f'distance-adaptive frontier {ds} {e}')
    rows['Dist.-adaptive'] = [ratio(d[e], stair[e]) for e in E]
    for name, path in (('DARTH', getattr(a, f'darth_{ds}')), ('DARTH+', getattr(a, f'darth_plus_{ds}'))):
        x = json.load(open(path))['levels']
        for e in E:
            close(x[e]['frontier']['mean_visits'], stair[e], f'{name} frontier {ds} {e}')
        rows[name] = [ratio(x[e]['rule'], stair[e]) for e in E]
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for ds, _ in SETS:
        ap.add_argument(f'--levels-{ds}', nargs='+', required=True, metavar='JSON', help=f'baselines/levels.py output(s), {ds}, graph A')
        ap.add_argument(f'--distance-adaptive-{ds}', required=True, metavar='JSON', help=f'baselines/distance_adaptive.py calibrate output, {ds}, graph A')
        ap.add_argument(f'--darth-{ds}', required=True, metavar='JSON', help=f'baselines/darth.py report output, {ds}, graph A')
        ap.add_argument(f'--darth-plus-{ds}', required=True, metavar='JSON', help=f'baselines/darth_plus.py report output, {ds}, graph A')
    ap.add_argument('--out', required=True, help='output PDF')
    return ap.parse_args()


def main():
    a = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    import style as fs
    fs.setup()

    data = {ds: collect(ds, a) for ds, _ in SETS}
    for ds, _ in SETS:
        print(ds, {n: [None if y is None else round(y, 3) for y in v] for n, v in data[ds].items()})
    print('no calibrated rule:', [(ds, n, e) for ds, _ in SETS for n, v in data[ds].items() for e, y in zip(E, v) if y is None])

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.52), sharey=True, squeeze=False)
    for ax, (ds, title) in zip(axes.flat, SETS):
        rows = data[ds]
        ax.axhline(1.0, color=fs.AQUA, lw=0.9, zorder=1)
        ax.text(0.1035, 0.968, 'CaST (ours)', color=fs.INK, fontsize=fs.TICK, ha='right', va='top', zorder=4)
        for n in ORDER:
            if n not in rows:
                continue
            st = fs.RULES[n]
            xs = [x for x, y in zip(EF, rows[n]) if y is not None]
            ys = [y for y in rows[n] if y is not None]
            hollow = st.get('mfc') == 'white'
            ax.plot(xs, ys, color=st['hue'], ls=st['ls'], lw=0.9, marker=st['marker'], ms=3.2 if hollow else 3.0,
                    mfc=st.get('mfc', st['hue']), mec=st['hue'] if hollow else 'white', mew=0.6 if hollow else 0.3, label=n, zorder=3)
        ax.set_title(title, color=fs.INK, pad=2)
        fs.risk_axis(ax)
        ax.set_ylim(0.8, 2.12)
        ax.yaxis.set_major_locator(FixedLocator([1.0, 1.5, 2.0]))
        ax.yaxis.set_minor_locator(FixedLocator([1.25, 1.75]))
        ax.set_yticklabels([])  # set below on the first column
        fs.clean(ax)
        ax.grid(True, axis='y', which='minor', color=fs.GRID, lw=0.4)
    for ax in axes[:, 0]:
        ax.set_yticklabels(['1.0', '1.5', '2.0'])
        ax.set_ylabel('Visits over ours', labelpad=2)
    for ax in axes[-1]:
        ax.set_xlabel('Risk level $\\varepsilon$', labelpad=1)
    h, l = axes[0, 0].get_legend_handles_labels()
    idx = [0, 4, 1, 5, 2, 6, 3, 7]  # column-major fill: first group on the top row, the other rules below
    h = [h[i] for i in idx]; l = [l[i] for i in idx]
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.705, bottom=0.215, wspace=0.07, hspace=0.38)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

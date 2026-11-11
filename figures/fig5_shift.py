"""Failure rate over the target eps on EVAL of graph A after a change of the queries or of the index, at the nine risk
levels, one panel per dataset. The five changes are title queries, and new items inserted in a number of 1% or 5% of the
base items, or as many base items replaced in place. Lines: the frontier rule calibrated before the change; markers are filled where the one-sided exact
binomial test at level 0.05 / 9 rejects the target. Band: the range over the five changes of the same chain recalibrated
on CAL queries drawn after the change (Theorem 3).

Paper: Fig. 5.

Inputs, per dataset (graph A): outputs of analysis/shift_title.py (--title-<ds>) and of analysis/shift_updates.py
(--insert1-<ds>, --insert5-<ds>, --replace1-<ds>, --replace5-<ds>); keys before_recalibration[eps] and
recalibrated[eps], each {calibrated, fail_rate, reject, ...}.

Usage:
    python figures/fig5_shift.py --title-amazon TITLE.json --replace5-amazon REPLACE5.json --replace1-amazon REPLACE1.json \
        --insert5-amazon INSERT5.json --insert1-amazon INSERT1.json  (the same five for arxiv)  --out shift.pdf
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.1']   # the nine risk levels, as keys of the inputs
EF = [float(e) for e in E]
DATASETS = ('amazon', 'arxiv')
CHANGES = (('title', 'Title queries'), ('replace5', 'Replace 5%'), ('replace1', 'Replace 1%'), ('insert5', 'Insert 5%'), ('insert1', 'Insert 1%'))


def series(block):
    """(failure rate / eps, test rejects) per level, None where no rule is calibrated."""
    out = []
    for e in E:
        s = block.get(e, {})
        out.append((s['fail_rate'] / float(e), bool(s['reject'])) if s.get('calibrated') else None)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for ds in DATASETS:
        for tag, name in CHANGES:
            src = 'analysis/shift_title.py' if tag == 'title' else 'analysis/shift_updates.py'
            ap.add_argument(f'--{tag}-{ds}', required=True, metavar='JSON', help=f'{src} output, {ds}, graph A (' + name.replace('%', '%%') + ')')
    ap.add_argument('--out', required=True, help='output PDF')
    return ap.parse_args()


def main():
    a = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from matplotlib.lines import Line2D
    import style as fs
    fs.setup()

    data = {}
    for ds in DATASETS:
        for tag, name in CHANGES:
            d = json.load(open(getattr(a, f'{tag}_{ds}')))
            data[(ds, name)] = (series(d['before_recalibration']), series(d['recalibrated']))

    # Counts of rejections and the largest ratio, before and after recalibration.
    for (ds, name), (before, after) in data.items():
        rej_b = sum(1 for x in before if x and x[1]); rej_a = sum(1 for x in after if x and x[1]); none_a = sum(1 for x in after if x is None)
        print(f'{ds:7s} {name:13s} before: rejects {rej_b}/9  max {max(x[0] for x in before if x):.3f} | recalibrated: rejects {rej_a}, no rule {none_a}, max {max(x[0] for x in after if x):.3f}')

    STY = {'Title queries': dict(hue=fs.RED, ls='-', marker='D', ms=3.2),
           'Replace 5%': dict(hue=fs.AMBER, ls='-', marker='o', ms=3.6),
           'Replace 1%': dict(hue=fs.AMBER, ls=(0, (3.2, 1.6)), marker='o', ms=3.6),
           'Insert 5%': dict(hue=fs.GRAY, ls='-', marker='s', ms=3.4),
           'Insert 1%': dict(hue=fs.GRAY, ls=(0, (3.2, 1.6)), marker='s', ms=3.4)}
    NAMES = list(STY)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.52))
    for ax, (ds, title, ylim, yt) in zip(axes, (('amazon', 'Amazon, graph A', (0.76, 1.24), [0.8, 0.9, 1.0, 1.1, 1.2]),
                                                ('arxiv', 'arXiv, graph A', (0.72, 2.75), [1.0, 1.5, 2.0, 2.5]))):
        lo = [min(data[(ds, n)][1][i][0] for n in NAMES if data[(ds, n)][1][i]) for i in range(len(E))]
        hi = [max(data[(ds, n)][1][i][0] for n in NAMES if data[(ds, n)][1][i]) for i in range(len(E))]
        ax.fill_between(EF, lo, hi, color=fs.AQUA_FILL, lw=0, zorder=1, label='Recalibrated')
        ax.axhline(1.0, color=fs.INK, lw=0.8, ls=(0, (1, 1.2)), zorder=2, label='Target $\\varepsilon$')
        for n in NAMES:
            before = data[(ds, n)][0]; st = STY[n]
            ax.plot(EF, [x[0] for x in before], color=st['hue'], ls=st['ls'], lw=0.9, zorder=3, label=n)
            for xf, x in zip(EF, before):
                ax.plot([xf], [x[0]], ls='', marker=st['marker'], ms=st['ms'], mew=0.7, mec=st['hue'],
                        mfc=st['hue'] if x[1] else 'white', zorder=4)
        ax.set_title(title, color=fs.INK, pad=2)
        fs.risk_axis(ax)
        ax.set_ylim(*ylim); ax.yaxis.set_major_locator(FixedLocator(yt))
        fs.clean(ax)
        ax.set_xlabel('Risk level $\\varepsilon$', labelpad=1)
    axes[0].set_ylabel('Failure rate over $\\varepsilon$', labelpad=2)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ['Title queries', 'Target $\\varepsilon$', 'Replace 5%', 'Replace 1%', 'Insert 5%', 'Insert 1%', 'Recalibrated']]
    # legend markers are hollow, as in the panels where the test does not reject; one extra entry shows the filled marker
    hh = []
    for i in order:
        n = l[i]
        if n in STY:
            st = STY[n]; hh.append(Line2D([], [], color=st['hue'], ls=st['ls'], lw=0.9, marker=st['marker'], ms=st['ms'], mfc='white', mec=st['hue'], mew=0.7))
        else:
            hh.append(h[i])
    labels = [l[i] for i in order]
    hh.insert(7, Line2D([], [], ls='', marker='o', ms=3.6, mfc=fs.MUTED, mec=fs.MUTED)); labels.insert(7, 'Test rejects')
    fig.legend(hh, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.45, columnspacing=0.6)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.705, bottom=0.215, wspace=0.2)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

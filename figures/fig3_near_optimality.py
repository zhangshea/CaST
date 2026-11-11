"""Near-optimality on EVAL of the frontier rule, calibrated at level delta / (2 M).

(a) Optimality-gap bound U / L (Corollary 1) at the nine risk levels for the four settings of the main workload
    (Amazon and arXiv, graphs A and B), rounded up to three decimals; the axis is broken above 1.30 for the one value
    that exceeds it.
(b) Cost-risk frontier of graph A, one panel per dataset: the lower bound L(eps) of Theorem 4, the empirical class
    optimum (problem (3) solved exactly on EVAL) and the mean visits of our rule at each risk level where it is calibrated.

Paper: Fig. 3.

Inputs:
  --exact-bounds  outputs of pipeline/exact_bounds.py, one per setting (four files; the setting is read from the
                  fields ds and graph); keys EVAL.rules[eps].ratio_up6 for (a) and EVAL.L_curve for (b)
  --frontier      outputs of pipeline/frontier.py for graph A of each dataset (two files); keys config.levels,
                  rules.chain[eps].{calibrated, mean_visits} and bounds[eps].empirical_optimum

Usage:
    python figures/fig3_near_optimality.py --exact-bounds EXACT_amazon_A.json EXACT_amazon_B.json EXACT_arxiv_A.json \
        EXACT_arxiv_B.json --frontier FRONTIER_amazon_A.json FRONTIER_arxiv_A.json --out nearopt.pdf
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E9 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]       # the nine risk levels
DATASETS = (('amazon', 'Amazon'), ('arxiv', 'arXiv'))
GRAPHS = ('A', 'B')
PANEL_B_GRAPH = 'A'
YTOP = 1.30                                                        # top of the lower part of the broken axis of (a)


def up3(x):
    """x (a multiple of 1e-6, as written by pipeline/exact_bounds.py) rounded up to three decimals."""
    m = int(round(x * 10 ** 6))
    return -(-m // 1000) / 1000


def by_setting(paths):
    out = {}
    for p in paths:
        d = json.load(open(p)); key = (d['ds'], d['graph'])
        assert key not in out, f'two inputs for setting {key}'
        out[key] = d
    return out


def ratios(exact):
    """U / L at the nine levels, rounded up; NaN where no rule is calibrated or L is not positive."""
    out = []
    for e in E9:
        r = exact['EVAL']['rules'].get(str(e), {})
        v = r.get('ratio_up6') if r.get('calibrated') else None
        out.append(math.nan if v is None else up3(v))
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exact-bounds', nargs=4, required=True, metavar='JSON', help='pipeline/exact_bounds.py outputs of the four settings')
    ap.add_argument('--frontier', nargs=2, required=True, metavar='JSON', help='pipeline/frontier.py outputs of graph A of both datasets')
    ap.add_argument('--out', required=True, help='output PDF')
    return ap.parse_args()


def main():
    a = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    import style as fs
    fs.setup()

    EX = by_setting(a.exact_bounds); FR = by_setting(a.frontier)
    want = [(ds, g) for ds, _ in DATASETS for g in GRAPHS]
    assert sorted(EX) == sorted(want), f'--exact-bounds must cover the settings {want}, got {sorted(EX)}'
    assert sorted(FR) == sorted((ds, PANEL_B_GRAPH) for ds, _ in DATASETS), f'--frontier must cover graph {PANEL_B_GRAPH} of both datasets'

    # (a) data: U / L rounded up, in the legend order Amazon A, Amazon B, arXiv A, arXiv B
    NAME = dict(DATASETS)
    RATIO = {(NAME[ds], g): ratios(EX[(ds, g)]) for ds, g in want}
    for s_, v in RATIO.items():
        print('U/L', s_, v)

    W, H = fs.COLUMN, 196.0 * fs.PT
    fig = plt.figure(figsize=(W, H))
    TOPA, BOTA = 0.905, 0.665    # panel (a)
    TOPB, BOTB = 0.43, 0.1    # panels (b)
    LEFT, RIGHT = 0.125, 0.995

    # ---------------- (a) U / L, with a broken axis so that the one value above 1.30 is drawn where it is
    GAPA, HTOP = 0.014, 0.042            # gap and height of the upper strip, as fractions of the figure height
    ax = fig.add_axes([LEFT, BOTA, RIGHT - LEFT, TOPA - BOTA - GAPA - HTOP])
    axt = fig.add_axes([LEFT, TOPA - HTOP, RIGHT - LEFT, HTOP], sharex=ax)
    STY = {('Amazon', 'A'): dict(hue=fs.INK, ls='-', marker='o', ms=3.4), ('Amazon', 'B'): dict(hue=fs.INK, ls=(0, (3.2, 1.6)), marker='^', ms=3.4),
           ('arXiv', 'A'): dict(hue=fs.GRAY, ls='-', marker='s', ms=3.2), ('arXiv', 'B'): dict(hue=fs.GRAY, ls=(0, (3.2, 1.6)), marker='D', ms=3.0)}
    finite = [y for v in RATIO.values() for y in v if not math.isnan(y)]
    hi = max(finite)
    assert hi > YTOP and sum(1 for y in finite if y > YTOP) == 1, 'the upper strip of panel (a) is laid out for exactly one value above 1.30'
    for a_ in (ax, axt):
        for s_, v in RATIO.items():
            st = STY[s_]
            a_.plot(E9, v, color=st['hue'], ls=st['ls'], lw=1.0, marker=st['marker'], ms=st['ms'], mec='white', mew=0.4,
                    label=f'{s_[0]}, {s_[1]}', zorder=3)
    ax.set_ylim(1.0, YTOP); ax.yaxis.set_major_locator(FixedLocator([1.0, 1.1, 1.2, 1.3])); ax.yaxis.set_minor_locator(NullLocator())
    axt.set_ylim(hi - 0.03, hi + 0.012); axt.yaxis.set_major_locator(FixedLocator([round(hi, 2)])); axt.yaxis.set_minor_locator(NullLocator())
    axt.set_yticklabels([f'{round(hi, 2):.2f}'])
    ax.set_xlim(0.006, 0.104); ax.set_xticks(E9); ax.set_xticklabels(['0.10' if e == 0.1 else f'{e:.2f}' for e in E9])
    fs.clean(ax); fs.clean(axt, ygrid=False)
    axt.spines['bottom'].set_visible(False); axt.tick_params(axis='x', bottom=False, labelbottom=False)
    # break marks on the left spine
    d = 2.2  # points
    for a_, yv in ((ax, 1.0), (axt, 0.0)):
        a_.plot([0], [yv], transform=a_.transAxes, marker=[(-1, -0.5), (1, 0.5)], ms=d * 2, mew=0.6, color=fs.MUTED, clip_on=False, ls='')
    ax.set_xlabel('Risk level $\\varepsilon$', labelpad=1); ax.set_ylabel('$U/L$', labelpad=2)
    ax.yaxis.set_label_coords(-0.075, 0.6)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', ncol=4, bbox_to_anchor=(0.085, 1.0), handlelength=2.0, columnspacing=0.8)
    fig.text(0.004, 1.0, '(a)', ha='left', va='top', color=fs.INK)

    # ---------------- (b) cost-risk frontier of graph A
    GAP = 0.02
    wb = (RIGHT - LEFT - GAP) / 2
    axs = [fig.add_axes([LEFT, BOTB, wb, TOPB - BOTB])]
    axs.append(fig.add_axes([LEFT + wb + GAP, BOTB, wb, TOPB - BOTB], sharey=axs[0]))
    for ax, (ds, name) in zip(axs, DATASETS):
        title = f'{name}, graph {PANEL_B_GRAPH}'
        d = FR[(ds, PANEL_B_GRAPH)]
        E = d['config']['levels']
        srcL = EX[(ds, PANEL_B_GRAPH)]['EVAL']['L_curve']
        cur = [(e, l) for e, l in srcL if 0.01 <= e <= 0.12 and l > 0]
        ax.plot([e for e, _ in cur], [l for _, l in cur], color=fs.INK, lw=1.1, label='Lower bound $L(\\varepsilon)$', zorder=2)
        opt = [(e, d['bounds'][str(e)]['empirical_optimum']) for e in E]
        ax.plot([e for e, _ in opt], [v for _, v in opt], color=fs.GRAY, lw=1.0, ls=(0, (3.2, 1.6)), marker='s', ms=3.0, mec='white', mew=0.4,
                label='Empirical optimum', zorder=3)
        rule = [(e, d['rules']['chain'][str(e)]['mean_visits']) for e in E if d['rules']['chain'][str(e)].get('calibrated')]
        ax.plot([e for e, _ in rule], [v for _, v in rule], color=fs.AQUA, lw=0, marker='o', ms=3.7, mec=fs.INK, mew=0.5, label='CaST (ours)', zorder=4)
        print(ds, 'rule', [(e, round(v, 1)) for e, v in rule])
        print(ds, 'opt ', [(e, round(v, 1)) for e, v in opt])
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator([3000, 4000, 6000, 8000, 12000])); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v / 1000)}k'))
        fs.risk_axis(ax, ticks=(0.01, 0.05, 0.1), minor=(0.02, 0.03, 0.04, 0.06, 0.07, 0.08), xlim=(0.005, 0.125))
        fs.clean(ax)
        ax.text(0.97, 0.96, title, transform=ax.transAxes, ha='right', va='top', color=fs.INK)
        ax.set_xlabel('Risk level $\\varepsilon$', labelpad=1)
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[0].set_ylabel('Mean visits (log)', labelpad=2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', ncol=3, bbox_to_anchor=(0.085, 0.53), handlelength=1.6, columnspacing=0.8)
    fig.text(0.004, 0.53, '(b)', ha='left', va='top', color=fs.INK)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

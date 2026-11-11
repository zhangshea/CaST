"""Shared style of the paper figures (Figs. 3-5): palette, fonts, printed sizes and the line style of each compared rule.

Palette:
  ours                  aqua, used for nothing else
  reference quantities  neutral ink and grays (lower bound L, empirical class optimum, target line)
  compared rules        one fixed hue, dash pattern and marker per rule, the same in every figure;
                        first group (same observations as ours): grays with distinct markers and dashes;
                        list size blue, distance-adaptive orange, DARTH and DARTH+ plum (solid, dashed)
  query and index changes (Fig. 5): red and amber, which no rule uses, with distinct markers

Text is Times and mathematics Computer Modern, as in the paper. Every figure is drawn at its printed size, so the
point sizes below are printed sizes: labels, legends and panel titles 8 pt (the caption size of the paper), tick
labels 7 pt.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRAY, GRID = '#0b0b0b', '#52514e', '#8f8d87', '#e2e1dd'
AQUA, AQUA_FILL = '#1baf7a', '#bfe8d8'
BLUE, ORANGE, PLUM = '#2a78d6', '#eb6834', '#b0459a'
RED, AMBER = '#e34948', '#c98500'

PT = 1 / 72  # inches per PostScript point
COLUMN = 252.0 * PT  # column width of the paper, 3.5 in

LABEL, TICK = 8, 7

RULES = {  # compared rules: the same hue, dash pattern and marker in every figure
    'Fixed': dict(hue=GRAY, ls=(0, (1, 1.2)), marker='v'),
    'LAET-style': dict(hue=GRAY, ls=(0, (3.2, 1.6)), marker='D'),
    'Level': dict(hue=MUTED, ls='-', marker='s'),
    'Level+cap': dict(hue=INK, ls='-', marker='^'),
    'List size': dict(hue=BLUE, ls='-', marker='o'),
    'Dist.-adaptive': dict(hue=ORANGE, ls='-', marker='P'),
    'DARTH': dict(hue=PLUM, ls='-', marker='X'),
    'DARTH+': dict(hue=PLUM, ls=(0, (3.2, 1.6)), marker='d', mfc='white'),
}


def setup():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
        'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': False,
        'font.size': LABEL, 'axes.labelsize': LABEL, 'axes.titlesize': LABEL, 'legend.fontsize': LABEL,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.9, 'legend.borderaxespad': 0.0, 'legend.borderpad': 0.0, 'legend.labelspacing': 0.25,
        'pdf.fonttype': 42, 'savefig.dpi': 600,
    })
    plt.rc('axes', edgecolor=MUTED, labelcolor=INK)  # spines and axis labels
    plt.rc('xtick', color=MUTED)
    plt.rc('ytick', color=MUTED)


def clean(ax, ygrid=True):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if ygrid:
        ax.grid(True, axis='y', color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def risk_axis(ax, ticks=(0.01, 0.03, 0.05, 0.07, 0.1), minor=(0.02, 0.04, 0.06, 0.08), xlim=(0.005, 0.105)):
    """Horizontal axis of the risk level eps."""
    from matplotlib.ticker import FixedLocator
    ax.set_xlim(*xlim)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(['0.10' if t == 0.1 else f'{t:.2f}' for t in ticks])
    ax.xaxis.set_minor_locator(FixedLocator(list(minor)))

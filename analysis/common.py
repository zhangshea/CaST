"""Small helpers shared by the analysis scripts of Section VI-B: rounding of reported ratios and argument parsing."""
import json
import math


def up3(x):
    """Round up to three decimals, as the reported ratios U / L are rounded."""
    return math.ceil(x * 1000 - 1e-9) / 1000


def labelled_paths(args, n):
    """Parse arguments of the form label=PATH_1,...,PATH_n into [(label, [paths])]."""
    out = []
    for arg in args:
        lab, paths = arg.split('=', 1); paths = paths.split(',')
        if len(paths) != n: raise SystemExit(f'{arg}: expected {n} comma-separated paths')
        out.append((lab, paths))
    return out

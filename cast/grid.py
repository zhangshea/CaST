"""Checkpoint grid and horizon of the stopping rules.

Paper: Section III-B (checkpoints and cost) and the implementation details of Section VI-A.
"""
import numpy as np

# Entries of the candidate list of the logged best-first traversal.
LIST_SIZE = 8192

# Visit counts g_0 < ... < g_{J-1} of the J = 100 checkpoints: a geometric grid from 64 to 32,768.
GRID = np.unique(np.round(np.geomspace(64, 32768, 100)).astype(np.int64))
J = len(GRID)

# Horizon H: the last checkpoint plus the maximum out-degree of the bottom layer (one expansion adds at most 64 visits).
H = 32768 + 64

# Visit caps tried by the Level+cap rule (None: no cap before the last checkpoint).
CAPS = [1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, None]

# Marker of "never happens" in int32 visit counts.
NEVER = np.iinfo(np.int32).max


def cap_checkpoint(cap, grid=GRID):
    """Index of the first checkpoint whose visit count reaches `cap` (the last checkpoint when cap is None)."""
    n = len(grid)
    return n - 1 if cap is None else min(int(np.searchsorted(grid, cap, side='left')), n - 1)

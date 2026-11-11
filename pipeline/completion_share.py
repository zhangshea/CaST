"""Share of queries that do not complete by the last checkpoint, and the risk levels a setting can evaluate.

Paper: Section VI-A ("... do not complete within 32,768 visits") and Section VI-B, paragraph "Workload breadth" (YFCC-10M:
levels from 0.05 only; k = 50: the share exceeds every evaluated risk level). A rule that waits to the last checkpoint
fails on exactly these queries, so only risk levels above this share can be met.

The share is the fraction of queries whose completion time X exceeds the visit count c_J at the last checkpoint, over all
given groups, per group, and (layout 'pool') over the EVAL queries of the pool permutation.

With --config and --out-config the list of risk levels of a frontier configuration is restricted:
    --rule above    if the overall share r exceeds 0.03, keep the levels eps > r (otherwise the configuration is unchanged);
    --rule margin   keep the levels eps >= r_SEL + 0.01, with r_SEL the share of the group `sel`; report levels from 0.03.

Usage:
    python pipeline/completion_share.py --red WORK/reduced/k50_kappa1 --ds amazon --graph A --out OUT.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cast.grid import NEVER


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--red', required=True, help='directory of the reduced arrays')
    ap.add_argument('--ds', required=True); ap.add_argument('--graph', required=True)
    ap.add_argument('--layout', default='pool', choices=('pool', 'parts'))
    ap.add_argument('--groups', nargs='*', default=None, help="groups read (default: fitpool and chunk0..19, or fit sel cal eval)")
    ap.add_argument('--pool-seed', type=lambda x: int(x, 0), default=0x13528ef, help="layout 'pool': seed of the permutation of the 200,000 pool queries")
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=None, help='frontier configuration to restrict'); ap.add_argument('--out-config', default=None)
    ap.add_argument('--rule', default='above', choices=('above', 'margin'))
    return ap.parse_args()


def main():
    a = parse_args(); ds, g = a.ds, a.graph
    groups = a.groups or (['fitpool'] + [f'chunk{k}' for k in range(20)] if a.layout == 'pool' else ['fit', 'sel', 'cal', 'eval'])
    Z = {p: np.load(f'{a.red}/{ds}_{g}_{p}.npz') for p in groups}
    X = {p: Z[p]['X_exact'].astype(np.int64) for p in groups}; cJ = {p: Z[p]['c'][:, -1].astype(np.int64) for p in groups}
    Xa = np.concatenate([X[p] for p in groups]); ca = np.concatenate([cJ[p] for p in groups])
    r0 = float((Xa > ca).mean())
    out = dict(ds=ds, graph=g, layout=a.layout, groups=groups, n=int(len(Xa)), share_incomplete=r0,
               share_incomplete_by_group={p: float((X[p] > cJ[p]).mean()) for p in groups}, share_never_complete=float((Xa >= NEVER).mean()),
               X_median=float(np.percentile(np.where(Xa >= NEVER, np.inf, Xa), 50)), last_checkpoint=int(Z[groups[0]]['grid'][-1]))
    if a.layout == 'pool' and all(f'chunk{k}' in X for k in range(20)):
        Xf = np.concatenate([X[f'chunk{k}'] for k in range(20)]); cf = np.concatenate([cJ[f'chunk{k}'] for k in range(20)])
        Pf = np.random.default_rng(a.pool_seed).permutation(200000); EV = Pf[60000:180000]          # EVAL of cast.setting.SPLIT
        out.update(share_incomplete_eval=float((Xf[EV] > cf[EV]).mean()),
                   X_median_eval=float(np.percentile(np.where(Xf[EV] >= NEVER, np.inf, Xf[EV]), 50)),
                   X_p90_eval=float(np.percentile(np.where(Xf[EV] >= NEVER, np.inf, Xf[EV]), 90)))
    if a.config and a.out_config:
        cfg = json.load(open(a.config)); E0, EP0 = list(cfg['levels']), list(cfg['report_levels'])
        if a.rule == 'above':
            if r0 > 0.03:
                cfg['levels'] = [e for e in E0 if e > r0]; cfg['report_levels'] = [e for e in EP0 if e > r0]
        else:
            rs = out['share_incomplete_by_group']['sel']
            cfg['levels'] = [e for e in E0 if e >= rs + 0.01 - 1e-12]; cfg['report_levels'] = [e for e in cfg['levels'] if e >= 0.03 - 1e-12]
        json.dump(cfg, open(a.out_config, 'w'), indent=1)
        out.update(levels=cfg['levels'], report_levels=cfg['report_levels'], rule=a.rule)
    json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()

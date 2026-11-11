"""Margin for differences between the deployed rule and its offline replay.

Paper: Section VI-A, Implementation details. When the online stopping point differs from the offline one on d of n
checked queries, an exact one-sided binomial bound at level alpha limits the probability p of a difference. The deployed
rule then has risk at most eps + p and cost at most U + H p, and its ratio against the class optimum at eps + p is at most
(U + H p) / L(eps + p), where L(e) = max(0, max_k (ell_k - lambda_k e)) is the lower bound of Theorem 4.

Reads the output of pipeline/exact_bounds.py (U and the values ell_k of the multipliers, both rounded outward).

Usage:
    python analysis/deployment_margin.py --exact EXACT.json --differ 1 --checked 30000 [--eps 0.05] [--alpha 0.05]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exact', required=True, help='output of pipeline/exact_bounds.py')
    ap.add_argument('--differ', type=int, required=True, help='queries whose online stopping point differs from the offline one')
    ap.add_argument('--checked', type=int, required=True, help='queries checked')
    ap.add_argument('--eps', type=float, default=0.05); ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from scipy.stats import beta
    from cast.grid import H
    # Exact one-sided (Clopper-Pearson) upper bound on the probability of a difference.
    p = 1.0 if a.differ >= a.checked else float(beta.ppf(1 - a.alpha, a.differ + 1, a.checked - a.differ))
    E = json.load(open(a.exact))['EVAL']; r = E['rules'][str(a.eps)]
    L = lambda e: max(0.0, max(d['ell_down6'] - d['lam'] * e for d in E['ell']))
    U = r['U_up6']; base = U / L(a.eps); adjusted = (U + H * p) / L(a.eps + p)
    out = dict(differ=a.differ, checked=a.checked, alpha=a.alpha, p=p, eps=a.eps, U=U, L=L(a.eps), ratio=base,
               L_at_eps_plus_p=L(a.eps + p), ratio_deployed=adjusted, increase=adjusted - base)
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

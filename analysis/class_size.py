"""Number of staircase rules of the declared interface.

Paper: Theorem 1(d), N_Z = C(B + J - 1, J - 1) C(B + J, J - 1) / J, and the remark after it: with the interface of the
experiments, J = 100 checkpoints and B = 100 buckets, ln N_Z is about 266 (Sections I and IV).

Usage:
    python analysis/class_size.py [--J 100] [--B 100]
"""
import argparse
import math


def log_staircases(J, B):
    return math.log(math.comb(B + J - 1, J - 1)) + math.log(math.comb(B + J, J - 1)) - math.log(J)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--J', type=int, default=100, help='checkpoints'); ap.add_argument('--B', type=int, default=100, help='score buckets')
    a = ap.parse_args()
    print(f'J = {a.J}, B = {a.B}: ln N_Z = {log_staircases(a.J, a.B):.3f}')

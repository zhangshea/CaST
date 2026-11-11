"""Property checks of Sec. III-A on small random instances (computational checks, not proofs).

Pure Python reference (feasibility by Hall's condition over all subsets, independent of the C++ code):
  lemma_1         Lemma 1: Greedy(A) equals the exact answer S* if and only if S* is a subset of A.
  lemma_2         Lemma 2: the members of S* that lie in A belong to Greedy(A).
  least_distance  the sorted distances of S* are dominated by those of every feasible set of k items, so S* has
                  the least total distance (Sec. III-A).
  verify          the engine's independent feasibility check (cast_engine.verify) against Hall's condition, if
                  the module is importable.

Usage: python tests/test_properties.py [--quick] [--seed S] [--module-dir DIR]
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------- reference feasibility
def feasible(S, cat, kappa):
    """Hall's condition with capacities: |T| <= sum_{g in c(T)} kappa(g) for every T subset of S (Definition 1)."""
    S = list(S)
    for r in range(1, len(S) + 1):
        for T in itertools.combinations(S, r):
            cs = set().union(*[cat[v] for v in T])
            if len(T) > sum(kappa[g] for g in cs):
                return False
    return True


def greedy(order, cat, kappa, k):
    B = []
    for v in order:
        if len(B) == k:
            break
        if feasible(B + [v], cat, kappa):
            B.append(v)
    return B


def rand_instance(rng, n, m, s, kap_max=2):
    cat = {v: set(rng.choice(m, size=rng.integers(1, min(s, m) + 1), replace=False).tolist()) for v in range(n)}
    kappa = {g: int(rng.integers(1, kap_max + 1)) for g in range(m)}
    d = rng.integers(0, 12, size=n) / 12.0
    order = sorted(range(n), key=lambda v: (d[v], v))
    return cat, kappa, d, order


# ----------------------------------------------------------------------------- Lemma 1, Lemma 2
def lemmas_1_2(rng, trials):
    bad1 = 0; bad2 = 0; contain = 0
    for _ in range(trials):
        n = int(rng.integers(2, 12)); m = int(rng.integers(1, 6)); s = int(rng.integers(1, 4)); k = int(rng.integers(1, 5))
        cat, kappa, d, order = rand_instance(rng, n, m, s)
        Sstar = greedy(order, cat, kappa, k)
        A = set(np.nonzero(rng.random(n) < rng.random())[0].tolist()) | (set(Sstar) if rng.random() < 0.5 else set())
        GA = greedy([v for v in order if v in A], cat, kappa, k)
        exact = sorted(GA) == sorted(Sstar)
        sub = set(Sstar) <= A
        contain += sub
        if exact != sub:
            bad1 += 1
        if not (set(Sstar) & A) <= set(GA):
            bad2 += 1
    return bad1 == 0 and bad2 == 0, f"{trials} random visited sets ({contain} containing S*): {bad1} violations of " \
                                    f"'exact iff S* in A' (Lemma 1), {bad2} of 'S* and A inside Greedy(A)' (Lemma 2)"


def least_distance(rng, trials):
    bad = 0; checked = 0
    for _ in range(trials):
        n = int(rng.integers(2, 10)); m = int(rng.integers(1, 5)); s = int(rng.integers(1, 4)); k = int(rng.integers(1, 4))
        cat, kappa, d, order = rand_instance(rng, n, m, s)
        Sstar = greedy(order, cat, kappa, k)
        if len(Sstar) < k:
            continue
        ds = sorted(d[v] for v in Sstar)
        for T in itertools.combinations(range(n), k):
            if feasible(T, cat, kappa):
                checked += 1
                dt = sorted(d[v] for v in T)
                if any(a > b + 1e-12 for a, b in zip(ds, dt)):
                    bad += 1
    return bad == 0, f"{checked} feasible k-sets: {bad} domination violations"


# ----------------------------------------------------------------------------- engine cross-check
def load_module(module_dir):
    sys.path.insert(0, module_dir)
    try:
        import cast_engine
        return cast_engine
    except ImportError:
        return None


def to_csr(cat, n):
    ptr = np.zeros(n + 1, np.int64); ci = []
    for v in range(n):
        ci.extend(sorted(cat[v])); ptr[v + 1] = len(ci)
    return ptr, np.array(ci, np.int32)


def verify_module(rng, eng, trials):
    bad = 0
    for _ in range(trials):
        n = int(rng.integers(2, 12)); m = int(rng.integers(1, 6)); s = int(rng.integers(1, 4)); k = int(rng.integers(1, 6))
        cat, kappa, d, order = rand_instance(rng, n, m, s)
        ptr, ci = to_csr(cat, n)
        C = eng.Categories(ptr, ci, m, np.array([kappa[g] for g in range(m)], np.int32))
        T = rng.choice(n, size=min(k, n), replace=False).tolist()
        row = np.full((1, k), -1, np.int64); row[0, :len(T)] = T
        code = int(eng.verify(row, C)[0])
        ref = feasible(T, cat, kappa)
        exp = (1 if len(T) == k else 2) if ref else 0
        if code != exp:
            bad += 1
    return bad == 0, f"{trials} random sets: engine verify vs Hall's condition, {bad} mismatches"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='a tenth of the trials')
    ap.add_argument('--seed', type=int, default=1, help='seed of the random instances')
    ap.add_argument('--module-dir', default=os.path.join(here, '..', 'build'), help='directory that holds the built cast_engine module')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t = 400 if a.quick else 4000
    checks = [("Lemma 1 and Lemma 2", lambda: lemmas_1_2(rng, 5 * t)),
              ("least total distance of S*", lambda: least_distance(rng, t // 2))]
    eng = load_module(a.module_dir)
    if eng is not None:
        checks += [("verify (engine)", lambda: verify_module(rng, eng, 5 * t))]
    allok = True
    for name, fn in checks:
        t0 = time.time()
        ok, msg = fn()
        allok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}  [{time.time() - t0:.1f}s]", flush=True)
    if eng is None:
        print("note: cast_engine not importable; the engine cross-check was skipped")
    print("ALL PASS" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()

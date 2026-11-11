"""Measurement protocol of the latency experiments and the statistics formed from the per-query latencies.

Paper: implementation details of Section VI-A (one pinned core, passes in rotated order, ratios within one timing
window, "significantly slower" by a one-sided paired bootstrap test at the 2.5% level) and Table III.

The latency of a query under a rule is the median over the passes of the time spent inside the engine for that query
(features and models included). A window reports the mean of these medians and its ratio to the mean of our rule.
"""
import numpy as np

from engine_rules import same_answer


def timed_passes(run, names, Q, S, passes, warmup, log):
    """Warm every rule up on the first `warmup` queries, then run `passes` single-thread passes over all queries.

    The order of the rules rotates by one position per pass. run(name, Q, threads) returns the engine output.
    Returns per rule: the list of per-query times of each pass (ns), the visits and the exactness flags of the last pass.
    """
    NS = {nm: [] for nm in names}; VIS, COV = {}, {}
    for nm in names: run(nm, Q[:warmup], 1)
    for rep in range(passes):
        for nm in names[rep % len(names):] + names[:rep % len(names)]:
            o = run(nm, Q, 1); NS[nm].append(o['ns'].astype(np.int64)); VIS[nm] = o['visits']; COV[nm] = same_answer(o['ids'], S)
        log('timing pass', rep + 1, 'of', passes)
    return NS, VIS, COV


def median_latency(NS):
    """Per-query median over the passes, per rule (ns)."""
    return {nm: np.median(np.stack(NS[nm]), axis=0) for nm in NS}


def bootstrap_indices(n, seed, resamples, index_dtype):
    """Query indices of the paired bootstrap, drawn once per window and shared by all rules.

    The integer type is part of the random stream: the first two windows of Table III drew int64 indices (2,000
    resamples, seed 0), the third and fourth int32 indices (20,000 resamples, recorded seeds).
    """
    return np.random.default_rng(seed).integers(0, n, (resamples, n), dtype=np.dtype(index_dtype))


def rejection_count(n, eps, a):
    """Smallest failure count at which the one-sided exact binomial test at level a rejects a risk of at most eps."""
    from scipy.stats import binom
    f = int(n * eps)
    while binom.cdf(f - 1, n, eps) < 1 - a: f += 1
    return f


def summarize(names, MED, VIS, FAIL, bidx, reference, family=None, pass_means=None):
    """Per-rule statistics of one timing window.

    MED[rule] per-query median latency (ns), VIS[rule] per-query visits, FAIL[rule] number of queries whose answer
    differs from the exact one, bidx bootstrap indices. `ratio` is the mean latency over that of `reference`; `ci95` its
    2.5% and 97.5% bootstrap quantiles, so a rule is significantly slower when ci95[0] > 1. With `family`, the one-sided
    bounds at level 0.025 / family (Bonferroni over the ratios of the window) are added.
    """
    from scipy.stats import beta as beta_dist
    lat_ref = MED[reference]; n = len(lat_ref); TIM = {}
    for nm in names:
        lat = MED[nm]; rat = lat[bidx].mean(1) / lat_ref[bidx].mean(1); f_ = FAIL[nm]
        TIM[nm] = dict(mean_us=float(lat.mean() / 1e3), p50_us=float(np.percentile(lat, 50) / 1e3), p95_us=float(np.percentile(lat, 95) / 1e3),
                       p99_us=float(np.percentile(lat, 99) / 1e3), mean_visits=float(VIS[nm].mean()), us_per_visit=float(lat.sum() / 1e3 / VIS[nm].sum()),
                       fail=f_, fail_rate=f_ / n,
                       fail_rate_ci95=[float(beta_dist.ppf(.025, f_, n - f_ + 1)) if f_ else 0.0, float(beta_dist.ppf(.975, f_ + 1, n - f_))],
                       ratio=float(lat.mean() / lat_ref.mean()), visit_ratio=float(VIS[nm].mean() / VIS[reference].mean()),
                       ci95=[float(np.quantile(rat, .025)), float(np.quantile(rat, .975))])
        TIM[nm]['significantly_slower'] = bool(TIM[nm]['ci95'][0] > 1.0)
        if family:
            TIM[nm].update(lower_bound_one_sided_bonferroni=float(np.quantile(rat, 0.025 / family)),
                           upper_bound_one_sided_bonferroni=float(np.quantile(rat, 1 - 0.025 / family)))
        if pass_means is not None: TIM[nm]['pass_means_us'] = pass_means[nm]
    return TIM

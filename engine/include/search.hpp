// Search engine: best-first traversal of a proximity graph with a candidate list of L entries.
// Paper: Sec. III-A (graph search, completion time X), Sec. III-B (checkpoints and cost), Sec. III-C (declared
// interface, score, bucket, flag), Sec. VI-A (Compared methods, Implementation details).
//
// All searches run on the same Graph with the same distance kernel (hnswlib's inner product), the same
// visited tags and the same heaps, so they differ only in the stated way.
//
//   ANSWER  the traversal that maintains the greedy constrained answer Greedy(A_t) of the visited items A_t
//           in an answer queue (answer_queue.hpp). log_run records its trajectory; stop_run is the timed
//           online search with a stopping rule evaluated at the checkpoints.
//   LIST    search with a list of L entries, then Greedy over all visited items ("List size" of Sec. VI-A)
//   PLAIN   top-k of a search with a list of L entries (ignores the categories)
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "answer_queue.hpp"
#include "category_store.hpp"
#include "graph.hpp"

namespace cast {

enum Method : int { PLAIN = 0, LIST = 1, ANSWER = 2 };

struct Params {
  int method = PLAIN, k = 10, L = 10;
};

struct QueryResult {
  int nout = 0;
  int64_t ndist = 0, nvisit = 0, nexp = 0, ns = 0;
};

struct DI { float d; uint32_t i; };

// A flattened gradient-boosted forest (scikit-learn HistGradientBoosting classifier or regressor).
// raw(x) = base + sum of the leaf values; a non-leaf node sends x to the left child iff x[feat] <= thr, and NaN
// goes left iff miss_left (scikit-learn's predict semantics).
struct Forest {
  std::vector<int32_t> root, left, right; std::vector<int16_t> feat; std::vector<double> thr, val; std::vector<uint8_t> leaf, miss_left;
  double base = 0.0;
  double raw(const double* x) const {
    double s = base;
    for (size_t t = 0; t < root.size(); t++) {
      int32_t n = root[t];
      while (!leaf[n]) { const double v = x[feat[n]]; n = (std::isnan(v) ? miss_left[n] != 0 : v <= thr[n]) ? left[n] : right[n]; }
      s += val[n];
    }
    return s;
  }
};
// NumPy 1.26's float32 logarithm (simd_log_FLOAT in loops_exponent_log.dispatch.c.src, AVX512F / FMA3 paths) in
// scalar form, with the constants of that implementation (NumPy, BSD 3-Clause License, see THIRD_PARTY_NOTICES.md): frexp gives the same mantissa in [0.5, 1) and exponent as getmant(_MM_MANT_NORM_p5_1) and
// getexp + 1; same constants, same fused multiply-adds. The online features then equal the features computed
// offline with NumPy bit for bit.
inline float npy_logf(float v) {
  if (std::isnan(v)) return v;
  if (v < 0.0f) return -std::numeric_limits<float>::quiet_NaN();
  if (v == 0.0f) return -std::numeric_limits<float>::infinity();
  if (std::isinf(v)) return v;
  int e; float x = std::frexp(v, &e); float ex = (float)e;
  if (x <= 0.707106781186547524400844362104849039f) { x = x + x; ex = ex - 1.0f; }
  x = x - 1.0f;
  float num = std::fma(2.589979117907922693523e-02f, x, 3.808837741388407920751e-01f);
  num = std::fma(num, x, 1.480000633576506585156e+00f); num = std::fma(num, x, 2.112677543073053063722e+00f);
  num = std::fma(num, x, 9.999999999999998702752e-01f); num = std::fma(num, x, 0.0f);
  float den = std::fma(5.875095403124574342950e-03f, x, 1.546476374983906719538e-01f);
  den = std::fma(den, x, 9.864942958519418960339e-01f); den = std::fma(den, x, 2.453006071784736363091e+00f);
  den = std::fma(den, x, 2.612677543073109236779e+00f); den = std::fma(den, x, 1.0f);
  return std::fma(ex, 0.693147180559945309417232121458176568f, num / den);
}
// A stopping rule. Types 0 to 4 act at the checkpoints grid[0..J-1] (visit counts, Sec. III-B) and stop at the
// last checkpoint at the latest; types 6 to 10 act between checkpoints and have no terminal stop of their own.
//   0  no early stop (the search runs to its end, or to the last checkpoint)
//   1  staircase (our rule): stop iff bucket(m) <= beta[j][a], where m is the running minimum of the score
//      s = 1 - P(complete), bucket(m) = |{i : edges[i] <= m}| and a is the flag "the answer holds k items"
//   2  level with cap: stop iff m < lam or j >= jcap
//   3  fixed: stop at checkpoint j >= jfix (no model)
//   4  LAET-style: at checkpoint j0 predict the completion time X^ = exp(reg(features, flag)), then stop at the
//      first checkpoint j >= j0 with visits >= mult * X^
//   6  DARTH (Chatzakis et al., 2025), see stop_run
//   7  DARTH training log: never stop; once the answer is full, at every li-th distance computation record
//      the 11 answer features (19 with qstats) and the current recall of the answer
//   8  DARTH+ (Chatzakis et al., 2026), see stop_run
//   9  Patience (Teofili and Lin, 2025): stop at the first expansion at which the answer holds k items and
//      has not changed for pdelta visits, or at the first expansion with visits >= hvis
//  10  distance-adaptive (after Al-Jazzazi et al., 2025): stop at the first expansion at which the answer
//      holds k items and the closest unexpanded candidate is farther than (1 + gamma) times the k-th distance
//      of the answer, compared in float64
struct StopRule {
  int type = 0; std::vector<int32_t> grid; std::vector<double> edges; std::vector<int32_t> beta;
  double lam = 0.0; int jcap = 0, jfix = 0, j0 = 0; double mult = 0.0;
  const Forest* clf = nullptr; const Forest* reg = nullptr;
  std::vector<double> log_tab, log1p_tab;   // NumPy's float64 log(n) and log1p(n) for integer n (features 0 and 5), else std::log
  double Rt = 1.0; int ipi = 1, mpi = 1, li = 1;   // DARTH: recall target, initial and minimum prediction interval; log interval
  const Forest* qreg = nullptr; int qstats = 0;    // DARTH+: lower-quantile regressor; type 7 also records the query statistics
  int pdelta = 0, hvis = 0; double gamma = 0.0;    // Patience (type 9) and the distance-adaptive rule (type 10)
};
struct StopOut { int32_t stop_j = -1, nv = 0, calls = 0, qcalls = 0; int64_t hook_ns = 0; std::vector<double> feats, score; std::vector<uint8_t> full;
                 const int64_t* Sq = nullptr; std::vector<double> drows; };   // DARTH log rows: 11 (19 with qstats) features + recall (type 7)

struct MaxCmp { bool operator()(const DI& a, const DI& b) const { return prec(a.d, a.i, b.d, b.i); } };
struct MinCmp { bool operator()(const DI& a, const DI& b) const { return prec(b.d, b.i, a.d, a.i); } };

#define CAST_PREFETCH(p) __builtin_prefetch((const void*)(p), 0, 3)

class Searcher {
 public:
  Searcher(const Graph& g, const CategoryStore& cs, const Params& P)
      : g_(g), P_(P) {
    size_t N = g.n;
    vis_.assign(N, 0); exp_.assign(N, 0);
    greedy_.init(cs.m, P.k, cs.kappa.data(), cs.view());
    greedy_.set_sorted_input(true);
    if (P.method == ANSWER) D_.init(cs.m, P.k, cs.kappa.data(), cs.view());
  }

  // Trajectory log (never timed): one best-first run with a list of Lmax entries that maintains the answer
  // queue D = Greedy(A_t). Requires Params.method == ANSWER. At each expansion, before it happens, it records
  // the number of visits, the distance of the candidate to expand, the distance of the last answer item, the
  // k-th distance of the unconstrained top k, the answer size and the visit of the last answer change. It also
  // records the visit at which each member of the exact answer S was visited (their maximum is the completion
  // time X of Sec. III-A) and, for each list size in grid, the number of visits at which a search with that
  // list size ends (the stop of the rule "List size").
  struct LogOut {
    std::vector<int32_t> nv, asize, chg; std::vector<float> dx, dbk, dpk;
    std::vector<int32_t> rank_s, TL; int32_t final_nv = 0, ndesc = 0;
  };
  void log_run(const float* q, int Lmax, const int64_t* Sq, const std::vector<int>& grid, LogOut& o) {
    QueryResult R; new_epoch(); uint32_t ep; float dep; descend(q, ep, dep, R);
    o.ndesc = (int32_t)R.ndist;
    const uint16_t E = ep_; const int k = P_.k; const size_t G = grid.size();
    const float inf = std::numeric_limits<float>::infinity();
    D_.reset(); W_.clear(); C_.clear();
    std::vector<DI> pk; std::vector<std::vector<DI>> H(G); std::vector<uint8_t> stopped(G, 0);
    o.rank_s.assign(k, 0); o.TL.assign(G, 0);
    int32_t nv = 0, chg = 0;
    auto visit = [&](uint32_t u, float d) {
      nv++;
      for (int j = 0; j < k; j++) if (Sq[j] == (int64_t)u) o.rank_s[j] = nv;
      const int acc = D_.offer(u, d);
      if (acc) { chg = nv; }
      DI x{d, u};
      if ((int)pk.size() < k) { pk.push_back(x); std::push_heap(pk.begin(), pk.end(), MaxCmp()); }
      else if (prec(d, u, pk.front().d, pk.front().i)) { std::pop_heap(pk.begin(), pk.end(), MaxCmp()); pk.back() = x; std::push_heap(pk.begin(), pk.end(), MaxCmp()); }
      for (size_t g = 0; g < G; g++) {
        if (stopped[g]) continue;
        auto& h = H[g];
        if ((int)h.size() < grid[g]) { h.push_back(x); std::push_heap(h.begin(), h.end(), MaxCmp()); }
        else if (prec(d, u, h.front().d, h.front().i)) { std::pop_heap(h.begin(), h.end(), MaxCmp()); h.back() = x; std::push_heap(h.begin(), h.end(), MaxCmp()); }
      }
    };
    vis_[ep] = E; W_.push_back({dep, ep}); C_.push_back({dep, ep}); visit(ep, dep);
    float lower = dep;
    while (!C_.empty()) {
      DI top = C_.front();
      if ((int)W_.size() >= Lmax && top.d > lower) break;
      for (size_t g = 0; g < G; g++)
        if (!stopped[g] && (int)H[g].size() >= grid[g] && top.d > H[g].front().d) { stopped[g] = 1; o.TL[g] = nv; std::vector<DI>().swap(H[g]); }
      o.nv.push_back(nv); o.dx.push_back(top.d); o.dbk.push_back(D_.full() ? D_.max_dist() : inf);
      o.dpk.push_back((int)pk.size() == k ? pk.front().d : inf); o.asize.push_back(D_.size()); o.chg.push_back(chg);
      std::pop_heap(C_.begin(), C_.end(), MinCmp()); C_.pop_back();
      exp_[top.i] = E;
      const uint32_t* ll = g_.links0(top.i);
      uint32_t cnt = ll[0] & 0xFFFF;
      for (uint32_t j = 1; j <= cnt; j++) {
        uint32_t u = ll[j];
        if (vis_[u] == E) continue;
        vis_[u] = E;
        float d = g_.distance(q, u);
        visit(u, d);
        if ((int)W_.size() < Lmax || d < lower) {
          C_.push_back({d, u}); std::push_heap(C_.begin(), C_.end(), MinCmp());
          W_.push_back({d, u}); std::push_heap(W_.begin(), W_.end(), MaxCmp());
          if ((int)W_.size() > Lmax) { std::pop_heap(W_.begin(), W_.end(), MaxCmp()); W_.pop_back(); }
          lower = W_.front().d;
        }
      }
    }
    for (size_t g = 0; g < G; g++) if (!stopped[g]) o.TL[g] = nv;
    o.final_nv = nv; (void)inf;
  }

  // Timed online search: the trajectory of log_run (answer queue D, list of Lmax entries, same expansion and
  // insertion rules, with prefetching of the neighbors' tags and vectors), with the stopping rule evaluated at
  // every expansion whose visit count reaches the next checkpoint (Sec. III-B; several checkpoints reached at
  // one expansion are evaluated in order with the same state, as the offline reduction of the logs does).
  // The seven features of the score model at a checkpoint (Sec. VI-A), with nv the number of visits, dx the
  // distance of the candidate to expand, dbk the k-th distance of the answer, dpk the k-th distance of the
  // unconstrained top k and chg the visit of the last answer change:
  //   0 log(nv)   1 log(max(dx, 1e-9))   2 log(dbk / dx)   3 log(dpk / dx)   4 answer size
  //   5 log1p(nv - chg)   6 (nv - chg) / nv
  // Features 1 to 3 are float32 logarithms with +inf -> NaN (a missing value: the answer or the top k is not
  // full yet); they use NumPy's algorithm (npy_logf) and features 0 and 5 NumPy's tables, so the online features
  // equal, bit for bit, the features reduced offline with that NumPy build. x[7] is the flag a_j, read by the
  // LAET-style regressor.
  // dump: never stop; record the features, the score and the flag at every checkpoint.
  // instr: accumulate the time spent evaluating the rule.
  QueryResult stop_run(const float* q, int Lmax, const StopRule& rule, int64_t* out, StopOut& so, bool dump, bool instr) {
    QueryResult R; auto t0 = std::chrono::steady_clock::now();
    new_epoch(); uint32_t ep; float dep; descend(q, ep, dep, R);
    const uint16_t E = ep_; const int k = P_.k; const int J = (int)rule.grid.size();
    const float inf = std::numeric_limits<float>::infinity();
    const bool need_feat = dump || rule.type == 1 || rule.type == 2 || rule.type == 4;   // (DARTH uses its own features)
    D_.reset(); W_.clear(); C_.clear();
    std::vector<DI> pk; if (need_feat) pk.reserve(k + 1);
    int32_t nv = 0, chg = 0, ins = 0; int nj = 0; double m = std::numeric_limits<double>::infinity(), xpred = -1.0; double x[8];
    if (dump) { so.feats.assign((size_t)J * 7, 0.0); so.score.assign(J, 0.0); so.full.assign(J, 0); }
    auto visit = [&](uint32_t u, float d) {
      nv++;
      if (D_.offer(u, d)) { chg = nv; ins++; }
      if (need_feat) {
        DI z{d, u};
        if ((int)pk.size() < k) { pk.push_back(z); std::push_heap(pk.begin(), pk.end(), MaxCmp()); }
        else if (prec(d, u, pk.front().d, pk.front().i)) { std::pop_heap(pk.begin(), pk.end(), MaxCmp()); pk.back() = z; std::push_heap(pk.begin(), pk.end(), MaxCmp()); }
      }
    };
    auto features = [&](float dx) {
      const double cnv = (double)nv; const float dbk = D_.full() ? D_.max_dist() : inf, dpk = (int)pk.size() == k ? pk.front().d : inf;
      auto lr = [&](float num) {   // NumPy: log in float32, then nan_to_num(posinf=nan): NaN -> 0, +inf -> NaN, -inf -> -FLT_MAX
        const float v = npy_logf(num / dx);
        if (std::isnan(v)) return 0.0;
        if (std::isinf(v)) return v > 0 ? std::numeric_limits<double>::quiet_NaN() : (double)(-std::numeric_limits<float>::max());
        return (double)v; };
      const size_t a0 = (size_t)nv, a5 = (size_t)(nv - chg);
      x[0] = a0 < rule.log_tab.size() ? rule.log_tab[a0] : std::log(cnv); x[1] = (double)npy_logf(std::max(dx, 1e-9f)); x[2] = lr(dbk); x[3] = lr(dpk);
      x[4] = (double)D_.size(); x[5] = a5 < rule.log1p_tab.size() ? rule.log1p_tab[a5] : std::log1p(cnv - (double)chg); x[6] = (cnv - (double)chg) / cnv;
      x[7] = D_.size() >= (size_t)k ? 1.0 : 0.0;
    };
    auto bucket = [&](double v) { return (int)(std::upper_bound(rule.edges.begin(), rule.edges.end(), v) - rule.edges.begin()); };
    // DARTH: features of the returned answer (the answer queue D), in the training order of DARTH's reference code:
    // step (expansions), dists (distance computations), inserts (answers accepted), first_nn_dist (bottom-layer entry distance), nn_dist,
    // furthest_dist, avg_dist, variance (E d^2 - mean^2), percentile_25/50/75 (sorted index floor(p (k - 1)), as its nth_element).
    const bool darth = rule.type == 6 || rule.type == 7 || rule.type == 8; int32_t dint = 0, dpi = rule.ipi; bool dstop = false, qsw = false; double xd[19]; float dd[64];
    // DARTH+: the eight query statistics, computed once per query (mean, median, standard deviation, minimum, maximum, range,
    // L1 and L2 norm of the query's dimensions) as features 11-18.
    const bool qst = rule.type == 8 || (rule.type == 7 && rule.qstats);
    if (qst) {
      const size_t dm = g_.dim; std::vector<double> qv(q, q + dm); double sm = 0.0, l1 = 0.0, l2 = 0.0, mn = qv[0], mx = qv[0];
      for (double v : qv) { sm += v; l1 += std::fabs(v); l2 += v * v; mn = std::min(mn, v); mx = std::max(mx, v); }
      const double mean = sm / (double)dm; double ss = 0.0; for (double v : qv) ss += (v - mean) * (v - mean);
      std::sort(qv.begin(), qv.end()); const double med = dm % 2 ? qv[dm / 2] : 0.5 * (qv[dm / 2 - 1] + qv[dm / 2]);
      xd[11] = mean; xd[12] = med; xd[13] = std::sqrt(ss / (double)dm); xd[14] = mn; xd[15] = mx; xd[16] = mx - mn; xd[17] = l1; xd[18] = std::sqrt(l2);
    }
    const int nfd = qst ? 19 : 11;
    auto dfeat = [&]() {
      const int n = (int)D_.size(); double sm = 0.0, s2 = 0.0;
      for (int t = 0; t < n; t++) { dd[t] = D_.member_dist(t); sm += dd[t]; s2 += (double)dd[t] * dd[t]; }
      std::sort(dd, dd + n); const double mean = sm / n;
      xd[0] = (double)R.nexp; xd[1] = (double)nv; xd[2] = (double)ins; xd[3] = (double)dep; xd[4] = dd[0]; xd[5] = dd[n - 1]; xd[6] = mean;
      xd[7] = s2 / n - mean * mean; xd[8] = dd[(int)(0.25 * (n - 1))]; xd[9] = dd[(int)(0.5 * (n - 1))]; xd[10] = dd[(int)(0.75 * (n - 1))];
    };
    auto darth_after_distance = [&]() {   // called after every bottom-layer distance computation (DARTH checks per distance)
      dint++;
      if (rule.type == 7) {
        if (D_.full() && dint % rule.li == 0) {
          dfeat(); int hit = 0;
          for (int t = 0; t < (int)D_.size(); t++) for (int u = 0; u < k; u++) if ((int64_t)D_.member(t) == so.Sq[u]) { hit++; break; }
          for (int f = 0; f < nfd; f++) so.drows.push_back(xd[f]); so.drows.push_back((double)hit / k);
        }
        return;
      }
      if (dint != dpi) return;                                  // dint counts up from 0 and is reset at each prediction: same as DARTH's % test
      dint = 0; so.calls++;
      if (!D_.full()) return;                                   // DARTH's predictor returns 0 and keeps the interval
      std::chrono::steady_clock::time_point h0; if (instr) h0 = std::chrono::steady_clock::now();
      dfeat(); double rp;
      if (rule.type == 6) rp = std::min(1.0, std::max(0.0, rule.reg->raw(xd)));
      else {   // DARTH+: the recall regressor until its prediction first reaches Rt, then the lower-quantile model only;
               // stop once the lower bound reaches Rt; the next interval follows the prediction just made (the quantile one after the switch)
        if (!qsw) { rp = std::min(1.0, std::max(0.0, rule.reg->raw(xd))); if (rp >= rule.Rt) qsw = true; }
        if (qsw) { rp = std::min(1.0, std::max(0.0, rule.qreg->raw(xd))); so.qcalls++; }
      }
      if (instr) so.hook_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - h0).count();
      if (rp >= rule.Rt) { dstop = true; return; }
      dpi = (int)(rule.mpi + (rule.ipi - rule.mpi) * (rule.Rt - rp)); if (dpi < 1) dpi = 1;
    };
    auto decide = [&](int j, float dx) -> bool {   // true = stop at checkpoint j
      if (dump) {
        features(dx); const double sc = 1.0 / (1.0 + std::exp(rule.clf->raw(x)));   // s = 1 - expit(raw)
        for (int f = 0; f < 7; f++) so.feats[(size_t)j * 7 + f] = x[f];
        so.score[j] = sc; so.full[j] = x[7] > 0.5; return false;
      }
      if (j >= J - 1) return true;
      switch (rule.type) {
        case 1: { features(dx); m = std::min(m, 1.0 / (1.0 + std::exp(rule.clf->raw(x)))); const int a = x[7] > 0.5;
                  return bucket(m) <= rule.beta[(size_t)j * 2 + a]; }
        case 2: { features(dx); m = std::min(m, 1.0 / (1.0 + std::exp(rule.clf->raw(x)))); return m < rule.lam || j >= rule.jcap; }
        case 3: return j >= rule.jfix;
        case 4: { if (j < rule.j0) return false; if (xpred < 0) { features(dx); xpred = std::exp(rule.reg->raw(x)); }
                  return (double)nv >= rule.mult * xpred; }
        default: return false;
      }
    };
    vis_[ep] = E; W_.push_back({dep, ep}); C_.push_back({dep, ep}); visit(ep, dep); if (darth) darth_after_distance();
    float lower = dep; bool stop = false;
    while (!C_.empty() && !dstop) {
      DI top = C_.front();
      if ((int)W_.size() >= Lmax && top.d > lower) break;
      if (rule.type == 9 && ((rule.hvis > 0 && nv >= rule.hvis) || (D_.full() && nv - chg >= rule.pdelta))) { stop = true; break; }
      if (rule.type == 10 && D_.full() && (double)top.d > (1.0 + rule.gamma) * (double)D_.max_dist()) { stop = true; break; }
      if (!darth && rule.type != 9 && rule.type != 10 && nj < J && nv >= rule.grid[nj]) {
        std::chrono::steady_clock::time_point h0; if (instr) h0 = std::chrono::steady_clock::now();
        while (nj < J && nv >= rule.grid[nj]) { if (decide(nj, top.d)) { stop = true; break; } nj++; }
        if (instr) so.hook_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - h0).count();
        if (stop) break;
      }
      std::pop_heap(C_.begin(), C_.end(), MinCmp()); C_.pop_back();
      exp_[top.i] = E; R.nexp++;
      const uint32_t* ll = g_.links0(top.i);
      uint32_t cnt = ll[0] & 0xFFFF;
      CAST_PREFETCH(vis_.data() + ll[1]); CAST_PREFETCH(g_.vec(ll[1]));
      for (uint32_t j = 1; j <= cnt; j++) {
        uint32_t u = ll[j];
        if (j < cnt) { CAST_PREFETCH(vis_.data() + ll[j + 1]); CAST_PREFETCH(g_.vec(ll[j + 1])); }
        if (vis_[u] == E) continue;
        vis_[u] = E;
        float d = g_.distance(q, u); R.ndist++;
        visit(u, d);
        if ((int)W_.size() < Lmax || d < lower) {
          C_.push_back({d, u}); std::push_heap(C_.begin(), C_.end(), MinCmp());
          W_.push_back({d, u}); std::push_heap(W_.begin(), W_.end(), MaxCmp());
          if ((int)W_.size() > Lmax) { std::pop_heap(W_.begin(), W_.end(), MaxCmp()); W_.pop_back(); }
          lower = W_.front().d;
        }
        if (darth) { darth_after_distance(); if (dstop) break; }   // stop inside an expansion, as DARTH does
      }
    }
    so.stop_j = (stop || dstop) ? nj : -1; so.nv = nv; R.nvisit = nv;
    R.nout = emit(D_, out);
    for (int j = R.nout; j < k; j++) out[j] = -1;
    R.ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return R;
  }

  // Run one query with method LIST or PLAIN. out receives k ids (-1 padded) in (distance, id) order.
  QueryResult run(const float* q, int L, int64_t* out) {
    QueryResult R;
    new_epoch();
    uint32_t ep; float dep;
    descend(q, ep, dep, R);
    bool rec = P_.method == LIST;
    beam_init(ep, dep, rec, R);
    beam_run(q, L, rec, R);
    if (P_.method == PLAIN) R.nout = topk_of_W(P_.k, out);
    else { pool_.assign(V_.begin(), V_.end()); R.nout = greedy_pool(out); }
    for (int j = R.nout; j < P_.k; j++) out[j] = -1;
    return R;
  }

 private:
  // ----------------------------------------------------------------- tags
  void new_epoch() {
    if (++ep_ == 0) {
      std::fill(vis_.begin(), vis_.end(), 0); std::fill(exp_.begin(), exp_.end(), 0);
      ep_ = 1;
    }
  }

  // ----------------------------------------------------------------- upper layers
  void descend(const float* q, uint32_t& cur, float& dcur, QueryResult& R) {
    cur = g_.entry;
    dcur = g_.distance(q, cur); R.ndist++;
    for (int lvl = g_.maxlevel; lvl > 0; lvl--) {
      bool changed = true;
      while (changed) {
        changed = false;
        const uint32_t* ll = g_.links_up(cur, lvl);
        uint32_t cnt = ll[0] & 0xFFFF;
        for (uint32_t j = 1; j <= cnt; j++) {
          uint32_t c = ll[j];
          float d = g_.distance(q, c); R.ndist++;
          if (d < dcur) { dcur = d; cur = c; changed = true; }
        }
      }
    }
  }

  // ----------------------------------------------------------------- search with a list of L entries
  void beam_init(uint32_t ep, float dep, bool rec, QueryResult& R) {
    W_.clear(); C_.clear(); V_.clear();
    vis_[ep] = ep_;
    W_.push_back({dep, ep}); C_.push_back({dep, ep});
    if (rec) V_.push_back({dep, ep});
    R.nvisit = 1;
  }
  void beam_run(const float* q, int ef, bool rec, QueryResult& R) {
    float lower = W_.empty() ? std::numeric_limits<float>::infinity() : W_.front().d;
    const uint16_t E = ep_;
    while (!C_.empty()) {
      DI top = C_.front();
      if ((int)W_.size() >= ef && top.d > lower) break;
      std::pop_heap(C_.begin(), C_.end(), MinCmp()); C_.pop_back();
      exp_[top.i] = E;
      R.nexp++;
      const uint32_t* ll = g_.links0(top.i);
      uint32_t cnt = ll[0] & 0xFFFF;
      CAST_PREFETCH(vis_.data() + ll[1]);
      CAST_PREFETCH(g_.vec(ll[1]));
      for (uint32_t j = 1; j <= cnt; j++) {
        uint32_t u = ll[j];
        if (j < cnt) { CAST_PREFETCH(vis_.data() + ll[j + 1]); CAST_PREFETCH(g_.vec(ll[j + 1])); }
        if (vis_[u] == E) continue;
        vis_[u] = E;
        float d = g_.distance(q, u); R.ndist++; R.nvisit++;
        if (rec) V_.push_back({d, u});
        if ((int)W_.size() < ef || d < lower) {
          C_.push_back({d, u}); std::push_heap(C_.begin(), C_.end(), MinCmp());
          W_.push_back({d, u}); std::push_heap(W_.begin(), W_.end(), MaxCmp());
          if ((int)W_.size() > ef) { std::pop_heap(W_.begin(), W_.end(), MaxCmp()); W_.pop_back(); }
          lower = W_.front().d;
        }
      }
    }
  }
  int topk_of_W(int k, int64_t* out) {
    std::sort(W_.begin(), W_.end(), MaxCmp());
    int n = std::min<int>(k, (int)W_.size());
    for (int j = 0; j < n; j++) out[j] = W_[j].i;
    return n;
  }
  // Greedy over pool_ in (distance, id) order, truncated at k items
  int greedy_pool(int64_t* out) {
    std::make_heap(pool_.begin(), pool_.end(), MinCmp());
    greedy_.reset();
    size_t n = pool_.size();
    while (n > 0 && greedy_.size() < P_.k) {
      std::pop_heap(pool_.begin(), pool_.begin() + n, MinCmp());
      DI e = pool_[--n];
      greedy_.offer(e.i, e.d);
    }
    return emit(greedy_, out);
  }
  // the answer items of a queue in (distance, id) order
  int emit(const AnswerQueue& Qu, int64_t* out) {
    tmp_.clear();
    for (int s = 0; s < Qu.size(); s++) tmp_.push_back({Qu.member_dist(s), Qu.member(s)});
    std::sort(tmp_.begin(), tmp_.end(), MaxCmp());
    int n = 0;
    for (const DI& x : tmp_) out[n++] = (int64_t)x.i;
    return n;
  }

  const Graph& g_;
  Params P_;
  uint16_t ep_ = 0;
  std::vector<uint16_t> vis_, exp_;   // visited and expanded tags of the current query (epoch stamps)
  std::vector<DI> W_, C_, V_, pool_, tmp_;
  AnswerQueue greedy_, D_;
};

}  // namespace cast

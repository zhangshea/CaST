// Differential test of the answer queue (include/answer_queue.hpp) against Greedy computed from scratch.
// Paper: Sec. III-A (the search maintains Greedy(A_t) exactly as visits arrive; Definition 1).
//
// Each trace draws a random instance (n items, m categories, |c(v)| <= s, capacities kappa, distances with
// deliberate ties broken by id), an answer size K, and a random offer order. After every checked offer the
// answer of the queue must equal Greedy of the offered prefix A_t, computed from scratch (sort A_t by
// (distance, id), keep an item iff a matching computed anew assigns the kept set plus the item, stop at K
// items); the assignment of the queue must be valid.
//
//   test_answer_queue [traces] [seed]
// Runs offers in random order on `traces` small traces (every offer checked) and on larger traces, then
// offers in sorted order (the scan of Greedy over a list). Exit code 0 iff no mismatch.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "../include/answer_queue.hpp"

using namespace cast;

struct Inst {
  int n, m, K;
  std::vector<int64_t> off; std::vector<uint32_t> col; std::vector<int32_t> kappa; std::vector<float> d;
};

static Inst draw(std::mt19937_64& rng, int nmax, int mmax, int smax, int kmax, int kapmax) {
  Inst I;
  I.n = 1 + rng() % nmax; I.m = 1 + rng() % mmax; int s = 1 + rng() % smax; I.K = 1 + rng() % kmax;
  I.kappa.resize(I.m);
  bool uniform = rng() % 2;
  int ku = 1 + rng() % kapmax;
  for (int g = 0; g < I.m; g++) I.kappa[g] = uniform ? ku : 1 + rng() % kapmax;
  I.off.push_back(0);
  for (int v = 0; v < I.n; v++) {
    int deg = 1 + rng() % std::min(s, I.m);
    std::vector<uint32_t> cs;
    while ((int)cs.size() < deg) { uint32_t g = rng() % I.m; if (std::find(cs.begin(), cs.end(), g) == cs.end()) cs.push_back(g); }
    for (auto g : cs) I.col.push_back(g);
    I.off.push_back(I.col.size());
  }
  int levels = 2 + rng() % 20;   // few distinct values -> many ties
  I.d.resize(I.n);
  for (int v = 0; v < I.n; v++) I.d[v] = (float)(rng() % levels) / levels;
  return I;
}

// reference: Kuhn's algorithm on unit category slots, written independently of the queue
static bool ref_feasible(const Inst& I, const std::vector<int>& S) {
  std::vector<std::vector<int>> slots(I.m);
  std::vector<int> owner;   // per slot
  std::vector<int> slot_cat;
  for (int g = 0; g < I.m; g++) for (int t = 0; t < I.kappa[g]; t++) { slots[g].push_back(owner.size()); owner.push_back(-1); slot_cat.push_back(g); }
  std::vector<char> seen;
  std::function<bool(int)> aug = [&](int u) -> bool {
    int v = S[u];
    for (int64_t p = I.off[v]; p < I.off[v + 1]; p++)
      for (int t : slots[I.col[p]]) {
        if (seen[t]) continue;
        seen[t] = 1;
        if (owner[t] < 0 || aug(owner[t])) { owner[t] = u; return true; }
      }
    return false;
  };
  for (int u = 0; u < (int)S.size(); u++) { seen.assign(owner.size(), 0); if (!aug(u)) return false; }
  return true;
}

static std::vector<int> ref_greedy(const Inst& I, std::vector<int> U, int K) {
  std::sort(U.begin(), U.end(), [&](int a, int b) { return prec(I.d[a], a, I.d[b], b); });
  std::vector<int> B;
  for (int v : U) {
    if ((int)B.size() == K) break;
    B.push_back(v);
    if (!ref_feasible(I, B)) B.pop_back();
  }
  std::sort(B.begin(), B.end());
  return B;
}

static bool check_phi(const Inst& I, AnswerQueue& Q) {
  std::vector<int> load(I.m, 0);
  for (int s = 0; s < Q.size(); s++) {
    uint32_t v = Q.member(s); int g = Q.member_category(s);
    bool ok = false;
    for (int64_t p = I.off[v]; p < I.off[v + 1]; p++) ok |= ((int)I.col[p] == g);
    if (!ok) return false;
    if (++load[g] > I.kappa[g]) return false;
  }
  return true;
}

struct Result { long traces = 0, checks = 0, mismatches = 0; };

static Result run(bool sorted, long traces, int nmax, int mmax, int smax, int kmax, int kapmax,
                  int check_every, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Result R;
  for (long t = 0; t < traces; t++) {
    Inst I = draw(rng, nmax, mmax, smax, kmax, kapmax);
    CategoryView cv{I.off.data(), I.col.data()};
    AnswerQueue Q;
    Q.init(I.m, I.K, I.kappa.data(), cv);
    Q.set_sorted_input(sorted);
    Q.reset();
    std::vector<int> order(I.n);
    for (int v = 0; v < I.n; v++) order[v] = v;
    if (sorted) std::sort(order.begin(), order.end(), [&](int a, int b) { return prec(I.d[a], a, I.d[b], b); });
    else std::shuffle(order.begin(), order.end(), rng);
    int len = sorted ? I.n : 1 + rng() % I.n;
    std::vector<int> U;
    for (int i = 0; i < len; i++) {
      int v = order[i];
      Q.offer(v, I.d[v]);
      U.push_back(v);
      if (i % check_every == 0 || i == len - 1) {
        std::vector<int> ref = ref_greedy(I, U, I.K);
        std::vector<uint32_t> got(Q.size());
        Q.members(got.data());
        std::sort(got.begin(), got.end());
        bool same = got.size() == ref.size();
        for (size_t j = 0; same && j < got.size(); j++) same = (int)got[j] == ref[j];
        same = same && check_phi(I, Q);
        R.checks++;
        if (!same) {
          if (R.mismatches < 5) {
            std::fprintf(stderr, "MISMATCH sorted=%d trace=%ld step=%d n=%d m=%d K=%d\n", (int)sorted, t, i, I.n, I.m, I.K);
          }
          R.mismatches++;
          break;
        }
      }
    }
    R.traces++;
  }
  return R;
}

int main(int argc, char** argv) {
  long traces = argc > 1 ? std::atol(argv[1]) : 1000000;
  uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  long total_mis = 0;
  auto report = [&](const char* name, const Result& r, double sec) {
    std::printf("%-34s traces=%ld checks=%ld mismatches=%ld  %.1fs\n", name, r.traces, r.checks, r.mismatches, sec);
    total_mis += r.mismatches;
  };
  auto timed = [&](const char* name, bool sorted, long tr, int nmax, int mmax, int smax, int kmax, int kap,
                   int every, uint64_t sd) {
    auto t0 = std::chrono::steady_clock::now();
    Result r = run(sorted, tr, nmax, mmax, smax, kmax, kap, every, sd);
    report(name, r, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  };
  // the offsets of the seed keep the four runs on distinct random streams
  timed("any order, small, every offer", false, traces, 16, 8, 4, 10, 3, 1, seed);
  timed("any order, large, every 3rd offer", false, traces / 10, 60, 14, 4, 24, 3, 3, seed + 1);
  timed("any order, large K", false, traces / 20, 80, 20, 3, 60, 8, 7, seed + 2);
  timed("sorted order, small", true, traces / 5, 30, 8, 4, 10, 3, 1, seed + 3);
  std::printf("TOTAL mismatches=%ld\n", total_mis);
  return total_mis ? 1 : 0;
}

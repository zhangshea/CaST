// Build a category-agnostic HNSW graph with hnswlib (inner-product space, distance 1 - <q, x>).
// Paper: Sec. VI-A, Implementation details (graphs A and B: degree parameter 32, construction list 200, two
// independent draws of the insertion levels).
//
//   build_hnsw --base emb_base.npy --out index.hnsw [--info info.json] [--M 32] [--efc 200] [--seed S] [--threads 1]
//
// The graph has one node per base row. With --threads 1 (default) the rows are inserted in id order, so
// node id = row id and the graph is a deterministic function of (data, M, efc, seed); hnswlib's generator of
// the insertion levels is seeded with `seed`. With several threads the insertion order, and with it the node
// ids, depend on the schedule: the tool then reports the number of nodes whose id differs from their label and
// exits with a non-zero code, and the caller renumbers the items by node id.
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "../include/npy.hpp"
#include "hnswlib.h"

using namespace cast;

static double cpu_seconds() {
  rusage ru; getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}
static long peak_rss_kb() { rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss; }

int main(int argc, char** argv) {
  std::string base, out, info;
  size_t M = 32, efc = 200, seed = 100; int threads = 1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
    if (a == "--base") base = nx(); else if (a == "--out") out = nx(); else if (a == "--info") info = nx();
    else if (a == "--M") M = std::stoull(nx());
    else if (a == "--efc") efc = std::stoull(nx()); else if (a == "--seed") seed = std::stoull(nx());
    else if (a == "--threads") threads = std::stoi(nx());
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (base.empty() || out.empty()) { std::fprintf(stderr, "need --base and --out\n"); return 2; }
  auto t_load0 = std::chrono::steady_clock::now();
  NpyArray X = read_npy(base);
  if (X.dtype != "<f4" || X.shape.size() != 2) { std::fprintf(stderr, "base must be float32 [n, d]\n"); return 2; }
  size_t n = X.rows(), d = X.cols();
  size_t N = n;
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
  hnswlib::InnerProductSpace space(d);
  auto* index = new hnswlib::HierarchicalNSW<float>(&space, N, M, efc, seed);
  const float* Xd = X.data<float>();
  auto row = [&](size_t c) -> const float* { return Xd + c * d; };
  double c0 = cpu_seconds();
  auto t0 = std::chrono::steady_clock::now();
  if (threads <= 1) {
    for (size_t c = 0; c < N; c++) {
      index->addPoint(row(c), (hnswlib::labeltype)c);
      if ((c + 1) % 200000 == 0) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double dg = 0;
        for (size_t u = 0; u <= c; u++) dg += index->getListCount(index->get_linklist0((hnswlib::tableint)u));
        std::fprintf(stderr, "added %zu / %zu  %.0fs  mean level-0 degree so far %.3f\n", c + 1, N, el, dg / (double)(c + 1));
      }
    }
  } else {
    // multi-threaded insertion (not deterministic); the first point is inserted alone
    index->addPoint(row(0), 0);
    std::atomic<size_t> next(1);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back([&]() {
      for (;;) { size_t c = next.fetch_add(1); if (c >= N) break; index->addPoint(row(c), (hnswlib::labeltype)c); }
    });
    for (auto& th : pool) th.join();
  }
  double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double build_cpu = cpu_seconds() - c0;
  index->saveIndex(out);
  // label check: the search engine needs internal id == label for every node
  size_t bad = 0;
  for (size_t c = 0; c < N; c++) if (index->getExternalLabel((hnswlib::tableint)c) != c) bad++;
  double deg0 = 0;
  for (size_t c = 0; c < N; c++) deg0 += index->getListCount(index->get_linklist0((hnswlib::tableint)c));
  deg0 /= (double)N;
  size_t upper = 0, nlev = 0;
  for (size_t c = 0; c < N; c++) { int l = index->element_levels_[c]; if (l > 0) { upper += (size_t)l * index->size_links_per_element_; nlev++; } }
  if (!info.empty()) {
    std::ofstream js(info);
    js << "{\n \"n_nodes\": " << N << ",\n \"n_base\": " << n << ",\n \"dim\": " << d << ",\n \"M\": " << M
       << ",\n \"efc\": " << efc << ",\n \"seed\": " << seed << ",\n \"threads\": " << threads
       << ",\n \"load_s\": " << load_s << ",\n \"build_s\": " << build_s << ",\n \"build_cpu_s\": " << build_cpu
       << ",\n \"peak_rss_kb\": " << peak_rss_kb() << ",\n \"maxlevel\": " << index->maxlevel_
       << ",\n \"entry\": " << index->enterpoint_node_ << ",\n \"size_data_per_element\": " << index->size_data_per_element_
       << ",\n \"level0_bytes\": " << N * index->size_data_per_element_ << ",\n \"upper_bytes\": " << upper
       << ",\n \"n_upper_nodes\": " << nlev << ",\n \"label_mismatch\": " << bad << ",\n \"mean_degree0\": " << deg0
       << "\n}\n";
  }
  std::fprintf(stderr, "done: %zu nodes, build %.1fs (cpu %.1fs), peak rss %ld kB, label mismatches %zu, mean level-0 degree %.3f\n", N, build_s,
               build_cpu, peak_rss_kb(), bad, deg0);
  delete index;
  return bad ? 1 : 0;
}

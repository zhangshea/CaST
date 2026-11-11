// Apply an index update to an existing hnswlib index with the library's public calls only (one thread).
// Paper: Sec. VI-B, "Query shift and index updates" (inserts of further items, in-place replacements of base items).
//
//   update_hnsw --in A.hnsw --out B.hnsw --vec items.npy --mode insert|replace [--slots slots.npy] --seed S [--info info.json]
//
// insert : the index is loaded with room for n + m points and addPoint(item_i, label n + i) is called in order, so node id = label
//          (the engine's requirement); hnswlib's generator of the insertion levels is seeded with S first (after a load it is
//          default-seeded).
// replace: addPoint(item_i, label slots_i) for an existing label, which hnswlib performs as an in-place update (updatePoint with
//          neighbor-update probability 1: the vector is replaced and the neighborhood repaired), i.e. the slot of a base item is
//          reused by the replacing item; ids do not change.
// The label check of build_hnsw (internal id == label for every node) is repeated on the result.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../include/npy.hpp"
#include "hnswlib.h"

using namespace cast;

int main(int argc, char** argv) {
  std::string in, out, vec, mode, slots, info; size_t seed = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
    if (a == "--in") in = nx(); else if (a == "--out") out = nx(); else if (a == "--vec") vec = nx(); else if (a == "--mode") mode = nx();
    else if (a == "--slots") slots = nx(); else if (a == "--seed") seed = std::stoull(nx()); else if (a == "--info") info = nx();
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (in.empty() || out.empty() || vec.empty() || (mode != "insert" && mode != "replace")) { std::fprintf(stderr, "need --in --out --vec --mode\n"); return 2; }
  NpyArray V = read_npy(vec);
  if (V.dtype != "<f4" || V.shape.size() != 2) { std::fprintf(stderr, "vectors must be float32 [m, d]\n"); return 2; }
  const size_t m = V.rows(), d = V.cols();
  std::vector<int64_t> S;
  if (mode == "replace") {
    NpyArray P = read_npy(slots);
    if (P.dtype != "<i8" || P.rows() != m) { std::fprintf(stderr, "slots must be int64 [m]\n"); return 2; }
    S.assign(P.data<int64_t>(), P.data<int64_t>() + m);
  }
  hnswlib::InnerProductSpace space(d);
  auto* probe = new hnswlib::HierarchicalNSW<float>(&space, in);   // to read the stored size
  const size_t n0 = probe->cur_element_count; delete probe;
  auto* index = new hnswlib::HierarchicalNSW<float>(&space, in, false, mode == "insert" ? n0 + m : n0);
  index->level_generator_.seed(seed);
  const float* X = V.data<float>(); auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < m; i++) {
    if (mode == "insert") index->addPoint(X + i * d, (hnswlib::labeltype)(n0 + i));
    else {
      if (S[i] < 0 || (size_t)S[i] >= n0) { std::fprintf(stderr, "slot out of range\n"); return 2; }
      index->addPoint(X + i * d, (hnswlib::labeltype)S[i]);
    }
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  index->saveIndex(out);
  const size_t N = index->cur_element_count; size_t bad = 0, del = 0; double deg0 = 0;
  for (size_t c = 0; c < N; c++) {
    if (index->getExternalLabel((hnswlib::tableint)c) != c) bad++;
    if (index->isMarkedDeleted((hnswlib::tableint)c)) del++;
    deg0 += index->getListCount(index->get_linklist0((hnswlib::tableint)c));
  }
  deg0 /= (double)N;
  if (!info.empty()) {
    std::ofstream js(info);
    js << "{\n \"mode\": \"" << mode << "\",\n \"n_before\": " << n0 << ",\n \"m\": " << m << ",\n \"n_after\": " << N << ",\n \"dim\": " << d
       << ",\n \"seed\": " << seed << ",\n \"secs\": " << secs << ",\n \"label_mismatch\": " << bad << ",\n \"marked_deleted\": " << del
       << ",\n \"mean_degree0\": " << deg0 << ",\n \"maxlevel\": " << index->maxlevel_ << ",\n \"entry\": " << index->enterpoint_node_ << "\n}\n";
  }
  std::fprintf(stderr, "done: %s %zu points in %.1fs, %zu -> %zu nodes, label mismatches %zu, marked deleted %zu, mean level-0 degree %.3f\n",
               mode.c_str(), m, secs, n0, N, bad, del, deg0);
  delete index;
  return bad || del ? 1 : 0;
}

// Proximity-graph storage of the search engine. Paper: Sec. III-A (graph search) and Sec. VI-A, Implementation
// details (HNSW graphs A and B, Vamana graphs).
//
// The bottom layer uses hnswlib's interleaved element layout (link count, maxM0 neighbor ids, vector, 8-byte
// label), so one cache-line stream serves both the adjacency list and the vector of a node, exactly as in
// hnswlib. Upper layers are kept per node as fixed-size lists [count, maxM ids].
// Two loaders: an hnswlib index file (parsed directly), or arrays exported from another graph
// implementation (used for the Vamana graphs). The distance is hnswlib's inner-product kernel,
// d(q, x) = 1 - <q, x>.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib.h"

namespace cast {

struct Graph {
  size_t n = 0, dim = 0, maxM0 = 0, maxM = 0;
  size_t stride = 0, off_data = 0, off_label = 0;
  char* blob = nullptr;
  std::vector<int32_t> level;        // top layer of each node
  std::vector<int64_t> upper_ptr;    // offset into `upper` (uint32 units) or -1
  std::vector<uint32_t> upper;       // per node, per layer 1..level: [count, maxM ids]
  uint32_t entry = 0;
  int maxlevel = 0;
  std::unique_ptr<hnswlib::InnerProductSpace> space;
  hnswlib::DISTFUNC<float> dist = nullptr;
  void* dist_param = nullptr;

  Graph() = default;
  Graph(const Graph&) = delete;
  ~Graph() { std::free(blob); }

  inline const uint32_t* links0(uint32_t i) const { return reinterpret_cast<const uint32_t*>(blob + i * stride); }
  inline const char* vec(uint32_t i) const { return blob + i * stride + off_data; }
  inline const uint32_t* links_up(uint32_t i, int lvl) const { return upper.data() + upper_ptr[i] + (size_t)(lvl - 1) * (maxM + 1); }
  inline float distance(const void* q, uint32_t i) const { return dist(q, vec(i), dist_param); }

  void alloc(size_t n_, size_t dim_, size_t maxM0_, size_t maxM_) {
    n = n_; dim = dim_; maxM0 = maxM0_; maxM = maxM_;
    off_data = sizeof(uint32_t) * (maxM0 + 1);
    off_label = off_data + dim * sizeof(float);
    stride = off_label + sizeof(size_t);
    size_t bytes = n * stride;
    blob = static_cast<char*>(std::aligned_alloc(64, ((bytes + 63) / 64) * 64));
    if (!blob) throw std::runtime_error("out of memory");
    space.reset(new hnswlib::InnerProductSpace(dim));
    dist = space->get_dist_func(); dist_param = space->get_dist_func_param();
    level.assign(n, 0); upper_ptr.assign(n, -1);
  }

  // mean out-degree on the bottom layer
  double mean_degree0() const {
    double s = 0; for (size_t i = 0; i < n; i++) s += (links0(i)[0] & 0xFFFF); return n ? s / n : 0;
  }
};

template <class T> static void rd(std::ifstream& in, T& x) { in.read(reinterpret_cast<char*>(&x), sizeof(T)); }

// Parse an hnswlib (v0.8.0) index file written by HierarchicalNSW::saveIndex.
inline std::unique_ptr<Graph> load_hnswlib(const std::string& path, size_t dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  size_t offsetLevel0, max_elements, cur_count, size_per_elem, label_offset, offset_data, maxM, maxM0, M, efc;
  int maxlevel; uint32_t entry; double mult;
  rd(in, offsetLevel0); rd(in, max_elements); rd(in, cur_count); rd(in, size_per_elem); rd(in, label_offset);
  rd(in, offset_data); rd(in, maxlevel); rd(in, entry); rd(in, maxM); rd(in, maxM0); rd(in, M); rd(in, mult); rd(in, efc);
  auto g = std::make_unique<Graph>();
  g->alloc(cur_count, dim, maxM0, maxM);
  if (g->stride != size_per_elem || g->off_data != offset_data || g->off_label != label_offset || offsetLevel0 != 0)
    throw std::runtime_error("unexpected hnswlib layout (dim mismatch?)");
  in.read(g->blob, cur_count * size_per_elem);
  size_t per = (maxM + 1) * sizeof(uint32_t);
  for (size_t i = 0; i < cur_count; i++) {
    uint32_t sz; rd(in, sz);
    if (sz) {
      int L = (int)(sz / per);
      g->level[i] = L;
      g->upper_ptr[i] = (int64_t)g->upper.size();
      size_t at = g->upper.size();
      g->upper.resize(at + sz / sizeof(uint32_t));
      in.read(reinterpret_cast<char*>(g->upper.data() + at), sz);
    }
  }
  if (!in) throw std::runtime_error("truncated index " + path);
  g->entry = entry; g->maxlevel = maxlevel;
  // node id must equal the external label (the build tool inserts in id order)
  for (size_t i = 0; i < cur_count; i += 997) {
    size_t lab; std::memcpy(&lab, g->blob + i * g->stride + g->off_label, sizeof(size_t));
    if (lab != i) throw std::runtime_error("label != node id in " + path);
  }
  return g;
}

// Build from exported arrays: X [n, dim] float32, nbr0 [n, maxM0] int32 (-1 padded), level [n],
// upper_ptr [n] int64 and upper (uint32, per node and layer [count, maxM ids]).
inline std::unique_ptr<Graph> graph_from_arrays(const float* X, size_t n, size_t dim, const int32_t* nbr0, size_t maxM0,
                                                const int32_t* level, const int64_t* upper_ptr, const uint32_t* upper,
                                                size_t upper_len, size_t maxM, uint32_t entry, int maxlevel) {
  auto g = std::make_unique<Graph>();
  g->alloc(n, dim, maxM0, maxM);
  for (size_t i = 0; i < n; i++) {
    uint32_t* ll = reinterpret_cast<uint32_t*>(g->blob + i * g->stride);
    uint32_t c = 0;
    for (size_t j = 0; j < maxM0; j++) { int32_t v = nbr0[i * maxM0 + j]; if (v < 0) break; ll[1 + c++] = (uint32_t)v; }
    for (size_t j = c; j < maxM0; j++) ll[1 + j] = 0;
    ll[0] = c;
    std::memcpy(g->blob + i * g->stride + g->off_data, X + i * dim, dim * sizeof(float));
    size_t lab = i; std::memcpy(g->blob + i * g->stride + g->off_label, &lab, sizeof(size_t));
    g->level[i] = level[i]; g->upper_ptr[i] = upper_ptr[i];
  }
  g->upper.assign(upper, upper + upper_len);
  g->entry = entry; g->maxlevel = maxlevel;
  return g;
}

}  // namespace cast

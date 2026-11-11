// Category store: the category lists c(v) of all items in compressed sparse row form (offsets int64[n + 1],
// category ids uint32) and the capacities kappa. The graph never stores categories. Paper: Sec. III-A.
#pragma once
#include <cstdint>
#include <vector>

#include "answer_queue.hpp"

namespace cast {

struct CategoryStore {
  size_t n = 0;                  // number of items
  uint32_t m = 0;                // number of categories
  std::vector<int64_t> off;      // c(v) = col[off[v] .. off[v + 1])
  std::vector<uint32_t> col;
  std::vector<int32_t> kappa;    // capacity of each category

  CategoryStore(const int64_t* ptr, size_t n_, const int32_t* idx, uint32_t m_, const int32_t* kap)
      : n(n_), m(m_), off(ptr, ptr + n_ + 1), kappa(kap, kap + m_) {
    int64_t base = off[0];
    for (auto& o : off) o -= base;
    col.resize(off[n]);
    for (int64_t t = 0; t < off[n]; t++) col[t] = (uint32_t)idx[base + t];
  }
  CategoryView view() const { return CategoryView{off.data(), col.data()}; }
};

}  // namespace cast

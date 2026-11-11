// Independent feasibility check of an answer (paper Sec. III-A, Definition 1): is there an assignment
// phi: S -> categories with phi(v) in c(v) and |phi^{-1}(gamma)| <= kappa(gamma)?
// Written separately from the answer queue: categories are relabelled locally, each category gamma is expanded
// into kappa(gamma) unit slots, and a maximum bipartite matching is grown by depth-first augmenting paths
// (Kuhn's algorithm).
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace cast {

enum VerifyCode : int8_t { kInfeasible = 0, kComplete = 1, kIncomplete = 2 };

class Verifier {
 public:
  // ids: k entries, negative = empty. Returns kComplete if the non-empty entries are distinct, feasible
  // and exactly k; kIncomplete if feasible with fewer than k; kInfeasible otherwise.
  template <class CategoryFn>
  int8_t check(const int64_t* ids, int k, const int32_t* kappa, CategoryFn categories) {
    items_.clear();
    for (int a = 0; a < k; a++) if (ids[a] >= 0) items_.push_back(ids[a]);
    std::vector<int64_t> sorted(items_);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return kInfeasible;
    // local category slots
    cats_.clear();
    for (int64_t v : items_) { auto r = categories(v); for (auto p = r.first; p != r.second; ++p) cats_.push_back(*p); }
    std::sort(cats_.begin(), cats_.end()); cats_.erase(std::unique(cats_.begin(), cats_.end()), cats_.end());
    slot_start_.assign(cats_.size() + 1, 0);
    for (size_t j = 0; j < cats_.size(); j++) {
      int cap = std::min<int>(kappa[cats_[j]], (int)items_.size());
      slot_start_[j + 1] = slot_start_[j] + cap;
    }
    owner_.assign(slot_start_.back(), -1);
    adj_.assign(items_.size(), {});
    for (size_t u = 0; u < items_.size(); u++) {
      auto r = categories(items_[u]);
      for (auto p = r.first; p != r.second; ++p) {
        size_t j = std::lower_bound(cats_.begin(), cats_.end(), *p) - cats_.begin();
        for (int t = slot_start_[j]; t < slot_start_[j + 1]; t++) adj_[u].push_back(t);
      }
    }
    for (size_t u = 0; u < items_.size(); u++) {
      seen_.assign(owner_.size(), 0);
      if (!augment((int)u)) return kInfeasible;
    }
    return (int)items_.size() == k ? kComplete : kIncomplete;
  }

 private:
  bool augment(int u) {
    for (int t : adj_[u]) {
      if (seen_[t]) continue;
      seen_[t] = 1;
      if (owner_[t] < 0 || augment(owner_[t])) { owner_[t] = u; return true; }
    }
    return false;
  }
  std::vector<int64_t> items_;
  std::vector<uint32_t> cats_;
  std::vector<int> slot_start_, owner_;
  std::vector<std::vector<int>> adj_;
  std::vector<char> seen_;
};

}  // namespace cast

// Answer queue: maintains the greedy constrained answer Greedy(A) of the items offered so far, exactly, for
// offers in any order. Paper: Sec. III-A (Definition 1, Lemma 1, Lemma 2).
//
// The queue holds at most k answer items in slots 0..k-1 together with one assignment phi (Definition 1):
// phi(slot) is a category of the item in the slot, and no category gamma is assigned more than kappa(gamma)
// items. Per category it keeps the load |phi^{-1}(gamma)| and the list of the slots assigned to it. Category
// arrays are dense (one entry per category) and are reset in O(1) per query by epoch stamps.
//
// An arriving item is decided by an alternating-path search between answer items and category slots: a step
// goes from a category gamma to an item assigned to gamma and on to another category of that item, to which the
// item could be reassigned. Two labels summarise these paths for the current answer:
//   open(gamma)  a full category from which an alternating path reaches a category with spare capacity, so
//                reassigning the items along the path frees one slot of gamma;
//   last(gamma)  for a full category that is not open, the last answer item, in the order (distance, id),
//                among the items on alternating paths from gamma.
// The labels are recomputed lazily, at most once per change of the answer: one breadth-first search for open
// and one pass over the strongly connected components of the remaining categories for last.
//
// After every offer the answer equals Greedy of all items offered so far, with ties broken by item id.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace cast {

// The order of the paper: increasing distance, ties broken by item id.
inline bool prec(float da, uint32_t ia, float db, uint32_t ib) { return da < db || (da == db && ia < ib); }

// Category access for an item id: [cbegin(e), cend(e)) is the category list c(e), in compressed sparse row form.
struct CategoryView {
  const int64_t* ptr = nullptr;
  const uint32_t* idx = nullptr;
  inline const uint32_t* cbegin(uint32_t e) const { return idx + ptr[e]; }
  inline const uint32_t* cend(uint32_t e) const { return idx + ptr[e + 1]; }
};

class AnswerQueue {
 public:
  // m categories, at most K answer items, capacities kappa[0..m-1], category lists cv.
  void init(uint32_t m, int K, const int32_t* kappa, const CategoryView& cv) {
    m_ = m; K_ = K; kappa_ = kappa; cv_ = cv;
    auto rs32 = [&](std::vector<int32_t>& v, int32_t x) { v.assign(m, x); };
    auto rsu = [&](std::vector<uint32_t>& v) { v.assign(m, 0u); };
    rsu(cstamp_); rs32(load_, 0); rs32(head_, -1);
    rsu(open_stamp_); rs32(nxt_slot_, -1); rs32(nxt_cat_, -1);
    rs32(last_slot_, -1);
    rsu(asg_stamp_); rsu(car_stamp_); rs32(car_head_, -1);
    rsu(tj_stamp_); rs32(tj_index_, 0); rs32(tj_low_, 0); rs32(tj_comp_, -1); rsu(onst_stamp_);
    rsu(bstamp_); rs32(bpar_cat_, -1); rs32(bpar_slot_, -1);
    int S = K + 2;
    item_.assign(S, 0); dist_.assign(S, 0.f); phi_.assign(S, -1); lnext_.assign(S, -1);
    hpos_.assign(S, -1); heap_.assign(S, 0);
    cepoch_ = 1; lab_epoch_ = 1; bepoch_ = 1;
    size_ = 0; hsize_ = 0; dirty_ = true;
  }
  // Offers arrive in increasing (distance, id) order: the queue is then the scan of Greedy over a list.
  void set_sorted_input(bool s) { sorted_input_ = s; }

  void reset() {
    if (++cepoch_ == 0) { std::fill(cstamp_.begin(), cstamp_.end(), 0u); cepoch_ = 1; }
    size_ = 0; hsize_ = 0; dirty_ = true;
  }
  int size() const { return size_; }
  int capacity() const { return K_; }
  bool full() const { return size_ == K_; }
  uint32_t member(int slot) const { return item_[slot]; }
  float member_dist(int slot) const { return dist_[slot]; }
  // Distance of the last answer item (the k-th distance of the answer once it is full).
  float max_dist() const { return hsize_ ? dist_[heap_[0]] : std::numeric_limits<float>::infinity(); }

  // Offer item e at distance d. Returns 1 if e entered the answer, 0 otherwise.
  int offer(uint32_t e, float d) {
    if (size_ == K_) {
      int top = heap_[0];
      if (!prec(d, e, dist_[top], item_[top])) return 0;   // not before the last of k answer items: O(1)
    }
    const uint32_t* cb = cv_.cbegin(e); const uint32_t* ce = cv_.cend(e);
    if (cb == ce) return 0;   // an item without categories belongs to no feasible answer (Definition 1)
    // a category of e with spare capacity: assign e to it directly
    for (const uint32_t* p = cb; p != ce; ++p) {
      uint32_t g = *p;
      if (ld(g) < kappa_[g]) { insert_at(e, d, (int32_t)g); return 1; }
    }
    ensure_labels();
    for (const uint32_t* p = cb; p != ce; ++p) {
      uint32_t g = *p;
      if (open_stamp_[g] == lab_epoch_) {   // open full category: reassign along the alternating path
        int tmp = K_;   // temporary slot
        shift_from(g);
        place(tmp, e, d, (int32_t)g);
        finish_insert(tmp);
        return 1;
      }
    }
    // the answer plus e is not feasible (Definition 1)
    if (sorted_input_) return 0;   // every answer item precedes e
    // e after every answer item: nothing it could replace precedes it
    if (hsize_ && prec(dist_[heap_[0]], item_[heap_[0]], d, e)) return 0;
    ensure_last();
    // e can replace exactly the answer items on alternating paths from c(e); the last of them is the
    // maximum of last(gamma) over the categories gamma of e
    int ys = -1;
    for (const uint32_t* p = cb; p != ce; ++p) {
      int s = last_slot_[*p];
      if (ys < 0 || prec(dist_[ys], item_[ys], dist_[s], item_[s])) ys = s;
    }
    if (prec(dist_[ys], item_[ys], d, e)) return 0;
    swap_in(e, d, ys);
    return 1;
  }

  // Copies the ids of the answer items to out, returns their number (for tests).
  int members(uint32_t* out) const { for (int i = 0; i < size_; i++) out[i] = item_[i]; return size_; }
  // The category phi assigns to the item in a slot (for tests).
  int member_category(int slot) const { return phi_[slot]; }

 private:
  // ----------------------------------------------------------------- per-category state
  inline void touch(uint32_t g) { if (cstamp_[g] != cepoch_) { cstamp_[g] = cepoch_; load_[g] = 0; head_[g] = -1; } }
  inline int32_t ld(uint32_t g) const { return cstamp_[g] == cepoch_ ? load_[g] : 0; }
  inline int32_t hd(uint32_t g) const { return cstamp_[g] == cepoch_ ? head_[g] : -1; }
  inline void link(int slot, int32_t g) { touch(g); phi_[slot] = g; lnext_[slot] = head_[g]; head_[g] = slot; load_[g]++; }
  inline void unlink(int slot) {
    int32_t g = phi_[slot];
    int32_t* pp = &head_[g];
    while (*pp != slot) pp = &lnext_[*pp];
    *pp = lnext_[slot];
    load_[g]--;
  }
  inline void move_to(int slot, int32_t g2) { unlink(slot); link(slot, g2); }
  inline void place(int slot, uint32_t e, float d, int32_t g) { item_[slot] = e; dist_[slot] = d; link(slot, g); }

  // ----------------------------------------------------------------- indexed max-heap of slots
  inline bool hless(int a, int b) const { return prec(dist_[a], item_[a], dist_[b], item_[b]); }
  void hswap(int i, int j) { std::swap(heap_[i], heap_[j]); hpos_[heap_[i]] = i; hpos_[heap_[j]] = j; }
  void hup(int i) { while (i > 0) { int p = (i - 1) >> 1; if (hless(heap_[p], heap_[i])) { hswap(i, p); i = p; } else break; } }
  void hdown(int i) {
    for (;;) {
      int l = 2 * i + 1, r = l + 1, c = i;
      if (l < hsize_ && hless(heap_[c], heap_[l])) c = l;
      if (r < hsize_ && hless(heap_[c], heap_[r])) c = r;
      if (c == i) break;
      hswap(i, c); i = c;
    }
  }
  void hpush(int slot) { heap_[hsize_] = slot; hpos_[slot] = hsize_; hsize_++; hup(hsize_ - 1); }
  void hremove(int slot) {
    int i = hpos_[slot]; hsize_--;
    if (i != hsize_) { heap_[i] = heap_[hsize_]; hpos_[heap_[i]] = i; hup(i); hdown(hpos_[heap_[i]]); }
    hpos_[slot] = -1;
  }

  // relocate the item in slot `from` to the free slot `to` (keeps the category lists consistent)
  void relocate(int from, int to) {
    int32_t g = phi_[from];
    item_[to] = item_[from]; dist_[to] = dist_[from]; phi_[to] = g; lnext_[to] = lnext_[from];
    int32_t* pp = &head_[g];
    while (*pp != from) pp = &lnext_[*pp];
    *pp = to;
  }

  // e was placed in the temporary slot tmp (the answer plus e is feasible); keep it, and if the answer then
  // holds more than k items, remove the last one.
  // spare_g >= 0: e was assigned to category spare_g, which had spare capacity, and no other item moved.
  // The labels depend only on the full categories and on the items assigned to them, so they stay valid
  // when spare_g still has spare capacity afterwards and the removed item (if any) sat on a category with
  // spare capacity.
  void finish_insert(int tmp, int32_t spare_g = -1) {
    bool keep = spare_g >= 0;
    if (size_ < K_) {
      int s = size_++;
      relocate(tmp, s); hpush(s);
    } else {
      int ys = heap_[0];
      int32_t h = phi_[ys];
      keep = keep && load_[h] < kappa_[h];   // load before the removal (includes e if h == spare_g)
      hremove(ys); unlink(ys);
      relocate(tmp, ys); hpush(ys);
    }
    keep = keep && load_[spare_g] < kappa_[spare_g];
    if (!keep) dirty_ = true;
  }
  void insert_at(uint32_t e, float d, int32_t g) {
    int tmp = K_;
    place(tmp, e, d, g);
    finish_insert(tmp, g);
  }

  // ----------------------------------------------------------------- reassignment along the open labels
  void shift_from(uint32_t g) {
    // g is full and open: the item nxt_slot_[g] moves to nxt_cat_[g], and so on until a category that had
    // spare capacity when the labels were computed takes the unit (the label path is a simple path of the
    // search tree)
    int32_t cur = (int32_t)g;
    for (;;) {
      int32_t y = nxt_slot_[cur], g2 = nxt_cat_[cur];
      bool spare = ld(g2) < kappa_[g2];
      move_to(y, g2);
      if (spare) break;
      cur = g2;
    }
  }

  // e replaces the answer item in slot ys along an alternating path from c(e) to phi(ys)
  void swap_in(uint32_t e, float d, int ys) {
    int32_t t = phi_[ys];
    int32_t g1 = bfs_path(e, t);   // fills bpar_* ; returns the category of e that starts the path
    hremove(ys); unlink(ys);
    // reassign the items along the path, from its end
    int32_t cur = t;
    while (cur != g1) {
      int32_t y = bpar_slot_[cur], pc = bpar_cat_[cur];
      move_to(y, cur);
      cur = pc;
    }
    place(ys, e, d, g1); hpush(ys);
    dirty_ = true;
  }

  // Breadth-first alternating-path search from c(e) until category `target`; returns the first category
  // of the path.
  int32_t bfs_path(uint32_t e, int32_t target) {
    if (++bepoch_ == 0) { std::fill(bstamp_.begin(), bstamp_.end(), 0u); bepoch_ = 1; }
    bq_.clear();
    for (const uint32_t* p = cv_.cbegin(e); p != cv_.cend(e); ++p) {
      uint32_t g = *p;
      if (bstamp_[g] == bepoch_) continue;
      bstamp_[g] = bepoch_; bpar_cat_[g] = -1; bpar_slot_[g] = -1; bq_.push_back((int32_t)g);
      if ((int32_t)g == target) return target;
    }
    for (size_t h = 0; h < bq_.size(); h++) {
      int32_t g = bq_[h];
      for (int32_t y = hd(g); y != -1; y = lnext_[y]) {
        for (const uint32_t* p = cv_.cbegin(item_[y]); p != cv_.cend(item_[y]); ++p) {
          uint32_t g2 = *p;
          if ((int32_t)g2 == g || bstamp_[g2] == bepoch_) continue;
          bstamp_[g2] = bepoch_; bpar_cat_[g2] = g; bpar_slot_[g2] = y;
          if ((int32_t)g2 == target) { return start_of(target); }
          bq_.push_back((int32_t)g2);
        }
      }
    }
    return -1;  // not reached when slot ys lies on an alternating path from c(e)
  }
  int32_t start_of(int32_t g) const { while (bpar_cat_[g] >= 0) g = bpar_cat_[g]; return g; }

  // ----------------------------------------------------------------- labels
  void ensure_labels() {
    if (!dirty_) return;
    dirty_ = false;
    if (++lab_epoch_ == 0) {
      std::fill(open_stamp_.begin(), open_stamp_.end(), 0u);
      std::fill(asg_stamp_.begin(), asg_stamp_.end(), 0u); std::fill(car_stamp_.begin(), car_stamp_.end(), 0u);
      std::fill(tj_stamp_.begin(), tj_stamp_.end(), 0u); std::fill(onst_stamp_.begin(), onst_stamp_.end(), 0u);
      std::fill(spare_at_label_.begin(), spare_at_label_.end(), 0u);
      lab_epoch_ = 1;
    }
    if (spare_at_label_.size() != m_) spare_at_label_.assign(m_, 0u);
    const uint32_t E = lab_epoch_;
    // assigned categories, and for every other category g' of an answer item y the entry (g', y): y could
    // be reassigned to g'
    acats_.clear(); car_slot_.clear(); car_next_.clear(); gq_.clear();
    for (int y = 0; y < size_; y++) {
      int32_t py = phi_[y];
      if (asg_stamp_[py] != E) { asg_stamp_[py] = E; acats_.push_back(py); }
      for (const uint32_t* p = cv_.cbegin(item_[y]); p != cv_.cend(item_[y]); ++p) {
        uint32_t g2 = *p;
        if ((int32_t)g2 == py) continue;
        if (car_stamp_[g2] != E) { car_stamp_[g2] = E; car_head_[g2] = -1; }
        car_slot_.push_back(y); car_next_.push_back(car_head_[g2]); car_head_[g2] = (int32_t)car_slot_.size() - 1;
        if (ld(g2) < kappa_[g2] && spare_at_label_[g2] != E) { spare_at_label_[g2] = E; gq_.push_back((int32_t)g2); }
      }
    }
    // breadth-first search for open, backwards from the categories with spare capacity
    for (size_t h = 0; h < gq_.size(); h++) {
      int32_t g = gq_[h];
      if (car_stamp_[g] != E) continue;
      for (int32_t c = car_head_[g]; c != -1; c = car_next_[c]) {
        int32_t y = car_slot_[c]; int32_t g0 = phi_[y];
        if (open_stamp_[g0] == E || ld(g0) < kappa_[g0]) continue;
        open_stamp_[g0] = E; nxt_slot_[g0] = y; nxt_cat_[g0] = g;
        gq_.push_back(g0);
      }
    }
    last_dirty_ = true;   // the last labels are computed on demand (ensure_last), from these open labels
  }
  void ensure_last() {
    if (!last_dirty_) return;
    last_dirty_ = false;
    compute_last();
  }

  // Strongly connected components (Tarjan) of the full categories that are not open; last(gamma) is the
  // last answer item reachable from gamma.
  struct Frame { int32_t g; int32_t y; const uint32_t* p; };
  inline bool closed(int32_t g) const { return ld(g) >= kappa_[g] && open_stamp_[g] != lab_epoch_; }
  void next_arc(Frame& f) {
    // advance f to the next successor category; f.y == -1 when exhausted
    for (;;) {
      if (f.y < 0) return;
      const uint32_t* ce = cv_.cend(item_[f.y]);
      while (f.p != ce) { if ((int32_t)*f.p != f.g) return; ++f.p; }
      f.y = lnext_[f.y];
      if (f.y >= 0) f.p = cv_.cbegin(item_[f.y]);
    }
  }
  void compute_last() {
    const uint32_t E = lab_epoch_;
    int32_t idx = 0, comp = 0;
    stk_.clear();
    for (int32_t root : acats_) {
      if (!closed(root) || tj_stamp_[root] == E) continue;
      frames_.clear();
      auto enter = [&](int32_t g) {
        tj_stamp_[g] = E; tj_index_[g] = tj_low_[g] = idx++; tj_comp_[g] = -1;
        stk_.push_back(g); onst_stamp_[g] = E;
        Frame f{g, head_[g], nullptr};
        if (f.y >= 0) f.p = cv_.cbegin(item_[f.y]);
        next_arc(f);
        frames_.push_back(f);
      };
      enter(root);
      while (!frames_.empty()) {
        Frame& f = frames_.back();
        if (f.y >= 0) {
          int32_t w = (int32_t)*f.p; ++f.p; next_arc(f);
          if (tj_stamp_[w] != E) { enter(w); continue; }
          if (onst_stamp_[w] == E) tj_low_[f.g] = std::min(tj_low_[f.g], tj_index_[w]);
          continue;
        }
        int32_t g = f.g;
        frames_.pop_back();
        if (!frames_.empty()) tj_low_[frames_.back().g] = std::min(tj_low_[frames_.back().g], tj_low_[g]);
        if (tj_low_[g] == tj_index_[g]) {
          // pop the component rooted at g
          size_t start = stk_.size();
          do { start--; } while (stk_[start] != g);
          for (size_t i = start; i < stk_.size(); i++) { tj_comp_[stk_[i]] = comp; onst_stamp_[stk_[i]] = 0; }
          int best = -1;
          for (size_t i = start; i < stk_.size(); i++) {
            int32_t c = stk_[i];
            for (int32_t y = head_[c]; y != -1; y = lnext_[y]) {
              if (best < 0 || hless(best, y)) best = y;
              for (const uint32_t* p = cv_.cbegin(item_[y]); p != cv_.cend(item_[y]); ++p) {
                int32_t w = (int32_t)*p;
                if (w == c || tj_comp_[w] == comp) continue;
                int s = last_slot_[w];   // a finished component (successors finish first)
                if (best < 0 || hless(best, s)) best = s;
              }
            }
          }
          for (size_t i = start; i < stk_.size(); i++) last_slot_[stk_[i]] = best;
          stk_.resize(start);
          comp++;
        }
      }
    }
  }

  uint32_t m_ = 0; int K_ = 0; const int32_t* kappa_ = nullptr; CategoryView cv_;
  bool sorted_input_ = false, dirty_ = true, last_dirty_ = true;
  uint32_t cepoch_ = 1, lab_epoch_ = 1, bepoch_ = 1;
  int size_ = 0, hsize_ = 0;
  std::vector<uint32_t> cstamp_; std::vector<int32_t> load_, head_;
  std::vector<uint32_t> open_stamp_; std::vector<int32_t> nxt_slot_, nxt_cat_;
  std::vector<int32_t> last_slot_;
  std::vector<uint32_t> asg_stamp_, car_stamp_, spare_at_label_; std::vector<int32_t> car_head_, car_slot_, car_next_;
  std::vector<int32_t> acats_, gq_;
  std::vector<uint32_t> tj_stamp_, onst_stamp_; std::vector<int32_t> tj_index_, tj_low_, tj_comp_, stk_;
  std::vector<Frame> frames_;
  std::vector<uint32_t> bstamp_; std::vector<int32_t> bpar_cat_, bpar_slot_, bq_;
  std::vector<uint32_t> item_; std::vector<float> dist_; std::vector<int32_t> phi_, lnext_, hpos_, heap_;
};

}  // namespace cast

// Python bindings (pybind11) of the search engine, module cast_engine: graph loading, category store, the
// trajectory log, the timed search with a stopping rule, the list search, the independent feasibility check
// and the forest evaluation. Paper: Sec. III-A, III-B and Sec. VI-A (Compared methods, Implementation details).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "include/category_store.hpp"
#include "include/graph.hpp"
#include "include/search.hpp"
#include "include/verify.hpp"

namespace py = pybind11;
using namespace cast;

template <class T> using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

static int method_id(const std::string& s) {
  if (s == "plain") return PLAIN;
  if (s == "list") return LIST;
  if (s == "answer") return ANSWER;
  throw std::runtime_error("unknown method " + s);
}

static Params parse(const py::dict& d) {
  Params P;
  for (auto kv : d) {
    std::string k = py::str(kv.first);
    auto v = kv.second;
    if (k == "method") P.method = method_id(v.cast<std::string>());
    else if (k == "k") P.k = v.cast<int>();
    else if (k == "L") P.L = v.cast<int>();
    else throw std::runtime_error("unknown parameter " + k);
  }
  return P;
}

PYBIND11_MODULE(cast_engine, m) {
  m.doc() = "Search engine: best-first graph traversal that maintains the greedy constrained answer, with stopping rules";

  py::class_<Graph, std::unique_ptr<Graph>>(m, "Graph")
      .def_readonly("n", &Graph::n)
      .def_readonly("dim", &Graph::dim)
      .def_readonly("maxM0", &Graph::maxM0)
      .def_readonly("maxM", &Graph::maxM)
      .def_readonly("maxlevel", &Graph::maxlevel)
      .def_readonly("entry", &Graph::entry)
      .def("mean_degree0", &Graph::mean_degree0);

  m.def("load_hnswlib", [](const std::string& path, size_t dim) { return load_hnswlib(path, dim); },
        py::arg("path"), py::arg("dim"));
  m.def("graph_from_arrays",
        [](arr<float> X, arr<int32_t> nbr0, arr<int32_t> level, arr<int64_t> upper_ptr, arr<uint32_t> upper, size_t maxM,
           uint32_t entry, int maxlevel) {
          return graph_from_arrays(X.data(), X.shape(0), X.shape(1), nbr0.data(), nbr0.shape(1), level.data(),
                                   upper_ptr.data(), upper.data(), upper.size(), maxM, entry, maxlevel);
        });

  py::class_<CategoryStore>(m, "Categories")
      .def(py::init([](arr<int64_t> ptr, arr<int32_t> idx, uint32_t n_categories, arr<int32_t> kappa) {
             if ((size_t)kappa.size() != n_categories) throw std::runtime_error("kappa must have n_categories entries");
             return new CategoryStore(ptr.data(), ptr.size() - 1, idx.data(), n_categories, kappa.data());
           }),
           py::arg("ptr"), py::arg("idx"), py::arg("n_categories"), py::arg("kappa"))
      .def_readonly("n", &CategoryStore::n)
      .def_readonly("n_categories", &CategoryStore::m);

  m.def("search",   // list search (method 'list') and plain top-k (method 'plain')
        [](const Graph& g, CategoryStore& cs, arr<float> Q, py::dict params, py::object Ls, int threads) {
          Params P = parse(params);
          if (P.method == ANSWER) throw std::runtime_error("search runs methods 'list' and 'plain'; method 'answer' runs through log_search and stop_search");
          size_t nq = Q.shape(0), d = Q.shape(1);
          if (d != g.dim) throw std::runtime_error("query dimension mismatch");
          std::vector<int32_t> Lq(nq, P.L);
          if (!Ls.is_none()) {
            arr<int32_t> a = Ls.cast<arr<int32_t>>();
            if ((size_t)a.size() != nq) throw std::runtime_error("Ls must have one entry per query");
            std::memcpy(Lq.data(), a.data(), nq * sizeof(int32_t));
          }
          const int k = P.k;
          arr<int64_t> ids({nq, (size_t)k});
          arr<int64_t> nvisit(nq);
          int64_t* pid = ids.mutable_data();
          int64_t* pnv = nvisit.mutable_data();
          const float* q = Q.data();
          threads = std::max(1, threads);
          {
            py::gil_scoped_release nogil;
            auto work = [&](int t) {
              Searcher S(g, cs, P);
              for (size_t i = t; i < nq; i += threads) {
                QueryResult R = S.run(q + i * d, Lq[i], pid + i * k);
                pnv[i] = R.nvisit;
              }
            };
            if (threads == 1) work(0);
            else {
              std::vector<std::thread> pool_t;
              for (int t = 0; t < threads; t++) pool_t.emplace_back(work, t);
              for (auto& th : pool_t) th.join();
            }
          }
          py::dict out;
          out["ids"] = ids; out["visits"] = nvisit;
          return out;
        },
        py::arg("graph"), py::arg("categories"), py::arg("Q"), py::arg("params"), py::arg("Ls") = py::none(),
        py::arg("threads") = 1);

  m.def("log_search",   // trajectory log, never timed
        [](const Graph& g, CategoryStore& cs, arr<float> Q, arr<int64_t> S, py::dict params, int Lmax, arr<int32_t> grid, int threads) {
          Params P = parse(params);
          if (P.method != ANSWER) throw std::runtime_error("log_search needs method 'answer'");
          size_t nq = Q.shape(0), d = Q.shape(1), G = grid.size(); const int k = P.k;
          if (d != g.dim) throw std::runtime_error("query dimension mismatch");
          std::vector<int> gr(grid.data(), grid.data() + G);
          std::vector<Searcher::LogOut> L(nq);
          const float* q = Q.data(); const int64_t* Sp = S.data();
          threads = std::max(1, threads);
          {
            py::gil_scoped_release nogil;
            std::vector<std::thread> pool_t;
            for (int t = 0; t < threads; t++) pool_t.emplace_back([&, t]() {
              Searcher Sr(g, cs, P);
              for (size_t i = t; i < nq; i += threads) Sr.log_run(q + i * d, Lmax, Sp + i * k, gr, L[i]);
            });
            for (auto& th : pool_t) th.join();
          }
          size_t tot = 0; for (auto& o : L) tot += o.nv.size();
          arr<int64_t> ptr(nq + 1); arr<int32_t> nv(tot), asize(tot), chg(tot), fin(nq), nds(nq); arr<float> dx(tot), dbk(tot), dpk(tot);
          arr<int32_t> rs({nq, (size_t)k}), tl({nq, G});
          int64_t* pp = ptr.mutable_data(); size_t c = 0; pp[0] = 0;
          for (size_t i = 0; i < nq; i++) {
            auto& o = L[i]; size_t n = o.nv.size();
            std::memcpy(nv.mutable_data() + c, o.nv.data(), n * 4); std::memcpy(asize.mutable_data() + c, o.asize.data(), n * 4);
            std::memcpy(chg.mutable_data() + c, o.chg.data(), n * 4); std::memcpy(dx.mutable_data() + c, o.dx.data(), n * 4);
            std::memcpy(dbk.mutable_data() + c, o.dbk.data(), n * 4); std::memcpy(dpk.mutable_data() + c, o.dpk.data(), n * 4);
            std::memcpy(rs.mutable_data() + i * k, o.rank_s.data(), k * 4); std::memcpy(tl.mutable_data() + i * G, o.TL.data(), G * 4);
            fin.mutable_data()[i] = o.final_nv; nds.mutable_data()[i] = o.ndesc; c += n; pp[i + 1] = (int64_t)c;
          }
          py::dict out;
          out["ptr"] = ptr; out["visits"] = nv; out["answer_size"] = asize; out["chg"] = chg; out["dx"] = dx; out["dbk"] = dbk; out["dpk"] = dpk;
          out["visit_of_member"] = rs; out["list_stop"] = tl; out["final_visits"] = fin; out["ndesc"] = nds;
          return out;
        },
        py::arg("graph"), py::arg("categories"), py::arg("Q"), py::arg("S"), py::arg("params"), py::arg("Lmax"), py::arg("grid"), py::arg("threads") = 1);

  // Flattened gradient-boosted forests and the timed search with a stopping rule.
  auto forest_of = [](py::object o, Forest& F) {
    if (o.is_none()) return false;
    py::dict d = o.cast<py::dict>();
    auto vi = [&](const char* k) { arr<int32_t> a = d[k].cast<arr<int32_t>>(); return std::vector<int32_t>(a.data(), a.data() + a.size()); };
    auto vd = [&](const char* k) { arr<double> a = d[k].cast<arr<double>>(); return std::vector<double>(a.data(), a.data() + a.size()); };
    F.root = vi("root"); F.left = vi("left"); F.right = vi("right"); F.thr = vd("thr"); F.val = vd("val");
    { arr<int16_t> a = d["feat"].cast<arr<int16_t>>(); F.feat.assign(a.data(), a.data() + a.size()); }
    { arr<uint8_t> a = d["leaf"].cast<arr<uint8_t>>(); F.leaf.assign(a.data(), a.data() + a.size()); }
    { arr<uint8_t> a = d["miss_left"].cast<arr<uint8_t>>(); F.miss_left.assign(a.data(), a.data() + a.size()); }
    F.base = d["base"].cast<double>(); return true;
  };
  m.def("forest_raw",   // raw outputs of a flattened forest on given rows
        [forest_of](py::dict f, arr<double> X) {
          Forest F; forest_of(f, F); size_t n = X.shape(0), m_ = X.shape(1); arr<double> out(n); const double* x = X.data();
          for (size_t i = 0; i < n; i++) out.mutable_data()[i] = F.raw(x + i * m_);
          return out;
        });
  m.def("npy_logf_vec", [](arr<float> x) {   // the scalar form of NumPy's float32 logarithm used by the online features
          arr<float> out(x.size()); for (ssize_t i = 0; i < x.size(); i++) out.mutable_data()[i] = npy_logf(x.data()[i]); return out; });
  m.def("stop_search",
        [forest_of](const Graph& g, CategoryStore& cs, arr<float> Q, py::dict params, int Lmax, py::dict rule, py::object clf, py::object reg,
                    int threads, bool dump, bool instr, py::object S) {
          Params P = parse(params);
          if (P.method != ANSWER) throw std::runtime_error("stop_search needs method 'answer'");
          size_t nq = Q.shape(0), d = Q.shape(1); if (d != g.dim) throw std::runtime_error("query dimension mismatch");
          Forest Fc, Fr; StopRule Rl;
          Rl.type = rule["type"].cast<int>();
          { arr<int32_t> a = rule["grid"].cast<arr<int32_t>>(); Rl.grid.assign(a.data(), a.data() + a.size()); }
          if (rule.contains("edges")) { arr<double> a = rule["edges"].cast<arr<double>>(); Rl.edges.assign(a.data(), a.data() + a.size()); }
          if (rule.contains("beta")) { arr<int32_t> a = rule["beta"].cast<arr<int32_t>>(); Rl.beta.assign(a.data(), a.data() + a.size()); }
          if (rule.contains("lam")) Rl.lam = rule["lam"].cast<double>();
          if (rule.contains("jcap")) Rl.jcap = rule["jcap"].cast<int>();
          if (rule.contains("jfix")) Rl.jfix = rule["jfix"].cast<int>();
          if (rule.contains("j0")) Rl.j0 = rule["j0"].cast<int>();
          if (rule.contains("mult")) Rl.mult = rule["mult"].cast<double>();
          if (rule.contains("log_tab")) { arr<double> a = rule["log_tab"].cast<arr<double>>(); Rl.log_tab.assign(a.data(), a.data() + a.size()); }
          if (rule.contains("log1p_tab")) { arr<double> a = rule["log1p_tab"].cast<arr<double>>(); Rl.log1p_tab.assign(a.data(), a.data() + a.size()); }
          if (rule.contains("Rt")) Rl.Rt = rule["Rt"].cast<double>();
          if (rule.contains("ipi")) Rl.ipi = std::max(1, rule["ipi"].cast<int>());
          if (rule.contains("mpi")) Rl.mpi = std::max(1, rule["mpi"].cast<int>());
          if (rule.contains("li")) Rl.li = std::max(1, rule["li"].cast<int>());
          if (rule.contains("qstats")) Rl.qstats = rule["qstats"].cast<int>();
          if (rule.contains("pdelta")) Rl.pdelta = rule["pdelta"].cast<int>();
          if (rule.contains("hvis")) Rl.hvis = rule["hvis"].cast<int>();
          if (rule.contains("gamma")) Rl.gamma = rule["gamma"].cast<double>();
          if (Rl.type == 9 && Rl.pdelta <= 0) throw std::runtime_error("Patience needs pdelta > 0");
          Forest Fq;                                                     // DARTH+: the lower-quantile regressor
          if (rule.contains("qforest") && forest_of(rule["qforest"], Fq)) Rl.qreg = &Fq;
          if (forest_of(clf, Fc)) Rl.clf = &Fc;
          if (forest_of(reg, Fr)) Rl.reg = &Fr;
          if ((Rl.type == 1 || Rl.type == 2 || dump) && !Rl.clf) throw std::runtime_error("rule needs the classifier");
          if (Rl.type == 1 && Rl.beta.size() != (Rl.grid.size() - 1) * 2) throw std::runtime_error("beta must be (J-1) x 2");
          if ((Rl.type == 4 || Rl.type == 6 || Rl.type == 8) && !Rl.reg) throw std::runtime_error("rule needs the regressor");
          if (Rl.type == 8 && !Rl.qreg) throw std::runtime_error("DARTH+ needs the quantile regressor (rule['qforest'])");
          const int64_t* Sp = nullptr; arr<int64_t> Sarr;
          if (Rl.type == 7) { if (S.is_none()) throw std::runtime_error("the DARTH log needs S"); Sarr = S.cast<arr<int64_t>>(); Sp = Sarr.data(); }
          const int k = P.k; const size_t J = Rl.grid.size(); const size_t RW = Rl.qstats ? 20 : 12;   // DARTH log row width
          arr<int64_t> ids({nq, (size_t)k}); arr<int64_t> ns(nq), hook(nq); arr<int32_t> nv(nq), sj(nq), calls(nq), qcalls(nq);
          std::vector<std::vector<double>> drows(Rl.type == 7 ? nq : 0);
          arr<double> feats({dump ? nq : 1, dump ? J : 1, (size_t)7}), score({dump ? nq : 1, dump ? J : 1}); arr<uint8_t> full({dump ? nq : 1, dump ? J : 1});
          int64_t* pid = ids.mutable_data(); const float* q = Q.data(); threads = std::max(1, threads);
          int64_t *pns = ns.mutable_data(), *phk = hook.mutable_data(); int32_t *pnv = nv.mutable_data(), *psj = sj.mutable_data(), *pcl = calls.mutable_data(), *pqc = qcalls.mutable_data();
          double *pft = feats.mutable_data(), *psc = score.mutable_data(); uint8_t* pfl = full.mutable_data();
          {
            py::gil_scoped_release nogil;
            auto work = [&](int t) {
              Searcher Sr(g, cs, P); StopOut so;
              for (size_t i = t; i < nq; i += threads) {
                so.hook_ns = 0; so.calls = 0; so.qcalls = 0; so.drows.clear(); so.Sq = Sp ? Sp + i * k : nullptr;
                QueryResult R = Sr.stop_run(q + i * d, Lmax, Rl, pid + i * k, so, dump, instr);
                pns[i] = R.ns; phk[i] = so.hook_ns; pnv[i] = so.nv; psj[i] = so.stop_j; pcl[i] = so.calls; pqc[i] = so.qcalls;
                if (Rl.type == 7) drows[i] = so.drows;
                if (dump) {
                  std::memcpy(pft + i * J * 7, so.feats.data(), J * 7 * sizeof(double));
                  std::memcpy(psc + i * J, so.score.data(), J * sizeof(double));
                  std::memcpy(pfl + i * J, so.full.data(), J);
                }
              }
            };
            if (threads == 1) work(0);
            else { std::vector<std::thread> pool_t; for (int t = 0; t < threads; t++) pool_t.emplace_back(work, t); for (auto& th : pool_t) th.join(); }
          }
          py::dict out; out["ids"] = ids; out["ns"] = ns; out["hook_ns"] = hook; out["visits"] = nv; out["stop_j"] = sj; out["calls"] = calls;
          if (Rl.type == 8) out["qcalls"] = qcalls;
          if (Rl.type == 7) {
            size_t tot = 0; for (auto& v : drows) tot += v.size() / RW;
            arr<double> rows({tot, RW}); arr<int64_t> ptr(nq + 1); double* pr = rows.mutable_data(); int64_t* pp = ptr.mutable_data(); size_t c = 0; pp[0] = 0;
            for (size_t i = 0; i < nq; i++) { std::memcpy(pr + c * RW, drows[i].data(), drows[i].size() * sizeof(double)); c += drows[i].size() / RW; pp[i + 1] = (int64_t)c; }
            out["drows"] = rows; out["dptr"] = ptr;
          }
          if (dump) { out["features"] = feats; out["score"] = score; out["full"] = full; }
          return out;
        },
        py::arg("graph"), py::arg("categories"), py::arg("Q"), py::arg("params"), py::arg("Lmax"), py::arg("rule"), py::arg("clf") = py::none(),
        py::arg("reg") = py::none(), py::arg("threads") = 1, py::arg("dump") = false, py::arg("instr") = false, py::arg("S") = py::none());

  m.def("verify",
        [](arr<int64_t> ids, const CategoryStore& cs) {
          size_t nq = ids.shape(0), k = ids.shape(1);
          arr<int8_t> res(nq);
          int8_t* r = res.mutable_data();
          const int64_t* p = ids.data();
          {
            py::gil_scoped_release nogil;
            Verifier V;
            auto categories = [&](int64_t v) {
              return std::make_pair(cs.col.data() + cs.off[v], cs.col.data() + cs.off[v + 1]);
            };
            for (size_t i = 0; i < nq; i++) r[i] = V.check(p + i * k, (int)k, cs.kappa.data(), categories);
          }
          return res;
        },
        "Independent check: 1 = feasible with k items, 2 = feasible with fewer items, 0 = infeasible or duplicate");

  m.def("rank_capped",
        [](const CategoryStore& cs, int64_t cap) {
          AnswerQueue G;
          G.init(cs.m, (int)cap, cs.kappa.data(), cs.view());
          G.set_sorted_input(true);
          G.reset();
          for (size_t v = 0; v < cs.n && G.size() < cap; v++) G.offer((uint32_t)v, 0.f);
          return G.size();
        },
        "Size of the largest feasible set of items (the rank of the constraint), truncated at cap");

  m.def("build_flags", []() {
    std::string s;
#ifdef __VERSION__
    s += std::string("compiler ") + __VERSION__;
#endif
#ifdef __AVX512F__
    s += "; AVX512F";
#endif
#ifdef __AVX2__
    s += "; AVX2";
#endif
#ifdef __FAST_MATH__
    s += "; FAST_MATH";
#endif
    return s;
  });
}

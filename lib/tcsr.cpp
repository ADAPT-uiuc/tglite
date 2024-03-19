#include "tglite/tcsr.h"

namespace tglite {

struct ETuple {
  IdI32 nbr;
  IdI32 eid;
  TsF32 ets;
  static bool cmp_ts(const ETuple &a, const ETuple &b) {
    return a.ets < b.ets;
  }
};

TCSR create_tcsr(py::array_t<IdI32> &edges, py::array_t<TsF32> &times, size_t num_nodes) {
  auto *edges_ptr = static_cast<IdI32 *>(edges.request().ptr);
  auto *times_ptr = static_cast<TsF32 *>(times.request().ptr);

  std::vector<std::vector<ETuple>> adj_list(num_nodes);
  for (IdI32 eid = 0; eid < edges.shape(0); eid++) {
    IdI32 src = edges_ptr[eid * 2];
    IdI32 dst = edges_ptr[eid * 2 + 1];
    TsF32 ets = times_ptr[eid];
    adj_list[src].push_back({dst, eid, ets});
    adj_list[dst].push_back({src, eid, ets});
  }

  TCSR tcsr;
  for (auto &adj : adj_list) {
    std::sort(adj.begin(), adj.end(), ETuple::cmp_ts);
    tcsr.ind.push_back(tcsr.ind.back() + adj.size());
    for (auto &tuple : adj) {
      tcsr.nbr.push_back(tuple.nbr);
      tcsr.eid.push_back(tuple.eid);
      tcsr.ets.push_back(tuple.ets);
    }
    adj.clear();
    adj.shrink_to_fit();
  }

  return tcsr;
}

} // namespace tglite

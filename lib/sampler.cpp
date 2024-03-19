#include "tglite/sampler.h"
#include <omp.h>

namespace tglite {

TemporalSampler::
TemporalSampler(int num_threads, int num_nbrs, bool recent)
    : _num_threads(num_threads), _num_nbrs(num_nbrs), _recent(recent) { }

TemporalBlock TemporalSampler::
sample(TCSR &tcsr, py::array_t<IdI32> &nodes, py::array_t<TsF32> &times) {
  omp_set_num_threads(_num_threads);

  const IdI32 *nodes_ptr = static_cast<const IdI32 *>(nodes.request().ptr);
  const TsF32 *times_ptr = static_cast<const TsF32 *>(times.request().ptr);
  size_t size = nodes.size();

  TemporalBlock block;
  sample_layer(tcsr, block, nodes_ptr, times_ptr, size);

  return block;
}

void TemporalSampler::
sample_layer(TCSR &tcsr, TemporalBlock &block,
             const IdI32 *nodes_ptr, const TsF32 *times_ptr, size_t size) {
  std::vector<IdI32> *eid[_num_threads];
  std::vector<TsF32> *ets[_num_threads];
  std::vector<IdI32> *srcnodes[_num_threads];
  std::vector<IdI32> *dstindex[_num_threads];
  std::vector<IdI32> out_nodes(_num_threads, 0);

  int nodes_per_thread = int(ceil(static_cast<float>(size) / _num_threads));
  int reserve_capacity = nodes_per_thread * _num_nbrs;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;

    eid[tid] = new std::vector<IdI32>;
    ets[tid] = new std::vector<TsF32>;
    srcnodes[tid] = new std::vector<IdI32>;
    dstindex[tid] = new std::vector<IdI32>;

    eid[tid]->reserve(reserve_capacity);
    ets[tid]->reserve(reserve_capacity);
    srcnodes[tid]->reserve(reserve_capacity);
    dstindex[tid]->reserve(reserve_capacity);

    #pragma omp for schedule(static, nodes_per_thread)
    for (size_t j = 0; j < size; j++) {
      IdI32 nid = nodes_ptr[j];
      TsF32 nts = times_ptr[j];

      IdI32 s_search = tcsr.ind[nid];
      auto e_it = std::lower_bound(tcsr.ets.begin() + s_search,
                                   tcsr.ets.begin() + tcsr.ind[nid + 1], nts);
      IdI32 e_search = std::max(int(e_it - tcsr.ets.begin()) - 1, s_search);

      if (_recent || (e_search - s_search + 1 < _num_nbrs)) {
        for (IdI32 k = e_search; k >= std::max(s_search, e_search - _num_nbrs + 1); k--) {
          if (tcsr.ets[k] < nts - 1e-7f) {
            add_neighbor(tcsr, eid[tid], ets[tid], srcnodes[tid], dstindex[tid], k, out_nodes[tid]);
          }
        }
      } else {
        for (int k = 0; k < _num_nbrs; k++) {
          IdI32 picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
          if (tcsr.ets[picked] < nts - 1e-7f) {
            add_neighbor(tcsr, eid[tid], ets[tid], srcnodes[tid], dstindex[tid], picked, out_nodes[tid]);
          }
        }
      }

      out_nodes[tid] += 1;
    }
  }

  combine_coo(block, eid, ets, srcnodes, dstindex, out_nodes);
}

inline void TemporalSampler::
add_neighbor(TCSR &tcsr,
    std::vector<IdI32> *eid, std::vector<TsF32> *ets,
    std::vector<IdI32> *srcnodes, std::vector<IdI32> *dstindex,
    IdI32 &k, IdI32 &dst_idx) {
  eid->push_back(tcsr.eid[k]);
  ets->push_back(tcsr.ets[k]);
  srcnodes->push_back(tcsr.nbr[k]);
  dstindex->push_back(dst_idx);
}

inline void TemporalSampler::
combine_coo(TemporalBlock &block,
    std::vector<IdI32> **eid,
    std::vector<TsF32> **ets,
    std::vector<IdI32> **srcnodes,
    std::vector<IdI32> **dstindex,
    std::vector<IdI32> &out_nodes) {

  std::vector<IdI32> scan_nodes;
  std::vector<IdI32> scan_edges;
  scan_nodes.push_back(0);
  scan_edges.push_back(0);
  for (int tid = 0; tid < _num_threads; tid++) {
    scan_nodes.push_back(scan_nodes.back() + out_nodes[tid]);
    scan_edges.push_back(scan_edges.back() + eid[tid]->size());
  }

  IdI32 num_edges = scan_edges.back();
  block.dstindex.resize(num_edges);
  block.srcnodes.resize(num_edges);
  block.eid.resize(num_edges);
  block.ets.resize(num_edges);
  block.num_edges = num_edges;

  #pragma omp parallel for schedule(static, 1)
  for (int tid = 0; tid < _num_threads; tid++) {
    std::transform(dstindex[tid]->begin(), dstindex[tid]->end(),
        dstindex[tid]->begin(), [&](auto &v) { return v + scan_nodes[tid]; });
    std::copy(eid[tid]->begin(), eid[tid]->end(), block.eid.begin() + scan_edges[tid]);
    std::copy(ets[tid]->begin(), ets[tid]->end(), block.ets.begin() + scan_edges[tid]);
    std::copy(srcnodes[tid]->begin(), srcnodes[tid]->end(), block.srcnodes.begin() + scan_edges[tid]);
    std::copy(dstindex[tid]->begin(), dstindex[tid]->end(), block.dstindex.begin() + scan_edges[tid]);
    delete eid[tid];
    delete ets[tid];
    delete srcnodes[tid];
    delete dstindex[tid];
  }
}

} // namespace tglite

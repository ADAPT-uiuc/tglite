#pragma once

#include "tglite/core.h"
#include "tglite/tcsr.h"

namespace tglite {

class TemporalBlock {
public:
  size_t num_edges = 0;
  std::vector<IdI32> dstindex;
  std::vector<IdI32> srcnodes;
  std::vector<IdI32> eid;
  std::vector<TsF32> ets;

  TemporalBlock() {}

  py::array_t<IdI32> dstindex_copy() const { return to_pyarray_copy(dstindex); }
  py::array_t<IdI32> srcnodes_copy() const { return to_pyarray_copy(srcnodes); }
  py::array_t<IdI32> eid_copy() const { return to_pyarray_copy(eid); }
  py::array_t<TsF32> ets_copy() const { return to_pyarray_copy(ets); }

  // py::array_t<IdI32> dstindex_owned() {
  //   auto *ptr = dstindex;
  //   dstindex = nullptr;
  //   return to_pyarray_owned(ptr);
  // }

  // py::array_t<IdI32> srcnodes_owned() {
  //   auto *ptr = srcnodes;
  //   srcnodes = nullptr;
  //   return to_pyarray_owned(ptr);
  // }

  // py::array_t<IdI32> eid_owned() {
  //   auto *ptr = eid;
  //   eid = nullptr;
  //   return to_pyarray_owned(ptr);
  // }

  // py::array_t<TsF32> ets_owned() {
  //   auto *ptr = ets;
  //   ets = nullptr;
  //   return to_pyarray_owned(ptr);
  // }
};

class TemporalSampler {
public:
  TemporalSampler(int num_threads, int num_nbrs, bool recent);

  TemporalBlock sample(TCSR &tcsr,
      py::array_t<IdI32> &nodes,
      py::array_t<TsF32> &times);

private:
  int _num_threads;
  int _num_nbrs;
  bool _recent;

  void sample_layer(TCSR &tcsr, TemporalBlock &block,
      const IdI32 *nodes_ptr, const TsF32 *times_ptr, size_t size);

  void add_neighbor(TCSR &tcsr,
      std::vector<IdI32> *eid,
      std::vector<TsF32> *ets,
      std::vector<IdI32> *srcnodes,
      std::vector<IdI32> *dstindex,
      IdI32 &k, IdI32 &dst_idx);

  void combine_coo(
      TemporalBlock &block,
      std::vector<IdI32> **eid,
      std::vector<TsF32> **ets,
      std::vector<IdI32> **srcnodes,
      std::vector<IdI32> **dstindex,
      std::vector<IdI32> &out_nodes);
};

} // namespace tglite

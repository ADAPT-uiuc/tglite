#pragma once

#include "tglite/core.h"

namespace tglite {

py::tuple dedup_targets(py::array_t<IdI32> &nodes, py::array_t<TsF32> &times);

// py::tuple dedup_indices(torch::Tensor &nodes, torch::Tensor &times);

// bool dedup_targets(
//   const IdI32 *nodes_ptr,
//   const TsF32 *times_ptr,
//   size_t len,
//   const IdI32 *pre_nodes,
//   const TsF32 *pre_times,
//   size_t pre_len,
//   std::vector<IdI32> &uniq_nodes,
//   std::vector<TsF32> &uniq_times,
//   std::vector<IdI32> &inv_idx
// );

} // namespace tglite

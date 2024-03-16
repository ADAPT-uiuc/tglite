#include "tglite/dedup.h"
#include "tglite/utils.h"

namespace tglite {

py::tuple dedup_targets(py::array_t<IdI32> &nodes, py::array_t<TsF32> &times) {
  ssize_t size = nodes.size();
  auto inv_idx = py::array_t<IdI32>(size);

  auto *node_ptr = static_cast<IdI32 *>(nodes.request().ptr);
  auto *time_ptr = static_cast<TsF32 *>(times.request().ptr);
  auto *inv_ptr = static_cast<IdI32 *>(inv_idx.request().ptr);

  std::unordered_map<int64_t, ssize_t> key2idx;
  auto *uniq_node = new std::vector<IdI32>;
  auto *uniq_time = new std::vector<TsF32>;
  uniq_node->reserve(size);
  uniq_time->reserve(size);
  key2idx.reserve(size);

  bool has_dups = false;
  for (ssize_t i = 0; i < size; i++) {
    IdI32 nid = node_ptr[i];
    TsF32 nts = time_ptr[i];
    int64_t key = opt_hash(nid, nts);
    auto iter = key2idx.find(key);
    if (iter != key2idx.end()) {
      auto uniq_idx = iter->second;
      inv_ptr[i] = uniq_idx;
      has_dups = true;
    } else {
      auto idx = uniq_node->size();
      uniq_node->push_back(nid);
      uniq_time->push_back(nts);
      key2idx.emplace(key, idx);
      inv_ptr[i] = idx;
    }
  }

  py::array_t<IdI32> res_nodes = to_pyarray_owned(uniq_node);
  py::array_t<TsF32> res_times = to_pyarray_owned(uniq_time);
  return py::make_tuple(has_dups, res_nodes, res_times, inv_idx);
}

// py::tuple dedup_indices(torch::Tensor &nodes, torch::Tensor &times) {
//   ssize_t size = nodes.size(0);
//
//   auto nodes_ptr = nodes.accessor<IdI32, 1>().data();
//   auto times_ptr = times.accessor<TsF32, 1>().data();
//
//   auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64);
//   auto inv_idx = torch::zeros(size, opt_i64);
//   auto inv_ptr = inv_idx.accessor<int64_t, 1>().data();
//
//   std::unordered_map<int64_t, ssize_t> key2idx;
//   std::vector<int64_t> indices;
//   key2idx.reserve(size);
//
//   for (ssize_t i = 0; i < size; i++) {
//     IdI32 nid = nodes_ptr[i];
//     TsF32 nts = times_ptr[i];
//
//     int64_t key = opt_hash(nid, nts);
//     auto it = key2idx.find(key);
//
//     if (it != key2idx.end()) {
//       auto uniq_idx = it->second;
//       inv_ptr[i] = uniq_idx;
//     } else {
//       auto idx = indices.size();
//       indices.push_back(i);
//       key2idx.emplace(key, idx);
//       inv_ptr[i] = idx;
//     }
//   }
//
//   py::array_t<int64_t> filter_idx = py::cast(indices);
//   return py::make_tuple(filter_idx, inv_idx);
// }

// bool dedup_targets(
//     const IdI32 *nodes_ptr, const TsF32 *times_ptr, size_t len,
//     const IdI32 *pre_nodes, const TsF32 *pre_times, size_t pre_len,
//     std::vector<IdI32> &uniq_nodes,
//     std::vector<TsF32> &uniq_times,
//     std::vector<IdI32> &inv_idx) {
//
//   std::unordered_map<int64_t, size_t> key2idx;
//   key2idx.reserve(len + pre_len);
//   bool has_dups = false;
//
//   for (size_t i = 0; i < len + pre_len; i++) {
//     IdI32 nid = i < pre_len ? pre_nodes[i] : nodes_ptr[i - pre_len];
//     TsF32 nts = i < pre_len ? pre_times[i] : times_ptr[i - pre_len];
//
//     int64_t key = opt_hash(nid, nts);
//     auto it = key2idx.find(key);
//
//     if (it != key2idx.end()) {
//       auto uniq_idx = it->second;
//       inv_idx.push_back(uniq_idx);
//       has_dups = true;
//     } else {
//       auto idx = uniq_nodes.size();
//       uniq_nodes.push_back(nid);
//       uniq_times.push_back(nts);
//       key2idx.emplace(key, idx);
//       inv_idx.push_back(idx);
//     }
//   }
//
//   return has_dups;
// }

} // namespace tglite

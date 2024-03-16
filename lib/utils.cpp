#include "tglite/utils.h"
#include <omp.h>

namespace tglite {

// torch::Tensor index_pinned(torch::Tensor &input, torch::Tensor &index) {
//   int64_t nrows = index.size(0);
//   int64_t nfeats = input.size(1);
//
//   auto opts = torch::TensorOptions()
//       .device(input.device())
//       .dtype(input.dtype())
//       .pinned_memory(true);
//   auto out = torch::zeros({nrows, nfeats}, opts);
//
//   auto *out_ptr = out.accessor<float, 2>().data();
//   auto *inp_ptr = input.accessor<float, 2>().data();
//   auto *idx_ptr = index.accessor<int64_t, 1>().data();
//
//   #pragma omp parallel for
//   for (int64_t i = 0; i < nrows; i++) {
//     auto idx = idx_ptr[i];
//     auto inp_start = inp_ptr + idx * nfeats;
//     std::copy(inp_start, inp_start + nfeats, out_ptr + i * nfeats);
//   }
//
//   return out;
// }

// py::tuple find_last_message(py::array_t<IdI32> &uniq_nodes, py::array_t<IdI32> &edges) {
//   ssize_t num_nodes = uniq_nodes.size();
//   ssize_t num_edges = edges.shape(0);
//
//   auto *msg_order = new std::vector<IdI32>;
//   auto *msg_index = new std::vector<IdI32>;
//   msg_order->resize(num_nodes * 2);
//   msg_index->resize(num_nodes);
//
//   auto *nodes_ptr = static_cast<IdI32 *>(uniq_nodes.request().ptr);
//   auto *edges_ptr = static_cast<IdI32 *>(edges.request().ptr);
//
//   #pragma omp parallel for schedule(static)
//   for (ssize_t i = 0; i < num_nodes; i++) {
//     IdI32 nid = nodes_ptr[i];
//     for (ssize_t e = num_edges - 1; e >= 0; e--) {
//       if (edges_ptr[e * 2 + 0] == nid) {
//         // is src node, order is same
//         (*msg_order)[i * 2 + 0] = edges_ptr[e * 2 + 0];
//         (*msg_order)[i * 2 + 1] = edges_ptr[e * 2 + 1];
//         (*msg_index)[i] = e;
//         break;
//       } else if (edges_ptr[e * 2 + 1] == nid) {
//         // is dst node, order is flipped
//         (*msg_order)[i * 2 + 0] = edges_ptr[e * 2 + 1];
//         (*msg_order)[i * 2 + 1] = edges_ptr[e * 2 + 0];
//         (*msg_index)[i] = e;
//         break;
//       }
//     }
//   }
//
//   py::array_t<IdI32> res_order = to_pyarray_owned(msg_order);
//   py::array_t<IdI32> res_index = to_pyarray_owned(msg_index);
//   return py::make_tuple(res_order, res_index);
// }

py::array_t<IdI32> find_latest_uniq(py::array_t<IdI32> &uniq, py::array_t<IdI32> &nodes, py::array_t<TsF32> &times) {
  ssize_t num_uniq = uniq.size();
  ssize_t num_nodes = nodes.size();

  auto *index = new std::vector<IdI32>;
  index->resize(num_uniq);

  auto *uniq_ptr = static_cast<IdI32 *>(uniq.request().ptr);
  auto *node_ptr = static_cast<IdI32 *>(nodes.request().ptr);
  auto *time_ptr = static_cast<TsF32 *>(times.request().ptr);

  #pragma omp parallel for schedule(static)
  for (ssize_t i = 0; i < num_uniq; i++) {
    IdI32 nid = uniq_ptr[i];
    TsF32 max = -1.0f;
    for (ssize_t j = num_nodes - 1; j >= 0; j--) {
      if (node_ptr[j] == nid && time_ptr[j] > max) {
        max = time_ptr[j];
        (*index)[i] = j;
      }
    }
  }

  py::array_t<IdI32> res = to_pyarray_owned(index);
  return res;
}

} // namespace tglite

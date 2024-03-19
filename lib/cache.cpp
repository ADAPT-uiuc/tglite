#include "tglite/cache.h"
#include "tglite/utils.h"

namespace tglite {

py::tuple find_dedup_time_hits(
    torch::Tensor &times,
    torch::Tensor &time_table,
    int time_window
) {
  auto tup = torch::_unique(times.flatten(), /*sorted=*/true, /*return_inverse=*/true);

  times = std::get<0>(tup);
  auto inv_idx = std::get<1>(tup);

  auto delta_int = times.to(torch::kInt64);
  auto hit_idx = (delta_int == times) & (0 <= delta_int) & (delta_int <= time_window);
  auto hit_delta = delta_int.index({hit_idx});

  int64_t hit_count = torch::sum(hit_idx).item().toLong();
  int64_t uniq_size = times.size(0);

  torch::Tensor out_embeds;
  if (hit_count == uniq_size) {
    out_embeds = time_table.index({hit_delta});
  } else {
    int64_t time_dim = time_table.size(1);
    auto opts = torch::TensorOptions().device(time_table.device());
    out_embeds = torch::zeros({uniq_size, time_dim}, opts);
    out_embeds.index_put_({hit_idx}, time_table.index({hit_delta}));
  }

  return py::make_tuple(hit_count, hit_idx, out_embeds, times, inv_idx);
}

py::array_t<int64_t> compute_cache_keys(py::array_t<IdI32> &nodes, py::array_t<TsF32> &times) {
  ssize_t size = nodes.size();
  auto keys = py::array_t<int64_t>(size);
  auto keys_ptr = static_cast<int64_t *>(keys.request().ptr);
  auto node_ptr = static_cast<IdI32 *>(nodes.request().ptr);
  auto time_ptr = static_cast<TsF32 *>(times.request().ptr);
  for (ssize_t i = 0; i < size; i++) {
    keys_ptr[i] = opt_hash(node_ptr[i], time_ptr[i]);
  }
  return keys;
}

EmbedTable::
EmbedTable(ssize_t dim_emb, ssize_t limit)
    : _dim_emb(dim_emb), _limit(limit) {
  ssize_t start_capacity = std::min((ssize_t)1024, limit);
  _table = torch::zeros({start_capacity, dim_emb}, torch::TensorOptions().dtype(torch::kFloat32));
  _key2idx.reserve(start_capacity);
  _keys.resize(start_capacity);
}

py::tuple EmbedTable::
lookup(py::array_t<int64_t> &keys, torch::Device &device) {
  ssize_t size = keys.size();
  auto output = torch::zeros({size, _dim_emb});
  auto hit_idx = torch::zeros(size, torch::TensorOptions().dtype(torch::kBool));

  auto *keys_ptr = static_cast<int64_t *>(keys.request().ptr);
  auto *hit_ptr = hit_idx.accessor<bool, 1>().data();
  std::vector<int64_t> indices;
  indices.reserve(size);

  for (ssize_t i = 0; i < size; i++) {
    auto it = _key2idx.find(keys_ptr[i]);
    if (it != _key2idx.end()) {
      indices.push_back(it->second);
      hit_ptr[i] = true;
    }
  }

  output.index_put_({hit_idx}, _table.index(
      {torch::from_blob(indices.data(), indices.size(),
          torch::TensorOptions().dtype(torch::kLong))}));

  output = output.to(device);
  hit_idx = hit_idx.to(device);
  return py::make_tuple(hit_idx, output);
}

void EmbedTable::
store(py::array_t<int64_t> &keys, torch::Tensor &values) {
  ssize_t size = keys.size();
  auto *keys_ptr = static_cast<int64_t *>(keys.request().ptr);
  auto embeds = values.detach().cpu();

  ssize_t nrows = _table.size(0);
  if (_start + size > nrows && nrows < _limit) {
    ssize_t grow_size = std::max(_start + size, nrows * 2);
    grow_size = std::min(grow_size, _limit);
    _table.resize_({grow_size, _dim_emb});
    _keys.resize(grow_size);
    nrows = grow_size;
  }

  ssize_t inp_idx = 0;
  while (size > 0) {
    ssize_t nslots = std::min(nrows - _start, size);
    ssize_t inp_end = inp_idx + nslots;
    ssize_t out_end = _start + nslots;

    _table.index_put_({torch::indexing::Slice(_start, out_end)},
        embeds.index({torch::indexing::Slice(inp_idx, inp_end)}));

    for (ssize_t i = 0; i < nslots; i++) {
      ssize_t slot = _start + i;
      int64_t old_key = _keys[slot];
      int64_t new_key = keys_ptr[inp_idx + i];
      _key2idx.erase(old_key);
      _key2idx.emplace(new_key, slot);
      _keys[slot] = new_key;
    }

    _start = out_end % nrows;
    inp_idx = inp_end;
    size -= nslots;
  }
}

} // namespace tglite

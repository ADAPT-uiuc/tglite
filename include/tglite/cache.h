#pragma once

#include "tglite/core.h"

namespace tglite {

py::tuple find_dedup_time_hits(
    torch::Tensor &times,
    torch::Tensor &time_table,
    int time_window);

py::array_t<int64_t> compute_cache_keys(py::array_t<IdI32> &nodes, py::array_t<TsF32> &times);

/// Table for caching computed embeddings.
class EmbedTable {
public:
  EmbedTable(ssize_t dim_emb, ssize_t limit);

  py::tuple lookup(py::array_t<int64_t> &keys, torch::Device &device);

  void store(py::array_t<int64_t> &keys, torch::Tensor &values);

private:
  ssize_t _dim_emb;
  ssize_t _limit;

  ssize_t _start = 0;
  torch::Tensor _table;
  std::vector<int64_t> _keys;
  std::unordered_map<int64_t, ssize_t> _key2idx;
  // torch::Tensor _pin;
};

} // namespace tglite

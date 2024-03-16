#pragma once

#include "tglite/core.h"

namespace tglite {

// torch::Tensor index_pinned(torch::Tensor &input, torch::Tensor &index);

// py::tuple find_last_message(py::array_t<IdI32> &uniq_nodes, py::array_t<IdI32> &edges);
py::array_t<IdI32> find_latest_uniq(py::array_t<IdI32> &uniq, py::array_t<IdI32> &nodes, py::array_t<TsF32> &times);

/// Custom hash function for collision-free keys.
inline int64_t opt_hash(int32_t &s, float &t) {
  return (static_cast<int64_t>(s) << 32) | static_cast<int32_t>(t);
}

template <typename T>
inline py::array_t<typename T::value_type> to_pyarray_view(const T &seq) {
    if (seq.size() > 0) {
        auto capsule = py::capsule(&seq, [](void* p) { /* borrowed */ });
        return py::array(seq.size(), seq.data(), capsule);
    } else {
        return py::array();
    }
}

template <typename T>
inline py::array_t<typename T::value_type> to_pyarray_copy(const T &seq) {
    if (seq.size() > 0) {
        T* copy_ptr = new T(seq);
        auto capsule = py::capsule(copy_ptr, [](void* p) { delete reinterpret_cast<T*>(p); });
        return py::array(copy_ptr->size(), copy_ptr->data(), capsule);
    } else {
        return py::array();
    }
}

template <typename T>
inline py::array_t<typename T::value_type> to_pyarray_owned(T *seq_ptr) {
    if (seq_ptr && seq_ptr->size() > 0) {
        auto capsule = py::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<T*>(p); });
        return py::array(seq_ptr->size(), seq_ptr->data(), capsule);
    } else {
        return py::array();
    }
}

} // namespace tglite

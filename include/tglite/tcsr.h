#pragma once

#include "tglite/core.h"
#include "tglite/utils.h"

namespace tglite {

class TCSR {
public:
  std::vector<IdI32> ind;
  std::vector<IdI32> nbr;
  std::vector<IdI32> eid;
  std::vector<TsF32> ets;

  TCSR() {
    ind.push_back(0);
  }

  py::array_t<IdI32> ind_view() const { return to_pyarray_view(ind); }
  py::array_t<IdI32> nbr_view() const { return to_pyarray_view(nbr); }
  py::array_t<IdI32> eid_view() const { return to_pyarray_view(eid); }
  py::array_t<TsF32> ets_view() const { return to_pyarray_view(ets); }
};

TCSR create_tcsr(py::array_t<IdI32> &edges, py::array_t<TsF32> &times, size_t num_nodes);

} // namespace tglite

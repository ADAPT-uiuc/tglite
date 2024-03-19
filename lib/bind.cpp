#include "tglite/core.h"
#include "tglite/cache.h"
#include "tglite/dedup.h"
#include "tglite/sampler.h"
#include "tglite/tcsr.h"
#include "tglite/utils.h"

using namespace tglite;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<TCSR>(m, "TCSR")
    .def(py::init<>())
    .def_property_readonly("ind", &TCSR::ind_view)
    .def_property_readonly("nbr", &TCSR::nbr_view)
    .def_property_readonly("eid", &TCSR::eid_view)
    .def_property_readonly("ets", &TCSR::ets_view);

  py::class_<TemporalBlock>(m, "TemporalBlock")
    .def(py::init<>())
    .def("copy_eid", &TemporalBlock::eid_copy)
    .def("copy_ets", &TemporalBlock::ets_copy)
    .def("copy_srcnodes", &TemporalBlock::srcnodes_copy)
    .def("copy_dstindex", &TemporalBlock::dstindex_copy)
    .def("num_edges", [](const TemporalBlock &b) { return b.num_edges; });

  py::class_<TemporalSampler>(m, "TemporalSampler")
    .def(py::init<int, int, bool>())
    .def("sample", &TemporalSampler::sample);

  py::class_<EmbedTable>(m, "EmbedTable")
    .def(py::init<ssize_t, ssize_t>())
    // .def("_table", [](const EmbedTable &et) { return et._table; })
    // .def("_keys", [](const EmbedTable &et) { return et._keys; })
    // .def("_map", [](const EmbedTable &et) { return et._key2idx; })
    .def("lookup", &EmbedTable::lookup, "tglite::EmbedTable::lookup")
    .def("store", &EmbedTable::store, "tglite::EmbedTable::store");

  m.def("create_tcsr", &create_tcsr, "tglite::create_tcsr");
  m.def("dedup_targets", &dedup_targets, "tglite::dedup_targets");
  m.def("find_latest_uniq", &find_latest_uniq, "tglite::find_latest_uniq");
  // m.def("find_last_message", &find_last_message, "tglite::find_last_message");
  m.def("find_dedup_time_hits", &find_dedup_time_hits, "tglite::find_dedup_time_hits");
  m.def("compute_cache_keys", &compute_cache_keys, "tglite::compute_cache_keys");

  // m.def("dedup_indices", &dedup_indices, "tglite::dedup_indices");
  // m.def("index_pinned", &index_pinned, "tglite::index_pinned");
}

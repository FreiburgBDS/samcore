// nanobind module definition for the samcore package.
//
// The actual bindings live in per-type translation units under
// src/bindings/ (see common.hpp for the shared helpers and entry points).

#include "common.hpp"

// module

NB_MODULE(_samcore, m) {
    m.doc() = "samcore: C++ (libsamcore) backend for SAM data processing";

    bind_header(m);
    bind_labels(m);
    bind_scan(m);
    bind_dataset(m);
    bind_submodules(m);
}

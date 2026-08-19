#pragma once

// Internal: lazy (mmap-style) h5sam loading.  The HDF5 file stays open and
// the data dataset is read on first access.  Not installed; not public API.

#include <H5Cpp.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <samcore/sam_header.hpp>
#include <samcore/sam_labels.hpp>

namespace samcore::io {

struct h5sam_lazy_state {
    H5::H5File file;
    H5::DataSet dset;
    size_t rows = 0;
    size_t cols = 0;
};

struct h5sam_lazy_handle {
    sam_header header;
    sam_labels labels;
    std::optional<std::vector<std::int32_t>> starts;
    std::unique_ptr<h5sam_lazy_state> data;
};

// Open an .h5sam file, parse header/labels/starts eagerly and leave the
// signal data dataset open for on-demand reads.
[[nodiscard]] h5sam_lazy_handle read_h5sam_lazy(
    const std::filesystem::path& path);

} // namespace samcore::io

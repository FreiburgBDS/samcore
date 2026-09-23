#pragma once

// Internal: lazy (mmap-style) h5sam/h5samd loading.  The HDF5 file stays
// open and the data datasets are read on first access.  Not installed; not
// public API.

#include <H5Cpp.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <samcore/array.hpp>
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

// --- .h5samd -------------------------------------------------------------

// Open HDF5 handles for a .h5samd file.  X/Z/V stay on disk until
// read_h5samd_lazy_data is called.
struct h5samd_lazy_state {
    H5::H5File file;
    std::optional<H5::DataSet> x;
    std::optional<H5::DataSet> z;
    std::optional<H5::DataSet> v;
    size_t x_rows = 0;
    size_t x_cols = 0;
    size_t z_cols = 0;
    size_t v_cols = 0;
};

struct h5samd_lazy_handle {
    std::optional<sam_labels> labels;
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes;
    std::vector<double> cube_resolutions;
    std::vector<std::int32_t> scanlens;
    bool unsupervised = false;
    std::unique_ptr<h5samd_lazy_state> data;
};

// Owned X/Z/V read out of a lazy handle.
struct h5samd_lazy_data {
    array2d<float> x;
    std::optional<array2d<float>> z;
    std::optional<array2d<float>> v;
};

// Open a .h5samd file, parse labels/shapes/resolutions/scanlens eagerly and
// leave X/Z/V open for on-demand reads.
[[nodiscard]] h5samd_lazy_handle read_h5samd_lazy(
    const std::filesystem::path& path);

// Read X/Z/V out of an open lazy state.
[[nodiscard]] h5samd_lazy_data read_h5samd_lazy_data(h5samd_lazy_state& state);

} // namespace samcore::io

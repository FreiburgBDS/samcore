#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/sam_header.hpp>
#include <samcore/sam_labels.hpp>
#include <samcore/sam_merge.hpp>

namespace samcore::io {

struct h5sam_result {
    array2d<std::int8_t> data;
    sam_header header;
    sam_labels labels;
    std::optional<std::vector<std::int32_t>> starts;
};

// Read a .h5sam (HDF5) file.
[[nodiscard]] h5sam_result read_h5sam(const std::filesystem::path& path);

// Write a .h5sam (HDF5) file: a `header` group
// attributes, gzip-compressed `data` (int8), `labels` (int8),
// `label_names` (fixed-length ASCII), optional `starts` (int32).
void write_h5sam(const std::filesystem::path& path,
                 const array2d<std::int8_t>& data,
                 const sam_header& header,
                 const sam_labels& labels,
                 const std::optional<std::vector<std::int32_t>>& starts);

// Partial row read of a .h5sam file's data dataset (first..first+count).
// Partial row read (counterpart of a lazy-mode scan).
[[nodiscard]] array2d<std::int8_t> read_h5sam_rows(
    const std::filesystem::path& path, size_t first, size_t count);

struct h5samd_result {
    array2d<float> x;
    std::optional<sam_labels> labels; // nullopt when unsupervised
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes;
    std::vector<double> cube_resolutions;
    std::vector<std::int32_t> scanlens;
    std::vector<std::int32_t> valid_scanlens;
    std::vector<double> cube_samplerates;
    merge_mode merge = merge_mode::pad; // legacy files were padded
    float pad_value = 0.0f;
    bool unsupervised = false;
    std::optional<array2d<float>> z;
    std::optional<array2d<float>> v;
};

// Read a .h5samd (HDF5) file.
// including h5py bool datasets for `unsupervised`.
[[nodiscard]] h5samd_result read_h5samd(const std::filesystem::path& path);

// Write a .h5samd file (see sam_dataset.save).
void write_h5samd(const std::filesystem::path& path,
                  const array2d<float>& x,
                  const std::optional<sam_labels>& labels,
                  const std::vector<std::pair<std::int32_t, std::int32_t>>& cube_shapes,
                  const std::vector<double>& cube_resolutions,
                  const std::vector<std::int32_t>& scanlens,
                  const std::vector<std::int32_t>& valid_scanlens,
                  const std::vector<double>& cube_samplerates,
                  merge_mode merge, float pad_value, bool unsupervised,
                  const std::optional<array2d<float>>& z,
                  const std::optional<array2d<float>>& v);

// Compatibility overload for direct low-level writers (legacy padded files).
inline void write_h5samd(
    const std::filesystem::path& path, const array2d<float>& x,
    const std::optional<sam_labels>& labels,
    const std::vector<std::pair<std::int32_t, std::int32_t>>& cube_shapes,
    const std::vector<double>& cube_resolutions,
    const std::vector<std::int32_t>& scanlens, bool unsupervised,
    const std::optional<array2d<float>>& z,
    const std::optional<array2d<float>>& v) {
    write_h5samd(path, x, labels, cube_shapes, cube_resolutions, scanlens,
                 scanlens, {}, merge_mode::pad, 0.0f, unsupervised, z, v);
}

// Convert multiple .h5sam files into a single .h5samd, streaming data
// per file to bound memory usage.  Global xcorr needs one extra signal pass
// because the final common overlap depends on all lags.
void convert_h5sam_to_h5samd(
    const std::vector<std::filesystem::path>& input_paths,
    const std::filesystem::path& output_path,
    const dataset_merge_options& options = {},
    std::optional<bool> unsupervised = std::nullopt);

// Compatibility overload preserving the old explicit-pad calling convention.
inline void convert_h5sam_to_h5samd(
    const std::vector<std::filesystem::path>& input_paths,
    const std::filesystem::path& output_path, float pad_value,
    std::optional<bool> unsupervised = std::nullopt) {
    dataset_merge_options options;
    options.mode = merge_mode::pad;
    options.pad_value = pad_value;
    convert_h5sam_to_h5samd(input_paths, output_path, options, unsupervised);
}

} // namespace samcore::io

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/sam_labels.hpp>
#include <samcore/sam_scan.hpp>

namespace samcore {

// Spatial provenance of one sample: source cube index and pixel-centre
// coordinates in mm (x = column, y = line), computed from the cube's
// resolution.
struct spatial_record {
    std::int32_t idx;
    float x;
    float y;
};

// Parameters forwarded to sam_dataset::preprocess strategies.
struct preprocess_args {
    double cutoff = 0.0;
    double cutoff_low = 0.0;
    double cutoff_high = 0.0;
    double fs = 0.0;
    std::string mode = "minmax";
    size_t window_length = 5;
    size_t polyorder = 2;
    size_t kernel_size = 3;
    size_t start = 0;
    size_t end = 0;
    size_t window = 5;
};

// Collection of SAM A-scans pooled from one or more scan cubes,
// zero-padded to a common length with spatial provenance per row.
// Provides: construction
// from handlers, padding/concatenation, label merging, serialization
// as .h5samd, preprocessing and cube extraction.  ML iteration
// (batches/splits/patches) is deliberately not part of the C++ API.
class sam_dataset {
public:
    sam_dataset() = default;

    // Deep copy of the dataset (X, labels, provenance, Z/V, splits).
    [[nodiscard]] sam_dataset copy() const;

    // Build from one or more scan handlers.  Signals are converted to
    // float32 and zero-padded (pad_value) to the maximum scan length;
    // labels are merged by name.  `unsupervised` is auto-detected when
    // nullopt (unsupervised unless every cube is labeled).
    explicit sam_dataset(std::vector<sam_scan> handlers, float pad_value = 0.0f,
                         std::optional<bool> unsupervised = std::nullopt);

    // IO

    // Save as .h5samd; throws std::invalid_argument unless the path ends
    // with .h5samd.
    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static sam_dataset load(const std::filesystem::path& path);

    // accessors

    [[nodiscard]] const array2d<float>& X() const noexcept { return x_; }
    [[nodiscard]] array2d<float>& X() noexcept { return x_; }
    [[nodiscard]] const std::optional<sam_labels>& labels() const noexcept {
        return labels_;
    }
    [[nodiscard]] std::optional<sam_labels>& labels() noexcept { return labels_; }
    [[nodiscard]] const std::optional<array2d<float>>& Z() const noexcept { return z_; }
    [[nodiscard]] std::optional<array2d<float>>& Z() noexcept { return z_; }
    [[nodiscard]] const std::optional<array2d<float>>& V() const noexcept { return v_; }
    [[nodiscard]] std::optional<array2d<float>>& V() noexcept { return v_; }
    [[nodiscard]] bool unsupervised() const noexcept { return unsupervised_; }
    [[nodiscard]] float pad_value() const noexcept { return pad_value_; }
    [[nodiscard]] size_t num_samples() const noexcept { return x_.rows(); }
    [[nodiscard]] size_t num_features() const noexcept {
        return z_ ? z_->cols() : 0;
    }
    [[nodiscard]] size_t maxlen() const noexcept { return x_.cols(); }
    [[nodiscard]] const std::vector<std::pair<std::int32_t, std::int32_t>>&
    cube_shapes() const noexcept { return cube_shapes_; }
    [[nodiscard]] const std::vector<double>& cube_resolutions() const noexcept {
        return cube_resolutions_;
    }
    [[nodiscard]] const std::vector<std::int32_t>& scanlens() const noexcept {
        return scanlens_;
    }

    // Spatial provenance for every sample, computed from cube shapes and
    // resolutions (mm).
    [[nodiscard]] std::vector<spatial_record> spatial() const;

    [[nodiscard]] size_t num_classes() const;

    // preprocessing

    // Apply a built-in strategy to X in place; strategies: "lp", "bp",
    // "normalize", "savgol", "medfilt", "gate", "detrend", "envelope",
    // "moving_average".  Unknown strategies throw std::invalid_argument.
    void preprocess(const std::string& strategy, const preprocess_args& args = {});

    // label delegation

    [[nodiscard]] std::map<std::string, size_t> class_distribution() const;
    [[nodiscard]] std::vector<std::int8_t> to_binary(
        std::variant<std::int8_t, std::string> positive_label) const;
    [[nodiscard]] array2d<float> to_one_hot() const;
    void relabel(const std::map<
                 std::variant<std::int64_t, std::string>,
                 std::variant<std::int64_t, std::string>>& mapping);

    // cube extraction

    // split bookkeeping (set/updated by the Python layer)

    [[nodiscard]] const std::vector<std::int64_t>& train_indices() const noexcept {
        return train_indices_;
    }
    [[nodiscard]] std::vector<std::int64_t>& train_indices() noexcept {
        return train_indices_;
    }
    [[nodiscard]] const std::vector<std::int64_t>& test_indices() const noexcept {
        return test_indices_;
    }
    [[nodiscard]] std::vector<std::int64_t>& test_indices() noexcept {
        return test_indices_;
    }
    [[nodiscard]] bool shuffled() const noexcept { return shuffled_; }
    void set_shuffled(bool v) noexcept { shuffled_ = v; }

    // cube extraction

    // Signal data of cube idx reshaped to (nlines, scanspline, scanlen).
    [[nodiscard]] array3d<float> get_cube_X(std::int32_t idx) const;
    // Feature data of cube idx reshaped to (nlines, scanspline, n_features).
    [[nodiscard]] array3d<float> get_cube_Z(std::int32_t idx) const;
    // Low-dimensional coordinates of cube idx (nlines, scanspline, n_dims).
    [[nodiscard]] array3d<float> get_cube_V(std::int32_t idx) const;
    // Label grid of cube idx (nlines, scanspline).
    [[nodiscard]] array2d<std::int8_t> get_cube_labels(std::int32_t idx) const;

private:
    array2d<float> x_;
    std::optional<sam_labels> labels_;
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes_;
    std::vector<double> cube_resolutions_;
    std::vector<std::int32_t> scanlens_;
    float pad_value_ = 0.0f;
    bool unsupervised_ = false;
    std::optional<array2d<float>> z_;
    std::optional<array2d<float>> v_;
    std::vector<std::int64_t> train_indices_;
    std::vector<std::int64_t> test_indices_;
    bool shuffled_ = false;
};

} // namespace samcore

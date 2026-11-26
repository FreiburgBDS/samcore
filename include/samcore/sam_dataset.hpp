#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/sam_labels.hpp>
#include <samcore/sam_merge.hpp>
#include <samcore/sam_scan.hpp>

namespace samcore {

namespace io {
struct h5samd_lazy_state; // defined in src/io/h5_lazy.hpp
}

// Spatial provenance of one sample: source cube index and pixel-centre
// coordinates in mm (x = column, y = line), computed from the cube's
// resolution (µm per pixel).
struct spatial_record {
    std::int32_t idx;
    float x; // mm
    float y; // mm
};

// Parameters forwarded to sam_dataset::preprocess strategies.  cutoff* and
// fs must share one frequency unit (MHz to match sam_header::samplerate;
// only their ratio is used); the window/sample fields count samples.
struct preprocess_args {
    double cutoff = 0.0;      // frequency unit of fs
    double cutoff_low = 0.0;  // frequency unit of fs
    double cutoff_high = 0.0; // frequency unit of fs
    double fs = 0.0;          // frequency unit of cutoff*
    std::string mode = "minmax";
    size_t window_length = 5; // samples
    size_t polyorder = 2;
    size_t kernel_size = 3;   // samples
    size_t start = 0;         // sample index
    size_t end = 0;           // sample index (0 = full length)
    size_t window = 5;        // samples
};

// Collection of SAM A-scans pooled from one or more scan cubes.  Scans are
// combined by the selected merge policy (crop to the shortest, pad, global
// xcorr alignment, or zgate alignment) with spatial provenance per row.
// Provides: construction from handlers, label merging, serialization as
// .h5samd, preprocessing, alignment, spectral analysis and cube extraction.
// ML iteration (batches/splits/patches) is deliberately not part of the C++
// API.
class sam_dataset {
public:
    sam_dataset() = default;
    ~sam_dataset();
    sam_dataset(const sam_dataset&);
    sam_dataset& operator=(const sam_dataset&);
    sam_dataset(sam_dataset&&) noexcept;
    sam_dataset& operator=(sam_dataset&&) noexcept;

    // Deep copy of the dataset (X, labels, provenance, Z/V, splits).
    // A lazy-mode dataset is materialized first.
    [[nodiscard]] sam_dataset copy() const;

    // Build from one or more scan handlers using the merge options.
    // `unsupervised` is auto-detected when nullopt (unsupervised unless
    // every cube is labeled).
    explicit sam_dataset(std::vector<sam_scan> handlers,
                         const dataset_merge_options& options,
                         std::optional<bool> unsupervised = std::nullopt);

    // Compatibility overload using the default crop policy.  `pad_value` is
    // retained for source compatibility and is ignored by crop; use the
    // options overload with merge_mode::pad for padding.
    explicit sam_dataset(std::vector<sam_scan> handlers,
                         float pad_value = 0.0f,
                         std::optional<bool> unsupervised = std::nullopt);

    // IO

    // Save as .h5samd; throws std::invalid_argument unless the path ends
    // with .h5samd.  A lazy-mode dataset is materialized first.
    void save(const std::filesystem::path& path) const;

    // Load a .h5samd file.  With lazy = true the X/Z/V datasets stay on
    // disk (the file handle is kept open) and are decoded in cached row
    // blocks on first access; labels, cube shapes, resolutions and scan
    // lengths are always loaded eagerly.  The data is chunked + compressed,
    // so this is paged lazy reading, not memory mapping.
    [[nodiscard]] static sam_dataset load(const std::filesystem::path& path,
                                          bool lazy = false);

    // Whether the data has been loaded into memory (false while a
    // lazy-mode dataset is still backed by the on-disk HDF5 file).
    [[nodiscard]] bool loaded() const noexcept { return lazy_ == nullptr; }
    // Alias of loaded(); backing() is "eager" or "lazy".
    [[nodiscard]] bool materialized() const noexcept { return loaded(); }
    [[nodiscard]] std::string backing() const {
        return lazy_ ? "lazy" : "eager";
    }

    // Materialize X/Z/V (no-op when already loaded).  Every data accessor
    // calls this implicitly.
    void load() const { ensure_loaded(); }

    // Read selected rows of X (or Z with use_z) without materializing a
    // lazy dataset; only the blocks containing those rows are decoded.
    [[nodiscard]] array2d<float> read_rows(
        const std::vector<std::int64_t>& indices, bool use_z = false) const;
    // Number of X blocks decoded from disk so far (0 when materialized).
    [[nodiscard]] size_t blocks_read() const noexcept;

    // accessors

    // Data accessors materialize a lazy-mode dataset on first use.
    [[nodiscard]] const array2d<float>& X() const;
    [[nodiscard]] array2d<float>& X();
    [[nodiscard]] const std::optional<sam_labels>& labels() const noexcept {
        return labels_;
    }
    [[nodiscard]] std::optional<sam_labels>& labels() noexcept { return labels_; }
    [[nodiscard]] const std::optional<array2d<float>>& Z() const;
    [[nodiscard]] std::optional<array2d<float>>& Z();
    [[nodiscard]] const std::optional<array2d<float>>& V() const;
    [[nodiscard]] std::optional<array2d<float>>& V();
    [[nodiscard]] bool has_z() const noexcept;
    [[nodiscard]] bool has_v() const noexcept;
    [[nodiscard]] bool unsupervised() const noexcept { return unsupervised_; }
    [[nodiscard]] float pad_value() const noexcept { return pad_value_; }
    // Metadata only: these never materialize a lazy-mode dataset.
    [[nodiscard]] size_t num_samples() const noexcept;
    [[nodiscard]] size_t num_features() const noexcept;
    [[nodiscard]] size_t maxlen() const noexcept;
    [[nodiscard]] const std::vector<std::pair<std::int32_t, std::int32_t>>&
    cube_shapes() const noexcept { return cube_shapes_; }
    // Lateral resolution of each cube in µm/pixel.
    [[nodiscard]] const std::vector<double>& cube_resolutions() const noexcept {
        return cube_resolutions_;
    }
    // Original A-scan lengths of each source cube in samples.
    [[nodiscard]] const std::vector<std::int32_t>& scanlens() const noexcept {
        return scanlens_;
    }
    // Valid signal samples currently represented by each cube in X.  This
    // equals the original scanlen under pad and the common merged width
    // after crop/xcorr/zgate.
    [[nodiscard]] const std::vector<std::int32_t>& valid_scanlens() const noexcept {
        return valid_scanlens_;
    }
    // Source sampling rate of each cube in MHz.  Empty for legacy files.
    [[nodiscard]] const std::vector<double>& cube_samplerates() const noexcept {
        return cube_samplerates_;
    }
    // Common sampling rate in MHz, or nullopt when unknown/mixed.
    [[nodiscard]] std::optional<double> samplerate() const noexcept;
    // Merge policy used to construct the dataset.
    [[nodiscard]] merge_mode merge() const noexcept { return merge_mode_; }
    [[nodiscard]] std::string merge_mode_name() const {
        return to_string(merge_mode_);
    }

    // Spatial provenance for every sample, computed from the cube shapes and
    // resolutions (µm/pixel); x/y are pixel-centre coordinates in mm.
    [[nodiscard]] std::vector<spatial_record> spatial() const;

    [[nodiscard]] size_t num_classes() const;

    // preprocessing

    // Apply a built-in strategy to X in place; strategies: "lp", "bp",
    // "normalize", "savgol", "medfilt", "gate", "detrend", "envelope",
    // "moving_average".  Unknown strategies throw std::invalid_argument.
    void preprocess(const std::string& strategy, const preprocess_args& args = {});

    // alignment over the whole pooled dataset (rows of X)

    // Extract `scanlen` samples starting at each row's relative start.
    // -1 marks an unaligned row and zero-fills it.
    void align_manual(const std::vector<std::int32_t>& starts,
                      std::int64_t scanlen);
    // Align every row to the pooled reference row; keeps the common overlap.
    void align_xcorr(size_t reference = 0, std::int64_t max_shift = 0,
                     double max_gate_loss = 1.0);
    // Keep `length` samples from each row's first threshold crossing.
    void align_zgate(double threshold = 0.2, std::int64_t length = 2000);
    // Align each row's envelope peak in [start_ns, start_ns+gate_ns) to the
    // reference row.  Requires a common samplerate and valid signal length.
    void align_tof(double gate_ns, size_t reference = 0,
                   double start_ns = 0.0, double max_gate_loss = 1.0);

    // spectral analysis over the whole pooled dataset (rows of X)

    // STFT of all rows.  Requires a common sampling rate and a common valid
    // length (use compute_stft_by_cube otherwise).  Times are relative to
    // the start of each signal; the dataset stores no tzero.
    [[nodiscard]] signal::stft_result compute_stft(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    [[nodiscard]] std::vector<signal::stft_result> compute_stft_by_cube(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    // Welch PSD of all rows.  Different valid lengths are allowed.
    [[nodiscard]] signal::psd_result psd(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    [[nodiscard]] std::vector<signal::psd_result> psd_by_cube(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    // Power spectrogram of all rows; same constraints as compute_stft().
    [[nodiscard]] signal::spectrogram_result power_spectrogram(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    [[nodiscard]] std::vector<signal::spectrogram_result>
    power_spectrogram_by_cube(size_t nperseg = 256, size_t noverlap = 128,
                              double f_min = 0.0, double f_max = 0.0,
                              signal::window_kind window =
                                  signal::window_kind::hann) const;
    // Per-row STFT peak frequency/time.  Times are relative to each row's
    // signal start.
    [[nodiscard]] std::vector<float> stft_peak_frequency(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    [[nodiscard]] std::vector<float> stft_peak_time(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0,
        signal::window_kind window = signal::window_kind::hann) const;
    // Flattened, optionally L1-normalized spectrogram features over all rows.
    // Requires a common valid length across cubes.  `fs_mhz` overrides the
    // stored sampling rate (useful for legacy files) and must agree with it
    // when metadata is present.
    [[nodiscard]] signal::spectrogram_features_result spectrogram_features(
        const signal::spectrogram_feature_options& options = {},
        std::optional<double> fs_mhz = std::nullopt) const;
    // One-sided FFT magnitude of all rows.  Requires a common sampling rate
    // and common valid length (use spectrum_by_cube otherwise).
    [[nodiscard]] signal::spectrum_result spectrum() const;
    [[nodiscard]] std::vector<signal::spectrum_result> spectrum_by_cube() const;

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
    void ensure_loaded() const;
    [[nodiscard]] double require_samplerate() const;
    [[nodiscard]] std::pair<size_t, size_t> cube_row_range_internal(
        std::int32_t idx) const;
    [[nodiscard]] array2d<float> cube_valid_rows(std::int32_t idx) const;

    // Data members are mutable: ensure_loaded() materializes a lazy-mode
    // dataset through const accessors (same pattern as sam_scan::data_).
    mutable array2d<float> x_;
    std::optional<sam_labels> labels_;
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes_;
    std::vector<double> cube_resolutions_;
    std::vector<std::int32_t> scanlens_;
    std::vector<std::int32_t> valid_scanlens_;
    std::vector<double> cube_samplerates_;
    merge_mode merge_mode_ = merge_mode::crop;
    float pad_value_ = 0.0f;
    bool unsupervised_ = false;
    mutable std::optional<array2d<float>> z_;
    mutable std::optional<array2d<float>> v_;
    std::vector<std::int64_t> train_indices_;
    std::vector<std::int64_t> test_indices_;
    bool shuffled_ = false;
    mutable std::unique_ptr<io::h5samd_lazy_state> lazy_;
};

} // namespace samcore

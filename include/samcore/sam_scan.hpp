#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/io/fileio.hpp>
#include <samcore/sam_header.hpp>
#include <samcore/sam_labels.hpp>
#include <samcore/signal/kernels.hpp>

namespace samcore {

namespace io {
struct h5sam_lazy_state; // forward declaration (defined in src/io/h5_lazy.hpp)
}

enum class image_mode { max, absmax, power };
enum class downsample_mode { decimate, mean, median, sample };
enum class mirror_axis { x, y };

// Result of image(mode): int8 for 'max',
// int16 for 'absmax', float32 for 'power').
using image_result = std::variant<array2d<std::int8_t>,
                                  array2d<std::int16_t>,
                                  array2d<float>>;

// Handler for SAM scan data; supports
// loading from .h5sam files, per-scan
// processing (image, downsample, rotate, mirror, selects, zgate) and
// spectral estimation (STFT / PSD / spectrogram).
class sam_scan {
public:
    sam_scan();
    ~sam_scan();
    sam_scan(sam_scan&&) noexcept;
    sam_scan& operator=(sam_scan&&) noexcept;
    sam_scan(const sam_scan&);
    sam_scan& operator=(const sam_scan&);

    // Load from a .h5sam file (same as from_file).
    explicit sam_scan(const std::string& path, bool mmap = false);

    // Load a .h5sam file.  With mmap = true the signal data stays on
    // disk (the file handle is kept open) and is read on first access;
    // header, labels and starts are always loaded eagerly.  Throws
    // std::invalid_argument for other extensions and std::runtime_error
    // for IO/parse errors.
    [[nodiscard]] static sam_scan from_file(const std::filesystem::path& path,
                                            bool mmap = false);

    // Build from data + header with shape validation
    // (data.ndim == 2, rows == nlines*scanspline, cols == scanlen).
    // Throws std::invalid_argument on mismatch.
    [[nodiscard]] static sam_scan from_data(
        array2d<std::int8_t> data, sam_header header,
        std::optional<std::vector<std::int32_t>> starts = std::nullopt,
        std::optional<sam_labels> labels = std::nullopt);

    // accessors

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] std::string& path() noexcept { return path_; }

    // Whether the signal data has been loaded into memory (false while a
    // mmap-mode scan is still backed by the on-disk HDF5 file).
    [[nodiscard]] bool loaded() const noexcept { return lazy_ == nullptr; }

    // Materialize the signal data (no-op when already loaded or not in
    // mmap mode).  Every data accessor calls this implicitly.
    void load() const { ensure_loaded(); }

    [[nodiscard]] const array2d<std::int8_t>& data() const {
        ensure_loaded();
        return data_;
    }
    [[nodiscard]] array2d<std::int8_t>& data() {
        ensure_loaded();
        return data_;
    }
    [[nodiscard]] const sam_header& header() const noexcept { return header_; }
    [[nodiscard]] sam_header& header() noexcept { return header_; }
    [[nodiscard]] const sam_labels& samlabels() const noexcept { return labels_; }
    [[nodiscard]] sam_labels& samlabels() noexcept { return labels_; }

    [[nodiscard]] const std::optional<std::vector<std::int32_t>>& starts() const noexcept {
        return starts_;
    }
    [[nodiscard]] std::optional<std::vector<std::int32_t>>& starts() noexcept {
        return starts_;
    }

    [[nodiscard]] std::int64_t nlines() const noexcept { return header_.nlines; }
    [[nodiscard]] std::int64_t cols() const noexcept { return header_.scanspline; }
    [[nodiscard]] std::int64_t scanlen() const noexcept { return header_.scanlen; }
    [[nodiscard]] double samplerate() const noexcept { return header_.samplerate; }
    [[nodiscard]] double downsample_factor() const noexcept { return header_.downsample_factor; }
    [[nodiscard]] std::pair<std::int64_t, std::int64_t> shape() const noexcept {
        return { header_.nlines, header_.scanspline };
    }
    // Scan by flat index, or by (line, col) spatial index.
    [[nodiscard]] std::span<const std::int8_t> scan(size_t index) const;
    [[nodiscard]] std::span<const std::int8_t> scan(std::int64_t line, std::int64_t col) const;
    [[nodiscard]] size_t num_scans() const noexcept;

    // IO

    // Save as .h5sam (HDF5).  Throws std::invalid_argument when the path
    // does not end in .h5sam.
    void to_h5sam(const std::filesystem::path& path) const;

    // time

    // Time index in nanoseconds; with a scan index, uses that scan's
    // start when the handler carries per-scan starts (gated data).
    [[nodiscard]] std::vector<double> time(std::optional<size_t> index = {}) const;
    [[nodiscard]] double samplespacing() const noexcept {
        return 1.0 / samplerate() * 1e3;
    }
    // Hash of the header metadata.
    [[nodiscard]] size_t header_hash() const { return header_.hash(); }

    // reductions

    // Reduce every scan to a single value, reshaped to (nlines, cols).
    [[nodiscard]] image_result image(image_mode mode) const;

    // Data in [-1, 127/128] as float32.
    [[nodiscard]] array2d<float> normalized_data() const;

    // labels

    void set_labels(sam_labels labels);
    void set_labels(std::vector<std::int8_t> labels,
                    std::vector<std::string> label_names = {});

    // Deep copy of the scan (data, header, labels, starts).
    [[nodiscard]] sam_scan copy() const {
        sam_scan h;
        h.data_ = data();
        h.header_ = header_;
        h.labels_ = labels_;
        h.starts_ = starts_;
        h.path_ = path_;
        return h;
    }

    // processing (mutating + const-copy duals)

    void downsample(size_t factor, downsample_mode mode = downsample_mode::decimate);
    [[nodiscard]] sam_scan downsampled(size_t factor,
                                       downsample_mode mode = downsample_mode::decimate) const;

    // Spatial rotation by 90/180/270 degrees clockwise; per-scan labels
    // and starts follow the moved signals.
    void rotate(int degrees);
    [[nodiscard]] sam_scan rotated(int degrees) const;

    // Spatial mirror along the x (column) or y (line) axis.
    void mirror(mirror_axis axis);
    [[nodiscard]] sam_scan mirrored(mirror_axis axis) const;

    // Rectangular spatial region [line_start, line_end) x [col_start, col_end).
    [[nodiscard]] sam_scan rectangle_select(std::int64_t line_start,
                                            std::int64_t line_end,
                                            std::int64_t col_start,
                                            std::int64_t col_end) const;
    void rectangle_select_ip(std::int64_t line_start, std::int64_t line_end,
                             std::int64_t col_start, std::int64_t col_end);

    // Time range in nanoseconds; adjusts tzero and scanlen, drops starts.
    [[nodiscard]] sam_scan time_range_select(double start_time,
                                             double end_time) const;
    void time_range_select_ip(double start_time, double end_time);

    // Threshold-based gating of every scan; returns a scan with starts
    // accumulated on top of existing ones.
    [[nodiscard]] sam_scan zgate(double threshold = 0.2, std::int64_t length = 2000) const;
    void zgate_ip(double threshold, std::int64_t length);

    // spectral

    // One-sided STFT of every scan (fs = samplerate in Hz); t aligned to
    // the handler's time scale in seconds.
    [[nodiscard]] signal::stft_result compute_stft(size_t nperseg = 256,
                                                   size_t noverlap = 128) const;
    // Welch PSD (scipy.signal.welch, detrend=False).
    [[nodiscard]] signal::psd_result psd(size_t nperseg = 256,
                                         size_t noverlap = 128) const;
    // Per-frame power spectral density (hann window, density scaling).
    [[nodiscard]] signal::spectrogram_result power_spectrogram(
        size_t nperseg = 256, size_t noverlap = 128) const;

    // iteration

    [[nodiscard]] std::span<const std::int8_t> operator[](size_t index) const {
        return scan(index);
    }

private:
    void ensure_loaded() const;
    void align_manual(const std::vector<std::int32_t>& new_starts,
                      std::int64_t new_scanlen);

    mutable array2d<std::int8_t> data_;
    sam_header header_;
    sam_labels labels_;
    std::optional<std::vector<std::int32_t>> starts_;
    std::string path_;
    mutable std::unique_ptr<io::h5sam_lazy_state> lazy_;
};

} // namespace samcore

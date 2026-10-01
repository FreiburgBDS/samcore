#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <samcore/array.hpp>
#include <samcore/constants.hpp>
#include <samcore/io/fileio.hpp>
#include <samcore/sam_header.hpp>
#include <samcore/sam_labels.hpp>
#include <samcore/signal/kernels.hpp>

namespace samcore {

namespace io {
struct h5sam_lazy_state; // forward declaration (defined in src/io/h5_lazy.hpp)
}

enum class downsample_mode { decimate, mean, median, sample };
enum class mirror_axis { x, y };

// Layered gating (XGate) result: one scalar per gate per scan plus the
// picked gate start of every scan (-1 when the pick failed).
struct xgate_result {
    array3d<float> values;            // (nlines, cols, n_gates)
    std::vector<std::int32_t> starts; // (n_signals,)
};

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
    explicit sam_scan(const std::string& path, bool lazy = false);

    // Load a .h5sam file.  With lazy = true the signal data stays on disk
    // (the file handle is kept open) and is decoded in cached row blocks on
    // first access; header, labels and starts are always loaded eagerly.
    // Note: the data is chunked + compressed, so this is paged lazy reading,
    // not memory mapping.  Throws std::invalid_argument for other extensions
    // and std::runtime_error for IO/parse errors.
    [[nodiscard]] static sam_scan from_file(const std::filesystem::path& path,
                                            bool lazy = false);

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
    // lazy-mode scan is still backed by the on-disk HDF5 file).
    [[nodiscard]] bool loaded() const noexcept { return lazy_ == nullptr; }
    // Alias of loaded(); backing() is "eager" or "lazy".
    [[nodiscard]] bool materialized() const noexcept { return loaded(); }
    [[nodiscard]] std::string backing() const {
        return lazy_ ? "lazy" : "eager";
    }

    // Materialize the signal data (no-op when already loaded or not in
    // lazy mode).  Every data accessor calls this implicitly.
    void load() const { ensure_loaded(); }

    // Row access that never materializes a lazy scan: owned copies are
    // returned, and only the blocks containing the requested rows are
    // decoded from disk.
    [[nodiscard]] std::vector<std::int8_t> read_row(size_t index) const;
    [[nodiscard]] array2d<std::int8_t> read_rows(size_t first,
                                                 size_t count) const;
    // Arbitrary rows (one row per index), owned copy, without materializing.
    [[nodiscard]] array2d<std::int8_t> read_selected(
        const std::vector<std::int64_t>& indices) const;
    // Number of data blocks decoded from disk so far (0 when materialized).
    [[nodiscard]] size_t blocks_read() const noexcept;

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
    // Sampling rate in MHz.
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

    // Time index in ns; with a scan index, uses that scan's
    // start when the handler carries per-scan starts (gated data).
    [[nodiscard]] std::vector<double> time(std::optional<size_t> index = {}) const;
    // Sample spacing in ns (1e3 / samplerate).
    [[nodiscard]] double samplespacing() const noexcept {
        return 1.0 / samplerate() * 1e3;
    }
    // Shared (relative) time axis; identical to time() without an index.
    [[nodiscard]] std::vector<double> relative_time() const;
    // Absolute time axis of one scan (offset by its start when present);
    // identical to time(index).  Throws for an out-of-range index when the
    // handler carries starts.
    [[nodiscard]] std::vector<double> absolute_time(size_t index) const;
    // Sample index nearest to time_ns on the handler's time scale
    // (round-to-nearest-even, the same rule as time_range_select).  For gated
    // scans this is relative to the shared tzero; per-scan origins are
    // handled by absolute_time().
    [[nodiscard]] std::int64_t sample_index(double time_ns) const;
    // Time in ns of a sample index on the handler's time scale.
    [[nodiscard]] double sample_time(std::int64_t index) const;
    // Hash of the header metadata.
    [[nodiscard]] size_t header_hash() const { return header_.hash(); }

    // reductions

    // Reduce every scan to a single value, reshaped to (nlines, cols).
    // dtypes: int8 ('max', 'absmax'; absmax saturates abs(-128) to 127),
    // float32 ('power', sum of squared samples).
    [[nodiscard]] array2d<std::int8_t> image_max() const;
    [[nodiscard]] array2d<std::int8_t> image_absmax() const;
    [[nodiscard]] array2d<float> image_power() const;

    // Data in [-1, 127/128] as float32.
    [[nodiscard]] array2d<float> normalized_data() const;

    // Conventional time of flight: the sample of the analytic-envelope peak
    // (Hilbert magnitude) of every A-scan within the [start, end) window,
    // expressed in ns on the handler's time scale (tzero + sample *
    // samplespacing).  With sub_sample the peak position is refined by a
    // parabolic fit through the envelope maximum and its neighbours.
    // Windows without an envelope peak (silent) return no_tof (-1.0).
    [[nodiscard]] array2d<float> tof(std::int64_t start = 0,
                                     std::int64_t end = 0,
                                     bool sub_sample = true) const;

    // Pulse-echo thickness map from tof(): tof_ns * 1e-9 * v / 2.  Returns
    // meters (sound_speed_m_s in m/s); windows without an envelope peak
    // keep the no_tof sentinel (-1.0).
    [[nodiscard]] array2d<float> thickness(double sound_speed_m_s,
                                           std::int64_t start = 0,
                                           std::int64_t end = 0,
                                           bool sub_sample = true) const;

    // Layered gating (XGate): pick a start per A-scan, then reduce n_gates
    // consecutive non-overlapping windows of gate_ns each to one scalar.
    // pick:
    //   "tof"       - analytic-envelope peak (rounded to a sample),
    //   "threshold" - first sample with value > threshold * full_scale
    //                 (positive samples only),
    //   "none"      - start at sample 0.
    // mode: "max", "absmax" (max |value|) or "power" (sum of squares).
    // gate_ns is the window length in ns.  Scans whose pick fails
    // (silent / no crossing) keep zero values and start -1; gates that do
    // not fit within the scan are zero.
    [[nodiscard]] xgate_result xgate(double gate_ns, size_t n_gates = 50,
                                     const std::string& pick = "tof",
                                     double threshold = -1.0,
                                     const std::string& mode = "max") const;

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

    // Sample-index range [start_idx, end_idx); adjusts scanlen and preserves
    // per-scan alignment: valid starts advance by start_idx so absolute times
    // are unchanged, and a uniform valid starts vector is folded into tzero.
    // Unaligned scans shift tzero instead.  Out-of-range indices are clamped.
    [[nodiscard]] sam_scan index_range_select(std::int64_t start_idx,
                                              std::int64_t end_idx) const;
    void index_range_select_ip(std::int64_t start_idx, std::int64_t end_idx);

    // Time range in ns; adjusts tzero and scanlen.  Per-scan starts
    // are preserved and advanced the same way as index_range_select (the
    // range is measured from the shared tzero, i.e. relative time).
    [[nodiscard]] sam_scan time_range_select(double start_time,
                                             double end_time) const;
    void time_range_select_ip(double start_time, double end_time);

    // Threshold-based gating of every scan; returns a scan with starts
    // accumulated on top of existing ones.
    [[nodiscard]] sam_scan zgate(double threshold = 0.2, std::int64_t length = 2000) const;
    void zgate_ip(double threshold, std::int64_t length);

    // Public manual alignment (also used by zgate): `new_starts` are
    // accumulated on top of existing starts (or set directly when none
    // exist); scans with a -1 start are zero-filled.  `new_scanlen` is the
    // length of the already-extracted windows.
    void align_manual(const std::vector<std::int32_t>& new_starts,
                      std::int64_t new_scanlen);

    // Align every A-scan to the reference scan by integer cross-correlation
    // lag within +-max_shift samples (0 = scanlen/4).  The sample data is
    // shifted so features coincide with the reference; `starts`, when
    // present, advance by the applied shift (clamped at 0) so absolute
    // feature times are preserved.  The reference row is left unchanged.
    void align_xcorr(size_t reference = 0, std::int64_t max_shift = 0);
    [[nodiscard]] sam_scan aligned_xcorr(size_t reference = 0,
                                         std::int64_t max_shift = 0) const;

    // Classic ToF alignment: shift every A-scan so the echo picked by its
    // analytic-envelope peak lands at the reference scan's peak.  The ToF
    // gate is [start_ns, start_ns + gate_ns) on the time axis; gate_ns is
    // required and given in ns.  Integer sample shifts are applied
    // to the data and existing `starts` advance by the applied shift
    // (clamped at 0).  Scans without an envelope peak in the gate are left
    // unchanged.  Throws when the reference scan has no peak in the gate.
    void align_tof(double gate_ns, size_t reference = 0,
                   double start_ns = 0.0);
    [[nodiscard]] sam_scan aligned_tof(double gate_ns, size_t reference = 0,
                                       double start_ns = 0.0) const;

    // spectral

    // One-sided STFT of every scan (fs = samplerate in Hz); t aligned to
    // the handler's time scale in seconds.  With f_max > 0 only bins in
    // [f_min, f_max] Hz are returned (f_max = 0 means Nyquist).
    [[nodiscard]] signal::stft_result compute_stft(size_t nperseg = 256,
                                                   size_t noverlap = 128,
                                                   double f_min = 0.0,
                                                   double f_max = 0.0) const;
    // Welch PSD (scipy.signal.welch, detrend=False); f in Hz, f_min/f_max
    // in Hz.
    [[nodiscard]] signal::psd_result psd(size_t nperseg = 256,
                                         size_t noverlap = 128,
                                         double f_min = 0.0,
                                         double f_max = 0.0) const;
    // Per-frame power spectral density (hann window, density scaling);
    // f in Hz, t in seconds, f_min/f_max in Hz.
    [[nodiscard]] signal::spectrogram_result power_spectrogram(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0) const;
    // Per-scan STFT magnitude peaks: the frequency (Hz) of the strongest
    // max-over-frames bin and the time (seconds) of the strongest
    // max-over-frequencies frame, reshaped to (nlines, cols).  Band-limited
    // like compute_stft().
    [[nodiscard]] array2d<float> stft_peak_frequency(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0) const;
    [[nodiscard]] array2d<float> stft_peak_time(
        size_t nperseg = 256, size_t noverlap = 128, double f_min = 0.0,
        double f_max = 0.0) const;
    // One-sided FFT magnitude of every A-scan (numpy rfft parity); the
    // frequencies are in Hz.
    [[nodiscard]] signal::spectrum_result spectrum() const;

    // iteration

    [[nodiscard]] std::span<const std::int8_t> operator[](size_t index) const {
        return scan(index);
    }

private:
    void ensure_loaded() const;

    // Shared implementation of index_range_select / time_range_select.
    [[nodiscard]] sam_scan slice_index_range(std::int64_t start_idx,
                                             std::int64_t end_idx) const;

    mutable array2d<std::int8_t> data_;
    sam_header header_;
    sam_labels labels_;
    std::optional<std::vector<std::int32_t>> starts_;
    std::string path_;
    mutable std::unique_ptr<io::h5sam_lazy_state> lazy_;
};

} // namespace samcore

#include <samcore/sam_scan.hpp>

#include "io/h5_lazy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef SAMCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace samcore {

namespace {

std::string extension_lower(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

// Swap two equal-size contiguous byte ranges through a scratch buffer.
void swap_blocks(std::int8_t* a, std::int8_t* b, size_t bytes,
                 std::int8_t* scratch) {
    std::memcpy(scratch, a, bytes);
    std::memcpy(a, b, bytes);
    std::memcpy(b, scratch, bytes);
}

// First sample whose |value| reaches thresh_val; -1 when none.  numpy
// parity: np.abs() on int8 wraps |-128| to -128, so a -128 sample never
// counts as a threshold crossing.
std::int64_t first_crossing(std::span<const std::int8_t> scan,
                            double thresh_val) {
    for (size_t k = 0; k < scan.size(); ++k) {
        const auto a = static_cast<std::int8_t>(std::abs(scan[k]));
        if (static_cast<double>(a) >= thresh_val) {
            return static_cast<std::int64_t>(k);
        }
    }
    return -1;
}

// Fold a uniform, fully valid starts vector into tzero and clear it.
bool collapse_starts(sam_header& header,
                     std::optional<std::vector<std::int32_t>>& starts) {
    if (!starts.has_value() || starts->empty()) return false;
    const std::int32_t first = starts->front();
    if (first < 0) return false;
    for (auto s : *starts) {
        if (s != first) return false;
    }
    header.tzero += static_cast<std::int64_t>(std::nearbyint(
        static_cast<double>(first) / header.samplerate * 1e3));
    starts = std::nullopt;
    return true;
}

// int8 signal cube -> float32 staging buffer (used by the spectral APIs).
array2d<float> to_float(const array2d<std::int8_t>& data) {
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            out[i][j] = static_cast<float>(data[i][j]);
        }
    }
    return out;
}

// Per-signal values -> (nlines, cols) image.
array2d<float> reshape_signals(const std::vector<float>& values,
                               size_t nlines, size_t ncols) {
    array2d<float> out(nlines, ncols);
    for (size_t i = 0; i < values.size(); ++i) {
        out[i / ncols][i % ncols] = values[i];
    }
    return out;
}

// Rebuild labels/starts so that output index o takes the value at
// in_index(o), the inverse permutation of the data movement.
template <class F>
void permute_labels_starts(sam_labels& labels,
                           std::optional<std::vector<std::int32_t>>& starts,
                           size_t total, const F& in_index) {
    if (labels.size() == total) {
        const auto& old = labels.labels();
        std::vector<std::int8_t> out(total);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (total > 8) schedule(static)
#endif
        for (size_t o = 0; o < total; ++o) out[o] = old[in_index(o)];
        labels.set_labels(std::move(out));
    }
    if (starts.has_value() && starts->size() == total) {
        const auto& old = *starts;
        std::vector<std::int32_t> out(total);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (total > 8) schedule(static)
#endif
        for (size_t o = 0; o < total; ++o) out[o] = old[in_index(o)];
        starts = std::move(out);
    }
}

// In-place permutation of whole signal rows along the cycles of in_index
// (one scratch row, no full-size temporary).
template <class F>
void rotate_rows_inplace(array2d<std::int8_t>& data, size_t total, size_t sl,
                         const F& in_index) {
    std::vector<std::uint8_t> visited(total, 0);
    std::vector<std::int8_t> tmp(sl);
    for (size_t s = 0; s < total; ++s) {
        if (visited[s]) continue;
        std::memcpy(tmp.data(), data[s].data(), sl);
        size_t cur = s;
        while (true) {
            const size_t prev = in_index(cur);
            if (prev == s) break;
            std::memcpy(data[cur].data(), data[prev].data(), sl);
            visited[cur] = 1;
            cur = prev;
        }
        std::memcpy(data[cur].data(), tmp.data(), sl);
        visited[cur] = 1;
        visited[s] = 1;
    }
}

} // namespace

sam_scan sam_scan::from_file(const std::filesystem::path& path, bool lazy) {
    const std::string ext = extension_lower(path);
    sam_scan scan;
    if (ext == ".h5sam") {
        if (lazy) {
            auto res = io::read_h5sam_lazy(path);
            scan.header_ = std::move(res.header);
            scan.labels_ = std::move(res.labels);
            scan.starts_ = std::move(res.starts);
            scan.lazy_ = std::move(res.data);
        } else {
            auto res = io::read_h5sam(path);
            scan = from_data(std::move(res.data), std::move(res.header),
                             std::move(res.starts), std::move(res.labels));
        }
    } else {
        throw std::invalid_argument("File format not supported: " +
                                    path.string());
    }
    scan.path_ = path.string();
    return scan;
}

sam_scan sam_scan::from_data(array2d<std::int8_t> data, sam_header header,
                             std::optional<std::vector<std::int32_t>> starts,
                             std::optional<sam_labels> labels) {
    if (data.rows() != static_cast<size_t>(header.nlines * header.scanspline) ||
        data.cols() != static_cast<size_t>(header.scanlen)) {
        throw std::invalid_argument("Data mismatch with header.");
    }
    sam_scan scan;
    scan.data_ = std::move(data);
    // The handler always owns its data: materialize a non-owning input view
    // so copy()/to_h5sam() cannot alias or silently drop borrowed memory.
    if (scan.data_.is_view()) {
        scan.data_ = array2d<std::int8_t>(scan.data_);
    }
    scan.header_ = std::move(header);
    scan.path_ = {};
    scan.starts_ = std::move(starts);
    if (labels) {
        scan.labels_ = std::move(*labels);
    } else {
        scan.labels_ = sam_labels::create_unlabeled(scan.data_.rows());
    }
    return scan;
}

std::span<const std::int8_t> sam_scan::scan(size_t index) const {
    ensure_loaded();
    return data_[index];
}

std::span<const std::int8_t> sam_scan::scan(std::int64_t line,
                                            std::int64_t col) const {
    ensure_loaded();
    return data_[static_cast<size_t>(line * cols() + col)];
}

void sam_scan::to_h5sam(const std::filesystem::path& path) const {
    ensure_loaded();
    const std::string ext = extension_lower(path);
    if (ext != ".h5sam") {
        throw std::invalid_argument("Output path must end with .h5sam");
    }
    io::write_h5sam(path, data_, header_, labels_, starts_);
}

std::vector<double> sam_scan::time(std::optional<size_t> index) const {
    if (starts_.has_value() && index.has_value() &&
        (*starts_)[*index] > 0) {
        const auto start = static_cast<std::int64_t>((*starts_)[*index]);
        return header_.time(start, start + header_.scanlen);
    }
    return header_.time();
}

std::vector<double> sam_scan::relative_time() const {
    return header_.time();
}

std::vector<double> sam_scan::absolute_time(size_t index) const {
    if (starts_.has_value() && index >= starts_->size()) {
        throw std::out_of_range("absolute_time: scan index out of range.");
    }
    return time(index);
}

std::int64_t sam_scan::sample_index(double time_ns) const {
    return static_cast<std::int64_t>(
        std::nearbyint((time_ns - header_.tzero) / samplespacing()));
}

double sam_scan::sample_time(std::int64_t index) const {
    return header_.tzero + static_cast<double>(index) * samplespacing();
}

namespace {

// Reduce every scan to a single value and reshape to (nlines, cols).
template <typename T, typename F>
array2d<T> reduce_image(const array2d<std::int8_t>& data, size_t nlines,
                        size_t ncols, const F& reduce) {
    const size_t n = data.rows();
    array2d<T> img(nlines, ncols);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        img[i / ncols][i % ncols] = reduce(data[i]);
    }
    return img;
}

// Same, but streaming row blocks from a lazy (paged) reader: the full cube is
// never materialized.
template <typename T, typename F>
array2d<T> reduce_image_paged(io::paged_reader<std::int8_t>& reader,
                              size_t nlines, size_t ncols, const F& reduce) {
    const size_t n = reader.rows();
    const size_t block = std::max<size_t>(1, reader.block_rows());
    array2d<T> img(nlines, ncols);
    std::vector<std::int8_t> buf(block * reader.cols());
    for (size_t first = 0; first < n; first += block) {
        const size_t count = std::min(block, n - first);
        reader.read_rows(first, count, buf.data());
        for (size_t r = 0; r < count; ++r) {
            const size_t i = first + r;
            img[i / ncols][i % ncols] = reduce(std::span<const std::int8_t>(
                buf.data() + r * reader.cols(), reader.cols()));
        }
    }
    return img;
}

std::int8_t row_max(std::span<const std::int8_t> row) {
    return *std::max_element(row.begin(), row.end());
}

// absmax = max(|min|, |max|), saturated to int8 (abs(-128) -> 127).
std::int8_t row_absmax(std::span<const std::int8_t> row) {
    std::int8_t mx = row[0], mn = row[0];
    for (auto v : row) {
        mx = std::max(mx, v);
        mn = std::min(mn, v);
    }
    const int am = std::max(std::abs(static_cast<int>(mx)),
                            std::abs(static_cast<int>(mn)));
    return static_cast<std::int8_t>(std::min(am, 127));
}

// power: sum of squares, float32.
float row_power(std::span<const std::int8_t> row) {
    double acc = 0.0;
    for (auto v : row) acc += static_cast<double>(v) * v;
    return static_cast<float>(acc);
}

} // namespace

array2d<std::int8_t> sam_scan::image_max() const {
    const size_t nl = static_cast<size_t>(header_.nlines);
    const size_t nc = static_cast<size_t>(header_.scanspline);
    if (lazy_) return reduce_image_paged<std::int8_t>(lazy_->reader, nl, nc,
                                                      row_max);
    ensure_loaded();
    return reduce_image<std::int8_t>(data_, nl, nc, row_max);
}

array2d<std::int8_t> sam_scan::image_absmax() const {
    const size_t nl = static_cast<size_t>(header_.nlines);
    const size_t nc = static_cast<size_t>(header_.scanspline);
    if (lazy_) return reduce_image_paged<std::int8_t>(lazy_->reader, nl, nc,
                                                      row_absmax);
    ensure_loaded();
    return reduce_image<std::int8_t>(data_, nl, nc, row_absmax);
}

array2d<float> sam_scan::image_power() const {
    const size_t nl = static_cast<size_t>(header_.nlines);
    const size_t nc = static_cast<size_t>(header_.scanspline);
    if (lazy_) return reduce_image_paged<float>(lazy_->reader, nl, nc,
                                                row_power);
    ensure_loaded();
    return reduce_image<float>(data_, nl, nc, row_power);
}

array2d<float> sam_scan::normalized_data() const {
    ensure_loaded();
    array2d<float> out(data_.rows(), data_.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data_.rows() > 8)
#endif
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            out[i][j] = static_cast<float>(data_[i][j]) / 128.0f;
        }
    }
    return out;
}

array2d<float> sam_scan::tof(std::int64_t start, std::int64_t end,
                             bool sub_sample) const {
    ensure_loaded();
    const std::int64_t n = scanlen();
    if (end == 0) end = n;
    if (start < 0 || end > n || start >= end) {
        throw std::invalid_argument(
            "tof: invalid sample range [" + std::to_string(start) + ", " +
            std::to_string(end) + ") for scan length " + std::to_string(n) +
            ".");
    }
    const double spacing = samplespacing();
    const size_t nc = static_cast<size_t>(cols());
    const size_t count = data_.rows();
    array2d<float> out(static_cast<size_t>(nlines()), nc);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (count > 8)
#endif
    for (size_t i = 0; i < count; ++i) {
        // Conventional pick: maximum of the analytic envelope (Hilbert
        // magnitude) within the gate.  No amplitude threshold is involved.
        const std::vector<double> row(data_[i].begin(), data_[i].end());
        const std::vector<double> env = signal::hilbert_envelope(row);
        std::int64_t peak = -1;
        double best = 0.0;
        for (std::int64_t k = start; k < end; ++k) {
            const double v = env[static_cast<size_t>(k)];
            if (v > best) {
                best = v;
                peak = k;
            }
        }
        if (peak >= 0 && best > 0.0) {
            double pos = static_cast<double>(peak);
            if (sub_sample && peak > start && peak + 1 < end) {
                const double y0 = env[static_cast<size_t>(peak - 1)];
                const double y2 = env[static_cast<size_t>(peak + 1)];
                const double denom = y0 - 2.0 * best + y2;
                if (denom != 0.0) {
                    pos += std::clamp(0.5 * (y0 - y2) / denom, -0.5, 0.5);
                }
            }
            out[i / nc][i % nc] =
                static_cast<float>(header_.tzero + pos * spacing);
        } else {
            // Silent window: documented no-echo sentinel (see no_tof).
            out[i / nc][i % nc] = no_tof;
        }
    }
    return out;
}

array2d<float> sam_scan::thickness(double sound_speed_m_s, std::int64_t start,
                                   std::int64_t end, bool sub_sample) const {
    if (sound_speed_m_s <= 0.0) {
        throw std::invalid_argument(
            "thickness: sound speed must be positive.");
    }
    array2d<float> out = tof(start, end, sub_sample);
    const float scale = static_cast<float>(sound_speed_m_s * 1e-9 / 2.0);
    for (auto& v : out.flat()) {
        // The no-echo sentinel stays -1; real ToF values are scaled to m.
        if (v != no_tof) v *= scale;
    }
    return out;
}

xgate_result sam_scan::xgate(double gate_ns, size_t n_gates,
                             const std::string& pick, double threshold,
                             const std::string& mode) const {
    ensure_loaded();
    if (!(gate_ns > 0.0)) {
        throw std::invalid_argument("xgate: gate_ns must be positive.");
    }
    if (pick != "tof" && pick != "threshold" && pick != "none") {
        throw std::invalid_argument(
            "xgate: pick must be 'tof', 'threshold' or 'none'.");
    }
    if (mode != "max" && mode != "absmax" && mode != "power") {
        throw std::invalid_argument(
            "xgate: mode must be 'max', 'absmax' or 'power'.");
    }
    if (pick == "threshold" && (threshold < 0.0 || threshold > 1.0)) {
        throw std::invalid_argument(
            "xgate: threshold must be in [0, 1] for pick='threshold'.");
    }
    const double spacing = samplespacing();
    const std::int64_t gate_samples = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(std::llround(gate_ns / spacing)));
    const size_t n = data_.rows();
    const size_t nc = static_cast<size_t>(cols());
    const std::int64_t sl = scanlen();

    xgate_result result;
    result.values = array3d<float>(static_cast<size_t>(nlines()), nc, n_gates);
    result.starts.assign(n, -1);

    // The envelope is expensive, so compute the ToF picks once up front.
    std::vector<double> tof_pos;
    if (pick == "tof") {
        const array2d<float> t = tof(0, 0, true); // ns; no_tof when silent
        tof_pos.assign(n, -1.0);
        for (size_t i = 0; i < n; ++i) {
            const float v = t[i / nc][i % nc];
            if (v != no_tof) {
                tof_pos[i] = (static_cast<double>(v) - header_.tzero) / spacing;
            }
        }
    }

#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        std::int64_t start = -1;
        if (pick == "tof") {
            if (tof_pos[i] >= 0.0) {
                start = static_cast<std::int64_t>(std::llround(tof_pos[i]));
            }
        } else if (pick == "threshold") {
            const double thresh = threshold * full_scale;
            for (std::int64_t j = 0; j < sl; ++j) {
                if (static_cast<double>(data_[i][static_cast<size_t>(j)]) >
                    thresh) {
                    start = j;
                    break;
                }
            }
        } else {
            start = 0;
        }
        if (start < 0 || start >= sl) {
            continue; // failed pick: zero values, start stays -1
        }
        result.starts[i] = static_cast<std::int32_t>(start);
        const size_t max_gates =
            static_cast<size_t>((sl - start) / gate_samples);
        const size_t actual = std::min(n_gates, max_gates);
        auto plane = result.values.plane(i / nc);
        const size_t base = (i % nc) * n_gates;
        for (size_t g = 0; g < actual; ++g) {
            const auto window = data_[i].subspan(
                static_cast<size_t>(start) + g * static_cast<size_t>(gate_samples),
                static_cast<size_t>(gate_samples));
            float v = 0.0f;
            if (mode == "max") {
                v = static_cast<float>(*std::max_element(window.begin(),
                                                         window.end()));
            } else if (mode == "absmax") {
                int best = 0;
                for (auto s : window) {
                    best = std::max(best, std::abs(static_cast<int>(s)));
                }
                v = static_cast<float>(best);
            } else { // power
                double acc = 0.0;
                for (auto s : window) {
                    acc += static_cast<double>(s) * s;
                }
                v = static_cast<float>(acc);
            }
            plane[base + g] = v;
        }
    }
    return result;
}

void sam_scan::set_labels(sam_labels labels) {
    ensure_loaded();
    if (labels.size() != data_.rows()) {
        throw std::invalid_argument(
            "Length of labels must match the number of scans (nlines * cols).");
    }
    if (!labels.label_names().empty() &&
        labels.label_names().size() < static_cast<size_t>(labels.max_label()) + 1) {
        throw std::invalid_argument(
            "Length of label_names must be greater than the maximum label index.");
    }
    labels.verify_integrity();
    labels_ = std::move(labels);
}

void sam_scan::set_labels(std::vector<std::int8_t> labels,
                          std::vector<std::string> label_names) {
    set_labels(sam_labels(std::move(labels), std::move(label_names)));
}

void sam_scan::downsample(size_t factor, downsample_mode mode) {
    ensure_loaded();
    if (factor < 1) {
        throw std::invalid_argument("Downsampling factor must be at least 1.");
    }
    if (factor > static_cast<size_t>(header_.scanlen)) {
        throw std::invalid_argument(
            "Downsampling factor cannot be greater than scan length.");
    }
    const size_t new_len = static_cast<size_t>(header_.scanlen) / factor;
    const size_t trimmed_len = new_len * factor;
    const size_t num_signals = data_.rows();

    array2d<std::int8_t> downsampled(num_signals, new_len);
    if (mode == downsample_mode::decimate) {
        if (factor < 2) {
            for (size_t s = 0; s < num_signals; ++s) {
                std::memcpy(downsampled[s].data(), data_[s].data(),
                            trimmed_len);
            }
        } else {
            // Design the anti-aliasing filter once, then filter every
            // signal in one fused cascade pass (int8 input, no staging
            // buffer).
            const std::vector<signal::sos> sections =
                signal::decimate_sos(factor);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (num_signals > 8)
#endif
            for (size_t s = 0; s < num_signals; ++s) {
                const auto dec = signal::decimate(
                    data_[s].first(trimmed_len), factor, sections);
                for (size_t i = 0; i < new_len; ++i) {
                    const double v = std::clamp(dec[i], -128.0, 127.0);
                    downsampled[s][i] = static_cast<std::int8_t>(v); // truncation
                }
            }
        }
    } else if (mode == downsample_mode::mean) {
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (num_signals > 8)
#endif
        for (size_t s = 0; s < num_signals; ++s) {
            for (size_t i = 0; i < new_len; ++i) {
                double acc = 0.0;
                for (size_t k = 0; k < factor; ++k) {
                    acc += data_[s][i * factor + k];
                }
                downsampled[s][i] = static_cast<std::int8_t>(acc / static_cast<double>(factor));
            }
        }
    } else if (mode == downsample_mode::median) {
        std::vector<double> seg(factor);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (num_signals > 8) firstprivate(seg)
#endif
        for (size_t s = 0; s < num_signals; ++s) {
            for (size_t i = 0; i < new_len; ++i) {
                for (size_t k = 0; k < factor; ++k) {
                    seg[k] = data_[s][i * factor + k];
                }
                const size_t half = factor / 2;
                double median;
                if (factor % 2 == 1) {
                    // odd: median is the middle order statistic
                    std::nth_element(seg.begin(),
                                     seg.begin() + static_cast<std::ptrdiff_t>(half),
                                     seg.end());
                    median = seg[half];
                } else {
                    // even segments: average of the two middle order statistics
                    std::nth_element(seg.begin(),
                                     seg.begin() + static_cast<std::ptrdiff_t>(half - 1),
                                     seg.end());
                    const double second =
                        *std::min_element(seg.begin() + static_cast<std::ptrdiff_t>(half),
                                          seg.end());
                    median = (seg[half - 1] + second) / 2.0;
                }
                downsampled[s][i] = static_cast<std::int8_t>(median);
            }
        }
    } else { // sample
        for (size_t s = 0; s < num_signals; ++s) {
            for (size_t i = 0; i < new_len; ++i) {
                downsampled[s][i] = data_[s][i * factor];
            }
        }
    }

    data_ = std::move(downsampled);
    header_.samplerate /= static_cast<double>(factor);
    header_.scanlen = static_cast<std::int64_t>(new_len);
    header_.downsample_factor *= static_cast<std::int64_t>(factor);
    if (starts_.has_value()) {
        for (auto& s : *starts_) {
            s = s < 0 ? -1 : s / static_cast<std::int32_t>(factor);
        }
    }
}

sam_scan sam_scan::downsampled(size_t factor, downsample_mode mode) const {
    sam_scan h = copy();
    h.downsample(factor, mode);
    return h;
}

void sam_scan::rotate(int degrees) {
    ensure_loaded();
    int d = degrees % 360;
    if (d < 0) d += 360;
    if (d != 0 && d != 90 && d != 180 && d != 270) {
        throw std::invalid_argument("degrees must be 90, 180, or 270.");
    }
    if (d == 0) return;

    const size_t nl = static_cast<size_t>(nlines());
    const size_t nc = static_cast<size_t>(cols());
    const size_t total = nl * nc;
    const size_t sl = data_.cols();

    if (d == 180) {
        // In place: output signal o takes input signal total-1-o (samples
        // within each signal stay in order).
        const size_t pairs = total / 2;
        if (pairs > 0) {
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel if (pairs > 8)
#endif
            {
                std::vector<std::int8_t> scratch(sl);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp for schedule(static)
#endif
                for (size_t p = 0; p < pairs; ++p) {
                    swap_blocks(data_[p].data(), data_[total - 1 - p].data(),
                                sl, scratch.data());
                }
            }
        }
        permute_labels_starts(labels_, starts_, total, [nl, nc](size_t o) {
            const size_t i = o / nc;
            const size_t j = o % nc;
            return (nl - 1 - i) * nc + (nc - 1 - j);
        });
        return; // shape unchanged
    }

    const bool cw = (d == 90);
    // Output grid is (nc, nl); output flat index o = i*nl + j takes the
    // signal at input flat index in_index(o).
    auto in_index = [nl, nc, cw](size_t o) -> size_t {
        const size_t i = o / nl;
        const size_t j = o % nl;
        return cw ? (nl - 1 - j) * nc + i : j * nc + (nc - 1 - i);
    };
    rotate_rows_inplace(data_, total, sl, in_index);
    permute_labels_starts(labels_, starts_, total, in_index);

    header_.nlines = static_cast<std::int64_t>(nc);
    header_.scanspline = static_cast<std::int64_t>(nl);
}

sam_scan sam_scan::rotated(int degrees) const {
    sam_scan h = copy();
    h.rotate(degrees);
    return h;
}

void sam_scan::mirror(mirror_axis axis) {
    ensure_loaded();
    const size_t nl = static_cast<size_t>(nlines());
    const size_t nc = static_cast<size_t>(cols());
    const size_t total = nl * nc;
    const size_t sl = data_.cols();

    if (axis == mirror_axis::x) {
        // Output flat index o = i*nc + j takes input i*nc + (nc-1-j):
        // reverse the signal order within each line.
        const size_t pairs_per_line = nc / 2;
        const size_t pairs = nl * pairs_per_line;
        if (pairs > 0) {
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel if (pairs > 8)
#endif
            {
                std::vector<std::int8_t> scratch(sl);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp for schedule(static)
#endif
                for (size_t p = 0; p < pairs; ++p) {
                    const size_t line = p / pairs_per_line;
                    const size_t k = p % pairs_per_line;
                    swap_blocks(data_[line * nc + k].data(),
                                data_[line * nc + (nc - 1 - k)].data(), sl,
                                scratch.data());
                }
            }
        }
        permute_labels_starts(labels_, starts_, total, [nc](size_t o) {
            const size_t i = o / nc;
            const size_t j = o % nc;
            return i * nc + (nc - 1 - j);
        });
    } else {
        // Reverse the line order: swap whole lines (nc signals, one
        // contiguous block each) keeping the column order within a line.
        const size_t pairs = nl / 2;
        if (pairs > 0) {
            const size_t line_bytes = nc * sl;
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel if (pairs > 8)
#endif
            {
                std::vector<std::int8_t> scratch(line_bytes);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp for schedule(static)
#endif
                for (size_t p = 0; p < pairs; ++p) {
                    swap_blocks(data_[p * nc].data(),
                                data_[(nl - 1 - p) * nc].data(), line_bytes,
                                scratch.data());
                }
            }
        }
        permute_labels_starts(labels_, starts_, total, [nl, nc](size_t o) {
            const size_t i = o / nc;
            const size_t j = o % nc;
            return (nl - 1 - i) * nc + j;
        });
    }
}

sam_scan sam_scan::mirrored(mirror_axis axis) const {
    sam_scan h = copy();
    h.mirror(axis);
    return h;
}

sam_scan sam_scan::rectangle_select(std::int64_t line_start,
                                    std::int64_t line_end,
                                    std::int64_t col_start,
                                    std::int64_t col_end) const {
    ensure_loaded();
    if (!(0 <= line_start && line_start < line_end && line_end <= nlines())) {
        throw std::invalid_argument("Line range out of bounds for nlines=" +
                                    std::to_string(nlines()));
    }
    if (!(0 <= col_start && col_start < col_end && col_end <= cols())) {
        throw std::invalid_argument("Column range out of bounds for cols=" +
                                    std::to_string(cols()));
    }
    const std::int64_t new_nlines = line_end - line_start;
    const std::int64_t new_cols = col_end - col_start;
    const size_t count = static_cast<size_t>(new_nlines * new_cols);

    const size_t sl = static_cast<size_t>(scanlen());
    const size_t ncols = static_cast<size_t>(new_cols);
    const size_t lines = static_cast<size_t>(new_nlines);
    const size_t line_bytes = ncols * sl;
    const size_t in_cols = static_cast<size_t>(cols());

    // The selected columns of one line are consecutive signals, so each
    // line is a single contiguous memcpy; parallelize over lines.
    array2d<std::int8_t> new_data(count, sl);
    std::vector<size_t> idx(count);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (lines > 1) schedule(static)
#endif
    for (size_t li = 0; li < lines; ++li) {
        const size_t first =
            (static_cast<size_t>(line_start) + li) * in_cols +
            static_cast<size_t>(col_start);
        std::memcpy(new_data.data() + li * line_bytes,
                    data_.data() + first * sl, line_bytes);
        size_t* out_idx = idx.data() + li * ncols;
        for (size_t c = 0; c < ncols; ++c) out_idx[c] = first + c;
    }

    sam_header new_header = header_;
    new_header.nlines = new_nlines;
    new_header.scanspline = new_cols;

    std::optional<std::vector<std::int32_t>> new_starts;
    if (starts_.has_value()) {
        std::vector<std::int32_t> s(count);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (count > 8) schedule(static)
#endif
        for (size_t r = 0; r < count; ++r) s[r] = (*starts_)[idx[r]];
        new_starts = std::move(s);
    }

    return from_data(std::move(new_data), std::move(new_header),
                     std::move(new_starts), labels_.take(idx));
}

void sam_scan::rectangle_select_ip(std::int64_t line_start,
                                   std::int64_t line_end,
                                   std::int64_t col_start,
                                   std::int64_t col_end) {
    *this = rectangle_select(line_start, line_end, col_start, col_end);
}

sam_scan sam_scan::slice_index_range(std::int64_t start_idx,
                                     std::int64_t end_idx) const {
    ensure_loaded();
    if (start_idx < 0) start_idx = 0;
    if (end_idx > scanlen()) end_idx = scanlen();
    if (start_idx >= end_idx) {
        throw std::invalid_argument(
            "Invalid sample range [" + std::to_string(start_idx) + ", " +
            std::to_string(end_idx) + ") for scan length " +
            std::to_string(scanlen()) + ".");
    }
    const std::int64_t new_scanlen = end_idx - start_idx;
    array2d<std::int8_t> new_data(data_.rows(),
                                  static_cast<size_t>(new_scanlen));
    for (size_t s = 0; s < data_.rows(); ++s) {
        std::memcpy(new_data[s].data(),
                    data_[s].data() + static_cast<std::ptrdiff_t>(start_idx),
                    static_cast<size_t>(new_scanlen));
    }
    sam_header new_header = header_;
    new_header.scanlen = new_scanlen;

    std::optional<std::vector<std::int32_t>> new_starts;
    if (starts_.has_value() && starts_->size() == data_.rows()) {
        // Per-scan alignment: the retained window starts start_idx samples
        // later in every scan, so valid starts advance by start_idx.
        new_starts = *starts_;
        for (auto& s : *new_starts) {
            if (s != -1) s += static_cast<std::int32_t>(start_idx);
        }
        collapse_starts(new_header, new_starts);
    } else {
        // Unaligned: the shared origin moves with the slice.
        new_header.tzero = static_cast<std::int64_t>(
            std::nearbyint(new_header.tzero + start_idx * samplespacing()));
    }
    return from_data(std::move(new_data), std::move(new_header),
                     std::move(new_starts), labels_);
}

sam_scan sam_scan::index_range_select(std::int64_t start_idx,
                                      std::int64_t end_idx) const {
    return slice_index_range(start_idx, end_idx);
}

void sam_scan::index_range_select_ip(std::int64_t start_idx,
                                     std::int64_t end_idx) {
    *this = slice_index_range(start_idx, end_idx);
}

sam_scan sam_scan::time_range_select(double start_time,
                                     double end_time) const {
    ensure_loaded();
    const double sp = samplespacing();
    std::int64_t start_idx =
        static_cast<std::int64_t>(std::nearbyint((start_time - header_.tzero) / sp));
    std::int64_t end_idx =
        static_cast<std::int64_t>(std::nearbyint((end_time - header_.tzero) / sp));
    if (start_idx < 0) start_idx = 0;
    if (end_idx > scanlen()) end_idx = scanlen();
    if (start_idx >= end_idx) {
        throw std::invalid_argument(
            "Invalid time range [" + std::to_string(start_time) + ", " +
            std::to_string(end_time) + "] ns");
    }
    return slice_index_range(start_idx, end_idx);
}

void sam_scan::time_range_select_ip(double start_time, double end_time) {
    *this = time_range_select(start_time, end_time);
}

sam_scan::sam_scan() = default;

sam_scan::sam_scan(const std::string& path, bool lazy) {
    *this = from_file(path, lazy);
}

sam_scan::~sam_scan() = default;
sam_scan::sam_scan(sam_scan&&) noexcept = default;
sam_scan& sam_scan::operator=(sam_scan&&) noexcept = default;

sam_scan::sam_scan(const sam_scan& o) {
    o.ensure_loaded();
    data_ = o.data_;
    header_ = o.header_;
    labels_ = o.labels_;
    starts_ = o.starts_;
    path_ = o.path_;
}

sam_scan& sam_scan::operator=(const sam_scan& o) {
    if (this == &o) return *this;
    o.ensure_loaded();
    data_ = o.data_;
    header_ = o.header_;
    labels_ = o.labels_;
    starts_ = o.starts_;
    path_ = o.path_;
    lazy_.reset();
    return *this;
}

size_t sam_scan::num_scans() const noexcept {
    return lazy_ ? lazy_->reader.rows() : data_.rows();
}

std::vector<std::int8_t> sam_scan::read_row(size_t index) const {
    if (index >= num_scans()) {
        throw std::out_of_range("read_row: scan index out of range.");
    }
    if (!lazy_) {
        const auto row = data_[index];
        return std::vector<std::int8_t>(row.begin(), row.end());
    }
    return lazy_->reader.read_row(index);
}

array2d<std::int8_t> sam_scan::read_rows(size_t first, size_t count) const {
    if (first + count > num_scans()) {
        throw std::out_of_range("read_rows: scan range out of bounds.");
    }
    if (!lazy_) {
        array2d<std::int8_t> out(count, data_.cols());
        for (size_t r = 0; r < count; ++r) {
            std::memcpy(out[r].data(), data_[first + r].data(), data_.cols());
        }
        return out;
    }
    array2d<std::int8_t> out(count, lazy_->reader.cols());
    if (count > 0) {
        lazy_->reader.read_rows(first, count, out.data());
    }
    return out;
}

size_t sam_scan::blocks_read() const noexcept {
    return lazy_ ? lazy_->reader.blocks_read() : 0;
}

array2d<std::int8_t> sam_scan::read_selected(
    const std::vector<std::int64_t>& indices) const {
    if (!lazy_) {
        array2d<std::int8_t> out(indices.size(), data_.cols());
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto row = static_cast<size_t>(indices[i]);
            if (row >= data_.rows()) {
                throw std::out_of_range("read_selected: row out of range.");
            }
            std::memcpy(out[i].data(), data_[row].data(), data_.cols());
        }
        return out;
    }
    array2d<std::int8_t> out(indices.size(), lazy_->reader.cols());
    if (!indices.empty()) {
        lazy_->reader.read_selected(indices, out.data());
    }
    return out;
}

void sam_scan::ensure_loaded() const {
    if (!lazy_) return;
    const size_t rows = lazy_->reader.rows();
    const size_t cols = lazy_->reader.cols();
    array2d<std::int8_t> loaded(rows, cols);
    if (rows > 0 && cols > 0) {
        lazy_->reader.read_rows(0, rows, loaded.data());
    }
    data_ = std::move(loaded);
    lazy_.reset(); // closes the file handle
}

void sam_scan::align_manual(const std::vector<std::int32_t>& new_starts,
                            std::int64_t new_scanlen) {
    ensure_loaded();
    if (new_scanlen <= 0) {
        throw std::invalid_argument(
            "align_manual: scanlen must be a positive integer.");
    }
    if (new_scanlen > scanlen()) {
        throw std::invalid_argument(
            "align_manual: scanlen cannot be greater than scan length.");
    }
    const size_t n = data_.rows();
    if (new_starts.size() != n) {
        throw std::invalid_argument(
            "align_manual: starts length must equal the number of scans.");
    }
    const std::int64_t max_start = scanlen() - new_scanlen;
    const bool has_starts = starts_.has_value() && starts_->size() == n;

    // Absolute origin (offset from tzero) of each output window; -1 marks an
    // unaligned scan.  `new_starts` are relative to the current window and
    // accumulate on top of any existing per-scan origin.
    std::vector<std::int32_t> origins(n, -1);
    for (size_t i = 0; i < n; ++i) {
        const std::int64_t rel = new_starts[i];
        if (rel == -1 || (has_starts && (*starts_)[i] == -1)) {
            continue;
        }
        if (rel < 0 || rel > max_start) {
            throw std::invalid_argument(
                "Start indices must be in the range [-1, " +
                std::to_string(max_start) + "] for scan length " +
                std::to_string(scanlen()) + " and new scan length " +
                std::to_string(new_scanlen) + ".");
        }
        const std::int64_t base = has_starts ? (*starts_)[i] : 0;
        origins[i] = static_cast<std::int32_t>(base + rel);
    }

    // Extract the aligned windows; unaligned scans are zero-filled.
    array2d<std::int8_t> out(n, static_cast<size_t>(new_scanlen));
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        auto dst = out[i];
        if (origins[i] == -1) {
            std::fill(dst.begin(), dst.end(), std::int8_t{0});
        } else {
            const std::int64_t rel = new_starts[i];
            std::copy(data_[i].begin() + rel,
                      data_[i].begin() + rel + new_scanlen, dst.begin());
        }
    }

    data_ = std::move(out);
    starts_ = std::move(origins);
    header_.scanlen = new_scanlen;
    collapse_starts(header_, starts_);
}

void sam_scan::align_zgate(double threshold, std::int64_t length) {
    ensure_loaded();
    if (length <= 0) {
        throw std::invalid_argument("Length must be a positive integer.");
    }
    if (length > scanlen()) {
        throw std::invalid_argument(
            "Zgate length cannot be greater than scan length.");
    }
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Threshold must be between 0.0 and 1.0.");
    }
    const std::int64_t max_start = scanlen() - length;
    const double thresh_val = threshold * full_scale;
    const size_t n = data_.rows();

    std::vector<std::int32_t> starts(n, -1);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        std::int64_t start = first_crossing(data_[i], thresh_val);
        if (start >= 0) {
            if (start > max_start) start = max_start;
            starts[i] = static_cast<std::int32_t>(start);
        }
    }
    align_manual(starts, length);
}

void sam_scan::align_xcorr(size_t reference, std::int64_t max_shift,
                           double max_gate_loss) {
    ensure_loaded();
    const size_t n = data_.rows();
    if (reference >= n) {
        throw std::out_of_range(
            "align_xcorr: reference scan index out of range.");
    }
    const size_t sl = data_.cols();
    std::int64_t ms = max_shift;
    if (ms == 0) ms = static_cast<std::int64_t>(sl / 4);
    if (ms < 0 || static_cast<size_t>(ms) >= sl) {
        throw std::invalid_argument(
            "align_xcorr: max_shift must be in [0, scanlen).");
    }
    const auto ref = data_[reference];
    std::vector<std::int32_t> shifts(n, 0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        shifts[i] = static_cast<std::int32_t>(
            signal::xcorr_lag(data_[i], ref, ms));
    }

    // Keep the overlap common to every scan: the output window is
    // `scanlen - spread` long and each scan's origin moves by the common
    // front crop `-min_shift`, so absolute feature times are preserved.
    const std::int32_t min_shift =
        *std::min_element(shifts.begin(), shifts.end());
    const std::int32_t max_shift_v =
        *std::max_element(shifts.begin(), shifts.end());
    const std::int64_t spread =
        static_cast<std::int64_t>(max_shift_v - min_shift);
    if (max_gate_loss < 1.0 &&
        static_cast<double>(spread) >
            max_gate_loss * static_cast<double>(sl)) {
        throw std::invalid_argument(
            "align_xcorr: aligned shift spread exceeds max_gate_loss.");
    }
    const std::int64_t gate_len = static_cast<std::int64_t>(sl) - spread;
    if (gate_len <= 0) {
        throw std::invalid_argument(
            "align_xcorr: lag spread exceeds the scan length.");
    }
    std::vector<std::int32_t> offsets(n);
    for (size_t i = 0; i < n; ++i) {
        offsets[i] = static_cast<std::int32_t>(shifts[i] - min_shift);
    }
    align_manual(offsets, gate_len);
}

void sam_scan::align_tof(double gate_ns, size_t reference, double start_ns,
                         double max_gate_loss) {
    ensure_loaded();
    if (!(gate_ns > 0.0)) {
        throw std::invalid_argument("align_tof: gate_ns must be positive.");
    }
    if (start_ns < 0.0) {
        throw std::invalid_argument(
            "align_tof: start_ns must be non-negative.");
    }
    const size_t n = data_.rows();
    if (reference >= n) {
        throw std::out_of_range(
            "align_tof: reference scan index out of range.");
    }
    const size_t sl = data_.cols();
    const double spacing = samplespacing();
    const std::int64_t start_idx = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::llround(start_ns / spacing)), 0,
        static_cast<std::int64_t>(sl) - 1);
    const std::int64_t gate_samples = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(std::llround(gate_ns / spacing)));
    const std::int64_t end_idx = std::min<std::int64_t>(
        static_cast<std::int64_t>(sl), start_idx + gate_samples);
    if (start_idx >= end_idx) {
        throw std::invalid_argument("align_tof: gate is outside the scan.");
    }

    // Envelope-peak ToF (ns) per scan, converted back to sample positions.
    const array2d<float> peaks = tof(start_idx, end_idx, true);
    const size_t nc = static_cast<size_t>(cols());
    std::vector<double> pos(n, 0.0);
    std::vector<bool> valid(n, false);
    for (size_t i = 0; i < n; ++i) {
        const float v = peaks[i / nc][i % nc];
        if (v != no_tof) {
            pos[i] = (static_cast<double>(v) - header_.tzero) / spacing;
            valid[i] = true;
        }
    }
    if (!valid[reference]) {
        throw std::invalid_argument(
            "align_tof: reference scan has no envelope peak in the gate.");
    }

    // Integer shift relative to the reference; unaligned scans stay -1.
    std::vector<std::int32_t> shifts(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (valid[i]) {
            shifts[i] = static_cast<std::int32_t>(
                std::llround(pos[i] - pos[reference]));
        }
    }

    // Common overlap of every scan that has an envelope peak; scans without
    // one (valid[i] == false) keep start -1 and are zero-filled.
    std::int32_t min_shift = 0;
    std::int32_t max_shift = 0;
    bool any = false;
    for (size_t i = 0; i < n; ++i) {
        if (!valid[i]) continue;
        if (!any) {
            min_shift = max_shift = shifts[i];
            any = true;
        } else {
            min_shift = std::min(min_shift, shifts[i]);
            max_shift = std::max(max_shift, shifts[i]);
        }
    }
    const std::int64_t spread =
        static_cast<std::int64_t>(max_shift - min_shift);
    if (max_gate_loss < 1.0 &&
        static_cast<double>(spread) >
            max_gate_loss * static_cast<double>(sl)) {
        throw std::invalid_argument(
            "align_tof: peak spread exceeds max_gate_loss.");
    }
    const std::int64_t gate_len = static_cast<std::int64_t>(sl) - spread;
    if (gate_len <= 0) {
        throw std::invalid_argument(
            "align_tof: peak spread exceeds the scan length.");
    }
    std::vector<std::int32_t> offsets(n, -1);
    for (size_t i = 0; i < n; ++i) {
        if (valid[i]) {
            offsets[i] = static_cast<std::int32_t>(shifts[i] - min_shift);
        }
    }
    align_manual(offsets, gate_len);
}

signal::stft_result sam_scan::compute_stft(size_t nperseg, size_t noverlap,
                                           double f_min, double f_max) const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    const double fs = samplerate() * 1e6;
    auto res = signal::stft(fdata, fs, nperseg, noverlap, f_min, f_max);
    // Align to the handler time scale: t_aligned = t + timescale[0] * 1e-9
    const auto ts = time();
    const double offset = (ts.empty() ? 0.0 : ts[0]) * 1e-9;
    for (auto& t : res.t) t += static_cast<float>(offset);
    return res;
}

signal::psd_result sam_scan::psd(size_t nperseg, size_t noverlap,
                                 double f_min, double f_max) const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    return signal::welch_psd(fdata, samplerate() * 1e6, nperseg, noverlap,
                             f_min, f_max);
}

signal::spectrogram_result sam_scan::power_spectrogram(
    size_t nperseg, size_t noverlap, double f_min, double f_max) const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    return signal::spectrogram_psd(fdata, samplerate() * 1e6, nperseg,
                                   noverlap, f_min, f_max);
}

array2d<float> sam_scan::stft_peak_frequency(size_t nperseg, size_t noverlap,
                                             double f_min, double f_max) const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    const auto res = signal::stft_peaks(fdata, samplerate() * 1e6, nperseg,
                                        noverlap, f_min, f_max);
    return reshape_signals(res.peak_frequency, static_cast<size_t>(nlines()),
                           static_cast<size_t>(cols()));
}

array2d<float> sam_scan::stft_peak_time(size_t nperseg, size_t noverlap,
                                        double f_min, double f_max) const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    const auto res = signal::stft_peaks(fdata, samplerate() * 1e6, nperseg,
                                        noverlap, f_min, f_max);
    // Align to the handler time scale like compute_stft().
    const auto ts = time();
    const double offset = (ts.empty() ? 0.0 : ts[0]) * 1e-9;
    std::vector<float> aligned = res.peak_time;
    for (auto& t : aligned) t += static_cast<float>(offset);
    return reshape_signals(aligned, static_cast<size_t>(nlines()),
                           static_cast<size_t>(cols()));
}

signal::spectrum_result sam_scan::spectrum() const {
    ensure_loaded();
    const array2d<float> fdata = to_float(data_);
    return signal::fft_spectrum(fdata, 1.0 / (samplerate() * 1e6));
}

} // namespace samcore

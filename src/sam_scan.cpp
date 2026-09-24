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

sam_scan sam_scan::from_file(const std::filesystem::path& path, bool mmap) {
    const std::string ext = extension_lower(path);
    sam_scan scan;
    if (ext == ".h5sam") {
        if (mmap) {
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

} // namespace

array2d<std::int8_t> sam_scan::image_max() const {
    ensure_loaded();
    return reduce_image<std::int8_t>(
        data_, static_cast<size_t>(header_.nlines),
        static_cast<size_t>(header_.scanspline),
        [](std::span<const std::int8_t> row) {
            return *std::max_element(row.begin(), row.end());
        });
}

array2d<std::int8_t> sam_scan::image_absmax() const {
    ensure_loaded();
    // absmax = max(|min|, |max|), saturated to int8 (abs(-128) -> 127) so
    // the result stays in the same 8-bit domain as the input.
    return reduce_image<std::int8_t>(
        data_, static_cast<size_t>(header_.nlines),
        static_cast<size_t>(header_.scanspline),
        [](std::span<const std::int8_t> row) {
            std::int8_t mx = row[0], mn = row[0];
            for (auto v : row) {
                mx = std::max(mx, v);
                mn = std::min(mn, v);
            }
            const int am = std::max(std::abs(static_cast<int>(mx)),
                                    std::abs(static_cast<int>(mn)));
            return static_cast<std::int8_t>(std::min(am, 127));
        });
}

array2d<float> sam_scan::image_power() const {
    ensure_loaded();
    // power: sum of squares, float32
    return reduce_image<float>(
        data_, static_cast<size_t>(header_.nlines),
        static_cast<size_t>(header_.scanspline),
        [](std::span<const std::int8_t> row) {
            double acc = 0.0;
            for (auto v : row) acc += static_cast<double>(v) * v;
            return static_cast<float>(acc);
        });
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
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (num_signals > 8)
#endif
        for (size_t s = 0; s < num_signals; ++s) {
            std::vector<double> row(trimmed_len);
            for (size_t i = 0; i < trimmed_len; ++i) row[i] = data_[s][i];
            const auto dec = signal::decimate(row, factor);
            for (size_t i = 0; i < new_len; ++i) {
                const double v = std::clamp(dec[i], -128.0, 127.0);
                downsampled[s][i] = static_cast<std::int8_t>(v); // truncation
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

sam_scan sam_scan::time_range_select(double start_time, double end_time) const {
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
    const std::int64_t new_scanlen = end_idx - start_idx;
    array2d<std::int8_t> new_data(data_.rows(), static_cast<size_t>(new_scanlen));
    for (size_t s = 0; s < data_.rows(); ++s) {
        std::memcpy(new_data[s].data(),
                    data_[s].data() + static_cast<std::ptrdiff_t>(start_idx),
                    static_cast<size_t>(new_scanlen));
    }
    sam_header new_header = header_;
    new_header.scanlen = new_scanlen;
    new_header.tzero =
        static_cast<std::int64_t>(std::nearbyint(header_.tzero + start_idx * sp));
    // starts are dropped
    return from_data(std::move(new_data), std::move(new_header), std::nullopt,
                     labels_);
}

void sam_scan::time_range_select_ip(double start_time, double end_time) {
    *this = time_range_select(start_time, end_time);
}

sam_scan::sam_scan() = default;

sam_scan::sam_scan(const std::string& path, bool mmap) {
    *this = from_file(path, mmap);
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
    return lazy_ ? lazy_->rows : data_.rows();
}

void sam_scan::ensure_loaded() const {
    if (!lazy_) return;
    H5::DataSpace space = lazy_->dset.getSpace();
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    array2d<std::int8_t> loaded(static_cast<size_t>(dims[0]),
                                static_cast<size_t>(dims[1]));
    if (dims[0] > 0 && dims[1] > 0) {
        lazy_->dset.read(loaded.data(), H5::PredType::NATIVE_INT8);
    }
    data_ = std::move(loaded);
    lazy_.reset(); // closes the file handle
}

void sam_scan::align_manual(const std::vector<std::int32_t>& new_starts,
                            std::int64_t new_scanlen) {
    const std::int64_t max_start = scanlen() - new_scanlen;
    for (auto s : new_starts) {
        if (s > max_start && s != -1) {
            throw std::invalid_argument(
                "Start indices must be in the range [-1, " +
                std::to_string(max_start) + "] for scan length " +
                std::to_string(scanlen()) + " and new scan length " +
                std::to_string(new_scanlen) + ".");
        }
    }
    std::vector<uint8_t> valid_new(new_starts.size());
    for (size_t i = 0; i < new_starts.size(); ++i) {
        valid_new[i] = new_starts[i] != -1 ? 1 : 0;
    }
    if (!starts_.has_value()) {
        for (size_t i = 0; i < valid_new.size(); ++i) {
            if (!valid_new[i]) {
                std::fill(data_[i].begin(), data_[i].end(), 0);
            }
        }
        starts_ = new_starts;
    } else {
        std::vector<uint8_t> existing_valid(starts_->size());
        for (size_t i = 0; i < starts_->size(); ++i) {
            existing_valid[i] = (*starts_)[i] != -1 ? 1 : 0;
        }
        for (size_t i = 0; i < starts_->size(); ++i) {
            if (valid_new[i] && existing_valid[i]) {
                (*starts_)[i] += new_starts[i];
            } else {
                (*starts_)[i] = -1;
                std::fill(data_[i].begin(), data_[i].end(), 0);
            }
        }
    }
}

void sam_scan::zgate_ip(double threshold, std::int64_t length) {
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
    const double thresh_val = threshold * 127.0;
    const size_t n = data_.rows();
    const size_t len = static_cast<size_t>(length);

    std::vector<std::int32_t> starts(n);
    array2d<std::int8_t> gated(n, len);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        const auto scan = data_[i];
        std::int64_t start = -1;
        for (size_t k = 0; k < scan.size(); ++k) {
            // numpy parity: np.abs() on int8 wraps |-128| to -128, so a
            // -128 sample never counts as a threshold crossing.
            const auto a = static_cast<std::int8_t>(std::abs(scan[k]));
            if (static_cast<double>(a) >= thresh_val) {
                start = static_cast<std::int64_t>(k);
                break;
            }
        }
        if (start >= 0) {
            if (start > max_start) start = max_start;
            starts[i] = static_cast<std::int32_t>(start);
            std::copy(scan.begin() + static_cast<std::ptrdiff_t>(start),
                      scan.begin() + static_cast<std::ptrdiff_t>(start + len),
                      gated[i].begin());
        } else {
            starts[i] = -1;
            std::fill(gated[i].begin(), gated[i].end(), 0);
        }
    }

    data_ = std::move(gated);
    align_manual(starts, length);

    if (starts_.has_value() && !starts_->empty()) {
        const bool all_equal =
            std::all_of(starts_->begin(), starts_->end(),
                        [&](std::int32_t v) { return v == (*starts_)[0]; });
        if (all_equal && (*starts_)[0] >= 0) {
            header_.tzero += static_cast<std::int64_t>(std::nearbyint(
                static_cast<double>((*starts_)[0]) / samplerate() * 1e3));
            starts_ = std::nullopt;
        }
    }
    header_.scanlen = length;
}

sam_scan sam_scan::zgate(double threshold, std::int64_t length) const {
    sam_scan h = copy();
    h.zgate_ip(threshold, length);
    return h;
}

signal::stft_result sam_scan::compute_stft(size_t nperseg,
                                           size_t noverlap) const {
    ensure_loaded();
    array2d<float> fdata(data_.rows(), data_.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data_.rows() > 8)
#endif
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            fdata[i][j] = static_cast<float>(data_[i][j]);
        }
    }
    const double fs = samplerate() * 1e6;
    auto res = signal::stft(fdata, fs, nperseg, noverlap);
    // Align to the handler time scale: t_aligned = t + timescale[0] * 1e-9
    const auto ts = time();
    const double offset = (ts.empty() ? 0.0 : ts[0]) * 1e-9;
    for (auto& t : res.t) t += static_cast<float>(offset);
    return res;
}

signal::psd_result sam_scan::psd(size_t nperseg, size_t noverlap) const {
    ensure_loaded();
    array2d<float> fdata(data_.rows(), data_.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data_.rows() > 8)
#endif
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            fdata[i][j] = static_cast<float>(data_[i][j]);
        }
    }
    return signal::welch_psd(fdata, samplerate() * 1e6, nperseg, noverlap);
}

signal::spectrogram_result sam_scan::power_spectrogram(
    size_t nperseg, size_t noverlap) const {
    ensure_loaded();
    array2d<float> fdata(data_.rows(), data_.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data_.rows() > 8)
#endif
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            fdata[i][j] = static_cast<float>(data_[i][j]);
        }
    }
    return signal::spectrogram_psd(fdata, samplerate() * 1e6, nperseg,
                                   noverlap);
}

} // namespace samcore

#include <samcore/sam_dataset.hpp>

#include <samcore/constants.hpp>
#include <samcore/preprocessing.hpp>
#include <samcore/signal/kernels.hpp>

#include "io/h5_lazy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace samcore {

namespace {

dataset_merge_options compatibility_options(float pad_value) {
    dataset_merge_options options;
    options.mode = merge_mode::crop;
    options.pad_value = pad_value;
    return options;
}

// First sample whose |value| reaches thresh_val; -1 when none.
std::int64_t first_crossing(std::span<const float> signal, double thresh_val) {
    for (size_t k = 0; k < signal.size(); ++k) {
        if (std::abs(static_cast<double>(signal[k])) >= thresh_val) {
            return static_cast<std::int64_t>(k);
        }
    }
    return -1;
}

// int8 variant preserving the historical numpy abs(-128) parity.
std::int64_t first_crossing(std::span<const std::int8_t> signal,
                            double thresh_val) {
    for (size_t k = 0; k < signal.size(); ++k) {
        const auto a = static_cast<std::int8_t>(std::abs(signal[k]));
        if (static_cast<double>(a) >= thresh_val) {
            return static_cast<std::int64_t>(k);
        }
    }
    return -1;
}

// Analytic-envelope peak inside [start, end), optionally refined by a
// parabolic fit.  Returns the sample position or nullopt for a silent window.
std::optional<double> envelope_peak_position(std::span<const float> signal,
                                             std::int64_t start,
                                             std::int64_t end,
                                             bool sub_sample = true) {
    const std::vector<double> row(signal.begin(), signal.end());
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
    if (peak < 0 || best <= 0.0) return std::nullopt;
    double pos = static_cast<double>(peak);
    if (sub_sample && peak > start && peak + 1 < end) {
        const double y0 = env[static_cast<size_t>(peak - 1)];
        const double y2 = env[static_cast<size_t>(peak + 1)];
        const double denom = y0 - 2.0 * best + y2;
        if (denom != 0.0) {
            pos += std::clamp(0.5 * (y0 - y2) / denom, -0.5, 0.5);
        }
    }
    return pos;
}

// Rows belonging to cube idx: [start, end) within the concatenated X.
std::pair<size_t, size_t> cube_row_range_impl(
    const std::vector<std::pair<std::int32_t, std::int32_t>>& shapes,
    std::int32_t idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= shapes.size()) {
        throw std::out_of_range("cube index out of range");
    }
    size_t offset = 0;
    for (std::int32_t i = 0; i < idx; ++i) {
        offset += static_cast<size_t>(shapes[static_cast<size_t>(i)].first) *
                  static_cast<size_t>(shapes[static_cast<size_t>(i)].second);
    }
    const auto& shape = shapes[static_cast<size_t>(idx)];
    return {offset, offset + static_cast<size_t>(shape.first) *
                              static_cast<size_t>(shape.second)};
}

void append_3d(array3d<float>& dst, size_t& row, array3d<float>&& src) {
    const size_t n = src.size0();
    if (n == 0) return;
    if (row + n > dst.size0() || src.size1() != dst.size1() ||
        src.size2() != dst.size2()) {
        throw std::invalid_argument("spectral results have incompatible shapes");
    }
    std::copy(src.flat().begin(), src.flat().end(),
              dst.flat().begin() + static_cast<std::ptrdiff_t>(
                                       row * src.size1() * src.size2()));
    row += n;
}

void append_3d_complex(array3d<std::complex<float>>& dst, size_t& row,
                       array3d<std::complex<float>>&& src) {
    const size_t n = src.size0();
    if (n == 0) return;
    if (row + n > dst.size0() || src.size1() != dst.size1() ||
        src.size2() != dst.size2()) {
        throw std::invalid_argument("spectral results have incompatible shapes");
    }
    std::copy(src.flat().begin(), src.flat().end(),
              dst.flat().begin() + static_cast<std::ptrdiff_t>(
                                       row * src.size1() * src.size2()));
    row += n;
}

} // namespace

sam_dataset::sam_dataset(const sam_dataset& o) { *this = o; }

sam_dataset& sam_dataset::operator=(const sam_dataset& o) {
    if (this == &o) return *this;
    o.ensure_loaded(); // a lazy-mode source is materialized once, here
    x_ = o.x_;
    labels_ = o.labels_;
    cube_shapes_ = o.cube_shapes_;
    cube_resolutions_ = o.cube_resolutions_;
    scanlens_ = o.scanlens_;
    valid_scanlens_ = o.valid_scanlens_;
    cube_samplerates_ = o.cube_samplerates_;
    merge_mode_ = o.merge_mode_;
    pad_value_ = o.pad_value_;
    unsupervised_ = o.unsupervised_;
    z_ = o.z_;
    v_ = o.v_;
    train_indices_ = o.train_indices_;
    test_indices_ = o.test_indices_;
    shuffled_ = o.shuffled_;
    lazy_.reset();
    return *this;
}

sam_dataset::sam_dataset(sam_dataset&&) noexcept = default;
sam_dataset& sam_dataset::operator=(sam_dataset&&) noexcept = default;
sam_dataset::~sam_dataset() = default;

void sam_dataset::ensure_loaded() const {
    if (!lazy_) return;
    io::h5samd_lazy_data data = io::read_h5samd_lazy_data(*lazy_);
    x_ = std::move(data.x);
    z_ = std::move(data.z);
    v_ = std::move(data.v);
    lazy_.reset(); // closes the file handle
}

array2d<float> sam_dataset::read_rows(
    const std::vector<std::int64_t>& indices, bool use_z) const {
    if (use_z && !has_z()) {
        throw std::runtime_error("Z has not been built yet.");
    }
    const size_t cols = use_z ? num_features() : maxlen();
    array2d<float> out(indices.size(), cols);
    if (!lazy_) {
        const array2d<float>& src = use_z ? *z_ : x_;
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto row = static_cast<size_t>(indices[i]);
            if (row >= src.rows()) {
                throw std::out_of_range("read_rows: row index out of range.");
            }
            std::memcpy(out[i].data(), src[row].data(), cols * sizeof(float));
        }
        return out;
    }
    if (indices.empty()) return out;
    auto& reader = use_z ? *lazy_->z : *lazy_->x;
    reader.read_selected(indices, out.data());
    return out;
}

size_t sam_dataset::blocks_read() const noexcept {
    return (lazy_ && lazy_->x) ? lazy_->x->blocks_read() : 0;
}

const array2d<float>& sam_dataset::X() const {
    ensure_loaded();
    return x_;
}

array2d<float>& sam_dataset::X() {
    ensure_loaded();
    return x_;
}

const std::optional<array2d<float>>& sam_dataset::Z() const {
    ensure_loaded();
    return z_;
}

std::optional<array2d<float>>& sam_dataset::Z() {
    ensure_loaded();
    return z_;
}

const std::optional<array2d<float>>& sam_dataset::V() const {
    ensure_loaded();
    return v_;
}

std::optional<array2d<float>>& sam_dataset::V() {
    ensure_loaded();
    return v_;
}

bool sam_dataset::has_z() const noexcept {
    return lazy_ ? lazy_->z.has_value() : z_.has_value();
}

bool sam_dataset::has_v() const noexcept {
    return lazy_ ? lazy_->v.has_value() : v_.has_value();
}

size_t sam_dataset::num_samples() const noexcept {
    return lazy_ ? lazy_->x_rows : x_.rows();
}

size_t sam_dataset::num_features() const noexcept {
    return lazy_ ? lazy_->z_cols : (z_ ? z_->cols() : 0);
}

size_t sam_dataset::maxlen() const noexcept {
    return lazy_ ? lazy_->x_cols : x_.cols();
}

std::optional<double> sam_dataset::samplerate() const noexcept {
    if (cube_samplerates_.empty()) return std::nullopt;
    const double first = cube_samplerates_.front();
    for (double fs : cube_samplerates_) {
        if (fs != first) return std::nullopt;
    }
    return first;
}

double sam_dataset::require_samplerate() const {
    if (cube_samplerates_.empty()) {
        throw std::runtime_error(
            "Dataset has no sampling-rate metadata (legacy .h5samd file); "
            "rebuild it from .h5sam scans to use spectral methods.");
    }
    const double fs = cube_samplerates_.front();
    if (!(fs > 0.0) || !std::isfinite(fs)) {
        throw std::runtime_error("Dataset sampling rate is not positive.");
    }
    for (size_t i = 1; i < cube_samplerates_.size(); ++i) {
        if (cube_samplerates_[i] != fs) {
            throw std::invalid_argument(
                "Dataset spectral methods require a uniform samplerate; "
                "cube " + std::to_string(i) + " has " +
                std::to_string(cube_samplerates_[i]) +
                " MHz while cube 0 has " + std::to_string(fs) + " MHz.");
        }
    }
    return fs;
}

sam_dataset::sam_dataset(std::vector<sam_scan> handlers, float pad_value,
                         std::optional<bool> unsupervised)
    : sam_dataset(std::move(handlers), compatibility_options(pad_value),
                  unsupervised) {}

sam_dataset::sam_dataset(std::vector<sam_scan> handlers,
                         const dataset_merge_options& options,
                         std::optional<bool> unsupervised) {
    if (handlers.empty()) {
        throw std::invalid_argument(
            "SAMDataset must be initialized with at least one SAMHandler.");
    }

    std::vector<sam_labels> labels_list;
    std::vector<std::pair<std::int32_t, std::int32_t>> shapes;
    std::vector<double> resolutions;
    std::vector<double> samplerates;
    std::vector<std::int32_t> lens;
    std::vector<size_t> row_cube;
    std::vector<size_t> row_local;

    size_t total = 0;
    for (size_t k = 0; k < handlers.size(); ++k) {
        auto& h = handlers[k];
        h.load(); // materialize before the parallel build below
        const size_t n = static_cast<size_t>(h.nlines() * h.cols());
        const size_t sl = static_cast<size_t>(h.scanlen());
        if (sl == 0) {
            throw std::invalid_argument("SAMScan has zero scan length.");
        }
        total += n;
        shapes.emplace_back(static_cast<std::int32_t>(h.nlines()),
                            static_cast<std::int32_t>(h.cols()));
        resolutions.push_back(h.header().resolution);
        samplerates.push_back(h.header().samplerate);
        lens.push_back(static_cast<std::int32_t>(sl));
        labels_list.push_back(h.samlabels().copy());
        row_cube.resize(total, k);
        row_local.resize(total);
        for (size_t r = 0; r < n; ++r) row_local[total - n + r] = r;
    }
    if (total == 0) {
        throw std::invalid_argument("SAMDataset cannot be empty.");
    }

    cube_shapes_ = std::move(shapes);
    cube_resolutions_ = std::move(resolutions);
    scanlens_ = std::move(lens);
    cube_samplerates_ = std::move(samplerates);
    merge_mode_ = options.mode;
    pad_value_ = options.pad_value;

    const auto source_length = [&](size_t r) {
        return static_cast<std::int64_t>(scanlens_[row_cube[r]]);
    };
    const auto min_scanlen = static_cast<std::int64_t>(
        *std::min_element(scanlens_.begin(), scanlens_.end()));

    std::vector<std::int32_t> offsets(total, 0);
    std::int64_t out_cols = 0;

    switch (options.mode) {
    case merge_mode::crop:
        out_cols = min_scanlen;
        break;

    case merge_mode::pad:
        out_cols = *std::max_element(scanlens_.begin(), scanlens_.end());
        break;

    case merge_mode::zgate: {
        if (options.zgate_length <= 0) {
            throw std::invalid_argument(
                "zgate_length must be a positive integer.");
        }
        if (options.threshold < 0.0 || options.threshold > 1.0) {
            throw std::invalid_argument(
                "zgate threshold must be between 0.0 and 1.0.");
        }
        if (options.zgate_length > min_scanlen) {
            throw std::invalid_argument(
                "zgate_length cannot exceed the shortest scan length (" +
                std::to_string(min_scanlen) + ").");
        }
        out_cols = options.zgate_length;
        const double thresh_val = options.threshold * full_scale;
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (total > 8)
#endif
        for (size_t r = 0; r < total; ++r) {
            const auto& src = handlers[row_cube[r]].data()[row_local[r]];
            std::int64_t start = first_crossing(src, thresh_val);
            if (start >= 0) {
                const std::int64_t max_start =
                    source_length(r) - options.zgate_length;
                start = std::min(start, max_start);
                offsets[r] = static_cast<std::int32_t>(start);
            } else {
                offsets[r] = -1;
            }
        }
        break;
    }

    case merge_mode::xcorr: {
        if (options.reference >= total) {
            throw std::out_of_range(
                "xcorr reference row is out of range.");
        }
        if (options.max_gate_loss <= 0.0 || options.max_gate_loss > 1.0) {
            throw std::invalid_argument(
                "max_gate_loss must be in (0, 1].");
        }
        const size_t common = static_cast<size_t>(min_scanlen);
        std::int64_t max_shift = options.max_shift;
        if (max_shift == 0) max_shift = static_cast<std::int64_t>(common / 4);
        if (max_shift < 0 || static_cast<size_t>(max_shift) >= common) {
            throw std::invalid_argument(
                "max_shift must be in [0, min_scanlen).");
        }
        const auto& ref_data =
            handlers[row_cube[options.reference]].data()[row_local[options.reference]];
        const std::span<const std::int8_t> ref(ref_data.data(), common);
        std::vector<std::int32_t> shifts(total, 0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (total > 8)
#endif
        for (size_t r = 0; r < total; ++r) {
            const auto& src = handlers[row_cube[r]].data()[row_local[r]];
            shifts[r] = static_cast<std::int32_t>(
                signal::xcorr_lag(std::span<const std::int8_t>(
                                      src.data(), common),
                                  ref, max_shift));
        }
        const auto [min_it, max_it] =
            std::minmax_element(shifts.begin(), shifts.end());
        const std::int64_t spread =
            static_cast<std::int64_t>(*max_it) - *min_it;
        if (static_cast<double>(spread) >
            options.max_gate_loss * static_cast<double>(common)) {
            throw std::invalid_argument(
                "xcorr: aligned shift spread exceeds max_gate_loss.");
        }
        for (size_t r = 0; r < total; ++r) {
            offsets[r] = shifts[r] - *min_it;
        }
        out_cols = std::numeric_limits<std::int64_t>::max();
        for (size_t r = 0; r < total; ++r) {
            out_cols = std::min(out_cols, source_length(r) - offsets[r]);
        }
        if (out_cols <= 0) {
            throw std::invalid_argument(
                "xcorr: lag spread exceeds every source scan length.");
        }
        break;
    }
    }

    if (out_cols <= 0) {
        throw std::invalid_argument("Merged dataset has zero signal length.");
    }

    x_ = array2d<float>(total, static_cast<size_t>(out_cols));
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (total > 8) schedule(static)
#endif
    for (size_t r = 0; r < total; ++r) {
        const auto& src = handlers[row_cube[r]].data()[row_local[r]];
        auto dst = x_[r];
        if (options.mode == merge_mode::pad) {
            std::fill(dst.begin(), dst.end(), options.pad_value);
            const size_t n = std::min(src.size(), dst.size());
            std::transform(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(n),
                           dst.begin(),
                           [](std::int8_t v) { return static_cast<float>(v); });
        } else if (offsets[r] < 0) {
            std::fill(dst.begin(), dst.end(), 0.0f);
        } else {
            const auto first = src.begin() + offsets[r];
            std::transform(first, first + static_cast<std::ptrdiff_t>(out_cols),
                           dst.begin(),
                           [](std::int8_t v) { return static_cast<float>(v); });
        }
    }

    valid_scanlens_.resize(handlers.size());
    for (size_t k = 0; k < handlers.size(); ++k) {
        valid_scanlens_[k] = (options.mode == merge_mode::pad)
                                 ? scanlens_[k]
                                 : static_cast<std::int32_t>(out_cols);
    }

    if (unsupervised.has_value()) {
        unsupervised_ = *unsupervised;
    } else {
        unsupervised_ = !std::all_of(labels_list.begin(), labels_list.end(),
                                     [](const sam_labels& l) {
                                         return l.is_labeled();
                                     });
    }

    if (!unsupervised_) {
        for (size_t i = 0; i < labels_list.size(); ++i) {
            if (!labels_list[i].is_labeled()) {
                throw std::invalid_argument(
                    "Cube " + std::to_string(i) +
                    " is unlabeled but dataset is supervised. Either set "
                    "unsupervised=true or ensure all cubes have labels.");
            }
        }
        labels_ = labels_list.size() >= 2 ? merge_labels(labels_list)
                                          : labels_list[0].copy();
    }

    train_indices_.resize(x_.rows());
    std::iota(train_indices_.begin(), train_indices_.end(), 0);
    test_indices_.clear();
    shuffled_ = false;
}

sam_dataset sam_dataset::copy() const {
    return sam_dataset(*this);
}

void sam_dataset::save(const std::filesystem::path& path) const {
    ensure_loaded();
    io::write_h5samd(path, x_, labels_, cube_shapes_, cube_resolutions_,
                     scanlens_, valid_scanlens_, cube_samplerates_,
                     merge_mode_, pad_value_, unsupervised_, z_, v_);
}

sam_dataset sam_dataset::load(const std::filesystem::path& path, bool lazy) {
    if (lazy) {
        io::h5samd_lazy_handle res = io::read_h5samd_lazy(path);
        sam_dataset ds;
        ds.labels_ = std::move(res.labels);
        ds.cube_shapes_ = std::move(res.cube_shapes);
        ds.cube_resolutions_ = std::move(res.cube_resolutions);
        ds.scanlens_ = std::move(res.scanlens);
        ds.valid_scanlens_ = std::move(res.valid_scanlens);
        ds.cube_samplerates_ = std::move(res.cube_samplerates);
        ds.merge_mode_ = res.merge;
        ds.unsupervised_ = res.unsupervised;
        ds.pad_value_ = res.pad_value;
        ds.lazy_ = std::move(res.data);
        ds.train_indices_.resize(ds.num_samples());
        std::iota(ds.train_indices_.begin(), ds.train_indices_.end(), 0);
        ds.test_indices_.clear();
        ds.shuffled_ = false;
        return ds;
    }
    io::h5samd_result res = io::read_h5samd(path);
    sam_dataset ds;
    ds.x_ = std::move(res.x);
    ds.labels_ = std::move(res.labels);
    ds.cube_shapes_ = std::move(res.cube_shapes);
    ds.cube_resolutions_ = std::move(res.cube_resolutions);
    ds.scanlens_ = std::move(res.scanlens);
    ds.valid_scanlens_ = std::move(res.valid_scanlens);
    ds.cube_samplerates_ = std::move(res.cube_samplerates);
    ds.merge_mode_ = res.merge;
    ds.unsupervised_ = res.unsupervised;
    ds.z_ = std::move(res.z);
    ds.v_ = std::move(res.v);
    ds.pad_value_ = res.pad_value;
    ds.train_indices_.resize(ds.x_.rows());
    std::iota(ds.train_indices_.begin(), ds.train_indices_.end(), 0);
    ds.test_indices_.clear();
    ds.shuffled_ = false;
    return ds;
}

std::vector<spatial_record> sam_dataset::spatial() const {
    std::vector<spatial_record> out;
    out.reserve(num_samples());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        const auto& [nlines, cols] = cube_shapes_[i];
        const size_t n = static_cast<size_t>(nlines) * cols;
        const double res_mm = cube_resolutions_[i] / 1000.0;
        for (size_t k = 0; k < n; ++k) {
            spatial_record rec;
            rec.idx = static_cast<std::int32_t>(i);
            rec.x = static_cast<float>((k % static_cast<size_t>(cols)) * res_mm);
            rec.y = static_cast<float>((k / static_cast<size_t>(cols)) * res_mm);
            out.push_back(rec);
        }
    }
    return out;
}

size_t sam_dataset::num_classes() const {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    return labels_->num_classes();
}

void sam_dataset::preprocess(const std::string& strategy,
                             const preprocess_args& args) {
    ensure_loaded();
    if (strategy == "lp") {
        if (args.cutoff <= 0.0 || args.fs <= 0.0) {
            throw std::invalid_argument("'cutoff' and 'fs' are required for 'lp'.");
        }
        x_ = preprocessing::lp(x_, args.cutoff, args.fs);
    } else if (strategy == "bp") {
        if (args.cutoff_low <= 0.0 || args.cutoff_high <= 0.0 || args.fs <= 0.0) {
            throw std::invalid_argument(
                "'cutoff_low', 'cutoff_high' and 'fs' are required for 'bp'.");
        }
        x_ = preprocessing::bp(x_, args.cutoff_low, args.cutoff_high, args.fs);
    } else if (strategy == "normalize") {
        x_ = preprocessing::normalize(x_, args.mode);
    } else if (strategy == "savgol") {
        x_ = preprocessing::savgol(x_, args.window_length, args.polyorder);
    } else if (strategy == "medfilt") {
        x_ = preprocessing::medfilt(x_, args.kernel_size);
    } else if (strategy == "gate") {
        const size_t n = x_.cols();
        const size_t end = args.end == 0 ? n : args.end;
        x_ = preprocessing::gate(x_, args.start, args.end);
        for (auto& valid : valid_scanlens_) {
            const size_t old = static_cast<size_t>(valid);
            const size_t s = std::min(args.start, old);
            const size_t e = std::min(end, old);
            valid = static_cast<std::int32_t>(e > s ? e - s : 0);
        }
    } else if (strategy == "detrend") {
        x_ = preprocessing::detrend(x_);
    } else if (strategy == "envelope") {
        x_ = preprocessing::envelope(x_);
    } else if (strategy == "moving_average") {
        x_ = preprocessing::moving_average(x_, args.window);
    } else {
        throw std::invalid_argument("Unknown preprocess strategy: " + strategy);
    }
}

void sam_dataset::align_manual(const std::vector<std::int32_t>& starts,
                               std::int64_t new_scanlen) {
    ensure_loaded();
    if (new_scanlen <= 0) {
        throw std::invalid_argument(
            "align_manual: scanlen must be a positive integer.");
    }
    if (starts.size() != x_.rows()) {
        throw std::invalid_argument(
            "align_manual: starts length must equal the number of rows.");
    }

    std::vector<std::int32_t> valid_per_row(x_.rows(), 0);
    {
        size_t row = 0;
        for (size_t c = 0; c < cube_shapes_.size(); ++c) {
            const size_t n = static_cast<size_t>(cube_shapes_[c].first) *
                             static_cast<size_t>(cube_shapes_[c].second);
            for (size_t k = 0; k < n; ++k) {
                valid_per_row[row++] = valid_scanlens_[c];
            }
        }
    }
    for (size_t i = 0; i < starts.size(); ++i) {
        const std::int64_t valid = valid_per_row[i];
        if (new_scanlen > valid) {
            throw std::invalid_argument(
                "align_manual: scanlen cannot exceed the valid signal length "
                "of row " + std::to_string(i) + ".");
        }
        const std::int64_t max_start = valid - new_scanlen;
        if (starts[i] < -1 || starts[i] > max_start) {
            throw std::invalid_argument(
                "start indices must be in [-1, " + std::to_string(max_start) +
                "] for row " + std::to_string(i) + ".");
        }
    }

    array2d<float> out(x_.rows(), static_cast<size_t>(new_scanlen));
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (x_.rows() > 8)
#endif
    for (size_t i = 0; i < x_.rows(); ++i) {
        auto dst = out[i];
        if (starts[i] < 0) {
            std::fill(dst.begin(), dst.end(), 0.0f);
        } else {
            std::copy(x_[i].begin() + starts[i],
                      x_[i].begin() + starts[i] + new_scanlen, dst.begin());
        }
    }
    x_ = std::move(out);
    valid_scanlens_.assign(cube_shapes_.size(),
                           static_cast<std::int32_t>(new_scanlen));
    z_.reset();
    v_.reset();
}

void sam_dataset::align_xcorr(size_t reference, std::int64_t max_shift,
                              double max_gate_loss) {
    ensure_loaded();
    const size_t n = x_.rows();
    if (reference >= n) {
        throw std::out_of_range(
            "align_xcorr: reference scan index out of range.");
    }
    const auto min_valid_it =
        std::min_element(valid_scanlens_.begin(), valid_scanlens_.end());
    const size_t common = static_cast<size_t>(*min_valid_it);
    if (common == 0) {
        throw std::runtime_error("align_xcorr: dataset has zero valid length.");
    }
    std::int64_t ms = max_shift;
    if (ms == 0) ms = static_cast<std::int64_t>(common / 4);
    if (ms < 0 || static_cast<size_t>(ms) >= common) {
        throw std::invalid_argument(
            "align_xcorr: max_shift must be in [0, min_valid_scanlen).");
    }
    if (max_gate_loss <= 0.0 || max_gate_loss > 1.0) {
        throw std::invalid_argument(
            "align_xcorr: max_gate_loss must be in (0, 1].");
    }
    const std::span<const float> ref(x_[reference].data(), common);
    std::vector<std::int32_t> shifts(n, 0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (n > 8)
#endif
    for (size_t i = 0; i < n; ++i) {
        shifts[i] = static_cast<std::int32_t>(
            signal::xcorr_lag(std::span<const float>(x_[i].data(), common),
                              ref, ms));
    }
    const auto [min_it, max_it] =
        std::minmax_element(shifts.begin(), shifts.end());
    const std::int64_t spread =
        static_cast<std::int64_t>(*max_it) - *min_it;
    if (spread > max_gate_loss * static_cast<double>(common)) {
        throw std::invalid_argument(
            "align_xcorr: aligned shift spread exceeds max_gate_loss.");
    }
    std::vector<std::int32_t> offsets(n);
    for (size_t i = 0; i < n; ++i) {
        offsets[i] = static_cast<std::int32_t>(shifts[i] - *min_it);
    }
    std::int64_t newlen = std::numeric_limits<std::int64_t>::max();
    size_t row = 0;
    for (size_t c = 0; c < cube_shapes_.size(); ++c) {
        const size_t count = static_cast<size_t>(cube_shapes_[c].first) *
                             static_cast<size_t>(cube_shapes_[c].second);
        for (size_t k = 0; k < count; ++k, ++row) {
            newlen = std::min(newlen,
                              static_cast<std::int64_t>(valid_scanlens_[c]) -
                                  offsets[row]);
        }
    }
    if (newlen <= 0) {
        throw std::invalid_argument(
            "align_xcorr: lag spread exceeds the valid scan length.");
    }
    align_manual(offsets, newlen);
}

void sam_dataset::align_zgate(double threshold, std::int64_t length) {
    ensure_loaded();
    if (length <= 0) {
        throw std::invalid_argument("Length must be a positive integer.");
    }
    const auto min_valid_it =
        std::min_element(valid_scanlens_.begin(), valid_scanlens_.end());
    if (length > *min_valid_it) {
        throw std::invalid_argument(
            "Zgate length cannot be greater than the shortest valid scan "
            "length.");
    }
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Threshold must be between 0.0 and 1.0.");
    }
    const double thresh_val = threshold * full_scale;
    std::vector<std::int32_t> valid_per_row(x_.rows(), 0);
    size_t row = 0;
    for (size_t c = 0; c < cube_shapes_.size(); ++c) {
        const size_t count = static_cast<size_t>(cube_shapes_[c].first) *
                             static_cast<size_t>(cube_shapes_[c].second);
        for (size_t k = 0; k < count; ++k, ++row) {
            valid_per_row[row] = valid_scanlens_[c];
        }
    }

    std::vector<std::int32_t> starts(x_.rows(), -1);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (x_.rows() > 8)
#endif
    for (size_t i = 0; i < x_.rows(); ++i) {
        const std::int64_t valid = valid_per_row[i];
        const std::int64_t max_start = valid - length;
        std::int64_t start = first_crossing(
            std::span<const float>(x_[i].data(),
                                   static_cast<size_t>(valid)),
            thresh_val);
        if (start >= 0) {
            starts[i] = static_cast<std::int32_t>(
                std::min(start, max_start));
        }
    }
    align_manual(starts, length);
}

void sam_dataset::align_tof(double gate_ns, size_t reference,
                            double start_ns, double max_gate_loss) {
    ensure_loaded();
    if (!(gate_ns > 0.0)) {
        throw std::invalid_argument("align_tof: gate_ns must be positive.");
    }
    if (start_ns < 0.0) {
        throw std::invalid_argument(
            "align_tof: start_ns must be non-negative.");
    }
    if (reference >= x_.rows()) {
        throw std::out_of_range(
            "align_tof: reference scan index out of range.");
    }
    if (max_gate_loss <= 0.0 || max_gate_loss > 1.0) {
        throw std::invalid_argument(
            "align_tof: max_gate_loss must be in (0, 1].");
    }
    const double samplerate_mhz = require_samplerate();
    const auto min_valid_it =
        std::min_element(valid_scanlens_.begin(), valid_scanlens_.end());
    const auto max_valid_it =
        std::max_element(valid_scanlens_.begin(), valid_scanlens_.end());
    if (*min_valid_it != *max_valid_it) {
        throw std::invalid_argument(
            "align_tof: all cubes must have the same valid signal length.");
    }
    const std::int64_t sl = *min_valid_it;
    const double spacing = 1.0 / samplerate_mhz * 1e3;
    const std::int64_t start_idx = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::llround(start_ns / spacing)), 0,
        sl - 1);
    const std::int64_t gate_samples = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(std::llround(gate_ns / spacing)));
    const std::int64_t end_idx =
        std::min<std::int64_t>(sl, start_idx + gate_samples);
    if (start_idx >= end_idx) {
        throw std::invalid_argument("align_tof: gate is outside the scan.");
    }

    std::vector<std::optional<double>> positions(x_.rows());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (x_.rows() > 8)
#endif
    for (size_t i = 0; i < x_.rows(); ++i) {
        positions[i] = envelope_peak_position(
            std::span<const float>(x_[i].data(), static_cast<size_t>(sl)),
            start_idx, end_idx);
    }
    if (!positions[reference].has_value()) {
        throw std::invalid_argument(
            "align_tof: reference scan has no envelope peak in the gate.");
    }

    std::vector<std::int32_t> shifts(x_.rows(), 0);
    for (size_t i = 0; i < x_.rows(); ++i) {
        if (positions[i].has_value()) {
            shifts[i] = static_cast<std::int32_t>(std::llround(
                *positions[i] - *positions[reference]));
        }
    }
    std::int32_t min_shift = 0;
    std::int32_t max_shift = 0;
    bool any = false;
    for (size_t i = 0; i < x_.rows(); ++i) {
        if (!positions[i].has_value()) continue;
        if (!any) {
            min_shift = max_shift = shifts[i];
            any = true;
        } else {
            min_shift = std::min(min_shift, shifts[i]);
            max_shift = std::max(max_shift, shifts[i]);
        }
    }
    const std::int64_t spread = max_shift - min_shift;
    if (spread > max_gate_loss * static_cast<double>(sl)) {
        throw std::invalid_argument(
            "align_tof: peak spread exceeds max_gate_loss.");
    }
    const std::int64_t newlen = sl - spread;
    if (newlen <= 0) {
        throw std::invalid_argument(
            "align_tof: peak spread exceeds the scan length.");
    }
    std::vector<std::int32_t> offsets(x_.rows(), -1);
    for (size_t i = 0; i < x_.rows(); ++i) {
        if (positions[i].has_value()) {
            offsets[i] = shifts[i] - min_shift;
        }
    }
    align_manual(offsets, newlen);
}

std::pair<size_t, size_t> sam_dataset::cube_row_range_internal(
    std::int32_t idx) const {
    return cube_row_range_impl(cube_shapes_, idx);
}

array2d<float> sam_dataset::cube_valid_rows(std::int32_t idx) const {
    ensure_loaded();
    const auto [start, end] = cube_row_range_internal(idx);
    const size_t n = end - start;
    const auto len = static_cast<size_t>(
        valid_scanlens_.at(static_cast<size_t>(idx)));
    if (len > x_.cols()) {
        throw std::runtime_error(
            "Dataset valid scan length exceeds the stored signal width.");
    }
    array2d<float> out(n, len);
    for (size_t r = 0; r < n; ++r) {
        std::copy(x_[start + r].begin(),
                  x_[start + r].begin() + static_cast<std::ptrdiff_t>(len),
                  out[r].begin());
    }
    return out;
}

signal::stft_result sam_dataset::compute_stft(size_t nperseg, size_t noverlap,
                                              double f_min, double f_max,
                                              signal::window_kind window) const {
    auto parts =
        compute_stft_by_cube(nperseg, noverlap, f_min, f_max, window);
    if (parts.empty()) {
        throw std::runtime_error("Dataset is empty.");
    }
    const auto& first = parts.front();
    signal::stft_result out;
    out.f = first.f;
    out.t = first.t;
    const size_t nframes = first.t.size();
    out.zxx = array3d<std::complex<float>>(
        num_samples(), first.f.size(), nframes);
    size_t row = 0;
    for (auto& part : parts) {
        if (part.f != out.f || part.t.size() != nframes) {
            throw std::invalid_argument(
                "Dataset STFT requires a common valid length across cubes; "
                "use compute_stft_by_cube() for ragged results.");
        }
        append_3d_complex(out.zxx, row, std::move(part.zxx));
    }
    return out;
}

std::vector<signal::stft_result> sam_dataset::compute_stft_by_cube(
    size_t nperseg, size_t noverlap, double f_min, double f_max,
    signal::window_kind window) const {
    const double fs = require_samplerate() * 1e6;
    std::vector<signal::stft_result> out;
    out.reserve(cube_shapes_.size());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        out.push_back(signal::stft(rows, fs, nperseg, noverlap, f_min, f_max,
                                   window));
    }
    return out;
}

signal::psd_result sam_dataset::psd(size_t nperseg, size_t noverlap,
                                    double f_min, double f_max,
                                    signal::window_kind window) const {
    auto parts = psd_by_cube(nperseg, noverlap, f_min, f_max, window);
    if (parts.empty()) {
        throw std::runtime_error("Dataset is empty.");
    }
    signal::psd_result out;
    out.f = parts.front().f;
    out.psd = array2d<float>(num_samples(), out.f.size());
    size_t row = 0;
    for (auto& part : parts) {
        if (part.f != out.f || part.psd.cols() != out.f.size()) {
            throw std::invalid_argument(
                "Dataset PSD cubes produced incompatible frequency axes.");
        }
        for (size_t r = 0; r < part.psd.rows(); ++r) {
            std::copy(part.psd[r].begin(), part.psd[r].end(),
                      out.psd[row + r].begin());
        }
        row += part.psd.rows();
    }
    return out;
}

std::vector<signal::psd_result> sam_dataset::psd_by_cube(
    size_t nperseg, size_t noverlap, double f_min, double f_max,
    signal::window_kind window) const {
    const double fs = require_samplerate() * 1e6;
    std::vector<signal::psd_result> out;
    out.reserve(cube_shapes_.size());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        out.push_back(signal::welch_psd(rows, fs, nperseg, noverlap, f_min,
                                        f_max, window));
    }
    return out;
}

signal::spectrogram_result sam_dataset::power_spectrogram(
    size_t nperseg, size_t noverlap, double f_min, double f_max,
    signal::window_kind window) const {
    auto parts = power_spectrogram_by_cube(nperseg, noverlap, f_min, f_max,
                                           window);
    if (parts.empty()) {
        throw std::runtime_error("Dataset is empty.");
    }
    const auto& first = parts.front();
    signal::spectrogram_result out;
    out.f = first.f;
    out.t = first.t;
    const size_t nframes = first.t.size();
    out.sxx = array3d<float>(num_samples(), first.f.size(), nframes);
    size_t row = 0;
    for (auto& part : parts) {
        if (part.f != out.f || part.t.size() != nframes) {
            throw std::invalid_argument(
                "Dataset spectrogram requires a common valid length across "
                "cubes; use power_spectrogram_by_cube() for ragged results.");
        }
        append_3d(out.sxx, row, std::move(part.sxx));
    }
    return out;
}

std::vector<signal::spectrogram_result>
sam_dataset::power_spectrogram_by_cube(size_t nperseg, size_t noverlap,
                                       double f_min, double f_max,
                                       signal::window_kind window) const {
    const double fs = require_samplerate() * 1e6;
    std::vector<signal::spectrogram_result> out;
    out.reserve(cube_shapes_.size());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        out.push_back(signal::spectrogram_psd(rows, fs, nperseg, noverlap,
                                              f_min, f_max, window));
    }
    return out;
}

std::vector<float> sam_dataset::stft_peak_frequency(
    size_t nperseg, size_t noverlap, double f_min, double f_max,
    signal::window_kind window) const {
    const double fs = require_samplerate() * 1e6;
    std::vector<float> out;
    out.reserve(num_samples());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        auto peaks = signal::stft_peaks(rows, fs, nperseg, noverlap, f_min,
                                        f_max, window);
        out.insert(out.end(), peaks.peak_frequency.begin(),
                   peaks.peak_frequency.end());
    }
    return out;
}

std::vector<float> sam_dataset::stft_peak_time(
    size_t nperseg, size_t noverlap, double f_min, double f_max,
    signal::window_kind window) const {
    const double fs = require_samplerate() * 1e6;
    std::vector<float> out;
    out.reserve(num_samples());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        auto peaks = signal::stft_peaks(rows, fs, nperseg, noverlap, f_min,
                                        f_max, window);
        out.insert(out.end(), peaks.peak_time.begin(), peaks.peak_time.end());
    }
    return out;
}

signal::spectrogram_features_result sam_dataset::spectrogram_features(
    const signal::spectrogram_feature_options& options,
    std::optional<double> fs_mhz) const {
    double fs = 0.0;
    if (fs_mhz.has_value()) {
        if (!(*fs_mhz > 0.0) || !std::isfinite(*fs_mhz)) {
            throw std::invalid_argument(
                "spectrogram_features: fs_mhz must be finite and positive.");
        }
        for (size_t i = 0; i < cube_samplerates_.size(); ++i) {
            const double a = cube_samplerates_[i];
            const double b = *fs_mhz;
            if (std::abs(a - b) > 1e-9 * std::max(1.0, std::abs(a))) {
                throw std::invalid_argument(
                    "spectrogram_features: fs_mhz=" + std::to_string(b) +
                    " contradicts cube " + std::to_string(i) +
                    " samplerate " + std::to_string(a) + " MHz.");
            }
        }
        fs = *fs_mhz;
    } else {
        fs = require_samplerate();
    }

    const auto min_valid_it =
        std::min_element(valid_scanlens_.begin(), valid_scanlens_.end());
    const auto max_valid_it =
        std::max_element(valid_scanlens_.begin(), valid_scanlens_.end());
    if (min_valid_it == valid_scanlens_.end() ||
        *min_valid_it != *max_valid_it) {
        throw std::invalid_argument(
            "Dataset spectrogram_features requires a common valid length "
            "across cubes.");
    }

    signal::spectrogram_features_result out;
    size_t row = 0;
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        auto part =
            signal::spectrogram_features(rows, fs * 1e6, options);
        if (i == 0) {
            out.f = part.f;
            out.t = part.t;
            out.features =
                array2d<float>(num_samples(), out.f.size() * out.t.size());
        } else if (part.f != out.f || part.t.size() != out.t.size()) {
            throw std::invalid_argument(
                "Dataset spectrogram_features cubes produced incompatible "
                "frequency/time axes.");
        }
        for (size_t r = 0; r < part.features.rows(); ++r, ++row) {
            std::copy(part.features[r].begin(), part.features[r].end(),
                      out.features[row].begin());
        }
    }
    return out;
}

signal::spectrum_result sam_dataset::spectrum() const {
    auto parts = spectrum_by_cube();
    if (parts.empty()) {
        throw std::runtime_error("Dataset is empty.");
    }
    signal::spectrum_result out;
    out.f = parts.front().f;
    out.mag = array2d<float>(num_samples(), out.f.size());
    size_t row = 0;
    for (auto& part : parts) {
        if (part.f != out.f || part.mag.cols() != out.f.size()) {
            throw std::invalid_argument(
                "Dataset spectrum requires a common valid length across "
                "cubes; use spectrum_by_cube() for ragged results.");
        }
        for (size_t r = 0; r < part.mag.rows(); ++r) {
            std::copy(part.mag[r].begin(), part.mag[r].end(),
                      out.mag[row + r].begin());
        }
        row += part.mag.rows();
    }
    return out;
}

std::vector<signal::spectrum_result> sam_dataset::spectrum_by_cube() const {
    const double fs = require_samplerate() * 1e6;
    std::vector<signal::spectrum_result> out;
    out.reserve(cube_shapes_.size());
    for (size_t i = 0; i < cube_shapes_.size(); ++i) {
        auto rows = cube_valid_rows(static_cast<std::int32_t>(i));
        out.push_back(signal::fft_spectrum(rows, 1.0 / fs));
    }
    return out;
}

std::map<std::string, size_t> sam_dataset::class_distribution() const {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    return labels_->class_distribution();
}

std::vector<std::int8_t> sam_dataset::to_binary(
    std::variant<std::int8_t, std::string> positive_label) const {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    return labels_->to_binary(std::move(positive_label));
}

array2d<float> sam_dataset::to_one_hot() const {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    return labels_->to_one_hot();
}

void sam_dataset::relabel(
    const std::map<std::variant<std::int64_t, std::string>,
                   std::variant<std::int64_t, std::string>>& mapping) {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    labels_->relabel(mapping);
}

array3d<float> sam_dataset::get_cube_X(std::int32_t idx) const {
    ensure_loaded();
    const auto [start, end] = cube_row_range_internal(idx);
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array3d<float> out(static_cast<size_t>(nlines), static_cast<size_t>(cols),
                       x_.cols());
    std::copy(x_.flat().begin() + static_cast<std::ptrdiff_t>(start * x_.cols()),
              x_.flat().begin() + static_cast<std::ptrdiff_t>(end * x_.cols()),
              out.flat().begin());
    return out;
}

array3d<float> sam_dataset::get_cube_Z(std::int32_t idx) const {
    ensure_loaded();
    if (!z_) {
        throw std::runtime_error("Z has not been built yet.");
    }
    const auto [start, end] = cube_row_range_internal(idx);
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array3d<float> out(static_cast<size_t>(nlines), static_cast<size_t>(cols),
                       z_->cols());
    std::copy(z_->flat().begin() + static_cast<std::ptrdiff_t>(start * z_->cols()),
              z_->flat().begin() + static_cast<std::ptrdiff_t>(end * z_->cols()),
              out.flat().begin());
    return out;
}

array3d<float> sam_dataset::get_cube_V(std::int32_t idx) const {
    ensure_loaded();
    if (!v_) {
        throw std::runtime_error("V has not been set.");
    }
    const auto [start, end] = cube_row_range_internal(idx);
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array3d<float> out(static_cast<size_t>(nlines), static_cast<size_t>(cols),
                       v_->cols());
    std::copy(v_->flat().begin() + static_cast<std::ptrdiff_t>(start * v_->cols()),
              v_->flat().begin() + static_cast<std::ptrdiff_t>(end * v_->cols()),
              out.flat().begin());
    return out;
}

array2d<std::int8_t> sam_dataset::get_cube_labels(std::int32_t idx) const {
    if (!labels_) {
        throw std::runtime_error(
            "This dataset is unsupervised. Labels are not available.");
    }
    const auto [start, end] = cube_row_range_internal(idx);
    (void)end;
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array2d<std::int8_t> out(static_cast<size_t>(nlines),
                             static_cast<size_t>(cols));
    size_t k = 0;
    for (size_t r = 0; r < static_cast<size_t>(nlines); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(cols); ++c) {
            out[r][c] = labels_->labels()[start + k++];
        }
    }
    return out;
}

} // namespace samcore

#include <samcore/sam_dataset.hpp>

#include <samcore/preprocessing.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace samcore {

sam_dataset::sam_dataset(std::vector<sam_scan> handlers, float pad_value,
                         std::optional<bool> unsupervised) {
    if (handlers.empty()) {
        throw std::invalid_argument(
            "SAMDataset must be initialized with at least one SAMHandler.");
    }

    std::vector<sam_labels> labels_list;
    std::vector<std::pair<std::int32_t, std::int32_t>> shapes;
    std::vector<double> resolutions;
    std::vector<std::int32_t> lens;
    std::vector<array2d<float>> parts;

    size_t maxlen = 0;
    for (const auto& h : handlers) {
        const size_t n = static_cast<size_t>(h.nlines() * h.cols());
        const size_t sl = static_cast<size_t>(h.scanlen());
        array2d<float> f(n, sl);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < sl; ++j) {
                f[i][j] = static_cast<float>(h.data()[i][j]);
            }
        }
        maxlen = std::max(maxlen, sl);
        parts.push_back(std::move(f));
        shapes.emplace_back(static_cast<std::int32_t>(h.nlines()),
                            static_cast<std::int32_t>(h.cols()));
        resolutions.push_back(h.header().resolution);
        lens.push_back(static_cast<std::int32_t>(sl));
        labels_list.push_back(h.samlabels().copy());
    }
    cube_shapes_ = std::move(shapes);
    cube_resolutions_ = std::move(resolutions);
    scanlens_ = std::move(lens);
    pad_value_ = pad_value;

    // Pad to the common length and concatenate.
    size_t total = 0;
    for (const auto& p : parts) total += p.rows();
    x_ = array2d<float>(total, maxlen);
    size_t offset = 0;
    for (auto& p : parts) {
        for (size_t i = 0; i < p.rows(); ++i) {
            auto dst = x_[offset + i];
            std::fill(dst.begin(), dst.end(), pad_value);
            std::copy(p[i].begin(), p[i].end(), dst.begin());
        }
        offset += p.rows();
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
        if (labels_list.size() >= 2) {
            labels_ = merge_labels(labels_list);
        } else {
            labels_ = labels_list[0].copy();
        }
    }

    train_indices_.resize(x_.rows());
    std::iota(train_indices_.begin(), train_indices_.end(), 0);
    test_indices_.clear();
    shuffled_ = false;
}

sam_dataset sam_dataset::copy() const {
    sam_dataset ds;
    ds.x_ = x_;
    if (labels_) {
        ds.labels_ = labels_->copy();
    }
    ds.cube_shapes_ = cube_shapes_;
    ds.cube_resolutions_ = cube_resolutions_;
    ds.scanlens_ = scanlens_;
    ds.pad_value_ = pad_value_;
    ds.unsupervised_ = unsupervised_;
    if (z_) {
        ds.z_ = z_;
    }
    if (v_) {
        ds.v_ = v_;
    }
    ds.train_indices_ = train_indices_;
    ds.test_indices_ = test_indices_;
    ds.shuffled_ = shuffled_;
    return ds;
}

void sam_dataset::save(const std::filesystem::path& path) const {
    io::write_h5samd(path, x_, labels_, cube_shapes_, cube_resolutions_,
                     scanlens_, unsupervised_, z_, v_);
}

sam_dataset sam_dataset::load(const std::filesystem::path& path) {
    io::h5samd_result res = io::read_h5samd(path);
    sam_dataset ds;
    ds.x_ = std::move(res.x);
    ds.labels_ = std::move(res.labels);
    ds.cube_shapes_ = std::move(res.cube_shapes);
    ds.cube_resolutions_ = std::move(res.cube_resolutions);
    ds.scanlens_ = std::move(res.scanlens);
    ds.unsupervised_ = res.unsupervised;
    ds.z_ = std::move(res.z);
    ds.v_ = std::move(res.v);
    ds.pad_value_ = 0.0f;
    ds.train_indices_.resize(ds.x_.rows());
    std::iota(ds.train_indices_.begin(), ds.train_indices_.end(), 0);
    ds.test_indices_.clear();
    ds.shuffled_ = false;
    return ds;
}

std::vector<spatial_record> sam_dataset::spatial() const {
    std::vector<spatial_record> out;
    out.reserve(x_.rows());
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
        x_ = preprocessing::gate(x_, args.start, args.end);
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

namespace {

// Rows belonging to cube idx: [start, end) within the concatenated X.
void cube_row_range(const sam_dataset& ds, std::int32_t idx,
                    size_t& start, size_t& end) {
    const auto& shapes = ds.cube_shapes();
    if (idx < 0 || static_cast<size_t>(idx) >= shapes.size()) {
        throw std::out_of_range("cube index out of range");
    }
    size_t offset = 0;
    for (std::int32_t i = 0; i < idx; ++i) {
        offset += static_cast<size_t>(shapes[static_cast<size_t>(i)].first) *
                  static_cast<size_t>(shapes[static_cast<size_t>(i)].second);
    }
    start = offset;
    end = offset + static_cast<size_t>(shapes[static_cast<size_t>(idx)].first) *
                       static_cast<size_t>(shapes[static_cast<size_t>(idx)].second);
}

} // namespace

array3d<float> sam_dataset::get_cube_X(std::int32_t idx) const {
    size_t start, end;
    cube_row_range(*this, idx, start, end);
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array3d<float> out(static_cast<size_t>(nlines), static_cast<size_t>(cols),
                       x_.cols());
    std::copy(x_.flat().begin() + static_cast<std::ptrdiff_t>(start * x_.cols()),
              x_.flat().begin() + static_cast<std::ptrdiff_t>(end * x_.cols()),
              out.flat().begin());
    return out;
}

array3d<float> sam_dataset::get_cube_Z(std::int32_t idx) const {
    if (!z_) {
        throw std::runtime_error("Z has not been built yet.");
    }
    size_t start, end;
    cube_row_range(*this, idx, start, end);
    const auto& [nlines, cols] = cube_shapes_[static_cast<size_t>(idx)];
    array3d<float> out(static_cast<size_t>(nlines), static_cast<size_t>(cols),
                       z_->cols());
    std::copy(z_->flat().begin() + static_cast<std::ptrdiff_t>(start * z_->cols()),
              z_->flat().begin() + static_cast<std::ptrdiff_t>(end * z_->cols()),
              out.flat().begin());
    return out;
}

array3d<float> sam_dataset::get_cube_V(std::int32_t idx) const {
    if (!v_) {
        throw std::runtime_error("V has not been set.");
    }
    size_t start, end;
    cube_row_range(*this, idx, start, end);
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
    size_t start, end;
    cube_row_range(*this, idx, start, end);
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

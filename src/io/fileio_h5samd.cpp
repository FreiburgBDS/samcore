#include <samcore/io/fileio.hpp>

#include <H5Cpp.h>

#include <samcore/constants.hpp>
#include <samcore/signal/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>

#include "h5_common.hpp"
#include "h5_lazy.hpp"

namespace samcore::io {

namespace {
using namespace detail;

bool read_unsupervised(H5::H5File& file) {
    if (!file.nameExists("unsupervised")) return false;
    H5::DataSet dset = file.openDataSet("unsupervised");
    std::uint8_t v = 0;
    dset.read(&v, H5::PredType::NATIVE_UINT8);
    return v != 0;
}

void write_unsupervised(H5::H5File& file, bool value) {
    const std::uint8_t v = value ? 1 : 0;
    H5::DataSpace space(H5S_SCALAR);
    H5::DSetCreatPropList plist;
    H5::DataSet dset =
        file.createDataSet("unsupervised", H5::PredType::NATIVE_UINT8, space, plist);
    dset.write(&v, H5::PredType::NATIVE_UINT8);
}

std::vector<std::pair<std::int32_t, std::int32_t>> read_cube_shapes(
    H5::H5File& file) {
    if (!file.nameExists("cube_shapes")) return {};
    H5::DataSet dset = file.openDataSet("cube_shapes");
    H5::DataSpace space = dset.getSpace();
    if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("'cube_shapes' dataset must be 2-D");
    }
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    const size_t n = static_cast<size_t>(dims[0]);
    std::vector<std::int32_t> buf(n * 2);
    if (n > 0) dset.read(buf.data(), H5::PredType::NATIVE_INT32);
    std::vector<std::pair<std::int32_t, std::int32_t>> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(buf[2 * i], buf[2 * i + 1]);
    }
    return out;
}

void write_cube_shapes(H5::H5File& file,
                       const std::vector<std::pair<std::int32_t, std::int32_t>>& shapes) {
    const hsize_t dims[2] = {static_cast<hsize_t>(shapes.size()), 2};
    H5::DataSpace space(2, dims);
    H5::DSetCreatPropList plist;
    std::vector<std::int32_t> buf;
    buf.reserve(shapes.size() * 2);
    for (const auto& [nl, nc] : shapes) {
        buf.push_back(nl);
        buf.push_back(nc);
    }
    H5::DataSet dset = file.createDataSet("cube_shapes", H5::PredType::NATIVE_INT32,
                                          space, plist);
    if (!buf.empty()) dset.write(buf.data(), H5::PredType::NATIVE_INT32);
}

void write_i32_1d(H5::H5File& file, const std::string& name,
                  const std::vector<std::int32_t>& values) {
    const hsize_t dims = values.size();
    H5::DataSpace space(1, &dims);
    H5::DSetCreatPropList plist;
    H5::DataSet dset =
        file.createDataSet(name, H5::PredType::NATIVE_INT32, space, plist);
    if (!values.empty()) {
        dset.write(values.data(), H5::PredType::NATIVE_INT32);
    }
}

void write_f64_1d(H5::H5File& file, const std::string& name,
                  const std::vector<double>& values) {
    const hsize_t dims = values.size();
    H5::DataSpace space(1, &dims);
    H5::DSetCreatPropList plist;
    H5::DataSet dset =
        file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space, plist);
    if (!values.empty()) {
        dset.write(values.data(), H5::PredType::NATIVE_DOUBLE);
    }
}

std::int64_t first_crossing_i8(std::span<const std::int8_t> signal,
                                double thresh_val) {
    for (size_t k = 0; k < signal.size(); ++k) {
        const auto a = static_cast<std::int8_t>(std::abs(signal[k]));
        if (static_cast<double>(a) >= thresh_val) {
            return static_cast<std::int64_t>(k);
        }
    }
    return -1;
}

// Shape of a 2-D dataset (same validation as detail::read_2d).
std::pair<size_t, size_t> dataset_dims(H5::DataSet& dset) {
    H5::DataSpace space = dset.getSpace();
    if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("expected a 2-D dataset");
    }
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    return {static_cast<size_t>(dims[0]), static_cast<size_t>(dims[1])};
}

// Everything except X/Z/V, shared by the eager and lazy readers.  x_cols is
// used for the uniform-scanlen fallback when the file has no `scanlens`.
struct h5samd_meta {
    std::optional<sam_labels> labels;
    std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes;
    std::vector<double> cube_resolutions;
    std::vector<std::int32_t> scanlens;
    std::vector<std::int32_t> valid_scanlens;
    std::vector<double> cube_samplerates;
    merge_mode merge = merge_mode::pad;
    float pad_value = 0.0f;
    bool unsupervised = false;
};

std::vector<double> read_f64_1d(H5::DataSet& dset) {
    H5::DataSpace space = dset.getSpace();
    std::vector<double> buf(static_cast<size_t>(space.getSimpleExtentNpoints()));
    if (!buf.empty()) dset.read(buf.data(), H5::PredType::NATIVE_DOUBLE);
    return buf;
}

h5samd_meta read_h5samd_meta(H5::H5File& file, size_t x_cols) {
    h5samd_meta meta;
    meta.unsupervised = read_unsupervised(file);

    if (file.attrExists("merge_mode")) {
        H5::Attribute attr = file.openAttribute("merge_mode");
        meta.merge = merge_mode_from_string(detail::read_string_attr(attr));
    }
    if (file.attrExists("pad_value")) {
        H5::Attribute attr = file.openAttribute("pad_value");
        attr.read(H5::PredType::NATIVE_FLOAT, &meta.pad_value);
    }

    std::vector<std::int8_t> labels_arr;
    if (file.nameExists("labels")) {
        H5::DataSet dset = file.openDataSet("labels");
        labels_arr.resize(static_cast<size_t>(dset.getSpace().getSimpleExtentNpoints()));
        if (!labels_arr.empty()) dset.read(labels_arr.data(), H5::PredType::NATIVE_INT8);
    }
    std::vector<std::string> label_names;
    if (file.nameExists("label_names")) {
        H5::DataSet ndset = file.openDataSet("label_names");
        label_names = detail::read_strings_1d(ndset);
    }
    if (!meta.unsupervised && !labels_arr.empty()) {
        meta.labels = sam_labels(std::move(labels_arr), std::move(label_names));
    }

    meta.cube_shapes = read_cube_shapes(file);

    if (file.nameExists("cube_resolutions")) {
        H5::DataSet dset = file.openDataSet("cube_resolutions");
        meta.cube_resolutions = read_f64_1d(dset);
    }

    if (file.nameExists("cube_samplerates")) {
        H5::DataSet dset = file.openDataSet("cube_samplerates");
        meta.cube_samplerates = read_f64_1d(dset);
    }

    if (file.nameExists("scanlens")) {
        H5::DataSet dset = file.openDataSet("scanlens");
        meta.scanlens = detail::read_i32_1d(dset);
    } else {
        // fallback: uniform scan length
        for (size_t i = 0; i < meta.cube_shapes.size(); ++i) {
            meta.scanlens.push_back(static_cast<std::int32_t>(x_cols));
        }
    }

    if (file.nameExists("valid_scanlens")) {
        H5::DataSet dset = file.openDataSet("valid_scanlens");
        meta.valid_scanlens = detail::read_i32_1d(dset);
    } else {
        // Legacy files stored every valid sample (padding only extends rows).
        meta.valid_scanlens = meta.scanlens;
    }
    return meta;
}

} // namespace

h5samd_result read_h5samd(const std::filesystem::path& path) {
    if (path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Path must end with .h5samd");
    }
    try {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);

        detail::warn_if_newer_format(file, "read_h5samd(" + path.string() + ")");

        h5samd_result result;
        size_t x_cols = 0;
        if (file.nameExists("X")) {
            H5::DataSet dset = file.openDataSet("X");
            result.x = detail::read_2d<float>(dset, H5::PredType::NATIVE_FLOAT);
            x_cols = result.x.cols();
        }

        h5samd_meta meta = read_h5samd_meta(file, x_cols);
        result.unsupervised = meta.unsupervised;
        result.labels = std::move(meta.labels);
        result.cube_shapes = std::move(meta.cube_shapes);
        result.cube_resolutions = std::move(meta.cube_resolutions);
        result.scanlens = std::move(meta.scanlens);
        result.valid_scanlens = std::move(meta.valid_scanlens);
        result.cube_samplerates = std::move(meta.cube_samplerates);
        result.merge = meta.merge;
        result.pad_value = meta.pad_value;

        if (file.nameExists("Z")) {
            H5::DataSet dset = file.openDataSet("Z");
            result.z = detail::read_2d<float>(dset, H5::PredType::NATIVE_FLOAT);
        }
        if (file.nameExists("V")) {
            H5::DataSet dset = file.openDataSet("V");
            result.v = detail::read_2d<float>(dset, H5::PredType::NATIVE_FLOAT);
        }
        return result;
    } catch (const H5::Exception&) {
        detail::rethrow_h5("read_h5samd(" + path.string() + ")");
    }
}

h5samd_lazy_handle read_h5samd_lazy(const std::filesystem::path& path) {
    if (path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Path must end with .h5samd");
    }
    try {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);
        detail::warn_if_newer_format(file, "read_h5samd_lazy(" + path.string() + ")");
        auto state = std::make_unique<h5samd_lazy_state>();

        if (file.nameExists("X")) {
            H5::DataSet dset = file.openDataSet("X");
            const auto [rows, cols] = dataset_dims(dset);
            state->x_rows = rows;
            state->x_cols = cols;
            state->x.emplace(file, std::move(dset), rows, cols,
                             H5::PredType::NATIVE_FLOAT);
        }

        h5samd_lazy_handle handle;
        h5samd_meta meta = read_h5samd_meta(file, state->x_cols);
        handle.unsupervised = meta.unsupervised;
        handle.labels = std::move(meta.labels);
        handle.cube_shapes = std::move(meta.cube_shapes);
        handle.cube_resolutions = std::move(meta.cube_resolutions);
        handle.scanlens = std::move(meta.scanlens);
        handle.valid_scanlens = std::move(meta.valid_scanlens);
        handle.cube_samplerates = std::move(meta.cube_samplerates);
        handle.merge = meta.merge;
        handle.pad_value = meta.pad_value;

        if (file.nameExists("Z")) {
            H5::DataSet dset = file.openDataSet("Z");
            state->z_cols = dataset_dims(dset).second;
            state->z.emplace(file, std::move(dset), state->x_rows,
                             state->z_cols, H5::PredType::NATIVE_FLOAT);
        }
        if (file.nameExists("V")) {
            H5::DataSet dset = file.openDataSet("V");
            state->v_cols = dataset_dims(dset).second;
            state->v.emplace(file, std::move(dset), state->x_rows,
                             state->v_cols, H5::PredType::NATIVE_FLOAT);
        }

        handle.data = std::move(state);
        return handle;
    } catch (const H5::Exception&) {
        detail::rethrow_h5("read_h5samd_lazy(" + path.string() + ")");
    }
}

h5samd_lazy_data read_h5samd_lazy_data(h5samd_lazy_state& state) {
    h5samd_lazy_data out;
    if (state.x) {
        out.x = array2d<float>(state.x->rows(), state.x->cols());
        state.x->read_rows(0, state.x->rows(), out.x.data());
    }
    if (state.z) {
        out.z = array2d<float>(state.z->rows(), state.z->cols());
        state.z->read_rows(0, state.z->rows(), out.z->data());
    }
    if (state.v) {
        out.v = array2d<float>(state.v->rows(), state.v->cols());
        state.v->read_rows(0, state.v->rows(), out.v->data());
    }
    return out;
}

void write_h5samd(const std::filesystem::path& path, const array2d<float>& x,
                  const std::optional<sam_labels>& labels,
                  const std::vector<std::pair<std::int32_t, std::int32_t>>& cube_shapes,
                  const std::vector<double>& cube_resolutions,
                  const std::vector<std::int32_t>& scanlens,
                  const std::vector<std::int32_t>& valid_scanlens,
                  const std::vector<double>& cube_samplerates,
                  merge_mode merge, float pad_value, bool unsupervised,
                  const std::optional<array2d<float>>& z,
                  const std::optional<array2d<float>>& v) {
    if (path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Path must end with .h5samd");
    }
    try {
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        detail::write_format_version(file);
        detail::write_string_attr(file, "merge_mode", to_string(merge));
        detail::write_double_attr(file, "pad_value",
                                  static_cast<double>(pad_value));

        detail::write_2d_gzip(file, "X", x, H5::PredType::NATIVE_FLOAT);

        const std::vector<std::int8_t>& label_vals =
            labels ? labels->labels() : std::vector<std::int8_t>{};
        {
            std::vector<std::int8_t> buf;
            if (labels) {
                buf.assign(label_vals.begin(), label_vals.end());
            } else {
                buf.assign(x.rows(), -1);
            }
            const hsize_t dims = buf.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            if (!buf.empty()) {
                const hsize_t chunk = std::min<hsize_t>(dims, 65536);
                plist.setChunk(1, &chunk);
                plist.setDeflate(3);
            }
            H5::DataSet dset = file.createDataSet("labels", H5::PredType::NATIVE_INT8,
                                                  space, plist);
            if (!buf.empty()) dset.write(buf.data(), H5::PredType::NATIVE_INT8);
        }

        detail::write_strings_1d(file, "label_names",
                                 labels ? labels->label_names()
                                        : std::vector<std::string>{});

        write_cube_shapes(file, cube_shapes);
        write_f64_1d(file, "cube_resolutions", cube_resolutions);
        write_f64_1d(file, "cube_samplerates", cube_samplerates);
        write_i32_1d(file, "scanlens", scanlens);
        write_i32_1d(file, "valid_scanlens", valid_scanlens);

        write_unsupervised(file, unsupervised);

        if (z.has_value()) {
            detail::write_2d_gzip(file, "Z", *z, H5::PredType::NATIVE_FLOAT);
        }
        if (v.has_value()) {
            detail::write_2d_gzip(file, "V", *v, H5::PredType::NATIVE_FLOAT);
        }
    } catch (const H5::Exception&) {
        detail::rethrow_h5("write_h5samd(" + path.string() + ")");
    }
}

void convert_h5sam_to_h5samd(
    const std::vector<std::filesystem::path>& input_paths,
    const std::filesystem::path& output_path,
    const dataset_merge_options& options,
    std::optional<bool> unsupervised) {
    if (output_path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Output path must end with .h5samd");
    }
    if (input_paths.empty()) {
        throw std::invalid_argument("At least one input path is required.");
    }
    for (const auto& p : input_paths) {
        const std::string ext = p.extension().string();
        if (ext != ".h5sam") {
            throw std::invalid_argument("Input path must be a .h5sam file: " +
                                        p.string());
        }
    }

    try {
        // Phase 1: collect metadata without loading signal data.
        std::vector<sam_labels> labels_list;
        std::vector<std::pair<std::int32_t, std::int32_t>> cube_shapes;
        std::vector<double> cube_resolutions;
        std::vector<double> cube_samplerates;
        std::vector<std::int32_t> scanlens;
        std::vector<std::int64_t> scan_counts;

        for (const auto& path : input_paths) {
            // Metadata-only open: header/labels/starts are read eagerly and
            // the signal dataset is not touched in this pass.
            io::h5sam_lazy_handle meta = io::read_h5sam_lazy(path);
            if (!meta.data) {
                throw std::runtime_error(
                    "convert_h5sam_to_h5samd: missing data handle for " +
                    path.string());
            }
            scan_counts.push_back(
                static_cast<std::int64_t>(meta.data->reader.rows()));
            scanlens.push_back(
                static_cast<std::int32_t>(meta.data->reader.cols()));
            cube_shapes.emplace_back(static_cast<std::int32_t>(meta.header.nlines),
                                     static_cast<std::int32_t>(meta.header.scanspline));
            cube_resolutions.push_back(meta.header.resolution);
            cube_samplerates.push_back(meta.header.samplerate);
            labels_list.push_back(std::move(meta.labels));
        }

        const std::int64_t total_signals =
            std::accumulate(scan_counts.begin(), scan_counts.end(), std::int64_t{0});
        const std::int64_t max_scanlen =
            *std::max_element(scanlens.begin(), scanlens.end());
        const std::int64_t min_scanlen =
            *std::min_element(scanlens.begin(), scanlens.end());

        sam_labels merged_labels =
            labels_list.size() >= 2 ? merge_labels(labels_list) : labels_list[0].copy();

        bool unsup = unsupervised.value_or(
            !std::all_of(labels_list.begin(), labels_list.end(),
                         [](const sam_labels& l) { return l.is_labeled(); }));
        if (!unsup) {
            for (size_t i = 0; i < labels_list.size(); ++i) {
                if (!labels_list[i].is_labeled()) {
                    throw std::invalid_argument(
                        "Cube " + std::to_string(i) +
                        " is unlabeled but dataset is supervised. Either set "
                        "unsupervised=true or ensure all cubes have labels.");
                }
            }
        }

        // Resolve the global alignment plan.  crop/pad/zgate widths are known
        // from metadata; xcorr needs a planning pass over the signals.
        std::int64_t out_cols = 0;
        std::vector<std::vector<std::int32_t>> cube_offsets(input_paths.size());
        if (options.mode == merge_mode::crop) {
            out_cols = min_scanlen;
        } else if (options.mode == merge_mode::pad) {
            out_cols = max_scanlen;
        } else if (options.mode == merge_mode::zgate) {
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
        } else {
            if (options.reference >= static_cast<size_t>(total_signals)) {
                throw std::out_of_range(
                    "xcorr reference row is out of range.");
            }
            if (options.max_gate_loss <= 0.0 || options.max_gate_loss > 1.0) {
                throw std::invalid_argument(
                    "max_gate_loss must be in (0, 1].");
            }
            const size_t common = static_cast<size_t>(min_scanlen);
            std::int64_t max_shift = options.max_shift;
            if (max_shift == 0) {
                max_shift = static_cast<std::int64_t>(common / 4);
            }
            if (max_shift < 0 || static_cast<size_t>(max_shift) >= common) {
                throw std::invalid_argument(
                    "max_shift must be in [0, min_scanlen).");
            }

            std::size_t ref_cube = 0;
            std::size_t ref_row = options.reference;
            while (ref_row >= static_cast<size_t>(scan_counts[ref_cube])) {
                ref_row -= static_cast<size_t>(scan_counts[ref_cube]);
                ++ref_cube;
            }
            h5sam_result ref_res = read_h5sam(input_paths[ref_cube]);
            std::vector<std::int8_t> ref_buf(common);
            std::copy_n(ref_res.data[ref_row].begin(),
                        static_cast<std::ptrdiff_t>(common), ref_buf.begin());
            const std::span<const std::int8_t> ref(ref_buf.data(), common);

            std::int32_t min_shift = 0;
            std::int32_t max_shift_v = 0;
            bool first_shift = true;
            for (size_t i = 0; i < input_paths.size(); ++i) {
                h5sam_result res = read_h5sam(input_paths[i]);
                cube_offsets[i].resize(static_cast<size_t>(scan_counts[i]));
                for (size_t r = 0; r < cube_offsets[i].size(); ++r) {
                    const auto& src = res.data[r];
                    const std::int32_t lag = static_cast<std::int32_t>(
                        signal::xcorr_lag(
                            std::span<const std::int8_t>(src.data(), common),
                            ref, max_shift));
                    cube_offsets[i][r] = lag;
                    if (first_shift) {
                        min_shift = max_shift_v = lag;
                        first_shift = false;
                    } else {
                        min_shift = std::min(min_shift, lag);
                        max_shift_v = std::max(max_shift_v, lag);
                    }
                }
            }
            const std::int64_t spread =
                static_cast<std::int64_t>(max_shift_v) - min_shift;
            if (static_cast<double>(spread) >
                options.max_gate_loss * static_cast<double>(common)) {
                throw std::invalid_argument(
                    "xcorr: aligned shift spread exceeds max_gate_loss.");
            }
            out_cols = std::numeric_limits<std::int64_t>::max();
            for (size_t i = 0; i < input_paths.size(); ++i) {
                std::int32_t max_offset = 0;
                for (auto& offset : cube_offsets[i]) {
                    offset -= min_shift;
                    max_offset = std::max(max_offset, offset);
                }
                out_cols = std::min(out_cols,
                                    static_cast<std::int64_t>(scanlens[i]) -
                                        max_offset);
            }
            if (out_cols <= 0) {
                throw std::invalid_argument(
                    "xcorr: lag spread exceeds every source scan length.");
            }
        }

        if (out_cols <= 0) {
            throw std::invalid_argument("Merged dataset has zero signal length.");
        }

        std::vector<std::int32_t> valid_scanlens(scanlens.size());
        for (size_t i = 0; i < scanlens.size(); ++i) {
            valid_scanlens[i] = options.mode == merge_mode::pad
                                    ? scanlens[i]
                                    : static_cast<std::int32_t>(out_cols);
        }

        // Phase 2: create the output and stream data per file.
        H5::H5File out(output_path.string(), H5F_ACC_TRUNC);
        detail::write_format_version(out);
        detail::write_string_attr(out, "merge_mode", to_string(options.mode));
        detail::write_double_attr(out, "pad_value",
                                  static_cast<double>(options.pad_value));

        {
            const hsize_t dims[2] = {static_cast<hsize_t>(total_signals),
                                     static_cast<hsize_t>(out_cols)};
            H5::DataSpace space(2, dims);
            H5::DSetCreatPropList plist =
                detail::gzip_plist(static_cast<size_t>(total_signals),
                                   static_cast<size_t>(out_cols), sizeof(float));
            H5::DataSet dset = out.createDataSet("X", H5::PredType::NATIVE_FLOAT,
                                                 space, plist);
            std::int64_t offset = 0;
            for (size_t i = 0; i < input_paths.size(); ++i) {
                h5sam_result res = read_h5sam(input_paths[i]);
                const std::int64_t n = static_cast<std::int64_t>(res.data.rows());
                std::vector<float> cube(static_cast<size_t>(n * out_cols),
                                        options.pad_value);
                const double thresh_val = options.threshold * full_scale;
                for (std::int64_t r = 0; r < n; ++r) {
                    float* dst = cube.data() + static_cast<std::ptrdiff_t>(
                                                  r * out_cols);
                    const auto src = res.data[static_cast<size_t>(r)];
                    if (options.mode == merge_mode::pad) {
                        for (std::int64_t c = 0; c < scanlens[i]; ++c) {
                            dst[c] = static_cast<float>(src[static_cast<size_t>(c)]);
                        }
                    } else if (options.mode == merge_mode::zgate) {
                        std::int64_t start = first_crossing_i8(src, thresh_val);
                        if (start >= 0) {
                            start = std::min(start, scanlens[i] - out_cols);
                            for (std::int64_t c = 0; c < out_cols; ++c) {
                                dst[c] = static_cast<float>(
                                    src[static_cast<size_t>(start + c)]);
                            }
                        } else {
                            std::fill(dst, dst + out_cols, 0.0f);
                        }
                    } else {
                        const std::int64_t start =
                            options.mode == merge_mode::xcorr
                                ? cube_offsets[i][static_cast<size_t>(r)]
                                : 0;
                        for (std::int64_t c = 0; c < out_cols; ++c) {
                            dst[c] = static_cast<float>(
                                src[static_cast<size_t>(start + c)]);
                        }
                    }
                }
                const hsize_t sel[2] = {static_cast<hsize_t>(n),
                                        static_cast<hsize_t>(out_cols)};
                const hsize_t off[2] = {static_cast<hsize_t>(offset), 0};
                H5::DataSpace fspace = dset.getSpace();
                fspace.selectHyperslab(H5S_SELECT_SET, sel, off);
                H5::DataSpace mspace(2, sel);
                dset.write(cube.data(), H5::PredType::NATIVE_FLOAT, mspace, fspace);
                offset += n;
            }
        }

        {
            std::vector<std::int8_t> label_vals(merged_labels.labels().begin(),
                                                merged_labels.labels().end());
            const hsize_t dims = label_vals.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            if (!label_vals.empty()) {
                const hsize_t chunk = std::min<hsize_t>(dims, 65536);
                plist.setChunk(1, &chunk);
                plist.setDeflate(3);
            }
            H5::DataSet dset = out.createDataSet("labels", H5::PredType::NATIVE_INT8,
                                                 space, plist);
            if (!label_vals.empty()) dset.write(label_vals.data(), H5::PredType::NATIVE_INT8);
        }

        detail::write_strings_1d(out, "label_names", merged_labels.label_names());
        write_cube_shapes(out, cube_shapes);
        write_f64_1d(out, "cube_resolutions", cube_resolutions);
        write_f64_1d(out, "cube_samplerates", cube_samplerates);
        write_i32_1d(out, "scanlens", scanlens);
        write_i32_1d(out, "valid_scanlens", valid_scanlens);
        write_unsupervised(out, unsup);
    } catch (const H5::Exception&) {
        detail::rethrow_h5("convert_h5sam_to_h5samd(" + output_path.string() + ")");
    }
}

} // namespace samcore::io

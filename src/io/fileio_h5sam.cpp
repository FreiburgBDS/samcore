#include <samcore/io/fileio.hpp>

#include <H5Cpp.h>

#include <algorithm>
#include <cstring>
#include <map>

#include "h5_common.hpp"
#include "h5_lazy.hpp"

namespace samcore::io {

namespace {
using namespace detail;

void require_extension(const std::filesystem::path& path,
                       const std::vector<std::string>& exts) {
    std::string ext = path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const auto& e : exts) {
        if (ext == e) return;
    }
    throw std::invalid_argument("Unsupported file extension for path: " +
                                path.string());
}

// Typed coercions over attribute values, mirroring the header parser's
// behavior (numeric fields accept any scalar; bools are truthy ints).

std::int64_t as_int(const extra_value& v) {
    if (std::holds_alternative<bool>(v)) return std::get<bool>(v) ? 1 : 0;
    if (std::holds_alternative<std::int64_t>(v)) return std::get<std::int64_t>(v);
    if (std::holds_alternative<double>(v)) return static_cast<std::int64_t>(std::get<double>(v));
    return 0;
}

double as_double(const extra_value& v) {
    if (std::holds_alternative<double>(v)) return std::get<double>(v);
    if (std::holds_alternative<std::int64_t>(v)) return static_cast<double>(std::get<std::int64_t>(v));
    if (std::holds_alternative<bool>(v)) return std::get<bool>(v) ? 1.0 : 0.0;
    return 0.0;
}

bool as_bool(const extra_value& v) {
    if (std::holds_alternative<bool>(v)) return std::get<bool>(v);
    return as_int(v) != 0;
}

const std::string* as_string(const extra_value& v) {
    return std::get_if<std::string>(&v);
}

sam_header read_header_group(H5::Group& group) {
    sam_header header;
    const int n = group.getNumAttrs();
    for (int i = 0; i < n; ++i) {
        H5::Attribute attr = group.openAttribute(static_cast<unsigned>(i));
        const std::string name = attr.getName();
        if (name == "headerlen" || name == "bytes_p_sample" ||
            name == "version") {
            continue; // backwards compat: ignore legacy fields
        }
        extra_value v = read_scalar_attr(attr);
        if (name == "scanspline") {
            header.scanspline = as_int(v);
        } else if (name == "nlines") {
            header.nlines = as_int(v);
        } else if (name == "scanlen") {
            header.scanlen = as_int(v);
        } else if (name == "samplerate") {
            header.samplerate = as_double(v);
        } else if (name == "tzero") {
            header.tzero = as_int(v);
        } else if (name == "resolution") {
            header.resolution = as_double(v);
        } else if (name == "interpolated") {
            header.interpolated = as_bool(v);
        } else if (name == "quality") {
            header.quality = as_bool(v);
        } else if (name == "mode") {
            if (const auto* s = as_string(v)) header.mode = *s;
        } else if (name == "transducer_in") {
            if (const auto* s = as_string(v)) header.transducer_in = *s;
        } else if (name == "transducer_through") {
            if (const auto* s = as_string(v)) header.transducer_through = *s;
        } else if (name == "cellid") {
            if (const auto* s = as_string(v)) header.cellid = *s;
        } else if (name == "downsample_factor") {
            header.downsample_factor = as_int(v);
        } else {
            header.extra[name] = std::move(v);
        }
    }
    return header;
}

void write_header_group(H5::Group& group, const sam_header& header) {
    write_int_attr(group, "scanspline", header.scanspline);
    write_int_attr(group, "nlines", header.nlines);
    write_bool_attr(group, "interpolated", header.interpolated);
    write_int_attr(group, "scanlen", header.scanlen);
    write_double_attr(group, "samplerate", header.samplerate);
    write_int_attr(group, "tzero", header.tzero);
    write_bool_attr(group, "quality", header.quality);
    write_double_attr(group, "resolution", header.resolution);
    write_string_attr(group, "mode", header.mode);
    write_string_attr(group, "transducer_in", header.transducer_in);
    write_string_attr(group, "transducer_through", header.transducer_through);
    write_string_attr(group, "cellid", header.cellid);
    write_int_attr(group, "downsample_factor", header.downsample_factor);
    for (const auto& [key, value] : header.extra) {
        if (std::holds_alternative<std::int64_t>(value)) {
            write_int_attr(group, key, std::get<std::int64_t>(value));
        } else if (std::holds_alternative<double>(value)) {
            write_double_attr(group, key, std::get<double>(value));
        } else if (std::holds_alternative<bool>(value)) {
            write_bool_attr(group, key, std::get<bool>(value));
        } else {
            write_string_attr(group, key, std::get<std::string>(value));
        }
    }
}

sam_labels read_labels_datasets(H5::H5File& file, size_t n_signals) {
    if (!file.nameExists("labels")) {
        return sam_labels::create_unlabeled(n_signals);
    }
    H5::DataSet dset = file.openDataSet("labels");
    std::vector<std::int8_t> labels(static_cast<size_t>(
        dset.getSpace().getSimpleExtentNpoints()));
    if (!labels.empty()) dset.read(labels.data(), H5::PredType::NATIVE_INT8);

    std::vector<std::string> names;
    if (file.nameExists("label_names")) {
        H5::DataSet ndset = file.openDataSet("label_names");
        names = read_strings_1d(ndset);
    }
    if (labels.empty()) {
        return sam_labels::create_unlabeled(n_signals);
    }
    return sam_labels(std::move(labels), std::move(names));
}

} // namespace

h5sam_lazy_handle read_h5sam_lazy(const std::filesystem::path& path) {
    require_extension(path, {".h5sam"});
    try {
        h5sam_lazy_handle handle;
        auto state = std::make_unique<h5sam_lazy_state>();

        H5::H5File file(path.string(), H5F_ACC_RDONLY);
        H5::Group header_group = file.openGroup("header");
        handle.header = read_header_group(header_group);

        H5::DataSet dset = file.openDataSet("data");
        H5::DataSpace space = dset.getSpace();
        if (space.getSimpleExtentNdims() != 2) {
            throw std::runtime_error("'data' dataset must be 2-D");
        }
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);
        state->rows = static_cast<size_t>(dims[0]);
        state->cols = static_cast<size_t>(dims[1]);

        handle.labels = read_labels_datasets(file, state->rows);
        if (file.nameExists("starts")) {
            H5::DataSet sset = file.openDataSet("starts");
            handle.starts = read_i32_1d(sset);
        }

        state->file = std::move(file);
        state->dset = std::move(dset);
        handle.data = std::move(state);
        return handle;
    } catch (const H5::Exception&) {
        detail::rethrow_h5("read_h5sam_lazy(" + path.string() + ")");
    }
}

h5sam_result read_h5sam(const std::filesystem::path& path) {
    require_extension(path, {".h5sam"});
    try {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);
        H5::Group header_group = file.openGroup("header");
        sam_header header = read_header_group(header_group);

        H5::DataSet dset = file.openDataSet("data");
        H5::DataSpace space = dset.getSpace();
        if (space.getSimpleExtentNdims() != 2) {
            throw std::runtime_error("'data' dataset must be 2-D");
        }
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);

        array2d<std::int8_t> data(static_cast<size_t>(dims[0]),
                                  static_cast<size_t>(dims[1]));
        if (dims[0] > 0 && dims[1] > 0) {
            dset.read(data.data(), H5::PredType::NATIVE_INT8);
        }

        sam_labels labels = read_labels_datasets(file, data.rows());

        std::optional<std::vector<std::int32_t>> starts;
        if (file.nameExists("starts")) {
            H5::DataSet sset = file.openDataSet("starts");
            starts = read_i32_1d(sset);
        }

        return h5sam_result{std::move(data), std::move(header),
                            std::move(labels), std::move(starts)};
    } catch (const H5::Exception&) {
        detail::rethrow_h5("read_h5sam(" + path.string() + ")");
    }
}

void write_h5sam(const std::filesystem::path& path,
                 const array2d<std::int8_t>& data, const sam_header& header,
                 const sam_labels& labels,
                 const std::optional<std::vector<std::int32_t>>& starts) {
    require_extension(path, {".h5sam"});
    try {
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

        H5::Group header_group = file.createGroup("header");
        write_header_group(header_group, header);

        write_2d_gzip(file, "data", data, H5::PredType::NATIVE_INT8);

        {
            std::vector<std::int8_t> label_buf(labels.labels().begin(),
                                               labels.labels().end());
            const hsize_t dims = label_buf.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            if (!label_buf.empty()) {
                const hsize_t chunk = std::min<hsize_t>(dims, 65536);
                plist.setChunk(1, &chunk);
                plist.setDeflate(3);
            }
            H5::DataSet dset = file.createDataSet("labels", H5::PredType::NATIVE_INT8,
                                                  space, plist);
            if (!label_buf.empty()) dset.write(label_buf.data(), H5::PredType::NATIVE_INT8);
        }

        write_strings_1d(file, "label_names", labels.label_names());

        if (starts.has_value()) {
            const auto& s = *starts;
            const hsize_t dims = s.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            if (!s.empty()) {
                const hsize_t chunk = std::min<hsize_t>(dims, 65536);
                plist.setChunk(1, &chunk);
                plist.setDeflate(3);
            }
            H5::DataSet dset = file.createDataSet("starts", H5::PredType::NATIVE_INT32,
                                                  space, plist);
            if (!s.empty()) dset.write(s.data(), H5::PredType::NATIVE_INT32);
        }
    } catch (const H5::Exception&) {
        detail::rethrow_h5("write_h5sam(" + path.string() + ")");
    }
}

array2d<std::int8_t> read_h5sam_rows(const std::filesystem::path& path,
                                     size_t first, size_t count) {
    try {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);
        H5::DataSet dset = file.openDataSet("data");
        H5::DataSpace fspace = dset.getSpace();
        if (fspace.getSimpleExtentNdims() != 2) {
            throw std::runtime_error("'data' dataset must be 2-D");
        }
        hsize_t dims[2];
        fspace.getSimpleExtentDims(dims);
        if (first + count > dims[0]) {
            throw std::invalid_argument("read_h5sam_rows: range out of bounds");
        }
        const hsize_t offset[2] = {static_cast<hsize_t>(first), 0};
        const hsize_t sel[2] = {static_cast<hsize_t>(count), dims[1]};
        fspace.selectHyperslab(H5S_SELECT_SET, sel, offset);

        array2d<std::int8_t> out(count, static_cast<size_t>(dims[1]));
        if (count > 0 && dims[1] > 0) {
            H5::DataSpace mspace(2, sel);
            dset.read(out.data(), H5::PredType::NATIVE_INT8, mspace, fspace);
        }
        return out;
    } catch (const H5::Exception&) {
        detail::rethrow_h5("read_h5sam_rows(" + path.string() + ")");
    }
}

} // namespace samcore::io

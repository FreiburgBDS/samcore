#include <samcore/io/fileio.hpp>

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "h5_common.hpp"

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

} // namespace

h5samd_result read_h5samd(const std::filesystem::path& path) {
    if (path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Path must end with .h5samd");
    }
    try {
        H5::H5File file(path.string(), H5F_ACC_RDONLY);

        h5samd_result result;
        if (file.nameExists("X")) {
            H5::DataSet dset = file.openDataSet("X");
            result.x = detail::read_2d<float>(dset, H5::PredType::NATIVE_FLOAT);
        }

        result.unsupervised = read_unsupervised(file);

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
        if (!result.unsupervised && !labels_arr.empty()) {
            result.labels = sam_labels(std::move(labels_arr), std::move(label_names));
        }

        result.cube_shapes = read_cube_shapes(file);

        if (file.nameExists("cube_resolutions")) {
            H5::DataSet dset = file.openDataSet("cube_resolutions");
            H5::DataSpace space = dset.getSpace();
            std::vector<double> buf(static_cast<size_t>(space.getSimpleExtentNpoints()));
            if (!buf.empty()) dset.read(buf.data(), H5::PredType::NATIVE_DOUBLE);
            result.cube_resolutions = std::move(buf);
        }

        if (file.nameExists("scanlens")) {
            H5::DataSet dset = file.openDataSet("scanlens");
            result.scanlens = detail::read_i32_1d(dset);
        } else {
            // fallback: uniform scan length
            for (size_t i = 0; i < result.cube_shapes.size(); ++i) {
                result.scanlens.push_back(static_cast<std::int32_t>(result.x.cols()));
            }
        }

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

void write_h5samd(const std::filesystem::path& path, const array2d<float>& x,
                  const std::optional<sam_labels>& labels,
                  const std::vector<std::pair<std::int32_t, std::int32_t>>& cube_shapes,
                  const std::vector<double>& cube_resolutions,
                  const std::vector<std::int32_t>& scanlens, bool unsupervised,
                  const std::optional<array2d<float>>& z,
                  const std::optional<array2d<float>>& v) {
    if (path.extension().string() != ".h5samd") {
        throw std::invalid_argument("Path must end with .h5samd");
    }
    try {
        H5::H5File file(path.string(), H5F_ACC_TRUNC);

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

        {
            const hsize_t dims = cube_resolutions.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            H5::DataSet dset = file.createDataSet("cube_resolutions",
                                                  H5::PredType::NATIVE_DOUBLE, space, plist);
            if (!cube_resolutions.empty()) {
                dset.write(cube_resolutions.data(), H5::PredType::NATIVE_DOUBLE);
            }
        }

        {
            const hsize_t dims = scanlens.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            H5::DataSet dset = file.createDataSet("scanlens", H5::PredType::NATIVE_INT32,
                                                  space, plist);
            if (!scanlens.empty()) dset.write(scanlens.data(), H5::PredType::NATIVE_INT32);
        }

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
    const std::filesystem::path& output_path, float pad_value,
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
        std::vector<std::int32_t> scanlens;
        std::vector<std::int64_t> scan_counts;

        for (const auto& path : input_paths) {
            h5sam_result res = read_h5sam(path);
            scan_counts.push_back(static_cast<std::int64_t>(res.data.rows()));
            scanlens.push_back(static_cast<std::int32_t>(res.data.cols()));
            cube_shapes.emplace_back(static_cast<std::int32_t>(res.header.nlines),
                                     static_cast<std::int32_t>(res.header.scanspline));
            cube_resolutions.push_back(res.header.resolution);
            labels_list.push_back(std::move(res.labels));
        }

        const std::int64_t total_signals =
            std::accumulate(scan_counts.begin(), scan_counts.end(), std::int64_t{0});
        const std::int64_t max_scanlen =
            *std::max_element(scanlens.begin(), scanlens.end());

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

        // Phase 2: create the output and stream data per file.
        H5::H5File out(output_path.string(), H5F_ACC_TRUNC);

        {
            const hsize_t dims[2] = {static_cast<hsize_t>(total_signals),
                                     static_cast<hsize_t>(max_scanlen)};
            H5::DataSpace space(2, dims);
            H5::DSetCreatPropList plist =
                detail::gzip_plist(static_cast<size_t>(total_signals),
                                   static_cast<size_t>(max_scanlen), sizeof(float));
            H5::DataSet dset = out.createDataSet("X", H5::PredType::NATIVE_FLOAT,
                                                 space, plist);
            std::int64_t offset = 0;
            for (size_t i = 0; i < input_paths.size(); ++i) {
                h5sam_result res = read_h5sam(input_paths[i]);
                const std::int64_t n = static_cast<std::int64_t>(res.data.rows());
                const std::int64_t sl = static_cast<std::int64_t>(res.data.cols());
                std::vector<float> cube(n * max_scanlen, pad_value);
                for (std::int64_t r = 0; r < n; ++r) {
                    for (std::int64_t c = 0; c < sl; ++c) {
                        cube[static_cast<size_t>(r * max_scanlen + c)] =
                            static_cast<float>(res.data[static_cast<size_t>(r)]
                                                       [static_cast<size_t>(c)]);
                    }
                }
                const hsize_t sel[2] = {static_cast<hsize_t>(n),
                                        static_cast<hsize_t>(max_scanlen)};
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

        {
            const hsize_t dims = cube_resolutions.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            H5::DataSet dset = out.createDataSet("cube_resolutions",
                                                 H5::PredType::NATIVE_DOUBLE, space, plist);
            if (!cube_resolutions.empty()) {
                dset.write(cube_resolutions.data(), H5::PredType::NATIVE_DOUBLE);
            }
        }
        {
            const hsize_t dims = scanlens.size();
            H5::DataSpace space(1, &dims);
            H5::DSetCreatPropList plist;
            H5::DataSet dset = out.createDataSet("scanlens", H5::PredType::NATIVE_INT32,
                                                 space, plist);
            if (!scanlens.empty()) dset.write(scanlens.data(), H5::PredType::NATIVE_INT32);
        }
        write_unsupervised(out, unsup);
    } catch (const H5::Exception&) {
        detail::rethrow_h5("convert_h5sam_to_h5samd(" + output_path.string() + ")");
    }
}

} // namespace samcore::io

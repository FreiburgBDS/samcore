#pragma once

// Internal HDF5 helpers shared by the h5sam / h5samd readers and writers.
// Not installed; not part of the public API.

#include <H5Cpp.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <samcore/sam_header.hpp>

namespace samcore::io::detail {

[[noreturn]] inline void rethrow_h5(const std::string& ctx) {
    try {
        throw;
    } catch (const H5::Exception& e) {
        const std::string msg = e.getDetailMsg();
        throw std::runtime_error(ctx + ": " +
                                 (msg.empty() ? std::string(e.getFuncName())
                                              : msg));
    } catch (...) {
        throw;
    }
}

// attribute helpers

// Read a string attribute, handling both fixed- and variable-length storage.
// Reads with the file's own datatype as the memory type: HDF5 has no
// conversion path between string types with different character sets
// (h5py writes UTF-8), so identity conversion is the only portable route.
[[nodiscard]] inline std::string read_string_attr(H5::Attribute& attr) {
    H5::DataType t = attr.getDataType();
    if (H5Tis_variable_str(t.getId())) {
        char* buf = nullptr;
        attr.read(t, &buf);
        std::string s(buf ? buf : "");
        if (buf) H5free_memory(buf);
        return s;
    }
    const size_t size = t.getSize();
    std::vector<char> buf(size, '\0');
    attr.read(t, buf.data());
    const char* begin = buf.data();
    const char* end = begin + size;
    while (end > begin && end[-1] == '\0') --end;
    return std::string(begin, end);
}

// Read an integer/float/bool/string attribute into a typed extra value.
// h5py stores Python bools as HDF5 enums; reading them as integers works
// via HDF5 type conversion, so bools arrive as int64 (matching the
// pre-existing behavior).
[[nodiscard]] inline extra_value read_scalar_attr(H5::Attribute& attr) {
    H5::DataType t = attr.getDataType();
    const H5T_class_t cls = t.getClass();
    if (cls == H5T_STRING) {
        return read_string_attr(attr);
    }
    if (cls == H5T_FLOAT) {
        double v = 0.0;
        attr.read(H5::PredType::NATIVE_DOUBLE, &v);
        return v;
    }
    // INTEGER, ENUM (bool), BITFIELD.
    std::int64_t v = 0;
    try {
        attr.read(H5::PredType::NATIVE_INT64, &v);
    } catch (const H5::Exception&) {
        std::uint8_t b = 0;
        attr.read(H5::PredType::NATIVE_UINT8, &b);
        v = b;
    }
    return v;
}

// Write a string attribute as variable-length UTF-8, exactly like h5py.
inline void write_string_attr(H5::Group& group, const std::string& name,
                              const std::string& value) {
    H5::DataSpace dspace(H5S_SCALAR);
    H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
    H5Tset_cset(stype.getId(), H5T_CSET_UTF8);
    H5::Attribute attr = group.createAttribute(name, stype, dspace);
    const char* cstr = value.c_str();
    attr.write(stype, &cstr);
}

inline void write_int_attr(H5::Group& group, const std::string& name,
                           std::int64_t value) {
    H5::DataSpace dspace(H5S_SCALAR);
    H5::Attribute attr =
        group.createAttribute(name, H5::PredType::NATIVE_INT64, dspace);
    attr.write(H5::PredType::NATIVE_INT64, &value);
}

inline void write_double_attr(H5::Group& group, const std::string& name,
                              double value) {
    H5::DataSpace dspace(H5S_SCALAR);
    H5::Attribute attr =
        group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, dspace);
    attr.write(H5::PredType::NATIVE_DOUBLE, &value);
}

inline void write_bool_attr(H5::Group& group, const std::string& name,
                            bool value) {
    // Stored as uint8; h5py reads it back as np.uint8, which is
    // tolerated for boolean header fields.
    std::uint8_t v = value ? 1 : 0;
    H5::DataSpace dspace(H5S_SCALAR);
    H5::Attribute attr =
        group.createAttribute(name, H5::PredType::NATIVE_UINT8, dspace);
    attr.write(H5::PredType::NATIVE_UINT8, &v);
}

// dataset helpers

// Chunk rows so that rows*cols*itemsize is roughly chunk_target bytes.
[[nodiscard]] inline hsize_t chunk_rows(size_t rows, size_t cols,
                                        size_t itemsize,
                                        size_t chunk_target = 1024 * 1024) {
    if (rows == 0) return 1;
    const size_t row_bytes = cols * itemsize;
    if (row_bytes == 0) return 1;
    hsize_t cr = static_cast<hsize_t>(chunk_target / row_bytes);
    if (cr < 1) cr = 1;
    if (cr > rows) cr = rows;
    return cr;
}

[[nodiscard]] inline H5::DSetCreatPropList gzip_plist(size_t rows, size_t cols,
                                                      size_t itemsize) {
    H5::DSetCreatPropList plist;
    const hsize_t cr = chunk_rows(rows, cols, itemsize);
    const hsize_t chunk[2] = {cr, static_cast<hsize_t>(cols)};
    plist.setChunk(2, chunk);
    plist.setDeflate(3);
    return plist;
}

[[nodiscard]] inline std::vector<std::int32_t> read_i32_1d(H5::DataSet& dset) {
    H5::DataSpace space = dset.getSpace();
    const hsize_t n = space.getSimpleExtentNpoints();
    std::vector<std::int32_t> out(n);
    if (n > 0) dset.read(out.data(), H5::PredType::NATIVE_INT32);
    return out;
}

[[nodiscard]] inline std::vector<std::string> read_strings_1d(
    H5::DataSet& dset) {
    H5::DataSpace space = dset.getSpace();
    const hsize_t n = space.getSimpleExtentNpoints();
    H5::DataType t = dset.getDataType();
    if (t.getClass() != H5T_STRING) {
        throw std::runtime_error("expected a string dataset");
    }
    std::vector<std::string> out;
    if (n == 0) return out;
    out.reserve(n);
    // Identity-conversion reads (see read_string_attr for the cset rationale).
    if (H5Tis_variable_str(t.getId())) {
        std::vector<char*> buf(static_cast<size_t>(n));
        dset.read(buf.data(), t);
        for (size_t i = 0; i < n; ++i) {
            out.emplace_back(buf[i] ? buf[i] : "");
            if (buf[i]) H5free_memory(buf[i]);
        }
    } else {
        const size_t size = t.getSize();
        std::vector<char> buf(static_cast<size_t>(n) * size);
        dset.read(buf.data(), t);
        for (size_t i = 0; i < n; ++i) {
            const char* begin = buf.data() + i * size;
            const char* end = begin + size;
            while (end > begin && end[-1] == '\0') --end;
            out.emplace_back(begin, end);
        }
    }
    return out;
}

inline void write_strings_1d(H5::H5File& file, const std::string& name,
                             const std::vector<std::string>& values) {
    size_t max_len = 1;
    for (const auto& v : values) max_len = std::max(max_len, v.size());

    const hsize_t dims = static_cast<hsize_t>(values.size());
    H5::DataSpace space(1, &dims);
    H5::StrType stype(H5::PredType::C_S1, static_cast<hsize_t>(max_len));
    H5::DSetCreatPropList plist;
    if (!values.empty()) {
        const hsize_t chunk = 1;
        plist.setChunk(1, &chunk);
        plist.setDeflate(3);
    }
    H5::DataSet dset = file.createDataSet(name, stype, space, plist);
    if (!values.empty()) {
        std::vector<char> buf(values.size() * max_len, '\0');
        for (size_t i = 0; i < values.size(); ++i) {
            std::memcpy(buf.data() + i * max_len, values[i].data(),
                        values[i].size());
        }
        dset.write(buf.data(), stype);
    }
}

// Read a 2-D dataset of the given element type into an array2d<T>.
template <class T, class H5Type>
[[nodiscard]] array2d<T> read_2d(H5::DataSet& dset, H5Type mem_type) {
    H5::DataSpace space = dset.getSpace();
    if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("expected a 2-D dataset");
    }
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    array2d<T> out(static_cast<size_t>(dims[0]), static_cast<size_t>(dims[1]));
    if (dims[0] > 0 && dims[1] > 0) {
        dset.read(out.data(), mem_type);
    }
    return out;
}

// Write a 2-D dataset with gzip level 3 compression and explicit chunking.
template <class T, class H5Type>
inline void write_2d_gzip(H5::H5File& file, const std::string& name,
                          const array2d<T>& data, H5Type mem_type) {
    const hsize_t dims[2] = {static_cast<hsize_t>(data.rows()),
                             static_cast<hsize_t>(data.cols())};
    H5::DataSpace space(2, dims);
    H5::DSetCreatPropList plist =
        gzip_plist(data.rows(), data.cols(), sizeof(T));
    H5::DataSet dset = file.createDataSet(name, mem_type, space, plist);
    if (data.size() > 0) dset.write(data.data(), mem_type);
}

} // namespace samcore::io::detail

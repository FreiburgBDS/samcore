// nanobind bindings for libsamcore: the backend of the `samcore` package.
//
// Zero-copy conventions:
//  * property getters of internally-owned buffers return numpy views with
//    keep-alive to the parent object (nanobind's default reference_internal
//    policy for properties)
//  * functions returning freshly computed buffers transfer ownership to
//    Python via an nb::capsule (no copy on the way out)
//  * function inputs accept any dtype and are converted once at the boundary
//
// Methods whose Python API needs a wrapper (in_place flags, bool masks,
// numpy-typed starts, ...) are bound under underscore-prefixed names and
// re-exposed by samcore/__init__.py.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <samcore/samcore.hpp>

namespace nb = nanobind;
using namespace samcore;

// input array aliases (by-value ndarrays trigger dtype/layout conversion)

using in_i8_2 = nb::ndarray<const std::int8_t, nb::numpy, nb::ndim<2>, nb::c_contig>;
using in_i8_1 = nb::ndarray<const std::int8_t, nb::numpy, nb::ndim<1>, nb::c_contig>;
using in_f32_2 = nb::ndarray<const float, nb::numpy, nb::ndim<2>, nb::c_contig>;
using in_f32_1 = nb::ndarray<const float, nb::numpy, nb::ndim<1>, nb::c_contig>;

// output helpers: transfer freshly computed buffers to Python (zero-copy)

template <typename T>
nb::object to_numpy(array2d<T>&& a) {
    auto* buf = new array2d<T>(std::move(a));
    nb::capsule owner(buf,
                      [](void* p) noexcept { delete static_cast<array2d<T>*>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, T>(buf->data(), {buf->rows(), buf->cols()}, owner));
}

template <typename T>
nb::object to_numpy(std::vector<T>&& v) {
    auto* buf = new std::vector<T>(std::move(v));
    nb::capsule owner(buf,
                      [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, T>(buf->data(), {buf->size()}, owner));
}

template <typename T>
nb::object to_numpy3(array3d<T>&& a) {
    auto* buf = new array3d<T>(std::move(a));
    nb::capsule owner(buf,
                      [](void* p) noexcept { delete static_cast<array3d<T>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T>(
        buf->data(), {buf->size0(), buf->size1(), buf->size2()}, owner));
}

// Zero-copy view over the (possibly converted) input array.  Safe because
// by-value ndarrays keep their storage alive for the whole call.
template <typename T>
array2d<T> view_in(nb::ndarray<const T, nb::numpy, nb::ndim<2>, nb::c_contig> a) {
    return array2d<T>(const_cast<T*>(a.data()), a.shape(0), a.shape(1),
                      non_owning);
}

template <typename T>
array2d<T> copy_in(nb::ndarray<const T, nb::numpy, nb::ndim<2>, nb::c_contig> a) {
    array2d<T> out(a.shape(0), a.shape(1));
    std::memcpy(out.data(), a.data(), a.size() * sizeof(T));
    return out;
}

template <typename T>
std::vector<T> copy_in(nb::ndarray<const T, nb::numpy, nb::ndim<1>, nb::c_contig> a) {
    return std::vector<T>(a.data(), a.data() + a.shape(0));
}

// misc helpers


// nlohmann::json -> Python object
inline nb::object json_to_py(const nlohmann::json& j) {
    if (j.is_null()) return nb::none();
    if (j.is_boolean()) return nb::cast(j.get<bool>());
    if (j.is_number_integer() || j.is_number_unsigned()) {
        return nb::cast(j.get<std::int64_t>());
    }
    if (j.is_number_float()) return nb::cast(j.get<double>());
    if (j.is_string()) return nb::cast(j.get<std::string>());
    if (j.is_array()) {
        nb::list out;
        for (const auto& v : j) out.append(json_to_py(v));
        return out;
    }
    nb::dict out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        out[it.key().c_str()] = json_to_py(it.value());
    }
    return out;
}

// Python object -> nlohmann::json (best effort, scalar values only)
inline nlohmann::json py_to_json(nb::handle h) {
    if (h.is_none()) return nullptr;
    if (nb::isinstance<nb::bool_>(h)) return nb::cast<bool>(h);
    if (nb::isinstance<nb::int_>(h)) return nb::cast<std::int64_t>(h);
    if (nb::isinstance<nb::float_>(h)) return nb::cast<double>(h);
    if (nb::isinstance<nb::str>(h)) return nb::cast<std::string>(h);
    if (nb::isinstance<nb::list>(h)) {
        nlohmann::json arr = nlohmann::json::array();
        for (auto item : nb::cast<nb::list>(h)) arr.push_back(py_to_json(item));
        return arr;
    }
    if (nb::isinstance<nb::dict>(h)) {
        nlohmann::json obj = nlohmann::json::object();
        for (auto [k, v] : nb::cast<nb::dict>(h)) {
            obj[nb::cast<std::string>(k)] = py_to_json(v);
        }
        return obj;
    }
    throw std::invalid_argument("cannot convert value to JSON");
}

// extra_map <-> Python dict
inline nb::object extra_to_py(const extra_map& extra) {
    nb::dict out;
    for (const auto& [k, v] : extra) {
        if (std::holds_alternative<std::int64_t>(v)) {
            out[k.c_str()] = nb::cast(std::get<std::int64_t>(v));
        } else if (std::holds_alternative<double>(v)) {
            out[k.c_str()] = nb::cast(std::get<double>(v));
        } else if (std::holds_alternative<bool>(v)) {
            out[k.c_str()] = nb::cast(std::get<bool>(v));
        } else {
            const std::string& s = std::get<std::string>(v);
            if (!s.empty() && (s[0] == '[' || s[0] == '{')) {
                try {
                    out[k.c_str()] = json_to_py(nlohmann::json::parse(s));
                } catch (const nlohmann::json::parse_error&) {
                    out[k.c_str()] = nb::cast(s);
                }
            } else {
                out[k.c_str()] = nb::cast(s);
            }
        }
    }
    return out;
}

inline extra_map py_to_extra(nb::handle h) {
    extra_map out;
    if (h.is_none()) return out;
    nb::dict d = nb::cast<nb::dict>(h);
    for (auto [k, v] : d) {
        const std::string key = nb::cast<std::string>(k);
        if (nb::isinstance<nb::bool_>(v)) {
            out[key] = nb::cast<bool>(v);
        } else if (nb::isinstance<nb::int_>(v)) {
            out[key] = nb::cast<std::int64_t>(v);
        } else if (nb::isinstance<nb::float_>(v)) {
            out[key] = nb::cast<double>(v);
        } else if (nb::isinstance<nb::str>(v)) {
            out[key] = nb::cast<std::string>(v);
        } else {
            out[key] = py_to_json(v).dump();
        }
    }
    return out;
}

// numpy int8 1-D -> vector<int8_t>
inline std::vector<std::int8_t> labels_in(nb::handle h) {
    in_i8_1 a = nb::cast<in_i8_1>(h);
    return std::vector<std::int8_t>(a.data(), a.data() + a.shape(0));
}
// ---- binding entry points (one per translation unit under src/bindings/) ----

void bind_header(nb::module_& m);
void bind_labels(nb::module_& m);
void bind_scan(nb::module_& m);
void bind_dataset(nb::module_& m);
void bind_submodules(nb::module_& m);

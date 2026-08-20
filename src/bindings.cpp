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

namespace {

// nlohmann::json -> Python object
nb::object json_to_py(const nlohmann::json& j) {
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
nlohmann::json py_to_json(nb::handle h) {
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
nb::object extra_to_py(const extra_map& extra) {
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

extra_map py_to_extra(nb::handle h) {
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
std::vector<std::int8_t> labels_in(nb::handle h) {
    in_i8_1 a = nb::cast<in_i8_1>(h);
    return std::vector<std::int8_t>(a.data(), a.data() + a.shape(0));
}

} // namespace

// module

NB_MODULE(_samcore, m) {
    m.doc() = "samcore: C++ (libsamcore) backend for SAM data processing";

    // SAMHeader

    nb::class_<sam_header>(m, "SAMHeader", nb::dynamic_attr())
        .def(nb::init<>())
        .def("__init__",
             [](sam_header* h, std::int64_t scanspline, std::int64_t nlines,
                std::int64_t scanlen, double samplerate, std::int64_t tzero,
                double resolution, bool interpolated, bool quality,
                const std::string& mode, const std::string& transducer_in,
                const std::string& transducer_through,
                const std::string& cellid, std::int64_t downsample_factor,
                nb::object extra) {
                 new (h) sam_header(scanspline, nlines, scanlen, samplerate,
                                    tzero, resolution, interpolated, quality,
                                    mode, transducer_in, transducer_through,
                                    cellid, downsample_factor);
                 if (!extra.is_none()) {
                     h->extra = py_to_extra(extra);
                 }
             },
             nb::arg("scanspline"), nb::arg("nlines"), nb::arg("scanlen"),
             nb::arg("samplerate"), nb::arg("tzero"), nb::arg("resolution"),
             nb::arg("interpolated") = false, nb::arg("quality") = true,
             nb::arg("mode") = "", nb::arg("transducer_in") = "",
             nb::arg("transducer_through") = "", nb::arg("cellid") = "",
             nb::arg("downsample_factor") = 1, nb::arg("extra") = nb::none())
        .def_rw("scanspline", &sam_header::scanspline)
        .def_rw("nlines", &sam_header::nlines)
        .def_rw("interpolated", &sam_header::interpolated)
        .def_rw("scanlen", &sam_header::scanlen)
        .def_rw("samplerate", &sam_header::samplerate)
        .def_rw("tzero", &sam_header::tzero)
        .def_rw("quality", &sam_header::quality)
        .def_rw("resolution", &sam_header::resolution)
        .def_rw("mode", &sam_header::mode)
        .def_rw("transducer_in", &sam_header::transducer_in)
        .def_rw("transducer_through", &sam_header::transducer_through)
        .def_rw("cellid", &sam_header::cellid)
        .def_rw("downsample_factor", &sam_header::downsample_factor)
        .def_prop_rw("extra",
                     [](sam_header& h) { return extra_to_py(h.extra); },
                     [](sam_header& h, nb::object v) { h.extra = py_to_extra(v); })
        .def("to_json", [](const sam_header& h) { return json_to_py(h.to_json()); })
        .def("json_str", &sam_header::json_str)
        .def("time", &sam_header::time, nb::arg("start") = 0, nb::arg("end") = -1)
        .def("copy", [](const sam_header& h) { return h; })
        .def("__str__", [](const sam_header& h) { return h.json_str(); })
        .def("__hash__", &sam_header::hash)
        .def("__eq__",
             [](const sam_header& h, nb::object o) {
                 if (!nb::isinstance<sam_header>(o)) return false;
                 return h == nb::cast<const sam_header&>(o);
             });

    // SAMLabels

    nb::class_<sam_labels>(m, "SAMLabels", nb::dynamic_attr())
        .def(nb::init<>())
        .def(nb::init<std::vector<std::int8_t>, std::vector<std::string>>(),
             nb::arg("labels"), nb::arg("label_names") = std::vector<std::string>{})
        .def_prop_ro("labels", [](sam_labels& l) {
            const auto& v = l.labels();
            return nb::ndarray<nb::numpy, std::int8_t>(
                const_cast<std::int8_t*>(v.data()), {v.size()});
        })
        .def_prop_ro("label_names", [](const sam_labels& l) {
            return l.label_names();
        })
        .def("__len__", [](const sam_labels& l) { return l.size(); })
        .def("__getitem__", [](const sam_labels& l, std::int64_t i) {
            const auto n = static_cast<std::int64_t>(l.size());
            if (i < 0) i += n;
            if (i < 0 || i >= n) {
                throw std::out_of_range("label index out of range");
            }
            return static_cast<std::int64_t>(l[static_cast<size_t>(i)]);
        })
        .def("label_name", &sam_labels::label_name)
        .def("label_name_val", &sam_labels::label_name_val)
        .def("name_to_value", &sam_labels::name_to_value)
        .def("has_name", &sam_labels::has_name)
        .def("is_labeled", &sam_labels::is_labeled)
        .def("max_label", &sam_labels::max_label)
        .def("num_classes", &sam_labels::num_classes)
        .def("unique_labels", [](const sam_labels& l) { return l.unique_labels(); })
        .def("verify_integrity", &sam_labels::verify_integrity)
        .def("_healthy_mask", [](sam_labels& l) { return to_numpy(l.healthy_mask()); })
        .def("_labeled_mask", [](sam_labels& l) { return to_numpy(l.labeled_mask()); })
        .def("_unlabeled_mask", [](sam_labels& l) { return to_numpy(l.unlabeled_mask()); })
        .def("_mask",
             [](sam_labels& l, std::int8_t v) { return to_numpy(l.mask(v)); })
        .def("_mask",
             [](sam_labels& l, const std::string& n) { return to_numpy(l.mask(n)); })
        .def("to_binary",
             [](sam_labels& l,
                std::variant<std::int8_t, std::string> positive) {
                 return to_numpy(l.to_binary(std::move(positive)));
             })
        .def("class_distribution",
             [](const sam_labels& l) { return l.class_distribution(); })
        .def("to_one_hot", [](sam_labels& l) { return to_numpy(l.to_one_hot()); })
        .def("clean_labels", &sam_labels::clean_labels)
        .def("relabel",
             [](sam_labels& l, nb::dict mapping) {
                 std::map<std::variant<std::int64_t, std::string>,
                          std::variant<std::int64_t, std::string>> m;
                 for (auto [k, v] : mapping) {
                     std::variant<std::int64_t, std::string> key;
                     if (nb::isinstance<nb::int_>(k)) {
                         key = nb::cast<std::int64_t>(k);
                     } else {
                         key = nb::cast<std::string>(k);
                     }
                     std::variant<std::int64_t, std::string> val;
                     if (nb::isinstance<nb::int_>(v)) {
                         val = nb::cast<std::int64_t>(v);
                     } else {
                         val = nb::cast<std::string>(v);
                     }
                     m.emplace(std::move(key), std::move(val));
                 }
                 l.relabel(m);
             })
        .def("merge", [](const sam_labels& l, const sam_labels& o) {
            return l.merge(o);
        })
        .def("copy", [](const sam_labels& l) { return l.copy(); })
        .def("take", [](const sam_labels& l, std::vector<size_t> indices) {
            return l.take(indices);
        })
        .def_static("create_unlabeled", &sam_labels::create_unlabeled)
        .def("to_dict", [](const sam_labels& l) { return json_to_py(l.to_dict()); })
        .def_static("from_dict", [](nb::object d) {
            return sam_labels::from_dict(py_to_json(d));
        });

    m.def("merge_labels",
          [](const std::vector<sam_labels>& instances) {
              return merge_labels(instances);
          },
          nb::arg("instances"));

    // SAMScan

    nb::class_<sam_scan>(m, "SAMScan", nb::dynamic_attr())
        .def(nb::init<>())
        .def(nb::init<std::string, bool>(),
             nb::arg("path"), nb::arg("mmap") = false,
             "Load a SAM scan from a .h5sam file. With mmap=True the "
             "signal data stays on disk until first accessed.")
        .def_static("from_file",
                    [](const std::string& path, bool mmap) {
                        return sam_scan::from_file(path, mmap);
                    },
                    nb::arg("path"), nb::arg("mmap") = false)
        .def_static("from_data",
                    [](in_i8_2 data, sam_header header,
                       std::optional<std::vector<std::int32_t>> starts,
                       std::optional<sam_labels> labels) {
                        return sam_scan::from_data(copy_in<std::int8_t>(data),
                                                   std::move(header),
                                                   std::move(starts),
                                                   std::move(labels));
                    },
                    nb::arg("data"), nb::arg("header"),
                    nb::arg("starts") = nb::none(),
                    nb::arg("samlabels") = nb::none())
        .def_prop_ro("data", [](sam_scan& s) {
            auto& d = s.data();
            return nb::ndarray<nb::numpy, std::int8_t>(d.data(),
                                                       {d.rows(), d.cols()});
        })
        .def_prop_rw("header", [](sam_scan& s) -> sam_header& { return s.header(); },
                     [](sam_scan& s, sam_header h) { s.header() = std::move(h); })
        .def_prop_rw("samlabels",
                     [](sam_scan& s) -> sam_labels& { return s.samlabels(); },
                     [](sam_scan& s, sam_labels l) { s.samlabels() = std::move(l); })
        .def_prop_rw("_starts",
                     [](sam_scan& s) -> std::optional<std::vector<std::int32_t>> {
                         return s.starts();
                     },
                     [](sam_scan& s,
                        std::optional<std::vector<std::int32_t>> v) {
                         s.starts() = std::move(v);
                     })
        .def_prop_rw("path",
                     [](const sam_scan& s) { return s.path(); },
                     [](sam_scan& s, std::string p) { s.path() = std::move(p); })
        .def_prop_ro("loaded", [](const sam_scan& s) { return s.loaded(); })
        .def("load", [](sam_scan& s) { s.load(); })
        .def_prop_ro("nlines", [](const sam_scan& s) { return s.nlines(); })
        .def_prop_ro("rows", [](const sam_scan& s) { return s.nlines(); })
        .def_prop_ro("cols", [](const sam_scan& s) { return s.cols(); })
        .def_prop_ro("scanlen", [](const sam_scan& s) { return s.scanlen(); })
        .def_prop_ro("samplerate", [](const sam_scan& s) { return s.samplerate(); })
        .def_prop_ro("shape", [](const sam_scan& s) {
            auto [nl, nc] = s.shape();
            return std::make_tuple(nl, nc);
        })
        .def("__len__", [](const sam_scan& s) { return s.num_scans(); })
        .def("__getitem__",
             [](sam_scan& s, std::int64_t i) {
                 auto n = static_cast<std::int64_t>(s.num_scans());
                 if (i < 0) i += n;
                 if (i < 0 || i >= n) {
                     throw std::out_of_range("scan index out of range");
                 }
                 auto row = s.data()[static_cast<size_t>(i)];
                 return nb::ndarray<nb::numpy, std::int8_t>(row.data(),
                                                            {row.size()});
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("__getitem__",
             [](sam_scan& s, nb::slice sl) {
                 // step == 1 returns a zero-copy strided 2-D view, other
                 // steps copy the selected rows.
                 const auto n = s.num_scans();
                 auto [start, stop, step, len] = sl.compute(n);
                 auto& d = s.data();
                 const size_t cols = d.cols();
                 if (len == 0) {
                     return nb::ndarray<nb::numpy, std::int8_t>(
                         d.data(), {0, cols});
                 }
                 const size_t first = static_cast<size_t>(start);
                 if (step == 1) {
                     // zero-copy strided view (rv_policy::reference + keep_alive)
                     return nb::ndarray<nb::numpy, std::int8_t>(
                         d.data() + first * cols, {len, cols}, nb::handle(),
                         {static_cast<std::int64_t>(cols),
                          static_cast<std::int64_t>(sizeof(std::int8_t))});
                 }
                 // non-unit step: copy the selected rows into an owned buffer
                 array2d<std::int8_t> out(len, cols);
                 for (size_t i = 0; i < len; ++i) {
                     std::memcpy(out[i].data(),
                                 d[first + i * static_cast<size_t>(step)].data(),
                                 cols);
                 }
                 auto* buf = new array2d<std::int8_t>(std::move(out));
                 nb::capsule owner(
                     buf, [](void* p) noexcept {
                         delete static_cast<array2d<std::int8_t>*>(p);
                     });
                 return nb::ndarray<nb::numpy, std::int8_t>(
                     buf->data(), {buf->rows(), buf->cols()}, owner);
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("__getitem__",
             [](sam_scan& s, std::tuple<std::int64_t, std::int64_t> idx) {
                 const auto [l, c] = idx;
                 if (l < 0 || c < 0 || l >= s.nlines() || c >= s.cols()) {
                     throw std::out_of_range("scan index out of range");
                 }
                 auto row = s.data()[static_cast<size_t>(
                     l * s.cols() + c)];
                 return nb::ndarray<nb::numpy, std::int8_t>(row.data(),
                                                            {row.size()});
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("scan",
             [](sam_scan& s, size_t i) {
                 auto row = s.data()[i];
                 return nb::ndarray<nb::numpy, std::int8_t>(row.data(),
                                                            {row.size()});
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("scan",
             [](sam_scan& s, std::int64_t l, std::int64_t c) {
                 auto row = s.data()[static_cast<size_t>(l * s.cols() + c)];
                 return nb::ndarray<nb::numpy, std::int8_t>(row.data(),
                                                            {row.size()});
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("time",
             [](sam_scan& s, std::optional<size_t> index) {
                 return to_numpy(s.time(index));
             },
             nb::arg("index") = nb::none())
        .def_prop_ro("samplespacing",
                     [](const sam_scan& s) { return s.samplespacing(); })
        .def("header_hash", [](const sam_scan& s) { return s.header_hash(); })
        .def("image", [](sam_scan& s, const std::string& mode) {
            image_mode m;
            if (mode == "max") {
                m = image_mode::max;
            } else if (mode == "absmax") {
                m = image_mode::absmax;
            } else if (mode == "power") {
                m = image_mode::power;
            } else {
                throw std::invalid_argument("Unsupported image type: " + mode);
            }
            image_result r = s.image(m);
            if (std::holds_alternative<array2d<std::int8_t>>(r)) {
                return to_numpy(std::move(std::get<array2d<std::int8_t>>(r)));
            }
            if (std::holds_alternative<array2d<std::int16_t>>(r)) {
                return to_numpy(std::move(std::get<array2d<std::int16_t>>(r)));
            }
            return to_numpy(std::move(std::get<array2d<float>>(r)));
        }, nb::arg("mode"))
        .def("normalized_data", [](sam_scan& s) {
            return to_numpy(s.normalized_data());
        })
        .def("set_labels",
             [](sam_scan& s, nb::handle labels,
                std::vector<std::string> label_names) {
                 s.set_labels(labels_in(labels), std::move(label_names));
             },
             nb::arg("labels"), nb::arg("label_names") = std::vector<std::string>{})
        .def("_downsample",
             [](sam_scan& s, size_t factor, const std::string& mode) {
                 downsample_mode m = downsample_mode::decimate;
                 if (mode == "mean") {
                     m = downsample_mode::mean;
                 } else if (mode == "median") {
                     m = downsample_mode::median;
                 } else if (mode == "sample") {
                     m = downsample_mode::sample;
                 } else if (mode != "decimate") {
                     throw std::invalid_argument(
                         "Invalid mode. Choose from 'decimate', 'mean', "
                         "'median', or 'sample'.");
                 }
                 s.downsample(factor, m);
             },
             nb::arg("factor"), nb::arg("mode") = "decimate")
        .def("_downsampled",
             [](sam_scan& s, size_t factor, const std::string& mode) {
                 downsample_mode m = downsample_mode::decimate;
                 if (mode == "mean") {
                     m = downsample_mode::mean;
                 } else if (mode == "median") {
                     m = downsample_mode::median;
                 } else if (mode == "sample") {
                     m = downsample_mode::sample;
                 }
                 return s.downsampled(factor, m);
             },
             nb::arg("factor"), nb::arg("mode") = "decimate")
        .def("_rotate", [](sam_scan& s, int degrees) { s.rotate(degrees); })
        .def("_rotated",
             [](sam_scan& s, int degrees) { return s.rotated(degrees); })
        .def("_mirror",
             [](sam_scan& s, const std::string& orientation) {
                 std::string o = orientation;
                 std::transform(o.begin(), o.end(), o.begin(), ::tolower);
                 if (o == "x") {
                     s.mirror(mirror_axis::x);
                 } else if (o == "y") {
                     s.mirror(mirror_axis::y);
                 } else {
                     throw std::invalid_argument(
                         "orientation must be 'x' or 'y'");
                 }
             })
        .def("_mirrored",
             [](sam_scan& s, const std::string& orientation) {
                 std::string o = orientation;
                 std::transform(o.begin(), o.end(), o.begin(), ::tolower);
                 if (o == "x") {
                     return s.mirrored(mirror_axis::x);
                 }
                 if (o == "y") {
                     return s.mirrored(mirror_axis::y);
                 }
                 throw std::invalid_argument("orientation must be 'x' or 'y'");
             })
        .def("_rectangle_select", &sam_scan::rectangle_select,
             nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
             nb::arg("col_end"))
        .def("_rectangle_select_ip", &sam_scan::rectangle_select_ip,
             nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
             nb::arg("col_end"))
        .def("_time_range_select", &sam_scan::time_range_select,
             nb::arg("start_time"), nb::arg("end_time"))
        .def("_time_range_select_ip", &sam_scan::time_range_select_ip,
             nb::arg("start_time"), nb::arg("end_time"))
        .def("_zgate_copy", &sam_scan::zgate, nb::arg("threshold") = 0.2,
             nb::arg("length") = 2000)
        .def("_zgate_ip", &sam_scan::zgate_ip, nb::arg("threshold"),
             nb::arg("length"))
        .def("_compute_stft",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::stft_result r = s.compute_stft(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.zxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128)
        .def("psd",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::psd_result r = s.psd(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.psd)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128)
        .def("power_spectrogram",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::spectrogram_result r =
                     s.power_spectrogram(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.sxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128)
        .def("to_h5sam", [](sam_scan& s, const std::string& path) {
            s.to_h5sam(path);
        })
        .def("copy", [](const sam_scan& s) { return s.copy(); })
        .def("_assign", [](sam_scan& s, const sam_scan& o) { s = o; })
        .def("_load_file",
             [](sam_scan& s, const std::string& path) {
                 s = sam_scan::from_file(path);
             });

    // SAMDataset

    nb::class_<sam_dataset>(m, "SAMDataset", nb::dynamic_attr())
        .def(nb::init<std::vector<sam_scan>, float, std::optional<bool>>(),
             nb::arg("handlers"), nb::arg("pad_value") = 0.0f,
             nb::arg("unsupervised") = nb::none())
        .def_static("load", [](const std::string& path) {
            return sam_dataset::load(path);
        }, nb::arg("path"))
        .def("save", [](sam_dataset& d, const std::string& path) { d.save(path); })
        .def("copy", [](const sam_dataset& d) { return d.copy(); })
        .def_prop_ro("X", [](sam_dataset& d) {
            auto& x = d.X();
            return nb::ndarray<nb::numpy, float>(x.data(), {x.rows(), x.cols()});
        })
        .def_prop_ro("labels",
                     [](sam_dataset& d) -> sam_labels* {
                         return d.labels().has_value() ? &*d.labels() : nullptr;
                     },
                     nb::rv_policy::reference_internal)
        .def_prop_rw("Z",
                     [](sam_dataset& d) -> nb::object {
                         if (!d.Z()) return nb::none();
                         auto& z = *d.Z();
                         return nb::cast(nb::ndarray<nb::numpy, float>(
                             z.data(), {z.rows(), z.cols()}));
                     },
                     [](sam_dataset& d, nb::object z) {
                         if (z.is_none()) {
                             d.Z() = std::nullopt;
                             return;
                         }
                         in_f32_2 a = nb::cast<in_f32_2>(z);
                         d.Z() = copy_in<float>(a);
                     })
        .def_prop_rw("V",
                     [](sam_dataset& d) -> nb::object {
                         if (!d.V()) return nb::none();
                         auto& v = *d.V();
                         return nb::cast(nb::ndarray<nb::numpy, float>(
                             v.data(), {v.rows(), v.cols()}));
                     },
                     [](sam_dataset& d, nb::object v) {
                         if (v.is_none()) {
                             d.V() = std::nullopt;
                             return;
                         }
                         in_f32_2 a = nb::cast<in_f32_2>(v);
                         d.V() = copy_in<float>(a);
                     })
        .def_prop_ro("unsupervised",
                     [](const sam_dataset& d) { return d.unsupervised(); })
        .def_prop_ro("pad_value", [](const sam_dataset& d) { return d.pad_value(); })
        .def_prop_ro("num_samples",
                     [](const sam_dataset& d) { return d.num_samples(); })
        .def_prop_ro("num_features", [](const sam_dataset& d) -> nb::object {
            return d.Z().has_value()
                       ? nb::cast(static_cast<size_t>(d.num_features()))
                       : nb::none();
        })
        .def_prop_ro("maxlen", [](const sam_dataset& d) { return d.maxlen(); })
        .def_prop_ro("num_classes",
                     [](const sam_dataset& d) { return d.num_classes(); })
        .def_prop_ro("cube_shapes", [](const sam_dataset& d) {
            std::vector<std::pair<std::int64_t, std::int64_t>> out;
            for (const auto& [nl, nc] : d.cube_shapes()) {
                out.emplace_back(nl, nc);
            }
            return out;
        })
        .def_prop_ro("cube_resolutions", [](const sam_dataset& d) {
            return d.cube_resolutions();
        })
        .def_prop_ro("scanlens", [](const sam_dataset& d) { return d.scanlens(); })
        .def_prop_rw("_train_indices",
                     [](sam_dataset& d) -> std::vector<std::int64_t> {
                         return d.train_indices();
                     },
                     [](sam_dataset& d, std::vector<std::int64_t> v) {
                         d.train_indices() = std::move(v);
                     })
        .def_prop_rw("_test_indices",
                     [](sam_dataset& d) -> std::vector<std::int64_t> {
                         return d.test_indices();
                     },
                     [](sam_dataset& d, std::vector<std::int64_t> v) {
                         d.test_indices() = std::move(v);
                     })
        .def_prop_rw("_shuffled",
                     [](const sam_dataset& d) { return d.shuffled(); },
                     [](sam_dataset& d, bool v) { d.set_shuffled(v); })
        .def("_spatial_arrays", [](sam_dataset& d) {
            auto sp = d.spatial();
            std::vector<std::int32_t> idx(sp.size());
            std::vector<float> x(sp.size()), y(sp.size());
            for (size_t i = 0; i < sp.size(); ++i) {
                idx[i] = sp[i].idx;
                x[i] = sp[i].x;
                y[i] = sp[i].y;
            }
            return nb::make_tuple(to_numpy(std::move(idx)),
                                  to_numpy(std::move(x)),
                                  to_numpy(std::move(y)));
        })
        .def("_preprocess",
             [](sam_dataset& d, const std::string& strategy, double cutoff,
                double cutoff_low, double cutoff_high, double fs,
                const std::string& mode, size_t window_length, size_t polyorder,
                size_t kernel_size, size_t start, size_t end, size_t window) {
                 preprocess_args args;
                 args.cutoff = cutoff;
                 args.cutoff_low = cutoff_low;
                 args.cutoff_high = cutoff_high;
                 args.fs = fs;
                 args.mode = mode;
                 args.window_length = window_length;
                 args.polyorder = polyorder;
                 args.kernel_size = kernel_size;
                 args.start = start;
                 args.end = end;
                 args.window = window;
                 d.preprocess(strategy, args);
             },
             nb::arg("strategy"), nb::arg("cutoff") = 0.0,
             nb::arg("cutoff_low") = 0.0, nb::arg("cutoff_high") = 0.0,
             nb::arg("fs") = 0.0, nb::arg("mode") = "minmax",
             nb::arg("window_length") = 5, nb::arg("polyorder") = 2,
             nb::arg("kernel_size") = 3, nb::arg("start") = 0,
             nb::arg("end") = 0, nb::arg("window") = 5)
        .def("get_cube_X",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_X(idx));
             })
        .def("get_cube_Z",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_Z(idx));
             })
        .def("get_cube_V",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_V(idx));
             })
        .def("get_cube_labels",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy(d.get_cube_labels(idx));
             })
        .def("class_distribution", &sam_dataset::class_distribution)
        .def("to_binary",
             [](sam_dataset& d,
                std::variant<std::int8_t, std::string> positive_label) {
                 return to_numpy(d.to_binary(std::move(positive_label)));
             })
        .def("to_one_hot",
             [](sam_dataset& d) { return to_numpy(d.to_one_hot()); })
        .def("_relabel",
             [](sam_dataset& d, nb::dict mapping) {
                 std::map<std::variant<std::int64_t, std::string>,
                          std::variant<std::int64_t, std::string>> m;
                 for (auto [k, v] : mapping) {
                     std::variant<std::int64_t, std::string> key;
                     if (nb::isinstance<nb::int_>(k)) {
                         key = nb::cast<std::int64_t>(k);
                     } else {
                         key = nb::cast<std::string>(k);
                     }
                     std::variant<std::int64_t, std::string> val;
                     if (nb::isinstance<nb::int_>(v)) {
                         val = nb::cast<std::int64_t>(v);
                     } else {
                         val = nb::cast<std::string>(v);
                     }
                     m.emplace(std::move(key), std::move(val));
                 }
                 d.relabel(m);
             });

    // preprocessing

    nb::module_ pp = m.def_submodule("preprocessing");
    pp.def("lp",
           [](in_f32_2 data, double cutoff, double fs) {
               return to_numpy(preprocessing::lp(view_in<float>(data), cutoff, fs));
           },
           nb::arg("data"), nb::arg("cutoff"), nb::arg("fs"));
    pp.def("bp",
           [](in_f32_2 data, double cutoff_low, double cutoff_high, double fs) {
               return to_numpy(preprocessing::bp(view_in<float>(data), cutoff_low,
                                                 cutoff_high, fs));
           },
           nb::arg("data"), nb::arg("cutoff_low"), nb::arg("cutoff_high"),
           nb::arg("fs"));
    pp.def("normalize",
           [](in_f32_2 data, const std::string& mode) {
               return to_numpy(preprocessing::normalize(view_in<float>(data), mode));
           },
           nb::arg("data"), nb::arg("mode") = "minmax");
    pp.def("savgol",
           [](in_f32_2 data, size_t window_length, size_t polyorder) {
               return to_numpy(
                   preprocessing::savgol(view_in<float>(data), window_length,
                                         polyorder));
           },
           nb::arg("data"), nb::arg("window_length") = 5,
           nb::arg("polyorder") = 2);
    pp.def("medfilt",
           [](in_f32_2 data, size_t kernel_size) {
               return to_numpy(preprocessing::medfilt(view_in<float>(data),
                                                      kernel_size));
           },
           nb::arg("data"), nb::arg("kernel_size") = 3);
    pp.def("gate",
           [](in_f32_2 data, size_t start, size_t end) {
               return to_numpy(preprocessing::gate(view_in<float>(data), start, end));
           },
           nb::arg("data"), nb::arg("start") = 0, nb::arg("end") = 0);
    pp.def("detrend", [](in_f32_2 data) {
        return to_numpy(preprocessing::detrend(view_in<float>(data)));
    });
    pp.def("envelope", [](in_f32_2 data) {
        return to_numpy(preprocessing::envelope(view_in<float>(data)));
    });
    pp.def("moving_average",
           [](in_f32_2 data, size_t window) {
               return to_numpy(
                   preprocessing::moving_average(view_in<float>(data), window));
           },
           nb::arg("data"), nb::arg("window") = 5);

    // utils

    nb::module_ ut = m.def_submodule("utils");
    ut.def("kurt", [](in_f32_2 data) {
        return to_numpy(utils::kurt(view_in<float>(data)));
    });
    ut.def("time_index",
           [](double tzero, double delta_t, size_t num) {
               return to_numpy(utils::time_index(tzero, delta_t, num));
           },
           nb::arg("tzero"), nb::arg("delta_t"), nb::arg("num"));
    ut.def("fft_spec",
           [](in_f32_1 scan, double d) {
               std::vector<float> v = copy_in<float>(scan);
               auto [mag, freqs] = utils::fft_spec(v, d);
               return nb::make_tuple(to_numpy(std::move(mag)),
                                     to_numpy(std::move(freqs)));
           },
           nb::arg("scan"), nb::arg("d") = 1.0);
    ut.def("spectral_entropy",
           [](in_f32_2 psd, double base) {
               return to_numpy(utils::spectral_entropy(view_in<float>(psd), base));
           },
           nb::arg("psd"), nb::arg("base") = 2.0);
    ut.def("spectral_flatness", [](in_f32_2 psd) {
        return to_numpy(utils::spectral_flatness(view_in<float>(psd)));
    });
    ut.def("spectral_centroid",
           [](in_f32_1 freqs, in_f32_2 psd) {
               return to_numpy(utils::spectral_centroid(
                   copy_in<float>(freqs), view_in<float>(psd)));
           },
           nb::arg("freqs"), nb::arg("psd"));
    ut.def("spectral_energy_ratio",
           [](in_f32_1 freqs, in_f32_2 psd, double critical_freq) {
               return to_numpy(utils::spectral_energy_ratio(
                   copy_in<float>(freqs), view_in<float>(psd), critical_freq));
           },
           nb::arg("freqs"), nb::arg("psd"), nb::arg("critical_freq"));

    // file I/O

    nb::module_ io = m.def_submodule("io");
    io.def("read_h5sam", [](const std::string& path) {
        auto r = samcore::io::read_h5sam(path);
        return nb::make_tuple(to_numpy(std::move(r.data)),
                              nb::cast(std::move(r.header)),
                              nb::cast(std::move(r.labels)),
                              r.starts ? to_numpy(std::move(*r.starts))
                                       : nb::object(nb::none()));
    });
    io.def("write_h5sam",
           [](const std::string& path, in_i8_2 data, sam_header header,
              sam_labels labels,
              std::optional<std::vector<std::int32_t>> starts) {
               samcore::io::write_h5sam(path, copy_in<std::int8_t>(data),
                                        std::move(header), std::move(labels),
                                        std::move(starts));
           },
           nb::arg("path"), nb::arg("data"), nb::arg("header"),
           nb::arg("samlabels"), nb::arg("starts") = nb::none());
    io.def("read_h5samd", [](const std::string& path) {
        return sam_dataset::load(path);
    });
    io.def("convert_h5sam_to_h5samd",
           [](const std::vector<std::string>& input_paths,
              const std::string& output_path, float pad_value,
              std::optional<bool> unsupervised) {
               std::vector<std::filesystem::path> paths;
               for (const auto& p : input_paths) paths.emplace_back(p);
               samcore::io::convert_h5sam_to_h5samd(paths, output_path,
                                                    pad_value, unsupervised);
           },
           nb::arg("input_paths"), nb::arg("output_path"),
           nb::arg("pad_value") = 0.0f, nb::arg("unsupervised") = nb::none());
}

// nanobind bindings for SAMHeader.

#include "common.hpp"

void bind_header(nb::module_& m) {
    // SAMHeader

    nb::class_<sam_header>(m, "SAMHeader", nb::dynamic_attr(),
                           "Metadata describing a SAM acquisition: grid "
                                   "geometry, sampling, timing and acquisition "
                                   "settings.\n\n"
                                   "Attributes\n"
                                   "----------\n"
                                   "scanspline : int\n"
                                   "    Number of scans per line (grid columns).\n"
                                   "nlines : int\n"
                                   "    Number of scan lines (grid rows).\n"
                                   "scanlen : int\n"
                                   "    Length of each A-scan in samples.\n"
                                   "interpolated : bool\n"
                                   "    Whether the acquisition was interpolated.\n"
                                   "samplerate : float\n"
                                   "    Sampling rate in MHz.\n"
                                   "tzero : int\n"
                                   "    Time origin in nanoseconds.\n"
                                   "quality : bool\n"
                                   "    True for high acquisition quality, False "
                                   "for low quality.\n"
                                   "resolution : float\n"
                                   "    Lateral resolution in um per pixel.\n"
                                   "mode : str\n"
                                   "    Scan mode, 'echo', 'through' or an "
                                   "empty string.\n"
                                   "transducer_in : str\n"
                                   "    Description of the input transducer.\n"
                                   "transducer_through : str\n"
                                   "    Description of the through transducer.\n"
                                   "cellid : str\n"
                                   "    Identifier of the scanned cell.\n"
                                   "downsample_factor : int\n"
                                   "    Factor by which the acquisition was "
                                   "downsampled.\n"
                                   "extra : dict\n"
                                   "    Additional metadata as key-value pairs.")
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
             nb::arg("downsample_factor") = 1, nb::arg("extra") = nb::none(),
             "Create a SAM header.")
        .def_rw("scanspline", &sam_header::scanspline,
                "Scans per line (grid columns).")
        .def_rw("nlines", &sam_header::nlines,
                "Number of lines (grid rows).")
        .def_rw("interpolated", &sam_header::interpolated,
                "Whether the acquisition was interpolated.")
        .def_rw("scanlen", &sam_header::scanlen,
                "Samples per A-scan.")
        .def_rw("samplerate", &sam_header::samplerate,
                "Sampling rate in MHz.")
        .def_rw("tzero", &sam_header::tzero,
                "Time origin in ns.")
        .def_rw("quality", &sam_header::quality,
                "Quality flag of the acquisition.")
        .def_rw("resolution", &sam_header::resolution,
                "Lateral resolution in um per pixel.")
        .def_rw("mode", &sam_header::mode,
                "Acquisition mode.")
        .def_rw("transducer_in", &sam_header::transducer_in,
                "Input transducer.")
        .def_rw("transducer_through", &sam_header::transducer_through,
                "Through transducer.")
        .def_rw("cellid", &sam_header::cellid,
                "Cell identifier.")
        .def_rw("downsample_factor", &sam_header::downsample_factor,
                "Downsampling factor applied to the acquisition.")
        .def_prop_rw("extra",
                     [](sam_header& h) { return extra_to_py(h.extra); },
                     [](sam_header& h, nb::object v) { h.extra = py_to_extra(v); },
                     "Additional metadata as a dict.")
        .def("to_json", [](const sam_header& h) { return json_to_py(h.to_json()); },
             "Serialize the header to a JSON-compatible dict.")
        .def("json_str", &sam_header::json_str,
             "Serialize the header to a JSON string.")
        .def("time", &sam_header::time, nb::arg("start") = 0, nb::arg("end") = -1,
             "Time axis in ns for samples ``start..end``.")
        .def("copy", [](const sam_header& h) { return h; },
             "Return a copy of the header.")
        .def("__str__", [](const sam_header& h) { return h.json_str(); },
             "JSON string representation of the header.")
        .def("__hash__", &sam_header::hash,
             "Stable hash of the header.")
        .def("__eq__",
             [](const sam_header& h, nb::object o) {
                 if (!nb::isinstance<sam_header>(o)) return false;
                 return h == nb::cast<const sam_header&>(o);
             },
             "Compare two headers for equality.");

}

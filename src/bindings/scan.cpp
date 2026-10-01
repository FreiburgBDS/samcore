// nanobind bindings for SAMScan.

#include "common.hpp"

void bind_scan(nb::module_& m) {
    // SAMScan

    nb::class_<sam_scan>(m, "SAMScan", nb::dynamic_attr(),
                         "A single SAM acquisition: a rectangular grid "
                                 "of int8 A-scans together with its header, "
                                 "labels and optional per-scan start "
                                 "indices.\n\n"
                                 "``data`` is a zero-copy numpy view of the "
                                 "C++ buffer with shape "
                                 "(nlines * cols, scanlen).\n\n"
                                 "Units: time values in ns (``time``, "
                                 "``tof``, gates), sampling rate in MHz "
                                 "(``samplerate``), lateral resolution in "
                                 "µm/pixel (``header.resolution``); spectral "
                                 "frequency bins are in Hz and STFT time "
                                 "bins in seconds.\n\n"
                                 "Attributes\n"
                                 "----------\n"
                                 "data : ndarray (int8)\n"
                                 "    Signal matrix, shape (nlines * cols, "
                                 "scanlen).\n"
                                 "header : SAMHeader\n"
                                 "    Acquisition metadata.\n"
                                 "samlabels : SAMLabels\n"
                                 "    Per-scan labels and their names.\n"
                                 "starts : ndarray (int32) or None\n"
                                 "    Per-scan start indices; -1 marks an "
                                 "unaligned scan.\n"
                                 "path : str\n"
                                 "    Source file path (empty when built from "
                                 "data).")
        .def(nb::init<>())
        .def("__init__",
             [](sam_scan* self, const std::string& path, bool lazy) {
                 new (self) sam_scan(path, lazy);
             },
             nb::arg("path"), nb::arg("lazy") = false,
             "Load a SAM scan from a .h5sam file.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "path : str\n"
                     "    Path to a .h5sam file.\n"
                     "lazy : bool, optional\n"
                     "    With True the signal data stays on disk and is "
                     "decoded in cached row blocks on demand; header, labels "
                     "and starts are always loaded eagerly.  The data is "
                     "chunked and compressed, so this is paged lazy reading, "
                     "not memory mapping.")
        .def_static("from_file",
                    [](const std::string& path, bool lazy) {
                        return sam_scan::from_file(path, lazy);
                    },
                    nb::arg("path"), nb::arg("lazy") = false,
                    "Load a SAM scan from a .h5sam file.\n\n"
                            "Parameters\n"
                            "----------\n"
                            "path : str\n"
                            "    Path to a .h5sam file.\n"
                            "lazy : bool, optional\n"
                            "    Keep the signal data on disk and decode it "
                            "in cached row blocks on demand (paged lazy "
                            "reading; not memory mapping).")
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
                    nb::arg("samlabels") = nb::none(),
                    nb::sig(
                        "def from_data(data: numpy.typing.NDArray[numpy.int8], header: SAMHeader, starts: numpy.typing.NDArray[numpy.int32] | collections.abc.Sequence[int] | None = None, samlabels: SAMLabels | None = None) -> SAMScan"),
                    "Build a scan from an int8 signal array and a header.\n\n"
                            "Parameters\n"
                            "----------\n"
                            "data : ndarray (int8)\n"
                            "    Signals of shape (nlines * cols, scanlen).\n"
                            "header : SAMHeader\n"
                            "    Acquisition metadata; ``nlines * scanspline`` "
                            "must match ``data.shape[0]`` and ``scanlen`` "
                            "``data.shape[1]``.\n"
                            "starts : ndarray (int32) or None, optional\n"
                            "    Per-scan start indices; -1 marks an unaligned "
                            "scan.\n"
                            "samlabels : SAMLabels or None, optional\n"
                            "    Per-scan labels; unlabeled when omitted.")
        .def_static("handler_from_data",
                    [](in_i8_2 data, sam_header header,
                       std::optional<std::vector<std::int32_t>> starts,
                       std::optional<sam_labels> labels) {
                        sam_scan s = sam_scan::from_data(
                            copy_in<std::int8_t>(data), std::move(header),
                            std::move(starts), std::move(labels));
                        s.path() = "";
                        return s;
                    },
                    nb::arg("data"), nb::arg("header"),
                    nb::arg("starts") = nb::none(),
                    nb::arg("samlabels") = nb::none(),
                    nb::sig(
                        "def handler_from_data(data: numpy.typing.NDArray[numpy.int8], header: SAMHeader, starts: numpy.typing.NDArray[numpy.int32] | collections.abc.Sequence[int] | None = None, samlabels: SAMLabels | None = None) -> SAMScan"),
                    "Create a new scan from data and header, without loading "
                            "from file.\n\n"
                            "Parameters\n"
                            "----------\n"
                            "data : ndarray (int8)\n"
                            "    The signal array of shape "
                            "(nlines * cols, scanlen).\n"
                            "header : SAMHeader\n"
                            "    The scan header metadata.\n"
                            "starts : ndarray (int32) or sequence of int, "
                            "optional\n"
                            "    The start indices for each scan, if "
                            "applicable.\n"
                            "samlabels : SAMLabels or None, optional\n"
                            "    Per-scan labels; unlabeled when omitted.\n\n"
                            "Returns\n"
                            "-------\n"
                            "SAMScan\n"
                            "    A new instance with the provided data and "
                            "metadata.")
        .def_prop_ro("data", [](sam_scan& s) {
            auto& d = s.data();
            return nb::ndarray<nb::numpy, std::int8_t>(d.data(),
                                                       {d.rows(), d.cols()});
        }, "Signal data as an (n_signals, scanlen) int8 array.")
        .def_prop_rw("header", [](sam_scan& s) -> sam_header& { return s.header(); },
                     [](sam_scan& s, sam_header h) { s.header() = std::move(h); },
                     "The acquisition header.")
        .def_prop_rw("samlabels",
                     [](sam_scan& s) -> sam_labels& { return s.samlabels(); },
                     [](sam_scan& s, sam_labels l) { s.samlabels() = std::move(l); },
                     "The scan's labels.")
        .def_prop_ro("labels", [](sam_scan& s) {
            const auto& v = s.samlabels().labels();
            return nb::ndarray<nb::numpy, std::int8_t>(
                const_cast<std::int8_t*>(v.data()), {v.size()});
        }, "Per-scan label values as an int8 array.")
        .def_prop_ro("label_names", [](sam_scan& s) {
            return s.samlabels().label_names();
        }, "The list of class label names.")
        .def_prop_rw("starts",
                     [](sam_scan& s)
                         -> std::optional<nb::ndarray<nb::numpy, std::int32_t>> {
                         const auto& v = s.starts();
                         if (!v.has_value()) return std::nullopt;
                         return to_ndarray(std::vector<std::int32_t>(*v));
                     },
                     [](sam_scan& s,
                        std::optional<std::variant<
                            std::vector<std::int32_t>,
                            nb::ndarray<nb::numpy, std::int32_t>>> v) {
                         if (!v) {
                             s.starts() = std::nullopt;
                             return;
                         }
                         if (auto* vec =
                                 std::get_if<std::vector<std::int32_t>>(&*v)) {
                             s.starts() = std::move(*vec);
                         } else {
                             auto a =
                                 std::get<nb::ndarray<nb::numpy, std::int32_t>>(*v);
                             s.starts() = std::vector<std::int32_t>(
                                 a.data(), a.data() + a.size());
                         }
                     },
                     nb::rv_policy::move,
                     "Per-scan start indices (int32) or None; -1 marks "
                             "an unaligned scan.")
        .def_prop_ro("timescale", [](sam_scan& s) {
            return to_numpy(without_gil([&] { return s.time(); }));
        }, nb::sig(
               "def timescale(self) -> numpy.typing.NDArray[numpy.float64]"),
           "Time axis in ns for the full scan length.")
        .def_prop_ro("downsample_factor", [](sam_scan& s) {
            return s.header().downsample_factor;
        }, "Downsampling factor of the acquisition.")
        .def_prop_rw("path",
                     [](const sam_scan& s) { return s.path(); },
                     [](sam_scan& s, std::string p) { s.path() = std::move(p); },
                     "Source file path (empty if built from data).")
        .def_prop_ro("loaded", [](const sam_scan& s) { return s.loaded(); },
                     "Whether the signal data has been loaded into memory "
                             "(false in lazy mode until materialized).")
        .def_prop_ro("materialized",
                     [](const sam_scan& s) { return s.materialized(); },
                     "Alias of ``loaded``: True once the signal data is in "
                             "memory.")
        .def_prop_ro("backing", [](const sam_scan& s) { return s.backing(); },
                     "``'eager'`` or ``'lazy'`` depending on how the scan "
                             "was loaded.")
        .def_prop_ro("blocks_read",
                     [](const sam_scan& s) { return s.blocks_read(); },
                     "Number of data blocks decoded from disk so far (0 for "
                             "an eager scan); read accounting for lazy "
                             "access.")
        .def("load", [](sam_scan& s) { without_gil([&] { s.load(); }); },
             "Materialize the signal data into memory (no-op when already "
                     "loaded).")
        .def("read_row",
             [](sam_scan& s, size_t index) {
                 return to_numpy(s.read_row(index));
             },
             nb::arg("index"),
             "Read one A-scan without materializing the scan.\n\n"
                     "In lazy mode only the block containing the row is "
                     "decoded from disk; in eager mode a copy of the row is "
                     "returned.")
        .def("read_rows",
             [](sam_scan& s, size_t first, size_t count) {
                 return to_numpy(s.read_rows(first, count));
             },
             nb::arg("first"), nb::arg("count"),
             "Read ``count`` consecutive A-scans starting at ``first`` "
                     "without materializing the scan (owned copy).")
        .def_prop_ro("nlines", [](const sam_scan& s) { return s.nlines(); },
                     "Number of grid lines (rows).")
        .def_prop_ro("rows", [](const sam_scan& s) { return s.nlines(); },
                     "Number of grid lines (rows), identical to "
                             "``nlines``.")
        .def_prop_ro("cols", [](const sam_scan& s) { return s.cols(); },
                     "Number of scans per line (grid columns).")
        .def_prop_ro("scanlen", [](const sam_scan& s) { return s.scanlen(); },
                     "Samples per A-scan.")
        .def_prop_ro("samplerate", [](const sam_scan& s) { return s.samplerate(); },
                     "Sampling rate in MHz.")
        .def_prop_ro("shape", [](const sam_scan& s) {
            auto [nl, nc] = s.shape();
            return std::make_tuple(nl, nc);
        }, "Grid shape as ``(nlines, cols)``.")
        .def("__len__", [](const sam_scan& s) { return s.num_scans(); })
        .def("__getitem__",
             [](sam_scan& s, std::int64_t i) -> nb::object {
                 auto n = static_cast<std::int64_t>(s.num_scans());
                 if (i < 0) i += n;
                 if (i < 0 || i >= n) {
                     throw std::out_of_range("scan index out of range");
                 }
                 const size_t index = static_cast<size_t>(i);
                 if (!s.loaded()) {
                     return to_numpy(s.read_row(index));
                 }
                 auto row = s.data()[index];
                 return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                     row.data(), {row.size()}));
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("__getitem__",
             [](sam_scan& s, nb::slice sl) -> nb::object {
                 // Eager scans: step == 1 returns a zero-copy strided 2-D
                 // view, other steps copy the selected rows.  Lazy scans
                 // decode only the requested blocks and return an owned
                 // array.
                 const auto n = s.num_scans();
                 auto [start, stop, step, len] = sl.compute(n);
                 const size_t first = static_cast<size_t>(start);
                 if (!s.loaded()) {
                     if (len == 0) {
                         // read_selected({}) yields a (0, samples) array
                         // without decoding anything.
                         return to_numpy(s.read_selected({}));
                     }
                     if (step == 1) {
                         return to_numpy(s.read_rows(first, len));
                     }
                     std::vector<std::int64_t> indices(len);
                     for (size_t i = 0; i < len; ++i) {
                         indices[i] = static_cast<std::int64_t>(
                             first + i * static_cast<size_t>(step));
                     }
                     return to_numpy(s.read_selected(indices));
                 }
                 auto& d = s.data();
                 const size_t cols = d.cols();
                 if (len == 0) {
                     return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                         d.data(), {0, cols}));
                 }
                 if (step == 1) {
                     // zero-copy strided view (rv_policy::reference + keep_alive)
                     return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                         d.data() + first * cols, {len, cols}, nb::handle(),
                         {static_cast<std::int64_t>(cols),
                          static_cast<std::int64_t>(sizeof(std::int8_t))}));
                 }
                 // non-unit step: copy the selected rows into an owned buffer
                 array2d<std::int8_t> out(len, cols);
                 for (size_t i = 0; i < len; ++i) {
                     std::memcpy(out[i].data(),
                                 d[first + i * static_cast<size_t>(step)].data(),
                                 cols);
                 }
                 return to_numpy(std::move(out));
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("__getitem__",
             [](sam_scan& s, std::tuple<std::int64_t, std::int64_t> idx)
                 -> nb::object {
                 const auto [l, c] = idx;
                 if (l < 0 || c < 0 || l >= s.nlines() || c >= s.cols()) {
                     throw std::out_of_range("scan index out of range");
                 }
                 const size_t index = static_cast<size_t>(l * s.cols() + c);
                 if (!s.loaded()) {
                     return to_numpy(s.read_row(index));
                 }
                 auto row = s.data()[index];
                 return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                     row.data(), {row.size()}));
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("scan",
             [](sam_scan& s, size_t i) -> nb::object {
                 if (!s.loaded()) {
                     return to_numpy(s.read_row(i));
                 }
                 auto row = s.data()[i];
                 return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                     row.data(), {row.size()}));
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
        .def("scan",
             [](sam_scan& s, std::int64_t l, std::int64_t c) -> nb::object {
                 const size_t index = static_cast<size_t>(l * s.cols() + c);
                 if (!s.loaded()) {
                     return to_numpy(s.read_row(index));
                 }
                 auto row = s.data()[index];
                 return to_numpy_view(nb::ndarray<nb::numpy, std::int8_t>(
                     row.data(), {row.size()}));
             },
             nb::rv_policy::reference, nb::keep_alive<1, 0>())
         .def("time",
              [](sam_scan& s, std::optional<size_t> index) {
                  return to_numpy(without_gil([&] { return s.time(index); }));
              },
              nb::arg("index") = nb::none(),
              nb::sig(
                  "def time(self, index: int | None = None) -> numpy.typing.NDArray[numpy.float64]"),
              "Time axis in ns for the scan (or for one A-scan when "
                      "``index`` is given, offset by its start).")
         .def_prop_ro("samplespacing",
                      [](const sam_scan& s) { return s.samplespacing(); },
                      "Time between two samples in ns.")
         .def("relative_time", [](const sam_scan& s) {
             return to_numpy(without_gil([&] { return s.relative_time(); }));
         }, nb::sig("def relative_time(self) -> numpy.typing.NDArray[numpy.float64]"),
            "Shared (relative) time axis in ns; identical to ``time()`` "
                    "without an index.")
         .def("absolute_time", [](const sam_scan& s, size_t index) {
             return to_numpy(without_gil([&] { return s.absolute_time(index); }));
         }, nb::arg("index"),
            nb::sig("def absolute_time(self, index: int) -> numpy.typing.NDArray[numpy.float64]"),
            "Absolute time axis in ns of one A-scan, offset by its per-scan "
                    "start when the handler carries starts; identical to "
                    "``time(index)``.")
         .def("sample_index", &sam_scan::sample_index, nb::arg("time_ns"),
              "Sample index nearest to ``time_ns`` on the handler's time "
                      "scale (round-to-nearest-even, the same rule as "
                      "``time_range_select``).  For gated scans this is "
                      "relative to the shared ``tzero``.")
         .def("sample_time", &sam_scan::sample_time, nb::arg("index"),
              "Time in ns of a sample index on the handler's time scale.")
         .def("header_hash", [](const sam_scan& s) { return s.header_hash(); },
              "Stable hash of the header.")
         .def("image", [](sam_scan& s, const std::string& mode) {
             if (mode == "max") {
                 return to_numpy(without_gil([&] { return s.image_max(); }));
             }
             if (mode == "absmax") {
                 return to_numpy(without_gil([&] { return s.image_absmax(); }));
             }
             if (mode == "power") {
                 return to_numpy(without_gil([&] { return s.image_power(); }));
             }
             throw std::invalid_argument("Unsupported image type: " + mode);
         }, nb::arg("mode"),
            nb::sig(
                "def image(self, mode: str) -> numpy.typing.NDArray[numpy.float32] | numpy.typing.NDArray[numpy.int8]"),
            "Reduce every A-scan to a single value and reshape the result "
                    "into a C-scan image of shape (nlines, cols).\n\n"
                    "``mode`` selects the reduction:\n"
                    "  - 'max'    - maximum sample value of each scan\n"
                    "  - 'absmax' - maximum absolute sample value of each scan\n"
                    "  - 'power'  - sum of squared samples (signal energy)\n\n"
                    "Values are uncalibrated: 'max'/'absmax' are in raw "
                    "int8 sample units and 'power' in squared int8 sample "
                    "units.\n\n"
                    "Parameters\n"
                    "----------\n"
                    "mode : str\n"
                    "    One of 'max', 'absmax' or 'power'.\n\n"
                    "Returns\n"
                    "-------\n"
                    "numpy.ndarray\n"
                    "    Image of shape (nlines, cols) with dtype int8 "
                    "('max', 'absmax'; absmax saturates abs(-128) to 127) or "
                    "float32 ('power').")
         .def("normalized_data", [](sam_scan& s) {
             return to_numpy(without_gil([&] { return s.normalized_data(); }));
         }, nb::sig(
             "def normalized_data(self) -> numpy.typing.NDArray[numpy.float32]"),
            "Signal data normalized to the range [-1, 127/128] as float32 "
                    "(int8 samples divided by 128).")
         .def("set_labels",
              [](sam_scan& s, nb::handle labels,
                 std::vector<std::string> label_names) {
                  s.set_labels(labels_in(labels), std::move(label_names));
              },
              nb::arg("labels"), nb::arg("label_names") = std::vector<std::string>{},
              nb::sig(
                  "def set_labels(self, labels: numpy.typing.NDArray[numpy.int8], label_names: collections.abc.Sequence[str] = []) -> None"),
              "Set the per-scan labels.\n\n"
                      "``labels`` is an int8 array of length nlines * cols; "
                      "label 0 is reserved for 'healthy' and -1 for "
                      "'unlabeled'.  ``label_names`` maps each label value to "
                      "its name.")
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
                  without_gil([&] { s.downsample(factor, m); });
              },
              nb::arg("factor"), nb::arg("mode") = "decimate",
              "Downsample the signals in place by an integer ``factor``.\n\n"
                      "``mode`` selects the downsampling:\n"
                      "  - 'decimate' - anti-aliasing IIR filter, then "
                      "subsample\n"
                      "  - 'mean'     - mean of each non-overlapping segment\n"
                      "  - 'median'   - median of each non-overlapping "
                      "segment\n"
                      "  - 'sample'   - first sample of each segment\n\n"
                      "The samplerate, scanlen and downsample_factor are "
                      "updated; existing starts are divided by ``factor``.")
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
                  return without_gil([&] { return s.downsampled(factor, m); });
              },
              nb::arg("factor"), nb::arg("mode") = "decimate",
              "Return a downsampled copy of the scan.")
         .def("_rotate", [](sam_scan& s, int degrees) {
              without_gil([&] { s.rotate(degrees); });
          },
              nb::arg("degrees"),
              "Rotate the spatial grid clockwise in place by 90, 180 or 270 "
                      "degrees.\n\n"
                      "Only the spatial layout of the scans is rotated - the "
                      "individual time-domain signals are untouched.  For a "
                      "shape (nlines, cols) grid: 90/270 deg produce "
                      "(cols, nlines), 180 deg keeps (nlines, cols).  Labels "
                      "and starts follow the moved signals.")
         .def("_rotated",
              [](sam_scan& s, int degrees) {
              return without_gil([&] { return s.rotated(degrees); });
          },
              nb::arg("degrees"),
              "Return a copy with the spatial grid rotated clockwise by 90, "
                      "180 or 270 degrees (see ``_rotate``).")
         .def("_mirror",
              [](sam_scan& s, const std::string& orientation) {
                  std::string o = orientation;
                  std::transform(o.begin(), o.end(), o.begin(), ::tolower);
                  if (o == "x") {
                      without_gil([&] { s.mirror(mirror_axis::x); });
                  } else if (o == "y") {
                      without_gil([&] { s.mirror(mirror_axis::y); });
                  } else {
                      throw std::invalid_argument(
                          "orientation must be 'x' or 'y'");
                  }
              },
              nb::arg("orientation"),
              "Mirror the spatial grid in place.\n\n"
                      "Only the spatial layout is flipped; signals are "
                      "untouched and the shape is unchanged.  With "
                      "``orientation='x'`` the columns are reversed "
                      "(left/right), with ``'y'`` the lines are reversed "
                      "(top/bottom).  Labels and starts follow the moved "
                      "signals.")
         .def("_mirrored",
              [](sam_scan& s, const std::string& orientation) {
                  std::string o = orientation;
                  std::transform(o.begin(), o.end(), o.begin(), ::tolower);
                  if (o == "x") {
                      return without_gil([&] { return s.mirrored(mirror_axis::x); });
                  }
                  if (o == "y") {
                      return without_gil([&] { return s.mirrored(mirror_axis::y); });
                  }
                  throw std::invalid_argument("orientation must be 'x' or 'y'");
              },
              nb::arg("orientation"),
              "Return a copy mirrored across the 'x' or 'y' axis (see "
                      "``_mirror``).")
         .def("_rectangle_select",
              [](const sam_scan& s, std::int64_t line_start,
                 std::int64_t line_end, std::int64_t col_start,
                 std::int64_t col_end) {
                  return without_gil([&] {
                      return s.rectangle_select(line_start, line_end,
                                                col_start, col_end);
                  });
              },
              nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
              nb::arg("col_end"),
              "Select a rectangular spatial region and return it as a new "
                      "scan.\n\n"
                      "``line_start``/``col_start`` are inclusive and "
                      "``line_end``/``col_end`` exclusive.  The result has "
                      "shape (line_end-line_start, col_end-col_start).")
        .def("_rectangle_select_ip",
             [](sam_scan& s, std::int64_t line_start, std::int64_t line_end,
                std::int64_t col_start, std::int64_t col_end) {
                 without_gil([&] {
                     s.rectangle_select_ip(line_start, line_end, col_start,
                                           col_end);
                 });
             },
             nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
             nb::arg("col_end"),
             "Keep only the given sub-rectangle of the spatial grid in place "
                     "(see ``_rectangle_select``).")
        .def("_index_range_select",
             [](const sam_scan& s, std::int64_t start_idx,
                std::int64_t end_idx) {
                 return without_gil([&] {
                     return s.index_range_select(start_idx, end_idx);
                 });
             },
             nb::arg("start_idx"), nb::arg("end_idx"),
             "Slice the time axis by sample index and return a new scan.\n\n"
                     "``start_idx`` is inclusive and ``end_idx`` exclusive; "
                     "out-of-range indices are clamped.  Per-scan ``starts`` "
                     "are preserved and advanced by ``start_idx`` so absolute "
                     "times are unchanged; a uniform valid starts vector is "
                     "folded into ``tzero``.")
        .def("_index_range_select_ip",
             [](sam_scan& s, std::int64_t start_idx, std::int64_t end_idx) {
                 without_gil([&] {
                     s.index_range_select_ip(start_idx, end_idx);
                 });
             },
             nb::arg("start_idx"), nb::arg("end_idx"),
             "Slice the time axis by sample index in place (see "
                     "``_index_range_select``).")
        .def("_time_range_select",
             [](const sam_scan& s, double start_time, double end_time) {
                 return without_gil([&] {
                     return s.time_range_select(start_time, end_time);
                 });
             },
             nb::arg("start_time"), nb::arg("end_time"),
             "Return a copy truncated to the given time range in ns.")
        .def("_time_range_select_ip",
             [](sam_scan& s, double start_time, double end_time) {
                 without_gil([&] {
                     s.time_range_select_ip(start_time, end_time);
                 });
             },
             nb::arg("start_time"), nb::arg("end_time"),
             "Truncate the signals in place to the given time range in "
                     "ns.")
        .def("_zgate_copy",
             [](const sam_scan& s, double threshold, std::int64_t length) {
                 return without_gil([&] {
                     return s.zgate(threshold, length);
                 });
             },
             nb::arg("threshold") = 0.2, nb::arg("length") = 2000,
             "Apply threshold-based gating and return a copy.\n\n"
                     "For each A-scan the first sample whose absolute value "
                     "reaches ``threshold * 127`` is found and ``length`` "
                     "samples are extracted from there (clamped so the window "
                     "fits).  If the scan already has start indices, the new "
                     "relative starts are accumulated on top of them.  Scans "
                     "without a crossing yield -1 and are zero-filled.\n\n"
                     "Returns a gated scan whose ``scanlen`` is ``length``.")
        .def("_zgate_ip",
             [](sam_scan& s, double threshold, std::int64_t length) {
                 without_gil([&] { s.zgate_ip(threshold, length); });
             },
             nb::arg("threshold"), nb::arg("length"),
             "Apply threshold-based gating in place (see ``_zgate_copy``).")
        .def("_align_manual",
             [](sam_scan& s, in_i32_1 starts, std::int64_t scanlen) {
                 without_gil([&] {
                     s.align_manual(
                         std::vector<std::int32_t>(
                             starts.data(), starts.data() + starts.shape(0)),
                         scanlen);
                 });
             },
             nb::arg("starts"), nb::arg("scanlen"),
             "Apply manual per-scan start indices and a window length.\n\n"
                     "Used by ``zgate`` and cross-correlation alignment: the "
                     "values are accumulated on top of existing starts (or "
                     "set directly when none exist); -1 marks an unaligned "
                     "scan and zero-fills its samples.  ``starts`` are sample "
                     "indices and ``scanlen`` is a length in samples.")
        .def("_align_xcorr",
             [](sam_scan& s, size_t reference, std::int64_t max_shift) {
                 without_gil([&] { s.align_xcorr(reference, max_shift); });
             },
             nb::arg("reference") = 0, nb::arg("max_shift") = 0,
             "Align every A-scan to a reference by cross-correlation in "
             "place.\n\n"
                     "Integer lags within ``max_shift`` samples (0 = "
                     "scanlen/4) are applied to the sample data; existing "
                     "``starts`` advance by the applied shift so absolute "
                     "feature times are preserved.")
        .def("_aligned_xcorr",
             [](const sam_scan& s, size_t reference, std::int64_t max_shift) {
                 return without_gil([&] { return s.aligned_xcorr(reference, max_shift); });
             },
             nb::arg("reference") = 0, nb::arg("max_shift") = 0,
             "Return a copy aligned to the reference (see ``_align_xcorr``).")
        .def("_align_tof",
             [](sam_scan& s, double gate_ns, size_t reference,
                double start_ns) {
                 without_gil([&] { s.align_tof(gate_ns, reference, start_ns); });
             },
             nb::arg("gate_ns"), nb::arg("reference") = 0,
             nb::arg("start_ns") = 0.0,
             "Classic ToF alignment in place.\n\n"
                     "Each A-scan is shifted so the echo picked by its "
                     "analytic-envelope peak lands at the reference scan's "
                     "peak.  The ToF gate is ``[start_ns, start_ns + "
                     "gate_ns)`` on the time axis (both in ns); "
                     "existing ``starts`` advance by the applied integer "
                     "shift (clamped at 0).  Scans without an envelope peak "
                     "in the gate are left unchanged.")
        .def("_aligned_tof",
             [](const sam_scan& s, double gate_ns, size_t reference,
                double start_ns) {
                 return without_gil([&] { return s.aligned_tof(gate_ns, reference, start_ns); });
             },
             nb::arg("gate_ns"), nb::arg("reference") = 0,
             nb::arg("start_ns") = 0.0,
             "Return a copy aligned by ToF (see ``_align_tof``).")
        .def("_tof",
             [](sam_scan& s, std::int64_t start, std::int64_t end,
                bool sub_sample) {
                 return to_numpy(without_gil([&] { return s.tof(start, end, sub_sample); }));
             },
             nb::arg("start") = 0, nb::arg("end") = 0,
             nb::arg("sub_sample") = true,
             nb::sig(
                 "def _tof(self, start: int = 0, end: int = 0, sub_sample: bool = True) -> numpy.typing.NDArray[numpy.float32]"),
             "Time-of-flight image: analytic-envelope peak per A-scan.\n\n"
                     "Each A-scan is converted to its analytic signal "
                     "(Hilbert transform) and the maximum of the envelope "
                     "magnitude within the ``[start, end)`` gate is taken as "
                     "the echo arrival (the conventional pick for pulse-echo "
                     "measurements; no amplitude threshold is involved).  "
                     "With ``sub_sample`` the peak is refined by a parabolic "
                     "fit through the maximum and its neighbours.  The result "
                     "is expressed in ns on the handler's time scale "
                     "(``tzero + sample * samplespacing``).  Windows "
                     "without an envelope peak return ``NO_TOF`` (-1.0) "
                     "instead of NaN.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "start : int, optional\n"
                     "    First sample of the gate.\n"
                     "end : int, optional\n"
                     "    End of the gate (exclusive); 0 means the full scan "
                     "length.\n"
                     "sub_sample : bool, optional\n"
                     "    Refine the envelope peak position with a parabolic "
                     "fit.  Default True.\n\n"
                     "Returns\n"
                     "-------\n"
                     "ndarray (float32)\n"
                     "    ToF map of shape (nlines, cols).")
        .def("_thickness",
             [](sam_scan& s, double sound_speed_m_s, std::int64_t start,
                std::int64_t end, bool sub_sample) {
                 return to_numpy(without_gil([&] {
                     return s.thickness(sound_speed_m_s, start, end,
                                        sub_sample);
                 }));
             },
             nb::arg("sound_speed_m_s"), nb::arg("start") = 0,
             nb::arg("end") = 0, nb::arg("sub_sample") = true,
             nb::sig(
                 "def _thickness(self, sound_speed_m_s: float, start: int = 0, end: int = 0, sub_sample: bool = True) -> numpy.typing.NDArray[numpy.float32]"),
             "Pulse-echo thickness image derived from :meth:`tof`.\n\n"
                     "``thickness = tof_ns * 1e-9 * sound_speed_m_s / 2``, "
                     "i.e. the result is in meters (m); windows without an "
                     "envelope peak keep the ``NO_TOF`` sentinel (-1.0) "
                     "instead of NaN.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "sound_speed_m_s : float\n"
                     "    Longitudinal sound speed in m/s.\n"
                     "start : int, optional\n"
                     "    First sample of the gate.\n"
                     "end : int, optional\n"
                     "    End of the gate (exclusive); 0 means the full scan "
                     "length.\n"
                     "sub_sample : bool, optional\n"
                     "    Refine the envelope peak position with a parabolic "
                     "fit.  Default True.")
        .def("_xgate",
             [](const sam_scan& s, double gate_ns, size_t n_gates,
                const std::string& pick, double threshold,
                const std::string& mode) {
                 xgate_result r = without_gil([&] {
                     return s.xgate(gate_ns, n_gates, pick, threshold, mode);
                 });
                 return nb::make_tuple(to_numpy3(std::move(r.values)),
                                       to_numpy(std::move(r.starts)));
             },
             nb::arg("gate_ns"), nb::arg("n_gates") = 50,
             nb::arg("pick") = "tof", nb::arg("threshold") = -1.0,
             nb::arg("mode") = "max",
             nb::sig(
                 "def _xgate(self, gate_ns: float, n_gates: int = 50, pick: str = 'tof', threshold: float = -1.0, mode: str = 'max') -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.int32]]"),
             "Layered gating (XGate): reduce consecutive gates to scalars.\n\n"
                     "Picks a start per A-scan and reduces ``n_gates`` "
                     "consecutive non-overlapping windows of ``gate_ns`` "
                     "each to one scalar, producing a depth-layer stack.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "gate_ns : float\n"
                     "    Window length in ns.\n"
                     "n_gates : int, optional\n"
                     "    Maximum number of gates per scan (upper bound; "
                     "gates that do not fit are zero).\n"
                     "pick : str, optional\n"
                     "    Start criterion: 'tof' (analytic-envelope peak), "
                     "'threshold' (first sample above threshold*127) or "
                     "'none' (start at sample 0).\n"
                     "threshold : float, optional\n"
                     "    Fraction of the int8 range for pick='threshold'.\n"
                     "mode : str, optional\n"
                     "    Gate reduction: 'max', 'absmax' or 'power'.\n\n"
                     "Returns\n"
                     "-------\n"
                     "(values, starts) : tuple of ndarray\n"
                     "    Values of shape (nlines, cols, n_gates) float32 and "
                     "the picked start sample per signal (int32, -1 when the "
                     "pick failed).")
        .def("_compute_stft",
             [](sam_scan& s, size_t nperseg, size_t noverlap, double f_min,
                double f_max) {
                 signal::stft_result r = without_gil([&] {
                     return s.compute_stft(nperseg, noverlap, f_min, f_max);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.zxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::sig(
                 "def _compute_stft(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.complex64]]"),
             "Compute the one-sided Short-Time Fourier Transform (STFT) of "
                     "every A-scan.\n\n"
                     "SAM data is real-valued, so only positive-frequency "
                     "bins are produced.  With ``f_max > 0`` only bins in "
                     "``[f_min, f_max]`` Hz are returned (``f_max = 0`` "
                     "means Nyquist).\n\n"
                     "Parameters\n"
                     "----------\n"
                     "nperseg : int, optional\n"
                     "    Samples per segment.\n"
                     "noverlap : int, optional\n"
                     "    Samples of overlap between segments.\n"
                     "f_min, f_max : float, optional\n"
                     "    Frequency band limits in Hz.\n\n"
                     "Returns\n"
                     "-------\n"
                     "(freqs, time, zxx) : tuple of ndarray\n"
                     "    Frequency bins in Hz (n_freqs,), time bins in "
                     "seconds (n_frames,) and the complex STFT of shape "
                     "(n_signals, n_freqs, n_frames).")
        .def("psd",
             [](sam_scan& s, size_t nperseg, size_t noverlap, double f_min,
                double f_max) {
                 signal::psd_result r = without_gil([&] {
                     return s.psd(nperseg, noverlap, f_min, f_max);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.psd)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::sig(
                 "def psd(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "Power spectral density of every A-scan via Welch's method.\n\n"
                     "With ``f_max > 0`` only bins in ``[f_min, f_max]`` Hz "
                     "are returned (``f_max = 0`` means Nyquist).\n\n"
                     "Parameters\n"
                     "----------\n"
                     "nperseg : int, optional\n"
                     "    Samples per segment.\n"
                     "noverlap : int, optional\n"
                     "    Samples of overlap between segments.\n"
                     "f_min, f_max : float, optional\n"
                     "    Frequency band limits in Hz.\n\n"
                     "Returns\n"
                     "-------\n"
                     "(freqs, psd) : tuple of ndarray\n"
                     "    Frequency bins (n_freqs,) and the PSD of shape "
                     "(n_signals, n_freqs).")
        .def("power_spectrogram",
             [](sam_scan& s, size_t nperseg, size_t noverlap, double f_min,
                double f_max) {
                 signal::spectrogram_result r = without_gil([&] {
                     return s.power_spectrogram(nperseg, noverlap, f_min,
                                                f_max);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.sxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::sig(
                 "def power_spectrogram(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "Compute the power spectrogram of every A-scan.\n\n"
                     "Uses the same window and segment parameters as the "
                     "STFT.  With ``f_max > 0`` only bins in "
                     "``[f_min, f_max]`` Hz are returned (``f_max = 0`` "
                     "means Nyquist).\n\n"
                     "Parameters\n"
                     "----------\n"
                     "nperseg : int, optional\n"
                     "    Samples per segment.\n"
                     "noverlap : int, optional\n"
                     "    Samples of overlap between segments.\n"
                     "f_min, f_max : float, optional\n"
                     "    Frequency band limits in Hz.\n\n"
                     "Returns\n"
                     "-------\n"
                     "(freqs, time, sxx) : tuple of ndarray\n"
                     "    Frequency bins in Hz (n_freqs,), time bins in "
                     "seconds (n_frames,) and the power spectrogram of shape "
                     "(n_signals, n_freqs, n_frames).")
        .def("stft_peak_frequency",
             [](sam_scan& s, size_t nperseg, size_t noverlap, double f_min,
                double f_max) {
                 return to_numpy(without_gil([&] {
                     return s.stft_peak_frequency(nperseg, noverlap, f_min,
                                                  f_max);
                 }));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::sig(
                 "def stft_peak_frequency(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0) -> numpy.typing.NDArray[numpy.float32]"),
             "Peak STFT frequency per A-scan (Hz), reshaped to "
                     "(nlines, cols).\n\n"
                     "For every scan the frequency with the largest "
                     "max-over-frames magnitude is selected; the band can be "
                     "limited with ``f_min``/``f_max``.")
        .def("stft_peak_time",
             [](sam_scan& s, size_t nperseg, size_t noverlap, double f_min,
                double f_max) {
                 return to_numpy(without_gil([&] {
                     return s.stft_peak_time(nperseg, noverlap, f_min, f_max);
                 }));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::sig(
                 "def stft_peak_time(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0) -> numpy.typing.NDArray[numpy.float32]"),
             "Peak STFT time per A-scan (seconds), reshaped to "
                     "(nlines, cols).\n\n"
                     "For every scan the frame with the largest "
                     "max-over-frequencies magnitude is selected; the band "
                     "can be limited with ``f_min``/``f_max``.")
        .def("spectrum",
             [](const sam_scan& s) {
                 signal::spectrum_result r =
                     without_gil([&] { return s.spectrum(); });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.mag)));
             },
             nb::sig(
                 "def spectrum(self) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "One-sided FFT magnitude of every A-scan (numpy rfft parity).\n\n"
                     "Returns\n"
                     "-------\n"
                     "(freqs, magnitude) : tuple of ndarray\n"
                     "    Frequency bins in Hz (n_freqs,) and the magnitude "
                     "of shape (n_signals, n_freqs).")
        .def("to_h5sam", [](sam_scan& s, const std::string& path) {
            without_gil([&] { s.to_h5sam(path); });
        }, "Write the scan (data, header, labels, starts) to a "
                   ".h5sam file.")
        .def("copy", [](const sam_scan& s) {
             return without_gil([&] { return s.copy(); });
         },
             "Return a deep copy of the scan.")
        .def("_assign", [](sam_scan& s, const sam_scan& o) { s = o; })
        .def("_load_file",
             [](sam_scan& s, const std::string& path) {
                 s = without_gil([&] { return sam_scan::from_file(path); });
             });

}

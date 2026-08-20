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
                                 "(nlines * cols, scanlen).")
        .def(nb::init<>())
        .def(nb::init<std::string, bool>(),
             nb::arg("path"), nb::arg("mmap") = false,
             "Load a SAM scan from a .h5sam file.  With mmap=True the "
                     "signal data stays on disk until first accessed.")
        .def_static("from_file",
                    [](const std::string& path, bool mmap) {
                        return sam_scan::from_file(path, mmap);
                    },
                    nb::arg("path"), nb::arg("mmap") = false,
                    "Load a SAM scan from a .h5sam file.")
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
                    "Build a scan from an int8 signal array (nlines x "
                            "cols rows, scanlen columns), a header, and "
                            "optional per-scan starts and labels.")
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
                     [](sam_scan& s) -> nb::object {
                         const auto& v = s.starts();
                         if (!v.has_value()) return nb::none();
                         return to_numpy(std::vector<std::int32_t>(*v));
                     },
                     [](sam_scan& s,
                        std::optional<std::vector<std::int32_t>> v) {
                         s.starts() = std::move(v);
                     },
                     "Per-scan start indices (int32) or None; -1 marks "
                             "an unaligned scan.")
        .def_prop_ro("timescale", [](sam_scan& s) {
            return to_numpy(s.time());
        }, "Time axis in ns for the full scan length.")
        .def_prop_ro("downsample_factor", [](sam_scan& s) {
            return s.header().downsample_factor;
        }, "Downsampling factor of the acquisition.")
        .def_prop_rw("path",
                     [](const sam_scan& s) { return s.path(); },
                     [](sam_scan& s, std::string p) { s.path() = std::move(p); },
                     "Source file path (empty if built from data).")
        .def_prop_ro("loaded", [](const sam_scan& s) { return s.loaded(); },
                     "Whether the signal data has been loaded into "
                             "memory (mmap mode).")
        .def("load", [](sam_scan& s) { s.load(); },
             "Load the signal data into memory (mmap mode).")
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
              nb::arg("index") = nb::none(),
              nb::sig(
                  "def time(self, index: int | None = None) -> numpy.typing.NDArray[numpy.float64]"),
              "Time axis in ns for the scan (or for one A-scan when "
                      "``index`` is given, offset by its start).")
         .def_prop_ro("samplespacing",
                      [](const sam_scan& s) { return s.samplespacing(); },
                      "Time between two samples in ns.")
         .def("header_hash", [](const sam_scan& s) { return s.header_hash(); },
              "Stable hash of the header.")
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
         }, nb::arg("mode"),
            nb::sig(
                "def image(self, mode: str) -> numpy.typing.NDArray[numpy.float32] | numpy.typing.NDArray[numpy.int16] | numpy.typing.NDArray[numpy.int8]"),
            "Reduce every A-scan to a single value and reshape the result "
                    "into a C-scan image of shape (nlines, cols).\n\n"
                    "``mode`` selects the reduction:\n"
                    "  - 'max'    - maximum sample value of each scan\n"
                    "  - 'absmax' - maximum absolute sample value of each scan\n"
                    "  - 'power'  - sum of squared samples (signal energy)\n\n"
                    "Returns an image with dtype int8 ('max'), int16 "
                    "('absmax') or float32 ('power').")
         .def("normalized_data", [](sam_scan& s) {
             return to_numpy(s.normalized_data());
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
                  s.downsample(factor, m);
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
                  return s.downsampled(factor, m);
              },
              nb::arg("factor"), nb::arg("mode") = "decimate",
              "Return a downsampled copy of the scan.")
         .def("_rotate", [](sam_scan& s, int degrees) { s.rotate(degrees); },
              nb::arg("degrees"),
              "Rotate the spatial grid clockwise in place by 90, 180 or 270 "
                      "degrees.\n\n"
                      "Only the spatial layout of the scans is rotated - the "
                      "individual time-domain signals are untouched.  For a "
                      "shape (nlines, cols) grid: 90/270 deg produce "
                      "(cols, nlines), 180 deg keeps (nlines, cols).  Labels "
                      "and starts follow the moved signals.")
         .def("_rotated",
              [](sam_scan& s, int degrees) { return s.rotated(degrees); },
              nb::arg("degrees"),
              "Return a copy with the spatial grid rotated clockwise by 90, "
                      "180 or 270 degrees (see ``_rotate``).")
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
                      return s.mirrored(mirror_axis::x);
                  }
                  if (o == "y") {
                      return s.mirrored(mirror_axis::y);
                  }
                  throw std::invalid_argument("orientation must be 'x' or 'y'");
              },
              nb::arg("orientation"),
              "Return a copy mirrored across the 'x' or 'y' axis (see "
                      "``_mirror``).")
         .def("_rectangle_select", &sam_scan::rectangle_select,
              nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
              nb::arg("col_end"),
              "Select a rectangular spatial region and return it as a new "
                      "scan.\n\n"
                      "``line_start``/``col_start`` are inclusive and "
                      "``line_end``/``col_end`` exclusive.  The result has "
                      "shape (line_end-line_start, col_end-col_start).")
        .def("_rectangle_select_ip", &sam_scan::rectangle_select_ip,
             nb::arg("line_start"), nb::arg("line_end"), nb::arg("col_start"),
             nb::arg("col_end"),
             "Keep only the given sub-rectangle of the spatial grid in place "
                     "(see ``_rectangle_select``).")
        .def("_time_range_select", &sam_scan::time_range_select,
             nb::arg("start_time"), nb::arg("end_time"),
             "Return a copy truncated to the given time range in nanoseconds.")
        .def("_time_range_select_ip", &sam_scan::time_range_select_ip,
             nb::arg("start_time"), nb::arg("end_time"),
             "Truncate the signals in place to the given time range in "
                     "nanoseconds.")
        .def("_zgate_copy", &sam_scan::zgate, nb::arg("threshold") = 0.2,
             nb::arg("length") = 2000,
             "Apply threshold-based gating and return a copy.\n\n"
                     "For each A-scan the first sample whose absolute value "
                     "reaches ``threshold * 127`` is found and ``length`` "
                     "samples are extracted from there (clamped so the window "
                     "fits).  If the scan already has start indices, the new "
                     "relative starts are accumulated on top of them.  Scans "
                     "without a crossing yield -1 and are zero-filled.\n\n"
                     "Returns a gated scan whose ``scanlen`` is ``length``.")
        .def("_zgate_ip", &sam_scan::zgate_ip, nb::arg("threshold"),
             nb::arg("length"),
             "Apply threshold-based gating in place (see ``_zgate_copy``).")
        .def("_compute_stft",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::stft_result r = s.compute_stft(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.zxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             "Compute the one-sided Short-Time Fourier Transform (STFT) of "
                     "every A-scan.\n\n"
                     "SAM data is real-valued, so only positive-frequency "
                     "bins are produced.\n\n"
                     "Returns ``(freqs, time, zxx)`` where zxx has shape "
                     "(n_signals, n_freqs, n_frames).")
        .def("psd",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::psd_result r = s.psd(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.psd)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             "Power spectral density of every A-scan via Welch's method.\n\n"
                     "Returns ``(freqs, psd)`` where psd has shape "
                     "(n_signals, n_freqs).")
        .def("power_spectrogram",
             [](sam_scan& s, size_t nperseg, size_t noverlap) {
                 signal::spectrogram_result r =
                     s.power_spectrogram(nperseg, noverlap);
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.sxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             "Compute the power spectrogram of every A-scan.\n\n"
                     "Uses the same window and segment parameters as the "
                     "STFT.\n\n"
                     "Returns ``(freqs, time, sxx)`` where sxx has shape "
                     "(n_signals, n_freqs, n_frames).")
        .def("to_h5sam", [](sam_scan& s, const std::string& path) {
            s.to_h5sam(path);
        }, "Write the scan (data, header, labels, starts) to a "
                   ".h5sam file.")
        .def("copy", [](const sam_scan& s) { return s.copy(); },
             "Return a deep copy of the scan.")
        .def("_assign", [](sam_scan& s, const sam_scan& o) { s = o; })
        .def("_load_file",
             [](sam_scan& s, const std::string& path) {
                 s = sam_scan::from_file(path);
             });

}

// nanobind bindings for SAMDataset.

#include "common.hpp"

#include <new>

void bind_dataset(nb::module_& m) {
    const auto make_options = [](const std::string& merge, float pad_value,
                                 size_t reference, std::int64_t max_shift,
                                 double max_gate_loss, double threshold,
                                 std::int64_t zgate_length) {
        dataset_merge_options options;
        options.mode = merge_mode_from_string(merge);
        options.pad_value = pad_value;
        options.reference = reference;
        options.max_shift = max_shift;
        options.max_gate_loss = max_gate_loss;
        options.threshold = threshold;
        options.zgate_length = zgate_length;
        return options;
    };

    // SAMDataset

nb::class_<sam_dataset>(m, "SAMDataset", nb::dynamic_attr(),
                           "Collection of SAM A-scans pooled from one or more "
                                   "scan cubes, ready for batching, "
                                   "splitting, preprocessing and supervised "
                                   "or unsupervised learning.\n\n"
                                   "Signals are combined to a common length "
                                   "using ``merge='crop'`` (default), ``'pad'``, "
                                   "``'xcorr'`` or ``'zgate'``; spatial provenance "
                                   "is tracked per row, and labels are merged "
                                   "across cubes.  The dataset serializes as a "
                                   "single .h5samd file.\n\n"
                                   "Attributes\n"
                                   "----------\n"
                                   "X : ndarray (float32)\n"
                                   "    Merged and concatenated signals of "
                                   "shape (num_samples, common_length).\n"
                                   "spatial : recarray\n"
                                   "    Structured array with fields ``idx`` "
                                   "(source cube), ``x`` (column position in "
                                   "mm) and ``y`` (row position in mm).\n"
                                   "labels : SAMLabels or None\n"
                                   "    Label metadata; None when "
                                   "unsupervised.\n"
                                   "Z : ndarray (float32) or None\n"
                                   "    Feature matrix built by "
                                   "``transform``; None until set.\n"
                                   "V : ndarray (float32) or None\n"
                                   "    Optional per-sample coordinates for "
                                   "visualization; None until set.\n"
                                   "unsupervised : bool\n"
                                   "    When True there are no labels and "
                                   "batches yield ``(X, spatial)`` instead "
                                   "of ``(X, y, spatial)``.\n\n"
                                   "Call ``preprocess`` to apply filters or "
                                   "normalization, and ``transform`` to "
                                   "build ``Z``.\n\n"
                                   "Units: ``spatial`` x/y in mm, "
                                   "``cube_resolutions`` in µm/pixel, "
                                   "``scanlens`` in samples; preprocessing "
                                   "``cutoff``/``fs`` share one frequency "
                                   "unit (MHz to match the header).")
        .def("__init__", [make_options](sam_dataset* self,
                std::vector<sam_scan> handlers, float pad_value,
                std::optional<bool> unsupervised, const std::string& merge,
                size_t reference, std::optional<std::int64_t> max_shift,
                double max_gate_loss, double threshold,
                std::int64_t zgate_length) {
                 const auto options = make_options(
                     merge, pad_value, reference, max_shift.value_or(0),
                     max_gate_loss, threshold, zgate_length);
                 auto built = without_gil([&] {
                     return sam_dataset(std::move(handlers), options,
                                        unsupervised);
                 });
                 new (self) sam_dataset(std::move(built));
             },
             nb::arg("handlers"), nb::arg("pad_value") = 0.0f,
             nb::arg("unsupervised") = nb::none(),
             nb::arg("merge") = "crop", nb::arg("reference") = 0,
             nb::arg("max_shift") = nb::none(),
             nb::arg("max_gate_loss") = 1.0, nb::arg("threshold") = 0.2,
             nb::arg("zgate_length") = 2000,
             "Build a dataset from a list of scans.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "handlers : sequence of SAMScan\n"
                     "    One or more scans providing data, headers and "
                     "labels.\n"
                     "pad_value : float, optional\n"
                     "    Value used by ``merge='pad'`` for shorter scans.\n"
                     "unsupervised : bool or None, optional\n"
                     "    When True labels are discarded; when False labels "
                     "are required.  None (default) auto-detects: supervised "
                     "only when all scans are labeled.\n"
                     "merge : str, optional\n"
                     "    Length/alignment policy: ``'crop'`` (default; keep "
                     "the leading ``min(scanlens)`` samples), ``'pad'`` "
                     "(legacy zero/constant padding to the longest scan), "
                     "``'xcorr'`` (align all pooled rows to one reference by "
                     "cross-correlation, then keep the common overlap), or "
                     "``'zgate'`` (threshold-align every row to "
                     "``zgate_length`` samples).\n"
                     "reference : int, optional\n"
                     "    Pooled row used as the xcorr reference.  Default 0.\n"
                     "max_shift : int or None, optional\n"
                     "    Maximum xcorr lag in samples; None means "
                     "``min(scanlens) // 4``.\n"
                     "max_gate_loss : float, optional\n"
                     "    Maximum xcorr lag spread as a fraction of the "
                     "common prefix.  Default 1.0.\n"
                     "threshold : float, optional\n"
                     "    Zgate fraction of the int8 full scale.  Default 0.2.\n"
                     "zgate_length : int, optional\n"
                     "    Zgate output window in samples.  Default 2000.")
        .def_static("load",
                    [](const std::string& path, bool lazy) {
                        return without_gil([&] {
                            return sam_dataset::load(path, lazy);
                        });
                    },
                    nb::arg("path"), nb::arg("lazy") = false,
           "Load a dataset from a .h5samd file.\n\n"
           "Parameters\n"
           "----------\n"
           "path : str\n"
           "    Path to the .h5samd file.\n"
           "lazy : bool, optional\n"
           "    With True X/Z/V stay on disk and are decoded in cached row "
           "blocks on demand (paged lazy reading, not memory mapping); "
           "metadata is always loaded eagerly.")
        .def_prop_ro("loaded", [](const sam_dataset& d) { return d.loaded(); },
                     "Whether the dataset data has been loaded into memory "
                     "(false in lazy mode until materialized).")
        .def_prop_ro("materialized",
                     [](const sam_dataset& d) { return d.materialized(); },
                     "Alias of ``loaded``: True once X/Z/V are in memory.")
        .def_prop_ro("backing", [](const sam_dataset& d) { return d.backing(); },
                     "``'eager'`` or ``'lazy'`` depending on how the "
                     "dataset was loaded.")
        .def_prop_ro("blocks_read",
                     [](const sam_dataset& d) { return d.blocks_read(); },
                     "Number of X blocks decoded from disk so far (0 for an "
                     "eager dataset).")
        .def("materialize", [](sam_dataset& d) { without_gil([&] { d.load(); }); },
             "Materialize X/Z/V into memory (no-op when already loaded).  "
             "Every data accessor calls this implicitly.")
        .def("_read_rows",
             [](const sam_dataset& d, in_i64_1 indices, bool use_z) {
                 std::vector<std::int64_t> idx(
                     indices.data(), indices.data() + indices.shape(0));
                 return to_numpy(without_gil(
                     [&] { return d.read_rows(idx, use_z); }));
             },
             nb::arg("indices"), nb::arg("use_z") = false,
             "Read selected rows of X (or Z with ``use_z=True``) without "
             "materializing a lazy dataset; only the blocks containing those "
             "rows are decoded.")
        .def("save", [](sam_dataset& d, const std::string& path) {
             without_gil([&] { d.save(path); });
         },
             "Save the dataset to a .h5samd file.")
        .def("copy", [](const sam_dataset& d) {
             return without_gil([&] { return d.copy(); });
         },
             "Return a deep copy of the dataset.")
        .def_prop_ro("X", [](sam_dataset& d) {
            without_gil([&] { d.load(); });
            auto& x = d.X();
            return nb::ndarray<nb::numpy, float>(x.data(), {x.rows(), x.cols()});
        }, "Signal matrix (num_samples, common_length) float32.")
        .def_prop_ro("labels",
                     [](sam_dataset& d) -> sam_labels* {
                         return d.labels().has_value() ? &*d.labels() : nullptr;
                     },
                     nb::rv_policy::reference_internal,
                     nb::sig("def labels(self) -> SAMLabels | None"),
                     "Merged labels, or None for unsupervised "
                             "datasets.")
        .def_prop_rw("Z",
                     [](sam_dataset& d)
                         -> std::optional<nb::ndarray<nb::numpy, float>> {
                         without_gil([&] { d.load(); });
                         if (!d.Z()) return std::nullopt;
                         auto& z = *d.Z();
                         return nb::ndarray<nb::numpy, float>(
                             z.data(), {z.rows(), z.cols()});
                     },
                     [](sam_dataset& d, std::optional<in_f32_2> z) {
                         if (!z) {
                             d.Z() = std::nullopt;
                             return;
                         }
                         d.Z() = without_gil([&] { return copy_in<float>(*z); });
                     },
                     "Transformed feature matrix (float32), or None.")
        .def_prop_rw("V",
                     [](sam_dataset& d)
                         -> std::optional<nb::ndarray<nb::numpy, float>> {
                         without_gil([&] { d.load(); });
                         if (!d.V()) return std::nullopt;
                         auto& v = *d.V();
                         return nb::ndarray<nb::numpy, float>(
                             v.data(), {v.rows(), v.cols()});
                     },
                     [](sam_dataset& d, std::optional<in_f32_2> v) {
                         if (!v) {
                             d.V() = std::nullopt;
                             return;
                         }
                         d.V() = without_gil([&] { return copy_in<float>(*v); });
                     },
                     "Optional extra per-sample vectors "
                             "(num_samples, features) float32, or None.")
        .def_prop_ro("unsupervised",
                     [](const sam_dataset& d) { return d.unsupervised(); },
                     "Whether the dataset carries no labels.")
        .def_prop_ro("pad_value", [](const sam_dataset& d) { return d.pad_value(); },
                     "Value used by ``merge='pad'`` for shorter scans.")
        .def_prop_ro("merge_mode", [](const sam_dataset& d) {
            return d.merge_mode_name();
        }, "Merge policy used to build the dataset: 'crop', 'pad', 'xcorr' "
           "or 'zgate'.")
        .def_prop_ro("valid_scanlens",
                     [](const sam_dataset& d) { return d.valid_scanlens(); },
                     "Valid signal length of each cube in samples.  Equal to "
                     "the original scanlen under 'pad' and to the common "
                     "merged width after crop/xcorr/zgate.")
        .def_prop_ro("cube_samplerates",
                     [](const sam_dataset& d) { return d.cube_samplerates(); },
                     "Sampling rate (MHz) of each source cube; empty for "
                     "legacy files without metadata.")
        .def_prop_ro("samplerate", [](const sam_dataset& d) -> nb::object {
            const auto fs = d.samplerate();
            return fs ? nb::cast(*fs) : nb::none();
        }, nb::sig("def samplerate(self) -> float | None"),
           "Common sampling rate in MHz, or None when unknown or mixed.")
        .def_prop_ro("num_samples",
                     [](const sam_dataset& d) { return d.num_samples(); },
                     "Total number of signals across all cubes.")
        .def_prop_ro("num_features", [](const sam_dataset& d) -> nb::object {
            return d.has_z()
                       ? nb::cast(static_cast<size_t>(d.num_features()))
                        : nb::none();
        }, nb::sig("def num_features(self) -> int | None"),
           "Number of columns of ``Z``, or None if not set.")
        .def_prop_ro("maxlen", [](const sam_dataset& d) { return d.maxlen(); },
                     "Common stored signal length of ``X`` (kept for API "
                     "compatibility; not necessarily the longest source scan).")
        .def_prop_ro("num_classes",
                     [](const sam_dataset& d) { return d.num_classes(); },
                     "Number of labeled classes.")
        .def_prop_ro("cube_shapes", [](const sam_dataset& d) {
            std::vector<std::pair<std::int64_t, std::int64_t>> out;
            for (const auto& [nl, nc] : d.cube_shapes()) {
                out.emplace_back(nl, nc);
            }
            return out;
        }, "Spatial shape ``(nlines, cols)`` of each cube.")
        .def_prop_ro("cube_resolutions", [](const sam_dataset& d) {
            return d.cube_resolutions();
        }, "Lateral resolution (µm/pixel) of each cube.")
        .def_prop_ro("scanlens", [](const sam_dataset& d) { return d.scanlens(); },
                     "Original scan length of each cube in samples.")
.def_prop_rw("train_indices",
                     [](sam_dataset& d) {
                         return to_ndarray(
                             std::vector<std::int64_t>(d.train_indices()));
                     },
                     [](sam_dataset& d,
                        std::variant<std::vector<std::int64_t>,
                                     nb::ndarray<nb::numpy, std::int64_t>> v) {
                         if (auto* vec =
                                 std::get_if<std::vector<std::int64_t>>(&v)) {
                             d.train_indices() = std::move(*vec);
                         } else {
                             auto a = std::get<nb::ndarray<nb::numpy, std::int64_t>>(v);
                             d.train_indices() = std::vector<std::int64_t>(
                                 a.data(), a.data() + a.size());
                         }
                     },
                      nb::rv_policy::move,
                      "Row indices of ``X`` used for training.  Returns a "
                      "fresh copy: in-place numpy mutation is not persisted.")
        .def_prop_rw("test_indices",
                     [](sam_dataset& d) {
                         return to_ndarray(
                             std::vector<std::int64_t>(d.test_indices()));
                     },
                     [](sam_dataset& d,
                        std::variant<std::vector<std::int64_t>,
                                     nb::ndarray<nb::numpy, std::int64_t>> v) {
                         if (auto* vec =
                                 std::get_if<std::vector<std::int64_t>>(&v)) {
                             d.test_indices() = std::move(*vec);
                         } else {
                             auto a = std::get<nb::ndarray<nb::numpy, std::int64_t>>(v);
                             d.test_indices() = std::vector<std::int64_t>(
                                 a.data(), a.data() + a.size());
                         }
                     },
                      nb::rv_policy::move,
                      "Row indices of ``X`` used for testing.  Returns a "
                      "fresh copy: in-place numpy mutation is not persisted.")
        .def_prop_rw("shuffled",
                     [](const sam_dataset& d) { return d.shuffled(); },
                     [](sam_dataset& d, bool v) { d.set_shuffled(v); },
                     "Whether the last split randomized the sample indices "
                             "(informational; reset to False on load).")
        .def_prop_ro("spatial", [](sam_dataset& d) {
            auto sp = without_gil([&] { return d.spatial(); });
            std::vector<std::int32_t> idx(sp.size());
            std::vector<float> x(sp.size()), y(sp.size());
            for (size_t i = 0; i < sp.size(); ++i) {
                idx[i] = sp[i].idx;
                x[i] = sp[i].x;
                y[i] = sp[i].y;
            }
            nb::module_ np = nb::module_::import_("numpy");
            nb::object i = to_numpy(std::move(idx));
            nb::object xx = to_numpy(std::move(x));
            nb::object yy = to_numpy(std::move(y));
            return np.attr("rec").attr("fromarrays")(
                nb::make_tuple(i, xx, yy), nb::arg("names") = "idx,x,y");
        }, nb::sig("def spatial(self) -> numpy.recarray"),
           "Recarray with fields ``idx`` (cube id), ``x`` and ``y`` "
                   "(pixel-centre position in mm) per signal.")
        .def_prop_ro("dataset_label_names", [](sam_dataset& d) {
            if (!d.labels().has_value()) {
                throw std::runtime_error(
                    "This dataset is unsupervised. Labels are not available.");
            }
            return d.labels()->label_names();
        }, "The list of class label names (requires labels).")
        .def_prop_ro("handler_ids", [](sam_dataset& d) {
            auto sp = without_gil([&] { return d.spatial(); });
            std::vector<std::int32_t> idx(sp.size());
            for (size_t i = 0; i < sp.size(); ++i) idx[i] = sp[i].idx;
            return to_numpy(std::move(idx));
        }, nb::sig("def handler_ids(self) -> numpy.typing.NDArray[numpy.int32]"),
           "Cube id of each signal (int32).")
        .def_prop_ro("cube_counts", [](const sam_dataset& d) {
            std::map<std::int32_t, std::int32_t> out;
            std::int32_t i = 0;
            for (const auto& [nl, nc] : d.cube_shapes()) {
                out[i++] = static_cast<std::int32_t>(nl * nc);
            }
            return out;
        }, "Number of signals per cube id.")
        .def_static("convert_from_paths",
            [make_options](const std::vector<std::string>& input_paths,
               const std::string& output_path, float pad_value,
               std::optional<bool> unsupervised, const std::string& merge,
               size_t reference, std::optional<std::int64_t> max_shift,
               double max_gate_loss, double threshold,
               std::int64_t zgate_length) {
                const auto options = make_options(
                    merge, pad_value, reference, max_shift.value_or(0),
                    max_gate_loss, threshold, zgate_length);
                std::vector<std::filesystem::path> paths;
                for (const auto& p : input_paths) paths.emplace_back(p);
                without_gil([&] {
                    samcore::io::convert_h5sam_to_h5samd(
                        paths, output_path, options, unsupervised);
                });
                return without_gil(
                    [&] { return sam_dataset::load(output_path); });
            },
            nb::arg("input_paths"), nb::arg("output_path"),
            nb::arg("pad_value") = 0.0f,
            nb::arg("unsupervised") = nb::none(),
            nb::arg("merge") = "crop", nb::arg("reference") = 0,
            nb::arg("max_shift") = nb::none(),
            nb::arg("max_gate_loss") = 1.0, nb::arg("threshold") = 0.2,
            nb::arg("zgate_length") = 2000,
            nb::sig(
                "def convert_from_paths(input_paths: collections.abc.Sequence[str], output_path: str, pad_value: float = 0.0, unsupervised: bool | None = None, merge: str = 'crop', reference: int = 0, max_shift: int | None = None, max_gate_loss: float = 1.0, threshold: float = 0.2, zgate_length: int = 2000) -> SAMDataset"),
            "Convert multiple .h5sam files into a single .h5samd file.\n\n"
                    "Reads headers and labels from all files first, then "
                    "streams signal data one file at a time.  Global xcorr "
                    "uses one extra signal pass because the final overlap "
                    "depends on every lag.\n\n"
                    "Parameters\n"
                    "----------\n"
                    "input_paths : sequence of str\n"
                    "    Paths to source .h5sam files.\n"
                    "output_path : str\n"
                    "    Destination .h5samd path (must end with "
                    "``.h5samd``).\n"
                    "pad_value : float, optional\n"
                    "    Value used by ``merge='pad'``.\n"
                    "unsupervised : bool or None, optional\n"
                    "    If None, auto-detected from whether any cube is "
                    "labeled.\n"
                    "merge : str, optional\n"
                    "    'crop' (default), 'pad', 'xcorr' or 'zgate'.\n"
                    "reference : int, optional\n"
                    "    Pooled row used as the xcorr reference.\n"
                    "max_shift : int or None, optional\n"
                    "    Maximum xcorr lag; None means min scanlen // 4.\n"
                    "max_gate_loss : float, optional\n"
                    "    Maximum xcorr lag-spread fraction.\n"
                    "threshold : float, optional\n"
                    "    Zgate threshold fraction of full scale.\n"
                    "zgate_length : int, optional\n"
                    "    Zgate output window in samples.\n\n"
                    "Returns\n"
                    "-------\n"
                    "SAMDataset\n"
                    "    The converted dataset, loaded from "
                    "``output_path``.")
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
                 without_gil([&] { d.preprocess(strategy, args); });
             },
             nb::arg("strategy"), nb::arg("cutoff") = 0.0,
             nb::arg("cutoff_low") = 0.0, nb::arg("cutoff_high") = 0.0,
             nb::arg("fs") = 0.0, nb::arg("mode") = "minmax",
             nb::arg("window_length") = 5, nb::arg("polyorder") = 2,
             nb::arg("kernel_size") = 3, nb::arg("start") = 0,
             nb::arg("end") = 0, nb::arg("window") = 5,
             "Apply a preprocessing strategy to ``X`` in place.\n\n"
                     "Supported strategies: 'lp', 'bp', 'normalize' (mode "
                     "max/zscore/minmax), 'savgol', 'medfilt', 'gate', "
                     "'detrend', 'envelope' or 'moving_average'.  Only the "
                     "parameters relevant to the chosen strategy are used.\n\n"
                     "Units: ``cutoff``/``cutoff_low``/``cutoff_high`` and "
                     "``fs`` share one frequency unit (MHz to match "
                     "``SAMHeader.samplerate``; only their ratio is used); "
                     "``window_length``, ``kernel_size``, ``window``, "
                     "``start`` and ``end`` are in samples.")
        .def("_align_manual",
             [](sam_dataset& d, in_i32_1 starts, std::int64_t scanlen) {
                 without_gil([&] {
                     d.align_manual(
                         std::vector<std::int32_t>(
                             starts.data(), starts.data() + starts.shape(0)),
                         scanlen);
                 });
             },
             nb::arg("starts"), nb::arg("scanlen"),
             "Align all dataset rows to manual relative window starts.")
        .def("_align_xcorr",
             [](sam_dataset& d, size_t reference, std::int64_t max_shift,
                double max_gate_loss) {
                 without_gil([&] {
                     d.align_xcorr(reference, max_shift, max_gate_loss);
                 });
             },
             nb::arg("reference") = 0, nb::arg("max_shift") = 0,
             nb::arg("max_gate_loss") = 1.0,
             "Align every dataset row to one pooled reference row by "
             "cross-correlation.")
        .def("_align_zgate",
             [](sam_dataset& d, double threshold, std::int64_t length) {
                 without_gil([&] { d.align_zgate(threshold, length); });
             },
             nb::arg("threshold") = 0.2, nb::arg("length") = 2000,
             "Threshold-align every dataset row to a common window length.")
        .def("_align_tof",
             [](sam_dataset& d, double gate_ns, size_t reference,
                double start_ns, double max_gate_loss) {
                 without_gil([&] {
                     d.align_tof(gate_ns, reference, start_ns, max_gate_loss);
                 });
             },
             nb::arg("gate_ns"), nb::arg("reference") = 0,
             nb::arg("start_ns") = 0.0, nb::arg("max_gate_loss") = 1.0,
             "Align every dataset row to the reference envelope peak.")
        .def("compute_stft",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto r = without_gil([&] {
                     return d.compute_stft(nperseg, noverlap, f_min, f_max,
                                           kind);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.zxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             nb::sig(
                 "def compute_stft(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0, window: str = 'hann') -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.complex64]]"),
             "STFT of every dataset row.\n\n"
                     "Requires a common ``samplerate`` and a common valid "
                     "length across cubes.  Times are relative to the start "
                     "of each signal.  Use :meth:`compute_stft_by_cube` for "
                     "ragged datasets.")
        .def("compute_stft_by_cube",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto results = without_gil([&] {
                     return d.compute_stft_by_cube(nperseg, noverlap, f_min,
                                                   f_max, kind);
                 });
                 nb::list out;
                 for (auto& r : results) {
                     out.append(nb::make_tuple(to_numpy(std::move(r.f)),
                                               to_numpy(std::move(r.t)),
                                               to_numpy3(std::move(r.zxx))));
                 }
                 return out;
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             "Per-cube STFT results as a list of (freqs, times, zxx).")
        .def("psd",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto r = without_gil([&] {
                     return d.psd(nperseg, noverlap, f_min, f_max, kind);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.psd)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             nb::sig(
                 "def psd(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0, window: str = 'hann') -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "Welch PSD of every dataset row, excluding padded samples.")
        .def("psd_by_cube",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto results = without_gil([&] {
                     return d.psd_by_cube(nperseg, noverlap, f_min, f_max,
                                          kind);
                 });
                 nb::list out;
                 for (auto& r : results) {
                     out.append(nb::make_tuple(to_numpy(std::move(r.f)),
                                               to_numpy(std::move(r.psd))));
                 }
                 return out;
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             "Per-cube Welch PSD results as a list of (freqs, psd).")
        .def("power_spectrogram",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto r = without_gil([&] {
                     return d.power_spectrogram(nperseg, noverlap, f_min,
                                                f_max, kind);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)),
                                       to_numpy3(std::move(r.sxx)));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             nb::sig(
                 "def power_spectrogram(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0, window: str = 'hann') -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "Power spectrogram of every dataset row.\n\n"
                     "Requires a common samplerate and common valid length; "
                     "times are relative to each signal's start.")
        .def("power_spectrogram_by_cube",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 auto results = without_gil([&] {
                     return d.power_spectrogram_by_cube(
                         nperseg, noverlap, f_min, f_max, kind);
                 });
                 nb::list out;
                 for (auto& r : results) {
                     out.append(nb::make_tuple(to_numpy(std::move(r.f)),
                                               to_numpy(std::move(r.t)),
                                               to_numpy3(std::move(r.sxx))));
                 }
                 return out;
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             "Per-cube power spectrograms as a list of (freqs, times, sxx).")
        .def("stft_peak_frequency",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 return to_numpy(without_gil([&] {
                     return d.stft_peak_frequency(nperseg, noverlap, f_min,
                                                  f_max, kind);
                 }));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             nb::sig(
                 "def stft_peak_frequency(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0, window: str = 'hann') -> numpy.typing.NDArray[numpy.float32]"),
             "Peak STFT frequency per dataset row (Hz).")
        .def("stft_peak_time",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                double f_min, double f_max, const std::string& window) {
                 const auto kind = signal::window_kind_from_string(window);
                 return to_numpy(without_gil([&] {
                     return d.stft_peak_time(nperseg, noverlap, f_min,
                                             f_max, kind);
                 }));
             },
             nb::arg("nperseg") = 256, nb::arg("noverlap") = 128,
             nb::arg("f_min") = 0.0, nb::arg("f_max") = 0.0,
             nb::arg("window") = "hann",
             nb::sig(
                 "def stft_peak_time(self, nperseg: int = 256, noverlap: int = 128, f_min: float = 0.0, f_max: float = 0.0, window: str = 'hann') -> numpy.typing.NDArray[numpy.float32]"),
             "Peak STFT time per dataset row (seconds, relative to each "
             "signal start).")
        .def("spectrogram_features",
             [](const sam_dataset& d, size_t nperseg, size_t noverlap,
                const std::string& window, double f_min, double f_max,
                bool l1_normalize, std::optional<double> fs_mhz) {
                 signal::spectrogram_feature_options options;
                 options.nperseg = nperseg;
                 options.noverlap = noverlap;
                 options.window = signal::window_kind_from_string(window);
                 options.f_min = f_min;
                 options.f_max = f_max;
                 options.l1_normalize = l1_normalize;
                 auto r = without_gil([&] {
                     return d.spectrogram_features(options, fs_mhz);
                 });
                 return nb::make_tuple(to_numpy(std::move(r.features)),
                                       to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.t)));
             },
             nb::arg("nperseg") = 128, nb::arg("noverlap") = 96,
             nb::arg("window") = "hann", nb::arg("f_min") = 0.0,
             nb::arg("f_max") = 0.0, nb::arg("l1_normalize") = false,
             nb::arg("fs_mhz") = nb::none(),
             nb::sig(
                 "def spectrogram_features(self, nperseg: int = 128, noverlap: int = 96, window: str = 'hann', f_min: float = 0.0, f_max: float = 0.0, l1_normalize: bool = False, fs_mhz: float | None = None) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]"),
             "Flattened scipy-compatible spectrogram features for the dataset.\n\n"
                     "Each row is ``Sxx[kept_bins, :].reshape(-1)`` in "
                     "frequency-major order, optionally L1-normalized.  "
                     "Requires a common sampling rate and valid length "
                     "across cubes.  ``fs_mhz`` overrides sampling-rate "
                     "metadata (for legacy files) and must agree with it "
                     "when present.\n\n"
                     "Returns\n"
                     "-------\n"
                     "(F, f, t) : tuple of ndarray\n"
                     "    Features (n_signals, n_kept_freqs * n_frames) "
                     "float32, frequency bins (float64) and frame-centre "
                     "times in seconds (float64).")
        .def("spectrum",
             [](const sam_dataset& d) {
                 auto r = without_gil([&] { return d.spectrum(); });
                 return nb::make_tuple(to_numpy(std::move(r.f)),
                                       to_numpy(std::move(r.mag)));
             },
             nb::sig(
                 "def spectrum(self) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]"),
             "One-sided FFT magnitude of every dataset row.\n\n"
                     "Requires common samplerate and common valid length; "
                     "use :meth:`spectrum_by_cube` for ragged datasets.")
        .def("spectrum_by_cube",
             [](const sam_dataset& d) {
                 auto results =
                     without_gil([&] { return d.spectrum_by_cube(); });
                 nb::list out;
                 for (auto& r : results) {
                     out.append(nb::make_tuple(to_numpy(std::move(r.f)),
                                               to_numpy(std::move(r.mag))));
                 }
                 return out;
             },
             "Per-cube one-sided FFT spectra as a list of (freqs, magnitude).")
        .def("get_cube_X",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(without_gil([&] { return d.get_cube_X(idx); }));
             }, nb::sig(
                 "def get_cube_X(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` as a (nlines, cols, scanlen) float32 "
                     "array.")
        .def("get_cube_Z",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(without_gil([&] { return d.get_cube_Z(idx); }));
             }, nb::sig(
                 "def get_cube_Z(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` of the transformed features ``Z``.")
        .def("get_cube_V",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(without_gil([&] { return d.get_cube_V(idx); }));
             }, nb::sig(
                 "def get_cube_V(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` of the extra per-sample vectors ``V``.")
        .def("get_cube_labels",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy(without_gil([&] { return d.get_cube_labels(idx); }));
             }, nb::sig(
                 "def get_cube_labels(self, idx: int) -> numpy.typing.NDArray[numpy.int8]"),
             "Cube ``idx`` labels as a (nlines, cols) int8 array.")
        .def("class_distribution",
             [](const sam_dataset& d) {
                 return without_gil([&] { return d.class_distribution(); });
             },
             "Counts of each label name across the dataset (requires "
                     "labels).")
        .def("to_binary",
             [](sam_dataset& d,
                std::variant<std::int8_t, std::string> positive_label) {
                 auto binary = without_gil(
                     [&] { return d.to_binary(std::move(positive_label)); });
                 return to_numpy(std::move(binary));
             },
             nb::arg("positive_label"),
             nb::sig(
                 "def to_binary(self, positive_label: int | str) -> numpy.typing.NDArray[numpy.int8]"),
             "Binary labels: 1 for the positive class, 0 for healthy, "
                     "-1 for unlabeled.  Accepts a numeric value or a name.")
        .def("to_one_hot",
             [](sam_dataset& d) {
                 return to_numpy(
                     without_gil([&] { return d.to_one_hot(); }));
             },
             nb::sig(
                 "def to_one_hot(self) -> numpy.typing.NDArray[numpy.float32]"),
             "One-hot matrix (num_samples, num_classes) float32.")
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
                 without_gil([&] { d.relabel(m); });
             },
             "Remap the dataset labels according to ``mapping``.");

}

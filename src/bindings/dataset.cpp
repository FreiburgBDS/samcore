// nanobind bindings for SAMDataset.

#include "common.hpp"

void bind_dataset(nb::module_& m) {
    // SAMDataset

nb::class_<sam_dataset>(m, "SAMDataset", nb::dynamic_attr(),
                           "Collection of SAM A-scans pooled from one or more "
                                   "scan cubes, ready for batching, "
                                   "splitting, preprocessing and supervised "
                                   "or unsupervised learning.\n\n"
                                   "Signals are padded to a common length, "
                                   "spatial provenance is tracked per row, "
                                   "and labels are merged across cubes.  The "
                                   "dataset serializes as a single .h5samd "
                                   "file.\n\n"
                                   "Attributes\n"
                                   "----------\n"
                                   "X : ndarray (float32)\n"
                                   "    Padded and concatenated signals of "
                                   "shape (num_samples, max_scanlen).\n"
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
                                   "build ``Z``.")
        .def(nb::init<std::vector<sam_scan>, float, std::optional<bool>>(),
             nb::arg("handlers"), nb::arg("pad_value") = 0.0f,
             nb::arg("unsupervised") = nb::none(),
             "Build a dataset from a list of scans.\n\n"
                     "Parameters\n"
                     "----------\n"
                     "handlers : sequence of SAMScan\n"
                     "    One or more scans providing data, headers and "
                     "labels.\n"
                     "pad_value : float, optional\n"
                     "    Value used to pad shorter scans to the common "
                     "length.\n"
                     "unsupervised : bool or None, optional\n"
                     "    When True labels are discarded; when False labels "
                     "are required.  None (default) auto-detects: supervised "
                     "only when all scans are labeled.")
        .def_static("load", [](const std::string& path) {
            return sam_dataset::load(path);
        }, nb::arg("path"),
           "Load a dataset from a .h5samd file.")
        .def("save", [](sam_dataset& d, const std::string& path) { d.save(path); },
             "Save the dataset to a .h5samd file.")
        .def("copy", [](const sam_dataset& d) { return d.copy(); },
             "Return a deep copy of the dataset.")
        .def_prop_ro("X", [](sam_dataset& d) {
            auto& x = d.X();
            return nb::ndarray<nb::numpy, float>(x.data(), {x.rows(), x.cols()});
        }, "Signal matrix (num_samples, maxlen) float32.")
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
                         d.Z() = copy_in<float>(*z);
                     },
                     "Transformed feature matrix (float32), or None.")
        .def_prop_rw("V",
                     [](sam_dataset& d)
                         -> std::optional<nb::ndarray<nb::numpy, float>> {
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
                         d.V() = copy_in<float>(*v);
                     },
                     "Optional extra per-sample vectors "
                             "(num_samples, features) float32, or None.")
        .def_prop_ro("unsupervised",
                     [](const sam_dataset& d) { return d.unsupervised(); },
                     "Whether the dataset carries no labels.")
        .def_prop_ro("pad_value", [](const sam_dataset& d) { return d.pad_value(); },
                     "Value used to pad shorter scans.")
        .def_prop_ro("num_samples",
                     [](const sam_dataset& d) { return d.num_samples(); },
                     "Total number of signals across all cubes.")
        .def_prop_ro("num_features", [](const sam_dataset& d) -> nb::object {
            return d.Z().has_value()
                       ? nb::cast(static_cast<size_t>(d.num_features()))
                        : nb::none();
        }, nb::sig("def num_features(self) -> int | None"),
           "Number of columns of ``Z``, or None if not set.")
        .def_prop_ro("maxlen", [](const sam_dataset& d) { return d.maxlen(); },
                     "Common (padded) scan length of ``X``.")
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
        }, "Lateral resolution (um/pixel) of each cube.")
        .def_prop_ro("scanlens", [](const sam_dataset& d) { return d.scanlens(); },
                     "Original scan length of each cube.")
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
            auto sp = d.spatial();
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
                   "(position in mm) per signal.")
        .def_prop_ro("dataset_label_names", [](sam_dataset& d) {
            if (!d.labels().has_value()) {
                throw std::runtime_error(
                    "This dataset is unsupervised. Labels are not available.");
            }
            return d.labels()->label_names();
        }, "The list of class label names (requires labels).")
        .def_prop_ro("handler_ids", [](sam_dataset& d) {
            auto sp = d.spatial();
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
            [](const std::vector<std::string>& input_paths,
               const std::string& output_path, float pad_value,
               std::optional<bool> unsupervised) {
                std::vector<std::filesystem::path> paths;
                for (const auto& p : input_paths) paths.emplace_back(p);
                samcore::io::convert_h5sam_to_h5samd(paths, output_path,
                                                     pad_value, unsupervised);
                return sam_dataset::load(output_path);
            },
            nb::arg("input_paths"), nb::arg("output_path"),
            nb::arg("pad_value") = 0.0f,
            nb::arg("unsupervised") = nb::none(),
            nb::sig(
                "def convert_from_paths(input_paths: collections.abc.Sequence[str], output_path: str, pad_value: float = 0.0, unsupervised: bool | None = None) -> SAMDataset"),
            "Convert multiple .h5sam files into a single .h5samd file.\n\n"
                    "Reads headers and labels from all files first to "
                    "pre-size the output, then streams signal data one file "
                    "at a time to bound memory usage.\n\n"
                    "Parameters\n"
                    "----------\n"
                    "input_paths : sequence of str\n"
                    "    Paths to source .h5sam files.\n"
                    "output_path : str\n"
                    "    Destination .h5samd path (must end with "
                    "``.h5samd``).\n"
                    "pad_value : float, optional\n"
                    "    Value used to pad shorter signals to the common max "
                    "length.\n"
                    "unsupervised : bool or None, optional\n"
                    "    If None, auto-detected from whether any cube is "
                    "labeled.\n\n"
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
                 d.preprocess(strategy, args);
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
                     "parameters relevant to the chosen strategy are used.")
        .def("get_cube_X",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_X(idx));
             }, nb::sig(
                 "def get_cube_X(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` as a (nlines, cols, scanlen) float32 "
                     "array.")
        .def("get_cube_Z",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_Z(idx));
             }, nb::sig(
                 "def get_cube_Z(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` of the transformed features ``Z``.")
        .def("get_cube_V",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy3(d.get_cube_V(idx));
             }, nb::sig(
                 "def get_cube_V(self, idx: int) -> numpy.typing.NDArray[numpy.float32]"),
             "Cube ``idx`` of the extra per-sample vectors ``V``.")
        .def("get_cube_labels",
             [](sam_dataset& d, std::int32_t idx) {
                 return to_numpy(d.get_cube_labels(idx));
             }, nb::sig(
                 "def get_cube_labels(self, idx: int) -> numpy.typing.NDArray[numpy.int8]"),
             "Cube ``idx`` labels as a (nlines, cols) int8 array.")
        .def("class_distribution", &sam_dataset::class_distribution,
             "Counts of each label name across the dataset (requires "
                     "labels).")
        .def("to_binary",
             [](sam_dataset& d,
                std::variant<std::int8_t, std::string> positive_label) {
                 return to_numpy(d.to_binary(std::move(positive_label)));
             },
             nb::arg("positive_label"),
             nb::sig(
                 "def to_binary(self, positive_label: int | str) -> numpy.typing.NDArray[numpy.int8]"),
             "Binary labels: 1 for the positive class, 0 for healthy, "
                     "-1 for unlabeled.  Accepts a numeric value or a name.")
        .def("to_one_hot",
             [](sam_dataset& d) { return to_numpy(d.to_one_hot()); },
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
                 d.relabel(m);
             },
             "Remap the dataset labels according to ``mapping``.");

}

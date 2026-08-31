// nanobind bindings for SAMLabels.

#include "common.hpp"

void bind_labels(nb::module_& m) {
    // SAMLabels

    nb::class_<sam_labels>(m, "SAMLabels", nb::dynamic_attr(),
                           "Per-scan class labels (int8; -1 = unlabeled) "
                                   "together with their names.  Label 0 is "
                                   "always named 'healthy'.")
        .def(nb::init<>())
        .def(nb::init<std::vector<std::int8_t>, std::vector<std::string>>(),
             nb::arg("labels"), nb::arg("label_names") = std::vector<std::string>{},
             "Create labels from an array of int8 class ids and an "
                     "optional list of label names.")
        .def_prop_ro("labels", [](sam_labels& l) {
            const auto& v = l.labels();
            return nb::ndarray<nb::numpy, std::int8_t>(
                const_cast<std::int8_t*>(v.data()), {v.size()});
        }, "The per-scan label values as an int8 array.")
        .def_prop_ro("label_names", [](const sam_labels& l) {
            return l.label_names();
        }, "The list of class label names.")
        .def("__len__", [](const sam_labels& l) { return l.size(); },
             "Number of labels.")
        .def("__getitem__", [](const sam_labels& l, std::int64_t i) {
            const auto n = static_cast<std::int64_t>(l.size());
            if (i < 0) i += n;
            if (i < 0 || i >= n) {
                throw std::out_of_range("label index out of range");
            }
            return static_cast<std::int64_t>(l[static_cast<size_t>(i)]);
        }, "Label value at index ``i`` (negative indices count from "
                   "the end).")
        .def("label_name", &sam_labels::label_name,
             "Name of the label of the scan at ``index``.")
        .def("label_name_val", &sam_labels::label_name_val,
             "Name of the given numeric label value.")
        .def("name_to_value", &sam_labels::name_to_value,
             nb::arg("name"),
             "Resolve a label name (case-insensitive) to its integer label "
             "value.\n\n"
             "The reserved names 'unlabeled' and 'healthy' map to -1 and 0. "
             "Auto-generated names of the form 'label<int>' are also resolved "
             "for positive values that appear in the data.  Unknown names "
             "return -1.")
        .def("has_name", &sam_labels::has_name,
             nb::arg("name"),
             "Return True when ``name`` (case-insensitive) is registered in "
             "``label_names``, including the reserved names 'healthy' and "
             "'unlabeled'.")
        .def("is_labeled", &sam_labels::is_labeled,
             "Whether any label is not unlabeled (-1).")
        .def("max_label", &sam_labels::max_label,
             "Largest numeric label value present.")
        .def_prop_ro("num_classes",
                     [](const sam_labels& l) { return l.num_classes(); },
                     "Number of distinct labeled classes (excludes "
                             "unlabeled).")
        .def_prop_ro("unique_labels",
                     [](const sam_labels& l) { return l.unique_labels(); },
                     "The distinct numeric label values present.")
        .def("verify_integrity", &sam_labels::verify_integrity,
             "Raise if the labels are inconsistent with their names.")
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
             },
             nb::arg("positive_label"),
             "Convert the labels to binary classification labels.\n\n"
             "Signals matching ``positive_label`` (by value or name, "
             "case-insensitive) become 1, healthy (0) signals become 0, and "
             "unlabeled (-1) signals stay -1.  Every other value maps to 0.\n\n"
             "Returns an int8 array of the same length as ``labels``.")
        .def("class_distribution",
             [](const sam_labels& l) { return l.class_distribution(); },
             "Count the number of signals in each class.\n\n"
             "Returns a dict mapping each label name to its count; unlabeled "
             "signals are reported under the key 'unlabeled'.")
        .def("to_one_hot", [](sam_labels& l) { return to_numpy(l.to_one_hot()); },
             "One-hot matrix (n, num_classes) float32; unlabeled rows "
                     "are all zero.")
        .def("clean_labels", &sam_labels::clean_labels,
             "Compact the label space in place so label values are "
             "consecutive integers starting from 0.\n\n"
             "Label -1 (unlabeled) is left unchanged and label 0 (healthy) "
             "stays at index 0 of ``label_names``.  Every other value that "
             "appears is remapped to 1, 2, 3, ... in ascending order, carrying "
             "over the corresponding name; names that never occur are dropped.")
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
             },
             "Remap label values and/or names according to ``mapping`` (keys "
             "and values may be integers or names).")
        .def("merge", [](const sam_labels& l, const sam_labels& o) {
            return l.merge(o);
        }, "Merge ``other`` into this object by label name (case-insensitive); "
           "labels sharing a name keep the same value in the result.")
        .def("copy", [](const sam_labels& l) { return l.copy(); },
             "Return a deep copy of the labels.")
        .def("take", [](const sam_labels& l, std::vector<size_t> indices) {
            return l.take(indices);
        }, "Return the labels at the given scan indices.")
        .def_static("create_unlabeled", &sam_labels::create_unlabeled,
                    nb::arg("num_signals"),
                    "Create ``num_signals`` unlabeled (-1) labels.")
        .def("to_dict", [](const sam_labels& l) {
            nb::dict d;
            d["labels"] = l.labels();
            d["label_names"] = l.label_names();
            return d;
        },
         "Serialize the labels to a dict.")
        .def_static("from_dict", [](nb::object data) {
            nb::dict d = nb::cast<nb::dict>(data);
            std::vector<std::string> names = {sam_labels::label_name_healthy};
            if (d.contains("label_names") && nb::isinstance<nb::list>(d["label_names"])) {
                names = nb::cast<std::vector<std::string>>(d["label_names"]);
            }
            std::vector<std::int8_t> labels;
            if (d.contains("labels") && nb::isinstance<nb::list>(d["labels"])) {
                labels = nb::cast<std::vector<std::int8_t>>(d["labels"]);
            }
            return sam_labels(std::move(labels), std::move(names));
        }, nb::arg("data"),
           "Recreate labels from a ``to_dict`` dict.");

    m.def("merge_labels",
          [](const std::vector<sam_labels>& instances) {
              return merge_labels(instances);
          },
          nb::arg("instances"),
          "Merge a list of label objects into a single one.");

}

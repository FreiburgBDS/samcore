#include <samcore/sam_labels.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <set>
#include <stdexcept>

namespace samcore {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

sam_labels::sam_labels(std::vector<std::int8_t> labels,
                       std::vector<std::string> label_names)
    : labels_(std::move(labels)), label_names_(std::move(label_names)) {
    if (labels_.empty()) {
        throw std::invalid_argument("Labels array cannot be empty.");
    }
    if (!label_names_.empty() &&
        lower(label_names_[0]) != label_name_healthy) {
        std::printf(
            "Warning: label_names provided but label 0 name is not '%s'. "
            "Changing first label name to '%s'.\n",
            label_name_healthy, label_name_healthy);
        label_names_[0] = label_name_healthy;
    }
}

std::string sam_labels::label_name(size_t index) const {
    const auto label = labels_[index];
    if (label == label_unlabeled) return label_name_unlabeled;
    if (label == label_healthy) return label_name_healthy;
    if (!label_names_.empty() && static_cast<size_t>(label) < label_names_.size()) {
        return label_names_[static_cast<size_t>(label)];
    }
    if (label > 0) return "label" + std::to_string(label);
    return "N/A";
}

std::string sam_labels::label_name_val(std::int8_t label_val) const {
    if (label_val == label_unlabeled) return label_name_unlabeled;
    if (label_val == label_healthy) return label_name_healthy;
    if (!label_names_.empty() && static_cast<size_t>(label_val) < label_names_.size()) {
        return label_names_[static_cast<size_t>(label_val)];
    }
    if (label_val > 0) return "label" + std::to_string(label_val);
    return "N/A";
}

std::int8_t sam_labels::name_to_value(const std::string& name) const {
    if (name.empty()) return label_unlabeled;
    const std::string l = lower(name);
    if (l == label_name_unlabeled) return label_unlabeled;
    if (l == label_name_healthy) return label_healthy;
    for (size_t i = 0; i < label_names_.size(); ++i) {
        if (lower(label_names_[i]) == l) return static_cast<std::int8_t>(i);
    }
    if (l.rfind("label", 0) == 0) {
        try {
            const int val = std::stoi(name.substr(5));
            if (val > 0 && std::find(labels_.begin(), labels_.end(),
                                     static_cast<std::int8_t>(val)) != labels_.end()) {
                return static_cast<std::int8_t>(val);
            }
        } catch (const std::exception&) {
            // fall through
        }
    }
    return label_unlabeled;
}

bool sam_labels::has_name(const std::string& name) const {
    const std::string l = lower(name);
    if (l == label_name_healthy || l == label_name_unlabeled) return true;
    return std::any_of(label_names_.begin(), label_names_.end(),
                       [&l](const std::string& n) { return lower(n) == l; });
}

bool sam_labels::is_labeled() const {
    return std::any_of(labels_.begin(), labels_.end(),
                       [](std::int8_t v) { return v != label_unlabeled; });
}

std::int8_t sam_labels::max_label() const {
    return *std::max_element(labels_.begin(), labels_.end());
}

size_t sam_labels::num_classes() const {
    std::set<std::int8_t> seen;
    for (auto v : labels_) {
        if (v >= 0) seen.insert(v);
    }
    return seen.size();
}

std::vector<std::int8_t> sam_labels::unique_labels() const {
    std::set<std::int8_t> seen;
    for (auto v : labels_) {
        if (v != label_unlabeled) seen.insert(v);
    }
    return std::vector<std::int8_t>(seen.begin(), seen.end());
}

void sam_labels::verify_integrity() const {
    if (labels_.empty()) {
        throw std::invalid_argument("Labels array empty.");
    }
    if (std::any_of(labels_.begin(), labels_.end(),
                    [](std::int8_t v) { return v < label_unlabeled; })) {
        throw std::invalid_argument(
            "Labels array contains invalid values (less than -1).");
    }
    if (!label_names_.empty() &&
        static_cast<size_t>(max_label()) >= label_names_.size()) {
        throw std::invalid_argument(
            "Label names list does not cover all label indices.");
    }
    if (!label_names_.empty() && label_names_[0] != label_name_healthy) {
        throw std::invalid_argument(
            "First label name must be 'healthy' if label names are provided.");
    }
    std::set<std::string> seen;
    for (const auto& name : label_names_) {
        if (!seen.insert(name).second) {
            throw std::invalid_argument("Duplicate label name found: " + name);
        }
    }
}

std::vector<uint8_t> sam_labels::healthy_mask() const {
    std::vector<uint8_t> m(labels_.size());
    for (size_t i = 0; i < labels_.size(); ++i) {
        m[i] = labels_[i] == label_healthy ? 1 : 0;
    }
    return m;
}

std::vector<uint8_t> sam_labels::mask(std::int8_t label) const {
    std::vector<uint8_t> m(labels_.size());
    for (size_t i = 0; i < labels_.size(); ++i) {
        m[i] = labels_[i] == label ? 1 : 0;
    }
    return m;
}

std::vector<uint8_t> sam_labels::mask(const std::string& name) const {
    return mask(name_to_value(name));
}

std::vector<uint8_t> sam_labels::labeled_mask() const {
    std::vector<uint8_t> m(labels_.size());
    for (size_t i = 0; i < labels_.size(); ++i) {
        m[i] = labels_[i] != label_unlabeled ? 1 : 0;
    }
    return m;
}

std::vector<uint8_t> sam_labels::unlabeled_mask() const {
    std::vector<uint8_t> m(labels_.size());
    for (size_t i = 0; i < labels_.size(); ++i) {
        m[i] = labels_[i] == label_unlabeled ? 1 : 0;
    }
    return m;
}

std::vector<std::int8_t> sam_labels::to_binary(
    std::variant<std::int8_t, std::string> positive_label) const {
    std::vector<std::int8_t> binary(labels_.size(), 0);
    for (size_t i = 0; i < labels_.size(); ++i) {
        if (labels_[i] == label_unlabeled) binary[i] = label_unlabeled;
    }
    if (std::holds_alternative<std::string>(positive_label)) {
        const auto& name = std::get<std::string>(positive_label);
        const auto val = name_to_value(name);
        if (val != label_unlabeled ||
            lower(name) == label_name_unlabeled) {
            for (size_t i = 0; i < labels_.size(); ++i) {
                if (labels_[i] == val) binary[i] = 1;
            }
        }
    } else if (std::holds_alternative<std::int8_t>(positive_label)) {
        const auto val = std::get<std::int8_t>(positive_label);
        for (size_t i = 0; i < labels_.size(); ++i) {
            if (labels_[i] == val) binary[i] = 1;
        }
    } else {
        throw std::invalid_argument("positive_label must be int or str.");
    }
    return binary;
}

std::map<std::string, size_t> sam_labels::class_distribution() const {
    std::set<std::int8_t> uniq(labels_.begin(), labels_.end());
    std::map<std::string, size_t> counts;
    for (auto val : uniq) {
        counts[label_name_val(val)] =
            static_cast<size_t>(std::count(labels_.begin(), labels_.end(), val));
    }
    return counts;
}

array2d<float> sam_labels::to_one_hot() const {
    const size_t n = labels_.size();
    const size_t n_classes = num_classes();
    if (n_classes == 0) {
        return array2d<float>(n, 1, 0.0f);
    }
    array2d<float> oh(n, n_classes, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const auto v = labels_[i];
        if (v >= 0 && static_cast<size_t>(v) < n_classes) {
            oh[i][static_cast<size_t>(v)] = 1.0f;
        }
    }
    return oh;
}

void sam_labels::clean_labels() {
    auto uniq = unique_labels();
    if (uniq.empty()) {
        label_names_ = {label_name_healthy};
        return;
    }
    std::map<std::int8_t, std::int8_t> old_to_new;
    std::vector<std::string> new_names = {label_name_healthy};
    std::int8_t next_idx = 1;
    for (auto old_val : uniq) {
        if (old_val == label_healthy) {
            old_to_new[old_val] = 0;
        } else {
            old_to_new[old_val] = next_idx;
            if (!label_names_.empty() &&
                static_cast<size_t>(old_val) < label_names_.size()) {
                new_names.push_back(label_names_[static_cast<size_t>(old_val)]);
            } else {
                new_names.push_back("label" + std::to_string(old_val));
            }
            ++next_idx;
        }
    }
    std::vector<std::int8_t> new_labels(labels_.size(), label_unlabeled);
    for (size_t i = 0; i < labels_.size(); ++i) {
        auto it = old_to_new.find(labels_[i]);
        if (it != old_to_new.end()) new_labels[i] = it->second;
    }
    labels_ = std::move(new_labels);
    label_names_ = std::move(new_names);
}

void sam_labels::relabel(
    const std::map<std::variant<std::int64_t, std::string>,
                   std::variant<std::int64_t, std::string>>& mapping) {
    std::map<std::int8_t, std::string> name_changes;
    std::map<std::int8_t, std::int8_t> value_remap;

    for (const auto& [old_key, new_val] : mapping) {
        std::int8_t old;
        if (std::holds_alternative<std::int64_t>(old_key)) {
            old = static_cast<std::int8_t>(std::get<std::int64_t>(old_key));
        } else if (std::holds_alternative<std::string>(old_key)) {
            const auto& name = std::get<std::string>(old_key);
            old = name_to_value(name);
            if (old == label_unlabeled && lower(name) != label_name_unlabeled) {
                throw std::out_of_range("Label name not found: " + name);
            }
        } else {
            throw std::invalid_argument("Mapping keys must be int or str.");
        }

        std::int8_t new_val_int;
        if (std::holds_alternative<std::int64_t>(new_val)) {
            new_val_int = static_cast<std::int8_t>(std::get<std::int64_t>(new_val));
        } else if (std::holds_alternative<std::string>(new_val)) {
            const auto& name = std::get<std::string>(new_val);
            auto new_ = name_to_value(name);
            if (new_ != label_unlabeled) {
                name_changes[new_] = name;
            }
            if (new_ == label_unlabeled) {
                new_ = old;
            }
            name_changes[new_] = name;
            new_val_int = new_;
        } else {
            throw std::invalid_argument("Mapping values must be int or str.");
        }

        if (old != new_val_int) {
            value_remap[old] = new_val_int;
        }
    }

    for (size_t i = 0; i < labels_.size(); ++i) {
        auto it = value_remap.find(labels_[i]);
        if (it != value_remap.end()) labels_[i] = it->second;
    }
    for (const auto& [idx, name] : name_changes) {
        if (static_cast<size_t>(idx) < label_names_.size()) {
            label_names_[static_cast<size_t>(idx)] = name;
        }
    }
    clean_labels();
}

sam_labels sam_labels::merge(const sam_labels& other) const {
    return merge_labels({*this, other});
}

sam_labels sam_labels::copy() const {
    return sam_labels(labels_, label_names_);
}

sam_labels sam_labels::take(const std::vector<size_t>& indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Cannot take an empty index set.");
    }
    std::vector<std::int8_t> out;
    out.reserve(indices.size());
    for (auto i : indices) out.push_back(labels_[i]);
    return sam_labels(std::move(out), label_names_);
}

sam_labels sam_labels::create_unlabeled(size_t num_signals) {
    return sam_labels(std::vector<std::int8_t>(num_signals, label_unlabeled),
                      {label_name_healthy});
}

nlohmann::json sam_labels::to_dict() const {
    return nlohmann::json{{"labels", labels_}, {"label_names", label_names_}};
}

sam_labels sam_labels::from_dict(const nlohmann::json& d) {
    std::vector<std::string> names;
    if (d.contains("label_names") && d["label_names"].is_array()) {
        for (const auto& v : d["label_names"]) names.push_back(v.get<std::string>());
    } else {
        names = {label_name_healthy};
    }
    std::vector<std::int8_t> labels;
    if (d.contains("labels") && d["labels"].is_array()) {
        for (const auto& v : d["labels"]) labels.push_back(v.get<std::int8_t>());
    }
    return sam_labels(std::move(labels), std::move(names));
}

sam_labels merge_labels(const std::vector<sam_labels>& instances) {
    if (instances.size() < 2) {
        throw std::invalid_argument(
            "At least two SAMLabels instances are required to merge.");
    }

    std::vector<std::string> unified_names = {sam_labels::label_name_healthy};

    auto add_name = [&unified_names](const std::string& name) -> std::int8_t {
        for (size_t i = 0; i < unified_names.size(); ++i) {
            if (lower(unified_names[i]) == lower(name)) {
                return static_cast<std::int8_t>(i);
            }
        }
        unified_names.push_back(name);
        return static_cast<std::int8_t>(unified_names.size() - 1);
    };

    auto get_name = [](const sam_labels& inst,
                       std::int8_t val) -> std::string {
        if (val == sam_labels::label_healthy) {
            return sam_labels::label_name_healthy;
        }
        const auto& names = inst.label_names();
        if (!names.empty() && static_cast<size_t>(val) < names.size()) {
            return names[static_cast<size_t>(val)];
        }
        return "label" + std::to_string(val);
    };

    std::vector<std::map<std::int8_t, std::int8_t>> all_remaps;
    for (const auto& inst : instances) {
        const auto& labels = inst.labels();
        std::set<std::int8_t> used(labels.begin(), labels.end());
        used.erase(sam_labels::label_unlabeled);
        std::map<std::int8_t, std::int8_t> mapping;
        for (auto old_val : used) {
            if (old_val == sam_labels::label_healthy) {
                mapping[old_val] = 0;
            } else {
                mapping[old_val] = add_name(get_name(inst, old_val));
            }
        }
        all_remaps.push_back(std::move(mapping));
    }

    std::vector<std::int8_t> combined;
    for (size_t k = 0; k < instances.size(); ++k) {
        const auto& labels = instances[k].labels();
        const auto& mapping = all_remaps[k];
        for (auto v : labels) {
            auto it = mapping.find(v);
            combined.push_back(it != mapping.end() ? it->second
                                                   : sam_labels::label_unlabeled);
        }
    }

    sam_labels result(std::move(combined), unified_names);
    result.clean_labels();
    return result;
}

} // namespace samcore

#pragma once

#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <samcore/array.hpp>

namespace samcore {

// Per-scan integer labels with a registry of label names.
// Per-scan labels with a name registry.
class sam_labels {
public:
    static constexpr std::int8_t label_unlabeled = -1;
    static constexpr std::int8_t label_healthy = 0;
    static constexpr const char* label_name_unlabeled = "unlabeled";
    static constexpr const char* label_name_healthy = "healthy";

    sam_labels() = default;

    // Throws std::invalid_argument on empty arrays; coerces first label
    // name to "healthy" with a warning print when names are given but
    // label 0 is not named "healthy".
    sam_labels(std::vector<std::int8_t> labels,
               std::vector<std::string> label_names = {});

    [[nodiscard]] const std::vector<std::int8_t>& labels() const noexcept {
        return labels_;
    }
    [[nodiscard]] const std::vector<std::string>& label_names() const noexcept {
        return label_names_;
    }
    void set_labels(std::vector<std::int8_t> labels) {
        labels_ = std::move(labels);
    }
    void set_label_names(std::vector<std::string> names) {
        label_names_ = std::move(names);
    }

    [[nodiscard]] size_t size() const noexcept { return labels_.size(); }

    [[nodiscard]] std::int8_t operator[](size_t i) const { return labels_[i]; }

    [[nodiscard]] std::string label_name(size_t index) const;
    [[nodiscard]] std::string label_name_val(std::int8_t label_val) const;

    // Resolve a name (case-insensitive) to its integer value; -1 when
    // unknown.  Handles the reserved names and the auto-generated
    // "label<N>" form.
    [[nodiscard]] std::int8_t name_to_value(const std::string& name) const;

    [[nodiscard]] bool has_name(const std::string& name) const;

    [[nodiscard]] bool is_labeled() const;
    [[nodiscard]] std::int8_t max_label() const;
    // Number of classes excluding -1 (unlabeled).
    [[nodiscard]] size_t num_classes() const;
    // Sorted unique label values excluding -1.
    [[nodiscard]] std::vector<std::int8_t> unique_labels() const;

    void verify_integrity() const;

    // Masks (vector<uint8_t> of 0/1, aligned with labels()).
    [[nodiscard]] std::vector<uint8_t> healthy_mask() const;
    [[nodiscard]] std::vector<uint8_t> mask(std::int8_t label) const;
    [[nodiscard]] std::vector<uint8_t> mask(const std::string& name) const;
    [[nodiscard]] std::vector<uint8_t> labeled_mask() const;
    [[nodiscard]] std::vector<uint8_t> unlabeled_mask() const;

    // Binary labels: 1 where labels == positive_label, 0 elsewhere,
    // -1 kept for unlabeled.
    [[nodiscard]] std::vector<std::int8_t> to_binary(
        std::variant<std::int8_t, std::string> positive_label) const;

    // label name (str) -> count (unlabeled under
    // "unlabeled").
    [[nodiscard]] std::map<std::string, size_t> class_distribution() const;

    // One-hot matrix (n, num_classes) float32; rows of unlabeled stay 0.
    [[nodiscard]] array2d<float> to_one_hot() const;

    // Compact label space in place: values become consecutive starting
    // at 0 (healthy), unused names dropped.
    void clean_labels();

    // Relabel by value or name; string targets are resolved through the
    // current registry.  In place, then clean_labels().
    void relabel(const std::map<
                 std::variant<std::int64_t, std::string>,
                 std::variant<std::int64_t, std::string>>& mapping);

    // Merge another instance's labels by name (case-insensitive).
    [[nodiscard]] sam_labels merge(const sam_labels& other) const;

    [[nodiscard]] sam_labels copy() const;
    [[nodiscard]] sam_labels take(const std::vector<size_t>& indices) const;

    static sam_labels create_unlabeled(size_t num_signals);

    [[nodiscard]] nlohmann::json to_dict() const;
    static sam_labels from_dict(const nlohmann::json& d);

private:
    std::vector<std::int8_t> labels_;
    std::vector<std::string> label_names_;
};

// Merge two or more label sets by name in a single pass; none are
// modified.  Throws on fewer than 2 instances.
[[nodiscard]] sam_labels merge_labels(const std::vector<sam_labels>& instances);

} // namespace samcore

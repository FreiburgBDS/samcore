#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace samcore {

// Policy used to combine scans with different lengths into one rectangular
// dataset.
enum class merge_mode {
    crop,  // keep the leading min(scanlen) samples (default)
    pad,   // pad every scan to max(scanlen) with pad_value
    xcorr, // align all pooled scans to one reference, then keep the overlap
    zgate  // threshold-align every scan to a common window length
};

[[nodiscard]] inline const char* to_string(merge_mode mode) noexcept {
    switch (mode) {
    case merge_mode::crop: return "crop";
    case merge_mode::pad: return "pad";
    case merge_mode::xcorr: return "xcorr";
    case merge_mode::zgate: return "zgate";
    }
    return "crop";
}

[[nodiscard]] inline merge_mode merge_mode_from_string(const std::string& s) {
    if (s == "crop") return merge_mode::crop;
    if (s == "pad") return merge_mode::pad;
    if (s == "xcorr") return merge_mode::xcorr;
    if (s == "zgate") return merge_mode::zgate;
    throw std::invalid_argument(
        "Unknown merge policy: '" + s +
        "' (expected 'crop', 'pad', 'xcorr' or 'zgate').");
}

// Options shared by the in-memory dataset constructor and the streaming
// .h5sam -> .h5samd converter.  Only the fields relevant to `mode` are used.
struct dataset_merge_options {
    merge_mode mode = merge_mode::crop;
    float pad_value = 0.0f;
    std::size_t reference = 0;      // xcorr: pooled row index
    std::int64_t max_shift = 0;     // xcorr: 0 means min_scanlen / 4
    double max_gate_loss = 1.0;     // xcorr: fraction of min_scanlen
    double threshold = 0.2;         // zgate: fraction of full_scale
    std::int64_t zgate_length = 2000; // zgate: output window in samples
};

} // namespace samcore

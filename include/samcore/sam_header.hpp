#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace samcore {

// Extra header metadata values.  Anything that is not a scalar is stored
// as its JSON encoding (non-scalar `extra` values are
// json.dumps'ed and json.loads'ed on read).
using extra_value = std::variant<std::int64_t, double, bool, std::string>;
using extra_map = std::map<std::string, extra_value>;

// Generalized header for a SAM scan.
class sam_header {
public:
    sam_header() = default;

    sam_header(std::int64_t scanspline, std::int64_t nlines,
               std::int64_t scanlen, double samplerate, std::int64_t tzero,
               double resolution, bool interpolated = false,
               bool quality = true, std::string mode = {},
               std::string transducer_in = {}, std::string transducer_through = {},
               std::string cellid = {}, std::int64_t downsample_factor = 1,
               extra_map extra = {});

    // Time axis in nanoseconds for the given sample range (linspace
    // semantics: start and end inclusive, num = end - start).
    [[nodiscard]] std::vector<double> time(std::int64_t start = 0,
                                           std::int64_t end = -1) const;

    [[nodiscard]] bool operator==(const sam_header& o) const;
    [[nodiscard]] bool operator!=(const sam_header& o) const { return !(*this == o); }

    // Deterministic content hash (equal headers hash equal).  NOTE: not
    // byte-identical to a JSON-based hash; hash values are never persisted.
    [[nodiscard]] size_t hash() const;

    std::int64_t scanspline = 0;
    std::int64_t nlines = 0;
    std::int64_t scanlen = 0;
    double samplerate = 0.0;
    std::int64_t tzero = 0;
    double resolution = 0.0;
    bool interpolated = false;
    bool quality = true;
    std::string mode;
    std::string transducer_in;
    std::string transducer_through;
    std::string cellid;
    std::int64_t downsample_factor = 1;
    extra_map extra;
};

} // namespace samcore

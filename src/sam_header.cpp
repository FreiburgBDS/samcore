#include <samcore/sam_header.hpp>

#include <cmath>
#include <stdexcept>

namespace samcore {

namespace {

std::int64_t as_int(const nlohmann::json& v) {
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    if (v.is_number_integer() || v.is_number_unsigned()) return v.get<std::int64_t>();
    if (v.is_number_float()) return static_cast<std::int64_t>(v.get<double>());
    return 0;
}

double as_double(const nlohmann::json& v) {
    if (v.is_number()) return v.get<double>();
    if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
    return 0.0;
}

bool as_bool(const nlohmann::json& v) {
    if (v.is_boolean()) return v.get<bool>();
    return as_int(v) != 0;
}

std::string as_string(const nlohmann::json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number() || v.is_boolean()) return v.dump();
    return {};
}

// A string that starts with '[' or '{' is treated as JSON and stored
// verbatim in `extra`.
bool looks_like_json(const std::string& s) {
    return !s.empty() && (s[0] == '[' || s[0] == '{');
}

} // namespace

sam_header::sam_header(std::int64_t scanspline, std::int64_t nlines,
                       std::int64_t scanlen, double samplerate,
                       std::int64_t tzero, double resolution,
                       bool interpolated, bool quality, std::string mode,
                       std::string transducer_in, std::string transducer_through,
                       std::string cellid, std::int64_t downsample_factor,
                       extra_map extra)
    : scanspline(scanspline), nlines(nlines),
      scanlen(scanlen), samplerate(samplerate), tzero(tzero),
      resolution(resolution), interpolated(interpolated), quality(quality),
      mode(std::move(mode)), transducer_in(std::move(transducer_in)),
      transducer_through(std::move(transducer_through)),
      cellid(std::move(cellid)), downsample_factor(downsample_factor),
      extra(std::move(extra)) {}

sam_header sam_header::from_json(const nlohmann::json& j) {
    if (!j.is_object()) {
        throw std::invalid_argument("sam_header: expected a JSON object");
    }
    sam_header h;
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        const nlohmann::json& v = it.value();
        if (key == "version") {
            // legacy field; no longer part of the header
            continue;
        } else if (key == "scanspline") {
            h.scanspline = as_int(v);
        } else if (key == "nlines") {
            h.nlines = as_int(v);
        } else if (key == "scanlen") {
            h.scanlen = as_int(v);
        } else if (key == "samplerate") {
            h.samplerate = as_double(v);
        } else if (key == "tzero") {
            h.tzero = as_int(v);
        } else if (key == "resolution") {
            h.resolution = as_double(v);
        } else if (key == "interpolated") {
            h.interpolated = as_bool(v);
        } else if (key == "quality") {
            h.quality = as_bool(v);
        } else if (key == "mode") {
            h.mode = as_string(v);
        } else if (key == "transducer_in") {
            h.transducer_in = as_string(v);
        } else if (key == "transducer_through") {
            h.transducer_through = as_string(v);
        } else if (key == "cellid") {
            h.cellid = as_string(v);
        } else if (key == "downsample_factor") {
            h.downsample_factor = as_int(v);
        } else if (key == "headerlen" || key == "bytes_p_sample") {
            // backwards compat: ignore
            continue;
        } else {
            if (v.is_string() && looks_like_json(v.get_ref<const std::string&>())) {
                // Keep the JSON text verbatim so round trips are lossless.
                h.extra[key] = v.get_ref<const std::string&>();
            } else if (v.is_boolean()) {
                h.extra[key] = v.get<bool>();
            } else if (v.is_number_integer() || v.is_number_unsigned()) {
                h.extra[key] = v.get<std::int64_t>();
            } else if (v.is_number_float()) {
                h.extra[key] = v.get<double>();
            } else if (v.is_string()) {
                h.extra[key] = v.get<std::string>();
            } else {
                // Nested structures arrive as pre-serialized JSON text.
                h.extra[key] = v.dump();
            }
        }
    }
    return h;
}

nlohmann::json sam_header::to_json() const {
    nlohmann::json j = nlohmann::json::object();
    j["scanspline"] = scanspline;
    j["nlines"] = nlines;
    j["interpolated"] = interpolated;
    j["scanlen"] = scanlen;
    j["samplerate"] = samplerate;
    j["tzero"] = tzero;
    j["quality"] = quality;
    j["resolution"] = resolution;
    j["mode"] = mode;
    j["transducer_in"] = transducer_in;
    j["transducer_through"] = transducer_through;
    j["cellid"] = cellid;
    j["downsample_factor"] = downsample_factor;
    for (const auto& [key, value] : extra) {
        if (std::holds_alternative<std::int64_t>(value)) {
            j[key] = std::get<std::int64_t>(value);
        } else if (std::holds_alternative<double>(value)) {
            j[key] = std::get<double>(value);
        } else if (std::holds_alternative<bool>(value)) {
            j[key] = std::get<bool>(value);
        } else {
            j[key] = std::get<std::string>(value);
        }
    }
    return j;
}

std::string sam_header::json_str() const { return to_json().dump(); }

std::vector<double> sam_header::time(std::int64_t start,
                                     std::int64_t end) const {
    if (end < start) end = scanlen;
    const std::int64_t num = end - start;
    if (num <= 0) return {};
    const double start_time = tzero + start / samplerate * 1e3;
    const double end_time = tzero + end / samplerate * 1e3;
    std::vector<double> out(static_cast<size_t>(num));
    if (num == 1) {
        out[0] = start_time;
        return out;
    }
    const double step = (end_time - start_time) / static_cast<double>(num - 1);
    for (std::int64_t i = 0; i < num; ++i) {
        out[static_cast<size_t>(i)] = start_time + step * static_cast<double>(i);
    }
    return out;
}

bool sam_header::operator==(const sam_header& o) const {
    return scanspline == o.scanspline &&
           nlines == o.nlines && scanlen == o.scanlen &&
           samplerate == o.samplerate && tzero == o.tzero &&
           resolution == o.resolution && interpolated == o.interpolated &&
           quality == o.quality && mode == o.mode &&
           transducer_in == o.transducer_in &&
           transducer_through == o.transducer_through &&
           cellid == o.cellid && downsample_factor == o.downsample_factor &&
           extra == o.extra;
}

size_t sam_header::hash() const {
    return std::hash<std::string>{}(json_str());
}

} // namespace samcore

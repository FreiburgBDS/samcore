#include <samcore/sam_header.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>

namespace samcore {

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
    size_t h = 0;
    auto mix = [&h](size_t x) { h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2); };
    mix(std::hash<std::int64_t>{}(scanspline));
    mix(std::hash<std::int64_t>{}(nlines));
    mix(std::hash<std::int64_t>{}(scanlen));
    mix(std::hash<double>{}(samplerate));
    mix(std::hash<std::int64_t>{}(tzero));
    mix(std::hash<double>{}(resolution));
    mix(std::hash<bool>{}(interpolated));
    mix(std::hash<bool>{}(quality));
    mix(std::hash<std::string>{}(mode));
    mix(std::hash<std::string>{}(transducer_in));
    mix(std::hash<std::string>{}(transducer_through));
    mix(std::hash<std::string>{}(cellid));
    mix(std::hash<std::int64_t>{}(downsample_factor));
    for (const auto& [key, value] : extra) {
        mix(std::hash<std::string>{}(key));
        std::visit(
            [&](const auto& v) {
                mix(std::hash<std::decay_t<decltype(v)>>{}(v));
            },
            value);
    }
    return h;
}

} // namespace samcore

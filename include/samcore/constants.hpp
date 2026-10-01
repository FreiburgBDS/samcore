#pragma once

// Shared constants of the SAM data model.

namespace samcore {

// Full-scale magnitude of the int8 SAM samples: |x| <= full_scale (127).
// Used by threshold-relative APIs such as align_zgate().
inline constexpr int full_scale = 127;

// Marker returned by tof() and thickness() for gates without an envelope
// peak (no echo), in place of a NaN.  Release builds run with -ffast-math,
// which makes NaN values unreliable across compilers, so "no echo" is a
// finite sentinel outside the valid time/thickness range.
inline constexpr float no_tof = -1.0f;

} // namespace samcore

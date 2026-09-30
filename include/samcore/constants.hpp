#pragma once

// Shared constants of the SAM data model.

namespace samcore {

// Full-scale magnitude of the int8 SAM samples: |x| <= full_scale (127).
// Used by threshold-relative APIs such as zgate().
inline constexpr int full_scale = 127;

} // namespace samcore

#pragma once

#include <samcore/array.hpp>

namespace samcore::preprocessing {

// All functions operate on 2-D signal arrays (n_signals, scanlen) in
// float32 and are OpenMP-parallel over signals.

// 3rd-order Butterworth low-pass filter.  cutoff and fs in Hz.
[[nodiscard]] array2d<float> lp(const array2d<float>& data, double cutoff,
                                double fs);

// 3rd-order Butterworth band-pass filter.
[[nodiscard]] array2d<float> bp(const array2d<float>& data, double cutoff_low,
                                double cutoff_high, double fs);

// Per-signal amplitude normalization.  mode: "max", "zscore", "minmax".
[[nodiscard]] array2d<float> normalize(const array2d<float>& data,
                                       const std::string& mode = "minmax");

// Savitzky-Golay smoothing (window_length odd, polyorder < window).
[[nodiscard]] array2d<float> savgol(const array2d<float>& data,
                                    size_t window_length = 5,
                                    size_t polyorder = 2);

// 1-D median filter (kernel_size odd, zero-padded edges).
[[nodiscard]] array2d<float> medfilt(const array2d<float>& data,
                                     size_t kernel_size = 3);

// Time-gate: samples [start, end); end == 0 means the full length.
// Throws on invalid ranges.
[[nodiscard]] array2d<float> gate(const array2d<float>& data, size_t start = 0,
                                  size_t end = 0);

// Remove the best-fit line from each signal.
[[nodiscard]] array2d<float> detrend(const array2d<float>& data);

// Hilbert-transform envelope magnitude of each signal.
[[nodiscard]] array2d<float> envelope(const array2d<float>& data);

// Boxcar moving average, mode='same' convolution semantics.
[[nodiscard]] array2d<float> moving_average(const array2d<float>& data,
                                            size_t window = 5);

} // namespace samcore::preprocessing

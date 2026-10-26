#pragma once

#include <samcore/array.hpp>

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace samcore::preprocessing {

// All functions operate on 2-D signal arrays (n_signals, scanlen) in
// float32 and are OpenMP-parallel over signals.
//
// Frequency parameters (cutoff*, fs) must share one frequency unit; the
// filters only use their ratio, so Hz or MHz both work as long as they
// match.  MHz matches sam_header::samplerate.

// 3rd-order Butterworth low-pass filter.
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

// 1-D median filter (kernel_size odd, whole-sample reflect padding;
// see signal::medfilt1d for the exact edge semantics).
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

// Analytic signal of each signal: z = x + i*H{x} (scipy.signal.hilbert
// parity), returned as complex64.  envelope(data) == abs(analytic(data))
// and the per-sample phase is arg(analytic(data)).
[[nodiscard]] array2d<std::complex<float>> analytic(const array2d<float>& data);

// Circular statistics of the instantaneous phase of each signal.
//
// For every signal, phase = arg(analytic(signal)) and
//
//     mean_sin = mean(sin(phase))
//     mean_cos = mean(cos(phase))
//     phase_mean = atan2(mean_sin, mean_cos)
//     phase_var  = 1 - hypot(mean_sin, mean_cos)
//
// phase_mean is the circular mean direction in radians; phase_var in [0, 1]
// measures the spread of the phases (0 = all equal, 1 = uniformly spread).
//
// Degenerate cases: a silent (all-zero) signal yields phase 0 at every
// sample (consistent with arg(0) = 0), hence phase_mean 0 and phase_var 0.
// If the phases cancel so that the resultant length
// hypot(mean_sin, mean_cos) is numerically zero (< 1e-12), the mean
// direction is undefined; phase_mean is then 0 and phase_var is 1.
// Returns {phase_mean, phase_var}, each of length data.rows().
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>> phase_stats(
    const array2d<float>& data);

// Boxcar moving average, mode='same' convolution semantics.
[[nodiscard]] array2d<float> moving_average(const array2d<float>& data,
                                            size_t window = 5);

} // namespace samcore::preprocessing

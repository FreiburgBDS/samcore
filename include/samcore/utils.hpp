#pragma once

#include <complex>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <samcore/array.hpp>

namespace samcore::utils {

// Pearson kurtosis (bias-corrected = False, fisher = False) per signal,
// matching scipy.stats.kurtosis(..., fisher=False).  Constant
// (zero-variance) signals return 0 instead of scipy's NaN: -ffast-math
// makes NaN values unreliable.
[[nodiscard]] std::vector<double> kurt(const array2d<float>& data);

// Sample-spaced time index tzero + i * delta_t for i in [0, num)
// (end-exclusive; delta_t is the sample spacing, not a span).  All
// three are in the same time unit (ns for SAM data).
[[nodiscard]] std::vector<double> time_index(double tzero, double delta_t,
                                             size_t num);

// Magnitude spectrum and frequency bins of a single signal (numpy
// rfft/rfftfreq parity).
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>> fft_spec(
    std::span<const float> scan, double d = 1.0);

// One-sided FFT magnitude of every signal (numpy rfft parity), float32.
// d is the sample spacing (1/fs) for the frequency axis; the returned
// frequencies are in 1/d units (d in ns -> GHz).  Returns
// {magnitude (n_signals, n_freqs), frequencies (n_freqs,)}.
[[nodiscard]] std::pair<array2d<float>, std::vector<float>> fft_spectrum(
    const array2d<float>& data, double d = 1.0);

// Normalised Shannon spectral entropy along the last axis.  Input is a
// PSD of shape (n_signals, n_freqs); silent rows return 0.
[[nodiscard]] std::vector<double> spectral_entropy(const array2d<float>& psd,
                                                   double base = 2.0);

// Spectral flatness (geometric / arithmetic mean) per row.
[[nodiscard]] std::vector<double> spectral_flatness(const array2d<float>& psd);

// PSD-weighted mean frequency per row; silent rows return 0.
[[nodiscard]] std::vector<double> spectral_centroid(
    std::span<const float> freqs, const array2d<float>& psd);

// Ratio of PSD energy at/above critical_freq to energy below it; rows
// with zero below-energy return 0.
[[nodiscard]] std::vector<double> spectral_energy_ratio(
    std::span<const float> freqs, const array2d<float>& psd,
    double critical_freq);

} // namespace samcore::utils

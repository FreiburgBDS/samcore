#pragma once

#include <complex>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <samcore/array.hpp>

namespace samcore::signal {

struct stft_result {
    std::vector<float> f;    // (n_freqs,) one-sided bins
    std::vector<float> t;    // (n_frames,) seconds
    array3d<std::complex<float>> zxx; // (n_signals, n_freqs, n_frames)
};

struct psd_result {
    std::vector<float> f;      // (n_freqs,)
    array2d<float> psd;        // (n_signals, n_freqs)
};

struct spectrogram_result {
    std::vector<float> f;      // (n_freqs,)
    std::vector<float> t;      // (n_frames,) seconds
    array3d<float> sxx;        // (n_signals, n_freqs, n_frames)
};

// FFT / windows

// Real-input one-sided FFT of a signal.  Complex output size n/2+1.
[[nodiscard]] std::vector<std::complex<double>> rfft(
    std::span<const double> x);

// Inverse of rfft: length-n real signal from the one-sided spectrum.
[[nodiscard]] std::vector<double> irfft(
    std::span<const std::complex<double>> X, size_t n);

// Periodic ("fftbins=True") Hann window of length n: 0.5-0.5*cos(2*pi*k/n).
[[nodiscard]] std::vector<double> hann_window(size_t n);

// Spectral estimators (scipy.signal parity)

// One-sided STFT, scaling='spectrum', detrend=False, boundary=None,
// padded=False.  fs in Hz.
[[nodiscard]] stft_result stft(const array2d<float>& data, double fs,
                               size_t nperseg, size_t noverlap);

// Welch PSD, scaling='density', detrend=False, averaged over frames.
[[nodiscard]] psd_result welch_psd(const array2d<float>& data, double fs,
                                   size_t nperseg, size_t noverlap);

// Per-frame power spectral density (scipy.signal.spectrogram, mode='psd').
[[nodiscard]] spectrogram_result spectrogram_psd(const array2d<float>& data,
                                                 double fs, size_t nperseg,
                                                 size_t noverlap);

// IIR filters

struct sos { // second-order section, ba form
    double b0, b1, b2, a0, a1, a2;
};

// 3rd-order Butterworth low/band-pass in ba form (analog prototype +
// bilinear), scipy.signal.butter(3, ...) parity.
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>>
butter_lowpass(double cutoff, double fs);
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>>
butter_bandpass(double cutoff_low, double cutoff_high, double fs);

// Chebyshev type I filter in SOS form, scipy.signal.cheby1 parity.
[[nodiscard]] std::vector<sos> cheby1_sos(size_t order, double rp, double wn);

// Direct form II transposed (lfilter with zero initial conditions),
// double precision, matching scipy.signal.lfilter on float64 inputs.
[[nodiscard]] std::vector<double> lfilter(const std::vector<double>& b,
                                          const std::vector<double>& a,
                                          std::span<const double> x);

// Zero-phase forward-backward filtering with scipy's default odd
// signal extension and padlen = 3*max(len(a), len(b)).
[[nodiscard]] std::vector<double> filtfilt(const std::vector<double>& b,
                                           const std::vector<double>& a,
                                           std::span<const double> x);

// scipy.signal.decimate(x, q, ftype='iir') parity on a single signal.
[[nodiscard]] std::vector<double> decimate(std::span<const double> x,
                                           size_t q);

// FIR / nonlinear

// Analytic-signal magnitude via FFT (scipy.signal.hilbert parity).
[[nodiscard]] std::vector<double> hilbert_envelope(std::span<const double> x);

// Savitzky-Golay filter coefficients for a window (odd) and polyorder,
// from a least-squares polynomial fit (scipy.savgol_coeffs parity).
[[nodiscard]] std::vector<double> savgol_coeffs(size_t window_length,
                                                size_t polyorder);

// 1-D median filter with zero-padded edges, odd kernel (scipy.ndimage
// semantics for medfilt).
[[nodiscard]] std::vector<double> medfilt1d(std::span<const double> x,
                                            size_t kernel_size);

} // namespace samcore::signal

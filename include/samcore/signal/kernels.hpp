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

// One-sided FFT magnitude of every signal (numpy.fft.rfft parity).
struct spectrum_result {
    std::vector<float> f;   // (n_freqs,) Hz (d is the sample spacing)
    array2d<float> mag;     // (n_signals, n_freqs)
};

// Per-signal STFT magnitude peaks.
struct stft_peaks_result {
    std::vector<float> peak_frequency; // (n_signals,) Hz
    std::vector<float> peak_time;      // (n_signals,) seconds
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
// padded=False.  fs in Hz.  With f_max > 0 only bins in [f_min, f_max] are
// returned (f_max = 0 means Nyquist).
[[nodiscard]] stft_result stft(const array2d<float>& data, double fs,
                               size_t nperseg, size_t noverlap,
                               double f_min = 0.0, double f_max = 0.0);

// Welch PSD, scaling='density', detrend=False, averaged over frames.
[[nodiscard]] psd_result welch_psd(const array2d<float>& data, double fs,
                                   size_t nperseg, size_t noverlap,
                                   double f_min = 0.0, double f_max = 0.0);

// Per-frame power spectral density (scipy.signal.spectrogram, mode='psd').
[[nodiscard]] spectrogram_result spectrogram_psd(const array2d<float>& data,
                                                 double fs, size_t nperseg,
                                                 size_t noverlap,
                                                 double f_min = 0.0,
                                                 double f_max = 0.0);

// Per-signal peaks of the STFT magnitude: the frequency with the largest
// max-over-frames magnitude and the time with the largest max-over-frequencies
// magnitude.  Band-limited like stft().
[[nodiscard]] stft_peaks_result stft_peaks(const array2d<float>& data,
                                           double fs, size_t nperseg,
                                           size_t noverlap, double f_min = 0.0,
                                           double f_max = 0.0);

// One-sided FFT magnitude of every signal, batched through pocketfft.  d is
// the sample spacing (1/fs) for the frequency axis; values match
// np.abs(np.fft.rfft(data, axis=1)).
[[nodiscard]] spectrum_result fft_spectrum(const array2d<float>& data,
                                           double d = 1.0);

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

// Design the anti-aliasing filter used by scipy.signal.decimate (Chebyshev
// type I, order 8, 0.05 dB ripple, cutoff 0.8/q).  Exposed so callers can
// filter many signals with one design instead of redesigning per signal.
[[nodiscard]] std::vector<sos> decimate_sos(size_t q);

// scipy.signal.decimate(x, q, ftype='iir') parity on a single signal.
[[nodiscard]] std::vector<double> decimate(std::span<const double> x,
                                           size_t q);

// int8 fast path (SAM data) with a pre-computed design; a single fused
// cascade pass per signal, no int8->double staging buffer.
[[nodiscard]] std::vector<double> decimate(std::span<const std::int8_t> x,
                                           size_t q,
                                           std::span<const sos> sections);

// FIR / nonlinear

// Analytic-signal magnitude via FFT (scipy.signal.hilbert parity).
[[nodiscard]] std::vector<double> hilbert_envelope(std::span<const double> x);

// Savitzky-Golay filter coefficients for a window (odd) and polyorder,
// from a least-squares polynomial fit (scipy.savgol_coeffs parity).
[[nodiscard]] std::vector<double> savgol_coeffs(size_t window_length,
                                                size_t polyorder);

// 1-D median filter with an odd kernel and whole-sample reflect padding
// (numpy.pad mode='reflect' semantics: x[-1] mirrors x[1]).  Interior
// samples match scipy.signal.medfilt / scipy.ndimage.median_filter; the
// edge padding deliberately differs from scipy.signal.medfilt (which
// zero-pads) and from scipy.ndimage's mode='reflect' (which duplicates the
// edge sample).
[[nodiscard]] std::vector<double> medfilt1d(std::span<const double> x,
                                            size_t kernel_size);

// Integer lag of `x` relative to `ref` maximizing the mean-subtracted
// cross-correlation over lags [-max_shift, max_shift].  A positive lag means
// `x` is delayed by that many samples relative to `ref`.  Both signals must
// have the same length.  FFT-based (zero-padded to 2n); ties prefer the
// positive lag and then the smaller |lag|.
[[nodiscard]] std::int64_t xcorr_lag(std::span<const std::int8_t> x,
                                     std::span<const std::int8_t> ref,
                                     std::int64_t max_shift);

} // namespace samcore::signal

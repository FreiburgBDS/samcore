#include <samcore/signal/kernels.hpp>

#include <pocketfft_hdronly.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>

#ifdef SAMCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace samcore::signal {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

using pocketfft::shape_t;
using pocketfft::stride_t;

// One-sided frequency bins: k*fs/n for k = 0..n/2 (np.fft.rfftfreq parity).
std::vector<float> rfftfreq(size_t n, double d) {
    std::vector<float> f(n / 2 + 1);
    for (size_t k = 0; k <= n / 2; ++k) {
        f[k] = static_cast<float>(static_cast<double>(k) / (d * static_cast<double>(n)));
    }
    return f;
}

// Batched one-sided FFT over windowed frames: input (n_signals, n_frames,
// nperseg) float32 -> output (n_signals, n_frames, nfreqs) complex64,
// transformed along the last axis in a single pocketfft call (the same
// batching strategy numpy/scipy use; scipy also computes in single
// precision when the input is int8/float32).
void batch_rfft_frames(const std::vector<float>& in, size_t nsig,
                       size_t nframes, size_t nperseg,
                       std::vector<std::complex<float>>& out) {
    const size_t nfreqs = nperseg / 2 + 1;
    out.resize(nsig * nframes * nfreqs);
    const shape_t shape{nsig, nframes, nperseg};
    const stride_t stride_in{
        static_cast<ptrdiff_t>(nframes * nperseg * sizeof(float)),
        static_cast<ptrdiff_t>(nperseg * sizeof(float)),
        static_cast<ptrdiff_t>(sizeof(float))};
    const stride_t stride_out{
        static_cast<ptrdiff_t>(nframes * nfreqs * sizeof(std::complex<float>)),
        static_cast<ptrdiff_t>(nfreqs * sizeof(std::complex<float>)),
        static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
    pocketfft::r2c(shape, stride_in, stride_out, shape_t{2}, true, in.data(),
                   out.data(), 1.0f);
}

// Fill a (chunk, nframes, nperseg) buffer with windowed frames of the
// signals [s0, s0+chunk).
void fill_windowed(const array2d<float>& data, size_t s0, size_t chunk,
                   size_t nperseg, size_t step,
                   const std::vector<float>& win, std::vector<float>& buf) {
    const size_t nframes = buf.size() / (chunk * nperseg);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 4)
#endif
    for (size_t i = 0; i < chunk; ++i) {
        const auto row = data[s0 + i];
        for (size_t fr = 0; fr < nframes; ++fr) {
            const float* src = row.data() + fr * step;
            float* dst = buf.data() + (i * nframes + fr) * nperseg;
            for (size_t k = 0; k < nperseg; ++k) {
                dst[k] = src[k] * win[k];
            }
        }
    }
}

// Signals per FFT batch (bounds temporary memory).
constexpr size_t BATCH_SIGNALS = 256;

// Inclusive bin range [k0, k1) for the [f_min, f_max] band of a one-sided
// frequency axis.  f_max <= 0 means the Nyquist bin.
std::pair<size_t, size_t> band_bins(const std::vector<float>& f, double f_min,
                                    double f_max) {
    if (f.empty()) return {0, 0};
    if (f_min < 0.0) {
        throw std::invalid_argument("spectral band: f_min must be >= 0");
    }
    const double nyquist = f.back();
    if (f_max <= 0.0) f_max = nyquist;
    if (f_max > nyquist * (1.0 + 1e-9)) {
        throw std::invalid_argument("spectral band: f_max exceeds Nyquist");
    }
    if (f_min > f_max) {
        throw std::invalid_argument("spectral band: f_min must be <= f_max");
    }
    size_t k0 = 0;
    while (k0 < f.size() && static_cast<double>(f[k0]) < f_min) ++k0;
    size_t k1 = k0;
    while (k1 < f.size() && static_cast<double>(f[k1]) <= f_max) ++k1;
    if (k0 >= k1) {
        throw std::invalid_argument(
            "spectral band contains no frequency bins");
    }
    return {k0, k1};
}

// Sample times of the STFT frame centres.
std::vector<float> frame_times(size_t n, size_t nperseg, size_t step,
                               double fs) {
    const double start = static_cast<double>(nperseg) / 2.0;
    const double stop = static_cast<double>(n) -
                        static_cast<double>(nperseg) / 2.0 + 1.0;
    const size_t nt =
        static_cast<size_t>(std::ceil((stop - start) / static_cast<double>(step)));
    std::vector<float> t(nt);
    for (size_t i = 0; i < nt; ++i) {
        t[i] = static_cast<float>((start + static_cast<double>(i * step)) / fs);
    }
    return t;
}

} // namespace

std::vector<std::complex<double>> rfft(std::span<const double> x) {
    const shape_t shape{x.size()};
    const stride_t stride_in{sizeof(double)};
    const stride_t stride_out{sizeof(std::complex<double>)};
    std::vector<std::complex<double>> out(x.size() / 2 + 1);
    pocketfft::r2c(shape, stride_in, stride_out, 0, true, x.data(), out.data(),
                   1.0);
    return out;
}

std::vector<double> irfft(std::span<const std::complex<double>> X, size_t n) {
    const shape_t shape{n};
    const stride_t stride_in{sizeof(std::complex<double>)};
    const stride_t stride_out{sizeof(double)};
    std::vector<double> out(n);
    pocketfft::c2r(shape, stride_in, stride_out, 0, false, X.data(), out.data(),
                   1.0 / static_cast<double>(n));
    return out;
}

std::vector<double> hann_window(size_t n) {
    std::vector<double> w(n);
    for (size_t k = 0; k < n; ++k) {
        w[k] = 0.5 - 0.5 * std::cos(2.0 * PI * static_cast<double>(k) /
                                    static_cast<double>(n));
    }
    return w;
}

stft_result stft(const array2d<float>& data, double fs, size_t nperseg,
                 size_t noverlap, double f_min, double f_max) {
    if (nperseg < 1 || noverlap >= nperseg) {
        throw std::invalid_argument("stft: invalid nperseg/noverlap");
    }
    const size_t n = data.cols();
    if (n < nperseg) {
        throw std::invalid_argument("stft: nperseg cannot be greater than the length of the signals.");
    }
    const size_t step = nperseg - noverlap;
    const size_t nframes = (n - nperseg) / step + 1;
    const size_t nfreqs = nperseg / 2 + 1;
    const auto all_f = rfftfreq(nperseg, 1.0 / fs);
    const auto [k0, k1] = band_bins(all_f, f_min, f_max);
    const size_t nkept = k1 - k0;
    const auto win_d = hann_window(nperseg);
    std::vector<float> win(win_d.begin(), win_d.end());
    float win_sum = 0.0f;
    for (float w : win) win_sum += w;

    stft_result result;
    result.f.assign(all_f.begin() + static_cast<std::ptrdiff_t>(k0),
                    all_f.begin() + static_cast<std::ptrdiff_t>(k1));
    result.t = frame_times(n, nperseg, step, fs);
    result.zxx = array3d<std::complex<float>>(data.rows(), nkept, nframes);

    std::vector<float> buf;
    std::vector<std::complex<float>> spec;
    for (size_t s0 = 0; s0 < data.rows(); s0 += BATCH_SIGNALS) {
        const size_t chunk = std::min<size_t>(BATCH_SIGNALS, data.rows() - s0);
        buf.resize(chunk * nframes * nperseg);
        fill_windowed(data, s0, chunk, nperseg, step, win, buf);
        batch_rfft_frames(buf, chunk, nframes, nperseg, spec);
        const float scale = 1.0f / win_sum;
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 4)
#endif
        for (size_t i = 0; i < chunk; ++i) {
            const size_t sig = s0 + i;
            for (size_t fr = 0; fr < nframes; ++fr) {
                for (size_t k = k0; k < k1; ++k) {
                    const auto v = spec[(i * nframes + fr) * nfreqs + k];
                    result.zxx.flat()[(sig * nkept + (k - k0)) * nframes + fr] =
                        std::complex<float>(v.real() * scale, v.imag() * scale);
                }
            }
        }
    }
    return result;
}

psd_result welch_psd(const array2d<float>& data, double fs, size_t nperseg,
                     size_t noverlap, double f_min, double f_max) {
    if (nperseg < 1 || noverlap >= nperseg) {
        throw std::invalid_argument("welch_psd: invalid nperseg/noverlap");
    }
    const size_t n = data.cols();
    if (n < nperseg) {
        throw std::invalid_argument("welch_psd: nperseg cannot be greater than the length of the signals.");
    }
    const size_t step = nperseg - noverlap;
    const size_t nframes = (n - nperseg) / step + 1;
    const size_t nfreqs = nperseg / 2 + 1;
    const auto all_f = rfftfreq(nperseg, 1.0 / fs);
    const auto [k0, k1] = band_bins(all_f, f_min, f_max);
    const size_t nkept = k1 - k0;
    const auto win_d = hann_window(nperseg);
    std::vector<float> win(win_d.begin(), win_d.end());
    float win2_sum = 0.0f;
    for (float w : win) win2_sum += w * w;
    const double scale = 1.0 / (fs * win2_sum);
    const bool even = nperseg % 2 == 0;

    psd_result result;
    result.f.assign(all_f.begin() + static_cast<std::ptrdiff_t>(k0),
                    all_f.begin() + static_cast<std::ptrdiff_t>(k1));
    result.psd = array2d<float>(data.rows(), nkept, 0.0f);

    std::vector<float> buf;
    std::vector<std::complex<float>> spec;
    std::vector<float> acc(nkept);
    for (size_t s0 = 0; s0 < data.rows(); s0 += BATCH_SIGNALS) {
        const size_t chunk = std::min<size_t>(BATCH_SIGNALS, data.rows() - s0);
        buf.resize(chunk * nframes * nperseg);
        fill_windowed(data, s0, chunk, nperseg, step, win, buf);
        batch_rfft_frames(buf, chunk, nframes, nperseg, spec);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 4) firstprivate(acc)
#endif
        for (size_t i = 0; i < chunk; ++i) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t fr = 0; fr < nframes; ++fr) {
                for (size_t k = k0; k < k1; ++k) {
                    acc[k - k0] +=
                        std::norm(spec[(i * nframes + fr) * nfreqs + k]);
                }
            }
            auto out = result.psd[s0 + i];
            for (size_t k = k0; k < k1; ++k) {
                float v = acc[k - k0] / static_cast<float>(nframes) *
                          static_cast<float>(scale);
                if (k > 0 && (even ? k + 1 < nfreqs : true)) v *= 2.0f;
                out[k - k0] = v;
            }
        }
    }
    return result;
}

spectrogram_result spectrogram_psd(const array2d<float>& data, double fs,
                                   size_t nperseg, size_t noverlap,
                                   double f_min, double f_max) {
    if (nperseg < 1 || noverlap >= nperseg) {
        throw std::invalid_argument("spectrogram_psd: invalid nperseg/noverlap");
    }
    const size_t n = data.cols();
    if (n < nperseg) {
        throw std::invalid_argument("spectrogram_psd: nperseg cannot be greater than the length of the signals.");
    }
    const size_t step = nperseg - noverlap;
    const size_t nframes = (n - nperseg) / step + 1;
    const size_t nfreqs = nperseg / 2 + 1;
    const auto all_f = rfftfreq(nperseg, 1.0 / fs);
    const auto [k0, k1] = band_bins(all_f, f_min, f_max);
    const size_t nkept = k1 - k0;
    const auto win_d = hann_window(nperseg);
    std::vector<float> win(win_d.begin(), win_d.end());
    float win2_sum = 0.0f;
    for (float w : win) win2_sum += w * w;
    const double scale = 1.0 / (fs * win2_sum);
    const bool even = nperseg % 2 == 0;

    spectrogram_result result;
    result.f.assign(all_f.begin() + static_cast<std::ptrdiff_t>(k0),
                    all_f.begin() + static_cast<std::ptrdiff_t>(k1));
    result.t = frame_times(n, nperseg, step, fs);
    result.sxx = array3d<float>(data.rows(), nkept, nframes);

    std::vector<float> buf;
    std::vector<std::complex<float>> spec;
    for (size_t s0 = 0; s0 < data.rows(); s0 += BATCH_SIGNALS) {
        const size_t chunk = std::min<size_t>(BATCH_SIGNALS, data.rows() - s0);
        buf.resize(chunk * nframes * nperseg);
        fill_windowed(data, s0, chunk, nperseg, step, win, buf);
        batch_rfft_frames(buf, chunk, nframes, nperseg, spec);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 4)
#endif
        for (size_t i = 0; i < chunk; ++i) {
            const size_t sig = s0 + i;
            for (size_t fr = 0; fr < nframes; ++fr) {
                for (size_t k = k0; k < k1; ++k) {
                    float v = std::norm(spec[(i * nframes + fr) * nfreqs + k]) *
                              static_cast<float>(scale);
                    if (k > 0 && (even ? k + 1 < nfreqs : true)) v *= 2.0f;
                    result.sxx.flat()[(sig * nkept + (k - k0)) * nframes + fr] = v;
                }
            }
        }
    }
    return result;
}

stft_peaks_result stft_peaks(const array2d<float>& data, double fs,
                             size_t nperseg, size_t noverlap, double f_min,
                             double f_max) {
    if (nperseg < 1 || noverlap >= nperseg) {
        throw std::invalid_argument("stft_peaks: invalid nperseg/noverlap");
    }
    const size_t n = data.cols();
    if (n < nperseg) {
        throw std::invalid_argument("stft_peaks: nperseg cannot be greater than the length of the signals.");
    }
    const size_t step = nperseg - noverlap;
    const size_t nframes = (n - nperseg) / step + 1;
    const size_t nfreqs = nperseg / 2 + 1;
    const auto all_f = rfftfreq(nperseg, 1.0 / fs);
    const auto [k0, k1] = band_bins(all_f, f_min, f_max);
    const size_t nkept = k1 - k0;
    const auto times = frame_times(n, nperseg, step, fs);
    const auto win_d = hann_window(nperseg);
    std::vector<float> win(win_d.begin(), win_d.end());

    stft_peaks_result result;
    result.peak_frequency.resize(data.rows());
    result.peak_time.resize(data.rows());

    std::vector<float> buf;
    std::vector<std::complex<float>> spec;
    for (size_t s0 = 0; s0 < data.rows(); s0 += BATCH_SIGNALS) {
        const size_t chunk = std::min<size_t>(BATCH_SIGNALS, data.rows() - s0);
        buf.resize(chunk * nframes * nperseg);
        fill_windowed(data, s0, chunk, nperseg, step, win, buf);
        batch_rfft_frames(buf, chunk, nframes, nperseg, spec);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 4)
#endif
        for (size_t i = 0; i < chunk; ++i) {
            std::vector<float> best_bin(nkept, 0.0f);
            std::vector<float> best_frame(nframes, 0.0f);
            for (size_t fr = 0; fr < nframes; ++fr) {
                for (size_t k = k0; k < k1; ++k) {
                    const float mag =
                        std::abs(spec[(i * nframes + fr) * nfreqs + k]);
                    float& b = best_bin[k - k0];
                    if (mag > b) b = mag;
                    if (mag > best_frame[fr]) best_frame[fr] = mag;
                }
            }
            size_t bin_idx = 0;
            for (size_t j = 1; j < nkept; ++j) {
                if (best_bin[j] > best_bin[bin_idx]) bin_idx = j;
            }
            size_t frame_idx = 0;
            for (size_t j = 1; j < nframes; ++j) {
                if (best_frame[j] > best_frame[frame_idx]) frame_idx = j;
            }
            result.peak_frequency[s0 + i] = all_f[k0 + bin_idx];
            result.peak_time[s0 + i] = times[frame_idx];
        }
    }
    return result;
}

spectrum_result fft_spectrum(const array2d<float>& data, double d) {
    const size_t n = data.cols();
    const size_t nfreqs = n / 2 + 1;
    spectrum_result result;
    result.f = rfftfreq(n, d);
    result.mag = array2d<float>(data.rows(), nfreqs);
    if (data.rows() == 0 || n == 0) return result;

    std::vector<std::complex<float>> spec(BATCH_SIGNALS * nfreqs);
    for (size_t s0 = 0; s0 < data.rows(); s0 += BATCH_SIGNALS) {
        const size_t chunk = std::min<size_t>(BATCH_SIGNALS, data.rows() - s0);
        const shape_t shape{chunk, n};
        const stride_t stride_in{
            static_cast<ptrdiff_t>(n * sizeof(float)),
            static_cast<ptrdiff_t>(sizeof(float))};
        const stride_t stride_out{
            static_cast<ptrdiff_t>(nfreqs * sizeof(std::complex<float>)),
            static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
        pocketfft::r2c(shape, stride_in, stride_out, shape_t{1}, true,
                       data.data() + s0 * n, spec.data(), 1.0f);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (chunk > 8)
#endif
        for (size_t i = 0; i < chunk; ++i) {
            auto row = result.mag[s0 + i];
            const auto* src = spec.data() + i * nfreqs;
            for (size_t k = 0; k < nfreqs; ++k) {
                row[k] = std::abs(src[k]);
            }
        }
    }
    return result;
}

std::vector<double> hilbert_envelope(std::span<const double> x) {
    const size_t n = x.size();
    if (n == 0) return {};
    // One-sided spectrum, then the analytic signal (scipy.signal.hilbert):
    // X_a[0] = X[0]; X_a[k] = 2*X[k] for 1 <= k < n/2; X_a[n/2] = X[n/2]
    // (n even); X_a[k] = 0 for k > n/2.
    const auto X = rfft(x);
    std::vector<std::complex<double>> analytic_spec(n, {0.0, 0.0});
    analytic_spec[0] = X[0];
    for (size_t k = 1; k < n / 2; ++k) analytic_spec[k] = 2.0 * X[k];
    if (n % 2 == 0) {
        analytic_spec[n / 2] = X[n / 2];
    } else {
        // n odd: double the Nyquist-adjacent bin too (no Nyquist bin).
        analytic_spec[n / 2] = 2.0 * X[n / 2];
    }
    const shape_t shape{n};
    const stride_t strides{sizeof(std::complex<double>)};
    std::vector<std::complex<double>> analytic(n);
    pocketfft::c2c(shape, strides, strides, shape_t{0}, false,
                   analytic_spec.data(), analytic.data(),
                   1.0 / static_cast<double>(n));
    std::vector<double> env(n);
    for (size_t k = 0; k < n; ++k) env[k] = std::abs(analytic[k]);
    return env;
}

} // namespace samcore::signal

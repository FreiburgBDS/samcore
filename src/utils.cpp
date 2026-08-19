#include <samcore/utils.hpp>

#include <samcore/signal/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef SAMCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace samcore::utils {

std::vector<double> kurt(const array2d<float>& data) {
    std::vector<double> out(data.rows());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        const size_t n = data.cols();
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) mean += data[s][i];
        mean /= static_cast<double>(n);
        double m2 = 0.0, m4 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(data[s][i]) - mean;
            const double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= static_cast<double>(n);
        m4 /= static_cast<double>(n);
        // scipy parity: kurtosis of a constant (zero-variance) signal is NaN
        out[s] = m2 == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                           : m4 / (m2 * m2);
    }
    return out;
}

std::vector<double> time_index(double tzero, double delta_t, size_t num) {
    if (num == 0) return {};
    if (num == 1) return {tzero};
    std::vector<double> out(num);
    for (size_t i = 0; i < num; ++i) {
        out[i] = tzero + delta_t * static_cast<double>(i) /
                              static_cast<double>(num - 1);
    }
    return out;
}

std::pair<std::vector<double>, std::vector<double>> fft_spec(
    std::span<const float> scan, double d) {
    const size_t n = scan.size();
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = scan[i];
    const auto spec = signal::rfft(x);
    std::vector<double> mag(spec.size());
    for (size_t i = 0; i < spec.size(); ++i) mag[i] = std::abs(spec[i]);
    std::vector<double> freqs(spec.size());
    for (size_t k = 0; k < spec.size(); ++k) {
        freqs[k] = static_cast<double>(k) / (d * static_cast<double>(n));
    }
    return {std::move(mag), std::move(freqs)};
}

std::vector<double> spectral_entropy(const array2d<float>& psd, double base) {
    const size_t rows = psd.rows();
    const size_t n = psd.cols();
    std::vector<double> out(rows, 0.0);
    const double inv_log_base = 1.0 / std::log(base);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (size_t s = 0; s < rows; ++s) {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i) total += psd[s][i];
        if (total == 0.0) continue; // silent row -> entropy 0
        double ent = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double p = psd[s][i] / total;
            if (p > 0.0) ent += p * std::log(p);
        }
        out[s] = -ent * inv_log_base;
    }
    return out;
}

std::vector<double> spectral_flatness(const array2d<float>& psd) {
    const size_t rows = psd.rows();
    const size_t n = psd.cols();
    std::vector<double> out(rows, 0.0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (size_t s = 0; s < rows; ++s) {
        double am = 0.0;
        double gm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            am += psd[s][i];
            gm += std::log(static_cast<double>(psd[s][i]) + 1e-10);
        }
        am /= static_cast<double>(n);
        if (am == 0.0) continue; // silent row -> flatness 0
        gm = std::exp(gm / static_cast<double>(n));
        out[s] = gm / am;
    }
    return out;
}

std::vector<double> spectral_centroid(std::span<const float> freqs,
                                      const array2d<float>& psd) {
    const size_t rows = psd.rows();
    const size_t n = psd.cols();
    if (freqs.size() != n) {
        throw std::invalid_argument("spectral_centroid: freqs/psd length mismatch");
    }
    std::vector<double> out(rows, 0.0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (size_t s = 0; s < rows; ++s) {
        double total = 0.0, weighted = 0.0;
        for (size_t i = 0; i < n; ++i) {
            total += psd[s][i];
            weighted += freqs[i] * psd[s][i];
        }
        out[s] = total == 0.0 ? 0.0 : weighted / total;
    }
    return out;
}

std::vector<double> spectral_energy_ratio(std::span<const float> freqs,
                                          const array2d<float>& psd,
                                          double critical_freq) {
    const size_t rows = psd.rows();
    const size_t n = psd.cols();
    if (freqs.size() != n) {
        throw std::invalid_argument("spectral_energy_ratio: freqs/psd length mismatch");
    }
    std::vector<double> out(rows, 0.0);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (size_t s = 0; s < rows; ++s) {
        double above = 0.0, below = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (freqs[i] >= critical_freq) {
                above += psd[s][i];
            } else {
                below += psd[s][i];
            }
        }
        out[s] = below == 0.0 ? 0.0 : above / below;
    }
    return out;
}

} // namespace samcore::utils

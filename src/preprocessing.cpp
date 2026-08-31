#include <samcore/preprocessing.hpp>

#include <samcore/signal/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef SAMCORE_HAS_OPENMP
#include <omp.h>
#endif

namespace samcore::preprocessing {

namespace {

void validate_rows(const array2d<float>& data) {
    if (data.cols() == 0) {
        throw std::invalid_argument("preprocessing: empty signal rows");
    }
}

// scipy.signal.savgol_filter mode='interp': convolution with 'constant'
// (zero) padding, then refit the edges with a least-squares polynomial.
array2d<float> savgol_filter_impl(const array2d<float>& data,
                                  size_t window_length, size_t polyorder) {
    const size_t n = data.cols();
    const size_t rows = data.rows();
    if (window_length > n) {
        throw std::invalid_argument(
            "If mode is 'interp', window_length must be less than or equal to the size of x.");
    }
    const std::vector<double> coeffs =
        signal::savgol_coeffs(window_length, polyorder);
    const size_t half = window_length / 2;

    array2d<float> out(rows, n);
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 4)
#endif
    for (size_t s = 0; s < rows; ++s) {
        const auto in = data[s];
        auto o = out[s];
        // Convolve with zero padding: out[i] = sum_k c[k] * x[i + k - half].
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t k = 0; k < window_length; ++k) {
                const int64_t idx = static_cast<int64_t>(i) +
                                    static_cast<int64_t>(k) -
                                    static_cast<int64_t>(half);
                if (idx >= 0 && static_cast<size_t>(idx) < n) {
                    acc += coeffs[k] * in[static_cast<size_t>(idx)];
                }
            }
            o[i] = static_cast<float>(acc);
        }
    }

    // Edge refit: least-squares polyfit over the first/last window_length
    // samples evaluated at the outer half-window (scipy _fit_edge parity).
    auto fit_edge = [&](size_t window_start, size_t interp_start,
                        size_t interp_stop) {
        const size_t p = polyorder + 1;
        const size_t w = window_length;
        // Vandermonde on x = 0..w-1, powers 0..polyorder (col-major fit
        // solving min ||A c - y|| via normal equations).
        std::vector<double> A(w * p);
        for (size_t i = 0; i < w; ++i) {
            double v = 1.0;
            for (size_t j = 0; j < p; ++j) {
                A[i * p + j] = v;
                v *= static_cast<double>(i);
            }
        }
        std::vector<double> AtA(p * p, 0.0);
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                double acc = 0.0;
                for (size_t i = 0; i < w; ++i) {
                    acc += A[i * p + r] * A[i * p + c];
                }
                AtA[r * p + c] = acc;
            }
        }
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (rows > 4)
#endif
        for (size_t s = 0; s < rows; ++s) {
            std::vector<double> Atb(p, 0.0);
            for (size_t r = 0; r < p; ++r) {
                double acc = 0.0;
                for (size_t i = 0; i < w; ++i) {
                    acc += A[i * p + r] * data[s][window_start + i];
                }
                Atb[r] = acc;
            }
            // Solve with plain Gaussian elimination (n <= polyorder+1 <= 6).
            std::vector<double> M = AtA;
            std::vector<double> rhs = Atb;
            for (size_t col = 0; col < p; ++col) {
                size_t piv = col;
                double best = std::fabs(M[col * p + col]);
                for (size_t r = col + 1; r < p; ++r) {
                    if (std::fabs(M[r * p + col]) > best) {
                        best = std::fabs(M[r * p + col]);
                        piv = r;
                    }
                }
                if (piv != col) {
                    for (size_t c = 0; c < p; ++c) std::swap(M[col * p + c], M[piv * p + c]);
                    std::swap(rhs[col], rhs[piv]);
                }
                for (size_t r = col + 1; r < p; ++r) {
                    const double f = M[r * p + col] / M[col * p + col];
                    if (f == 0.0) continue;
                    for (size_t c = col; c < p; ++c) M[r * p + c] -= f * M[col * p + c];
                    rhs[r] -= f * rhs[col];
                }
            }
            std::vector<double> c(p);
            for (size_t r = p; r-- > 0;) {
                double acc = rhs[r];
                for (size_t cc = r + 1; cc < p; ++cc) acc -= M[r * p + cc] * c[cc];
                c[r] = acc / M[r * p + r];
            }
            // Evaluate polynomial at t = i - window_start for i in
            // [interp_start, interp_stop).  c holds x^0..x^p coefficients
            // (lowest first); Horner must run highest-degree first
            // (np.polyval parity).
            for (size_t i = interp_start; i < interp_stop; ++i) {
                const double t = static_cast<double>(i - window_start);
                double v = 0.0;
                for (size_t r = p; r-- > 0;) {
                    v = v * t + c[r];
                }
                out[s][i] = static_cast<float>(v);
            }
        }
    };

    fit_edge(0, 0, half);
    fit_edge(n - window_length, n - half, n);
    return out;
}

} // namespace

array2d<float> lp(const array2d<float>& data, double cutoff, double fs) {
    validate_rows(data);
    const auto filter = signal::butter_lowpass(cutoff, fs);
    const auto& b = filter.first;
    const auto& a = filter.second;
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        std::vector<double> row(data.cols());
        for (size_t i = 0; i < data.cols(); ++i) row[i] = data[s][i];
        const auto y = signal::lfilter(b, a, row);
        for (size_t i = 0; i < data.cols(); ++i) out[s][i] = static_cast<float>(y[i]);
    }
    return out;
}

array2d<float> bp(const array2d<float>& data, double cutoff_low,
                  double cutoff_high, double fs) {
    validate_rows(data);
    const auto filter = signal::butter_bandpass(cutoff_low, cutoff_high, fs);
    const auto& b = filter.first;
    const auto& a = filter.second;
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        std::vector<double> row(data.cols());
        for (size_t i = 0; i < data.cols(); ++i) row[i] = data[s][i];
        const auto y = signal::lfilter(b, a, row);
        for (size_t i = 0; i < data.cols(); ++i) out[s][i] = static_cast<float>(y[i]);
    }
    return out;
}

array2d<float> normalize(const array2d<float>& data, const std::string& mode) {
    if (mode != "max" && mode != "zscore" && mode != "minmax") {
        throw std::invalid_argument("Unknown normalization mode: " + mode);
    }
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        const size_t n = data.cols();
        auto o = out[s];
        if (mode == "max") {
            double m = 0.0;
            for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(data[s][i])));
            if (m == 0.0) m = 1.0;
            for (size_t i = 0; i < n; ++i) o[i] = static_cast<float>(data[s][i] / m);
        } else if (mode == "zscore") {
            double mean = 0.0;
            for (size_t i = 0; i < n; ++i) mean += data[s][i];
            mean /= static_cast<double>(n);
            double var = 0.0;
            for (size_t i = 0; i < n; ++i) var += (data[s][i] - mean) * (data[s][i] - mean);
            double stdv = std::sqrt(var / static_cast<double>(n));
            if (stdv == 0.0) stdv = 1.0;
            for (size_t i = 0; i < n; ++i) o[i] = static_cast<float>((data[s][i] - mean) / stdv);
        } else if (mode == "minmax") {
            double dmin = data[s][0], dmax = data[s][0];
            for (size_t i = 1; i < n; ++i) {
                dmin = std::min(dmin, static_cast<double>(data[s][i]));
                dmax = std::max(dmax, static_cast<double>(data[s][i]));
            }
            double denom = dmax - dmin;
            if (denom == 0.0) denom = 1.0;
            for (size_t i = 0; i < n; ++i) o[i] = static_cast<float>((data[s][i] - dmin) / denom);
        }
    }
    return out;
}

array2d<float> savgol(const array2d<float>& data, size_t window_length,
                      size_t polyorder) {
    validate_rows(data);
    return savgol_filter_impl(data, window_length, polyorder);
}

array2d<float> medfilt(const array2d<float>& data, size_t kernel_size) {
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("kernel_size must be odd");
    }
    validate_rows(data);
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        std::vector<double> row(data.cols());
        for (size_t i = 0; i < data.cols(); ++i) row[i] = data[s][i];
        const auto y = signal::medfilt1d(row, kernel_size);
        for (size_t i = 0; i < data.cols(); ++i) out[s][i] = static_cast<float>(y[i]);
    }
    return out;
}

array2d<float> gate(const array2d<float>& data, size_t start, size_t end) {
    const size_t n = data.cols();
    if (end == 0) end = n;
    if (start > n || end > n || start >= end) {
        throw std::invalid_argument(
            "Invalid gate range [" + std::to_string(start) + ":" +
            std::to_string(end) + "] for signal length " + std::to_string(n) + ".");
    }
    array2d<float> out(data.rows(), end - start);
    for (size_t s = 0; s < data.rows(); ++s) {
        std::copy(data[s].begin() + static_cast<std::ptrdiff_t>(start),
                  data[s].begin() + static_cast<std::ptrdiff_t>(end),
                  out[s].begin());
    }
    return out;
}

array2d<float> detrend(const array2d<float>& data) {
    const size_t n = data.cols();
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<double>(i);
    double xm = 0.0;
    for (double v : x) xm += v;
    xm /= static_cast<double>(n);
    std::vector<double> xx(n);
    double denom = 0.0;
    for (size_t i = 0; i < n; ++i) {
        xx[i] = x[i] - xm;
        denom += xx[i] * xx[i];
    }
    array2d<float> out(data.rows(), n);
    if (denom == 0.0) {
        for (size_t s = 0; s < data.rows(); ++s) {
            for (size_t i = 0; i < n; ++i) out[s][i] = data[s][i];
        }
        return out;
    }
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        double y_mean = 0.0;
        for (size_t i = 0; i < n; ++i) y_mean += data[s][i];
        y_mean /= static_cast<double>(n);
        double slope = 0.0;
        for (size_t i = 0; i < n; ++i) {
            slope += xx[i] * (static_cast<double>(data[s][i]) - y_mean);
        }
        slope /= denom;
        const double intercept = y_mean - slope * xm;
        for (size_t i = 0; i < n; ++i) {
            out[s][i] = static_cast<float>(data[s][i] - (slope * x[i] + intercept));
        }
    }
    return out;
}

array2d<float> envelope(const array2d<float>& data) {
    validate_rows(data);
    array2d<float> out(data.rows(), data.cols());
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        std::vector<double> row(data.cols());
        for (size_t i = 0; i < data.cols(); ++i) row[i] = data[s][i];
        const auto y = signal::hilbert_envelope(row);
        for (size_t i = 0; i < data.cols(); ++i) out[s][i] = static_cast<float>(y[i]);
    }
    return out;
}

array2d<float> moving_average(const array2d<float>& data, size_t window) {
    const size_t n = data.cols();
    array2d<float> out(data.rows(), n);
    if (n < window) {
        for (size_t s = 0; s < data.rows(); ++s) {
            for (size_t i = 0; i < n; ++i) out[s][i] = data[s][i];
        }
        return out;
    }
    std::vector<double> kernel(window, 1.0 / static_cast<double>(window));
    const size_t half = (window - 1) / 2;
#ifdef SAMCORE_HAS_OPENMP
#pragma omp parallel for if (data.rows() > 8)
#endif
    for (size_t s = 0; s < data.rows(); ++s) {
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t k = 0; k < window; ++k) {
                const int64_t idx = static_cast<int64_t>(i) + static_cast<int64_t>(k) -
                                    static_cast<int64_t>(half);
                if (idx >= 0 && static_cast<size_t>(idx) < n) {
                    acc += kernel[k] * data[s][static_cast<size_t>(idx)];
                }
            }
            out[s][i] = static_cast<float>(acc);
        }
    }
    return out;
}

} // namespace samcore::preprocessing

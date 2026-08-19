#include <samcore/signal/kernels.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace samcore::signal {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

// small dense linear algebra (double)

// Gaussian elimination with partial pivoting for A x = b (A is n x n).
std::vector<double> solve_linear(const std::vector<double>& A, size_t n,
                                 const std::vector<double>& b) {
    std::vector<double> M = A;
    std::vector<double> rhs = b;
    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        double best = std::fabs(M[col * n + col]);
        for (size_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(M[r * n + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (pivot != col) {
            for (size_t c = 0; c < n; ++c) {
                std::swap(M[col * n + c], M[pivot * n + c]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        const double piv = M[col * n + col];
        if (piv == 0.0) {
            throw std::runtime_error("solve_linear: singular system");
        }
        for (size_t r = col + 1; r < n; ++r) {
            const double f = M[r * n + col] / piv;
            if (f == 0.0) continue;
            for (size_t c = col; c < n; ++c) {
                M[r * n + c] -= f * M[col * n + c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    std::vector<double> x(n);
    for (size_t r = n; r-- > 0;) {
        double acc = rhs[r];
        for (size_t c = r + 1; c < n; ++c) {
            acc -= M[r * n + c] * x[c];
        }
        x[r] = acc / M[r * n + r];
    }
    return x;
}

// scipy.linalg.companion(a).T with a0 = 1: first column -a[1:], superdiagonal
// ones.  State-space A of the transposed direct form II implementation.
std::vector<double> companion_t(size_t n, const std::vector<double>& a) {
    std::vector<double> A(n * n, 0.0);
    for (size_t i = 0; i + 1 < n; ++i) {
        A[i * n + (i + 1)] = 1.0;
    }
    for (size_t i = 0; i < n; ++i) {
        A[i * n] = -a[i + 1];
    }
    return A;
}

// lfilter_zi(b, a): initial state for which the step response is constant
// (scipy.signal.lfilter_zi parity).
std::vector<double> lfilter_zi(const std::vector<double>& b,
                               const std::vector<double>& a) {
    const size_t n = std::max(b.size(), a.size()) - 1;
    if (n == 0) return {};
    std::vector<double> bb(n + 1, 0.0), aa(n + 1, 0.0);
    std::copy(b.begin(), b.end(), bb.begin());
    std::copy(a.begin(), a.end(), aa.begin());
    const double a0 = aa[0];
    for (auto& v : aa) v /= a0;
    for (auto& v : bb) v /= a0;
    const std::vector<double> A = companion_t(n, aa);
    std::vector<double> IminusA(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) IminusA[i * n + i] = 1.0;
    for (size_t i = 0; i < n * n; ++i) IminusA[i] -= A[i];
    std::vector<double> B(n);
    for (size_t i = 0; i < n; ++i) {
        B[i] = bb[i + 1] - aa[i + 1] * bb[0];
    }
    return solve_linear(IminusA, n, B);
}

// filter design

// Expand monic polynomial from roots: poly(z) = prod(z - r_i), returned in
// descending power order (leading coefficient first).
std::vector<double> poly_from_roots(const std::vector<std::complex<double>>& roots) {
    std::vector<std::complex<double>> poly(1, {1.0, 0.0});
    for (const auto& r : roots) {
        std::vector<std::complex<double>> next(poly.size() + 1, {0.0, 0.0});
        for (size_t i = 0; i < poly.size(); ++i) {
            next[i + 1] += poly[i];
            next[i] -= poly[i] * r;
        }
        poly = std::move(next);
    }
    std::vector<double> out(poly.size());
    for (size_t i = 0; i < poly.size(); ++i) {
        out[i] = poly[poly.size() - 1 - i].real();
    }
    return out;
}

struct design_result {
    std::vector<double> b;
    std::vector<double> a;
};

// Digital filter from analog poles (scaled by w0 in the design caller) and
// zeros in the s-plane (specified by counts of zeros at s=0 and at infinity),
// via the bilinear transform s = c*(z-1)/(z+1), c = 2*fs.  Normalized so
// the magnitude response is 1 at z0 (DC for lowpass, band center for
// bandpass).
design_result bilinear_design(std::vector<std::complex<double>> apoles,
                              size_t zeros_at_zero, size_t zeros_at_inf,
                              double w0, double c, std::complex<double> z0) {
    std::vector<std::complex<double>> poles;
    poles.reserve(apoles.size());
    for (auto& p : apoles) {
        p *= w0;
        poles.push_back((p + c) / (c - p));
    }
    std::vector<std::complex<double>> zroots(zeros_at_zero, {1.0, 0.0});
    for (size_t i = 0; i < zeros_at_inf; ++i) {
        zroots.emplace_back(-1.0, 0.0);
    }

    // Unit-gain response, |H(z0)| = |prod(z0 - z_i)| / |prod(z0 - p_i)|.
    std::complex<double> num = 1.0;
    for (const auto& z : zroots) num *= (z0 - z);
    std::complex<double> den = 1.0;
    for (const auto& p : poles) den *= (z0 - p);
    const double k = std::abs(den / num);

    design_result r;
    r.b = poly_from_roots(zroots);
    r.a = poly_from_roots(poles);
    for (auto& v : r.b) v *= k;
    const double a0 = r.a[0];
    for (auto& v : r.a) v /= a0;
    for (auto& v : r.b) v /= a0;
    return r;
}

// DF2T filter application with optional initial state.
std::vector<double> lfilter_impl(const std::vector<double>& b,
                                 const std::vector<double>& a,
                                 std::span<const double> x,
                                 const std::vector<double>* zi) {
    const size_t m = std::max(b.size(), a.size()) - 1;
    std::vector<double> state(m, 0.0);
    if (zi) {
        if (zi->size() != m) {
            throw std::invalid_argument("lfilter: zi length mismatch");
        }
        state = *zi;
    }
    std::vector<double> y(x.size());
    for (size_t n = 0; n < x.size(); ++n) {
        y[n] = b[0] * x[n] + state[0];
        for (size_t i = 0; i + 1 < m; ++i) {
            state[i] = state[i + 1] + b[i + 1] * x[n] - a[i + 1] * y[n];
        }
        if (m > 0) {
            state[m - 1] = b[m] * x[n] - a[m] * y[n];
        }
    }
    return y;
}

// Odd extension of length `edge` on both sides (scipy odd_ext parity).
std::vector<double> odd_ext(std::span<const double> x, size_t edge) {
    std::vector<double> ext(2 * edge + x.size());
    for (size_t i = 0; i < edge; ++i) {
        ext[i] = 2.0 * x[0] - x[edge - i];
    }
    std::copy(x.begin(), x.end(), ext.begin() + edge);
    for (size_t i = 0; i < edge; ++i) {
        ext[edge + x.size() + i] = 2.0 * x[x.size() - 1] - x[x.size() - 2 - i];
    }
    return ext;
}

} // namespace

std::pair<std::vector<double>, std::vector<double>> butter_lowpass(
    double cutoff, double fs) {
    constexpr size_t order = 3;
    if (cutoff <= 0.0 || cutoff >= fs / 2.0) {
        throw std::invalid_argument("butter_lowpass: cutoff must be in (0, fs/2)");
    }
    const double c = 2.0 * fs;
    const double w0 = c * std::tan(PI * cutoff / fs);
    std::vector<std::complex<double>> poles;
    for (size_t k = 0; k < order; ++k) {
        const double angle = PI / 2.0 + (2.0 * k + 1.0) * PI / (2.0 * order);
        poles.emplace_back(std::cos(angle), std::sin(angle));
    }
    const design_result d = bilinear_design(poles, 0, order, w0, c, {1.0, 0.0});
    return {d.b, d.a};
}

std::pair<std::vector<double>, std::vector<double>> butter_bandpass(
    double cutoff_low, double cutoff_high, double fs) {
    constexpr size_t order = 3;
    if (cutoff_low <= 0.0 || cutoff_high <= cutoff_low ||
        cutoff_high >= fs / 2.0) {
        throw std::invalid_argument(
            "butter_bandpass: cutoffs must satisfy 0 < lo < hi < fs/2");
    }
    const double c = 2.0 * fs;
    const double wl = c * std::tan(PI * cutoff_low / fs);
    const double wh = c * std::tan(PI * cutoff_high / fs);
    const double bw = wh - wl;
    const double w0sq = wl * wh;

    // Analog lowpass prototype poles (order 3).
    std::vector<std::complex<double>> lp_poles;
    for (size_t k = 0; k < order; ++k) {
        const double angle = PI / 2.0 + (2.0 * k + 1.0) * PI / (2.0 * order);
        lp_poles.emplace_back(std::cos(angle), std::sin(angle));
    }
    // lp2bp: s -> (s^2 + w0^2) / (s * bw); each prototype pole p gives two
    // bandpass poles solving s^2 - p*bw*s + w0^2 = 0.
    std::vector<std::complex<double>> bp_poles;
    for (const auto& p : lp_poles) {
        const std::complex<double> disc = std::sqrt(p * p * bw * bw - 4.0 * w0sq);
        bp_poles.push_back((p * bw + disc) / 2.0);
        bp_poles.push_back((p * bw - disc) / 2.0);
    }
    // Bandpass center: digital image of the analog geometric mean w0.
    const double wc = std::sqrt(w0sq);
    const std::complex<double> z0 = std::polar(1.0, 2.0 * std::atan2(wc, c));
    const design_result d = bilinear_design(bp_poles, order, order, 1.0, c, z0);
    return {d.b, d.a};
}

std::vector<sos> cheby1_sos(size_t order, double rp, double wn) {
    if (order == 0 || wn <= 0.0 || wn >= 1.0) {
        throw std::invalid_argument("cheby1_sos: invalid order or wn");
    }
    const double eps = std::sqrt(std::pow(10.0, rp / 10.0) - 1.0);
    const double phi = std::asinh(1.0 / eps) / static_cast<double>(order);
    const double c = 4.0; // normalized fs = 2 (scipy convention)
    const double w0 = c * std::tan(PI * wn / 2.0);

    std::vector<std::complex<double>> apoles;
    for (size_t k = 0; k < order; ++k) {
        const double theta = PI * (2.0 * k + 1.0) / (2.0 * order);
        apoles.emplace_back(-std::sinh(phi) * std::sin(theta),
                            std::cosh(phi) * std::cos(theta));
    }
    // Chebyshev lowpass: no finite zeros, order zeros at z = -1.
    std::vector<std::complex<double>> poles;
    for (auto& p : apoles) {
        p *= w0;
        poles.push_back((p + c) / (c - p));
    }
    const std::vector<std::complex<double>> zroots(order, {-1.0, 0.0});
    // DC gain must equal 10^(-rp/20): the passband ripple trough (scipy
    // cheb1ap normalization), not 1.
    const std::complex<double> z0{1.0, 0.0};
    std::complex<double> num = 1.0;
    for (const auto& z : zroots) num *= (z0 - z);
    std::complex<double> den = 1.0;
    for (const auto& p : poles) den *= (z0 - p);
    const double k = std::abs(den / num) * std::pow(10.0, -rp / 20.0);

    // Order poles by increasing frequency for stable section pairing.
    std::vector<size_t> pole_order(order);
    for (size_t i = 0; i < order; ++i) pole_order[i] = i;
    std::sort(pole_order.begin(), pole_order.end(), [&](size_t i, size_t j) {
        return std::abs(poles[i].imag()) < std::abs(poles[j].imag());
    });

    std::vector<sos> sections;
    const size_t n_sec = (order + 1) / 2;
    sections.reserve(n_sec);
    // scipy's zpk2sos puts the ENTIRE gain in the first section and leaves
    // the rest with unity numerator [1, 2, 1]; replicate that for bit-level
    // parity of the sos array.
    for (size_t s = 0; s < n_sec; ++s) {
        sos sec{};
        std::vector<std::complex<double>> sec_poles;
        sec_poles.push_back(poles[pole_order[2 * s]]);
        if (2 * s + 1 < order) sec_poles.push_back(poles[pole_order[2 * s + 1]]);
        const auto za = poly_from_roots(sec_poles);
        const size_t z_count = sec_poles.size();
        const double g = (s == 0) ? k : 1.0;
        std::vector<std::complex<double>> zroots_s(z_count, {-1.0, 0.0});
        const auto zb = poly_from_roots(zroots_s);
        sec.a0 = za[0];
        sec.a1 = za[1];
        sec.a2 = za.size() > 2 ? za[2] : 0.0;
        sec.b0 = zb[0] * g;
        sec.b1 = zb.size() > 1 ? zb[1] * g : 0.0;
        sec.b2 = zb.size() > 2 ? zb[2] * g : 0.0;
        sections.push_back(sec);
    }
    return sections;
}

std::vector<double> lfilter(const std::vector<double>& b,
                            const std::vector<double>& a,
                            std::span<const double> x) {
    return lfilter_impl(b, a, x, nullptr);
}

std::vector<double> filtfilt(const std::vector<double>& b,
                             const std::vector<double>& a,
                             std::span<const double> x) {
    const size_t ntaps = std::max(b.size(), a.size());
    const size_t edge = 3 * ntaps;
    if (x.size() <= edge) {
        throw std::invalid_argument(
            "filtfilt: input length must be greater than 3*max(len(a), len(b))");
    }
    const std::vector<double> ext = odd_ext(x, edge);
    const std::vector<double> zi = lfilter_zi(b, a);
    std::vector<double> state = zi;
    for (auto& s : state) s *= ext[0];

    std::vector<double> y = lfilter_impl(b, a, ext, &state);
    state = zi;
    for (auto& s : state) s *= y.back();
    std::vector<double> yrev(y.rbegin(), y.rend());
    y = lfilter_impl(b, a, yrev, &state);
    std::reverse(y.begin(), y.end());
    return std::vector<double>(y.begin() + edge, y.end() - edge);
}

std::vector<double> decimate(std::span<const double> x, size_t q) {
    if (q < 2) return std::vector<double>(x.begin(), x.end());
    const auto sections = cheby1_sos(8, 0.05, 0.8 / static_cast<double>(q));

    const size_t n_sections = sections.size();
    const size_t edge = 3 * (2 * n_sections + 1);
    if (x.size() <= edge) {
        throw std::invalid_argument("decimate: input too short");
    }
    const std::vector<double> ext = odd_ext(x, edge);

    auto apply_sections = [&](std::span<const double> in,
                              std::vector<std::array<double, 2>>& states) {
        std::vector<double> cur(in.begin(), in.end());
        std::vector<double> out(in.size());
        for (size_t s = 0; s < n_sections; ++s) {
            const sos& sec = sections[s];
            const double inv_a0 = 1.0 / sec.a0;
            const double a1 = sec.a1 * inv_a0;
            const double a2 = sec.a2 * inv_a0;
            const double b0 = sec.b0 * inv_a0;
            const double b1 = sec.b1 * inv_a0;
            const double b2 = sec.b2 * inv_a0;
            double s0 = states[s][0];
            double s1 = states[s][1];
            for (size_t n = 0; n < in.size(); ++n) {
                out[n] = b0 * cur[n] + s0;
                s0 = b1 * cur[n] - a1 * out[n] + s1;
                s1 = b2 * cur[n] - a2 * out[n];
            }
            states[s][0] = s0;
            states[s][1] = s1;
            cur.swap(out);
        }
        return cur;
    };

    // Per-section steady-state zi (scipy sosfilt_zi parity).  Each section's
    // zi = lfilter_zi(section) scaled by the cumulative DC gain of all
    // PRECEDING sections (the signal levels grow as they pass through the
    // cascade), then the whole thing is scaled by the first input sample.
    std::vector<std::array<double, 2>> states(n_sections, {0.0, 0.0});
    double cum_gain = 1.0;
    for (size_t s = 0; s < n_sections; ++s) {
        const sos& sec = sections[s];
        const double inv_a0 = 1.0 / sec.a0;
        const double b0 = sec.b0 * inv_a0, b1 = sec.b1 * inv_a0;
        const double b2 = sec.b2 * inv_a0;
        const double a1 = sec.a1 * inv_a0, a2 = sec.a2 * inv_a0;
        // lfilter_zi of the section (2x2 solve, scipy companion parity).
        const double B0 = b1 - a1 * b0;
        const double B1 = b2 - a2 * b0;
        const double z0 = (B0 + B1) / (1.0 + a1 + a2);
        const double z1 = B1 - a2 * z0;
        states[s][0] = cum_gain * z0;
        states[s][1] = cum_gain * z1;
        cum_gain *= (b0 + b1 + b2) / (1.0 + a1 + a2);
    }

    // Forward pass with the steady-state initial conditions.
    const std::vector<std::array<double, 2>> zi_states = states;
    for (size_t s = 0; s < n_sections; ++s) {
        states[s][0] *= ext[0];
        states[s][1] *= ext[0];
    }
    std::vector<double> y = apply_sections(ext, states);
    // Backward pass: re-derive the zi from the ORIGINAL steady-state
    // vectors scaled by the last sample of the forward output (scipy
    // sosfiltfilt parity).
    for (size_t s = 0; s < n_sections; ++s) {
        states[s][0] = zi_states[s][0] * y.back();
        states[s][1] = zi_states[s][1] * y.back();
    }
    std::vector<double> yrev(y.rbegin(), y.rend());
    y = apply_sections(yrev, states);
    std::reverse(y.begin(), y.end());

    // Trim the pads, then subsample every q-th sample (scipy decimate
    // parity: y[::q] on the filtfilt output).
    const size_t n = y.size() - 2 * edge;
    const size_t n_out = (n + q - 1) / q;
    std::vector<double> out;
    out.reserve(n_out);
    for (size_t j = 0; j < n_out; ++j) {
        out.push_back(y[edge + j * q]);
    }
    return out;
}

std::vector<double> savgol_coeffs(size_t window_length, size_t polyorder) {
    if (polyorder >= window_length) {
        throw std::invalid_argument("polyorder must be less than window_length.");
    }
    const size_t halflen = window_length / 2;
    // x = arange(-pos, window_length - pos), pos = halflen (odd window).
    const double pos = static_cast<double>(halflen);
    std::vector<double> x(window_length);
    for (size_t i = 0; i < window_length; ++i) {
        x[i] = static_cast<double>(i) - pos;
    }
    // Least squares: min ||A c - y|| with A[i][j] = x[i]^j, y = unit vector
    // at the window center (scipy savgol_coeffs parity).
    const size_t p = polyorder + 1;
    std::vector<double> A(window_length * p);
    for (size_t i = 0; i < window_length; ++i) {
        double v = 1.0;
        for (size_t j = 0; j < p; ++j) {
            A[i * p + j] = v;
            v *= x[i];
        }
    }
    std::vector<double> y(window_length, 0.0);
    y[halflen] = 1.0;
    // Normal equations + solve (well-conditioned for small windows).
    std::vector<double> AtA(p * p, 0.0);
    std::vector<double> Atb(p, 0.0);
    for (size_t i = 0; i < window_length; ++i) {
        for (size_t c = 0; c < p; ++c) {
            Atb[c] += A[i * p + c] * y[i];
            for (size_t r = 0; r <= c; ++r) {
                AtA[c * p + r] += A[i * p + c] * A[i * p + r];
            }
        }
    }
    for (size_t r = 0; r < p; ++r) {
        for (size_t c = r + 1; c < p; ++c) {
            AtA[r * p + c] = AtA[c * p + r];
        }
    }
    const std::vector<double> c = solve_linear(AtA, p, Atb);
    // Coefficients are the least-squares projection of the unit vector
    // onto the column space: coeffs = A * c (scipy savgol_coeffs parity),
    // then reversed for use='conv' (convolution ordering).
    std::vector<double> coeffs(window_length);
    for (size_t i = 0; i < window_length; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < p; ++j) {
            acc += A[i * p + j] * c[j];
        }
        coeffs[i] = acc;
    }
    return std::vector<double>(coeffs.rbegin(), coeffs.rend());
}

std::vector<double> medfilt1d(std::span<const double> x, size_t kernel_size) {
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("kernel_size must be odd");
    }
    const size_t n = x.size();
    const size_t half = kernel_size / 2;
    std::vector<double> out(n);
    std::vector<double> window(kernel_size);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < kernel_size; ++j) {
            // numpy 'reflect' padding (scipy.ndimage median_filter default).
            int64_t idx = static_cast<int64_t>(i) + static_cast<int64_t>(j) -
                          static_cast<int64_t>(half);
            if (idx < 0) idx = -idx;
            if (idx >= static_cast<int64_t>(n)) {
                idx = 2 * static_cast<int64_t>(n) - 2 - idx;
            }
            window[j] = x[static_cast<size_t>(idx)];
        }
        const size_t mid = half;
        std::nth_element(window.begin(), window.begin() + mid, window.end());
        out[i] = window[mid];
    }
    return out;
}

} // namespace samcore::signal

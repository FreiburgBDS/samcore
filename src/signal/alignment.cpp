// Cross-correlation alignment kernels.

#include <samcore/signal/kernels.hpp>

#include <complex>
#include <stdexcept>
#include <vector>

namespace samcore::signal {

std::int64_t xcorr_lag(std::span<const std::int8_t> x,
                       std::span<const std::int8_t> ref,
                       std::int64_t max_shift) {
    const size_t n = x.size();
    if (ref.size() != n) {
        throw std::invalid_argument("xcorr_lag: signal lengths differ");
    }
    if (n == 0) {
        throw std::invalid_argument("xcorr_lag: empty signal");
    }
    if (max_shift < 0) max_shift = 0;
    if (static_cast<size_t>(max_shift) >= n) {
        max_shift = static_cast<std::int64_t>(n) - 1;
    }

    // Mean-subtract to remove any DC baseline, then zero-pad to 2n so the
    // linear cross-correlation has no circular wraparound within
    // +-max_shift.  c[lag] = sum_j x[j + lag] * ref[j].
    const size_t nn = 2 * n;
    std::vector<double> a(nn, 0.0), b(nn, 0.0);
    double mean_x = 0.0, mean_ref = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_ref += ref[i];
    }
    mean_x /= static_cast<double>(n);
    mean_ref /= static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<double>(x[i]) - mean_x;
        b[i] = static_cast<double>(ref[i]) - mean_ref;
    }

    auto A = rfft(a);
    const auto B = rfft(b);
    for (size_t k = 0; k < A.size(); ++k) {
        A[k] *= std::conj(B[k]);
    }
    const std::vector<double> c = irfft(A, nn);
    auto at = [&](std::int64_t lag) -> double {
        return lag >= 0 ? c[static_cast<size_t>(lag)]
                        : c[nn + static_cast<size_t>(lag)];
    };

    std::int64_t best = 0;
    double best_val = at(0);
    for (std::int64_t lag = 1; lag <= max_shift; ++lag) {
        const double positive = at(lag);
        if (positive > best_val) {
            best_val = positive;
            best = lag;
        }
        const double negative = at(-lag);
        if (negative > best_val) {
            best_val = negative;
            best = -lag;
        }
    }
    return best;
}

} // namespace samcore::signal

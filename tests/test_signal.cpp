#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include <samcore/signal/kernels.hpp>


using namespace samcore;
using namespace samcore::signal;

TEST(signal, RfftOfImpulse) {
    std::vector<double> x(8, 0.0);
    x[0] = 1.0;
    auto X = rfft(x);
    ASSERT_EQ(X.size(), 5);
    for (const auto& v : X) {
        EXPECT_NEAR(v.real(), 1.0, 1e-12);
        EXPECT_NEAR(v.imag(), 0.0, 1e-12);
    }
}

TEST(signal, RfftIrfftRoundTrip) {
    std::vector<double> x;
    for (int i = 0; i < 16; ++i) {
        x.push_back(std::sin(0.3 * i) + 0.5 * std::cos(1.1 * i));
    }
    auto X = rfft(x);
    auto y = irfft(X, x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y[i], x[i], 1e-9);
    }
}

TEST(signal, RfftSymmetricReal) {
    std::vector<double> x{1, 2, 3, 4};
    auto X = rfft(x);
    // known rfft of [1,2,3,4]
    EXPECT_NEAR(X[0].real(), 10.0, 1e-12);
    EXPECT_NEAR(X[1].real(), -2.0, 1e-12);
    EXPECT_NEAR(X[1].imag(), 2.0, 1e-12);
    EXPECT_NEAR(X[2].real(), -2.0, 1e-12);
    EXPECT_NEAR(X[2].imag(), 0.0, 1e-12);
}

TEST(signal, HannWindow) {
    auto w = hann_window(8);
    EXPECT_NEAR(w[0], 0.0, 1e-12);
    EXPECT_NEAR(w[4], 1.0, 1e-12);
    EXPECT_NEAR(w[7], 0.5 - 0.5 * std::cos(2.0 * 3.141592653589793 * 7.0 / 8.0),
                1e-12);
}

TEST(signal, ButterLowpassDcGain) {
    auto [b, a] = butter_lowpass(10.0, 100.0);
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    // DC gain must be 1.
    double num = 0.0, den = 0.0;
    for (double v : b) num += v;
    for (double v : a) den += v;
    EXPECT_NEAR(num / den, 1.0, 1e-9);
}

TEST(signal, ButterLowpassConstantSignal) {
    auto [b, a] = butter_lowpass(10.0, 100.0);
    std::vector<double> x(200, 42.0);
    auto y = lfilter(b, a, x);
    for (size_t i = 50; i < y.size(); ++i) {
        // asymptotic convergence of the transient
        EXPECT_NEAR(y[i], 42.0, 1e-4);
    }
}

TEST(signal, ButterBandpassCenterGain) {
    auto [b, a] = butter_bandpass(5.0, 20.0, 100.0);
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    // Response magnitude at the digital center frequency should be 1.
    const double w0 = 2.0 * 3.141592653589793 * std::sqrt(5.0 * 20.0) / 100.0;
    std::complex<double> z = std::polar(1.0, w0);
    std::complex<double> num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        num += b[i] * std::pow(z, static_cast<double>(b.size() - 1 - i));
        den += a[i] * std::pow(z, static_cast<double>(a.size() - 1 - i));
    }
    EXPECT_NEAR(std::abs(num / den), 1.0, 1e-6);
}

TEST(signal, SavgolCoeffsKnown) {
    auto c = savgol_coeffs(5, 2);
    // scipy.savgol_coeffs(5, 2) = [-0.0857, 0.3429, 0.4857, 0.3429, -0.0857]
    EXPECT_NEAR(c[0], -0.08571428571428572, 1e-12);
    EXPECT_NEAR(c[1], 0.3428571428571429, 1e-12);
    EXPECT_NEAR(c[2], 0.4857142857142857, 1e-12);
    EXPECT_NEAR(c[3], 0.3428571428571429, 1e-12);
    EXPECT_NEAR(c[4], -0.08571428571428572, 1e-12);
}

TEST(signal, Medfilt1dKnown) {
    std::vector<double> x{0, 3, 1, 2, 9, 1, 4, 5, 7};
    auto y = medfilt1d(x, 3);
    // scipy.signal.medfilt(x, 3): reflect-padded edges
    EXPECT_NEAR(y[0], 3.0, 1e-12);
    EXPECT_NEAR(y[1], 1.0, 1e-12);
    EXPECT_NEAR(y[2], 2.0, 1e-12);
    EXPECT_NEAR(y[3], 2.0, 1e-12);
    EXPECT_NEAR(y[4], 2.0, 1e-12);
    EXPECT_NEAR(y[5], 4.0, 1e-12);
    EXPECT_NEAR(y[6], 4.0, 1e-12);
    EXPECT_NEAR(y[7], 5.0, 1e-12);
    EXPECT_NEAR(y[8], 5.0, 1e-12);
}

TEST(signal, StftShape) {
    array2d<float> data(3, 2000, 0.0f);
    auto res = stft(data, 100.0e6, 256, 128);
    EXPECT_EQ(res.f.size(), 129);
    EXPECT_EQ(res.t.size(), 14);
    EXPECT_EQ(res.zxx.size0(), 3);
    EXPECT_EQ(res.zxx.size1(), 129);
    EXPECT_EQ(res.zxx.size2(), 14);
    EXPECT_NEAR(res.f.back(), 50.0e6, 1e-3);
}

TEST(signal, StftWindowSumNormalization) {
    // A constant signal produces a spectrum concentrated at DC with
    // amplitude equal to the signal level (scaling='spectrum').
    array2d<float> data(1, 256, 42.0f);
    auto res = stft(data, 100.0, 256, 128);
    EXPECT_NEAR(res.zxx.flat()[0].real(), 42.0f, 1e-3);
}


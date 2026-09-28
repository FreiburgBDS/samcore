#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include <samcore/signal/kernels.hpp>


using namespace samcore;
using namespace samcore::signal;

TEST(signal, DecimateMatchesScipy) {
    // x[i] = (i % 7 - 3) + 0.25 * (i % 5); references generated with
    // scipy.signal.decimate(x, q) (cheby1(8, 0.05, 0.8/q), zero phase).
    std::vector<double> x(64);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(static_cast<int>(i % 7) - 3) +
               0.25 * static_cast<double>(i % 5);
    }
    const std::vector<double> expect_q2 = {
        -2.9699878030795648, -0.62247609089086764, 2.2456443225503264,
        1.2301675666009513, -1.3352379766343991, 0.22702727534005168,
        2.8024114706684, -0.33810202401937933, -1.6252884379870944,
        2.6431145785569852, 1.2440057390472243, -1.7101556631615309,
        0.70004708517900394, 2.3658203020176147, -0.12727940280753591,
        -1.535432681861626, 2.3079189865552889, 1.6807073716192471,
        -2.0736268888089984, 0.85012700507989214, 2.4828958201791003,
        -0.45817283487197319, -1.1306981003235979, 1.9990870738318836,
        1.7658585516611307, -1.9090768241655294, 0.51598091783322797,
        2.8631247880142459, -0.63112841893137628, -1.2454351155855574,
        2.4757195393747282, 1.1701950723716872};
    const std::vector<double> expect_q5 = {
        -2.9614181885079924, 1.236551675232203, 0.252413704454232,
        0.55443073093767536, 0.59543989687743459, 0.37464500888444408,
        0.64629847941177376, 0.44137808132587392, 0.48148575877977406,
        0.64582685167757981, 0.2621387976145077, 0.83316599604173747,
        0.16584556711563594};
    for (auto [q, expect] :
         std::initializer_list<std::pair<size_t, const std::vector<double>*>>{
             {2, &expect_q2}, {5, &expect_q5}}) {
        const auto y = decimate(std::span<const double>(x), q);
        ASSERT_EQ(y.size(), expect->size());
        for (size_t i = 0; i < y.size(); ++i) {
            EXPECT_NEAR(y[i], (*expect)[i], 1e-11);
        }
    }
}

TEST(signal, DecimateInt8MatchesDouble) {
    std::vector<std::int8_t> xi(200);
    for (size_t i = 0; i < xi.size(); ++i) {
        xi[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 7 + 13) % 255) - 127);
    }
    std::vector<double> xd(xi.begin(), xi.end());
    const auto sections = decimate_sos(5);
    const auto yd = decimate(std::span<const double>(xd), 5);
    const auto yi = decimate(std::span<const std::int8_t>(xi), 5, sections);
    ASSERT_EQ(yi.size(), yd.size());
    for (size_t i = 0; i < yi.size(); ++i) {
        EXPECT_NEAR(yi[i], yd[i], 1e-12);
    }
    // q < 2 is an identity copy; decimate_sos rejects it.
    EXPECT_THROW((void)decimate_sos(1), std::invalid_argument);
    const auto y1 = decimate(std::span<const double>(xd), 1);
    EXPECT_EQ(y1, xd);
}

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


#include <gtest/gtest.h>

#include <cstring>

#include <samcore/signal/kernels.hpp>
#include <samcore/utils.hpp>


using namespace samcore;
using namespace samcore::utils;

namespace {

array2d<float> make_signal(size_t rows, size_t cols) {
    array2d<float> d(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            d[i][j] = static_cast<float>(std::sin(0.1 * static_cast<double>(j)) *
                                         (1.0 + 0.1 * static_cast<double>(i)));
        }
    }
    return d;
}

} // namespace

TEST(utils, KurtOfGaussianApprox) {
    // uniform-ish data has kurtosis below 3, pure tone ~ 1.5
    array2d<float> d(2, 2000);
    for (size_t j = 0; j < 2000; ++j) {
        d[0][j] = static_cast<float>(std::sin(0.3 * static_cast<double>(j)));
        d[1][j] = static_cast<float>(static_cast<int>(j % 100) - 50);
    }
    auto k = kurt(d);
    // scipy.stats.kurtosis(sin, fisher=False) ≈ 1.5
    EXPECT_NEAR(k[0], 1.5, 0.05);
    EXPECT_NEAR(k[1], 1.8, 0.1); // discrete uniform -> 1.8
}

TEST(utils, TimeIndex) {
    auto t = time_index(100.0, 10.0, 5);
    ASSERT_EQ(t.size(), 5);
    EXPECT_DOUBLE_EQ(t[0], 100.0);
    EXPECT_DOUBLE_EQ(t[4], 110.0);
    EXPECT_DOUBLE_EQ(t[2], 105.0);
}

TEST(utils, FftSpec) {
    std::vector<float> scan(128, 0.0f);
    scan[0] = 1.0f;
    auto [mag, freqs] = fft_spec(scan, 1.0);
    ASSERT_EQ(mag.size(), 65);
    ASSERT_EQ(freqs.size(), 65);
    EXPECT_NEAR(mag[0], 1.0, 1e-12);
    EXPECT_NEAR(freqs[1], 1.0 / 128.0, 1e-15);
}

TEST(utils, SpectralEntropyBasic) {
    array2d<float> psd(3, 8);
    // row 0: all energy in one bin -> entropy 0
    for (size_t j = 0; j < 8; ++j) psd[0][j] = j == 2 ? 1.0f : 0.0f;
    // row 1: uniform -> entropy 1 (base 2)
    for (size_t j = 0; j < 8; ++j) psd[1][j] = 1.0f;
    // row 2: silent -> 0
    for (size_t j = 0; j < 8; ++j) psd[2][j] = 0.0f;
    auto e = spectral_entropy(psd, 2.0);
    EXPECT_NEAR(e[0], 0.0, 1e-12);
    EXPECT_NEAR(e[1], 3.0, 1e-12); // log2(8)
    EXPECT_NEAR(e[2], 0.0, 1e-12);
}

TEST(utils, SpectralFlatnessBasic) {
    array2d<float> psd(2, 4);
    for (size_t j = 0; j < 4; ++j) psd[0][j] = 1.0f; // flat -> 1
    psd[1][0] = 1.0f;                                // peaked -> ~0
    auto f = spectral_flatness(psd);
    EXPECT_NEAR(f[0], 1.0, 1e-6); // +1e-10 log floor perturbs slightly
    EXPECT_NEAR(f[1], 0.0, 1e-6);
}

TEST(utils, SpectralCentroid) {
    array2d<float> psd(1, 3);
    std::vector<float> freqs{1.0f, 2.0f, 3.0f};
    psd[0][0] = 0.0f;
    psd[0][1] = 1.0f;
    psd[0][2] = 1.0f;
    auto c = spectral_centroid(freqs, psd);
    EXPECT_NEAR(c[0], 2.5, 1e-12);
}

TEST(utils, SpectralEnergyRatio) {
    array2d<float> psd(1, 4);
    std::vector<float> freqs{1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t j = 0; j < 4; ++j) psd[0][j] = 1.0f;
    auto r = spectral_energy_ratio(freqs, psd, 3.0f);
    EXPECT_NEAR(r[0], 2.0 / 2.0, 1e-12); // >= 3.0 -> 2 bins, below -> 2
    auto r2 = spectral_energy_ratio(freqs, psd, 5.0f);
    EXPECT_NEAR(r2[0], 0.0, 1e-12); // nothing above -> 0
}


TEST(utils, KurtConstantRowIsNan) {
    // scipy parity: kurtosis of a zero-variance row is NaN.  The check
    // compares the bit pattern because -ffast-math makes std::isnan
    // always return false.
    array2d<float> d(2, 16);
    for (size_t j = 0; j < 16; ++j) {
        d[0][j] = 3.0f;
        d[1][j] = static_cast<float>(j);
    }
    auto k = kurt(d);
    std::uint64_t bits;
    std::memcpy(&bits, &k[0], sizeof(bits));
    // Quiet NaN with the sign bit masked out: -ffast-math makes std::isnan
    // always return false, and the sign of the NaN varies by compiler.
    EXPECT_EQ(bits & 0x7fffffffffffffffull, 0x7ff8000000000000ull);
    EXPECT_NEAR(k[1], 1.7905882352941176, 1e-12);
}

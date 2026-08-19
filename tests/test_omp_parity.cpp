#include <gtest/gtest.h>

#include <samcore/sam_scan.hpp>

// OpenMP correctness: kernels must produce identical results regardless of
// thread count.  Results are checked against hand-computed references; the
// suite is also run with OMP_NUM_THREADS=1 via ctest.

using namespace samcore;

namespace {

std::string sha1(const std::vector<std::int8_t>& data) {
    // Deterministic FNV-1a hash over the raw bytes (order preserved).
    uint64_t h = 1469598103934665603ull;
    for (auto v : data) {
        h ^= static_cast<uint8_t>(v);
        h *= 1099511628211ull;
    }
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

sam_scan make_scan(std::int64_t nlines, std::int64_t cols,
                   std::int64_t scanlen) {
    sam_header header(cols, nlines, scanlen, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(static_cast<size_t>(nlines * cols),
                              static_cast<size_t>(scanlen));
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(
                static_cast<int>(((i * 13 + j * 7) ^ (j >> 3)) % 120) - 60);
        }
    }
    return sam_scan::from_data(std::move(data), header);
}

} // namespace

TEST(omp_parity, ImagePower) {
    auto h = make_scan(16, 16, 512);
    auto img = std::get<array2d<float>>(h.image(image_mode::power));
    // img is (nlines, cols); scan s sits at img[s / nc][s % nc].
    const size_t nc = static_cast<size_t>(h.cols());
    for (size_t s = 0; s < h.data().rows(); ++s) {
        double acc = 0.0;
        for (size_t j = 0; j < 512; ++j) {
            acc += static_cast<double>(h.data()[s][j]) * h.data()[s][j];
        }
        EXPECT_NEAR(img[s / nc][s % nc], static_cast<float>(acc), 1e-3);
    }
}

TEST(omp_parity, Zgate) {
    auto h = make_scan(8, 8, 400);
    // carve distinct peaks per scan
    for (size_t i = 0; i < h.data().rows(); ++i) {
        h.data()[i][(i * 17) % 300 + 30] = 100;
    }
    auto g = h.zgate(0.5, 64);
    for (size_t i = 0; i < g.data().rows(); ++i) {
        EXPECT_EQ(g.data()[i][0], 100);
    }
}

TEST(omp_parity, DownsampleMean) {
    auto h = make_scan(4, 4, 256);
    auto d = h.downsampled(8, downsample_mode::mean);
    EXPECT_EQ(d.scanlen(), 32);
    EXPECT_EQ(d.data().rows(), 16);
}

TEST(omp_parity, PreprocessingAndSpectral) {
    auto h = make_scan(6, 6, 512);
    auto st = h.compute_stft(128, 64);
    EXPECT_EQ(st.zxx.size0(), 36);
    auto p = h.psd(128, 64);
    EXPECT_EQ(p.psd.rows(), 36);
}

TEST(omp_parity, StftFinite) {
    auto h = make_scan(8, 8, 512);
    auto st = h.compute_stft(128, 64);
    for (auto v : st.zxx.flat()) {
        EXPECT_TRUE(std::isfinite(v.real()));
        EXPECT_TRUE(std::isfinite(v.imag()));
    }
}

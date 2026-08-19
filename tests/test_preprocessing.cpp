#include <gtest/gtest.h>

#include <cmath>

#include <samcore/preprocessing.hpp>


using namespace samcore;

namespace {

array2d<float> make_data(size_t rows, size_t cols) {
    array2d<float> d(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            d[i][j] = static_cast<float>(
                std::sin(0.05 * static_cast<double>(j)) +
                0.2 * std::cos(0.4 * static_cast<double>(j)) +
                0.1 * static_cast<double>(static_cast<int>(i % 3)));
        }
    }
    return d;
}

} // namespace

TEST(preprocessing, LpReducesHighFreqs) {
    array2d<float> d(2, 1000);
    for (size_t j = 0; j < 1000; ++j) {
        d[0][j] = static_cast<float>(std::sin(2.0 * j) + std::sin(0.05 * static_cast<double>(j)));
        d[1][j] = d[0][j];
    }
    auto out = preprocessing::lp(d, 5.0, 100.0);
    // filtered must be smooth: total variation shrinks
    double var_in = 0.0, var_out = 0.0;
    for (size_t j = 1; j < 1000; ++j) {
        var_in += std::fabs(d[0][j] - d[0][j - 1]);
        var_out += std::fabs(out[0][j] - out[0][j - 1]);
    }
    EXPECT_LT(var_out, var_in * 0.5);
}

TEST(preprocessing, NormalizeModes) {
    auto d = make_data(4, 100);
    auto mx = preprocessing::normalize(d, "max");
    for (size_t i = 0; i < d.rows(); ++i) {
        double m = 0.0;
        for (size_t j = 0; j < d.cols(); ++j) {
            m = std::max(m, static_cast<double>(std::fabs(mx[i][j])));
        }
        EXPECT_NEAR(m, 1.0, 1e-6);
    }
    auto mm = preprocessing::normalize(d, "minmax");
    for (size_t i = 0; i < d.rows(); ++i) {
        double mn = mm[i][0], mx2 = mm[i][0];
        for (size_t j = 1; j < d.cols(); ++j) {
            mn = std::min(mn, static_cast<double>(mm[i][j]));
            mx2 = std::max(mx2, static_cast<double>(mm[i][j]));
        }
        EXPECT_NEAR(mn, 0.0, 1e-6);
        EXPECT_NEAR(mx2, 1.0, 1e-6);
    }
    auto zs = preprocessing::normalize(d, "zscore");
    for (size_t i = 0; i < d.rows(); ++i) {
        double mean = 0.0;
        for (size_t j = 0; j < d.cols(); ++j) mean += zs[i][j];
        mean /= static_cast<double>(d.cols());
        EXPECT_NEAR(mean, 0.0, 1e-5);
    }
    EXPECT_THROW((void)preprocessing::normalize(d, "nope"), std::invalid_argument);
}

TEST(preprocessing, SavgolSmooths) {
    auto d = make_data(2, 200);
    auto out = preprocessing::savgol(d, 9, 3);
    EXPECT_EQ(out.rows(), d.rows());
    EXPECT_EQ(out.cols(), d.cols());
    EXPECT_THROW((void)preprocessing::savgol(d, 3, 5), std::invalid_argument);
}

TEST(preprocessing, MedfiltSpikeRemoval) {
    array2d<float> d(1, 10, 0.0f);
    d[0][5] = 100.0f;
    auto out = preprocessing::medfilt(d, 3);
    EXPECT_NEAR(out[0][5], 0.0f, 1e-6);
    EXPECT_NEAR(out[0][4], 0.0f, 1e-6);
    EXPECT_THROW((void)preprocessing::medfilt(d, 4), std::invalid_argument);
}

TEST(preprocessing, GateSlices) {
    auto d = make_data(3, 50);
    auto out = preprocessing::gate(d, 10, 20);
    EXPECT_EQ(out.cols(), 10);
    EXPECT_EQ(out[0][0], d[0][10]);
    auto full = preprocessing::gate(d, 0, 0);
    EXPECT_EQ(full.cols(), 50);
    EXPECT_THROW((void)preprocessing::gate(d, 20, 10), std::invalid_argument);
    EXPECT_THROW((void)preprocessing::gate(d, 45, 55), std::invalid_argument);
}

TEST(preprocessing, DetrendRemovesLine) {
    array2d<float> d(1, 100);
    for (size_t j = 0; j < 100; ++j) d[0][j] = static_cast<float>(3.0 * static_cast<double>(j) + 5.0);
    auto out = preprocessing::detrend(d);
    for (size_t j = 0; j < 100; ++j) {
        EXPECT_NEAR(out[0][j], 0.0f, 1e-4);
    }
}

TEST(preprocessing, EnvelopeNonnegative) {
    auto d = make_data(2, 128);
    auto out = preprocessing::envelope(d);
    for (size_t j = 0; j < 128; ++j) {
        EXPECT_GE(out[0][j], 0.0f);
    }
}

TEST(preprocessing, MovingAverage) {
    auto d = make_data(1, 100);
    auto out = preprocessing::moving_average(d, 5);
    EXPECT_EQ(out.cols(), 100);
    // constant signal: interior preserved; edges dip (zero padding like
    // np.convolve mode='same')
    array2d<float> c(1, 100, 7.0f);
    auto oc = preprocessing::moving_average(c, 5);
    for (size_t j = 2; j < 98; ++j) EXPECT_NEAR(oc[0][j], 7.0f, 1e-5);
    EXPECT_NEAR(oc[0][0], 7.0f * 3.0f / 5.0f, 1e-5);
}

TEST(preprocessing, RowIndependent) {
    auto d = make_data(4, 64);
    auto out = preprocessing::lp(d, 10.0, 100.0);
    auto single = preprocessing::lp(preprocessing::gate(d, 0, 64), 10.0, 100.0);
    // row 2 of the batched result equals row 2 of the single-call result
    for (size_t j = 0; j < 64; ++j) {
        EXPECT_EQ(out[2][j], single[2][j]);
    }
}

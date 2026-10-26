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

TEST(preprocessing, AnalyticMatchesEnvelope) {
    auto d = make_data(3, 97);
    auto z = preprocessing::analytic(d);
    auto env = preprocessing::envelope(d);
    ASSERT_EQ(z.rows(), d.rows());
    ASSERT_EQ(z.cols(), d.cols());
    for (size_t s = 0; s < d.rows(); ++s) {
        for (size_t i = 0; i < d.cols(); ++i) {
            EXPECT_NEAR(std::abs(z[s][i]), env[s][i], 1e-6);
        }
    }
}

TEST(preprocessing, AnalyticNegativeConstant) {
    array2d<float> d(1, 32, -2.0f);
    auto z = preprocessing::analytic(d);
    for (size_t i = 0; i < d.cols(); ++i) {
        EXPECT_NEAR(z[0][i].real(), -2.0f, 1e-5);
        EXPECT_NEAR(z[0][i].imag(), 0.0f, 1e-5);
        EXPECT_NEAR(std::abs(z[0][i]), 2.0f, 1e-5);
    }
}

TEST(preprocessing, AnalyticSingleSample) {
    array2d<float> d(1, 1, 5.0f);
    auto z = preprocessing::analytic(d);
    EXPECT_NEAR(z[0][0].real(), 5.0f, 1e-6);
    EXPECT_NEAR(z[0][0].imag(), 0.0f, 1e-6);
    auto [mean, var] = preprocessing::phase_stats(d);
    EXPECT_NEAR(mean[0], 0.0, 1e-9);
    EXPECT_NEAR(var[0], 0.0, 1e-9);
}

TEST(preprocessing, PhaseStatsConstantSignals) {
    array2d<float> d(2, 64, 3.0f);
    for (size_t i = 0; i < d.cols(); ++i) d[1][i] = -3.0f;
    auto [mean, var] = preprocessing::phase_stats(d);
    ASSERT_EQ(mean.size(), 2u);
    ASSERT_EQ(var.size(), 2u);
    EXPECT_NEAR(mean[0], 0.0, 1e-9);
    EXPECT_NEAR(var[0], 0.0, 1e-9);
    EXPECT_NEAR(std::abs(mean[1]), 3.14159265358979323846, 1e-6);
    EXPECT_NEAR(var[1], 0.0, 1e-9);
}

TEST(preprocessing, PhaseStatsSilent) {
    array2d<float> d(1, 32, 0.0f);
    auto [mean, var] = preprocessing::phase_stats(d);
    ASSERT_EQ(mean.size(), 1u);
    EXPECT_DOUBLE_EQ(mean[0], 0.0);
    EXPECT_DOUBLE_EQ(var[0], 0.0);
}

TEST(preprocessing, PhaseStatsUndefinedMean) {
    // Two samples with analytic phases 0 and pi: the mean components
    // cancel (up to rounding), so the mean direction is undefined
    // (reported as 0) and the phase is maximally spread (1).
    array2d<float> d(1, 2);
    d[0][0] = 1.0f;
    d[0][1] = -1.0f;
    auto [mean, var] = preprocessing::phase_stats(d);
    EXPECT_DOUBLE_EQ(mean[0], 0.0);
    EXPECT_DOUBLE_EQ(var[0], 1.0);
}

TEST(preprocessing, PhaseStatsMatchesManualReduction) {
    auto d = make_data(4, 128);
    auto [mean, var] = preprocessing::phase_stats(d);
    auto z = preprocessing::analytic(d);
    for (size_t s = 0; s < d.rows(); ++s) {
        double sum_sin = 0.0;
        double sum_cos = 0.0;
        for (size_t i = 0; i < d.cols(); ++i) {
            const double phase =
                std::arg(std::complex<double>(z[s][i].real(), z[s][i].imag()));
            sum_sin += std::sin(phase);
            sum_cos += std::cos(phase);
        }
        sum_sin /= static_cast<double>(d.cols());
        sum_cos /= static_cast<double>(d.cols());
        // analytic() is complex64, so the reference is limited by float
        // precision even though phase_stats() accumulates in double.
        EXPECT_NEAR(mean[s], std::atan2(sum_sin, sum_cos), 1e-6);
        EXPECT_NEAR(var[s], 1.0 - std::hypot(sum_sin, sum_cos), 1e-6);
        EXPECT_GE(var[s], 0.0);
        EXPECT_LE(var[s], 1.0);
    }
}

TEST(preprocessing, PhaseStatsFullCycleIsSpread) {
    // A pure tone spanning an integer number of cycles visits every phase
    // direction equally, so the circular variance approaches 1.
    array2d<float> d(1, 128);
    for (size_t i = 0; i < d.cols(); ++i) {
        d[0][i] = static_cast<float>(
            std::cos(2.0 * 3.14159265358979323846 * 8.0 *
                     static_cast<double>(i) / 128.0));
    }
    auto [mean, var] = preprocessing::phase_stats(d);
    EXPECT_NEAR(var[0], 1.0, 1e-6);
    EXPECT_TRUE(std::isfinite(mean[0]));
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

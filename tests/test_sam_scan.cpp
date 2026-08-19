#include <gtest/gtest.h>

#include <samcore/sam_scan.hpp>

using namespace samcore;

namespace {

sam_scan make_scan(std::int64_t nlines, std::int64_t cols, std::int64_t scanlen,
                   bool with_starts = false) {
    sam_header header(cols, nlines, scanlen, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(static_cast<size_t>(nlines * cols),
                              static_cast<size_t>(scanlen));
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(
                static_cast<int>((i * 7 + j * 3) % 128) - 64);
        }
    }
    std::optional<std::vector<std::int32_t>> starts;
    if (with_starts) {
        std::vector<std::int32_t> s(data.rows());
        for (size_t i = 0; i < s.size(); ++i) {
            s[i] = static_cast<std::int32_t>((i % 5 == 0) ? -1 : (i % 10));
        }
        starts = std::move(s);
    }
    sam_labels labels(std::vector<std::int8_t>(data.rows()), {"healthy"});
    return sam_scan::from_data(std::move(data), header, std::move(starts),
                               labels);
}

} // namespace

TEST(sam_scan, FromDataValidation) {
    sam_header header(4, 3, 100, 100.0, 0, 1.0);
    array2d<std::int8_t> good(12, 100);
    EXPECT_NO_THROW((void)sam_scan::from_data(good, header));
    array2d<std::int8_t> bad_rows(11, 100);
    EXPECT_THROW((void)sam_scan::from_data(bad_rows, header), std::invalid_argument);
    array2d<std::int8_t> bad_cols(12, 99);
    EXPECT_THROW((void)sam_scan::from_data(bad_cols, header), std::invalid_argument);
}

TEST(sam_scan, AccessorsAndShape) {
    auto h = make_scan(3, 4, 200);
    EXPECT_EQ(h.nlines(), 3);
    EXPECT_EQ(h.cols(), 4);
    EXPECT_EQ(h.scanlen(), 200);
    EXPECT_EQ(h.shape(), (std::pair<std::int64_t, std::int64_t>{3, 4}));
    EXPECT_EQ(h.num_scans(), 12);
    EXPECT_DOUBLE_EQ(h.samplespacing(), 10.0); // 1/100 MHz * 1e3 = 10 ns
    EXPECT_EQ(h.scan(1, 2).size(), 200);
    EXPECT_EQ(h.scan(6)[0], h.data()[6][0]);
}

TEST(sam_scan, TimeAxisDefault) {
    auto h = make_scan(2, 2, 100);
    auto t = h.time();
    ASSERT_EQ(t.size(), 100);
    EXPECT_DOUBLE_EQ(t[0], 1000.0); // tzero
    EXPECT_DOUBLE_EQ(t[99], 2000.0); // tzero + scanlen/samplerate*1e3
}

TEST(sam_scan, TimeAxisPerScanStarts) {
    auto h = make_scan(2, 2, 100, true);
    auto t = h.time(3); // starts[3] = 3 > 0
    ASSERT_EQ(t.size(), 100);
    EXPECT_DOUBLE_EQ(t[0], 1000.0 + 3 * 10.0);
    auto t0 = h.time(0); // starts[0] == -1 -> default
    EXPECT_DOUBLE_EQ(t0[0], 1000.0);
}

TEST(sam_scan, ImageModes) {
    sam_header header(2, 2, 4, 100.0, 0, 1.0);
    array2d<std::int8_t> data(4, 4);
    std::int8_t rows[4][4] = {{1, 2, 3, 4},
                              {-1, -2, -3, -4},
                              {5, 5, 5, 5},
                              {0, 0, 0, 0}};
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) data[i][j] = rows[i][j];
    }
    auto h = sam_scan::from_data(data, header);

    auto mx = std::get<array2d<std::int8_t>>(h.image(image_mode::max));
    EXPECT_EQ(mx.rows(), 2);
    EXPECT_EQ(mx.cols(), 2);
    EXPECT_EQ(mx[0][0], 4);
    EXPECT_EQ(mx[0][1], -1);
    EXPECT_EQ(mx[1][0], 5);
    EXPECT_EQ(mx[1][1], 0);

    auto am = std::get<array2d<std::int16_t>>(h.image(image_mode::absmax));
    EXPECT_EQ(am[0][0], 4);
    EXPECT_EQ(am[0][1], 4);
    EXPECT_EQ(am[1][0], 5);
    EXPECT_EQ(am[1][1], 0);

    auto pw = std::get<array2d<float>>(h.image(image_mode::power));
    EXPECT_FLOAT_EQ(pw[0][0], 30.0f);
    EXPECT_FLOAT_EQ(pw[0][1], 30.0f);
    EXPECT_FLOAT_EQ(pw[1][0], 100.0f);
    EXPECT_FLOAT_EQ(pw[1][1], 0.0f);
}

TEST(sam_scan, NormalizedData) {
    auto h = make_scan(1, 1, 4);
    h.data()[0][0] = 64;
    h.data()[0][1] = -64;
    h.data()[0][2] = 0;
    h.data()[0][3] = 127;
    auto nd = h.normalized_data();
    EXPECT_FLOAT_EQ(nd[0][0], 0.5f);
    EXPECT_FLOAT_EQ(nd[0][1], -0.5f);
    EXPECT_FLOAT_EQ(nd[0][2], 0.0f);
    EXPECT_FLOAT_EQ(nd[0][3], 127.0f / 128.0f);
}

TEST(sam_scan, SetLabelsValidation) {
    auto h = make_scan(2, 2, 8);
    EXPECT_THROW(h.set_labels({0, 1}, {"healthy", "d"}), std::invalid_argument);
    h.set_labels({0, 1, 1, 0}, {"healthy", "d"});
    EXPECT_EQ(h.samlabels().labels(), (std::vector<std::int8_t>{0, 1, 1, 0}));
}

TEST(sam_scan, DownsampleSampleMode) {
    auto h = make_scan(1, 1, 8);
    for (size_t j = 0; j < 8; ++j) {
        h.data()[0][j] = static_cast<std::int8_t>(j);
    }
    h.downsample(2, downsample_mode::sample);
    EXPECT_EQ(h.scanlen(), 4);
    EXPECT_EQ(h.data()[0][0], 0);
    EXPECT_EQ(h.data()[0][1], 2);
    EXPECT_EQ(h.data()[0][3], 6);
    EXPECT_EQ(h.samplerate(), 50.0);
    EXPECT_EQ(h.downsample_factor(), 2);
}

TEST(sam_scan, DownsampleMeanMode) {
    auto h = make_scan(1, 1, 8);
    for (size_t j = 0; j < 8; ++j) {
        h.data()[0][j] = static_cast<std::int8_t>(j);
    }
    h.downsample(2, downsample_mode::mean);
    EXPECT_EQ(h.data()[0][0], 0); // mean(0,1)=0.5 -> 0 (truncation)
    EXPECT_EQ(h.data()[0][1], 2); // mean(2,3)=2.5 -> 2
    EXPECT_EQ(h.data()[0][3], 6);
}

TEST(sam_scan, DownsampleMedianMode) {
    auto h = make_scan(1, 1, 8);
    for (size_t j = 0; j < 8; ++j) {
        h.data()[0][j] = static_cast<std::int8_t>(j);
    }
    h.downsample(4, downsample_mode::median);
    EXPECT_EQ(h.data()[0][0], 1); // median(0,1,2,3) = 1
    EXPECT_EQ(h.data()[0][1], 5); // median(4,5,6,7) = 5
}

TEST(sam_scan, DownsampleStartsFolded) {
    auto h = make_scan(2, 2, 100, true);
    h.downsample(10, downsample_mode::mean);
    if (h.starts().has_value()) {
        for (auto s : *h.starts()) {
            EXPECT_GE(s, -1);
            EXPECT_LT(s, 10);
        }
    }
}

TEST(sam_scan, DownsampleValidation) {
    auto h = make_scan(1, 1, 8);
    EXPECT_THROW(h.downsample(0), std::invalid_argument);
    EXPECT_THROW(h.downsample(9), std::invalid_argument);
}

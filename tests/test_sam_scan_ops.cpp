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

TEST(sam_scan, Rotate90ShapeAndData) {
    sam_header header(2, 3, 1, 100.0, 0, 1.0); // 3 lines, 2 cols
    array2d<std::int8_t> data(6, 1);
    for (size_t i = 0; i < 6; ++i) data[i][0] = static_cast<std::int8_t>(i);
    sam_labels labels(std::vector<std::int8_t>{10, 11, 12, 13, 14, 15},
                      {"healthy"});
    auto h = sam_scan::from_data(data, header, std::nullopt, labels);

    h.rotate(90);
    EXPECT_EQ(h.nlines(), 2);
    EXPECT_EQ(h.cols(), 3);
    // 90 CW on grid [[0,1],[2,3],[4,5]] -> [[4,2,0],[5,3,1]]
    EXPECT_EQ(h.data()[0][0], 4);
    EXPECT_EQ(h.data()[1][0], 2);
    EXPECT_EQ(h.data()[2][0], 0);
    EXPECT_EQ(h.data()[3][0], 5);
    EXPECT_EQ(h.data()[4][0], 3);
    EXPECT_EQ(h.data()[5][0], 1);
    // labels follow the same permutation
    EXPECT_EQ(h.samlabels().labels(),
              (std::vector<std::int8_t>{14, 12, 10, 15, 13, 11}));
}

TEST(sam_scan, Rotate180IdentityShape) {
    auto h = make_scan(2, 3, 2);
    auto orig = h.copy();
    h.rotate(180);
    EXPECT_EQ(h.nlines(), 2);
    EXPECT_EQ(h.cols(), 3);
    // out[i][j] = in[nl-1-i][nc-1-j]; the 180-degree permutation reverses
    // the flat scan order
    EXPECT_EQ(h.data()[0][0], orig.data()[5][0]);
    EXPECT_EQ(h.data()[5][0], orig.data()[0][0]);
    EXPECT_EQ(h.data()[1][0], orig.data()[4][0]);
}

TEST(sam_scan, Rotate270Shape) {
    auto h = make_scan(3, 2, 2);
    h.rotate(270);
    EXPECT_EQ(h.nlines(), 2);
    EXPECT_EQ(h.cols(), 3);
}

TEST(sam_scan, RotateNegativeDegrees) {
    auto h = make_scan(2, 3, 1);
    h.rotate(-90);
    EXPECT_EQ(h.nlines(), 3);
    EXPECT_EQ(h.cols(), 2);
}

TEST(sam_scan, RotateInvalidThrows) {
    auto h = make_scan(2, 2, 1);
    EXPECT_THROW(h.rotate(45), std::invalid_argument);
    EXPECT_THROW(h.rotate(360 + 45), std::invalid_argument);
}

TEST(sam_scan, RotateConstVariant) {
    auto h = make_scan(3, 2, 1);
    auto r = h.rotated(90);
    EXPECT_EQ(r.nlines(), 2);
    EXPECT_EQ(r.cols(), 3);
    EXPECT_EQ(h.nlines(), 3); // unchanged
}

TEST(sam_scan, RotateRoundTrip) {
    auto h = make_scan(3, 4, 5);
    auto h2 = h.rotated(90).rotated(90).rotated(90).rotated(90);
    EXPECT_EQ(h2.data(), h.data());
    EXPECT_EQ(h2.samlabels().labels(), h.samlabels().labels());
}

TEST(sam_scan, RotateStartsFollow) {
    auto h = make_scan(2, 3, 1, true);
    auto before = *h.starts();
    h.rotate(90);
    // permutation must be a bijection of the original starts
    auto after = *h.starts();
    std::vector<std::int32_t> sorted_b(before), sorted_a(after);
    std::sort(sorted_b.begin(), sorted_b.end());
    std::sort(sorted_a.begin(), sorted_a.end());
    EXPECT_EQ(sorted_a, sorted_b);
}

TEST(sam_scan, MirrorX) {
    sam_header header(2, 2, 1, 100.0, 0, 1.0);
    array2d<std::int8_t> data(4, 1);
    for (size_t i = 0; i < 4; ++i) data[i][0] = static_cast<std::int8_t>(i);
    sam_labels labels(std::vector<std::int8_t>{10, 11, 12, 13}, {"healthy"});
    auto h = sam_scan::from_data(data, header, std::nullopt, labels);
    h.mirror(mirror_axis::x);
    // [[0,1],[2,3]] -> [[1,0],[3,2]]
    EXPECT_EQ(h.data()[0][0], 1);
    EXPECT_EQ(h.data()[1][0], 0);
    EXPECT_EQ(h.data()[2][0], 3);
    EXPECT_EQ(h.data()[3][0], 2);
    EXPECT_EQ(h.samlabels().labels(),
              (std::vector<std::int8_t>{11, 10, 13, 12}));
    EXPECT_EQ(h.nlines(), 2);
}

TEST(sam_scan, MirrorY) {
    sam_header header(2, 2, 1, 100.0, 0, 1.0);
    array2d<std::int8_t> data(4, 1);
    for (size_t i = 0; i < 4; ++i) data[i][0] = static_cast<std::int8_t>(i);
    auto h = sam_scan::from_data(data, header);
    h.mirror(mirror_axis::y);
    EXPECT_EQ(h.data()[0][0], 2);
    EXPECT_EQ(h.data()[1][0], 3);
    EXPECT_EQ(h.data()[2][0], 0);
    EXPECT_EQ(h.data()[3][0], 1);
}

TEST(sam_scan, MirrorRoundTrip) {
    auto h = make_scan(3, 4, 5);
    auto h2 = h.mirrored(mirror_axis::x).mirrored(mirror_axis::x);
    EXPECT_EQ(h2.data(), h.data());
}

TEST(sam_scan, MirrorXYComposition) {
    // mirror x then y == 180-degree rotation
    auto h = make_scan(3, 4, 5);
    auto a = h.mirrored(mirror_axis::x).mirrored(mirror_axis::y);
    auto b = h.rotated(180);
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.samlabels().labels(), b.samlabels().labels());
}

TEST(sam_scan, RectangleSelect) {
    auto h = make_scan(4, 5, 3);
    auto sel = h.rectangle_select(1, 3, 1, 4);
    EXPECT_EQ(sel.nlines(), 2);
    EXPECT_EQ(sel.cols(), 3);
    EXPECT_EQ(sel.num_scans(), 6);
    // first selected scan is original flat index 1*5+1 = 6
    EXPECT_EQ(sel.data()[0][0], h.data()[6][0]);
    EXPECT_EQ(sel.data()[5][2], h.data()[13][2]);
    EXPECT_EQ(sel.samlabels().labels().size(), 6);
    // original unchanged
    EXPECT_EQ(h.num_scans(), 20);
}

TEST(sam_scan, RectangleSelectValidation) {
    auto h = make_scan(4, 5, 3);
    EXPECT_THROW((void)h.rectangle_select(3, 1, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)h.rectangle_select(0, 5, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)h.rectangle_select(0, 1, 4, 6), std::invalid_argument);
}

TEST(sam_scan, TimeRangeSelect) {
    auto h = make_scan(2, 2, 100);
    // samplerate 100 MHz -> 10 ns/sample; tzero 1000
    auto sel = h.time_range_select(1100.0, 1500.0);
    EXPECT_EQ(sel.scanlen(), 40);
    EXPECT_EQ(sel.header().tzero, 1100);
    EXPECT_EQ(sel.data()[0][0], h.data()[0][10]);
    // starts dropped
    EXPECT_FALSE(sel.starts().has_value());
}

TEST(sam_scan, TimeRangeSelectClamping) {
    auto h = make_scan(1, 1, 100);
    auto sel = h.time_range_select(500.0, 1500.0); // start clamps to 0
    EXPECT_EQ(sel.scanlen(), 50);
    EXPECT_EQ(sel.header().tzero, 1000);
}

TEST(sam_scan, TimeRangeSelectInvalidThrows) {
    auto h = make_scan(1, 1, 100);
    EXPECT_THROW((void)h.time_range_select(1500.0, 1100.0), std::invalid_argument);
}

TEST(sam_scan, ZgateBasic) {
    sam_header header(2, 2, 200, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(4, 200, 0);
    // scan 0: strong pulse at sample 20
    data[0][20] = 100;
    data[0][30] = -90;
    // scan 1: pulse late
    data[1][150] = 80;
    // scan 2: no crossing
    // scan 3: crossing beyond max_start -> clamped
    data[3][195] = 90;
    auto h = sam_scan::from_data(data, header);

    auto g = h.zgate(0.5, 50); // threshold 63.5
    EXPECT_EQ(g.scanlen(), 50);
    ASSERT_TRUE(g.starts().has_value());
    EXPECT_EQ((*g.starts())[0], 20);
    EXPECT_EQ(g.data()[0][0], 100);
    EXPECT_EQ(g.data()[0][10], -90);
    EXPECT_EQ((*g.starts())[1], 150);
    EXPECT_EQ((*g.starts())[2], -1);
    EXPECT_TRUE(std::all_of(g.data()[2].begin(), g.data()[2].end(),
                            [](std::int8_t v) { return v == 0; }));
    EXPECT_EQ((*g.starts())[3], 150); // clamped to 200-50
    EXPECT_EQ(g.data()[3][45], 90);
    // original unchanged (non-in-place)
    EXPECT_EQ(h.scanlen(), 200);
}

TEST(sam_scan, ZgateAllEqualFoldsTzero) {
    sam_header header(1, 1, 200, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(1, 200, 0);
    data[0][40] = 100;
    auto h = sam_scan::from_data(data, header);
    auto g = h.zgate(0.5, 50);
    EXPECT_FALSE(g.starts().has_value());
    // tzero += round(40 / 100 * 1e3) = 400
    EXPECT_EQ(g.header().tzero, 1400);
}

TEST(sam_scan, ZgateAccumulatesStarts) {
    sam_header header(2, 1, 200, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(2, 200, 0);
    data[0][10] = 100;
    data[1][20] = 100;
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{5, 5};
    auto h = sam_scan::from_data(data, header, starts);
    auto g = h.zgate(0.5, 50);
    ASSERT_TRUE(g.starts().has_value());
    EXPECT_EQ((*g.starts())[0], 15);
    EXPECT_EQ((*g.starts())[1], 25);
}

TEST(sam_scan, ZgateValidation) {
    auto h = make_scan(1, 1, 200);
    EXPECT_THROW((void)h.zgate(0.5, 0), std::invalid_argument);
    EXPECT_THROW((void)h.zgate(0.5, 300), std::invalid_argument);
    EXPECT_THROW((void)h.zgate(-0.1, 50), std::invalid_argument);
    EXPECT_THROW((void)h.zgate(1.5, 50), std::invalid_argument);
}

TEST(sam_scan, SpectralMethods) {
    auto h = make_scan(2, 2, 2000);
    auto st = h.compute_stft();
    EXPECT_EQ(st.zxx.size0(), 4);
    EXPECT_EQ(st.zxx.size1(), 129);
    EXPECT_EQ(st.zxx.size2(), 14);
    // t aligned to tzero*1e-9 seconds
    EXPECT_NEAR(st.t[0], 1000.0e-9 + 128.0 / 100.0e6, 1e-9);

    auto p = h.psd();
    EXPECT_EQ(p.psd.rows(), 4);
    EXPECT_EQ(p.psd.cols(), 129);

    auto s = h.power_spectrogram();
    EXPECT_EQ(s.sxx.size0(), 4);
    EXPECT_EQ(s.sxx.size1(), 129);
    EXPECT_EQ(s.sxx.size2(), 14);

    EXPECT_THROW((void)h.compute_stft(3000, 128), std::invalid_argument);
}

TEST(sam_scan, CopyIsDeep) {
    auto h = make_scan(2, 2, 8);
    auto c = h.copy();
    c.data()[0][0] = 99;
    EXPECT_NE(c.data()[0][0], h.data()[0][0]);
    c.header().scanlen = 42;
    EXPECT_EQ(h.scanlen(), 8);
}

TEST(sam_scan, HeaderHashStable) {
    auto h = make_scan(2, 2, 8);
    EXPECT_EQ(h.header_hash(), h.copy().header_hash());
}

TEST(sam_scan, ZgateMinus128NotACrossing) {
    // numpy parity: np.abs on int8 wraps |-128| to -128, so a -128 sample
    // must NOT be treated as a threshold crossing.
    sam_header header(2, 1, 200, 100.0, 0, 1.0);
    array2d<std::int8_t> data(2, 200, 0);
    data[0][10] = -128;      // skipped by the crossing search
    data[0][50] = 100;       // first real crossing of scan 0
    data[1][20] = 90;        // first real crossing of scan 1
    auto h = sam_scan::from_data(data, header);
    auto g = h.zgate(0.5, 50);
    ASSERT_TRUE(g.starts().has_value());
    EXPECT_EQ((*g.starts())[0], 50);
    EXPECT_EQ((*g.starts())[1], 20);
    EXPECT_EQ(g.data()[0][0], 100);
}

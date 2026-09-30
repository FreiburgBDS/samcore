#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <samcore/sam_scan.hpp>
#include <samcore/utils.hpp>

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

// Gaussian-modulated 20%-of-Nyquist tone centred at `center`; the analytic
// envelope of this waveform peaks at `center`.
std::int8_t burst_sample(std::int64_t j, double center, double width,
                         double amplitude) {
    const double t = static_cast<double>(j) - center;
    const double x = amplitude * std::exp(-0.5 * (t / width) * (t / width)) *
                     std::sin(2.0 * 3.14159265358979323846 * 0.2 * t);
    return static_cast<std::int8_t>(
        std::clamp(static_cast<int>(std::lround(x)), -128, 127));
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

TEST(sam_scan, Rotate180KeepsSamplesInOrder) {
    // Regression: rotating 180 must permute whole signals only; samples
    // inside a signal stay in time order.
    sam_header header(2, 2, 3, 100.0, 0, 1.0); // 2 lines x 2 cols
    array2d<std::int8_t> data(4, 3);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            data[i][j] = static_cast<std::int8_t>(i * 10 + j);
        }
    }
    auto h = sam_scan::from_data(data, header);
    h.rotate(180);
    EXPECT_EQ(h.data()[0][0], 30);
    EXPECT_EQ(h.data()[0][1], 31);
    EXPECT_EQ(h.data()[0][2], 32);
    EXPECT_EQ(h.data()[3][0], 0);
    EXPECT_EQ(h.data()[3][1], 1);
    EXPECT_EQ(h.data()[3][2], 2);
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

TEST(sam_scan, MirrorYKeepsColumnsInOrder) {
    // Regression: mirror y swaps whole lines; the column order (and the
    // samples of every signal) must be preserved.
    sam_header header(2, 2, 2, 100.0, 0, 1.0); // 2 lines x 2 cols
    array2d<std::int8_t> data(4, 2);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            data[i][j] = static_cast<std::int8_t>(i * 10 + j);
        }
    }
    auto h = sam_scan::from_data(data, header);
    h.mirror(mirror_axis::y);
    // line 0 <- line 1 (signals 20, 30), line 1 <- line 0 (signals 0, 10)
    EXPECT_EQ(h.data()[0][0], 20);
    EXPECT_EQ(h.data()[0][1], 21);
    EXPECT_EQ(h.data()[1][0], 30);
    EXPECT_EQ(h.data()[1][1], 31);
    EXPECT_EQ(h.data()[2][0], 0);
    EXPECT_EQ(h.data()[2][1], 1);
    EXPECT_EQ(h.data()[3][0], 10);
    EXPECT_EQ(h.data()[3][1], 11);
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

TEST(sam_scan, IndexRangeSelectPreservesStarts) {
    sam_header header(1, 3, 100, 100.0, 1000, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(3, 100);
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(i * 10 + j);
        }
    }
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{10, 20, -1};
    auto h = sam_scan::from_data(data, header, starts);

    auto sel = h.index_range_select(5, 15);
    EXPECT_EQ(sel.scanlen(), 10);
    ASSERT_TRUE(sel.starts().has_value());
    EXPECT_EQ((*sel.starts())[0], 15);
    EXPECT_EQ((*sel.starts())[1], 25);
    EXPECT_EQ((*sel.starts())[2], -1);
    EXPECT_EQ(sel.data()[0][0], h.data()[0][5]);
    EXPECT_EQ(sel.data()[1][9], h.data()[1][14]);
    // absolute times of the retained samples are unchanged for valid rows
    for (size_t i : {0, 1}) {
        const auto before = h.time(i);
        const auto after = sel.time(i);
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_DOUBLE_EQ(after[j], before[5 + j])
                << "row " << i << " sample " << j;
        }
    }
}

TEST(sam_scan, IndexRangeSelectCollapsesUniformStarts) {
    sam_header header(1, 2, 100, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(2, 100, 3);
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{10, 10};
    auto h = sam_scan::from_data(data, header, starts);

    auto sel = h.index_range_select(3, 8);
    EXPECT_EQ(sel.scanlen(), 5);
    EXPECT_FALSE(sel.starts().has_value());
    EXPECT_EQ(sel.header().tzero, 1130); // 1000 + (10 + 3) * 10
    EXPECT_DOUBLE_EQ(sel.time()[0], 1130.0);
}

TEST(sam_scan, IndexRangeSelectUnalignedShiftsTzero) {
    sam_header header(1, 2, 100, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(2, 100, 1);
    auto h = sam_scan::from_data(data, header);

    auto sel = h.index_range_select(10, 20);
    EXPECT_EQ(sel.scanlen(), 10);
    EXPECT_FALSE(sel.starts().has_value());
    EXPECT_EQ(sel.header().tzero, 1100);
    EXPECT_DOUBLE_EQ(sel.time()[0], 1100.0);
}

TEST(sam_scan, IndexRangeSelectValidationAndInPlace) {
    auto h = make_scan(2, 2, 100);
    EXPECT_THROW((void)h.index_range_select(50, 50), std::invalid_argument);
    EXPECT_THROW((void)h.index_range_select(60, 50), std::invalid_argument);
    // out-of-range indices are clamped
    auto clamped = h.index_range_select(-5, 200);
    EXPECT_EQ(clamped.scanlen(), 100);
    EXPECT_EQ(clamped.data(), h.data());

    auto ip = h.copy();
    ip.index_range_select_ip(10, 20);
    EXPECT_EQ(ip.scanlen(), 10);
    EXPECT_EQ(ip.data()[0][0], h.data()[0][10]);
    EXPECT_EQ(h.scanlen(), 100); // source untouched by the copy path
}

TEST(sam_scan, TimeRangeSelectPreservesStarts) {
    sam_header header(1, 2, 100, 100.0, 1000, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(2, 100, 4);
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{0, 20};
    auto h = sam_scan::from_data(data, header, starts);

    auto sel = h.time_range_select(1100.0, 1200.0); // local samples [10, 20)
    EXPECT_EQ(sel.scanlen(), 10);
    ASSERT_TRUE(sel.starts().has_value());
    EXPECT_EQ((*sel.starts())[0], 10);
    EXPECT_EQ((*sel.starts())[1], 30);
    EXPECT_EQ(sel.header().tzero, 1000);
    EXPECT_DOUBLE_EQ(sel.time(1)[0], 1300.0); // 1000 + 30 * 10
}

TEST(sam_scan, SampleTimeConversions) {
    auto h = make_scan(1, 1, 100); // 100 MHz -> 10 ns/sample, tzero 1000
    EXPECT_DOUBLE_EQ(h.sample_time(0), 1000.0);
    EXPECT_DOUBLE_EQ(h.sample_time(25), 1250.0);
    EXPECT_EQ(h.sample_index(1000.0), 0);
    EXPECT_EQ(h.sample_index(1249.0), 25);
    EXPECT_EQ(h.sample_index(1250.0), 25);
    // ties round to nearest even (std::nearbyint semantics)
    EXPECT_EQ(h.sample_index(1005.0), 0); // 0.5 -> 0
    EXPECT_EQ(h.sample_index(1015.0), 2); // 1.5 -> 2
    for (std::int64_t i : {0, 1, 7, 99}) {
        EXPECT_EQ(h.sample_index(h.sample_time(i)), i);
    }
}

TEST(sam_scan, RelativeAndAbsoluteTime) {
    auto h = make_scan(2, 2, 100, true); // starts[3] = 3
    EXPECT_EQ(h.relative_time(), h.time());
    EXPECT_EQ(h.absolute_time(3), h.time(3));
    EXPECT_DOUBLE_EQ(h.absolute_time(3)[0], 1000.0 + 3 * 10.0);
    EXPECT_THROW((void)h.absolute_time(99), std::out_of_range);

    // without starts both accessors return the shared axis
    auto plain = make_scan(1, 1, 100);
    EXPECT_EQ(plain.relative_time(), plain.time());
    EXPECT_EQ(plain.absolute_time(0), plain.time());
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

TEST(sam_scan, SpectrumAndSpectralBand) {
    auto h = make_scan(2, 2, 2000); // samplerate 100 MHz -> fs = 1e8 Hz
    const auto spec = h.spectrum();
    EXPECT_EQ(spec.mag.rows(), 4u);
    EXPECT_EQ(spec.mag.cols(), 1001u);
    EXPECT_NEAR(spec.f.back(), 50.0e6f, 1.0f);
    // per-row agreement with the single-signal utility
    for (size_t i = 0; i < 4; ++i) {
        std::vector<float> row(h.data()[i].begin(), h.data()[i].end());
        auto [m1, f1] = utils::fft_spec(row, 1.0 / 1.0e8);
        for (size_t k = 0; k < spec.mag.cols(); ++k) {
            // float32 batched FFT vs the double reference: ~1e-4 relative
            EXPECT_NEAR(spec.mag[i][k], m1[k],
                        5e-4 * (1.0 + std::abs(m1[k])))
                << "row " << i << " bin " << k;
            EXPECT_NEAR(spec.f[k], f1[k], 10.0f) << "bin " << k;
        }
    }

    auto [f_band, psd_band] = h.psd(256, 128, 10.0e6, 20.0e6);
    ASSERT_GT(f_band.size(), 0u);
    for (float f : f_band) {
        EXPECT_GE(f, 10.0e6f);
        EXPECT_LE(f, 20.0e6f);
    }
    EXPECT_EQ(psd_band.cols(), f_band.size());

    auto pf = h.stft_peak_frequency(256, 128, 10.0e6, 20.0e6);
    EXPECT_EQ(pf.rows(), 2u);
    EXPECT_EQ(pf.cols(), 2u);
    auto pt = h.stft_peak_time(256, 128);
    EXPECT_EQ(pt.rows(), 2u);
    EXPECT_EQ(pt.cols(), 2u);
    EXPECT_THROW((void)h.psd(256, 128, 20.0e6, 10.0e6),
                 std::invalid_argument);
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

TEST(sam_scan, AlignXcorrShiftsRowsToReference) {
    constexpr size_t n = 200;
    sam_header header(1, 3, static_cast<std::int64_t>(n), 100.0, 1000, 1.0);
    std::vector<std::int8_t> tmpl(n);
    for (size_t i = 0; i < n; ++i) {
        tmpl[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 37 + 11) % 200) - 100);
    }
    array2d<std::int8_t> data(3, n, 0);
    for (size_t j = 0; j < n; ++j) data[0][j] = tmpl[j];
    for (size_t j = 7; j < n; ++j) data[1][j] = tmpl[j - 7];   // delayed 7
    for (size_t j = 0; j + 5 < n; ++j) data[2][j] = tmpl[j + 5]; // early 5

    auto h = sam_scan::from_data(data, header);
    h.align_xcorr(0, 30);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 20; j + 20 < n; ++j) {
            ASSERT_EQ(h.data()[i][j], h.data()[0][j]) << "row " << i
                                                      << " sample " << j;
        }
    }
}

TEST(sam_scan, AlignXcorrAdvancesStarts) {
    constexpr size_t n = 200;
    sam_header header(1, 3, static_cast<std::int64_t>(n), 100.0, 1000, 1.0);
    std::vector<std::int8_t> tmpl(n);
    for (size_t i = 0; i < n; ++i) {
        tmpl[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 41 + 3) % 180) - 90);
    }
    array2d<std::int8_t> data(3, n, 0);
    for (size_t j = 0; j < n; ++j) data[0][j] = tmpl[j];
    for (size_t j = 9; j < n; ++j) data[1][j] = tmpl[j - 9];
    for (size_t j = 0; j + 4 < n; ++j) data[2][j] = tmpl[j + 4];
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{100, 100, 100};

    auto h = sam_scan::from_data(data, header, starts);
    h.align_xcorr(0, 30);
    ASSERT_TRUE(h.starts().has_value());
    // shifts are [0, +9, -4]; starts advance by the applied shift
    EXPECT_EQ((*h.starts())[0], 100);
    EXPECT_EQ((*h.starts())[1], 109);
    EXPECT_EQ((*h.starts())[2], 96);
}

TEST(sam_scan, AlignXcorrCopyVariantAndValidation) {
    auto h = make_scan(2, 2, 64);
    auto before = h.copy();
    auto aligned = h.aligned_xcorr(0, 8);
    EXPECT_EQ(h.data(), before.data()); // original untouched
    EXPECT_EQ(aligned.data().rows(), before.data().rows());

    EXPECT_THROW(h.align_xcorr(99), std::out_of_range);
    EXPECT_THROW(h.align_xcorr(0, 64), std::invalid_argument);
    EXPECT_THROW(h.align_xcorr(0, -1), std::invalid_argument);
}

TEST(sam_scan, AlignTofAlignsEchoes) {
    sam_header header(1, 2, 400, 100.0, 1000, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(2, 400, 0);
    for (std::int64_t j = 0; j < 400; ++j) {
        data[0][j] = burst_sample(j, 100, 8.0, 100.0);
        data[1][j] = burst_sample(j, 130, 8.0, 100.0);
    }
    auto h = sam_scan::from_data(data, header);

    // 2500 ns = 250 samples from tzero; covers both echoes
    h.align_tof(2500.0, 0);
    // row 1 is shifted left by 30 samples so the echoes coincide
    for (size_t j = 40; j < 350; ++j) {
        ASSERT_EQ(h.data()[1][j], h.data()[0][j]) << "sample " << j;
    }
}

TEST(sam_scan, AlignTofAdvancesStartsAndCopyVariant) {
    sam_header header(1, 2, 400, 100.0, 1000, 1.0);
    array2d<std::int8_t> data(2, 400, 0);
    for (std::int64_t j = 0; j < 400; ++j) {
        data[0][j] = burst_sample(j, 100, 8.0, 100.0);
        data[1][j] = burst_sample(j, 130, 8.0, 100.0);
    }
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{50, 50};
    auto h = sam_scan::from_data(data, header, starts);

    auto aligned = h.aligned_tof(2500.0, 0);
    EXPECT_EQ(h.data(), data); // original untouched
    ASSERT_TRUE(aligned.starts().has_value());
    EXPECT_EQ((*aligned.starts())[0], 50);
    EXPECT_EQ((*aligned.starts())[1], 80); // delayed echo -> +30 shift
}

TEST(sam_scan, AlignTofRoundsSubSampleOffsets) {
    sam_header header(1, 2, 400, 100.0, 0, 1.0); // tzero 0, 10 ns/sample
    array2d<std::int8_t> data(2, 400, 0);
    for (std::int64_t j = 0; j < 400; ++j) {
        data[0][j] = burst_sample(j, 100, 8.0, 100.0);
        data[1][j] = burst_sample(j, 110.6, 8.0, 100.0);
    }
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{30, 30};
    auto h = sam_scan::from_data(data, header, starts);
    h.align_tof(2500.0, 0);
    // sub-sample peak offset 10.6 rounds to an 11-sample shift
    ASSERT_TRUE(h.starts().has_value());
    EXPECT_EQ((*h.starts())[1], 41);
}

TEST(sam_scan, AlignTofValidation) {
    sam_header header(1, 2, 200, 100.0, 1000, 1.0);
    array2d<std::int8_t> zeros(2, 200, 0);
    auto silent = sam_scan::from_data(zeros, header);
    // silent rows have no envelope peak in the gate
    EXPECT_THROW(silent.align_tof(100.0), std::invalid_argument);

    auto h = make_scan(1, 2, 200);
    EXPECT_THROW(h.align_tof(0.0), std::invalid_argument);
    EXPECT_THROW(h.align_tof(-5.0), std::invalid_argument);
    EXPECT_THROW(h.align_tof(100.0, 5), std::out_of_range);
    EXPECT_THROW(h.align_tof(100.0, 0, -1.0), std::invalid_argument);
}

TEST(sam_scan, TofEnvelopePeak) {
    sam_header header(2, 2, 400, 100.0, 1000, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(4, 400, 0);
    for (std::int64_t j = 0; j < 400; ++j) {
        // row 0 carries both echoes so the gate can select between them
        const int two = static_cast<int>(burst_sample(j, 100, 8.0, 100.0)) +
                        static_cast<int>(burst_sample(j, 250, 6.0, 60.0));
        data[0][j] = static_cast<std::int8_t>(std::clamp(two, -128, 127));
        data[1][j] = burst_sample(j, 250, 6.0, 60.0);
        data[2][j] = burst_sample(j, 100, 8.0, 100.0);
        // data[3] stays silent
    }
    auto h = sam_scan::from_data(data, header);

    auto t = h.tof();
    ASSERT_EQ(t.rows(), 2);
    ASSERT_EQ(t.cols(), 2);
    // tzero 1000 + 100 * 10 ns, refined to the exact envelope centre
    EXPECT_NEAR(t[0][0], 2000.0f, 0.05f);
    EXPECT_NEAR(t[0][1], 3500.0f, 0.05f);
    // silent window -> NaN (bit pattern; fast-math breaks std::isnan)
    std::uint32_t bits = 0;
    std::memcpy(&bits, &t[1][1], sizeof(bits));
    EXPECT_EQ(bits & 0x7fffffffu, 0x7fc00000u);

    // the gate selects the burst inside it (second burst only)
    auto g = h.tof(180, 0);
    EXPECT_NEAR(g[0][0], 3500.0f, 0.05f);

    // without sub-sample refinement the pick is the exact integer sample
    auto coarse = h.tof(0, 0, false);
    EXPECT_FLOAT_EQ(coarse[0][0], 2000.0f);
}

TEST(sam_scan, TofThicknessAndValidation) {
    sam_header header(1, 1, 100, 100.0, 0, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(1, 100, 0);
    for (std::int64_t j = 0; j < 100; ++j) {
        data[0][j] = burst_sample(j, 25, 4.0, 100.0);
    }
    auto h = sam_scan::from_data(data, header);

    auto t = h.tof();
    EXPECT_NEAR(t[0][0], 250.0f, 0.05f); // tzero 0 + 25*10
    auto thick = h.thickness(1500.0);
    // 250 ns * 1e-9 * 1500 m/s / 2 = 1.875e-4 m
    EXPECT_NEAR(thick[0][0], 1.875e-4f, 1e-9f);

    EXPECT_THROW((void)h.tof(50, 50), std::invalid_argument);
    EXPECT_THROW((void)h.tof(0, 200), std::invalid_argument);
    EXPECT_THROW((void)h.tof(-1, 0), std::invalid_argument);
    EXPECT_THROW((void)h.thickness(0.0), std::invalid_argument);
}

TEST(sam_scan, XgateThresholdPickAndModes) {
    sam_header header(1, 2, 20, 100.0, 0, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(2, 20, 0);
    for (std::int64_t j = 0; j < 20; ++j) {
        data[0][j] = static_cast<std::int8_t>(j);  // first > 12.7 at sample 13
        data[1][j] = static_cast<std::int8_t>(-j); // no positive crossing
    }
    auto h = sam_scan::from_data(data, header);

    // 50 ns = 5 samples; only one gate fits from sample 13
    auto r = h.xgate(50.0, 2, "threshold", 0.1, "max");
    ASSERT_EQ(r.values.size0(), 2u); // nlines
    ASSERT_EQ(r.values.size1(), 1u); // cols
    ASSERT_EQ(r.values.size2(), 2u);
    ASSERT_EQ(r.starts.size(), 2u);
    EXPECT_EQ(r.starts[0], 13);
    EXPECT_EQ(r.starts[1], -1);
    EXPECT_FLOAT_EQ(r.values.flat()[0], 17.0f); // max(13..17)
    EXPECT_FLOAT_EQ(r.values.flat()[1], 0.0f);  // second gate does not fit
    EXPECT_FLOAT_EQ(r.values.flat()[2], 0.0f);  // failed pick row
    EXPECT_FLOAT_EQ(r.values.flat()[3], 0.0f);

    // 30 ns = 3 samples -> two fitting gates (13..15, 16..18)
    auto r2 = h.xgate(30.0, 2, "threshold", 0.1, "max");
    EXPECT_EQ(r2.starts[0], 13);
    EXPECT_FLOAT_EQ(r2.values.flat()[0], 15.0f);
    EXPECT_FLOAT_EQ(r2.values.flat()[1], 18.0f);

    // pick none + absmax / power on the negative row
    auto rn = h.xgate(50.0, 4, "none", -1.0, "absmax");
    EXPECT_EQ(rn.starts[0], 0);
    EXPECT_EQ(rn.starts[1], 0);
    EXPECT_FLOAT_EQ(rn.values.flat()[4], 4.0f); // row 1 gate 0: max |0..4|
    auto rp = h.xgate(50.0, 4, "none", -1.0, "power");
    EXPECT_FLOAT_EQ(rp.values.flat()[4], 30.0f); // 0+1+4+9+16
}

TEST(sam_scan, XgateTofPickAndValidation) {
    constexpr size_t n = 200;
    sam_header header(1, 2, n, 100.0, 0, 1.0); // 10 ns/sample
    array2d<std::int8_t> data(2, n, 0);
    for (std::int64_t j = 0; j < static_cast<std::int64_t>(n); ++j) {
        data[0][j] = burst_sample(j, 100, 8.0, 100.0);
        data[1][j] = burst_sample(j, 50, 8.0, 100.0);
    }
    auto h = sam_scan::from_data(data, header);

    auto r = h.xgate(100.0, 3, "tof");
    EXPECT_NEAR(r.starts[0], 100, 1);
    EXPECT_NEAR(r.starts[1], 50, 1);
    EXPECT_GT(r.values.flat()[0], 0.0f);
    EXPECT_LE(r.values.flat()[0], 127.0f);

    // n_gates = 0 -> empty layer stack but starts still reported
    auto empty = h.xgate(100.0, 0, "tof");
    EXPECT_EQ(empty.values.size2(), 0u);
    EXPECT_NEAR(empty.starts[0], 100, 1);

    EXPECT_THROW((void)h.xgate(0.0), std::invalid_argument);
    EXPECT_THROW((void)h.xgate(100.0, 3, "bogus"), std::invalid_argument);
    EXPECT_THROW((void)h.xgate(100.0, 3, "none", -1.0, "bogus"),
                 std::invalid_argument);
    EXPECT_THROW((void)h.xgate(100.0, 3, "threshold", -1.0),
                 std::invalid_argument);
}

TEST(sam_scan, AlignManualPublic) {
    // align_manual accumulates starts and zero-fills -1 rows.
    sam_header header(1, 2, 100, 100.0, 0, 1.0);
    array2d<std::int8_t> data(2, 100, 5);
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{10, 20};
    auto h = sam_scan::from_data(data, header, starts);
    h.align_manual({3, -1}, 50);
    ASSERT_TRUE(h.starts().has_value());
    EXPECT_EQ((*h.starts())[0], 13);
    EXPECT_EQ((*h.starts())[1], -1);
    EXPECT_TRUE(std::all_of(h.data()[1].begin(), h.data()[1].end(),
                            [](std::int8_t v) { return v == 0; }));
    EXPECT_EQ(h.data()[0][0], 5); // valid rows keep their data
    EXPECT_THROW(h.align_manual({100, 0}, 50), std::invalid_argument);
}

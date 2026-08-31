#include <gtest/gtest.h>

#include <filesystem>

#include <samcore/sam_scan.hpp>

namespace {

std::filesystem::path data_dir() {
    return std::filesystem::path(SAMCORE_TEST_DATA_DIR);
}

std::filesystem::path h5sam_path() { return data_dir() / "testdata.h5sam"; }

std::filesystem::path tmp_file(const std::string& name) {
    std::filesystem::path dir = std::filesystem::temp_directory_path();
    return dir / name;
}

} // namespace

using samcore::sam_scan;

TEST(io_h5sam, ReadLegacyFile) {
    if (!std::filesystem::exists(h5sam_path())) GTEST_SKIP() << "no h5sam testdata";
    auto h = sam_scan::from_file(h5sam_path());
    EXPECT_EQ(h.data().rows(),
              static_cast<size_t>(h.header().nlines * h.header().scanspline));
    EXPECT_EQ(h.data().cols(), static_cast<size_t>(h.header().scanlen));
    EXPECT_GT(h.header().samplerate, 0.0);
    EXPECT_EQ(h.samlabels().labels().size(), h.data().rows());
}

TEST(io_h5sam, SampleTestdataHasStarts) {
    if (!std::filesystem::exists(h5sam_path())) GTEST_SKIP() << "no h5sam testdata";
    auto h = sam_scan::from_file(h5sam_path());
    ASSERT_TRUE(h.starts().has_value());
    EXPECT_EQ(h.starts()->size(), h.data().rows());
    for (const auto s : *h.starts()) {
        if (s != -1) {
            EXPECT_GE(s, 0);
            EXPECT_LT(s, static_cast<std::int32_t>(h.scanlen()));
        }
    }
}

TEST(io_h5sam, RoundTripPreservesEverything) {
    if (!std::filesystem::exists(h5sam_path())) GTEST_SKIP() << "no h5sam testdata";
    auto h = sam_scan::from_file(h5sam_path());

    const std::filesystem::path out = tmp_file("samcore_roundtrip.h5sam");
    h.to_h5sam(out);
    auto h2 = sam_scan::from_file(out);

    EXPECT_EQ(h2.data(), h.data());
    EXPECT_EQ(h2.header().scanlen, h.header().scanlen);
    EXPECT_EQ(h2.header().nlines, h.header().nlines);
    EXPECT_EQ(h2.header().scanspline, h.header().scanspline);
    EXPECT_EQ(h2.samlabels().labels(), h.samlabels().labels());
    EXPECT_EQ(h2.samlabels().label_names(), h.samlabels().label_names());
    std::filesystem::remove(out);
}

TEST(io_h5sam, RoundTripWithStarts) {
    // Round-trip per-scan starts through .h5sam using a scan built in
    // memory (the bundled sample testdata also carries starts, see
    // SampleTestdataHasStarts).
    samcore::sam_header header(4, 3, 128, 50.0, 10, 2.5);
    samcore::array2d<std::int8_t> data(12, 128);
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>((i + j) % 100 - 50);
        }
    }
    std::vector<std::int32_t> starts = {0, 5, -1, 3, 2, 1, 0, 7, -1, 2, 3, 4};
    auto h = samcore::sam_scan::from_data(data, header, starts);

    const std::filesystem::path out = tmp_file("samcore_roundtrip_starts.h5sam");
    h.to_h5sam(out);
    auto h2 = sam_scan::from_file(out);
    ASSERT_TRUE(h2.starts().has_value());
    EXPECT_EQ(*h2.starts(), starts);
    EXPECT_EQ(h2.data(), data);
    std::filesystem::remove(out);
}

TEST(io_h5sam, WriteThenReadBackFromScratch) {
    samcore::sam_header header(4, 3, 128, 50.0, 10, 2.5, false,
                               true, "echo", "t1", "t2", "cell", 1,
                               {{"extra_key", "extra_value"}});
    samcore::array2d<std::int8_t> data(12, 128);
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(
                static_cast<int>((i * 3 + j) % 100) - 50);
        }
    }
    samcore::sam_labels labels(
        std::vector<std::int8_t>{0, 1, -1, 2, 0, 1, 0, 2, -1, 0, 1, 1},
        {"healthy", "defect_a", "defect_b"});
    std::optional<std::vector<std::int32_t>> starts =
        std::vector<std::int32_t>{0, 5, -1, 3, 2, 1, 0, 7, -1, 2, 3, 4};

    auto scan = samcore::sam_scan::from_data(data, header, starts, labels);
    const std::filesystem::path out = tmp_file("samcore_scratch.h5sam");
    scan.to_h5sam(out);

    auto loaded = samcore::sam_scan::from_file(out);
    EXPECT_EQ(loaded.data(), data);
    EXPECT_EQ(loaded.header().scanlen, header.scanlen);
    EXPECT_EQ(loaded.header().samplerate, header.samplerate);
    EXPECT_EQ(loaded.samlabels().labels(), labels.labels());
    EXPECT_EQ(loaded.samlabels().label_names(), labels.label_names());
    EXPECT_EQ(std::get<std::string>(loaded.header().extra.at("extra_key")),
              "extra_value");
    ASSERT_TRUE(loaded.starts().has_value());
    EXPECT_EQ(*loaded.starts(), *starts);
    std::filesystem::remove(out);
}

TEST(io_h5sam, ExtraAttrsRoundTrip) {
    samcore::sam_header header(4, 3, 128, 50.0, 10, 2.5);
    header.extra["gain"] = std::int64_t{3};
    header.extra["gain_db"] = 6.5;
    header.extra["calibrated"] = std::string("[true, false]");
    samcore::array2d<std::int8_t> data(12, 128);
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>((i + j) % 100 - 50);
        }
    }
    auto scan = samcore::sam_scan::from_data(data, header);
    const std::filesystem::path out = tmp_file("samcore_extra_attrs.h5sam");
    scan.to_h5sam(out);
    auto loaded = samcore::sam_scan::from_file(out);
    EXPECT_EQ(std::get<std::int64_t>(loaded.header().extra.at("gain")), 3);
    EXPECT_DOUBLE_EQ(std::get<double>(loaded.header().extra.at("gain_db")), 6.5);
    EXPECT_EQ(std::get<std::string>(loaded.header().extra.at("calibrated")),
              "[true, false]");
    std::filesystem::remove(out);
}

TEST(io_h5sam, PartialRowRead) {
    if (!std::filesystem::exists(h5sam_path())) GTEST_SKIP() << "no h5sam testdata";
    auto h = sam_scan::from_file(h5sam_path());
    const auto rows = samcore::io::read_h5sam_rows(h5sam_path(), 0, 4);
    EXPECT_EQ(rows.rows(), 4);
    EXPECT_EQ(rows.cols(), h.data().cols());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(rows[i][0], h.data()[i][0]);
    }
}


TEST(io_h5sam, MmapLazyLoad) {
    if (!std::filesystem::exists(h5sam_path())) GTEST_SKIP() << "no h5sam testdata";
    // mmap mode: metadata available without loading the data
    samcore::sam_scan lazy = samcore::sam_scan::from_file(h5sam_path(), true);
    EXPECT_FALSE(lazy.loaded());
    EXPECT_EQ(lazy.num_scans(),
              static_cast<size_t>(lazy.header().nlines * lazy.header().scanspline));
    EXPECT_EQ(lazy.scanlen(), static_cast<size_t>(lazy.header().scanlen));
    EXPECT_EQ(lazy.samlabels().labels().size(), lazy.num_scans());

    // accessing data materializes it
    auto& d = lazy.data();
    EXPECT_TRUE(lazy.loaded());
    (void)d;

    // results identical to an eager load
    auto eager = samcore::sam_scan::from_file(h5sam_path());
    EXPECT_EQ(lazy.data(), eager.data());
    EXPECT_EQ(lazy.header(), eager.header());
    EXPECT_EQ(lazy.samlabels().labels(), eager.samlabels().labels());
}

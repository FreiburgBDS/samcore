#include <gtest/gtest.h>

#include <filesystem>

#include <samcore/sam_dataset.hpp>

namespace {

std::filesystem::path data_dir() {
    return std::filesystem::path(SAMCORE_TEST_DATA_DIR);
}

std::filesystem::path h5samd_path() { return data_dir() / "testdata.h5samd"; }

std::filesystem::path tmp_file(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

samcore::sam_scan make_cube(int nlines, int cols, int scanlen, int offset,
                            double resolution) {
    samcore::sam_header header(cols, nlines, scanlen, 100.0, 0,
                               resolution);
    samcore::array2d<std::int8_t> data(static_cast<size_t>(nlines * cols),
                                       static_cast<size_t>(scanlen));
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            data[i][j] = static_cast<std::int8_t>(
                static_cast<int>((i + offset) % 100) - 50);
        }
    }
    return samcore::sam_scan::from_data(std::move(data), std::move(header));
}

} // namespace

using namespace samcore;

TEST(io_h5samd, ReadLegacyFile) {
    if (!std::filesystem::exists(h5samd_path())) GTEST_SKIP() << "no h5samd testdata";
    auto ds = sam_dataset::load(h5samd_path());
    EXPECT_GT(ds.num_samples(), 0);
    EXPECT_GT(ds.maxlen(), 0);
    EXPECT_GE(ds.cube_shapes().size(), 1);
    EXPECT_EQ(ds.cube_resolutions().size(), ds.cube_shapes().size());
}

TEST(io_h5samd, RoundTrip) {
    sam_dataset ds({make_cube(3, 4, 100, 0, 1.0), make_cube(2, 5, 80, 7, 2.0)},
                   0.0f, true);
    EXPECT_EQ(ds.num_samples(), 3 * 4 + 2 * 5);
    EXPECT_EQ(ds.maxlen(), 100);
    EXPECT_TRUE(ds.unsupervised());

    const std::filesystem::path out = tmp_file("samcore_ds.h5samd");
    ds.save(out);
    auto ds2 = sam_dataset::load(out);
    EXPECT_EQ(ds2.X(), ds.X());
    EXPECT_EQ(ds2.cube_shapes(), ds.cube_shapes());
    EXPECT_EQ(ds2.cube_resolutions(), ds.cube_resolutions());
    EXPECT_EQ(ds2.scanlens(), ds.scanlens());
    EXPECT_TRUE(ds2.unsupervised());
    std::filesystem::remove(out);
}

TEST(io_h5samd, RoundTripSupervised) {
    auto scan1 = make_cube(2, 2, 64, 0, 1.0);
    scan1.set_labels({0, 1, 0, 1}, {"healthy", "defect"});
    auto scan2 = make_cube(2, 2, 64, 0, 1.0);
    scan2.set_labels({1, 0, 1, 0}, {"healthy", "defect"});
    sam_dataset ds({scan1, scan2});
    EXPECT_FALSE(ds.unsupervised());
    ASSERT_TRUE(ds.labels().has_value());
    EXPECT_EQ(ds.labels()->labels(),
              (std::vector<std::int8_t>{0, 1, 0, 1, 1, 0, 1, 0}));

    const std::filesystem::path out = tmp_file("samcore_ds_sup.h5samd");
    ds.save(out);
    auto ds2 = sam_dataset::load(out);
    ASSERT_TRUE(ds2.labels().has_value());
    EXPECT_EQ(ds2.labels()->labels(), ds.labels()->labels());
    EXPECT_EQ(ds2.labels()->label_names(), ds.labels()->label_names());
    std::filesystem::remove(out);
}

TEST(io_h5samd, ConvertFromH5sam) {
    if (!std::filesystem::exists(data_dir() / "testdata.h5sam")) {
        GTEST_SKIP() << "no h5sam testdata";
    }
    const std::filesystem::path out = tmp_file("samcore_conv.h5samd");
    io::convert_h5sam_to_h5samd({data_dir() / "testdata.h5sam"}, out);
    auto ds = sam_dataset::load(out);
    EXPECT_GT(ds.num_samples(), 0);
    EXPECT_EQ(ds.cube_shapes().size(), 1);
    std::filesystem::remove(out);
}

TEST(sam_dataset, SupervisedWithUnlabeledCubeThrows) {
    auto scan1 = make_cube(2, 2, 64, 0, 1.0);
    scan1.set_labels({0, 1, 0, 1}, {"healthy", "defect"});
    auto scan2 = make_cube(2, 2, 64, 0, 1.0); // unlabeled
    EXPECT_THROW(sam_dataset({scan1, scan2}, 0.0f, false), std::invalid_argument);
}

TEST(sam_dataset, Padding) {
    sam_dataset ds({make_cube(2, 2, 100, 0, 1.0), make_cube(2, 2, 50, 0, 1.0)},
                   0.0f, true);
    EXPECT_EQ(ds.maxlen(), 100);
    // second cube's rows are padded with 0 after sample 50
    EXPECT_FLOAT_EQ(ds.X()[4][99], 0.0f);
    EXPECT_NE(ds.X()[4][0], 0.0f);
}

TEST(sam_dataset, SpatialProvenance) {
    sam_dataset ds({make_cube(2, 3, 64, 0, 1000.0)}, 0.0f, true);
    auto sp = ds.spatial();
    ASSERT_EQ(sp.size(), 6);
    EXPECT_EQ(sp[0].idx, 0);
    EXPECT_FLOAT_EQ(sp[0].x, 0.0f);
    // shape (2 lines, 3 cols); res 1000 um/px = 1 mm/px
    EXPECT_FLOAT_EQ(sp[2].x, 2.0f);
    EXPECT_FLOAT_EQ(sp[3].x, 0.0f); // line 1, col 0
    EXPECT_FLOAT_EQ(sp[3].y, 1.0f);
    EXPECT_FLOAT_EQ(sp[4].y, 1.0f);
}

TEST(sam_dataset, CubeExtraction) {
    sam_dataset ds({make_cube(2, 3, 64, 0, 1.0)}, 0.0f, true);
    auto cube = ds.get_cube_X(0);
    EXPECT_EQ(cube.size0(), 2);
    EXPECT_EQ(cube.size1(), 3);
    EXPECT_EQ(cube.size2(), 64);
    EXPECT_FLOAT_EQ(cube.flat()[0], ds.X()[0][0]);
}

TEST(sam_dataset, CubeLabels) {
    auto scan = make_cube(2, 2, 64, 0, 1.0);
    scan.set_labels({0, 1, 2, 1}, {"healthy", "defect", "other"});
    sam_dataset ds({scan});
    auto grid = ds.get_cube_labels(0);
    EXPECT_EQ(grid[0][0], 0);
    EXPECT_EQ(grid[0][1], 1);
    EXPECT_EQ(grid[1][0], 2);
    EXPECT_EQ(grid[1][1], 1);
}

TEST(sam_dataset, PreprocessStrategies) {
    auto scan = make_cube(4, 4, 64, 0, 1.0);
    sam_dataset ds({scan}, 0.0f, true);
    preprocess_args args;
    args.cutoff = 10.0;
    args.fs = 100.0;
    ds.preprocess("lp", args);
    EXPECT_EQ(ds.X().rows(), 16);
    EXPECT_EQ(ds.X().cols(), 64);

    ds.preprocess("normalize", args);
    ds.preprocess("medfilt", args);
    EXPECT_THROW(ds.preprocess("bogus", args), std::invalid_argument);
}

TEST(sam_dataset, ZAccessors) {
    sam_dataset ds({make_cube(2, 2, 64, 0, 1.0)}, 0.0f, true);
    EXPECT_FALSE(ds.Z().has_value());
    ds.Z() = samcore::array2d<float>(4, 3, 1.0f);
    EXPECT_EQ(ds.num_features(), 3);
    auto zcube = ds.get_cube_Z(0);
    EXPECT_EQ(zcube.size0(), 2);
    EXPECT_EQ(zcube.size1(), 2);
    EXPECT_EQ(zcube.size2(), 3);
}

TEST(sam_dataset, UnsupervisedLabelGuards) {
    sam_dataset ds({make_cube(2, 2, 64, 0, 1.0)}, 0.0f, true);
    EXPECT_THROW((void)ds.num_classes(), std::runtime_error);
    EXPECT_THROW((void)ds.class_distribution(), std::runtime_error);
    EXPECT_THROW((void)ds.get_cube_labels(0), std::runtime_error);
}

TEST(sam_dataset, CopyIsDeep) {
    sam_dataset ds({make_cube(2, 2, 64, 0, 1.0)}, 0.0f, true);
    auto c = ds.copy();
    EXPECT_EQ(c.X(), ds.X());
    EXPECT_EQ(c.cube_shapes(), ds.cube_shapes());
    EXPECT_EQ(c.cube_resolutions(), ds.cube_resolutions());
    EXPECT_EQ(c.scanlens(), ds.scanlens());
    EXPECT_EQ(c.unsupervised(), ds.unsupervised());
    EXPECT_EQ(c.train_indices(), ds.train_indices());
    EXPECT_EQ(c.test_indices(), ds.test_indices());
    // mutations of the copy do not affect the original
    c.X()[0][0] = 12345.0f;
    EXPECT_NE(c.X()[0][0], ds.X()[0][0]);
}

TEST(sam_dataset, CopyPreservesZAndLabels) {
    auto scan = make_cube(2, 2, 64, 0, 1.0);
    scan.set_labels({0, 1, 0, 1}, {"healthy", "defect"});
    sam_dataset ds({scan});
    ds.Z() = samcore::array2d<float>(4, 3, 1.0f);
    auto c = ds.copy();
    ASSERT_TRUE(c.labels().has_value());
    EXPECT_EQ(c.labels()->labels(), ds.labels()->labels());
    EXPECT_EQ(c.labels()->label_names(), ds.labels()->label_names());
    ASSERT_TRUE(c.Z().has_value());
    EXPECT_EQ(*c.Z(), *ds.Z());
}

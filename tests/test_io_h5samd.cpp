#include <gtest/gtest.h>

#include <filesystem>

#include <H5Cpp.h>

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

samcore::sam_scan make_shifted_cube(
    const std::vector<std::vector<std::int8_t>>& rows,
    std::int64_t scanlen, double samplerate = 100.0) {
    samcore::sam_header header(1, static_cast<std::int64_t>(rows.size()),
                               scanlen, samplerate, 0, 1.0);
    samcore::array2d<std::int8_t> data(rows.size(),
                                       static_cast<size_t>(scanlen));
    for (size_t i = 0; i < rows.size(); ++i) {
        std::copy(rows[i].begin(), rows[i].end(), data[i].begin());
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
    EXPECT_EQ(ds.merge(), merge_mode::pad); // legacy files were padded
}

TEST(io_h5samd, RoundTrip) {
    sam_dataset ds({make_cube(3, 4, 100, 0, 1.0), make_cube(2, 5, 80, 7, 2.0)},
                   0.0f, true);
    EXPECT_EQ(ds.num_samples(), 3 * 4 + 2 * 5);
    EXPECT_EQ(ds.maxlen(), 80); // default merge crops to the shortest scan
    EXPECT_EQ(ds.merge(), merge_mode::crop);
    EXPECT_TRUE(ds.unsupervised());

    const std::filesystem::path out = tmp_file("samcore_ds.h5samd");
    ds.save(out);
    auto ds2 = sam_dataset::load(out);
    EXPECT_EQ(ds2.X(), ds.X());
    EXPECT_EQ(ds2.cube_shapes(), ds.cube_shapes());
    EXPECT_EQ(ds2.cube_resolutions(), ds.cube_resolutions());
    EXPECT_EQ(ds2.scanlens(), ds.scanlens());
    EXPECT_EQ(ds2.valid_scanlens(), ds.valid_scanlens());
    EXPECT_EQ(ds2.cube_samplerates(), ds.cube_samplerates());
    EXPECT_EQ(ds2.merge(), ds.merge());
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
    dataset_merge_options options;
    options.mode = merge_mode::pad;
    sam_dataset ds({make_cube(2, 2, 100, 0, 1.0), make_cube(2, 2, 50, 0, 1.0)},
                   options, true);
    EXPECT_EQ(ds.maxlen(), 100);
    // second cube's rows are padded with 0 after sample 50
    EXPECT_FLOAT_EQ(ds.X()[4][99], 0.0f);
    EXPECT_NE(ds.X()[4][0], 0.0f);
    EXPECT_EQ(ds.valid_scanlens(), (std::vector<std::int32_t>{100, 50}));
}

TEST(sam_dataset, DefaultCrop) {
    sam_dataset ds({make_cube(2, 2, 100, 0, 1.0), make_cube(2, 2, 50, 0, 1.0)},
                   0.0f, true);
    EXPECT_EQ(ds.maxlen(), 50);
    EXPECT_EQ(ds.merge(), merge_mode::crop);
    EXPECT_EQ(ds.valid_scanlens(), (std::vector<std::int32_t>{50, 50}));
}

TEST(sam_dataset, MergeZgate) {
    dataset_merge_options options;
    options.mode = merge_mode::zgate;
    options.threshold = 0.5;
    options.zgate_length = 20;
    sam_dataset ds({make_cube(2, 2, 100, 0, 1.0), make_cube(2, 2, 60, 7, 1.0)},
                   options, true);
    EXPECT_EQ(ds.maxlen(), 20);
    EXPECT_EQ(ds.valid_scanlens(), (std::vector<std::int32_t>{20, 20}));
    EXPECT_EQ(ds.merge(), merge_mode::zgate);
}

TEST(sam_dataset, MergeXcorrAlignsAcrossCubes) {
    constexpr std::int64_t n = 200;
    std::vector<std::int8_t> tmpl(n);
    for (std::int64_t i = 0; i < n; ++i) {
        tmpl[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 37 + 11) % 200) - 100);
    }
    std::vector<std::int8_t> delayed(n, 0), early(n, 0);
    for (std::int64_t j = 7; j < n; ++j) delayed[j] = tmpl[j - 7];
    for (std::int64_t j = 0; j + 5 < n; ++j) early[j] = tmpl[j + 5];

    auto cube0 = make_shifted_cube({tmpl, delayed}, n);
    auto cube1 = make_shifted_cube({early}, n);
    dataset_merge_options options;
    options.mode = merge_mode::xcorr;
    options.max_shift = 30;
    sam_dataset ds({cube0, cube1}, options, true);
    EXPECT_LT(ds.maxlen(), 200u);
    EXPECT_GT(ds.maxlen(), 180u);
    for (std::size_t r = 0; r < ds.num_samples(); ++r) {
        for (std::size_t j = 20; j + 20 < ds.maxlen(); ++j) {
            EXPECT_FLOAT_EQ(ds.X()[r][j], ds.X()[0][j])
                << "row " << r << " sample " << j;
        }
    }
}

TEST(sam_dataset, WholeDatasetAlignment) {
    constexpr std::int64_t n = 200;
    std::vector<std::int8_t> tmpl(n);
    for (std::int64_t i = 0; i < n; ++i) {
        tmpl[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 41 + 3) % 180) - 90);
    }
    std::vector<std::int8_t> delayed(n, 0);
    for (std::int64_t j = 9; j < n; ++j) delayed[j] = tmpl[j - 9];
    auto scan = make_shifted_cube({tmpl, delayed}, n);
    sam_dataset ds({scan}, 0.0f, true);
    auto aligned = ds.copy();
    aligned.align_xcorr(0, 30);
    EXPECT_EQ(aligned.maxlen(), 191);
    EXPECT_EQ(ds.maxlen(), 200);
    for (std::size_t j = 20; j + 20 < aligned.maxlen(); ++j) {
        EXPECT_FLOAT_EQ(aligned.X()[1][j], aligned.X()[0][j]);
    }

    std::vector<std::int32_t> starts{0, 5};
    auto manual = ds.copy();
    manual.align_manual(starts, 100);
    EXPECT_EQ(manual.maxlen(), 100);
    EXPECT_FLOAT_EQ(manual.X()[1][0], ds.X()[1][5]);

    auto zg = ds.copy();
    zg.align_zgate(0.5, 20);
    EXPECT_EQ(zg.maxlen(), 20);
}

TEST(sam_dataset, SpectralMethodsMatchCubes) {
    auto scan0 = make_shifted_cube(
        std::vector<std::vector<std::int8_t>>(
            2, std::vector<std::int8_t>(128, 10)),
        128);
    auto scan1 = make_shifted_cube(
        std::vector<std::vector<std::int8_t>>(
            2, std::vector<std::int8_t>(128, -5)),
        128);
    for (size_t j = 0; j < 128; ++j) {
        scan0.data()[0][j] = static_cast<std::int8_t>((j * 3) % 100 - 50);
        scan0.data()[1][j] = scan0.data()[0][j];
        scan1.data()[0][j] = static_cast<std::int8_t>((j * 5) % 100 - 50);
        scan1.data()[1][j] = scan1.data()[0][j];
    }
    sam_dataset ds({scan0, scan1}, 0.0f, true);
    ASSERT_TRUE(ds.samplerate().has_value());
    EXPECT_DOUBLE_EQ(*ds.samplerate(), 100.0);

    const auto p = ds.psd(32, 16);
    EXPECT_EQ(p.psd.rows(), 4);
    const auto parts = ds.psd_by_cube(32, 16);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].psd.rows(), 2u);
    EXPECT_EQ(parts[1].psd.rows(), 2u);

    const auto s = ds.spectrum();
    EXPECT_EQ(s.mag.rows(), 4);
    EXPECT_EQ(s.mag.cols(), 65);
    const auto spec_parts = ds.spectrum_by_cube();
    EXPECT_EQ(spec_parts.size(), 2u);

    const auto peaks = ds.stft_peak_frequency(32, 16);
    EXPECT_EQ(peaks.size(), 4u);

    auto scan_psd = scan0.psd(32, 16);
    for (size_t r = 0; r < 2; ++r) {
        ASSERT_EQ(p.psd.cols(), scan_psd.psd.cols());
        for (size_t k = 0; k < p.psd.cols(); ++k) {
            EXPECT_FLOAT_EQ(p.psd[r][k], scan_psd.psd[r][k]);
        }
    }
}

TEST(sam_dataset, SpatialProvenance) {
    sam_dataset ds({make_cube(2, 3, 64, 0, 1000.0)}, 0.0f, true);
    auto sp = ds.spatial();
    ASSERT_EQ(sp.size(), 6);
    EXPECT_EQ(sp[0].idx, 0);
    EXPECT_FLOAT_EQ(sp[0].x, 0.0f);
    // shape (2 lines, 3 cols); res 1000 µm/px = 1 mm/px
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

TEST(io_h5samd, LazyLoad) {
    sam_dataset ds({make_cube(2, 3, 8, 0, 1.0), make_cube(2, 3, 6, 3, 2.0)},
                   0.0f, true);
    ds.Z() = array2d<float>(ds.num_samples(), 2, 1.0f);
    ds.V() = array2d<float>(ds.num_samples(), 4, 2.0f);

    const std::filesystem::path out = tmp_file("samcore_ds_lazy.h5samd");
    ds.save(out);

    sam_dataset lazy = sam_dataset::load(out, true);
    EXPECT_FALSE(lazy.loaded());
    EXPECT_FALSE(lazy.materialized());
    EXPECT_EQ(lazy.backing(), "lazy");
    EXPECT_EQ(lazy.num_samples(), ds.num_samples());
    EXPECT_EQ(lazy.maxlen(), ds.maxlen());
    EXPECT_EQ(lazy.num_features(), ds.num_features());
    EXPECT_TRUE(lazy.has_z());
    EXPECT_TRUE(lazy.has_v());
    EXPECT_EQ(lazy.cube_shapes(), ds.cube_shapes());
    EXPECT_EQ(lazy.cube_resolutions(), ds.cube_resolutions());
    EXPECT_EQ(lazy.scanlens(), ds.scanlens());
    EXPECT_TRUE(lazy.unsupervised());
    EXPECT_EQ(lazy.train_indices().size(), ds.num_samples());

    // metadata-only accessors do not materialize
    auto sp = lazy.spatial();
    EXPECT_EQ(sp.size(), ds.num_samples());
    EXPECT_FALSE(lazy.loaded());

    // accessing data materializes it
    EXPECT_EQ(lazy.X(), ds.X());
    EXPECT_TRUE(lazy.loaded());
    EXPECT_EQ(lazy.backing(), "eager");
    ASSERT_TRUE(lazy.Z().has_value());
    EXPECT_EQ(*lazy.Z(), *ds.Z());
    ASSERT_TRUE(lazy.V().has_value());
    EXPECT_EQ(*lazy.V(), *ds.V());
    std::filesystem::remove(out);
}

TEST(io_h5samd, LazyPagedRowReads) {
    sam_dataset ds({make_cube(2, 3, 8, 0, 1.0), make_cube(2, 3, 6, 3, 2.0)},
                   0.0f, true);
    ds.Z() = array2d<float>(ds.num_samples(), 2, 1.0f);
    const std::filesystem::path out = tmp_file("samcore_ds_lazy_rows.h5samd");
    ds.save(out);

    sam_dataset lazy = sam_dataset::load(out, true);
    ASSERT_FALSE(lazy.loaded());
    EXPECT_EQ(lazy.blocks_read(), 0u);

    const std::vector<std::int64_t> idx{5, 0, 3, 5, 1};
    auto x = lazy.read_rows(idx);
    ASSERT_EQ(x.rows(), idx.size());
    EXPECT_EQ(x.cols(), ds.maxlen());
    for (size_t i = 0; i < idx.size(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            EXPECT_FLOAT_EQ(x[i][j], ds.X()[static_cast<size_t>(idx[i])][j]);
        }
    }
    EXPECT_FALSE(lazy.loaded());
    EXPECT_GE(lazy.blocks_read(), 1u);

    // Repeated reads reuse decoded blocks.
    const size_t after_first = lazy.blocks_read();
    (void)lazy.read_rows(idx);
    EXPECT_EQ(lazy.blocks_read(), after_first);

    // Z rows are paged too.
    auto z = lazy.read_rows(idx, /*use_z=*/true);
    ASSERT_EQ(z.rows(), idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        for (size_t j = 0; j < z.cols(); ++j) {
            EXPECT_FLOAT_EQ(z[i][j], (*ds.Z())[static_cast<size_t>(idx[i])][j]);
        }
    }
    EXPECT_FALSE(lazy.loaded());

    // Bounds and availability checks.
    EXPECT_THROW(lazy.read_rows({static_cast<std::int64_t>(lazy.num_samples())}),
                 std::out_of_range);

    sam_dataset no_z({make_cube(2, 2, 4, 0, 1.0)}, 0.0f, true);
    const std::filesystem::path out2 = tmp_file("samcore_ds_no_z.h5samd");
    no_z.save(out2);
    sam_dataset lazy2 = sam_dataset::load(out2, true);
    EXPECT_THROW(lazy2.read_rows({0}, /*use_z=*/true), std::runtime_error);
    std::filesystem::remove(out);
    std::filesystem::remove(out2);
}

TEST(io_h5samd, LazyMaterializesOnCopyAndSave) {
    sam_dataset ds({make_cube(2, 2, 8, 0, 1.0)}, 0.0f, true);
    const std::filesystem::path out = tmp_file("samcore_ds_lazy_copy.h5samd");
    ds.save(out);

    sam_dataset lazy = sam_dataset::load(out, true);
    ASSERT_FALSE(lazy.loaded());
    sam_dataset deep = lazy.copy();
    EXPECT_TRUE(lazy.loaded()); // the source was materialized by the copy
    EXPECT_TRUE(deep.loaded());
    EXPECT_EQ(deep.X(), ds.X());

    sam_dataset lazy2 = sam_dataset::load(out, true);
    const std::filesystem::path out2 = tmp_file("samcore_ds_lazy_rt.h5samd");
    lazy2.save(out2);
    EXPECT_TRUE(lazy2.loaded());
    auto reloaded = sam_dataset::load(out2);
    EXPECT_EQ(reloaded.X(), ds.X());
    std::filesystem::remove(out);
    std::filesystem::remove(out2);
}

TEST(sam_dataset, ViewInputIsMaterializedOnCopyAndSaved) {
    sam_dataset ds({make_cube(2, 3, 8, 0, 1.0)}, 0.0f, true);
    std::vector<float> ext(ds.num_samples() * ds.maxlen());
    for (size_t i = 0; i < ext.size(); ++i) {
        ext[i] = static_cast<float>(i + 1);
    }
    ds.X() = array2d<float>(ext.data(), ds.num_samples(), ds.maxlen(),
                            non_owning);
    ASSERT_TRUE(ds.X().is_view());

    sam_dataset ds2 = ds.copy(); // documented deep copy
    EXPECT_FALSE(ds2.X().is_view());

    ext[0] = 123.0f;
    EXPECT_FLOAT_EQ(ds2.X()[0][0], 1.0f); // copy did not alias

    // Cube extraction and save read through the view correctly.
    auto cube = ds.get_cube_X(0);
    EXPECT_FLOAT_EQ(cube.flat()[0], 123.0f);

    const std::filesystem::path out = tmp_file("samcore_ds_view.h5samd");
    ds.save(out);
    auto loaded = sam_dataset::load(out);
    EXPECT_FLOAT_EQ(loaded.X()[0][0], 123.0f);
    std::filesystem::remove(out);
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

TEST(io_h5samd, FormatVersionAttributes) {
    sam_dataset ds({make_cube(2, 2, 8, 0, 1.0)}, 0.0f, true);
    const std::filesystem::path out = tmp_file("samcore_ds_version.h5samd");
    ds.save(out);

    {
        H5::H5File f(out.string(), H5F_ACC_RDONLY);
        ASSERT_TRUE(f.attrExists("samcore_format_version"));
        std::int64_t v = 0;
        f.openAttribute("samcore_format_version")
            .read(H5::PredType::NATIVE_INT64, &v);
        EXPECT_EQ(v, 1);
        EXPECT_TRUE(f.attrExists("samcore_version"));
    }

    // A file from a newer library warns but still loads (eager and lazy).
    {
        H5::H5File rw(out.string(), H5F_ACC_RDWR);
        H5::Attribute a = rw.openAttribute("samcore_format_version");
        const std::int64_t future = 999;
        a.write(H5::PredType::NATIVE_INT64, &future);
    }
    auto loaded = sam_dataset::load(out);
    EXPECT_EQ(loaded.X(), ds.X());
    auto lazy = sam_dataset::load(out, true);
    EXPECT_EQ(lazy.X(), ds.X());
    std::filesystem::remove(out);
}

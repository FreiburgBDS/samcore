import os

import numpy as np
import pytest

from samcore import SAMDataset, SAMHeader, SAMLabels, SAMScan

from conftest import H5_PATH, H5SAMD_PATH, needs_data, needs_h5samd


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_header(n_signals, scanlen, cols=1):
    """Build a minimal SAMHeader for testing (samcore has no version field)."""
    return SAMHeader(scanspline=cols, nlines=n_signals // cols,
                     scanlen=scanlen, samplerate=1000.0, tzero=0,
                     resolution=1.0)


def _make_handler(n_signals=10, scanlen=100, cols=1, labels=None,
                  label_names=None):
    """Build a SAMScan handler for testing using handler_from_data."""
    data = np.random.default_rng(42).integers(
        -128, 127, size=(n_signals, scanlen)).astype(np.int8)
    if labels is None:
        labels = np.zeros(n_signals, dtype=np.int8)
    if label_names is None:
        label_names = ["healthy"]
    samlabels = SAMLabels(labels, label_names)
    header = _make_header(n_signals, scanlen, cols)
    return SAMScan.handler_from_data(data, header, samlabels=samlabels)


def _no_label_handler(n_signals=5, scanlen=50, cols=1):
    data = np.random.default_rng(0).integers(
        -128, 127, size=(n_signals, scanlen)).astype(np.int8)
    header = _make_header(n_signals, scanlen, cols)
    return SAMScan.handler_from_data(data, header)


@pytest.fixture
def single_handler():
    return [_make_handler(n_signals=20, scanlen=100, cols=1)]


@pytest.fixture
def labeled_handler():
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1, 0, 1,
                       0, 0, 1, 1, 2, 2, -1, -1, 0, 1], dtype=np.int8)
    return [_make_handler(n_signals=20, scanlen=100, cols=1,
                          labels=labels,
                          label_names=["healthy", "Defect_A", "Defect_B"])]


@pytest.fixture
def multi_handlers():
    labels1 = np.array([0, 0, 1, 1, 2, 2, -1, -1, 0, 1,
                        0, 0, 1, 1, 2, 2, -1, -1, 0, 1], dtype=np.int8)
    labels2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 3, 3, 3],
                       dtype=np.int8)
    samlabels1 = SAMLabels(labels1, ["healthy", "Defect_A", "Defect_B"])
    samlabels2 = SAMLabels(labels2, ["healthy", "Defect_A", "Defect_B",
                                     "Defect_C"])
    header1 = _make_header(20, 100, cols=1)
    header2 = _make_header(15, 100, cols=1)
    h1 = SAMScan.handler_from_data(
        np.random.default_rng(42).integers(-128, 127, size=(20, 100)).astype(np.int8),
        header1, samlabels=samlabels1)
    h2 = SAMScan.handler_from_data(
        np.random.default_rng(43).integers(-128, 127, size=(15, 100)).astype(np.int8),
        header2, samlabels=samlabels2)
    return [h1, h2]


# ── construction ─────────────────────────────────────────────────────────────

class TestConstruction:
    def test_single_handler(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        assert ds.X.shape == (20, 100)
        assert ds.X.dtype == np.float32
        assert ds.Z is None
        assert ds.spatial.shape == (20,)
        assert ds.spatial.dtype.names == ("idx", "x", "y")
        assert np.all(ds.spatial["idx"] == 0)
        assert ds.spatial[0].idx == 0
        assert ds.num_samples == 20

    def test_multi_handler(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert ds.X.shape == (35, 100)
        assert ds.X.dtype == np.float32
        assert ds.num_samples == 35

    def test_spatial_multi(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert ds.spatial.shape == (35,)
        assert np.sum(ds.spatial["idx"] == 0) == 20
        assert np.sum(ds.spatial["idx"] == 1) == 15

    def test_spatial_grid(self):
        nlines, cols = 5, 4
        data = np.random.default_rng(99).integers(
            -128, 127, size=(nlines * cols, 100)).astype(np.int8)
        header = _make_header(nlines * cols, 100, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler])
        assert ds.spatial.shape == (20,)
        np.testing.assert_array_equal(ds.spatial["idx"], 0)
        res_mm = np.float32(1.0 / 1000.0)
        np.testing.assert_array_equal(
            ds.spatial["x"], np.tile(np.arange(cols, dtype=np.float32) * res_mm,
                                     nlines))
        np.testing.assert_array_equal(
            ds.spatial["y"], np.repeat(np.arange(nlines, dtype=np.float32) * res_mm,
                                       cols))
        assert ds.spatial[3].x == 3 * res_mm
        assert ds.spatial[3].y == 0.0
        assert ds.spatial[4].x == 0.0
        assert ds.spatial[4].y == 1 * res_mm

    def test_empty_handlers_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SAMDataset([])

    def test_padding_applied(self):
        d1 = np.random.default_rng(0).integers(-128, 127, size=(5, 50)).astype(np.int8)
        d2 = np.random.default_rng(1).integers(-128, 127, size=(5, 100)).astype(np.int8)
        h1 = SAMScan.handler_from_data(d1, _make_header(5, 50))
        h2 = SAMScan.handler_from_data(d2, _make_header(5, 100))
        ds = SAMDataset([h1, h2], pad_value=-1.0)
        assert ds.X.shape == (10, 100)
        assert np.all(ds.X[:5, 50:] == -1.0)

    def test_float32_enforced(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        assert ds.X.dtype == np.float32

    def test_default_cols(self):
        data = np.random.default_rng(0).integers(-128, 127, size=(5, 50)).astype(np.int8)
        header = _make_header(5, 50)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler])
        assert ds.spatial.shape == (5,)
        res_mm = np.float32(1.0 / 1000.0)
        np.testing.assert_array_equal(ds.spatial["x"], np.zeros(5))
        np.testing.assert_array_almost_equal(ds.spatial["y"], np.arange(5) * res_mm)

    def test_scalar_cols_broadcast(self):
        d1 = np.random.default_rng(0).integers(-128, 127, size=(6, 50)).astype(np.int8)
        d2 = np.random.default_rng(1).integers(-128, 127, size=(9, 50)).astype(np.int8)
        h1 = SAMScan.handler_from_data(d1, _make_header(6, 50, 3))
        h2 = SAMScan.handler_from_data(d2, _make_header(9, 50, 3))
        ds = SAMDataset([h1, h2])
        res_mm = np.float32(1.0 / 1000.0)
        np.testing.assert_array_equal(
            ds.spatial[:6]["x"], ((np.arange(6, dtype=np.float32) % 3)) * res_mm)
        np.testing.assert_array_equal(
            ds.spatial[6:]["x"], ((np.arange(9, dtype=np.float32) % 3)) * res_mm)

    def test_no_labels_default(self):
        handler = _no_label_handler()
        ds = SAMDataset([handler])
        assert ds.unsupervised is True
        assert ds.labels is None

    def test_copy_is_deep(self, single_handler):
        ds = SAMDataset(single_handler)
        c = ds.copy()
        assert c.X.shape == ds.X.shape
        assert c.unsupervised == ds.unsupervised
        assert c.pad_value == ds.pad_value
        assert c.cube_shapes == ds.cube_shapes
        assert c.cube_resolutions == ds.cube_resolutions


# ── preprocess ───────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_lp(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("lp", cutoff=1e6, fs=25e6)
        assert ds.X.shape == (20, 100)
        assert ds.X.dtype == np.float32

    def test_bp(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("bp", cutoff_low=5e4, cutoff_high=5e5, fs=25e6)
        assert ds.X.shape == (20, 100)

    def test_normalize_max(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("normalize", mode="max")
        assert ds.X.shape == (20, 100)
        assert ds.X.dtype == np.float32
        assert np.all(np.max(np.abs(ds.X), axis=1) <= 1.0)

    def test_normalize_zscore(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("normalize", mode="zscore")
        assert ds.X.shape == (20, 100)
        means = np.mean(ds.X, axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-6)

    def test_normalize_minmax(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("normalize", mode="minmax")
        assert ds.X.shape == (20, 100)
        assert np.all(np.min(ds.X, axis=1) >= 0.0)
        assert np.all(np.max(ds.X, axis=1) <= 1.0)

    def test_savgol(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("savgol", window_length=5, polyorder=2)
        assert ds.X.shape == (20, 100)
        assert ds.X.dtype == np.float32

    def test_medfilt(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("medfilt", kernel_size=3)
        assert ds.X.shape == (20, 100)
        assert ds.X.dtype == np.float32

    def test_gate(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("gate", start=20, end=80)
        assert ds.X.shape == (20, 60)

    def test_detrend(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("detrend")
        assert ds.X.shape == (20, 100)

    def test_envelope(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("envelope")
        assert ds.X.shape == (20, 100)
        assert np.all(ds.X >= 0)

    def test_moving_average(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("moving_average", window=3)
        assert ds.X.shape == (20, 100)

    def test_no_preprocess(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        assert ds.X.shape == (20, 100)

    def test_chaining(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.preprocess("normalize").preprocess("envelope")
        assert ds.X.shape == (20, 100)

    def test_lp_missing_params_raises(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        with pytest.raises(ValueError, match="'cutoff' and 'fs'"):
            ds.preprocess("lp")

    def test_bp_missing_params_raises(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        with pytest.raises(ValueError, match="'cutoff_low'"):
            ds.preprocess("bp")

    def test_unknown_strategy_raises(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        with pytest.raises(ValueError, match="Unknown preprocess"):
            ds.preprocess("garbage")


# ── transform ────────────────────────────────────────────────────────────────

class TestTransform:
    def test_transform_basic(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        Z = ds.transform(lambda d: d[:, :10])
        assert Z.shape == (20, 10)
        np.testing.assert_array_equal(ds.Z, Z)
        assert ds.num_features == 10

    def test_transform_1d_output(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        Z = ds.transform(lambda d: d.mean(axis=1))
        assert Z.shape == (20, 1)
        assert ds.num_features == 1

    def test_transform_returns_float32(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        Z = ds.transform(lambda d: d[:, :5])
        assert Z.dtype == np.float32


# ── labels ───────────────────────────────────────────────────────────────────

class TestDatasetLabels:
    def test_single_handler_labels_preserved(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        np.testing.assert_array_equal(
            ds.labels.labels, handlers[0].samlabels.labels)

    def test_multi_handler_merged(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert len(ds.labels) == 35

    def test_class_distribution(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        dist = ds.class_distribution()
        assert dist["healthy"] == 6
        assert dist["Defect_A"] == 6
        assert dist["Defect_B"] == 4
        assert dist["unlabeled"] == 4

    def test_to_binary_int(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        b = ds.to_binary(1)
        np.testing.assert_array_equal(
            b, np.where(handlers[0].samlabels.labels == 1, 1,
                        np.where(handlers[0].samlabels.labels == -1, -1, 0)))

    def test_to_binary_str(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        b = ds.to_binary("Defect_B")
        np.testing.assert_array_equal(
            b, np.where(handlers[0].samlabels.labels == 2, 1,
                        np.where(handlers[0].samlabels.labels == -1, -1, 0)))


# ── splitting ────────────────────────────────────────────────────────────────

class TestSplitting:
    def test_train_test_split_ratio(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.train_test_split(test_size=0.5, shuffle=False)
        assert len(ds.train_indices) == 10
        assert len(ds.test_indices) == 10

    def test_split_by_label_int(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.split_by_label(1)
        expected_test = np.where(handlers[0].samlabels.labels == 1)[0]
        np.testing.assert_array_equal(np.sort(ds.test_indices),
                                      np.sort(expected_test))

    def test_split_by_label_str(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.split_by_label("Defect_A")
        expected_test = np.where(handlers[0].samlabels.labels == 1)[0]
        np.testing.assert_array_equal(np.sort(ds.test_indices),
                                      np.sort(expected_test))

    def test_split_by_label_case_insensitive(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.split_by_label("defect_a")
        expected_test = np.where(handlers[0].samlabels.labels == 1)[0]
        np.testing.assert_array_equal(np.sort(ds.test_indices),
                                      np.sort(expected_test))

    def test_split_by_label_with_test_size(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.split_by_label("Defect_A", test_size=0.5)
        assert len(ds.test_indices) >= 1

    def test_split_by_label_unknown_name_raises(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        with pytest.raises(ValueError, match="not found"):
            ds.split_by_label("nonexistent")

    def test_split_by_label_train_test_disjoint(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.split_by_label("Defect_A")
        assert len(np.intersect1d(ds.train_indices, ds.test_indices)) == 0
        assert len(ds.train_indices) + len(ds.test_indices) == len(ds)

    def test_split_by_label_bad_type_raises(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        with pytest.raises(TypeError):
            ds.split_by_label(1.5)


# ── batches ──────────────────────────────────────────────────────────────────

class TestBatches:
    def test_train_batches(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        batches = list(ds.batches("train", shuffle=False, batch_size=8))
        assert len(batches) == 3
        assert batches[0][0].shape == (8, 100)
        assert batches[-1][0].shape == (4, 100)

    def test_train_batches_shuffle(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        batches1 = list(ds.batches("train", shuffle=True, seed=42, batch_size=8))
        batches2 = list(ds.batches("train", shuffle=True, seed=42, batch_size=8))
        for (x1, y1, s1), (x2, y2, s2) in zip(batches1, batches2):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_iter(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        batches = list(ds.batches("train", batch_size=8))
        assert len(batches) == 3

    def test_test_batches(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.train_test_split(test_size=0.5, shuffle=False)
        batches = list(ds.batches("test", shuffle=False, batch_size=8))
        assert len(batches) == 2

    def test_num_batches_properties(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.train_test_split(test_size=0.5, shuffle=False)
        assert ds.num_train_batches(7) == 2
        assert ds.num_test_batches(7) == 2

    def test_use_z_requires_z(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        batches = list(ds.batches("train"))
        assert len(batches) > 0
        with pytest.raises(RuntimeError, match="Z has not been built"):
            list(ds.batches("train", use_z=True))

    def test_use_z_auto_detects_when_built(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.transform(lambda d: d[:, :5])
        batches = list(ds.batches("train"))
        assert batches[0][0].shape[1] == 5


# ── statistics ───────────────────────────────────────────────────────────────

class TestStatistics:
    def test_num_samples(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        assert ds.num_samples == 20

    def test_num_features_none_when_no_z(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        assert ds.num_features is None

    def test_num_classes(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        assert ds.num_classes == 3

    def test_dataset_label_names(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        assert ds.dataset_label_names == ["healthy", "Defect_A", "Defect_B"]

    def test_handler_ids(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert ds.handler_ids.shape == (35,)
        assert ds.handler_ids.dtype == np.int32
        assert set(np.unique(ds.handler_ids)) == {0, 1}

    def test_cube_counts(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert ds.cube_counts == {0: 20, 1: 15}


# ── export ───────────────────────────────────────────────────────────────────

class TestExport:
    def test_to_numpy_train(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        X, y, _ = ds.to_numpy("train")
        assert X.shape == ds.X.shape
        assert X.dtype == np.float32
        assert y.shape == (ds.num_samples,)

    def test_to_numpy_test(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.train_test_split(test_size=0.5, shuffle=False)
        X, y, _ = ds.to_numpy("test")
        assert X.shape == (10, 100)

    def test_to_numpy_use_z(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.transform(lambda d: d[:, :5])
        X, y, _ = ds.to_numpy("train", use_z=True)
        assert X.shape == (20, 5)

    def test_to_dict_keys(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        d = ds.to_dict("train")
        assert set(d.keys()) == {"data", "labels", "label_names",
                                 "handler_ids", "spatial"}
        assert d["data"].dtype == np.float32
        assert d["spatial"].shape == (20,)
        assert d["spatial"].dtype.names == ("idx", "x", "y")

    def test_to_numpy_is_copy(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        X, y, _ = ds.to_numpy("train")
        X[0, 0] = 999.0
        assert ds.X[0, 0] != 999.0

    def test_to_dict_is_copy(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        d = ds.to_dict("train")
        d["data"][0, 0] = 999.0
        assert ds.X[0, 0] != 999.0


# ── get_cube / cube_batches / spatial_patches ────────────────────────────────

class TestGetCube:
    def test_get_cube_X_shape(self):
        nlines, cols, scanlen = 5, 4, 100
        data = np.random.default_rng(0).integers(
            -128, 127, size=(nlines * cols, scanlen)).astype(np.int8)
        header = _make_header(nlines * cols, scanlen, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)

        cube = ds.get_cube_X(0)
        assert cube.shape == (nlines, cols, scanlen)

    def test_get_cube_X_multi(self, multi_handlers):
        ds = SAMDataset(multi_handlers, unsupervised=True)
        cube0 = ds.get_cube_X(0)
        cube1 = ds.get_cube_X(1)
        assert cube0.shape[0] == 20
        assert cube1.shape[0] == 15
        assert cube0.shape[2] == ds.X.shape[1]
        assert cube1.shape[2] == ds.X.shape[1]

    def test_get_cube_X_data(self):
        nlines, cols, scanlen = 3, 2, 50
        data = np.random.default_rng(42).integers(
            -128, 127, size=(nlines * cols, scanlen)).astype(np.int8)
        header = _make_header(nlines * cols, scanlen, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)

        cube = ds.get_cube_X(0)
        np.testing.assert_array_equal(cube.reshape(-1, scanlen), data)

    def test_get_cube_Z(self):
        nlines, cols, scanlen = 4, 3, 80
        data = np.random.default_rng(0).integers(
            -128, 127, size=(nlines * cols, scanlen)).astype(np.int8)
        header = _make_header(nlines * cols, scanlen, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)
        ds.transform(lambda d: d[:, :10])

        cube = ds.get_cube_Z(0)
        assert cube.shape == (nlines, cols, 10)

    def test_get_cube_V(self):
        nlines, cols, scanlen = 3, 2, 50
        data = np.random.default_rng(0).integers(
            -128, 127, size=(nlines * cols, scanlen)).astype(np.int8)
        header = _make_header(nlines * cols, scanlen, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)
        ds.V = np.random.default_rng(0).normal(
            size=(nlines * cols, 3)).astype(np.float32)

        cube = ds.get_cube_V(0)
        assert cube.shape == (nlines, cols, 3)
        np.testing.assert_array_equal(
            cube.reshape(-1, 3), ds.V)

    def test_get_cube_Z_not_built_raises(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        with pytest.raises(RuntimeError, match="Z has not been built"):
            ds.get_cube_Z(0)

    def test_get_cube_V_not_set_raises(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        with pytest.raises(RuntimeError, match="V has not been set"):
            ds.get_cube_V(0)

    def test_get_cube_bad_idx(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        with pytest.raises(IndexError):
            ds.get_cube_X(99)

    def test_cube_labels_shape(self, labeled_handler):
        nlines, cols = 20, 1
        ds = SAMDataset(labeled_handler)
        labels = ds.get_cube_labels(0)
        assert labels.shape == (nlines, cols)

    def test_cube_labels_match(self):
        nlines, cols = 3, 4
        labels_arr = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int8)
        samlabels = SAMLabels(labels_arr, ["healthy", "A", "B"])
        data = np.random.default_rng(0).integers(
            -128, 127, size=(nlines * cols, 50)).astype(np.int8)
        header = _make_header(nlines * cols, 50, cols)
        handler = SAMScan.handler_from_data(data, header, samlabels=samlabels)
        ds = SAMDataset([handler])
        cube_labels = ds.get_cube_labels(0)
        np.testing.assert_array_equal(cube_labels, labels_arr.reshape(nlines, cols))

    def test_cube_batches_unsupervised(self, multi_handlers):
        ds = SAMDataset(multi_handlers, unsupervised=True)
        batches = list(ds.cube_batches(shuffle=False))
        assert len(batches) == 2
        for batch in batches:
            X = batch[0]
            assert X.ndim == 4
            assert X.shape[0] == 1
            assert X.shape[3] == ds.X.shape[1]

    def test_cube_batches_supervised(self, multi_handlers):
        ds = SAMDataset(multi_handlers)
        batches = list(ds.cube_batches(shuffle=False))
        assert len(batches) == 2
        for X, y in batches:
            assert X.ndim == 4
            assert X.shape[0] == 1
            assert y.ndim == 3
            assert y.shape[0] == 1

    def test_cube_batches_use_z(self, multi_handlers):
        ds = SAMDataset(multi_handlers)
        ds.transform(lambda d: d[:, :10])
        for X, _ in ds.cube_batches(use_z=True, shuffle=False):
            assert X.shape[3] == 10

    def test_cube_batches_shuffle(self, multi_handlers):
        ds = SAMDataset(multi_handlers, unsupervised=True)
        b1 = [b[0].sum() for b in ds.cube_batches(shuffle=True, seed=42)]
        b2 = [b[0].sum() for b in ds.cube_batches(shuffle=True, seed=42)]
        assert b1 == b2

    def test_spatial_patches_shape(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        batches = list(ds.spatial_patches(patch_size=(5, 1), stride=(5, 1),
                                          shuffle=False))
        assert len(batches) == 4
        for Xp, in batches:
            assert Xp.shape == (5, 1, 100)
            assert Xp.dtype == np.float32

    def test_spatial_patches_supervised(self, labeled_handler):
        ds = SAMDataset(labeled_handler)
        batches = list(ds.spatial_patches(patch_size=(5, 1), stride=(5, 1),
                                          shuffle=False))
        for Xp, yp in batches:
            assert Xp.shape == (5, 1, 100)
            assert yp.shape == (5, 1)

    def test_spatial_patches_multi_cube(self, multi_handlers):
        ds = SAMDataset(multi_handlers, unsupervised=True)
        n_patches = 0
        for Xp, in ds.spatial_patches(patch_size=(4, 1), stride=(4, 1),
                                      shuffle=False):
            n_patches += 1
            assert Xp.shape[0] == 4 or Xp.shape[0] <= 4
            assert Xp.shape[1] == 1
        assert n_patches > 0

    def test_spatial_patches_stride(self):
        nlines, cols = 6, 3
        data = np.random.default_rng(0).integers(
            -128, 127, size=(nlines * cols, 50)).astype(np.int8)
        header = _make_header(nlines * cols, 50, cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)
        batches = list(ds.spatial_patches(patch_size=(4, 2), stride=(2, 1),
                                          shuffle=False))
        assert len(batches) == 4

    def test_spatial_patches_shuffle(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        s1 = sum(b[0].sum() for b in ds.spatial_patches(
            patch_size=(5, 1), stride=(5, 1), shuffle=True, seed=42))
        s2 = sum(b[0].sum() for b in ds.spatial_patches(
            patch_size=(5, 1), stride=(5, 1), shuffle=True, seed=42))
        assert s1 == s2

    def test_spatial_patches_too_large(self, single_handler):
        ds = SAMDataset(single_handler, unsupervised=True)
        batches = list(ds.spatial_patches(patch_size=(999, 999), stride=(1, 1),
                                          shuffle=False))
        assert len(batches) == 0


# ── stratified split ─────────────────────────────────────────────────────────

class TestStratifiedSplit:
    def test_preserves_label_proportions(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.stratified_train_test_split(test_size=0.5, random_state=42)
        train_labels = ds.labels.labels[ds.train_indices]
        test_labels = ds.labels.labels[ds.test_indices]
        for cls_val in [0, 1, 2]:
            total = int(np.sum(ds.labels.labels == cls_val))
            train_count = int(np.sum(train_labels == cls_val))
            test_count = int(np.sum(test_labels == cls_val))
            assert train_count + test_count == total
        assert len(ds.test_indices) > 0
        assert len(ds.train_indices) > 0

    def test_split_disjoint(self, single_handler):
        handlers = single_handler
        ds = SAMDataset(handlers)
        ds.stratified_train_test_split(test_size=0.3, random_state=42)
        assert len(np.intersect1d(ds.train_indices, ds.test_indices)) == 0
        assert len(ds.train_indices) + len(ds.test_indices) == len(ds)


# ── stratified split by cube (unsupervised-friendly) ────────────────────────

class TestStratifiedSplitByCube:
    def test_preserves_per_cube_proportions_unsupervised(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        assert ds.labels is None

        ds.stratified_split_by_cube(test_size=0.2, random_state=42)

        for ci, n_total in ds.cube_counts.items():
            n_test = int(np.sum(ds.spatial["idx"][ds.test_indices] == ci))
            n_train = int(np.sum(ds.spatial["idx"][ds.train_indices] == ci))
            assert n_test == max(1, int(n_total * 0.2))
            assert n_train + n_test == n_total

    def test_total_split_sizes(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        ds.stratified_split_by_cube(test_size=0.3, random_state=1)
        total = ds.num_samples
        assert len(ds.train_indices) + len(ds.test_indices) == total
        expected_test = sum(max(1, int(n * 0.3))
                            for n in ds.cube_counts.values())
        assert len(ds.test_indices) == expected_test

    def test_train_test_disjoint(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        ds.stratified_split_by_cube(test_size=0.5, random_state=7)
        assert len(np.intersect1d(ds.train_indices, ds.test_indices)) == 0
        assert len(ds.train_indices) + len(ds.test_indices) == len(ds)

    def test_shuffled_flag_false(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        ds.stratified_split_by_cube(test_size=0.2, random_state=42)
        assert ds.shuffled is False

    def test_reproducible_with_seed(self, multi_handlers):
        handlers = multi_handlers
        ds1 = SAMDataset(handlers, unsupervised=True)
        ds2 = SAMDataset(handlers, unsupervised=True)
        ds1.stratified_split_by_cube(test_size=0.25, random_state=99)
        ds2.stratified_split_by_cube(test_size=0.25, random_state=99)
        np.testing.assert_array_equal(ds1.train_indices, ds2.train_indices)
        np.testing.assert_array_equal(ds1.test_indices, ds2.test_indices)

    def test_different_seeds_give_different_splits(self, multi_handlers):
        handlers = multi_handlers
        ds1 = SAMDataset(handlers, unsupervised=True)
        ds2 = SAMDataset(handlers, unsupervised=True)
        ds1.stratified_split_by_cube(test_size=0.5, random_state=1)
        ds2.stratified_split_by_cube(test_size=0.5, random_state=2)
        assert not np.array_equal(ds1.test_indices, ds2.test_indices)

    def test_works_on_supervised_dataset(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers)
        assert ds.labels is not None
        ds.stratified_split_by_cube(test_size=0.5, random_state=3)
        assert len(ds.test_indices) > 0
        assert len(ds.train_indices) > 0
        for ci, n_total in ds.cube_counts.items():
            n_test = int(np.sum(ds.spatial["idx"][ds.test_indices] == ci))
            assert n_test == max(1, int(n_total * 0.5))

    def test_small_cube_guarantees_one_test_sample(self):
        h1 = _make_handler(n_signals=2, scanlen=50)
        h2 = _make_handler(n_signals=100, scanlen=50)
        ds = SAMDataset([h1, h2], unsupervised=True)
        ds.stratified_split_by_cube(test_size=0.2, random_state=0)
        n_test_cube0 = int(np.sum(ds.spatial["idx"][ds.test_indices] == 0))
        assert n_test_cube0 == 1

    def test_test_size_zero_raises(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.stratified_split_by_cube(test_size=0.0)

    def test_test_size_one_raises(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.stratified_split_by_cube(test_size=1.0)

    def test_test_size_negative_raises(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.stratified_split_by_cube(test_size=-0.1)

    def test_test_size_above_one_raises(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.stratified_split_by_cube(test_size=1.5)

    def test_batches_after_split(self, multi_handlers):
        handlers = multi_handlers
        ds = SAMDataset(handlers, unsupervised=True)
        ds.stratified_split_by_cube(test_size=0.3, random_state=42)
        train_batches = list(ds.batches("train", shuffle=False, batch_size=8))
        test_batches = list(ds.batches("test", shuffle=False, batch_size=8))
        n_train = sum(b[0].shape[0] for b in train_batches)
        n_test = sum(b[0].shape[0] for b in test_batches)
        assert n_train == len(ds.train_indices)
        assert n_test == len(ds.test_indices)


# ── to_one_hot / relabel delegation ──────────────────────────────────────────

class TestDatasetLabelDelegation:
    def test_to_one_hot(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        oh = ds.to_one_hot()
        assert oh.shape[0] == ds.num_samples
        assert oh.shape[1] == ds.num_classes

    def test_relabel(self, labeled_handler):
        handlers = labeled_handler
        ds = SAMDataset(handlers)
        ds.relabel({2: 1})
        unique = np.unique(ds.labels.labels)
        assert 2 not in unique


# ── integration: real SAM file → SAMDataset → roundtrip ─────────────────────

class TestIntegrationRealFiles:
    @needs_data
    def test_build_from_handler(self, h5):
        ds = SAMDataset([h5], unsupervised=True)
        assert ds.X.dtype == np.float32
        assert ds.num_samples == h5.data.shape[0]
        assert ds.spatial.shape == (ds.num_samples,)
        assert np.all(ds.spatial["idx"] == 0)

    @needs_data
    def test_save_load_roundtrip(self, h5, tmp_path):
        ds = SAMDataset([h5], pad_value=-1.0, unsupervised=True)
        path = str(tmp_path / "real.h5samd")
        ds.save(path)

        ds2 = SAMDataset.load(path)
        np.testing.assert_array_equal(ds.X, ds2.X)
        np.testing.assert_array_equal(ds.spatial, ds2.spatial)
        assert ds.labels is None and ds2.labels is None

    @needs_data
    def test_loaded_batches(self, h5, tmp_path):
        ds = SAMDataset([h5], unsupervised=True)
        ds.save(str(tmp_path / "batches.h5samd"))
        ds2 = SAMDataset.load(str(tmp_path / "batches.h5samd"))

        batches = list(ds2.batches("train", shuffle=False))
        total = sum(b[0].shape[0] for b in batches)
        assert total == ds2.num_samples

    @needs_data
    def test_loaded_split_and_export(self, h5, tmp_path):
        ds = SAMDataset([h5], unsupervised=True)
        ds.save(str(tmp_path / "split.h5samd"))
        ds2 = SAMDataset.load(str(tmp_path / "split.h5samd"))

        ds2.train_test_split(test_size=0.3, shuffle=False)
        X_train, s_train = ds2.to_numpy("train")
        X_test, s_test = ds2.to_numpy("test")
        assert X_train.shape[0] + X_test.shape[0] == ds2.num_samples
        assert s_train.shape[0] + s_test.shape[0] == ds2.num_samples

    @needs_data
    def test_spatial_coords_match_grid(self, h5):
        ds = SAMDataset([h5], unsupervised=True)
        mask = ds.spatial["idx"] == 0
        block_x = ds.spatial["x"][mask]
        block_y = ds.spatial["y"][mask]
        c = h5.cols
        n = int(np.sum(mask))
        res_mm = h5.header.resolution / 1000.0
        np.testing.assert_array_almost_equal(
            block_x, (np.arange(n, dtype=np.float32) % c) * res_mm)
        np.testing.assert_array_almost_equal(
            block_y, (np.arange(n, dtype=np.float32) // c) * res_mm)

    @needs_data
    def test_cube_counts_match(self, h5):
        ds = SAMDataset([h5], unsupervised=True)
        counts = ds.cube_counts
        assert counts[0] == h5.data.shape[0]


# ── h5samd file loading (deterministic) ─────────────────────────────────────

@needs_h5samd
class TestUnsupervisedFile:
    @classmethod
    def setup_class(cls):
        cls.ds = SAMDataset.load(H5SAMD_PATH)

    def test_load_unsupervised_flag(self):
        assert self.ds.unsupervised is True

    def test_load_num_samples(self):
        assert self.ds.num_samples == 1200

    def test_load_x_sum_deterministic(self):
        assert np.sum(self.ds.X) == 112734032.0

    def test_load_x_shape_and_dtype(self):
        assert self.ds.X.shape == (1200, 5000)
        assert self.ds.X.dtype == np.float32

    def test_load_cube_counts(self):
        assert self.ds.cube_counts == {0: 400, 1: 400, 2: 400}

    def test_load_handler_ids(self):
        ids = self.ds.handler_ids
        assert np.array_equal(np.unique(ids), [0, 1, 2])
        assert ids[0] == 0
        assert ids[399] == 0
        assert ids[400] == 1
        assert ids[799] == 1
        assert ids[800] == 2
        assert ids[1199] == 2

    def test_load_batches_yield_spatial(self):
        batches = list(self.ds.batches("train", shuffle=False, batch_size=16))
        assert len(batches) == 75
        first_x, first_s = batches[0]
        assert first_x.shape == (16, 5000)
        assert first_s.dtype.names == ("idx", "x", "y")
        np.testing.assert_array_equal(
            first_x[0, :5], np.array([19., 21., 17., 18., 14.], dtype=np.float32))
        assert first_s[0].idx == 0
        assert first_s[0].x == 0.0
        assert first_s[0].y == 0.0
        assert first_s[1].x == 0.4
        assert first_s[1].y == 0.0
        assert first_s[2].x == 0.8
        assert first_s[2].y == 0.0

    def test_load_last_batch_size(self):
        batches = list(self.ds.batches("train", shuffle=False, batch_size=16))
        last_x, _ = batches[-1]
        assert last_x.shape[0] == 16

    def test_load_train_test_split(self):
        ds = SAMDataset.load(H5SAMD_PATH)
        ds.train_test_split(test_size=0.3, shuffle=False)
        X_train, s_train = ds.to_numpy("train")
        X_test, s_test = ds.to_numpy("test")
        assert X_train.shape[0] == 840
        assert X_test.shape[0] == 360
        assert s_train.shape[0] == 840
        assert s_test.shape[0] == 360

    def test_load_to_dict_keys(self):
        d = self.ds.to_dict("train")
        assert set(d.keys()) == {"data", "handler_ids", "spatial"}
        assert d["data"].shape == (1200, 5000)
        assert d["spatial"].shape == (1200,)
        assert d["handler_ids"].shape == (1200,)

    def test_load_to_numpy(self):
        X, spatial = self.ds.to_numpy("train")
        assert X.shape == (1200, 5000)
        assert spatial.shape == (1200,)
        assert spatial.dtype.names == ("idx", "x", "y")

    @needs_data
    def test_load_build_matches_file(self, h5):
        built = SAMDataset([h5], unsupervised=True)
        np.testing.assert_array_equal(self.ds.X[:400], built.X)
        np.testing.assert_array_equal(self.ds.spatial[:400], built.spatial)
        assert self.ds.labels is None and built.labels is None
        assert self.ds.unsupervised == built.unsupervised


# ── unsupervised ─────────────────────────────────────────────────────────────

class TestUnsupervised:
    def test_auto_detect_unsupervised_when_all_unlabeled(self):
        handler = _no_label_handler()
        ds = SAMDataset([handler])
        assert ds.unsupervised is True

    def test_auto_detect_supervised_when_all_labeled(self):
        handler = _make_handler(n_signals=5, scanlen=50)
        ds = SAMDataset([handler])
        assert ds.unsupervised is False

    def test_auto_detect_unsupervised_when_mixed(self):
        h1 = _make_handler(n_signals=5, scanlen=50)
        data2 = np.random.default_rng(0).integers(
            -128, 127, size=(5, 50)).astype(np.int8)
        sl2 = SAMLabels.create_unlabeled(5)
        h2 = SAMScan.handler_from_data(data2, _make_header(5, 50),
                                       samlabels=sl2)
        ds = SAMDataset([h1, h2])
        assert ds.unsupervised is True
        assert ds.labels is None

    def test_explicit_supervised_with_all_labeled(self):
        handler = _make_handler(n_signals=5, scanlen=50)
        ds = SAMDataset([handler], unsupervised=False)
        assert ds.unsupervised is False

    def test_explicit_supervised_with_unlabeled_raises(self):
        sl = SAMLabels.create_unlabeled(5)
        data = np.random.default_rng(0).integers(
            -128, 127, size=(5, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(5, 50),
                                            samlabels=sl)
        with pytest.raises(ValueError, match="unlabeled but dataset is supervised"):
            SAMDataset([handler], unsupervised=False)

    def test_explicit_unsupervised_with_labeled_data_ok(self):
        handler = _make_handler(n_signals=5, scanlen=50)
        ds = SAMDataset([handler], unsupervised=True)
        assert ds.unsupervised is True

    def test_explicit_unsupervised_no_labels_ok(self):
        handler = _no_label_handler()
        ds = SAMDataset([handler], unsupervised=True)
        assert ds.unsupervised is True

    def test_unsupervised_batches_yield_spatial(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(10, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(10, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        batches = list(ds.batches("train", shuffle=False, batch_size=4))
        assert len(batches) == 3
        for x_chunk, spatial_chunk in batches:
            assert x_chunk.shape[0] == spatial_chunk.shape[0]
            assert spatial_chunk.dtype.names == ("idx", "x", "y")

    def test_unsupervised_batches_no_labels(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(10, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(10, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        for _, second in ds.batches("train", shuffle=False, batch_size=4):
            assert second.dtype.names == ("idx", "x", "y")

    def test_unsupervised_spatial_grid_correct(self):
        nlines, cols = 3, 4
        data = np.random.default_rng(99).integers(
            -128, 127, size=(nlines * cols, 100)).astype(np.int8)
        header = _make_header(nlines * cols, 100, cols=cols)
        handler = SAMScan.handler_from_data(data, header)
        ds = SAMDataset([handler], unsupervised=True)
        res_mm = np.float32(1.0 / 1000.0)
        for _, spatial_chunk in ds.batches("train", shuffle=False):
            np.testing.assert_array_equal(
                spatial_chunk["x"], np.tile(np.arange(cols, dtype=np.float32) * res_mm, nlines))
            np.testing.assert_array_equal(
                spatial_chunk["y"], np.repeat(np.arange(nlines, dtype=np.float32) * res_mm, cols))

    def test_unsupervised_to_numpy(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(6, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(6, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        X, spatial = ds.to_numpy("train")
        assert X.shape == (6, 50)
        assert spatial.dtype.names == ("idx", "x", "y")
        assert spatial.shape == (6,)

    def test_unsupervised_to_dict(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(5, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(5, 50, cols=1))
        ds = SAMDataset([handler], unsupervised=True)
        d = ds.to_dict("train")
        assert set(d.keys()) == {"data", "handler_ids", "spatial"}
        assert "labels" not in d
        assert "label_names" not in d
        assert d["spatial"].shape == (5,)

    def test_unsupervised_roundtrip(self, tmp_path):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(8, 100)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(8, 100, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        path = str(tmp_path / "unsup.h5samd")
        ds.save(path)

        ds2 = SAMDataset.load(path)
        assert ds2.unsupervised is True
        np.testing.assert_array_equal(ds.X, ds2.X)
        np.testing.assert_array_equal(ds.spatial, ds2.spatial)

        batches = list(ds2.batches("train", shuffle=False, batch_size=4))
        assert len(batches) == 2
        for x_chunk, spatial_chunk in batches:
            assert x_chunk.shape == (4, 100)
            assert spatial_chunk.dtype.names == ("idx", "x", "y")

    def test_unsupervised_iter(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(6, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(6, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        batches = list(ds.batches("train", batch_size=3))
        assert len(batches) == 2
        for _, second in batches:
            assert second.dtype.names == ("idx", "x", "y")

    def test_unsupervised_with_z(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(8, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(8, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        ds.transform(lambda d: d[:, :10])
        for x_chunk, spatial_chunk in ds.batches("train", shuffle=False,
                                                 use_z=True, batch_size=4):
            assert x_chunk.shape[1] == 10
            assert spatial_chunk.dtype.names == ("idx", "x", "y")

    def test_supervised_batches_still_yield_labels(self):
        handler = _make_handler(n_signals=6, scanlen=50, cols=2)
        ds = SAMDataset([handler])
        for _, labels, _ in ds.batches("train", shuffle=False, batch_size=3):
            assert labels.dtype == np.int8

    def test_unsupervised_multi_cube_spatial_ids(self):
        data1 = np.random.default_rng(0).integers(
            -128, 127, size=(4, 50)).astype(np.int8)
        data2 = np.random.default_rng(1).integers(
            -128, 127, size=(6, 50)).astype(np.int8)
        h1 = SAMScan.handler_from_data(data1, _make_header(4, 50, cols=2))
        h2 = SAMScan.handler_from_data(data2, _make_header(6, 50, cols=3))
        ds = SAMDataset([h1, h2], unsupervised=True)
        ids_per_batch = []
        for _, spatial_chunk in ds.batches("train", shuffle=False):
            ids_per_batch.extend(spatial_chunk["idx"].tolist())
        assert ids_per_batch[:4] == [0, 0, 0, 0]
        assert ids_per_batch[4:] == [1, 1, 1, 1, 1, 1]

    def test_unsupervised_to_numpy_test_split(self):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(10, 50)).astype(np.int8)
        handler = SAMScan.handler_from_data(data, _make_header(10, 50, cols=2))
        ds = SAMDataset([handler], unsupervised=True)
        ds.train_test_split(test_size=0.4, shuffle=False)
        X_test, spatial_test = ds.to_numpy("test")
        assert X_test.shape == (4, 50)
        assert spatial_test.shape == (4,)

    def test_labels_is_none_when_auto_unsupervised(self):
        handler = _no_label_handler()
        ds = SAMDataset([handler])
        assert ds.unsupervised is True
        assert ds.labels is None

    def test_labels_is_none_when_explicit_unsupervised(self):
        handler = _make_handler(n_signals=5, scanlen=50)
        ds = SAMDataset([handler], unsupervised=True)
        assert ds.unsupervised is True
        assert ds.labels is None

    def test_labels_stored_when_supervised(self):
        handler = _make_handler(n_signals=5, scanlen=50)
        ds = SAMDataset([handler])
        assert ds.unsupervised is False
        assert ds.labels is not None

    def test_label_methods_raise_on_unsupervised(self):
        handler = _no_label_handler()
        ds = SAMDataset([handler])

        with pytest.raises(RuntimeError, match="unsupervised"):
            _ = ds.num_classes
        with pytest.raises(RuntimeError, match="unsupervised"):
            _ = ds.dataset_label_names
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.stratified_train_test_split()
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.split_by_label(0)
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.to_one_hot()
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.to_binary(0)
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.class_distribution()
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.relabel({0: 1})
        with pytest.raises(RuntimeError, match="unsupervised"):
            ds.get_cube_labels(0)

    def test_save_load_preserves_labels_none(self, tmp_path):
        handler = _no_label_handler()
        ds = SAMDataset([handler])
        assert ds.labels is None

        path = str(tmp_path / "nolabel.h5samd")
        ds.save(path)
        ds2 = SAMDataset.load(path)
        assert ds2.unsupervised is True
        assert ds2.labels is None


# ── Streaming .h5sam → .h5samd converter ───────────────────────────────────

class TestConvertFromPaths:
    def _h5sam_path(self):
        if not os.path.exists(H5_PATH):
            pytest.skip("No H5 testdata")
        return H5_PATH

    def test_single_file(self, tmp_path):
        h5_path = self._h5sam_path()
        out = str(tmp_path / "conv.h5samd")
        SAMDataset.convert_from_paths([h5_path], out, unsupervised=True)
        ds = SAMDataset.load(out)
        assert ds.X.dtype == np.float32
        assert ds.X.shape[0] > 0
        assert ds.cube_shapes is not None

    def test_matches_inline_build(self, tmp_path):
        h5_path = self._h5sam_path()
        out = str(tmp_path / "conv_match.h5samd")
        SAMDataset.convert_from_paths([h5_path], out, unsupervised=True)
        loaded = SAMDataset.load(out)

        h5 = SAMScan(h5_path)
        built = SAMDataset([h5], unsupervised=True)

        np.testing.assert_array_equal(loaded.X, built.X)
        np.testing.assert_array_equal(loaded.spatial, built.spatial)
        assert loaded.labels is None and built.labels is None

    def test_different_scanlens_padded(self, tmp_path):
        h5_path = self._h5sam_path()
        h = SAMScan(h5_path)
        short_data = np.random.default_rng(0).integers(
            -128, 127, size=(h.data.shape[0], 50)).astype(np.int8)
        short = SAMScan.handler_from_data(
            short_data, _make_header(short_data.shape[0], 50, h.cols))
        short_path = str(tmp_path / "short.h5sam")
        short.to_h5sam(short_path)

        out = str(tmp_path / "conv_pad.h5samd")
        SAMDataset.convert_from_paths([h5_path, short_path], out,
                                      pad_value=-1.0, unsupervised=True)
        ds = SAMDataset.load(out)
        maxlen = max(h.scanlen, 50)
        assert ds.X.shape[1] == maxlen

    def test_with_labels(self, tmp_path):
        data = np.random.default_rng(0).integers(
            -128, 127, size=(10, 50)).astype(np.int8)
        header = _make_header(10, 50, cols=2)
        labels = SAMLabels(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8),
                           ["healthy", "Defect"])
        handler = SAMScan.handler_from_data(data, header, samlabels=labels)
        src = str(tmp_path / "src.h5sam")
        handler.to_h5sam(src)

        out = str(tmp_path / "conv_labels.h5samd")
        SAMDataset.convert_from_paths([src], out, unsupervised=False)
        ds = SAMDataset.load(out)
        assert ds.unsupervised is False
        assert ds.num_classes >= 0

    def test_unsupervised_auto_detect(self, tmp_path):
        h5_path = self._h5sam_path()
        out = str(tmp_path / "conv_auto.h5samd")
        SAMDataset.convert_from_paths([h5_path], out)
        ds = SAMDataset.load(out)
        assert isinstance(ds.unsupervised, bool)

    def test_empty_paths_raises(self, tmp_path):
        out = str(tmp_path / "empty.h5samd")
        with pytest.raises(ValueError, match="At least one"):
            SAMDataset.convert_from_paths([], out)

    def test_bad_output_extension_raises(self, tmp_path):
        h5_path = self._h5sam_path()
        with pytest.raises(ValueError, match="must end with .h5samd"):
            SAMDataset.convert_from_paths([h5_path],
                                          str(tmp_path / "bad.txt"))

    def test_spatial_fields(self, tmp_path):
        h5_path = self._h5sam_path()
        out = str(tmp_path / "conv_spatial.h5samd")
        SAMDataset.convert_from_paths([h5_path], out, unsupervised=True)
        ds = SAMDataset.load(out)
        assert ds.spatial.dtype.names == ("idx", "x", "y")
        assert np.all(ds.spatial["idx"] == 0)

    def test_cube_shapes_preserved(self, tmp_path):
        h5_path = self._h5sam_path()
        out = str(tmp_path / "conv_shapes.h5samd")
        SAMDataset.convert_from_paths([h5_path], out, unsupervised=True)
        ds = SAMDataset.load(out)
        assert len(ds.cube_shapes) == 1

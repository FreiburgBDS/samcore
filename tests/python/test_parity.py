"""samcore package test suite: behavioral checks against the C++ backend
plus zero-copy verification."""

import os
import sys

import numpy as np
import pytest

import samcore
from samcore import SAMDataset, SAMHeader, SAMLabels, SAMScan

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "SAMCORE_TEST_DATA_DIR", os.path.join(_REPO_ROOT, "tests", "data"))

H5 = os.path.join(DATA_DIR, "testdata.h5sam")

needs_data = pytest.mark.skipif(
    not os.path.exists(H5), reason="no SAM testdata")


@needs_data
def test_h5sam_load():
    h = SAMScan(H5)
    assert h.data.shape == (h.header.nlines * h.header.scanspline,
                            h.header.scanlen)
    assert h.data.dtype == np.int8
    assert h.samplerate > 0
    assert h.samlabels.labels.shape[0] == h.data.shape[0]
    assert h.starts is not None
    assert len(h.starts) == h.data.shape[0]


@needs_data
def test_zero_copy_data_view():
    h = SAMScan(H5)
    data = h.data
    assert data.flags["OWNDATA"] is False
    before = int(data[0, 0])
    h.data[0, 0] = 0
    assert int(data[0, 0]) == 0  # write through numpy reaches the C++ buffer
    data[0, 0] = before  # and vice versa
    assert int(h.data[0, 0]) == before


@needs_data
def test_zero_copy_survives_parent_deletion():
    h = SAMScan(H5)
    data = h.data
    del h
    assert data.shape[0] > 0
    assert np.all(np.abs(data.ravel()[:100]) <= 128)


@needs_data
def test_header_mutation_visible():
    h = SAMScan(H5)
    h.header.scanlen = 1234
    assert h.scanlen == 1234


@needs_data
def test_image_modes():
    h = SAMScan(H5)
    for mode in ("max", "absmax", "power"):
        img = h.image(mode)
        assert img.shape == h.shape
    assert h.image("max").dtype == np.int8
    assert h.image("absmax").dtype == np.int16
    assert h.image("power").dtype == np.float32
    with pytest.raises(ValueError):
        h.image("bogus")


@needs_data
def test_time_and_spacing():
    h = SAMScan(H5)
    t = h.time()
    assert len(t) == h.scanlen
    assert np.isclose(h.samplespacing, 1.0 / h.samplerate * 1e3)
    assert np.isclose(t[0], h.header.tzero)
    assert np.allclose(np.diff(t), np.diff(t)[0])


@needs_data
def test_normalized_data():
    h = SAMScan(H5)
    nd = h.normalized_data()
    assert nd.dtype == np.float32
    assert np.all(nd >= -1.0) and np.all(nd <= 127.0 / 128.0 + 1e-6)


@needs_data
def test_downsample_modes():
    h = SAMScan(H5)
    for mode in ("sample", "mean", "median", "decimate"):
        d = h.downsampled(8, mode)
        assert d.scanlen == h.scanlen // 8
        assert d.samplerate == h.samplerate / 8
        assert d.downsample_factor == 8
    h2 = SAMScan(H5)
    h2.downsample(4, "sample")  # in place
    assert h2.scanlen == h.scanlen // 4
    with pytest.raises(ValueError):
        h.downsample(0)
    with pytest.raises(ValueError):
        h.downsample(h.scanlen + 1)


@needs_data
def test_rotate_mirror_roundtrip():
    h = SAMScan(H5)
    for deg in (90, 180, 270):
        r = h.rotated(deg)
        assert r.shape == h.shape if deg == 180 else (h.cols, h.nlines)
    r4 = h.rotated(90).rotated(90).rotated(90).rotated(90)
    assert np.array_equal(r4.data, h.data)
    m = h.mirrored("x").mirrored("x")
    assert np.array_equal(m.data, h.data)
    assert np.array_equal(h.mirrored("x").mirrored("y").data,
                          h.rotated(180).data)
    with pytest.raises(ValueError):
        h.rotate(45)
    with pytest.raises(ValueError):
        h.mirror("z")


@needs_data
def test_selects():
    h = SAMScan(H5)
    sel = h.rectangle_select(0, 4, 0, 4)
    assert sel.shape == (4, 4)
    expected = [l * h.cols + c for l in range(4) for c in range(4)]
    assert np.array_equal(sel.data, h.data[expected])
    tsel = h.time_range_select(h.header.tzero + 1000.0,
                               h.header.tzero + 2000.0)
    assert tsel.scanlen < h.scanlen
    assert tsel.starts is None


@needs_data
def test_zgate():
    h = SAMScan(H5)
    g = h.zgate(0.5, 100)
    assert g.scanlen == 100
    assert g.starts is not None
    assert g.starts.dtype == np.int32
    with pytest.raises(ValueError):
        h.zgate(0.5, 0)
    with pytest.raises(ValueError):
        h.zgate(-0.1, 100)
    with pytest.raises(ValueError):
        h.zgate(0.5, h.scanlen + 1)


@needs_data
def test_stft_psd_spectrogram():
    h = SAMScan(H5)
    nframes = (h.scanlen - 256) // 128 + 1
    f, t, zxx = h.compute_stft(256, 128)
    assert zxx.shape == (h.num_scans(), 129, nframes)
    assert zxx.dtype == np.complex64
    assert f.shape == (129,)
    assert t.shape == (nframes,)
    assert np.all(np.isfinite(zxx))
    f, psd = h.psd(256, 128)
    assert psd.shape == (h.num_scans(), 129)
    f2, t2, sxx = h.power_spectrogram(256, 128)
    assert sxx.shape == (h.num_scans(), 129, nframes)
    with pytest.raises(ValueError):
        h.compute_stft(h.scanlen + 1, 64)


@needs_data
def test_labels_api():
    h = SAMScan(H5)
    assert h.labels.shape == (h.num_scans(),)
    h.set_labels(np.full(h.num_scans(), 1, dtype=np.int8),
                 ["healthy", "defect"])
    assert np.all(h.labels == 1)
    assert h.label_names == ["healthy", "defect"]
    assert h.samlabels.is_labeled()
    assert np.all(h.samlabels.mask(1))


@needs_data
def test_handler_from_data_and_copy():
    h = SAMScan(H5)
    h2 = SAMScan.handler_from_data(h.data, h.header.copy(), None,
                                   h.samlabels.copy())
    assert np.array_equal(h2.data, h.data)
    with pytest.raises(ValueError):
        SAMScan.handler_from_data(h.data[:, :10], h.header)


@needs_data
def test_mmap_lazy_load():
    h = SAMScan(H5, mmap=True)
    assert h.loaded is False
    assert h.num_scans() == h.header.nlines * h.header.scanspline
    assert len(h.samlabels.labels) == h.num_scans()
    _ = h.data  # materialize
    assert h.loaded is True
    eager = SAMScan(H5)
    assert np.array_equal(h.data, eager.data)
    # explicit load() is a no-op afterwards
    h.load()
    assert h.loaded is True


@needs_data
def test_dataset_copy():
    h1 = SAMScan(H5)
    h1.set_labels(np.where(np.arange(h1.num_scans()) % 2, 1, 0).astype(np.int8),
                  ["healthy", "defect"])
    ds = SAMDataset([h1])
    ds.Z = np.ones((h1.num_scans(), 3), dtype=np.float32)
    c = ds.copy()
    assert np.array_equal(c.X, ds.X)
    assert np.array_equal(np.asarray(c.labels.labels),
                          np.asarray(ds.labels.labels))
    assert np.array_equal(np.asarray(c.Z), np.asarray(ds.Z))
    assert np.array_equal(np.asarray(c.train_indices), np.asarray(ds.train_indices))
    c.X[0, 0] = 999.0
    assert ds.X[0, 0] != 999.0


@needs_data
def test_indexing_and_iteration():
    h = SAMScan(H5)
    n = h.num_scans()
    # integer indexing (incl. negative) matches numpy semantics
    assert np.array_equal(h[0], h.data[0])
    assert np.array_equal(h[-1], h.data[n - 1])
    assert np.array_equal(h[-3], h.data[n - 3])
    with pytest.raises(IndexError):
        h[n]
    with pytest.raises(IndexError):
        h[-(n + 1)]
    # tuple (line, col) indexing
    assert np.array_equal(h[3, 4], h.data[3 * h.cols + 4])
    with pytest.raises(IndexError):
        h[h.nlines, 0]
    with pytest.raises(IndexError):
        h[0, h.cols]
    # slice indexing
    sl = h[1:5]
    assert sl.shape == (4, h.scanlen)
    assert np.array_equal(sl, h.data[1:5])
    sl2 = h[10:2:-1]
    assert np.array_equal(sl2, h.data[10:2:-1])
    sl3 = h[::2]
    assert np.array_equal(sl3, h.data[::2])
    sl4 = h[-5:]
    assert np.array_equal(sl4, h.data[-5:])
    empty = h[5:2]
    assert empty.shape == (0, h.scanlen)
    # iteration yields the same rows as data
    rows = list(h)
    assert len(rows) == n
    assert np.array_equal(np.stack(rows), h.data)
    # sliced rows are zero-copy views of the same buffer
    sl5 = h[0:3]
    before = int(h.data[0, 0])
    sl5[0, 0] = 99
    assert int(h.data[0, 0]) == 99
    h.data[0, 0] = before


@needs_data
def test_h5sam_roundtrip(tmp_path):
    h = SAMScan(H5)
    out = str(tmp_path / "out.h5sam")
    h.to_h5sam(out)
    h2 = SAMScan(out)
    assert np.array_equal(h2.data, h.data)
    assert h2.header.scanlen == h.header.scanlen


@needs_data
def test_dataset(tmp_path):
    h1 = SAMScan(H5)
    h1.set_labels(np.where(np.arange(h1.num_scans()) % 2, 1, 0).astype(np.int8),
                  ["healthy", "defect"])
    h2 = SAMScan(H5)
    h2.set_labels(np.where(np.arange(h2.num_scans()) % 2, 1, 0).astype(np.int8),
                  ["healthy", "defect"])
    ds = SAMDataset([h1, h2])
    assert ds.num_samples == 2 * h1.num_scans()
    assert not ds.unsupervised
    assert ds.X.shape[1] == h1.scanlen
    sp = ds.spatial
    assert sp.dtype.names == ("idx", "x", "y")
    assert np.array_equal(np.unique(sp["idx"]), [0, 1])
    # splits
    ds.train_test_split(test_size=0.2, random_state=42)
    assert len(ds.train_indices) + len(ds.test_indices) == ds.num_samples
    batches = list(ds.batches(batch_size=32, shuffle=False))
    assert len(batches) == int(np.ceil(len(ds.train_indices) / 32))
    xb, yb, sb = batches[0]
    assert xb.dtype == np.float32
    assert xb.shape[0] == yb.shape[0] == sb.shape[0]
    # cubes
    cube = ds.get_cube_X(0)
    assert cube.shape == (h1.nlines, h1.cols, h1.scanlen)
    labels_grid = ds.get_cube_labels(0)
    assert labels_grid.shape == (h1.nlines, h1.cols)
    # h5samd roundtrip
    out = str(tmp_path / "ds.h5samd")
    ds.save(out)
    ds2 = SAMDataset.load(out)
    assert np.array_equal(ds2.X, ds.X)
    assert np.array_equal(np.asarray(ds2.labels.labels),
                          np.asarray(ds.labels.labels))
    assert ds2.cube_shapes == ds.cube_shapes
    # preprocessing + transform
    ds2.preprocess("normalize", mode="max")
    Z = ds2.transform(lambda x: np.max(x, axis=1))
    assert Z.shape[0] == ds2.num_samples
    assert ds2.num_features == 1


def test_header_without_version():
    hdr = SAMHeader(20, 10, 2000, 2500.0, 15000, 400.0)
    assert hdr.scanspline == 20
    assert hdr.nlines == 10
    assert hdr.scanlen == 2000
    assert hdr.samplerate == 2500.0
    j = hdr.to_json()
    assert "version" not in j


def test_labels_standalone():
    labels = SAMLabels(np.array([-1, 0, 1, 2], dtype=np.int8),
                       ["healthy", "defect_a", "defect_b"])
    assert labels.label_name(2) == "defect_a"
    assert labels.name_to_value("DEFECT_B") == 2
    assert labels.num_classes == 3
    assert np.array_equal(labels.to_binary("defect_a"),
                          np.array([-1, 0, 1, 0], dtype=np.int8))
    assert labels.healthy_mask().dtype == bool
    assert np.array_equal(labels.healthy_mask(),
                          np.array([False, True, False, False]))
    oh = labels.to_one_hot()
    assert oh.shape == (4, 3)
    merged = samcore.merge_labels(
        [SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "defect"]),
         SAMLabels(np.array([0, 1], dtype=np.int8), ["healthy", "defect"])])
    assert list(merged.label_names) == ["healthy", "defect"]
    d = labels.to_dict()
    assert list(SAMLabels.from_dict(d).label_names) == labels.label_names


def test_preprocessing_module():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 256)).astype(np.float32)
    out = samcore.preprocessing.lp(data, 10.0, 100.0)
    assert out.shape == data.shape
    assert out.dtype == np.float32
    assert samcore.preprocessing.bp(data, 5.0, 20.0, 100.0).shape == data.shape
    assert samcore.preprocessing.normalize(data, "max").shape == data.shape
    assert samcore.preprocessing.savgol(data, 7, 2).shape == data.shape
    assert samcore.preprocessing.medfilt(data, 5).shape == data.shape
    assert samcore.preprocessing.gate(data, 10, 100).shape == (8, 90)
    assert samcore.preprocessing.detrend(data).shape == data.shape
    assert samcore.preprocessing.envelope(data).shape == data.shape
    assert samcore.preprocessing.moving_average(data, 5).shape == data.shape
    with pytest.raises(ValueError):
        samcore.preprocessing.normalize(data, "bogus")


def test_utils_module():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 128)).astype(np.float32)
    k = samcore.utils.kurt(data)
    assert k.shape == (4,)
    mag, freqs = samcore.utils.fft_spec(data[0])
    assert mag.shape == (65,)
    psd = np.abs(data) ** 2
    e = samcore.utils.spectral_entropy(psd)
    assert e.shape == (4,)
    fl = samcore.utils.spectral_flatness(psd)
    assert fl.shape == (4,)
    c = samcore.utils.spectral_centroid(freqs[:65], psd[:, :65])
    assert c.shape == (4,)
    r = samcore.utils.spectral_energy_ratio(freqs[:65], psd[:, :65], 10.0)
    assert r.shape == (4,)
    t = samcore.utils.time_index(0.0, 10.0, 5)
    assert len(t) == 5


@needs_data
def test_io_functions():
    data, header, labels, starts = samcore.io.read_h5sam(H5)
    assert isinstance(labels, SAMLabels)
    assert data.shape[1] == header.scanlen
    assert data.shape[0] == header.nlines * header.scanspline

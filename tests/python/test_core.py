"""
Covers SAMScan core behaviour: integrity across loads, time index, images,
indexing, downsample, zgate, spatial/temporal selection, handler_from_data,
labels, copy semantics and header round trips.
"""

import numpy as np
import pytest

from samcore import SAMHeader, SAMLabels, SAMScan

from conftest import H5_PATH, needs_data


# ── integrity / time / images ────────────────────────────────────────────────

@needs_data
def test_integrity(h5):
    h5_again = SAMScan(H5_PATH)
    assert h5.nlines == h5_again.nlines
    assert h5.cols == h5_again.cols
    assert h5.data.shape == h5_again.data.shape
    for f_ in ("nlines", "scanspline", "samplerate", "scanlen", "tzero",
               "quality", "resolution"):
        assert getattr(h5.header, f_) == getattr(h5_again.header, f_)
    assert h5.samplespacing == h5_again.samplespacing
    np.testing.assert_array_equal(h5.data, h5_again.data)


@needs_data
def test_time_index(h5):
    time_index = h5.timescale
    assert len(time_index) == h5.scanlen
    duration = 1e3 * h5.header.scanlen / h5.header.samplerate
    assert np.isclose(time_index[0], h5.header.tzero)
    assert np.isclose(time_index[-1], h5.header.tzero + duration)
    assert np.all(np.diff(time_index) > 0)


@needs_data
def test_image(h5):
    img_max = h5.image("max")
    assert img_max.shape == (h5.nlines, h5.cols)
    np.testing.assert_array_equal(img_max.ravel(), h5.data.max(axis=-1))

    img_absmax = h5.image("absmax")
    assert img_absmax.shape == (h5.nlines, h5.cols)
    # C++ saturates abs(-128) to 127 to stay in int8
    np.testing.assert_array_equal(
        img_absmax.ravel(),
        np.minimum(np.abs(h5.data.astype(np.int64)).max(axis=-1), 127))

    img_power = h5.image("power")
    assert img_power.shape == (h5.nlines, h5.cols)
    assert np.all(img_power >= 0)
    np.testing.assert_array_equal(
        img_power.ravel(),
        np.sum(np.square(h5.data.astype(np.float32)), axis=-1))


@needs_data
def test_image_unsupported_kind(h5):
    with pytest.raises(ValueError, match="Unsupported image type"):
        h5.image("kurtosis")
    with pytest.raises(ValueError, match="Unsupported image type"):
        h5.image("tsne")


# ── indexing ────────────────────────────────────────────────────────────────

@needs_data
def test_getitem_flat(h5):
    scan = h5[0]
    assert scan.shape == (h5.scanlen,)
    np.testing.assert_array_equal(scan, h5.data[0])


@needs_data
def test_getitem_line_col(h5):
    line, col = 2, 3
    scan = h5[line, col]
    np.testing.assert_array_equal(scan, h5.data[line * h5.cols + col])


@needs_data
def test_getitem_slice(h5):
    scans = h5[5:10]
    assert scans.shape[0] == 5
    np.testing.assert_array_equal(scans, h5.data[5:10])


# ── downsample ──────────────────────────────────────────────────────────────

@needs_data
def test_downsample(h5):
    expected = h5.scanlen // 2

    handler = h5.copy()
    handler.downsample(2, mode="sample")
    assert handler.data.shape[1] == expected
    np.testing.assert_array_equal(h5.data[:, ::2], handler.data)

    handler = h5.copy()
    handler.downsample(2, mode="mean")
    assert handler.data.shape[1] == expected

    handler = h5.copy()
    handler.downsample(2, mode="decimate")
    assert handler.data.shape[1] == expected

    handler = h5.copy()
    handler.downsample(2, mode="median")
    assert handler.data.shape[1] == expected

    with pytest.raises(ValueError, match="cannot be greater than scan length"):
        h5.copy().downsample(h5.scanlen + 1, mode="sample")


@needs_data
def test_downsample_in_place(h5):
    handler = h5.copy()
    result = handler.downsample(2, mode="sample", in_place=True)
    assert result is handler
    assert handler.data.shape[1] == h5.scanlen // 2


@needs_data
def test_downsample_not_in_place(h5):
    original_shape = h5.data.shape
    result = h5.downsample(2, mode="sample", in_place=False)
    assert result is not h5
    assert h5.data.shape == original_shape
    assert result.data.shape[1] == original_shape[1] // 2
    np.testing.assert_array_equal(result.data, h5.data[:, ::2])


# ── zgate ───────────────────────────────────────────────────────────────────

@needs_data
def test_zgate(h5):
    # The sample testdata carries per-scan starts, which zgate offsets
    # against.  Clear them so this test isolates the raw crossing detection.
    h = h5.copy()
    h.starts = None
    zg = h.zgate(threshold=0.2, length=500)
    assert zg.data.shape[0] == h.data.shape[0]
    assert zg.data.shape[1] == 500
    assert zg.scanlen == 500
    assert zg.header.scanlen == 500
    assert zg.nlines == h.nlines
    assert zg.cols == h.cols

    assert zg.starts is not None
    assert len(zg.starts) == h.data.shape[0]

    max_start = h.scanlen - 500
    for i in [0, 5, 10]:
        scan = h.data[i].astype(int)
        crossings = np.where(np.abs(scan) >= 0.2 * 127)[0]
        expected_start = int(crossings[0]) if len(crossings) > 0 else -1
        if expected_start > max_start:
            expected_start = max_start

        assert zg.starts[i] == expected_start
        if expected_start >= 0:
            np.testing.assert_array_equal(
                zg.data[i], scan[expected_start:expected_start + 500])
            assert abs(int(zg.data[i, 0])) >= 0.2 * 127
        else:
            np.testing.assert_array_equal(zg.data[i], np.zeros(500, dtype=np.int8))


def test_zgate_synthetic():
    data = np.zeros((5, 1000), dtype=np.int8)
    data[:, 100] = 100
    hdr = SAMHeader(1, 5, 1000, 1000.0, 0, 1.0)
    h = SAMScan.handler_from_data(data, hdr)

    zg = h.zgate(threshold=0.2, length=300)
    assert zg.scanlen == 300
    assert zg.starts is None  # all equal, tzero adjusted
    np.testing.assert_array_equal(zg.data[:, 0], [100] * 5)


def test_zgate_no_crossing():
    data = np.zeros((3, 500), dtype=np.int8)
    hdr = SAMHeader(1, 3, 500, 1000.0, 0, 1.0)
    h = SAMScan.handler_from_data(data, hdr)

    zg = h.zgate(threshold=0.9, length=200)
    assert zg.scanlen == 200
    assert zg.starts is not None
    np.testing.assert_array_equal(zg.starts, [-1, -1, -1])
    np.testing.assert_array_equal(zg.data, np.zeros((3, 200), dtype=np.int8))


@needs_data
def test_zgate_length_too_large(h5):
    with pytest.raises(ValueError, match="cannot be greater than scan length"):
        h5.zgate(threshold=0.2, length=99999)


@needs_data
def test_zgate_already_gated(h5):
    zg = h5.zgate(threshold=0.2, length=500)
    zg2 = zg.zgate(threshold=0.2, length=200, in_place=True)
    assert zg2 is zg
    assert zg.scanlen == 200
    assert zg.starts is not None
    assert np.all(zg.starts >= -1)


@needs_data
def test_zgate_double_gate_new_handler(h5):
    zg = h5.zgate(threshold=0.2, length=500)
    zg2 = zg.zgate(threshold=0.2, length=200)
    assert zg2 is not zg
    assert zg2.scanlen == 200
    assert zg2.starts is not None


@needs_data
def test_zgate_in_place(h5):
    handler = h5.copy()
    n_signals = handler.data.shape[0]
    result = handler.zgate(threshold=0.2, length=500, in_place=True)
    assert result is handler
    assert handler.scanlen == 500
    assert handler.data.shape == (n_signals, 500)
    assert handler.starts is not None
    assert len(handler.starts) == n_signals


# ── selection ───────────────────────────────────────────────────────────────

@needs_data
def test_rectangle_select(h5):
    sel = h5.rectangle_select(0, 5, 0, 5)
    assert sel.nlines == 5
    assert sel.cols == 5
    assert sel.data.shape[0] == 25
    assert sel.data.shape[1] == h5.scanlen

    lines = np.arange(0, 5)
    cols = np.arange(0, 5)
    idx = (lines[:, None] * h5.cols + cols[None, :]).ravel()
    np.testing.assert_array_equal(sel.data, h5.data[idx])


@needs_data
def test_rectangle_select_full(h5):
    sel = h5.rectangle_select(0, h5.nlines, 0, h5.cols)
    np.testing.assert_array_equal(sel.data, h5.data)
    assert sel.nlines == h5.nlines
    assert sel.cols == h5.cols


@needs_data
def test_rectangle_select_in_place(h5):
    handler = h5.copy()
    result = handler.rectangle_select(0, 10, 0, 10, in_place=True)
    assert result is handler
    assert handler.nlines == 10
    assert handler.cols == 10
    assert handler.data.shape[0] == 100


@needs_data
def test_rectangle_select_bad_range(h5):
    with pytest.raises(ValueError, match="out of bounds"):
        h5.rectangle_select(0, 999, 0, 5)
    with pytest.raises(ValueError, match="out of bounds"):
        h5.rectangle_select(0, 5, 0, 999)


@needs_data
def test_time_range_select(h5):
    tz = h5.header.tzero
    sp = h5.samplespacing
    sel = h5.time_range_select(tz, tz + sp * 200)
    assert sel.scanlen == 200
    assert sel.data.shape[1] == 200
    np.testing.assert_array_equal(sel.data, h5.data[:, :200])


@needs_data
def test_time_range_select_mid(h5):
    tz = h5.header.tzero
    sp = h5.samplespacing
    sel = h5.time_range_select(tz + sp * 100, tz + sp * 300)
    assert sel.scanlen == 200
    assert sel.data.shape[1] == 200
    np.testing.assert_array_equal(sel.data, h5.data[:, 100:300])


@needs_data
def test_time_range_select_in_place(h5):
    handler = h5.copy()
    tz = handler.header.tzero
    sp = handler.samplespacing
    result = handler.time_range_select(tz, tz + sp * 500, in_place=True)
    assert result is handler
    assert handler.scanlen == 500
    assert handler.data.shape[1] == 500


@needs_data
def test_time_range_select_bad_range(h5):
    with pytest.raises(ValueError, match="Invalid time range"):
        h5.time_range_select(999999, 999999)


# ── handler_from_data ───────────────────────────────────────────────────────

@needs_data
def test_handler_from_data_valid(h5):
    new = SAMScan.handler_from_data(h5.data.copy(), h5.header.copy(),
                                    starts=None, samlabels=h5.samlabels.copy())
    assert new.data.shape == h5.data.shape
    assert new.header == h5.header
    assert new.nlines == h5.nlines
    assert new.cols == h5.cols
    np.testing.assert_array_equal(new.samlabels.labels, h5.samlabels.labels)


@needs_data
def test_handler_from_data_default_labels(h5):
    new = SAMScan.handler_from_data(h5.data.copy(), h5.header.copy())
    assert np.all(new.samlabels.labels == -1)


def test_handler_from_data_bad_shape_raises():
    hdr = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
    with pytest.raises(ValueError, match="Data mismatch"):
        SAMScan.handler_from_data(np.zeros((5, 50), dtype=np.int8), hdr)


# ── labels ──────────────────────────────────────────────────────────────────

@needs_data
def test_set_labels_valid(h5):
    h = h5.copy()
    n = h.data.shape[0]
    new_labels = np.zeros(n, dtype=np.int8)
    new_labels[0] = 1
    h.set_labels(new_labels, ["healthy", "Defect"])
    assert h.samlabels.labels[0] == 1
    assert h.samlabels.label_names == ["healthy", "Defect"]


@needs_data
def test_set_labels_wrong_length_raises(h5):
    h = h5.copy()
    with pytest.raises(ValueError, match="Length of labels"):
        h.set_labels(np.array([0, 1], dtype=np.int8))


@needs_data
def test_copy_is_deep(h5):
    c = h5.copy()
    assert c.data.shape == h5.data.shape
    assert c.header == h5.header
    assert c.path == h5.path
    c.data[0, 0] = 127
    assert h5.data[0, 0] != c.data[0, 0]
    c.samlabels.labels[0] = 0
    assert h5.samlabels.labels[0] != c.samlabels.labels[0]


@needs_data
def test_normalized_data(h5):
    nd = h5.normalized_data()
    assert nd.dtype == np.float32
    assert nd.shape == h5.data.shape
    assert np.all(nd >= -1.0)
    assert np.all(nd <= 127.0 / 128.0)


@needs_data
def test_header_hash(h5):
    assert h5.header_hash() == h5.copy().header_hash()


# ── SAMHeader ───────────────────────────────────────────────────────────────


class TestSAMHeader:
    def test_constructor_and_to_json(self):
        hdr = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        js = hdr.to_json()
        assert js["scanspline"] == 4
        assert js["samplerate"] == 2500
        assert js["downsample_factor"] == 1

    def test_constructor_with_optionals(self):
        hdr = SAMHeader(2, 5, 200, 1000.0, 0, 0.5, True, False,
                        "echo", "TX1", "RX1", "C001", 2)
        assert hdr.transducer_in == "TX1"
        assert hdr.transducer_through == "RX1"
        assert hdr.cellid == "C001"
        assert hdr.downsample_factor == 2

    def test_eq(self):
        h1 = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        h2 = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        assert h1 == h2

    def test_neq(self):
        h1 = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        h2 = SAMHeader(5, 10, 500, 2500.0, 100, 1.0)
        assert h1 != h2

    def test_eq_different_type(self):
        h = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        assert h != "not a header"

    def test_copy(self):
        h = SAMHeader(4, 10, 500, 2500.0, 100, 1.0, transducer_in="T1",
                      downsample_factor=3)
        c = h.copy()
        assert h == c
        assert h is not c
        c.samplerate = 9999
        assert h.samplerate != c.samplerate

    def test_time_default(self):
        h = SAMHeader(4, 10, 500, 2500.0, 0, 1.0)
        t = h.time()
        assert len(t) == 500
        assert t[0] == 0.0

    def test_time_with_bounds(self):
        h = SAMHeader(4, 10, 500, 2500.0, 100, 1.0)
        t = h.time(start=10, end=20)
        assert len(t) == 10
        assert abs(t[0] - (100 + 10 / 2500 * 1e3)) < 1e-9

    def test_hash(self):
        h1 = SAMHeader(4, 10, 500, 2500.0, 0, 1.0)
        h2 = h1.copy()
        assert hash(h1) == hash(h2)

    def test_str(self):
        h = SAMHeader(4, 10, 500, 2500.0, 0, 1.0)
        s = str(h)
        assert "samplerate" in s

    def test_extra_roundtrip(self, tmp_path):
        extra_in = {"operator": "Alice", "temperature_c": 23.5}
        hdr = SAMHeader(4, 10, 500, 2500.0, 0, 1.0, extra=extra_in)
        data = np.zeros((40, 500), dtype=np.int8)
        out = str(tmp_path / "extra.h5sam")
        samcore_io = __import__("samcore").io
        labels = SAMLabels.create_unlabeled(40)
        samcore_io.write_h5sam(out, data, hdr, labels)
        _, hdr2, _, _ = samcore_io.read_h5sam(out)
        assert hdr2.extra == extra_in

    def test_extra_roundtrip_complex(self, tmp_path):
        extra_in = {"tags": ["batch_A", "calibrated"]}
        hdr = SAMHeader(4, 10, 500, 2500.0, 0, 1.0, extra=extra_in)
        data = np.zeros((40, 500), dtype=np.int8)
        out = str(tmp_path / "extra_complex.h5sam")
        samcore_io = __import__("samcore").io
        labels = SAMLabels.create_unlabeled(40)
        samcore_io.write_h5sam(out, data, hdr, labels)
        _, hdr2, _, _ = samcore_io.read_h5sam(out)
        assert hdr2.extra == extra_in

    def test_extra_roundtrip_mixed(self, tmp_path):
        extra_in = {"operator": "Bob", "gain_db": 6.0, "notes": ["check"]}
        hdr = SAMHeader(4, 10, 500, 2500.0, 0, 1.0, extra=extra_in)
        data = np.zeros((40, 500), dtype=np.int8)
        out = str(tmp_path / "extra_mixed.h5sam")
        samcore_io = __import__("samcore").io
        labels = SAMLabels.create_unlabeled(40)
        samcore_io.write_h5sam(out, data, hdr, labels)
        _, hdr2, _, _ = samcore_io.read_h5sam(out)
        assert hdr2.extra == extra_in

    def test_extra_empty(self, tmp_path):
        hdr = SAMHeader(4, 10, 500, 2500.0, 0, 1.0)
        data = np.zeros((40, 500), dtype=np.int8)
        out = str(tmp_path / "extra_empty.h5sam")
        samcore_io = __import__("samcore").io
        labels = SAMLabels.create_unlabeled(40)
        samcore_io.write_h5sam(out, data, hdr, labels)
        _, hdr2, _, _ = samcore_io.read_h5sam(out)
        assert hdr2.extra == {}


# ── memmap (lazy loading) ───────────────────────────────────────────────────


class TestMemmap:
    @needs_data
    def test_mmap_lazy(self):
        handler = SAMScan(H5_PATH, mmap=True)
        assert handler.loaded is False
        assert handler.data.shape[0] > 0  # accessing data materializes it
        assert handler.loaded is True

    @needs_data
    def test_mmap_slicing_works(self):
        handler = SAMScan(H5_PATH, mmap=True)
        eager = SAMScan(H5_PATH)
        np.testing.assert_array_equal(handler[0], eager[0])
        np.testing.assert_array_equal(handler[:10], eager[:10])
        assert handler.loaded is True

    @needs_data
    def test_mmap_copy_is_deep(self):
        handler = SAMScan(H5_PATH, mmap=True)
        c = handler.copy()
        assert c.loaded is True  # copy materializes the data
        np.testing.assert_array_equal(c.data, handler.data)

    @needs_data
    def test_mmap_timescale(self):
        handler = SAMScan(H5_PATH, mmap=True)
        assert isinstance(handler.timescale, np.ndarray)
        assert len(handler.timescale) == handler.scanlen
        assert handler.samplespacing > 0

    @needs_data
    def test_mmap_zgate_not_in_place(self):
        handler = SAMScan(H5_PATH, mmap=True)
        result = handler.zgate(threshold=0.2, length=500, in_place=False)
        assert result is not handler
        assert isinstance(result.data, np.ndarray)

    @needs_data
    def test_mmap_rectangle_select_not_in_place(self):
        handler = SAMScan(H5_PATH, mmap=True)
        result = handler.rectangle_select(0, 10, 0, 10, in_place=False)
        assert result is not handler
        assert isinstance(result.data, np.ndarray)

    @needs_data
    def test_mmap_time_range_select_not_in_place(self):
        handler = SAMScan(H5_PATH, mmap=True)
        result = handler.time_range_select(
            handler.header.tzero,
            handler.header.tzero + handler.samplespacing * 500,
            in_place=False)
        assert result is not handler
        assert isinstance(result.data, np.ndarray)

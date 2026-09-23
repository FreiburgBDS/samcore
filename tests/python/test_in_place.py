"""``in_place`` contract coverage for the mutating transformation methods.

For every method with an ``in_place`` flag we check the full contract:

* ``in_place=True`` mutates and returns ``self``;
* ``in_place=False`` returns a new, independent object and leaves the source
  bit-identical (data, labels, starts, header);
* both paths produce identical results (data, labels, starts, shape);
* the default flag matches the documented behaviour.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from samcore import SAMHeader, SAMLabels, SAMScan

from conftest import H5_PATH, needs_data


def _make_handler(nlines: int = 3, cols: int = 4, scanlen: int = 120) -> SAMScan:
    """Small deterministic handler with labels and starts."""
    rng = np.random.default_rng(1234)
    n = nlines * cols
    data = rng.integers(-100, 100, size=(n, scanlen)).astype(np.int8)
    data[:, 5] = 100  # guaranteed threshold crossing for zgate
    flat = np.arange(n)
    labels = (flat % 3).astype(np.int8)
    starts = (flat * 2).astype(np.int32)
    header = SAMHeader(cols, nlines, scanlen, 1000.0, 0, 1.0)
    return SAMScan.handler_from_data(
        data, header, starts=starts,
        samlabels=SAMLabels(labels, ["healthy", "a", "b"]))


def _snapshot(h: SAMScan) -> Dict[str, object]:
    return {
        "data": h.data.copy(),
        "labels": np.array(h.samlabels.labels, copy=True),
        "starts": None if h.starts is None else np.array(h.starts, copy=True),
        "shape": tuple(h.shape),
        "scanlen": h.scanlen,
        "tzero": h.header.tzero,
        "samplerate": h.header.samplerate,
    }


def _assert_unchanged(h: SAMScan, snap: Dict[str, object]) -> None:
    np.testing.assert_array_equal(h.data, snap["data"])
    np.testing.assert_array_equal(np.asarray(h.samlabels.labels), snap["labels"])
    if snap["starts"] is None:
        assert h.starts is None
    else:
        np.testing.assert_array_equal(h.starts, snap["starts"])
    assert tuple(h.shape) == snap["shape"]
    assert h.scanlen == snap["scanlen"]
    assert h.header.tzero == snap["tzero"]
    assert h.header.samplerate == snap["samplerate"]


def _assert_same(a: SAMScan, b: SAMScan) -> None:
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(np.asarray(a.samlabels.labels),
                                  np.asarray(b.samlabels.labels))
    assert tuple(a.shape) == tuple(b.shape)
    assert a.scanlen == b.scanlen
    assert a.header.tzero == b.header.tzero
    assert a.header.samplerate == b.header.samplerate
    if a.starts is None or b.starts is None:
        assert a.starts is None and b.starts is None
    else:
        np.testing.assert_array_equal(a.starts, b.starts)


def _time_op(h: SAMScan, in_place: bool) -> SAMScan:
    tz = h.header.tzero
    return h.time_range_select(tz, tz + h.samplespacing * 60, in_place=in_place)


Op = Callable[[SAMScan, bool], SAMScan]
DefaultOp = Callable[[SAMScan], SAMScan]

# (name, explicit op, default-argument call, documented default)
OPS: List[Tuple[str, Op, DefaultOp, bool]] = [
    ("rotate90", lambda h, ip: h.rotate(90, in_place=ip),
     lambda h: h.rotate(90), True),
    ("rotate180", lambda h, ip: h.rotate(180, in_place=ip),
     lambda h: h.rotate(180), True),
    ("rotate270", lambda h, ip: h.rotate(270, in_place=ip),
     lambda h: h.rotate(270), True),
    ("mirror_x", lambda h, ip: h.mirror("x", in_place=ip),
     lambda h: h.mirror("x"), True),
    ("mirror_y", lambda h, ip: h.mirror("y", in_place=ip),
     lambda h: h.mirror("y"), True),
    ("downsample_sample", lambda h, ip: h.downsample(2, "sample", in_place=ip),
     lambda h: h.downsample(2, "sample"), True),
    ("downsample_mean", lambda h, ip: h.downsample(2, "mean", in_place=ip),
     lambda h: h.downsample(2, "mean"), True),
    ("downsample_median", lambda h, ip: h.downsample(3, "median", in_place=ip),
     lambda h: h.downsample(3, "median"), True),
    ("zgate", lambda h, ip: h.zgate(0.2, 100, in_place=ip),
     lambda h: h.zgate(0.2, 100), False),
    ("rectangle_select",
     lambda h, ip: h.rectangle_select(0, 2, 1, 3, in_place=ip),
     lambda h: h.rectangle_select(0, 2, 1, 3), False),
    ("time_range_select", _time_op,
     lambda h: h.time_range_select(h.header.tzero,
                                   h.header.tzero + h.samplespacing * 60),
     False),
]

IDS = [o[0] for o in OPS]


@pytest.mark.parametrize("name,op,default_call,default", OPS, ids=IDS)
class TestInPlaceContract:
    def test_in_place_true_returns_self(self, name: str, op: Op,
                                        default_call: DefaultOp,
                                        default: bool) -> None:
        h = _make_handler()
        assert op(h, True) is h

    def test_in_place_false_returns_independent_copy(self, name: str, op: Op,
                                                     default_call: DefaultOp,
                                                     default: bool) -> None:
        h = _make_handler()
        before = _snapshot(h)
        r = op(h, False)
        assert r is not h
        _assert_unchanged(h, before)
        assert not np.shares_memory(r.data, h.data)

    def test_mutating_result_does_not_touch_source(self, name: str, op: Op,
                                                   default_call: DefaultOp,
                                                   default: bool) -> None:
        h = _make_handler()
        before = _snapshot(h)
        r = op(h, False)
        r.data[0, 0] = np.int8((int(r.data[0, 0]) + 1) % 127)
        _assert_unchanged(h, before)

    def test_both_paths_agree(self, name: str, op: Op,
                              default_call: DefaultOp,
                              default: bool) -> None:
        a = op(_make_handler(), True)
        b = op(_make_handler(), False)
        _assert_same(a, b)

    def test_default_flag_matches_documented_behaviour(
            self, name: str, op: Op, default_call: DefaultOp,
            default: bool) -> None:
        # Call without the flag and check the documented default.
        h = _make_handler()
        before = _snapshot(h)
        r = default_call(h)
        if default:
            assert r is h
            _assert_same(r, op(_make_handler(), True))
        else:
            assert r is not h
            _assert_unchanged(h, before)
            _assert_same(r, op(_make_handler(), False))


def test_rotate_mirror_roundtrip_edge_shapes() -> None:
    for nlines, cols in [(1, 1), (1, 5), (5, 1), (4, 4), (3, 4), (4, 3), (2, 3)]:
        h = _make_handler(nlines=nlines, cols=cols, scanlen=32)
        for deg, back in ((90, 270), (180, 180), (270, 90)):
            np.testing.assert_array_equal(h.rotated(deg).rotated(back).data,
                                          h.data)
        for ax in ("x", "y"):
            np.testing.assert_array_equal(h.mirrored(ax).mirrored(ax).data,
                                          h.data)


def test_rotate_non_in_place_labels_and_starts() -> None:
    nlines, cols = 3, 4
    h = _make_handler(nlines=nlines, cols=cols, scanlen=32)
    flat = np.arange(nlines * cols)
    labels_grid = (flat % 3).reshape(nlines, cols)
    starts_grid = (flat * 2).reshape(nlines, cols)
    for deg, k in ((90, -1), (180, -2), (270, -3)):
        r = h.rotate(deg, in_place=False)
        expected_labels = np.rot90(labels_grid, k)
        assert tuple(r.shape) == expected_labels.shape
        np.testing.assert_array_equal(r.samlabels.labels.reshape(r.shape),
                                      expected_labels)
        np.testing.assert_array_equal(r.starts.reshape(r.shape),
                                      np.rot90(starts_grid, k))


def test_mirror_non_in_place_labels_and_starts() -> None:
    nlines, cols = 3, 4
    h = _make_handler(nlines=nlines, cols=cols, scanlen=32)
    flat = np.arange(nlines * cols)
    labels_grid = (flat % 3).reshape(nlines, cols)
    starts_grid = (flat * 2).reshape(nlines, cols)
    for ax, axis in (("x", 1), ("y", 0)):
        r = h.mirror(ax, in_place=False)
        np.testing.assert_array_equal(r.samlabels.labels.reshape(r.shape),
                                      np.flip(labels_grid, axis=axis))
        np.testing.assert_array_equal(r.starts.reshape(r.shape),
                                      np.flip(starts_grid, axis=axis))


def test_rectangle_select_metadata_both_paths() -> None:
    nlines, cols = 3, 4
    idx = [l * cols + c for l in range(2) for c in (1, 2)]
    for in_place in (False, True):
        h = _make_handler(nlines=nlines, cols=cols, scanlen=32)
        data_before = h.data.copy()
        labels_before = np.array(h.samlabels.labels, copy=True)
        starts_before = np.array(h.starts, copy=True)
        r = h.rectangle_select(0, 2, 1, 3, in_place=in_place)
        np.testing.assert_array_equal(r.data, data_before[idx])
        np.testing.assert_array_equal(r.samlabels.labels, labels_before[idx])
        np.testing.assert_array_equal(r.starts, starts_before[idx])


def test_time_range_select_drops_starts_both_paths() -> None:
    for in_place in (False, True):
        h = _make_handler()
        before = h.data.copy()
        tz, sp = h.header.tzero, h.samplespacing
        r = h.time_range_select(tz + sp * 10, tz + sp * 70, in_place=in_place)
        assert r.starts is None
        np.testing.assert_array_equal(r.data, before[:, 10:70])


@needs_data
class TestMmapInPlaceContract:
    def test_not_in_place_materializes_and_is_independent(self) -> None:
        ops: List[Tuple[str, Op]] = [
            ("rotate", lambda h, ip: h.rotate(90, in_place=ip)),
            ("mirror", lambda h, ip: h.mirror("y", in_place=ip)),
            ("downsample", lambda h, ip: h.downsample(2, "sample", in_place=ip)),
        ]
        for _, op in ops:
            eager = SAMScan(H5_PATH)
            lazy = SAMScan(H5_PATH, mmap=True)
            r = op(lazy, False)
            assert r is not lazy
            assert r.loaded is True
            assert lazy.loaded is True  # copy() materializes the source
            assert not np.shares_memory(r.data, lazy.data)
            np.testing.assert_array_equal(r.data, op(eager, False).data)

    def test_in_place_true_materializes_and_mutates(self) -> None:
        eager = SAMScan(H5_PATH)
        eager.rotate(90, in_place=True)
        lazy = SAMScan(H5_PATH, mmap=True)
        r = lazy.rotate(90, in_place=True)
        assert r is lazy
        assert lazy.loaded is True
        np.testing.assert_array_equal(lazy.data, eager.data)

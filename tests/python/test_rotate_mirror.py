"""
rotate/mirror permute the spatial layout of scans without touching the
time-domain signals.  Each scan's flat grid index is encoded in
``data[:, 0]`` so a single reshape exposes the spatial permutation;
labels / starts must follow the same permutation.
"""

import numpy as np
import pytest

from samcore import SAMHeader, SAMLabels, SAMScan

from conftest import needs_data


def _make_handler(nlines=3, cols=4, scanlen=5, with_labels=True,
                  with_starts=False):
    idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
    data = np.zeros((nlines * cols, scanlen), dtype=np.int8)
    data[:, 0] = idx_grid.reshape(-1)
    data[:, 1] = 7

    hdr = SAMHeader(cols, nlines, scanlen, 25.0, 0, 1.0)

    if with_labels:
        labels = (idx_grid.reshape(-1).astype(np.int8)) % 3
        samlabels = SAMLabels(labels, ["healthy", "b", "c"])
    else:
        samlabels = None

    starts = (idx_grid.reshape(-1) * 2).astype(np.int32) if with_starts else None

    return SAMScan.handler_from_data(data, hdr, starts=starts,
                                     samlabels=samlabels)


class TestRotateShape:
    def test_90_changes_shape(self):
        h = _make_handler(nlines=3, cols=4)
        h.rotate(90)
        assert h.shape == (4, 3)
        assert h.cols == 3
        assert h.nlines == 4

    def test_180_preserves_shape(self):
        h = _make_handler(nlines=3, cols=4)
        h.rotate(180)
        assert h.shape == (3, 4)

    def test_270_changes_shape(self):
        h = _make_handler(nlines=3, cols=4)
        h.rotate(270)
        assert h.shape == (4, 3)

    def test_360_is_noop(self):
        h = _make_handler(nlines=3, cols=4)
        before = h.data.copy()
        h.rotate(360)
        assert h.shape == (3, 4)
        np.testing.assert_array_equal(h.data, before)

    def test_0_is_noop(self):
        h = _make_handler(nlines=3, cols=4)
        before = h.data.copy()
        h.rotate(0)
        assert h.shape == (3, 4)
        np.testing.assert_array_equal(h.data, before)

    def test_negative_normalises(self):
        h1 = _make_handler().rotate(-90, in_place=False)
        h2 = _make_handler().rotate(270, in_place=False)
        assert h1.shape == h2.shape
        np.testing.assert_array_equal(h1.data, h2.data)


class TestRotateData:
    def test_90cw_matches_np_rot90(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        h.rotate(90)
        np.testing.assert_array_equal(
            h.data[:, 0].reshape(h.shape), np.rot90(idx_grid, k=-1))

    def test_180_matches_np_rot90(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        h.rotate(180)
        np.testing.assert_array_equal(
            h.data[:, 0].reshape(h.shape), np.rot90(idx_grid, k=-2))

    def test_270cw_matches_np_rot90(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        h.rotate(270)
        np.testing.assert_array_equal(
            h.data[:, 0].reshape(h.shape), np.rot90(idx_grid, k=-3))

    def test_full_rotation_returns_original(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        original = h.data.copy()
        for _ in range(4):
            h.rotate(90)
        assert h.shape == (nlines, cols)
        np.testing.assert_array_equal(h.data, original)


class TestRotatePermutation:
    def test_labels_follow_data(self):
        nlines, cols = 3, 4
        labels_grid = np.arange(nlines * cols).reshape(nlines, cols) % 3

        for degrees in (90, 180, 270):
            h2 = _make_handler(nlines=nlines, cols=cols)
            h2.rotate(degrees)
            k = -(degrees // 90)
            expected = np.rot90(labels_grid, k=k)
            np.testing.assert_array_equal(
                h2.samlabels.labels.reshape(h2.shape), expected)

    def test_starts_follow_data(self):
        nlines, cols = 3, 4
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        starts_grid = idx_grid * 2

        for degrees in (90, 180, 270):
            h2 = _make_handler(nlines=nlines, cols=cols, with_starts=True)
            h2.rotate(degrees)
            k = -(degrees // 90)
            expected = np.rot90(starts_grid, k=k).reshape(-1)
            np.testing.assert_array_equal(h2.starts, expected)

    def test_signals_untouched(self):
        nlines, cols, scanlen = 3, 4, 5
        data = np.zeros((nlines * cols, scanlen), dtype=np.int8)
        data[:] = np.arange(scanlen)
        hdr = SAMHeader(cols, nlines, scanlen, 25.0, 0, 1.0)
        h = SAMScan.handler_from_data(data, hdr)

        h.rotate(90).mirror("y").rotate(180)
        np.testing.assert_array_equal(h.data, np.tile(np.arange(scanlen),
                                                      (h.data.shape[0], 1)))


class TestRotateInPlace:
    def test_in_place_true_returns_self(self):
        h = _make_handler()
        assert h.rotate(90, in_place=True) is h

    def test_in_place_false_leaves_original(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        before = h.data.copy()
        h2 = h.rotate(90, in_place=False)
        assert h2 is not h
        assert h.shape == (nlines, cols)
        np.testing.assert_array_equal(h.data, before)
        assert h2.shape == (cols, nlines)


class TestRotateValidation:
    @pytest.mark.parametrize("bad", [45, 1, 100, -10, 720 + 45])
    def test_bad_degrees_raises(self, bad):
        with pytest.raises(ValueError, match="90, 180, or 270"):
            _make_handler().rotate(bad)


class TestMirrorShape:
    @pytest.mark.parametrize("orientation", ["x", "y", "X", "Y"])
    def test_shape_unchanged(self, orientation):
        h = _make_handler(nlines=3, cols=4)
        h.mirror(orientation)
        assert h.shape == (3, 4)


class TestMirrorData:
    def test_mirror_x_reverses_columns(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        h.mirror("x")
        np.testing.assert_array_equal(
            h.data[:, 0].reshape(h.shape), np.flip(idx_grid, axis=1))

    def test_mirror_y_reverses_lines(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        h.mirror("y")
        np.testing.assert_array_equal(
            h.data[:, 0].reshape(h.shape), np.flip(idx_grid, axis=0))

    def test_mirror_x_twice_is_identity(self):
        h1 = _make_handler().mirror("x")
        h1.mirror("x")
        h2 = _make_handler()
        np.testing.assert_array_equal(h1.data, h2.data)

    def test_mirror_y_twice_is_identity(self):
        h1 = _make_handler().mirror("y")
        h1.mirror("y")
        h2 = _make_handler()
        np.testing.assert_array_equal(h1.data, h2.data)


class TestMirrorPermutation:
    def test_labels_follow_data_x(self):
        nlines, cols = 3, 4
        labels_grid = np.arange(nlines * cols).reshape(nlines, cols) % 3
        h = _make_handler(nlines=nlines, cols=cols)
        h.mirror("x")
        np.testing.assert_array_equal(
            h.samlabels.labels.reshape(h.shape), np.flip(labels_grid, axis=1))

    def test_labels_follow_data_y(self):
        nlines, cols = 3, 4
        labels_grid = np.arange(nlines * cols).reshape(nlines, cols) % 3
        h = _make_handler(nlines=nlines, cols=cols)
        h.mirror("y")
        np.testing.assert_array_equal(
            h.samlabels.labels.reshape(h.shape), np.flip(labels_grid, axis=0))

    def test_starts_follow_data(self):
        nlines, cols = 3, 4
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        starts_grid = idx_grid * 2
        for orient in ("x", "y"):
            h = _make_handler(nlines=nlines, cols=cols, with_starts=True)
            h.mirror(orient)
            axis = 1 if orient == "x" else 0
            np.testing.assert_array_equal(
                h.starts, np.flip(starts_grid, axis=axis).reshape(-1))


class TestMirrorInPlaceAndValidation:
    def test_in_place_true_returns_self(self):
        h = _make_handler()
        assert h.mirror("x", in_place=True) is h

    def test_in_place_false_leaves_original(self):
        nlines, cols = 3, 4
        h = _make_handler(nlines=nlines, cols=cols)
        before = h.data.copy()
        h2 = h.mirror("x", in_place=False)
        assert h2 is not h
        np.testing.assert_array_equal(h.data, before)
        idx_grid = np.arange(nlines * cols).reshape(nlines, cols)
        np.testing.assert_array_equal(
            h2.data[:, 0].reshape(h2.shape), np.flip(idx_grid, axis=1))

    @pytest.mark.parametrize("bad", ["z", "horizontal", "vertical", ""])
    def test_bad_orientation_raises(self, bad):
        with pytest.raises(ValueError, match="x' or 'y'"):
            _make_handler().mirror(bad)


class TestRotateOnRealHandler:
    @needs_data
    def test_rotate_then_unrotate_restores_data(self, h5):
        h0 = h5.copy()
        original = h0.data.copy()
        h0.rotate(90).rotate(270)
        np.testing.assert_array_equal(h0.data, original)
        assert h0.shape == h5.shape

    @needs_data
    def test_mirror_x_twice_real(self, h5):
        h0 = h5.copy()
        original = h0.data.copy()
        h0.mirror("x").mirror("x")
        np.testing.assert_array_equal(h0.data, original)

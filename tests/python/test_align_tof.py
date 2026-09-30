"""Tests for the time-of-flight / thickness images and cross-correlation
alignment (both in-place and copy variants)."""

import numpy as np
import pytest

from samcore import SAMHeader, SAMScan


def _header(nlines, cols, scanlen, samplerate=100.0, tzero=1000):
    return SAMHeader(scanspline=cols, nlines=nlines, scanlen=scanlen,
                     samplerate=samplerate, tzero=tzero, resolution=1.0)


def _template(seed, n=200):
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 100, size=n).astype(np.int8)


# ── time of flight / thickness ──────────────────────────────────────────────


def _burst(j, center, width, amplitude):
    """Gaussian-modulated 20%-of-Nyquist tone; envelope peaks at center."""
    t = np.asarray(j, dtype=float) - center
    x = amplitude * np.exp(-0.5 * (t / width) ** 2) * np.sin(2 * np.pi * 0.2 * t)
    return np.clip(np.round(x), -128, 127).astype(np.int8)


def test_tof_envelope_peak():
    n = 400
    data = np.zeros((4, n), dtype=np.int8)
    j = np.arange(n)
    both = _burst(j, 100, 8.0, 100.0).astype(int) + \
        _burst(j, 250, 6.0, 60.0).astype(int)
    data[0] = np.clip(both, -128, 127).astype(np.int8)
    data[1] = _burst(j, 250, 6.0, 60.0)
    data[2] = _burst(j, 100, 8.0, 100.0)
    # row 3 stays silent
    h = SAMScan.handler_from_data(data, _header(2, 2, n))

    t = h.tof()
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    # tzero 1000 + sample * 10 ns, refined to the exact envelope centre
    np.testing.assert_allclose(t[0, 0], 2000.0, atol=0.05)
    np.testing.assert_allclose(t[0, 1], 3500.0, atol=0.05)
    assert np.isnan(t[1, 1])

    # the gate picks the burst inside it (second burst only)
    g = h.tof(start=180)
    np.testing.assert_allclose(g[0, 0], 3500.0, atol=0.05)

    # the coarse pick is the exact integer sample
    np.testing.assert_allclose(h.tof(sub_sample=False)[0, 0], 2000.0)


def test_tof_validation_and_silent_window():
    h = SAMScan.handler_from_data(
        np.zeros((1, 100), dtype=np.int8), _header(1, 1, 100))
    assert np.isnan(h.tof()[0, 0])  # no envelope peak

    with pytest.raises(ValueError):
        h.tof(start=50, end=50)
    with pytest.raises(ValueError):
        h.tof(start=0, end=500)
    with pytest.raises(ValueError):
        h.tof(start=-1)


def test_thickness_scales_with_sound_speed():
    n = 100
    data = np.zeros((1, n), dtype=np.int8)
    data[0] = _burst(np.arange(n), 25, 4.0, 100.0)
    h = SAMScan.handler_from_data(data, _header(1, 1, n, tzero=0))

    np.testing.assert_allclose(h.tof(), [[250.0]], atol=0.05)
    # 250 ns * 1e-9 * 1500 m/s / 2
    np.testing.assert_allclose(h.thickness(1500.0), [[1.875e-4]], rtol=1e-6)
    with pytest.raises(ValueError):
        h.thickness(0.0)


# ── cross-correlation alignment ─────────────────────────────────────────────


def test_align_xcorr_recovers_known_shifts():
    tmpl = _template(0)
    data = np.zeros((3, 200), dtype=np.int8)
    data[0] = tmpl
    data[1, 7:] = tmpl[:-7]     # delayed by 7
    data[2, :-4] = tmpl[4:]     # early by 4
    h = SAMScan.handler_from_data(data, _header(1, 3, 200))

    h.align_xcorr(0, 30)
    # interior samples coincide with the reference.  The delayed scan loses
    # its tail and the early scan loses its head: those samples lie outside
    # the buffer and are zero-filled.
    np.testing.assert_array_equal(h.data[1, :190], h.data[0, :190])
    np.testing.assert_array_equal(h.data[2, 4:190], h.data[0, 4:190])


def test_align_xcorr_advances_starts_and_copy_variant():
    tmpl = _template(1)
    data = np.zeros((2, 200), dtype=np.int8)
    data[0] = tmpl
    data[1, 9:] = tmpl[:-9]
    starts = np.array([100, 100], dtype=np.int32)
    h = SAMScan.handler_from_data(data, _header(1, 2, 200), starts)

    aligned = h.aligned_xcorr(0, 30)
    np.testing.assert_array_equal(h.data, data)  # original untouched
    assert list(aligned.starts) == [100, 109]


def test_align_xcorr_validation():
    h = SAMScan.handler_from_data(
        np.zeros((2, 64), dtype=np.int8), _header(2, 1, 64))
    with pytest.raises(IndexError):
        h.align_xcorr(5)
    with pytest.raises(ValueError):
        h.align_xcorr(0, 64)
    with pytest.raises(ValueError):
        h.align_xcorr(0, -1)


def test_align_tof_aligns_echoes_and_advances_starts():
    n = 400
    j = np.arange(n)
    data = np.zeros((2, n), dtype=np.int8)
    data[0] = _burst(j, 100, 8.0, 100.0)
    data[1] = _burst(j, 130, 8.0, 100.0)
    starts = np.array([50, 50], dtype=np.int32)
    h = SAMScan.handler_from_data(data, _header(1, 2, n), starts)

    aligned = h.aligned_tof(gate_ns=2500.0, reference=0)
    np.testing.assert_array_equal(h.data, data)  # original untouched
    assert list(aligned.starts) == [50, 80]      # delayed echo -> +30 shift
    np.testing.assert_array_equal(aligned.data[1, 40:350],
                                  aligned.data[0, 40:350])

    # in-place path without starts
    h2 = SAMScan.handler_from_data(data, _header(1, 2, n))
    h2.align_tof(gate_ns=2500.0)
    np.testing.assert_array_equal(h2.data[1, 40:350], h2.data[0, 40:350])


def test_align_tof_rounds_sub_sample_offsets():
    n = 400
    j = np.arange(n)
    data = np.zeros((2, n), dtype=np.int8)
    data[0] = _burst(j, 100, 8.0, 100.0)
    data[1] = _burst(j, 110.6, 8.0, 100.0)
    starts = np.array([30, 30], dtype=np.int32)
    h = SAMScan.handler_from_data(data, _header(1, 2, n), starts)

    h.align_tof(gate_ns=2500.0)
    assert list(h.starts) == [30, 41]  # 10.6 samples -> rounded shift 11


def test_align_tof_validation():
    silent = SAMScan.handler_from_data(
        np.zeros((2, 200), dtype=np.int8), _header(1, 2, 200))
    with pytest.raises(ValueError):
        silent.align_tof(gate_ns=100.0)  # no envelope peak in the gate

    h = SAMScan.handler_from_data(
        np.zeros((2, 200), dtype=np.int8), _header(1, 2, 200))
    with pytest.raises(ValueError):
        h.align_tof(gate_ns=0.0)
    with pytest.raises(ValueError):
        h.align_tof(gate_ns=-5.0)
    with pytest.raises(IndexError):
        h.align_tof(gate_ns=100.0, reference=5)
    with pytest.raises(ValueError):
        h.align_tof(gate_ns=100.0, start_ns=-1.0)


def test_align_manual_public():
    data = np.full((2, 100), 5, dtype=np.int8)
    starts = np.array([10, 20], dtype=np.int32)
    h = SAMScan.handler_from_data(data, _header(1, 2, 100), starts)

    h.align_manual([3, -1], 50)
    assert list(h.starts) == [13, -1]
    assert np.all(h.data[1] == 0)   # -1 rows are zero-filled
    assert np.all(h.data[0] == 5)   # valid rows keep their data
    with pytest.raises(ValueError):
        h.align_manual([100, 0], 50)

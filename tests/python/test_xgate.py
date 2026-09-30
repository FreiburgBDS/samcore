"""Tests for the native XGate (layered gating) implementation, checked
against a NumPy reference implementation."""

import numpy as np
import pytest

from samcore import SAMHeader, SAMScan


def _header(nlines, scanlen, samplerate=100.0, tzero=0):
    return SAMHeader(scanspline=1, nlines=nlines, scanlen=scanlen,
                     samplerate=samplerate, tzero=tzero, resolution=1.0)


def test_xgate_none_pick_matches_reference():
    rng = np.random.default_rng(0)
    n, sl = 6, 40
    data = rng.integers(-100, 100, size=(n, sl)).astype(np.int8)
    h = SAMScan.handler_from_data(data, _header(n, sl))

    gate_ns = 30.0  # 3 samples at 10 ns/sample
    n_gates = 4
    values, starts = h.xgate(gate_ns, n_gates, pick="none", mode="max")
    assert values.shape == (n, 1, n_gates)
    assert values.dtype == np.float32
    assert starts.dtype == np.int32
    np.testing.assert_array_equal(starts, np.zeros(n, dtype=np.int32))
    for i in range(n):
        for g in range(n_gates):
            np.testing.assert_allclose(values[i, 0, g],
                                       data[i, g * 3:(g + 1) * 3].max())

    v_abs, _ = h.xgate(gate_ns, n_gates, pick="none", mode="absmax")
    v_pow, _ = h.xgate(gate_ns, n_gates, pick="none", mode="power")
    for g in range(n_gates):
        seg = data[:, g * 3:(g + 1) * 3].astype(np.float32)
        np.testing.assert_allclose(v_abs[:, 0, g], np.abs(seg).max(axis=1))
        np.testing.assert_allclose(v_pow[:, 0, g], np.square(seg).sum(axis=1))


def test_xgate_threshold_pick_matches_reference():
    rng = np.random.default_rng(1)
    n, sl = 5, 50
    data = rng.integers(-100, 100, size=(n, sl)).astype(np.int8)
    data[:, 7] = 100  # guaranteed crossing for every row
    h = SAMScan.handler_from_data(data, _header(n, sl))

    gate_ns, n_gates = 40.0, 3  # 4 samples per gate
    values, starts = h.xgate(gate_ns, n_gates, pick="threshold",
                             threshold=0.2, mode="max")
    # reference: strictly greater than 0.2 * 127, positive only
    ref_starts = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        above = np.where(data[i] > 0.2 * 127)[0]
        if len(above):
            ref_starts[i] = above[0]
    np.testing.assert_array_equal(starts, ref_starts)
    for i in range(n):
        for g in range(n_gates):
            s = int(ref_starts[i]) + g * 4
            if ref_starts[i] >= 0 and s + 4 <= sl:
                np.testing.assert_allclose(values[i, 0, g],
                                           data[i, s:s + 4].max())
            else:
                assert values[i, 0, g] == 0.0

    # threshold=None falls back to pick="none" (start at sample 0)
    v_none, s_none = h.xgate(gate_ns, n_gates, pick="threshold",
                             threshold=None)
    np.testing.assert_array_equal(s_none, np.zeros(n, dtype=np.int32))
    v_none_ref, _ = h.xgate(gate_ns, n_gates, pick="none", mode="max")
    np.testing.assert_allclose(v_none, v_none_ref)


def test_xgate_tof_pick_matches_tof():
    n, sl = 3, 400
    j = np.arange(sl)

    def burst(c):
        x = 100 * np.exp(-0.5 * ((j - c) / 8.0) ** 2) * \
            np.sin(2 * np.pi * 0.2 * (j - c))
        return np.clip(np.round(x), -128, 127).astype(np.int8)

    data = np.stack([burst(100), burst(250), np.zeros(sl, dtype=np.int8)])
    h = SAMScan.handler_from_data(data, _header(n, sl))

    values, starts = h.xgate(100.0, 3, pick="tof")
    tof_ns = h.tof()
    valid = ~np.isnan(tof_ns[:, 0])
    # positive values round half away from zero (std::llround)
    ref = np.floor(tof_ns[valid, 0] / 10.0 + 0.5).astype(np.int32)
    np.testing.assert_array_equal(starts[valid], ref)
    np.testing.assert_array_equal(starts[~valid], -1)
    assert values[2].sum() == 0.0  # silent row


def test_xgate_validation():
    data = np.zeros((1, 32), dtype=np.int8)
    h = SAMScan.handler_from_data(data, _header(1, 32))
    with pytest.raises(ValueError):
        h.xgate(0.0)
    with pytest.raises(ValueError):
        h.xgate(50.0, pick="bogus")
    with pytest.raises(ValueError):
        h.xgate(50.0, mode="bogus")
    with pytest.raises(ValueError):
        h.xgate(50.0, pick="threshold", threshold=1.5)

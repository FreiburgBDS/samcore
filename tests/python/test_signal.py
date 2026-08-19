"""Covers numerical parity of the spectral reductions with per-row 1-D
references and edge cases (silent scans, degenerate inputs).  samcore's
utils operate on 2-D PSD arrays (float32 at the boundary), so only the
2-D cases are ported.
"""

import numpy as np
import pytest

from samcore import utils


def _entropy_1d_ref(psd, base=2.0):
    total = np.sum(psd)
    if total == 0:
        return 0.0
    psd_norm = psd / total
    psd_norm = psd_norm[np.where(psd_norm > 0)]
    return float(-np.sum(psd_norm * np.log(psd_norm)) / np.log(base))


def _flatness_1d_ref(psd):
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    arithmetic_mean = np.mean(psd)
    return float(geometric_mean / arithmetic_mean)


class TestSpectralEntropy:
    def test_2d_matches_per_row_1d(self):
        rng = np.random.default_rng(0)
        psd = rng.random((7, 50)) + 1e-3
        vec = utils.spectral_entropy(psd, base=2.0)
        assert vec.shape == (7,)
        for i in range(7):
            assert np.isclose(vec[i], _entropy_1d_ref(psd[i], base=2.0))

    def test_silent_row_returns_zero(self):
        psd = np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3, 0.4]])
        with np.errstate(all="raise"):
            out = utils.spectral_entropy(psd)
        assert out.shape == (2,)
        assert out[0] == 0.0
        assert np.isfinite(out[1])

    def test_base_e(self):
        rng = np.random.default_rng(2)
        psd = rng.random((4, 30)) + 1e-3
        out = utils.spectral_entropy(psd, base=np.e)
        for i in range(4):
            assert np.isclose(out[i], _entropy_1d_ref(psd[i], base=np.e))

    def test_no_warnings_on_zero_bins(self):
        psd = np.array([[0.0, 0.5, 0.0, 0.5]])
        with np.errstate(all="raise"):
            out = utils.spectral_entropy(psd)
        assert np.isfinite(out[0])
        assert np.isclose(out[0], 1.0)


class TestSpectralFlatness:
    def test_2d_matches_per_row_1d(self):
        rng = np.random.default_rng(3)
        psd = rng.random((6, 40)) + 1e-3
        vec = utils.spectral_flatness(psd)
        assert vec.shape == (6,)
        for i in range(6):
            assert np.isclose(vec[i], _flatness_1d_ref(psd[i]))

    def test_flat_psd_returns_one(self):
        psd = np.tile(np.array([1.0, 1.0, 1.0, 1.0]), (5, 1))
        out = utils.spectral_flatness(psd)
        assert out.shape == (5,)
        np.testing.assert_allclose(out, np.ones(5), atol=1e-6)

    def test_peaked_psd_returns_near_zero(self):
        psd = np.zeros((4, 16))
        psd[:, 3] = 1.0
        out = utils.spectral_flatness(psd)
        assert out.shape == (4,)
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-6)

    def test_silent_row_returns_zero(self):
        psd = np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3, 0.4]])
        with np.errstate(all="raise"):
            out = utils.spectral_flatness(psd)
        assert out.shape == (2,)
        assert out[0] == 0.0
        assert np.isfinite(out[1])


class TestSpectralCentroid:
    def test_shape_2d(self):
        rng = np.random.default_rng(4)
        freqs = np.linspace(0, 5e6, 64)
        psd = rng.random((11, 64))
        out = utils.spectral_centroid(freqs, psd)
        assert out.shape == (11,)
        assert np.all(np.isfinite(out))

    def test_centroid_of_impulse_at_known_bin(self):
        freqs = np.linspace(0, 50e6, 33)
        psd = np.zeros((5, len(freqs)))
        target_bin = 10
        psd[:, target_bin] = 1.0
        out = utils.spectral_centroid(freqs, psd)
        np.testing.assert_allclose(out, np.full(5, freqs[target_bin]))

    def test_centroid_weighted_mean(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0])
        psd = np.array([[2.0, 0.0, 0.0, 2.0],
                        [0.0, 0.0, 4.0, 0.0]])
        out = utils.spectral_centroid(freqs, psd)
        np.testing.assert_allclose(out, np.array([1.5, 2.0]))

    def test_silent_row_returns_zero(self):
        freqs = np.array([0.0, 1.0, 2.0])
        psd = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0]])
        with np.errstate(all="raise"):
            out = utils.spectral_centroid(freqs, psd)
        assert out.shape == (2,)
        assert out[0] == 0.0
        assert out[1] == 0.0


class TestSpectralEnergyRatio:
    def test_shape_2d(self):
        rng = np.random.default_rng(5)
        freqs = np.linspace(0, 5e6, 65)
        psd = rng.random((9, 65))
        out = utils.spectral_energy_ratio(freqs, psd, critical_freq=2.5e6)
        assert out.shape == (9,)
        assert np.all(np.isfinite(out))

    def test_known_split(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        psd = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [3.0, 1.0, 0.0, 0.0, 1.0, 2.0]])
        out = utils.spectral_energy_ratio(freqs, psd, critical_freq=3.0)
        np.testing.assert_allclose(out, np.array([1.0, 0.75]))

    def test_all_energy_above_returns_zero(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0])
        psd = np.array([[0.0, 0.0, 1.0, 1.0]])
        out = utils.spectral_energy_ratio(freqs, psd, critical_freq=2.0)
        assert out.shape == (1,)
        assert out[0] == 0.0

    def test_silent_row_returns_zero(self):
        freqs = np.array([0.0, 1.0, 2.0])
        psd = np.array([[0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]])
        with np.errstate(all="raise"):
            out = utils.spectral_energy_ratio(freqs, psd, critical_freq=1.0)
        assert out.shape == (2,)
        assert out[0] == 0.0
        assert np.isfinite(out[1])

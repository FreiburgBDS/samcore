"""Covers numerical parity of the spectral reductions with per-row 1-D
references and edge cases (silent scans, degenerate inputs).  samcore's
utils operate on 2-D PSD arrays (float32 at the boundary), so only the
2-D cases are ported.
"""

import numpy as np

from samcore import preprocessing, utils


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


class TestKurt:
    """Constant rows return the documented 0, not NaN (fast-math)."""

    def test_constant_row_is_zero(self):
        data = np.stack([np.full(16, 3.0, dtype=np.float32),
                         np.arange(16, dtype=np.float32)])
        k = utils.kurt(data)
        assert k[0] == 0.0
        assert np.isfinite(k[1])


class TestMedfilt:
    """Edge padding is whole-sample reflection, not zero padding."""

    def test_interior_matches_sliding_median(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 64)).astype(np.float32)
        got = preprocessing.medfilt(data, 5)
        # Away from the padded edges a median filter is a plain sliding median.
        ref = np.stack([np.array([np.median(row[i - 2:i + 3])
                                  for i in range(5, row.size - 5)])
                        for row in data])
        np.testing.assert_allclose(got[:, 5:-5], ref, atol=1e-6)

    def test_edges_use_whole_sample_reflection(self):
        data = np.array([[0.0, 3.0, 1.0, 2.0, 9.0, 1.0, 4.0, 5.0, 7.0]],
                        dtype=np.float32)
        got = preprocessing.medfilt(data, 3)[0]
        padded = np.pad(data[0], 1, mode="reflect")
        ref = np.array([np.median(padded[i:i + 3])
                        for i in range(data.shape[1])])
        np.testing.assert_array_equal(got, ref)
        # A zero-padding implementation would return 0.0 here.
        assert got[0] == 3.0


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


def _analytic_1d(x):
    """scipy.signal.hilbert reference: keep DC/Nyquist, double the rest."""
    n = x.size
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x) * h)


class TestAnalytic:
    def test_matches_reference(self):
        rng = np.random.default_rng(6)
        data = rng.standard_normal((5, 128)).astype(np.float32)
        got = preprocessing.analytic(data)
        assert got.shape == data.shape
        assert got.dtype == np.complex64
        ref = np.stack([_analytic_1d(row.astype(np.float64)) for row in data])
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_odd_length_matches_reference(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((3, 129)).astype(np.float32)
        got = preprocessing.analytic(data)
        ref = np.stack([_analytic_1d(row.astype(np.float64)) for row in data])
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_envelope_is_abs(self):
        rng = np.random.default_rng(8)
        data = rng.standard_normal((4, 100)).astype(np.float32)
        np.testing.assert_allclose(preprocessing.envelope(data),
                                   np.abs(preprocessing.analytic(data)),
                                   atol=1e-6)

    def test_single_sample_keeps_dc(self):
        got = preprocessing.analytic(np.array([[5.0]], dtype=np.float32))
        np.testing.assert_allclose(got, np.array([[5.0 + 0.0j]]))

    def test_negative_constant_is_real(self):
        data = np.full((1, 32), -2.0, dtype=np.float32)
        got = preprocessing.analytic(data)
        np.testing.assert_allclose(got.real, -2.0, atol=1e-5)
        np.testing.assert_allclose(got.imag, 0.0, atol=1e-5)


class TestPhaseStats:
    def test_matches_reference(self):
        rng = np.random.default_rng(9)
        data = rng.standard_normal((4, 128)).astype(np.float32)
        mean, var = preprocessing.phase_stats(data)
        assert mean.shape == (4,)
        assert var.shape == (4,)
        assert mean.dtype == np.float64
        assert var.dtype == np.float64
        for i, row in enumerate(data):
            phase = np.angle(_analytic_1d(row.astype(np.float64)))
            mean_sin = np.mean(np.sin(phase))
            mean_cos = np.mean(np.cos(phase))
            np.testing.assert_allclose(
                mean[i], np.arctan2(mean_sin, mean_cos), atol=1e-9)
            np.testing.assert_allclose(
                var[i], 1.0 - np.hypot(mean_sin, mean_cos), atol=1e-9)
            assert 0.0 <= var[i] <= 1.0

    def test_silent_row(self):
        data = np.zeros((2, 64), dtype=np.float32)
        data[1] = np.arange(64, dtype=np.float32)
        with np.errstate(all="raise"):
            mean, var = preprocessing.phase_stats(data)
        assert mean[0] == 0.0
        assert var[0] == 0.0
        assert np.isfinite(mean[1])
        assert np.isfinite(var[1])

    def test_negative_constant_has_pi_phase(self):
        data = np.full((1, 64), -3.0, dtype=np.float32)
        mean, var = preprocessing.phase_stats(data)
        assert np.isclose(abs(mean[0]), np.pi)
        assert np.isclose(var[0], 0.0)

    def test_undefined_mean(self):
        # Analytic phases 0 and pi cancel (up to rounding) in the circular
        # mean, so the mean direction is reported as undefined.
        data = np.array([[1.0, -1.0]], dtype=np.float32)
        mean, var = preprocessing.phase_stats(data)
        assert mean[0] == 0.0
        assert var[0] == 1.0

    def test_full_cycle_is_spread(self):
        n = 128
        data = np.cos(2.0 * np.pi * 8.0 * np.arange(n) / n)[None, :]
        mean, var = preprocessing.phase_stats(data.astype(np.float32))
        assert np.isclose(var[0], 1.0, atol=1e-6)
        assert np.isfinite(mean[0])

    def test_empty_rows_raise(self):
        with np.testing.assert_raises(ValueError):
            preprocessing.phase_stats(np.empty((2, 0), dtype=np.float32))


def test_fft_spectrum_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 128)).astype(np.float32)
    mag, freqs = utils.fft_spectrum(data, d=0.5)
    ref = np.abs(np.fft.rfft(data.astype(np.float64), axis=1))
    ref_f = np.fft.rfftfreq(128, d=0.5)
    np.testing.assert_allclose(freqs, ref_f, rtol=1e-6)
    np.testing.assert_allclose(mag, ref, rtol=1e-5, atol=1e-4)

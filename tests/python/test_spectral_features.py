"""SciPy parity for flattened spectrogram features.

These tests are optional when SciPy is unavailable, but the feature matrix
must match the NMF pipeline's definition exactly.
"""

import numpy as np
import pytest

import samcore

from conftest import H5SAMD_PATH, needs_h5samd

scipy_signal = pytest.importorskip("scipy.signal")


def _reference(X, fs, nperseg, noverlap, window, f_min, f_max, l1):
    f, t, Sxx = scipy_signal.spectrogram(
        np.asarray(X, dtype=np.float64),
        fs=fs,
        window=scipy_signal.get_window(window, nperseg, fftbins=True),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="psd",
    )
    if f_max <= 0:
        f_max = f[-1]
    keep = (f >= f_min) & (f <= f_max)
    F = Sxx[:, keep, :].reshape(Sxx.shape[0], -1)
    if l1:
        F = F / (F.sum(axis=1, keepdims=True) + 1e-12)
    return F.astype(np.float32), f[keep], t


@pytest.mark.parametrize("window", ["hann", "hamming", "cosine"])
@pytest.mark.parametrize("nperseg", [64, 128])
@pytest.mark.parametrize("overlap_fraction", [0.5, 0.75])
@pytest.mark.parametrize("l1", [False, True])
@pytest.mark.parametrize("f_max", [8e6, 15e6])
def test_matches_scipy_across_search_grid(window, nperseg, overlap_fraction,
                                          l1, f_max):
    fs = 125e6
    noverlap = int(nperseg * overlap_fraction)
    X = np.random.default_rng(1234).normal(size=(3, 512)).astype(np.float32)

    F, f, t = samcore.utils.spectrogram_features(
        X, fs, nperseg=nperseg, noverlap=noverlap, window=window,
        f_max=f_max, l1_normalize=l1)
    Fr, fr, tr = _reference(X, fs, nperseg, noverlap, window, 0.0, f_max, l1)

    assert F.shape == Fr.shape
    assert F.dtype == np.float32
    np.testing.assert_allclose(f, fr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t, tr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F, Fr, rtol=1e-4, atol=1e-8)
    if l1:
        np.testing.assert_allclose(
            F.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)


def test_canonical_nmf_config():
    fs = 125e6
    X = np.random.default_rng(99).normal(size=(4, 512)).astype(np.float32)
    F, f, t = samcore.utils.spectrogram_features(
        X, fs, nperseg=128, noverlap=96, window="cosine",
        f_max=8e6, l1_normalize=True)
    Fr, fr, tr = _reference(
        X, fs, 128, 96, "cosine", 0.0, 8e6, True)
    np.testing.assert_allclose(f, fr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t, tr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F, Fr, rtol=1e-4, atol=1e-8)
    assert F.shape == (4, fr.size * tr.size)


def test_noncontiguous_and_float64_inputs():
    fs = 20e6
    base = np.random.default_rng(7).normal(
        size=(2, 1024)).astype(np.float32)
    X_f32 = base[:, ::2]  # non-contiguous float32 view
    X_f64 = np.ascontiguousarray(X_f32).astype(np.float64)
    F32, f32, t32 = samcore.utils.spectrogram_features(
        X_f32, fs, nperseg=128, noverlap=64, window="hamming",
        f_max=5e6, l1_normalize=False)
    F64, f64, t64 = samcore.utils.spectrogram_features(
        X_f64, fs, nperseg=128, noverlap=64, window="hamming",
        f_max=5e6, l1_normalize=False)
    np.testing.assert_array_equal(f32, f64)
    np.testing.assert_array_equal(t32, t64)
    np.testing.assert_allclose(F32, F64, rtol=1e-6, atol=1e-8)


def test_nonzero_fmin_band():
    fs = 125e6
    X = np.random.default_rng(31).normal(size=(2, 512)).astype(np.float32)
    F, f, t = samcore.utils.spectrogram_features(
        X, fs, nperseg=128, noverlap=64, window="cosine",
        f_min=3e6, f_max=20e6, l1_normalize=True)
    Fr, fr, tr = _reference(
        X, fs, 128, 64, "cosine", 3e6, 20e6, True)
    np.testing.assert_allclose(f, fr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t, tr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F, Fr, rtol=1e-4, atol=1e-8)
    assert f[0] >= 3e6 and f[-1] <= 20e6


def test_empty_signal_set_and_validation():
    X = np.zeros((0, 512), dtype=np.float32)
    F, f, t = samcore.utils.spectrogram_features(X, 1e6)
    assert F.shape == (0, f.size * t.size)
    Fz, _, _ = samcore.utils.spectrogram_features(
        np.zeros((1, 512), dtype=np.float32), 1e6, l1_normalize=True)
    assert np.all(Fz == 0.0)
    with pytest.raises(ValueError, match="Unknown window"):
        samcore.utils.spectrogram_features(
            np.zeros((1, 512), dtype=np.float32), 1e6, window="boxcar")
    with pytest.raises(ValueError, match="fs must be"):
        samcore.utils.spectrogram_features(
            np.zeros((1, 512), dtype=np.float32), 0.0)


def test_dataset_method_and_samplerate_override():
    fs_mhz = 125.0
    data = np.random.default_rng(5).integers(
        -100, 100, size=(3, 512)).astype(np.int8)
    header = samcore.SAMHeader(
        scanspline=1, nlines=3, scanlen=512, samplerate=fs_mhz,
        tzero=0, resolution=1.0)
    scan = samcore.SAMScan.handler_from_data(data, header)
    ds = samcore.SAMDataset([scan], unsupervised=True)

    F, f, t = ds.spectrogram_features(
        nperseg=128, noverlap=96, window="cosine", f_max=8e6,
        l1_normalize=True, fs_mhz=fs_mhz)
    Fu, fu, tu = samcore.utils.spectrogram_features(
        ds.X, fs_mhz * 1e6, nperseg=128, noverlap=96, window="cosine",
        f_max=8e6, l1_normalize=True)
    np.testing.assert_array_equal(f, fu)
    np.testing.assert_array_equal(t, tu)
    np.testing.assert_allclose(F, Fu, rtol=1e-6, atol=1e-8)

    with pytest.raises(ValueError, match="contradicts"):
        ds.spectrogram_features(fs_mhz=100.0)


def test_default_window_is_bit_identical_on_scan_and_dataset():
    data = np.random.default_rng(8).integers(
        -100, 100, size=(4, 256)).astype(np.int8)
    header = samcore.SAMHeader(
        scanspline=1, nlines=4, scanlen=256, samplerate=100.0,
        tzero=0, resolution=1.0)
    scan = samcore.SAMScan.handler_from_data(data, header)
    ds = samcore.SAMDataset([scan], unsupervised=True)

    for fn in ("psd", "power_spectrogram", "compute_stft"):
        default = getattr(scan, fn)(nperseg=32, noverlap=16)
        explicit = getattr(scan, fn)(
            nperseg=32, noverlap=16, window="hann")
        for a, b in zip(default, explicit):
            np.testing.assert_array_equal(a, b)
        ds_default = getattr(ds, fn)(nperseg=32, noverlap=16)
        ds_explicit = getattr(ds, fn)(
            nperseg=32, noverlap=16, window="hann")
        for a, b in zip(ds_default, ds_explicit):
            np.testing.assert_array_equal(a, b)

    for fn in ("stft_peak_frequency", "stft_peak_time"):
        np.testing.assert_array_equal(
            getattr(scan, fn)(nperseg=32, noverlap=16),
            getattr(scan, fn)(nperseg=32, noverlap=16, window="hann"))
        np.testing.assert_array_equal(
            getattr(ds, fn)(nperseg=32, noverlap=16),
            getattr(ds, fn)(nperseg=32, noverlap=16, window="hann"))


@needs_h5samd
def test_legacy_dataset_accepts_explicit_fs_override():
    ds = samcore.SAMDataset.load(H5SAMD_PATH)
    assert ds.samplerate is None
    with pytest.raises(RuntimeError, match="sampling-rate metadata"):
        ds.spectrogram_features(nperseg=128, noverlap=64)
    F, f, t = ds.spectrogram_features(
        nperseg=128, noverlap=64, fs_mhz=100.0)
    assert F.shape == (ds.num_samples, f.size * t.size)
    assert np.isfinite(F).all()



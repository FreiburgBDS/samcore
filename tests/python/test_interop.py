"""NumPy ``__array__`` protocols and the lazy-torch-import contract."""

import subprocess
import sys

import numpy as np
import pytest

from samcore import SAMDataset, SAMHeader, SAMLabels, SAMScan


def _scan(n=6, sl=64):
    rng = np.random.default_rng(0)
    data = rng.integers(-100, 100, size=(n, sl)).astype(np.int8)
    header = SAMHeader(scanspline=1, nlines=n, scanlen=sl, samplerate=100.0,
                       tzero=0, resolution=1.0)
    return SAMScan.handler_from_data(data, header)


def test_scan_array_is_zero_copy_int8():
    h = _scan()
    arr = np.asarray(h)
    assert arr.dtype == np.int8
    assert arr.shape == h.data.shape
    assert np.shares_memory(arr, h.data)
    before = int(h.data[0, 0])
    arr[0, 0] = 42
    assert int(h.data[0, 0]) == 42  # writes through to the C++ buffer
    h.data[0, 0] = before


def test_scan_array_dtype_and_copy_flags():
    h = _scan()
    as_f32 = np.asarray(h, dtype=np.float32)
    assert as_f32.dtype == np.float32
    assert not np.shares_memory(as_f32, h.data)
    copied = np.array(h, copy=True)
    assert not np.shares_memory(copied, h.data)
    np.testing.assert_array_equal(copied, h.data)
    with pytest.raises(ValueError):
        np.asarray(h, dtype=np.float32, copy=False)


def test_labels_array_protocol():
    labels = SAMLabels(np.array([0, 1, -1], dtype=np.int8),
                       ["healthy", "defect"])
    arr = np.asarray(labels)
    assert arr.dtype == np.int8
    assert arr.shape == (3,)
    assert np.shares_memory(arr, labels.labels)
    arr[0] = 1
    assert int(labels.labels[0]) == 1


def test_dataset_array_protocol():
    h = _scan()
    ds = SAMDataset([h], unsupervised=True)
    arr = np.asarray(ds)
    assert arr.dtype == np.float32
    assert arr.shape == ds.X.shape
    assert np.shares_memory(arr, ds.X)


def test_import_samcore_does_not_import_torch():
    code = ("import sys, samcore; "
            "assert 'torch' not in sys.modules, "
            "'torch must not be imported by samcore'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_interop_module_is_importable_without_torch():
    from samcore import interop
    assert callable(interop.tensor)
    assert hasattr(interop, "TorchDataset")
    assert hasattr(interop, "TorchDataLoader")

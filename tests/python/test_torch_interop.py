"""PyTorch interop tests (the whole module skips when torch is absent)."""

import numpy as np
import pytest

from samcore import SAMDataset, SAMHeader, SAMLabels, SAMScan

torch = pytest.importorskip("torch")

from samcore.interop import TorchDataLoader, TorchDataset, tensor  # noqa: E402


def _scan(n=6, sl=64, labels=None):
    rng = np.random.default_rng(0)
    data = rng.integers(-100, 100, size=(n, sl)).astype(np.int8)
    header = SAMHeader(scanspline=1, nlines=n, scanlen=sl, samplerate=100.0,
                       tzero=0, resolution=1.0)
    return SAMScan.handler_from_data(data, header, samlabels=labels)


def _dataset():
    h1 = _scan(4, 32, SAMLabels(np.array([0, 1, 0, 1], dtype=np.int8),
                                ["healthy", "defect"]))
    h2 = _scan(4, 32, SAMLabels(np.array([1, 0, 1, 0], dtype=np.int8),
                                ["healthy", "defect"]))
    ds = SAMDataset([h1, h2])
    ds.train_test_split(test_size=0.5, random_state=0)
    return ds


def test_tensor_is_zero_copy():
    h = _scan()
    t = tensor(h)
    assert t.dtype == torch.int8
    assert tuple(t.shape) == h.data.shape
    assert t.data_ptr() == h.data.ctypes.data


def test_tensor_dtype_and_dataset():
    ds = _dataset()
    assert tensor(ds).data_ptr() == ds.X.ctypes.data
    t = tensor(ds, dtype=torch.float64)
    assert t.dtype == torch.float64
    assert tuple(t.shape) == ds.X.shape


def test_torch_dataset_supervised_items():
    ds = _dataset()
    td = TorchDataset(ds, split="train")
    assert len(td) == len(ds.train_indices)
    x, y, meta = td[0]
    assert x.dtype == torch.float32
    assert tuple(x.shape) == (ds.maxlen,)
    assert y.dtype == torch.int8
    assert int(y) == int(ds.labels.labels[int(ds.train_indices[0])])
    assert set(meta) == {"idx", "x", "y"}
    assert td.dataset is ds


def test_torch_dataloader_collates():
    ds = _dataset()
    loader = TorchDataLoader(ds, split="train", batch_size=2, shuffle=False)
    x, y, meta = next(iter(loader))
    assert tuple(x.shape) == (2, ds.maxlen)
    assert tuple(y.shape) == (2,)
    assert set(meta) == {"idx", "x", "y"}
    assert tuple(meta["x"].shape) == (2,)


def test_torch_dataset_unsupervised_and_no_spatial():
    h = _scan()
    ds = SAMDataset([h], unsupervised=True)
    td = TorchDataset(ds, split="train")
    x, meta = td[0]
    assert tuple(x.shape) == (ds.maxlen,)
    assert set(meta) == {"idx", "x", "y"}
    td2 = TorchDataset(ds, split="train", with_spatial=False)
    assert len(td2[0]) == 1


def test_torch_dataset_use_z_and_validation():
    ds = _dataset()
    ds.transform(lambda d: d[:, :5])
    td = TorchDataset(ds, split="train", use_z=True, with_spatial=False)
    x, y = td[0]
    assert tuple(x.shape) == (5,)

    plain = SAMDataset([_scan()], unsupervised=True)
    with pytest.raises(RuntimeError):
        TorchDataset(plain, split="train", use_z=True)
    with pytest.raises(ValueError):
        TorchDataset(plain, split="bogus")

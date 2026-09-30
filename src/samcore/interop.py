"""Optional interoperability helpers (NumPy and PyTorch).

PyTorch is imported lazily inside the functions, so ``import samcore`` never
requires it.  CPU tensors created here share memory with the samcore buffers
(zero-copy) whenever no dtype conversion or device transfer is needed.

Keras/TensorFlow users can wrap :meth:`SAMDataset.cube_batches` (which yields
``(1, H, W, C)`` grids) with ``tf.data.Dataset.from_generator``; no
TensorFlow-specific code is needed here.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from samcore._samcore import SAMDataset

__all__ = ["TorchDataLoader", "TorchDataset", "tensor"]


def _torch() -> Any:
    """Import torch lazily with a helpful error message."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required for samcore.interop; install it with "
            "`pip install torch`."
        ) from exc
    return torch


def tensor(obj: Any, *, dtype: Any = None, device: Any = None) -> Any:
    """Convert a scan/labels/dataset (or any array-like) to a torch tensor.

    Parameters
    ----------
    obj : SAMScan, SAMLabels, SAMDataset or array-like
        Converted through ``np.asarray`` (the ``__array__`` protocol), so a
        CPU tensor shares memory with the samcore buffer unless a dtype
        conversion is requested.
    dtype : torch dtype or numpy dtype, optional
        Target dtype.
    device : torch device, optional
        Target device; anything but CPU copies the data.

    Returns
    -------
    torch.Tensor
    """
    torch = _torch()
    array = np.asarray(obj)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else None
    if torch_dtype is None and dtype is not None:
        array = array.astype(dtype, copy=False)
    result = torch.from_numpy(np.ascontiguousarray(array))
    if torch_dtype is not None:
        result = result.to(torch_dtype)
    if device is not None:
        result = result.to(device)
    return result


class TorchDataset:
    """A ``torch.utils.data.Dataset`` over one :class:`SAMDataset` split.

    Each item is ``(x, y, meta)`` for supervised datasets and ``(x, meta)``
    for unsupervised ones.  ``meta`` is a dict of Python scalars
    (``idx``, ``x``, ``y``) so torch's default collate builds a dict of
    tensors without needing recarray support.  With ``with_spatial=False``
    the meta dict is omitted, yielding ``(x, y)`` / ``(x,)``.

    Parameters
    ----------
    dataset : SAMDataset
        Source dataset (not modified).
    split : str, optional
        ``'train'`` (default) or ``'test'``.
    use_z : bool or None, optional
        Use the feature matrix ``Z`` instead of ``X``; None auto-detects.
    with_spatial : bool, optional
        Include the provenance dict in each item.  Default True.
    """

    def __init__(self, dataset: SAMDataset, split: str = "train",
                 use_z: Optional[bool] = None,
                 with_spatial: bool = True) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'.")
        if use_z and dataset.Z is None:
            raise RuntimeError(
                "Z has not been built yet. Call transform() first.")
        if use_z is None:
            use_z = dataset.Z is not None
        data = dataset.Z if use_z else dataset.X
        if data is None:
            raise RuntimeError(
                f"{'Z' if use_z else 'X'} is not available in the dataset.")
        self.dataset = dataset
        self.split = split
        self.use_z = use_z
        self.with_spatial = with_spatial
        self._indices = np.asarray(
            dataset.train_indices if split == "train"
            else dataset.test_indices)
        self._data = data
        if dataset.unsupervised:
            self._labels: Optional[np.ndarray] = None
        else:
            assert dataset.labels is not None
            self._labels = np.asarray(dataset.labels.labels)
        self._spatial = dataset.spatial if with_spatial else None

    def __len__(self) -> int:
        """Number of samples in the split."""
        return int(self._indices.shape[0])

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        """Return ``(x, y, meta)`` / ``(x, meta)`` / ``(x, y)`` / ``(x,)``."""
        torch = _torch()
        row = int(self._indices[index])
        x = torch.from_numpy(np.ascontiguousarray(self._data[row]))
        meta: Optional[Dict[str, Any]] = None
        if self._spatial is not None:
            meta = {
                "idx": int(self._spatial["idx"][row]),
                "x": float(self._spatial["x"][row]),
                "y": float(self._spatial["y"][row]),
            }
        if self._labels is None:
            return (x, meta) if meta is not None else (x,)
        y = torch.tensor(int(self._labels[row]), dtype=torch.int8)
        return (x, y, meta) if meta is not None else (x, y)


def TorchDataLoader(dataset: SAMDataset, *, split: str = "train",
                    batch_size: int = 64, shuffle: Optional[bool] = None,
                    num_workers: int = 0, use_z: Optional[bool] = None,
                    with_spatial: bool = True, **kwargs: Any) -> Any:
    """Create a ``torch.utils.data.DataLoader`` over a SAMDataset split.

    Parameters
    ----------
    dataset : SAMDataset
        Source dataset.
    split : str, optional
        ``'train'`` (default) or ``'test'``.
    batch_size : int, optional
        Samples per batch.  Default 64.
    shuffle : bool or None, optional
        Defaults to True for the train split and False for test.
    num_workers : int, optional
        DataLoader worker processes.  Default 0.
    use_z : bool or None, optional
        Use ``Z`` instead of ``X``; None auto-detects.
    with_spatial : bool, optional
        Include the provenance dict in each batch.  Default True.
    **kwargs
        Forwarded to ``torch.utils.data.DataLoader``.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    torch = _torch()
    torch_dataset = TorchDataset(dataset, split=split, use_z=use_z,
                                 with_spatial=with_spatial)
    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers, **kwargs)

"""samcore._dataset - SAMDataset Python-side API layer.

Adds splits, batch/patch iteration and exports on top of the C++
``SAMDataset``.  The functions defined here are attached to the C++ class
at import time, so ``samcore.SAMDataset`` is the one and only
``SAMDataset`` type.  Their annotations are real (not postponed) so that
nanobind's stubgen, which runs against the assembled package at build
time, renders fully-typed, documented members inside the generated
``_samcore.pyi``.
"""

from typing import Callable, Iterator, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from samcore._samcore import SAMDataset

# Batch shapes yielded by the iterators.  Supervised datasets yield
# (X, y, spatial); unsupervised datasets yield (X, spatial).
_Batch = Union[
    Tuple[NDArray[np.float32], NDArray[np.int8], np.recarray],
    Tuple[NDArray[np.float32], np.recarray],
]
_CubeBatch = Union[
    Tuple[NDArray[np.float32], NDArray[np.int8]],
    Tuple[NDArray[np.float32]],
]


def preprocess(self: SAMDataset, strategy: str, **kwargs: object) -> SAMDataset:
    """Apply a built-in preprocessing strategy to ``X`` in place.

    Parameters
    ----------
    strategy : str
        One of ``'lp'``, ``'bp'``, ``'normalize'``, ``'savgol'``,
        ``'medfilt'``, ``'gate'``, ``'detrend'``, ``'envelope'``,
        ``'moving_average'``.
    **kwargs
        Forwarded to the strategy:

        - ``'lp'``: ``cutoff`` (float), ``fs`` (float)
        - ``'bp'``: ``cutoff_low`` (float), ``cutoff_high`` (float),
          ``fs`` (float)
        - ``'normalize'``: ``mode`` (str, default ``'minmax'``, one of
          ``'max'``, ``'zscore'``, ``'minmax'``)
        - ``'savgol'``: ``window_length`` (int, default 5),
          ``polyorder`` (int, default 2)
        - ``'medfilt'``: ``kernel_size`` (int, default 3)
        - ``'gate'``: ``start`` (int, default 0), ``end`` (int, default 0)
        - ``'detrend'``: none
        - ``'envelope'``: none
        - ``'moving_average'``: ``window`` (int, default 5)

    Returns
    -------
    SAMDataset
        Self, for chaining.
    """
    self._preprocess(  # type: ignore[attr-defined]
        strategy,
        cutoff=kwargs.get("cutoff", 0.0),
        cutoff_low=kwargs.get("cutoff_low", 0.0),
        cutoff_high=kwargs.get("cutoff_high", 0.0),
        fs=kwargs.get("fs", 0.0),
        mode=kwargs.get("mode", "max"),
        window_length=kwargs.get("window_length", 5),
        polyorder=kwargs.get("polyorder", 2),
        kernel_size=kwargs.get("kernel_size", 3),
        start=kwargs.get("start", 0),
        end=kwargs.get("end", 0),
        window=kwargs.get("window", 5),
    )
    return self


def transform(self: SAMDataset,
              fn: Callable[[NDArray[np.float32]], NDArray[np.float32]]
              ) -> NDArray[np.float32]:
    """Build a feature matrix ``Z`` by applying ``fn`` to ``X``.

    Parameters
    ----------
    fn : callable
        ``fn(data_2d) -> array``.  Receives the full padded signal array
        and must return a 1-D or 2-D array whose first dimension matches
        ``len(self)``.

    Returns
    -------
    ndarray (float32)
        The feature matrix ``Z`` (float32, at least 2-D).
    """
    Z = fn(self.X)
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
    if Z.shape[0] == 1:
        Z = Z.T
    self.Z = Z
    return Z


def train_test_split(self: SAMDataset, test_size: float = 0.2,
                     random_state: Optional[int] = None,
                     shuffle: bool = True) -> None:
    """Randomly split into train and test sets without stratification.

    Use for balanced datasets or when stratification is unnecessary (e.g.
    unsupervised pre-training, or when labels are not the primary concern).

    Parameters
    ----------
    test_size : float, optional
        Fraction of samples to allocate to the test set (0 < t < 1).
    random_state : int or None, optional
        Seed for reproducible shuffling.
    shuffle : bool, optional
        Whether to shuffle indices before splitting.  Default True.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    total = self.X.shape[0]
    indices = np.arange(total)
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(total * (1 - test_size))
    self.train_indices = indices[:split_idx]
    self.test_indices = indices[split_idx:]
    self.shuffled = shuffle


def stratified_train_test_split(self: SAMDataset, test_size: float = 0.2,
                                random_state: Optional[int] = None) -> None:
    """Randomly split while preserving the class distribution.

    Use when classes are imbalanced and each split must contain a
    proportional share of every class.

    Requires a supervised dataset.

    Parameters
    ----------
    test_size : float, optional
        Fraction of samples to allocate to the test set (0 < t < 1).
    random_state : int or None, optional
        Seed for reproducible shuffling.
    """
    self._require_labels()  # type: ignore[attr-defined]
    assert self.labels is not None
    self.train_indices, self.test_indices = self.labels.stratified_split(
        test_size=test_size, random_state=random_state)
    self.shuffled = True


def stratified_split_by_cube(self: SAMDataset, test_size: float = 0.2,
                             random_state: Optional[int] = None) -> None:
    """Split into train and test sets preserving per-cube proportions.

    Each source cube contributes the same fraction (``test_size``) of its
    signals to the test set.  This is the per-cube analogue of
    :meth:`stratified_train_test_split` and the recommended split for
    unsupervised datasets.  It also works on supervised datasets:
    stratification is by spatial provenance (cube index), not by label.

    Parameters
    ----------
    test_size : float, optional
        Fraction of each cube's signals to allocate to the test set
        (0 < t < 1).  Defaults to 0.2.
    random_state : int or None, optional
        Seed for reproducible shuffling.

    Notes
    -----
    Each cube contributes ``max(1, int(n_cube * test_size))`` test signals
    so small cubes always have at least one test sample.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    cube_ids = self.spatial["idx"]
    for ci in np.unique(cube_ids):
        idx = np.where(cube_ids == ci)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    # train_indices/test_indices are copy-returning properties, so the
    # shuffle must happen on local arrays before assignment.
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(test)
    self.train_indices = train
    self.test_indices = test
    self.shuffled = True


def split_by_label(self: SAMDataset, label: Union[int, str],
                   test_size: Optional[float] = None) -> None:
    """Isolate a specific label class into the test set (one-vs-rest).

    All samples matching ``label`` become the test set, everything else
    becomes the training set.  Useful for anomaly detection where one class
    represents the anomaly of interest.

    Requires a supervised dataset.

    Parameters
    ----------
    label : int or str
        Label value or name to isolate in the test set.
    test_size : float or None, optional
        If given, sample only this fraction of the matching class (useful
        when the target class is very large).  If None, all matching
        samples are placed in the test set.
    """
    self._require_labels()  # type: ignore[attr-defined]
    assert self.labels is not None
    if isinstance(label, str):
        if not self.labels.has_name(label):
            raise ValueError(
                f"Label name {label!r} not found in dataset labels "
                f"({self.labels.label_names}).")
        label_val = self.labels.name_to_value(label)
    elif isinstance(label, int):
        label_val = label
    else:
        raise TypeError("label must be int or str.")
    matching = np.where(np.asarray(self.labels.labels) == label_val)[0]
    if test_size is not None:
        if not (0 < test_size <= 1):
            raise ValueError("test_size must be between 0 and 1.")
        rng = np.random.default_rng()
        n = max(1, int(len(matching) * test_size))
        n = min(n, len(matching))
        test_idx = rng.choice(matching, size=n, replace=False)
    else:
        test_idx = matching
    self.test_indices = test_idx
    self.train_indices = np.setdiff1d(np.arange(len(self)), test_idx)
    self.shuffled = test_size is not None


def batches(self: SAMDataset, split: str = "train", shuffle: bool = True,
            seed: Optional[int] = None, use_z: Optional[bool] = None,
            batch_size: int = 64) -> Iterator[_Batch]:
    """Yield minibatches of ``(X, y, spatial)`` or ``(X, spatial)``.

    When the dataset is supervised each batch is a 3-tuple
    ``(X_batch, y_batch, spatial_batch)``; unsupervised datasets yield a
    2-tuple ``(X_batch, spatial_batch)``.

    Parameters
    ----------
    split : str, optional
        Which split to iterate: ``'train'`` or ``'test'``.
    shuffle : bool, optional
        Shuffle the split indices before batching.  Set to False for
        deterministic iteration order.
    seed : int or None, optional
        Seed for reproducible shuffling.  Ignored when ``shuffle=False``.
    use_z : bool or None, optional
        If True, use the feature matrix ``Z`` instead of ``X``.  If None
        (default), auto-detects: uses ``Z`` when available (after
        :meth:`transform`), otherwise falls back to ``X``.
    batch_size : int, optional
        Number of samples per batch.  Default 64.

    Yields
    ------
    tuple
        ``(X, y, spatial)`` for supervised datasets;
        ``(X, spatial)`` for unsupervised datasets.
    """
    if use_z and self.Z is None:
        raise RuntimeError("Z has not been built yet. Call transform() first.")
    indices = (self.train_indices.copy()
               if split == "train" else self.test_indices.copy())
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    if use_z is None:
        use_z = self.Z is not None
    data = self.Z if use_z else self.X
    if data is None:
        raise RuntimeError(
            f"{'Z' if use_z else 'X'} is not available in the dataset.")
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        Xb = data[batch_idx]
        if self.unsupervised:
            yield Xb, cast(np.recarray, self.spatial[batch_idx])
        else:
            assert self.labels is not None
            yield Xb, np.asarray(self.labels.labels)[batch_idx], \
                cast(np.recarray, self.spatial[batch_idx])


def cube_batches(self: SAMDataset, use_z: Optional[bool] = None,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> Iterator[_CubeBatch]:
    """Yield one entire scan cube at a time as a 2-D spatial grid.

    Each cube is shaped for direct use with 2-D CNNs:
    ``(1, H, W, features)`` where H = nlines, W = scanspline, and features
    is the scan length (or the Z dimension when ``use_z=True``).

    Parameters
    ----------
    use_z : bool or None, optional
        If True, use ``Z``; if None (default), auto-detects.
    shuffle : bool, optional
        Shuffle cube order.  Default True.
    seed : int or None, optional
        Seed for reproducible shuffling.

    Yields
    ------
    tuple
        ``(X_cube, y_cube)`` for supervised datasets, where X_cube has
        shape ``(1, H, W, features)`` and y_cube is a ``(1, H, W)`` label
        grid (batch dimension included for direct use with segmentation
        losses).  ``(X_cube,)`` for unsupervised datasets.
    """
    if use_z and self.Z is None:
        raise RuntimeError("Z has not been built yet. Call transform() first.")
    if use_z is None:
        use_z = self.Z is not None
    n_cubes = len(self.cube_shapes)
    order = np.arange(n_cubes)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    getter = self.get_cube_Z if use_z else self.get_cube_X
    for i in order:
        cube = getter(int(i))
        X = cube[np.newaxis, ...]
        if self.unsupervised:
            yield (X,)
        else:
            assert self.labels is not None
            y = self.get_cube_labels(int(i))
            yield X, y[np.newaxis, ...]


def spatial_patches(
        self: SAMDataset, patch_size: Tuple[int, int] = (32, 32),
        stride: Tuple[int, int] = (16, 16), use_z: Optional[bool] = None,
        shuffle: bool = True, seed: Optional[int] = None
) -> Iterator[_CubeBatch]:
    """Yield spatial patches across all cubes preserving the 2-D layout.

    Each patch comes from within a single cube -- patches never cross cube
    boundaries.

    Parameters
    ----------
    patch_size : tuple of int, optional
        Patch height and width ``(ph, pw)`` in pixels.
    stride : tuple of int, optional
        Vertical and horizontal stride ``(sh, sw)``.
    use_z : bool or None, optional
        If True, use ``Z``; if None (default), auto-detects.
    shuffle : bool, optional
        Shuffle patch order.  Default True.
    seed : int or None, optional
        Seed for reproducible shuffling.

    Yields
    ------
    tuple
        ``(X_patch, y_patch)`` for supervised datasets, where X_patch is
        ``(ph, pw, features)`` and y_patch is ``(ph, pw)``.
        ``(X_patch,)`` for unsupervised datasets.
    """
    if use_z and self.Z is None:
        raise RuntimeError("Z has not been built yet. Call transform() first.")
    if use_z is None:
        use_z = self.Z is not None
    ph, pw = patch_size
    sh, sw = stride
    positions = []
    for i, (nlines, cols) in enumerate(self.cube_shapes):
        h_patches = max(0, (nlines - ph) // sh + 1)
        w_patches = max(0, (cols - pw) // sw + 1)
        for r in range(h_patches):
            for c in range(w_patches):
                positions.append((i, r * sh, c * sw))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(positions)
    getter = self.get_cube_Z if use_z else self.get_cube_X
    for idx, row, col in positions:
        cube = getter(idx)
        Xp = cube[row:row + ph, col:col + pw, :]
        if self.unsupervised:
            yield (Xp,)
        else:
            y_cube = self.get_cube_labels(idx)
            yp = y_cube[row:row + ph, col:col + pw]
            yield Xp, yp


def to_numpy(self: SAMDataset, split: str = "train",
             use_z: Optional[bool] = None) -> _Batch:
    """Return the split data as numpy arrays.

    Parameters
    ----------
    split : str, optional
        ``'train'`` or ``'test'``.
    use_z : bool or None, optional
        If True, use ``Z``; if None (default), auto-detects.

    Returns
    -------
    tuple
        ``(X, y, spatial)`` for supervised datasets;
        ``(X, spatial)`` for unsupervised datasets.
    """
    indices = self.train_indices if split == "train" else self.test_indices
    if use_z is None:
        use_z = self.Z is not None
    data = self.Z if use_z else self.X
    if data is None:
        raise RuntimeError(
            f"{'Z' if use_z else 'X'} is not available in the dataset.")
    if self.unsupervised:
        return data[indices].copy(), \
            cast(np.recarray, self.spatial[indices].copy())
    assert self.labels is not None
    return (data[indices].copy(),
            np.asarray(self.labels.labels)[indices].copy(),
            cast(np.recarray, self.spatial[indices].copy()))


def to_dict(self: SAMDataset, split: str = "train",
            use_z: Optional[bool] = None) -> dict:
    """Return the split data as a dict.

    Parameters
    ----------
    split : str, optional
        ``'train'`` or ``'test'``.
    use_z : bool or None, optional
        If True, use ``Z``; if None (default), auto-detects.

    Returns
    -------
    dict
        Keys: ``data``, ``labels``, ``label_names``, ``handler_ids``,
        ``spatial`` (supervised); ``data``, ``handler_ids``, ``spatial``
        (unsupervised).
    """
    indices = self.train_indices if split == "train" else self.test_indices
    if use_z is None:
        use_z = self.Z is not None
    data = self.Z if use_z else self.X
    if data is None:
        raise RuntimeError(
            f"{'Z' if use_z else 'X'} is not available in the dataset.")
    if self.unsupervised:
        return {
            "data": data[indices].copy(),
            "handler_ids": self.spatial["idx"][indices].copy(),
            "spatial": self.spatial[indices].copy(),
        }
    assert self.labels is not None
    return {
        "data": data[indices].copy(),
        "labels": np.asarray(self.labels.labels)[indices].copy(),
        "label_names": list(self.labels.label_names).copy(),
        "handler_ids": self.spatial["idx"][indices].copy(),
        "spatial": self.spatial[indices].copy(),
    }


def relabel(self: SAMDataset,
            mapping: dict[Union[int, str], Union[int, str]]) -> None:
    """Remap the dataset labels according to ``mapping``.

    Requires a supervised dataset.

    Parameters
    ----------
    mapping : dict
        Old label value or name to new value or name.
    """
    self._require_labels()  # type: ignore[attr-defined]
    self._relabel(mapping)  # type: ignore[attr-defined]


def num_train_batches(self: SAMDataset, batch_size: int = 64) -> int:
    """Number of training batches of the given size."""
    return int(np.ceil(len(self.train_indices) / batch_size))


def num_test_batches(self: SAMDataset, batch_size: int = 64) -> int:
    """Number of test batches of the given size."""
    return int(np.ceil(len(self.test_indices) / batch_size))


def _require_labels(self: SAMDataset) -> None:
    """Raise when the dataset is unsupervised and labels are required."""
    if self.labels is None:
        raise RuntimeError(
            "This dataset is unsupervised. Labels are not available.")


def __len__(self: SAMDataset) -> int:
    """Number of signals in the dataset."""
    return self.X.shape[0]


def __iter__(self: SAMDataset) -> Iterator[_Batch]:
    """Iterate over the training batches."""
    return iter(self.batches("train"))


# Attach the convenience API to the C++ class.  Setting __module__ to
# "samcore._samcore" makes nanobind's stubgen render these members inside
# the generated class stub; the module-level names are deleted afterwards
# so the stub of this module stays clean.  ``SAMDataset`` is deleted from
# this module's namespace as well so that no reference cycle remains
# (class -> patched function -> module globals -> class); reference counting
# alone then releases everything at interpreter shutdown on every platform.
_PATCHED = (
    preprocess, transform, train_test_split, stratified_train_test_split,
    stratified_split_by_cube, split_by_label, batches, cube_batches,
    spatial_patches, to_numpy, to_dict, relabel, num_train_batches,
    num_test_batches, _require_labels, __len__, __iter__,
)
for _fn in _PATCHED:
    setattr(SAMDataset, _fn.__name__, _fn)
    _fn.__module__ = "samcore._samcore"
    # Materialize the annotations (PEP 649 on Python >= 3.14 defers them to
    # a lazy __annotate__ closure), so they stay valid after the class name
    # is deleted from this module's namespace below.
    _fn.__annotations__ = _fn.__annotations__

del (_PATCHED, _fn, preprocess, transform, train_test_split,
     stratified_train_test_split, stratified_split_by_cube, split_by_label,
     batches, cube_batches, spatial_patches, to_numpy, to_dict, relabel,
     num_train_batches, num_test_batches, _require_labels, __len__,
     __iter__, SAMDataset)
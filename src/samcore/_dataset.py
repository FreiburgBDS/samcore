# samcore._dataset - SAMDataset python-side API layer (spatial recarray,
# splits, batch/patch iteration, exports).

import numpy as np

from samcore._samcore import SAMDataset

_SPATIAL_DTYPE = np.dtype([("idx", "i4"), ("x", "f4"), ("y", "f4")])


def _dataset_train_indices_get(self):
    return np.asarray(self._train_indices, dtype=np.int64)


def _dataset_train_indices_set(self, value):
    self._train_indices = np.asarray(value, dtype=np.int64).tolist()


def _dataset_test_indices_get(self):
    return np.asarray(self._test_indices, dtype=np.int64)


def _dataset_test_indices_set(self, value):
    self._test_indices = np.asarray(value, dtype=np.int64).tolist()


def _dataset_spatial(self):
    idx, x, y = self._spatial_arrays()
    block = np.empty(len(idx), dtype=_SPATIAL_DTYPE)
    block["idx"] = idx
    block["x"] = x
    block["y"] = y
    return block.view(np.recarray)


def _dataset_preprocess(self, strategy, **kwargs):
    self._preprocess(
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


def _dataset_transform(self, fn):
    Z = fn(self.X)
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float32))
    if Z.shape[0] == 1:
        Z = Z.T
    self.Z = Z
    return self.Z


def _dataset_train_test_split(self, test_size=0.2, random_state=None,
                              shuffle=True):
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    total = self.X.shape[0]
    indices = np.arange(total)
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(indices)
        self.shuffled = True
    split_idx = int(total * (1 - test_size))
    self.train_indices = indices[:split_idx]
    self.test_indices = indices[split_idx:]


def _dataset_stratified_train_test_split(self, test_size=0.2,
                                         random_state=None):
    self._require_labels()
    self.train_indices, self.test_indices = self.labels.stratified_split(
        test_size=test_size, random_state=random_state)
    self.shuffled = False


def _dataset_stratified_split_by_cube(self, test_size=0.2, random_state=None):
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
    self.train_indices = np.concatenate(train_parts)
    self.test_indices = np.concatenate(test_parts)
    rng.shuffle(self.train_indices)
    rng.shuffle(self.test_indices)
    self.shuffled = False


def _dataset_split_by_label(self, label, test_size=None):
    self._require_labels()
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


def _dataset_batches(self, split="train", shuffle=True, seed=None,
                     use_z=None, batch_size=64):
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
            yield Xb, self.spatial[batch_idx]
        else:
            yield Xb, np.asarray(self.labels.labels)[batch_idx], \
                self.spatial[batch_idx]


def _dataset_cube_batches(self, use_z=None, shuffle=True, seed=None):
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
            y = self.get_cube_labels(int(i))
            yield X, y[np.newaxis, ...]


def _dataset_spatial_patches(self, patch_size=(32, 32), stride=(16, 16),
                             use_z=None, shuffle=True, seed=None):
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


def _dataset_to_numpy(self, split="train", use_z=None):
    indices = self.train_indices if split == "train" else self.test_indices
    if use_z is None:
        use_z = self.Z is not None
    data = self.Z if use_z else self.X
    if data is None:
        raise RuntimeError(
            f"{'Z' if use_z else 'X'} is not available in the dataset.")
    if self.unsupervised:
        return data[indices].copy(), self.spatial[indices].copy()
    return (data[indices].copy(),
            np.asarray(self.labels.labels)[indices].copy(),
            self.spatial[indices].copy())


def _dataset_to_dict(self, split="train", use_z=None):
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
    return {
        "data": data[indices].copy(),
        "labels": np.asarray(self.labels.labels)[indices].copy(),
        "label_names": list(self.labels.label_names).copy(),
        "handler_ids": self.spatial["idx"][indices].copy(),
        "spatial": self.spatial[indices].copy(),
    }


def _dataset_require_labels(self):
    if self.labels is None:
        raise RuntimeError(
            "This dataset is unsupervised. Labels are not available.")


def _dataset_relabel(self, mapping):
    self._require_labels()
    self._relabel(mapping)


def _dataset_num_train_batches(self, batch_size=64):
    return int(np.ceil(len(self.train_indices) / batch_size))


def _dataset_num_test_batches(self, batch_size=64):
    return int(np.ceil(len(self.test_indices) / batch_size))


def _dataset_dataset_label_names(self):
    self._require_labels()
    return list(self.labels.label_names)


def _dataset_handler_ids(self):
    return self.spatial["idx"]


def _dataset_cube_counts(self):
    return {int(i): int(n * c)
            for i, (n, c) in enumerate(self.cube_shapes)}


def _dataset_convert_from_paths(cls, input_paths, output_path, pad_value=0.0,
                                unsupervised=None):
    from samcore._io import io
    io.convert_h5sam_to_h5samd(list(input_paths), output_path, pad_value,
                               unsupervised)
    return cls.load(output_path)


SAMDataset.train_indices = property(_dataset_train_indices_get,
                                    _dataset_train_indices_set)
SAMDataset.test_indices = property(_dataset_test_indices_get,
                                   _dataset_test_indices_set)
SAMDataset.shuffled = property(lambda self: self._shuffled,
                               lambda self, v: setattr(self, "_shuffled",
                                                       bool(v)))
SAMDataset.spatial = property(_dataset_spatial)
SAMDataset.preprocess = _dataset_preprocess
SAMDataset.transform = _dataset_transform
SAMDataset.train_test_split = _dataset_train_test_split
SAMDataset.stratified_train_test_split = _dataset_stratified_train_test_split
SAMDataset.stratified_split_by_cube = _dataset_stratified_split_by_cube
SAMDataset.split_by_label = _dataset_split_by_label
SAMDataset.batches = _dataset_batches
SAMDataset.cube_batches = _dataset_cube_batches
SAMDataset.spatial_patches = _dataset_spatial_patches
SAMDataset.to_numpy = _dataset_to_numpy
SAMDataset.to_dict = _dataset_to_dict
SAMDataset.relabel = _dataset_relabel
SAMDataset.num_train_batches = _dataset_num_train_batches
SAMDataset.num_test_batches = _dataset_num_test_batches
SAMDataset.dataset_label_names = property(_dataset_dataset_label_names)
SAMDataset.handler_ids = property(_dataset_handler_ids)
SAMDataset.cube_counts = property(_dataset_cube_counts)
SAMDataset._require_labels = _dataset_require_labels
SAMDataset.convert_from_paths = classmethod(_dataset_convert_from_paths)
SAMDataset.__len__ = lambda self: self.X.shape[0]
SAMDataset.__iter__ = lambda self: iter(self.batches("train"))

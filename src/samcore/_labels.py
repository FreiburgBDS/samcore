# samcore._labels - SAMLabels python-side API layer (numpy-typed masks,
# class constants, stratified split).

import numpy as np

from samcore._samcore import SAMLabels

# reserved label values / names (kept for API compatibility)
LABEL_UNLABELED = -1
LABEL_NAME_UNLABELED = "unlabeled"
LABEL_HEALTHY = 0
LABEL_NAME_HEALTHY = "healthy"


def _labels_healthy_mask(self):
    return np.asarray(self._healthy_mask(), dtype=bool)


def _labels_labeled_mask(self):
    return np.asarray(self._labeled_mask(), dtype=bool)


def _labels_unlabeled_mask(self):
    return np.asarray(self._unlabeled_mask(), dtype=bool)


def _labels_mask(self, label):
    return np.asarray(self._mask(label), dtype=bool)


def _labels_stratified_split(self, test_size=0.2, random_state=None):
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    rng = np.random.default_rng(random_state)
    labels = np.asarray(self.labels)
    labeled_idx = np.where(labels != LABEL_UNLABELED)[0]
    unlabeled_idx = np.where(labels == LABEL_UNLABELED)[0]
    train_parts = [np.array([], dtype=np.int64)]
    test_parts = [np.array([], dtype=np.int64)]
    if len(labeled_idx) > 0:
        y_lab = labels[labeled_idx]
        for cls in np.unique(y_lab):
            cls_where = labeled_idx[y_lab == cls]
            rng.shuffle(cls_where)
            n_test = max(1, int(len(cls_where) * test_size))
            test_parts.append(cls_where[:n_test])
            train_parts.append(cls_where[n_test:])
    if len(unlabeled_idx) > 0:
        rng.shuffle(unlabeled_idx)
        n_test = max(1, int(len(unlabeled_idx) * test_size))
        test_parts.append(unlabeled_idx[:n_test])
        train_parts.append(unlabeled_idx[n_test:])
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


SAMLabels.LABEL_UNLABELED = LABEL_UNLABELED
SAMLabels.LABEL_NAME_UNLABELED = LABEL_NAME_UNLABELED
SAMLabels.LABEL_HEALTHY = LABEL_HEALTHY
SAMLabels.LABEL_NAME_HEALTHY = LABEL_NAME_HEALTHY
SAMLabels.healthy_mask = _labels_healthy_mask
SAMLabels.labeled_mask = _labels_labeled_mask
SAMLabels.unlabeled_mask = _labels_unlabeled_mask
SAMLabels.mask = _labels_mask
SAMLabels.stratified_split = _labels_stratified_split

_labels_relabel_impl = SAMLabels.relabel


def _labels_relabel(self, mapping):
    try:
        _labels_relabel_impl(self, mapping)
    except IndexError as e:
        raise KeyError(str(e)) from None


SAMLabels.relabel = _labels_relabel

_labels_num_classes = SAMLabels.num_classes
_labels_unique_labels = SAMLabels.unique_labels
SAMLabels.num_classes = property(lambda self: _labels_num_classes(self))
SAMLabels.unique_labels = property(lambda self: _labels_unique_labels(self))

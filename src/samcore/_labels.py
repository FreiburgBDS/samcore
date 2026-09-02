"""samcore._labels - SAMLabels Python-side API layer.

Adds numpy-typed mask helpers, class constants and a stratified split on
top of the C++ ``SAMLabels`` (the C++ class also provides ``relabel``, with
unknown names raising ``KeyError``).  The functions defined here are
attached to the C++ class at import time, so ``samcore.SAMLabels`` is the
one and only ``SAMLabels`` type.  Their annotations are real (not
postponed) so that nanobind's stubgen, which runs against the assembled
package at build time, renders fully-typed, documented members inside the
generated ``_samcore.pyi``.
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from samcore._samcore import SAMLabels

# reserved label values / names (kept for API compatibility)
LABEL_UNLABELED = -1
LABEL_NAME_UNLABELED = "unlabeled"
LABEL_HEALTHY = 0
LABEL_NAME_HEALTHY = "healthy"


def healthy_mask(self: SAMLabels) -> NDArray[np.bool_]:
    """Boolean mask of the healthy (0) scans.

    Returns
    -------
    ndarray (bool)
        Mask of shape (n_signals,).
    """
    return self.labels == LABEL_HEALTHY


def labeled_mask(self: SAMLabels) -> NDArray[np.bool_]:
    """Boolean mask of the labeled (not -1) scans.

    Returns
    -------
    ndarray (bool)
        Mask of shape (n_signals,).
    """
    return self.labels != LABEL_UNLABELED


def unlabeled_mask(self: SAMLabels) -> NDArray[np.bool_]:
    """Boolean mask of the unlabeled (-1) scans.

    Returns
    -------
    ndarray (bool)
        Mask of shape (n_signals,).
    """
    return self.labels == LABEL_UNLABELED


def mask(self: SAMLabels, label: Union[int, str]) -> NDArray[np.bool_]:
    """Boolean mask of the scans carrying the given label.

    Parameters
    ----------
    label : int or str
        Label value, or label name (case-insensitive).

    Returns
    -------
    ndarray (bool)
        Mask of shape (n_signals,).
    """
    val = self.name_to_value(label) if isinstance(label, str) else label
    return self.labels == val


def stratified_split(
        self: SAMLabels, test_size: float = 0.2,
        random_state: Optional[int] = None
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Stratified train/test split that preserves class proportions.

    Splits the labeled scans per class (each class contributes
    ``max(1, int(n_class * test_size))`` test samples) and the unlabeled
    scans separately, keeping the same class distribution in both sets.

    Parameters
    ----------
    test_size : float, optional
        Fraction of samples to allocate to the test set (0 < t < 1).
    random_state : int or None, optional
        Seed for reproducible shuffling.

    Returns
    -------
    train : ndarray (int64)
        Row indices for training.
    test : ndarray (int64)
        Row indices for testing.
    """
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


# Attach the convenience API to the C++ class.  Setting __module__ to
# "samcore._samcore" makes nanobind's stubgen render these members inside
# the generated class stub; the module-level names are deleted afterwards
# so the stub of this module stays clean.
#
# ``SAMLabels`` is deleted from this module's namespace after patching so
# that no reference cycle remains (class -> patched function -> module
# globals -> class).  Without it, reference counting alone releases
# everything at interpreter shutdown on every platform.
SAMLabels.LABEL_UNLABELED = LABEL_UNLABELED
SAMLabels.LABEL_NAME_UNLABELED = LABEL_NAME_UNLABELED
SAMLabels.LABEL_HEALTHY = LABEL_HEALTHY
SAMLabels.LABEL_NAME_HEALTHY = LABEL_NAME_HEALTHY
for _fn in (healthy_mask, labeled_mask, unlabeled_mask, mask,
            stratified_split):
    setattr(SAMLabels, _fn.__name__, _fn)
    _fn.__module__ = "samcore._samcore"
    # Materialize the annotations (PEP 649 on Python >= 3.14 defers them to
    # a lazy __annotate__ closure), so they stay valid after the class name
    # is deleted from this module's namespace below.
    _fn.__annotations__ = _fn.__annotations__

del (_fn, healthy_mask, labeled_mask, unlabeled_mask, mask, stratified_split,
     SAMLabels)
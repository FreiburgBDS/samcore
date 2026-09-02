"""samcore - SAM (Scanning Acoustic Microscopy) core data processing.

A library for Scanning Acoustic Microscopy (SAM) data processing and
analysis with a fast C++ (libsamcore) backend.  It provides the main data
containers -- :class:`SAMScan` (a single acquisition grid of A-scans),
:class:`SAMLabels` (per-scan class labels) and :class:`SAMDataset`
(a pooled, padded collection of scans ready for batching, splitting and
preprocessing) -- plus signal processing primitives (``preprocessing``),
spectral utilities (``utils``) and file I/O helpers (:class:`io`).

Typical workflow::

    import samcore

    scan = samcore.SAMScan("data.h5sam")
    dataset = samcore.SAMDataset([scan])
    dataset.preprocess("lp", cutoff=10.0, fs=2.5e3)
    dataset.train_test_split(test_size=0.2)
    for X, y, spatial in dataset.batches(batch_size=64):
        ...
"""

from samcore._samcore import (SAMDataset, SAMHeader, SAMLabels, SAMScan,
                              merge_labels, preprocessing, utils)

from samcore import _dataset, _labels, _scan
from samcore._io import io

try:
    from samcore._version import __version__  # generated from version.txt
except ImportError:  # pragma: no cover - running from the source tree
    import pathlib

    __version__ = (
        pathlib.Path(__file__).resolve().parents[2] / "version.txt"
    ).read_text().strip()
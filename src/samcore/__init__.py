# samcore - SAM (Scanning Acoustic Microscopy) data processing with a C++
# (libsamcore) backend.

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

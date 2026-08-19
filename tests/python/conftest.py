import os

import numpy as np
import pytest

from samcore import SAMHeader, SAMLabels, SAMScan

# testdata directory (tests/data under the repo root); overridable via env
_DATA_DIR = os.environ.get(
    "SAMCORE_TEST_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data"))
H5_PATH = os.path.join(_DATA_DIR, "testdata.h5sam")
H5SAMD_PATH = os.path.join(_DATA_DIR, "testdata.h5samd")

needs_data = pytest.mark.skipif(not os.path.exists(H5_PATH),
                                reason="no SAM testdata")
needs_h5samd = pytest.mark.skipif(not os.path.exists(H5SAMD_PATH),
                                  reason="no h5samd testdata")


@pytest.fixture(scope="module")
def h5():
    """A handler loaded from the sample .h5sam test data."""
    return SAMScan(H5_PATH)


def make_header(n_signals, scanlen, cols=1):
    """Minimal header for testing (samcore has no version field)."""
    return SAMHeader(scanspline=cols, nlines=n_signals // cols,
                     scanlen=scanlen, samplerate=1000.0, tzero=0,
                     resolution=1.0)


def make_handler(n_signals=10, scanlen=100, cols=1, labels=None,
                 label_names=None):
    """Build a handler with int8 data (samcore stores signal data as int8)."""
    data = np.random.default_rng(42).integers(
        -128, 127, size=(n_signals, scanlen)).astype(np.int8)
    if labels is None:
        labels = np.zeros(n_signals, dtype=np.int8)
    if label_names is None:
        label_names = ["healthy"]
    samlabels = SAMLabels(labels, label_names)
    header = make_header(n_signals, scanlen, cols)
    return SAMScan.handler_from_data(data, header, samlabels=samlabels)

"""samcore._io - file I/O helpers (backed by the C++ ``io`` submodule).

The :class:`io` class wraps the C++ ``samcore.io`` functions with typed
signatures and full documentation.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from samcore._samcore import SAMDataset
from samcore._samcore import SAMHeader
from samcore._samcore import SAMLabels
from samcore._samcore import io as _io


class io:
    """File I/O helpers for .h5sam (scan) and .h5samd (dataset) files.

    All members are static methods mirroring the C++ ``samcore.io``
    submodule.
    """

    @staticmethod
    def read_h5sam(
        path: str,
    ) -> Tuple[NDArray[np.int8], SAMHeader, SAMLabels,
               Optional[NDArray[np.int32]]]:
        """Read a .h5sam file.

        Parameters
        ----------
        path : str
            Path to the .h5sam file.

        Returns
        -------
        tuple
            ``(data, header, labels, starts)``: the int8 signal array of
            shape (n_signals, scanlen), the header, the labels and the
            per-scan start indices (or None).
        """
        return _io.read_h5sam(path)

    @staticmethod
    def write_h5sam(path: str, data: NDArray[np.int8], header: SAMHeader,
                    samlabels: SAMLabels,
                    starts: Optional[NDArray[np.int32]] = None) -> None:
        """Write an int8 signal array plus metadata to a .h5sam file.

        Parameters
        ----------
        path : str
            Destination path (must end with ``.h5sam``).
        data : ndarray (int8)
            Signals of shape (n_signals, scanlen).
        header : SAMHeader
            Acquisition metadata.
        samlabels : SAMLabels
            Per-scan labels.
        starts : ndarray (int32) or None, optional
            Per-scan start indices; -1 marks an unaligned scan.
        """
        _io.write_h5sam(path, data, header, samlabels, starts)

    @staticmethod
    def read_h5samd(path: str) -> SAMDataset:
        """Read a .h5samd file as a :class:`SAMDataset`.

        Parameters
        ----------
        path : str
            Path to the .h5samd file.

        Returns
        -------
        SAMDataset
            The loaded dataset.
        """
        return _io.read_h5samd(path)

    @staticmethod
    def convert_h5sam_to_h5samd(input_paths: List[str], output_path: str,
                                pad_value: float = 0.0,
                                unsupervised: Optional[bool] = None) -> None:
        """Convert one or more .h5sam files into a .h5samd dataset.

        Signals are padded to the longest scan length.

        Parameters
        ----------
        input_paths : list of str
            Paths to source .h5sam files.
        output_path : str
            Destination .h5samd path.
        pad_value : float, optional
            Value used to pad shorter signals.
        unsupervised : bool or None, optional
            When True labels are discarded; when False labels are required.
            None (default) auto-detects from whether any cube is labeled.
        """
        _io.convert_h5sam_to_h5samd(list(input_paths), output_path,
                                    pad_value, unsupervised)
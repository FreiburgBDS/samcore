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
    def read_h5samd(path: str, lazy: bool = False) -> SAMDataset:
        """Read a .h5samd file as a :class:`SAMDataset`.

        Parameters
        ----------
        path : str
            Path to the .h5samd file.
        lazy : bool, optional
            With True X/Z/V stay on disk and are decoded in cached row blocks
            on demand (paged lazy reading, not memory mapping); metadata is
            always loaded eagerly.

        Returns
        -------
        SAMDataset
            The loaded dataset.
        """
        return _io.read_h5samd(path, lazy=lazy)

    @staticmethod
    def convert_h5sam_to_h5samd(input_paths: List[str], output_path: str,
                                pad_value: float = 0.0,
                                unsupervised: Optional[bool] = None,
                                merge: str = "crop",
                                reference: int = 0,
                                max_shift: Optional[int] = None,
                                max_gate_loss: float = 1.0,
                                threshold: float = 0.2,
                                zgate_length: int = 2000) -> None:
        """Convert one or more .h5sam files into a .h5samd dataset.

        Signals are combined according to ``merge``: ``'crop'`` keeps the
        leading samples up to the shortest scan, ``'pad'`` retains the legacy
        longest-length padding, ``'xcorr'`` aligns all rows to one pooled
        reference, and ``'zgate'`` threshold-aligns every row.

        Parameters
        ----------
        input_paths : list of str
            Paths to source .h5sam files.
        output_path : str
            Destination .h5samd path.
        pad_value : float, optional
            Value used by ``merge='pad'``.
        unsupervised : bool or None, optional
            When True labels are discarded; when False labels are required.
            None (default) auto-detects from whether any cube is labeled.
        merge : str, optional
            Merge policy: 'crop' (default), 'pad', 'xcorr' or 'zgate'.
        reference : int, optional
            Pooled row used as the xcorr reference.  Default 0.
        max_shift : int or None, optional
            Maximum xcorr lag; None means ``min(scanlens) // 4``.
        max_gate_loss : float, optional
            Maximum xcorr lag spread as a fraction of the common prefix.
        threshold : float, optional
            Zgate threshold fraction of the int8 full scale.
        zgate_length : int, optional
            Zgate output window in samples.  Default 2000.
        """
        _io.convert_h5sam_to_h5samd(
            list(input_paths), output_path, pad_value, unsupervised, merge,
            reference, max_shift, max_gate_loss, threshold, zgate_length)
"""samcore._scan - SAMScan Python-side API layer.

The heavy lifting runs in libsamcore via nanobind (see ``samcore._samcore``);
this module layers the convenience API on top of the C++ ``SAMScan``:
``in_place`` flags, numpy-typed ``starts``/labels and convenience aliases.

The functions defined here are attached to the C++ class at import time
(``SAMScan.downsample`` etc.), so ``samcore.SAMScan`` is the one and only
``SAMScan`` type.  Their annotations are real (not postponed) so that
nanobind's stubgen, which runs against the assembled package at build time,
renders fully-typed, documented members inside the generated ``_samcore.pyi``.
"""

from typing import Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

from samcore._samcore import SAMScan

_STFT = Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]


def compute_stft(self: SAMScan, nperseg: int = 256,
                 noverlap: int = 128) -> _STFT:
    """Compute the one-sided Short-Time Fourier Transform (STFT) of every A-scan.

    SAM data is real-valued and single-channel, so a one-sided (real) STFT
    is used, producing only positive-frequency bins.

    Parameters
    ----------
    nperseg : int
        Number of samples per segment.
    noverlap : int
        Number of samples to overlap between segments.

    Returns
    -------
    f : ndarray (float32)
        Frequency bins, shape (n_freqs,).
    t : ndarray (float32)
        Time bins in seconds, shape (n_frames,).
    Zxx : ndarray (complex64)
        The STFT of shape (n_signals, n_freqs, n_frames).

    Raises
    ------
    ValueError
        If ``nperseg`` exceeds the signal length.
    """
    if self.data.shape[1] < nperseg:
        raise ValueError(
            "nperseg cannot be greater than the length of the signals.")
    return self._compute_stft(nperseg, noverlap)  # type: ignore[attr-defined]


def downsample(self: SAMScan, factor: int, mode: str = "decimate",
               in_place: bool = True) -> SAMScan:
    """Downsample the data by an integer factor.

    Parameters
    ----------
    factor : int
        The integer factor by which to downsample the data.
    mode : str, optional
        The mode to use for downsampling:

        - ``'decimate'``: anti-aliasing IIR filter, then subsample
        - ``'mean'``: mean of each non-overlapping segment
        - ``'median'``: median of each non-overlapping segment
        - ``'sample'``: first sample of each non-overlapping segment
    in_place : bool, optional
        If True, modify this scan in place and return ``self``.

    Returns
    -------
    SAMScan
        The downsampled scan.  When ``in_place`` is True this is ``self``.
    """
    if not in_place:
        h = self.copy()
        h._downsample(factor, mode)  # type: ignore[attr-defined]
        return h
    self._downsample(factor, mode)  # type: ignore[attr-defined]
    return self


def downsampled(self: SAMScan, factor: int, mode: str = "decimate") -> SAMScan:
    """Return a downsampled copy of the scan.

    Parameters
    ----------
    factor : int
        The integer factor by which to downsample the data.
    mode : str, optional
        Downsampling mode; see :meth:`downsample`.

    Returns
    -------
    SAMScan
        A new, downsampled scan.
    """
    return self._downsampled(factor, mode)  # type: ignore[attr-defined]


def rotate(self: SAMScan, degrees: int, in_place: bool = True) -> SAMScan:
    """Rotate the SAM cube clockwise by 90, 180, or 270 degrees.

    Only the *spatial* layout of scans is rotated -- the individual signals
    (time-domain waveforms) are untouched.  For a scan with
    ``shape == (nlines, cols)``:

    * 90 deg clockwise -> new shape ``(cols, nlines)``
    * 180 deg clockwise -> new shape ``(nlines, cols)``
    * 270 deg clockwise -> new shape ``(cols, nlines)``

    Per-scan metadata that is index-aligned with ``data`` (labels,
    ``starts``) is permuted to follow the moved signals.

    Parameters
    ----------
    degrees : int
        Rotation angle; must be one of 90, 180, 270 (negative angles are
        normalized modulo 360).
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default True.

    Returns
    -------
    SAMScan
        The rotated scan.  When ``in_place`` is True this is ``self``.
    """
    if not in_place:
        h = self.copy()
        h._rotate(degrees)  # type: ignore[attr-defined]
        return h
    self._rotate(degrees)  # type: ignore[attr-defined]
    return self


def rotated(self: SAMScan, degrees: int) -> SAMScan:
    """Return a copy with the spatial grid rotated clockwise.

    Parameters
    ----------
    degrees : int
        Rotation angle; see :meth:`rotate`.

    Returns
    -------
    SAMScan
        A new, rotated scan.
    """
    return self._rotated(degrees)  # type: ignore[attr-defined]


def mirror(self: SAMScan, orientation: str, in_place: bool = True) -> SAMScan:
    """Mirror the SAM cube along the x or y spatial axis.

    Only the *spatial* layout of scans is flipped -- the individual signals
    (time-domain waveforms) are untouched.  The shape of the scan is
    unchanged.

    ``orientation`` follows the convention of :attr:`shape` --
    ``(nlines, cols)`` corresponds to ``(y, x)``:

    * ``"x"``: flip left/right (reverse column order)
    * ``"y"``: flip top/bottom (reverse line order)

    Per-scan metadata that is index-aligned with ``data`` (labels,
    ``starts``) is permuted to follow the moved signals.

    Parameters
    ----------
    orientation : str
        One of ``"x"`` or ``"y"`` (case-insensitive).
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default True.

    Returns
    -------
    SAMScan
        The mirrored scan.  When ``in_place`` is True this is ``self``.
    """
    if not in_place:
        h = self.copy()
        h._mirror(orientation)  # type: ignore[attr-defined]
        return h
    self._mirror(orientation)  # type: ignore[attr-defined]
    return self


def mirrored(self: SAMScan, orientation: str) -> SAMScan:
    """Return a copy mirrored across the x or y axis.

    Parameters
    ----------
    orientation : str
        One of ``"x"`` or ``"y"`` (case-insensitive).

    Returns
    -------
    SAMScan
        A new, mirrored scan.
    """
    return self._mirrored(orientation)  # type: ignore[attr-defined]


def zgate(self: SAMScan, threshold: float = 0.2, length: int = 2000,
          in_place: bool = False) -> SAMScan:
    """Apply threshold-based signal gating to each A-scan.

    For each scan, finds the first sample whose absolute value exceeds
    ``threshold * 127`` and extracts ``length`` samples starting from that
    position.  If the threshold crossing occurs too late (after
    ``scanlen - length``) it is clamped so the extracted window fits.

    If the scan already has start indices (e.g. from a previous gate or ZGT
    data), the new relative starts are accumulated on top of the existing
    ones.

    Parameters
    ----------
    threshold : float, optional
        Fraction of the full int8 range (0.0-1.0).  The actual threshold
        value is ``threshold * 127``.
    length : int, optional
        Number of samples to extract after the threshold crossing.
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default False -- a
        new scan is returned.

    Returns
    -------
    SAMScan
        A scan containing the gated signals, with ``scanlen == length`` and
        an associated ``starts`` array.  When ``in_place`` is True this is
        ``self``.
    """
    if in_place:
        self._zgate_ip(threshold, length)  # type: ignore[attr-defined]
        return self
    return self._zgate_copy(threshold, length)  # type: ignore[attr-defined]


def rectangle_select(self: SAMScan, line_start: int, line_end: int,
                     col_start: int, col_end: int,
                     in_place: bool = False) -> SAMScan:
    """Select a rectangular spatial region from the scan.

    Parameters
    ----------
    line_start : int
        Starting line index (inclusive).
    line_end : int
        Ending line index (exclusive).
    col_start : int
        Starting column index (inclusive).
    col_end : int
        Ending column index (exclusive).
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default False -- a
        new scan is returned.

    Returns
    -------
    SAMScan
        A scan containing the selected data.  When ``in_place`` is True
        this is ``self``.
    """
    if in_place:
        self._rectangle_select_ip(  # type: ignore[attr-defined]
            line_start, line_end, col_start, col_end)
        return self
    return self._rectangle_select(  # type: ignore[attr-defined]
        line_start, line_end, col_start, col_end)


def time_range_select(self: SAMScan, start_time: float, end_time: float,
                      in_place: bool = False) -> SAMScan:
    """Select a time range from the scan.

    Parameters
    ----------
    start_time : float
        Start time in nanoseconds.
    end_time : float
        End time in nanoseconds.
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default False -- a
        new scan is returned.

    Returns
    -------
    SAMScan
        A scan containing the selected data.  When ``in_place`` is True
        this is ``self``.
    """
    if in_place:
        self._time_range_select_ip(  # type: ignore[attr-defined]
            start_time, end_time)
        return self
    return self._time_range_select(  # type: ignore[attr-defined]
        start_time, end_time)


def num_scans(self: SAMScan) -> int:
    """Number of A-scans in the data (``nlines * cols``)."""
    return len(self)


def __iter__(self: SAMScan) -> Iterator[NDArray[np.int8]]:
    """Iterate over the A-scans in the data."""
    return iter(self.data)


def __hash__(self: SAMScan) -> int:
    """Hash of this scan, based on the header hash and the data."""
    return hash((self.header_hash(), self.data.tobytes()))


# Attach the convenience API to the C++ class.  The class is bound with
# nb::dynamic_attr(), so the patched members keep working on every instance,
# including those returned by C++ (copy(), from_data(), zgate(), ...).
#
# Setting __module__ to "samcore._samcore" makes nanobind's stubgen render
# these members inside the generated class stub (with their annotations and
# docstrings) instead of dropping them.  The module-level names are deleted
# afterwards so the stub of this module stays clean.  ``SAMScan`` is deleted
# from this module's namespace as well so that no reference cycle remains
# (class -> patched function -> module globals -> class); reference counting
# alone then releases everything at interpreter shutdown on every platform.
_PATCHED = (
    compute_stft, downsample, downsampled, rotate, rotated, mirror, mirrored,
    zgate, rectangle_select, time_range_select, num_scans, __iter__, __hash__,
)
for _fn in _PATCHED:
    setattr(SAMScan, _fn.__name__, _fn)
    _fn.__module__ = "samcore._samcore"
    # Materialize the annotations (PEP 649 on Python >= 3.14 defers them to
    # a lazy __annotate__ closure), so they stay valid after the class name
    # is deleted from this module's namespace below.
    _fn.__annotations__ = _fn.__annotations__

del (_PATCHED, _fn, compute_stft, downsample, downsampled, rotate, rotated,
     mirror, mirrored, zgate, rectangle_select, time_range_select, num_scans,
     __iter__, __hash__, SAMScan)
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

from typing import Any, Iterator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from samcore._numpy import array_view
from samcore._samcore import SAMScan

_STFT = Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]


def compute_stft(self: SAMScan, nperseg: int = 256,
                 noverlap: int = 128, f_min: float = 0.0,
                 f_max: float = 0.0) -> _STFT:
    """Compute the one-sided Short-Time Fourier Transform (STFT) of every A-scan.

    SAM data is real-valued and single-channel, so a one-sided (real) STFT
    is used, producing only positive-frequency bins.

    Parameters
    ----------
    nperseg : int
        Number of samples per segment.
    noverlap : int
        Number of samples to overlap between segments.
    f_min, f_max : float
        Frequency band limits in Hz.  With ``f_max > 0`` only bins in
        ``[f_min, f_max]`` are returned; ``f_max = 0`` means Nyquist.

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
    return self._compute_stft(nperseg, noverlap, float(f_min), float(f_max))  # type: ignore[attr-defined]


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


def align_manual(self: SAMScan, starts: NDArray[np.int32],
                 scanlen: int) -> SAMScan:
    """Apply manual per-scan start indices and a window length.

    The values are accumulated on top of existing starts (or set directly
    when the scan has none); -1 marks an unaligned scan and zero-fills its
    samples.  ``scanlen`` is the length of the already-extracted windows
    (as produced by :meth:`zgate`).

    Parameters
    ----------
    starts : ndarray (int32)
        Per-scan start indices, length ``nlines * cols``.
    scanlen : int
        Window length the starts refer to, in samples.

    Returns
    -------
    SAMScan
        ``self``, modified in place.
    """
    self._align_manual(np.asarray(starts, dtype=np.int32), int(scanlen))  # type: ignore[attr-defined]
    return self


def align_xcorr(self: SAMScan, reference: int = 0,
                max_shift: Optional[int] = None,
                in_place: bool = True) -> SAMScan:
    """Align every A-scan to a reference by integer cross-correlation.

    For each scan the lag within ``max_shift`` samples that maximizes the
    mean-subtracted cross-correlation with the reference scan is applied to
    the sample data, so features coincide with the reference.  Existing
    ``starts`` advance by the applied shift (clamped at 0) so absolute
    feature times are preserved; the reference row itself is unchanged.

    Parameters
    ----------
    reference : int, optional
        Flat index of the reference A-scan.  Default 0.
    max_shift : int or None, optional
        Maximum absolute lag in samples; None (default) means
        ``scanlen // 4``.
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default True.

    Returns
    -------
    SAMScan
        The aligned scan.  When ``in_place`` is True this is ``self``.
    """
    shift = 0 if max_shift is None else int(max_shift)
    if not in_place:
        h = self.copy()
        h._align_xcorr(int(reference), shift)  # type: ignore[attr-defined]
        return h
    self._align_xcorr(int(reference), shift)  # type: ignore[attr-defined]
    return self


def align_tof(self: SAMScan, gate_ns: float, reference: int = 0,
              start_ns: float = 0.0, in_place: bool = True) -> SAMScan:
    """Classic time-of-flight alignment.

    Each A-scan is shifted so the echo picked by its analytic-envelope peak
    (see :meth:`tof`) lands at the reference scan's peak.  The ToF gate is
    ``[start_ns, start_ns + gate_ns)`` on the time axis, both in
    ns.  Existing ``starts`` advance by the applied integer shift
    (clamped at 0) so absolute feature times are preserved; scans without an
    envelope peak in the gate are left unchanged.

    Parameters
    ----------
    gate_ns : float
        Length of the ToF gate in ns.  Required.
    reference : int, optional
        Flat index of the reference A-scan.  Default 0.
    start_ns : float, optional
        Gate start offset from ``tzero`` in ns.  Default 0.
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default True.

    Returns
    -------
    SAMScan
        The aligned scan.  When ``in_place`` is True this is ``self``.
    """
    if not in_place:
        h = self.copy()
        h._align_tof(  # type: ignore[attr-defined]
            float(gate_ns), int(reference), float(start_ns))
        return h
    self._align_tof(  # type: ignore[attr-defined]
        float(gate_ns), int(reference), float(start_ns))
    return self


def tof(self: SAMScan, start: int = 0, end: int = 0,
        sub_sample: bool = True) -> NDArray[np.float32]:
    """Time-of-flight image from the analytic-envelope peak per A-scan.

    Each A-scan is converted to its analytic signal (Hilbert transform) and
    the maximum of the envelope magnitude within the gate ``[start, end)``
    is taken as the echo arrival.  This is the conventional pick for
    pulse-echo measurements -- no amplitude threshold is involved, so the
    result is independent of the int8 signal scale.  With ``sub_sample``
    the peak position is refined by a parabolic fit through the maximum and
    its neighbours.

    Parameters
    ----------
    start : int, optional
        First sample of the gate.
    end : int, optional
        End of the gate (exclusive); 0 means the full scan length.
    sub_sample : bool, optional
        Refine the envelope peak position with a parabolic fit.
        Default True.

    Returns
    -------
    ndarray (float32)
        ToF map of shape ``(nlines, cols)`` in ns on the handler's time
        scale (``tzero + sample * samplespacing``).  Windows without an
        envelope peak return ``samcore.NO_TOF`` (-1.0) instead of NaN.
    """
    return self._tof(int(start), int(end), bool(sub_sample))  # type: ignore[attr-defined]


def thickness(self: SAMScan, sound_speed_m_s: float, start: int = 0,
              end: int = 0, sub_sample: bool = True) -> NDArray[np.float32]:
    """Pulse-echo thickness image derived from :meth:`tof`.

    ``thickness = tof_ns * 1e-9 * sound_speed_m_s / 2``; windows without
    an envelope peak keep the ``samcore.NO_TOF`` sentinel (-1.0) instead
    of NaN.

    Parameters
    ----------
    sound_speed_m_s : float
        Longitudinal sound speed in m/s.
    start : int, optional
        First sample of the gate.
    end : int, optional
        End of the gate (exclusive); 0 means the full scan length.
    sub_sample : bool, optional
        Refine the envelope peak position with a parabolic fit.
        Default True.

    Returns
    -------
    ndarray (float32)
        Thickness map of shape ``(nlines, cols)`` in meters (m).
    """
    return self._thickness(  # type: ignore[attr-defined]
        float(sound_speed_m_s), int(start), int(end), bool(sub_sample))


def xgate(self: SAMScan, gate_ns: float, n_gates: int = 50,
          pick: str = "tof", threshold: Optional[float] = None,
          mode: str = "max") -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Layered gating (XGate): reduce consecutive gates to scalars.

    A start sample is picked for every A-scan and ``n_gates`` consecutive
    non-overlapping windows of ``gate_ns`` each are reduced to one scalar,
    producing a depth-layer stack for fast imaging of echo-pulse scans.

    Parameters
    ----------
    gate_ns : float
        Window length in ns.
    n_gates : int, optional
        Maximum number of gates per scan (upper bound; gates that do not
        fit within the scan are zero).
    pick : str, optional
        Start criterion:

        - ``'tof'``: analytic-envelope peak (see :meth:`tof`),
        - ``'threshold'``: first sample with value > ``threshold * 127``
          (positive samples only),
        - ``'none'``: start at sample 0.
    threshold : float or None, optional
        Fraction of the int8 range for ``pick='threshold'``.  ``None``
        falls back to ``pick='none'`` (start at 0).
    mode : str, optional
        Gate reduction: ``'max'``, ``'absmax'`` (max ``|value|``) or
        ``'power'`` (sum of squares).

    Returns
    -------
    values : ndarray (float32)
        Gate values of shape ``(nlines, cols, n_gates)``.
    starts : ndarray (int32)
        Picked start sample per signal (``nlines * cols``), ``-1`` when the
        pick failed (zero values).
    """
    if pick == "threshold" and threshold is None:
        pick = "none"  # no threshold means start at sample 0
    return self._xgate(  # type: ignore[attr-defined]
        float(gate_ns), int(n_gates), pick,
        -1.0 if threshold is None else float(threshold), mode)


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


def index_range_select(self: SAMScan, start_idx: int, end_idx: int,
                       in_place: bool = False) -> SAMScan:
    """Slice the time axis by sample index.

    ``start_idx`` is inclusive and ``end_idx`` exclusive; out-of-range
    indices are clamped.  Per-scan ``starts`` are preserved and advanced by
    ``start_idx`` so absolute times are unchanged; when every scan shares the
    same valid start, it is folded into ``tzero`` and ``starts`` is cleared.

    Parameters
    ----------
    start_idx : int
        First sample index to keep (inclusive).
    end_idx : int
        End of the sample range (exclusive).
    in_place : bool, optional
        If True, modify this scan and return ``self``.  Default False -- a
        new scan is returned.

    Returns
    -------
    SAMScan
        A scan with ``scanlen == end_idx - start_idx``.
    """
    if in_place:
        self._index_range_select_ip(int(start_idx), int(end_idx))  # type: ignore[attr-defined]
        return self
    return self._index_range_select(int(start_idx), int(end_idx))  # type: ignore[attr-defined]


def time_range_select(self: SAMScan, start_time: float, end_time: float,
                      in_place: bool = False) -> SAMScan:
    """Select a time range from the scan.

    The range is measured in ns from the shared ``tzero`` (relative
    time).  Per-scan ``starts`` are preserved and advanced by the sliced
    sample offset so absolute times are unchanged (a uniform valid start is
    folded into ``tzero``); use :meth:`index_range_select` when you want to
    drive the selection by sample index directly.

    Parameters
    ----------
    start_time : float
        Start time in ns.
    end_time : float
        End time in ns.
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


def __array__(self: SAMScan, dtype: Any = None,
              copy: Optional[bool] = None) -> NDArray[Any]:
    """Raw signal matrix as a NumPy array.

    ``np.asarray(scan)`` returns the raw int8 signals of shape
    ``(nlines * cols, scanlen)`` as a zero-copy view when no dtype
    conversion is requested; use :meth:`normalized_data` for the
    ``[-1, 1)`` float representation.
    """
    return array_view(self.data, dtype, copy)


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
    zgate, align_manual, align_xcorr, align_tof,
    tof, thickness, xgate, index_range_select, rectangle_select,
    time_range_select, num_scans, __iter__, __hash__, __array__,
)
for _fn in _PATCHED:
    setattr(SAMScan, _fn.__name__, _fn)
    _fn.__module__ = "samcore._samcore"
    # Materialize the annotations (PEP 649 on Python >= 3.14 defers them to
    # a lazy __annotate__ closure), so they stay valid after the class name
    # is deleted from this module's namespace below.
    _fn.__annotations__ = _fn.__annotations__

del (_PATCHED, _fn, compute_stft, downsample, downsampled, rotate, rotated,
     mirror, mirrored, zgate, align_manual, align_xcorr, align_tof, tof,
     thickness, xgate, index_range_select, rectangle_select,
     time_range_select, num_scans, __iter__, __hash__, __array__, SAMScan)
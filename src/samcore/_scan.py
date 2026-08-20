# samcore._scan - SAMScan python-side API layer.
#
# The heavy lifting runs in libsamcore via nanobind (see _samcore); this
# module layers the convenience API on top: in_place flags, numpy-typed
# starts/labels, convenience aliases and the raw compute_stft wrapper.

import numpy as np

from samcore._samcore import SAMScan


def _scan_handler_from_data(cls, data, header, starts=None, samlabels=None):
    new_sh = cls.from_data(np.asarray(data, dtype=np.int8), header,
                           starts, samlabels)
    new_sh.path = ""
    return new_sh


def _scan_downsample(self, factor, mode="decimate", in_place=True):
    if not in_place:
        h = self.copy()
        h._downsample(factor, mode)
        return h
    self._downsample(factor, mode)
    return self


def _scan_rotate(self, degrees, in_place=True):
    if not in_place:
        h = self.copy()
        h._rotate(degrees)
        return h
    self._rotate(degrees)
    return self


def _scan_mirror(self, orientation, in_place=True):
    if not in_place:
        h = self.copy()
        h._mirror(orientation)
        return h
    self._mirror(orientation)
    return self


def _scan_zgate(self, threshold=0.2, length=2000, in_place=False):
    if in_place:
        self._zgate_ip(threshold, length)
        return self
    return self._zgate_copy(threshold, length)


def _scan_rectangle_select(self, line_start, line_end, col_start, col_end,
                           in_place=False):
    if in_place:
        self._rectangle_select_ip(line_start, line_end, col_start, col_end)
        return self
    return self._rectangle_select(line_start, line_end, col_start, col_end)


def _scan_time_range_select(self, start_time, end_time, in_place=False):
    if in_place:
        self._time_range_select_ip(start_time, end_time)
        return self
    return self._time_range_select(start_time, end_time)


def _scan_compute_stft(self, nperseg=256, noverlap=128):
    """One-sided STFT of every scan.

    Returns
    -------
    f : np.ndarray, shape (n_freqs,)
    t : np.ndarray, shape (n_frames,)
    Zxx : np.ndarray, shape (n_signals, n_freqs, n_frames), complex64
    """
    if self.data.shape[1] < nperseg:
        raise ValueError(
            "nperseg cannot be greater than the length of the signals.")
    return self._compute_stft(nperseg, noverlap)


def _scan_downsampled(self, factor, mode="decimate"):
    return self._downsampled(factor, mode)


def _scan_rotated(self, degrees):
    return self._rotated(degrees)


def _scan_mirrored(self, orientation):
    return self._mirrored(orientation)


def _scan_hash(self):
    return hash((self.header_hash(), self.data.tobytes()))


def _scan_num_scans(self):
    return len(self)


def _scan_iter(self):
    return iter(self.data)


SAMScan.compute_stft = _scan_compute_stft
SAMScan.downsample = _scan_downsample
SAMScan.rotate = _scan_rotate
SAMScan.mirror = _scan_mirror
SAMScan.zgate = _scan_zgate
SAMScan.rectangle_select = _scan_rectangle_select
SAMScan.time_range_select = _scan_time_range_select
SAMScan.handler_from_data = classmethod(_scan_handler_from_data)
SAMScan.downsampled = _scan_downsampled
SAMScan.rotated = _scan_rotated
SAMScan.mirrored = _scan_mirrored
SAMScan.__hash__ = _scan_hash
SAMScan.num_scans = _scan_num_scans
SAMScan.__iter__ = _scan_iter

"""GIL-release check: long C++ computations must not block other Python
threads (desktop UIs run samcore work in worker threads)."""

import threading
import time

import numpy as np

from samcore import SAMHeader, SAMScan


def test_gil_released_during_long_computation():
    n, sl = 512, 20000
    h = SAMScan.handler_from_data(
        np.zeros((n, sl), dtype=np.int8),
        SAMHeader(scanspline=1, nlines=n, scanlen=sl, samplerate=100.0,
                  tzero=0, resolution=1.0))

    started = threading.Event()
    worker_done_at = []

    def worker():
        started.set()
        time.sleep(0.01)  # requires the GIL only to append
        worker_done_at.append(time.perf_counter())

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait()
    t0 = time.perf_counter()
    h.compute_stft(nperseg=256, noverlap=128)
    t1 = time.perf_counter()
    thread.join(timeout=2.0)

    assert t1 - t0 > 0.02, (
        f"workload too small ({t1 - t0:.3f} s) to test GIL release")
    assert worker_done_at, "worker thread never finished"
    # The worker must have completed *while* the C++ call was running; if
    # the GIL were held for the whole call it could only finish afterwards.
    assert worker_done_at[0] < t1, (
        "worker did not run during compute_stft; the GIL is not released")

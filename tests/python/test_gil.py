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

    stop = threading.Event()
    ticks = [0]

    def worker():
        # Every iteration needs the GIL, so the worker can only make progress
        # while the main thread is inside the C++ call (which must release the
        # GIL).  A busy loop avoids time.sleep, whose wakeup macOS coalesces to
        # tens of milliseconds and made this test flaky.
        while not stop.is_set():
            ticks[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.05)          # let the worker get scheduled at least once
    before = ticks[0]         # snapshot while the main thread holds the GIL
    t0 = time.perf_counter()
    h.compute_stft(nperseg=256, noverlap=128)
    t1 = time.perf_counter()
    during = ticks[0] - before
    stop.set()
    thread.join(timeout=2.0)

    assert t1 - t0 > 0.02, (
        f"workload too small ({t1 - t0:.3f} s) to test GIL release")
    # The main thread holds the GIL from the `before` snapshot until the C++
    # call releases it, and again once it returns, so any progress here can
    # only have happened while the call was running with the GIL released.
    assert during > 0, (
        "worker made no progress during compute_stft; the GIL is not released")

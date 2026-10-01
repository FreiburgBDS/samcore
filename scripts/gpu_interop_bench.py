#!/usr/bin/env python3
"""GPU / interop efficiency benchmark for samcore.

Runs against real acquisitions and measures:

1. eager load time and RSS footprint,
2. zero-copy tensor/array creation vs explicit copies,
3. host->device transfer bandwidth and peak VRAM,
4. GPU reductions (absmax / power C-scans, FFT spectrum) vs the samcore
   C++ implementations and NumPy,
5. chunked streaming of a cube that does not fit in VRAM,
6. ``TorchDataLoader`` throughput vs ``SAMDataset.batches()``.

Usage::

    python scripts/gpu_interop_bench.py [--data-dir DIR] [--quick]
"""

import argparse
import os
import time
from typing import Callable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    raise SystemExit("torch is required for this benchmark")

from samcore import SAMDataset, SAMScan, utils
from samcore.interop import TorchDataLoader, tensor

DEFAULT_DIR = "/home/ahmet/work/acoustic"
THRU = "PT5_Thru_multiple Scan11 ZGate1.h5sam"
ECHO = "PT5_ECHO_multiple Scan11 ZGate1.h5sam"
BIG = "260916_umherum Scan6 ZGate1.h5sam"


def rss_gb() -> float:
    """Resident set size of this process in GiB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable[[], object]) -> float:
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def gpu_info() -> bool:
    if not torch.cuda.is_available():
        print("torch.cuda is not available; GPU sections will be skipped")
        return False
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {p.name}, {total / 2**30:.1f} GiB total, "
          f"{free / 2**30:.1f} GiB free, CC {p.major}.{p.minor}, "
          f"torch {torch.__version__}")
    return True


def bench_scan(path: str, has_cuda: bool, chunk_cols: int = 2048) -> None:
    print("=" * 88)
    print(f"{os.path.basename(path)}  ({os.path.getsize(path) / 1e6:.0f} MB file)")
    t0 = time.perf_counter()
    scan = SAMScan(path)
    t_load = time.perf_counter() - t0
    n, sl = scan.data.shape
    raw_gb = n * sl / 1e9
    print(f"  load: {t_load:.2f} s, {raw_gb:.2f} G samples "
          f"({scan.nlines}x{scan.cols} grid, scanlen {sl}), RSS {rss_gb():.2f} GiB")

    # --- zero-copy vs copies -------------------------------------------------
    t_arr = timed(lambda: np.asarray(scan))
    t_zero = timed(lambda: tensor(scan))
    t = tensor(scan)
    same = t.data_ptr() == scan.data.ctypes.data
    print(f"  np.asarray(scan): {t_arr * 1e3:.3f} ms | tensor(scan): "
          f"{t_zero * 1e3:.3f} ms, zero-copy: {same}")
    t_copy = timed(lambda: torch.tensor(scan.data.copy()))
    print(f"  torch.tensor(scan.data.copy()): {t_copy * 1e3:.1f} ms "
          f"({raw_gb:.2f} GiB copied)")
    t_f32 = timed(lambda: torch.from_numpy(
        np.ascontiguousarray(scan.data, dtype=np.float32)))
    print(f"  int8 -> float32 numpy copy: {t_f32 * 1e3:.1f} ms "
          f"({raw_gb * 4:.2f} GiB)")

    # --- samcore C++ references ---------------------------------------------
    t0 = time.perf_counter(); img_abs = scan.image("absmax")
    t_abs_cpu = time.perf_counter() - t0
    t0 = time.perf_counter(); img_pow = scan.image("power")
    t_pow_cpu = time.perf_counter() - t0
    t0 = time.perf_counter(); _, spec_mag = scan.spectrum()
    t_spec_cpu = time.perf_counter() - t0
    print(f"  samcore C++: absmax {t_abs_cpu * 1e3:.1f} ms | power "
          f"{t_pow_cpu * 1e3:.1f} ms | spectrum {t_spec_cpu * 1e3:.1f} ms")

    if not has_cuda:
        del scan, t
        return

    # --- full-cube H2D + reductions -----------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g = t.cuda()
    t_h2d = time.perf_counter() - t0
    print(f"  H2D int8 transfer: {t_h2d * 1e3:.1f} ms ({raw_gb / t_h2d:.2f} GB/s), "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    t0 = time.perf_counter()
    # int16 avoids torch's int8 |-128| wrap; clamp to match samcore's saturation
    gpu_abs = g.to(torch.int16).abs().amax(dim=1).clamp_max(127).to(torch.int8)
    t_abs_gpu = time.perf_counter() - t0
    match = np.array_equal(gpu_abs.cpu().numpy(),
                           np.asarray(img_abs).ravel())
    print(f"  GPU absmax: {t_abs_gpu * 1e3:.1f} ms "
          f"(speedup {t_abs_cpu / t_abs_gpu:.1f}x), matches samcore: {match}")

    t0 = time.perf_counter()
    acc = torch.zeros(n, dtype=torch.float64, device="cuda")
    for c0 in range(0, sl, chunk_cols):
        block = g[:, c0:c0 + chunk_cols].float()
        acc += torch.einsum("ij,ij->i", block, block)
        del block
    gpu_pow = acc.float().cpu().numpy()
    t_pow_gpu = time.perf_counter() - t0
    match = np.allclose(gpu_pow, np.asarray(img_pow).ravel(), rtol=1e-5,
                        atol=1.0)
    print(f"  GPU power (col chunks of {chunk_cols}): {t_pow_gpu * 1e3:.1f} ms "
          f"(speedup {t_pow_cpu / t_pow_gpu:.1f}x), matches: {match}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # --- GPU FFT on a chunk vs samcore.utils.fft_spectrum -------------------
    m = min(n, 16384)
    xf = g[:m].float()
    t0 = time.perf_counter()
    mag_gpu = torch.fft.rfft(xf, dim=1).abs().cpu().numpy()
    t_fft_gpu = time.perf_counter() - t0
    mag_cpu, _ = utils.fft_spectrum(np.ascontiguousarray(scan.data[:m]), d=1.0)
    err = np.max(np.abs(mag_gpu - mag_cpu)) / max(np.max(mag_cpu), 1e-9)
    print(f"  GPU rfft [{m} scans]: {t_fft_gpu * 1e3:.1f} ms, "
          f"max rel err vs samcore: {err:.2e}, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del xf, mag_gpu, g, acc
    torch.cuda.empty_cache()

    # --- TorchDataLoader vs batches -----------------------------------------
    del spec_mag  # free the 2.3 GiB spectrum before building the dataset
    ds = SAMDataset([scan], unsupervised=True)
    loader = TorchDataLoader(ds, batch_size=4096, shuffle=False)
    t0 = time.perf_counter(); count = 0
    for xb, _ in loader:
        count += xb.shape[0]
    t_loader = time.perf_counter() - t0
    t0 = time.perf_counter(); count2 = 0
    for xb, _ in ds.batches("train", batch_size=4096, shuffle=False):
        count2 += xb.shape[0]
    t_batches = time.perf_counter() - t0
    print(f"  TorchDataLoader: {count} samples in {t_loader:.2f} s "
          f"({count / t_loader / 1e6:.1f} M samples/s)")
    print(f"  ds.batches():    {count2} samples in {t_batches:.2f} s "
          f"({count2 / t_batches / 1e6:.1f} M samples/s)")
    del ds, loader, scan, t
    torch.cuda.empty_cache()


def bench_chunked_big(path: str, has_cuda: bool, lines_per_chunk: int = 8) -> None:
    print("=" * 88)
    print(f"{os.path.basename(path)}  (chunked streaming, "
          f"{lines_per_chunk} lines per chunk)")
    t0 = time.perf_counter()
    scan = SAMScan(path)
    t_load = time.perf_counter() - t0
    n, sl = scan.data.shape
    raw_gb = n * sl / 1e9
    print(f"  load: {t_load:.2f} s, grid {scan.nlines}x{scan.cols}, scanlen {sl}, "
          f"raw {raw_gb:.2f} GB, RSS {rss_gb():.2f} GiB")

    t0 = time.perf_counter(); img_ref = scan.image("absmax")
    t_cpu = time.perf_counter() - t0
    print(f"  samcore absmax (all {n} scans): {t_cpu:.2f} s")

    if not has_cuda:
        del scan
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows_per_line = scan.cols
    out = np.empty((scan.nlines, scan.cols), dtype=np.int16)
    t0 = time.perf_counter()
    for l0 in range(0, scan.nlines, lines_per_chunk):
        l1 = min(l0 + lines_per_chunk, scan.nlines)
        block = scan.data[l0 * rows_per_line:l1 * rows_per_line]  # view
        g = torch.from_numpy(np.ascontiguousarray(block)).cuda()
        am = g.to(torch.int16).abs().amax(dim=1).clamp_max(127)
        out[l0 * rows_per_line:l1 * rows_per_line] = \
            am.cpu().numpy().reshape(-1, rows_per_line)
        del g, am
    t_stream = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    match = np.array_equal(out, np.asarray(img_ref))
    print(f"  chunked H2D + absmax: {t_stream:.2f} s "
          f"({raw_gb / t_stream:.2f} GB/s), peak VRAM {peak:.2f} GiB, "
          f"matches samcore: {match}")

    free, _ = torch.cuda.mem_get_info()
    if n * sl <= free * 0.9:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        g = tensor(scan).cuda()
        torch.cuda.synchronize()
        t_full = time.perf_counter() - t0
        print(f"  whole-cube H2D: {t_full:.2f} s ({raw_gb / t_full:.2f} GB/s), "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del g
    else:
        print(f"  whole-cube transfer skipped: needs {n * sl / 2**30:.2f} GiB, "
              f"only {free / 2**30:.2f} GiB free")
    del out, scan
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--quick", action="store_true",
                    help="skip the big chunked-streaming file")
    args = ap.parse_args()

    has_cuda = gpu_info()
    for name in (THRU, ECHO):
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            bench_scan(p, has_cuda)
        else:
            print("missing:", p)
    if not args.quick:
        p = os.path.join(args.data_dir, BIG)
        if os.path.exists(p):
            bench_chunked_big(p, has_cuda)
        else:
            print("missing:", p)


if __name__ == "__main__":
    main()

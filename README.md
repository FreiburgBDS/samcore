# samcore

**Python package and C++ library for Scanning Acoustic Microscopy (SAM) data processing.**

samcore is a library for non-destructive testing (NDT) with Scanning Acoustic
Microscopy (SAM), a domain where it is mainly used for the acoustic inspection
of battery cells (internal structure, defects, wetting and layer integrity)
and, more generally, for materials and component testing.  A focused
ultrasound transducer is raster-scanned over a sample and records an A-scan at
every pixel, either in echo (reflection) or through-transmission mode, so
a single measurement yields a 3-D signal cube.  samcore turns those cubes into
C-scan images, spectra and ready-to-train machine-learning datasets.

## Overview

### Features

- **`.h5sam` / `.h5samd` file formats**: native HDF5-based I/O for single
  acquisition cubes and pooled datasets, including lazy/memory-mapped reading
  of signal data (`SAMScan(path, mmap=True)`).
- **C-scan imaging**: reduce every A-scan to one pixel
  (`scan.image("max" | "absmax" | "power")`) and normalize the raw int8
  signals to `[-1, 1)` (`scan.normalized_data()`).
- **Signal processing**: built-in strategies `lp`, `bp`, `normalize`,
  `savgol`, `medfilt`, `gate`, `detrend`, `envelope` and `moving_average`.
- **Spectral analysis**: one-sided STFT, Welch PSD and per-frame power
  spectrograms of every A-scan.
- **Cube manipulation**: downsample, rotate, mirror, rectangular/time-range
  selection and Z-gating, all with in-place and copy variants.
- **Labels**: per-scan class labels (`SAMLabels`) with label names, masks
  (healthy/labeled/unlabeled) and class distributions.
- **Datasets for ML**: pool one or more cubes into a padded `SAMDataset`,
  merge labels, split train/test (random, stratified, or by cube), apply
  feature transforms (`Z`) and iterate minibatches or spatial patches.
- **Fast C++ core**: the same functionality is available from C++ as `libsamcore`, parallelized with OpenMP.

### File formats

The two formats are the core of the package:

**`.h5sam`: one SAM acquisition** (a single scan cube) stored as an HDF5 file:

- `header` group holds acquisition metadata as attributes: `nlines`,
  `scanspline` (scans per line), `scanlen` (samples per A-scan), `samplerate`,
  `tzero`, `resolution`, `interpolated`, `quality`, `mode`, `transducer_in`,
  `transducer_through`, `cellid`, `downsample_factor`, plus arbitrary extra
  attributes, which are preserved on round-trip.
- `data`: the raw int8 signals, shape `(nlines * scanspline, scanlen)`,
  gzip-compressed.  With `mmap=True` the array stays in the file until first
  access.
- `labels` / `label_names`: optional per-scan integer labels and their names.
- `starts`: optional int32 per-scan start index for time-aligned or
  Z-gated data (`-1` marks an unaligned scan).

**`.h5samd`: a SAM dataset** pooled from one or more `.h5sam` cubes and used
for training and analysis:

- `X`: float32 signals of shape `(n_samples, maxlen)`, zero-padded to the
  longest A-scan.
- `labels` / `label_names`: optional per-sample labels, plus an
  `unsupervised` flag.
- `cube_shapes`, `cube_resolutions`, `scanlens`: provenance that maps every
  sample back to its source cube and pixel coordinates in mm
  (`SAMDataset.spatial`).
- `Z`: an optional feature matrix, and `V` an optional low-dimensional
  embedding.

## Installation

```sh
pip install samcore
```
To build from source instead, see [DEVELOPMENT.md](https://github.com/FreiburgBDS/samcore/blob/main/DEVELOPMENT.md).

## Python usage

```python
import samcore

# Load one acquisition grid; mmap=True keeps the signals on disk until first use.
scan = samcore.SAMScan("cell.h5sam", mmap=True)

print(scan.header.cellid, scan.nlines, scan.cols, scan.scanlen)
print("time axis [ns]:", scan.time()[:5])

# Reduce every A-scan to a single pixel -> C-scan image of shape (nlines, cols)
cscan = scan.image("absmax")

# Pool one or more cubes into a padded, label-aware dataset
dataset = samcore.SAMDataset([scan])
dataset.preprocess("lp", cutoff=10.0, fs=2.5e3)   # low-pass filter
dataset.train_test_split(test_size=0.2, random_state=0)

for X, y, spatial in dataset.batches(batch_size=64):
    # X: (batch, maxlen) float32 signals; y: int8 labels;
    # spatial: recarray with the source cube index (idx) and the
    # pixel-centre coordinates in mm (x, y).
    # Unsupervised datasets yield (X, spatial) instead.
    ...

dataset.save("cells.h5samd")
loaded = samcore.SAMDataset.load("cells.h5samd")
```

Several `.h5sam` files can be converted into a dataset in one call:

```python
samcore.io.convert_h5sam_to_h5samd(
    ["cell_a.h5sam", "cell_b.h5sam"], "cells.h5samd"
)
```

`scan.data` and `dataset.X` are zero-copy numpy views of the C++ buffers, so
inspection and slicing do not duplicate the signal data.  `SAMDataset` also
provides `cube_batches()` and `spatial_patches()` iterators for 2-D CNN
workflows, and `transform()` to build a feature matrix `Z`.

## C++ usage

```cpp
#include <samcore/sam_scan.hpp>
#include <samcore/sam_dataset.hpp>

int main() {
    auto scan = samcore::sam_scan::from_file("cell.h5sam"); // .h5sam files

    auto cscan = scan.image_absmax(); // C-scan image

    samcore::sam_dataset dataset({scan});
    samcore::preprocess_args args;
    args.cutoff = 10.0;
    args.fs = 2.5e3;
    dataset.preprocess("lp", args);

    dataset.save("cells.h5samd");
}
```

Install the library with `cmake --install build` and consume it in downstream
CMake projects:

```cmake
find_package(samcore REQUIRED)
target_link_libraries(myapp PRIVATE samcore::samcore)
```

Note: OpenMP is an implementation detail of the static library and is not
propagated through the CMake package.  Consumers linking `libsamcore.a`
enable it themselves (`find_package(OpenMP)` + link `OpenMP::OpenMP_CXX`).

## Development

For building from source, system dependencies, C++ build options, the optional executables, 
stub regeneration and the test suites, see
[DEVELOPMENT.md](https://github.com/FreiburgBDS/samcore/blob/main/DEVELOPMENT.md).

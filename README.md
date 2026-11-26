# samcore
[![PyPI](https://img.shields.io/pypi/v/samcore)](https://pypi.org/project/samcore)

**Python package and C++ library for Scanning Acoustic Microscopy (SAM) data processing.**

samcore is a library for non-destructive testing (NDT) with Scanning Acoustic
Microscopy (SAM), mainly used for the acoustic inspection of battery cells
and materials. A focused ultrasound transducer is raster-scanned over a sample
to record a 3-D signal cube, which samcore turns into C-scan images, spectra
and ready-to-train machine-learning datasets.

## Features

- **`.h5sam` / `.h5samd` I/O**: native HDF5 formats for single cubes and pooled
  datasets, with lazy signal loading.
- **Imaging & analysis**: C-scan reduction, filtering, gating, envelope and
  phase statistics, STFT and Welch PSD.
- **ML-ready datasets**: common-length `SAMDataset`s with labels, splits,
  batch/patch iteration, NumPy and optional PyTorch interop.
- **Fast C++ core**: the same functionality as `libsamcore`,
  OpenMP-parallelized and GIL-releasing for Python.

The on-disk layout of both formats is documented in
[DEVELOPMENT.md](https://github.com/FreiburgBDS/samcore/blob/main/DEVELOPMENT.md#file-formats).

## Installation

```sh
pip install samcore
```

To build from source, see [DEVELOPMENT.md](https://github.com/FreiburgBDS/samcore/blob/main/DEVELOPMENT.md).

## Python usage

```python
import samcore

scan = samcore.SAMScan("cell.h5sam")             # one acquisition grid
cscan = scan.image("absmax")                     # C-scan, shape (nlines, cols)

dataset = samcore.SAMDataset([scan])             # crops to the shortest scan
dataset.preprocess("lp", cutoff=10.0, fs=625.0)  # MHz
dataset.train_test_split(test_size=0.2, random_state=0)

for X, y, spatial in dataset.batches(batch_size=64):
    ...  # X: signals, y: labels, spatial: source cube and x/y in mm

dataset.save("cells.h5samd")
```

Use `merge="pad"` to keep longest-length padding instead of cropping.
`scan.data` and `dataset.X` are zero-copy NumPy views, and
`samcore.utils.spectrogram_features()` builds flattened features for NMF
pipelines.

## C++ usage

```cpp
#include <samcore/sam_scan.hpp>
#include <samcore/sam_dataset.hpp>

int main() {
    auto scan = samcore::sam_scan::from_file("cell.h5sam");
    auto cscan = scan.image_absmax();

    samcore::sam_dataset dataset({scan});
    samcore::preprocess_args args;
    args.cutoff = 10.0;  // MHz
    args.fs = 625.0;     // MHz
    dataset.preprocess("lp", args);

    dataset.save("cells.h5samd");
}
```

Install the library with `cmake --install build` and link `samcore::samcore`
via `find_package(samcore REQUIRED)`.

## Development

See [DEVELOPMENT.md](https://github.com/FreiburgBDS/samcore/blob/main/DEVELOPMENT.md)
for build instructions, dependencies, file formats and the test suites.

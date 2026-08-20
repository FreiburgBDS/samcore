# samcore

Python package and C++ library for Scanning Acoustic Microscopy (SAM) core data processing.

## Cloning

The repository uses a git submodule for the bundled FFT backend
(`external/pocketfft`, a header-only dependency).  Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:FreiburgBDS/samcore.git
```

If you already cloned without them (or want to update them):

```bash
git submodule update --init --recursive
```

## Installation & Dependencies

Install the system dependencies for your platform (below), then either
`pip install .` for the Python package or build the C++ library with CMake.

### Dependencies

Requires a C++20 compiler and CMake.  Dependencies:

- **HDF5** (C++ bindings): used for `.h5sam`/`.h5samd` file I/O.  Obtained
  via the nix dev shell, a distro package, or `HDF5_ROOT`.
- **nlohmann_json**: JSON serialization of headers and labels.  Used when
  found as a system package, otherwise fetched automatically (header-only).
- **PocketFFT**: FFT backend, vendored as a git submodule
  (`external/pocketfft`).
- **nanobind** + **numpy**: build-time dependencies of the Python package
  (installed automatically by pip).

**Nix:**
```bash
nix-shell
```

### Arch Linux

```bash
sudo pacman -S --needed --noconfirm base-devel cmake ninja hdf5 python-pip
```

### Debian / Ubuntu
```bash
sudo apt-get update && sudo apt-get -y install cmake ninja-build build-essential \
  libhdf5-dev pkg-config python3-dev python3-pip
```

### Fedora
```bash
sudo dnf install -y cmake ninja-build gcc-c++ hdf5-devel python3-devel python3-pip
```

### MacOS and Windows
Not yet tested.

## Python package (samcore)

### Installation

```sh
pip install .
```

### Usage
```python
from samcore import SAMScan, SAMDataset

scan = SAMScan("testdata.h5sam")       # .h5sam files
img = scan.image("absmax")         # (nlines, cols)

dataset = SAMDataset([scan])
dataset.preprocess("lp", cutoff=10.0, fs=2.5e3)
```

`scan.data` and `dataset.X` are zero-copy numpy views of the C++ buffers;
`SAMScan(path, mmap=True)` keeps h5sam signal data on disk
until first accessed.

The wheel ships `.pyi` type stubs plus a `py.typed` marker (generated at
build time by nanobind's stubgen), so static type checkers and IDEs see the
full typed API.

## C++ library (libsamcore)

### Build

```sh
cmake -B build -G Ninja
cmake --build build
```

Optional executables (`bench`, `gen_data`, see below):

```sh
cmake -B build -G Ninja -DSAMCORE_BUILD_EXECUTABLES=ON
cmake --build build
```

### Usage
```cpp
#include <samcore/sam_scan.hpp>
#include <samcore/sam_dataset.hpp>

int main() {
    auto scan = samcore::sam_scan::from_file("testdata.h5sam"); // .h5sam files
    auto img = scan.image(samcore::image_mode::absmax);         // (nlines, cols)

    samcore::sam_dataset dataset({scan});
    samcore::preprocess_args args;
    args.cutoff = 10.0;
    args.fs = 2.5e3;
    dataset.preprocess("lp", args);
}
```

Install with `cmake --install build` and consume in downstream CMake projects using
`find_package(samcore REQUIRED)` and linking `samcore::samcore`.

Note: OpenMP is an implementation detail of the static library and is not
propagated through the CMake package.  Consumers linking `libsamcore.a`
enable it themselves (`find_package(OpenMP)` + link `OpenMP::OpenMP_CXX`).

## Developers

### Generate Stubs

The wheel ships `.pyi` type stubs plus a `py.typed` marker.  They are
generated automatically at build time by nanobind's stubgen: a CMake custom
command in the `SAMCORE_BUILD_PYTHON` block stages an importable copy of the
package (pure-Python modules + generated `_version.py` + the compiled
extension) and runs `python -m nanobind.stubgen`.  No manual step is needed
when running `pip install .`.

To regenerate manually after installing `nanobind` into your environment:

```sh
python -m nanobind.stubgen -m samcore -m samcore._samcore -m samcore._scan \
  -m samcore._labels -m samcore._dataset -m samcore._io -M py.typed
```

### Executables

Optional tools, built with `-DSAMCORE_BUILD_EXECUTABLES=ON`:

- `bench` - OpenMP scaling smoke benchmark (`./bench [scans_x scans_y scanlen]`).
- `gen_data` - deterministic random test-data generator (gated sine burst +
  gaussian noise, int8), writing `.h5sam` (with per-scan starts) or a
  multi-cube `.h5samd`.  Used to (re)generate `tests/data/testdata.h5sam`
  and `tests/data/testdata.h5samd`.

### Tests

```sh
# C++ (googletest) - debug build with sanitizers
cmake -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
ctest --test-dir build-debug --output-on-failure

# Python (pytest)
pip install .
pytest tests/python
```

Test data ships in `tests/data/` (override with `SAMCORE_TEST_DATA_DIR`).
Release builds use `-O2 -ffast-math -flto`; Debug enables ASan+UBSan+LSan.

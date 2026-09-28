# Development & Build Guide

This guide covers building samcore from source, its dependencies and build
options, maintenance tasks and the test suites.  End users should install the
published package instead (see [README.md](README.md)):

```sh
pip install samcore
```

## Prerequisites & Dependencies

samcore requires a C++20 compiler and CMake >= 3.22.

- **HDF5** (C++ bindings): used for `.h5sam`/`.h5samd` file I/O.  Required and
  not fetched from the network; obtained via the nix dev shell, a distro
  package, or `HDF5_ROOT` / `CMAKE_PREFIX_PATH`.
- **PocketFFT**: FFT backend, header-only, vendored as the git submodule
  `external/pocketfft`.  Overridable with `-DPOCKETFFT_SOURCE_DIR=<dir>`.
- **OpenMP**: enabled by default (`SAMCORE_ENABLE_OPENMP`); required unless
  turned off.
- **googletest**: tests only.  Found via `find_package(GTest)`; when missing,
  CMake fetches a pinned googletest release with `FetchContent` (needs network
  access).
- **nanobind** + **numpy**: build-time dependencies of the Python package
  (installed automatically by pip).
- **Python >= 3.9** and **pytest**: for the Python test suite
  (`pip install ".[test]"`).

### Cloning

The repository uses a git submodule for the bundled FFT backend
(`external/pocketfft`, a header-only dependency).  Clone with submodules:

```sh
git clone --recurse-submodules git@github.com:FreiburgBDS/samcore.git
```

If you already cloned without them (or want to update them):

```sh
git submodule update --init --recursive
```

### Platform setup

Install the system dependencies for your platform (below), then either
`pip install .` for the Python package or build the C++ library with CMake.

**Nix:**
```sh
nix-shell
```

**Arch Linux:**
```sh
sudo pacman -S --needed --noconfirm base-devel cmake ninja hdf5 python-pip
```

**Debian / Ubuntu:**
```sh
sudo apt-get update && sudo apt-get -y install cmake ninja-build build-essential \
  libhdf5-dev pkg-config python3-dev python3-pip
```

**Fedora:**
```sh
sudo dnf install -y cmake ninja-build gcc-c++ hdf5-devel python3-devel python3-pip
```

**Windows / macOS:**
Both platforms are built with LLVM Clang and Ninja via
[conda-forge](https://conda-forge.org/): AppleClang on macOS ships no OpenMP,
and MSVC only implements the outdated OpenMP 2.0.

```sh
conda install -c conda-forge clangxx llvm-openmp hdf5 cmake ninja
```

On macOS, also point the build at the active conda prefix so CMake finds
HDF5/OpenMP and the compiler finds `omp.h`:

```sh
export CC="$CONDA_PREFIX/bin/clang"
export CXX="$CONDA_PREFIX/bin/clang++"
export HDF5_ROOT="$CONDA_PREFIX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
```

## Building from Source

### Python package (samcore)

```sh
pip install .
```

The Python package is built with scikit-build-core + nanobind; `pyproject.toml`
configures the CMake build with `SAMCORE_BUILD_PYTHON=ON`,
`SAMCORE_BUILD_TESTS=OFF` and `SAMCORE_BUILD_EXECUTABLES=OFF`.  Type stubs are
generated automatically (see [Python type stubs](#python-type-stubs)).

If you followed the recommended Conda setup on Windows or macOS, point CMake
at the Clang toolchain:

Windows (PowerShell):
```powershell
pip install . -C cmake.args="-G Ninja" -C cmake.args="-DCMAKE_CXX_COMPILER=clang++" -C cmake.args="-DCMAKE_C_COMPILER=clang"
```

macOS (with the environment variables from above exported):
```sh
pip install . -C cmake.args="-G Ninja"
```

For a Debug build of the extension (ASan + UBSan, see the test section for the
runtime settings):

```sh
pip install --config-settings=cmake.build-type=Debug .
```

### C++ library (libsamcore)

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

On Windows or macOS, set up the conda toolchain from above first (including
the exported variables on macOS) so CMake finds Clang and HDF5.

Downstream CMake projects consume the installed library with:

```cmake
find_package(samcore REQUIRED)
target_link_libraries(myapp PRIVATE samcore::samcore)
```

`HDF5` is a public dependency and is found by the generated package config, so
downstream builds need HDF5 development files available as well.

Note: OpenMP is an implementation detail of the static library and is not
propagated through the CMake package.  Consumers linking `libsamcore.a`
enable it themselves (`find_package(OpenMP)` + link `OpenMP::OpenMP_CXX`).

### Executables

Optional tools, built with `-DSAMCORE_BUILD_EXECUTABLES=ON`:

```sh
cmake -B build -G Ninja -DSAMCORE_BUILD_EXECUTABLES=ON
cmake --build build
```

- `bench` - OpenMP scaling smoke benchmark (`./bench [scans_x scans_y scanlen]`).
- `gen_data` - deterministic random test-data generator (gated sine burst +
  gaussian noise, int8), writing `.h5sam` (with per-scan starts) or a
  multi-cube `.h5samd`.  Used to (re)generate `tests/data/testdata.h5sam`
  and `tests/data/testdata.h5samd`.

### Build options

| Option | Default | Description |
| --- | --- | --- |
| `SAMCORE_BUILD_TESTS` | `ON` (top-level) | Build the googletest suite. |
| `SAMCORE_BUILD_EXECUTABLES` | `OFF` | Build `bench` and `gen_data`. |
| `SAMCORE_BUILD_PYTHON` | `OFF` | Build the nanobind Python extension (pip enables it). |
| `SAMCORE_ENABLE_OPENMP` | `ON` | Enable OpenMP parallelism. |
| `SAMCORE_NATIVE_ARCH` | `OFF` | Compile with `-march=native` (local builds only). |
| `SAMCORE_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors. |
| `SAMCORE_ENABLE_SANITIZERS` | `ON` | Enable ASan+UBSan in Debug builds (GCC/Clang). |
| `SAMCORE_ENABLE_FAST_MATH` | `ON` | Enable `-ffast-math` in Release builds. |
| `SAMCORE_ENABLE_LTO` | `ON` | Enable `-flto` in Release builds (forced off on musl). |

## Maintenance & Testing

### Python type stubs

The wheel ships `.pyi` type stubs plus a `py.typed` marker.  They are
generated automatically at build time by nanobind's stubgen: a CMake custom
command in the `SAMCORE_BUILD_PYTHON` block stages an importable copy of the
package (pure-Python modules + generated `_version.py` + the compiled
extension) and runs `python -m nanobind.stubgen`.  No manual step is needed
when running `pip install .`; stub generation is optional tooling and never
fails the build.

To regenerate manually after installing `nanobind` into your environment:

```sh
python -m nanobind.stubgen -m samcore -m samcore._samcore -m samcore._scan \
  -m samcore._labels -m samcore._dataset -m samcore._io -M py.typed
```

Type checking runs against the installed stubs, so install the package first:

```sh
pip install .
mypy
```

### C++ tests (googletest)

The test suite is built with `SAMCORE_BUILD_TESTS` (on by default for
top-level builds) and run through CTest.  Debug builds enable
AddressSanitizer + UBSanitizer + LeakSanitizer:

```sh
cmake -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
ctest --test-dir build-debug --output-on-failure
```

Release build (also builds and runs the tests unless
`-DSAMCORE_BUILD_TESTS=OFF`):

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The OpenMP kernels are additionally exercised single-threaded
(`samcore_tests_single_threaded`, run with `OMP_NUM_THREADS=1`) to verify that
results agree regardless of thread count.

### Python tests (pytest)

```sh
pip install ".[test]"
pytest tests/python
```

To run the Python extension under sanitizers, install the Debug configuration
and preload the ASan runtime:

```sh
pip install --config-settings=cmake.build-type=Debug .
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
pytest tests/python
```

### Test data

Sample files are kept in the repository under `tests/data/`
(`testdata.h5sam`, `testdata.h5samd`).  The Python tests read the directory
from an environment variable:

```sh
export SAMCORE_TEST_DATA_DIR=/path/to/data
```

The C++ tests take the same path as a CMake cache variable:

```sh
cmake -B build -DSAMCORE_TEST_DATA_DIR=/path/to/data
```

The `.h5sam` test data can be regenerated with the `gen_data` executable (see
above).

### Compiler flags & build modes

Release builds use `-O2 -ffast-math -flto` (`-flto` is disabled on musl
targets); Debug builds enable ASan+UBSan with `-fno-omit-frame-pointer`.
`SAMCORE_NATIVE_ARCH=ON` adds `-march=native` and `SAMCORE_WARNINGS_AS_ERRORS=ON`
adds `-Werror` (non-MSVC); the default warning level is `-Wall -Wextra
-Wpedantic` (`/W4 /permissive-` on MSVC).

When building locally with a custom HDF5 prefix:

```sh
cmake -B build -G Ninja -DHDF5_ROOT=/path/to/prefix
```

Note: on hybrid-core CPUs set `OMP_WAIT_POLICY=passive` for sane OpenMP
scaling.

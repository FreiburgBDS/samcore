# Dependency resolution: prefer packages provided by the environment
# (nix dev shell, system packages), fall back to pinned FetchContent
# downloads for the header-only libraries.

# ---- OpenMP ---------------------------------------------------------------
if(SAMCORE_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# ---- HDF5 (C++ bindings) --------------------------------------------------
# Required. Not fetched from the network because a full HDF5 build is heavy;
# obtain it via the nix dev shell (`nix develop`), a distro package, or set
# HDF5_ROOT to a custom prefix.
find_package(HDF5 COMPONENTS CXX REQUIRED)

# ---- nlohmann_json --------------------------------------------------------
find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found, fetching via FetchContent")
  include(FetchContent)
  # Fetch the headers WITHOUT add_subdirectory (self-contained Populate form).
  # nlohmann/json's own CMake project defines a plain in-tree target which
  # cannot be referenced from install(EXPORT samcoreTargets).  Exposing the
  # fetched headers as an IMPORTED interface target keeps the exported samcore
  # package identical to the system-package case: the export references
  # nlohmann_json::nlohmann_json by name, resolved at consume time by the
  # find_dependency(nlohmann_json) in samcoreConfig.cmake.in.
  FetchContent_Populate(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
  set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_SOURCE_DIR}/include")
endif()

# ---- pocketfft (FFT, header-only) -----------------------------------------
# Provided as a git submodule (external/pocketfft, containing
# pocketfft_hdronly.h at its root).  A custom copy can be selected with
# POCKETFFT_SOURCE_DIR (e.g. from a nix dev shell).  No network fetch is
# needed.
set(POCKETFFT_SOURCE_DIR "" CACHE PATH
  "Directory containing pocketfft_hdronly.h (defaults to the git submodule)")
if(POCKETFFT_SOURCE_DIR)
  set(SAMCORE_POCKETFFT_INCLUDE_DIR "${POCKETFFT_SOURCE_DIR}")
else()
  set(SAMCORE_POCKETFFT_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/external/pocketfft")
endif()
message(STATUS "pocketfft headers: ${SAMCORE_POCKETFFT_INCLUDE_DIR}")

# ---- googletest (tests only) ----------------------------------------------
if(SAMCORE_BUILD_TESTS)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "googletest not found, fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

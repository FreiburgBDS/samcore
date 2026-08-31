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

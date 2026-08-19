cmake_minimum_required(VERSION 3.22)

# Read version from version.txt; register it as a configure dependency so
# the build re-runs automatically when the file changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/version.txt")
file(READ "${CMAKE_SOURCE_DIR}/version.txt" VERSION_STRING)
string(STRIP "${VERSION_STRING}" PROJECT_VERSION)

project(samcore VERSION ${PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SAMCORE_BUILD_TESTS "Build the libsamcore test suite" ${PROJECT_IS_TOP_LEVEL})
option(SAMCORE_BUILD_EXECUTABLES "Build executables (bench, gen_data)" OFF)
option(SAMCORE_BUILD_PYTHON "Build the samcore Python bindings (nanobind)" OFF)
option(SAMCORE_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(SAMCORE_NATIVE_ARCH "Compile with -march=native" OFF)
option(SAMCORE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(SAMCORE_ENABLE_SANITIZERS "Enable ASan+UBSan in Debug builds" ON)
option(SAMCORE_ENABLE_FAST_MATH "Enable -ffast-math in Release builds" ON)
option(SAMCORE_ENABLE_LTO "Enable link-time optimization in Release builds" ON)

if(SAMCORE_NATIVE_ARCH)
  add_compile_options(-march=native)
endif()

include(cmake/dependencies.cmake)

add_library(samcore
  src/sam_header.cpp
  src/sam_labels.cpp
  src/sam_scan.cpp
  src/sam_dataset.cpp
  src/io/fileio_h5sam.cpp
  src/io/fileio_h5samd.cpp
  src/signal/filters.cpp
  src/signal/spectral.cpp
  src/preprocessing.cpp
  src/utils.cpp
)
add_library(samcore::samcore ALIAS samcore)

target_include_directories(samcore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # pocketfft is an internal dependency: no public header includes it, so
    # it must not leak into the exported interface.
    ${SAMCORE_POCKETFFT_INCLUDE_DIR})

target_link_libraries(samcore
  PUBLIC
    HDF5::HDF5
    nlohmann_json::nlohmann_json
  PRIVATE
    # OpenMP is an implementation detail: apply it only in the build tree
    # so the exported package does not reference OpenMP::OpenMP_CXX.
    $<$<BOOL:${SAMCORE_ENABLE_OPENMP}>:$<BUILD_INTERFACE:OpenMP::OpenMP_CXX>>)

target_compile_definitions(samcore PUBLIC SAMCORE_VERSION="${PROJECT_VERSION}")

if(SAMCORE_ENABLE_OPENMP)
  target_compile_definitions(samcore PUBLIC SAMCORE_HAS_OPENMP=1)
endif()

target_compile_options(samcore PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${SAMCORE_WARNINGS_AS_ERRORS}>>:-Werror>)

# ---- build modes ---------------------------------------------------------
# Build-interface targets so the flags reach every final link target
# (tests, executables, python module) without leaking into the installed
# package interface.
add_library(samcore_debug_flags INTERFACE)
target_compile_options(samcore_debug_flags INTERFACE
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<CONFIG:Debug>,$<BOOL:${SAMCORE_ENABLE_SANITIZERS}>>:-fsanitize=address,undefined;-fno-omit-frame-pointer>)
target_link_options(samcore_debug_flags INTERFACE
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<CONFIG:Debug>,$<BOOL:${SAMCORE_ENABLE_SANITIZERS}>>:-fsanitize=address,undefined>)

add_library(samcore_release_flags INTERFACE)
target_compile_options(samcore_release_flags INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:$<$<CONFIG:Release>:-O2>>
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<CONFIG:Release>,$<BOOL:${SAMCORE_ENABLE_FAST_MATH}>>:-ffast-math>
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<CONFIG:Release>,$<BOOL:${SAMCORE_ENABLE_LTO}>>:-flto>)
target_link_options(samcore_release_flags INTERFACE
  $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<CONFIG:Release>,$<BOOL:${SAMCORE_ENABLE_LTO}>>:-flto>)

target_link_libraries(samcore PUBLIC
  $<BUILD_INTERFACE:samcore_debug_flags>
  $<BUILD_INTERFACE:samcore_release_flags>)

set_target_properties(samcore PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# ---- install / export ----------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS samcore
  EXPORT samcoreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT samcoreTargets
  FILE samcoreTargets.cmake
  NAMESPACE samcore::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samcore)

configure_package_config_file(
  cmake/samcoreConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/samcoreConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samcore)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/samcoreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/samcoreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/samcoreConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samcore)

# ---- tests & executables ---------------------------------------------------
if(SAMCORE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SAMCORE_BUILD_EXECUTABLES)
  add_subdirectory(executables)
endif()

if(SAMCORE_BUILD_PYTHON)
  # ---- nanobind extension (samcore python package) ------------------------
  # Built via scikit-build-core (pip install .), which exposes nanobind to
  # CMake automatically; the execute_process lookup is only a fallback for
  # direct -DSAMCORE_BUILD_PYTHON=ON builds outside pip.
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  find_package(nanobind CONFIG QUIET)
  if(NOT nanobind_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NB_CMAKE_DIR)
      list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
    endif()
    find_package(nanobind CONFIG REQUIRED)
  endif()

  # Per-version extension (no abi3): keeps wheel builds uniform across the
  # whole CPython range.
  nanobind_add_module(_samcore NB_DOMAIN numpy src/bindings.cpp)
  target_link_libraries(_samcore PRIVATE samcore)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_samcore PRIVATE -w)
  endif()

  install(TARGETS _samcore LIBRARY DESTINATION samcore)

  # package version, generated from version.txt (single source of truth)
  configure_file("${CMAKE_SOURCE_DIR}/cmake/_version.py.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
          DESTINATION samcore)
endif()

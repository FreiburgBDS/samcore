include(GoogleTest)

set(SAMCORE_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE STRING
  "Directory with the sample test data (testdata.h5sam/.h5samd)")

add_executable(samcore_tests
  test_array.cpp
  test_header.cpp
  test_labels.cpp
  test_io_h5sam.cpp
  test_io_h5samd.cpp
  test_signal.cpp
  test_preprocessing.cpp
  test_utils.cpp
  test_sam_scan.cpp
  test_sam_scan_ops.cpp
  test_omp_parity.cpp
)
target_link_libraries(samcore_tests PRIVATE samcore GTest::gtest_main)
target_compile_definitions(samcore_tests PRIVATE
  SAMCORE_TEST_DATA_DIR="${SAMCORE_TEST_DATA_DIR}")

gtest_discover_tests(samcore_tests)

# Run the full suite again single-threaded: kernels must agree regardless of
# OMP_NUM_THREADS (results are verified against hand-computed references).
if(SAMCORE_ENABLE_OPENMP)
  add_test(NAME samcore_tests_single_threaded
    COMMAND samcore_tests --gtest_filter=omp_parity.*)
  set_tests_properties(samcore_tests_single_threaded PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1")
endif()


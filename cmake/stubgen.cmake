# cmake/stubgen.cmake - run nanobind's stubgen non-fatally.
#
# Invoked with `cmake -P` from the samcore_stub custom target (see
# CMakeLists.txt).  Runs `python -m nanobind.stubgen` against a staged,
# importable copy of the samcore package.
#
# The script deliberately does NOT touch PYTHONPATH: the command inherits
# the build environment, so nanobind must be importable by ${PYTHON} there
# (e.g. from the pip build-isolation overlay); the staged package is added
# to the import path with -i instead.
#
# Stub generation is optional tooling: if it fails for any reason, a warning
# is printed and the script still exits successfully so it can never fail
# the wheel build.  The .pyi files are installed with install(FILES ... OPTIONAL),
# so a failed run simply ships a wheel without stubs.

if(NOT DEFINED STUB_DIR OR NOT DEFINED STUB_PKG OR NOT DEFINED PYTHON)
    message(FATAL_ERROR
        "stubgen.cmake: STUB_DIR, STUB_PKG and PYTHON must be defined")
endif()

set(_modules samcore samcore._samcore samcore._scan samcore._labels
             samcore._dataset samcore._io)

set(_args -i "${STUB_DIR}" -M "${STUB_PKG}/py.typed" -q)
foreach(_m IN LISTS _modules)
    list(APPEND _args -m "${_m}")
endforeach()

execute_process(
    COMMAND "${PYTHON}" -m nanobind.stubgen ${_args}
    RESULT_VARIABLE _res
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)

if(NOT _res EQUAL 0)
    message(WARNING
        "nanobind stub generation failed (exit ${_res}); "
        "skipping .pyi stubs for this build")
    if(_out)
        message(STATUS "stubgen stdout: ${_out}")
    endif()
    if(_err)
        message(STATUS "stubgen stderr: ${_err}")
    endif()
endif()

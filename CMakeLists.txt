# Native build for lauelab.
#
# This is the single maintained build definition for the CPU native
# components. It is consumed both directly
# (cmake -S . -B build) and through scikit-build-core during
# `python -m pip install .`.  Install destinations are relative to the
# install prefix so that scikit-build-core can point them into the wheel.

cmake_minimum_required(VERSION 3.24)

project(lauelab
  VERSION 0.1.0
  DESCRIPTION "Native components for Laue diffraction analysis"
  LANGUAGES C)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LAUE_BUILD_LAUEGO "Build the LaueGo programs peaksearch, pix2qs, and euler" ON)
option(LAUE_BUILD_RECONSTRUCTION "Build the CPU reconstruction program reconstructN_cpu" ON)
option(LAUE_NATIVE_OPTIMIZATION
  "Tune for the build machine (-march=native).  Local use only; never redistribute these artifacts." OFF)
# The supported alpha install places the package under
# <conda-prefix>/lib/pythonX.Y/site-packages. From either package bin directory,
# this relative path resolves to <conda-prefix>/lib without recording an
# absolute build prefix.
set(LAUE_INSTALL_RPATH "$ORIGIN/../../../../.."
  CACHE STRING "RPATH for the supported Conda/source-install layout; resolves from a package bin directory to <prefix>/lib")

# ---------------------------------------------------------------------------
# Global compile settings (numerical contract: -O2, SSE2 baseline, no fast-math)
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(FATAL_ERROR "lauelab currently supports Linux x86-64 only (got ${CMAKE_SYSTEM_PROCESSOR})")
endif()
if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(WARNING "lauelab is validated with GCC; ${CMAKE_C_COMPILER_ID} is unsupported")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# The makefiles built with plain -O2 (no -DNDEBUG, no -O3); keep that exactly.
set(CMAKE_C_FLAGS_RELEASE "-O2" CACHE STRING "Release flags" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "RelWithDebInfo flags" FORCE)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)          # sources rely on gnu99
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RPATH policy for the supported Conda/source-install workflow: write only the
# package-relative entry. Build-tree binaries use the same value, with no
# install-time relink or CMake-generated build RPATH. A toolchain can append
# its own prefix; conda-forge GCC can add <prefix>/lib, but this project does
# not add that absolute entry.
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
set(CMAKE_INSTALL_RPATH "${LAUE_INSTALL_RPATH}")

add_library(laue_common_flags INTERFACE)
# -ffp-contract=off: never fuse multiply-add.  GCC's default ("fast" in gnu
# mode) is inert on the SSE2 baseline, which has no FMA instructions, but
# would change results under LAUE_NATIVE_OPTIMIZATION on FMA-capable CPUs.
target_compile_options(laue_common_flags INTERFACE -msse2 -ffp-contract=off)
if(LAUE_NATIVE_OPTIMIZATION)
  message(STATUS "LAUE_NATIVE_OPTIMIZATION=ON: building with -march=native; do not redistribute")
  target_compile_options(laue_common_flags INTERFACE -march=native)
endif()

# ---------------------------------------------------------------------------
# Dependencies (capabilities, not package managers)
# ---------------------------------------------------------------------------
# Prefer the dependency prefixes that the active toolchain exposes on PATH
# (h5cc).  This is a hint only: it keeps a conda env, /usr/local,
# or an HPC module tree ahead of a stray system copy without branching on
# which package manager provided it.  CMAKE_PREFIX_PATH and *_ROOT still win.
# The Python interpreter we are installing into (set by scikit-build-core)
# identifies the target environment even when no shell activated it, so its
# prefix is searched first for tools and libraries.
set(LAUE_TOOL_HINTS)
foreach(pyvar IN ITEMS Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE)
  if(DEFINED ${pyvar} AND EXISTS "${${pyvar}}")
    get_filename_component(_py_bindir "${${pyvar}}" DIRECTORY)
    get_filename_component(_py_prefix "${_py_bindir}" DIRECTORY)
    list(APPEND LAUE_TOOL_HINTS "${_py_bindir}")
    list(APPEND CMAKE_PREFIX_PATH "${_py_prefix}")
    break()
  endif()
endforeach()

foreach(tool IN ITEMS h5cc)
  find_program(LAUE_${tool}_PATH NAMES ${tool} HINTS ${LAUE_TOOL_HINTS})
  if(LAUE_${tool}_PATH)
    get_filename_component(_bindir "${LAUE_${tool}_PATH}" DIRECTORY)
    get_filename_component(_prefix "${_bindir}" DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
  endif()
endforeach()
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

find_package(OpenMP REQUIRED COMPONENTS C)
if(LAUE_BUILD_LAUEGO OR LAUE_BUILD_RECONSTRUCTION)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  find_package(ZLIB REQUIRED)
  # FindHDF5 copies the definitions HDF5 itself was compiled with (from
  # `h5cc -show`, e.g. -DNDEBUG -D_FORTIFY_SOURCE=2 on conda-forge) onto the
  # imported targets.  Consumers do not need them and they must not leak
  # into this project's compile lines, so keep only H5_* build markers.
  foreach(tgt IN ITEMS hdf5::hdf5 hdf5::hdf5_hl)
    if(TARGET ${tgt})
      get_target_property(_defs ${tgt} INTERFACE_COMPILE_DEFINITIONS)
      if(_defs)
        list(FILTER _defs INCLUDE REGEX "^H5_")
        set_target_properties(${tgt} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_defs}")
      endif()
    endif()
  endforeach()
endif()
include(CheckIPOSupported)
check_ipo_supported(RESULT LAUE_IPO_SUPPORTED OUTPUT LAUE_IPO_MESSAGE LANGUAGES C)

# ---------------------------------------------------------------------------
# Source locations
# ---------------------------------------------------------------------------
set(LAUE_PKG src/lauelab)
set(PEAKSEARCH_SRC ${LAUE_PKG}/indexing/src/peaksearch/src)
set(PIXELS2QS_SRC  ${LAUE_PKG}/indexing/src/pixels2qs/src)
set(EULER_SRC      ${LAUE_PKG}/indexing/src/euler/src)
set(LIBLAUE_SRC    ${LAUE_PKG}/indexing/src/liblaue)
set(MINPACK_SRC    ${LAUE_PKG}/indexing/src/minpack)
set(RECON_CPU_SRC  ${LAUE_PKG}/reconstruct/source/recon_cpu/source)

# Levenberg-Marquardt least squares for the peak fits (MINPACK, see minpack/README.md)
set(MINPACK_SOURCES
  ${MINPACK_SRC}/lmsder.c
  ${MINPACK_SRC}/lmpar.c
  ${MINPACK_SRC}/qrfac.c
  ${MINPACK_SRC}/qrsolv.c
  ${MINPACK_SRC}/enorm.c
  ${MINPACK_SRC}/dpmpar.c)

set(LAUE_INDEXING_BIN_DIR   lauelab/indexing/bin)
set(LAUE_RECONSTRUCT_BIN_DIR lauelab/reconstruct/bin)

# ---------------------------------------------------------------------------
# liblaue.so — in-process indexing library (required)
# Compiles the shared LaueGo sources a second time with -DLIBLAUE_BUILD, so
# it cannot reuse the executables' objects.
# ---------------------------------------------------------------------------
add_library(laue SHARED
  ${LIBLAUE_SRC}/liblaue.c
  ${LIBLAUE_SRC}/liblaue_recon.c
  ${RECON_CPU_SRC}/cosmicFilter.c
  ${PIXELS2QS_SRC}/readGeoN.c
  ${PIXELS2QS_SRC}/keyLists.c
  ${PIXELS2QS_SRC}/checkFileType.c
  ${PIXELS2QS_SRC}/xmlUtility.c
  ${PIXELS2QS_SRC}/main.c
  ${PEAKSEARCH_SRC}/peaksearch.c
  ${PEAKSEARCH_SRC}/grid.c
  ${PEAKSEARCH_SRC}/grid_operations.c
  ${PEAKSEARCH_SRC}/fitToFunction.c
  ${PEAKSEARCH_SRC}/fitToGaussian.c
  ${PEAKSEARCH_SRC}/point.c
  ${PEAKSEARCH_SRC}/list.c
  ${PEAKSEARCH_SRC}/peak.c
  ${PEAKSEARCH_SRC}/genfileinf.c
  ${PEAKSEARCH_SRC}/calibparam.c
  ${PEAKSEARCH_SRC}/ccdTable.c
  ${EULER_SRC}/fast.c
  ${EULER_SRC}/EulerTestFaster.c
  ${EULER_SRC}/lattice.c
  ${EULER_SRC}/mathUtil.c
  ${EULER_SRC}/nmsimplex.c
  ${EULER_SRC}/optimize.c
  ${MINPACK_SOURCES})
set_target_properties(laue PROPERTIES
  OUTPUT_NAME laue
  PREFIX "lib"
  SUFFIX ".so"
  NO_SONAME ON
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_compile_definitions(laue PRIVATE LIBLAUE_BUILD)
target_compile_options(laue PRIVATE -ffunction-sections -fdata-sections)
target_link_options(laue PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)
target_include_directories(laue PRIVATE ${LIBLAUE_SRC} ${PIXELS2QS_SRC} ${PEAKSEARCH_SRC} ${EULER_SRC} ${MINPACK_SRC})
target_link_libraries(laue PRIVATE laue_common_flags OpenMP::OpenMP_C m)
if(LAUE_IPO_SUPPORTED)
  set_property(TARGET laue PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
else()
  message(WARNING "LTO not supported for liblaue: ${LAUE_IPO_MESSAGE}")
endif()
install(TARGETS laue LIBRARY DESTINATION ${LAUE_INDEXING_BIN_DIR})

# ---------------------------------------------------------------------------
# LaueGo programs
# ---------------------------------------------------------------------------
if(LAUE_BUILD_LAUEGO)
  add_executable(peaksearch
    ${PEAKSEARCH_SRC}/WinViewHeader.c
    ${PEAKSEARCH_SRC}/WinViewImage.c
    ${PEAKSEARCH_SRC}/WinViewRead.c
    ${PEAKSEARCH_SRC}/calibparam.c
    ${PEAKSEARCH_SRC}/ccdTable.c
    ${PEAKSEARCH_SRC}/fitToFunction.c
    ${PEAKSEARCH_SRC}/fitToGaussian.c
    ${PEAKSEARCH_SRC}/genfileinf.c
    ${PEAKSEARCH_SRC}/grid.c
    ${PEAKSEARCH_SRC}/grid_operations.c
    ${PEAKSEARCH_SRC}/list.c
    ${PEAKSEARCH_SRC}/main.c
    ${PEAKSEARCH_SRC}/microHDF5.c
    ${PEAKSEARCH_SRC}/peak.c
    ${PEAKSEARCH_SRC}/peaksearch.c
    ${PEAKSEARCH_SRC}/point.c
    ${MINPACK_SOURCES})
  target_include_directories(peaksearch PRIVATE ${PEAKSEARCH_SRC} ${MINPACK_SRC})
  target_link_libraries(peaksearch PRIVATE laue_common_flags hdf5::hdf5_hl hdf5::hdf5 ZLIB::ZLIB m)

  add_executable(pix2qs
    ${PIXELS2QS_SRC}/checkFileType.c
    ${PIXELS2QS_SRC}/keyLists.c
    ${PIXELS2QS_SRC}/lattice.c
    ${PIXELS2QS_SRC}/main.c
    ${PIXELS2QS_SRC}/readGeoN.c
    ${PIXELS2QS_SRC}/xmlUtility.c)
  target_include_directories(pix2qs PRIVATE ${PIXELS2QS_SRC})
  target_link_libraries(pix2qs PRIVATE laue_common_flags m)

  add_executable(euler
    ${EULER_SRC}/EulerTestFaster.c
    ${EULER_SRC}/checkFileType.c
    ${EULER_SRC}/fast.c
    ${EULER_SRC}/lattice.c
    ${EULER_SRC}/main.c
    ${EULER_SRC}/mathUtil.c
    ${EULER_SRC}/nmsimplex.c
    ${EULER_SRC}/optimize.c)
  target_include_directories(euler PRIVATE ${EULER_SRC})
  target_link_libraries(euler PRIVATE laue_common_flags m)

  install(TARGETS peaksearch pix2qs euler RUNTIME DESTINATION ${LAUE_INDEXING_BIN_DIR})
endif()

# ---------------------------------------------------------------------------
# CPU reconstruction
# ---------------------------------------------------------------------------
if(LAUE_BUILD_RECONSTRUCTION)
  add_executable(reconstructN_cpu
    ${RECON_CPU_SRC}/WireScan.c
    ${RECON_CPU_SRC}/checkFileType.c
    ${RECON_CPU_SRC}/cosmicFilter.c
    ${RECON_CPU_SRC}/hardwareSpecific.c
    ${RECON_CPU_SRC}/mathUtil.c
    ${RECON_CPU_SRC}/microHDF5.c
    ${RECON_CPU_SRC}/misc.c
    ${RECON_CPU_SRC}/readGeoN.c
    ${RECON_CPU_SRC}/xmlUtility.c)
  target_compile_definitions(reconstructN_cpu PRIVATE RECONSTRUCT_BACKWARDS MULTI_IMAGE_FILE)
  target_include_directories(reconstructN_cpu PRIVATE ${RECON_CPU_SRC})
  target_link_libraries(reconstructN_cpu PRIVATE laue_common_flags OpenMP::OpenMP_C
    hdf5::hdf5_hl hdf5::hdf5 ZLIB::ZLIB m)
  install(TARGETS reconstructN_cpu RUNTIME DESTINATION ${LAUE_RECONSTRUCT_BIN_DIR})
endif()

# ---------------------------------------------------------------------------
# Post-install RPATH normalization
# conda-forge GCC appends <prefix>/lib to the RPATH of linked files. Rewrite
# each installed native file with the relative Conda/source-install RPATH so
# the artifact does not contain an absolute build prefix. Failure stops the
# installation.
# ---------------------------------------------------------------------------
find_program(LAUE_PATCHELF NAMES patchelf HINTS ${LAUE_TOOL_HINTS})
if(NOT LAUE_PATCHELF)
  message(FATAL_ERROR "patchelf is required to normalise installed RPATHs "
    "(pip install patchelf, or conda install -c conda-forge patchelf)")
endif()
set(LAUE_INSTALLED_NATIVE_FILES "${LAUE_INDEXING_BIN_DIR}/liblaue.so")
if(LAUE_BUILD_LAUEGO)
  list(APPEND LAUE_INSTALLED_NATIVE_FILES
    "${LAUE_INDEXING_BIN_DIR}/peaksearch" "${LAUE_INDEXING_BIN_DIR}/pix2qs" "${LAUE_INDEXING_BIN_DIR}/euler")
endif()
if(LAUE_BUILD_RECONSTRUCTION)
  list(APPEND LAUE_INSTALLED_NATIVE_FILES "${LAUE_RECONSTRUCT_BIN_DIR}/reconstructN_cpu")
endif()
install(CODE "
  foreach(rel IN ITEMS ${LAUE_INSTALLED_NATIVE_FILES})
    set(f \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/\${rel}\")
    execute_process(COMMAND \"${LAUE_PATCHELF}\" --set-rpath \"${LAUE_INSTALL_RPATH}\" \"\${f}\"
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR \"patchelf failed on \${f}: \${err}\")
    endif()
    message(STATUS \"RPATH set to ${LAUE_INSTALL_RPATH}: \${f}\")
  endforeach()
")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "lauelab native build:")
message(STATUS "  build type            ${CMAKE_BUILD_TYPE}")
message(STATUS "  LAUE_BUILD_LAUEGO     ${LAUE_BUILD_LAUEGO}")
message(STATUS "  LAUE_BUILD_RECONSTRUCTION ${LAUE_BUILD_RECONSTRUCTION}")
message(STATUS "  LAUE_NATIVE_OPTIMIZATION  ${LAUE_NATIVE_OPTIMIZATION}")
message(STATUS "  install RPATH         ${CMAKE_INSTALL_RPATH}")
if(HDF5_FOUND)
  message(STATUS "  HDF5                  ${HDF5_VERSION} (${HDF5_C_INCLUDE_DIRS})")
endif()

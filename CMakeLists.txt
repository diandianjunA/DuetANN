cmake_minimum_required(VERSION 3.18)

option(DUETANN_STORAGE_NODE_ONLY
    "Build storage-host runtime and CPU offline tools without CUDA/GPU dependencies" OFF)
if (NOT DUETANN_STORAGE_NODE_ONLY AND
    NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
    "$ENV{CUDAARCHS}" STREQUAL "")
    # Select the default before project(... CUDA); CMake otherwise initializes
    # it from nvcc's compiler default and may select pre-SM80 code.
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif ()

if (DUETANN_STORAGE_NODE_ONLY)
    project(duetann CXX)
else ()
    project(duetann CXX CUDA)
endif ()

set(DUETANN_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH
    "${DUETANN_SOURCE_ROOT}"
    "${DUETANN_SOURCE_ROOT}/cmake")

include(DuetANNOptions)
include(DuetANNCompiler)
include(DuetANNSources)
include(DuetANNDependencies)
include(DuetANNTargetHelpers)
include(DuetANNOfflineTools)

if (DUETANN_BUILD_OFFLINE_TOOLS)
    find_package(FaissCPU REQUIRED)
endif ()

include_directories(
    ${DUETANN_SOURCE_ROOT}/src
    ${DUETANN_SOURCE_ROOT}/thirdparty)
add_subdirectory(rdma-library)

add_executable(duetann_memory_node
    src/memory_node_main.cc
    ${DUETANN_MEMORY_NODE_SOURCES}
    src/common/core_assignment.cc
    src/common/timing.cc
    src/service/index_metadata.cc)
duetann_target_common_includes(duetann_memory_node)
target_link_libraries(duetann_memory_node rdma_library)

if (DUETANN_STORAGE_NODE_ONLY)
    message(STATUS
        "Storage-node build: CUDA and DOCA are disabled; CPU offline tools=${DUETANN_BUILD_OFFLINE_TOOLS}")
    if (DUETANN_BUILD_OFFLINE_TOOLS)
        duetann_add_offline_tools()
    endif ()
    if (DUETANN_BUILD_TESTS)
        enable_testing()
        include(DuetANNTestHelpers)
        include(test/core_tests.cmake)
        include(test/offline_tests.cmake)
    endif ()
else ()
    include(DuetANNDoca)
    include(DuetANNGpuTargets)
    include(DuetANNComputeTargets)

    get_target_property(compile_options duetann_runtime COMPILE_OPTIONS)
    message(STATUS "Compile options: ${compile_options}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "DOCA GPUNetIO backend: ${DUETANN_HAVE_GPUNETIO}")
endif ()

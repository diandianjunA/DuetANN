cmake_minimum_required(VERSION 3.18)

option(DVSTOR_STORAGE_NODE_ONLY
    "Build storage-host runtime and CPU offline tools without CUDA/GPU dependencies" OFF)
if (NOT DVSTOR_STORAGE_NODE_ONLY AND
    NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
    "$ENV{CUDAARCHS}" STREQUAL "")
    # Select the default before project(... CUDA); CMake otherwise initializes
    # it from nvcc's compiler default and may select pre-SM80 code.
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif ()

if (DVSTOR_STORAGE_NODE_ONLY)
    project(dvstor CXX)
else ()
    project(dvstor CXX CUDA)
endif ()

set(DVSTOR_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH
    "${DVSTOR_SOURCE_ROOT}"
    "${DVSTOR_SOURCE_ROOT}/cmake")

include(DvstorOptions)
include(DvstorCompiler)
include(DvstorSources)
include(DvstorDependencies)
include(DvstorTargetHelpers)
include(DvstorOfflineTools)

if (DVSTOR_BUILD_OFFLINE_TOOLS)
    find_package(FaissCPU REQUIRED)
endif ()

include_directories(
    ${DVSTOR_SOURCE_ROOT}/src
    ${DVSTOR_SOURCE_ROOT}/thirdparty)
add_subdirectory(rdma-library)

add_executable(dvstor_memory_node
    src/memory_node_main.cc
    ${DVSTOR_MEMORY_NODE_SOURCES}
    src/common/core_assignment.cc
    src/common/timing.cc
    src/service/index_metadata.cc)
dvstor_target_common_includes(dvstor_memory_node)
target_link_libraries(dvstor_memory_node rdma_library)

if (DVSTOR_STORAGE_NODE_ONLY)
    message(STATUS
        "Storage-node build: CUDA and DOCA are disabled; CPU offline tools=${DVSTOR_BUILD_OFFLINE_TOOLS}")
    if (DVSTOR_BUILD_OFFLINE_TOOLS)
        dvstor_add_offline_tools()
    endif ()
    if (DVSTOR_BUILD_TESTS)
        enable_testing()
        include(DvstorTestHelpers)
        include(test/core_tests.cmake)
        include(test/offline_tests.cmake)
    endif ()
else ()
    include(DvstorDoca)
    include(DvstorGpuTargets)
    include(DvstorComputeTargets)

    get_target_property(compile_options dvstor_runtime COMPILE_OPTIONS)
    message(STATUS "Compile options: ${compile_options}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "DOCA GPUNetIO backend: ${DVSTOR_HAVE_GPUNETIO}")
endif ()

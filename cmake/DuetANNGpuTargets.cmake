include_guard(GLOBAL)

set(DUETANN_GPU_SOURCES src/gpu_search/host_distance_kernel.cu)
if (DUETANN_HAVE_GPUNETIO)
    list(APPEND DUETANN_GPU_SOURCES
        src/gpu_search/persistent_kernel.cu
        src/gpu/gpunetio_probe.cu)
endif ()

add_library(duetann_gpu_kernels STATIC ${DUETANN_GPU_SOURCES})
duetann_target_common_includes(duetann_gpu_kernels PUBLIC)
target_link_libraries(duetann_gpu_kernels CUDA::cudart rdma_library)
target_compile_definitions(duetann_gpu_kernels PUBLIC
    DUETANN_PERSISTENT_WIDE_SORT=$<BOOL:${DUETANN_PERSISTENT_WIDE_SORT}>
    DUETANN_PERSISTENT_LEGACY_MERGE=$<BOOL:${DUETANN_PERSISTENT_LEGACY_MERGE}>
    DUETANN_PERSISTENT_COMPACT_LEGACY_SCRATCH=$<BOOL:${DUETANN_PERSISTENT_COMPACT_LEGACY_SCRATCH}>
    DUETANN_PERSISTENT_MAX_BEAM=${DUETANN_PERSISTENT_MAX_BEAM}
    DUETANN_PERSISTENT_MAX_EXACT=${DUETANN_PERSISTENT_MAX_EXACT})
if (DUETANN_HAVE_GPUNETIO)
    target_include_directories(duetann_gpu_kernels PUBLIC ${DUETANN_DOCA_INCLUDE_DIR})
    target_compile_definitions(duetann_gpu_kernels PUBLIC DUETANN_HAVE_GPUNETIO=1)
endif ()
set_target_properties(duetann_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(duetann_gpu_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

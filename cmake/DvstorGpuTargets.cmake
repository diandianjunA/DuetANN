include_guard(GLOBAL)

set(DVSTOR_GPU_SOURCES src/gpu_search/host_distance_kernel.cu)
if (DVSTOR_HAVE_GPUNETIO)
    list(APPEND DVSTOR_GPU_SOURCES
        src/gpu_search/persistent_kernel.cu
        src/gpu/gpunetio_probe.cu)
endif ()

add_library(dvstor_gpu_kernels STATIC ${DVSTOR_GPU_SOURCES})
dvstor_target_common_includes(dvstor_gpu_kernels PUBLIC)
target_link_libraries(dvstor_gpu_kernels CUDA::cudart rdma_library)
target_compile_definitions(dvstor_gpu_kernels PUBLIC
    DVSTOR_PERSISTENT_WIDE_SORT=$<BOOL:${DVSTOR_PERSISTENT_WIDE_SORT}>
    DVSTOR_PERSISTENT_LEGACY_MERGE=$<BOOL:${DVSTOR_PERSISTENT_LEGACY_MERGE}>
    DVSTOR_PERSISTENT_MAX_BEAM=${DVSTOR_PERSISTENT_MAX_BEAM}
    DVSTOR_PERSISTENT_MAX_EXACT=${DVSTOR_PERSISTENT_MAX_EXACT})
if (DVSTOR_HAVE_GPUNETIO)
    target_include_directories(dvstor_gpu_kernels PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
    target_compile_definitions(dvstor_gpu_kernels PUBLIC DVSTOR_HAVE_GPUNETIO=1)
endif ()
set_target_properties(dvstor_gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(dvstor_gpu_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

set(DUETANN_TEST_SOURCE_DIR "${DUETANN_SOURCE_ROOT}/test/gpu")

if (NOT TARGET duetann_gpu_kernels)
    return()
endif ()

add_executable(host_distance_kernel_test
    ${DUETANN_TEST_SOURCE_DIR}/host_distance_kernel_test.cu)
target_link_libraries(host_distance_kernel_test
    PRIVATE duetann_gpu_kernels CUDA::cudart)
add_test(NAME host_distance_kernel_test COMMAND host_distance_kernel_test)
set_tests_properties(host_distance_kernel_test PROPERTIES
    TIMEOUT 15
    RESOURCE_LOCK duetann_gpu
    SKIP_REGULAR_EXPRESSION "^SKIP:")

duetann_add_cuda_test(gpu_centroid_route_ranking_test 15)
duetann_add_cuda_test(gpu_warp_reduction_test 15)
duetann_add_cuda_test(gpu_compact_beam_merge_test 15)
duetann_add_cuda_test(gpu_pq_packed_scoring_test 15)
duetann_add_cuda_test(gpu_pq_scoring_microbench 120)
duetann_add_cuda_test(gpu_opq_rotation_microbench 60)
duetann_add_cuda_test(gpu_beam_merge_equivalence_test 15)
duetann_add_cuda_test(gpu_stable_run_merge_microbench 30)
duetann_add_cuda_test(gpu_stable_frontier_preview_test 30)
duetann_add_cuda_test(gpu_partition_bounded_certificate_test 30)
duetann_add_cuda_test(gpu_dominance_envelope_certificate_test 60)
duetann_add_cuda_test(gpu_fused_prefix_materialization_test 30)
duetann_add_cuda_test(gpu_frontier_rob_test 30)

add_executable(gpu_persistent_dispatch_test
    ${DUETANN_TEST_SOURCE_DIR}/gpu_persistent_dispatch_test.cc)
target_link_libraries(gpu_persistent_dispatch_test PRIVATE duetann_runtime)
add_test(NAME gpu_persistent_dispatch_test COMMAND gpu_persistent_dispatch_test)
add_test(NAME gpu_persistent_dispatch_pq32_test
    COMMAND gpu_persistent_dispatch_test 1024 10 8 32)
add_test(NAME gpu_persistent_dispatch_pq20_test
    COMMAND gpu_persistent_dispatch_test 128 10 8 20)
add_test(NAME gpu_persistent_dispatch_pq25_test
    COMMAND gpu_persistent_dispatch_test 128 10 8 25)
add_test(NAME gpu_persistent_dispatch_256_test
    COMMAND gpu_persistent_dispatch_test 128 10 8 32 256)
# Persistent kernels must not share device 0 with another test process.
set_tests_properties(
    gpu_persistent_dispatch_test
    gpu_persistent_dispatch_pq32_test
    gpu_persistent_dispatch_pq20_test
    gpu_persistent_dispatch_pq25_test
    gpu_persistent_dispatch_256_test PROPERTIES
    TIMEOUT 180
    RESOURCE_LOCK duetann_gpu
    SKIP_REGULAR_EXPRESSION "^SKIP:")

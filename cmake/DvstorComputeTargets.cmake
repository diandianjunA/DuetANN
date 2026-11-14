include_guard(GLOBAL)

if (DVSTOR_HAVE_GPUNETIO)
    list(APPEND DVSTOR_RUNTIME_SOURCES src/gpu/gpunetio_transport.cc)
    set_source_files_properties(src/gpu/gpunetio_transport.cc PROPERTIES
        COMPILE_OPTIONS "-Wno-deprecated-declarations;-Wno-pedantic")
else ()
    list(APPEND DVSTOR_RUNTIME_SOURCES src/gpu_search/persistent_kernel_stub.cc)
endif ()

add_library(dvstor_runtime STATIC ${DVSTOR_RUNTIME_SOURCES})
dvstor_target_common_includes(dvstor_runtime PUBLIC)
target_link_libraries(dvstor_runtime rdma_library CUDA::cudart)
if (DVSTOR_HAVE_GPUNETIO)
    target_include_directories(dvstor_runtime PUBLIC ${DVSTOR_DOCA_INCLUDE_DIR})
    target_compile_definitions(dvstor_runtime PUBLIC DVSTOR_HAVE_GPUNETIO=1)
    target_link_libraries(dvstor_runtime
        ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
        ${DVSTOR_DOCA_VERBS_LIBRARY}
        ${DVSTOR_DOCA_COMMON_LIBRARY}
        ${DVSTOR_DOCA_RDMA_LIBRARY}
        ${DVSTOR_MLX5_LIBRARY})
endif ()
target_link_libraries(dvstor_runtime dvstor_gpu_kernels)
target_compile_options(dvstor_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

if (DVSTOR_BUILD_EXECUTABLES OR DVSTOR_BUILD_TESTS)
    add_library(dvstor_breakdown_support STATIC ${DVSTOR_BREAKDOWN_SUPPORT_SOURCES})
    dvstor_target_tool_includes(dvstor_breakdown_support)
    target_link_libraries(dvstor_breakdown_support PUBLIC dvstor_runtime)
endif ()

if (DVSTOR_BUILD_EXECUTABLES)
    add_executable(dvstor_compute_node src/main.cc)
    target_link_libraries(dvstor_compute_node dvstor_runtime)

    if (DVSTOR_HAVE_GPUNETIO)
        add_executable(dvstor_gpunetio_probe tools/probes/gpunetio_probe.cc)
        dvstor_target_common_includes(dvstor_gpunetio_probe)
        target_include_directories(dvstor_gpunetio_probe PRIVATE ${DVSTOR_DOCA_INCLUDE_DIR})
        target_compile_definitions(dvstor_gpunetio_probe PRIVATE ALLOW_EXPERIMENTAL_API=1)
        target_compile_options(dvstor_gpunetio_probe PRIVATE
            -Wno-deprecated-declarations
            -Wno-pedantic)
        target_link_libraries(dvstor_gpunetio_probe
            rdma_library
            CUDA::cudart
            ${DVSTOR_DOCA_GPUNETIO_LIBRARY}
            ${DVSTOR_DOCA_VERBS_LIBRARY}
            ${DVSTOR_DOCA_COMMON_LIBRARY}
            ${DVSTOR_MLX5_LIBRARY})

        add_executable(dvstor_gpunetio_loopback_probe
            tools/probes/gpunetio_loopback_probe.cc
            tools/breakdown_benchmark/args.cc)
        dvstor_target_tool_includes(dvstor_gpunetio_loopback_probe)
        target_link_libraries(dvstor_gpunetio_loopback_probe dvstor_runtime)
    endif ()

    add_executable(dvstor_breakdown_benchmark
        tools/breakdown_benchmark/dvstor_breakdown_benchmark.cc
        ${DVSTOR_BREAKDOWN_BENCHMARK_SOURCES})
    dvstor_target_tool_includes(dvstor_breakdown_benchmark)
    target_link_libraries(dvstor_breakdown_benchmark dvstor_breakdown_support)

    add_executable(dvstor_base_extent_churn
        experiment/contribution/c2_sensitivity/base_extent_churn.cc)
    dvstor_target_tool_includes(dvstor_base_extent_churn)

    add_executable(dvstor_long_insert_recall
        tools/long_insert_recall/dvstor_long_insert_recall.cc
        experiment/contribution/c1_locality/locality_sampler.cc
        tools/breakdown_benchmark/args.cc
        tools/breakdown_benchmark/progress.cc)
    dvstor_target_tool_includes(dvstor_long_insert_recall)
    target_link_libraries(dvstor_long_insert_recall dvstor_runtime)
    find_package(FaissCPU QUIET)
    if (FaissCPU_FOUND)
        target_link_libraries(dvstor_long_insert_recall FaissCPU::FaissCPU)
        target_compile_definitions(dvstor_long_insert_recall PRIVATE
            DVSTOR_LONG_INSERT_WITH_FAISS_CPU=1)
    endif ()
endif ()

if (DVSTOR_BUILD_OFFLINE_TOOLS)
    dvstor_add_offline_tools()
endif ()

if (DVSTOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

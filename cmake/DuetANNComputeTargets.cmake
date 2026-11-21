include_guard(GLOBAL)

if (DUETANN_HAVE_GPUNETIO)
    list(APPEND DUETANN_RUNTIME_SOURCES src/gpu/gpunetio_transport.cc)
    set_source_files_properties(src/gpu/gpunetio_transport.cc PROPERTIES
        COMPILE_OPTIONS "-Wno-deprecated-declarations;-Wno-pedantic")
else ()
    list(APPEND DUETANN_RUNTIME_SOURCES src/gpu_search/persistent_kernel_stub.cc)
endif ()

add_library(duetann_runtime STATIC ${DUETANN_RUNTIME_SOURCES})
duetann_target_common_includes(duetann_runtime PUBLIC)
target_link_libraries(duetann_runtime rdma_library CUDA::cudart)
if (DUETANN_HAVE_GPUNETIO)
    target_include_directories(duetann_runtime PUBLIC ${DUETANN_DOCA_INCLUDE_DIR})
    target_compile_definitions(duetann_runtime PUBLIC DUETANN_HAVE_GPUNETIO=1)
    target_link_libraries(duetann_runtime
        ${DUETANN_DOCA_GPUNETIO_LIBRARY}
        ${DUETANN_DOCA_VERBS_LIBRARY}
        ${DUETANN_DOCA_COMMON_LIBRARY}
        ${DUETANN_DOCA_RDMA_LIBRARY}
        ${DUETANN_MLX5_LIBRARY})
endif ()
target_link_libraries(duetann_runtime duetann_gpu_kernels)
target_compile_options(duetann_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

if (DUETANN_BUILD_EXECUTABLES OR DUETANN_BUILD_TESTS)
    add_library(duetann_breakdown_support STATIC ${DUETANN_BREAKDOWN_SUPPORT_SOURCES})
    duetann_target_tool_includes(duetann_breakdown_support)
    target_link_libraries(duetann_breakdown_support PUBLIC duetann_runtime)
endif ()

if (DUETANN_BUILD_EXECUTABLES)
    add_executable(duetann_compute_node src/main.cc)
    target_link_libraries(duetann_compute_node duetann_runtime)

    if (DUETANN_HAVE_GPUNETIO)
        add_executable(duetann_gpunetio_probe tools/probes/gpunetio_probe.cc)
        duetann_target_common_includes(duetann_gpunetio_probe)
        target_include_directories(duetann_gpunetio_probe PRIVATE ${DUETANN_DOCA_INCLUDE_DIR})
        target_compile_definitions(duetann_gpunetio_probe PRIVATE ALLOW_EXPERIMENTAL_API=1)
        target_compile_options(duetann_gpunetio_probe PRIVATE
            -Wno-deprecated-declarations
            -Wno-pedantic)
        target_link_libraries(duetann_gpunetio_probe
            rdma_library
            CUDA::cudart
            ${DUETANN_DOCA_GPUNETIO_LIBRARY}
            ${DUETANN_DOCA_VERBS_LIBRARY}
            ${DUETANN_DOCA_COMMON_LIBRARY}
            ${DUETANN_MLX5_LIBRARY})

        add_executable(duetann_gpunetio_loopback_probe
            tools/probes/gpunetio_loopback_probe.cc
            tools/breakdown_benchmark/args.cc)
        duetann_target_tool_includes(duetann_gpunetio_loopback_probe)
        target_link_libraries(duetann_gpunetio_loopback_probe duetann_runtime)
    endif ()

    add_executable(duetann_breakdown_benchmark
        tools/breakdown_benchmark/duetann_breakdown_benchmark.cc
        ${DUETANN_BREAKDOWN_BENCHMARK_SOURCES})
    duetann_target_tool_includes(duetann_breakdown_benchmark)
    target_link_libraries(duetann_breakdown_benchmark duetann_breakdown_support)

    add_executable(duetann_base_extent_churn
        experiment/contribution/c2_sensitivity/base_extent_churn.cc)
    duetann_target_tool_includes(duetann_base_extent_churn)

    add_executable(duetann_long_insert_recall
        tools/long_insert_recall/duetann_long_insert_recall.cc
        experiment/contribution/c1_locality/locality_sampler.cc
        tools/breakdown_benchmark/args.cc
        tools/breakdown_benchmark/progress.cc)
    duetann_target_tool_includes(duetann_long_insert_recall)
    target_link_libraries(duetann_long_insert_recall duetann_runtime)
    find_package(FaissCPU QUIET)
    if (FaissCPU_FOUND)
        target_link_libraries(duetann_long_insert_recall FaissCPU::FaissCPU)
        target_compile_definitions(duetann_long_insert_recall PRIVATE
            DUETANN_LONG_INSERT_WITH_FAISS_CPU=1)
    endif ()
endif ()

if (DUETANN_BUILD_OFFLINE_TOOLS)
    duetann_add_offline_tools()
endif ()

if (DUETANN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

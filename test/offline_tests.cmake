set(DVSTOR_TEST_SOURCE_DIR "${DVSTOR_SOURCE_ROOT}/test/offline")

dvstor_add_test(offline_local_id_set_test)
dvstor_target_tool_includes(offline_local_id_set_test)
set_tests_properties(offline_local_id_set_test PROPERTIES TIMEOUT 10)

if (DVSTOR_BUILD_OFFLINE_TOOLS)
    dvstor_add_test(shard_publication_test
        ${DVSTOR_TEST_SOURCE_DIR}/shard_publication_test.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/checkpoint.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/dataset_io.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/graph.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/partitioning.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/shard_writer.cc)
    dvstor_target_tool_includes(shard_publication_test)
    target_link_libraries(shard_publication_test PRIVATE rdma_library)
    set_tests_properties(shard_publication_test PROPERTIES TIMEOUT 20)

    dvstor_add_test(vamana_offline_graph_test
        ${DVSTOR_TEST_SOURCE_DIR}/vamana_offline_graph_test.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/checkpoint.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/dataset_io.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/graph.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc)
    dvstor_target_tool_includes(vamana_offline_graph_test)
    target_link_libraries(vamana_offline_graph_test PRIVATE rdma_library)
    set_tests_properties(vamana_offline_graph_test PROPERTIES TIMEOUT 10)

    if (TARGET FaissCPU::FaissCPU)
        dvstor_add_test(metis_repartitioner_test
            ${DVSTOR_TEST_SOURCE_DIR}/metis_repartitioner_test.cc
            ${DVSTOR_METIS_REPARTITIONER_SOURCES})
        dvstor_target_tool_includes(metis_repartitioner_test)
        target_link_libraries(metis_repartitioner_test
            rdma_library FaissCPU::FaissCPU)
        target_link_metis(metis_repartitioner_test)
        set_tests_properties(metis_repartitioner_test PROPERTIES
            TIMEOUT 180
            SKIP_RETURN_CODE 77)
    endif ()

    dvstor_add_test(kmeans_partition_test
        ${DVSTOR_TEST_SOURCE_DIR}/kmeans_partition_test.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/dataset_io.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/partitioning.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc)
    dvstor_target_tool_includes(kmeans_partition_test)
    target_link_libraries(kmeans_partition_test PRIVATE rdma_library)
    set_tests_properties(kmeans_partition_test PROPERTIES TIMEOUT 60)
endif ()

if (NOT DVSTOR_STORAGE_NODE_ONLY AND TARGET FaissCPU::FaissCPU)
    dvstor_add_test(legacy_index_converter_test
        ${DVSTOR_TEST_SOURCE_DIR}/legacy_index_converter_test.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/legacy_index_converter.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/pq_indexer.cc
        ${DVSTOR_SOURCE_ROOT}/tools/vamana_offline/progress.cc
        ${DVSTOR_SOURCE_ROOT}/src/gpu_search/index_format.cc
        ${DVSTOR_SOURCE_ROOT}/src/gpu_search/pq_index.cc)
    dvstor_target_tool_includes(legacy_index_converter_test)
    target_link_libraries(legacy_index_converter_test PRIVATE
        rdma_library FaissCPU::FaissCPU)

    dvstor_add_test(pq_encoder_equivalence_test)
    target_link_libraries(pq_encoder_equivalence_test PRIVATE
        dvstor_runtime FaissCPU::FaissCPU)
endif ()

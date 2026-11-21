set(DUETANN_TEST_SOURCE_DIR "${DUETANN_SOURCE_ROOT}/test/core")

duetann_add_test(vector_dtype_test)

duetann_add_test(early_prune_equivalence_test
    ${DUETANN_TEST_SOURCE_DIR}/early_prune_equivalence.cc)
set_tests_properties(early_prune_equivalence_test PROPERTIES TIMEOUT 60)

duetann_add_test(index_metadata_test
    ${DUETANN_TEST_SOURCE_DIR}/index_metadata_test.cc
    ${DUETANN_SOURCE_ROOT}/src/service/index_metadata.cc
    ${DUETANN_SOURCE_ROOT}/src/gpu_search/index_format.cc)
set_tests_properties(index_metadata_test PROPERTIES TIMEOUT 10)

duetann_add_test(rdma_contract_test)
target_link_libraries(rdma_contract_test PRIVATE rdma_library)
set_tests_properties(rdma_contract_test PROPERTIES TIMEOUT 10)

duetann_add_test(server_connection_reconnect_test)
target_link_libraries(server_connection_reconnect_test PRIVATE rdma_library)
set_tests_properties(server_connection_reconnect_test PROPERTIES
    TIMEOUT 30
    SKIP_REGULAR_EXPRESSION "^SKIP:")

duetann_add_test(configuration_update_protocol_test)
target_sources(configuration_update_protocol_test PRIVATE
    ${DUETANN_SOURCE_ROOT}/src/common/configuration.cc)
target_link_libraries(configuration_update_protocol_test PRIVATE rdma_library)
if (DUETANN_STORAGE_NODE_ONLY)
    set_tests_properties(configuration_update_protocol_test PROPERTIES TIMEOUT 10)
endif ()

set(DVSTOR_TEST_SOURCE_DIR "${DVSTOR_SOURCE_ROOT}/test/core")

dvstor_add_test(vector_dtype_test)

dvstor_add_test(early_prune_equivalence_test
    ${DVSTOR_TEST_SOURCE_DIR}/early_prune_equivalence.cc)
set_tests_properties(early_prune_equivalence_test PROPERTIES TIMEOUT 60)

dvstor_add_test(index_metadata_test
    ${DVSTOR_TEST_SOURCE_DIR}/index_metadata_test.cc
    ${DVSTOR_SOURCE_ROOT}/src/service/index_metadata.cc
    ${DVSTOR_SOURCE_ROOT}/src/gpu_search/index_format.cc)
set_tests_properties(index_metadata_test PROPERTIES TIMEOUT 10)

dvstor_add_test(rdma_contract_test)
target_link_libraries(rdma_contract_test PRIVATE rdma_library)
set_tests_properties(rdma_contract_test PROPERTIES TIMEOUT 10)

dvstor_add_test(server_connection_reconnect_test)
target_link_libraries(server_connection_reconnect_test PRIVATE rdma_library)
set_tests_properties(server_connection_reconnect_test PROPERTIES
    TIMEOUT 30
    SKIP_REGULAR_EXPRESSION "^SKIP:")

dvstor_add_test(configuration_update_protocol_test)
target_sources(configuration_update_protocol_test PRIVATE
    ${DVSTOR_SOURCE_ROOT}/src/common/configuration.cc)
target_link_libraries(configuration_update_protocol_test PRIVATE rdma_library)
if (DVSTOR_STORAGE_NODE_ONLY)
    set_tests_properties(configuration_update_protocol_test PROPERTIES TIMEOUT 10)
endif ()

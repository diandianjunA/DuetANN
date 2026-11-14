set(DVSTOR_TEST_SOURCE_DIR "${DVSTOR_SOURCE_ROOT}/test/benchmark")

dvstor_add_test(breakdown_aggregate_test)
target_link_libraries(breakdown_aggregate_test PRIVATE dvstor_runtime)

add_test(NAME experiment_config_contract_test
         COMMAND python3
                 ${CMAKE_SOURCE_DIR}/test/benchmark/experiment_config_contract_test.py)

dvstor_add_test(breakdown_benchmark_support_test)
dvstor_target_tool_includes(breakdown_benchmark_support_test)
target_link_libraries(breakdown_benchmark_support_test PRIVATE dvstor_breakdown_support)

dvstor_add_test(breakdown_benchmark_control_test
    ${DVSTOR_TEST_SOURCE_DIR}/breakdown_benchmark_control_test.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/args.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/progress.cc)
dvstor_target_tool_includes(breakdown_benchmark_control_test)
target_link_libraries(breakdown_benchmark_control_test PRIVATE dvstor_breakdown_support)

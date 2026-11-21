set(DUETANN_TEST_SOURCE_DIR "${DUETANN_SOURCE_ROOT}/test/benchmark")

duetann_add_test(breakdown_aggregate_test)
target_link_libraries(breakdown_aggregate_test PRIVATE duetann_runtime)

add_test(NAME experiment_config_contract_test
         COMMAND python3
                 ${CMAKE_SOURCE_DIR}/test/benchmark/experiment_config_contract_test.py)

duetann_add_test(breakdown_benchmark_support_test)
duetann_target_tool_includes(breakdown_benchmark_support_test)
target_link_libraries(breakdown_benchmark_support_test PRIVATE duetann_breakdown_support)

duetann_add_test(breakdown_benchmark_control_test
    ${DUETANN_TEST_SOURCE_DIR}/breakdown_benchmark_control_test.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/args.cc
    ${CMAKE_SOURCE_DIR}/tools/breakdown_benchmark/progress.cc)
duetann_target_tool_includes(breakdown_benchmark_control_test)
target_link_libraries(breakdown_benchmark_control_test PRIVATE duetann_breakdown_support)

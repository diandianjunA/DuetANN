include_guard(GLOBAL)

# Keep assertions enabled in Release builds: the CPU tests use assert().
function(duetann_add_test target_name)
    if (ARGN)
        add_executable(${target_name} ${ARGN})
    else ()
        add_executable(${target_name} ${DUETANN_TEST_SOURCE_DIR}/${target_name}.cc)
    endif ()
    target_compile_options(${target_name} PRIVATE -UNDEBUG)
    duetann_target_common_includes(${target_name})
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

function(duetann_add_cuda_test target_name timeout)
    add_executable(${target_name} ${DUETANN_TEST_SOURCE_DIR}/${target_name}.cu)
    duetann_target_common_includes(${target_name})
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        TIMEOUT ${timeout}
        RESOURCE_LOCK duetann_gpu
        SKIP_REGULAR_EXPRESSION "^SKIP:")
endfunction()

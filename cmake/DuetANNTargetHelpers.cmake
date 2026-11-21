include_guard(GLOBAL)

function(duetann_target_common_includes target_name)
    set(_visibility PRIVATE)
    if (ARGC GREATER 1)
        set(_visibility ${ARGV1})
    endif ()
    target_include_directories(${target_name} ${_visibility}
        ${DUETANN_SOURCE_ROOT}/src
        ${DUETANN_SOURCE_ROOT}/rdma-library
        ${DUETANN_SOURCE_ROOT}/thirdparty
    )
    unset(_visibility)
endfunction()

function(duetann_target_tool_includes target_name)
    target_include_directories(${target_name} PRIVATE ${DUETANN_SOURCE_ROOT})
    duetann_target_common_includes(${target_name})
endfunction()

function(target_link_metis target_name)
    if (DUETANN_HAVE_METIS)
        target_include_directories(${target_name} PRIVATE ${METIS_INCLUDE_DIR})
        target_link_libraries(${target_name} ${METIS_LIBRARY})
        if (GKLIB_LIBRARY)
            target_link_libraries(${target_name} ${GKLIB_LIBRARY})
        endif ()
        target_compile_definitions(${target_name} PRIVATE DUETANN_HAVE_METIS=1)
    else ()
        target_compile_definitions(${target_name} PRIVATE DUETANN_HAVE_METIS=0)
    endif ()
endfunction()

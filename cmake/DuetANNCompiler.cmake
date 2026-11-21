include_guard(GLOBAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR
        "DuetANN requires GCC 11 or newer for standard C++20 coroutines and atomic_ref. "
        "Select a newer compiler with -DCMAKE_CXX_COMPILER=/path/to/g++.")
endif ()

# nvcc may use a different host GCC from the explicitly configured C++ compiler.
# Put the configured compiler's own runtime first instead of hard-coding a host path.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
        OUTPUT_VARIABLE DUETANN_CXX_STDLIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE DUETANN_CXX_STDLIB_RESULT)
    if (DUETANN_CXX_STDLIB_RESULT EQUAL 0 AND EXISTS "${DUETANN_CXX_STDLIB}")
        get_filename_component(DUETANN_CXX_STDLIB_DIR
            "${DUETANN_CXX_STDLIB}" DIRECTORY)
        add_link_options("-L${DUETANN_CXX_STDLIB_DIR}")
    endif ()
endif ()

# Use RPATH rather than RUNPATH so custom METIS/GKlib paths retain priority.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

if (NOT DUETANN_STORAGE_NODE_ONLY)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wfatal-errors>)
if (DUETANN_USE_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif ()
# Keep finite/NaN checks in routing and update-state code well-defined.
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-trapping-math>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
add_compile_options(-DNOHUGEPAGES)

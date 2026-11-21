include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(duetann_check_metis_link output_variable)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${METIS_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${METIS_LIBRARY}")
    if (GKLIB_LIBRARY)
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${GKLIB_LIBRARY}")
    endif ()
    unset(DUETANN_METIS_LINK_COMPATIBLE CACHE)
    check_cxx_source_compiles(
        "#include <metis.h>
         int main() {
             idx_t options[METIS_NOPTIONS];
             return METIS_SetDefaultOptions(options) == METIS_OK ? 0 : 1;
         }"
        DUETANN_METIS_LINK_COMPATIBLE)
    set(${output_variable} "${DUETANN_METIS_LINK_COMPATIBLE}" PARENT_SCOPE)
    cmake_pop_check_state()
endfunction()

if (NOT DUETANN_STORAGE_NODE_ONLY)
    find_package(CUDAToolkit 11.8 REQUIRED)

    if (DUETANN_CONDA_LIB_DIR AND EXISTS "${DUETANN_CONDA_LIB_DIR}")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -L${DUETANN_CONDA_LIB_DIR} -Wl,-rpath,${DUETANN_CONDA_LIB_DIR}")
    endif ()

endif ()

if (DUETANN_BUILD_OFFLINE_TOOLS)
    set(_DUETANN_METIS_HINTS)
    if (DUETANN_METIS_ROOT)
        list(APPEND _DUETANN_METIS_HINTS "${DUETANN_METIS_ROOT}")
    endif ()
    if (EXISTS "${DUETANN_SOURCE_ROOT}/thirdparty/metis64")
        list(APPEND _DUETANN_METIS_HINTS "${DUETANN_SOURCE_ROOT}/thirdparty/metis64")
    endif ()

    if (NOT DUETANN_METIS_PARTITION_MODE STREQUAL "OFF")
        find_path(METIS_INCLUDE_DIR metis.h
            HINTS ${_DUETANN_METIS_HINTS}
            PATH_SUFFIXES include)
        find_library(METIS_LIBRARY NAMES metis
            HINTS ${_DUETANN_METIS_HINTS}
            PATH_SUFFIXES lib lib64)
        find_library(GKLIB_LIBRARY NAMES GKlib
            HINTS ${_DUETANN_METIS_HINTS}
            PATH_SUFFIXES lib lib64)

        if (METIS_INCLUDE_DIR AND METIS_LIBRARY)
            duetann_check_metis_link(_DUETANN_METIS_LINKS)
            string(FIND "${METIS_LIBRARY}"
                "${DUETANN_SOURCE_ROOT}/thirdparty/metis64/"
                _DUETANN_BUNDLED_METIS_INDEX)
            if (NOT _DUETANN_METIS_LINKS AND _DUETANN_BUNDLED_METIS_INDEX EQUAL 0)
                set(_DUETANN_REJECTED_METIS "${METIS_LIBRARY};${GKLIB_LIBRARY}")
                foreach (_DUETANN_METIS_VARIABLE
                    METIS_INCLUDE_DIR METIS_LIBRARY GKLIB_LIBRARY)
                    unset(${_DUETANN_METIS_VARIABLE})
                    unset(${_DUETANN_METIS_VARIABLE} CACHE)
                endforeach ()
                set(_DUETANN_SYSTEM_LIBRARY_SUFFIXES lib lib64)
                if (CMAKE_LIBRARY_ARCHITECTURE)
                    list(APPEND _DUETANN_SYSTEM_LIBRARY_SUFFIXES
                        "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif ()
                find_path(METIS_INCLUDE_DIR metis.h
                    PATHS /usr /usr/local
                    PATH_SUFFIXES include
                    NO_DEFAULT_PATH)
                find_library(METIS_LIBRARY NAMES metis
                    PATHS /usr /usr/local
                    PATH_SUFFIXES ${_DUETANN_SYSTEM_LIBRARY_SUFFIXES}
                    NO_DEFAULT_PATH)
                find_library(GKLIB_LIBRARY NAMES GKlib
                    PATHS /usr /usr/local
                    PATH_SUFFIXES ${_DUETANN_SYSTEM_LIBRARY_SUFFIXES}
                    NO_DEFAULT_PATH)
                if (METIS_INCLUDE_DIR AND METIS_LIBRARY)
                    duetann_check_metis_link(_DUETANN_METIS_LINKS)
                    if (_DUETANN_METIS_LINKS)
                        message(STATUS
                            "Bundled METIS is not link-compatible; using system METIS instead")
                    endif ()
                endif ()
                unset(_DUETANN_SYSTEM_LIBRARY_SUFFIXES)
                unset(_DUETANN_METIS_VARIABLE)
            endif ()
            if (_DUETANN_METIS_LINKS)
                set(DUETANN_HAVE_METIS ON)
                if (GKLIB_LIBRARY)
                    message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY}, GKlib: ${GKLIB_LIBRARY})")
                else ()
                    message(STATUS "METIS partitioning: enabled (${METIS_LIBRARY})")
                endif ()
            elseif (DUETANN_METIS_PARTITION_MODE STREQUAL "ON")
                message(FATAL_ERROR
                    "METIS partitioning was requested, but the discovered libraries cannot be linked: "
                    "${_DUETANN_REJECTED_METIS};${METIS_LIBRARY};${GKLIB_LIBRARY}. "
                    "Rebuild/install METIS for this host, set "
                    "DUETANN_METIS_ROOT to a compatible CPU installation, or use "
                    "-DDUETANN_METIS_PARTITION=OFF.")
            else ()
                set(DUETANN_HAVE_METIS OFF)
                message(WARNING
                    "METIS partitioning: disabled because the discovered libraries cannot be linked. "
                    "This commonly means the bundled METIS/GKlib was built against a newer glibc. "
                    "The balanced and bfs partition strategies remain available.")
            endif ()
            unset(_DUETANN_BUNDLED_METIS_INDEX)
            unset(_DUETANN_REJECTED_METIS)
            unset(_DUETANN_METIS_LINKS)
        elseif (DUETANN_METIS_PARTITION_MODE STREQUAL "ON")
            message(FATAL_ERROR
                "METIS partitioning requested but metis.h/libmetis were not found. "
                "Install libmetis-dev, set DUETANN_METIS_ROOT, or set -DDUETANN_METIS_PARTITION=OFF.")
        else ()
            set(DUETANN_HAVE_METIS OFF)
            message(STATUS "METIS partitioning: disabled (metis.h/libmetis not found)")
        endif ()
    else ()
        set(DUETANN_HAVE_METIS OFF)
        message(STATUS "METIS partitioning: disabled")
    endif ()

    unset(_DUETANN_METIS_HINTS)
else ()
    set(DUETANN_HAVE_METIS OFF)
endif ()

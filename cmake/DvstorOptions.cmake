include_guard(GLOBAL)

option(DVSTOR_BUILD_EXECUTABLES "Build standalone DVSTOR executables and tools" ON)
option(DVSTOR_BUILD_TESTS "Build local DVSTOR smoke tests" ON)
option(DVSTOR_BUILD_OFFLINE_TOOLS
    "Build CPU-only index construction, partitioning, and PQ tools" ON)
option(DVSTOR_USE_NATIVE_ARCH "Compile C++ sources with -march=native" ON)
option(DVSTOR_PERSISTENT_WIDE_SORT "Build the wide persistent Beam sort for ablation" OFF)
option(DVSTOR_PERSISTENT_LEGACY_MERGE "Build the legacy persistent Beam merge for ablation" OFF)

set(DVSTOR_PERSISTENT_MAX_BEAM "256" CACHE STRING
    "Maximum traversal Beam compiled into the persistent kernel (128 or 256)")
set_property(CACHE DVSTOR_PERSISTENT_MAX_BEAM PROPERTY STRINGS 128 256)
if (NOT DVSTOR_PERSISTENT_MAX_BEAM MATCHES "^(128|256)$")
    message(FATAL_ERROR "DVSTOR_PERSISTENT_MAX_BEAM must be 128 or 256")
endif ()

set(DVSTOR_PERSISTENT_MAX_EXACT "512" CACHE STRING
    "Maximum final-rerank capacity compiled into the persistent kernel (256 or 512)")
set_property(CACHE DVSTOR_PERSISTENT_MAX_EXACT PROPERTY STRINGS 256 512)
if (NOT DVSTOR_PERSISTENT_MAX_EXACT MATCHES "^(256|512)$")
    message(FATAL_ERROR "DVSTOR_PERSISTENT_MAX_EXACT must be 256 or 512")
endif ()
math(EXPR DVSTOR_PERSISTENT_MIN_EXACT "${DVSTOR_PERSISTENT_MAX_BEAM} * 2")
if (DVSTOR_PERSISTENT_MAX_EXACT LESS DVSTOR_PERSISTENT_MIN_EXACT)
    message(FATAL_ERROR "DVSTOR_PERSISTENT_MAX_EXACT must be at least twice DVSTOR_PERSISTENT_MAX_BEAM")
endif ()

set(DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES "8" CACHE STRING
    "Number of graph edges in each Live/DynaExtent length class")
if (NOT DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES MATCHES "^[1-9][0-9]*$" OR
    DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES GREATER 127)
    message(FATAL_ERROR "DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES must be in [1, 127]")
endif ()
add_compile_definitions(
    DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES=${DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES})

if (DVSTOR_STORAGE_NODE_ONLY)
    set(Boost_NO_BOOST_CMAKE ON CACHE BOOL
        "Use CMake's FindBoost module on CPU-only storage hosts to avoid Conda RPATH injection")
else ()
    set(DVSTOR_DOCA_ROOT "/opt/mellanox/doca" CACHE PATH "DOCA SDK installation root")
endif ()

set(DVSTOR_METIS_PARTITION "AUTO" CACHE STRING
    "METIS graph partitioning support: AUTO, ON, or OFF")
set_property(CACHE DVSTOR_METIS_PARTITION PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${DVSTOR_METIS_PARTITION}" DVSTOR_METIS_PARTITION_MODE)
if (NOT DVSTOR_METIS_PARTITION_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "DVSTOR_METIS_PARTITION must be AUTO, ON, or OFF")
endif ()

set(DVSTOR_CONDA_LIB_DIR "" CACHE PATH
    "Optional libstdc++ search path for hosts whose system libstdc++ is too old for CUDA")
set(DVSTOR_METIS_ROOT "" CACHE PATH "Optional METIS installation root")
set(DVSTOR_FAISS_BLAS_VENDOR "Generic" CACHE STRING
    "BLAS vendor for CPU Faiss offline tools; Generic avoids nested MKL/OpenMP runtimes")

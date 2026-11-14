include_guard(GLOBAL)

find_path(DVSTOR_DOCA_INCLUDE_DIR doca_gpunetio.h
    HINTS "${DVSTOR_DOCA_ROOT}/include")
find_library(DVSTOR_DOCA_GPUNETIO_LIBRARY doca_gpunetio
    HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
find_library(DVSTOR_DOCA_VERBS_LIBRARY doca_verbs
    HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
find_library(DVSTOR_DOCA_COMMON_LIBRARY doca_common
    HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
find_library(DVSTOR_DOCA_RDMA_LIBRARY doca_rdma
    HINTS "${DVSTOR_DOCA_ROOT}/lib/x86_64-linux-gnu")
find_library(DVSTOR_MLX5_LIBRARY mlx5)

if (NOT DVSTOR_DOCA_INCLUDE_DIR OR
    NOT DVSTOR_DOCA_GPUNETIO_LIBRARY OR
    NOT DVSTOR_DOCA_VERBS_LIBRARY OR
    NOT DVSTOR_DOCA_COMMON_LIBRARY OR
    NOT DVSTOR_DOCA_RDMA_LIBRARY OR
    NOT DVSTOR_MLX5_LIBRARY)
    message(WARNING
        "DOCA GPUNetIO installation not found; building the compute-node "
        "runtime WITHOUT GPUNetIO. Only the coupled (host-orchestrated) "
        "query mode will be available.")
    set(DVSTOR_HAVE_GPUNETIO OFF)
else ()
    set(DVSTOR_HAVE_GPUNETIO ON)
endif ()

if (NOT DVSTOR_HAVE_GPUNETIO)
    return()
endif ()

# Persistent GPUNetIO requires Volta's independent-thread synchronization and
# system-scope operations. Reject unsupported architectures during configure.
if (NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "DVSTOR persistent GPUNetIO requires an explicit SM70+ CUDA "
        "architecture; CMAKE_CUDA_ARCHITECTURES is empty or OFF")
endif ()
foreach (DVSTOR_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (NOT DVSTOR_CUDA_ARCH MATCHES "^([0-9]+)(-real|-virtual)?$")
        message(FATAL_ERROR
            "DVSTOR persistent GPUNetIO requires an explicit numeric "
            "SM70+ CUDA architecture; unsupported entry: ${DVSTOR_CUDA_ARCH}")
    endif ()
    if (CMAKE_MATCH_1 LESS 70)
        message(FATAL_ERROR
            "DVSTOR persistent GPUNetIO requires SM70+; unsupported "
            "CMAKE_CUDA_ARCHITECTURES entry: ${DVSTOR_CUDA_ARCH}")
    endif ()
endforeach ()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <infiniband/verbs.h>
int main() { return static_cast<int>(IBV_WC_DRIVER1); }"
    DVSTOR_HAVE_IBV_WC_DRIVER1)
if (DVSTOR_HAVE_IBV_WC_DRIVER1)
    add_compile_definitions(DVSTOR_HAVE_IBV_WC_DRIVER1=1)
endif ()

set(CMAKE_REQUIRED_LIBRARIES ${DVSTOR_MLX5_LIBRARY})
check_cxx_source_compiles("
#include <infiniband/verbs.h>
#if !defined(IBV_WC_DRIVER1) && !defined(DVSTOR_HAVE_IBV_WC_DRIVER1)
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif
#include <infiniband/mlx5dv.h>
int main() { return mlx5dv_reg_dmabuf_mr(nullptr, 0, 0, 0, 0, 0, 0) ==
                     nullptr; }"
    DVSTOR_HAVE_MLX5DV_REG_DMABUF_MR)
if (DVSTOR_HAVE_MLX5DV_REG_DMABUF_MR)
    add_compile_definitions(DVSTOR_HAVE_MLX5DV_REG_DMABUF_MR=1)
endif ()

set(CMAKE_REQUIRED_LIBRARIES ${IBVerbs_LIBRARY})
check_cxx_source_compiles("
#include <infiniband/verbs.h>
int main() { return ibv_reg_dmabuf_mr(nullptr, 0, 0, 0, 0, 0, 0) ==
                     nullptr; }"
    DVSTOR_HAVE_IBV_REG_DMABUF_MR)
unset(CMAKE_REQUIRED_LIBRARIES)
if (DVSTOR_HAVE_IBV_REG_DMABUF_MR)
    add_compile_definitions(DVSTOR_HAVE_IBV_REG_DMABUF_MR=1)
endif ()

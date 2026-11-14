#pragma once

#include <cuda_runtime.h>

#include "gpu_search/types.hh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ u32 warp_reduce_add_sync(u32 mask, u32 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(mask, value);
#else
  if (mask == 0xffffffffu) {
#pragma unroll
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      value += __shfl_xor_sync(mask, value, offset);
    }
    return value;
  }
  u32 reduced = 0;
  for (u32 remaining = mask; remaining != 0; remaining &= remaining - 1u) {
    const u32 source = static_cast<u32>(__ffs(remaining) - 1);
    reduced += __shfl_sync(mask, value, source);
  }
  return reduced;
#endif
}

__device__ __forceinline__ u64 warp_reduce_add_sync(u32 mask, u64 value) {
#pragma unroll
  for (u32 offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_xor_sync(mask, value, offset);
  }
  return value;
}

__device__ __forceinline__ u32 warp_reduce_min_sync(u32 mask, u32 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_min_sync(mask, value);
#else
  if (mask == 0xffffffffu) {
#pragma unroll
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      value = min(value, __shfl_xor_sync(mask, value, offset));
    }
    return value;
  }
  u32 reduced = ~u32{0};
  for (u32 remaining = mask; remaining != 0; remaining &= remaining - 1u) {
    const u32 source = static_cast<u32>(__ffs(remaining) - 1);
    reduced = min(reduced, __shfl_sync(mask, value, source));
  }
  return reduced;
#endif
}

}  // namespace gpu_search::persistent_kernel_detail

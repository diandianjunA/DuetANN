#pragma once

#include "gpu_search/persistent_kernel/merge/candidate_scoring_base.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __noinline__ f32 opq_rotation_wide_fallback_row(
    const f32* __restrict__ opq_matrix, const f32* __restrict__ query,
    u32 row, u32 dim) {
  const f32* matrix_row = opq_matrix + static_cast<size_t>(row) * dim;
  double wide_value = 0.0;
  for (u32 column = 0; column < dim; ++column) {
    wide_value = fma(static_cast<double>(matrix_row[column]),
                     static_cast<double>(query[column]), wide_value);
  }
  return saturate_device_component(wide_value);
}

// OPQ rotation: dim x dim matvec per query, warp-cooperative over output
// rows.  In the row-major device layout lanes reading consecutive columns of
// one row are fully coalesced, and interleaving four rows per warp keeps four
// independent load streams in flight to hide L2 latency on a 128-thread
// query CTA.  The previous thread-per-row scan had each lane strided by dim
// floats, fanning every warp load out into 32 separate sectors and leaving a
// serial dim-step MAC chain per thread.
__device__ __forceinline__ void apply_opq_rotation(
    const f32* __restrict__ opq_matrix, const f32* __restrict__ query,
    f32* __restrict__ transformed, u32 dim) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  const u32 warp = threadIdx.x >> 5;
  const u32 warp_count = blockDim.x >> 5;
  const bool vectorized =
    (dim & 3u) == 0 &&
    (reinterpret_cast<size_t>(opq_matrix) & 15u) == 0 &&
    (reinterpret_cast<size_t>(query) & 15u) == 0;
  if (vectorized) {
    const u32 vec_count = dim >> 2;
    const float4* matrix4 = reinterpret_cast<const float4*>(opq_matrix);
    const float4* query4 = reinterpret_cast<const float4*>(query);
    u32 row = warp;
    for (; row + 3u * warp_count < dim; row += 4u * warp_count) {
      const float4* row0 = matrix4 + static_cast<size_t>(row) * vec_count;
      const float4* row1 =
        row0 + static_cast<size_t>(warp_count) * vec_count;
      const float4* row2 =
        row1 + static_cast<size_t>(warp_count) * vec_count;
      const float4* row3 =
        row2 + static_cast<size_t>(warp_count) * vec_count;
      f32 value0 = 0.0f;
      f32 value1 = 0.0f;
      f32 value2 = 0.0f;
      f32 value3 = 0.0f;
      for (u32 vec = lane; vec < vec_count; vec += 32u) {
        const float4 query_vec = query4[vec];
        const float4 matrix0 = __ldg(row0 + vec);
        const float4 matrix1 = __ldg(row1 + vec);
        const float4 matrix2 = __ldg(row2 + vec);
        const float4 matrix3 = __ldg(row3 + vec);
        value0 = fmaf(matrix0.x, query_vec.x, value0);
        value0 = fmaf(matrix0.y, query_vec.y, value0);
        value0 = fmaf(matrix0.z, query_vec.z, value0);
        value0 = fmaf(matrix0.w, query_vec.w, value0);
        value1 = fmaf(matrix1.x, query_vec.x, value1);
        value1 = fmaf(matrix1.y, query_vec.y, value1);
        value1 = fmaf(matrix1.z, query_vec.z, value1);
        value1 = fmaf(matrix1.w, query_vec.w, value1);
        value2 = fmaf(matrix2.x, query_vec.x, value2);
        value2 = fmaf(matrix2.y, query_vec.y, value2);
        value2 = fmaf(matrix2.z, query_vec.z, value2);
        value2 = fmaf(matrix2.w, query_vec.w, value2);
        value3 = fmaf(matrix3.x, query_vec.x, value3);
        value3 = fmaf(matrix3.y, query_vec.y, value3);
        value3 = fmaf(matrix3.z, query_vec.z, value3);
        value3 = fmaf(matrix3.w, query_vec.w, value3);
      }
      for (u32 offset = 16; offset != 0; offset >>= 1) {
        value0 += __shfl_down_sync(full_warp, value0, offset);
        value1 += __shfl_down_sync(full_warp, value1, offset);
        value2 += __shfl_down_sync(full_warp, value2, offset);
        value3 += __shfl_down_sync(full_warp, value3, offset);
      }
      if (lane == 0) {
        if (!finite_f32_bits(value0)) {
          value0 = opq_rotation_wide_fallback_row(
            opq_matrix, query, row, dim);
        }
        transformed[row] = value0;
        if (!finite_f32_bits(value1)) {
          value1 = opq_rotation_wide_fallback_row(
            opq_matrix, query, row + warp_count, dim);
        }
        transformed[row + warp_count] = value1;
        if (!finite_f32_bits(value2)) {
          value2 = opq_rotation_wide_fallback_row(
            opq_matrix, query, row + 2u * warp_count, dim);
        }
        transformed[row + 2u * warp_count] = value2;
        if (!finite_f32_bits(value3)) {
          value3 = opq_rotation_wide_fallback_row(
            opq_matrix, query, row + 3u * warp_count, dim);
        }
        transformed[row + 3u * warp_count] = value3;
      }
    }
    for (; row < dim; row += warp_count) {
      const float4* matrix_row4 =
        matrix4 + static_cast<size_t>(row) * vec_count;
      f32 value = 0.0f;
      for (u32 vec = lane; vec < vec_count; vec += 32u) {
        const float4 query_vec = query4[vec];
        const float4 matrix_vec = __ldg(matrix_row4 + vec);
        value = fmaf(matrix_vec.x, query_vec.x, value);
        value = fmaf(matrix_vec.y, query_vec.y, value);
        value = fmaf(matrix_vec.z, query_vec.z, value);
        value = fmaf(matrix_vec.w, query_vec.w, value);
      }
      for (u32 offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(full_warp, value, offset);
      }
      if (lane == 0) {
        if (!finite_f32_bits(value)) {
          value = opq_rotation_wide_fallback_row(opq_matrix, query, row, dim);
        }
        transformed[row] = value;
      }
    }
    return;
  }
  for (u32 row = warp; row < dim; row += warp_count) {
    const f32* matrix_row = opq_matrix + static_cast<size_t>(row) * dim;
    f32 value = 0.0f;
    for (u32 column = lane; column < dim; column += 32u) {
      value = fmaf(__ldg(matrix_row + column), query[column], value);
    }
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      value += __shfl_down_sync(full_warp, value, offset);
    }
    if (lane == 0) {
      if (!finite_f32_bits(value)) {
        value = opq_rotation_wide_fallback_row(opq_matrix, query, row, dim);
      }
      transformed[row] = value;
    }
  }
}

}  // namespace gpu_search::persistent_kernel_detail

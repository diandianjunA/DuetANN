#pragma once

#include "gpu_search/persistent_kernel/query/opq_rotation.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ bool query_descriptor_valid(
    const PersistentKernelParams& params, const QueryDescriptor& descriptor,
    u32 query_slot) {
  return query_slot < params.query_slots && descriptor.dim == params.dim &&
    descriptor.query_dtype <= 2 && descriptor.query_device_address != 0 &&
    descriptor.result_device_address != 0 &&
    (descriptor.query_dtype != 0 ||
     descriptor.query_device_address % alignof(f32) == 0) &&
    descriptor.result_device_address % alignof(u32) == 0 &&
    params.decoded_queries != nullptr && params.result_distances != nullptr &&
    params.navigation_candidate_handles != nullptr &&
    params.navigation_candidate_distances != nullptr &&
    descriptor.k != 0 && descriptor.k <= descriptor.result_capacity;
}

template <u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__device__ __forceinline__ void prepare_query_vectors(
    const PersistentKernelParams& params, const QueryDescriptor& descriptor,
    u32 query_slot, f32* query, f32* query_lut) {
  const u8* query_input =
    reinterpret_cast<const u8*>(descriptor.query_device_address);
  for (u32 dimension = threadIdx.x; dimension < params.dim;
       dimension += blockDim.x) {
    query[dimension] = query_component(
      query_input, descriptor.query_dtype, dimension);
  }
  __syncthreads();

  f32* transformed = params.transformed_queries +
    static_cast<size_t>(query_slot) * params.dim;
  if (params.opq_matrix == nullptr) {
    for (u32 row = threadIdx.x; row < params.dim; row += blockDim.x) {
      transformed[row] = query[row];
    }
  } else {
    apply_opq_rotation(params.opq_matrix, query, transformed, params.dim);
  }
  __syncthreads();

  const u32 pq_subquantizers =
    resolved_pq_subquantizers<PqSubquantizers>(params);
  const u32 table_entries = pq_subquantizers * 256;
  for (u32 index = threadIdx.x; index < table_entries;
       index += blockDim.x) {
    const u32 subquantizer = index / 256;
    const f32* query_subvector = transformed +
      static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
    const f32* centroid_subvector = params.pq_centroids +
      static_cast<size_t>(index) * params.pq_subvector_dim;
    f32 distance = 0.0f;
    for (u32 dimension = 0; dimension < params.pq_subvector_dim;
         ++dimension) {
      const f32 difference =
        query_subvector[dimension] - centroid_subvector[dimension];
      distance += difference * difference;
    }
    if (!finite_f32_bits(distance) || distance == FLT_MAX) {
      double wide_distance = 0.0;
      for (u32 dimension = 0; dimension < params.pq_subvector_dim;
           ++dimension) {
        const double difference =
          static_cast<double>(query_subvector[dimension]) -
          static_cast<double>(centroid_subvector[dimension]);
        wide_distance += difference * difference;
      }
      distance = saturate_device_squared_l2(wide_distance);
    }
    query_lut[index] = distance;
  }
  __syncthreads();
}

}  // namespace gpu_search::persistent_kernel_detail

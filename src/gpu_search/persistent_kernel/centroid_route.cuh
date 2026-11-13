#pragma once

#include "gpu_search/persistent_kernel/query_completion.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ u64 decode_tagged_raw(const u8* source) {
  return *reinterpret_cast<const u64*>(source);
}

struct CentroidRouteShardSnapshot {
  u64 version{};
  u64 vector_count{};
  f32 distance{FLT_MAX};
  u32 live_entry_count{};
  u64 remote_nodes[kCentroidRouteMaxLiveEntries]{};
};

template <typename T>
__device__ T centroid_route_atomic_load(const T& value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(
    const_cast<T&>(value));
  return reference.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ void centroid_route_atomic_store(T& destination, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(destination);
  reference.store(value, cuda::memory_order_relaxed);
}

__device__ bool snapshot_centroid_route_shard(
    const PersistentKernelParams& params, u32 shard,
    const f32* query, CentroidRouteShardSnapshot& result) {
  if (params.centroid_route_shards == nullptr ||
      params.centroid_route_entries == nullptr ||
      params.shard_centroids == nullptr ||
      params.centroid_route_entry_capacity == 0 ||
      params.centroid_route_entry_capacity > kCentroidRouteMaxLiveEntries ||
      shard >= params.num_shards ||
      shard >= params.centroid_route_shard_capacity) {
    return false;
  }
  const DeviceCentroidRouteShard& source =
    params.centroid_route_shards[shard];
  cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
    const_cast<u64&>(source.sequence));
  // A writer window is very short.  Two attempts recover from the common
  // boundary race without ever making a query wait on route publication.
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    const u64 before = sequence.load(cuda::memory_order_acquire);
    if ((before & 1u) != 0) continue;
    CentroidRouteShardSnapshot candidate{
      .version = centroid_route_atomic_load(source.version),
      .vector_count = centroid_route_atomic_load(source.vector_count),
      .live_entry_count =
        centroid_route_atomic_load(source.live_entry_count),
    };
    if (candidate.version == 0 ||
        candidate.vector_count == 0 || candidate.live_entry_count == 0 ||
        candidate.live_entry_count > params.centroid_route_entry_capacity) {
      return false;
    }
    f32 distance = 0.0f;
    const f32* centroid = params.shard_centroids +
      static_cast<size_t>(shard) * params.dim;
    for (u32 dimension = 0; dimension < params.dim; ++dimension) {
      const f32 difference = query[dimension] - centroid[dimension];
      // Match CPU update-home routing exactly: canonical FP32 centroids,
      // left-to-right FP32 fused accumulation, then physical-shard tie break.
      distance = fmaf(difference, difference, distance);
    }
    if (!finite_f32_bits(distance) || distance == FLT_MAX) {
      double wide_distance = 0.0;
      for (u32 dimension = 0; dimension < params.dim; ++dimension) {
        const double difference = static_cast<double>(query[dimension]) -
          static_cast<double>(centroid[dimension]);
        wide_distance = fma(difference, difference, wide_distance);
      }
      distance = saturate_device_squared_l2(wide_distance);
    }
    candidate.distance = distance;
    const DeviceCentroidRouteEntry* entries =
      params.centroid_route_entries +
      static_cast<size_t>(shard) * params.centroid_route_entry_capacity;
    for (u32 index = 0; index < candidate.live_entry_count; ++index) {
      const u64 remote_node = centroid_route_atomic_load(
        entries[index].remote_node);
      const u32 flags = centroid_route_atomic_load(entries[index].flags);
      if (remote_node == 0 || flags != kCentroidRouteLive ||
          remote_shard(remote_node) != shard) {
        return false;
      }
      candidate.remote_nodes[index] = remote_node;
    }
    const u64 after = sequence.load(cuda::memory_order_acquire);
    if (before != after || (after & 1u) != 0) continue;
    result = candidate;
    return true;
  }
  return false;
}

// All 64 RemotePtr-addressable shards participate in the same fixed sorting
// network. Invalid/unpublished shards sort after every valid shard, including
// a valid distance saturated near FLT_MAX. Every compare-exchange is owned by
// exactly one CUDA thread, and the barrier preserves the bitonic stage order.
__device__ void sort_centroid_route_shards(
    centroid_route_ranking::RankedShard* routes) {
  static_assert(kPersistentMaxShards == 64);
  for (u32 sequence = 2; sequence <= kPersistentMaxShards; sequence <<= 1) {
    for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (u32 index = threadIdx.x; index < kPersistentMaxShards;
           index += blockDim.x) {
        const u32 partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        if (!centroid_route_ranking::should_exchange(
              routes[index], routes[partner], ascending)) {
          continue;
        }
        const centroid_route_ranking::RankedShard temporary = routes[index];
        routes[index] = routes[partner];
        routes[partner] = temporary;
      }
      __syncthreads();
    }
  }
}


// Reconcile one completed issue wave with the next exact Stable-Run frontier.
// The preview reads the already sorted runs, so the core prefix is
// authoritative for the Beam that finish_approximate_stable_runs will publish.
// Only validated payloads are retained; COMMITTED payloads have already been
// decoded, and every unretained shadow payload is charged as waste exactly
// once. One warp performs matching, reclamation and allocation with ballot
// compaction, so no global queue or per-entry atomics are needed.
}  // namespace gpu_search::persistent_kernel_detail

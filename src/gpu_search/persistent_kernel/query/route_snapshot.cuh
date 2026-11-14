#pragma once

#include "gpu_search/persistent_kernel/query/centroid_route.cuh"

namespace gpu_search::persistent_kernel_detail {

struct QueryRouteSnapshotState {
  centroid_route_ranking::RankedShard ranked_routes[kPersistentMaxShards];
  u32 route_snapshot_entry_counts[kPersistentMaxShards];
  u64 route_snapshot_remote_nodes[
    kPersistentMaxShards * kCentroidRouteMaxLiveEntries];
  u32 route_entry_count;
  u32 ranked_route_count;
  u32 route_seed_failed;
  u32 route_snapshot_conflicted;
  u32 route_snapshot_timed_out;
  u32 route_snapshot_cancelled;
  u32 route_snapshot_retries;
  u32 route_snapshot_backoff_ns;
  u32 route_failure_reason;
  u64 route_epoch_before;
  u64 route_wait_started_ns;
};

enum class QueryRouteSnapshotResult : u8 {
  ready,
  cancelled,
  timed_out,
};

__device__ __forceinline__ QueryRouteSnapshotResult snapshot_query_route(
    const PersistentKernelParams& params, const QueryDescriptor& descriptor,
    const f32* query, u64* navigation_handles, u32 route_attempt,
    QueryRouteSnapshotState& state) {
  if (threadIdx.x == 0) {
    state.route_wait_started_ns = global_time_ns();
    state.route_snapshot_backoff_ns = 128u +
      ((descriptor.query_slot * 97u + route_attempt * 53u) & 255u);
  }
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) {
      state.route_entry_count = 0;
      state.ranked_route_count = 0;
      state.route_seed_failed = 0;
      state.route_snapshot_conflicted = 0;
      state.route_snapshot_timed_out = 0;
      state.route_snapshot_cancelled = 0;
      state.route_failure_reason = static_cast<u32>(
        QueryFailureReason::route_no_seed);
      if (params.centroid_route_epoch == nullptr) {
        state.route_epoch_before = 1;
        state.route_seed_failed = 1;
      } else {
        cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
          *params.centroid_route_epoch);
        state.route_epoch_before = route_epoch.load(cuda::memory_order_acquire);
        if ((state.route_epoch_before & 1u) != 0) {
          state.route_snapshot_conflicted = 1;
        }
      }
    }
    __syncthreads();

    if (state.route_snapshot_conflicted == 0) {
      // Parallelize across physical shards while preserving the exact scalar
      // fmaf recurrence within a shard. A dimension-wise tree reduction
      // would change FP32 rounding and could send inserts and queries to
      // different homes.
      for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
           shard += blockDim.x) {
        state.route_snapshot_entry_counts[shard] = 0;
        state.ranked_routes[shard] = centroid_route_ranking::RankedShard{
          .distance = FLT_MAX,
          .shard = shard,
          .valid = 0,
        };
        if (shard >= params.num_shards) continue;
        CentroidRouteShardSnapshot snapshot;
        if (!snapshot_centroid_route_shard(params, shard, query, snapshot)) {
          continue;
        }
        state.route_snapshot_entry_counts[shard] =
          snapshot.live_entry_count;
        for (u32 entry = 0; entry < snapshot.live_entry_count; ++entry) {
          state.route_snapshot_remote_nodes[
            static_cast<size_t>(shard) * kCentroidRouteMaxLiveEntries + entry] =
            snapshot.remote_nodes[entry];
        }
        state.ranked_routes[shard] = centroid_route_ranking::RankedShard{
          .distance = snapshot.distance,
          .shard = shard,
          .valid = 1,
        };
      }
      __syncthreads();
      sort_centroid_route_shards(state.ranked_routes);
      if (threadIdx.x == 0) {
        u64 route_epoch_after = 1;
        if (params.centroid_route_epoch != nullptr) {
          cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
            *params.centroid_route_epoch);
          route_epoch_after = route_epoch.load(cuda::memory_order_acquire);
        }
        if (params.centroid_route_epoch == nullptr ||
            !centroid_route_ranking::stable_publication_epoch(
              state.route_epoch_before, route_epoch_after)) {
          state.route_snapshot_conflicted = 1;
          state.route_entry_count = 0;
        } else {
          while (state.ranked_route_count < kPersistentMaxShards &&
                 state.ranked_routes[state.ranked_route_count].valid != 0) {
            ++state.ranked_route_count;
          }
        }
        if (state.route_snapshot_conflicted == 0 &&
            state.route_seed_failed == 0 && state.ranked_route_count != 0) {
          const u32 route_count = min(
            state.ranked_route_count, params.centroid_route_fanout);
          for (u32 route = 0; route < route_count; ++route) {
            const u32 shard = state.ranked_routes[route].shard;
            for (u32 local = 0;
                 local < state.route_snapshot_entry_counts[shard];
                 ++local) {
              const u64 remote_node = state.route_snapshot_remote_nodes[
                static_cast<size_t>(shard) * kCentroidRouteMaxLiveEntries + local];
              const u64 handle = handle_from_raw(params, remote_node);
              if (handle != kInvalidDeviceHandle) {
                navigation_handles[state.route_entry_count++] = handle;
              }
            }
          }
        }
      }
      __syncthreads();
    }

    if (state.route_snapshot_conflicted == 0) break;
    if (threadIdx.x == 0) {
      ++state.route_snapshot_retries;
      if (params.stop != nullptr &&
          *reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        state.route_snapshot_cancelled = 1;
      } else {
        const u64 timeout_ns = params.route_snapshot_timeout_ns == 0
          ? u64{100'000'000} : params.route_snapshot_timeout_ns;
        if (global_time_ns() - state.route_wait_started_ns >= timeout_ns) {
          state.route_snapshot_timed_out = 1;
        } else {
          device_ring_relax(state.route_snapshot_backoff_ns);
          state.route_snapshot_backoff_ns = min(
            state.route_snapshot_backoff_ns * 2u, 8192u);
        }
      }
    }
    __syncthreads();
    if (state.route_snapshot_cancelled != 0) {
      return QueryRouteSnapshotResult::cancelled;
    }
    if (state.route_snapshot_timed_out != 0) {
      return QueryRouteSnapshotResult::timed_out;
    }
  }
  __syncthreads();
  return QueryRouteSnapshotResult::ready;
}

}  // namespace gpu_search::persistent_kernel_detail

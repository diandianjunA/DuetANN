#pragma once

#include "gpu_search/persistent_kernel/io/graph_extent.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ bool issue_narrow_frontier_graph_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_begin,
  u32 slot_count, i32* batch_statuses, u64* batch_completion_timestamps_ns,
  DirectBatchPriority priority, u32* remote_batches, u32* total_remote_reads,
  u64* graph_read_bytes, u32* graph_live_extent_reads,
  u32* graph_full_record_reads, u32* classified_graph_reads,
  u64* classified_graph_bytes, u32* queue_rejects,
  u32* prefetch_reads = nullptr, u64* prefetch_bytes = nullptr,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  const size_t request_base = static_cast<size_t>(descriptor.query_slot) *
                                kPersistentFrontierRobCapacity +
                              slot_begin;
  u32* request_shards = params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes =
    params.speculative_graph_request_bytes == nullptr
      ? nullptr
      : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr
      : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* completion_statuses =
    batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (threadIdx.x < 32) {
    // One warp exclusively owns this query-local issue wave.  Accumulate all
    // per-shard accounting in registers and publish it once from lane zero;
    // atomically updating the same shared counters for every populated shard
    // only serializes the enqueue path and provides no inter-CTA protection.
    u32 admitted_batches = 0;
    u32 admitted_reads = 0;
    u32 admitted_short_reads = 0;
    u32 admitted_full_reads = 0;
    u32 admitted_dynamic_short_reads = 0;
    u32 admitted_dynamic_full_reads = 0;
    u32 rejected_reads = 0;
    u32 fatal_batches = 0;
    u64 admitted_bytes = 0;
    u64 admitted_dynamic_bytes = 0;
    if (lane == 0) {
      batch.active = 0;
      batch.fatal = 0;
      batch.rejected = 0;
      batch.finish_had_pending = 0;
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      batch.issue_timestamp_ns[shard] = 0;
    }
    if (lane < slot_count) {
      request_shards[lane] = UINT32_MAX;
      request_offsets[lane] = 0;
      request_local_iovas[lane] = 0;
      if (request_bytes != nullptr) {
        request_bytes[lane] = params.graph_entry_bytes;
      }
      if (request_handles != nullptr) {
        request_handles[lane] = kInvalidDeviceHandle;
      }
      if (validation_states != nullptr) {
        validation_states[lane] =
          static_cast<u8>(FrontierValidationState::unknown);
      }
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::issued)) {
        u32 acquired_slot = UINT32_MAX;
        u32 shard = UINT32_MAX;
        u64 offset = 0;
        u64 local_iova = 0;
        u32 transfer_bytes = params.graph_entry_bytes;
        if (prepare_graph_record_in_scratch(
              params, entry.node_handle, descriptor.query_slot,
              slot_begin + lane, acquired_slot, shard, offset, local_iova,
              transfer_bytes)) {
          entry.scratch_slot = static_cast<u8>(slot_begin + lane);
          entry.transfer_bytes = transfer_bytes;
          request_shards[lane] = shard;
          request_offsets[lane] = offset;
          request_local_iovas[lane] = local_iova;
          if (request_bytes != nullptr) {
            request_bytes[lane] = transfer_bytes;
          }
          if (request_handles != nullptr) {
            request_handles[lane] = entry.node_handle;
          }
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
        }
      }
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      completion_statuses[shard] = 0;
      completion_timestamps[shard] = 0;
    }
    __syncwarp(full_warp);

    // One lane owns one request.  Form same-shard groups directly in the
    // warp instead of assigning one lane per physical shard and rescanning
    // the whole ROB prefix for every shard.  Only each group leader publishes
    // an owner descriptor; all accounting remains a single warp reduction.
    const u32 shard = lane < slot_count ? request_shards[lane] : UINT32_MAX;
    const bool valid_request = lane < slot_count && shard != UINT32_MAX;
    const u32 lane_bytes =
      valid_request
        ? (live_extent_enabled ? request_bytes[lane] : params.graph_entry_bytes)
        : 0u;
    const u32 valid_mask = __ballot_sync(full_warp, valid_request);
    const u32 short_mask = __ballot_sync(
      full_warp, valid_request && lane_bytes < params.graph_entry_bytes);
    const bool dynamic_request =
      valid_request &&
      dynamic_graph_telemetry_handle(rob[slot_begin + lane].node_handle);
    const u32 dynamic_mask = __ballot_sync(full_warp, dynamic_request);
    if (valid_request) {
      const u32 shard_group = __match_any_sync(valid_mask, shard);
      const u32 leader = __ffs(shard_group) - 1u;
      const u32 matching = __popc(shard_group);
      const u32 payload_bytes = warp_reduce_add_sync(shard_group, lane_bytes);
      const u32 short_reads = __popc(shard_group & short_mask);
      const u32 full_reads = matching - short_reads;
      const u32 dynamic_matching = __popc(shard_group & dynamic_mask);
      const u32 dynamic_short_reads =
        __popc(shard_group & dynamic_mask & short_mask);
      const u32 dynamic_full_reads = dynamic_matching - dynamic_short_reads;
      const u32 dynamic_payload_bytes =
        warp_reduce_add_sync(shard_group, dynamic_request ? lane_bytes : 0u);
      if (lane == leader) {
        const u64 issue_timestamp_ns = global_time_ns();
        const i32 status = direct_fetch_batch(
          params, shard, request_shards, request_offsets, slot_count,
          params.graph_scratch, kPersistentGraphReadBytes,
          params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, completion_statuses + shard, true, nullptr,
          completion_timestamps + shard,
          live_extent_enabled ? request_bytes : nullptr, priority, true,
          matching);
        const bool admitted = status == 0 || status == -EINPROGRESS;
        if (admitted) {
          batch.issue_timestamp_ns[shard] = issue_timestamp_ns;
          admitted_batches = 1;
          admitted_reads = matching;
          admitted_bytes = payload_bytes;
          admitted_short_reads = short_reads;
          admitted_full_reads = full_reads;
          admitted_dynamic_bytes = dynamic_payload_bytes;
          admitted_dynamic_short_reads = dynamic_short_reads;
          admitted_dynamic_full_reads = dynamic_full_reads;
        } else {
          // Publish an immediate producer-side failure into the same
          // completion word consumed by the lane-local state transition.
          completion_statuses[shard] = status;
          if (status == -EAGAIN ||
              priority == DirectBatchPriority::speculative) {
            rejected_reads = matching;
          } else {
            fatal_batches = 1;
          }
        }
      }
    }
    __syncwarp(full_warp);
    if (valid_request) {
      const i32 status = completion_statuses[shard];
      if (status != 0 && status != -EINPROGRESS) {
        rob[slot_begin + lane].transfer_bytes = 0;
        if (priority == DirectBatchPriority::speculative) {
          request_shards[lane] = UINT32_MAX;
        }
      }
    }
    admitted_batches = warp_reduce_add_sync(full_warp, admitted_batches);
    admitted_reads = warp_reduce_add_sync(full_warp, admitted_reads);
    admitted_short_reads =
      warp_reduce_add_sync(full_warp, admitted_short_reads);
    admitted_full_reads = warp_reduce_add_sync(full_warp, admitted_full_reads);
    admitted_dynamic_short_reads =
      warp_reduce_add_sync(full_warp, admitted_dynamic_short_reads);
    admitted_dynamic_full_reads =
      warp_reduce_add_sync(full_warp, admitted_dynamic_full_reads);
    rejected_reads = warp_reduce_add_sync(full_warp, rejected_reads);
    fatal_batches = warp_reduce_add_sync(full_warp, fatal_batches);
    admitted_bytes = warp_reduce_add_sync(full_warp, admitted_bytes);
    admitted_dynamic_bytes =
      warp_reduce_add_sync(full_warp, admitted_dynamic_bytes);
    if (lane == 0) {
      batch.active = admitted_batches != 0 ? 1u : 0u;
      batch.rejected += rejected_reads;
      batch.fatal |= fatal_batches != 0 ? 1u : 0u;
      *remote_batches += admitted_batches;
      *total_remote_reads += admitted_reads;
      *classified_graph_reads += admitted_reads;
      *graph_read_bytes += admitted_bytes;
      *classified_graph_bytes += admitted_bytes;
      *graph_live_extent_reads += admitted_short_reads;
      *graph_full_record_reads += admitted_full_reads;
      add_dynamic_graph_read_telemetry(
        dynamic_telemetry, admitted_dynamic_short_reads,
        admitted_dynamic_full_reads, admitted_dynamic_bytes);
      *queue_rejects += rejected_reads;
      if (prefetch_reads != nullptr) {
        *prefetch_reads += admitted_reads;
      }
      if (prefetch_bytes != nullptr) {
        *prefetch_bytes += admitted_bytes;
      }
    }
    __syncwarp(full_warp);
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::issued)) {
        const u32 shard = request_shards[lane];
        if (shard != UINT32_MAX &&
            (completion_statuses[shard] == 0 ||
             completion_statuses[shard] == -EINPROGRESS)) {
          entry.state = static_cast<u8>(FrontierRequestState::inflight);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
        }
      }
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Build the complete issue frontier once and publish at most one descriptor
// per shard.  One warp lane owns one ROB request and `match_any` forms the
// per-shard groups.  This is deliberately request-centric: the former
// shard-centric implementation kept every request/telemetry accumulator live
// across direct_fetch_split_batch(), compiled at the architectural register
// limit, and repeatedly scanned the 32-slot ROB once per shard.  Group leaders
// now publish the descriptor while the other request lanes retain only their
// own metadata.  Query-local telemetry is reduced once by the warp; no atomics
// are needed.
//
// The owner still splits the descriptor into a strict-priority critical head
// and a speculative suffix.  This helper changes only GPU issue overhead, not
// request ordering, completion lifetime, or the Issue/Commit contract.
__device__ bool issue_split_frontier_graph_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& core_batch,
  FrontierGraphBatchState& tail_batch, u32 request_count, u32 core_slot_count,
  i32* core_batch_statuses, u64* core_batch_completion_timestamps_ns,
  i32* tail_batch_statuses, u64* tail_batch_completion_timestamps_ns,
  u32* remote_batches, u32* total_remote_reads, u64* graph_read_bytes,
  u32* graph_live_extent_reads, u32* graph_full_record_reads,
  u32* critical_graph_reads, u64* critical_graph_bytes,
  u32* speculative_graph_reads, u64* speculative_graph_bytes,
  u32* core_queue_rejects, u32* speculative_queue_rejects,
  u32* core_prefetch_reads, u64* core_prefetch_bytes, u32* tail_admitted,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  // `request_count` is the admitted frontier width for this epoch, not the
  // compile-time ROB capacity.  Keeping the descriptor and all shard scans
  // bounded by the live prefix avoids repeatedly walking unused ROB slots.
  request_count =
    min(request_count, static_cast<u32>(kPersistentFrontierRobCapacity));
  core_slot_count = min(core_slot_count, request_count);
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentFrontierRobCapacity;
  u32* request_shards = params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes =
    params.speculative_graph_request_bytes == nullptr
      ? nullptr
      : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr
      : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* core_statuses =
    core_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* core_timestamps =
    core_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  i32* tail_statuses =
    tail_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* tail_timestamps =
    tail_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (core_batch.active != 0 || tail_batch.active != 0 ||
      core_slot_count == 0 ||
      core_slot_count >= kPersistentFrontierRobCapacity ||
      core_batch_statuses == nullptr ||
      core_batch_completion_timestamps_ns == nullptr ||
      tail_batch_statuses == nullptr ||
      tail_batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) {
      core_batch.fatal = 1;
      tail_batch.fatal = 1;
    }
    __syncthreads();
    return false;
  }

  if (threadIdx.x < 32) {
    if (lane == 0) {
      core_batch.active = 0;
      core_batch.fatal = 0;
      core_batch.rejected = 0;
      core_batch.finish_had_pending = 0;
      tail_batch.active = 0;
      tail_batch.fatal = 0;
      tail_batch.rejected = 0;
      tail_batch.finish_had_pending = 0;
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      core_batch.issue_timestamp_ns[shard] = 0;
      tail_batch.issue_timestamp_ns[shard] = 0;
    }
    if (lane < request_count) {
      request_shards[lane] = UINT32_MAX;
      request_offsets[lane] = 0;
      request_local_iovas[lane] = 0;
      if (request_bytes != nullptr) {
        request_bytes[lane] = params.graph_entry_bytes;
      }
      if (request_handles != nullptr) {
        request_handles[lane] = kInvalidDeviceHandle;
      }
      if (validation_states != nullptr) {
        validation_states[lane] =
          static_cast<u8>(FrontierValidationState::unknown);
      }
    }
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      core_statuses[shard] = 0;
      core_timestamps[shard] = 0;
      tail_statuses[shard] = 0;
      tail_timestamps[shard] = 0;
    }
    __syncwarp(full_warp);

    if (lane < request_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::issued)) {
        u32 acquired_slot = UINT32_MAX;
        u32 shard = UINT32_MAX;
        u64 offset = 0;
        u64 local_iova = 0;
        u32 transfer_bytes = params.graph_entry_bytes;
        if (prepare_graph_record_in_scratch(
              params, entry.node_handle, descriptor.query_slot, lane,
              acquired_slot, shard, offset, local_iova, transfer_bytes)) {
          entry.scratch_slot = static_cast<u8>(lane);
          entry.transfer_bytes = transfer_bytes;
          request_shards[lane] = shard;
          request_offsets[lane] = offset;
          request_local_iovas[lane] = local_iova;
          if (request_bytes != nullptr) {
            request_bytes[lane] = transfer_bytes;
          }
          if (request_handles != nullptr) {
            request_handles[lane] = entry.node_handle;
          }
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
        }
      }
    }
    __syncwarp(full_warp);

    const u32 shard = lane < request_count ? request_shards[lane] : UINT32_MAX;
    const bool valid_request = lane < request_count && shard != UINT32_MAX;
    const u32 lane_bytes =
      valid_request
        ? (live_extent_enabled ? request_bytes[lane] : params.graph_entry_bytes)
        : 0u;
    const u32 valid_mask = __ballot_sync(full_warp, valid_request);
    const u32 core_mask =
      __ballot_sync(full_warp, valid_request && lane < core_slot_count);
    const u32 tail_mask = valid_mask & ~core_mask;
    const u32 short_mask = __ballot_sync(
      full_warp, valid_request && lane_bytes < params.graph_entry_bytes);
    const bool dynamic_request =
      valid_request && dynamic_graph_telemetry_handle(rob[lane].node_handle);
    const u32 dynamic_mask = __ballot_sync(full_warp, dynamic_request);

    u32 admitted_core_reads = 0;
    u32 admitted_tail_reads = 0;
    u32 admitted_core_bytes = 0;
    u32 admitted_tail_bytes = 0;
    u32 admitted_short_reads = 0;
    u32 admitted_full_reads = 0;
    u32 admitted_dynamic_short_reads = 0;
    u32 admitted_dynamic_full_reads = 0;
    u32 admitted_dynamic_bytes = 0;
    u32 admitted_core_batches = 0;
    u32 admitted_tail_batches = 0;
    u32 rejected_core_reads = 0;
    u32 rejected_tail_reads = 0;
    u32 fatal_core_batches = 0;

    if (valid_request) {
      const u32 shard_group = __match_any_sync(valid_mask, shard);
      const u32 leader = __ffs(shard_group) - 1u;
      const u32 core_matching = __popc(shard_group & core_mask);
      const u32 tail_matching = __popc(shard_group & tail_mask);
      const u32 core_payload_bytes = warp_reduce_add_sync(
        shard_group, lane < core_slot_count ? lane_bytes : 0u);
      const u32 tail_payload_bytes = warp_reduce_add_sync(
        shard_group, lane < core_slot_count ? 0u : lane_bytes);
      const u32 short_reads = __popc(shard_group & short_mask);
      const u32 full_reads = core_matching + tail_matching - short_reads;
      const u32 dynamic_matching = __popc(shard_group & dynamic_mask);
      const u32 dynamic_short_reads =
        __popc(shard_group & dynamic_mask & short_mask);
      const u32 dynamic_full_reads = dynamic_matching - dynamic_short_reads;
      const u32 dynamic_payload_bytes =
        warp_reduce_add_sync(shard_group, dynamic_request ? lane_bytes : 0u);

      if (lane == leader) {
        // Timestamp before publication: the exclusive owner may dequeue and
        // complete a tiny batch before this producer warp resumes. Sampling
        // after direct_fetch_split_batch() could therefore make completion
        // appear earlier than issue and underflow unsigned latency telemetry.
        const u64 issued_ns = global_time_ns();
        const i32 issue_status = direct_fetch_split_batch(
          params, shard, request_shards, request_offsets, request_count,
          core_slot_count, params.graph_entry_bytes,
          (descriptor.query_slot + shard) % params.direct_qps_per_node,
          request_local_iovas, core_statuses + shard, core_timestamps + shard,
          tail_statuses + shard, tail_timestamps + shard,
          live_extent_enabled ? request_bytes : nullptr);
        const bool admitted = issue_status == 0 || issue_status == -EINPROGRESS;
        if (admitted) {
          if (core_matching != 0) {
            core_batch.issue_timestamp_ns[shard] = issued_ns;
            admitted_core_reads = core_matching;
            admitted_core_bytes = core_payload_bytes;
            admitted_core_batches = 1;
          }
          if (tail_matching != 0) {
            tail_batch.issue_timestamp_ns[shard] = issued_ns;
            admitted_tail_reads = tail_matching;
            admitted_tail_bytes = tail_payload_bytes;
            admitted_tail_batches = 1;
          }
          admitted_short_reads = short_reads;
          admitted_full_reads = full_reads;
          admitted_dynamic_short_reads = dynamic_short_reads;
          admitted_dynamic_full_reads = dynamic_full_reads;
          admitted_dynamic_bytes = dynamic_payload_bytes;
        } else {
          if (core_matching != 0) {
            core_statuses[shard] = issue_status;
            if (issue_status == -EAGAIN) {
              rejected_core_reads = core_matching;
            } else {
              fatal_core_batches = 1;
            }
          }
          if (tail_matching != 0) {
            // Speculation is lossy by design. Convert every immediate
            // owner-side rejection to the same retryable stale state.
            tail_statuses[shard] = -EAGAIN;
            rejected_tail_reads = tail_matching;
          }
        }
      }
    }

    // The leader-only values above are query-local.  One warp reduction is
    // cheaper than atomically publishing once per populated shard and keeps
    // all counters out of the direct-fetch call's live range.
    admitted_core_reads = warp_reduce_add_sync(full_warp, admitted_core_reads);
    admitted_tail_reads = warp_reduce_add_sync(full_warp, admitted_tail_reads);
    admitted_core_bytes = warp_reduce_add_sync(full_warp, admitted_core_bytes);
    admitted_tail_bytes = warp_reduce_add_sync(full_warp, admitted_tail_bytes);
    admitted_short_reads =
      warp_reduce_add_sync(full_warp, admitted_short_reads);
    admitted_full_reads = warp_reduce_add_sync(full_warp, admitted_full_reads);
    admitted_dynamic_short_reads =
      warp_reduce_add_sync(full_warp, admitted_dynamic_short_reads);
    admitted_dynamic_full_reads =
      warp_reduce_add_sync(full_warp, admitted_dynamic_full_reads);
    admitted_dynamic_bytes =
      warp_reduce_add_sync(full_warp, admitted_dynamic_bytes);
    admitted_core_batches =
      warp_reduce_add_sync(full_warp, admitted_core_batches);
    admitted_tail_batches =
      warp_reduce_add_sync(full_warp, admitted_tail_batches);
    rejected_core_reads =
      warp_reduce_add_sync(full_warp, rejected_core_reads);
    rejected_tail_reads =
      warp_reduce_add_sync(full_warp, rejected_tail_reads);
    fatal_core_batches =
      warp_reduce_add_sync(full_warp, fatal_core_batches);
    if (lane == 0) {
      const u32 admitted_reads = admitted_core_reads + admitted_tail_reads;
      const u64 admitted_bytes =
        static_cast<u64>(admitted_core_bytes) + admitted_tail_bytes;
      core_batch.active = admitted_core_batches != 0 ? 1u : 0u;
      tail_batch.active = admitted_tail_batches != 0 ? 1u : 0u;
      core_batch.rejected = rejected_core_reads;
      tail_batch.rejected = rejected_tail_reads;
      core_batch.fatal = fatal_core_batches != 0 ? 1u : 0u;
      *remote_batches += admitted_core_batches + admitted_tail_batches;
      *total_remote_reads += admitted_reads;
      *graph_read_bytes += admitted_bytes;
      *graph_live_extent_reads += admitted_short_reads;
      *graph_full_record_reads += admitted_full_reads;
      add_dynamic_graph_read_telemetry(
        dynamic_telemetry, admitted_dynamic_short_reads,
        admitted_dynamic_full_reads, admitted_dynamic_bytes);
      *critical_graph_reads += admitted_core_reads;
      *critical_graph_bytes += admitted_core_bytes;
      *speculative_graph_reads += admitted_tail_reads;
      *speculative_graph_bytes += admitted_tail_bytes;
      *core_queue_rejects += rejected_core_reads;
      *speculative_queue_rejects += rejected_tail_reads;
      *core_prefetch_reads += admitted_core_reads;
      *core_prefetch_bytes += admitted_core_bytes;
      // This is the single physical admission point for a split tail.
      // Counting resident INFLIGHT slots in later core-only rounds would
      // charge the same descriptor repeatedly and bias queue-pressure
      // feedback against contraction.
      *tail_admitted += admitted_tail_reads;
    }
    __syncwarp(full_warp);

    if (lane < request_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::issued)) {
        const i32 status = shard == UINT32_MAX
                             ? -EINVAL
                             : (lane < core_slot_count ? core_statuses[shard]
                                                       : tail_statuses[shard]);
        // An owner can dequeue the just-published split descriptor and reject
        // its suffix before this producer warp reaches the state transition.
        // A nonzero tail issue token proves that the descriptor itself was
        // admitted and its reads/bytes were counted optimistically. Preserve
        // INFLIGHT in that race so the terminal tail drain performs the one
        // complete admission correction. An immediate ring-push rejection
        // has no token and still takes the ordinary STALE path below.
        const bool admitted_tail_rejected_by_owner =
          lane >= core_slot_count && status == -EAGAIN && shard != UINT32_MAX &&
          tail_batch.issue_timestamp_ns[shard] != 0;
        if (status == 0 || status == -EINPROGRESS ||
            admitted_tail_rejected_by_owner) {
          entry.state = static_cast<u8>(FrontierRequestState::inflight);
        } else {
          // Rejected requests did not consume network bytes and must not be
          // charged as speculative waste when their terminal slot is reclaimed.
          entry.transfer_bytes = 0;
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
        }
      }
    }
  }
  __syncthreads();
  return core_batch.fatal == 0 && tail_batch.fatal == 0;
}

// Prepare and enqueue one shadow-frontier wave without waiting for CQ
// completion. Request metadata, completion words, and payload slots are
// disjoint from every critical/dynamic-code path and remain immutable until
// finish_frontier_graph_batch() drains the admitted descriptors.
__device__ bool issue_frontier_graph_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_begin,
  u32 slot_count, i32* batch_statuses, u64* batch_completion_timestamps_ns,
  DirectBatchPriority priority, bool nonblocking_enqueue, u32* remote_batches,
  u32* total_remote_reads, u64* graph_read_bytes, u32* graph_live_extent_reads,
  u32* graph_full_record_reads, u32* speculative_graph_reads,
  u64* speculative_graph_bytes, u32* speculative_queue_rejects,
  u32* prefetch_reads = nullptr, u64* prefetch_bytes = nullptr,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  // Completion/status/request storage is wave-owned. Reusing it while an
  // owner still holds the prior descriptor is an ABA corruption, not
  // backpressure. The query pipeline normally defers new tail admission; keep
  // this guard as a fail-closed lifetime invariant.
  if (batch.active != 0) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (slot_begin > kPersistentFrontierRobCapacity ||
      slot_count > kPersistentFrontierRobCapacity - slot_begin ||
      batch_statuses == nullptr || batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (blockDim.x == kApproximateSortThreadsCompact &&
      slot_count <= kPersistentFrontierRobCapacity &&
      speculative_graph_reads != nullptr &&
      speculative_graph_bytes != nullptr && remote_batches != nullptr &&
      total_remote_reads != nullptr && graph_read_bytes != nullptr &&
      graph_live_extent_reads != nullptr &&
      graph_full_record_reads != nullptr &&
      speculative_queue_rejects != nullptr) {
    return issue_narrow_frontier_graph_batch(
      params, descriptor, rob, batch, slot_begin, slot_count, batch_statuses,
      batch_completion_timestamps_ns, priority, remote_batches,
      total_remote_reads, graph_read_bytes, graph_live_extent_reads,
      graph_full_record_reads, speculative_graph_reads, speculative_graph_bytes,
      speculative_queue_rejects, prefetch_reads, prefetch_bytes,
      dynamic_telemetry);
  }
  const size_t request_base = static_cast<size_t>(descriptor.query_slot) *
                                kPersistentFrontierRobCapacity +
                              slot_begin;
  u32* request_shards = params.speculative_graph_request_shards + request_base;
  u64* request_offsets =
    params.speculative_graph_request_offsets + request_base;
  u64* request_local_iovas =
    params.speculative_graph_request_local_iovas + request_base;
  u32* request_bytes =
    params.speculative_graph_request_bytes == nullptr
      ? nullptr
      : params.speculative_graph_request_bytes + request_base;
  u64* request_handles =
    params.speculative_graph_request_handles == nullptr
      ? nullptr
      : params.speculative_graph_request_handles + request_base;
  u8* validation_states =
    params.speculative_graph_validation_states == nullptr
      ? nullptr
      : params.speculative_graph_validation_states + request_base;
  i32* completion_statuses =
    batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const bool live_extent_enabled = request_bytes != nullptr;

  if (threadIdx.x == 0) {
    batch.active = 0;
    batch.fatal = 0;
    batch.rejected = 0;
    batch.finish_had_pending = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    batch.issue_timestamp_ns[shard] = 0;
  }
  __syncthreads();
  for (u32 local = threadIdx.x; local < slot_count; local += blockDim.x) {
    request_shards[local] = UINT32_MAX;
    request_offsets[local] = 0;
    request_local_iovas[local] = 0;
    if (request_bytes != nullptr) {
      request_bytes[local] = params.graph_entry_bytes;
    }
    if (request_handles != nullptr) {
      request_handles[local] = kInvalidDeviceHandle;
    }
    if (validation_states != nullptr) {
      validation_states[local] =
        static_cast<u8>(FrontierValidationState::unknown);
    }
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state != static_cast<u8>(FrontierRequestState::issued)) {
      continue;
    }
    u32 acquired_slot = UINT32_MAX;
    u32 shard = UINT32_MAX;
    u64 offset = 0;
    u64 local_iova = 0;
    u32 transfer_bytes = params.graph_entry_bytes;
    const u32 scratch_slot = slot;
    if (!prepare_graph_record_in_scratch(
          params, entry.node_handle, descriptor.query_slot, scratch_slot,
          acquired_slot, shard, offset, local_iova, transfer_bytes)) {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
      continue;
    }
    entry.scratch_slot = static_cast<u8>(scratch_slot);
    entry.transfer_bytes = transfer_bytes;
    request_shards[local] = shard;
    request_offsets[local] = offset;
    request_local_iovas[local] = local_iova;
    if (request_bytes != nullptr) request_bytes[local] = transfer_bytes;
    if (request_handles != nullptr) {
      request_handles[local] = entry.node_handle;
    }
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    completion_statuses[shard] = 0;
    completion_timestamps[shard] = 0;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    u32 matching = 0;
    u32 payload_bytes = 0;
    u32 short_reads = 0;
    u32 full_reads = 0;
    u32 dynamic_short_reads = 0;
    u32 dynamic_full_reads = 0;
    u64 dynamic_payload_bytes = 0;
    for (u32 local = 0; local < slot_count; ++local) {
      if (request_shards[local] != shard) continue;
      ++matching;
      const u32 bytes =
        live_extent_enabled ? request_bytes[local] : params.graph_entry_bytes;
      payload_bytes += bytes;
      if (bytes < params.graph_entry_bytes) {
        ++short_reads;
      } else {
        ++full_reads;
      }
      if (dynamic_graph_telemetry_handle(rob[slot_begin + local].node_handle)) {
        dynamic_payload_bytes += bytes;
        if (bytes < params.graph_entry_bytes) {
          ++dynamic_short_reads;
        } else {
          ++dynamic_full_reads;
        }
      }
    }
    if (matching == 0) continue;
    const u64 issue_timestamp_ns = global_time_ns();
    const i32 status = direct_fetch_batch(
      params, shard, request_shards, request_offsets, slot_count,
      params.graph_scratch, kPersistentGraphReadBytes, params.graph_entry_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iovas, completion_statuses + shard, true, nullptr,
      completion_timestamps + shard,
      live_extent_enabled ? request_bytes : nullptr, priority,
      nonblocking_enqueue);
    const bool admitted = status == 0 || status == -EINPROGRESS;
    if (admitted) {
      batch.issue_timestamp_ns[shard] = issue_timestamp_ns;
      atomicExch(&batch.active, 1u);
      atomicAdd(remote_batches, 1u);
      atomicAdd(total_remote_reads, matching);
      atomicAdd(speculative_graph_reads, matching);
      if (prefetch_reads != nullptr) {
        atomicAdd(prefetch_reads, matching);
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(graph_read_bytes),
                static_cast<unsigned long long>(payload_bytes));
      atomicAdd(reinterpret_cast<unsigned long long*>(speculative_graph_bytes),
                static_cast<unsigned long long>(payload_bytes));
      if (prefetch_bytes != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(prefetch_bytes),
                  static_cast<unsigned long long>(payload_bytes));
      }
      if (short_reads != 0) {
        atomicAdd(graph_live_extent_reads, short_reads);
      }
      if (full_reads != 0) {
        atomicAdd(graph_full_record_reads, full_reads);
      }
      add_dynamic_graph_read_telemetry(dynamic_telemetry, dynamic_short_reads,
                                       dynamic_full_reads,
                                       dynamic_payload_bytes);
    } else if (status == -EAGAIN) {
      atomicAdd(&batch.rejected, matching);
      atomicAdd(speculative_queue_rejects, matching);
      // This payload never entered an owner queue. Keep request-count
      // rejection telemetry, but do not later classify its bytes as fetched
      // speculative waste.
      for (u32 local = 0; local < slot_count; ++local) {
        if (request_shards[local] == shard) {
          rob[slot_begin + local].transfer_bytes = 0;
        }
      }
    } else {
      // This function is used for the speculative tail in the persistent
      // query path.  Treat any enqueue rejection as a stale probe; only the
      // critical owner path is allowed to make a query fail-stop.
      if (priority == DirectBatchPriority::speculative) {
        atomicAdd(&batch.rejected, matching);
        atomicAdd(speculative_queue_rejects, matching);
        for (u32 local = 0; local < slot_count; ++local) {
          if (request_shards[local] == shard) {
            request_shards[local] = UINT32_MAX;
            rob[slot_begin + local].transfer_bytes = 0;
          }
        }
      } else {
        atomicExch(&batch.fatal, 1u);
      }
    }
  }
  __syncthreads();

  for (u32 local = threadIdx.x; local < slot_count; local += blockDim.x) {
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state != static_cast<u8>(FrontierRequestState::issued)) {
      continue;
    }
    const u32 shard = request_shards[local];
    if (shard == UINT32_MAX) continue;
    const i32 status = completion_statuses[shard];
    if (status == 0 || status == -EINPROGRESS) {
      entry.state = static_cast<u8>(FrontierRequestState::inflight);
    } else {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

__device__ __forceinline__ bool finish_core_frontier_graph_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_count,
  i32* batch_statuses, u64* batch_completion_timestamps_ns, u64* wait_cycles,
  u64* completion_latency_ns, u64* completion_groups, u32* arrived, u32* stale,
  u32* ready_waves, DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  // No descriptor was admitted for this epoch. Avoid rescanning stale
  // per-shard completion words and the core ROB prefix; in a stable wide
  // frontier this is the common case because retained tail records already
  // cover the next commit prefix.
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses =
    batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 core_wait_started_cycles;
  __shared__ u32 core_finish_had_pending;
  __shared__ u32 core_finish_failed;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      core_wait_started_cycles = clock64();
      core_finish_had_pending = 0;
      core_finish_failed = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    bool lane_failed = false;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      if (completion_statuses[shard] == -EINPROGRESS) {
        lane_pending = true;
        completion_statuses[shard] =
          wait_direct_batch(params, completion_statuses + shard);
      }
      if (completion_statuses[shard] != 0 &&
          completion_statuses[shard] != -EAGAIN) {
        lane_failed = true;
      }
      if (completion_statuses[shard] == 0 &&
          batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        atomicAdd(reinterpret_cast<unsigned long long*>(completion_latency_ns),
                  static_cast<unsigned long long>(
                    completed_ns - batch.issue_timestamp_ns[shard]));
        atomicAdd(reinterpret_cast<unsigned long long*>(completion_groups),
                  1ULL);
        batch.issue_timestamp_ns[shard] = 0;
      }
    }
    const u32 pending_mask = __ballot_sync(full_warp, lane_pending);
    const u32 failed_mask = __ballot_sync(full_warp, lane_failed);
    if (lane == 0) {
      core_finish_had_pending = pending_mask != 0 ? 1u : 0u;
      core_finish_failed = failed_mask != 0 ? 1u : 0u;
      batch.finish_had_pending |= core_finish_had_pending;
    }
    __syncwarp(full_warp);

    bool local_arrived = false;
    bool local_stale = false;
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::inflight)) {
        const u32 shard = request_shards[lane];
        if (shard == UINT32_MAX || completion_statuses[shard] != 0) {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
          local_stale = true;
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::arrived);
          local_arrived = true;
          const FrontierValidationState validation =
            validate_frontier_record_local(params, descriptor.query_slot, entry,
                                           dynamic_telemetry);
          if (validation == FrontierValidationState::valid) {
            entry.state = static_cast<u8>(FrontierRequestState::validated);
            entry.validation = static_cast<u8>(FrontierValidationState::valid);
          } else {
            entry.state = static_cast<u8>(FrontierRequestState::stale);
            entry.validation = static_cast<u8>(validation);
            local_stale = true;
          }
        }
      }
    }
    const u32 arrived_mask = __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask = __ballot_sync(full_warp, local_stale);
    if (lane == 0) {
      *arrived += __popc(arrived_mask);
      *stale += __popc(stale_mask);
      if (core_finish_had_pending == 0) ++*ready_waves;
      *wait_cycles += clock64() - core_wait_started_cycles;
      batch.active = 0;
      batch.fatal |= core_finish_failed;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Consume every shard group that is READY at one observation boundary.
// wait_if_none selects between two progress modes:
//
//   * true: wait for the first completion, then harvest the complete visible
//     CQ backlog;
//   * false: return zero immediately when the CQ is empty so the caller can
//     consume already READY query-private parent work during that network gap.
//
// The tri-state result is -1 for a fatal batch, zero for a nonblocking empty
// observation, and one after useful progress. A single-group consumer paid
// two full shard scans, a wide call ABI, and four CTA barriers even when the
// completed shard exposed only parents behind a lower-rank hole. Batched
// harvest changes only private ROB arrival/validation state; Beam publication
// remains outside this function and therefore frontier-consistent.
__device__ __noinline__ i32 finish_next_core_frontier_group(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_count,
  const CoreFrontierTelemetry& telemetry, bool wait_if_none = true) {
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses =
    params.core_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    params.core_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 ready_shards;
  __shared__ u64 group_exposed_wait_cycles;

  if (threadIdx.x == 0) {
    ready_shards = 0;
    group_exposed_wait_cycles = 0;
  }
  __syncthreads();
  if (batch.active == 0) return batch.fatal == 0 ? 1 : -1;

  // One control lane observes at most 64 completion words. Other warps remain
  // quiescent until there is useful validation/decode work.
  if (threadIdx.x == 0) {
    u64 wait_started_cycles = 0;
    for (;;) {
      u64 observed_ready = 0;
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        if (batch.issue_timestamp_ns[shard] == 0) continue;
        const i32 status =
          *reinterpret_cast<volatile i32*>(completion_statuses + shard);
        if (status != -EINPROGRESS) {
          observed_ready |= u64{1} << shard;
        }
      }
      if (observed_ready != 0) {
        ready_shards = observed_ready;
        if (wait_started_cycles != 0) {
          group_exposed_wait_cycles = clock64() - wait_started_cycles;
        }
        break;
      }
      batch.finish_had_pending = 1;
      if (!wait_if_none) break;
      if (wait_started_cycles == 0) {
        wait_started_cycles = clock64();
      }
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        group_exposed_wait_cycles = clock64() - wait_started_cycles;
        break;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        group_exposed_wait_cycles = clock64() - wait_started_cycles;
        break;
      }
      device_ring_relax(128);
    }
    // A nonblocking empty observation is ordinary backpressure: preserve the
    // live batch so the caller can cover the gap with buffered score work.
    // Only a blocking wait can leave without a completion after stop/direct
    // disable and therefore turn the batch fatal.
    if (ready_shards == 0 && wait_if_none) {
      batch.fatal = 1;
      batch.active = 0;
    }
  }
  __syncthreads();
  if (ready_shards == 0) {
    if (!wait_if_none && batch.active != 0 && batch.fatal == 0) {
      return 0;
    }
    if (threadIdx.x == 0) {
      *telemetry.wait_cycles += group_exposed_wait_cycles;
    }
    __syncthreads();
    return -1;
  }

  bool local_arrived = false;
  bool local_stale = false;
  if (threadIdx.x < 32 && lane < slot_count) {
    const u32 shard = request_shards[lane];
    const bool in_ready_group =
      shard < kPersistentMaxShards && (ready_shards & (u64{1} << shard)) != 0;
    FrontierRobEntry& entry = rob[lane];
    if (in_ready_group &&
        entry.state == static_cast<u8>(FrontierRequestState::inflight)) {
      const i32 status = completion_statuses[shard];
      if (status != 0) {
        entry.state = static_cast<u8>(FrontierRequestState::stale);
        entry.validation =
          static_cast<u8>(FrontierValidationState::transport_rejected);
        local_stale = true;
      } else {
        entry.state = static_cast<u8>(FrontierRequestState::arrived);
        local_arrived = true;
        const FrontierValidationState validation =
          validate_frontier_record_local(params, descriptor.query_slot, entry,
                                         telemetry.dynamic_graph);
        if (validation == FrontierValidationState::valid) {
          entry.state = static_cast<u8>(FrontierRequestState::validated);
          entry.validation = static_cast<u8>(FrontierValidationState::valid);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(validation);
          local_stale = true;
        }
      }
    }
  }
  if (threadIdx.x < 32) {
    const u32 arrived_mask = __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask = __ballot_sync(full_warp, local_stale);
    if (lane == 0) {
      *telemetry.arrived += __popc(arrived_mask);
      *telemetry.stale += __popc(stale_mask);
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        if ((ready_shards & (u64{1} << shard)) == 0) continue;
        const i32 status = completion_statuses[shard];
        if (status != 0 && status != -EAGAIN) {
          batch.fatal = 1;
        }
        const u64 issue_ns = batch.issue_timestamp_ns[shard];
        if (status == 0 && issue_ns != 0) {
          const u64 owner_completed_ns = completion_timestamps[shard];
          const u64 completed_ns =
            owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
          *telemetry.completion_latency_ns += completed_ns - issue_ns;
          ++*telemetry.completion_groups;
        }
        batch.issue_timestamp_ns[shard] = 0;
      }
      bool remaining = false;
      for (u32 shard = 0; shard < params.num_shards; ++shard) {
        remaining |= batch.issue_timestamp_ns[shard] != 0;
      }
      batch.active = remaining ? 1u : 0u;
      if (batch.active == 0 && batch.finish_had_pending == 0) {
        ++*telemetry.ready_waves;
      }
      *telemetry.wait_cycles += group_exposed_wait_cycles;
    }
  }
  __syncthreads();
  return batch.fatal == 0 ? 1 : -1;
}

// Finish a narrow speculative wave with one warp.  Tail waves use at most the
// 32-entry query-local ROB, so the generic CTA path's global atomics and
// repeated block barriers are pure overhead.  Lane i owns tail slot i and
// ballot/reduction publishes the complete wave telemetry once.
__device__ __forceinline__ bool finish_narrow_speculative_frontier_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_begin,
  u32 slot_count, i32* batch_statuses, u64* batch_completion_timestamps_ns,
  bool wait_for_completion, u32* speculative_arrived, u32* speculative_stale,
  u64* speculative_wait_cycles, u64* speculative_completion_latency_ns,
  u64* speculative_completion_groups, u64* speculative_wasted_bytes,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses =
    batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base = static_cast<size_t>(descriptor.query_slot) *
                                kPersistentFrontierRobCapacity +
                              slot_begin;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 tail_finish_started_cycles;
  __shared__ u32 tail_finish_pending;
  __shared__ u32 tail_finish_failed;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      tail_finish_started_cycles = clock64();
      tail_finish_pending = 0;
      tail_finish_failed = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    u64 lane_completion_latency_ns = 0;
    u32 lane_completion_groups = 0;
    u32 lane_rejected_batches = 0;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      i32 status =
        *reinterpret_cast<volatile i32*>(completion_statuses + shard);
      if (status == -EINPROGRESS) {
        lane_pending = true;
        if (wait_for_completion) {
          status = wait_direct_batch(params, completion_statuses + shard);
        }
      }
      // This is the speculative tail.  Any owner-side completion error is
      // converted to a stale ROB entry below; the exact critical path will
      // retry the handle.  A speculative probe must not turn a recoverable
      // QP/transport condition into a query failure.
      if (status == 0 && batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        lane_completion_latency_ns +=
          completed_ns - batch.issue_timestamp_ns[shard];
        ++lane_completion_groups;
        batch.issue_timestamp_ns[shard] = 0;
      } else if (status == -EAGAIN && batch.issue_timestamp_ns[shard] != 0) {
        // The owner rejected this entire shard suffix before posting any WQE.
        // Clear its issue token so repeated nonblocking observations cannot
        // apply the physical-telemetry correction twice.
        ++lane_rejected_batches;
        batch.issue_timestamp_ns[shard] = 0;
      }
    }
    const u32 pending_mask = __ballot_sync(full_warp, lane_pending);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      lane_completion_latency_ns +=
        __shfl_down_sync(full_warp, lane_completion_latency_ns, offset);
      lane_completion_groups +=
        __shfl_down_sync(full_warp, lane_completion_groups, offset);
    }
    if (lane == 0) {
      tail_finish_pending = pending_mask != 0 ? 1u : 0u;
      tail_finish_failed = 0;
      *speculative_completion_latency_ns += lane_completion_latency_ns;
      *speculative_completion_groups += lane_completion_groups;
    }
  }
  __syncthreads();
  if (!wait_for_completion && tail_finish_pending != 0) {
    return batch.fatal == 0;
  }

  if (threadIdx.x < 32) {
    bool local_arrived = false;
    bool local_stale = false;
    u64 local_wasted_bytes = 0;
    if (lane < slot_count) {
      FrontierRobEntry& entry = rob[slot_begin + lane];
      if (entry.state == static_cast<u8>(FrontierRequestState::inflight)) {
        const u32 shard = request_shards[lane];
        const i32 status =
          shard == UINT32_MAX ? -EINVAL : completion_statuses[shard];
        if (status != 0) {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation =
            static_cast<u8>(FrontierValidationState::transport_rejected);
          local_stale = true;
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::arrived);
          local_arrived = true;
          const FrontierValidationState validation =
            validate_frontier_record_local(params, descriptor.query_slot, entry,
                                           dynamic_telemetry);
          if (validation == FrontierValidationState::valid) {
            entry.state = static_cast<u8>(FrontierRequestState::validated);
            entry.validation = static_cast<u8>(FrontierValidationState::valid);
          } else {
            entry.state = static_cast<u8>(FrontierRequestState::stale);
            entry.validation = static_cast<u8>(validation);
            local_stale = true;
          }
        }
        if (local_stale) local_wasted_bytes = entry.transfer_bytes;
      }
    }
    const u32 arrived_mask = __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask = __ballot_sync(full_warp, local_stale);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      local_wasted_bytes +=
        __shfl_down_sync(full_warp, local_wasted_bytes, offset);
    }
    if (lane == 0) {
      *speculative_arrived += __popc(arrived_mask);
      *speculative_stale += __popc(stale_mask);
      *speculative_wasted_bytes += local_wasted_bytes;
      *speculative_wait_cycles += clock64() - tail_finish_started_cycles;
      batch.active = 0;
      batch.fatal |= tail_finish_failed;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Query-tail specialization of the narrow completion path.  Every persistent
// query uses the complete 32-slot ROB address space and the dedicated tail
// completion arrays, so carrying slot bounds, array pointers, six counter
// pointers, and a runtime wait flag through the generic dispatcher only
// enlarges the call ABI and spills them before its fast-path predicates.
//
// WaitForCompletion is a compile-time property: the per-round observation is
// nonblocking, while failure/final cleanup is blocking.  Both instantiations
// retain the same completion accounting, immutable-snapshot validation, and
// lossy speculative-error semantics as the former narrow dispatcher path.
template <bool WaitForCompletion>
__device__ __noinline__ bool finish_query_tail_frontier_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch,
  const TailFrontierTelemetry& telemetry) {
  if (batch.active == 0) return batch.fatal == 0;
  constexpr u32 full_warp = 0xffffffffu;
  const u32 lane = threadIdx.x & 31u;
  i32* completion_statuses =
    params.tail_batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    params.tail_batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentFrontierRobCapacity;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  __shared__ u64 query_tail_finish_started_cycles;
  __shared__ u32 query_tail_finish_pending;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      query_tail_finish_started_cycles = clock64();
      query_tail_finish_pending = 0;
    }
    __syncwarp(full_warp);
    bool lane_pending = false;
    u64 lane_completion_latency_ns = 0;
    u32 lane_completion_groups = 0;
    u32 lane_rejected_batches = 0;
    for (u32 shard = lane; shard < params.num_shards; shard += 32) {
      i32 status =
        *reinterpret_cast<volatile i32*>(completion_statuses + shard);
      if (status == -EINPROGRESS) {
        lane_pending = true;
        if constexpr (WaitForCompletion) {
          status = wait_direct_batch(params, completion_statuses + shard);
        }
      }
      if (status == 0 && batch.issue_timestamp_ns[shard] != 0) {
        const u64 owner_completed_ns = completion_timestamps[shard];
        const u64 completed_ns =
          owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
        lane_completion_latency_ns +=
          completed_ns - batch.issue_timestamp_ns[shard];
        ++lane_completion_groups;
        batch.issue_timestamp_ns[shard] = 0;
      } else if (status == -EAGAIN && batch.issue_timestamp_ns[shard] != 0) {
        // Keep the token until the complete tail wave is terminal. A
        // nonblocking observation may see this rejected shard before another
        // shard completes; consuming it here would apply the optimistic
        // physical-read correction twice.
        ++lane_rejected_batches;
      }
    }
    const u32 pending_mask = __ballot_sync(full_warp, lane_pending);
    const bool terminal_wave = WaitForCompletion || pending_mask == 0;
    if (terminal_wave) {
      for (u32 shard = lane; shard < params.num_shards; shard += 32) {
        if (completion_statuses[shard] == -EAGAIN &&
            batch.issue_timestamp_ns[shard] != 0) {
          batch.issue_timestamp_ns[shard] = 0;
        }
      }
    }
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      lane_completion_latency_ns +=
        __shfl_down_sync(full_warp, lane_completion_latency_ns, offset);
      lane_completion_groups +=
        __shfl_down_sync(full_warp, lane_completion_groups, offset);
      lane_rejected_batches +=
        __shfl_down_sync(full_warp, lane_rejected_batches, offset);
    }
    if (lane == 0) {
      query_tail_finish_pending = pending_mask != 0 ? 1u : 0u;
      *telemetry.completion_latency_ns += lane_completion_latency_ns;
      *telemetry.completion_groups += lane_completion_groups;
      if (terminal_wave) {
        telemetry.admission_correction->rejected_batches +=
          lane_rejected_batches;
      }
    }
  }
  __syncthreads();
  if constexpr (!WaitForCompletion) {
    if (query_tail_finish_pending != 0) {
      return batch.fatal == 0;
    }
  }

  if (threadIdx.x < 32) {
    bool local_arrived = false;
    bool local_stale = false;
    bool local_admission_rejected = false;
    u64 local_wasted_bytes = 0;
    u64 local_rejected_bytes = 0;
    u32 local_rejected_live_extent_reads = 0;
    u32 local_rejected_full_record_reads = 0;
    u64 local_rejected_dynamic_bytes = 0;
    u32 local_rejected_dynamic_short_reads = 0;
    u32 local_rejected_dynamic_full_reads = 0;
    FrontierRobEntry& entry = rob[lane];
    if (entry.state == static_cast<u8>(FrontierRequestState::inflight)) {
      const u32 shard = request_shards[lane];
      const i32 status =
        shard == UINT32_MAX ? -EINVAL : completion_statuses[shard];
      if (status != 0) {
        local_admission_rejected = status == -EAGAIN;
        entry.state = static_cast<u8>(FrontierRequestState::stale);
        entry.validation =
          static_cast<u8>(FrontierValidationState::transport_rejected);
        local_stale = true;
      } else {
        entry.state = static_cast<u8>(FrontierRequestState::arrived);
        local_arrived = true;
        const FrontierValidationState validation =
          validate_frontier_record_local(params, descriptor.query_slot, entry,
                                         telemetry.dynamic_graph);
        if (validation == FrontierValidationState::valid) {
          entry.state = static_cast<u8>(FrontierRequestState::validated);
          entry.validation = static_cast<u8>(FrontierValidationState::valid);
        } else {
          entry.state = static_cast<u8>(FrontierRequestState::stale);
          entry.validation = static_cast<u8>(validation);
          local_stale = true;
        }
      }
      if (local_admission_rejected) {
        local_rejected_bytes = entry.transfer_bytes;
        local_rejected_live_extent_reads =
          entry.transfer_bytes < params.graph_entry_bytes ? 1u : 0u;
        local_rejected_full_record_reads =
          entry.transfer_bytes < params.graph_entry_bytes ? 0u : 1u;
        if (dynamic_graph_telemetry_handle(entry.node_handle)) {
          local_rejected_dynamic_bytes = entry.transfer_bytes;
          local_rejected_dynamic_short_reads =
            entry.transfer_bytes < params.graph_entry_bytes ? 1u : 0u;
          local_rejected_dynamic_full_reads =
            entry.transfer_bytes < params.graph_entry_bytes ? 0u : 1u;
        }
        // This payload never reached the NIC. Keep the terminal ROB state for
        // exact retry, but do not later charge it as speculative waste.
        entry.transfer_bytes = 0;
      } else if (local_stale) {
        local_wasted_bytes = entry.transfer_bytes;
      }
    }
    const u32 arrived_mask = __ballot_sync(full_warp, local_arrived);
    const u32 stale_mask =
      __ballot_sync(full_warp, local_stale && !local_admission_rejected);
    const u32 rejected_mask =
      __ballot_sync(full_warp, local_admission_rejected);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      local_wasted_bytes +=
        __shfl_down_sync(full_warp, local_wasted_bytes, offset);
      local_rejected_bytes +=
        __shfl_down_sync(full_warp, local_rejected_bytes, offset);
      local_rejected_live_extent_reads +=
        __shfl_down_sync(full_warp, local_rejected_live_extent_reads, offset);
      local_rejected_full_record_reads +=
        __shfl_down_sync(full_warp, local_rejected_full_record_reads, offset);
      local_rejected_dynamic_bytes +=
        __shfl_down_sync(full_warp, local_rejected_dynamic_bytes, offset);
      local_rejected_dynamic_short_reads +=
        __shfl_down_sync(full_warp, local_rejected_dynamic_short_reads, offset);
      local_rejected_dynamic_full_reads +=
        __shfl_down_sync(full_warp, local_rejected_dynamic_full_reads, offset);
    }
    if (lane == 0) {
      *telemetry.arrived += __popc(arrived_mask);
      *telemetry.stale += __popc(stale_mask);
      *telemetry.wasted_bytes += local_wasted_bytes;
      telemetry.admission_correction->rejected_reads += __popc(rejected_mask);
      telemetry.admission_correction->rejected_bytes += local_rejected_bytes;
      telemetry.admission_correction->rejected_live_extent_reads +=
        local_rejected_live_extent_reads;
      telemetry.admission_correction->rejected_full_record_reads +=
        local_rejected_full_record_reads;
      // Split admission initially counts the complete core+tail descriptor.
      // If the owner later rejects only the tail before posting any WQE, remove
      // its dynamic traffic here exactly once alongside the existing deferred
      // aggregate correction. This keeps the extent counters physical even
      // if callers use an older TailAdmissionCorrection consumer.
      if (telemetry.dynamic_graph != nullptr) {
        telemetry.dynamic_graph->read_bytes -= min(
          telemetry.dynamic_graph->read_bytes, local_rejected_dynamic_bytes);
        telemetry.dynamic_graph->short_reads -=
          min(telemetry.dynamic_graph->short_reads,
              local_rejected_dynamic_short_reads);
        telemetry.dynamic_graph->full_reads -=
          min(telemetry.dynamic_graph->full_reads,
              local_rejected_dynamic_full_reads);
      }
      *telemetry.wait_cycles += clock64() - query_tail_finish_started_cycles;
      batch.active = 0;
    }
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Drain an admitted speculative wave and validate each immutable local
// snapshot. Snapshot invalidity is speculative waste, not a transport failure:
// if the handle later becomes critical it will use the full authoritative
// retry path. A real owner/QP error remains fail-stop.
__device__ bool finish_frontier_graph_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  FrontierRobEntry* rob, FrontierGraphBatchState& batch, u32 slot_begin,
  u32 slot_count, i32* batch_statuses, u64* batch_completion_timestamps_ns,
  bool wait_for_completion, u32* speculative_arrived, u32* speculative_stale,
  u64* speculative_wait_cycles, u64* speculative_completion_latency_ns,
  u64* speculative_completion_groups, u64* speculative_wasted_bytes,
  u32* classified_arrived = nullptr, u32* classified_stale = nullptr,
  u64* classified_wasted_bytes = nullptr, u32* classified_ready_waves = nullptr,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  if (batch.active == 0) return batch.fatal == 0;
  if (slot_begin > kPersistentFrontierRobCapacity ||
      slot_count > kPersistentFrontierRobCapacity - slot_begin ||
      batch_statuses == nullptr || batch_completion_timestamps_ns == nullptr) {
    if (threadIdx.x == 0) batch.fatal = 1;
    __syncthreads();
    return false;
  }
  if (blockDim.x == kApproximateSortThreadsCompact &&
      slot_count <= kPersistentFrontierRobCapacity &&
      speculative_arrived != nullptr && speculative_stale != nullptr &&
      speculative_wait_cycles != nullptr &&
      speculative_completion_latency_ns != nullptr &&
      speculative_completion_groups != nullptr &&
      speculative_wasted_bytes != nullptr && classified_arrived == nullptr &&
      classified_stale == nullptr && classified_wasted_bytes == nullptr &&
      classified_ready_waves == nullptr) {
    return finish_narrow_speculative_frontier_batch(
      params, descriptor, rob, batch, slot_begin, slot_count, batch_statuses,
      batch_completion_timestamps_ns, wait_for_completion, speculative_arrived,
      speculative_stale, speculative_wait_cycles,
      speculative_completion_latency_ns, speculative_completion_groups,
      speculative_wasted_bytes, dynamic_telemetry);
  }
  if (blockDim.x == kApproximateSortThreadsCompact && wait_for_completion &&
      slot_begin == 0 && slot_count <= kPersistentScoreChunk &&
      speculative_arrived == nullptr && speculative_stale == nullptr &&
      speculative_wasted_bytes == nullptr && classified_arrived != nullptr &&
      classified_stale != nullptr && classified_ready_waves != nullptr &&
      speculative_wait_cycles != nullptr &&
      speculative_completion_latency_ns != nullptr &&
      speculative_completion_groups != nullptr) {
    return finish_core_frontier_graph_batch(
      params, descriptor, rob, batch, slot_count, batch_statuses,
      batch_completion_timestamps_ns, speculative_wait_cycles,
      speculative_completion_latency_ns, speculative_completion_groups,
      classified_arrived, classified_stale, classified_ready_waves,
      dynamic_telemetry);
  }
  i32* completion_statuses =
    batch_statuses +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  u64* completion_timestamps =
    batch_completion_timestamps_ns +
    static_cast<size_t>(descriptor.query_slot) * params.num_shards;
  __shared__ u64 speculative_wait_started_cycles;
  __shared__ u32 speculative_finish_failed;
  __shared__ u32 speculative_finish_pending;
  if (threadIdx.x == 0) {
    speculative_wait_started_cycles = clock64();
    speculative_finish_failed = 0;
    speculative_finish_pending = 0;
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (completion_statuses[shard] == -EINPROGRESS) {
      atomicExch(&speculative_finish_pending, 1u);
      if (wait_for_completion) {
        completion_statuses[shard] =
          wait_direct_batch(params, completion_statuses + shard);
      } else {
        continue;
      }
    }
    // The generic path is also a speculative-tail drain.  Completion errors
    // are handled as stale entries in the slot pass below; they must not
    // fail-stop the query or strand the exact critical retry.
    if (completion_statuses[shard] == 0 &&
        batch.issue_timestamp_ns[shard] != 0 &&
        speculative_completion_latency_ns != nullptr) {
      // The owner publishes the completion timestamp before the release
      // store to completion_status.  Consume that timestamp exactly once:
      // a nonblocking tail poll may observe one shard complete while another
      // shard in the same wave is still in flight.
      const u64 owner_completed_ns = completion_timestamps[shard];
      const u64 completed_ns =
        owner_completed_ns == 0 ? global_time_ns() : owner_completed_ns;
      atomicAdd(reinterpret_cast<unsigned long long*>(
                  speculative_completion_latency_ns),
                static_cast<unsigned long long>(
                  completed_ns - batch.issue_timestamp_ns[shard]));
      if (speculative_completion_groups != nullptr) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(speculative_completion_groups),
          1ULL);
      }
      batch.issue_timestamp_ns[shard] = 0;
    }
  }
  __syncthreads();
  if (!wait_for_completion && speculative_finish_pending != 0) {
    return batch.fatal == 0;
  }

  const size_t request_base = static_cast<size_t>(descriptor.query_slot) *
                                kPersistentFrontierRobCapacity +
                              slot_begin;
  const u32* request_shards =
    params.speculative_graph_request_shards + request_base;
  for (u32 local = threadIdx.x; local < slot_count; local += blockDim.x) {
    const u32 slot = slot_begin + local;
    FrontierRobEntry& entry = rob[slot];
    if (entry.state != static_cast<u8>(FrontierRequestState::inflight)) {
      continue;
    }
    const u32 shard = request_shards[local];
    if (shard == UINT32_MAX || completion_statuses[shard] != 0) {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation =
        static_cast<u8>(FrontierValidationState::transport_rejected);
      if (speculative_stale != nullptr) {
        atomicAdd(speculative_stale, 1u);
      }
      if (classified_stale != nullptr) {
        atomicAdd(classified_stale, 1u);
      }
      if (speculative_wasted_bytes != nullptr && entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(speculative_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      if (classified_wasted_bytes != nullptr && entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(classified_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      continue;
    }
    entry.state = static_cast<u8>(FrontierRequestState::arrived);
    if (speculative_arrived != nullptr) {
      atomicAdd(speculative_arrived, 1u);
    }
    if (classified_arrived != nullptr) {
      atomicAdd(classified_arrived, 1u);
    }
    const FrontierValidationState validation = validate_frontier_record_local(
      params, descriptor.query_slot, entry, dynamic_telemetry);
    if (validation == FrontierValidationState::valid) {
      entry.state = static_cast<u8>(FrontierRequestState::validated);
      entry.validation = static_cast<u8>(FrontierValidationState::valid);
    } else {
      entry.state = static_cast<u8>(FrontierRequestState::stale);
      entry.validation = static_cast<u8>(validation);
      if (speculative_stale != nullptr) {
        atomicAdd(speculative_stale, 1u);
      }
      if (classified_stale != nullptr) {
        atomicAdd(classified_stale, 1u);
      }
      if (speculative_wasted_bytes != nullptr && entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(speculative_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
      if (classified_wasted_bytes != nullptr && entry.transfer_bytes != 0) {
        atomicAdd(
          reinterpret_cast<unsigned long long*>(classified_wasted_bytes),
          static_cast<unsigned long long>(entry.transfer_bytes));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && wait_for_completion &&
      classified_ready_waves != nullptr && speculative_finish_pending == 0) {
    ++*classified_ready_waves;
  }
  if (threadIdx.x == 0) {
    *speculative_wait_cycles += clock64() - speculative_wait_started_cycles;
    batch.active = 0;
    batch.fatal |= speculative_finish_failed;
  }
  __syncthreads();
  return batch.fatal == 0;
}

// Returns zero on success. A nonzero value is a failure detail consumed only
// by the query's fail-stop diagnostic. Preparation failures encode a reason
// in bits [3:0], the parent index in [8:4], and the destination scratch slot
// in [14:9]. Reason 1 is an unresolved handle, 2 is a stop request, 3 is a
// missing scratch arena, 4 is an out-of-range destination, and 6 is an
// otherwise impossible preparation rejection. Snapshot validation is reason
// 5. Bit 16 plus errno magnitude denotes an owner/direct completion failure.
__device__ u32 fetch_graph_records_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  const u64* handles, u32 count, u32* acquired_slots,
  const u32* destination_scratch_slots, u32* remote_reads, u32* remote_batches,
  u32* graph_read_retries, u64* graph_read_bytes, u32* graph_live_extent_reads,
  u32* graph_full_record_reads, u32* graph_extent_fallback_reads,
  u32* graph_extent_underhint_reads, u32* graph_extent_hint_promotions,
  u32 route_attempt, u32 search_round, bool trace_enabled,
  GraphFetchCycleBreakdown* cycle_breakdown,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr,
  const u8* force_full_underhint = nullptr) {
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u64 shard_issue_timestamp_ns[kPersistentMaxShards];
  __shared__ u64 stage_started_cycles;
  __shared__ u32 trace_event_start;
  __shared__ u32 trace_batch_count;
  __shared__ u32 failed;
  __shared__ u32 retry_pending;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iovas =
    params.dynamic_code_request_local_iovas + request_base;
  u32* request_bytes =
    params.graph_request_bytes == nullptr
      ? nullptr
      : params.graph_request_bytes +
          static_cast<size_t>(descriptor.query_slot) * kPersistentMaxPrefetch;
  const bool live_extent_enabled = request_bytes != nullptr;
  const bool header_neighbor =
    params.graph_read_policy ==
    static_cast<u32>(GraphReadPolicy::header_neighbor);

  if (threadIdx.x == 0) failed = 0;
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    acquired_slots[index] = UINT32_MAX;
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iovas[index] = 0;
    if (request_bytes != nullptr) {
      request_bytes[index] = params.graph_entry_bytes;
    }
    remote_reads[index] = 0;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    shard_status[shard] = 0;
    shard_issue_timestamp_ns[shard] = 0;
  }
  __syncthreads();

  constexpr u32 warp_width = 32;
  // `remote_reads` is query-local scratch. Bit zero preserves its original
  // logical-read counter contract; the remaining bits carry retry provenance
  // across physical attempts without publishing an unverified short header.
  const u32 warp = threadIdx.x / warp_width;
  const u32 lane_in_warp = threadIdx.x % warp_width;
  const u32 warp_count = max(1u, blockDim.x / warp_width);
  if (lane_in_warp == 0) {
    for (u32 index = warp; index < count; index += warp_count) {
      u32 transfer_bytes = params.graph_entry_bytes;
      const u32 scratch_slot = destination_scratch_slots == nullptr
                                 ? index
                                 : destination_scratch_slots[index];
      if (!prepare_graph_record_in_scratch(
            params, handles[index], descriptor.query_slot, scratch_slot,
            acquired_slots[index], request_shards[index],
            request_offsets[index], request_local_iovas[index],
            transfer_bytes)) {
        // This branch is fail-stop only, so preserve the exact rejected
        // invariant without adding address resolution to successful queries.
        u32 reason = handles[index] == kInvalidDeviceHandle ? 7u
                     : handles[index] == 0                  ? 8u
                                                            : 1u;
        if (scratch_slot >= kPersistentGraphScratchSlots) {
          reason = 4u;
        } else if (params.graph_scratch == nullptr) {
          reason = 3u;
        } else if (*reinterpret_cast<volatile u32*>(params.stop) != 0) {
          reason = 2u;
        } else {
          u64 raw = 0;
          u64 graph_offset = 0;
          u32 shard = 0;
          if (resolve_handle(params, handles[index], raw, shard,
                             graph_offset)) {
            reason = 6u;
          }
        }
        const u32 detail = reason | ((index & 0x1fu) << 4) |
                           ((min(scratch_slot, 0x3fu) & 0x3fu) << 9);
        atomicCAS(&failed, 0u, detail);
      } else {
        const bool continue_header_neighbor = header_neighbor &&
                                              force_full_underhint != nullptr &&
                                              force_full_underhint[index] != 0;
        bool continue_header_neighbor_body = false;
        if (continue_header_neighbor) {
          const u8* prior_header = graph_record_pointer(
            params, descriptor.query_slot, acquired_slots[index]);
          u32 required_bytes = 0;
          if (graph_record_validation::required_live_extent_bytes(
                prior_header, graph_record_validation::kGraphRecordHeaderBytes,
                params.graph_degree, params.graph_entry_capacity,
                required_bytes) &&
              required_bytes >
                graph_record_validation::kGraphRecordHeaderBytes) {
            request_offsets[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            transfer_bytes =
              required_bytes - graph_record_validation::kGraphRecordHeaderBytes;
            continue_header_neighbor_body = true;
          }
        }
        const bool force_full = !header_neighbor &&
                                force_full_underhint != nullptr &&
                                force_full_underhint[index] != 0;
        remote_reads[index] = prepare_graph_read_attempt_state(
          params.graph_entry_bytes, force_full, transfer_bytes);
        if (continue_header_neighbor) {
          // The speculative header and this exact-prefix transfer form one
          // logical adjacency access.  Keep physical WQE/byte accounting for
          // both stages but do not count a second logical graph read.
          remote_reads[index] &= ~kGraphReadLogical;
          if (continue_header_neighbor_body) {
            remote_reads[index] |= kGraphReadHeaderNeighborBody;
          }
        }
        if (request_bytes != nullptr) {
          request_bytes[index] = transfer_bytes;
        }
      }
    }
  }
  __syncthreads();
  if (failed != 0) {
    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      acquired_slots[index] = UINT32_MAX;
    }
    __syncthreads();
    return failed;
  }

  // A compact graph entry is updated in-place by stage2/reverse-edge workers.
  // Its checksum is therefore an optimistic snapshot validator: a successful
  // RDMA completion can still overlap a legal publication and contain a torn
  // mix of the old and new entry. Storage CPU readers already retry this same
  // condition. Re-read only invalid entries and reserve fail-stop for a record
  // that remains invalid after the bounded snapshot attempts.
  constexpr u32 kFullGraphSnapshotAttempts = 3;
  // Live-Extent needs at most one optimistic prefix plus the legacy three
  // full attempts. Header->Neighbor normally completes in two partial reads;
  // if they cross a mutation, preserve all three authoritative full attempts
  // after the two-read optimistic sequence instead of alternating forever.
  constexpr u32 kHeaderNeighborOptimisticAttempts = 2;
  const u32 maximum_batch_attempts =
    header_neighbor
      ? kFullGraphSnapshotAttempts + kHeaderNeighborOptimisticAttempts
      : live_extent_enabled ? kFullGraphSnapshotAttempts + 1u
                            : kFullGraphSnapshotAttempts;
  for (u32 attempt = 0; attempt < maximum_batch_attempts; ++attempt) {
    if (threadIdx.x == 0) retry_pending = 0;
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      shard_status[shard] = 0;
      shard_issue_timestamp_ns[shard] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      stage_started_cycles = clock64();
      trace_event_start = 0;
      trace_batch_count = 0;
      if (trace_enabled && params.query_rdma_trace_headers != nullptr) {
        for (u32 shard = 0; shard < params.num_shards; ++shard) {
          for (u32 index = 0; index < count; ++index) {
            if (request_shards[index] == shard) {
              ++trace_batch_count;
              break;
            }
          }
        }
        QueryRdmaTraceHeader& header =
          params.query_rdma_trace_headers[descriptor.query_slot];
        trace_event_start = header.event_count;
        header.event_count += trace_batch_count;
        if (header.event_count > params.query_rdma_trace_events_per_query) {
          header.overflow = 1;
        }
      }
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      u32 matching = 0;
      u32 payload_bytes = 0;
      u32 minimum_bytes = UINT32_MAX;
      u32 maximum_bytes = 0;
      u32 short_reads = 0;
      u32 full_reads = 0;
      u32 fallback_reads = 0;
      u32 underhint_reads = 0;
      u32 retry_reads = 0;
      u32 dynamic_short_reads = 0;
      u32 dynamic_full_reads = 0;
      u32 dynamic_fallback_reads = 0;
      u64 dynamic_payload_bytes = 0;
      for (u32 index = 0; index < count; ++index) {
        if (request_shards[index] != shard) continue;
        ++matching;
        const u32 transfer_bytes =
          live_extent_enabled ? request_bytes[index] : params.graph_entry_bytes;
        const bool full_record_read =
          transfer_bytes >= params.graph_entry_bytes;
        const GraphReadAdmissionAccounting admission =
          classify_graph_read_admission(remote_reads[index], full_record_read,
                                        attempt);
        retry_reads += admission.retry_reads;
        if (live_extent_enabled) {
          payload_bytes += transfer_bytes;
          minimum_bytes = min(minimum_bytes, transfer_bytes);
          maximum_bytes = max(maximum_bytes, transfer_bytes);
          if (transfer_bytes < params.graph_entry_bytes) {
            ++short_reads;
          } else {
            ++full_reads;
            fallback_reads += admission.fallback_reads;
            underhint_reads += admission.underhint_reads;
          }
        }
        if (dynamic_graph_telemetry_handle(handles[index])) {
          dynamic_payload_bytes += transfer_bytes;
          if (transfer_bytes < params.graph_entry_bytes) {
            ++dynamic_short_reads;
          } else {
            ++dynamic_full_reads;
            dynamic_fallback_reads += admission.fallback_reads;
          }
        }
      }
      if (!live_extent_enabled && matching != 0) {
        payload_bytes = matching * params.graph_entry_bytes;
        minimum_bytes = params.graph_entry_bytes;
        maximum_bytes = params.graph_entry_bytes;
        full_reads = matching;
      }
      if (matching != 0) {
        if (trace_enabled && params.query_rdma_trace_events != nullptr) {
          u32 ordinal = 0;
          for (u32 prior = 0; prior < shard; ++prior) {
            for (u32 index = 0; index < count; ++index) {
              if (request_shards[index] == prior) {
                ++ordinal;
                break;
              }
            }
          }
          const u32 event_index = trace_event_start + ordinal;
          if (event_index < params.query_rdma_trace_events_per_query) {
            params.query_rdma_trace_events
              [static_cast<size_t>(descriptor.query_slot) *
                 params.query_rdma_trace_events_per_query +
               event_index] = {
              .request_id = descriptor.request_id,
              .issue_timestamp_ns = global_time_ns(),
              .route_attempt = route_attempt,
              .search_round = search_round,
              .snapshot_attempt = attempt,
              .target_shard = shard,
              .parent_count = matching,
              .payload_bytes = payload_bytes,
              .minimum_bytes_per_parent = minimum_bytes,
              .maximum_bytes_per_parent = maximum_bytes,
            };
          }
        }
      }
      i32* owner_completion =
        params.direct_batch_statuses == nullptr
          ? nullptr
          : params.direct_batch_statuses +
              static_cast<size_t>(descriptor.query_slot) * params.num_shards +
              shard;
      u64* owner_completion_timestamp_ns =
        !trace_enabled ||
            params.direct_batch_completion_timestamps_ns == nullptr
          ? nullptr
          : params.direct_batch_completion_timestamps_ns +
              static_cast<size_t>(descriptor.query_slot) * params.num_shards +
              shard;
      shard_issue_timestamp_ns[shard] = global_time_ns();
      shard_status[shard] = direct_fetch_batch(
        params, shard, request_shards, request_offsets, count,
        params.graph_scratch, kPersistentGraphReadBytes,
        params.graph_entry_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iovas, owner_completion, true, nullptr,
        owner_completion_timestamp_ns,
        live_extent_enabled ? request_bytes : nullptr);
      // Account only work that completed synchronously or was successfully
      // admitted to an owner queue. Parameter/enqueue/transport failures that
      // return before admission did not issue the advertised graph payload.
      const bool admitted =
        shard_status[shard] == 0 || shard_status[shard] == -EINPROGRESS;
      if (matching != 0 && admitted) {
        atomicAdd(remote_batches, 1u);
        if (graph_read_bytes != nullptr) {
          atomicAdd(reinterpret_cast<unsigned long long*>(graph_read_bytes),
                    static_cast<unsigned long long>(payload_bytes));
        }
        if (graph_live_extent_reads != nullptr && short_reads != 0) {
          atomicAdd(graph_live_extent_reads, short_reads);
        }
        if (graph_full_record_reads != nullptr && full_reads != 0) {
          atomicAdd(graph_full_record_reads, full_reads);
        }
        if (graph_extent_fallback_reads != nullptr && fallback_reads != 0) {
          atomicAdd(graph_extent_fallback_reads, fallback_reads);
        }
        if (graph_extent_underhint_reads != nullptr && underhint_reads != 0) {
          atomicAdd(graph_extent_underhint_reads, underhint_reads);
        }
        add_dynamic_graph_read_telemetry(
          dynamic_telemetry, dynamic_short_reads, dynamic_full_reads,
          dynamic_payload_bytes, dynamic_fallback_reads);
        // Count one fallback only when its first full WQE is admitted. Further
        // full snapshot retries remain graph_read_retries but are not another
        // stale-hint fallback.
        if (fallback_reads != 0) {
          for (u32 index = 0; index < count; ++index) {
            if (request_shards[index] == shard) {
              remote_reads[index] =
                graph_read_state_after_fallback_admission(remote_reads[index]);
            }
          }
        }
        if (graph_read_retries != nullptr) {
          // A query-local force-full is attempt zero only inside this helper;
          // physically it follows the already-admitted async short and must
          // retain the original short->full retry count. Later checksum full
          // attempts continue to contribute once each.
          if (retry_reads != 0) {
            atomicAdd(graph_read_retries, retry_reads);
          }
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (trace_enabled && params.query_rdma_trace_events != nullptr) {
        const u64 wait_phase_start_ns = global_time_ns();
        const u32 available =
          trace_event_start >= params.query_rdma_trace_events_per_query
            ? 0
            : min(trace_batch_count,
                  params.query_rdma_trace_events_per_query - trace_event_start);
        for (u32 ordinal = 0; ordinal < available; ++ordinal) {
          QueryRdmaTraceEvent& event =
            params.query_rdma_trace_events
              [static_cast<size_t>(descriptor.query_slot) *
                 params.query_rdma_trace_events_per_query +
               trace_event_start + ordinal];
          event.wait_phase_start_timestamp_ns = wait_phase_start_ns;
        }
      }
      if (cycle_breakdown != nullptr) {
        cycle_breakdown->issue += clock64() - stage_started_cycles;
        stage_started_cycles = clock64();
      }
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != -EINPROGRESS) continue;
      i32* owner_completion =
        params.direct_batch_statuses +
        static_cast<size_t>(descriptor.query_slot) * params.num_shards + shard;
      shard_status[shard] = wait_direct_batch(params, owner_completion);
    }
    __syncthreads();
    for (u32 shard = threadIdx.x; shard < params.num_shards;
         shard += blockDim.x) {
      if (shard_status[shard] != 0 || shard_issue_timestamp_ns[shard] == 0 ||
          cycle_breakdown == nullptr) {
        continue;
      }
      const u64 completion_timestamp_ns = global_time_ns();
      atomicAdd(reinterpret_cast<unsigned long long*>(
                  &cycle_breakdown->completion_latency_ns),
                static_cast<unsigned long long>(
                  completion_timestamp_ns - shard_issue_timestamp_ns[shard]));
      atomicAdd(reinterpret_cast<unsigned long long*>(
                  &cycle_breakdown->completion_groups),
                1ULL);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (cycle_breakdown != nullptr) {
        cycle_breakdown->wait += clock64() - stage_started_cycles;
      }
      const u64 process_start_ns = trace_enabled ? global_time_ns() : 0;
      if (trace_enabled && params.query_rdma_trace_headers != nullptr &&
          params.query_rdma_trace_events != nullptr) {
        const u32 available =
          trace_event_start >= params.query_rdma_trace_events_per_query
            ? 0
            : min(trace_batch_count,
                  params.query_rdma_trace_events_per_query - trace_event_start);
        for (u32 ordinal = 0; ordinal < available; ++ordinal) {
          QueryRdmaTraceEvent& event =
            params.query_rdma_trace_events
              [static_cast<size_t>(descriptor.query_slot) *
                 params.query_rdma_trace_events_per_query +
               trace_event_start + ordinal];
          event.completion_timestamp_ns =
            params.direct_batch_completion_timestamps_ns
              [static_cast<size_t>(descriptor.query_slot) * params.num_shards +
               event.target_shard];
          event.batch_process_start_timestamp_ns = process_start_ns;
        }
      }
      stage_started_cycles = clock64();
    }
    __syncthreads();

    for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
      const u32 shard = request_shards[index];
      if (shard == UINT32_MAX) continue;
      const u32 slot = acquired_slots[index];
      u8* record = graph_record_pointer(params, descriptor.query_slot, slot);
      const i32 status = shard_status[shard];
      const u32 physical_transfer_bytes =
        live_extent_enabled ? request_bytes[index] : params.graph_entry_bytes;
      const bool header_neighbor_body =
        header_neighbor &&
        (remote_reads[index] & kGraphReadHeaderNeighborBody) != 0;
      const u32 available_bytes =
        physical_transfer_bytes +
        (header_neighbor_body ? graph_record_validation::kGraphRecordHeaderBytes
                              : 0u);
      const bool partial_read = available_bytes < params.graph_entry_bytes;
      u32 required_bytes = 0;
      const bool prefix_valid =
        status == 0 && graph_record_validation::required_live_extent_bytes(
                         record, available_bytes, params.graph_degree,
                         params.graph_entry_capacity, required_bytes);
      // Capacity is checked before checksum acceptance. This prevents a
      // truncated counted prefix from being accepted even in the event of a
      // checksum collision. Any other invalid reconstructed short read also
      // upgrades to the authoritative full record rather than repeating the
      // same insufficient request.
      const bool short_read_requires_full =
        status == 0 && partial_read &&
        (!prefix_valid || required_bytes > available_bytes);
      const bool extent_underhint = status == 0 && partial_read &&
                                    prefix_valid &&
                                    required_bytes > available_bytes;
      const graph_record_validation::SnapshotState snapshot =
        status == 0 && !short_read_requires_full
          ? (partial_read
               ? classify_short_graph_record(params, record, available_bytes,
                                             handles[index])
               : classify_graph_record(params, record, handles[index]))
          : graph_record_validation::SnapshotState::invalid;
      const bool started_with_short =
        (remote_reads[index] & kGraphReadStartedWithShortExtent) != 0;
      const bool header_neighbor_full_fallback =
        header_neighbor &&
        (remote_reads[index] & kGraphReadHeaderNeighborFullFallback) != 0;
      // A logical bit cleared on entry means the header was already admitted
      // by the caller's asynchronous prefix path. In that case only the body
      // attempt belongs to this batch; the ordinary path owns both partial
      // attempts here.
      const u32 optimistic_partial_attempts =
        header_neighbor_full_fallback
          ? (remote_reads[index] & kGraphReadLogical) != 0
              ? kHeaderNeighborOptimisticAttempts
              : 1u
          : started_with_short ? 1u : 0u;
      const bool attempts_remain =
        graph_record_validation::snapshot_retry_available(
          attempt, optimistic_partial_attempts, partial_read,
          maximum_batch_attempts, kFullGraphSnapshotAttempts);
      const graph_record_validation::ReadAction action =
        short_read_requires_full
          ? (attempts_remain ? graph_record_validation::ReadAction::retry
                             : graph_record_validation::ReadAction::fail)
          : graph_record_validation::decide_read_action(status == 0, snapshot,
                                                        attempts_remain);
      if (action == graph_record_validation::ReadAction::accept) {
        account_full_record_avoidance_hit(
          params, available_bytes, snapshot, dynamic_telemetry);
        const GraphExtentAdaptation adaptation =
          adapt_graph_extent_class(params, handles[index], required_bytes);
        if (adaptation == GraphExtentAdaptation::promoted) {
          if (graph_extent_hint_promotions != nullptr) {
            atomicAdd(graph_extent_hint_promotions, 1u);
          }
          if (dynamic_telemetry != nullptr &&
              dynamic_graph_telemetry_handle(handles[index])) {
            atomicAdd(&dynamic_telemetry->hint_promotions, 1u);
          }
        } else if (adaptation == GraphExtentAdaptation::demoted &&
                   dynamic_telemetry != nullptr) {
          atomicAdd(&dynamic_telemetry->hint_demotions, 1u);
        }
        // UINT32_MAX removes this entry from subsequent per-shard retry
        // batches while leaving acquired_slots intact for traversal.
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::discard_stale) {
        // Slot reuse is an ordinary read-committed race.  The tagged handle no
        // longer names this record, so drop only this expansion; disabling the
        // direct path would turn normal update churn into a service failure.
        acquired_slots[index] = UINT32_MAX;
        request_shards[index] = UINT32_MAX;
        continue;
      }
      if (action == graph_record_validation::ReadAction::retry) {
        if (header_neighbor) {
          if (header_neighbor_body) {
            request_offsets[index] -=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] -=
              graph_record_validation::kGraphRecordHeaderBytes;
            remote_reads[index] =
              graph_read_state_after_header_neighbor_conflict(
                remote_reads[index]);
            request_bytes[index] = params.graph_entry_bytes;
          } else if (header_neighbor_full_fallback) {
            request_bytes[index] = params.graph_entry_bytes;
          } else if (extent_underhint) {
            request_offsets[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_local_iovas[index] +=
              graph_record_validation::kGraphRecordHeaderBytes;
            request_bytes[index] =
              required_bytes - graph_record_validation::kGraphRecordHeaderBytes;
            remote_reads[index] |= kGraphReadHeaderNeighborBody;
          } else {
            request_bytes[index] =
              graph_record_validation::kGraphRecordHeaderBytes;
          }
        } else if (partial_read) {
          request_bytes[index] = params.graph_entry_bytes;
          remote_reads[index] |= kGraphReadNeedsExtentFallback;
          if (extent_underhint) {
            remote_reads[index] |= kGraphReadExtentUnderhint;
          }
        }
        atomicAdd(&retry_pending, 1u);
        continue;
      }

      acquired_slots[index] = UINT32_MAX;
      request_shards[index] = UINT32_MAX;
      const u32 status_magnitude =
        status < 0 ? min(static_cast<u32>(-(status + 1)) + 1u, 0xffffu)
                   : min(static_cast<u32>(status), 0xffffu);
      const u32 previous_failure = atomicCAS(&failed, 0u,
                status == 0 ? 5u : (u32{1} << 16) | status_magnitude);
      if (status == 0) {
        if (previous_failure == 0) {
          printf("[graph-snapshot-debug] handle=%llu version=%u stored_checksum=%u actual_checksum=%u stable=%u provisional=%u generation=%u incarnation=%u\n",
                 static_cast<unsigned long long>(handles[index]),
                 graph_record_validation::load_u32(record + 12),
                 static_cast<u32>(record[2] | (record[3] << 8)),
                 static_cast<u32>(graph_record_validation::checksum16(
                   record, params.graph_entry_bytes)),
                 static_cast<u32>(record[0]),
                 static_cast<u32>((record[1] >> 4) & 0xfu),
                 graph_record_validation::load_u32(record + 4),
                 graph_record_validation::load_u32(record + 8));
        }
        if (params.direct_error != nullptr) {
          atomicCAS(params.direct_error, 0, -EBADMSG);
        }
        atomicExch(params.direct_disabled, 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
      cycle_breakdown->validation += clock64() - stage_started_cycles;
    }
    if (retry_pending == 0) break;
    if (threadIdx.x == 0) device_ring_relax(128);
    __syncthreads();
  }
  return failed;
}

}  // namespace gpu_search::persistent_kernel_detail

#pragma once

#include "gpu_search/persistent_kernel/io/direct_transport.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ bool exact_record_visible(const PersistentKernelParams& params,
                                     const u8* record, u64 handle) {
  const u64 before = *reinterpret_cast<const u64*>(record);
  const u64 after = *reinterpret_cast<const u64*>(
    record + exact_record_trailer_offset(params.node_record_bytes));
  const u32 expected_incarnation = remote_incarnation(handle);
  const u32 stored_incarnation =
    *reinterpret_cast<const u32*>(record + params.node_incarnation_offset);
  return before == after &&
         (before & (kNodeLockMask | kNodeDeletedMask)) == 0 &&
         static_cast<u32>(before >> kNodeHeaderIncarnationShift) ==
           expected_incarnation &&
         stored_incarnation == expected_incarnation;
}

struct ExactRerankDiagnostics {
  u32 resolved{};
  u32 equal_headers{};
  u32 visible{};
  u64 first_handle{kInvalidDeviceHandle};
  u64 first_before{};
  u64 first_after{};
  u32 first_stored_incarnation{};
  u32 first_expected_incarnation{};
};

__device__ ExactRerankDiagnostics diagnose_exact_rerank(
    const PersistentKernelParams& params, const QueryDescriptor& descriptor,
    const u64* rerank_handles, u32 rerank_count) {
  ExactRerankDiagnostics diagnostics;
  for (u32 index = 0; index < rerank_count; ++index) {
    const u64 handle = rerank_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    const bool resolved =
      resolve_handle(params, handle, raw, shard, graph_offset);
    diagnostics.resolved += resolved ? 1u : 0u;
    const u8* record = params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width + index) *
        params.node_record_stride;
    const u64 before = *reinterpret_cast<const u64*>(record);
    const u64 after = *reinterpret_cast<const u64*>(
      record + exact_record_trailer_offset(params.node_record_bytes));
    diagnostics.equal_headers += before == after ? 1u : 0u;
    const bool visible =
      resolved && exact_record_visible(params, record, handle);
    diagnostics.visible += visible ? 1u : 0u;
    if (!visible && diagnostics.first_handle == kInvalidDeviceHandle) {
      diagnostics.first_handle = handle;
      diagnostics.first_before = before;
      diagnostics.first_after = after;
      diagnostics.first_stored_incarnation =
        *reinterpret_cast<const u32*>(
          record + params.node_incarnation_offset);
      diagnostics.first_expected_incarnation = remote_incarnation(handle);
    }
  }
  return diagnostics;
}

// A terminal-horizon exact prefetch has a one-round lifetime: it starts only
// after the next critical graph wave has been published and is consumed after
// that final authoritative commit.  Keep only scalar control in shared
// memory.  The request SoA, per-shard completion words, and cached records
// live in the otherwise-unused second half of this query slot's exact-record
// region, so the optimization adds neither a device allocation nor a
// query-slot-sized shared array.
struct TerminalExactCacheState {
  u32 attempted{};
  u32 active{};
  // Set before any cache record can be handed to the ordinary miss path.
  // That path reuses exact_records[0, K), so no later exact attempt may match
  // the retained metadata against those overwritten payload slots.
  u32 consumed{};
  u32 candidate_count{};
  u32 issued_records{};
  u32 arrived_records{};
  u32 promoted_records{};
  u32 queue_rejects{};
  u32 miss_count{};
  u64 wasted_bytes{};
};

struct TerminalExactCacheScratch {
  u32* request_shards{};
  u64* remote_offsets{};
  u64* local_iova_offsets{};
  i32* shard_status{};
  u8* record_base{};
  u8* metadata_base{};
  size_t metadata_bytes{};
  bool valid{};
};

__device__ __forceinline__ TerminalExactCacheScratch
terminal_exact_cache_scratch(const PersistentKernelParams& params,
                             const QueryDescriptor& descriptor) {
  TerminalExactCacheScratch scratch{};
  if (params.exact_records == nullptr || params.node_record_stride == 0 ||
      params.exact_width < 2u * kPersistentMaxBeam) {
    return scratch;
  }
  scratch.record_base =
    params.exact_records + static_cast<size_t>(descriptor.query_slot) *
                             params.exact_width * params.node_record_stride;
  scratch.metadata_base =
    scratch.record_base +
    static_cast<size_t>(kPersistentMaxBeam) * params.node_record_stride;
  scratch.metadata_bytes =
    static_cast<size_t>(params.exact_width - kPersistentMaxBeam) *
    params.node_record_stride;

  uintptr_t cursor = reinterpret_cast<uintptr_t>(scratch.metadata_base);
  const uintptr_t end = cursor + scratch.metadata_bytes;
  constexpr u32 terminal_snapshot_requests = 2u * kPersistentMaxBeam;
  scratch.request_shards = reinterpret_cast<u32*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u32);
  cursor =
    (cursor + alignof(u64) - 1u) & ~static_cast<uintptr_t>(alignof(u64) - 1u);
  scratch.remote_offsets = reinterpret_cast<u64*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u64);
  scratch.local_iova_offsets = reinterpret_cast<u64*>(cursor);
  cursor += static_cast<uintptr_t>(terminal_snapshot_requests) * sizeof(u64);
  cursor =
    (cursor + alignof(i32) - 1u) & ~static_cast<uintptr_t>(alignof(i32) - 1u);
  scratch.shard_status = reinterpret_cast<i32*>(cursor);
  cursor += static_cast<uintptr_t>(kPersistentMaxShards) * sizeof(i32);
  scratch.valid = cursor <= end;
  return scratch;
}

// A fixed-record payload is immutable after one physical incarnation is
// published: upsert allocates a different incarnation rather than changing
// id/vector bytes in place. The terminal wave is a fenced
// [full record -> current header] train, so equality of its two headers and
// the stored incarnation is a complete exact snapshot obtained during this
// query. A later update linearizes after that snapshot and cannot mutate the
// retained physical incarnation.
__device__ __forceinline__ bool terminal_exact_cache_payload_valid(
  const PersistentKernelParams& params, const u8* record, u64 handle) {
  const u64 header = *reinterpret_cast<const u64*>(record);
  const u64 trailer = *reinterpret_cast<const u64*>(
    record + exact_record_trailer_offset(params.node_record_bytes));
  const u32 expected_incarnation = remote_incarnation(handle);
  const u32 stored_incarnation =
    *reinterpret_cast<const u32*>(record + params.node_incarnation_offset);
  return header == trailer &&
         (header & (kNodeLockMask | kNodeDeletedMask)) == 0 &&
         static_cast<u32>(header >> kNodeHeaderIncarnationShift) ==
           expected_incarnation &&
         stored_incarnation == expected_incarnation;
}

// Publish one optional fenced snapshot wave without waiting. This descriptor is
// deliberately enqueued only after the already-certified next graph core, so
// it cannot overtake the query's dependency. The query-side producer applies
// the same bounded critical backpressure as ordinary exact snapshots; any
// transport failure still degrades to the established exact fallback.
__device__ void begin_terminal_exact_cache_prefetch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  const u64* beam_handles, u32 beam_count, TerminalExactCacheState& state) {
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (threadIdx.x == 0) {
    state.attempted = 1;
    state.active = 0;
    state.consumed = 0;
    state.candidate_count =
      min(beam_count, static_cast<u32>(kPersistentMaxBeam));
    state.issued_records = 0;
    state.arrived_records = 0;
    state.promoted_records = 0;
    state.queue_rejects = 0;
    state.miss_count = 0;
    state.wasted_bytes = 0;
  }
  __syncthreads();
  if (!scratch.valid || state.candidate_count == 0 ||
      params.direct_batch_queues == nullptr) {
    return;
  }

  for (u32 index = threadIdx.x; index < 2u * state.candidate_count;
       index += blockDim.x) {
    scratch.request_shards[index] = UINT32_MAX;
    scratch.remote_offsets[index] = 0;
    scratch.local_iova_offsets[index] =
      index < state.candidate_count
        ? 0u
        : exact_record_trailer_offset(params.node_record_bytes);
  }
  for (u32 index = threadIdx.x; index < state.candidate_count;
       index += blockDim.x) {
    const u64 handle = beam_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
        shard >= params.num_shards) {
      continue;
    }
    scratch.request_shards[index] = shard;
    scratch.remote_offsets[index] =
      remote_byte_offset(raw) + params.node_meta_offset;
    u8* destination = scratch.record_base +
                      static_cast<size_t>(index) * params.node_record_stride;
    scratch.local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
    const u32 trailer = state.candidate_count + index;
    scratch.request_shards[trailer] = shard;
    scratch.remote_offsets[trailer] = scratch.remote_offsets[index];
    scratch.local_iova_offsets[trailer] =
      reinterpret_cast<u64>(
        destination + exact_record_trailer_offset(params.node_record_bytes)) -
      params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
       shard += blockDim.x) {
    scratch.shard_status[shard] = shard < params.num_shards ? 0 : -EINVAL;
  }
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    const i32 issue_status = direct_fetch_fenced_snapshot_batch(
      params, shard, scratch.request_shards, scratch.remote_offsets,
      state.candidate_count, params.node_record_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      scratch.local_iova_offsets, scratch.shard_status + shard, nullptr);
    // The owner may publish success before the enqueue helper returns.
    // Never overwrite that completion with a stale -EINPROGRESS return.
    if (issue_status != -EINPROGRESS) {
      scratch.shard_status[shard] = issue_status;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < state.candidate_count; ++index) {
      const u32 shard = scratch.request_shards[index];
      if (shard >= params.num_shards) continue;
      const i32 status =
        *reinterpret_cast<volatile i32*>(scratch.shard_status + shard);
      if (status == 0 || status == -EINPROGRESS) {
        ++state.issued_records;
      } else if (status == -EAGAIN) {
        ++state.queue_rejects;
      }
    }
    state.active = state.issued_records != 0 ? 1u : 0u;
  }
  __syncthreads();
}

// Always call this before exact scratch or the query slot can be reused.
// Cache transport failures are not query failures: the authoritative fenced
// exact read remains the correctness fallback.
__device__ void drain_terminal_exact_cache_prefetch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  TerminalExactCacheState& state) {
  if (state.active == 0) return;
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (!scratch.valid) {
    if (threadIdx.x == 0) state.active = 0;
    __syncthreads();
    return;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    const i32 status =
      *reinterpret_cast<volatile i32*>(scratch.shard_status + shard);
    if (status == -EINPROGRESS) {
      scratch.shard_status[shard] =
        wait_direct_batch(params, scratch.shard_status + shard);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state.arrived_records = 0;
    for (u32 index = 0; index < state.candidate_count; ++index) {
      const u32 shard = scratch.request_shards[index];
      if (shard < params.num_shards && scratch.shard_status[shard] == 0) {
        ++state.arrived_records;
      }
    }
    state.active = 0;
  }
  __syncthreads();
}

template <u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__device__ bool approximate_handles_batch(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  const f32* query_lut, u64* handles, u32 count, f32* distances,
  u64* total_dynamic_cycles, u32* total_dynamic_candidates,
  u32* total_dynamic_reads, u32* total_incarnation_rejects,
  u32* total_cache_hits, u32* total_batch_deduplicated,
  u32* total_cache_publish_successes, u32* total_cache_first_occupancies,
  u32* total_cache_publish_races, u32* total_lookup_probe_exhaustions,
  u32* total_publish_probe_exhaustions, u32* total_lookup_probes,
  u32* max_lookup_probes) {
  const u32 pq_code_bytes =
    resolved_pq_subquantizers<PqSubquantizers>(params);
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 failed;
  __shared__ u32 call_dynamic_candidates;
  __shared__ u32 call_dynamic_reads;
  __shared__ u32 call_incarnation_rejects;
  __shared__ u32 call_cache_hits;
  __shared__ u32 call_batch_deduplicated;
  __shared__ u32 call_cache_publish_successes;
  __shared__ u32 call_cache_first_occupancies;
  __shared__ u32 call_cache_publish_races;
  __shared__ u32 call_lookup_probe_exhaustions;
  __shared__ u32 call_publish_probe_exhaustions;
  __shared__ u32 call_lookup_probes;
  __shared__ u32 call_max_lookup_probes;
  __shared__ u64 call_rdma_started_cycles;
  __shared__ u64 call_rdma_cycles;
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  if (threadIdx.x == 0) {
    failed = 0;
    call_dynamic_candidates = 0;
    call_dynamic_reads = 0;
    call_incarnation_rejects = 0;
    call_cache_hits = 0;
    call_batch_deduplicated = 0;
    call_cache_publish_successes = 0;
    call_cache_first_occupancies = 0;
    call_cache_publish_races = 0;
    call_lookup_probe_exhaustions = 0;
    call_publish_probe_exhaustions = 0;
    call_lookup_probes = 0;
    call_max_lookup_probes = 0;
    call_rdma_started_cycles = 0;
    call_rdma_cycles = 0;
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u64 handle = handles[index];
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_local_iova_offsets[index] = 0;
    distances[index] = FLT_MAX;
    if (handle == kInvalidDeviceHandle) continue;
    u32 static_ordinal = 0;
    if (static_ordinal_from_raw(params, handle, static_ordinal)) {
      if (static_ordinal < params.num_nodes) {
        distances[index] = approximate_entry<PqSubquantizers>(
          params, query_lut,
          params.pq_codes +
            static_cast<size_t>(static_ordinal) * pq_code_bytes);
      }
      continue;
    }

    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    // Dynamic PQ is authoritative in the storage record. Centroid-route seeds
    // and discovered neighbors therefore use the same one-sided RDMA path.
    if (params.dynamic_code_records == nullptr || shard >= params.num_shards)
      continue;
    atomicAdd(&call_dynamic_candidates, 1u);
    // Dynamic records use the same physical-slot ordinal on storage and GPU.
    // There is no hash table, collision chain, or capacity-dependent lookup.
    const DeviceShardRegion& region = params.shards[shard];
    const u64 node_offset = remote_byte_offset(raw);
    const bool arena_available = params.dynamic_code_arena_states != nullptr &&
                                 params.dynamic_code_arena_records != nullptr;
    u64 arena_slot = 0;
    if (arena_available && dynamic_code_arena_slot_from_offset(
                             region, node_offset,
                             params.dynamic_code_arena_capacity, arena_slot)) {
      auto* state = params.dynamic_code_arena_states + arena_slot;
      const u32 observed = dynamic_arena_state_load(state);
      atomicAdd(&call_lookup_probes, 1u);
      atomicMax(&call_max_lookup_probes, 1u);
      if (dynamic_code_arena_state_matches(observed,
                                           remote_incarnation(handle))) {
        const u8* resident =
          params.dynamic_code_arena_records +
          static_cast<size_t>(arena_slot) * pq_code_bytes;
        distances[index] = approximate_entry<PqSubquantizers>(
          params, query_lut, resident);
        // Keep every payload load before the second state sample. Together
        // with the publisher's release store this is the device-scope
        // seqlock read side; legacy relaxed atomics are intentionally avoided.
        cuda::atomic_thread_fence(cuda::memory_order_acquire,
                                  cuda::thread_scope_device);
        const u32 after = dynamic_arena_state_load(state);
        if (dynamic_code_arena_read_stable(observed, after,
                                           remote_incarnation(handle))) {
          atomicAdd(&call_cache_hits, 1u);
          continue;
        }
        // A replacement reserved or republished the physical slot while its
        // payload was being scored.  Never attribute those bytes to the old
        // handle; retry through the incarnation+checksum-validated storage
        // path in this same scoring call.
        distances[index] = FLT_MAX;
        atomicAdd(&call_incarnation_rejects, 1u);
      }
      if (dynamic_code_arena_stale_handle(observed,
                                          remote_incarnation(handle))) {
        atomicAdd(&call_incarnation_rejects, 1u);
        continue;
      }
    }
    // The merge frontier can contain the same tagged handle through multiple
    // graph parents.  Deduplicate only inside this finite scoring call: one
    // stable incarnation-checked record is scattered to all identical
    // consumers below.  No value survives the call, so slot reuse needs no
    // cache invalidation protocol and ABA behavior is unchanged.
    u32 duplicate_of = UINT32_MAX;
    for (u32 prior = 0; prior < index; ++prior) {
      if (handles[prior] == handle) {
        duplicate_of = prior;
        break;
      }
    }
    if (duplicate_of != UINT32_MAX) {
      request_shards[index] = UINT32_MAX - 1u;
      request_offsets[index] = duplicate_of;
      atomicAdd(&call_batch_deduplicated, 1u);
      continue;
    }
    atomicAdd(&call_dynamic_reads, 1u);
    request_shards[index] = shard;
    request_offsets[index] =
      node_offset + params.shards[shard].dynamic_code_offset;
    u8* destination =
      params.dynamic_code_records +
      (request_base + index) * params.dynamic_code_record_stride;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  // The steady-state path is all cache hits.  Do not enter the shard fan-out,
  // CQ polling, publication, or duplicate-scatter machinery when there is no
  // authoritative read to issue.
  if (call_dynamic_reads == 0) {
    if (threadIdx.x == 0) {
      if (total_dynamic_candidates != nullptr) {
        *total_dynamic_candidates += call_dynamic_candidates;
      }
      if (total_cache_hits != nullptr) {
        *total_cache_hits += call_cache_hits;
      }
      if (total_incarnation_rejects != nullptr) {
        *total_incarnation_rejects += call_incarnation_rejects;
      }
      if (total_lookup_probe_exhaustions != nullptr) {
        *total_lookup_probe_exhaustions += call_lookup_probe_exhaustions;
      }
      if (total_lookup_probes != nullptr) {
        *total_lookup_probes += call_lookup_probes;
      }
      if (max_lookup_probes != nullptr) {
        *max_lookup_probes = max(*max_lookup_probes, call_max_lookup_probes);
      }
    }
    __syncthreads();
    return true;
  }

  if (threadIdx.x == 0) call_rdma_started_cycles = clock64();
  __syncthreads();

  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    i32* owner_completion =
      params.direct_batch_statuses == nullptr
        ? nullptr
        : params.direct_batch_statuses +
            static_cast<size_t>(descriptor.query_slot) * params.num_shards +
            shard;
    shard_status[shard] = direct_fetch_batch(
      params, shard, request_shards, request_offsets, count,
      params.dynamic_code_records +
        request_base * params.dynamic_code_record_stride,
      params.dynamic_code_record_stride, params.dynamic_code_record_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iova_offsets, owner_completion, true);
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
  if (threadIdx.x == 0) {
    call_rdma_cycles = clock64() - call_rdma_started_cycles;
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard >= params.num_shards) continue;
    if (shard_status[shard] != 0) {
      atomicExch(&failed, 1u);
      continue;
    }
    const u8* code = params.dynamic_code_records +
                     (request_base + index) * params.dynamic_code_record_stride;
    const u32 dynamic_tag = vamana::dynamic_navigation_code::load_u32_le(code);
    const u8* payload = code + sizeof(u32);
    const u8* checksum = payload + pq_code_bytes;
    const bool checksum_valid = vamana::dynamic_navigation_code::validate(
      dynamic_tag, payload, pq_code_bytes, checksum);
    // The transport request is complete, so reuse its IOVA slot to carry the
    // validation result into the single-threaded cache publication pass.
    request_local_iova_offsets[index] = checksum_valid ? 1u : 0u;
    if (dynamic_code_tag_incarnation(dynamic_tag) ==
          remote_incarnation(handles[index]) &&
        checksum_valid) {
      distances[index] = approximate_entry<PqSubquantizers>(
        params, query_lut, payload);
    } else {
      atomicAdd(&call_incarnation_rejects, 1u);
    }
  }
  __syncthreads();
  // Serialize publication within this CTA. Globally, old-incarnation ->
  // BUSY|new-incarnation reserves the physical slot and payload -> fence ->
  // new-incarnation publishes it. Incarnations only increase, so a delayed
  // reader can never replace a newer payload (the slot-reuse ABA case).
  if (threadIdx.x == 0 && params.dynamic_code_arena_states != nullptr &&
      params.dynamic_code_arena_records != nullptr) {
    for (u32 index = 0; index < count; ++index) {
      const u32 shard = request_shards[index];
      if (shard >= params.num_shards || shard_status[shard] != 0) {
        continue;
      }
      const u64 handle = handles[index];
      const u8* source =
        params.dynamic_code_records +
        (request_base + index) * params.dynamic_code_record_stride;
      const u32 source_tag =
        vamana::dynamic_navigation_code::load_u32_le(source);
      const u32 requested_incarnation = remote_incarnation(handle);
      if (!dynamic_code_arena_snapshot_cacheable(
            source_tag, requested_incarnation,
            request_local_iova_offsets[index] != 0)) {
        continue;
      }
      const u32 desired_incarnation = dynamic_code_tag_incarnation(source_tag);
      const u64 node_offset = remote_byte_offset(handle);
      const DeviceShardRegion& region = params.shards[shard];
      u64 arena_slot = 0;
      if (!dynamic_code_arena_slot_from_offset(
            region, node_offset, params.dynamic_code_arena_capacity,
            arena_slot)) {
        continue;
      }
      const u8 desired_extent = dynamic_code_arena_bootstrap_extent(
        source_tag, params.dynamic_graph_extent_bootstrap_enabled);
      const u32 desired =
        make_dynamic_code_tag(desired_incarnation, desired_extent);
      auto* state = params.dynamic_code_arena_states + arena_slot;
      u32 observed = dynamic_arena_state_load(state);
      if (dynamic_code_arena_state_matches(observed, desired_incarnation)) {
        // With tag bootstrap enabled, concurrent reads of one immutable PQ
        // payload may refine UNKNOWN or retain the larger class. With bootstrap
        // disabled, the UNKNOWN desired extent preserves any online-learned
        // class. Neither path republishes the PQ payload.
        u32 promoted = observed;
        while (dynamic_code_arena_refined_unknown_extent_state(
                 observed, desired_incarnation, desired_extent, promoted) ||
               dynamic_code_arena_promoted_extent_state(
                 observed, desired_incarnation, desired_extent, promoted)) {
          const u32 prior =
            dynamic_arena_state_compare_exchange(state, observed, promoted);
          if (prior == observed) break;
          observed = prior;
        }
        ++call_cache_publish_races;
        continue;
      }
      if (!dynamic_code_arena_can_publish(observed, desired_incarnation) ||
          dynamic_arena_state_compare_exchange(
            state, observed, kPersistentDynamicCodeArenaBusy | desired) !=
            observed) {
        ++call_cache_publish_races;
        continue;
      }
      u8* destination = params.dynamic_code_arena_records +
                        static_cast<size_t>(arena_slot) * pq_code_bytes;
      for (u32 byte = 0; byte < pq_code_bytes; ++byte) {
        destination[byte] = source[sizeof(u32) + byte];
      }
      dynamic_arena_state_publish(state, desired);
      ++call_cache_publish_successes;
      if (dynamic_code_arena_first_occupancy(observed)) {
        ++call_cache_first_occupancies;
      }
    }
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < count; index += blockDim.x) {
    if (request_shards[index] != UINT32_MAX - 1u) continue;
    const u64 source_index = request_offsets[index];
    if (source_index < index) distances[index] = distances[source_index];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (total_dynamic_cycles != nullptr) {
      *total_dynamic_cycles += call_rdma_cycles;
    }
    if (total_dynamic_candidates != nullptr) {
      *total_dynamic_candidates += call_dynamic_candidates;
    }
    if (total_dynamic_reads != nullptr) {
      *total_dynamic_reads += call_dynamic_reads;
    }
    if (total_incarnation_rejects != nullptr) {
      *total_incarnation_rejects += call_incarnation_rejects;
    }
    if (total_cache_hits != nullptr) {
      *total_cache_hits += call_cache_hits;
    }
    if (total_batch_deduplicated != nullptr) {
      *total_batch_deduplicated += call_batch_deduplicated;
    }
    if (total_cache_publish_successes != nullptr) {
      *total_cache_publish_successes += call_cache_publish_successes;
    }
    if (total_cache_first_occupancies != nullptr) {
      *total_cache_first_occupancies += call_cache_first_occupancies;
    }
    if (total_cache_publish_races != nullptr) {
      *total_cache_publish_races += call_cache_publish_races;
    }
    if (total_lookup_probe_exhaustions != nullptr) {
      *total_lookup_probe_exhaustions += call_lookup_probe_exhaustions;
    }
    if (total_publish_probe_exhaustions != nullptr) {
      *total_publish_probe_exhaustions += call_publish_probe_exhaustions;
    }
    if (total_lookup_probes != nullptr) {
      *total_lookup_probes += call_lookup_probes;
    }
    if (max_lookup_probes != nullptr) {
      *max_lookup_probes = max(*max_lookup_probes, call_max_lookup_probes);
    }
  }
  __syncthreads();
  return failed == 0;
}

__device__ bool exactify_into_beam(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  const f32* query, u64* candidate_handles, u32* candidate_ids,
  f32* candidate_distances, u32 candidate_count, u64* beam_handles,
  u32* beam_ids, f32* beam_distances, u8* beam_expanded, u32& beam_count,
  u32* exact_reads, u32* exact_snapshot_train_batches,
  u32* exact_snapshot_train_fallbacks, u32 beam_capacity, bool reset_beam,
  u64* merge_handles, u32* merge_ids, f32* merge_distances,
  u16* merge_expanded) {
  constexpr i32 kFencedSnapshotComplete = 1;
  constexpr i32 kNoSnapshotRequests = 2;
  __shared__ i32 shard_status[kPersistentMaxShards];
  __shared__ u32 transport_failed;
  if (candidate_count == 0) {
    if (reset_beam) {
      if (threadIdx.x == 0) beam_count = 0;
    } else {
      // Terminal-cache hits already carry exact distances. They still require
      // the authoritative exact ordering, but not an empty per-shard RDMA
      // protocol or a Beam->merge->Beam round trip.
      sort_candidates(beam_handles, beam_ids, beam_distances, beam_expanded,
                      beam_count);
      if (threadIdx.x == 0) {
        beam_count = min(beam_count, beam_capacity);
      }
    }
    __syncthreads();
    return true;
  }
  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    request_shards[index] = UINT32_MAX;
    request_offsets[index] = 0;
    request_shards[candidate_count + index] = UINT32_MAX;
    request_offsets[candidate_count + index] = 0;
    request_local_iova_offsets[candidate_count + index] = 0;
    candidate_ids[index] = UINT32_MAX;
    candidate_distances[index] = FLT_MAX;
    const u64 handle = candidate_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset)) continue;
    request_offsets[index] = remote_byte_offset(raw) + params.node_meta_offset;
    request_shards[index] = shard;
    const u8* destination =
      params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width +
       index) *
        params.node_record_stride;
    request_local_iova_offsets[index] =
      reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
    const u32 trailer_index = candidate_count + index;
    request_shards[trailer_index] = shard;
    request_offsets[trailer_index] = request_offsets[index];
    request_local_iova_offsets[trailer_index] =
      reinterpret_cast<u64>(
        destination + exact_record_trailer_offset(params.node_record_bytes)) -
      params.direct_local_iova_base;
  }
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    shard_status[shard] = 0;
  }
  __syncthreads();

  // One owner descriptor owns both snapshots. Every populated shard gets
  // exactly one success CQE after its fenced trailer run; request metadata
  // remains immutable until that final completion is observed.
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    bool has_request = false;
    for (u32 index = 0; index < candidate_count; ++index) {
      if (request_shards[index] == shard) {
        has_request = true;
        break;
      }
    }
    if (!has_request) {
      shard_status[shard] = kNoSnapshotRequests;
      continue;
    }
    if (exact_snapshot_train_batches != nullptr) {
      // "batches" is the attempt denominator: exactly one increment for each
      // populated shard that enters the mandatory train path. Every attempt
      // either reaches kFencedSnapshotComplete or increments fallbacks below.
      atomicAdd(exact_snapshot_train_batches, 1u);
    }
    i32* owner_completion =
      params.direct_batch_statuses == nullptr
        ? nullptr
        : params.direct_batch_statuses +
            static_cast<size_t>(descriptor.query_slot) * params.num_shards +
            shard;
    shard_status[shard] =
      params.direct_batch_queues == nullptr || owner_completion == nullptr
        ? -EAGAIN
        : direct_fetch_fenced_snapshot_batch(
            params, shard, request_shards, request_offsets, candidate_count,
            params.node_record_bytes,
            (descriptor.query_slot + shard) % params.direct_qps_per_node,
            request_local_iova_offsets, owner_completion);
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
    if (shard_status[shard] == 0) {
      shard_status[shard] = kFencedSnapshotComplete;
    }
  }
  __syncthreads();

  // A transport/capacity failure falls back per shard to the established
  // two-publication protocol. Successful trains never reread either
  // snapshot. Mark their trailer entries inactive so the common fallback
  // trailer pass is a no-op for those shards.
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard != UINT32_MAX && shard_status[shard] == kFencedSnapshotComplete) {
      request_shards[candidate_count + index] = UINT32_MAX;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] == kFencedSnapshotComplete ||
        shard_status[shard] == kNoSnapshotRequests) {
      continue;
    }
    if (exact_snapshot_train_fallbacks != nullptr) {
      atomicAdd(exact_snapshot_train_fallbacks, 1u);
    }
    i32* owner_completion =
      params.direct_batch_statuses == nullptr
        ? nullptr
        : params.direct_batch_statuses +
            static_cast<size_t>(descriptor.query_slot) * params.num_shards +
            shard;
    shard_status[shard] = direct_fetch_batch(
      params, shard, request_shards, request_offsets, candidate_count,
      params.exact_records + static_cast<size_t>(descriptor.query_slot) *
                               params.exact_width * params.node_record_stride,
      params.node_record_stride, params.node_record_bytes,
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iova_offsets, owner_completion, true);
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
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard != UINT32_MAX && shard_status[shard] != 0) {
      request_shards[candidate_count + index] = UINT32_MAX;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (shard_status[shard] != 0) continue;
    i32* owner_completion =
      params.direct_batch_statuses == nullptr
        ? nullptr
        : params.direct_batch_statuses +
            static_cast<size_t>(descriptor.query_slot) * params.num_shards +
            shard;
    shard_status[shard] = direct_fetch_batch(
      params, shard, request_shards + candidate_count,
      request_offsets + candidate_count, candidate_count,
      params.exact_records +
        static_cast<size_t>(descriptor.query_slot) * params.exact_width *
          params.node_record_stride +
        exact_record_trailer_offset(params.node_record_bytes),
      params.node_record_stride, sizeof(u64),
      (descriptor.query_slot + shard) % params.direct_qps_per_node,
      request_local_iova_offsets + candidate_count, owner_completion, true);
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
    if (shard_status[shard] == kFencedSnapshotComplete ||
        shard_status[shard] == kNoSnapshotRequests) {
      shard_status[shard] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    transport_failed = 0;
    for (u32 shard = 0; shard < params.num_shards; ++shard) {
      if (exact_snapshot_transport_failed(shard_status[shard])) {
        transport_failed = 1;
        break;
      }
    }
  }
  __syncthreads();
  // Do not rank or publish a subset when any populated shard exhausted the
  // legacy fallback. Visibility rejects are evaluated below only after every
  // transport completed and remain ordinary per-candidate filtering.
  if (transport_failed != 0) return false;

  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u32 shard = request_shards[index];
    if (shard == UINT32_MAX || shard_status[shard] != 0) continue;
    const u8* record =
      params.exact_records +
      (static_cast<size_t>(descriptor.query_slot) * params.exact_width +
       index) *
        params.node_record_stride;
    if (exact_record_visible(params, record, candidate_handles[index])) {
      candidate_ids[index] =
        *reinterpret_cast<const u32*>(record + kNodeIdOffset);
      candidate_distances[index] = exact_storage_distance(
        params, query, record + params.node_vector_offset);
    }
    atomicAdd(exact_reads, 1u);
  }
  __syncthreads();
  const u32 existing_count = reset_beam ? 0 : beam_count;
  const u32 merge_count = existing_count + candidate_count;
  if (!reset_beam && merge_count <= beam_capacity) {
    // The terminal cache is the only append caller. Its hits already occupy
    // Beam and its strict misses are disjoint candidate storage, so append the
    // misses and run the same authoritative sorter in place. This preserves
    // the exact comparator and visibility filtering while avoiding two full
    // Beam copies for the common small-miss case.
    for (u32 index = threadIdx.x; index < candidate_count;
         index += blockDim.x) {
      const u32 destination = existing_count + index;
      beam_handles[destination] = candidate_handles[index];
      beam_ids[destination] = candidate_ids[index];
      beam_distances[destination] = candidate_distances[index];
      beam_expanded[destination] = 0;
    }
    __syncthreads();
    sort_candidates(beam_handles, beam_ids, beam_distances, beam_expanded,
                    merge_count);
    if (threadIdx.x == 0) {
      u32 valid = 0;
      while (
        valid < merge_count && beam_handles[valid] != kInvalidDeviceHandle &&
        isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
        ++valid;
      }
      beam_count = min(valid, beam_capacity);
    }
    __syncthreads();
    return true;
  }
  for (u32 index = threadIdx.x; index < existing_count; index += blockDim.x) {
    merge_handles[index] = beam_handles[index];
    merge_ids[index] = beam_ids[index];
    merge_distances[index] = beam_distances[index];
    merge_expanded[index] = beam_expanded[index];
  }
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u32 destination = existing_count + index;
    merge_handles[destination] = candidate_handles[index];
    merge_ids[destination] = candidate_ids[index];
    merge_distances[destination] = candidate_distances[index];
    merge_expanded[destination] = 0;
  }
  __syncthreads();
  sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                  merge_count);
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (
      valid < merge_count && merge_handles[valid] != kInvalidDeviceHandle &&
      isfinite(merge_distances[valid]) && merge_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = min(valid, beam_capacity);
  }
  __syncthreads();
  for (u32 index = threadIdx.x; index < beam_count; index += blockDim.x) {
    beam_handles[index] = merge_handles[index];
    beam_ids[index] = merge_ids[index];
    beam_distances[index] = merge_distances[index];
    beam_expanded[index] = merge_expanded[index];
  }
  __syncthreads();
  return true;
}

// Commit a terminal-horizon cache without weakening exact visibility:
//
//   1. Match the final authoritative Beam to immutable cached incarnations.
//   2. Validate the fenced second header captured by the terminal wave.
//   3. Materialize valid hits into an exact Beam.
//   4. Pass every miss to the unchanged fenced-snapshot implementation and
//      merge it with those hits.
//
// A cache transport error, a reused slot, or an invalid early payload is just
// a miss.  A successfully observed final locked/deleted/stale header is an
// authoritative visibility rejection and must not be retried at a later
// linearization point.
__device__ bool exactify_into_beam_with_terminal_cache(
  const PersistentKernelParams& params, const QueryDescriptor& descriptor,
  const f32* query, u64* candidate_handles, u32* candidate_ids,
  f32* candidate_distances, u32 candidate_count, u64* beam_handles,
  u32* beam_ids, f32* beam_distances, u8* beam_expanded, u32& beam_count,
  u32* exact_reads, u32* exact_snapshot_train_batches,
  u32* exact_snapshot_train_fallbacks, u32 beam_capacity, bool reset_beam,
  u64* merge_handles, u32* merge_ids, f32* merge_distances,
  u16* merge_expanded, TerminalExactCacheState& cache) {
  // The production terminal call resets Beam. Keep compatibility callers on
  // the original implementation rather than creating a second append policy.
  if (cache.attempted == 0 || cache.issued_records == 0 ||
      cache.consumed != 0 || !reset_beam) {
    return exactify_into_beam(
      params, descriptor, query, candidate_handles, candidate_ids,
      candidate_distances, candidate_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, exact_reads,
      exact_snapshot_train_batches, exact_snapshot_train_fallbacks,
      beam_capacity, reset_beam, merge_handles, merge_ids, merge_distances,
      merge_expanded);
  }

  drain_terminal_exact_cache_prefetch(params, descriptor, cache);
  if (threadIdx.x == 0) cache.consumed = 1;
  __syncthreads();
  const TerminalExactCacheScratch scratch =
    terminal_exact_cache_scratch(params, descriptor);
  if (!scratch.valid || cache.arrived_records == 0) {
    return exactify_into_beam(
      params, descriptor, query, candidate_handles, candidate_ids,
      candidate_distances, candidate_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, exact_reads,
      exact_snapshot_train_batches, exact_snapshot_train_fallbacks,
      beam_capacity, reset_beam, merge_handles, merge_ids, merge_distances,
      merge_expanded);
  }

  const size_t request_base =
    static_cast<size_t>(descriptor.query_slot) * kPersistentMaxMergeCandidates;
  u32* request_shards = params.dynamic_code_request_shards + request_base;
  u64* request_offsets = params.dynamic_code_request_offsets + request_base;
  u64* request_local_iova_offsets =
    params.dynamic_code_request_local_iovas + request_base;
  constexpr u32 kTerminalMissSlotBit = 1u << 31;
  constexpr u32 kTerminalSlotMask = ~kTerminalMissSlotBit;
  constexpr u32 kTerminalCandidateMetadataBase = 3u * kPersistentMaxBeam;
  static_assert(kTerminalCandidateMetadataBase + kPersistentMaxExact <=
                kPersistentMaxMergeCandidates);
  __shared__ u32 terminal_cache_used_slots[(kPersistentMaxBeam + 31u) / 32u];
  __shared__ u32 terminal_cache_mixed_misses;

  // candidate_ids temporarily stores the matched cache slot.  The key is the
  // exact remote fixed-record address; payload/header incarnation validation
  // below prevents physical-slot ABA from becoming a false hit.
  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    candidate_ids[index] = UINT32_MAX;
    candidate_distances[index] = FLT_MAX;
    merge_expanded[index] = 2;
    request_shards[kTerminalCandidateMetadataBase + index] = UINT32_MAX;
    request_offsets[kTerminalCandidateMetadataBase + index] = 0;
    const u64 handle = candidate_handles[index];
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
        shard >= params.num_shards) {
      // This is an authoritative invalid handle, not a transport miss.
      candidate_distances[index] = -1.0f;
      continue;
    }
    const u64 remote_offset = remote_byte_offset(raw) + params.node_meta_offset;
    request_shards[kTerminalCandidateMetadataBase + index] = shard;
    request_offsets[kTerminalCandidateMetadataBase + index] = remote_offset;
    u32 matched_cache_slot = UINT32_MAX;
    // Frontier turnover is normally positional near convergence. Test the
    // issue-time Beam rank first, preserving the full address/incarnation
    // checks. Only the small rank-drift subset pays the general search.
    if (index < cache.candidate_count) {
      const u32 cache_shard = scratch.request_shards[index];
      const u8* record = scratch.record_base +
                         static_cast<size_t>(index) * params.node_record_stride;
      if (cache_shard == shard &&
          scratch.remote_offsets[index] == remote_offset &&
          scratch.shard_status[cache_shard] == 0 &&
          terminal_exact_cache_payload_valid(params, record, handle)) {
        matched_cache_slot = index;
      }
    }
    for (u32 cache_slot = 0;
         matched_cache_slot == UINT32_MAX && cache_slot < cache.candidate_count;
         ++cache_slot) {
      if (cache_slot == index) continue;
      const u32 cache_shard = scratch.request_shards[cache_slot];
      if (cache_shard != shard ||
          scratch.remote_offsets[cache_slot] != remote_offset ||
          scratch.shard_status[cache_shard] != 0) {
        continue;
      }
      const u8* record = scratch.record_base + static_cast<size_t>(cache_slot) *
                                                 params.node_record_stride;
      if (!terminal_exact_cache_payload_valid(params, record, handle)) {
        continue;
      }
      matched_cache_slot = cache_slot;
      break;
    }
    if (matched_cache_slot != UINT32_MAX) {
      candidate_ids[index] = matched_cache_slot;
    }
  }
  __syncthreads();

  // Claim every cache hit exactly once, then inject every pre-known miss into
  // a distinct unclaimed record slot. With F <= B == 128, unique H hit slots
  // leave B-H >= F-H slots, so this never needs another device allocation.
  // The defensive no-slot state remains a strict fallback.
  if (threadIdx.x == 0) {
    for (u32 word = 0; word < (kPersistentMaxBeam + 31u) / 32u; ++word) {
      terminal_cache_used_slots[word] = 0;
    }
    for (u32 index = 0; index < candidate_count; ++index) {
      const u32 slot = candidate_ids[index];
      if (slot >= cache.candidate_count || slot >= kPersistentMaxBeam) {
        candidate_ids[index] = UINT32_MAX;
        continue;
      }
      const u32 bit = 1u << (slot & 31u);
      u32& used = terminal_cache_used_slots[slot >> 5u];
      if ((used & bit) != 0) {
        candidate_ids[index] = UINT32_MAX;
      } else {
        used |= bit;
      }
    }
    u32 miss_count = 0;
    u32 next_free = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      if (candidate_distances[index] < 0.0f) continue;
      if (candidate_ids[index] != UINT32_MAX) {
        merge_expanded[index] = 0;
        continue;
      }
      while (next_free < kPersistentMaxBeam &&
             (terminal_cache_used_slots[next_free >> 5u] &
              (1u << (next_free & 31u))) != 0) {
        ++next_free;
      }
      if (next_free == kPersistentMaxBeam) continue;
      terminal_cache_used_slots[next_free >> 5u] |= 1u << (next_free & 31u);
      candidate_ids[index] = kTerminalMissSlotBit | next_free;
      merge_expanded[index] = 1;
      ++miss_count;
      ++next_free;
    }
    terminal_cache_mixed_misses = miss_count;

    // One immutable SoA is shared by every shard descriptor. Strict misses
    // are represented by full-record prefixes followed by address-identical
    // fenced header trailers. Cache hits already carry that exact train from
    // the terminal-horizon wave and require no second query-side round trip.
    u32 miss_index = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      const u16 origin = merge_expanded[index];
      if (origin > 1) continue;
      const u32 shard = request_shards[kTerminalCandidateMetadataBase + index];
      const u64 remote_offset =
        request_offsets[kTerminalCandidateMetadataBase + index];
      const u32 slot = candidate_ids[index] & kTerminalSlotMask;
      u8* record = scratch.record_base +
                   static_cast<size_t>(slot) * params.node_record_stride;
      if (origin == 1) {
        const u32 prefix = miss_index++;
        const u32 trailer = terminal_cache_mixed_misses + prefix;
        request_shards[prefix] = shard;
        request_offsets[prefix] = remote_offset;
        request_local_iova_offsets[prefix] =
          reinterpret_cast<u64>(record) - params.direct_local_iova_base;
        request_shards[trailer] = shard;
        request_offsets[trailer] = remote_offset;
        request_local_iova_offsets[trailer] =
          reinterpret_cast<u64>(
            record + exact_record_trailer_offset(params.node_record_bytes)) -
          params.direct_local_iova_base;
      }
    }
  }
  __syncthreads();

  const u32 mixed_request_count = 2u * terminal_cache_mixed_misses;
  // Prefetch status is no longer needed after the mapping barrier. Reuse the
  // same completion words for one mixed final validation wave.
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    scratch.shard_status[shard] = 0;
    bool has_miss = false;
    for (u32 index = 0; index < terminal_cache_mixed_misses; ++index) {
      has_miss |= request_shards[index] == shard;
    }
    i32 issue_status = 0;
    if (has_miss) {
      if (exact_snapshot_train_batches != nullptr) {
        atomicAdd(exact_snapshot_train_batches, 1u);
      }
      issue_status = direct_fetch_split_batch(
        params, shard, request_shards, request_offsets, mixed_request_count,
        terminal_cache_mixed_misses, params.node_record_bytes,
        (descriptor.query_slot + shard) % params.direct_qps_per_node,
        request_local_iova_offsets, scratch.shard_status + shard, nullptr,
        nullptr, nullptr, nullptr, kDirectBatchFlagMixedMandatoryFencedTail);
    }
    if (issue_status != -EINPROGRESS) {
      scratch.shard_status[shard] = issue_status;
    }
  }
  __syncthreads();
  for (u32 shard = threadIdx.x; shard < params.num_shards;
       shard += blockDim.x) {
    if (scratch.shard_status[shard] == -EINPROGRESS) {
      scratch.shard_status[shard] =
        wait_direct_batch(params, scratch.shard_status + shard);
    }
  }
  __syncthreads();

  for (u32 index = threadIdx.x; index < candidate_count; index += blockDim.x) {
    const u16 origin = merge_expanded[index];
    if (origin > 1) continue;
    const u32 cache_slot = candidate_ids[index] & kTerminalSlotMask;
    const u32 shard = request_shards[kTerminalCandidateMetadataBase + index];
    if (shard >= params.num_shards || scratch.shard_status[shard] != 0) {
      // Validation transport failure: preserve the ordinary exact fallback.
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = FLT_MAX;
      continue;
    }
    const u8* record = scratch.record_base + static_cast<size_t>(cache_slot) *
                                               params.node_record_stride;
    const bool visible =
      origin == 1
        ? exact_record_visible(params, record, candidate_handles[index])
        : terminal_exact_cache_payload_valid(params, record,
                                             candidate_handles[index]);
    if (!visible) {
      // The fenced trailer is the visibility linearization point. Negative
      // distances are private markers and never reach a sorter.
      candidate_ids[index] = UINT32_MAX;
      candidate_distances[index] = -1.0f;
      continue;
    }
    candidate_ids[index] =
      *reinterpret_cast<const u32*>(record + kNodeIdOffset);
    const f32 distance =
      exact_storage_distance(params, query, record + params.node_vector_offset);
    candidate_distances[index] =
      finite_f32_bits(distance) && distance != FLT_MAX ? distance : -1.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    u32 hit_count = 0;
    u32 miss_count = 0;
    u32 cache_resolved = 0;
    u32 promoted_count = 0;
    for (u32 index = 0; index < candidate_count; ++index) {
      const f32 distance = candidate_distances[index];
      if (distance >= 0.0f && distance != FLT_MAX &&
          candidate_ids[index] != UINT32_MAX) {
        beam_handles[hit_count] = candidate_handles[index];
        beam_ids[hit_count] = candidate_ids[index];
        beam_distances[hit_count] = distance;
        beam_expanded[hit_count] = 0;
        ++hit_count;
        ++cache_resolved;
        if (merge_expanded[index] == 0) ++promoted_count;
      } else if (distance == FLT_MAX) {
        // Forward compaction is overlap-safe: destination never exceeds the
        // source index already consumed by this scalar stable scan.
        candidate_handles[miss_count++] = candidate_handles[index];
      } else {
        // A coherent final header rejected this candidate.
        // Invalid/unresolvable handles never issued an exact read, matching
        // the established exactify_into_beam telemetry contract.
        if (merge_expanded[index] <= 1) ++cache_resolved;
      }
    }
    beam_count = hit_count;
    cache.promoted_records = promoted_count;
    cache.miss_count = miss_count;
    cache.wasted_bytes =
      static_cast<u64>(cache.arrived_records > promoted_count
                         ? cache.arrived_records - promoted_count
                         : 0u) *
      params.node_record_bytes;
    if (exact_reads != nullptr) {
      *exact_reads += cache_resolved;
    }
  }
  __syncthreads();

  // With reset_beam=false the established implementation merges exact cache
  // hits with all strict misses, performs the one authoritative exact sort,
  // and preserves its transport/failure semantics unchanged.
  return exactify_into_beam(
    params, descriptor, query, candidate_handles, candidate_ids,
    candidate_distances, cache.miss_count, beam_handles, beam_ids,
    beam_distances, beam_expanded, beam_count, exact_reads,
    exact_snapshot_train_batches, exact_snapshot_train_fallbacks, beam_capacity,
    false, merge_handles, merge_ids, merge_distances, merge_expanded);
}

}  // namespace gpu_search::persistent_kernel_detail

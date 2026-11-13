#pragma once

#include "gpu_search/persistent_kernel/exact_record_io.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ graph_record_validation::SnapshotState classify_graph_record(
  const PersistentKernelParams& params, const u8* record, u64 handle) {
  return graph_record_validation::classify_snapshot(
    record, params.graph_entry_bytes, params.graph_degree,
    params.graph_entry_capacity, remote_incarnation(handle));
}

__device__ graph_record_validation::SnapshotState classify_short_graph_record(
  const PersistentKernelParams& params, const u8* record, u32 transferred_bytes,
  u64 handle) {
  return graph_record_validation::classify_zero_extended_snapshot(
    record, transferred_bytes, params.graph_entry_bytes, params.graph_degree,
    params.graph_entry_capacity, remote_incarnation(handle));
}

// Validate only descriptors backed by the dedicated frontier-request SoA.
// Pointer identity is the type tag, avoiding a larger queue descriptor and
// preserving the generic exact-vector/dynamic-code transport ABI. One owner
// warp lane validates one graph record after the final CQE and before the
// release publication of completion_status.
__device__ void validate_frontier_owner_batch(
  const PersistentKernelParams& params, const DirectBatchDescriptor& descriptor,
  u32 memory_node, u32 lane) {
  if (params.speculative_graph_request_shards == nullptr ||
      params.speculative_graph_request_offsets == nullptr ||
      params.speculative_graph_request_local_iovas == nullptr ||
      params.speculative_graph_request_handles == nullptr ||
      params.speculative_graph_validation_states == nullptr ||
      descriptor.request_shards == nullptr ||
      descriptor.remote_offsets == nullptr ||
      descriptor.local_iova_offsets == nullptr) {
    return;
  }
  const size_t capacity =
    static_cast<size_t>(params.query_slots) * kPersistentFrontierRobCapacity;
  const uintptr_t shards_begin =
    reinterpret_cast<uintptr_t>(params.speculative_graph_request_shards);
  const uintptr_t shards_end = shards_begin + capacity * sizeof(u32);
  const uintptr_t descriptor_shards =
    reinterpret_cast<uintptr_t>(descriptor.request_shards);
  if (descriptor_shards < shards_begin || descriptor_shards >= shards_end ||
      ((descriptor_shards - shards_begin) % sizeof(u32)) != 0) {
    return;
  }
  const size_t request_base = (descriptor_shards - shards_begin) / sizeof(u32);
  if (request_base + descriptor.request_count > capacity ||
      descriptor.remote_offsets !=
        params.speculative_graph_request_offsets + request_base ||
      descriptor.local_iova_offsets !=
        params.speculative_graph_request_local_iovas + request_base ||
      (params.speculative_graph_request_bytes != nullptr &&
       descriptor.request_bytes !=
         params.speculative_graph_request_bytes + request_base)) {
    return;
  }

  for (u32 index = lane; index < descriptor.request_count; index += 32) {
    if (descriptor.request_shards[index] != memory_node) continue;
    const u32 transferred_bytes = descriptor.request_bytes == nullptr
                                    ? descriptor.bytes
                                    : descriptor.request_bytes[index];
    const u8* record = reinterpret_cast<const u8*>(
      params.direct_local_iova_base + descriptor.local_iova_offsets[index]);
    const u64 handle =
      params.speculative_graph_request_handles[request_base + index];
    u32 required_bytes = 0;
    const bool prefix_valid =
      graph_record_validation::required_live_extent_bytes(
        record, transferred_bytes, params.graph_degree,
        params.graph_entry_capacity, required_bytes);
    const bool partial = transferred_bytes < params.graph_entry_bytes;
    const bool extent_underhint =
      partial && prefix_valid && required_bytes > transferred_bytes;
    const graph_record_validation::SnapshotState snapshot =
      prefix_valid && required_bytes <= transferred_bytes
        ? (partial ? classify_short_graph_record(params, record,
                                                 transferred_bytes, handle)
                   : classify_graph_record(params, record, handle))
        : graph_record_validation::SnapshotState::invalid;
    const FrontierValidationState validation =
      extent_underhint ? FrontierValidationState::extent_underhint
      : snapshot == graph_record_validation::SnapshotState::valid
        ? FrontierValidationState::valid
      : snapshot == graph_record_validation::SnapshotState::stale_incarnation
        ? FrontierValidationState::stale_incarnation
        : FrontierValidationState::invalid_snapshot;
    params.speculative_graph_validation_states[request_base + index] =
      static_cast<u8>(validation);
  }
  // Every owner lane publishes a disjoint validation byte. The owner warp
  // synchronizes before lane zero releases completion_status, but the release
  // flag belongs to a different thread; make each producer's global write
  // visible to the query CTA before that cross-thread publication.
  __threadfence();
}

__device__ __forceinline__ u8 load_graph_extent_class(
  const PersistentKernelParams& params, u32 static_ordinal) {
  if (params.graph_extent_class_words == nullptr ||
      static_ordinal >= params.num_nodes) {
    return graph_record_validation::kGraphExtentClassUnknown;
  }
  const u32 word = *reinterpret_cast<volatile const u32*>(
    params.graph_extent_class_words + static_ordinal / sizeof(u32));
  return graph_record_validation::packed_graph_extent_class(
    word, static_ordinal % sizeof(u32));
}

// Resolve the incarnation-scoped dynamic arena state for a graph handle.  PQ
// discovery publishes this word as payload -> fence -> tagged state.  An
// exact incarnation match is therefore sufficient to reuse its extent hint;
// BUSY, unknown, and recycled slots all fall back to the full graph record.
__device__ __forceinline__ bool dynamic_graph_extent_state(
  const PersistentKernelParams& params, u64 handle, u32*& state,
  u32& observed) {
  state = nullptr;
  observed = 0;
  if (params.dynamic_graph_extent_enabled == 0 ||
      params.dynamic_code_arena_states == nullptr ||
      remote_incarnation(handle) == 0) {
    return false;
  }
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  if (!resolve_handle(params, handle, raw, shard, graph_offset) ||
      shard >= params.num_shards) {
    return false;
  }
  const DeviceShardRegion& region = params.shards[shard];
  u64 arena_slot = 0;
  if (!dynamic_code_arena_slot_from_offset(region, remote_byte_offset(raw),
                                           params.dynamic_code_arena_capacity,
                                           arena_slot)) {
    return false;
  }
  state = params.dynamic_code_arena_states + arena_slot;
  observed = dynamic_arena_state_load(state);
  return dynamic_code_arena_state_matches(observed, remote_incarnation(handle));
}

__device__ __forceinline__ u8 load_dynamic_graph_extent_class(
  const PersistentKernelParams& params, u64 handle) {
  u32* state = nullptr;
  u32 observed = 0;
  return dynamic_graph_extent_state(params, handle, state, observed)
           ? dynamic_code_tag_extent_class(observed)
           : kPersistentDynamicCodeArenaUnknownExtent;
}

using GraphExtentAdaptation = vamana::graph_extent::Adaptation;

__device__ GraphExtentAdaptation adapt_graph_extent_class(
  const PersistentKernelParams& params, u64 handle, u32 required_bytes) {
  const u8 observed_graph_class =
    graph_record_validation::graph_extent_class_for_required_bytes(
      required_bytes, params.graph_entry_capacity);
  if (observed_graph_class ==
      graph_record_validation::kGraphExtentClassUnknown) {
    return GraphExtentAdaptation::none;
  }

  u32 static_ordinal = 0;
  if (static_ordinal_from_raw(params, handle, static_ordinal)) {
    if (params.graph_extent_class_words == nullptr ||
        static_ordinal >= params.num_nodes) {
      return GraphExtentAdaptation::none;
    }
    u32* const word =
      params.graph_extent_class_words + static_ordinal / sizeof(u32);
    const u32 byte_index = static_ordinal % sizeof(u32);
    u32 observed = *reinterpret_cast<volatile u32*>(word);
    while (true) {
      u32 desired = observed;
      const auto transition = graph_record_validation::adapted_graph_extent_word(
        observed, byte_index, observed_graph_class, desired);
      if (transition.adaptation == GraphExtentAdaptation::none) {
        return GraphExtentAdaptation::none;
      }
      const u32 prior = atomicCAS(word, observed, desired);
      if (prior == observed) return transition.adaptation;
      observed = prior;
    }
  }

  u32* state = nullptr;
  u32 observed = 0;
  const u32 incarnation = remote_incarnation(handle);
  if (!dynamic_graph_extent_state(params, handle, state, observed)) {
    return GraphExtentAdaptation::none;
  }
  while (true) {
    u32 desired = observed;
    const auto transition = dynamic_code_arena_adapted_extent_state(
      observed, incarnation, observed_graph_class, desired);
    if (transition.adaptation == GraphExtentAdaptation::none) {
      return GraphExtentAdaptation::none;
    }
    const u32 prior =
      dynamic_arena_state_compare_exchange(state, observed, desired);
    if (prior == observed) return transition.adaptation;
    observed = prior;
  }
}

__device__ bool prepare_graph_record_in_scratch(
  const PersistentKernelParams& params, u64 handle, u32 query_slot,
  u32 scratch_slot, u32& acquired_slot, u32& request_shard, u64& request_offset,
  u64& request_local_iova, u32& request_bytes) {
  acquired_slot = UINT32_MAX;
  request_shard = UINT32_MAX;
  request_offset = 0;
  request_local_iova = 0;
  request_bytes = params.graph_entry_bytes;
  u64 raw = 0;
  u64 graph_offset = 0;
  u32 shard = 0;
  u32 static_ordinal = 0;
  const bool static_record =
    params.graph_extent_class_words != nullptr &&
    params.graph_request_bytes != nullptr &&
    static_ordinal_from_raw(params, handle, static_ordinal);
  if (!resolve_handle(params, handle, raw, shard, graph_offset)) return false;
  // Every policy ultimately fetches and validates the authoritative record.
  if (*reinterpret_cast<volatile u32*>(params.stop) != 0 ||
      params.graph_scratch == nullptr ||
      scratch_slot >= kPersistentGraphScratchSlots) {
    return false;
  }
  u8* destination = params.graph_scratch + (static_cast<size_t>(query_slot) *
                                              kPersistentGraphScratchSlots +
                                            scratch_slot) *
                                             kPersistentGraphReadBytes;
  acquired_slot = kGraphScratchBit | scratch_slot;
  request_shard = shard;
  request_offset = graph_offset;
  request_local_iova =
    reinterpret_cast<u64>(destination) - params.direct_local_iova_base;
  const bool header_neighbor =
    params.graph_read_policy ==
    static_cast<u32>(GraphReadPolicy::header_neighbor);
  // Both bootstrap sources produce hints only. An insufficient short read is
  // upgraded to an authoritative full retry; dynamic UNKNOWN is normalized to
  // the common sentinel before computing the transfer length.
  if (header_neighbor && params.graph_request_bytes != nullptr) {
    request_bytes = graph_record_validation::kGraphRecordHeaderBytes;
  } else if (static_record && params.graph_extent_class_words != nullptr &&
             params.graph_request_bytes != nullptr &&
             (params.graph_entry_bytes & (sizeof(u64) - 1u)) == 0) {
    request_bytes = graph_record_validation::graph_extent_bytes_for_class(
      load_graph_extent_class(params, static_ordinal), params.graph_entry_bytes,
      params.graph_entry_capacity);
  } else if (!static_record && params.graph_request_bytes != nullptr &&
             (params.graph_entry_bytes & (sizeof(u64) - 1u)) == 0) {
    const u8 dynamic_class = load_dynamic_graph_extent_class(params, handle);
    request_bytes = graph_record_validation::graph_extent_bytes_for_class(
      dynamic_class == kPersistentDynamicCodeArenaUnknownExtent
        ? graph_record_validation::kGraphExtentClassUnknown
        : dynamic_class,
      params.graph_entry_bytes, params.graph_entry_capacity);
  }
  return true;
}

__device__ bool prepare_graph_record(const PersistentKernelParams& params,
                                     u64 handle, u32 query_slot,
                                     u32 request_index, u32& acquired_slot,
                                     u32& request_shard, u64& request_offset,
                                     u64& request_local_iova,
                                     u32& request_bytes) {
  if (request_index >= kPersistentMaxPrefetch) return false;
  return prepare_graph_record_in_scratch(
    params, handle, query_slot, request_index, acquired_slot, request_shard,
    request_offset, request_local_iova, request_bytes);
}

__device__ u8* graph_record_pointer(const PersistentKernelParams& params,
                                    u32 query_slot, u32 acquired_slot) {
  if ((acquired_slot & kGraphScratchBit) != 0) {
    const u32 request_index = acquired_slot & ~kGraphScratchBit;
    if (request_index >= kPersistentGraphScratchSlots) return nullptr;
    return params.graph_scratch +
           (static_cast<size_t>(query_slot) * kPersistentGraphScratchSlots +
            request_index) *
             kPersistentGraphReadBytes;
  }
  return nullptr;
}

// Query-local state carried beside one authoritative graph fetch. A force-full
// retry is produced only after an asynchronous short prefix has already shown
// that the exact handle needs more bytes. The first admitted full attempt is
// therefore the full half of that one fallback; later checksum retries keep
// the underhint provenance for safe promotion but are not another fallback.
inline constexpr u32 kGraphReadLogical = 1u;
inline constexpr u32 kGraphReadStartedWithShortExtent = 2u;
inline constexpr u32 kGraphReadNeedsExtentFallback = 4u;
inline constexpr u32 kGraphReadExtentUnderhint = 8u;
inline constexpr u32 kGraphReadHeaderNeighborBody = 16u;
inline constexpr u32 kGraphReadHeaderNeighborFullFallback = 32u;

#ifdef __CUDACC__
__host__ __device__
#endif
  inline constexpr u32
  prepare_graph_read_attempt_state(u32 full_record_bytes,
                                   bool force_full_underhint,
                                   u32& transfer_bytes) {
  if (force_full_underhint) {
    transfer_bytes = full_record_bytes;
    return kGraphReadLogical | kGraphReadNeedsExtentFallback |
           kGraphReadExtentUnderhint;
  }
  return kGraphReadLogical |
         (transfer_bytes < full_record_bytes ? kGraphReadStartedWithShortExtent
                                             : 0u);
}

#ifdef __CUDACC__
__host__ __device__
#endif
  inline constexpr u32
  graph_read_state_after_fallback_admission(u32 state) {
  return state & ~kGraphReadNeedsExtentFallback;
}

#ifdef __CUDACC__
__host__ __device__
#endif
  inline constexpr u32
  graph_read_state_after_header_neighbor_conflict(u32 state) {
  return (state & ~kGraphReadHeaderNeighborBody) |
         kGraphReadNeedsExtentFallback |
         kGraphReadHeaderNeighborFullFallback;
}

struct GraphReadAdmissionAccounting {
  u32 fallback_reads{};
  u32 underhint_reads{};
  u32 retry_reads{};
};

// Classify one admitted physical read. Attempt zero may still be a physical
// retry when this helper was entered after an asynchronous short under-hint;
// later admitted attempts are ordinary checksum retries. Clearing the
// needs-fallback bit after admission makes the fallback count one-shot.
#ifdef __CUDACC__
__host__ __device__
#endif
  inline constexpr GraphReadAdmissionAccounting
  classify_graph_read_admission(u32 state, bool full_record_read,
                                u32 batch_attempt) {
  const bool fallback =
    full_record_read && (state & kGraphReadNeedsExtentFallback) != 0;
  return GraphReadAdmissionAccounting{
    .fallback_reads = fallback ? 1u : 0u,
    .underhint_reads =
      fallback && (state & kGraphReadExtentUnderhint) != 0 ? 1u : 0u,
    .retry_reads =
      batch_attempt != 0 ||
          (fallback && (state & kGraphReadStartedWithShortExtent) == 0)
        ? 1u
        : 0u,
  };
}

// Per-query physical accounting for dynamic graph traffic.  The structure is
// intended to live in shared memory and is passed by pointer through the RDMA
// helpers, avoiding six additional wide call arguments. Counts include every
// admitted physical snapshot attempt (including full retries), while fallback
// counts only the first admitted short->full upgrade. The separate hit pointer
// counts logical reads accepted as short prefixes without a full-record retry.
struct DynamicGraphTelemetry {
  u64 read_bytes{};
  u32 short_reads{};
  u32 full_reads{};
  u32 fallback_reads{};
  u32 hint_promotions{};
  u32 hint_demotions{};
  u32* full_record_avoidance_hits{};
};

__device__ __forceinline__ void account_full_record_avoidance_hit(
    const PersistentKernelParams& params, u32 transfer_bytes,
    graph_record_validation::SnapshotState snapshot,
    DynamicGraphTelemetry* telemetry) {
  const bool live_extent =
    params.graph_read_policy == static_cast<u32>(GraphReadPolicy::live_extent);
  if (telemetry == nullptr || telemetry->full_record_avoidance_hits == nullptr ||
      !graph_record_validation::is_full_record_avoidance_hit(
        live_extent, transfer_bytes < params.graph_entry_bytes, snapshot)) {
    return;
  }
  atomicAdd(telemetry->full_record_avoidance_hits, 1u);
}

__device__ __forceinline__ bool dynamic_graph_telemetry_handle(u64 handle) {
  return handle != 0 && handle != kInvalidDeviceHandle &&
         remote_incarnation(handle) != 0;
}

__device__ __forceinline__ void add_dynamic_graph_read_telemetry(
  DynamicGraphTelemetry* telemetry, u32 short_reads, u32 full_reads,
  u64 read_bytes, u32 fallback_reads = 0) {
  if (telemetry == nullptr) return;
  if (short_reads != 0) atomicAdd(&telemetry->short_reads, short_reads);
  if (full_reads != 0) atomicAdd(&telemetry->full_reads, full_reads);
  if (read_bytes != 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&telemetry->read_bytes),
              static_cast<unsigned long long>(read_bytes));
  }
  if (fallback_reads != 0) {
    atomicAdd(&telemetry->fallback_reads, fallback_reads);
  }
}

// Validate an asynchronously fetched frontier payload in the consuming query
// CTA. Publishing the CQ completion no longer depends on owner-side checksum
// work or a cross-thread system fence; the same snapshot/incarnation and
// Live-Extent rules are applied after the completion acquire.
__device__ FrontierValidationState validate_frontier_record_local(
  const PersistentKernelParams& params, u32 query_slot,
  const FrontierRobEntry& entry,
  DynamicGraphTelemetry* dynamic_telemetry = nullptr) {
  const u8* record = graph_record_pointer(
    params, query_slot,
    kGraphScratchBit | static_cast<u32>(entry.scratch_slot));
  if (record == nullptr || entry.transfer_bytes == 0) {
    return FrontierValidationState::invalid_snapshot;
  }
  u32 required_bytes = 0;
  const bool prefix_valid = graph_record_validation::required_live_extent_bytes(
    record, entry.transfer_bytes, params.graph_degree,
    params.graph_entry_capacity, required_bytes);
  if (!prefix_valid) {
    return FrontierValidationState::invalid_snapshot;
  }
  if (entry.transfer_bytes < params.graph_entry_bytes &&
      required_bytes > entry.transfer_bytes) {
    // This short prefix is not checksum-authoritative. Preserve only a
    // query-local force-full classification; the full retry below is the
    // first point allowed to repair the global extent hint.
    return FrontierValidationState::extent_underhint;
  }
  if (required_bytes > entry.transfer_bytes) {
    return FrontierValidationState::invalid_snapshot;
  }
  const graph_record_validation::SnapshotState snapshot =
    entry.transfer_bytes < params.graph_entry_bytes
      ? classify_short_graph_record(params, record, entry.transfer_bytes,
                                    entry.node_handle)
      : classify_graph_record(params, record, entry.node_handle);
  if (snapshot == graph_record_validation::SnapshotState::valid) {
    account_full_record_avoidance_hit(
      params, entry.transfer_bytes, snapshot, dynamic_telemetry);
    // Only a checksum-valid exact snapshot may move a dynamic hint down. The
    // helper applies hysteresis and rejects BUSY/recycled incarnations.
    if (adapt_graph_extent_class(params, entry.node_handle, required_bytes) ==
          GraphExtentAdaptation::demoted &&
        dynamic_telemetry != nullptr) {
      atomicAdd(&dynamic_telemetry->hint_demotions, 1u);
    }
    return FrontierValidationState::valid;
  }
  return snapshot == graph_record_validation::SnapshotState::stale_incarnation
           ? FrontierValidationState::stale_incarnation
           : FrontierValidationState::invalid_snapshot;
}

struct FrontierGraphBatchState {
  u32 active{};
  u32 fatal{};
  u32 rejected{};
  // Cumulative wait observation classifies a completely READY wave. Ordered
  // scoring uses READY-before-block directly and no longer needs a
  // most-recent wait flag.
  u32 finish_had_pending{};
  u64 issue_timestamp_ns[kPersistentMaxShards]{};
};

struct CoreFrontierTelemetry {
  u64* wait_cycles{};
  u64* completion_latency_ns{};
  u64* completion_groups{};
  u32* arrived{};
  u32* stale{};
  u32* ready_waves{};
  DynamicGraphTelemetry* dynamic_graph{};
};

// A split descriptor is visible to the query before the owner knows whether
// its speculative suffix fits the already-formed critical SQ train. Issue
// accounting therefore records the suffix optimistically. If the owner
// returns -EAGAIN without posting a tail WQE, the completion path publishes
// this exact one-shot correction so final graph-read telemetry remains
// physical rather than attempted.
struct TailAdmissionCorrection {
  u32 rejected_batches{};
  u32 rejected_reads{};
  u32 rejected_live_extent_reads{};
  u32 rejected_full_record_reads{};
  u64 rejected_bytes{};
};

struct TailFrontierTelemetry {
  u32* arrived{};
  u32* stale{};
  u64* wait_cycles{};
  u64* completion_latency_ns{};
  u64* completion_groups{};
  u64* wasted_bytes{};
  TailAdmissionCorrection* admission_correction{};
  DynamicGraphTelemetry* dynamic_graph{};
};

// A narrow core or tail frontier is owned by one warp. This avoids the
// general builder's CTA-wide setup barriers for both priority classes while
// preserving separate core/tail completion words inside one critical owner
// descriptor and one owner queue.
}  // namespace gpu_search::persistent_kernel_detail

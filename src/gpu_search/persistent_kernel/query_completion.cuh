#pragma once

#include "gpu_search/adaptive_frontier.hh"
#include "gpu_search/centroid_route_ranking.hh"
#include "gpu_search/persistent_kernel/rdma_read.cuh"

#include <cuda/atomic>

namespace gpu_search::persistent_kernel_detail {

struct TailFrontierFeedback {
  u32 promoted{};
  u32 retained{};
  u32 stale{};
  u32 queue_rejects{};
  u32 core_hits{};
  u32 core_misses{};
  u32 tail_admitted{};
  u32 commit_waves_observed{};
  u32 commit_waves_covered{};
};

__device__ __forceinline__ void apply_tail_admission_correction(
    TailAdmissionCorrection& correction, TailFrontierFeedback& feedback,
    u32& total_remote_batches, u32& total_remote_reads,
    u64& total_graph_read_bytes, u32& total_graph_live_extent_reads,
    u32& total_graph_full_record_reads, u32& speculative_graph_reads,
    u64& speculative_graph_bytes, u32& speculative_queue_rejects) {
  if (threadIdx.x != 0 || (correction.rejected_batches == 0 &&
                           correction.rejected_reads == 0)) {
    return;
  }
  const u32 rejected_batches = correction.rejected_batches;
  const u32 rejected_reads = correction.rejected_reads;
  const u64 rejected_bytes = correction.rejected_bytes;
  const u32 rejected_live = correction.rejected_live_extent_reads;
  const u32 rejected_full = correction.rejected_full_record_reads;

  // Split issue accounting is intentionally optimistic because the producer
  // cannot know owner SQ slack. Reconcile it exactly once when -EAGAIN proves
  // that no tail WQE was posted. Saturation guards keep diagnostics bounded
  // if a future caller violates the one-shot issue/completion contract.
  total_remote_batches -= min(total_remote_batches, rejected_batches);
  total_remote_reads -= min(total_remote_reads, rejected_reads);
  total_graph_read_bytes -= min(total_graph_read_bytes, rejected_bytes);
  total_graph_live_extent_reads -=
    min(total_graph_live_extent_reads, rejected_live);
  total_graph_full_record_reads -=
    min(total_graph_full_record_reads, rejected_full);
  speculative_graph_reads -=
    min(speculative_graph_reads, rejected_reads);
  speculative_graph_bytes -=
    min(speculative_graph_bytes, rejected_bytes);
  speculative_queue_rejects += rejected_reads;
  feedback.queue_rejects += rejected_reads;
  feedback.tail_admitted -= min(feedback.tail_admitted, rejected_reads);
  correction = {};
}

// The exact Stable-Run issue certificate already maps each next-commit handle
// to its query-local ROB slot.  Preserve that map across the communication
// overlap window instead of repeating selected×ROB and ROB×Beam searches at
// the next epoch.  All fields point to CTA-private shared storage.
struct CertifiedCommitReconcileContext {
  const u64* selected_handles{};
  const u32* certified_rob_slots{};
  FrontierRobEntry* frontier_rob{};
  u32* commit_rob_slots{};
  u32* graph_record_slots{};
  u64* critical_fetch_handles{};
  u32* critical_fetch_to_commit{};
  u32* critical_fetch_count{};
  u32* critical_rob_hits{};
  u32* critical_misses{};
  u32* speculative_promoted{};
  u32* core_prefetch_promoted{};
  u32* core_prefetch_stale{};
  TailFrontierFeedback* feedback{};
};

enum class UnderhintLookupMode : u8 {
  positional = 0,
  certified = 1,
  associative = 2,
};

__device__ __forceinline__ bool is_exact_underhint_evidence(
    const FrontierRobEntry& entry, u64 node_handle) {
  return entry.node_handle == node_handle &&
    entry.state == static_cast<u8>(FrontierRequestState::stale) &&
    entry.validation ==
      static_cast<u8>(FrontierValidationState::extent_underhint);
}

// Snapshot under-hint evidence before reconciliation clears stale ROB entries.
// Positional core and reusable-certificate paths already own exact position to
// ROB maps, so each lane checks only that one slot. Only the general/shadow
// path pays the bounded associative lookup. Exact tagged-handle equality keeps
// evidence query-local and prevents it from being mapped to a different
// incarnation. A warp-wide precheck skips all handle matching in the common
// case where this ROB contains no underhint evidence.
__device__ __forceinline__ void identify_selected_underhint_force_full(
    const u64* selected_handles,
    u32 selected_count,
    const FrontierRobEntry* frontier_rob,
    const u32* certified_rob_slots,
    UnderhintLookupMode lookup_mode,
    u8* selected_force_full,
    u32* any_force_full) {
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    const FrontierRobEntry& lane_entry = frontier_rob[position];
    const bool lane_has_underhint =
      lane_entry.state == static_cast<u8>(FrontierRequestState::stale) &&
      lane_entry.validation ==
        static_cast<u8>(FrontierValidationState::extent_underhint);
    const u32 underhint_mask = __ballot_sync(
      0xffffffffu, lane_has_underhint);
    if (position == 0) *any_force_full = underhint_mask != 0 ? 1u : 0u;
    bool force_full = false;
    if (underhint_mask != 0 && position < selected_count) {
      const u64 handle = selected_handles[position];
      if (lookup_mode == UnderhintLookupMode::positional) {
        force_full =
          is_exact_underhint_evidence(frontier_rob[position], handle);
      } else if (lookup_mode == UnderhintLookupMode::certified) {
        const u32 slot = certified_rob_slots[position];
        force_full = slot < kPersistentFrontierRobCapacity &&
          is_exact_underhint_evidence(frontier_rob[slot], handle);
      } else {
        for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
          if (is_exact_underhint_evidence(frontier_rob[slot], handle)) {
            force_full = true;
            break;
          }
        }
      }
    }
    if (underhint_mask != 0) {
      selected_force_full[position] = force_full ? 1u : 0u;
    }
  }
  __syncthreads();
}

// Every reconcile path publishes critical_fetch_to_commit. Re-indexing the
// position-local evidence only after compaction avoids threading another flag
// through each path and guarantees baseline misses are explicitly zeroed.
__device__ __forceinline__ void remap_critical_underhint_force_full(
    const u8* selected_force_full,
    u32 selected_count,
    const u32* critical_fetch_to_commit,
    u32 critical_fetch_count,
    u8* critical_force_full) {
  for (u32 fetch = threadIdx.x; fetch < critical_fetch_count;
       fetch += blockDim.x) {
    const u32 position = critical_fetch_to_commit[fetch];
    critical_force_full[fetch] =
      position < selected_count && selected_force_full[position] != 0
        ? 1u : 0u;
  }
  __syncthreads();
}

__device__ void set_query_trace_completion(
    const PersistentKernelParams& params, u32 query_slot,
    CompletionDescriptor& completion) {
  if (params.query_rdma_trace_headers == nullptr ||
      query_slot >= params.query_slots) return;
  const QueryRdmaTraceHeader& header =
    params.query_rdma_trace_headers[query_slot];
  if (header.request_id != completion.request_id || header.enabled == 0) return;
  completion.trace_event_count = min(
    header.event_count, params.query_rdma_trace_events_per_query);
  completion.trace_overflow = header.overflow;
}

__device__ void set_dynamic_code_cache_completion(
    CompletionDescriptor& completion, u32 cache_hits,
    u32 batch_deduplicated, u32 publish_successes,
    u32 first_occupancies, u32 publish_races,
    u32 lookup_probe_exhaustions, u32 publish_probe_exhaustions,
    u32 lookup_probes, u32 max_lookup_probes) {
  completion.dynamic_code_cache_hits = cache_hits;
  completion.dynamic_code_batch_deduplicated = batch_deduplicated;
  completion.dynamic_code_cache_publish_successes = publish_successes;
  completion.dynamic_code_cache_first_occupancies = first_occupancies;
  completion.dynamic_code_cache_publish_races = publish_races;
  completion.dynamic_code_cache_lookup_probe_exhaustions =
    lookup_probe_exhaustions;
  completion.dynamic_code_cache_publish_probe_exhaustions =
    publish_probe_exhaustions;
  completion.dynamic_code_cache_lookup_probes = lookup_probes;
  completion.dynamic_code_cache_max_lookup_probes = max_lookup_probes;
}

__device__ __forceinline__ void set_dynamic_graph_completion(
    CompletionDescriptor& completion,
    const DynamicGraphTelemetry& telemetry, u32 logical_graph_reads,
    u32 graph_read_policy) {
  completion.dynamic_graph_read_bytes = telemetry.read_bytes;
  completion.dynamic_graph_short_reads = telemetry.short_reads;
  completion.dynamic_graph_full_reads = telemetry.full_reads;
  completion.dynamic_graph_fallback_reads = telemetry.fallback_reads;
  completion.dynamic_graph_hint_promotions = telemetry.hint_promotions;
  completion.dynamic_graph_hint_demotions = telemetry.hint_demotions;
  if (graph_read_policy == static_cast<u32>(GraphReadPolicy::live_extent) &&
      telemetry.full_record_avoidance_hits != nullptr) {
    completion.graph_full_record_avoidance_hits = *telemetry.full_record_avoidance_hits;
    completion.graph_full_record_avoidance_misses =
      graph_record_validation::full_record_avoidance_misses(
        logical_graph_reads, completion.graph_full_record_avoidance_hits);
  }
}

__device__ void set_beam_merge_completion(
    CompletionDescriptor& completion,
    const BeamMergeCycleBreakdown& breakdown) {
  completion.beam_merge_prepare_cycles = breakdown.prepare;
  completion.beam_merge_sort_cycles = breakdown.sort;
  completion.beam_merge_materialize_cycles = breakdown.materialize;
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_terminal_exact_cache_completion(
    CompletionDescriptor& completion,
    const TerminalExactCacheState& cache) {
  if constexpr (EnableAsfe) {
    completion.terminal_exact_cache_attempted_queries =
      cache.attempted != 0 ? 1u : 0u;
    completion.terminal_exact_cache_issued_records = cache.issued_records;
    completion.terminal_exact_cache_promoted_records =
      cache.promoted_records;
    completion.terminal_exact_cache_wasted_bytes = cache.wasted_bytes;
    completion.terminal_exact_cache_queue_rejects = cache.queue_rejects;
    completion.terminal_exact_cache_miss_records = cache.miss_count;
  } else {
    completion.terminal_exact_cache_attempted_queries = 0;
    completion.terminal_exact_cache_issued_records = 0;
    completion.terminal_exact_cache_promoted_records = 0;
    completion.terminal_exact_cache_wasted_bytes = 0;
    completion.terminal_exact_cache_queue_rejects = 0;
    completion.terminal_exact_cache_miss_records = 0;
  }
}

__device__ __noinline__ void set_frontier_completion_full(
    CompletionDescriptor& completion,
    u32 logical_expansions,
    u32 critical_graph_reads,
    u64 critical_graph_bytes,
    u32 speculative_graph_reads,
    u64 speculative_graph_bytes,
    u32 speculative_arrived,
    u32 speculative_promoted,
    u32 speculative_stale,
    u64 speculative_wasted_bytes,
    u32 speculative_queue_rejects,
    u32 issue_epochs,
    u32 commit_epochs,
    u64 issue_width_sum,
    u64 issue_width_capacity_sum,
    u64 commit_width_sum,
    u32 max_issue_width,
    u32 max_commit_width,
    u32 critical_rob_hits,
    u32 critical_misses,
    u64 speculative_wait_cycles,
    u64 rdma_completion_latency_ns,
    u64 speculative_completion_latency_ns,
    u64 rdma_completion_groups,
    u64 speculative_completion_groups,
    u64 core_prefetch_bytes,
    u32 core_prefetch_reads,
    u32 core_prefetch_arrived,
    u32 core_prefetch_promoted,
    u32 core_prefetch_stale,
    u32 core_prefetch_queue_rejects,
    u32 core_prefetch_waves,
    u32 core_ready_waves) {
  completion.logical_expansions = logical_expansions;
  completion.critical_graph_reads = critical_graph_reads;
  completion.critical_graph_bytes = critical_graph_bytes;
  completion.speculative_graph_reads = speculative_graph_reads;
  completion.speculative_graph_bytes = speculative_graph_bytes;
  completion.speculative_arrived = speculative_arrived;
  completion.speculative_promoted = speculative_promoted;
  completion.speculative_stale = speculative_stale;
  completion.speculative_wasted_bytes = speculative_wasted_bytes;
  completion.speculative_queue_rejects = speculative_queue_rejects;
  completion.issue_epochs = issue_epochs;
  completion.commit_epochs = commit_epochs;
  completion.issue_width_sum = issue_width_sum;
  completion.issue_width_capacity_sum = issue_width_capacity_sum;
  completion.commit_width_sum = commit_width_sum;
  completion.max_issue_width = max_issue_width;
  completion.max_commit_width = max_commit_width;
  completion.critical_rob_hits = critical_rob_hits;
  completion.critical_misses = critical_misses;
  completion.speculative_wait_cycles = speculative_wait_cycles;
  completion.rdma_completion_latency_ns = rdma_completion_latency_ns;
  completion.speculative_completion_latency_ns =
    speculative_completion_latency_ns;
  completion.rdma_completion_groups = rdma_completion_groups;
  completion.speculative_completion_groups = speculative_completion_groups;
  completion.core_prefetch_bytes = core_prefetch_bytes;
  completion.core_prefetch_reads = core_prefetch_reads;
  completion.core_prefetch_arrived = core_prefetch_arrived;
  completion.core_prefetch_promoted = core_prefetch_promoted;
  completion.core_prefetch_stale = core_prefetch_stale;
  completion.core_prefetch_queue_rejects = core_prefetch_queue_rejects;
  completion.core_prefetch_waves = core_prefetch_waves;
  completion.core_ready_waves = core_ready_waves;
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_frontier_completion(
    CompletionDescriptor& completion,
    u32 logical_expansions,
    u32 critical_graph_reads,
    u64 critical_graph_bytes,
    u32 speculative_graph_reads,
    u64 speculative_graph_bytes,
    u32 speculative_arrived,
    u32 speculative_promoted,
    u32 speculative_stale,
    u64 speculative_wasted_bytes,
    u32 speculative_queue_rejects,
    u32 issue_epochs,
    u32 commit_epochs,
    u64 issue_width_sum,
    u64 issue_width_capacity_sum,
    u64 commit_width_sum,
    u32 max_issue_width,
    u32 max_commit_width,
    u32 critical_rob_hits,
    u32 critical_misses,
    u64 speculative_wait_cycles,
    u64 rdma_completion_latency_ns,
    u64 speculative_completion_latency_ns,
    u64 rdma_completion_groups,
    u64 speculative_completion_groups,
    u64 core_prefetch_bytes,
    u32 core_prefetch_reads,
    u32 core_prefetch_arrived,
    u32 core_prefetch_promoted,
    u32 core_prefetch_stale,
    u32 core_prefetch_queue_rejects,
    u32 core_prefetch_waves,
    u32 core_ready_waves) {
  if constexpr (EnableAsfe) {
    set_frontier_completion_full(
      completion, logical_expansions, critical_graph_reads,
      critical_graph_bytes, speculative_graph_reads,
      speculative_graph_bytes, speculative_arrived,
      speculative_promoted, speculative_stale,
      speculative_wasted_bytes, speculative_queue_rejects,
      issue_epochs, commit_epochs, issue_width_sum,
      issue_width_capacity_sum, commit_width_sum, max_issue_width,
      max_commit_width, critical_rob_hits, critical_misses,
      speculative_wait_cycles, rdma_completion_latency_ns,
      speculative_completion_latency_ns, rdma_completion_groups,
      speculative_completion_groups, core_prefetch_bytes,
      core_prefetch_reads, core_prefetch_arrived, core_prefetch_promoted,
      core_prefetch_stale, core_prefetch_queue_rejects,
      core_prefetch_waves, core_ready_waves);
  } else {
    completion.logical_expansions = logical_expansions;
    completion.critical_graph_reads = critical_graph_reads;
    completion.critical_graph_bytes = critical_graph_bytes;
    completion.speculative_graph_reads = 0;
    completion.speculative_graph_bytes = 0;
    completion.speculative_arrived = 0;
    completion.speculative_promoted = 0;
    completion.speculative_stale = 0;
    completion.speculative_wasted_bytes = 0;
    completion.speculative_queue_rejects = 0;
    completion.issue_epochs = 0;
    completion.commit_epochs = commit_epochs;
    completion.issue_width_sum = 0;
    completion.issue_width_capacity_sum = 0;
    completion.commit_width_sum = commit_width_sum;
    completion.max_issue_width = 0;
    completion.max_commit_width = max_commit_width;
    completion.critical_rob_hits = 0;
    completion.critical_misses = critical_misses;
    completion.speculative_wait_cycles = 0;
    completion.rdma_completion_latency_ns = rdma_completion_latency_ns;
    completion.speculative_completion_latency_ns = 0;
    completion.rdma_completion_groups = rdma_completion_groups;
    completion.speculative_completion_groups = 0;
    completion.core_prefetch_bytes = 0;
    completion.core_prefetch_reads = 0;
    completion.core_prefetch_arrived = 0;
    completion.core_prefetch_promoted = 0;
    completion.core_prefetch_stale = 0;
    completion.core_prefetch_queue_rejects = 0;
    completion.core_prefetch_waves = 0;
    completion.core_ready_waves = 0;
  }
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_frontier_certificate_completion(
    CompletionDescriptor& completion, u32 reusable_certificates,
    u32 streamed_candidate_runs, u32 ordered_score_batches,
    u32 ordered_score_candidates, u32 reusable_prefix_ranks,
    u32 reusable_full_prefix_certificates,
    u32 reusable_issued_certificates,
    u32 ooo_bypassed_parents,
    u32 certificate_rejects) {
  if constexpr (EnableAsfe) {
    // DEEC remains available as a focused proof/test primitive; production
    // uses the sort-once reusable PFEC certificate below.
    completion.frontier_telemetry_reserved0 = certificate_rejects;
    // The second retired DEEC slot is a success-only counter. Failure paths
    // keep it zero until they optionally overwrite both reserved words with
    // a rejected-handle diagnostic.
    completion.frontier_telemetry_reserved1 =
      completion.status == 0 ? ooo_bypassed_parents : 0u;
    completion.frontier_reusable_certificates = reusable_certificates;
    completion.frontier_streamed_candidate_runs =
      streamed_candidate_runs;
    completion.ordered_score_batches = ordered_score_batches;
    completion.ordered_score_candidates = ordered_score_candidates;
    completion.frontier_reusable_prefix_ranks =
      reusable_prefix_ranks;
    completion.frontier_reusable_full_prefix_certificates =
      reusable_full_prefix_certificates;
    completion.frontier_reusable_issued_certificates =
      reusable_issued_certificates;
  } else {
    completion.completion_score_batches = 0;
    completion.completion_score_candidates = 0;
    completion.frontier_telemetry_reserved0 = 0;
    completion.frontier_telemetry_reserved1 = 0;
    completion.frontier_reusable_certificates = 0;
    completion.frontier_streamed_candidate_runs = 0;
    completion.ordered_score_batches = 0;
    completion.ordered_score_candidates = 0;
    completion.frontier_reusable_prefix_ranks = 0;
    completion.frontier_reusable_full_prefix_certificates = 0;
    completion.frontier_reusable_issued_certificates = 0;
  }
}

}  // namespace gpu_search::persistent_kernel_detail

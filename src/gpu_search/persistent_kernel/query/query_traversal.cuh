#pragma once

#include "gpu_search/persistent_kernel/frontier/frontier_commit.cuh"
#include "gpu_search/persistent_kernel/query/query_preparation.cuh"
#include "gpu_search/persistent_kernel/query/route_snapshot.cuh"

namespace gpu_search::persistent_kernel_detail {

template <bool EnableAsfe,
          u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__device__ __forceinline__ void process_query(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    adaptive_frontier::ControllerState& frontier_controller) {
  // query_slot is consumed throughout the entire traversal.  Keeping the
  // descriptor field in a per-thread scalar extends one register (and, in
  // the 128-thread ASFE specialization, one spill value) across every RDMA,
  // scoring, and merge phase.  A single CTA-resident copy is broadcast by
  // shared memory and piggybacks on the completion-initialization barrier.
  __shared__ u32 query_slot;
  __shared__ u64 query_started_cycles;
  // Completion is produced by the CTA control lane only.  Keeping one copy
  // in shared memory prevents the large telemetry object from extending every
  // thread's local stack across the full ASFE query state machine.  That
  // local lifetime previously forced the 128-thread ASFE specialization to
  // spill even though all frontier metadata itself was CTA-resident.
  __shared__ CompletionDescriptor shared_completion;
  CompletionDescriptor& completion = shared_completion;
  if (threadIdx.x == 0) {
    query_slot = descriptor.query_slot;
    query_started_cycles = clock64();
    completion = CompletionDescriptor{
      .request_id = descriptor.request_id,
      .query_slot = query_slot,
    };
  }
  __syncthreads();
  if (!query_descriptor_valid(params, descriptor, query_slot)) {
    if (threadIdx.x == 0) {
      completion.status = -EINVAL;
      completion.diagnostic = make_query_diagnostic(
        QueryFailureReason::invalid_descriptor);
      completion.gpu_cycles = clock64() - query_started_cycles;
      set_query_trace_completion(params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  __shared__ u64 prepare_started_cycles;
  __shared__ u64 graph_phase_cycles;
  __shared__ u64 score_phase_cycles;
  __shared__ u64 beam_phase_cycles;
  __shared__ u64 exact_phase_cycles;
  __shared__ u64 dynamic_code_cycles;
  __shared__ u64 beam_selection_cycles;
  __shared__ u64 rdma_issue_cycles;
  __shared__ u64 frontier_preview_cycles;
  __shared__ u64 frontier_prepare_cycles;
  __shared__ u64 frontier_enqueue_cycles;
  __shared__ u64 frontier_subphase_started_cycles;
  __shared__ u64 rdma_wait_cycles;
  __shared__ u64 graph_validation_cycles;
  __shared__ u64 neighbor_decode_cycles;
  __shared__ u64 pq_score_cycles;
  __shared__ u64 visited_cycles;
  __shared__ u64 beam_merge_cycles;
  __shared__ BeamMergeCycleBreakdown beam_merge_breakdown;
  __shared__ BeamMergeCycleBreakdown beam_merge_round_breakdown;
  __shared__ u32 rdma_trace_enabled;
  __shared__ GraphFetchCycleBreakdown graph_fetch_breakdown;
  __shared__ u32 dynamic_code_candidates;
  __shared__ u32 dynamic_code_reads;
  __shared__ u32 dynamic_code_incarnation_rejects;
  __shared__ u32 dynamic_code_cache_hits;
  __shared__ u32 dynamic_code_batch_deduplicated;
  __shared__ u32 dynamic_code_cache_publish_successes;
  __shared__ u32 dynamic_code_cache_first_occupancies;
  __shared__ u32 dynamic_code_cache_publish_races;
  __shared__ u32 dynamic_code_cache_lookup_probe_exhaustions;
  __shared__ u32 dynamic_code_cache_publish_probe_exhaustions;
  __shared__ u32 dynamic_code_cache_lookup_probes;
  __shared__ u32 dynamic_code_cache_max_lookup_probes;
  __shared__ u64 phase_started_cycles;
  if (threadIdx.x == 0) {
    prepare_started_cycles = clock64();
    graph_phase_cycles = 0;
    score_phase_cycles = 0;
    beam_phase_cycles = 0;
    exact_phase_cycles = 0;
    dynamic_code_cycles = 0;
    beam_selection_cycles = 0;
    rdma_issue_cycles = 0;
    frontier_preview_cycles = 0;
    frontier_prepare_cycles = 0;
    frontier_enqueue_cycles = 0;
    frontier_subphase_started_cycles = 0;
    rdma_wait_cycles = 0;
    graph_validation_cycles = 0;
    neighbor_decode_cycles = 0;
    pq_score_cycles = 0;
    visited_cycles = 0;
    beam_merge_cycles = 0;
    beam_merge_breakdown = {};
    beam_merge_round_breakdown = {};
    rdma_trace_enabled =
      params.query_rdma_trace_mode ==
        static_cast<u32>(QueryRdmaTraceMode::full) ||
      (params.query_rdma_trace_mode ==
         static_cast<u32>(QueryRdmaTraceMode::sampled) &&
       params.query_rdma_trace_sample_rate != 0 &&
       descriptor.request_id % params.query_rdma_trace_sample_rate == 0);
    if (params.query_rdma_trace_headers != nullptr) {
      params.query_rdma_trace_headers[query_slot] = {
        .request_id = descriptor.request_id,
        .event_count = 0,
        .overflow = 0,
        .enabled = rdma_trace_enabled,
      };
    }
    dynamic_code_candidates = 0;
    dynamic_code_reads = 0;
    dynamic_code_incarnation_rejects = 0;
    dynamic_code_cache_hits = 0;
    dynamic_code_batch_deduplicated = 0;
    dynamic_code_cache_publish_successes = 0;
    dynamic_code_cache_first_occupancies = 0;
    dynamic_code_cache_publish_races = 0;
    dynamic_code_cache_lookup_probe_exhaustions = 0;
    dynamic_code_cache_publish_probe_exhaustions = 0;
    dynamic_code_cache_lookup_probes = 0;
    dynamic_code_cache_max_lookup_probes = 0;
  }
  __syncthreads();
  f32* query = params.decoded_queries + static_cast<size_t>(query_slot) * params.dim;
  f32* query_lut = params.query_luts +
    static_cast<size_t>(query_slot) *
      resolved_pq_subquantizers<PqSubquantizers>(params) * 256;
  prepare_query_vectors<PqSubquantizers>(
    params, descriptor, query_slot, query, query_lut);

  __shared__ u64 shared_beam_handles[kPersistentMaxBeam];
  __shared__ u32 shared_beam_ids[kPersistentMaxBeam];
  __shared__ f32 shared_beam_distances[kPersistentMaxBeam];
  __shared__ u8 shared_beam_expanded[kPersistentMaxBeam];
  __shared__ CandidateWorkspace candidate_workspace;
  __shared__ StableMergePreparedState stable_merge_state;
  u64* merge_handles = candidate_workspace.arrays.handles;
  // Stable-Run no longer needs its origin/candidate handle tail after graph
  // traversal. Reuse that dead lifetime for final exact-rerank IDs instead of
  // reserving a disjoint 512-entry shared array throughout the query.
  u32* merge_ids = reinterpret_cast<u32*>(
    candidate_workspace.arrays.handles + kPersistentMaxBeam);
  f32* merge_distances = candidate_workspace.arrays.distances;
  u16* merge_expanded = candidate_workspace.arrays.expanded;
  u64* navigation_handles = params.navigation_candidate_handles +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  f32* navigation_distances = params.navigation_candidate_distances +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  u64* beam_handles = shared_beam_handles;
  u32* beam_ids = shared_beam_ids;
  f32* beam_distances = shared_beam_distances;
  u8* beam_expanded = shared_beam_expanded;
  const u32 traversal_capacity = min(kPersistentMaxBeam, params.traversal_beam_width);
  u64* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < traversal_capacity; index += blockDim.x) {
    beam_handles[index] = kInvalidDeviceHandle;
    beam_ids[index] = UINT32_MAX;
    beam_distances[index] = FLT_MAX;
    beam_expanded[index] = 0;
  }
  for (u32 index = threadIdx.x; index < params.visited_capacity; index += blockDim.x) {
    visited[index] = kInvalidDeviceHandle;
  }
  __syncthreads();

  __shared__ u32 beam_count;
  // The compact 128-thread Stable-Run path retains four independently sorted
  // candidate prefixes until its exact preview has issued RDMA. Flags are
  // binary, so overlay their byte array with the wider IDs needed only by the
  // later exact-rerank phase. This keeps the four-run overlap below the 48-KiB
  // static shared-memory limit without adding global traffic.
  union RerankMetadataScratch {
    u8 flags[kPersistentStableRunScratch];
    u32 ids[kPersistentMaxExact];
  };
  __shared__ u64 rerank_handles[kPersistentStableRunScratch];
  __shared__ RerankMetadataScratch rerank_metadata;
  __shared__ f32 rerank_distances[kPersistentStableRunScratch];
  u8* rerank_flags = rerank_metadata.flags;
  u32* rerank_ids = rerank_metadata.ids;
  __shared__ u32 rerank_count;
  __shared__ u32 total_exact_reads;
  __shared__ QueryRouteSnapshotState route_state;
  auto& route_entry_count = route_state.route_entry_count;
  auto& route_seed_failed = route_state.route_seed_failed;
  auto& route_snapshot_retries = route_state.route_snapshot_retries;
  auto& route_failure_reason = route_state.route_failure_reason;
  if (threadIdx.x == 0) {
    rerank_count = 0;
    total_exact_reads = 0;
    route_snapshot_retries = 0;
  }
  __syncthreads();

  // A route publication may replace every seed after this CTA snapshots it.
  // Epoch contention is not an I/O error: wait for one complete authoritative
  // table transaction. The occupancy planner reserves a resident control CTA,
  // so the writer can always close the short odd-epoch window. A genuinely
  // stuck writer is bounded by the independent route-control deadline below.
  // Separately, re-snapshot once when a once-valid seed becomes stale after the
  // transaction; no immutable entry table is used as a fallback.
  for (u32 route_attempt = 0; route_attempt < 2; ++route_attempt) {
    if (threadIdx.x == 0) {
      beam_count = 0;
      rerank_count = 0;
    }
    const auto route_snapshot = snapshot_query_route(
      params, descriptor, query, navigation_handles, route_attempt, route_state);
    if (route_snapshot == QueryRouteSnapshotResult::cancelled) {
      return;
    }
    if (route_snapshot == QueryRouteSnapshotResult::timed_out) {
      if (threadIdx.x == 0) {
        completion.status = -ETIMEDOUT;
        completion.diagnostic = make_query_diagnostic(
          QueryFailureReason::route_snapshot_timeout,
          route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    if (route_entry_count != 0 &&
        !approximate_handles_batch<PqSubquantizers>(
          params, descriptor, query_lut, navigation_handles,
          route_entry_count, navigation_distances, &dynamic_code_cycles,
          &dynamic_code_candidates, &dynamic_code_reads,
          &dynamic_code_incarnation_rejects, &dynamic_code_cache_hits,
          &dynamic_code_batch_deduplicated,
          &dynamic_code_cache_publish_successes,
          &dynamic_code_cache_first_occupancies,
          &dynamic_code_cache_publish_races,
          &dynamic_code_cache_lookup_probe_exhaustions,
          &dynamic_code_cache_publish_probe_exhaustions,
          &dynamic_code_cache_lookup_probes,
          &dynamic_code_cache_max_lookup_probes)) {
      if (threadIdx.x == 0) {
        route_seed_failed = 1;
        route_failure_reason = static_cast<u32>(
          QueryFailureReason::dynamic_code_fetch);
      }
    }
    __syncthreads();
    if (route_seed_failed == 0) {
      for (u32 index = threadIdx.x; index < route_entry_count;
           index += blockDim.x) {
        merge_handles[index] = navigation_handles[index];
        merge_ids[index] = UINT32_MAX;
        merge_distances[index] = navigation_distances[index];
        merge_expanded[index] = 0;
      }
      __syncthreads();
      if (route_entry_count != 0) {
        sort_candidates(merge_handles, nullptr, merge_distances,
                        merge_expanded, route_entry_count);
      }
      if (threadIdx.x == 0) {
        const u32 initial_seed_capacity = min(
          route_entry_count, traversal_capacity);
        u32 unique_count = 0;
        for (u32 input = 0;
             input < route_entry_count &&
               unique_count < initial_seed_capacity;
             ++input) {
          const u64 handle = merge_handles[input];
          if (handle == kInvalidDeviceHandle ||
              !isfinite(merge_distances[input]) ||
              merge_distances[input] == FLT_MAX) {
            continue;
          }
          bool duplicate = false;
          for (u32 prior = 0; prior < unique_count; ++prior) {
            if (merge_handles[prior] == handle) {
              duplicate = true;
              break;
            }
          }
          if (duplicate) continue;
          if (unique_count != input) {
            merge_handles[unique_count] = handle;
            merge_distances[unique_count] = merge_distances[input];
            merge_expanded[unique_count] = 0;
          }
          ++unique_count;
        }
        beam_count = unique_count;
        for (u32 index = 0; index < beam_count; ++index) {
          beam_handles[index] = merge_handles[index];
          beam_ids[index] = UINT32_MAX;
          beam_distances[index] = merge_distances[index];
          beam_expanded[index] = 0;
          insert_visited(visited, params.visited_capacity,
                         beam_handles[index]);
        }
      }
      __syncthreads();
    }
    if (beam_count == 0) {
      if (route_attempt == 0) {
        if (threadIdx.x == 0) device_ring_relax(64);
        __syncthreads();
        continue;
      }
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.diagnostic = make_query_diagnostic(
          static_cast<QueryFailureReason>(route_failure_reason),
          route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

  __shared__ u64 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_beam_ranks[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  // The exact Stable-Run preview at the end of a round already identifies
  // the next authoritative commit prefix.  Carry that certificate across
  // the round boundary so the next iteration does not rescan the materialized
  // Beam on lane zero.  The certificate is query-local and is consumed only
  // after the preceding merge has passed its CTA barrier.
  __shared__ u32 next_commit_ready;
  __shared__ u32 next_commit_count;
  __shared__ u32 selection_from_certificate;
  __shared__ u64 critical_fetch_handles[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_slots[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_to_commit[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_destination_slots[kPersistentMaxPrefetch];
  __shared__ u8 selected_underhint_force_full[kPersistentMaxPrefetch];
  __shared__ u8 critical_fetch_force_full[kPersistentMaxPrefetch];
  // Tail completion has finished before underhint mapping starts. Reuse its
  // phase-local scalar for the uniform fast-path gate instead of growing the
  // persistent CTA's already occupancy-sensitive shared-memory footprint.
  union TailUnderhintPhaseScratch {
    u32 tail_stale_before;
    u32 selected_underhint_any;
  };
  __shared__ TailUnderhintPhaseScratch tail_underhint_scratch;
  __shared__ u32 commit_rob_slots[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_count;
  __shared__ u64 shadow_frontier_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 shadow_frontier_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 shadow_frontier_count;
  // 0: no reusable leaf state; 2: complete immutable leaves may be sealed as
  // ordered CQ/PQ progresses; 1: all four compact leaf slots (including a
  // final partial leaf and absent-leaf padding) are sorted; 3: only the first
  // compact leaf is ready and its remaining leaves are deferred past issue;
  // 5: an exact frontier certificate exists. The prepared-state flags record
  // whether complete leaves were produced eagerly or remain deferred past issue.
  __shared__ u32 stable_runs_prepared_before_issue;
  __shared__ u32 frontier_reusable_certificates;
  __shared__ u32 frontier_streamed_candidate_runs;
  __shared__ u32 ordered_score_batches;
  __shared__ u32 ordered_score_candidates;
  __shared__ u32 ooo_bypassed_parents;
  __shared__ u32 frontier_reusable_prefix_ranks;
  __shared__ u32 frontier_reusable_full_prefix_certificates;
  __shared__ u32 frontier_reusable_issued_certificates;
  __shared__ u32 frontier_certificate_rejects;
  __shared__ u32 issue_rob_slots[kPersistentFrontierRobCapacity];
  // Number of physical ROB slots covered by the current issue descriptor.
  // This is intentionally independent of the logical certificate width:
  // retained/promoted records can make the physical mapping sparse.
  __shared__ u32 physical_issue_span;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_record_slots[kPersistentMaxPrefetch];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_remote_batches;
  __shared__ u32 total_graph_read_retries;
  __shared__ u64 total_graph_read_bytes;
  __shared__ u32 total_graph_live_extent_reads;
  __shared__ u32 total_graph_full_record_reads;
  __shared__ u32 total_graph_extent_fallback_reads;
  __shared__ u32 total_graph_extent_underhint_reads;
  __shared__ u32 total_graph_extent_hint_promotions;
  __shared__ u32 total_graph_full_record_avoidance_hits;
  __shared__ u32 total_graph_rounds;
  __shared__ u32 graph_failed;
  __shared__ FrontierRobEntry frontier_rob[kPersistentFrontierRobCapacity];
  __shared__ FrontierGraphBatchState core_batch;
  __shared__ FrontierGraphBatchState tail_batch;
  __shared__ TerminalExactCacheState terminal_exact_cache;
  __shared__ u32 terminal_exact_horizon_ready;
  __shared__ TailFrontierFeedback tail_feedback;
  __shared__ u32 frontier_issue_epoch;
  __shared__ u32 logical_expansions;
  __shared__ u32 critical_graph_reads;
  __shared__ u64 critical_graph_bytes;
  __shared__ u32 speculative_graph_reads;
  __shared__ u64 speculative_graph_bytes;
  __shared__ u32 speculative_arrived;
  __shared__ u32 speculative_promoted;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 speculative_queue_rejects;
  __shared__ u32 issue_epochs;
  __shared__ u32 commit_epochs;
  __shared__ u64 issue_width_sum;
  __shared__ u64 issue_width_capacity_sum;
  __shared__ u64 commit_width_sum;
  __shared__ u32 max_issue_width;
  __shared__ u32 max_commit_width;
  __shared__ u32 critical_rob_hits;
  __shared__ u32 critical_misses;
  __shared__ u64 speculative_wait_cycles;
  __shared__ u64 rdma_completion_latency_ns;
  __shared__ u64 speculative_completion_latency_ns;
  __shared__ u64 rdma_completion_groups;
  __shared__ u64 speculative_completion_groups;
  __shared__ u64 core_prefetch_bytes;
  __shared__ u32 core_prefetch_reads;
  __shared__ u32 core_prefetch_arrived;
  __shared__ u32 core_prefetch_promoted;
  __shared__ u32 core_prefetch_stale;
  __shared__ u32 core_prefetch_queue_rejects;
  __shared__ u32 core_prefetch_waves;
  __shared__ u32 core_ready_waves;
  __shared__ DynamicGraphTelemetry dynamic_graph_telemetry;
  __shared__ CoreFrontierTelemetry core_telemetry;
  __shared__ TailFrontierTelemetry tail_telemetry;
  __shared__ TailAdmissionCorrection tail_admission_correction;
  __shared__ u64 shadow_issue_started_cycles;
  // Completion harvesting is independent of Beam rank. Candidate decode and
  // scoring deliberately remain one canonical full-width batch after every
  // mandatory parent has arrived; fragmenting that GPU work per CQ group is
  // substantially more expensive than the network wait it can hide.
  __shared__ u32 ooo_completed_parent_mask;
  __shared__ u32 early_queue_rejects_before;
  __shared__ u32 core_batch_positional;
  __shared__ u32 certified_mapping_ready;
  __shared__ u32 reconciled_positional_core;
  __shared__ u32 core_only_issue_epoch;
  __shared__ CertifiedCommitReconcileContext
    certified_commit_context;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_remote_batches = 0;
    total_graph_read_retries = 0;
    total_graph_read_bytes = 0;
    total_graph_live_extent_reads = 0;
    total_graph_full_record_reads = 0;
    total_graph_extent_fallback_reads = 0;
    total_graph_extent_underhint_reads = 0;
    total_graph_extent_hint_promotions = 0;
    total_graph_full_record_avoidance_hits = 0;
    total_graph_rounds = 0;
    graph_failed = 0;
    core_batch.active = 0;
    core_batch.fatal = 0;
    core_batch.rejected = 0;
    core_batch.finish_had_pending = 0;
    tail_batch.active = 0;
    tail_batch.fatal = 0;
    tail_batch.rejected = 0;
    tail_batch.finish_had_pending = 0;
    terminal_exact_cache = {};
    terminal_exact_horizon_ready = 0;
    next_commit_ready = 0;
    next_commit_count = 0;
    selection_from_certificate = 0;
    tail_underhint_scratch.selected_underhint_any = 0;
    stable_runs_prepared_before_issue = 0;
    ooo_completed_parent_mask = 0;
    frontier_reusable_certificates = 0;
    ooo_bypassed_parents = 0;
    if constexpr (EnableAsfe) {
      frontier_streamed_candidate_runs = 0;
      ordered_score_batches = 0;
      ordered_score_candidates = 0;
      frontier_reusable_prefix_ranks = 0;
      frontier_reusable_full_prefix_certificates = 0;
      frontier_reusable_issued_certificates = 0;
      frontier_certificate_rejects = 0;
      // The controller is owned by the persistent CTA and deliberately
      // survives this function. Advance its collapsed re-probe cadence once
      // per valid query instead of restarting at commit+1 per submission.
      adaptive_frontier::begin_query(frontier_controller);
    }
    tail_feedback = {};
    frontier_issue_epoch = 0;
    logical_expansions = 0;
    critical_graph_reads = 0;
    critical_graph_bytes = 0;
    speculative_graph_reads = 0;
    speculative_graph_bytes = 0;
    speculative_arrived = 0;
    speculative_promoted = 0;
    speculative_stale = 0;
    speculative_wasted_bytes = 0;
    speculative_queue_rejects = 0;
    issue_epochs = 0;
    commit_epochs = 0;
    issue_width_sum = 0;
    issue_width_capacity_sum = 0;
    commit_width_sum = 0;
    max_issue_width = 0;
    max_commit_width = 0;
    critical_rob_hits = 0;
    critical_misses = 0;
    speculative_wait_cycles = 0;
    rdma_completion_latency_ns = 0;
    speculative_completion_latency_ns = 0;
    rdma_completion_groups = 0;
    speculative_completion_groups = 0;
    core_prefetch_bytes = 0;
    core_prefetch_reads = 0;
    core_prefetch_arrived = 0;
    core_prefetch_promoted = 0;
    core_prefetch_stale = 0;
    core_prefetch_queue_rejects = 0;
    core_prefetch_waves = 0;
    core_ready_waves = 0;
    dynamic_graph_telemetry = {};
    dynamic_graph_telemetry.full_record_avoidance_hits = &total_graph_full_record_avoidance_hits;
    core_batch_positional = 0;
    certified_mapping_ready = 0;
    reconciled_positional_core = 0;
    core_only_issue_epoch = 0;
    physical_issue_span = 0;
    core_telemetry = CoreFrontierTelemetry{
      .wait_cycles = &speculative_wait_cycles,
      .completion_latency_ns = &rdma_completion_latency_ns,
      .completion_groups = &rdma_completion_groups,
      .arrived = &core_prefetch_arrived,
      .stale = &core_prefetch_stale,
      .ready_waves = &core_ready_waves,
      .dynamic_graph = &dynamic_graph_telemetry,
    };
    tail_telemetry = TailFrontierTelemetry{
      .arrived = &speculative_arrived,
      .stale = &speculative_stale,
      .wait_cycles = &speculative_wait_cycles,
      .completion_latency_ns = &speculative_completion_latency_ns,
      .completion_groups = &speculative_completion_groups,
      .wasted_bytes = &speculative_wasted_bytes,
      .admission_correction = &tail_admission_correction,
      .dynamic_graph = &dynamic_graph_telemetry,
    };
    tail_admission_correction = {};
    certified_commit_context = CertifiedCommitReconcileContext{
      .selected_handles = selected_handles,
      .certified_rob_slots = issue_rob_slots,
      .frontier_rob = frontier_rob,
      .commit_rob_slots = commit_rob_slots,
      .graph_record_slots = graph_record_slots,
      .critical_fetch_handles = critical_fetch_handles,
      .critical_fetch_to_commit = critical_fetch_to_commit,
      .critical_fetch_count = &critical_fetch_count,
      .critical_rob_hits = &critical_rob_hits,
      .critical_misses = &critical_misses,
      .speculative_promoted = &speculative_promoted,
      .core_prefetch_promoted = &core_prefetch_promoted,
      .core_prefetch_stale = &core_prefetch_stale,
      .feedback = &tail_feedback,
    };
  }
  if constexpr (EnableAsfe) {
    for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
         shard += blockDim.x) {
      core_batch.issue_timestamp_ns[shard] = 0;
      tail_batch.issue_timestamp_ns[shard] = 0;
    }
    for (u32 slot = threadIdx.x;
         slot < kPersistentFrontierRobCapacity;
         slot += blockDim.x) {
      frontier_rob[slot] = {};
    }
  }
  // Fixed-record policy leaves graph_request_bytes null, so no asynchronous
  // short read can produce extent-underhint evidence. Avoid even the one-time
  // flag stores for that baseline; the fixed path also skips both per-round
  // helper barriers below and passes no force-full vector to the fetcher.
  if (params.graph_request_bytes != nullptr) {
    for (u32 slot = threadIdx.x; slot < kPersistentMaxPrefetch;
         slot += blockDim.x) {
      selected_underhint_force_full[slot] = 0;
      critical_fetch_force_full[slot] = 0;
    }
  }
  __syncthreads();

  __shared__ u32 expansions;
  if (threadIdx.x == 0) expansions = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    completion.prepare_cycles = clock64() - prepare_started_cycles;
  }
  __syncthreads();
  while (expansions < params.max_expansions) {
    __shared__ u64 speculative_wait_before;
    if constexpr (EnableAsfe) {
      // Freeze the authoritative commit prefix before consuming any payload.
      // Completed shard groups may retire ROB entries in any order, but
      // candidate decode, visited, PQ, Beam, and expanded bits remain
      // unchanged until the complete dependency set has been resolved.
      if (threadIdx.x == 0) {
        phase_started_cycles = clock64();
        graph_failed = 0;
        // A candidate is valid for exactly one preview/materialization pair.
        // Clear it even on rounds that cannot construct an exact preview.
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        selection_from_certificate = 0;
        const bool certificate_candidate =
          next_commit_ready != 0 &&
          next_commit_count != 0 &&
          next_commit_count <=
            min(params.commit_width, params.max_expansions - expansions);
        // The certificate and Beam publication are the two outputs of the
        // same exact Stable-Run inputs inside this CTA. finish() completes
        // before the loop-back barrier, and no other actor can mutate this
        // query-local Beam. Re-scanning up to K ranks here duplicated the
        // merge proof serially on lane zero. Bitwise equivalence is enforced
        // by the focused DEEC/PBEC/PFEC and publish tests instead.
        const bool certificate_valid = certificate_candidate;
        if (certificate_valid) {
          // The preview is an exact stable merge certificate.  Its first
          // commit-width entries are therefore the same handles/ranks that a
          // fresh scan of the just-materialized authoritative Beam would
          // select.  Keep the rank mapping and reset ROB destinations here;
          // plan_commit_frontier() will reconcile any retained tail below.
          selected_count = next_commit_count;
          shadow_frontier_count = 0;
          for (u32 position = 0; position < selected_count; ++position) {
            commit_rob_slots[position] = UINT32_MAX;
          }
          ++commit_epochs;
          commit_width_sum += selected_count;
          max_commit_width = max(max_commit_width, selected_count);
          selection_from_certificate = 1;
          next_commit_ready = 0;
          next_commit_count = 0;
        } else {
          if (certificate_candidate) {
            ++frontier_certificate_rejects;
          }
          next_commit_ready = 0;
          next_commit_count = 0;
        }
      }
      __syncthreads();
      if (selection_from_certificate == 0) {
        select_commit_frontier(
          beam_handles, beam_expanded, beam_count,
          min(params.commit_width, params.max_expansions - expansions),
          selected_handles, selected_beam_ranks, commit_rob_slots,
          selected_count, shadow_frontier_count, commit_epochs,
          commit_width_sum, max_commit_width);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        beam_selection_cycles += clock64() - phase_started_cycles;
        core_batch.rejected = 0;
        speculative_wait_before = speculative_wait_cycles;
        ooo_completed_parent_mask = 0;
        stable_runs_prepared_before_issue = 0;
      }
      __syncthreads();
      const u32 single_chunk_parent_capacity =
        persistent_score_chunk_capacity(
          params.graph_entry_capacity, traversal_capacity);
      const bool canonical_core_mapping_ready =
        core_batch_positional != 0 ||
        (selection_from_certificate != 0 &&
         certified_mapping_ready != 0);
      const bool ooo_pipeline_enabled =
        core_batch.active != 0 && canonical_core_mapping_ready &&
        selected_count != 0 &&
        selected_count <= single_chunk_parent_capacity &&
        blockDim.x == kApproximateSortThreadsCompact;
      const bool frontier_stable_run_enabled =
        ooo_pipeline_enabled &&
        EnableAsfe &&
        params.beam_merge_policy ==
          static_cast<u32>(BeamMergePolicy::stable_run) &&
        expansions + selected_count < params.max_expansions;
      if (frontier_stable_run_enabled) {
        if (threadIdx.x == 0) {
          stable_merge_state = {};
          stable_merge_state.original_count = beam_count;
          stable_merge_state.compact = 1;
          stable_merge_state.origin_copied = 0;
          stable_merge_state.phase_started = clock64();
          beam_merge_round_breakdown = {};
          stable_runs_prepared_before_issue = 2;
        }
        __syncthreads();
      }
      if (ooo_pipeline_enabled) {
        constexpr u32 completion_warp_width = 32;
        const u32 completion_lane =
          threadIdx.x % completion_warp_width;

        while (core_batch.active != 0 && graph_failed == 0) {
          const i32 completion_progress =
            finish_next_core_frontier_group(
              params, descriptor, frontier_rob, core_batch,
              min(params.commit_width,
                  static_cast<u32>(kPersistentFrontierRobCapacity)),
              core_telemetry, true);
          if (completion_progress < 0) {
            if (threadIdx.x == 0) {
              graph_failed = frontier_graph_failure_code(
                params, descriptor, 1u);
            }
            __syncthreads();
            break;
          }

          // The completion helper returns at a CTA barrier. One warp can now
          // retire every newly VALIDATED logical parent without another
          // block-wide readiness scan or publication barrier. Other warps
          // rendezvous at the helper's first barrier in the next iteration.
          if (threadIdx.x < completion_warp_width) {
            const u32 position = completion_lane;
            const u32 rob_slot = position < selected_count
              ? (core_batch_positional != 0
                   ? position : issue_rob_slots[position])
              : UINT32_MAX;
            FrontierRobEntry* entry =
              rob_slot < kPersistentFrontierRobCapacity
                ? frontier_rob + rob_slot : nullptr;
            const bool ready =
              position < selected_count &&
              (ooo_completed_parent_mask &
                 (u32{1} << position)) == 0 &&
              entry != nullptr &&
              entry->state ==
                static_cast<u8>(FrontierRequestState::validated) &&
              entry->node_handle == selected_handles[position];
            const u32 ready_parent_mask =
              __ballot_sync(0xffffffffu, ready);
            if (ready) {
              entry->state =
                static_cast<u8>(FrontierRequestState::committed);
            }
            if (completion_lane == 0) {
              const u32 selected_mask =
                selected_count == completion_warp_width
                  ? 0xffffffffu : (u32{1} << selected_count) - 1u;
              const u32 not_ready_mask =
                selected_mask & ~ooo_completed_parent_mask &
                ~ready_parent_mask;
              if (not_ready_mask != 0) {
                const u32 first_hole =
                  static_cast<u32>(__ffs(not_ready_mask) - 1);
                const u32 prefix_through_hole =
                  first_hole + 1u == completion_warp_width
                    ? 0xffffffffu
                    : (u32{1} << (first_hole + 1u)) - 1u;
                ooo_bypassed_parents += __popc(
                  ready_parent_mask & ~prefix_through_hole);
              }
              ooo_completed_parent_mask |= ready_parent_mask;
            }
          }
        }
        // The final completion clears core_batch.active, so there is no next
        // helper-entry barrier at which non-owner warps can rendezvous.
        __syncthreads();
      }

      if (!ooo_pipeline_enabled &&
          !finish_query_core_frontier_batch(
                   params, descriptor, frontier_rob, core_batch,
                   min(params.commit_width,
                       static_cast<u32>(
                         kPersistentFrontierRobCapacity)),
                   core_telemetry)) {
        if (threadIdx.x == 0) {
          graph_failed = frontier_graph_failure_code(
            params, descriptor, 3u);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        rdma_wait_cycles +=
          speculative_wait_cycles - speculative_wait_before;
        speculative_wait_before = speculative_wait_cycles;
      }
      __syncthreads();
      // The tail state is CTA-resident and therefore a uniform predicate.
      // Skip the out-of-line completion frame entirely when no speculative
      // descriptor is active; this is the steady state after the controller
      // collapses an unprofitable suffix.
      if (tail_batch.active != 0) {
        if (threadIdx.x == 0) {
          tail_underhint_scratch.tail_stale_before = speculative_stale;
        }
        __syncthreads();
        if (!finish_query_tail_frontier_batch<false>(
              params, descriptor, frontier_rob, tail_batch,
              tail_telemetry)) {
          if (threadIdx.x == 0) graph_failed = 4u;
        }
        apply_tail_admission_correction(
          tail_admission_correction, tail_feedback,
          total_remote_batches, total_remote_reads,
          total_graph_read_bytes, total_graph_live_extent_reads,
          total_graph_full_record_reads, speculative_graph_reads,
          speculative_graph_bytes, speculative_queue_rejects);
        // Every active path through finish_query_tail_frontier_batch ends at
        // a CTA barrier, so lane zero can consume its counters immediately.
        if (threadIdx.x == 0) {
          const u64 tail_wait_delta =
            speculative_wait_cycles - speculative_wait_before;
          rdma_wait_cycles += tail_wait_delta;
          speculative_wait_before = speculative_wait_cycles;
          tail_feedback.stale +=
            speculative_stale -
              tail_underhint_scratch.tail_stale_before;
        }
      }
      __syncthreads();

      if (params.graph_request_bytes != nullptr) {
        const UnderhintLookupMode underhint_lookup_mode =
          core_batch_positional != 0
            ? UnderhintLookupMode::positional
            : ((selection_from_certificate != 0 &&
                certified_mapping_ready != 0)
                 ? UnderhintLookupMode::certified
                 : UnderhintLookupMode::associative);
        identify_selected_underhint_force_full(
          selected_handles, selected_count, frontier_rob, issue_rob_slots,
          underhint_lookup_mode, selected_underhint_force_full,
          &tail_underhint_scratch.selected_underhint_any);
      }

      if (core_batch_positional != 0) {
        reconcile_exact_core_frontier(
          selected_handles, selected_count, frontier_rob,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          commit_rob_slots, graph_record_slots,
          critical_fetch_handles, critical_fetch_to_commit,
          critical_fetch_destination_slots, critical_fetch_count,
          critical_rob_hits, critical_misses,
          core_prefetch_promoted, core_prefetch_stale);
        if (threadIdx.x == 0) {
          reconciled_positional_core = 1;
          core_batch_positional = 0;
          certified_mapping_ready = 0;
        }
      } else if (selection_from_certificate != 0 &&
                 certified_mapping_ready != 0) {
        reconcile_certified_commit_frontier(
          certified_commit_context, selected_count,
          issue_epochs != 0);
        if (threadIdx.x == 0) {
          // Arbitrary retained slots still use the ordinary free-slot
          // allocator below for any mandatory critical retry.
          reconciled_positional_core = 0;
          certified_mapping_ready = 0;
        }
      } else {
        plan_commit_frontier(
          beam_handles, beam_expanded, beam_count,
          min(params.commit_width, params.max_expansions - expansions),
          frontier_rob, frontier_controller, issue_epochs != 0,
          0, 0, selected_handles, selected_beam_ranks,
          commit_rob_slots, selected_count,
          critical_fetch_handles, critical_fetch_to_commit,
          critical_fetch_count, graph_record_slots,
          shadow_frontier_count, speculative_stale,
          speculative_wasted_bytes, speculative_promoted,
          core_prefetch_stale, core_prefetch_promoted,
          critical_rob_hits, critical_misses,
          commit_epochs, commit_width_sum, max_commit_width,
          tail_feedback, true);
        if (threadIdx.x == 0) {
          reconciled_positional_core = 0;
          certified_mapping_ready = 0;
        }
      }
      __syncthreads();

      if (reconciled_positional_core == 0) {
        reserve_critical_fetch_destinations(
          frontier_rob, critical_fetch_count,
          critical_fetch_destination_slots, graph_failed,
          speculative_stale, speculative_wasted_bytes,
          core_prefetch_stale, tail_feedback);
      }
      __syncthreads();
    } else if (threadIdx.x == 0) {
      graph_failed = 0;
      selected_count = 0;
      const u32 target =
        min(params.commit_width, params.max_expansions - expansions);
      for (u32 index = 0;
           index < beam_count && selected_count < target; ++index) {
        if (beam_expanded[index] != 0) continue;
        const u32 position = selected_count++;
        selected_handles[position] = beam_handles[index];
        selected_beam_ranks[position] = index;
        critical_fetch_handles[position] = beam_handles[index];
        critical_fetch_to_commit[position] = position;
      }
      critical_fetch_count = selected_count;
      if (selected_count != 0) {
        ++commit_epochs;
        commit_width_sum += selected_count;
        max_commit_width = max(max_commit_width, selected_count);
        critical_misses += selected_count;
      }
    }
    if constexpr (!EnableAsfe) {
      for (u32 fetch = threadIdx.x; fetch < critical_fetch_count;
           fetch += blockDim.x) {
        critical_fetch_destination_slots[fetch] =
          critical_fetch_to_commit[fetch];
      }
    }
    __syncthreads();
    if (params.graph_request_bytes != nullptr &&
        tail_underhint_scratch.selected_underhint_any != 0) {
      remap_critical_underhint_force_full(
        selected_underhint_force_full, selected_count,
        critical_fetch_to_commit, critical_fetch_count,
        critical_fetch_force_full);
    }
    if (threadIdx.x == 0) {
      phase_started_cycles = clock64();
    }
    if (selected_count == 0) break;
    if (threadIdx.x == 0) ++total_graph_rounds;
    __syncthreads();
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    __shared__ u64 critical_bytes_before;
    if (threadIdx.x == 0) {
      graph_fetch_breakdown = {};
      critical_bytes_before = total_graph_read_bytes;
    }
    __syncthreads();
    if (critical_fetch_count != 0 && graph_failed == 0) {
      const u32 fetch_failure = fetch_graph_records_batch(
            params, descriptor, critical_fetch_handles,
            critical_fetch_count, critical_fetch_slots,
            critical_fetch_destination_slots,
            remote_reads_by_lane, &total_remote_batches,
            &total_graph_read_retries, &total_graph_read_bytes,
            &total_graph_live_extent_reads,
            &total_graph_full_record_reads,
            &total_graph_extent_fallback_reads,
            &total_graph_extent_underhint_reads,
            &total_graph_extent_hint_promotions,
            route_attempt, total_graph_rounds - 1,
            rdma_trace_enabled != 0, &graph_fetch_breakdown,
            &dynamic_graph_telemetry,
            params.graph_request_bytes != nullptr &&
                tail_underhint_scratch.selected_underhint_any != 0
              ? critical_fetch_force_full : nullptr);
      if (fetch_failure != 0) {
        if (threadIdx.x == 0) {
          const u32 contextual_failure =
            fetch_failure |
            (selection_from_certificate != 0 &&
                 (fetch_failure & (u32{1} << 16)) == 0
               ? u32{1} << 15 : 0u);
          graph_failed = 6u | (contextual_failure << 8);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rdma_issue_cycles += graph_fetch_breakdown.issue;
      rdma_wait_cycles += graph_fetch_breakdown.wait;
      graph_validation_cycles += graph_fetch_breakdown.validation;
      rdma_completion_latency_ns +=
        graph_fetch_breakdown.completion_latency_ns;
      rdma_completion_groups += graph_fetch_breakdown.completion_groups;
      graph_phase_cycles += clock64() - phase_started_cycles;
      critical_graph_bytes +=
        total_graph_read_bytes - critical_bytes_before;
      for (u32 selected = 0; selected < critical_fetch_count; ++selected) {
        // fetch_graph_records_batch may use upper bits as private per-parent
        // retry metadata. Bit zero alone denotes the one logical graph read
        // selected by the search algorithm.
        total_remote_reads += remote_reads_by_lane[selected] & 1u;
        critical_graph_reads += remote_reads_by_lane[selected] & 1u;
        graph_record_slots[critical_fetch_to_commit[selected]] =
          critical_fetch_slots[selected];
      }
    }
    __syncthreads();
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        if (lane_in_warp == 0) graph_record_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      // A failed critical fetch may race with an already-admitted ASFE wave.
      // The query slot (and its graph scratch) cannot be reused until both
      // owner descriptors have been synchronously drained.  This mirrors the
      // normal end-of-query cleanup and closes the late-DMA overwrite window.
      if constexpr (EnableAsfe) {
        (void)finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry);
        if (tail_batch.active != 0) {
          (void)finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch, tail_telemetry);
          apply_tail_admission_correction(
            tail_admission_correction, tail_feedback,
            total_remote_batches, total_remote_reads,
            total_graph_read_bytes, total_graph_live_extent_reads,
            total_graph_full_record_reads, speculative_graph_reads,
            speculative_graph_bytes, speculative_queue_rejects);
        }
        __syncthreads();
        // A terminal exact train is launched at the end of the preceding
        // epoch and intentionally remains active while this epoch advances.
        // Therefore this early authoritative-fetch failure can observe an
        // older terminal train even though it occurs before this epoch's
        // terminal-horizon code below. The completion releases query_slot;
        // synchronously retire that train first so its full-record/header
        // DMA cannot land in the next incarnation of the slot.
        drain_terminal_exact_cache_prefetch(
          params, descriptor, terminal_exact_cache);
      }
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.result_count = graph_failed;
        completion.diagnostic = make_query_diagnostic(
          QueryFailureReason::graph_fetch, route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_read_bytes = total_graph_read_bytes;
        completion.graph_full_record_avoidance_hits = total_graph_full_record_avoidance_hits;
        completion.graph_live_extent_reads =
          total_graph_live_extent_reads;
        completion.graph_full_record_reads =
          total_graph_full_record_reads;
        completion.graph_extent_fallback_reads =
          total_graph_extent_fallback_reads;
        completion.graph_extent_underhint_reads =
          total_graph_extent_underhint_reads;
        completion.graph_extent_hint_promotions =
          total_graph_extent_hint_promotions;
        set_dynamic_graph_completion(
          completion, dynamic_graph_telemetry,
          critical_graph_reads + speculative_graph_reads,
          params.graph_read_policy);
        set_frontier_completion<EnableAsfe>(
          completion, logical_expansions, critical_graph_reads,
          critical_graph_bytes, speculative_graph_reads,
          speculative_graph_bytes, speculative_arrived,
          speculative_promoted, speculative_stale,
          speculative_wasted_bytes, speculative_queue_rejects,
          issue_epochs, commit_epochs, issue_width_sum,
          issue_width_capacity_sum, commit_width_sum, max_issue_width,
          max_commit_width, critical_rob_hits, critical_misses,
          speculative_wait_cycles,
          rdma_completion_latency_ns, speculative_completion_latency_ns,
          rdma_completion_groups, speculative_completion_groups,
          core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
          core_prefetch_promoted, core_prefetch_stale,
          core_prefetch_queue_rejects, core_prefetch_waves,
          core_ready_waves);
        set_frontier_certificate_completion<EnableAsfe>(
          completion, frontier_reusable_certificates,
          frontier_streamed_candidate_runs, ordered_score_batches,
          ordered_score_candidates, frontier_reusable_prefix_ranks,
          frontier_reusable_full_prefix_certificates,
          frontier_reusable_issued_certificates,
          ooo_bypassed_parents,
          frontier_certificate_rejects);
        // Failure-only provenance: preserve the exact handle rejected by the
        // authoritative fetch preparer.  The two reserved words normally
        // carry no ABI-visible result data; on a failed query they let the
        // host distinguish a malformed Beam value from a bad destination or
        // transport error without adding any work to successful queries.
        const u32 fetch_detail = graph_failed >> 8;
        if ((fetch_detail & (u32{1} << 16)) == 0) {
          const u32 failed_index = (fetch_detail >> 4) & 0x1fu;
          if (failed_index < critical_fetch_count) {
            const u64 failed_handle = critical_fetch_handles[failed_index];
            completion.frontier_telemetry_reserved0 =
              static_cast<u32>(failed_handle);
            completion.frontier_telemetry_reserved1 =
              static_cast<u32>(failed_handle >> 32);
          }
        }
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_terminal_exact_cache_completion<EnableAsfe>(
          completion, terminal_exact_cache);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    // Commit becomes authoritative only after every missing dependency has
    // completed the full critical Live-Extent/retry path. The frozen Beam
    // ranks preserve the legacy whole-batch expansion order.
    if (threadIdx.x == 0 && graph_failed == 0) {
      for (u32 position = 0; position < selected_count; ++position) {
        beam_expanded[selected_beam_ranks[position]] = 1;
      }
      phase_started_cycles = clock64();
    }
    __syncthreads();
    const u32 score_chunk_capacity = persistent_score_chunk_capacity(
      params.graph_entry_capacity, traversal_capacity);
    if (score_chunk_capacity == 0) {
      if (threadIdx.x == 0) graph_failed = 7u;
      __syncthreads();
    }
    for (u32 chunk_begin = 0;
         graph_failed == 0 && chunk_begin < selected_count;
         chunk_begin += score_chunk_capacity) {
      const u32 chunk_count = min(score_chunk_capacity,
                                  selected_count - chunk_begin);
      const bool final_parent_chunk =
        chunk_begin + chunk_count >= selected_count;
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_record_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        if (lane_in_warp == 0) {
          const u32 stable_count = record == nullptr ? 0 : record[0];
          const u32 provisional_count = record == nullptr
            ? 0 : (record[1] >> 4) & 0xfu;
          neighbor_counts[local] =
            record != nullptr && (record[1] & 1u) == 0
              ? min(stable_count + provisional_count,
                    params.graph_entry_capacity)
              : 0;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_offsets[0] = 0;
        for (u32 local = 0; local < chunk_count; ++local) {
          const u32 count = neighbor_counts[local];
          const u32 extent_class = min(
            (count + 7u) / 8u, kGraphDegreeHistogramBuckets - 1u);
          ++completion.expanded_parent_count;
          completion.expanded_neighbor_count_sum += count;
          ++completion.expanded_degree_histogram[extent_class];
          // selected_handles names the exact, already validated version that
          // is about to be expanded.  Record its true degree after the read;
          // this is observational Oracle telemetry and is never consulted by
          // request preparation.
          if (dynamic_graph_telemetry_handle(
                selected_handles[chunk_begin + local])) {
            ++completion.dynamic_expanded_parent_count;
            completion.dynamic_expanded_neighbor_count_sum += count;
            ++completion.dynamic_expanded_degree_histogram[extent_class];
          }
          neighbor_offsets[local + 1] =
            neighbor_offsets[local] + count;
        }
        flattened_neighbors = neighbor_offsets[chunk_count];
        phase_started_cycles = clock64();
      }
      __syncthreads();
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_record_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        __syncwarp();
        const u32 count = neighbor_counts[local];
        for (u32 neighbor = lane_in_warp; neighbor < count;
             neighbor += warp_width) {
          const u64 raw = decode_tagged_raw(
            record + 16 + neighbor * sizeof(u64));
          navigation_handles[neighbor_offsets[local] + neighbor] =
            handle_from_raw(params, raw);
        }
        __syncwarp();
        if (lane_in_warp == 0) {
          graph_record_slots[selected] = UINT32_MAX;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_decode_cycles += clock64() - phase_started_cycles;
      }
      if constexpr (EnableAsfe) {
        if (final_parent_chunk) {
          // Every committed payload has now been copied into the flat
          // navigation stream. Release its ROB/scratch ownership before the
          // candidate Stable-Run launches the next shadow wave; the RDMA
          // destination may then be overwritten because visited/PQ consume
          // only the copied handles.
          for (u32 slot = threadIdx.x;
               slot < kPersistentFrontierRobCapacity;
               slot += blockDim.x) {
            if (frontier_rob[slot].state ==
                static_cast<u8>(FrontierRequestState::committed)) {
              frontier_rob[slot] = {};
            }
          }
        }
        __syncthreads();
      }
      const u32 candidate_count = flattened_neighbors;
      if (candidate_count != 0) {
        if (threadIdx.x == 0) phase_started_cycles = clock64();
        __syncthreads();
        for (u32 flat = threadIdx.x;
             flat < candidate_count; flat += blockDim.x) {
          const u64 handle = navigation_handles[flat];
          if (handle == kInvalidDeviceHandle ||
              !insert_visited(visited, params.visited_capacity, handle)) {
            navigation_handles[flat] = kInvalidDeviceHandle;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          visited_cycles += clock64() - phase_started_cycles;
        }
      }
      if constexpr (EnableAsfe) {
        if (threadIdx.x == 0 && candidate_count != 0) {
          // All speculative/critical graph dependencies have been reconciled
          // and every mandatory miss has completed before this point.
          ++completion.completion_score_batches;
          completion.completion_score_candidates += candidate_count;
        }
      }
      if (threadIdx.x == 0) phase_started_cycles = clock64();
      if (candidate_count != 0 &&
          !approximate_handles_batch<PqSubquantizers>(
            params, descriptor, query_lut,
            navigation_handles, candidate_count,
            navigation_distances,
            &dynamic_code_cycles,
            &dynamic_code_candidates, &dynamic_code_reads,
            &dynamic_code_incarnation_rejects, &dynamic_code_cache_hits,
            &dynamic_code_batch_deduplicated,
            &dynamic_code_cache_publish_successes,
            &dynamic_code_cache_first_occupancies,
            &dynamic_code_cache_publish_races,
            &dynamic_code_cache_lookup_probe_exhaustions,
            &dynamic_code_cache_publish_probe_exhaustions,
            &dynamic_code_cache_lookup_probes,
            &dynamic_code_cache_max_lookup_probes)) {
        if (threadIdx.x == 0) graph_failed = 8u;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        if (candidate_count != 0) {
          pq_score_cycles += clock64() - phase_started_cycles;
        }
        score_phase_cycles = neighbor_decode_cycles +
          visited_cycles + pq_score_cycles;
        phase_started_cycles = clock64();
        // rerank_count is otherwise dead throughout traversal. Preserve the
        // non-empty old Beam cardinality across this merge so an impossible
        // topK(old Beam U candidates) -> empty transition can be attributed
        // to the exact Stable-Run publication path that produced it.
        rerank_count = beam_count;
      }
      __syncthreads();
      if (graph_failed != 0) break;
      u64* candidate_input_handles = navigation_handles;
      f32* candidate_input_distances = navigation_distances;
      const bool asfe_stable_run =
        EnableAsfe &&
        blockDim.x == kApproximateSortThreadsCompact &&
        params.beam_merge_policy ==
          static_cast<u32>(BeamMergePolicy::stable_run);
      if (stable_runs_prepared_before_issue == 3) {
        if (threadIdx.x == 0) {
          // Exclude descriptor publication from Stable-Run accounting; only
          // the mandatory remaining leaf work belongs to Beam merge.
          stable_merge_state.phase_started = clock64();
        }
        __syncthreads();
        complete_compact_approximate_stable_runs(
          candidate_input_handles, candidate_input_distances,
          candidate_count, traversal_capacity,
          rerank_handles, rerank_flags, rerank_distances,
          candidate_workspace, stable_merge_state,
          &beam_merge_round_breakdown, false);
        if (threadIdx.x == 0) {
          stable_runs_prepared_before_issue = 1;
        }
        __syncthreads();
      }
      if (stable_runs_prepared_before_issue == 2) {
        constexpr u32 stable_leaf_capacity =
          kApproximateSortThreadsCompact *
          kApproximateSortItemsCompactFinal256;
        // Sort every complete/partial canonical leaf after visited and
        // dynamic PQ have resolved. An absent leaf is the identity
        // (+infinity) run, so retain the logical arity instead of writing and
        // later merging its 128 sentinel entries.
        const u32 final_candidate_run_count = min(
          4u,
          (candidate_count + stable_leaf_capacity - 1u) /
            stable_leaf_capacity);
        while (stable_merge_state.candidate_run_count <
               final_candidate_run_count) {
          const u32 pass =
            stable_merge_state.candidate_run_count;
          const u32 input_offset =
            pass * stable_leaf_capacity;
          const u32 output_offset =
            pass * traversal_capacity;
          if (threadIdx.x == 0) {
            stable_merge_state.phase_started = clock64();
          }
          __syncthreads();
          stable_sort_candidate_run(
            candidate_workspace.sort,
            candidate_input_handles, candidate_input_distances,
            candidate_count, input_offset,
            rerank_handles, rerank_flags, rerank_distances,
            output_offset, traversal_capacity,
            &beam_merge_round_breakdown,
            &stable_merge_state.phase_started);
          if (threadIdx.x == 0) {
            ++stable_merge_state.candidate_run_count;
            ++frontier_streamed_candidate_runs;
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          stable_merge_state.prepared = 1;
          stable_runs_prepared_before_issue = 1;
        }
        __syncthreads();
      }
      const bool exact_frontier_issue_enabled =
        asfe_stable_run && final_parent_chunk &&
        expansions + selected_count < params.max_expansions;
      if (exact_frontier_issue_enabled) {
        const u32 remaining_search_budget =
          params.max_expansions - (expansions + selected_count);
        if (remaining_search_budget != 0 &&
            core_batch.active == 0) {
          // There is no future Commit Frontier when this round consumes the
          // remaining expansion budget. Do not issue a tail that would only
          // be synchronously drained on query completion.
          const u32 issue_capacity_limit =
            remaining_search_budget > params.commit_width
              ? frontier_controller.current_issue_width
              : params.commit_width;
          const u32 preview_capacity = min(
            issue_capacity_limit, remaining_search_budget);
          const u32 certificate_target = min(
            params.commit_width, remaining_search_budget);

          if (threadIdx.x == 0) {
            shadow_issue_started_cycles = clock64();
          }
          __syncthreads();
          if (stable_runs_prepared_before_issue == 0) {
            // PFEC sorts the four immutable leaves once, emits an exact
            // certificate, and retains the reusable private merge prefix.
            // Beam/expanded/visited remain untouched until finish below.
            // The leaf sort is register-only, so no shared storage overlay
            // with the candidate workspace is needed here.
            prepare_warp_leaf_fused_frontier_certificate(
              candidate_input_handles, candidate_input_distances,
              candidate_count, beam_handles, beam_distances,
              beam_expanded, beam_count, traversal_capacity,
              rerank_handles, rerank_flags, rerank_distances,
              candidate_workspace.arrays,
              preview_capacity, shadow_frontier_handles,
              shadow_frontier_ranks, shadow_frontier_count,
              stable_merge_state, &beam_merge_round_breakdown);
            if (threadIdx.x == 0) {
              stable_runs_prepared_before_issue = 5;
              frontier_streamed_candidate_runs += 4;
            }
            __syncthreads();
          } else {
            prepare_deferred_fused_frontier_certificate(
              beam_handles, beam_distances, beam_expanded,
              traversal_capacity, stable_merge_state.original_count,
              rerank_handles, rerank_flags, rerank_distances,
              stable_merge_state.candidate_run_count,
              candidate_workspace.arrays, preview_capacity,
              shadow_frontier_handles, shadow_frontier_ranks,
              shadow_frontier_count, stable_merge_state, nullptr);
          }
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_preview_cycles +=
              now - shadow_issue_started_cycles;
            shadow_issue_started_cycles = now;
            ++frontier_reusable_certificates;
            frontier_reusable_prefix_ranks +=
              shadow_frontier_count;
            // Certificate construction is complete. Start communication
            // accounting only after its exact frontier exists.
            frontier_subphase_started_cycles = now;
          }
          // Cache the next authoritative commit prefix before ROB admission
          // compacts or rejects the speculative suffix.  The preview is over
          // the exact four-run merge and therefore remains valid even when
          // the tail is throttled; it is consumed only after this round's
          // materialization barrier at the top of the next round.
          if (threadIdx.x == 0) {
            if (shadow_frontier_count >= certificate_target &&
                certificate_target != 0) {
              for (u32 position = 0; position < certificate_target;
                   ++position) {
                selected_handles[position] =
                  shadow_frontier_handles[position];
                selected_beam_ranks[position] =
                  static_cast<u32>(shadow_frontier_ranks[position]);
                commit_rob_slots[position] = UINT32_MAX;
              }
              next_commit_count = certificate_target;
              next_commit_ready = 1;
            } else {
              next_commit_count = 0;
              next_commit_ready = 0;
            }
          }
          // Tail completion is consumed once, immediately before the next
          // Commit Frontier is reconciled.  A second end-of-round poll used
          // to scan the same per-shard words solely to enable immediate tail
          // refill.  Leaving an unfinished tail resident is both safe (core
          // metadata/scratch are disjoint) and a direct, latency-derived
          // admission throttle: the next exact critical prefix can still be
          // issued, while no further speculative wave is admitted until the
          // current one has actually completed.
          if (shadow_frontier_count != 0) {
            // Do not round a valid shadow suffix down to a full Commit
            // Frontier.  A one-entry probe is intentional: it gives the
            // online controller a query-local promotion/staleness sample and
            // lets the owner pipeline hide even a small amount of latency.
            // The controller, rather than a width heuristic, decides whether
            // subsequent suffixes grow or shrink.
            __syncthreads();
            const u32 positional_core_width = min(
              params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity));
            if (threadIdx.x < 32) {
              const u32 lane = threadIdx.x;
              const bool live_tail =
                lane >= positional_core_width &&
                frontier_rob[lane].state !=
                  static_cast<u8>(FrontierRequestState::init);
              const u32 live_tail_mask =
                __ballot_sync(0xffffffffu, live_tail);
              if (lane == 0) {
                const bool controller_feedback_empty =
                  tail_feedback.promoted == 0 &&
                  tail_feedback.retained == 0 &&
                  tail_feedback.stale == 0 &&
                  tail_feedback.queue_rejects == 0 &&
                  tail_feedback.tail_admitted == 0 &&
                  tail_feedback.commit_waves_observed == 0 &&
                  tail_feedback.commit_waves_covered == 0;
                core_only_issue_epoch =
                  tail_batch.active == 0 &&
                  live_tail_mask == 0 &&
                  controller_feedback_empty &&
                  shadow_frontier_count <= positional_core_width
                    ? 1u : 0u;
              }
            }
            __syncthreads();
            if (threadIdx.x == 0) physical_issue_span = 0;
            __syncthreads();
            if (core_only_issue_epoch != 0) {
              prepare_exact_core_frontier(
                shadow_frontier_handles, shadow_frontier_ranks,
                shadow_frontier_count, frontier_rob,
                positional_core_width, frontier_issue_epoch,
                issue_epochs, issue_width_sum,
                issue_width_capacity_sum,
                frontier_controller.current_issue_width, max_issue_width);
            } else {
              // Only an empty physical tail may accept a fresh speculative
              // suffix.  A resident tail is still eligible for exact
              // promotion/mapping above, but it must not be overwritten or
              // allowed to delay this round's critical prefix.
              const bool may_admit_exact_tail = tail_batch.active == 0;
              prepare_issue_frontier_entries(
                shadow_frontier_handles, shadow_frontier_ranks,
                shadow_frontier_count, frontier_rob,
                frontier_issue_epoch, frontier_controller,
                positional_core_width,
                tail_feedback, speculative_stale,
                speculative_wasted_bytes, core_prefetch_stale,
                issue_epochs, issue_width_sum,
                issue_width_capacity_sum, max_issue_width,
                issue_rob_slots, physical_issue_span,
                // PFEC has exposed the exact next frontier before the
                // authoritative merge.  Its suffix therefore has the full
                // merge + next-round overlap window and is safe to issue
                // whenever no older tail occupies the speculative WQEs.
                may_admit_exact_tail,
                // Reconcile and train on the preceding wave exactly once
                // before admitting this wave.  The updated width applies to
                // the following certificate, avoiding self-training.
                true);
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_prepare_cycles +=
              now - frontier_subphase_started_cycles;
            frontier_subphase_started_cycles = now;
            early_queue_rejects_before =
              speculative_queue_rejects;
          }
          __syncthreads();
          if (shadow_frontier_count != 0) {
            const u32 core_slot_count = min(
              params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity - 1));
            const bool split_issue =
              physical_issue_span > core_slot_count;
            bool issue_ok = true;
            if (tail_batch.active == 0 && split_issue) {
              issue_ok = issue_split_frontier_graph_batch(
                params, descriptor, frontier_rob,
                core_batch, tail_batch, physical_issue_span,
                core_slot_count,
                params.core_batch_statuses,
                params.core_batch_completion_timestamps_ns,
                params.tail_batch_statuses,
                params.tail_batch_completion_timestamps_ns,
                &total_remote_batches, &total_remote_reads,
                &total_graph_read_bytes,
                &total_graph_live_extent_reads,
                &total_graph_full_record_reads,
                &critical_graph_reads, &critical_graph_bytes,
                &speculative_graph_reads, &speculative_graph_bytes,
                &core_prefetch_queue_rejects,
                &speculative_queue_rejects,
                &core_prefetch_reads, &core_prefetch_bytes,
                &tail_feedback.tail_admitted,
                &dynamic_graph_telemetry);
            } else {
              // The narrow core-only path preserves the critical-first
              // publication cost of the coupled engine. A resident tail does
              // not block this independent critical prefix.
              issue_ok = issue_frontier_graph_batch(
                params, descriptor, frontier_rob, core_batch,
                0, core_slot_count,
                params.core_batch_statuses,
                params.core_batch_completion_timestamps_ns,
                DirectBatchPriority::critical, true,
                &total_remote_batches, &total_remote_reads,
                &total_graph_read_bytes,
                &total_graph_live_extent_reads,
                &total_graph_full_record_reads,
                &critical_graph_reads, &critical_graph_bytes,
                &core_prefetch_queue_rejects,
                &core_prefetch_reads, &core_prefetch_bytes,
                &dynamic_graph_telemetry);
            }
            if (!issue_ok && threadIdx.x == 0) {
              graph_failed = frontier_graph_failure_code(
                params, descriptor, 9u);
            }
            if (threadIdx.x == 0) {
              core_batch_positional =
                core_only_issue_epoch != 0 ? 1u : 0u;
              certified_mapping_ready =
                core_only_issue_epoch == 0 &&
                next_commit_ready != 0 ? 1u : 0u;
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_enqueue_cycles +=
              now - frontier_subphase_started_cycles;
            if (core_batch.active != 0) {
              // The enqueue subphase timestamp is dead after this point;
              // retain the completed doorbell time through Beam publication.
              frontier_subphase_started_cycles = now;
              ++core_prefetch_waves;
              ++frontier_reusable_issued_certificates;
            } else {
              frontier_subphase_started_cycles = 0;
            }
            tail_feedback.queue_rejects +=
              speculative_queue_rejects -
                early_queue_rejects_before;
            const u64 frontier_issue_cycles =
              now - shadow_issue_started_cycles;
            rdma_issue_cycles += frontier_issue_cycles;
            graph_phase_cycles += frontier_issue_cycles;
            phase_started_cycles = clock64();
          }
          __syncthreads();
        }
        if (graph_failed != 0) break;
        if (threadIdx.x == 0) {
          // Do not charge certificate preview/enqueue/RDMA issue time to
          // Stable-Run. Prepared leaves remain immutable across issue.
          stable_merge_state.phase_started = clock64();
        }
        __syncthreads();
        finish_approximate_stable_runs(
          beam_handles, beam_ids, beam_distances,
          beam_expanded, beam_count, traversal_capacity,
          rerank_handles, rerank_flags, rerank_distances,
          candidate_workspace, stable_merge_state,
          &beam_merge_round_breakdown, true);
        if (threadIdx.x == 0) {
          const u64 beam_published_at = clock64();
          completion.frontier_prefix_to_beam_publish_cycles +=
            beam_published_at - shadow_issue_started_cycles;
          if (frontier_subphase_started_cycles != 0) {
            completion.frontier_issue_to_beam_publish_cycles +=
              beam_published_at - frontier_subphase_started_cycles;
          }
        }
        __syncthreads();
      } else {
        if (asfe_stable_run) {
          if (stable_runs_prepared_before_issue == 0) {
            prepare_approximate_stable_runs(
              candidate_input_handles, candidate_input_distances,
              candidate_count,
              beam_handles, beam_ids, beam_distances, beam_expanded,
              beam_count, traversal_capacity,
              rerank_handles, rerank_flags, rerank_distances,
              candidate_workspace, stable_merge_state,
              &beam_merge_round_breakdown, false);
          } else {
            if (threadIdx.x == 0) {
              stable_merge_state.phase_started = clock64();
            }
            __syncthreads();
          }
          finish_approximate_stable_runs(
            beam_handles, beam_ids, beam_distances,
            beam_expanded, beam_count, traversal_capacity,
            rerank_handles, rerank_flags, rerank_distances,
            candidate_workspace, stable_merge_state,
            &beam_merge_round_breakdown, true);
        } else {
          merge_approximate_into_beam(
            candidate_input_handles, candidate_input_distances,
            candidate_count, beam_handles, beam_ids, beam_distances,
            beam_expanded, beam_count, traversal_capacity,
            rerank_handles, rerank_flags, rerank_distances,
            candidate_workspace,
            static_cast<BeamMergePolicy>(params.beam_merge_policy),
            params.beam_merge_policy ==
                static_cast<u32>(BeamMergePolicy::stable_run)
              ? &beam_merge_round_breakdown : nullptr);
        }
      }
      if (threadIdx.x == 0) {
        if (rerank_count != 0 && beam_count == 0) {
          route_failure_reason =
            0x80000000u |
            ((stable_runs_prepared_before_issue & 0x3u) << 28) |
            ((stable_merge_state.prepared & 0x1u) << 27) |
            ((stable_merge_state.original_count & 0xffu) << 19) |
            ((candidate_count & 0x7ffu) << 8) |
            (total_graph_rounds & 0xffu);
        }
        const u64 merge_cycles = exact_frontier_issue_enabled
          ? beam_merge_round_breakdown.prepare +
              beam_merge_round_breakdown.sort +
              beam_merge_round_breakdown.materialize
          : clock64() - phase_started_cycles;
        beam_merge_cycles += merge_cycles;
        beam_phase_cycles += merge_cycles;
        if (params.beam_merge_policy ==
            static_cast<u32>(BeamMergePolicy::stable_run)) {
          beam_merge_breakdown.prepare +=
            beam_merge_round_breakdown.prepare;
          beam_merge_breakdown.sort += beam_merge_round_breakdown.sort;
          beam_merge_breakdown.materialize +=
            beam_merge_round_breakdown.materialize;
        }
      }
      __syncthreads();
      __syncthreads();
    }
    if constexpr (EnableAsfe) {
      // The complete Stable-Run publication above is the only authority for
      // traversal state.  Once its remaining unexpanded frontier fits in one
      // Commit wave, the following epoch is the earliest observable terminal
      // horizon: either that wave exhausts the frontier or it publishes the
      // final replacement candidates.  Enqueue a read-only exact snapshot
      // now, behind the already-issued critical graph core, and overlap it
      // with that last graph epoch.  The terminal exactifier revalidates every
      // cached record against the final Beam/header and falls back to the
      // established fenced path for all churn, deletes, and updates.
      if (threadIdx.x == 0) {
        terminal_exact_horizon_ready = 0;
        if (params.terminal_exact_prefetch_enabled != 0 &&
            terminal_exact_cache.attempted == 0) {
          u32 remaining = 0;
          for (u32 rank = 0; rank < beam_count; ++rank) {
            remaining += beam_expanded[rank] == 0 ? 1u : 0u;
          }
          // graph_failed is known before this point.  Never publish a new
          // terminal RDMA descriptor for a query that is about to take the
          // failure exit below: its exact-record scratch is query-slot-owned
          // and must not outlive the completion that releases that slot.
          terminal_exact_horizon_ready = graph_failed == 0 &&
            (remaining <= params.commit_width ||
             expansions + selected_count >= params.max_expansions);
        }
      }
      __syncthreads();
      if (terminal_exact_horizon_ready != 0) {
        begin_terminal_exact_cache_prefetch(
          params, descriptor, beam_handles, beam_count,
          terminal_exact_cache);
      }
      __syncthreads();
    }
    if (graph_failed != 0) {
      __shared__ u64 failed_speculative_wait_before;
      if constexpr (EnableAsfe) {
        if (threadIdx.x == 0) {
          failed_speculative_wait_before = speculative_wait_cycles;
        }
        __syncthreads();
        (void)finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry);
        if (tail_batch.active != 0) {
          (void)finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch, tail_telemetry);
          apply_tail_admission_correction(
            tail_admission_correction, tail_feedback,
            total_remote_batches, total_remote_reads,
            total_graph_read_bytes, total_graph_live_extent_reads,
            total_graph_full_record_reads, speculative_graph_reads,
            speculative_graph_bytes, speculative_queue_rejects);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          rdma_wait_cycles +=
            speculative_wait_cycles - failed_speculative_wait_before;
        }
        __syncthreads();
        // A terminal exact wave may have been admitted by an earlier epoch.
        // The graph-failure completion releases descriptor.query_slot to the
        // host, so every DMA targeting that slot must be quiescent first.
        // Treat its result as unused; the authoritative query is failing for
        // the graph/dynamic-code reason recorded below.
        drain_terminal_exact_cache_prefetch(
          params, descriptor, terminal_exact_cache);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.result_count = graph_failed;
        completion.diagnostic = make_query_diagnostic(
          core_batch.fatal != 0
            ? QueryFailureReason::graph_fetch
            : QueryFailureReason::dynamic_code_fetch,
          route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_read_bytes = total_graph_read_bytes;
        completion.graph_full_record_avoidance_hits = total_graph_full_record_avoidance_hits;
        completion.graph_live_extent_reads =
          total_graph_live_extent_reads;
        completion.graph_full_record_reads =
          total_graph_full_record_reads;
        completion.graph_extent_fallback_reads =
          total_graph_extent_fallback_reads;
        completion.graph_extent_underhint_reads =
          total_graph_extent_underhint_reads;
        completion.graph_extent_hint_promotions =
          total_graph_extent_hint_promotions;
        set_dynamic_graph_completion(
          completion, dynamic_graph_telemetry,
          critical_graph_reads + speculative_graph_reads,
          params.graph_read_policy);
        set_frontier_completion<EnableAsfe>(
          completion, logical_expansions, critical_graph_reads,
          critical_graph_bytes, speculative_graph_reads,
          speculative_graph_bytes, speculative_arrived,
          speculative_promoted, speculative_stale,
          speculative_wasted_bytes, speculative_queue_rejects,
          issue_epochs, commit_epochs, issue_width_sum,
          issue_width_capacity_sum, commit_width_sum, max_issue_width,
          max_commit_width, critical_rob_hits, critical_misses,
          speculative_wait_cycles,
          rdma_completion_latency_ns, speculative_completion_latency_ns,
          rdma_completion_groups, speculative_completion_groups,
          core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
          core_prefetch_promoted, core_prefetch_stale,
          core_prefetch_queue_rejects, core_prefetch_waves,
          core_ready_waves);
        set_frontier_certificate_completion<EnableAsfe>(
          completion, frontier_reusable_certificates,
          frontier_streamed_candidate_runs, ordered_score_batches,
          ordered_score_candidates, frontier_reusable_prefix_ranks,
          frontier_reusable_full_prefix_certificates,
          frontier_reusable_issued_certificates,
          ooo_bypassed_parents,
          frontier_certificate_rejects);
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_terminal_exact_cache_completion<EnableAsfe>(
          completion, terminal_exact_cache);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    if constexpr (EnableAsfe) {
      for (u32 slot = threadIdx.x; slot < kPersistentFrontierRobCapacity;
           slot += blockDim.x) {
        if (frontier_rob[slot].state ==
            static_cast<u8>(FrontierRequestState::committed)) {
          frontier_rob[slot] = {};
        }
      }
    }
    if (threadIdx.x == 0) {
      expansions += selected_count;
      logical_expansions += selected_count;
    }
    __syncthreads();
  }

  // The last commit epoch may have launched a shadow wave immediately before
  // max_expansions was reached. Drain it before exact rerank or query-slot
  // reuse; otherwise a late DMA could overwrite the next query's scratch.
  __shared__ u32 final_speculative_drain_failed;
  __shared__ u64 final_speculative_wait_before;
  if constexpr (EnableAsfe) {
    if (threadIdx.x == 0) {
      final_speculative_drain_failed = 0;
      final_speculative_wait_before = speculative_wait_cycles;
    }
    __syncthreads();
    if (!finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry)) {
      if (threadIdx.x == 0) final_speculative_drain_failed = 1;
    }
    if (tail_batch.active != 0) {
      if (!finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch,
            tail_telemetry)) {
        if (threadIdx.x == 0) final_speculative_drain_failed = 1;
      }
      apply_tail_admission_correction(
        tail_admission_correction, tail_feedback,
        total_remote_batches, total_remote_reads,
        total_graph_read_bytes, total_graph_live_extent_reads,
        total_graph_full_record_reads, speculative_graph_reads,
        speculative_graph_bytes, speculative_queue_rejects);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rdma_wait_cycles +=
        speculative_wait_cycles - final_speculative_wait_before;
      // No shadow payload is needed after the last authoritative commit.
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        FrontierRobEntry& entry = frontier_rob[slot];
        if (entry.state ==
            static_cast<u8>(FrontierRequestState::validated)) {
          const bool speculative =
            entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative);
          if (speculative && entry.transfer_bytes != 0) {
            speculative_wasted_bytes += entry.transfer_bytes;
          }
          if (speculative) {
            ++speculative_stale;
            // Query termination right-censors this record: no future Commit
            // Frontier exists in which it could either promote or become a
            // true turnover miss. Charge its physical bytes to telemetry,
            // but do not train the CTA-persistent controller with an outcome
            // created solely by the configured search budget. Width learning
            // therefore transfers across queries and datasets using only
            // observable frontier utility/pressure, not query length.
          } else {
            ++core_prefetch_stale;
          }
        }
        if (entry.state !=
            static_cast<u8>(FrontierRequestState::init)) {
          entry = {};
        }
      }
      adaptive_frontier::update_issue_width(
        frontier_controller,
        adaptive_frontier::Feedback{
          .promoted = tail_feedback.promoted,
          .retained = tail_feedback.retained,
          .stale = tail_feedback.stale,
          .queue_rejects = tail_feedback.queue_rejects,
          .critical_misses = tail_feedback.core_misses,
          .tail_admitted = tail_feedback.tail_admitted,
          .commit_waves_observed =
            tail_feedback.commit_waves_observed,
          .commit_waves_covered =
            tail_feedback.commit_waves_covered,
      });
      tail_feedback = {};
    }
    __syncthreads();
  } else if (threadIdx.x == 0) {
    final_speculative_drain_failed = 0;
  }
  __syncthreads();
  if (final_speculative_drain_failed != 0) {
    if constexpr (EnableAsfe) {
      // Preserve the same query-slot lifetime rule on this second exceptional
      // exit.  Without the drain, an admitted terminal exact train can write
      // into a later query generation after this completion is observed.
      drain_terminal_exact_cache_prefetch(
        params, descriptor, terminal_exact_cache);
    }
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.result_count = frontier_graph_failure_code(
        params, descriptor, 10u);
      completion.diagnostic = make_query_diagnostic(
        QueryFailureReason::graph_fetch, route_snapshot_retries);
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_read_bytes = total_graph_read_bytes;
      completion.graph_full_record_avoidance_hits = total_graph_full_record_avoidance_hits;
      completion.graph_live_extent_reads = total_graph_live_extent_reads;
      completion.graph_full_record_reads = total_graph_full_record_reads;
      completion.graph_extent_fallback_reads =
        total_graph_extent_fallback_reads;
      completion.graph_extent_underhint_reads =
        total_graph_extent_underhint_reads;
      completion.graph_extent_hint_promotions =
        total_graph_extent_hint_promotions;
      set_dynamic_graph_completion(
          completion, dynamic_graph_telemetry,
          critical_graph_reads + speculative_graph_reads,
          params.graph_read_policy);
      set_frontier_completion<EnableAsfe>(
        completion, logical_expansions, critical_graph_reads,
        critical_graph_bytes, speculative_graph_reads,
        speculative_graph_bytes, speculative_arrived,
        speculative_promoted, speculative_stale,
        speculative_wasted_bytes, speculative_queue_rejects,
        issue_epochs, commit_epochs, issue_width_sum,
        issue_width_capacity_sum, commit_width_sum, max_issue_width,
        max_commit_width, critical_rob_hits, critical_misses,
        speculative_wait_cycles,
        rdma_completion_latency_ns, speculative_completion_latency_ns,
        rdma_completion_groups, speculative_completion_groups,
        core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
        core_prefetch_promoted, core_prefetch_stale,
        core_prefetch_queue_rejects, core_prefetch_waves,
        core_ready_waves);
      set_frontier_certificate_completion<EnableAsfe>(
        completion, frontier_reusable_certificates,
        frontier_streamed_candidate_runs, ordered_score_batches,
        ordered_score_candidates, frontier_reusable_prefix_ranks,
        frontier_reusable_full_prefix_certificates,
        frontier_reusable_issued_certificates,
        ooo_bypassed_parents,
        frontier_certificate_rejects);
      completion.graph_rounds = total_graph_rounds;
      set_terminal_exact_cache_completion<EnableAsfe>(
        completion, terminal_exact_cache);
      set_query_trace_completion(params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }

  const bool terminal_beam_already_sorted =
    params.beam_merge_policy ==
      static_cast<u32>(BeamMergePolicy::stable_run);
  if (terminal_beam_already_sorted) {
    // Stable-Run publishes a complete distance-sorted authoritative Beam on
    // every round. Avoid copying it to merge scratch, sorting it again, and
    // then serially copying 128 entries from lane zero. Exactification still
    // receives the identical ordered handle set; merge_* remains its private
    // output workspace below.
    if (threadIdx.x == 0) {
      rerank_count = min(beam_count, kPersistentMaxExact);
    }
    __syncthreads();
    for (u32 index = threadIdx.x; index < rerank_count;
         index += blockDim.x) {
      rerank_handles[index] = beam_handles[index];
      rerank_ids[index] = UINT32_MAX;
      rerank_distances[index] = beam_distances[index];
    }
  } else {
    for (u32 index = threadIdx.x; index < beam_count;
         index += blockDim.x) {
      merge_handles[index] = beam_handles[index];
      merge_distances[index] = beam_distances[index];
      merge_expanded[index] = 0;
    }
    __syncthreads();
    sort_candidates(
      merge_handles, nullptr, merge_distances, merge_expanded, beam_count);
    if (threadIdx.x == 0) {
      rerank_count = min(beam_count, kPersistentMaxExact);
      for (u32 index = 0; index < rerank_count; ++index) {
        rerank_handles[index] = merge_handles[index];
        rerank_ids[index] = UINT32_MAX;
        rerank_distances[index] = merge_distances[index];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Remote fixed-record headers are the source of truth for delete/upsert
    // visibility; exactify the full traversal Beam so tombstoned prefix
    // candidates can be replaced instead of returning fewer than k.
    phase_started_cycles = clock64();
  }
  __syncthreads();
  bool exact_fetch_succeeded = false;
  if constexpr (EnableAsfe) {
    exact_fetch_succeeded = exactify_into_beam_with_terminal_cache(
      params, descriptor, query, rerank_handles, rerank_ids,
      rerank_distances, rerank_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, &total_exact_reads,
      &completion.exact_snapshot_train_batches,
      &completion.exact_snapshot_train_fallbacks,
      min(kPersistentMaxExact,
          max(params.final_rerank_width,
              static_cast<u32>(descriptor.k))),
      true, merge_handles, merge_ids, merge_distances, merge_expanded,
      terminal_exact_cache);
  } else {
    exact_fetch_succeeded = exactify_into_beam(
      params, descriptor, query, rerank_handles, rerank_ids,
      rerank_distances, rerank_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, &total_exact_reads,
      &completion.exact_snapshot_train_batches,
      &completion.exact_snapshot_train_fallbacks,
      min(kPersistentMaxExact,
          max(params.final_rerank_width,
              static_cast<u32>(descriptor.k))),
      true, merge_handles, merge_ids, merge_distances, merge_expanded);
  }
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();
  if constexpr (EnableAsfe) {
    // Completion publication is the query-slot reuse boundary. Keep this
    // final idempotent guard even though the terminal-aware exactifier drains
    // its train internally: it makes both the success and exact-error exits
    // safe if a future fallback returns before consuming the cache.
    drain_terminal_exact_cache_prefetch(
      params, descriptor, terminal_exact_cache);
  }

  // Transport failure is terminal even on the first route attempt; only a
  // completed-but-empty exact snapshot retains the legacy one-time reroute.
  if (!exact_fetch_succeeded || beam_count == 0) {
    if (exact_rerank_should_retry_route(
          exact_fetch_succeeded, route_attempt)) {
      for (u32 index = threadIdx.x; index < params.visited_capacity;
           index += blockDim.x) {
        visited[index] = kInvalidDeviceHandle;
      }
      if (threadIdx.x == 0) device_ring_relax(64);
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      // Empty exact rerank is a fail-stop invariant, not a normal search
      // outcome.  Inspect the already-fetched snapshots only on that path so
      // successful queries pay no extra scan.  The packed counts and first
      // rejected record reuse failure-only completion fields and distinguish
      // an empty traversal Beam, a malformed handle, and a snapshot
      // visibility failure without enlarging CompletionDescriptor.
      ExactRerankDiagnostics exact_diagnostics;
      if (exact_fetch_succeeded) {
        exact_diagnostics = diagnose_exact_rerank(
          params, descriptor, rerank_handles, rerank_count);
      }
      completion.status = -EIO;
      completion.diagnostic = make_query_diagnostic(
        exact_fetch_succeeded
          ? QueryFailureReason::exact_rerank_empty
          : QueryFailureReason::exact_fetch,
        route_snapshot_retries);
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.graph_cycles = graph_phase_cycles;
      completion.score_cycles = score_phase_cycles;
      completion.beam_cycles = beam_phase_cycles;
      completion.exact_cycles = exact_phase_cycles;
      completion.beam_selection_cycles = beam_selection_cycles;
      completion.rdma_issue_cycles = rdma_issue_cycles;
      completion.frontier_preview_cycles = frontier_preview_cycles;
      completion.frontier_prepare_cycles = frontier_prepare_cycles;
      completion.frontier_enqueue_cycles = frontier_enqueue_cycles;
      completion.rdma_wait_cycles = rdma_wait_cycles;
      completion.graph_validation_cycles = graph_validation_cycles;
      completion.neighbor_decode_cycles = neighbor_decode_cycles;
      completion.pq_score_cycles = pq_score_cycles;
      completion.visited_cycles = visited_cycles;
      completion.beam_merge_cycles = beam_merge_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_read_bytes = total_graph_read_bytes;
      completion.graph_full_record_avoidance_hits = total_graph_full_record_avoidance_hits;
      completion.graph_live_extent_reads =
        total_graph_live_extent_reads;
      completion.graph_full_record_reads =
        total_graph_full_record_reads;
      completion.graph_extent_fallback_reads =
        total_graph_extent_fallback_reads;
      completion.graph_extent_underhint_reads =
        total_graph_extent_underhint_reads;
      completion.graph_extent_hint_promotions =
        total_graph_extent_hint_promotions;
      set_dynamic_graph_completion(
          completion, dynamic_graph_telemetry,
          critical_graph_reads + speculative_graph_reads,
          params.graph_read_policy);
      set_frontier_completion<EnableAsfe>(
        completion, logical_expansions, critical_graph_reads,
        critical_graph_bytes, speculative_graph_reads,
        speculative_graph_bytes, speculative_arrived,
        speculative_promoted, speculative_stale,
        speculative_wasted_bytes, speculative_queue_rejects,
        issue_epochs, commit_epochs, issue_width_sum,
        issue_width_capacity_sum, commit_width_sum, max_issue_width,
        max_commit_width, critical_rob_hits, critical_misses,
        speculative_wait_cycles,
        rdma_completion_latency_ns, speculative_completion_latency_ns,
        rdma_completion_groups, speculative_completion_groups,
        core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
        core_prefetch_promoted, core_prefetch_stale,
        core_prefetch_queue_rejects, core_prefetch_waves,
        core_ready_waves);
      set_frontier_certificate_completion<EnableAsfe>(
        completion, frontier_reusable_certificates,
        frontier_streamed_candidate_runs, ordered_score_batches,
        ordered_score_candidates, frontier_reusable_prefix_ranks,
        frontier_reusable_full_prefix_certificates,
        frontier_reusable_issued_certificates,
        ooo_bypassed_parents,
        frontier_certificate_rejects);
      completion.graph_rounds = total_graph_rounds;
      completion.exact_vectors = total_exact_reads;
      completion.dynamic_code_cycles = dynamic_code_cycles;
      completion.dynamic_code_candidates = dynamic_code_candidates;
      completion.dynamic_code_reads = dynamic_code_reads;
      completion.dynamic_code_incarnation_rejects =
        dynamic_code_incarnation_rejects;
      set_dynamic_code_cache_completion(
        completion, dynamic_code_cache_hits,
        dynamic_code_batch_deduplicated,
        dynamic_code_cache_publish_successes,
        dynamic_code_cache_first_occupancies,
        dynamic_code_cache_publish_races,
        dynamic_code_cache_lookup_probe_exhaustions,
        dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
      set_beam_merge_completion(completion, beam_merge_breakdown);
      set_terminal_exact_cache_completion<EnableAsfe>(
        completion, terminal_exact_cache);
      if (exact_fetch_succeeded) {
        completion.result_count =
          (rerank_count & 0xffu) |
          ((exact_diagnostics.resolved & 0xffu) << 8) |
          ((exact_diagnostics.equal_headers & 0xffu) << 16) |
          ((exact_diagnostics.visible & 0xffu) << 24);
        completion.frontier_telemetry_reserved0 =
          static_cast<u32>(exact_diagnostics.first_handle);
        completion.frontier_telemetry_reserved1 =
          static_cast<u32>(exact_diagnostics.first_handle >> 32);
        completion.rdma_completion_latency_ns = exact_diagnostics.first_before;
        completion.speculative_completion_latency_ns = exact_diagnostics.first_after;
        completion.issue_width_sum =
          (static_cast<u64>(exact_diagnostics.first_expected_incarnation) << 32) |
          exact_diagnostics.first_stored_incarnation;
        completion.commit_width_sum = route_failure_reason;
      }
      set_query_trace_completion(params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  // exactify_into_beam() has already sorted the complete exact candidate run
  // and copied its valid prefix back to Beam.  Re-sorting that byte-identical
  // run here duplicated a full block-wide radix pass on every successful
  // query and cannot change either result order or visibility filtering.
  if (threadIdx.x == 0) {
    completion.result_count = publish_query_results(
      params, descriptor, query_slot, beam_ids, beam_distances, beam_count);
    completion.status = 0;
    completion.diagnostic = make_query_diagnostic(
      QueryFailureReason::none, route_snapshot_retries);
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.graph_cycles = graph_phase_cycles;
    completion.score_cycles = score_phase_cycles;
    completion.beam_cycles = beam_phase_cycles;
    completion.exact_cycles = exact_phase_cycles;
    completion.beam_selection_cycles = beam_selection_cycles;
    completion.rdma_issue_cycles = rdma_issue_cycles;
    completion.frontier_preview_cycles = frontier_preview_cycles;
    completion.frontier_prepare_cycles = frontier_prepare_cycles;
    completion.frontier_enqueue_cycles = frontier_enqueue_cycles;
    completion.rdma_wait_cycles = rdma_wait_cycles;
    completion.graph_validation_cycles = graph_validation_cycles;
    completion.neighbor_decode_cycles = neighbor_decode_cycles;
    completion.pq_score_cycles = pq_score_cycles;
    completion.visited_cycles = visited_cycles;
    completion.beam_merge_cycles = beam_merge_cycles;
    completion.remote_pages = total_remote_reads;
    completion.remote_batches = total_remote_batches;
    completion.graph_read_retries = total_graph_read_retries;
    completion.graph_read_bytes = total_graph_read_bytes;
    completion.graph_full_record_avoidance_hits = total_graph_full_record_avoidance_hits;
    completion.graph_live_extent_reads =
      total_graph_live_extent_reads;
    completion.graph_full_record_reads =
      total_graph_full_record_reads;
    completion.graph_extent_fallback_reads =
      total_graph_extent_fallback_reads;
    completion.graph_extent_underhint_reads =
      total_graph_extent_underhint_reads;
    completion.graph_extent_hint_promotions =
      total_graph_extent_hint_promotions;
    set_dynamic_graph_completion(
          completion, dynamic_graph_telemetry,
          critical_graph_reads + speculative_graph_reads,
          params.graph_read_policy);
    set_frontier_completion<EnableAsfe>(
      completion, logical_expansions, critical_graph_reads,
      critical_graph_bytes, speculative_graph_reads,
      speculative_graph_bytes, speculative_arrived,
      speculative_promoted, speculative_stale,
      speculative_wasted_bytes, speculative_queue_rejects,
      issue_epochs, commit_epochs, issue_width_sum,
      issue_width_capacity_sum, commit_width_sum, max_issue_width,
      max_commit_width, critical_rob_hits, critical_misses,
      speculative_wait_cycles,
      rdma_completion_latency_ns, speculative_completion_latency_ns,
      rdma_completion_groups, speculative_completion_groups,
      core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
      core_prefetch_promoted, core_prefetch_stale,
      core_prefetch_queue_rejects, core_prefetch_waves,
      core_ready_waves);
    set_frontier_certificate_completion<EnableAsfe>(
      completion, frontier_reusable_certificates,
      frontier_streamed_candidate_runs, ordered_score_batches,
      ordered_score_candidates, frontier_reusable_prefix_ranks,
      frontier_reusable_full_prefix_certificates,
      frontier_reusable_issued_certificates,
      ooo_bypassed_parents,
      frontier_certificate_rejects);
    set_terminal_exact_cache_completion<EnableAsfe>(
      completion, terminal_exact_cache);
    completion.graph_rounds = total_graph_rounds;
    completion.exact_vectors = total_exact_reads;
    completion.dynamic_code_cycles = dynamic_code_cycles;
    completion.dynamic_code_candidates = dynamic_code_candidates;
    completion.dynamic_code_reads = dynamic_code_reads;
    completion.dynamic_code_incarnation_rejects =
      dynamic_code_incarnation_rejects;
    set_dynamic_code_cache_completion(
      completion, dynamic_code_cache_hits,
      dynamic_code_batch_deduplicated,
      dynamic_code_cache_publish_successes,
      dynamic_code_cache_first_occupancies,
      dynamic_code_cache_publish_races,
      dynamic_code_cache_lookup_probe_exhaustions,
      dynamic_code_cache_publish_probe_exhaustions,
      dynamic_code_cache_lookup_probes,
      dynamic_code_cache_max_lookup_probes);
    set_beam_merge_completion(completion, beam_merge_breakdown);
    set_query_trace_completion(params, query_slot, completion);
    device_ring_push(params.completions, completion);
  }
  __syncthreads();
  return;
  }
}

}  // namespace gpu_search::persistent_kernel_detail

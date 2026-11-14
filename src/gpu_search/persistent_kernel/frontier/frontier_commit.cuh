#pragma once

#include "gpu_search/persistent_kernel/query/centroid_route.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __noinline__ void prepare_issue_frontier_entries(
    const u64* issue_handles, const u16* issue_ranks,
    u32& issue_count, FrontierRobEntry* frontier_rob,
    u32& issue_epoch, adaptive_frontier::ControllerState& controller,
    u32 core_slot_count, TailFrontierFeedback& feedback,
    u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& core_prefetch_stale,
    u32& issue_epochs, u64& issue_width_sum,
    u64& issue_width_capacity_sum, u32& observed_max_issue_width,
    u32* issue_rob_slots, u32& physical_issue_span,
    bool allow_new_tail = true,
    bool apply_controller_feedback = true,
    bool start_new_issue_epoch = true) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    if (threadIdx.x == 0) {
      // The caller normally materializes exactly this many preview entries.
      // Keep the helper defensive without letting the certificate update
      // below retroactively resize an already-generated wave.
      issue_count = min(issue_count, controller.current_issue_width);
    }
    __syncwarp(full_warp);
  }
  // Attribute benefit only at an exact Stable-Run certificate, never at the
  // round in which a shadow request happened to be admitted.  Early shadow
  // predicts this preview and split tail predicts a later one, so checking
  // here gives both paths the same unambiguous wave identity.  ARRIVED and
  // INFLIGHT entries deliberately do not count: the entire actual Commit
  // prefix must already be validated when authority is established. Fold
  // this certificate into the mandatory handle-to-ROB mapping below rather
  // than adding a second Commit-by-ROB scan.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 issue_position = threadIdx.x;
    u32 matching_slot = UINT32_MAX;
    if (issue_position < issue_count) {
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = frontier_rob[slot];
        const bool resident_tail =
          !allow_new_tail && slot >= core_slot_count &&
          (entry.state ==
               static_cast<u8>(FrontierRequestState::inflight) ||
           entry.state ==
               static_cast<u8>(FrontierRequestState::arrived));
        if ((entry.state ==
               static_cast<u8>(FrontierRequestState::validated) ||
             resident_tail) &&
            entry.node_handle == issue_handles[issue_position]) {
          matching_slot = slot;
          break;
        }
      }
    }
    issue_rob_slots[issue_position] = matching_slot;

    const bool observe_wave =
      apply_controller_feedback && feedback.tail_admitted != 0 &&
      feedback.commit_waves_observed == 0;
    const u32 commit_prefix_count = min(issue_count, core_slot_count);
    const bool exact_speculative_match =
      observe_wave && issue_position < commit_prefix_count &&
      matching_slot != UINT32_MAX &&
      frontier_rob[matching_slot].state ==
        static_cast<u8>(FrontierRequestState::validated) &&
      frontier_rob[matching_slot].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const u32 covered_mask =
      __ballot_sync(full_warp, exact_speculative_match);
    if (issue_position == 0 && observe_wave) {
      feedback.commit_waves_observed = 1;
      const u32 required_mask =
        commit_prefix_count == kPersistentFrontierRobCapacity
          ? full_warp
          : ((u32{1} << commit_prefix_count) - 1u);
      feedback.commit_waves_covered =
        commit_prefix_count != 0 &&
        (covered_mask & required_mask) == required_mask
          ? 1u : 0u;
    }
  }
  if (threadIdx.x == 0) {
    if (start_new_issue_epoch) ++issue_epoch;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = frontier_rob[slot];
    bool discarded_speculative = false;
    bool discarded_core = false;
    bool retained_speculative = false;
    u64 discarded_bytes = 0;
    if (entry.state ==
          static_cast<u8>(FrontierRequestState::committed) ||
        entry.state == static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::validated)) {
      bool retained = false;
      for (u32 issue_position = 0; issue_position < issue_count;
           ++issue_position) {
        if (issue_rob_slots[issue_position] != slot) continue;
        retained = true;
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = issue_ranks[issue_position];
        break;
      }
      if (!retained) {
        discarded_speculative = entry.priority ==
          static_cast<u8>(DirectBatchPriority::speculative);
        discarded_core = !discarded_speculative;
        discarded_bytes =
          discarded_speculative ? entry.transfer_bytes : 0;
        entry = {};
      } else {
        const bool newly_accounted =
          entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative) &&
          (entry.flags & kFrontierRobFlagUtilityAccounted) == 0;
        // The entry has now survived the exact post-merge certificate.  It
        // is no longer merely a pre-merge prediction, so clear the provenance
        // bit and account this physical read at most once before feeding
        // frontier-stability evidence to the controller.
        entry.flags &= static_cast<u8>(~kFrontierRobFlagEarlyShadow);
        if (newly_accounted) {
          entry.flags |= kFrontierRobFlagUtilityAccounted;
        }
        retained_speculative = newly_accounted;
      }
    }
    const u32 discarded_speculative_mask =
      __ballot_sync(full_warp, discarded_speculative);
    const u32 discarded_core_mask =
      __ballot_sync(full_warp, discarded_core);
    const u32 retained_speculative_mask =
      __ballot_sync(full_warp, retained_speculative);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 speculative_count =
        __popc(discarded_speculative_mask);
      speculative_stale += speculative_count;
      core_prefetch_stale += __popc(discarded_core_mask);
      speculative_wasted_bytes += discarded_bytes;
      feedback.stale += speculative_count;
      // A validated tail record that survives the exact next Stable-Run
      // preview is frontier-stability evidence, but it has not yet eliminated
      // a critical read. The controller may use retention to preserve a width
      // or strengthen whole-wave coverage; retention alone cannot bootstrap
      // another shadow slot. Stale/rejected bytes still drive contraction.
      feedback.retained += __popc(retained_speculative_mask);
    }
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 issue_position = threadIdx.x;
    const bool missing =
      issue_position < issue_count &&
      issue_rob_slots[issue_position] == UINT32_MAX &&
      (issue_position < core_slot_count || allow_new_tail);
    const bool core_issue = issue_position < core_slot_count;
    const bool free_core =
      issue_position < core_slot_count &&
      frontier_rob[issue_position].state ==
        static_cast<u8>(FrontierRequestState::init);
    const bool free_tail =
      issue_position >= core_slot_count &&
      frontier_rob[issue_position].state ==
        static_cast<u8>(FrontierRequestState::init);
    const u32 missing_core_mask = __ballot_sync(
      full_warp, missing && core_issue);
    const u32 missing_tail_mask = __ballot_sync(
      full_warp, missing && !core_issue);
    const u32 free_core_mask = __ballot_sync(full_warp, free_core);
    const u32 free_tail_mask = __ballot_sync(full_warp, free_tail);
    u32 destination = issue_rob_slots[issue_position];
    if (missing) {
      const u32 lower_issue_positions =
        issue_position == 0
          ? 0u : ((u32{1} << issue_position) - 1u);
      u32 free_ordinal = __popc(
        (core_issue ? missing_core_mask : missing_tail_mask) &
        lower_issue_positions);
      const u32 begin = core_issue ? 0u : core_slot_count;
      const u32 end = core_issue
        ? core_slot_count
        : static_cast<u32>(kPersistentFrontierRobCapacity);
      const u32 free_mask =
        core_issue ? free_core_mask : free_tail_mask;
      for (u32 slot = begin; slot < end; ++slot) {
        if ((free_mask & (u32{1} << slot)) == 0) continue;
        if (free_ordinal == 0) {
          destination = slot;
          break;
        }
        --free_ordinal;
      }
      if (destination != UINT32_MAX) {
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = issue_handles[issue_position];
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = issue_ranks[issue_position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority = static_cast<u8>(
          core_issue
            ? DirectBatchPriority::critical
            : DirectBatchPriority::speculative);
      }
    }
    issue_rob_slots[issue_position] = destination;
    const u32 admitted_mask = __ballot_sync(
      full_warp,
      issue_position < issue_count && destination != UINT32_MAX);
    u32 mapped_span =
      issue_position < issue_count && destination != UINT32_MAX
        ? destination + 1u : 0u;
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      mapped_span = max(
        mapped_span,
        __shfl_down_sync(full_warp, mapped_span, offset));
    }
    if (issue_position == 0) {
      const u32 admitted_issue_count = __popc(admitted_mask);
      // Logical certificate positions and physical ROB slots cease to be a
      // dense identity map as soon as a validated tail record is promoted,
      // or when a resident tail leaves a logical hole.  Preserve issue_count
      // as the immutable certificate span: every later lookup is explicitly
      // through issue_rob_slots[position].  Only accounting uses the admitted
      // popcount, while the descriptor uses the physical high-water mark.
      physical_issue_span = mapped_span;
      if (start_new_issue_epoch) {
        ++issue_epochs;
        issue_width_sum += admitted_issue_count;
        issue_width_capacity_sum += controller.current_issue_width;
      }
      observed_max_issue_width =
        max(observed_max_issue_width, admitted_issue_count);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && apply_controller_feedback) {
    // The exact certificate, completion state, and retention/staleness
    // classification above all describe the same physical wave. Train once
    // on that complete observation, then reset before the newly admitted
    // wave can contribute feedback. The preview was generated at the old
    // width, so the updated width deliberately takes effect on the next
    // preview; both growth and contraction are one-wave delayed and policy
    // shrink cannot relabel a valid retained record as stale.
    adaptive_frontier::update_issue_width(
      controller,
      adaptive_frontier::Feedback{
        .promoted = feedback.promoted,
        .retained = feedback.retained,
        .stale = feedback.stale,
        .queue_rejects = feedback.queue_rejects,
        .critical_misses = feedback.core_misses,
        .tail_admitted = feedback.tail_admitted,
        .commit_waves_observed = feedback.commit_waves_observed,
        .commit_waves_covered = feedback.commit_waves_covered,
      });
    feedback = {};
  }
  __syncthreads();
}

struct BeamShadowFrontierContext {
  FrontierRobEntry* frontier_rob{};
  adaptive_frontier::ControllerState* controller{};
  TailFrontierFeedback* feedback{};
  u32* issue_epoch{};
  u64* shadow_handles{};
  u16* shadow_ranks{};
  u32* shadow_count{};
  u32* shadow_rob_slots{};
  u32* new_shadow_count{};
  u32* speculative_stale{};
  u64* speculative_wasted_bytes{};
  u32* issue_epochs{};
  u64* issue_width_sum{};
  u64* issue_width_capacity_sum{};
  u32* observed_max_issue_width{};
};

// Admit one predicted issue wave proposed by the partial Stable-Run merge.
// RDMA issue overlaps the unchanged authoritative materialization; the next
// epoch still validates every entry against the newly merged Beam before it
// can become authoritative. The whole query-local ROB is available because
// exact misses are discovered only after this wave has been reconciled.
//
// A query starts from the current commit demand plus a bounded online probe.
// Promotion/retention and stale/rejected bytes adjust the following wave
// proportionally, without a dataset-specific predictor width.
__device__ __noinline__ void prepare_candidate_shadow_frontier(
    u32 proposed_count, u32 remaining_after_commit,
    const BeamShadowFrontierContext& context) {
  constexpr u32 full_warp = 0xffffffffu;
  adaptive_frontier::ControllerState& controller = *context.controller;
  TailFrontierFeedback& feedback = *context.feedback;
  u32& issue_epoch = *context.issue_epoch;
  u32& shadow_count = *context.shadow_count;
  u32& new_shadow_count = *context.new_shadow_count;
  if (threadIdx.x == 0) {
    adaptive_frontier::update_issue_width(
      controller,
      adaptive_frontier::Feedback{
        .promoted = feedback.promoted,
        .retained = feedback.retained,
        .stale = feedback.stale,
        .queue_rejects = feedback.queue_rejects,
        .critical_misses = feedback.core_misses,
        .tail_admitted = feedback.tail_admitted,
        .commit_waves_observed = feedback.commit_waves_observed,
        .commit_waves_covered = feedback.commit_waves_covered,
      });
    u32 shadow_budget = min(
      controller.current_issue_width,
      static_cast<u32>(kPersistentFrontierRobCapacity));
    shadow_budget = min(shadow_budget, remaining_after_commit);

    shadow_count = min(proposed_count, shadow_budget);
    ++issue_epoch;
    feedback = {};
    new_shadow_count = 0;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    u32 matching_slot = UINT32_MAX;
    if (position < shadow_count) {
      for (u32 slot = 0;
           slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = context.frontier_rob[slot];
        if (entry.state ==
              static_cast<u8>(FrontierRequestState::validated) &&
            entry.node_handle == context.shadow_handles[position]) {
          matching_slot = slot;
          break;
        }
      }
    }
    context.shadow_rob_slots[position] = matching_slot;
  }
  __syncthreads();

  // Reclaim validated tail payloads that fell outside the new shadow prefix.
  // Committed slots still feed the current expansion and cannot be reused
  // until graph decode has consumed them.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = context.frontier_rob[slot];
    bool discarded = false;
    bool retained = false;
    u64 discarded_bytes = 0;
    if (entry.state ==
        static_cast<u8>(FrontierRequestState::validated)) {
      for (u32 position = 0; position < shadow_count; ++position) {
        if (context.shadow_rob_slots[position] != slot) continue;
        retained = true;
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = context.shadow_ranks[position];
        break;
      }
      if (!retained) {
        discarded = true;
        discarded_bytes = entry.transfer_bytes;
        entry = {};
      }
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    }
    const u32 discarded_mask = __ballot_sync(full_warp, discarded);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 discarded_count = __popc(discarded_mask);
      *context.speculative_stale += discarded_count;
      *context.speculative_wasted_bytes += discarded_bytes;
      feedback.stale += discarded_count;
    }
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    const bool missing =
      position < shadow_count &&
      context.shadow_rob_slots[position] == UINT32_MAX;
    const bool free_tail =
      context.frontier_rob[threadIdx.x].state ==
        static_cast<u8>(FrontierRequestState::init);
    const u32 missing_mask = __ballot_sync(full_warp, missing);
    const u32 free_mask = __ballot_sync(full_warp, free_tail);
    u32 destination = context.shadow_rob_slots[position];
    if (missing) {
      u32 free_ordinal = __popc(
        missing_mask &
        (position == 0 ? 0u : ((u32{1} << position) - 1u)));
      for (u32 slot = 0;
           slot < kPersistentFrontierRobCapacity; ++slot) {
        if ((free_mask & (u32{1} << slot)) == 0) continue;
        if (free_ordinal == 0) {
          destination = slot;
          break;
        }
        --free_ordinal;
      }
      if (destination != UINT32_MAX) {
        FrontierRobEntry& entry = context.frontier_rob[destination];
        entry = {};
        entry.node_handle = context.shadow_handles[position];
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = context.shadow_ranks[position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
      }
    }
    context.shadow_rob_slots[position] = destination;
    const bool admitted =
      position < shadow_count && destination != UINT32_MAX;
    const u32 admitted_mask = __ballot_sync(full_warp, admitted);
    const u32 new_mask =
      __ballot_sync(full_warp, admitted && missing);
    if (position == 0) {
      shadow_count = __popc(admitted_mask);
      new_shadow_count = __popc(new_mask);
      ++*context.issue_epochs;
      const u32 logical_issue_width = shadow_count;
      *context.issue_width_sum += logical_issue_width;
      *context.issue_width_capacity_sum += controller.current_issue_width;
      *context.observed_max_issue_width =
        max(*context.observed_max_issue_width, logical_issue_width);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool finish_query_core_frontier_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* frontier_rob,
    FrontierGraphBatchState& core_batch,
    u32 slot_count,
    const CoreFrontierTelemetry& telemetry) {
  return finish_core_frontier_graph_batch(
    params, descriptor, frontier_rob, core_batch, slot_count,
    params.core_batch_statuses,
    params.core_batch_completion_timestamps_ns,
    telemetry.wait_cycles, telemetry.completion_latency_ns,
    telemetry.completion_groups, telemetry.arrived, telemetry.stale,
    telemetry.ready_waves, telemetry.dynamic_graph);
}

// Failure completions normally collapse every query-local graph dependency
// error into QueryFailureReason::graph_fetch.  Preserve a compact, zero-
// allocation postmortem in result_count (which is otherwise meaningless for
// a failed query): bits [7:0] identify the call site and bits [23:8] contain
// the magnitude of the first terminal core completion status.  The scan runs
// only after a fatal transition, so the successful query path is unchanged.
__device__ __forceinline__ u32 frontier_graph_failure_code(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    u32 stage) {
  i32 first_status = 0;
  if (params.core_batch_statuses != nullptr) {
    const i32* statuses = params.core_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
    for (u32 shard = 0; shard < params.num_shards; ++shard) {
      const i32 status =
        *reinterpret_cast<const volatile i32*>(statuses + shard);
      if (status == 0 || status == -EINPROGRESS || status == -EAGAIN) {
        continue;
      }
      first_status = status;
      break;
    }
  }
  const u32 magnitude = first_status < 0
    ? min(static_cast<u32>(-(first_status + 1)) + 1u, 0xffffu)
    : min(static_cast<u32>(first_status), 0xffffu);
  return (stage & 0xffu) | (magnitude << 8);
}

// Keep frontier selection and ROB reconciliation out of process_query's long
// live range. The metadata is query-CTA local; lane zero freezes the exact
// authoritative prefix and the first warp performs ROB matching, stale
// reconciliation, and miss compaction without a global queue.
__device__ __forceinline__ void select_commit_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 target, u64* selected_handles, u32* selected_beam_ranks,
    u32* commit_rob_slots, u32& selected_count, u32& shadow_frontier_count,
    u32& commit_epochs, u64& commit_width_sum, u32& max_commit_width) {
  if (threadIdx.x == 0) {
    selected_count = 0;
    for (u32 index = 0;
         index < beam_count && selected_count < target; ++index) {
      if (beam_expanded[index] != 0) continue;
      selected_handles[selected_count] = beam_handles[index];
      selected_beam_ranks[selected_count] = index;
      commit_rob_slots[selected_count] = UINT32_MAX;
      ++selected_count;
    }
    shadow_frontier_count = 0;
    if (selected_count != 0) {
      ++commit_epochs;
      commit_width_sum += selected_count;
      max_commit_width = max(max_commit_width, selected_count);
    }
  }
  __syncthreads();
}

__device__ __noinline__ void plan_commit_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 target, FrontierRobEntry* frontier_rob,
    adaptive_frontier::ControllerState& controller,
    bool issue_wave, u32 feedback_stale,
    u32 feedback_queue_rejects,
    u64* selected_handles, u32* selected_beam_ranks,
    u32* commit_rob_slots, u32& selected_count,
    u64* critical_fetch_handles, u32* critical_fetch_to_commit,
    u32& critical_fetch_count, u32* graph_record_slots,
    u32& shadow_frontier_count, u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& speculative_promoted,
    u32& core_prefetch_stale, u32& core_prefetch_promoted,
    u32& critical_rob_hits, u32& critical_misses,
    u32& commit_epochs, u64& commit_width_sum, u32& max_commit_width,
    TailFrontierFeedback& tail_feedback,
    bool selection_prepared = false) {
  if (!selection_prepared) {
    select_commit_frontier(
      beam_handles, beam_expanded, beam_count, target,
      selected_handles, selected_beam_ranks, commit_rob_slots,
      selected_count, shadow_frontier_count, commit_epochs,
      commit_width_sum, max_commit_width);
  }
  if (threadIdx.x == 0) critical_fetch_count = 0;
  __syncthreads();

  // Coupled Stable-Run + Live-Extent baseline: no ROB state exists and every
  // authoritative commit item is a critical fetch. Keep this path to one
  // ballot/compaction warp so ASFE instrumentation cannot tax the reference
  // execution.
  if (controller.max_issue_width == controller.commit_width) {
    if (threadIdx.x < kPersistentFrontierRobCapacity) {
      constexpr u32 full_warp = 0xffffffffu;
      const u32 position = threadIdx.x;
      const bool selected = position < selected_count;
      const u32 selected_mask = __ballot_sync(full_warp, selected);
      const u32 lower_lanes =
        position == 0 ? 0u : ((u32{1} << position) - 1u);
      if (selected) {
        const u32 fetch = __popc(selected_mask & lower_lanes);
        critical_fetch_to_commit[fetch] = position;
        critical_fetch_handles[fetch] = selected_handles[position];
      }
      if (position == 0) {
        critical_fetch_count = __popc(selected_mask);
        critical_misses += critical_fetch_count;
      }
    }
    __syncthreads();
    return;
  }

  // The overwhelmingly common steady state has no resident tail: the exact
  // preview issued commit-width records into positional core slots. Commit
  // rank therefore maps directly to scratch/ROB rank. Avoid all associative
  // ROB searches and retention scans in that state; the general path remains
  // for a real shadow tail or promoted tail record.
  if (controller.max_issue_width > controller.commit_width &&
      threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 lane = threadIdx.x;
    const bool live_tail =
      lane >= controller.commit_width &&
      frontier_rob[lane].state !=
        static_cast<u8>(FrontierRequestState::init);
    const u32 tail_mask = __ballot_sync(full_warp, live_tail);
    const bool speculative_core =
      lane < controller.commit_width &&
      frontier_rob[lane].state !=
        static_cast<u8>(FrontierRequestState::init) &&
      frontier_rob[lane].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const u32 speculative_core_mask =
      __ballot_sync(full_warp, speculative_core);
    if (lane == 0) {
      shadow_frontier_count =
        __popc(tail_mask) + (speculative_core_mask == 0 ? 0u : 1u);
    }
  }
  __syncthreads();
  if (controller.max_issue_width > controller.commit_width &&
      shadow_frontier_count == 0) {
    if (threadIdx.x < kPersistentFrontierRobCapacity) {
      constexpr u32 full_warp = 0xffffffffu;
      const u32 position = threadIdx.x;
      const bool selected = position < selected_count;
      FrontierRobEntry& entry = frontier_rob[position];
      const bool hit =
        selected && position < controller.commit_width &&
        entry.state == static_cast<u8>(FrontierRequestState::validated) &&
        entry.node_handle == selected_handles[position];
      const bool stale_core =
        position < controller.commit_width &&
        entry.state != static_cast<u8>(FrontierRequestState::init) &&
        !hit;
      if (hit) {
        commit_rob_slots[position] = position;
        entry.state = static_cast<u8>(FrontierRequestState::committed);
        graph_record_slots[position] =
          kGraphScratchBit | static_cast<u32>(entry.scratch_slot);
      } else if (stale_core) {
        entry = {};
      }
      const u32 miss_mask =
        __ballot_sync(full_warp, selected && !hit);
      const u32 lower_lanes =
        position == 0 ? 0u : ((u32{1} << position) - 1u);
      if (selected && !hit) {
        const u32 fetch = __popc(miss_mask & lower_lanes);
        critical_fetch_to_commit[fetch] = position;
        critical_fetch_handles[fetch] = selected_handles[position];
      }
      const u32 hit_mask = __ballot_sync(full_warp, hit);
      const u32 stale_mask = __ballot_sync(full_warp, stale_core);
      if (position == 0) {
        const u32 hits = __popc(hit_mask);
        const u32 misses = __popc(miss_mask);
        critical_fetch_count = __popc(miss_mask);
        critical_misses += critical_fetch_count;
        critical_rob_hits += hits;
        core_prefetch_promoted += hits;
        core_prefetch_stale += __popc(stale_mask);
        tail_feedback.stale += feedback_stale;
        tail_feedback.queue_rejects += feedback_queue_rejects;
        if (issue_wave) {
          tail_feedback.core_hits += hits;
          tail_feedback.core_misses += misses;
        }
      }
    }
    __syncthreads();
    return;
  }

  // The coupled baseline never creates ROB entries. Avoid making it pay the
  // candidate's bounded reconciliation work so the A/B comparison isolates
  // issue/commit decoupling. In the candidate, one warp owns the complete
  // 32-entry ROB: one lane per commit position/slot, with no scalar 32x32
  // nesting on the query's control lane.
  if (controller.max_issue_width > controller.commit_width &&
      threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 lane = threadIdx.x;
    if (lane < selected_count) {
      u32 matching_slot = UINT32_MAX;
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = frontier_rob[slot];
        if (entry.state ==
              static_cast<u8>(FrontierRequestState::validated) &&
            entry.node_handle == selected_handles[lane]) {
          matching_slot = slot;
          break;
        }
      }
      commit_rob_slots[lane] = matching_slot;
    }
    __syncwarp(full_warp);

    FrontierRobEntry& entry = frontier_rob[lane];
    bool stale_speculative = false;
    bool stale_core = false;
    bool retained_speculative = false;
    u64 stale_bytes = 0;
    if (entry.state == static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::validated)) {
      bool selected = false;
      for (u32 position = 0; position < selected_count; ++position) {
        if (commit_rob_slots[position] == lane) {
          selected = true;
          break;
        }
      }
      if (!selected) {
        bool still_unexpanded = false;
        u32 frontier_ordinal = 0;
        for (u32 rank = 0; rank < beam_count; ++rank) {
          if (beam_expanded[rank] != 0) continue;
          if (frontier_ordinal >= controller.current_issue_width) break;
          ++frontier_ordinal;
          if (beam_handles[rank] == entry.node_handle) {
            still_unexpanded = true;
            break;
          }
        }
        if (!still_unexpanded) {
          stale_speculative =
            entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative);
          stale_core = !stale_speculative;
          stale_bytes = stale_speculative ? entry.transfer_bytes : 0;
          entry = {};
        } else {
          const bool newly_accounted =
            entry.priority ==
              static_cast<u8>(DirectBatchPriority::speculative) &&
            (entry.flags & kFrontierRobFlagUtilityAccounted) == 0;
          // Exact Beam inspection has confirmed this previously early-issued
          // handle is still in the authoritative unexpanded frontier.
          // Promote its metadata from prediction to ordinary retained
          // evidence only after that validation point.
          entry.flags &= static_cast<u8>(~kFrontierRobFlagEarlyShadow);
          if (newly_accounted) {
            entry.flags |= kFrontierRobFlagUtilityAccounted;
          }
          retained_speculative = newly_accounted;
        }
      }
    }
    const u32 speculative_mask =
      __ballot_sync(full_warp, stale_speculative);
    const u32 core_mask = __ballot_sync(full_warp, stale_core);
    const u32 retained_mask =
      __ballot_sync(full_warp, retained_speculative);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      stale_bytes += __shfl_down_sync(full_warp, stale_bytes, offset);
    }
    if (lane == 0) {
      const u32 newly_stale = __popc(speculative_mask);
      speculative_stale += newly_stale;
      speculative_wasted_bytes += stale_bytes;
      core_prefetch_stale += __popc(core_mask);
      tail_feedback.stale += feedback_stale + newly_stale;
      tail_feedback.retained += __popc(retained_mask);
      tail_feedback.queue_rejects += feedback_queue_rejects;
    }
  } else if (threadIdx.x == 0) {
    tail_feedback.stale += feedback_stale;
    tail_feedback.queue_rejects += feedback_queue_rejects;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    const u32 slot = selected ? commit_rob_slots[position] : UINT32_MAX;
    const bool hit = selected && slot != UINT32_MAX;
    const bool speculative_hit =
      hit && frontier_rob[slot].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool controller_utility_hit =
      speculative_hit &&
      (frontier_rob[slot].flags &
         kFrontierRobFlagUtilityAccounted) == 0;
    if (hit) {
      if (controller_utility_hit) {
        frontier_rob[slot].flags |=
          kFrontierRobFlagUtilityAccounted;
      }
      frontier_rob[slot].state =
        static_cast<u8>(FrontierRequestState::committed);
      graph_record_slots[position] =
        kGraphScratchBit |
        static_cast<u32>(frontier_rob[slot].scratch_slot);
    }
    const u32 miss_mask = __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      critical_fetch_to_commit[fetch] = position;
      critical_fetch_handles[fetch] = selected_handles[position];
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 speculative_hit_mask =
      __ballot_sync(full_warp, speculative_hit);
    const u32 controller_utility_hit_mask =
      __ballot_sync(full_warp, controller_utility_hit);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      const u32 misses = __popc(miss_mask);
      critical_fetch_count = __popc(miss_mask);
      critical_misses += critical_fetch_count;
      critical_rob_hits += hits;
      const u32 promoted = __popc(speculative_hit_mask);
      speculative_promoted += promoted;
      core_prefetch_promoted += hits - promoted;
      // Promotion remains a physical/usefulness telemetry event, but a
      // request already credited when it survived an exact certificate must
      // not train the width controller a second time.
      tail_feedback.promoted +=
        __popc(controller_utility_hit_mask);
      if (issue_wave) {
        tail_feedback.core_hits += hits;
        tail_feedback.core_misses += misses;
      }
    }
  }
  __syncthreads();
}

// Publish the exact next commit prefix into positional core ROB slots. The
// current round has already consumed every COMMITTED payload, so those slots
// can be recycled without an associative ROB pass. One lane owns one slot;
// no atomics or global queue are involved in this metadata transition.
__device__ __forceinline__ void prepare_exact_core_frontier(
    const u64* issue_handles, const u16* issue_ranks, u32& issue_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count, u32& issue_epoch,
    u32& issue_epochs, u64& issue_width_sum,
    u64& issue_width_capacity_sum, u32 issue_width_capacity,
    u32& observed_max_issue_width) {
  constexpr u32 full_warp = 0xffffffffu;
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  if (threadIdx.x < 32) {
    const u32 lane = threadIdx.x;
    const u32 admitted = min(issue_count, core_slot_count);
    if (lane < core_slot_count) {
      FrontierRobEntry& entry = frontier_rob[lane];
      entry = {};
      if (lane < admitted) {
        entry.node_handle = issue_handles[lane];
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank = issue_ranks[lane];
        entry.scratch_slot = static_cast<u8>(lane);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::critical);
      }
    }
    __syncwarp(full_warp);
    if (lane == 0) {
      ++issue_epoch;
      ++issue_epochs;
      issue_count = admitted;
      issue_width_sum += admitted;
      issue_width_capacity_sum += issue_width_capacity;
      observed_max_issue_width =
        max(observed_max_issue_width, admitted);
    }
  }
  __syncthreads();
}

// Build a bounded look-ahead suffix from the *pre-merge* authoritative Beam.
// The first commit_count unexpanded entries are being expanded by the current
// epoch, so the following entries are the only legal candidates for an early
// shadow read.  Their handles/ranks are merely ROB metadata: this helper never
// touches Beam distances, expanded bits, or visited state.  The exact
// Stable-Run merge later reconciles these entries by handle; a displaced
// prediction is therefore harmless and is charged as speculative waste.
//
// This path is deliberately one warp and one ROB-tail allocation.  It runs
// after the current critical records have been admitted, so critical requests
// retain queue priority while the shadow suffix is in flight during
// neighbor/PQ scoring and authoritative merge.
__device__ __noinline__ void prepare_early_shadow_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 commit_count, u32 requested_count, FrontierRobEntry* frontier_rob,
    u32 core_slot_count, u32& issue_count, u32& issue_epoch) {
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  requested_count = min(
    requested_count,
    static_cast<u32>(kPersistentFrontierRobCapacity) - core_slot_count);

  // Reclaim terminal metadata before calculating the free tail mask.  No
  // payload can still be owned by a terminal slot at this point: the caller
  // has drained the preceding tail batch before entering the commit epoch.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = frontier_rob[threadIdx.x];
    if (threadIdx.x >= core_slot_count &&
        (entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    issue_count = 0;
    bool tail_empty = true;
    for (u32 slot = core_slot_count;
         slot < kPersistentFrontierRobCapacity; ++slot) {
      if (frontier_rob[slot].state !=
          static_cast<u8>(FrontierRequestState::init)) {
        tail_empty = false;
        break;
      }
    }
    if (tail_empty) {
      u32 skipped = 0;
      for (u32 rank = 0;
           rank < beam_count && issue_count < requested_count; ++rank) {
        if (beam_expanded[rank] != 0) continue;
        if (skipped < commit_count) {
          ++skipped;
          continue;
        }
        const u32 destination = core_slot_count + issue_count;
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = beam_handles[rank];
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank =
          static_cast<u16>(min(rank, static_cast<u32>(UINT16_MAX)));
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
        entry.flags = kFrontierRobFlagEarlyShadow;
        ++issue_count;
      }
      if (issue_count != 0) ++issue_epoch;
    }
  }
  __syncthreads();
}

// Admit a candidate-only shadow frontier into the ROB tail.  Unlike
// prepare_early_shadow_frontier(), the input is produced from a sorted,
// distance-ready Stable-Run leaf.  The entries are still predictions: only a
// later exact Stable-Run certificate may promote them.  Keeping admission in
// the contiguous ROB suffix preserves coalesced scratch addressing and lets
// the existing split critical/speculative descriptor path consume it without
// a global queue or another mapping structure.
__device__ __noinline__ void prepare_candidate_shadow_frontier(
    const u64* candidate_handles, const u16* candidate_ranks,
    u32 candidate_count, u32 requested_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    u32& issue_count, u32& issue_epoch) {
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  requested_count = min(
    min(requested_count, candidate_count),
    static_cast<u32>(kPersistentFrontierRobCapacity) - core_slot_count);

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = frontier_rob[threadIdx.x];
    if (threadIdx.x >= core_slot_count &&
        (entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    issue_count = 0;
    bool tail_empty = true;
    for (u32 slot = core_slot_count;
         slot < kPersistentFrontierRobCapacity; ++slot) {
      if (frontier_rob[slot].state !=
          static_cast<u8>(FrontierRequestState::init)) {
        tail_empty = false;
        break;
      }
    }
    if (tail_empty) {
      for (u32 position = 0; position < requested_count; ++position) {
        const u64 handle = candidate_handles[position];
        if (handle == kInvalidDeviceHandle) continue;
        const u32 destination = core_slot_count + issue_count;
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = handle;
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank = candidate_ranks[position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
        entry.flags = kFrontierRobFlagEarlyShadow;
        ++issue_count;
      }
      if (issue_count != 0) ++issue_epoch;
    }
  }
  __syncthreads();
}

// Close the utility epoch of the preceding pre-merge shadow wave before
// admitting another one.  A speculative payload that was selected by the
// exact certificate has already transitioned to COMMITTED and is released
// only after its neighbors have been copied.  Consequently every remaining
// VALIDATED tail entry at this point is a real prediction miss and can be
// charged/reclaimed without inspecting Beam or touching authoritative state.
//
// The controller update is deliberately query-CTA local.  It consumes one
// complete issue -> exact-certificate -> commit observation and then resets
// the feedback before the next wave is formed; no global queue, atomics, or
// CPU scheduler participate.
__device__ __forceinline__ void close_early_shadow_epoch(
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    adaptive_frontier::ControllerState& controller,
    TailFrontierFeedback& feedback, u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& core_prefetch_stale) {
  constexpr u32 full_warp = 0xffffffffu;
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = frontier_rob[slot];
    const bool tail_slot = slot >= core_slot_count;
    const bool unconsumed =
      tail_slot &&
      entry.state ==
        static_cast<u8>(FrontierRequestState::validated);
    const bool discarded_speculative =
      unconsumed &&
      entry.priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool discarded_core = unconsumed && !discarded_speculative;
    u64 discarded_bytes =
      discarded_speculative ? entry.transfer_bytes : 0;
    if (tail_slot &&
        (unconsumed ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
    const u32 speculative_mask =
      __ballot_sync(full_warp, discarded_speculative);
    const u32 core_mask = __ballot_sync(full_warp, discarded_core);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 newly_stale = __popc(speculative_mask);
      speculative_stale += newly_stale;
      speculative_wasted_bytes += discarded_bytes;
      core_prefetch_stale += __popc(core_mask);
      feedback.stale += newly_stale;
      adaptive_frontier::update_issue_width(
        controller,
        adaptive_frontier::Feedback{
          .promoted = feedback.promoted,
          .retained = feedback.retained,
          .stale = feedback.stale,
          .queue_rejects = feedback.queue_rejects,
          .critical_misses = feedback.core_misses,
          .tail_admitted = feedback.tail_admitted,
          .commit_waves_observed = feedback.commit_waves_observed,
          .commit_waves_covered = feedback.commit_waves_covered,
        });
      feedback = {};
    }
  }
  __syncthreads();
}

// Reconcile the exact post-merge preview with the next authoritative commit.
// Core entries are positional: preview rank i owns ROB/scratch slot i. A
// rejected or invalid preview therefore releases exactly its own scratch slot
// for the critical fallback. Keeping this transition out of process_query
// shortens the persistent kernel's live range and avoids an associative ROB
// scan on the steady-state path.
__device__ __forceinline__ void reconcile_exact_core_frontier(
    const u64* selected_handles, u32 selected_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    u32* commit_rob_slots, u32* graph_record_slots,
    u64* critical_fetch_handles, u32* critical_fetch_to_commit,
    u32* critical_fetch_destination_slots, u32& critical_fetch_count,
    u32& critical_rob_hits, u32& critical_misses,
    u32& core_prefetch_promoted, u32& core_prefetch_stale) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < 32) {
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    FrontierRobEntry* entry =
      position < core_slot_count ? frontier_rob + position : nullptr;
    const bool hit =
      selected && entry != nullptr &&
      (entry->state ==
         static_cast<u8>(FrontierRequestState::validated) ||
       entry->state ==
         static_cast<u8>(FrontierRequestState::committed)) &&
      entry->node_handle == selected_handles[position];
    const bool stale =
      entry != nullptr &&
      entry->state != static_cast<u8>(FrontierRequestState::init) &&
      !hit;
    commit_rob_slots[position] = hit ? position : UINT32_MAX;
    if (selected) {
      graph_record_slots[position] = hit
        ? kGraphScratchBit | static_cast<u32>(entry->scratch_slot)
        : UINT32_MAX;
    }
    if (hit) {
      entry->state = static_cast<u8>(FrontierRequestState::committed);
    } else if (stale) {
      *entry = {};
    }

    const u32 miss_mask =
      __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      critical_fetch_handles[fetch] = selected_handles[position];
      critical_fetch_to_commit[fetch] = position;
      critical_fetch_destination_slots[fetch] = position;
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 stale_mask = __ballot_sync(full_warp, stale);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      critical_fetch_count = __popc(miss_mask);
      critical_rob_hits += hits;
      critical_misses += critical_fetch_count;
      core_prefetch_promoted += hits;
      core_prefetch_stale += __popc(stale_mask);
    }
  }
  __syncthreads();
}

// Consume an arbitrary-slot mapping produced by
// prepare_issue_frontier_entries().  A mapping is evidence, not authority:
// only a completed, validated payload whose handle still equals the frozen
// Commit Frontier may transition to COMMITTED.  Every other case is compacted
// into the mandatory critical retry path without waiting on speculative I/O.
__device__ __noinline__ void reconcile_certified_commit_frontier(
    const CertifiedCommitReconcileContext& context,
    u32 selected_count, bool issue_wave) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < 32) {
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    const u32 slot = selected
      ? context.certified_rob_slots[position] : UINT32_MAX;
    FrontierRobEntry* entry =
      slot < kPersistentFrontierRobCapacity
        ? context.frontier_rob + slot : nullptr;
    const bool hit =
      selected && entry != nullptr &&
      (entry->state == static_cast<u8>(
         FrontierRequestState::validated) ||
       entry->state == static_cast<u8>(
         FrontierRequestState::committed)) &&
      entry->node_handle == context.selected_handles[position];
    const bool speculative_hit =
      hit && entry->priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool controller_utility_hit =
      speculative_hit &&
      (entry->flags & kFrontierRobFlagUtilityAccounted) == 0;
    const bool stale_core =
      selected && entry != nullptr && !hit &&
      entry->state != static_cast<u8>(FrontierRequestState::init) &&
      entry->state != static_cast<u8>(FrontierRequestState::inflight) &&
      entry->priority ==
        static_cast<u8>(DirectBatchPriority::critical);

    context.commit_rob_slots[position] = hit ? slot : UINT32_MAX;
    if (selected) {
      context.graph_record_slots[position] = hit
        ? kGraphScratchBit | static_cast<u32>(entry->scratch_slot)
        : UINT32_MAX;
    }
    if (hit) {
      if (controller_utility_hit) {
        entry->flags |= kFrontierRobFlagUtilityAccounted;
      }
      entry->state =
        static_cast<u8>(FrontierRequestState::committed);
    }

    const u32 miss_mask =
      __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      context.critical_fetch_handles[fetch] =
        context.selected_handles[position];
      context.critical_fetch_to_commit[fetch] = position;
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 speculative_hit_mask =
      __ballot_sync(full_warp, speculative_hit);
    const u32 controller_utility_hit_mask =
      __ballot_sync(full_warp, controller_utility_hit);
    const u32 stale_core_mask =
      __ballot_sync(full_warp, stale_core);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      const u32 misses = __popc(miss_mask);
      const u32 promoted = __popc(speculative_hit_mask);
      *context.critical_fetch_count = misses;
      *context.critical_rob_hits += hits;
      *context.critical_misses += misses;
      *context.speculative_promoted += promoted;
      *context.core_prefetch_promoted += hits - promoted;
      *context.core_prefetch_stale += __popc(stale_core_mask);
      context.feedback->promoted +=
        __popc(controller_utility_hit_mask);
      if (issue_wave) {
        context.feedback->core_hits += hits;
        context.feedback->core_misses += misses;
      }
    }
  }
  __syncthreads();
}

// Reserve scratch destinations for authoritative misses after every
// asynchronous wave has been drained. Critical reads must never wait behind
// validated shadow records merely because those records still occupy ROB
// slots. One warp allocates in strict priority order:
//
//   empty/stale -> speculative validated -> non-selected critical validated
//
// COMMITTED records remain protected until their neighbors have been copied,
// and any unexpectedly live request remains protected against a late DMA.
// Thus critical-first scheduling is enforced without a global queue, an
// atomic allocator, or an additional query-level synchronization point.
__device__ __noinline__ void reserve_critical_fetch_destinations(
    FrontierRobEntry* frontier_rob, u32 critical_fetch_count,
    u32* critical_fetch_destination_slots, u32& graph_failed,
    u32& speculative_stale, u64& speculative_wasted_bytes,
    u32& core_prefetch_stale, TailFrontierFeedback& feedback) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 lane = threadIdx.x;
    const FrontierRobEntry& resident = frontier_rob[lane];
    const u8 state = resident.state;
    const bool immediately_free =
      state == static_cast<u8>(FrontierRequestState::init) ||
      state == static_cast<u8>(FrontierRequestState::stale);
    const bool validated =
      state == static_cast<u8>(FrontierRequestState::validated);
    const bool speculative_victim =
      validated &&
      resident.priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool critical_victim = validated && !speculative_victim;

    const u32 free_mask =
      __ballot_sync(full_warp, immediately_free);
    const u32 speculative_mask =
      __ballot_sync(full_warp, speculative_victim);
    const u32 critical_mask =
      __ballot_sync(full_warp, critical_victim);
    const u32 free_count = __popc(free_mask);
    const u32 speculative_count = __popc(speculative_mask);

    u32 destination = UINT32_MAX;
    if (lane < critical_fetch_count) {
      u32 candidate_mask = 0;
      u32 ordinal = lane;
      if (ordinal < free_count) {
        candidate_mask = free_mask;
      } else {
        ordinal -= free_count;
        if (ordinal < speculative_count) {
          candidate_mask = speculative_mask;
        } else {
          ordinal -= speculative_count;
          candidate_mask = critical_mask;
        }
      }
      for (u32 prior = 0;
           prior < ordinal && candidate_mask != 0; ++prior) {
        candidate_mask &= candidate_mask - 1u;
      }
      if (candidate_mask != 0) {
        destination =
          static_cast<u32>(__ffs(static_cast<int>(candidate_mask)) - 1);
      }
      critical_fetch_destination_slots[lane] = destination;
    }

    const bool allocation_failed =
      lane < critical_fetch_count && destination == UINT32_MAX;
    const u32 allocation_failed_mask =
      __ballot_sync(full_warp, allocation_failed);
    if (lane == 0 && allocation_failed_mask != 0) {
      graph_failed = 5u;
    }

    bool evicted_speculative = false;
    bool evicted_critical = false;
    u64 evicted_bytes = 0;
    if (destination != UINT32_MAX) {
      FrontierRobEntry& victim = frontier_rob[destination];
      evicted_speculative =
        victim.state ==
          static_cast<u8>(FrontierRequestState::validated) &&
        victim.priority ==
          static_cast<u8>(DirectBatchPriority::speculative);
      evicted_critical =
        victim.state ==
          static_cast<u8>(FrontierRequestState::validated) &&
        !evicted_speculative;
      evicted_bytes =
        evicted_speculative ? victim.transfer_bytes : 0;
      victim = {};
    }
    const u32 evicted_speculative_mask =
      __ballot_sync(full_warp, evicted_speculative);
    const u32 evicted_critical_mask =
      __ballot_sync(full_warp, evicted_critical);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      evicted_bytes +=
        __shfl_down_sync(full_warp, evicted_bytes, offset);
    }
    if (lane == 0) {
      const u32 evicted_speculative_count =
        __popc(evicted_speculative_mask);
      speculative_stale += evicted_speculative_count;
      speculative_wasted_bytes += evicted_bytes;
      core_prefetch_stale += __popc(evicted_critical_mask);
      feedback.stale += evicted_speculative_count;
    }
  }
  __syncthreads();
}

// f64 fallback for a row whose f32 accumulation overflowed.  Kept out of
// line so its registers never pressure the hot rotation loop.
}  // namespace gpu_search::persistent_kernel_detail

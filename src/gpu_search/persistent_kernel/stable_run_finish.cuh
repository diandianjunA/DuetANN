#pragma once

#include "gpu_search/persistent_kernel/frontier_certificate.cuh"

namespace gpu_search::persistent_kernel_detail {

template <class OriginExpanded>
__device__ __noinline__ void finish_approximate_stable_runs_impl(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, const u64* origin_handles,
  const OriginExpanded* origin_expanded, u64* scratch_handles,
  u8* scratch_flags, f32* scratch_distances, CandidateWorkspace& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool fused_compact_materialize = false) {
  if (state.prepared == 0) {
    if (threadIdx.x == 0) beam_count = 0;
    __syncthreads();
    return;
  }
  const u32 origin_count = state.original_count;
  const bool finished_prepared_fused_tree =
    state.compact != 0 && fused_compact_materialize &&
    state.origin_copied == 0 && state.fused_tree_prepared != 0;
  if (state.compact != 0 && fused_compact_materialize &&
      state.origin_copied == 0) {
    if (state.fused_tree_prepared != 0) {
      finish_fused_stable_frontier_materialization(
        beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
        beam_capacity, scratch_handles, scratch_flags, scratch_distances,
        state.candidate_run_count, workspace.arrays, state, cycle_breakdown);
    } else {
      materialize_fused_stable_candidate_runs(
        beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
        beam_capacity, state.original_count, beam_handles, beam_expanded,
        scratch_handles, scratch_flags, scratch_distances,
        state.candidate_run_count, workspace.arrays);
    }
  } else if (state.compact != 0) {
    // First fold old Beam + candidate runs 0/1, then fold candidate runs 2/3.
    // Both levels execute after RDMA issue, providing useful communication
    // overlap without changing the Stable-Run order or tie semantics.
    materialize_stable_candidate_runs(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, state.original_count, origin_handles, origin_expanded,
      origin_count, scratch_handles, scratch_flags, scratch_distances, 2u,
      workspace.arrays, false);
    // The second fold uses the freshly published Beam as its origin; its
    // bitmap is always the u8 on-Beam storage, even when the first fold's
    // origin snapshot lives in the u16 workspace metadata.
    materialize_stable_candidate_runs<u8>(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, beam_count, beam_handles, beam_expanded, beam_count,
      scratch_handles + 2u * beam_capacity, scratch_flags + 2u * beam_capacity,
      scratch_distances + 2u * beam_capacity, 2u, workspace.arrays, false);
  } else {
    materialize_stable_candidate_runs(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, state.original_count, origin_handles, origin_expanded,
      origin_count, scratch_handles, scratch_flags, scratch_distances,
      state.candidate_run_count, workspace.arrays, false);
  }
  if (threadIdx.x == 0 && cycle_breakdown != nullptr &&
      !finished_prepared_fused_tree) {
    cycle_breakdown->materialize += clock64() - state.phase_started;
  }
  __syncthreads();
}

// The origin snapshot lives in the workspace (u16 metadata) when the caller
// copied it, and in the authoritative Beam (u8 bitmap) otherwise. Dispatch on
// the two storage types so the merged body stays generic.
__device__ __noinline__ void finish_approximate_stable_runs(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, CandidateWorkspace& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool fused_compact_materialize = false) {
  if (state.origin_copied != 0) {
    finish_approximate_stable_runs_impl(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, workspace.arrays.handles + beam_capacity,
      workspace.arrays.expanded + beam_capacity, scratch_handles, scratch_flags,
      scratch_distances, workspace, state, cycle_breakdown,
      fused_compact_materialize);
  } else {
    finish_approximate_stable_runs_impl(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, beam_handles, beam_expanded, scratch_handles,
      scratch_flags, scratch_distances, workspace, state, cycle_breakdown,
      fused_compact_materialize);
  }
}

__device__ __noinline__ void merge_approximate_stable_runs(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, CandidateWorkspace& workspace,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  __shared__ StableMergePreparedState state;
  prepare_approximate_stable_runs(
    candidate_handles, candidate_distances, candidate_count, beam_handles,
    beam_ids, beam_distances, beam_expanded, beam_count, beam_capacity,
    scratch_handles, scratch_flags, scratch_distances, workspace, state,
    cycle_breakdown);
  finish_approximate_stable_runs(
    beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
    beam_capacity, scratch_handles, scratch_flags, scratch_distances, workspace,
    state, cycle_breakdown);
}

__device__ void merge_approximate_into_beam(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u64* compact_scratch_handles,
  u8* compact_scratch_expanded, f32* compact_scratch_distances,
  CandidateWorkspace& workspace,
  BeamMergePolicy policy = BeamMergePolicy::legacy,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  const u32 existing_count = beam_count;
  const u32 merge_count = existing_count + candidate_count;
  if (blockDim.x != kApproximateSortThreadsWide &&
      blockDim.x != kApproximateSortThreadsCompact) {
    if (threadIdx.x == 0) beam_count = 0;
    __syncthreads();
    return;
  }
  if (policy == BeamMergePolicy::stable_run) {
    merge_approximate_stable_runs(
      candidate_handles, candidate_distances, candidate_count, beam_handles,
      beam_ids, beam_distances, beam_expanded, beam_count, beam_capacity,
      compact_scratch_handles, compact_scratch_expanded,
      compact_scratch_distances, workspace, cycle_breakdown);
    return;
  }
  if (blockDim.x == kApproximateSortThreadsWide) {
#if DVSTOR_PERSISTENT_WIDE_SORT
    merge_approximate_radix<ApproximateBlockSortWide,
                            kApproximateSortItemsWide>(
      candidate_handles, candidate_distances, candidate_count, beam_handles,
      beam_ids, beam_distances, beam_expanded, beam_count, beam_capacity,
      existing_count, merge_count, workspace.sort.radix_sort_wide);
#else
    // Slim builds reject 256-thread launches on the host; trap rather than
    // silently corrupt the Beam if one ever slips through.
    __trap();
#endif
  } else {
#if DVSTOR_PERSISTENT_LEGACY_MERGE
    merge_approximate_compact(
      candidate_handles, candidate_distances, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, beam_capacity, existing_count,
      merge_count, compact_scratch_handles, compact_scratch_expanded,
      compact_scratch_distances, workspace);
#else
    // Slim builds reject the legacy beam-merge policy on the host.
    __trap();
#endif
  }
}

}  // namespace gpu_search::persistent_kernel_detail

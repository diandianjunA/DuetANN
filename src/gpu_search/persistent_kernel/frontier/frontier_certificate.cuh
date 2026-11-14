#pragma once

#include "gpu_search/persistent_kernel/merge/stable_run_merge.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ void extend_fused_stable_candidate_tree(
  const f32* beam_distances, u32 beam_capacity, u32 origin_count,
  const u64* origin_handles, const u8* origin_expanded,
  const u64* scratch_handles, const u8* scratch_flags,
  const f32* scratch_distances, u32 candidate_run_count,
  CandidateWorkspaceArrays& output, u32 begin_rank, u32 end_rank) {
  candidate_run_count = min(candidate_run_count, 4u);
  const u32 run0_count = candidate_run_count > 0 ? beam_capacity : 0u;
  const u32 run1_count = candidate_run_count > 1 ? beam_capacity : 0u;
  const u32 run2_count = candidate_run_count > 2 ? beam_capacity : 0u;
  begin_rank = min(begin_rank, beam_capacity);
  end_rank = min(max(end_rank, begin_rank), beam_capacity);

  // Stage 1A: old Beam + run0 -> output[0, K).
  // Stage 1B: run1 + run2 -> output[K, 2K).
  for (u32 rank = begin_rank + threadIdx.x; rank < end_rank;
       rank += blockDim.x) {
    const u32 old_index = stable_merge_a_corank(
      rank, beam_distances, origin_count, scratch_distances, run0_count);
    const u32 run0_index = rank - old_index;
    const bool take_old =
      old_index < origin_count &&
      (run0_index >= run0_count ||
       !(scratch_distances[run0_index] < beam_distances[old_index]));
    if (take_old) {
      output.handles[rank] = origin_handles[old_index];
      output.distances[rank] = beam_distances[old_index];
      output.expanded[rank] = origin_expanded[old_index];
    } else if (run0_index < run0_count) {
      output.handles[rank] = scratch_handles[run0_index];
      output.distances[rank] = scratch_distances[run0_index];
      output.expanded[rank] = scratch_flags[run0_index];
    } else {
      output.handles[rank] = kInvalidDeviceHandle;
      output.distances[rank] = FLT_MAX;
      output.expanded[rank] = 0;
    }

    const u32 destination = beam_capacity + rank;
    if (run2_count == 0 && rank < run1_count) {
      const u32 source = beam_capacity + rank;
      output.handles[destination] = scratch_handles[source];
      output.distances[destination] = scratch_distances[source];
      output.expanded[destination] = scratch_flags[source];
    } else {
      const u32 run1_index = stable_merge_a_corank(
        rank, scratch_distances + beam_capacity, run1_count,
        scratch_distances + 2u * beam_capacity, run2_count);
      const u32 run2_index = rank - run1_index;
      const bool take_run1 =
        run1_index < run1_count &&
        (run2_index >= run2_count ||
         !(scratch_distances[2u * beam_capacity + run2_index] <
           scratch_distances[beam_capacity + run1_index]));
      if (take_run1) {
        const u32 source = beam_capacity + run1_index;
        output.handles[destination] = scratch_handles[source];
        output.distances[destination] = scratch_distances[source];
        output.expanded[destination] = scratch_flags[source];
      } else if (run2_index < run2_count) {
        const u32 source = 2u * beam_capacity + run2_index;
        output.handles[destination] = scratch_handles[source];
        output.distances[destination] = scratch_distances[source];
        output.expanded[destination] = scratch_flags[source];
      } else {
        output.handles[destination] = kInvalidDeviceHandle;
        output.distances[destination] = FLT_MAX;
        output.expanded[destination] = 0;
      }
    }
  }
  __syncthreads();

  // Stage 2: (old, run0) + (run1, run2) -> output[2K, 3K).
  for (u32 rank = begin_rank + threadIdx.x; rank < end_rank;
       rank += blockDim.x) {
    if (run1_count == 0) {
      const u32 destination = 2u * beam_capacity + rank;
      output.handles[destination] = output.handles[rank];
      output.distances[destination] = output.distances[rank];
      output.expanded[destination] = output.expanded[rank];
      continue;
    }
    const u32 left_index =
      stable_merge_a_corank(rank, output.distances, end_rank,
                            output.distances + beam_capacity, end_rank);
    const u32 right_index = rank - left_index;
    const bool take_left = left_index < end_rank &&
                           (right_index >= end_rank ||
                            !(output.distances[beam_capacity + right_index] <
                              output.distances[left_index]));
    const u32 destination = 2u * beam_capacity + rank;
    const u32 source = take_left ? left_index : beam_capacity + right_index;
    output.handles[destination] = output.handles[source];
    output.distances[destination] = output.distances[source];
    output.expanded[destination] = output.expanded[source];
  }
  __syncthreads();
}

// Compatibility/full-materialization entry point.  ASFE uses the incremental
// certificate path below; callers outside that path retain the original
// complete-tree behavior.
__device__ __noinline__ void prepare_fused_stable_candidate_tree(
  const f32* beam_distances, u32 beam_capacity, u32 origin_count,
  const u64* origin_handles, const u8* origin_expanded,
  const u64* scratch_handles, const u8* scratch_flags,
  const f32* scratch_distances, u32 candidate_run_count,
  CandidateWorkspaceArrays& output) {
  extend_fused_stable_candidate_tree(
    beam_distances, beam_capacity, origin_count, origin_handles,
    origin_expanded, scratch_handles, scratch_flags, scratch_distances,
    candidate_run_count, output, 0, beam_capacity);
}

// Publish a disjoint final-rank interval from the prepared tree. The prefix
// wins ties, preserving old < run0 < run1 < run2 < run3.
__device__ __noinline__ void materialize_fused_stable_candidate_range(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32 beam_capacity, u32 begin_rank, u32 end_rank, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, const CandidateWorkspaceArrays& output) {
  candidate_run_count = min(candidate_run_count, 4u);
  const u32 run3_count = candidate_run_count > 3 ? beam_capacity : 0u;
  // Stable-Run sorting places every semantic-valid item before the invalid
  // padding tail.  Therefore an invalid head proves that the complete fourth
  // leaf is empty.  Merging A with an empty B is exactly A, so avoid K
  // independent co-rank searches in the common <=3-leaf case while retaining
  // the original path whenever dynamic updates populate leaf 3.
  const bool run3_empty =
    run3_count == 0 ||
    !stable_run_item_valid(scratch_handles[3u * beam_capacity],
                           scratch_distances[3u * beam_capacity]);
  begin_rank = min(begin_rank, beam_capacity);
  end_rank = min(max(end_rank, begin_rank), beam_capacity);
  for (u32 rank = begin_rank + threadIdx.x; rank < end_rank;
       rank += blockDim.x) {
    if (run3_empty) {
      const u32 source = 2u * beam_capacity + rank;
      beam_handles[rank] = output.handles[source];
      beam_distances[rank] = output.distances[source];
      beam_expanded[rank] = output.expanded[source];
      beam_ids[rank] = UINT32_MAX;
      continue;
    }
    const u32 prefix_index = stable_merge_a_corank(
      rank, output.distances + 2u * beam_capacity, beam_capacity,
      scratch_distances + 3u * beam_capacity, run3_count);
    const u32 run3_index = rank - prefix_index;
    const bool take_prefix =
      prefix_index < beam_capacity &&
      (run3_index >= run3_count ||
       !(scratch_distances[3u * beam_capacity + run3_index] <
         output.distances[2u * beam_capacity + prefix_index]));
    if (take_prefix) {
      const u32 source = 2u * beam_capacity + prefix_index;
      beam_handles[rank] = output.handles[source];
      beam_distances[rank] = output.distances[source];
      beam_expanded[rank] = output.expanded[source];
    } else {
      const u32 source = 3u * beam_capacity + run3_index;
      beam_handles[rank] = scratch_handles[source];
      beam_distances[rank] = scratch_distances[source];
      beam_expanded[rank] = scratch_flags[source];
    }
    beam_ids[rank] = UINT32_MAX;
  }
  __syncthreads();
}

__device__ __forceinline__ void finalize_fused_stable_candidate_runs(
  const u64* beam_handles, const f32* beam_distances, u32& beam_count,
  u32 beam_capacity) {
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < beam_capacity &&
           stable_run_item_valid(beam_handles[valid], beam_distances[valid])) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}

// Unsplit compatibility path used outside the ASFE phase pipeline.
__device__ __noinline__ void materialize_fused_stable_candidate_runs(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u32 origin_count,
  const u64* origin_handles, const u8* origin_expanded,
  const u64* scratch_handles, const u8* scratch_flags,
  const f32* scratch_distances, u32 candidate_run_count,
  CandidateWorkspaceArrays& output) {
  prepare_fused_stable_candidate_tree(
    beam_distances, beam_capacity, origin_count, origin_handles,
    origin_expanded, scratch_handles, scratch_flags, scratch_distances,
    candidate_run_count, output);
  materialize_fused_stable_candidate_range(
    beam_handles, beam_ids, beam_distances, beam_expanded, beam_capacity, 0,
    beam_capacity, scratch_handles, scratch_flags, scratch_distances,
    candidate_run_count, output);
  finalize_fused_stable_candidate_runs(beam_handles, beam_distances, beam_count,
                                       beam_capacity);
}

// Materialize the smallest warp-sized authoritative prefix containing the
// requested number of unexpanded entries from an already prepared merge
// tree.  Keeping tree construction out of this function is deliberate: the
// tree inputs (old Beam/origin metadata) are dead before the rank publisher
// starts, so ptxas need not spill both input and output argument sets across
// the same call frame.
__device__ __noinline__ void materialize_fused_stable_frontier_prefix(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32 beam_capacity, const u64* scratch_handles, const u8* scratch_flags,
  const f32* scratch_distances, u32 candidate_run_count,
  const CandidateWorkspaceArrays& output, u32 issue_capacity,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  constexpr u32 tile_width = 32;
  candidate_run_count = min(candidate_run_count, 4u);
  const u32 run3_count = candidate_run_count > 3 ? beam_capacity : 0u;
  const bool run3_empty =
    run3_count == 0 ||
    !stable_run_item_valid(scratch_handles[3u * beam_capacity],
                           scratch_distances[3u * beam_capacity]);
  issue_capacity = min(issue_capacity, beam_capacity);
  if (threadIdx.x == 0) {
    issue_count = 0;
    state.fused_tree_prefix = beam_capacity;
    state.deferred_prefix = 0;
    state.materialized_prefix = 0;
  }
  __syncthreads();

  for (u32 tile_begin = 0; tile_begin < beam_capacity;
       tile_begin += tile_width) {
    const u32 tile_end = min(tile_begin + tile_width, beam_capacity);
    // Inline the final fold for the narrow tile instead of nesting a device
    // call.  The nested call forced ptxas to preserve every Beam/scratch/ROB
    // pointer across each tile and dominated this control path's local-memory
    // traffic.
    for (u32 rank = tile_begin + threadIdx.x; rank < tile_end;
         rank += blockDim.x) {
      if (run3_empty) {
        const u32 source = 2u * beam_capacity + rank;
        beam_handles[rank] = output.handles[source];
        beam_distances[rank] = output.distances[source];
        beam_expanded[rank] = output.expanded[source];
        beam_ids[rank] = UINT32_MAX;
        continue;
      }
      const u32 prefix_index = stable_merge_a_corank(
        rank, output.distances + 2u * beam_capacity, beam_capacity,
        scratch_distances + 3u * beam_capacity, run3_count);
      const u32 run3_index = rank - prefix_index;
      const bool take_prefix =
        prefix_index < beam_capacity &&
        (run3_index >= run3_count ||
         !(scratch_distances[3u * beam_capacity + run3_index] <
           output.distances[2u * beam_capacity + prefix_index]));
      if (take_prefix) {
        const u32 source = 2u * beam_capacity + prefix_index;
        beam_handles[rank] = output.handles[source];
        beam_distances[rank] = output.distances[source];
        beam_expanded[rank] = output.expanded[source];
      } else {
        const u32 source = 3u * beam_capacity + run3_index;
        beam_handles[rank] = scratch_handles[source];
        beam_distances[rank] = scratch_distances[source];
        beam_expanded[rank] = scratch_flags[source];
      }
      beam_ids[rank] = UINT32_MAX;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (u32 rank = tile_begin;
           rank < tile_end && issue_count < issue_capacity; ++rank) {
        if (!stable_run_item_valid(beam_handles[rank], beam_distances[rank]) ||
            beam_expanded[rank] != 0) {
          continue;
        }
        issue_handles[issue_count] = beam_handles[rank];
        issue_ranks[issue_count] = static_cast<u16>(rank);
        ++issue_count;
      }
      state.materialized_prefix = tile_end;
    }
    __syncthreads();
    if (issue_count >= issue_capacity) break;
  }
  if (threadIdx.x == 0) {
    state.fused_tree_prepared = 1;
    if (cycle_breakdown != nullptr) {
      const u64 now = clock64();
      cycle_breakdown->materialize += now - state.phase_started;
      state.phase_started = now;
    }
  }
  __syncthreads();
}

__device__ __noinline__ void preview_tree_stable_unexpanded_frontier(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_count, u32 beam_capacity, const u64* scratch_handles,
  const f32* scratch_distances, u32 candidate_run_count, u32 issue_capacity,
  CandidateWorkspaceArrays& workspace, u64* output_handles, u16* output_ranks,
  u32& output_count);

// Dominance-Envelope Exact Certificate (DEEC).
//
// Let O be the old Beam after removing expanded entries, and let O[m-1] be
// an anchor with distance T.  Because the Stable-Run merge places the whole
// old run before every candidate run on an equal-distance tie, the set
//
//   O[0:m] union { raw candidate a | distance(a) < T }
//
// sorted by the production Stable-Run order is an exact prefix of the next
// compact-unexpanded stream.  Among anchors relevant to the controller's
// desired Issue Frontier, we choose the widest complete envelope which fits
// in the GPU-resident ROB and covers the next Commit Frontier.  Thus stable
// frontiers naturally expose an exact speculative tail, while high-turnover
// frontiers contract to a narrow core or fall back to the complete prepared
// Stable-Run preview.  The rule uses no dataset-specific threshold.
//
// Four warps first count candidate membership for every possible anchor using
// coalesced 512-item leaf scans.  A second scan compacts only the selected
// envelope candidates. At most 32 combined old/candidate items are sorted by
// one warp using (distance, source-stable token). The same warp converts each
// compact position to its authoritative Beam rank using an expanded-prefix
// table and an old-Beam upper bound for candidate ties. No Beam, expanded, or
// visited state is modified. If no complete envelope fits, the caller must
// use a full exact Stable-Run certificate.
struct DominanceEnvelopeCertificateContext {
  const u64* candidate_handles;
  const f32* candidate_distances;
  const u64* beam_handles;
  const f32* beam_distances;
  const u8* beam_expanded;
  u64* prefix_handles;
  f32* prefix_distances;
  CandidateWorkspaceArrays* workspace;
  u64* output_handles;
  u16* output_ranks;
  u32* output_count;
  u32* envelope_size_out;
  u32 candidate_count;
  u32 beam_count;
  u32 beam_capacity;
  u32 commit_capacity;
  u32 issue_capacity;
};

static_assert(sizeof(DominanceEnvelopeCertificateContext) <= 128);

__device__ __noinline__ bool prepare_dominance_envelope_exact_certificate(
  const DominanceEnvelopeCertificateContext& context) {
  const u64* candidate_handles = context.candidate_handles;
  const f32* candidate_distances = context.candidate_distances;
  const u64* beam_handles = context.beam_handles;
  const f32* beam_distances = context.beam_distances;
  const u8* beam_expanded = context.beam_expanded;
  u64* prefix_handles = context.prefix_handles;
  f32* prefix_distances = context.prefix_distances;
  CandidateWorkspaceArrays& workspace = *context.workspace;
  u64* output_handles = context.output_handles;
  u16* output_ranks = context.output_ranks;
  u32& output_count = *context.output_count;
  u32& envelope_size_out = *context.envelope_size_out;
  // Keep the multi-warp candidate reservation counter separate from the
  // published certificate count.  Reusing output_count here makes every CTA
  // thread read it for the defensive compaction check and warp 0 overwrite it
  // with the final count without an intervening CTA barrier (a real shared
  // memory WAR race).  This aligned workspace slot is otherwise unused by the
  // at-most-32-item dominance envelope.
  auto* reservation_count = reinterpret_cast<u32*>(
    workspace.distances + kPersistentFrontierRobCapacity);
  u32 candidate_count = context.candidate_count;
  u32 beam_count = context.beam_count;
  u32 beam_capacity = context.beam_capacity;
  u32 commit_capacity = context.commit_capacity;
  u32 issue_capacity = context.issue_capacity;
  constexpr u32 kWarpWidth = 32;
  constexpr u32 kFullWarp = 0xffffffffu;
  constexpr u32 kWarpCount = kApproximateSortThreadsCompact / kWarpWidth;
  constexpr u32 kCandidatesPerWarp = kPersistentMaxMergeCandidates / kWarpCount;
  constexpr u32 kItemsPerLane = kCandidatesPerWarp / kWarpWidth;
  constexpr u32 kWarpCounterBase = kPersistentMaxExact * 2u - 2u * kWarpCount;
  static_assert(kWarpCount == 4);
  static_assert(kCandidatesPerWarp * kWarpCount ==
                kPersistentMaxMergeCandidates);
  static_assert(kItemsPerLane * kWarpWidth == kCandidatesPerWarp);

  candidate_count =
    min(candidate_count, static_cast<u32>(kPersistentMaxMergeCandidates));
  commit_capacity = min(min(commit_capacity, beam_capacity),
                        static_cast<u32>(kPersistentFrontierRobCapacity));
  issue_capacity = min(min(issue_capacity, beam_capacity),
                       static_cast<u32>(kPersistentFrontierRobCapacity));
  if (blockDim.x != kApproximateSortThreadsCompact || commit_capacity == 0 ||
      issue_capacity < commit_capacity) {
    if (threadIdx.x == 0) {
      output_count = 0;
      envelope_size_out = 0;
    }
    __syncthreads();
    return false;
  }

  const u32 index = threadIdx.x;
  const u32 warp = index / kWarpWidth;
  const u32 lane = index % kWarpWidth;

  // Compact only the old distances which can become anchors.  The total old
  // unexpanded count is retained as well so a short old run can use a virtual
  // +infinity anchor when all remaining candidates fit in the envelope.
  // beam_count may reach 256 with only 128 threads, so scan in 128-entry
  // chunks; the running cross-chunk bases are published through the two dead
  // slots just below the per-warp counters because register locals updated by
  // thread 0 alone are invisible to the other threads in the next chunk.
  const u32 lower_lanes = lane == 0 ? 0u : (u32{1} << lane) - 1u;
  u32 old_base = 0;
  u32 expanded_base = 0;
  for (u32 base = 0; base <= beam_count;
       base += kApproximateSortThreadsCompact) {
    const u32 chunk_index = base + index;
    const bool retain_old =
      chunk_index < beam_count && beam_expanded[chunk_index] == 0 &&
      stable_run_item_valid(beam_handles[chunk_index],
                            beam_distances[chunk_index]);
    const u32 retained_mask = __ballot_sync(kFullWarp, retain_old);
    const u32 expanded_mask =
      __ballot_sync(kFullWarp, chunk_index < beam_count &&
                                 beam_expanded[chunk_index] != 0);
    if (lane == 0) {
      workspace.expanded[kWarpCounterBase + warp] =
        static_cast<u16>(__popc(retained_mask));
      workspace.expanded[kWarpCounterBase + kWarpCount + warp] =
        static_cast<u16>(__popc(expanded_mask));
    }
    __syncthreads();

    u32 old_warp_base = old_base;
    u32 expanded_warp_base = expanded_base;
#pragma unroll
    for (u32 prior = 0; prior < kWarpCount; ++prior) {
      if (prior >= warp) break;
      old_warp_base += workspace.expanded[kWarpCounterBase + prior];
      expanded_warp_base +=
        workspace.expanded[kWarpCounterBase + kWarpCount + prior];
    }
    const u32 old_destination =
      old_warp_base + __popc(retained_mask & lower_lanes);
    if (retain_old && old_destination < issue_capacity) {
      prefix_handles[old_destination] = beam_handles[chunk_index];
      workspace.distances[old_destination] = beam_distances[chunk_index];
      prefix_distances[old_destination] = beam_distances[chunk_index];
      // Old tokens occupy [0, K); candidate tokens set the high bit.  This
      // single integer therefore encodes old-before-candidate ties as well
      // as stable order within each source run.
      workspace.handles[old_destination] = static_cast<u64>(chunk_index);
    }
    const u32 expanded_before =
      expanded_warp_base + __popc(expanded_mask & lower_lanes);
    if (chunk_index <= beam_count) {
      workspace.expanded[beam_capacity + chunk_index] =
        static_cast<u16>(expanded_before);
    }
    if (index == 0) {
      u32 retained_chunk = 0;
      u32 expanded_chunk = 0;
#pragma unroll
      for (u32 source_warp = 0; source_warp < kWarpCount; ++source_warp) {
        retained_chunk += workspace.expanded[kWarpCounterBase + source_warp];
        expanded_chunk +=
          workspace.expanded[kWarpCounterBase + kWarpCount + source_warp];
      }
      workspace.expanded[kWarpCounterBase - 2u] =
        static_cast<u16>(old_base + retained_chunk);
      workspace.expanded[kWarpCounterBase - 1u] =
        static_cast<u16>(expanded_base + expanded_chunk);
    }
    __syncthreads();
    old_base = workspace.expanded[kWarpCounterBase - 2u];
    expanded_base = workspace.expanded[kWarpCounterBase - 1u];
  }
  if (index == 0) {
    output_count = old_base;
    envelope_size_out = 0;
  }
  __syncthreads();
  const u32 old_unexpanded_count = output_count;
  const u32 anchor_count = min(old_unexpanded_count, issue_capacity);

  // One u16 counter per (warp, anchor), plus one valid-item counter per warp.
  // The temporary aliases the handle workspace only during the count pass;
  // the bounded preview has not started and therefore owns none of it yet.
  auto* anchor_histogram =
    reinterpret_cast<u16*>(workspace.handles + kPersistentFrontierRobCapacity);
  const u32 valid_count_base = kWarpCount * anchor_count;
  const u32 histogram_entries = valid_count_base + kWarpCount;
  for (u32 entry = index; entry < histogram_entries; entry += blockDim.x) {
    anchor_histogram[entry] = 0;
  }
  __syncthreads();

  const u32 leaf_begin = warp * kCandidatesPerWarp;
#pragma unroll
  for (u32 item = 0; item < kItemsPerLane; ++item) {
    const u32 ordinal = leaf_begin + lane + item * kWarpWidth;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (ordinal < candidate_count) {
      handle = candidate_handles[ordinal];
      distance = candidate_distances[ordinal];
    }
    const bool valid = stable_run_item_valid(handle, distance);
    const u32 valid_mask = __ballot_sync(kFullWarp, valid);
    if (lane == 0) {
      anchor_histogram[valid_count_base + warp] +=
        static_cast<u16>(__popc(valid_mask));
    }
    // Anchors are a stable subsequence of the distance-sorted old Beam.
    // Bucket each candidate at the first strict upper anchor in O(log I)
    // comparisons, then aggregate equal buckets within the warp. Prefixing
    // these bucket counts reconstructs every q_m without the former
    // O(N*I) ballot loop.
    u32 bucket = anchor_count;
    if (valid) {
      u32 low = 0;
      u32 high = anchor_count;
      while (low < high) {
        const u32 middle = low + ((high - low) >> 1);
        if (distance < workspace.distances[middle]) {
          high = middle;
        } else {
          low = middle + 1u;
        }
      }
      bucket = low;
    }
    const u32 bucket_peers = __match_any_sync(kFullWarp, bucket);
    const u32 bucket_leader = static_cast<u32>(__ffs(bucket_peers) - 1);
    if (bucket < anchor_count && lane == bucket_leader) {
      // Each warp owns a disjoint histogram segment and match_any elects one
      // writer per bucket for this item, so no atomic is needed.
      anchor_histogram[warp * anchor_count + bucket] +=
        static_cast<u16>(__popc(bucket_peers));
    }
    // match_any synchronizes participation but is not a memory barrier.
    // Consecutive items may elect different lanes for the same bucket, so
    // order their shared-memory RMWs explicitly within the owning warp.
    __syncwarp(kFullWarp);
  }
  __syncthreads();

  if (index == 0) {
    u32 total_valid_candidates = 0;
    for (u32 source_warp = 0; source_warp < kWarpCount; ++source_warp) {
      total_valid_candidates +=
        anchor_histogram[valid_count_base + source_warp];
    }

    u32 selected_anchor_count = 0;
    u32 selected_candidate_count = 0;
    bool selected = false;
    bool virtual_anchor = false;
    u32 dominated_candidates = 0;
    for (u32 anchor = 0; anchor < anchor_count; ++anchor) {
      for (u32 source_warp = 0; source_warp < kWarpCount; ++source_warp) {
        dominated_candidates +=
          anchor_histogram[source_warp * anchor_count + anchor];
      }
      const u32 envelope_size = anchor + 1u + dominated_candidates;
      if (envelope_size > kPersistentFrontierRobCapacity) {
        // Envelope size is monotone in the anchor.
        break;
      }
      if (envelope_size >= commit_capacity) {
        selected_anchor_count = anchor + 1u;
        selected_candidate_count = dominated_candidates;
        envelope_size_out = envelope_size;
        selected = true;
      }
    }

    if (!selected && old_unexpanded_count < commit_capacity) {
      // There are too few finite old anchors.  The complete remaining stream
      // is still certifiable when old plus every valid candidate fits.
      const u32 complete_size = old_unexpanded_count + total_valid_candidates;
      if (complete_size >= commit_capacity &&
          complete_size <= kPersistentFrontierRobCapacity) {
        selected_anchor_count = old_unexpanded_count;
        selected_candidate_count = total_valid_candidates;
        envelope_size_out = complete_size;
        selected = true;
        virtual_anchor = true;
      }
    }
    if (!selected) {
      envelope_size_out = 0;
      output_count = 0;
    } else {
      anchor_histogram[0] = static_cast<u16>(selected_anchor_count);
      anchor_histogram[1] = static_cast<u16>(selected_candidate_count);
      anchor_histogram[2] = static_cast<u16>(virtual_anchor ? 1u : 0u);
      *reservation_count = 0;
    }
  }
  __syncthreads();
  if (envelope_size_out == 0) return false;

  const u32 selected_anchor_count = anchor_histogram[0];
  const u32 expected_candidate_count = anchor_histogram[1];
  const bool virtual_infinity_anchor = anchor_histogram[2] != 0;
  const f32 selected_threshold =
    selected_anchor_count == 0
      ? FLT_MAX
      : workspace.distances[selected_anchor_count - 1u];

  // Keep the selected old prefix in [0, m) and compact the strict-dominance
  // candidate set immediately after it. Every slot in the certified envelope
  // is overwritten below, and the sorter never reads its disposable tail, so
  // clearing [envelope_size, K) would add two shared stores and a CTA barrier
  // without contributing to correctness.
#pragma unroll
  for (u32 item = 0; item < kItemsPerLane; ++item) {
    const u32 ordinal = leaf_begin + lane + item * kWarpWidth;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (ordinal < candidate_count) {
      handle = candidate_handles[ordinal];
      distance = candidate_distances[ordinal];
    }
    const bool valid = stable_run_item_valid(handle, distance);
    const bool dominated =
      valid && (virtual_infinity_anchor || distance < selected_threshold);
    const u32 dominated_mask = __ballot_sync(kFullWarp, dominated);
    const u32 dominated_count = __popc(dominated_mask);
    u32 reservation = 0;
    if (lane == 0 && dominated_count != 0) {
      reservation = atomicAdd(reservation_count, dominated_count);
    }
    reservation = __shfl_sync(kFullWarp, reservation, 0);
    if (dominated) {
      const u32 local_rank = __popc(dominated_mask & lower_lanes);
      const u32 destination = selected_anchor_count + reservation + local_rank;
      if (destination < kPersistentFrontierRobCapacity) {
        prefix_handles[destination] = handle;
        prefix_distances[destination] = distance;
        // The high bit places every old Beam item before every candidate
        // on a distance tie; low bits preserve raw Stable-Run input order.
        workspace.handles[destination] =
          static_cast<u64>(0x80000000u | ordinal);
      }
    }
  }
  __syncthreads();
  if (*reservation_count != expected_candidate_count) {
    if (index == 0) {
      output_count = 0;
      envelope_size_out = 0;
    }
    __syncthreads();
    return false;
  }

  // Sort the complete at-most-32 dominance envelope. Old tokens precede all
  // candidate tokens, so the bitonic key exactly matches
  //     old Beam < raw candidate ordinal
  // ties in the authoritative Stable-Run merge. Carry only a source slot
  // through the network; all lanes load payload before any output is written.
  if (warp == 0) {
    f32 key_distance =
      lane < envelope_size_out ? prefix_distances[lane] : FLT_MAX;
    u32 key_token = lane < envelope_size_out
                      ? static_cast<u32>(workspace.handles[lane])
                      : UINT32_MAX;
    u32 source_slot = lane < envelope_size_out ? lane : UINT32_MAX;
    for (u32 sequence = 2; sequence <= kWarpWidth; sequence <<= 1) {
      for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
        const f32 partner_distance =
          __shfl_xor_sync(kFullWarp, key_distance, stride);
        const u32 partner_token = __shfl_xor_sync(kFullWarp, key_token, stride);
        const u32 partner_slot =
          __shfl_xor_sync(kFullWarp, source_slot, stride);
        const bool partner_precedes =
          partner_distance < key_distance ||
          (partner_distance == key_distance && partner_token < key_token);
        const bool self_precedes =
          key_distance < partner_distance ||
          (key_distance == partner_distance && key_token < partner_token);
        const bool ascending = (lane & sequence) == 0;
        const bool lower = (lane & stride) == 0;
        const bool keep_minimum = lower == ascending;
        const bool exchange = keep_minimum ? partner_precedes : self_precedes;
        if (exchange) {
          key_distance = partner_distance;
          key_token = partner_token;
          source_slot = partner_slot;
        }
      }
    }
    u64 sorted_handle = kInvalidDeviceHandle;
    f32 sorted_distance = FLT_MAX;
    if (lane < envelope_size_out && source_slot != UINT32_MAX) {
      sorted_handle = prefix_handles[source_slot];
      sorted_distance = prefix_distances[source_slot];
    }

    const u32 output_limit = min(issue_capacity, envelope_size_out);
    u32 expanded_prefix_end = 0;
    if (lane < output_limit &&
        stable_run_item_valid(sorted_handle, sorted_distance)) {
      if ((key_token & 0x80000000u) == 0) {
        expanded_prefix_end = key_token;
      } else {
        // Every old item at an equal distance precedes a candidate. Locate
        // the old upper bound, then add exactly the expanded items hidden
        // from the compact-unexpanded envelope.
        u32 low = 0;
        u32 high = beam_count;
        while (low < high) {
          const u32 middle = low + ((high - low) >> 1);
          if (!(sorted_distance < beam_distances[middle])) {
            low = middle + 1u;
          } else {
            high = middle;
          }
        }
        expanded_prefix_end = low;
      }
    }
    const u32 authoritative_rank =
      lane +
      static_cast<u32>(workspace.expanded[beam_capacity + expanded_prefix_end]);
    const bool output_valid =
      lane < output_limit &&
      stable_run_item_valid(sorted_handle, sorted_distance) &&
      authoritative_rank < beam_capacity;
    if (lane < issue_capacity) {
      output_handles[lane] =
        output_valid ? sorted_handle : kInvalidDeviceHandle;
      output_ranks[lane] =
        output_valid ? static_cast<u16>(authoritative_rank) : UINT16_MAX;
    }
    const u32 valid_output_mask = __ballot_sync(kFullWarp, output_valid);
    if (lane == 0) output_count = __popc(valid_output_mask);
  }
  __syncthreads();
  return true;
}

// Partition-Bounded Exact Certificate (PBEC).
//
// Compact Stable-Run sorts the raw candidate stream as four consecutive
// 512-item stable leaves, retaining K items from each leaf before the fixed
// merge tree.  For an Issue Frontier of I <= 32, only the first I valid items
// of every leaf can contribute: leaf item I+1 already has I unexpanded items
// from that same leaf ahead of it.  Four warps therefore extract those exact
// stable prefixes directly from the scored input, without running CUB sort.
// The proof is independent of the distance distribution and dataset.
// The existing bounded tree then merges them with the old Beam and maps every
// unexpanded result back to its authoritative rank < K.
//
// This is a read-only certificate.  The regular Stable-Run sort and complete
// authoritative merge still execute unchanged after RDMA publication.
// Production query traversal guarantees that visited insertion made every
// valid candidate distinct from the old Beam, so candidates carry expanded=0;
// callers which permit duplicate-old handles must use the general prepared-run
// preview instead.
__device__ __noinline__ void prepare_partition_bounded_exact_certificate(
  const u64* candidate_handles, const f32* candidate_distances,
  u32 candidate_count, const u64* beam_handles, const f32* beam_distances,
  const u8* beam_expanded, u32 beam_count, u32 beam_capacity,
  u64* prefix_handles, f32* prefix_distances,
  CandidateWorkspaceArrays& workspace, u32 issue_capacity, u64* output_handles,
  u16* output_ranks, u32& output_count) {
  constexpr u32 kWarpWidth = 32;
  constexpr u32 kLeafCount = 4;
  constexpr u32 kLeafCapacity =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactFinal256;
  constexpr u32 kItemsPerLane = kLeafCapacity / kWarpWidth;
  static_assert(kLeafCount * kLeafCapacity == kPersistentMaxMergeCandidates);
  static_assert(kItemsPerLane * kWarpWidth == kLeafCapacity);

  candidate_count =
    min(candidate_count, static_cast<u32>(kPersistentMaxMergeCandidates));
  issue_capacity = min(min(issue_capacity, beam_capacity),
                       static_cast<u32>(kPersistentFrontierRobCapacity));
  if (blockDim.x != kApproximateSortThreadsCompact) {
    if (threadIdx.x == 0) output_count = 0;
    __syncthreads();
    return;
  }

  constexpr u32 full_warp = 0xffffffffu;
  const u32 warp = threadIdx.x / kWarpWidth;
  const u32 lane = threadIdx.x % kWarpWidth;
  const u32 leaf_begin = warp * kLeafCapacity;
  f32 lane_distances[kItemsPerLane];
  u32 valid_items = 0;
#pragma unroll
  for (u32 item = 0; item < kItemsPerLane; ++item) {
    const u32 ordinal = leaf_begin + lane + item * kWarpWidth;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (ordinal < candidate_count) {
      handle = candidate_handles[ordinal];
      distance = candidate_distances[ordinal];
    }
    const bool valid = stable_run_item_valid(handle, distance);
    lane_distances[item] = valid ? distance : FLT_MAX;
    if (valid) valid_items |= u32{1} << item;
  }

  // Every co-rank queried by the bounded preview has diagonal < I, so it can
  // inspect only leaf positions [0, I).  Clearing a complete K-entry leaf
  // would add 4*K stores to every round without changing the certificate.
  for (u32 position = lane; position < issue_capacity; position += kWarpWidth) {
    const u32 destination = warp * beam_capacity + position;
    prefix_handles[destination] = kInvalidDeviceHandle;
    prefix_distances[destination] = FLT_MAX;
  }
  __syncwarp(full_warp);

  bool have_previous = false;
  f32 previous_distance = 0.0f;
  u32 previous_ordinal = 0;
  for (u32 selected = 0; selected < issue_capacity; ++selected) {
    f32 local_distance = FLT_MAX;
    u32 local_ordinal = UINT32_MAX;
#pragma unroll
    for (u32 item = 0; item < kItemsPerLane; ++item) {
      if ((valid_items & (u32{1} << item)) == 0) continue;
      const u32 ordinal = leaf_begin + lane + item * kWarpWidth;
      const f32 distance = lane_distances[item];
      const bool after_previous =
        !have_previous || previous_distance < distance ||
        (distance == previous_distance && ordinal > previous_ordinal);
      if (!after_previous) continue;
      if (local_ordinal == UINT32_MAX || distance < local_distance ||
          (distance == local_distance && ordinal < local_ordinal)) {
        local_distance = distance;
        local_ordinal = ordinal;
      }
    }

    const u32 distance_bits = local_ordinal == UINT32_MAX ? UINT32_MAX
                              : local_distance == 0.0f
                                ? 0u
                                : __float_as_uint(local_distance);
    const u32 ordered_distance = local_ordinal == UINT32_MAX ? UINT32_MAX
                                 : (distance_bits & 0x80000000u) != 0
                                   ? ~distance_bits
                                   : (distance_bits ^ 0x80000000u);
    const u32 minimum_distance = warp_reduce_min_sync(full_warp, ordered_distance);
    const u32 minimum_ordinal = warp_reduce_min_sync(
      full_warp,
      ordered_distance == minimum_distance ? local_ordinal : UINT32_MAX);
    if (minimum_ordinal == UINT32_MAX) break;
    const u32 winner_mask =
      __ballot_sync(full_warp, local_ordinal == minimum_ordinal);
    const u32 winner_lane = static_cast<u32>(__ffs(winner_mask) - 1);
    const f32 selected_distance =
      __shfl_sync(full_warp, local_distance, winner_lane);
    u64 selected_handle = kInvalidDeviceHandle;
    if (lane == winner_lane) {
      selected_handle = candidate_handles[minimum_ordinal];
    }
    selected_handle = __shfl_sync(full_warp, selected_handle, winner_lane);
    if (lane == 0) {
      const u32 destination = warp * beam_capacity + selected;
      prefix_handles[destination] = selected_handle;
      prefix_distances[destination] = selected_distance;
    }
    have_previous = true;
    previous_distance = selected_distance;
    previous_ordinal = minimum_ordinal;
  }
  __syncthreads();

  preview_tree_stable_unexpanded_frontier(
    beam_handles, beam_distances, beam_expanded, beam_count, beam_capacity,
    prefix_handles, prefix_distances, kLeafCount, issue_capacity, workspace,
    output_handles, output_ranks, output_count);
}

// Keep the production pre-sort certificate behind one narrow call boundary.
// Without this wrapper, ptxas keeps both DEEC and PBEC argument sets live in
// process_query across the fallback branch, which can reduce persistent-CTA
// residency even though the two algorithms never execute concurrently.
__device__ __noinline__ void prepare_presort_exact_frontier_certificate(
  const DominanceEnvelopeCertificateContext& context) {
  if (prepare_dominance_envelope_exact_certificate(context)) return;
  prepare_partition_bounded_exact_certificate(
    context.candidate_handles, context.candidate_distances,
    context.candidate_count, context.beam_handles, context.beam_distances,
    context.beam_expanded, context.beam_count, context.beam_capacity,
    context.prefix_handles, context.prefix_distances, *context.workspace,
    context.issue_capacity, context.output_handles, context.output_ranks,
    *context.output_count);
}

__device__ __noinline__ void prepare_reusable_fused_frontier_certificate(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_capacity, u32 origin_count, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, CandidateWorkspaceArrays& output, u32 issue_capacity,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state, BeamMergeCycleBreakdown* cycle_breakdown,
  bool candidate_flags_known_clear);

// Prepare-Fused Exact Certificate (PFEC). Four warps sort the immutable
// candidate leaves and then construct the exact unexpanded Issue frontier.
__device__ __noinline__ void prepare_warp_leaf_fused_frontier_certificate(
  const u64* candidate_handles, const f32* candidate_distances,
  u32 candidate_count, const u64* beam_handles, const f32* beam_distances,
  const u8* beam_expanded, u32 beam_count, u32 beam_capacity,
  u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
  CandidateWorkspaceArrays& workspace, u32 issue_capacity, u64* output_handles,
  u16* output_ranks, u32& output_count, StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  constexpr u64 invalid_key = UINT64_MAX;
  if (threadIdx.x == 0) {
    state = {};
    state.original_count = beam_count;
    state.compact = 1;
    state.origin_copied = 0;
    state.phase_started = clock64();
    output_count = 0;
    if (cycle_breakdown != nullptr) *cycle_breakdown = {};
  }
  __syncthreads();
  if (blockDim.x != kApproximateSortThreadsCompact ||
      beam_capacity > kPersistentMaxBeam) {
    return;
  }
  candidate_count =
    min(candidate_count, static_cast<u32>(kPersistentMaxMergeCandidates));
  issue_capacity = min(min(issue_capacity, beam_capacity),
                       static_cast<u32>(kPersistentFrontierRobCapacity));

  const u32 warp = threadIdx.x / kWarpLeafSortThreads;
  const u32 lane = threadIdx.x % kWarpLeafSortThreads;
  const u32 leaf_begin = warp * kWarpLeafSortCapacity;
  u64 keys[kWarpLeafSortItemsPerThread];
#pragma unroll
  for (u32 item = 0; item < kWarpLeafSortItemsPerThread; ++item) {
    const u32 ordinal = leaf_begin + lane * kWarpLeafSortItemsPerThread + item;
    u64 key = invalid_key;
    if (ordinal < candidate_count) {
      const u64 handle = candidate_handles[ordinal];
      const f32 distance = candidate_distances[ordinal];
      if (stable_run_item_valid(handle, distance)) {
        const u32 bits = distance == 0.0f ? 0u : __float_as_uint(distance);
        const u32 ordered_distance =
          (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
        key = (static_cast<u64>(ordered_distance) << 32u) | ordinal;
      }
    }
    keys[item] = key;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->prepare += now - state.phase_started;
    state.phase_started = now;
  }
  __syncthreads();

  if (leaf_begin < candidate_count) {
    warp_bitonic_sort_blocked<kWarpLeafSortItemsPerThread>(keys);
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->sort += now - state.phase_started;
    state.phase_started = now;
  }
  __syncthreads();

#pragma unroll
  for (u32 item = 0; item < kWarpLeafSortItemsPerThread; ++item) {
    const u32 position = lane * kWarpLeafSortItemsPerThread + item;
    if (position >= beam_capacity) continue;
    const u32 destination = warp * beam_capacity + position;
    const u64 key = keys[item];
    if (key == invalid_key) {
      scratch_handles[destination] = kInvalidDeviceHandle;
      scratch_distances[destination] = FLT_MAX;
    } else {
      const u32 ordinal = static_cast<u32>(key);
      scratch_handles[destination] = candidate_handles[ordinal];
      scratch_distances[destination] = candidate_distances[ordinal];
    }
    scratch_flags[destination] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state.candidate_run_count =
      min(kWarpLeafSortWarps, (candidate_count + kWarpLeafSortCapacity - 1u) /
                                kWarpLeafSortCapacity);
    state.prepared = 1;
    state.fused_tree_prefix = 0;
    state.materialized_prefix = 0;
    state.deferred_prefix = 1;
    state.fused_tree_prepared = 1;
    if (cycle_breakdown != nullptr) {
      const u64 now = clock64();
      cycle_breakdown->materialize += now - state.phase_started;
      state.phase_started = now;
    }
  }
  __syncthreads();
  preview_tree_stable_unexpanded_frontier(
    beam_handles, beam_distances, beam_expanded, beam_count, beam_capacity,
    scratch_handles, scratch_distances, state.candidate_run_count,
    issue_capacity, workspace, output_handles, output_ranks, output_count);
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->materialize += now - state.phase_started;
    state.phase_started = now;
  }
  __syncthreads();
}

// Produce an exact Issue-Frontier certificate without publishing a partial
// Beam. Candidate runs contain no expanded entries, so the bounded preview
// compacts that metadata out of the old Beam and evaluates only W ranks of a
// balanced stable tree. This replaces the former lane-zero rank-by-rank
// four-head chain, whose cost grew with every expanded old-Beam entry late in
// a query and erased the communication overlap it enabled. The preview maps
// each compact rank back to its exact authoritative rank and preserves
//     old < run0 < run1 < run2 < run3
// ties. Its workspace is private and disposable: the complete authoritative
// tree is rebuilt from the still-immutable old Beam only after critical RDMA
// has been issued.
__device__ __noinline__ void prepare_deferred_fused_frontier_certificate(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_capacity, u32 origin_count, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, CandidateWorkspaceArrays& output, u32 issue_capacity,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  static_assert(kPersistentMaxExact * 2 >= kPersistentMaxBeam * 4);
  candidate_run_count = min(candidate_run_count, 4u);
  issue_capacity = min(issue_capacity, beam_capacity);
  (void)scratch_flags;
  preview_tree_stable_unexpanded_frontier(
    beam_handles, beam_distances, beam_expanded, origin_count, beam_capacity,
    scratch_handles, scratch_distances, candidate_run_count, issue_capacity,
    output, issue_handles, issue_ranks, issue_count);
  if (threadIdx.x == 0) {
    // The preview deliberately leaves no reusable authoritative tree prefix:
    // its old-Beam input was compacted. Finish therefore evaluates [0, K)
    // from the unchanged full old Beam while the issued RDMA wave is live.
    state.fused_tree_prefix = 0;
    state.materialized_prefix = 0;
    state.deferred_prefix = 1;
    state.fused_tree_prepared = 1;
    if (cycle_breakdown != nullptr) {
      const u64 now = clock64();
      cycle_breakdown->materialize += now - state.phase_started;
      state.phase_started = now;
    }
  }
  __syncthreads();
}

// Build an exact read-only Issue-Frontier certificate from already sorted
// Stable-Run leaves and retain every internal result needed by that
// certificate.  Unlike preview_tree_stable_unexpanded_frontier(), this helper
// does not compact the old Beam.  It evaluates a dependency-closed prefix of
// the authoritative merge tree, caches the final ranks in output[3*K, 4*K),
// and lets finish_fused_stable_frontier_materialization() extend only the
// missing suffix after RDMA publication.
//
// Candidate flags are included in the prefix bound so this remains exact for
// compatibility callers which propagate an old duplicate's expanded bit.
// Production candidates are disjoint and unexpanded, reducing the bound to
// issue_capacity plus the number of expanded old-Beam entries.  No Beam,
// visited, or expanded state is written before finish.
__device__ __noinline__ void prepare_reusable_fused_frontier_certificate(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_capacity, u32 origin_count, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, CandidateWorkspaceArrays& output, u32 issue_capacity,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool candidate_flags_known_clear = false) {
  static_assert(kPersistentMaxExact * 2 >= kPersistentMaxBeam * 4);
  constexpr u32 kWarpWidth = 32;
  constexpr u32 kFullWarp = 0xffffffffu;
  candidate_run_count = min(candidate_run_count, 4u);
  issue_capacity = min(min(issue_capacity, beam_capacity),
                       static_cast<u32>(kPersistentFrontierRobCapacity));
  origin_count = min(origin_count, beam_capacity);

  // Count an upper bound on expanded entries in the final Beam. Production
  // candidates have already passed visited insertion and Stable-Run writes
  // their flags as zero, so its hot path reads only the old Beam. Generic
  // compatibility callers may request the candidate-flag scan as well.
  // Per-warp counts temporarily reuse issue_ranks and avoid shared atomics.
  if (threadIdx.x == 0) state.materialized_prefix = 0;
  __syncthreads();
  u32 local_expanded = 0;
  // Strided scans: origin_count and beam_capacity may both reach 256 while
  // the CTA has only 128 threads.
  for (u32 scan = threadIdx.x; scan < origin_count; scan += blockDim.x) {
    if (stable_run_item_valid(beam_handles[scan], beam_distances[scan]) &&
        beam_expanded[scan] != 0) {
      ++local_expanded;
    }
  }
  if (!candidate_flags_known_clear) {
    for (u32 scan = threadIdx.x; scan < beam_capacity; scan += blockDim.x) {
#pragma unroll
      for (u32 run = 0; run < 4u; ++run) {
        if (run >= candidate_run_count) break;
        const u32 source = run * beam_capacity + scan;
        if (scratch_flags[source] != 0 &&
            stable_run_item_valid(scratch_handles[source],
                                  scratch_distances[source])) {
          ++local_expanded;
        }
      }
    }
  }
  if (blockDim.x % kWarpWidth == 0) {
    const u32 lane = threadIdx.x & (kWarpWidth - 1u);
    const u32 warp = threadIdx.x / kWarpWidth;
    const u32 warp_count = blockDim.x / kWarpWidth;
    const u32 warp_expanded = warp_reduce_add_sync(kFullWarp, local_expanded);
    if (lane == 0) {
      issue_ranks[warp] = static_cast<u16>(warp_expanded);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      u32 total = 0;
      for (u32 source_warp = 0; source_warp < warp_count; ++source_warp) {
        total += static_cast<u32>(issue_ranks[source_warp]);
      }
      state.materialized_prefix = total;
    }
  } else if (local_expanded != 0) {
    atomicAdd(&state.materialized_prefix, local_expanded);
  }
  __syncthreads();

  const u32 expanded_upper_bound =
    min(state.materialized_prefix, beam_capacity);
  const u32 reusable_prefix =
    min(beam_capacity, issue_capacity + expanded_upper_bound);
  extend_fused_stable_candidate_tree(
    beam_distances, beam_capacity, origin_count, beam_handles, beam_expanded,
    scratch_handles, scratch_flags, scratch_distances, candidate_run_count,
    output, 0, reusable_prefix);

  // Cache the exact final fold without publishing it.  The fourth K-entry
  // workspace segment is reserved for this lifetime and is already consumed
  // by finish_fused_stable_frontier_materialization() when
  // materialized_prefix is nonzero.
  const u32 run3_count = candidate_run_count > 3u ? beam_capacity : 0u;
  const bool run3_empty =
    run3_count == 0 ||
    !stable_run_item_valid(scratch_handles[3u * beam_capacity],
                           scratch_distances[3u * beam_capacity]);
  for (u32 rank = threadIdx.x; rank < reusable_prefix; rank += blockDim.x) {
    const u32 destination = 3u * beam_capacity + rank;
    if (run3_empty) {
      const u32 source = 2u * beam_capacity + rank;
      output.handles[destination] = output.handles[source];
      output.distances[destination] = output.distances[source];
      output.expanded[destination] = output.expanded[source];
      continue;
    }
    const u32 prefix_index = stable_merge_a_corank(
      rank, output.distances + 2u * beam_capacity, reusable_prefix,
      scratch_distances + 3u * beam_capacity, run3_count);
    const u32 run3_index = rank - prefix_index;
    const bool take_prefix =
      prefix_index < reusable_prefix &&
      (run3_index >= run3_count ||
       !(scratch_distances[3u * beam_capacity + run3_index] <
         output.distances[2u * beam_capacity + prefix_index]));
    if (take_prefix) {
      const u32 source = 2u * beam_capacity + prefix_index;
      output.handles[destination] = output.handles[source];
      output.distances[destination] = output.distances[source];
      output.expanded[destination] = output.expanded[source];
    } else {
      const u32 source = 3u * beam_capacity + run3_index;
      output.handles[destination] = scratch_handles[source];
      output.distances[destination] = scratch_distances[source];
      output.expanded[destination] = scratch_flags[source];
    }
  }
  __syncthreads();

  // Compact unexpanded cached ranks into the Issue Frontier.  The first four
  // rank slots are temporary per-warp counters; a CTA barrier snapshots every
  // prefix sum before those same slots become ordinary certificate outputs.
  if (blockDim.x == kApproximateSortThreadsCompact) {
    static_assert(kPersistentMaxBeam <=
                  2u * kApproximateSortThreadsCompact);
    constexpr u32 warp_count = kApproximateSortThreadsCompact / kWarpWidth;
    const u32 lane = threadIdx.x & (kWarpWidth - 1u);
    const u32 warp = threadIdx.x / kWarpWidth;
    // Cover up to 256 cached ranks with two ranks per thread.  The per-warp
    // ballot counts double as temporary counters and are dead once every
    // thread snapshots its prefix bases across the barrier.
    const u32 rank0 = threadIdx.x;
    const u32 rank1 = threadIdx.x + kApproximateSortThreadsCompact;
    const u32 source0 = 3u * beam_capacity + rank0;
    const u32 source1 = 3u * beam_capacity + rank1;
    const bool issue0 =
      rank0 < reusable_prefix && output.expanded[source0] == 0 &&
      stable_run_item_valid(output.handles[source0],
                            output.distances[source0]);
    const bool issue1 =
      rank1 < reusable_prefix && output.expanded[source1] == 0 &&
      stable_run_item_valid(output.handles[source1],
                            output.distances[source1]);
    const u32 issue_mask0 = __ballot_sync(kFullWarp, issue0);
    const u32 issue_mask1 = __ballot_sync(kFullWarp, issue1);
    if (lane == 0) {
      issue_ranks[warp] = static_cast<u16>(__popc(issue_mask0));
      issue_ranks[warp_count + warp] = static_cast<u16>(__popc(issue_mask1));
    }
    __syncthreads();
    u32 base0 = 0;
    u32 total0 = 0;
#pragma unroll
    for (u32 prior = 0; prior < warp_count; ++prior) {
      const u32 count = static_cast<u32>(issue_ranks[prior]);
      total0 += count;
      if (prior < warp) base0 += count;
    }
    u32 base1 = total0;
#pragma unroll
    for (u32 prior = 0; prior < warp_count; ++prior) {
      if (prior >= warp) break;
      base1 += static_cast<u32>(issue_ranks[warp_count + prior]);
    }
    if (threadIdx.x == 0) {
      u32 total1 = 0;
#pragma unroll
      for (u32 source_warp = 0; source_warp < warp_count; ++source_warp) {
        total1 += static_cast<u32>(issue_ranks[warp_count + source_warp]);
      }
      issue_count = min(total0 + total1, issue_capacity);
    }
    __syncthreads();
    const u32 lower_lanes = lane == 0 ? 0u : (u32{1} << lane) - 1u;
    if (issue0) {
      const u32 destination = base0 + __popc(issue_mask0 & lower_lanes);
      if (destination < issue_capacity) {
        issue_handles[destination] = output.handles[source0];
        issue_ranks[destination] = static_cast<u16>(rank0);
      }
    }
    if (issue1) {
      const u32 destination = base1 + __popc(issue_mask1 & lower_lanes);
      if (destination < issue_capacity) {
        issue_handles[destination] = output.handles[source1];
        issue_ranks[destination] = static_cast<u16>(rank1);
      }
    }
  } else if (threadIdx.x == 0) {
    issue_count = 0;
    for (u32 rank = 0; rank < reusable_prefix && issue_count < issue_capacity;
         ++rank) {
      const u32 source = 3u * beam_capacity + rank;
      if (output.expanded[source] != 0 ||
          !stable_run_item_valid(output.handles[source],
                                 output.distances[source])) {
        continue;
      }
      issue_handles[issue_count] = output.handles[source];
      issue_ranks[issue_count] = static_cast<u16>(rank);
      ++issue_count;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    state.fused_tree_prepared = 1;
    state.fused_tree_prefix = reusable_prefix;
    state.deferred_prefix = 1;
    state.materialized_prefix = reusable_prefix;
    const u64 now = clock64();
    if (cycle_breakdown != nullptr) {
      cycle_breakdown->materialize += now - state.phase_started;
    }
    state.phase_started = now;
  }
  __syncthreads();
}

// Compatibility wrapper used by the focused fused-prefix test and by callers
// that do not expose an explicit communication phase.  Production query
// traversal invokes the two narrow stages directly so their live ranges do
// not overlap.
__device__ __noinline__ void begin_fused_stable_frontier_materialization(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32 beam_capacity, u32 origin_count, const u64* origin_handles,
  const u8* origin_expanded, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, CandidateWorkspaceArrays& output, u32 issue_capacity,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  prepare_fused_stable_candidate_tree(
    beam_distances, beam_capacity, origin_count, origin_handles,
    origin_expanded, scratch_handles, scratch_flags, scratch_distances,
    candidate_run_count, output);
  materialize_fused_stable_frontier_prefix(
    beam_handles, beam_ids, beam_distances, beam_expanded, beam_capacity,
    scratch_handles, scratch_flags, scratch_distances, candidate_run_count,
    output, issue_capacity, issue_handles, issue_ranks, issue_count, state,
    cycle_breakdown);
}

__device__ __noinline__ void finish_fused_stable_frontier_materialization(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, CandidateWorkspaceArrays& output,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  candidate_run_count = min(candidate_run_count, 4u);
  const u32 run3_count = candidate_run_count > 3 ? beam_capacity : 0u;
  const bool run3_empty =
    run3_count == 0 ||
    !stable_run_item_valid(scratch_handles[3u * beam_capacity],
                           scratch_distances[3u * beam_capacity]);
  const bool deferred_prefix = state.deferred_prefix != 0;
  const bool bounded_stage1_certificate = state.deferred_prefix == 2;
  const u32 cached_prefix = min(state.materialized_prefix, beam_capacity);
  // Active-Arity Frontier Commit (AAFC): production rounds normally expose
  // only one or two 512-candidate leaves. Their deferred certificate leaves
  // Beam immutable, so materialize the exact two-level stable network and
  // publish its second level directly. The generic five-run tree otherwise
  // writes an identity Stage-2 result and copies it once more through the
  // absent run3 fold. This branch preserves the identical old < run0 < run1
  // tie order while removing that shared-memory pass and CTA barrier.
  if (deferred_prefix && !bounded_stage1_certificate && cached_prefix == 0 &&
      candidate_run_count <= 2u) {
    const u32 run0_count = candidate_run_count > 0 ? beam_capacity : 0u;
    const u32 run1_count = candidate_run_count > 1 ? beam_capacity : 0u;
    for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
      const u32 old_index =
        stable_merge_a_corank(rank, beam_distances, state.original_count,
                              scratch_distances, run0_count);
      const u32 run0_index = rank - old_index;
      const bool take_old =
        old_index < state.original_count &&
        (run0_index >= run0_count ||
         !(scratch_distances[run0_index] < beam_distances[old_index]));
      if (take_old) {
        output.handles[rank] = beam_handles[old_index];
        output.distances[rank] = beam_distances[old_index];
        output.expanded[rank] = beam_expanded[old_index];
      } else if (run0_index < run0_count) {
        output.handles[rank] = scratch_handles[run0_index];
        output.distances[rank] = scratch_distances[run0_index];
        output.expanded[rank] = scratch_flags[run0_index];
      } else {
        output.handles[rank] = kInvalidDeviceHandle;
        output.distances[rank] = FLT_MAX;
        output.expanded[rank] = 0;
      }
    }
    __syncthreads();

    for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
      u32 source = rank;
      bool take_left = true;
      if (run1_count != 0) {
        const u32 left_index =
          stable_merge_a_corank(rank, output.distances, beam_capacity,
                                scratch_distances + beam_capacity, run1_count);
        const u32 run1_index = rank - left_index;
        take_left = left_index < beam_capacity &&
                    (run1_index >= run1_count ||
                     !(scratch_distances[beam_capacity + run1_index] <
                       output.distances[left_index]));
        source = take_left ? left_index : beam_capacity + run1_index;
      }
      if (take_left) {
        beam_handles[rank] = output.handles[source];
        beam_distances[rank] = output.distances[source];
        beam_expanded[rank] = output.expanded[source];
      } else {
        beam_handles[rank] = scratch_handles[source];
        beam_distances[rank] = scratch_distances[source];
        beam_expanded[rank] = scratch_flags[source];
      }
      beam_ids[rank] = UINT32_MAX;
    }
    __syncthreads();
    finalize_fused_stable_candidate_runs(beam_handles, beam_distances,
                                         beam_count, beam_capacity);
    if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
      cycle_breakdown->materialize += clock64() - state.phase_started;
    }
    if (threadIdx.x == 0) {
      state.fused_tree_prepared = 0;
      state.fused_tree_prefix = 0;
      state.deferred_prefix = 0;
      state.materialized_prefix = 0;
    }
    __syncthreads();
    return;
  }
  if (bounded_stage1_certificate) {
    const u32 stage1_prefix = min(state.fused_tree_prefix, beam_capacity);
    const u32 run0_count = candidate_run_count > 0 ? beam_capacity : 0u;
    const u32 run1_count = candidate_run_count > 1 ? beam_capacity : 0u;
    const u32 run2_count = candidate_run_count > 2 ? beam_capacity : 0u;

    // Reuse the certified Stage-1A prefix. Complete its suffix and all of
    // Stage 1B while Beam is still the immutable old origin.
    for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
      if (rank >= stage1_prefix) {
        const u32 old_index =
          stable_merge_a_corank(rank, beam_distances, state.original_count,
                                scratch_distances, run0_count);
        const u32 run0_index = rank - old_index;
        const bool take_old =
          old_index < state.original_count &&
          (run0_index >= run0_count ||
           !(scratch_distances[run0_index] < beam_distances[old_index]));
        if (take_old) {
          output.handles[rank] = beam_handles[old_index];
          output.distances[rank] = beam_distances[old_index];
          output.expanded[rank] = beam_expanded[old_index];
        } else if (run0_index < run0_count) {
          output.handles[rank] = scratch_handles[run0_index];
          output.distances[rank] = scratch_distances[run0_index];
          output.expanded[rank] = scratch_flags[run0_index];
        } else {
          output.handles[rank] = kInvalidDeviceHandle;
          output.distances[rank] = FLT_MAX;
          output.expanded[rank] = 0;
        }
      }

      const u32 run1_index = stable_merge_a_corank(
        rank, scratch_distances + beam_capacity, run1_count,
        scratch_distances + 2u * beam_capacity, run2_count);
      const u32 run2_index = rank - run1_index;
      const bool take_run1 =
        run1_index < run1_count &&
        (run2_index >= run2_count ||
         !(scratch_distances[2u * beam_capacity + run2_index] <
           scratch_distances[beam_capacity + run1_index]));
      const u32 destination = beam_capacity + rank;
      if (take_run1) {
        const u32 source = beam_capacity + run1_index;
        output.handles[destination] = scratch_handles[source];
        output.distances[destination] = scratch_distances[source];
        output.expanded[destination] = scratch_flags[source];
      } else if (run2_index < run2_count) {
        const u32 source = 2u * beam_capacity + run2_index;
        output.handles[destination] = scratch_handles[source];
        output.distances[destination] = scratch_distances[source];
        output.expanded[destination] = scratch_flags[source];
      } else {
        output.handles[destination] = kInvalidDeviceHandle;
        output.distances[destination] = FLT_MAX;
        output.expanded[destination] = 0;
      }
    }
    __syncthreads();

    // Stage 2 now observes two complete immutable child runs.
    for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
      const u32 left_index =
        stable_merge_a_corank(rank, output.distances, beam_capacity,
                              output.distances + beam_capacity, beam_capacity);
      const u32 right_index = rank - left_index;
      const bool take_left = left_index < beam_capacity &&
                             (right_index >= beam_capacity ||
                              !(output.distances[beam_capacity + right_index] <
                                output.distances[left_index]));
      const u32 destination = 2u * beam_capacity + rank;
      const u32 source = take_left ? left_index : beam_capacity + right_index;
      output.handles[destination] = output.handles[source];
      output.distances[destination] = output.distances[source];
      output.expanded[destination] = output.expanded[source];
    }
    __syncthreads();
  } else if (deferred_prefix) {
    // The certificate never overwrote Beam, so the old origin remains a
    // valid immutable Stage-1A input. Complete every missing internal rank
    // before publishing any authoritative output.
    extend_fused_stable_candidate_tree(
      beam_distances, beam_capacity, state.original_count, beam_handles,
      beam_expanded, scratch_handles, scratch_flags, scratch_distances,
      candidate_run_count, output, min(state.fused_tree_prefix, beam_capacity),
      beam_capacity);
  }
  const u32 begin_rank = deferred_prefix ? 0u : cached_prefix;
  // This is a leaf operation on the production overlap path.  Keeping the
  // final fold here avoids preserving the whole suffix context across a
  // nested range call on every query round.
  for (u32 rank = begin_rank + threadIdx.x; rank < beam_capacity;
       rank += blockDim.x) {
    if (deferred_prefix && !bounded_stage1_certificate &&
        rank < cached_prefix) {
      const u32 source = 3u * beam_capacity + rank;
      beam_handles[rank] = output.handles[source];
      beam_distances[rank] = output.distances[source];
      beam_expanded[rank] = output.expanded[source];
      beam_ids[rank] = UINT32_MAX;
      continue;
    }
    if (run3_empty) {
      const u32 source = 2u * beam_capacity + rank;
      beam_handles[rank] = output.handles[source];
      beam_distances[rank] = output.distances[source];
      beam_expanded[rank] = output.expanded[source];
      beam_ids[rank] = UINT32_MAX;
      continue;
    }
    const u32 prefix_index = stable_merge_a_corank(
      rank, output.distances + 2u * beam_capacity, beam_capacity,
      scratch_distances + 3u * beam_capacity, run3_count);
    const u32 run3_index = rank - prefix_index;
    const bool take_prefix =
      prefix_index < beam_capacity &&
      (run3_index >= run3_count ||
       !(scratch_distances[3u * beam_capacity + run3_index] <
         output.distances[2u * beam_capacity + prefix_index]));
    if (take_prefix) {
      const u32 source = 2u * beam_capacity + prefix_index;
      beam_handles[rank] = output.handles[source];
      beam_distances[rank] = output.distances[source];
      beam_expanded[rank] = output.expanded[source];
    } else {
      const u32 source = 3u * beam_capacity + run3_index;
      beam_handles[rank] = scratch_handles[source];
      beam_distances[rank] = scratch_distances[source];
      beam_expanded[rank] = scratch_flags[source];
    }
    beam_ids[rank] = UINT32_MAX;
  }
  __syncthreads();
  finalize_fused_stable_candidate_runs(beam_handles, beam_distances, beam_count,
                                       beam_capacity);
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    // Account the suffix once. Query traversal also uses the unchanged
    // phase_started timestamp as the overlap credit for the tail controller.
    cycle_breakdown->materialize += clock64() - state.phase_started;
  }
  if (threadIdx.x == 0) {
    state.fused_tree_prepared = 0;
    state.fused_tree_prefix = 0;
    state.deferred_prefix = 0;
    state.materialized_prefix = 0;
  }
  __syncthreads();
}

// Read-only exact preview of the final Stable-Run Beam. The old Beam and up to
// four candidate inputs are already stable runs and scratch_flags has already
// inherited old-Beam expanded metadata. No authoritative array is modified.
// Five lanes cooperatively own the five run heads. This retains the bounded
// prefix work of the scalar k-way merge while parallelizing head loads and
// winner selection; no all-candidate rank pass or extra workspace is needed.
__device__ __noinline__ void preview_stable_unexpanded_frontier(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_count, u32 beam_capacity, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, u32 issue_capacity, u64* output_handles,
  u16* output_ranks, u32& output_count) {
  candidate_run_count = min(candidate_run_count, 4u);
  const u32 run_count = candidate_run_count + 1u;
  issue_capacity = min(issue_capacity, beam_capacity);
  if (blockDim.x != kApproximateSortThreadsCompact) {
    // On the wide 256-thread kernel the measured warp hand-off cost exceeds
    // five scalar head comparisons. Keep the same bounded exact algorithm
    // without regressing that architecture variant.
    if (threadIdx.x == 0) {
      u32 heads[5]{0, 0, 0, 0, 0};
      output_count = 0;
      for (u32 rank = 0; rank < beam_capacity && output_count < issue_capacity;
           ++rank) {
        u32 selected_run = UINT32_MAX;
        f32 selected_distance = FLT_MAX;
        for (u32 run = 0; run < run_count; ++run) {
          const u32 head = heads[run];
          const u32 count = run == 0 ? beam_count : beam_capacity;
          if (head >= count) continue;
          const u64 handle =
            run == 0 ? beam_handles[head]
                     : scratch_handles[(run - 1u) * beam_capacity + head];
          const f32 distance =
            run == 0 ? beam_distances[head]
                     : scratch_distances[(run - 1u) * beam_capacity + head];
          if (!stable_run_item_valid(handle, distance)) continue;
          if (selected_run == UINT32_MAX ||
              stable_run_head_precedes(distance, run, selected_distance,
                                       selected_run)) {
            selected_run = run;
            selected_distance = distance;
          }
        }
        if (selected_run == UINT32_MAX) break;
        const u32 head = heads[selected_run]++;
        const u64 handle =
          selected_run == 0
            ? beam_handles[head]
            : scratch_handles[(selected_run - 1u) * beam_capacity + head];
        const bool expanded =
          selected_run == 0
            ? beam_expanded[head] != 0
            : scratch_flags[(selected_run - 1u) * beam_capacity + head] != 0;
        if (expanded) continue;
        output_handles[output_count] = handle;
        output_ranks[output_count] = static_cast<u16>(rank);
        ++output_count;
      }
    }
  } else if (threadIdx.x < 32) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 lane = threadIdx.x;
    u32 head = 0;
    u32 emitted = 0;
    for (u32 rank = 0; rank < beam_capacity && emitted < issue_capacity;
         ++rank) {
      const bool run_active =
        lane < run_count && head < (lane == 0 ? beam_count : beam_capacity);
      const u32 offset = lane == 0 ? 0u : (lane - 1u) * beam_capacity;
      const u64 handle =
        run_active
          ? (lane == 0 ? beam_handles[head] : scratch_handles[offset + head])
          : kInvalidDeviceHandle;
      const f32 distance = run_active
                             ? (lane == 0 ? beam_distances[head]
                                          : scratch_distances[offset + head])
                             : FLT_MAX;
      const bool valid = run_active && stable_run_item_valid(handle, distance);
      const u32 active_mask = __ballot_sync(full_warp, valid);
      if (active_mask == 0) break;

      // Squared distances are non-negative, but use a full IEEE monotonic
      // transform and canonicalize both signed zeros so the reduction exactly
      // matches floating-point `<`/`==` ordering. SM80+ uses one native warp
      // reduction; the SM70 path preserves the same ordering with shuffles.
      // ballot/ffs preserves the lower-run tie rule.
      const u32 bits = distance == 0.0f ? 0u : __float_as_uint(distance);
      const u32 ordered =
        (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
      const u32 minimum =
        warp_reduce_min_sync(full_warp, valid ? ordered : UINT32_MAX);
      const u32 winner_mask =
        __ballot_sync(full_warp, valid && ordered == minimum);
      const u32 winner_lane = static_cast<u32>(__ffs(winner_mask) - 1);
      const u64 selected_handle = __shfl_sync(full_warp, handle, winner_lane);
      const u32 selected_head = __shfl_sync(full_warp, head, winner_lane);
      const bool selected_expanded =
        winner_lane == 0
          ? beam_expanded[selected_head] != 0
          : scratch_flags[(winner_lane - 1u) * beam_capacity + selected_head] !=
              0;
      const u32 expanded = __shfl_sync(
        full_warp, static_cast<u32>(selected_expanded), winner_lane);
      if (lane == winner_lane) ++head;
      if (lane == 0 && expanded == 0) {
        output_handles[emitted] = selected_handle;
        output_ranks[emitted] = static_cast<u16>(rank);
        ++emitted;
      }
      emitted = __shfl_sync(full_warp, emitted, 0);
    }
    if (lane == 0) output_count = emitted;
  }
  __syncthreads();
}

// The exact issue prefix is at most one ROB (32 entries) across five stable
// runs.  For this narrow control problem, one lane's register-resident heads
// are cheaper than a warp reduction and barrier for every emitted rank.  The
// authoritative merge remains fully parallel; this helper is read-only.
__device__ __noinline__ void preview_serial_stable_unexpanded_frontier(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_count, u32 beam_capacity, const u64* scratch_handles,
  const u8* scratch_flags, const f32* scratch_distances,
  u32 candidate_run_count, u32 issue_capacity, u64* output_handles,
  u16* output_ranks, u32& output_count) {
  if (threadIdx.x == 0) {
    u32 heads[5]{0, 0, 0, 0, 0};
    candidate_run_count = min(candidate_run_count, 4u);
    const u32 run_count = candidate_run_count + 1u;
    issue_capacity = min(issue_capacity, beam_capacity);
    output_count = 0;
    for (u32 rank = 0; rank < beam_capacity && output_count < issue_capacity;
         ++rank) {
      u32 selected_run = UINT32_MAX;
      f32 selected_distance = FLT_MAX;
      for (u32 run = 0; run < run_count; ++run) {
        const u32 head = heads[run];
        const u32 count = run == 0 ? beam_count : beam_capacity;
        if (head >= count) continue;
        const u64 handle =
          run == 0 ? beam_handles[head]
                   : scratch_handles[(run - 1u) * beam_capacity + head];
        const f32 distance =
          run == 0 ? beam_distances[head]
                   : scratch_distances[(run - 1u) * beam_capacity + head];
        if (!stable_run_item_valid(handle, distance)) continue;
        if (selected_run == UINT32_MAX ||
            stable_run_head_precedes(distance, run, selected_distance,
                                     selected_run)) {
          selected_run = run;
          selected_distance = distance;
        }
      }
      if (selected_run == UINT32_MAX) break;
      const u32 head = heads[selected_run]++;
      const u64 handle =
        selected_run == 0
          ? beam_handles[head]
          : scratch_handles[(selected_run - 1u) * beam_capacity + head];
      const bool expanded =
        selected_run == 0
          ? beam_expanded[head] != 0
          : scratch_flags[(selected_run - 1u) * beam_capacity + head] != 0;
      if (expanded) continue;
      output_handles[output_count] = handle;
      output_ranks[output_count] = static_cast<u16>(rank);
      ++output_count;
    }
  }
  __syncthreads();
}

// Exact bounded frontier extraction for the production Stable-Run pipeline.
// Candidate handles have already passed visited insertion, so only the old
// Beam can carry expanded entries.  Compact that one run, then use the same
// balanced stable merge tree as authoritative materialization to produce only
// the requested unexpanded prefix.  One CUDA lane owns one output rank; this
// removes the prior rank-by-rank warp reduction and its O(width)
// synchronization chain without changing old < run0 < run1 < run2 < run3 tie
// order.
//
// Expanded old-Beam entries are omitted from the tree, but each output is
// mapped back to its exact rank in the full merged stream.  An item at rank
// >= beam_capacity is not part of the next authoritative Beam and therefore
// cannot enter the issue frontier.
__device__ __noinline__ void preview_tree_stable_unexpanded_frontier(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_count, u32 beam_capacity, const u64* scratch_handles,
  const f32* scratch_distances, u32 candidate_run_count, u32 issue_capacity,
  CandidateWorkspaceArrays& workspace, u64* output_handles, u16* output_ranks,
  u32& output_count) {
  static_assert(kPersistentMaxBeam <= 256);
  constexpr u16 kCandidateSourceTag = UINT16_MAX;
  candidate_run_count = min(candidate_run_count, 4u);
  issue_capacity = min(issue_capacity, beam_capacity);
  if (blockDim.x == kApproximateSortThreadsCompact) {
    constexpr u32 kWarpWidth = 32;
    constexpr u32 kFullWarp = 0xffffffffu;
    constexpr u32 kWarpCount = kApproximateSortThreadsCompact / kWarpWidth;
    const u32 lane = threadIdx.x & (kWarpWidth - 1u);
    const u32 warp = threadIdx.x / kWarpWidth;
    const u32 lower_lanes = lane == 0 ? 0u : (u32{1} << lane) - 1u;
    // Compact the unexpanded old Beam and build its expanded-prefix table in
    // one pass. output_ranks is still dead and provides per-warp counters.
    u32 compact_base = 0;
    u32 expanded_carry = 0;
    for (u32 base = 0; base < beam_count;
         base += kApproximateSortThreadsCompact) {
      const u32 index = base + threadIdx.x;
      const bool in_range = index < beam_count;
      const bool retain =
        in_range && beam_expanded[index] == 0 &&
        stable_run_item_valid(beam_handles[index], beam_distances[index]);
      const u32 retained_mask = __ballot_sync(kFullWarp, retain);
      const u32 expanded_mask = __ballot_sync(
        kFullWarp, in_range && beam_expanded[index] != 0);
      if (lane == 0) {
        workspace.expanded[3u * beam_capacity + warp] =
          static_cast<u16>(__popc(retained_mask));
        output_ranks[warp] = static_cast<u16>(__popc(expanded_mask));
      }
      __syncthreads();

      u32 warp_base = compact_base;
      u32 expanded_base = expanded_carry;
      for (u32 prior = 0; prior < warp; ++prior) {
        warp_base += workspace.expanded[3u * beam_capacity + prior];
        expanded_base += output_ranks[prior];
      }
      const u32 destination =
        warp_base + __popc(retained_mask & lower_lanes);
      if (retain) {
        workspace.handles[destination] = beam_handles[index];
        workspace.distances[destination] = beam_distances[index];
        workspace.expanded[destination] = static_cast<u16>(index);
      }
      if (in_range) {
        workspace.old_expanded_prefix[index] = static_cast<u16>(
          expanded_base + __popc(expanded_mask & lower_lanes));
      }
      if (threadIdx.x == 0) {
        u32 retained = 0;
        u32 total_expanded = 0;
        for (u32 source_warp = 0; source_warp < kWarpCount; ++source_warp) {
          retained += workspace.expanded[3u * beam_capacity + source_warp];
          total_expanded += output_ranks[source_warp];
        }
        workspace.expanded[3u * beam_capacity + kWarpCount] =
          static_cast<u16>(compact_base + retained);
        output_ranks[kWarpCount] =
          static_cast<u16>(expanded_carry + total_expanded);
      }
      __syncthreads();
      compact_base = workspace.expanded[3u * beam_capacity + kWarpCount];
      expanded_carry = output_ranks[kWarpCount];
    }
    if (threadIdx.x == 0) {
      workspace.old_expanded_prefix[beam_count] =
        static_cast<u16>(expanded_carry);
      output_count = compact_base;
    }
  } else if (threadIdx.x == 0) {
    output_count = 0;
    u16 expanded_prefix = 0;
    for (u32 index = 0; index < beam_count; ++index) {
      workspace.old_expanded_prefix[index] = expanded_prefix;
      expanded_prefix += beam_expanded[index] != 0 ? 1u : 0u;
      if (beam_expanded[index] != 0 ||
          !stable_run_item_valid(beam_handles[index], beam_distances[index])) {
        continue;
      }
      const u32 destination = output_count++;
      workspace.handles[destination] = beam_handles[index];
      workspace.distances[destination] = beam_distances[index];
      workspace.expanded[destination] = static_cast<u16>(index);
    }
    workspace.old_expanded_prefix[beam_count] = expanded_prefix;
  }
  __syncthreads();
  const u32 unexpanded_old_count = output_count;
  const u32 run0_count = candidate_run_count > 0 ? beam_capacity : 0u;
  const u32 run1_count = candidate_run_count > 1 ? beam_capacity : 0u;
  const u32 run2_count = candidate_run_count > 2 ? beam_capacity : 0u;
  const u32 run3_count = candidate_run_count > 3 ? beam_capacity : 0u;

  // Stage 1A: compact unexpanded old Beam + run0.
  // Stage 1B: run1 + run2.
  for (u32 rank = threadIdx.x; rank < issue_capacity; rank += blockDim.x) {
    const u32 old_index =
      stable_merge_a_corank(rank, workspace.distances, unexpanded_old_count,
                            scratch_distances, run0_count);
    const u32 run0_index = rank - old_index;
    const bool take_old =
      old_index < unexpanded_old_count &&
      (run0_index >= run0_count ||
       !(scratch_distances[run0_index] < workspace.distances[old_index]));
    const u32 left_destination = beam_capacity + rank;
    if (take_old) {
      workspace.handles[left_destination] = workspace.handles[old_index];
      workspace.distances[left_destination] = workspace.distances[old_index];
      workspace.expanded[left_destination] = workspace.expanded[old_index];
    } else if (run0_index < run0_count) {
      workspace.handles[left_destination] = scratch_handles[run0_index];
      workspace.distances[left_destination] = scratch_distances[run0_index];
      workspace.expanded[left_destination] = kCandidateSourceTag;
    } else {
      workspace.handles[left_destination] = kInvalidDeviceHandle;
      workspace.distances[left_destination] = FLT_MAX;
      workspace.expanded[left_destination] = kCandidateSourceTag;
    }

    const u32 right_destination = 2u * beam_capacity + rank;
    if (run1_count == 0) {
      // Identity child: Stage 2 consumes only the left run, so no sentinel
      // materialization is required for an absent candidate leaf.
    } else if (run2_count == 0 && rank < run1_count) {
      workspace.handles[right_destination] =
        scratch_handles[beam_capacity + rank];
      workspace.distances[right_destination] =
        scratch_distances[beam_capacity + rank];
    } else {
      const u32 run1_index = stable_merge_a_corank(
        rank, scratch_distances + beam_capacity, run1_count,
        scratch_distances + 2u * beam_capacity, run2_count);
      const u32 run2_index = rank - run1_index;
      const bool take_run1 =
        run1_index < run1_count &&
        (run2_index >= run2_count ||
         !(scratch_distances[2u * beam_capacity + run2_index] <
           scratch_distances[beam_capacity + run1_index]));
      if (take_run1) {
        workspace.handles[right_destination] =
          scratch_handles[beam_capacity + run1_index];
        workspace.distances[right_destination] =
          scratch_distances[beam_capacity + run1_index];
      } else if (run2_index < run2_count) {
        workspace.handles[right_destination] =
          scratch_handles[2u * beam_capacity + run2_index];
        workspace.distances[right_destination] =
          scratch_distances[2u * beam_capacity + run2_index];
      } else {
        workspace.handles[right_destination] = kInvalidDeviceHandle;
        workspace.distances[right_destination] = FLT_MAX;
      }
    }
    if (run1_count != 0) {
      workspace.expanded[right_destination] = kCandidateSourceTag;
    }
  }
  __syncthreads();

  // Stage 2: (old-unexpanded, run0) + (run1, run2).
  for (u32 rank = threadIdx.x; rank < issue_capacity; rank += blockDim.x) {
    if (run1_count == 0) {
      const u32 source = beam_capacity + rank;
      const u32 destination =
        run3_count == 0 ? rank : 3u * beam_capacity + rank;
      workspace.handles[destination] = workspace.handles[source];
      workspace.distances[destination] = workspace.distances[source];
      workspace.expanded[destination] = workspace.expanded[source];
      continue;
    }
    const u32 left_index = stable_merge_a_corank(
      rank, workspace.distances + beam_capacity, issue_capacity,
      workspace.distances + 2u * beam_capacity, issue_capacity);
    const u32 right_index = rank - left_index;
    const bool take_left =
      left_index < issue_capacity &&
      (right_index >= issue_capacity ||
       !(workspace.distances[2u * beam_capacity + right_index] <
         workspace.distances[beam_capacity + left_index]));
    const u32 source =
      take_left ? beam_capacity + left_index : 2u * beam_capacity + right_index;
    const u32 destination = run3_count == 0 ? rank : 3u * beam_capacity + rank;
    workspace.handles[destination] = workspace.handles[source];
    workspace.distances[destination] = workspace.distances[source];
    workspace.expanded[destination] = workspace.expanded[source];
  }
  __syncthreads();

  // Stage 3 exists only when the fourth candidate leaf exists. Stage 2 writes
  // directly into the final compact prefix otherwise, eliminating one
  // query-wide shared-memory pass and barrier for the common 1--3 leaf case.
  if (run3_count != 0) {
    for (u32 rank = threadIdx.x; rank < issue_capacity; rank += blockDim.x) {
      const u32 prefix_index = stable_merge_a_corank(
        rank, workspace.distances + 3u * beam_capacity, issue_capacity,
        scratch_distances + 3u * beam_capacity, run3_count);
      const u32 run3_index = rank - prefix_index;
      const bool take_prefix =
        prefix_index < issue_capacity &&
        (run3_index >= run3_count ||
         !(scratch_distances[3u * beam_capacity + run3_index] <
           workspace.distances[3u * beam_capacity + prefix_index]));
      if (take_prefix) {
        workspace.handles[rank] =
          workspace.handles[3u * beam_capacity + prefix_index];
        workspace.distances[rank] =
          workspace.distances[3u * beam_capacity + prefix_index];
        workspace.expanded[rank] =
          workspace.expanded[3u * beam_capacity + prefix_index];
      } else if (run3_index < run3_count) {
        workspace.handles[rank] =
          scratch_handles[3u * beam_capacity + run3_index];
        workspace.distances[rank] =
          scratch_distances[3u * beam_capacity + run3_index];
        workspace.expanded[rank] = kCandidateSourceTag;
      } else {
        workspace.handles[rank] = kInvalidDeviceHandle;
        workspace.distances[rank] = FLT_MAX;
        workspace.expanded[rank] = kCandidateSourceTag;
      }
    }
    __syncthreads();
  }

  for (u32 rank = threadIdx.x; rank < issue_capacity; rank += blockDim.x) {
    const u64 handle = workspace.handles[rank];
    const f32 distance = workspace.distances[rank];
    const u16 source_tag = workspace.expanded[rank];
    u32 expanded_prefix_end = 0;
    if (source_tag != kCandidateSourceTag) {
      // An old item wins ties against every candidate run. Its source ordinal
      // therefore identifies the exact expanded prefix preceding it.
      expanded_prefix_end = static_cast<u32>(source_tag);
    } else if (stable_run_item_valid(handle, distance)) {
      // For a candidate, all old items with distance <= candidate distance
      // precede it (including signed-zero ties). Locate that stable
      // upper-bound, then read the precomputed number which are expanded.
      u32 low = 0;
      u32 high = beam_count;
      while (low < high) {
        const u32 middle = low + ((high - low) >> 1);
        if (!(distance < beam_distances[middle])) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      expanded_prefix_end = low;
    }
    const u32 expanded_before =
      workspace.old_expanded_prefix[expanded_prefix_end];
    const u32 authoritative_rank = rank + expanded_before;
    if (stable_run_item_valid(handle, distance) &&
        authoritative_rank < beam_capacity) {
      output_handles[rank] = handle;
      output_ranks[rank] = static_cast<u16>(authoritative_rank);
    } else {
      output_handles[rank] = kInvalidDeviceHandle;
      output_ranks[rank] = UINT16_MAX;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    output_count = 0;
    while (output_count < issue_capacity &&
           output_handles[output_count] != kInvalidDeviceHandle) {
      ++output_count;
    }
  }
  __syncthreads();
}

// Cheap early predictor for the partial Stable-Run state (old Beam + run0).
// Only two sorted heads participate, so a scalar stable merge is substantially
// cheaper than invoking the general five-way warp reduction once per rank.
// This is read-only and its rank is advisory: the later exact four-run merge
// remains the sole source of authoritative Beam positions.
__device__ __noinline__ void preview_partial_stable_shadow_frontier(
  const u64* beam_handles, const f32* beam_distances, const u8* beam_expanded,
  u32 beam_count, u32 beam_capacity, const u64* run_handles,
  const u8* run_flags, const f32* run_distances, u32 shadow_capacity,
  u64* output_handles, u16* output_ranks, u32& output_count) {
  if (threadIdx.x == 0) {
    u32 beam_head = 0;
    u32 run_head = 0;
    u32 merged_rank = 0;
    output_count = 0;
    shadow_capacity = min(shadow_capacity, beam_capacity);
    while (output_count < shadow_capacity) {
      while (beam_head < beam_count &&
             !stable_run_item_valid(beam_handles[beam_head],
                                    beam_distances[beam_head])) {
        beam_head = beam_count;
      }
      while (run_head < beam_capacity &&
             !stable_run_item_valid(run_handles[run_head],
                                    run_distances[run_head])) {
        run_head = beam_capacity;
      }
      if (beam_head >= beam_count && run_head >= beam_capacity) break;
      const bool take_beam =
        beam_head < beam_count &&
        (run_head >= beam_capacity ||
         !(run_distances[run_head] < beam_distances[beam_head]));
      const u64 handle =
        take_beam ? beam_handles[beam_head] : run_handles[run_head];
      const bool expanded = take_beam ? beam_expanded[beam_head++] != 0
                                      : run_flags[run_head++] != 0;
      if (!expanded) {
        output_handles[output_count] = handle;
        output_ranks[output_count] =
          static_cast<u16>(min(merged_rank, static_cast<u32>(UINT16_MAX)));
        ++output_count;
      }
      ++merged_rank;
    }
  }
  __syncthreads();
}

// Select only from the newly scored Stable-Run candidate streams.  This is a
// speculative predictor, not an authoritative merge: the normal Stable-Run
// materialization below remains the sole Beam publication path.  Candidate
// runs are individually sorted already, so a bounded four-head merge costs
// O(run_count * shadow_capacity) and never scans the expanded old-Beam prefix.
__device__ __noinline__ void preview_candidate_run_shadow_frontier(
  const u64* scratch_handles, const u8* scratch_flags,
  const f32* scratch_distances, u32 beam_capacity, u32 candidate_run_count,
  u32 shadow_capacity, u64* output_handles, u16* output_ranks,
  u32& output_count) {
  if (threadIdx.x == 0) {
    candidate_run_count = min(candidate_run_count, 4u);
    shadow_capacity = min(shadow_capacity, beam_capacity);
    u32 heads[4]{0, 0, 0, 0};
    output_count = 0;
    while (output_count < shadow_capacity) {
      u32 selected_run = UINT32_MAX;
      f32 selected_distance = FLT_MAX;
      for (u32 run = 0; run < candidate_run_count; ++run) {
        u32& head = heads[run];
        const u32 offset = run * beam_capacity;
        while (head < beam_capacity) {
          const u64 handle = scratch_handles[offset + head];
          const f32 distance = scratch_distances[offset + head];
          if (!stable_run_item_valid(handle, distance)) {
            head = beam_capacity;
            break;
          }
          if (scratch_flags[offset + head] == 0) break;
          ++head;
        }
        if (head >= beam_capacity) continue;
        const f32 distance = scratch_distances[offset + head];
        if (selected_run == UINT32_MAX ||
            stable_run_head_precedes(distance, run, selected_distance,
                                     selected_run)) {
          selected_run = run;
          selected_distance = distance;
        }
      }
      if (selected_run == UINT32_MAX) break;
      const u32 head = heads[selected_run]++;
      const u32 offset = selected_run * beam_capacity + head;
      output_handles[output_count] = scratch_handles[offset];
      // The node is not authoritative yet; record its exact rank in the
      // candidate-only issue frontier. Validation later maps by handle.
      output_ranks[output_count] = static_cast<u16>(output_count);
      ++output_count;
    }
  }
  __syncthreads();
}

}  // namespace gpu_search::persistent_kernel_detail

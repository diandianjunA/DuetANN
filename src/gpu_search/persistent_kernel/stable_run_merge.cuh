#pragma once

#include "gpu_search/persistent_kernel/candidate_scoring_base.cuh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ bool stable_run_item_valid(u64 handle,
                                                      f32 distance) {
  return handle != kInvalidDeviceHandle && isfinite(distance) &&
         distance != FLT_MAX;
}

__device__ __forceinline__ bool stable_run_head_precedes(f32 lhs_distance,
                                                         u32 lhs_run,
                                                         f32 rhs_distance,
                                                         u32 rhs_run) {
  return lhs_distance < rhs_distance ||
         (lhs_distance == rhs_distance && lhs_run < rhs_run);
}

// Return how many items from A occur in the first `diagonal` outputs of a
// stable merge.  A wins an equal-distance tie, matching the concatenated input
// order used by CUB's stable radix sort.  BlockRadixSort also treats -0/+0 as
// equal, so the boundary deliberately uses floating-point comparisons rather
// than comparing transformed key bits.
__device__ __forceinline__ u32 stable_merge_a_corank(u32 diagonal,
                                                     const f32* a_distances,
                                                     u32 a_count,
                                                     const f32* b_distances,
                                                     u32 b_count) {
  u32 low = diagonal > b_count ? diagonal - b_count : 0u;
  u32 high = min(diagonal, a_count);
  while (low <= high) {
    const u32 a = low + ((high - low) >> 1);
    const u32 b = diagonal - a;
    if (a != 0 && b < b_count && b_distances[b] < a_distances[a - 1]) {
      high = a - 1;
      continue;
    }
    if (b != 0 && a < a_count && !(b_distances[b - 1] < a_distances[a])) {
      low = a + 1;
      continue;
    }
    return a;
  }
  return min(low, a_count);
}

// Sort one contiguous 512-item candidate input window with the same stable,
// distance-only ordering as CUB's BlockRadixSort and write the run's first
// beam_capacity entries.  Each key is (ordered distance, raw window ordinal);
// the ordinal makes keys unique, so the register bitonic network reproduces
// the stable radix order exactly, including equal-distance candidates and
// CUB's treatment of -0/+0 as one key.  The four warp-local sorted 128-entry
// runs are reduced on warp 0.  For beam_capacity <= 128 a sequence of bitonic
// min-merges over the shared exchange buffer keeps exactly the global top-128;
// only those entries can contribute to the global top-k, and
// topK(topK(A u B) u topK(C u D)) == topK(A u B u C u D), so discarding the
// losing half of each pairwise merge is exact.  For beam_capacity <= 256 each
// pairwise step instead keeps the complete 256-entry merge (a full bitonic
// merge of the concatenated run pair), because a 256-entry Beam may need up
// to 256 entries from a single warp run; the final step then keeps the exact
// top-256 of the two merged runs.
__device__ __noinline__ void stable_sort_candidate_run(
  CandidateSortWorkspace& sort_workspace, const u64* candidate_handles,
  const f32* candidate_distances, u32 candidate_count, u32 input_offset,
  u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
  u32 output_offset, u32 beam_capacity,
  BeamMergeCycleBreakdown* cycle_breakdown, u64* phase_started) {
  constexpr u32 items_per_thread = kApproximateSortItemsCompactFinal256;
  constexpr u64 invalid_key = UINT64_MAX;
  static_assert(kPersistentMaxBeam <= 2 * kApproximateSortThreadsCompact);
  // Construction validates beam_capacity <= kPersistentMaxBeam, so a wider
  // request here means an unsupported test/ablation configuration.
  if (beam_capacity > kPersistentMaxBeam) {
    __trap();
  }
  const u32 lane = threadIdx.x % 32;
  const u32 warp = threadIdx.x / 32;
  u64 keys[items_per_thread];
#pragma unroll
  for (u32 item = 0; item < items_per_thread; ++item) {
    const u32 ordinal = threadIdx.x * items_per_thread + item;
    const u32 index = input_offset + ordinal;
    u64 key = invalid_key;
    if (index < candidate_count) {
      const u64 handle = candidate_handles[index];
      const f32 distance = candidate_distances[index];
      if (handle != kInvalidDeviceHandle && finite_f32_bits(distance) &&
          distance != FLT_MAX) {
        // CUB's floating radix order treats signed zero as one equivalent
        // key.  Canonicalizing both zeros preserves that behavior.
        const u32 bits = distance == 0.0f ? 0u : __float_as_uint(distance);
        const u32 ordered_distance =
          (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
        key = (static_cast<u64>(ordered_distance) << 32u) | ordinal;
      }
    }
    keys[item] = key;
  }
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->prepare += now - *phase_started;
    *phase_started = now;
  }
  warp_bitonic_sort_blocked<items_per_thread>(keys);
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->sort += now - *phase_started;
    *phase_started = now;
  }
  if (beam_capacity <= kApproximateSortThreadsCompact) {
    // Reduce the four ascending warp runs to the global top-128 on warp 0.
    // min(A[i], B[127-i]) keeps exactly the 128 smallest keys of A u B in
    // bitonic order; the half-cleaner merge then restores ascending order.
    for (u32 partner = 1; partner < 4; ++partner) {
      __syncthreads();
      if (warp == partner) {
#pragma unroll
        for (u32 item = 0; item < items_per_thread; ++item) {
          sort_workspace.stable_run_exchange[lane * items_per_thread + item] =
            keys[item];
        }
      }
      __syncthreads();
      if (warp == 0) {
#pragma unroll
        for (u32 item = 0; item < items_per_thread; ++item) {
          const u32 position = lane * items_per_thread + item;
          const u64 other =
            sort_workspace.stable_run_exchange[127u - position];
          keys[item] = keys[item] < other ? keys[item] : other;
        }
        warp_bitonic_merge_blocked<items_per_thread>(keys);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
      const u64 now = clock64();
      cycle_breakdown->sort += now - *phase_started;
      *phase_started = now;
    }
    if (warp == 0) {
#pragma unroll
      for (u32 item = 0; item < items_per_thread; ++item) {
        const u32 output = lane * items_per_thread + item;
        if (output >= beam_capacity) continue;
        const u32 destination = output_offset + output;
        const u64 key = keys[item];
        u64 handle = kInvalidDeviceHandle;
        f32 distance = FLT_MAX;
        if (key != invalid_key) {
          const u32 ordinal = static_cast<u32>(key);
          const u64 resolved_handle =
            candidate_handles[input_offset + ordinal];
          const f32 resolved_distance =
            candidate_distances[input_offset + ordinal];
          if (resolved_handle != kInvalidDeviceHandle &&
              finite_f32_bits(resolved_distance) &&
              resolved_distance != FLT_MAX) {
            handle = resolved_handle;
            distance = resolved_distance;
          }
        }
        scratch_handles[destination] = handle;
        scratch_distances[destination] = distance;
        scratch_flags[destination] = 0;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
      const u64 now = clock64();
      cycle_breakdown->materialize += now - *phase_started;
      *phase_started = now;
    }
    return;
  }

  // Wide (beam 129..256) reduction: pairwise full bitonic 256-merges.
  // C = A ++ reverse(B) is bitonic, so one half-cleaner merge network over
  // the concatenated 256-entry sequence restores the complete ascending order
  // of A u B; a 256-entry Beam may need every entry of a merged pair, so
  // unlike the narrow path no half is discarded here.  Round A merges warp
  // runs (0,1) on warp 0 and round B merges (2,3) on warp 2.  Both rounds
  // share the single 256-entry exchange buffer, so CTA barriers serialize
  // them.  Round C then keeps exactly the top-256 of the two sorted
  // 256-entry results on warp 0 via C[i] = min(A[i], B[255-i]) -- itself a
  // bitonic sequence -- followed by one final merge network.
  constexpr u32 wide_items_per_thread = 2 * items_per_thread;
  u64 wide[wide_items_per_thread];
  __syncthreads();
  if (warp == 0 || warp == 1) {
    const u32 base = warp == 0 ? 0u : 128u;
#pragma unroll
    for (u32 item = 0; item < items_per_thread; ++item) {
      sort_workspace.stable_run_exchange[base + lane * items_per_thread +
                                         item] = keys[item];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (u32 item = 0; item < wide_items_per_thread; ++item) {
      const u32 position = lane * wide_items_per_thread + item;
      wide[item] =
        sort_workspace
          .stable_run_exchange[position < 128u ? position : 383u - position];
    }
    warp_bitonic_merge_blocked<wide_items_per_thread>(wide);
  }
  __syncthreads();
  if (warp == 2 || warp == 3) {
    const u32 base = warp == 2 ? 0u : 128u;
#pragma unroll
    for (u32 item = 0; item < items_per_thread; ++item) {
      sort_workspace.stable_run_exchange[base + lane * items_per_thread +
                                         item] = keys[item];
    }
  }
  __syncthreads();
  if (warp == 2) {
#pragma unroll
    for (u32 item = 0; item < wide_items_per_thread; ++item) {
      const u32 position = lane * wide_items_per_thread + item;
      wide[item] =
        sort_workspace
          .stable_run_exchange[position < 128u ? position : 383u - position];
    }
    warp_bitonic_merge_blocked<wide_items_per_thread>(wide);
#pragma unroll
    for (u32 item = 0; item < wide_items_per_thread; ++item) {
      sort_workspace.stable_run_exchange[lane * wide_items_per_thread + item] =
        wide[item];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->sort += now - *phase_started;
    *phase_started = now;
  }
  if (warp == 0) {
    u64 final_keys[wide_items_per_thread];
#pragma unroll
    for (u32 item = 0; item < wide_items_per_thread; ++item) {
      const u32 position = lane * wide_items_per_thread + item;
      // Top-256 of the two ascending 256-entry runs: C[i] = min(A[i],
      // B[255-i]) is bitonic, so one merge network yields the exact top half
      // of A u B in ascending order.
      const u64 other = sort_workspace.stable_run_exchange[255u - position];
      final_keys[item] = wide[item] < other ? wide[item] : other;
    }
    warp_bitonic_merge_blocked<wide_items_per_thread>(final_keys);
#pragma unroll
    for (u32 item = 0; item < wide_items_per_thread; ++item) {
      const u32 output = lane * wide_items_per_thread + item;
      if (output >= beam_capacity) continue;
      const u32 destination = output_offset + output;
      const u64 key = final_keys[item];
      u64 handle = kInvalidDeviceHandle;
      f32 distance = FLT_MAX;
      if (key != invalid_key) {
        const u32 ordinal = static_cast<u32>(key);
        const u64 resolved_handle =
          candidate_handles[input_offset + ordinal];
        const f32 resolved_distance =
          candidate_distances[input_offset + ordinal];
        if (resolved_handle != kInvalidDeviceHandle &&
            finite_f32_bits(resolved_distance) &&
            resolved_distance != FLT_MAX) {
          handle = resolved_handle;
          distance = resolved_distance;
        }
      }
      scratch_handles[destination] = handle;
      scratch_distances[destination] = distance;
      scratch_flags[destination] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->materialize += now - *phase_started;
    *phase_started = now;
  }
}

#if DVSTOR_PERSISTENT_WIDE_SORT
// CUB radix variant retained for 256-thread ablation builds.
template <class BlockSort, u32 ItemsPerThread>
__device__ __noinline__ void stable_sort_candidate_run_radix(
  typename BlockSort::TempStorage& radix_storage, const u64* candidate_handles,
  const f32* candidate_distances, u32 candidate_count, u32 input_offset,
  u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
  u32 output_offset, u32 beam_capacity,
  BeamMergeCycleBreakdown* cycle_breakdown, u64* phase_started) {
  f32 local_distances[ItemsPerThread];
  u64 local_values[ItemsPerThread];
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = input_offset + threadIdx.x * ItemsPerThread + item;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (index < candidate_count) {
      handle = candidate_handles[index];
      distance = candidate_distances[index];
    }
    if (handle == kInvalidDeviceHandle || !finite_f32_bits(distance) ||
        distance == FLT_MAX) {
      handle = kInvalidDeviceHandle;
      distance = FLT_MAX;
    }
    local_distances[item] = distance;
    local_values[item] = handle;
  }
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->prepare += now - *phase_started;
    *phase_started = now;
  }
  BlockSort(radix_storage).Sort(local_distances, local_values);
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->sort += now - *phase_started;
    *phase_started = now;
  }
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    const u32 destination = output_offset + output;
    scratch_handles[destination] = local_values[item];
    scratch_distances[destination] = local_distances[item];
    scratch_flags[destination] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->materialize += now - *phase_started;
    *phase_started = now;
  }
}
#endif  // DVSTOR_PERSISTENT_WIDE_SORT

__device__ void clear_stable_candidate_run(
  u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
  u32 output_offset, u32 beam_capacity,
  BeamMergeCycleBreakdown* cycle_breakdown, u64* phase_started) {
  for (u32 index = threadIdx.x; index < beam_capacity; index += blockDim.x) {
    const u32 destination = output_offset + index;
    scratch_handles[destination] = kInvalidDeviceHandle;
    scratch_flags[destination] = 0;
    scratch_distances[destination] = FLT_MAX;
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->materialize += now - *phase_started;
    *phase_started = now;
  }
}

template <class OriginExpanded>
__device__ __noinline__ void restore_stable_candidate_flags(
  const u64* origin_handles, const OriginExpanded* origin_expanded,
  u32 origin_count, u64* scratch_handles, u8* scratch_flags,
  const f32* scratch_distances, u32 candidate_slots) {
  for (u32 index = threadIdx.x; index < candidate_slots; index += blockDim.x) {
    const u64 handle = scratch_handles[index];
    const f32 distance = scratch_distances[index];
    u8 expanded = 0;
    if (stable_run_item_valid(handle, distance)) {
      for (u32 prior = 0; prior < origin_count; ++prior) {
        if (origin_handles[prior] != handle) continue;
        expanded = origin_expanded[prior] != 0 ? u8{1} : u8{0};
        break;
      }
    }
    scratch_flags[index] = expanded;
  }
  __syncthreads();
}

// Materialize the exact top-k of three individually stable runs:
//   old Beam, candidate pass 0, candidate pass 1.
// The old run wins equal-distance ties, followed by candidate input order.
template <class OriginExpanded>
__device__ __noinline__ void materialize_stable_candidate_runs(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u32 existing_count,
  const u64* origin_handles, const OriginExpanded* origin_expanded,
  u32 origin_count, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, u32 candidate_run_count,
  CandidateWorkspaceArrays& output, bool restore_flags = true) {
  // Restore expanded metadata while the authoritative old Beam is still
  // intact. Production candidates are disjoint from the old Beam through
  // visited, but direct merge callers historically allow duplicates: a
  // duplicate inherits the first old entry's expanded bit.
  const u32 candidate_slots = candidate_run_count * beam_capacity;
  if (restore_flags) {
    restore_stable_candidate_flags(origin_handles, origin_expanded,
                                   origin_count, scratch_handles, scratch_flags,
                                   scratch_distances, candidate_slots);
  }

  // Stage 1: exact stable top-K of old Beam and candidate run 0.  One thread
  // owns each output rank (or a small stride for K=256/128-thread CTAs), using
  // co-rank to avoid both a block-wide second sort and a serial k-way merge.
  for (u32 index = threadIdx.x; index < beam_capacity; index += blockDim.x) {
    const u32 old_index = stable_merge_a_corank(
      index, beam_distances, existing_count, scratch_distances, beam_capacity);
    const u32 candidate_index = index - old_index;
    const bool take_old =
      old_index < existing_count &&
      (candidate_index >= beam_capacity ||
       !(scratch_distances[candidate_index] < beam_distances[old_index]));
    if (take_old) {
      output.handles[index] = beam_handles[old_index];
      output.distances[index] = beam_distances[old_index];
      output.expanded[index] = beam_expanded[old_index];
    } else {
      output.handles[index] = scratch_handles[candidate_index];
      output.distances[index] = scratch_distances[candidate_index];
      output.expanded[index] = scratch_flags[candidate_index];
    }
  }
  __syncthreads();

  // Stage 2: merge the stable (old, run0) prefix with run1.  The intermediate
  // run wins equal keys, preserving the global old < run0 < run1 order.
  const u32 second_count = candidate_run_count > 1 ? beam_capacity : 0u;
  for (u32 index = threadIdx.x; index < beam_capacity; index += blockDim.x) {
    const u32 intermediate_index =
      stable_merge_a_corank(index, output.distances, beam_capacity,
                            scratch_distances + beam_capacity, second_count);
    const u32 candidate_index = index - intermediate_index;
    const bool take_intermediate =
      intermediate_index < beam_capacity &&
      (candidate_index >= second_count ||
       !(scratch_distances[beam_capacity + candidate_index] <
         output.distances[intermediate_index]));
    if (take_intermediate) {
      beam_handles[index] = output.handles[intermediate_index];
      beam_distances[index] = output.distances[intermediate_index];
      beam_expanded[index] = output.expanded[intermediate_index];
    } else {
      const u32 source = beam_capacity + candidate_index;
      beam_handles[index] = scratch_handles[source];
      beam_distances[index] = scratch_distances[source];
      beam_expanded[index] = scratch_flags[source];
    }
    beam_ids[index] = UINT32_MAX;
  }
  __syncthreads();

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

// State shared between the sort/preview and authoritative materialize phases.
// Keeping the phase boundary explicit lets query traversal issue the next RDMA
// wave while the current Stable-Run materialization is still pending.
struct StableMergePreparedState {
  u32 original_count{};
  u32 candidate_run_count{};
  u32 compact{};
  u32 origin_copied{};
  u32 prepared{};
  // ASFE may publish only the exact authoritative prefix needed by the next
  // Issue Frontier, then finish the Beam suffix while RDMA is in flight.
  u32 fused_tree_prepared{};
  // Number of exact ranks already evaluated at every internal level of the
  // fixed Stable-Run tree.  Prefix evaluation is safe because every leaf is
  // an immutable sorted run and the first N outputs of a stable merge depend
  // only on the first N inputs of either child.  The remaining ranks are
  // evaluated from the same snapshots after RDMA publication.
  u32 fused_tree_prefix{};
  // Nonzero means the Issue certificate is read-only rather than an
  // authoritative Beam prefix. Value 2 denotes the bounded Stage-1A/four-head
  // certificate used by the production hot path. In either case
  // expanded/visited/termination remain untouched until finish publishes the
  // complete merge.
  u32 deferred_prefix{};
  u32 materialized_prefix{};
  // Commit-Ordered Streaming Stable Fold (COSSF). Candidate leaves are
  // folded, in raw Stable-Run order, into a private ping-pong top-K
  // accumulator while later core CQ groups are still in flight.
  u32 streaming_fold{};
  u32 streaming_accumulator_segment{};
  u32 streaming_candidate_offset{};
  // At most one sub-leaf interval may be sealed before finalization. Full
  // 512-item leaves remain unrestricted. This prevents a sequence of narrow
  // CQ groups from turning one necessary Stable-Run leaf into O(commit_width)
  // independent sorts/folds.
  u32 streaming_partial_sealed{};
  u64 phase_started{};
};

// Prepare the first compact Stable-Run independently. Query traversal may
// later publish a private, CTA-local authoritative prefix and launch a
// read-only shadow RDMA wave before the remaining merge ranks. No next-round
// selector or termination check can observe that partial prefix until the
// suffix finish barrier has completed.
__device__ __noinline__ void begin_compact_approximate_stable_runs(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u64* beam_handles, u8* beam_expanded, u32 beam_count, u32 beam_capacity,
  u64* scratch_handles, u8* scratch_flags, f32* scratch_distances,
  CandidateWorkspace& workspace, StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool restore_candidate_flags = true) {
  if (threadIdx.x == 0) {
    state = {};
    state.original_count = beam_count;
    state.phase_started = clock64();
    if (cycle_breakdown != nullptr) *cycle_breakdown = {};
  }
  __syncthreads();

  if (blockDim.x != kApproximateSortThreadsCompact) return;

  if (restore_candidate_flags) {
    for (u32 index = threadIdx.x; index < beam_count; index += blockDim.x) {
      workspace.arrays.handles[beam_capacity + index] = beam_handles[index];
      workspace.arrays.expanded[beam_capacity + index] = beam_expanded[index];
    }
  }
  __syncthreads();

  if (candidate_count != 0) {
    stable_sort_candidate_run(
      workspace.sort, candidate_handles,
      candidate_distances, candidate_count, 0, scratch_handles, scratch_flags,
      scratch_distances, 0, beam_capacity, cycle_breakdown,
      &state.phase_started);
  } else {
    clear_stable_candidate_run(scratch_handles, scratch_flags,
                               scratch_distances, 0, beam_capacity,
                               cycle_breakdown, &state.phase_started);
  }
  if (threadIdx.x == 0) {
    state.candidate_run_count = 1;
    state.compact = 1;
    state.origin_copied = restore_candidate_flags ? 1u : 0u;
  }
  __syncthreads();
}

// Initialize a Stable-Run whose immutable 512-item candidate leaves may be
// sealed as ordered parent groups are scored.  This is the merge-side half of
// frontier decoupling: the query CTA performs work that the authoritative
// merge must eventually do while a later critical CQ group is still in
// flight.  No Beam entry is published and a leaf is never sealed until every
// input position in that leaf is final.
__device__ __noinline__ void begin_streaming_compact_stable_runs(
  u64* beam_handles, const f32* beam_distances, u8* beam_expanded,
  u32 beam_count, const u32* selected_beam_ranks, u32 selected_count,
  u32 beam_capacity, CandidateWorkspace& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  if (threadIdx.x == 0) {
    state = {};
    state.original_count = beam_count;
    state.compact = 1;
    state.origin_copied = 0;
    state.streaming_fold = 1;
    state.streaming_accumulator_segment = 0;
    state.streaming_candidate_offset = 0;
    state.phase_started = clock64();
    if (cycle_breakdown != nullptr) *cycle_breakdown = {};
  }
  __syncthreads();

  if (blockDim.x != kApproximateSortThreadsCompact) return;
  // Snapshot the immutable old run into the first private accumulator. The
  // selected ranks have not yet been published as expanded while ordered CQ
  // scoring is active, so overlay that frozen commit metadata only in the
  // private copy. A failed critical dependency discards the accumulator.
  for (u32 index = threadIdx.x; index < beam_capacity; index += blockDim.x) {
    const bool valid =
      index < beam_count &&
      stable_run_item_valid(beam_handles[index], beam_distances[index]);
    workspace.arrays.handles[index] =
      valid ? beam_handles[index] : kInvalidDeviceHandle;
    workspace.arrays.distances[index] = valid ? beam_distances[index] : FLT_MAX;
    workspace.arrays.expanded[index] =
      static_cast<u8>(valid && beam_expanded[index] != 0 ? 1u : 0u);
  }
  __syncthreads();
  // Frozen commit ranks are unique. One thread per selected position updates
  // the private flag after the bulk copy, avoiding K*C rank comparisons.
  for (u32 position = threadIdx.x; position < selected_count;
       position += blockDim.x) {
    const u32 rank = selected_beam_ranks[position];
    if (rank < beam_capacity) {
      workspace.arrays.expanded[rank] = 1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && cycle_breakdown != nullptr) {
    const u64 now = clock64();
    cycle_breakdown->prepare += now - state.phase_started;
    state.phase_started = now;
  }
  __syncthreads();
}

// Seal every complete immutable leaf visible in candidate_count.  On the
// final call, also sort the last partial leaf, clear absent leaves, and mark
// the ordinary Stable-Run state prepared.  candidate_run_count is both the
// number of sealed leaves and the next raw input partition; this makes repeat
// calls idempotent and prevents rescanning candidates already covered during
// RDMA progress.
__device__ __noinline__ void extend_streaming_compact_stable_runs(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u32 beam_capacity, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, CandidateWorkspace& workspace,
  StableMergePreparedState& state, bool finalize,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool seal_partial = false) {
  if (blockDim.x != kApproximateSortThreadsCompact || state.compact == 0 ||
      state.prepared != 0) {
    return;
  }
  constexpr u32 leaf_capacity =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactFinal256;
  candidate_count =
    min(candidate_count, static_cast<u32>(kPersistentMaxMergeCandidates));
  const u32 runs_before = state.candidate_run_count;
  while (state.streaming_candidate_offset < candidate_count) {
    const u32 pass = state.candidate_run_count;
    const u32 input_offset = state.streaming_candidate_offset;
    const u32 available = candidate_count - input_offset;
    const bool complete_leaf = available >= leaf_capacity;
    // Seal a partial microbatch only when no complete leaf from this progress
    // interval was available. This exposes useful work without needlessly
    // fragmenting a large interval into an extra Stable-Run.
    const bool partial_allowed =
      finalize || (seal_partial && state.streaming_partial_sealed == 0 &&
                   state.candidate_run_count == runs_before);
    if (!complete_leaf && !partial_allowed) break;
    const u32 input_end = min(input_offset + leaf_capacity, candidate_count);
    const u32 output_offset = 0;
    // Time only the necessary leaf work itself.  The interval since the
    // preceding progress call belongs to RDMA/decode/PQ, not Beam prepare.
    if (threadIdx.x == 0) state.phase_started = clock64();
    __syncthreads();
    stable_sort_candidate_run(
      workspace.sort, candidate_handles,
      candidate_distances, input_end,
      // input_end is the immutable end of this run. Later candidates must
      // never be pulled into a previously sealed partial microbatch.
      input_offset, scratch_handles, scratch_flags, scratch_distances,
      output_offset, beam_capacity, cycle_breakdown, &state.phase_started);

    // topK(topK(A) union B) == topK(A union B).  The accumulator is the
    // stable total order of the old Beam and every earlier raw candidate
    // leaf; it wins equal-distance ties against this later leaf. Therefore a
    // left fold is bitwise equivalent to the balanced authoritative
    // Stable-Run tree while making each completed leaf immediately reusable.
    const u32 source_base = state.streaming_accumulator_segment * beam_capacity;
    const u32 destination_segment = state.streaming_accumulator_segment ^ 1u;
    const u32 destination_base = destination_segment * beam_capacity;
    for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
      const u32 accumulator_index = stable_merge_a_corank(
        rank, workspace.arrays.distances + source_base, beam_capacity,
        scratch_distances + output_offset, beam_capacity);
      const u32 candidate_index = rank - accumulator_index;
      const bool take_accumulator =
        accumulator_index < beam_capacity &&
        (candidate_index >= beam_capacity ||
         !(scratch_distances[output_offset + candidate_index] <
           workspace.arrays.distances[source_base + accumulator_index]));
      if (take_accumulator) {
        const u32 source = source_base + accumulator_index;
        workspace.arrays.handles[destination_base + rank] =
          workspace.arrays.handles[source];
        workspace.arrays.distances[destination_base + rank] =
          workspace.arrays.distances[source];
        workspace.arrays.expanded[destination_base + rank] =
          workspace.arrays.expanded[source];
      } else {
        const u32 source = output_offset + candidate_index;
        workspace.arrays.handles[destination_base + rank] =
          scratch_handles[source];
        workspace.arrays.distances[destination_base + rank] =
          scratch_distances[source];
        workspace.arrays.expanded[destination_base + rank] =
          scratch_flags[source];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      state.candidate_run_count = pass + 1u;
      state.streaming_accumulator_segment = destination_segment;
      state.streaming_candidate_offset = input_end;
      if (!complete_leaf && !finalize) {
        state.streaming_partial_sealed = 1;
      }
      if (cycle_breakdown != nullptr) {
        const u64 now = clock64();
        cycle_breakdown->materialize += now - state.phase_started;
        state.phase_started = now;
      }
    }
    __syncthreads();
  }

  if (!finalize) return;
  if (threadIdx.x == 0) {
    state.prepared = 1;
    state.phase_started = clock64();
  }
  __syncthreads();
}

// Extract the exact unexpanded Issue Frontier from the completed private
// COSSF accumulator.  This stage is read-only with respect to Beam, visited,
// and expanded; issue_ranks are authoritative ranks in the future Beam.
__device__ __noinline__ void prepare_streaming_stable_fold_frontier_certificate(
  u32 beam_capacity, u32 issue_capacity, CandidateWorkspaceArrays& workspace,
  u64* issue_handles, u16* issue_ranks, u32& issue_count,
  StableMergePreparedState& state) {
  constexpr u32 warp_width = 32;
  constexpr u32 full_warp = 0xffffffffu;
  issue_capacity = min(min(issue_capacity, beam_capacity),
                       static_cast<u32>(kPersistentFrontierRobCapacity));
  const u32 source_base = state.streaming_accumulator_segment * beam_capacity;
  if (blockDim.x == kApproximateSortThreadsCompact) {
    static_assert(kPersistentMaxBeam <=
                  2u * kApproximateSortThreadsCompact);
    constexpr u32 warp_count = kApproximateSortThreadsCompact / warp_width;
    const u32 lane = threadIdx.x & (warp_width - 1u);
    const u32 warp = threadIdx.x / warp_width;
    // Cover up to 256 accumulator ranks with two ranks per thread.  The
    // per-warp ballot counts double as temporary counters and are dead once
    // every thread snapshots its prefix bases across the barrier.
    const u32 rank0 = threadIdx.x;
    const u32 rank1 = threadIdx.x + kApproximateSortThreadsCompact;
    const bool issue0 =
      rank0 < beam_capacity &&
      workspace.expanded[source_base + rank0] == 0 &&
      stable_run_item_valid(workspace.handles[source_base + rank0],
                            workspace.distances[source_base + rank0]);
    const bool issue1 =
      rank1 < beam_capacity &&
      workspace.expanded[source_base + rank1] == 0 &&
      stable_run_item_valid(workspace.handles[source_base + rank1],
                            workspace.distances[source_base + rank1]);
    const u32 issue_mask0 = __ballot_sync(full_warp, issue0);
    const u32 issue_mask1 = __ballot_sync(full_warp, issue1);
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
        issue_handles[destination] = workspace.handles[source_base + rank0];
        issue_ranks[destination] = static_cast<u16>(rank0);
      }
    }
    if (issue1) {
      const u32 destination = base1 + __popc(issue_mask1 & lower_lanes);
      if (destination < issue_capacity) {
        issue_handles[destination] = workspace.handles[source_base + rank1];
        issue_ranks[destination] = static_cast<u16>(rank1);
      }
    }
  } else if (threadIdx.x == 0) {
    issue_count = 0;
    for (u32 rank = 0; rank < beam_capacity && issue_count < issue_capacity;
         ++rank) {
      if (workspace.expanded[source_base + rank] != 0 ||
          !stable_run_item_valid(workspace.handles[source_base + rank],
                                 workspace.distances[source_base + rank])) {
        continue;
      }
      issue_handles[issue_count] = workspace.handles[source_base + rank];
      issue_ranks[issue_count] = static_cast<u16>(rank);
      ++issue_count;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The exact full private fold is necessary work accumulated across CQ
    // intervals, not a synchronous full-prefix preview at this point.
    state.fused_tree_prepared = 1;
    state.fused_tree_prefix = beam_capacity;
    state.materialized_prefix = beam_capacity;
  }
  __syncthreads();
}

// Publish the completed COSSF accumulator after its RDMA descriptor has
// become visible to the owner. This single coalesced copy is the only point
// at which the streaming path changes the authoritative Beam.
__device__ __noinline__ void finish_streaming_stable_fold(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, CandidateWorkspaceArrays& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr) {
  const u32 source_base = state.streaming_accumulator_segment * beam_capacity;
  for (u32 rank = threadIdx.x; rank < beam_capacity; rank += blockDim.x) {
    beam_handles[rank] = workspace.handles[source_base + rank];
    beam_ids[rank] = UINT32_MAX;
    beam_distances[rank] = workspace.distances[source_base + rank];
    beam_expanded[rank] = workspace.expanded[source_base + rank];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < beam_capacity &&
           stable_run_item_valid(beam_handles[valid], beam_distances[valid])) {
      ++valid;
    }
    beam_count = valid;
    if (cycle_breakdown != nullptr) {
      cycle_breakdown->materialize += clock64() - state.phase_started;
    }
    state.streaming_fold = 0;
    state.streaming_accumulator_segment = 0;
    state.fused_tree_prepared = 0;
    state.fused_tree_prefix = 0;
    state.materialized_prefix = 0;
    state.prepared = 0;
  }
  __syncthreads();
}

// Complete compact Stable-Run preparation after an optional communication
// issue interval.  Callers reset state.phase_started immediately before this
// function when that interval must not be charged to Beam merge time.
__device__ __noinline__ void complete_compact_approximate_stable_runs(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u32 beam_capacity, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, CandidateWorkspace& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool restore_candidate_flags = true) {
  if (blockDim.x != kApproximateSortThreadsCompact || state.compact == 0 ||
      state.candidate_run_count != 1) {
    return;
  }

  constexpr u32 compact_run_capacity =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactFinal256;
  for (u32 pass = 1; pass < 4; ++pass) {
    const u32 input_offset = pass * compact_run_capacity;
    const u32 output_offset = pass * beam_capacity;
    if (candidate_count > input_offset) {
      stable_sort_candidate_run(
        workspace.sort, candidate_handles,
        candidate_distances, candidate_count, input_offset, scratch_handles,
        scratch_flags, scratch_distances, output_offset, beam_capacity,
        cycle_breakdown, &state.phase_started);
    } else {
      clear_stable_candidate_run(
        scratch_handles, scratch_flags, scratch_distances, output_offset,
        beam_capacity, cycle_breakdown, &state.phase_started);
    }
  }
  if (restore_candidate_flags) {
    restore_stable_candidate_flags(
      workspace.arrays.handles + beam_capacity,
      workspace.arrays.expanded + beam_capacity, state.original_count,
      scratch_handles, scratch_flags, scratch_distances, 4u * beam_capacity);
  }
  if (threadIdx.x == 0) {
    state.candidate_run_count = 4;
    state.prepared = 1;
  }
  __syncthreads();
}

__device__ __noinline__ void prepare_approximate_stable_runs(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u64* scratch_handles, u8* scratch_flags,
  f32* scratch_distances, CandidateWorkspace& workspace,
  StableMergePreparedState& state,
  BeamMergeCycleBreakdown* cycle_breakdown = nullptr,
  bool restore_candidate_flags = true) {
  if (blockDim.x == kApproximateSortThreadsCompact) {
    begin_compact_approximate_stable_runs(
      candidate_handles, candidate_distances, candidate_count, beam_handles,
      beam_expanded, beam_count, beam_capacity, scratch_handles, scratch_flags,
      scratch_distances, workspace, state, cycle_breakdown,
      restore_candidate_flags);
    complete_compact_approximate_stable_runs(
      candidate_handles, candidate_distances, candidate_count, beam_capacity,
      scratch_handles, scratch_flags, scratch_distances, workspace, state,
      cycle_breakdown, restore_candidate_flags);
    return;
  }

  const u32 existing_count = beam_count;
  if (threadIdx.x == 0) {
    state = {};
    state.original_count = existing_count;
    state.phase_started = clock64();
    if (cycle_breakdown != nullptr) *cycle_breakdown = {};
  }
  __syncthreads();

  if (blockDim.x == kApproximateSortThreadsWide) {
#if DVSTOR_PERSISTENT_WIDE_SORT
    stable_sort_candidate_run_radix<ApproximateBlockSortWideRun,
                              kApproximateSortItemsWideRun>(
      workspace.sort.radix_sort_wide_run, candidate_handles,
      candidate_distances, candidate_count, 0, scratch_handles, scratch_flags,
      scratch_distances, 0, beam_capacity, cycle_breakdown,
      &state.phase_started);
    if (candidate_count > kApproximateSortCapacityCompactPass) {
      stable_sort_candidate_run_radix<ApproximateBlockSortWideRun,
                                kApproximateSortItemsWideRun>(
        workspace.sort.radix_sort_wide_run, candidate_handles,
        candidate_distances, candidate_count,
        kApproximateSortCapacityCompactPass, scratch_handles, scratch_flags,
        scratch_distances, beam_capacity, beam_capacity, cycle_breakdown,
        &state.phase_started);
    } else {
      clear_stable_candidate_run(
        scratch_handles, scratch_flags, scratch_distances, beam_capacity,
        beam_capacity, cycle_breakdown, &state.phase_started);
    }
    if (restore_candidate_flags) {
      restore_stable_candidate_flags(
        beam_handles, beam_expanded, existing_count, scratch_handles,
        scratch_flags, scratch_distances, 2u * beam_capacity);
    }
    if (threadIdx.x == 0) {
      state.candidate_run_count = 2;
      state.compact = 0;
      state.prepared = 1;
    }
#endif  // DVSTOR_PERSISTENT_WIDE_SORT
  } else {
    if (threadIdx.x == 0) beam_count = 0;
    __syncthreads();
    return;
  }
  __syncthreads();
}

// Evaluate one rank interval at the first two levels of the exact five-run
// Stable-Run merge tree.  All children are immutable sorted runs. Therefore
// the [0, N) output of every parent depends only on [0, N) of either child;
// evaluating consecutive prefixes is exactly the same comparison network as
// evaluating [0, K) in one call, including stable equal-distance tie order.
}  // namespace gpu_search::persistent_kernel_detail

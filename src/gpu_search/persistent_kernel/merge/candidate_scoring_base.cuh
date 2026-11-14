#pragma once

#include "gpu_search/persistent_kernel/context.cuh"

namespace gpu_search::persistent_kernel_detail {

struct CandidateWorkspaceArrays {
  u64 handles[kPersistentMaxExact * 2];
  f32 distances[kPersistentMaxExact * 2];
  // Wider than the on-Beam u8 expanded bitmap: preview/certificate stages
  // store old-Beam ordinals (up to beam_capacity-1) and inclusive expanded
  // prefix counts (up to beam_count) here, both of which need 9 bits at
  // beam 256. The candidate-source sentinel is UINT16_MAX, so no ordinal can
  // alias it.
  u16 expanded[kPersistentMaxExact * 2];
  u16 old_expanded_prefix[kPersistentMaxBeam + 1];
};

static_assert(kPersistentMaxExact >= kPersistentMaxBeam);

union CandidateSortWorkspace {
  // Exchange buffer for the custom block top-k reduction in
  // stable_sort_candidate_run: one sorted 128-entry warp run per merge step,
  // or a full 256-entry run pair when beam_capacity > 128.
  u64 stable_run_exchange[2 * kApproximateSortThreadsCompact];
#if DVSTOR_PERSISTENT_WIDE_SORT
  ApproximateBlockSortWide::TempStorage radix_sort_wide;
  ApproximateBlockSortWideRun::TempStorage radix_sort_wide_run;
#endif
#if DVSTOR_PERSISTENT_LEGACY_MERGE
  ApproximateBlockSortCompactPass::TempStorage radix_sort_compact_pass;
  ApproximateBlockSortCompactFinal::TempStorage radix_sort_compact_final;
  ApproximateBlockSortCompactFinal256::TempStorage radix_sort_compact_final_256;
#endif
};

struct CandidateWorkspace {
  CandidateWorkspaceArrays arrays;
  CandidateSortWorkspace sort;
};

inline constexpr u32 kMergeFlagExpanded = 1u;

struct BeamMergeCycleBreakdown {
  // One stable-run call writes a call-local breakdown.  The query keeps a
  // separate instance and explicitly accumulates these fields across rounds;
  // legacy leaves them untouched.
  u64 prepare{};
  u64 sort{};
  u64 materialize{};
};

constexpr u32 kGraphScratchBit = 0x80000000u;
constexpr u64 kNodeLockMask = 1ull;
constexpr u64 kNodeDeletedMask = 1ull << 24;
constexpr u32 kNodeIdOffset = 8;
constexpr f32 kDeviceMaxValidSquaredL2 = 0x1.fffffcp+127f;

__device__ __forceinline__ bool finite_f32_bits(f32 value) {
  return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ f32 saturate_device_squared_l2(double value) {
  return value >= static_cast<double>(kDeviceMaxValidSquaredL2)
           ? kDeviceMaxValidSquaredL2
           : static_cast<f32>(value);
}

__device__ __forceinline__ f32 saturate_device_component(double value) {
  constexpr double maximum = static_cast<double>(FLT_MAX);
  if (value >= maximum) return FLT_MAX;
  if (value <= -maximum) return -FLT_MAX;
  return static_cast<f32>(value);
}

__device__ __forceinline__ u32 hash32(u32 value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ u32 hash64(u64 value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return static_cast<u32>(value ^ (value >> 32));
}

__device__ __forceinline__ u64 load_cg(const u64* address) {
  u64 value = 0;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ u32 load_cg(const u32* address) {
  u32 value = 0;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ u64 global_time_ns() {
  u64 value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ f32 query_component(const u8* query, u8 dtype,
                                               u32 index) {
  switch (dtype) {
    case 0:
      return reinterpret_cast<const f32*>(query)[index];
    case 1:
      return static_cast<f32>(query[index]);
    case 2:
      return static_cast<f32>(
        reinterpret_cast<const std::int8_t*>(query)[index]);
    default:
      return 0.0f;
  }
}

#ifdef DVSTOR_HAVE_GPUNETIO
__device__ i32 poll_direct_cq(doca_gpu_dev_verbs_cq* completion_queue,
                              doca_gpu_dev_verbs_ticket_t ticket,
                              u64 timeout_ns, const u32* stop,
                              const u32* direct_disabled) {
  auto* completion_base = reinterpret_cast<mlx5_cqe64*>(
    __ldg(reinterpret_cast<uintptr_t*>(&completion_queue->cqe_daddr)));
  const u32 completion_count = __ldg(&completion_queue->cqe_num);
  const u32 completion_index = ticket & (completion_count - 1);
  auto* completion = completion_base + completion_index;
  const u64 started = global_time_ns();
  for (;;) {
    const u64 consumer = doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      &completion_queue->cqe_ci);
    const u8 owner = doca_gpu_dev_verbs_load_relaxed_sys_global(
      reinterpret_cast<u8*>(&completion->op_own));
    if (!((consumer <= ticket) &&
          ((owner & MLX5_CQE_OWNER_MASK) ^ !!(ticket & completion_count)))) {
      const u8 opcode = owner >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
      const i32 status = opcode == MLX5_CQE_REQ ? 0 : -EIO;
      doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      // Advancing only the device-side consumer index is insufficient: the
      // NIC uses the CQ doorbell record to decide whether a wrapped CQE slot
      // has been reclaimed. Without this commit every QP eventually stops
      // producing CQEs after enough successful submissions. One exclusive
      // owner consumes this CQ, so exactly one CQE is committed here. Error
      // CQEs are consumed as well before the engine enters fail-stop mode.
      doca_gpu_dev_verbs_cq_update_dbrec<false>(completion_queue, 1);
      return status;
    }
    if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
        *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
      return -ECANCELED;
    }
    if (global_time_ns() - started >= timeout_ns) return -ETIMEDOUT;
    device_ring_relax(128);
  }
}

__device__ i32 lock_direct_qp(i32* lock, const u32* stop,
                              const u32* direct_disabled) {
  if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
      *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
    return -ECANCELED;
  }
  while (atomicCAS(lock, 0, 1) != 0) {
    if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
        *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
      return -ECANCELED;
    }
    device_ring_relax(128);
  }
  if (*reinterpret_cast<const volatile u32*>(stop) != 0 ||
      *reinterpret_cast<const volatile u32*>(direct_disabled) != 0) {
    atomicExch(lock, 0);
    return -ECANCELED;
  }
  __threadfence();
  return 0;
}

__device__ void unlock_direct_qp(i32* lock) {
  __threadfence();
  atomicExch(lock, 0);
}
#endif

__device__ bool insert_visited(u64* table, u32 capacity, u64 handle) {
  const u32 mask = capacity - 1;
  u32 slot = hash64(handle) & mask;
  for (u32 probe = 0; probe < capacity; ++probe) {
    const u64 old =
      atomicCAS(reinterpret_cast<unsigned long long*>(table + slot),
                static_cast<unsigned long long>(kInvalidDeviceHandle),
                static_cast<unsigned long long>(handle));
    if (old == kInvalidDeviceHandle) return true;
    if (old == handle) return false;
    slot = (slot + 1) & mask;
  }
  return false;
}

__device__ const DeviceShardRegion* shard_for_ordinal(
  const PersistentKernelParams& params, u32 ordinal, u64* slot_out = nullptr) {
  for (u32 shard = 0; shard < params.num_shards; ++shard) {
    const DeviceShardRegion& region = params.shards[shard];
    if (ordinal >= region.ordinal_base &&
        static_cast<u64>(ordinal) - region.ordinal_base < region.node_count) {
      if (slot_out != nullptr)
        *slot_out = static_cast<u64>(ordinal) - region.ordinal_base;
      return params.shards + shard;
    }
  }
  return nullptr;
}

__device__ __forceinline__ u32 remote_shard(u64 raw) {
  return static_cast<u32>((raw >> kRemoteOffsetUnitBits) & kRemoteShardMask);
}

__device__ __forceinline__ u64 remote_byte_offset(u64 raw) {
  return (raw & kRemoteOffsetUnitMask) << 4;
}

__device__ __forceinline__ u32 remote_incarnation(u64 raw) {
  return static_cast<u32>(raw >> kRemoteIncarnationShift);
}

__device__ __forceinline__ u64 make_remote_raw(u32 shard, u64 offset,
                                               u32 incarnation) {
  return (static_cast<u64>(incarnation) << kRemoteIncarnationShift) |
         (static_cast<u64>(shard) << kRemoteOffsetUnitBits) | (offset >> 4);
}

__device__ bool static_ordinal_from_raw(const PersistentKernelParams& params,
                                        u64 raw, u32& ordinal) {
  if (raw == 0 || remote_incarnation(raw) != 0) return false;
  const u32 shard = remote_shard(raw);
  const u64 offset = remote_byte_offset(raw);
  if (shard >= params.num_shards) return false;
  const DeviceShardRegion& region = params.shards[shard];
  if (offset < region.node_base_offset || region.node_stride == 0) return false;
  const u64 relative = offset - region.node_base_offset;
  if (relative % region.node_stride != 0) return false;
  const u64 slot = relative / region.node_stride;
  if (slot >= region.node_count) return false;
  ordinal = static_cast<u32>(region.ordinal_base + slot);
  return true;
}

__device__ u64 handle_from_raw(const PersistentKernelParams& params, u64 raw) {
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, raw, ordinal)) return raw;
  const u32 shard = remote_shard(raw);
  const u64 offset = remote_byte_offset(raw);
  const u32 incarnation = remote_incarnation(raw);
  if (raw == 0 || raw == kInvalidDeviceHandle || incarnation == 0 ||
      incarnation > kRemoteMaxIncarnation || shard >= params.num_shards) {
    return kInvalidDeviceHandle;
  }
  const DeviceShardRegion& region = params.shards[shard];
  if (!dynamic_record_range_from_offset(
        region, offset, region.dynamic_hot_offset, params.graph_entry_bytes)) {
    return kInvalidDeviceHandle;
  }
  return raw;
}

__device__ bool resolve_handle(const PersistentKernelParams& params, u64 handle,
                               u64& raw, u32& shard, u64& graph_offset) {
  raw = handle;
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, raw, ordinal)) {
    u64 slot = 0;
    const DeviceShardRegion* region = shard_for_ordinal(params, ordinal, &slot);
    if (region == nullptr) return false;
    shard = region->memory_node;
    graph_offset = region->graph_base_offset + slot * params.graph_entry_bytes;
    return true;
  }
  if (handle_from_raw(params, raw) == kInvalidDeviceHandle) return false;
  shard = remote_shard(raw);
  const DeviceShardRegion& region = params.shards[shard];
  graph_offset = remote_byte_offset(raw) + region.dynamic_hot_offset;
  return true;
}

__device__ __forceinline__ f32 accumulate_packed_pq_word(
    const f32* query_lut, u32 subquantizer, u32 packed, f32 distance) {
  // Keep the four additions in scalar subquantizer order. Besides preserving
  // the existing distance/tie contract, this lets one naturally aligned load
  // serve four independent LUT lookups.
  distance +=
    query_lut[static_cast<size_t>(subquantizer) * 256u + (packed & 0xffu)];
  distance += query_lut[static_cast<size_t>(subquantizer + 1u) * 256u +
                        ((packed >> 8u) & 0xffu)];
  distance += query_lut[static_cast<size_t>(subquantizer + 2u) * 256u +
                        ((packed >> 16u) & 0xffu)];
  distance += query_lut[static_cast<size_t>(subquantizer + 3u) * 256u +
                        (packed >> 24u)];
  return distance;
}

template <u32 Subquantizers>
__device__ __forceinline__ f32 approximate_entry_aligned_packed(
    const f32* query_lut, const u8* code) {
  static_assert(Subquantizers >= 4);
  constexpr u32 kPackedWords = Subquantizers / 4u;
  constexpr u32 kTail = Subquantizers % 4u;
  const u32* packed_code = reinterpret_cast<const u32*>(code);
  f32 distance = 0.0f;
#pragma unroll
  for (u32 word = 0; word < kPackedWords; ++word) {
    distance = accumulate_packed_pq_word(
      query_lut, word * 4u, packed_code[word], distance);
  }
  if constexpr (kTail != 0) {
#pragma unroll
    for (u32 tail = 0; tail < kTail; ++tail) {
      const u32 subquantizer = kPackedWords * 4u + tail;
      distance += query_lut[static_cast<size_t>(subquantizer) * 256u +
                            code[subquantizer]];
    }
  }
  return distance;
}

template <u32 Subquantizers>
__device__ __forceinline__ f32 approximate_entry_mixed_alignment_packed(
    const f32* query_lut, const u8* code) {
  static_assert(Subquantizers >= 4);
  // PQ25 codes are tightly packed: consecutive immutable and arena entries
  // rotate through all four byte alignments. Consume at most three leading
  // bytes to reach a natural u32 boundary, score the aligned body four codes
  // per load, then consume at most three trailing bytes. This avoids both
  // undefined/misaddressed unaligned u32 reads and padding every 100M-vector
  // code entry solely for the GPU fast path.
  const u32 misalignment =
    static_cast<u32>(reinterpret_cast<uintptr_t>(code) & (alignof(u32) - 1u));
  const u32 prefix = misalignment == 0 ? 0u : alignof(u32) - misalignment;
  f32 distance = 0.0f;
#pragma unroll
  for (u32 index = 0; index < alignof(u32) - 1u; ++index) {
    if (index < prefix) {
      distance += query_lut[static_cast<size_t>(index) * 256u + code[index]];
    }
  }

  const u32 packed_words = (Subquantizers - prefix) / 4u;
  const u32* packed_code = reinterpret_cast<const u32*>(code + prefix);
#pragma unroll
  for (u32 word = 0; word < Subquantizers / 4u; ++word) {
    if (word < packed_words) {
      distance = accumulate_packed_pq_word(
        query_lut, prefix + word * 4u, packed_code[word], distance);
    }
  }

  const u32 tail_begin = prefix + packed_words * 4u;
#pragma unroll
  for (u32 tail = 0; tail < alignof(u32) - 1u; ++tail) {
    const u32 subquantizer = tail_begin + tail;
    if (subquantizer < Subquantizers) {
      distance += query_lut[static_cast<size_t>(subquantizer) * 256u +
                            code[subquantizer]];
    }
  }
  return distance;
}

template <u32 PqSubquantizers>
__device__ __forceinline__ u32 resolved_pq_subquantizers(
    const PersistentKernelParams& params) {
  static_assert(
    PqSubquantizers == kPersistentRuntimePqSubquantizers ||
    PqSubquantizers == kPersistentPq20Subquantizers ||
    PqSubquantizers == kPersistentPq25Subquantizers ||
    PqSubquantizers == kPersistentPq32Subquantizers ||
    PqSubquantizers == kPersistentPq64Subquantizers ||
    PqSubquantizers == kPersistentPq128Subquantizers ||
    PqSubquantizers == kPersistentPq256Subquantizers);
  if constexpr (PqSubquantizers == kPersistentRuntimePqSubquantizers) {
    return params.pq_subquantizers;
  }
  return PqSubquantizers;
}

__device__ __noinline__ f32 approximate_entry_wide_fallback(
    const f32* query_lut, const u8* code, u32 subquantizers) {
  double wide_distance = 0.0;
  for (u32 subquantizer = 0; subquantizer < subquantizers; ++subquantizer) {
    wide_distance += static_cast<double>(
      query_lut[static_cast<size_t>(subquantizer) * 256u +
                code[subquantizer]]);
  }
  return saturate_device_squared_l2(wide_distance);
}

template <u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__device__ __forceinline__ f32 approximate_entry(
    const PersistentKernelParams& params, const f32* query_lut,
    const u8* code) {
  const u32 subquantizers =
    resolved_pq_subquantizers<PqSubquantizers>(params);
  f32 distance = 0.0f;
  if constexpr (PqSubquantizers == kPersistentPq20Subquantizers ||
                PqSubquantizers == kPersistentPq32Subquantizers ||
                PqSubquantizers == kPersistentPq64Subquantizers ||
                PqSubquantizers == kPersistentPq128Subquantizers ||
                PqSubquantizers == kPersistentPq256Subquantizers) {
    // cudaMalloc bases, the dynamic arena, and the RDMA scratch records are
    // naturally aligned. These strides are multiples of four bytes and
    // preserve that alignment, and the RDMA payload begins after a four-byte
    // aligned incarnation tag. Removing the redundant per-candidate
    // alignment/PQ dispatch is the purpose of the production
    // specializations.
    distance = approximate_entry_aligned_packed<PqSubquantizers>(
      query_lut, code);
  } else if constexpr (PqSubquantizers ==
                       kPersistentPq25Subquantizers) {
    const bool aligned =
      reinterpret_cast<uintptr_t>(code) % alignof(u32) == 0;
    distance = aligned
      ? approximate_entry_aligned_packed<kPersistentPq25Subquantizers>(
          query_lut, code)
      : approximate_entry_mixed_alignment_packed<
          kPersistentPq25Subquantizers>(query_lut, code);
  } else {
    for (u32 subquantizer = 0; subquantizer < subquantizers; ++subquantizer) {
      distance +=
        query_lut[static_cast<size_t>(subquantizer) * 256u +
                  code[subquantizer]];
    }
  }
  if (finite_f32_bits(distance) && distance < FLT_MAX) return distance;
  // Keep the exceptional non-finite recovery out of every production hot
  // path. A single runtime-sized noinline body avoids triplicating/unrolling
  // the double-precision loop in PQ20/PQ25/PQ32 kernels.
  return approximate_entry_wide_fallback(
    query_lut, code, subquantizers);
}

// Score every valid static-code candidate in handles[0, count) with the
// per-query LUT staged tile by tile in shared memory.  Each tile is one
// contiguous kPersistentPqLutTileSubquantizers x 256 x f32 slice of the
// global LUT.  The next tile's global loads are issued into registers one
// tile ahead so their latency hides behind the current tile's scoring; the
// barriers then only publish shared-memory stores.
//
// Per-candidate partial sums live in registers across the tile loop and are
// published to distances[] once at the end; invalid candidates keep the
// caller's FLT_MAX sentinel untouched.  Every candidate's contributions are
// appended to one f32 accumulator in ascending subquantizer order, which is
// the identical operation sequence (and therefore bitwise result) of the
// scalar per-candidate loop; the final publish applies approximate_entry's
// exceptional-path fallback under the same condition.  lut_tile must cover
// kPersistentPqLutTileBytes of shared memory.
//
// Contains CTA-wide barriers: every thread of the block must execute this
// with the same uniform arguments.
template <u32 PqSubquantizers = kPersistentRuntimePqSubquantizers>
__device__ void score_static_candidates_lut_tiled(
    const PersistentKernelParams& params, const f32* query_lut,
    const u64* handles, u32 count, f32* distances, f32* lut_tile) {
  const u32 subquantizers = resolved_pq_subquantizers<PqSubquantizers>(params);
  constexpr u32 kTile = kPersistentPqLutTileSubquantizers;
  // Slot space covers the 128-thread production CTA; a 256-thread ablation
  // build simply leaves the upper half unused.
  constexpr u32 kMaxSlots =
    kPersistentMaxMergeCandidates / kApproximateSortThreadsCompact;
  // Resolve the per-slot validity and code origin once.  Keeping it in
  // registers avoids re-reading the shard table on every tile.
  u32 ordinals[kMaxSlots];
  u32 static_mask = 0;
#pragma unroll
  for (u32 slot = 0; slot < kMaxSlots; ++slot) {
    const u32 index = threadIdx.x + slot * blockDim.x;
    ordinals[slot] = UINT32_MAX;
    u32 ordinal = 0;
    if (index < count &&
        static_ordinal_from_raw(params, handles[index], ordinal) &&
        ordinal < params.num_nodes) {
      ordinals[slot] = ordinal;
      static_mask |= 1u << slot;
    }
  }
  // A dynamic-only wave must not pay for staging the complete LUT.
  if (__syncthreads_or(static_mask != 0 ? 1 : 0) == 0) return;
  f32 partials[kMaxSlots];
  // Each thread stages the same one or two float4 words of every tile.  The
  // second word exists only for the 128-thread production CTA (each tile has
  // 256 f32 words); a 256-thread ablation build covers the tile with word0.
  const u32 word0 = threadIdx.x;
  const u32 word1 = blockDim.x == kApproximateSortThreadsCompact
                      ? threadIdx.x + kApproximateSortThreadsCompact
                      : UINT32_MAX;
  float4 prefetch0;
  float4 prefetch1;
  auto issue_tile_loads = [&](u32 tile_base, u32 tile_rows) {
    const f32* source = query_lut + static_cast<size_t>(tile_base) * 256u;
    if (word0 < tile_rows * 64u) {
      prefetch0 = reinterpret_cast<const float4*>(source)[word0];
    }
    if (word1 < tile_rows * 64u) {
      prefetch1 = reinterpret_cast<const float4*>(source)[word1];
    }
  };
  issue_tile_loads(0, min(kTile, subquantizers));
  for (u32 tile_base = 0; tile_base < subquantizers; tile_base += kTile) {
    const u32 tile_rows = min(kTile, subquantizers - tile_base);
    if (tile_base != 0) __syncthreads();
    if (word0 < tile_rows * 64u) {
      reinterpret_cast<float4*>(lut_tile)[word0] = prefetch0;
    }
    if (word1 < tile_rows * 64u) {
      reinterpret_cast<float4*>(lut_tile)[word1] = prefetch1;
    }
    __syncthreads();
    // Kick off the next tile's global loads now so they overlap this tile's
    // scoring; the registers are consumed by the store at the top of the
    // next iteration.
    if (tile_base + kTile < subquantizers) {
      issue_tile_loads(tile_base + kTile,
                       min(kTile, subquantizers - tile_base - kTile));
    }
#pragma unroll
    for (u32 slot = 0; slot < kMaxSlots; ++slot) {
      if ((static_mask & (1u << slot)) == 0) continue;
      const u8* code = params.pq_codes +
                       static_cast<size_t>(ordinals[slot]) * subquantizers;
      f32 distance = tile_base == 0 ? 0.0f : partials[slot];
      if (tile_rows == kTile &&
          (reinterpret_cast<uintptr_t>(code) & (alignof(u32) - 1u)) == 0) {
        distance = accumulate_packed_pq_word(
          lut_tile, 0, *reinterpret_cast<const u32*>(code + tile_base),
          distance);
      } else {
        for (u32 row = 0; row < tile_rows; ++row) {
          distance += lut_tile[static_cast<size_t>(row) * 256u +
                               code[tile_base + row]];
        }
      }
      partials[slot] = distance;
    }
  }
#pragma unroll
  for (u32 slot = 0; slot < kMaxSlots; ++slot) {
    if ((static_mask & (1u << slot)) == 0) continue;
    const u32 index = threadIdx.x + slot * blockDim.x;
    f32 distance = partials[slot];
    if (!(finite_f32_bits(distance) && distance < FLT_MAX)) {
      distance = approximate_entry_wide_fallback(
        query_lut,
        params.pq_codes + static_cast<size_t>(ordinals[slot]) * subquantizers,
        subquantizers);
    }
    distances[index] = distance;
  }
}

__device__ void beam_insert(u64* handles, u32* ids, f32* distances,
                            u8* expanded, u32& count, u32 capacity, u64 handle,
                            u32 id, f32 distance) {
  if (handle == kInvalidDeviceHandle || !isfinite(distance) ||
      distance == FLT_MAX)
    return;
  if (count < capacity) {
    handles[count] = handle;
    ids[count] = id;
    distances[count] = distance;
    expanded[count] = 0;
    ++count;
    return;
  }
  u32 worst = 0;
  for (u32 index = 1; index < count; ++index) {
    if (distances[index] > distances[worst]) worst = index;
  }
  if (distance >= distances[worst]) return;
  handles[worst] = handle;
  ids[worst] = id;
  distances[worst] = distance;
  expanded[worst] = 0;
}

__device__ __forceinline__ bool candidate_less(u64 lhs_handle, f32 lhs_distance,
                                               u64 rhs_handle,
                                               f32 rhs_distance) {
  return lhs_distance < rhs_distance ||
         (lhs_distance == rhs_distance && lhs_handle < rhs_handle);
}

__device__ u32 candidate_sort_capacity(u32 count) {
  u32 capacity = 1;
  while (capacity < count) capacity <<= 1;
  return capacity;
}

template <class Expanded>
__device__ void sort_candidates(u64* handles, u32* ids, f32* distances,
                                Expanded* expanded, u32 count) {
  const u32 capacity = candidate_sort_capacity(max(1u, count));
  for (u32 index = count + threadIdx.x; index < capacity; index += blockDim.x) {
    handles[index] = kInvalidDeviceHandle;
    if (ids != nullptr) ids[index] = UINT32_MAX;
    distances[index] = FLT_MAX;
    expanded[index] = 0;
  }
  __syncthreads();
  for (u32 sequence = 2; sequence <= capacity; sequence <<= 1) {
    for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (u32 index = threadIdx.x; index < capacity; index += blockDim.x) {
        const u32 partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        const bool exchange =
          ascending ? candidate_less(handles[partner], distances[partner],
                                     handles[index], distances[index])
                    : candidate_less(handles[index], distances[index],
                                     handles[partner], distances[partner]);
        if (!exchange) continue;
        const u64 handle = handles[index];
        handles[index] = handles[partner];
        handles[partner] = handle;
        if (ids != nullptr) {
          const u32 id = ids[index];
          ids[index] = ids[partner];
          ids[partner] = id;
        }
        const f32 distance = distances[index];
        distances[index] = distances[partner];
        distances[partner] = distance;
        const Expanded was_expanded = expanded[index];
        expanded[index] = expanded[partner];
        expanded[partner] = was_expanded;
      }
      __syncthreads();
    }
  }
}

#if DVSTOR_PERSISTENT_WIDE_SORT
template <class BlockSort, u32 ItemsPerThread>
__device__ void merge_approximate_radix(
  u64* candidate_handles, f32* candidate_distances, u32 candidate_count,
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u32 existing_count, u32 merge_count,
  typename BlockSort::TempStorage& radix_storage) {
  f32 local_distances[ItemsPerThread];
  u64 local_values[ItemsPerThread];
  u8 sorted_flags[ItemsPerThread];
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (index < existing_count) {
      handle = beam_handles[index];
      distance = beam_distances[index];
    } else if (index < merge_count) {
      const u32 candidate = index - existing_count;
      handle = candidate_handles[candidate];
      distance = candidate_distances[candidate];
    }
    if (handle == kInvalidDeviceHandle || !finite_f32_bits(distance) ||
        distance == FLT_MAX) {
      handle = kInvalidDeviceHandle;
      distance = FLT_MAX;
    }
    local_distances[item] = distance;
    local_values[item] = handle;
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(local_distances, local_values);
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    u32 flags = 0;
    for (u32 prior = 0; prior < existing_count; ++prior) {
      if (beam_handles[prior] == local_values[item]) {
        flags = beam_expanded[prior] != 0 ? kMergeFlagExpanded : 0u;
        break;
      }
    }
    sorted_flags[item] = static_cast<u8>(flags);
  }
  __syncthreads();
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = local_values[item];
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = local_distances[item];
    beam_expanded[output] =
      static_cast<u8>((sorted_flags[item] & kMergeFlagExpanded) != 0 ? 1u : 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    const u32 limit = min(merge_count, beam_capacity);
    while (valid < limit && beam_handles[valid] != kInvalidDeviceHandle &&
           isfinite(beam_distances[valid]) &&
           beam_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}

#endif  // DVSTOR_PERSISTENT_WIDE_SORT

#if DVSTOR_PERSISTENT_LEGACY_MERGE
template <u32 ItemsPerThread, class BlockSort>
__device__ void merge_approximate_compact_final(
  u64* beam_handles, u32* beam_ids, f32* beam_distances, u8* beam_expanded,
  u32& beam_count, u32 beam_capacity, u64* scratch_handles,
  u8* scratch_expanded, f32* scratch_distances,
  typename BlockSort::TempStorage& radix_storage) {
  const u32 scratch_count = beam_capacity * 2;
  f32 final_distances[ItemsPerThread];
  u64 final_values[ItemsPerThread];
  u8 final_flags[ItemsPerThread];
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 index = threadIdx.x * ItemsPerThread + item;
    u64 handle = kInvalidDeviceHandle;
    f32 distance = FLT_MAX;
    if (index < scratch_count) {
      handle = scratch_handles[index];
      distance = scratch_distances[index];
    }
    final_distances[item] = distance;
    final_values[item] = handle;
  }
  __syncthreads();
  BlockSort(radix_storage).Sort(final_distances, final_values);
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    final_flags[item] = 0;
    for (u32 prior = 0; prior < scratch_count; ++prior) {
      if (scratch_handles[prior] == final_values[item]) {
        final_flags[item] = static_cast<u8>(scratch_expanded[prior]);
        break;
      }
    }
  }
  __syncthreads();
  for (u32 item = 0; item < ItemsPerThread; ++item) {
    const u32 output = threadIdx.x * ItemsPerThread + item;
    if (output >= beam_capacity) continue;
    beam_handles[output] = final_values[item];
    beam_ids[output] = UINT32_MAX;
    beam_distances[output] = final_distances[item];
    beam_expanded[output] =
      static_cast<u8>((final_flags[item] & kMergeFlagExpanded) != 0 ? 1u : 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (
      valid < beam_capacity && beam_handles[valid] != kInvalidDeviceHandle &&
      isfinite(beam_distances[valid]) && beam_distances[valid] != FLT_MAX) {
      ++valid;
    }
    beam_count = valid;
  }
  __syncthreads();
}

__device__ void merge_approximate_compact(
  u64* candidate_handles, f32* candidate_distances, u64* beam_handles,
  u32* beam_ids, f32* beam_distances, u8* beam_expanded, u32& beam_count,
  u32 beam_capacity, u32 existing_count, u32 merge_count, u64* scratch_handles,
  u8* scratch_expanded, f32* scratch_distances, CandidateWorkspace& workspace) {
  constexpr u32 pass_items =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;
  for (u32 pass = 0; pass < 2; ++pass) {
    f32 local_distances[kApproximateSortItemsCompactPass];
    u64 local_values[kApproximateSortItemsCompactPass];
    u8 sorted_flags[kApproximateSortItemsCompactPass];
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 index = pass * pass_items +
                        threadIdx.x * kApproximateSortItemsCompactPass + item;
      u64 handle = kInvalidDeviceHandle;
      f32 distance = FLT_MAX;
      if (index < existing_count) {
        handle = beam_handles[index];
        distance = beam_distances[index];
      } else if (index < merge_count) {
        const u32 candidate = index - existing_count;
        handle = candidate_handles[candidate];
        distance = candidate_distances[candidate];
      }
      if (handle == kInvalidDeviceHandle || !isfinite(distance)) {
        handle = kInvalidDeviceHandle;
        distance = FLT_MAX;
      }
      local_distances[item] = distance;
      local_values[item] = handle;
    }
    __syncthreads();
    ApproximateBlockSortCompactPass(workspace.sort.radix_sort_compact_pass)
      .Sort(local_distances, local_values);
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      sorted_flags[item] = 0;
      for (u32 prior = 0; prior < existing_count; ++prior) {
        if (beam_handles[prior] == local_values[item]) {
          sorted_flags[item] = static_cast<u8>(
            beam_expanded[prior] != 0 ? kMergeFlagExpanded : 0u);
          break;
        }
      }
    }
    __syncthreads();
    for (u32 item = 0; item < kApproximateSortItemsCompactPass; ++item) {
      const u32 output = threadIdx.x * kApproximateSortItemsCompactPass + item;
      if (output >= beam_capacity) continue;
      const u32 destination = pass * beam_capacity + output;
      scratch_handles[destination] = local_values[item];
      scratch_expanded[destination] = sorted_flags[item];
      scratch_distances[destination] = local_distances[item];
    }
    __syncthreads();
  }

  constexpr u32 compact_final_capacity =
    kApproximateSortThreadsCompact * kApproximateSortItemsCompactFinal;
  if (beam_capacity * 2 <= compact_final_capacity) {
    merge_approximate_compact_final<kApproximateSortItemsCompactFinal,
                                    ApproximateBlockSortCompactFinal>(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, scratch_handles, scratch_expanded, scratch_distances,
      workspace.sort.radix_sort_compact_final);
  } else {
    merge_approximate_compact_final<kApproximateSortItemsCompactFinal256,
                                    ApproximateBlockSortCompactFinal256>(
      beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
      beam_capacity, scratch_handles, scratch_expanded, scratch_distances,
      workspace.sort.radix_sort_compact_final_256);
  }
}
#endif  // DVSTOR_PERSISTENT_LEGACY_MERGE

}  // namespace gpu_search::persistent_kernel_detail

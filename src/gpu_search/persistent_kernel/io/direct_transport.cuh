#pragma once

#include <cuda/atomic>

#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/persistent_kernel/merge/candidate_scoring.cuh"
#include "vamana/dynamic_navigation_code.hh"

namespace gpu_search::persistent_kernel_detail {

__device__ __forceinline__ u32 dynamic_arena_state_load(const u32* state) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(
    *const_cast<u32*>(state));
  return reference.load(cuda::memory_order_acquire);
}

// Match atomicCAS's return-value convention while making the publication
// order explicit.  Successful reservations/extent transitions are acq_rel;
// failure returns an acquire observation for the caller's retry loop.
__device__ __forceinline__ u32
dynamic_arena_state_compare_exchange(u32* state, u32 expected, u32 desired) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(*state);
  u32 observed = expected;
  (void)reference.compare_exchange_strong(
    observed, desired, cuda::memory_order_acq_rel, cuda::memory_order_acquire);
  return observed;
}

__device__ __forceinline__ void dynamic_arena_state_publish(u32* state,
                                                            u32 desired) {
  cuda::atomic_ref<u32, cuda::thread_scope_device> reference(*state);
  reference.store(desired, cuda::memory_order_release);
}

struct GraphFetchCycleBreakdown {
  u64 issue{};
  u64 wait{};
  u64 validation{};
  // Final-CQE owner submission-group latency. Keeping this separate from the
  // clock64 phase counters makes the signal available even when tracing is
  // disabled, without pretending that one CQE exposes per-WQE latency.
  u64 completion_latency_ns{};
  u64 completion_groups{};
};

#ifdef DVSTOR_HAVE_GPUNETIO
__device__ __forceinline__ void record_owner_watchdog_counter(
  unsigned long long* counter) {
  if (counter == nullptr) return;
  atomicAdd(counter, 1ULL);
}
#endif

__device__ __forceinline__ f32 storage_component(
  const PersistentKernelParams& params, const u8* vector, u32 dimension) {
  if (params.vector_dtype == 0)
    return reinterpret_cast<const f32*>(vector)[dimension];
  if (params.vector_dtype == 1) return static_cast<f32>(vector[dimension]);
  return static_cast<f32>(reinterpret_cast<const int8_t*>(vector)[dimension]);
}

__device__ f32 exact_storage_distance(const PersistentKernelParams& params,
                                      const f32* query, const u8* vector) {
  f32 distance = 0.0f;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const f32 component = storage_component(params, vector, dimension);
    const f32 difference = query[dimension] - component;
    distance += difference * difference;
  }
  if (finite_f32_bits(distance) && distance < FLT_MAX) return distance;
  double wide_distance = 0.0;
  for (u32 dimension = 0; dimension < params.dim; ++dimension) {
    const double component =
      static_cast<double>(storage_component(params, vector, dimension));
    const double difference = static_cast<double>(query[dimension]) - component;
    wide_distance += difference * difference;
  }
  return saturate_device_squared_l2(wide_distance);
}

__device__ i32 direct_fetch(const PersistentKernelParams& params,
                            u32 memory_node, u64 remote_offset, u8* destination,
                            u32 bytes, u32 lane) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (memory_node >= params.direct_region_count ||
      params.direct_qps == nullptr || params.direct_regions == nullptr ||
      params.direct_qp_locks == nullptr ||
      params.direct_qps_per_node == 0 || params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0)
    return -EHOSTDOWN;
  const u32 qp_index =
    (lane % params.direct_qps_per_node) * params.direct_region_count +
    memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  auto* qp =
    reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  if (!direct_remote_range_valid(region.bytes, remote_offset, bytes)) {
    return -ERANGE;
  }
  doca_gpu_dev_verbs_addr remote{.addr = region.address + remote_offset,
                                 .key = region.rkey};
  doca_gpu_dev_verbs_addr local{
    .addr = reinterpret_cast<u64>(destination) - params.direct_local_iova_base,
    .key = params.direct_local_mkey,
  };
  doca_gpu_dev_verbs_addr dump{
    .addr =
      reinterpret_cast<u64>(params.direct_dump) - params.direct_local_iova_base,
    .key = params.direct_local_mkey,
  };
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr)
      atomicCAS(params.direct_error, 0, status);
    if (status != -ECANCELED) atomicExch(params.direct_disabled, 1u);
    return status;
  }
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
    if (params.direct_error != nullptr)
      atomicCAS(params.direct_error, 0, -E2BIG);
    atomicExch(params.direct_disabled, 1u);
    unlock_direct_qp(params.direct_qp_locks + qp_index);
    return -E2BIG;
  }
  const doca_gpu_dev_verbs_ticket_t read_ticket = qp->sq_wqe_pi;
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const doca_gpu_dev_verbs_ticket_t completion_ticket =
    doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      &completion_queue->cqe_ci);
  auto* read_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, read_ticket);
  const bool need_dump = qp->need_dump;
  doca_gpu_dev_verbs_wqe_prepare_read(
    qp, read_wqe, read_ticket,
    need_dump ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE
              : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
    remote.addr, remote.key, local.addr, local.key, bytes);
  doca_gpu_dev_verbs_ticket_t final_ticket = read_ticket;
  if (need_dump) {
    final_ticket = read_ticket + 1;
    auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_ticket);
    doca_gpu_dev_verbs_wqe_prepare_dump(qp, dump_wqe, final_ticket,
                                        DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                        dump.addr, dump.key, 1);
  }
  doca_gpu_dev_verbs_submit<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, final_ticket + 1);
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr)
      atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
  unlock_direct_qp(params.direct_qp_locks + qp_index);
  return status;
#else
  (void)params;
  (void)memory_node;
  (void)remote_offset;
  (void)destination;
  (void)bytes;
  (void)lane;
  return -ENOTSUP;
#endif
}

__device__ i32 direct_fetch_batch(
  const PersistentKernelParams& params, u32 memory_node,
  const u32* request_shards, const u64* remote_offsets, u32 request_count,
  u8* destination_base, u32 destination_stride, u32 bytes, u32 lane,
  const u64* local_iova_offsets = nullptr, i32* owner_completion = nullptr,
  bool defer_owner_wait = false, u32* owner_progress = nullptr,
  u64* owner_completion_timestamp_ns = nullptr,
  const u32* request_bytes = nullptr,
  DirectBatchPriority priority = DirectBatchPriority::critical,
  bool nonblocking_enqueue = false, u32 prepared_matching = UINT32_MAX) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (request_shards == nullptr || remote_offsets == nullptr ||
      request_count == 0 || request_count > UINT16_MAX || bytes == 0 ||
      bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
    return bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? -E2BIG : -EINVAL;
  }
  u32 matching = prepared_matching;
  if (matching == UINT32_MAX) {
    matching = 0;
    for (u32 index = 0; index < request_count; ++index) {
      if (request_shards[index] == memory_node) ++matching;
    }
  }
  if (matching == 0) return 0;
  if (memory_node >= params.direct_region_count ||
      params.direct_qps == nullptr || params.direct_regions == nullptr ||
      params.direct_qp_locks == nullptr ||
      params.direct_qps_per_node == 0 || params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0)
    return -EHOSTDOWN;
  // Narrow-frontier callers already pass a normalized QP lane. Avoid a second
  // runtime integer remainder in every populated shard publication.
  u32 qp_lane =
    prepared_matching == UINT32_MAX ? lane % params.direct_qps_per_node : lane;
  const u32 qp_index = qp_lane * params.direct_region_count + memory_node;
  if (params.direct_qps[qp_index] == nullptr) return -EINVAL;
  // Critical and standalone speculative descriptors use disjoint rings.
  // The exclusive owner always drains the critical ring first and admits at
  // most one bounded speculative descriptor only at an otherwise idle
  // service boundary. This preserves critical WQE/CQ headroom without
  // forcing an early shadow read to wait for an unrelated critical prefix
  // solely to share its doorbell.
  const DeviceRingView<DirectBatchDescriptor>* selected_queues =
    priority == DirectBatchPriority::speculative
      ? params.direct_speculative_batch_queues
      : params.direct_batch_queues;
  if (priority == DirectBatchPriority::speculative &&
      selected_queues == nullptr) {
    if (owner_completion_timestamp_ns != nullptr) {
      *owner_completion_timestamp_ns = 0;
    }
    if (owner_completion != nullptr) {
      atomicExch(owner_completion, -EAGAIN);
    }
    return -EAGAIN;
  }
  if (selected_queues != nullptr && owner_completion != nullptr) {
    if (qp_index >= params.direct_batch_queue_count) return -EINVAL;
    DirectOwnerProgress* watchdog_progress =
      params.direct_owner_progress == nullptr
        ? nullptr
        : params.direct_owner_progress + qp_index;
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 2;
      __threadfence_system();
    }
    if (prepared_matching == UINT32_MAX) {
      atomicExch(owner_completion, -EINPROGRESS);
    } else {
      // The narrow frontier wave has exclusive ownership of this completion
      // word until its batch is drained. Its caller also warp-synchronizes the
      // request SoA producers before entering this helper. The ring's
      // system-scope release below therefore publishes both that transitive
      // metadata and this initialization before an owner can consume the
      // descriptor, so a contended RMW and a separate fence are unnecessary.
      *reinterpret_cast<volatile i32*>(owner_completion) = -EINPROGRESS;
    }
    if (owner_completion_timestamp_ns != nullptr) {
      *owner_completion_timestamp_ns = 0;
    }
    if (prepared_matching == UINT32_MAX) {
      __threadfence();
    }
    const DirectBatchDescriptor descriptor{
      .request_shards = request_shards,
      .remote_offsets = remote_offsets,
      .local_iova_offsets = local_iova_offsets,
      .request_bytes = request_bytes,
      .completion_status = owner_completion,
      .completion_timestamp_ns = owner_completion_timestamp_ns,
      .request_count = static_cast<u16>(request_count),
      .memory_node = static_cast<u16>(memory_node),
      .bytes = bytes,
      .priority = static_cast<u8>(priority),
    };
    // Announce before attempting the bounded enqueue. This also covers an
    // owner that stopped before dequeueing enough entries to free a ring slot.
    // Cancellation before publication balances the monotonic counters below.
    if (watchdog_progress != nullptr) {
      record_owner_watchdog_counter(&watchdog_progress->announced);
    }
    bool pushed = device_ring_try_push(selected_queues[qp_index], descriptor);
    if (!pushed && nonblocking_enqueue) {
      atomicExch(owner_completion, -EAGAIN);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -EAGAIN;
    }
    while (!pushed) {
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
        atomicExch(owner_completion, -ECANCELED);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -ECANCELED;
      }
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        atomicExch(owner_completion, -EHOSTDOWN);
        if (watchdog_progress != nullptr) {
          record_owner_watchdog_counter(&watchdog_progress->completed);
        }
        return -EHOSTDOWN;
      }
      // A full descriptor ring is bounded backpressure, not evidence that the
      // QP has failed.  The owner warp is the sole WQE/CQ authority and its CQ
      // watchdog below is responsible for declaring a transport failure.
      device_ring_relax(128);
      pushed = device_ring_try_push(selected_queues[qp_index], descriptor);
    }
    if (owner_progress != nullptr) {
      *reinterpret_cast<volatile u32*>(owner_progress) = 3;
      __threadfence_system();
    }
    if (defer_owner_wait) return -EINPROGRESS;
    for (;;) {
      const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
      if (status != -EINPROGRESS) return status;
      if (*reinterpret_cast<const volatile u32*>(params.stop) != 0)
        return -ECANCELED;
      if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
        return -EHOSTDOWN;
      }
      // Waiting behind already admitted bounded work is normal under load.
      // Only the owner that posts the WQE may turn a missing/error CQE into a
      // global fail-stop transition.
      device_ring_relax(128);
    }
  }
  auto* qp =
    reinterpret_cast<doca_gpu_dev_verbs_qp*>(params.direct_qps[qp_index]);
  if (bytes == 0) return -EINVAL;
  if (bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ||
      matching + (qp->need_dump ? 1u : 0u) > qp->sq_wqe_num)
    return -E2BIG;
  // Queued persistent batches are range-checked by the exclusive owner while
  // it counts the descriptor's matching requests.  Only the inline fallback
  // reaches this point, so validate here before it directly builds any WQE.
  // Keeping this below the owner-queue return avoids making every producer
  // rescan a descriptor that the owner must validate authoritatively anyway.
  const DirectRemoteRegion& region = params.direct_regions[memory_node];
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    const u32 request_length =
      request_bytes == nullptr ? bytes : request_bytes[index];
    if (request_length == 0 || request_length > bytes ||
        request_length > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      return request_length > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? -E2BIG
                                                                    : -EINVAL;
    }
    if (!direct_remote_range_valid(
          region.bytes, remote_offsets[index], request_length)) {
      return -ERANGE;
    }
  }
  i32 status = lock_direct_qp(params.direct_qp_locks + qp_index, params.stop,
                              params.direct_disabled);
  if (status != 0) return status;
  auto* completion_queue = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  const doca_gpu_dev_verbs_ticket_t completion_ticket =
    doca_gpu_dev_verbs_load_relaxed<
      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      &completion_queue->cqe_ci);
  const doca_gpu_dev_verbs_ticket_t first_wqe = qp->sq_wqe_pi;
  u32 posted = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    const doca_gpu_dev_verbs_ticket_t ticket = first_wqe + posted;
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
    const bool last_read = posted + 1 == matching;
    const auto flags = !qp->need_dump && last_read
                         ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
                         : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    const u64 local_iova = local_iova_offsets != nullptr
                             ? local_iova_offsets[index]
                             : reinterpret_cast<u64>(destination_base) +
                                 static_cast<u64>(index) * destination_stride -
                                 params.direct_local_iova_base;
    doca_gpu_dev_verbs_wqe_prepare_read(
      qp, wqe, ticket, flags, region.address + remote_offsets[index],
      region.rkey, local_iova, params.direct_local_mkey,
      request_bytes == nullptr ? bytes : request_bytes[index]);
    ++posted;
  }
  doca_gpu_dev_verbs_ticket_t final_wqe = first_wqe + posted - 1;
  if (qp->need_dump) {
    final_wqe = first_wqe + posted;
    auto* dump_wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, final_wqe);
    doca_gpu_dev_verbs_wqe_prepare_dump(
      qp, dump_wqe, final_wqe, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
      reinterpret_cast<u64>(params.direct_dump) - params.direct_local_iova_base,
      params.direct_local_mkey, 1);
  }
  doca_gpu_dev_verbs_submit<
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, final_wqe + 1);
  status = poll_direct_cq(completion_queue, completion_ticket,
                          params.direct_timeout_ns, params.stop,
                          params.direct_disabled);
  if (status != 0) {
    if (params.direct_error != nullptr)
      atomicCAS(params.direct_error, 0, status);
    atomicExch(params.direct_disabled, 1u);
  }
  unlock_direct_qp(params.direct_qp_locks + qp_index);
  return status;
#else
  (void)params;
  (void)memory_node;
  (void)request_shards;
  (void)remote_offsets;
  (void)request_count;
  (void)destination_base;
  (void)destination_stride;
  (void)bytes;
  (void)lane;
  (void)local_iova_offsets;
  (void)owner_completion;
  (void)owner_progress;
  (void)owner_completion_timestamp_ns;
  (void)request_bytes;
  (void)priority;
  (void)nonblocking_enqueue;
  (void)prepared_matching;
  return -ENOTSUP;
#endif
}

// Publish a critical prefix and speculative suffix with one query-side ring
// transaction. The exclusive owner always places the prefix first. When the
// critical queue is drained and SQ credit remains, it posts
// [critical][CQ fence][tail][CQ fence] with one doorbell; otherwise the tail
// is rejected fail-soft without entering another queue. This removes the
// second query-side system fence/ring publication while preserving separate
// completion and validation lifetimes.
__device__ i32 direct_fetch_split_batch(
  const PersistentKernelParams& params, u32 memory_node,
  const u32* request_shards, const u64* remote_offsets, u32 request_count,
  u32 critical_request_count, u32 bytes, u32 lane,
  const u64* local_iova_offsets, i32* critical_completion,
  u64* critical_completion_timestamp_ns, i32* speculative_completion,
  u64* speculative_completion_timestamp_ns, const u32* request_bytes,
  u8 descriptor_flags = 0) {
#ifdef DVSTOR_HAVE_GPUNETIO
  const bool exact_snapshot_tail =
    (descriptor_flags & kDirectBatchFlagMandatoryFencedTail) != 0;
  const bool mixed_mandatory_tail =
    (descriptor_flags & kDirectBatchFlagMixedMandatoryFencedTail) != 0;
  const bool mandatory_fenced_tail =
    exact_snapshot_tail || mixed_mandatory_tail;
  if (critical_request_count == 0 || critical_request_count > request_count ||
      (descriptor_flags & ~kDirectBatchKnownFlags) != 0 ||
      (exact_snapshot_tail && mixed_mandatory_tail) ||
      (exact_snapshot_tail && request_count != 2u * critical_request_count) ||
      (mixed_mandatory_tail && request_count < 2u * critical_request_count) ||
      (mandatory_fenced_tail && request_bytes != nullptr) ||
      request_shards == nullptr || remote_offsets == nullptr ||
      local_iova_offsets == nullptr || critical_completion == nullptr ||
      (!mandatory_fenced_tail && speculative_completion == nullptr) ||
      (mandatory_fenced_tail && speculative_completion != nullptr) ||
      params.direct_batch_queues == nullptr ||
      memory_node >= params.direct_region_count ||
      params.direct_qps == nullptr || params.direct_qps_per_node == 0 ||
      params.direct_disabled == nullptr ||
      *reinterpret_cast<volatile u32*>(params.direct_disabled) != 0) {
    return -EINVAL;
  }
  if (mandatory_fenced_tail) {
    for (u32 index = 0; index < critical_request_count; ++index) {
      if (request_shards[index] !=
            request_shards[critical_request_count + index] ||
          remote_offsets[index] !=
            remote_offsets[critical_request_count + index] ||
          !exact_snapshot_local_layout_matches(
            local_iova_offsets[index],
            local_iova_offsets[critical_request_count + index], bytes)) {
        return -EINVAL;
      }
    }
  }
  u32 critical_matching = 0;
  u32 speculative_matching = 0;
  for (u32 index = 0; index < request_count; ++index) {
    if (request_shards[index] != memory_node) continue;
    if (index < critical_request_count) {
      ++critical_matching;
    } else {
      ++speculative_matching;
    }
  }
  if (critical_matching == 0) {
    if (speculative_matching == 0) return 0;
    if (mandatory_fenced_tail) return -EINVAL;
    // This shard has no critical WQE train whose spare SQ credit can be
    // stolen, so publishing its tail would require a marginal doorbell.
    return direct_fetch_batch(
      params, memory_node, request_shards + critical_request_count,
      remote_offsets + critical_request_count,
      request_count - critical_request_count, params.graph_scratch,
      kPersistentGraphReadBytes, bytes, lane,
      local_iova_offsets + critical_request_count, speculative_completion, true,
      nullptr, speculative_completion_timestamp_ns,
      request_bytes == nullptr ? nullptr
                               : request_bytes + critical_request_count,
      DirectBatchPriority::speculative, true);
  }
  if (speculative_matching == 0) {
    if (mandatory_fenced_tail) return -EINVAL;
    return direct_fetch_batch(
      params, memory_node, request_shards, remote_offsets,
      critical_request_count, params.graph_scratch, kPersistentGraphReadBytes,
      bytes, lane, local_iova_offsets, critical_completion, true, nullptr,
      critical_completion_timestamp_ns, request_bytes,
      DirectBatchPriority::critical, true);
  }

  const u32 qp_lane = lane % params.direct_qps_per_node;
  const u32 qp_index = qp_lane * params.direct_region_count + memory_node;
  if (qp_index >= params.direct_batch_queue_count ||
      params.direct_qps[qp_index] == nullptr) {
    return -EINVAL;
  }
  DirectOwnerProgress* watchdog_progress =
    params.direct_owner_progress == nullptr
      ? nullptr
      : params.direct_owner_progress + qp_index;
  atomicExch(critical_completion, -EINPROGRESS);
  if (speculative_completion != nullptr) {
    atomicExch(speculative_completion, -EINPROGRESS);
  }
  if (critical_completion_timestamp_ns != nullptr) {
    *critical_completion_timestamp_ns = 0;
  }
  if (speculative_completion_timestamp_ns != nullptr) {
    *speculative_completion_timestamp_ns = 0;
  }
  __threadfence();
  const DirectBatchDescriptor descriptor{
    .request_shards = request_shards,
    .remote_offsets = remote_offsets,
    .local_iova_offsets = local_iova_offsets,
    .request_bytes = request_bytes,
    .completion_status = critical_completion,
    .completion_timestamp_ns = critical_completion_timestamp_ns,
    .speculative_completion_status = speculative_completion,
    .speculative_completion_timestamp_ns = speculative_completion_timestamp_ns,
    .request_count = static_cast<u16>(request_count),
    .memory_node = static_cast<u16>(memory_node),
    .bytes = bytes,
    .priority = static_cast<u8>(DirectBatchPriority::critical),
    .flags = descriptor_flags,
    .critical_request_count = static_cast<u16>(critical_request_count),
  };
  if (watchdog_progress != nullptr) {
    record_owner_watchdog_counter(&watchdog_progress->announced);
  }
  bool pushed =
    device_ring_try_push(params.direct_batch_queues[qp_index], descriptor);
  while (!pushed && mandatory_fenced_tail) {
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      atomicExch(critical_completion, -ECANCELED);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      atomicExch(critical_completion, -EHOSTDOWN);
      if (watchdog_progress != nullptr) {
        record_owner_watchdog_counter(&watchdog_progress->completed);
      }
      return -EHOSTDOWN;
    }
    // This is correctness-critical work, so transient ring pressure is
    // bounded backpressure just like direct_fetch_batch(), not a reason to
    // fall back to two independently published batches.
    device_ring_relax(128);
    pushed =
      device_ring_try_push(params.direct_batch_queues[qp_index], descriptor);
  }
  if (!pushed) {
    atomicExch(critical_completion, -EAGAIN);
    if (speculative_completion != nullptr) {
      atomicExch(speculative_completion, -EAGAIN);
    }
    if (watchdog_progress != nullptr) {
      record_owner_watchdog_counter(&watchdog_progress->completed);
    }
    return -EAGAIN;
  }
  return -EINPROGRESS;
#else
  (void)params;
  (void)memory_node;
  (void)request_shards;
  (void)remote_offsets;
  (void)request_count;
  (void)critical_request_count;
  (void)bytes;
  (void)lane;
  (void)local_iova_offsets;
  (void)critical_completion;
  (void)critical_completion_timestamp_ns;
  (void)speculative_completion;
  (void)speculative_completion_timestamp_ns;
  (void)request_bytes;
  (void)descriptor_flags;
  return -ENOTSUP;
#endif
}

// Publish one exact-record snapshot train. The request SoA contains two
// equal-sized runs: complete records followed by second-header trailers.
// The owner reserves both runs as critical work, puts a hardware initiator
// fence on the first trailer READ, and signals only the final trailer (or one
// final dump WQE). Thus the caller observes one completion for
// [full snapshots -> fence -> validation snapshots] and never performs the
// old query-side wait/enqueue/wait round trip.
__device__ __forceinline__ i32 direct_fetch_fenced_snapshot_batch(
  const PersistentKernelParams& params, u32 memory_node,
  const u32* request_shards, const u64* remote_offsets, u32 snapshot_count,
  u32 bytes, u32 lane, const u64* local_iova_offsets, i32* completion,
  u64* completion_timestamp_ns = nullptr) {
  static_assert(2u * kPersistentMaxExact <= kPersistentMaxMergeCandidates);
  if (snapshot_count == 0 || snapshot_count > kPersistentMaxExact) {
    return -EINVAL;
  }
  return direct_fetch_split_batch(
    params, memory_node, request_shards, remote_offsets, 2u * snapshot_count,
    snapshot_count, bytes, lane, local_iova_offsets, completion,
    completion_timestamp_ns, nullptr, nullptr, nullptr,
    kDirectBatchFlagMandatoryFencedTail);
}

__device__ i32 wait_direct_batch(const PersistentKernelParams& params,
                                 i32* owner_completion) {
#ifdef DVSTOR_HAVE_GPUNETIO
  if (owner_completion == nullptr) return -EINVAL;
  for (;;) {
    const i32 status = *reinterpret_cast<volatile i32*>(owner_completion);
    if (status != -EINPROGRESS) return status;
    if (*reinterpret_cast<const volatile u32*>(params.stop) != 0) {
      return -ECANCELED;
    }
    if (*reinterpret_cast<const volatile u32*>(params.direct_disabled) != 0) {
      return -EHOSTDOWN;
    }
    // This descriptor is already owned by a bounded owner queue.  Queueing
    // delay must not poison every query; the posting owner enforces the actual
    // CQ timeout and publishes a transport-wide failure when appropriate.
    device_ring_relax(128);
  }
#else
  (void)params;
  (void)owner_completion;
  return -ENOTSUP;
#endif
}

}  // namespace gpu_search::persistent_kernel_detail

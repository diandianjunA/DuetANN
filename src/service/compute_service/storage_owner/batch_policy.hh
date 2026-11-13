#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool tail_escape{};
  bool max_wait_flush{};
  u32 take{};
};

// A busy ACK means the authority has exhausted its accepted-batch contexts;
// sending the same logical tokens again at the roughly 10 us RPC round trip
// cannot create a context. Back off per authority until either this deadline
// expires or a real token completion clears it. The short initial delay keeps
// transient races cheap, while the cap is small relative to foreground Stage1
// service time and therefore does not reduce a saturated writer's goodput.
inline u64 storage_owner_busy_retry_delay_ns(u32 consecutive_busy_batches) {
  constexpr u64 kInitialDelayNs = 50'000;
  constexpr u64 kMaximumDelayNs = 2'000'000;
  if (consecutive_busy_batches == 0) return 0;
  const u32 shift = std::min<u32>(consecutive_busy_batches - 1, 6);
  return std::min<u64>(kMaximumDelayNs, kInitialDelayNs << shift);
}

inline bool storage_owner_retry_deadline_elapsed(
    u64 retry_not_before_ns,
    u64 now_ns) {
  return retry_not_before_ns == 0 || now_ns >= retry_not_before_ns;
}

// Accepted maintenance debt used by the time-based admission controller.
// Include completion targets observed since the last watermark refresh so a
// stale storage snapshot cannot hide this client's accepted work.
inline u64 storage_owner_repair_backlog(
    u64 next_sequence,
    u64 target_sequence,
    u64 durable_sequence) {
  const u64 accepted = std::max(
    target_sequence, next_sequence == 0 ? 0 : next_sequence - 1);
  return accepted > durable_sequence ? accepted - durable_sequence : 0;
}

// Deadline admission is a service-level objective, not a guarantee about
// future hardware speed. Unknown rates hold further submissions until a real
// measurement exists; measured rates reserve time to drain accepted debt.
inline bool storage_owner_repair_deadline_blocked(
    u64 oldest_age_ns, u64 backlog, double durable_rate_per_sec,
    u64 deadline_ns) {
  if (deadline_ns == 0 || backlog == 0) return false;
  if (oldest_age_ns >= deadline_ns) return true;
  if (!(durable_rate_per_sec > 0.0) ||
      !std::isfinite(durable_rate_per_sec)) {
    return true;
  }
  const long double remaining_ns =
    static_cast<long double>(backlog) * 1'000'000'000.0L /
    static_cast<long double>(durable_rate_per_sec);
  return remaining_ns >
    static_cast<long double>(deadline_ns - oldest_age_ns);
}

inline bool storage_owner_repair_deadline_exceeded(
    u64 repair_delay_ns, u64 deadline_ns) {
  return deadline_ns != 0 && repair_delay_ns > deadline_ns;
}

inline bool storage_owner_batch_wait_elapsed(
    u64 oldest_ready_since_ns,
    u64 now_ns,
    u64 wait_ns) {
  return wait_ns == 0 ||
    (oldest_ready_since_ns != 0 &&
     now_ns >= oldest_ready_since_ns &&
     now_ns - oldest_ready_since_ns >= wait_ns);
}

// Any published work remaining after a dequeue may belong to the same oldest
// batch (including an MPMC prefix hidden behind a publication hole), so it
// inherits the old timestamp. Only observing the published queue empty ends
// that coalescing interval.
inline u64 next_storage_owner_batch_observed_ns(
    u64 previous_observed_ns,
    u32 remaining_tasks,
    u64 dequeued_at_ns) {
  if (remaining_tasks == 0) return 0;
  // Every task left behind by this dequeue may already have shared the
  // previous oldest timestamp. Resetting its age merely because the requested
  // prefix was fully visible lets an old tail exceed max_wait repeatedly
  // (e.g. dequeue 32 from 44, then restart the remaining 12 at age zero).
  // Keep the conservative old timestamp until the published queue is observed
  // empty. If only newly-arrived work remains this can flush it slightly early,
  // but never violates the configured latency bound.
  return previous_observed_ns != 0
    ? previous_observed_ns : dequeued_at_ns;
}

// Queue::push_wait() is linearizable, but a Vyukov MPMC producer reserves its
// FIFO position before publishing that cell.  A later producer may therefore
// publish position N+1 (and increment the external published-task count) while
// a preempted producer still owns the invisible position N.  The single
// consumer must treat the counter as an admission/batching hint, not as proof
// that the whole requested FIFO prefix is immediately dequeue-visible.
//
// Return the exact visible prefix.  In particular, an invisible first cell
// consumes neither an RPC slot nor published-task credit; a partial prefix is
// safe to send and is charged by its actual size.
template <class Queue, class Output>
inline u32 dequeue_storage_owner_visible_prefix(
    Queue& queue, u32 requested, Output& output) {
  u32 popped = 0;
  while (popped < requested) {
    u32 task_id = 0;
    if (!queue.try_pop(task_id)) break;
    output.push_back(task_id);
    ++popped;
  }
  return popped;
}

// A synchronous caller population forms a closed queueing loop. If every RPC
// slot consumes the first item it sees, N callers are permanently fragmented
// across rpc_depth tiny requests (N / rpc_depth items each). Conversely, a
// policy that waits for a full batch can strand a finite tail forever while
// unrelated RPCs or producers remain active.
//
// Keep RPC depth as a safety ceiling, not a target occupancy.  A synchronous
// caller population is a closed queueing loop: distributing its fixed number
// of callers over every free transport slot permanently fragments later
// batches (N callers / rpc_depth items each) without increasing RPC goodput.
//
// The queue is therefore the bounded batch assembler.  A full batch is sent
// immediately; a truly isolated finite tail is sent immediately; otherwise a
// partial batch remains intact until its hard maximum wait.  Multiple full or
// expired batches may still occupy independent slots, so real offered load
// exposes transport concurrency without allowing slot count to determine
// batch shape.  This is the usual max-batch/max-delay contract and neither
// changes mutation semantics nor hides acknowledged Stage2 debt.
inline StorageOwnerBatchDecision decide_storage_owner_batch(
    u32 ready_tasks,
    u32 active_rpcs,
    u32 free_rpc_slots,
    u32 pending_producers,
    u32 batch_max,
    u64 oldest_ready_since_ns,
    u64 now_ns,
    u64 max_wait_ns) {
  if (batch_max == 0) return {};
  if (free_rpc_slots == 0 || ready_tasks == 0) {
    return {};
  }

  if (ready_tasks >= batch_max) {
    return {.take = batch_max};
  }

  const bool isolated_tail = active_rpcs == 0 && pending_producers == 0;
  const bool max_wait_flush = !isolated_tail &&
    storage_owner_batch_wait_elapsed(
      oldest_ready_since_ns, now_ns, max_wait_ns);
  if (!isolated_tail && !max_wait_flush) return {};

  return {
    .tail_escape = isolated_tail,
    .max_wait_flush = max_wait_flush,
    .take = std::min(ready_tasks, batch_max),
  };
}

}  // namespace compute_service_detail

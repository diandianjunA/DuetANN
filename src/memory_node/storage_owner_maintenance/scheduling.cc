#include "memory_node/memory_node.hh"

#include <bit>
#include <limits>

void MemoryNode::notify_storage_owner_maintenance() {
  // Foreground queue/admission and shutdown waits use this condition variable
  // directly. Executors wait on a separate channel so their broadcast wakeup
  // cannot recursively wake every other executor on each context release.
  storage_owner_maintenance_cv_.notify_all();
  notify_storage_owner_maintenance_executors();
}

void MemoryNode::notify_storage_owner_maintenance_capacity() {
  // The publishing executor continues its own work-conserving loop. This
  // channel is only for foreground queue/admission and shutdown-drain waits;
  // it deliberately leaves the executor epoch unchanged.
  storage_owner_maintenance_cv_.notify_all();
}

void MemoryNode::notify_storage_owner_maintenance_executor(u32 worker_id) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_id >= worker_count) {
    // A stale/foreign registry tag must fail closed to all active executors.
    // It cannot be silently dropped because its true owner is unknown.
    notify_storage_owner_maintenance_executors();
    return;
  }
  storage_owner_maintenance_targeted_wakes_.fetch_add(
    1, std::memory_order_relaxed);
  auto& channel = storage_owner_maintenance_wake_channels_[worker_id];
  channel.epoch.fetch_add(1, std::memory_order_release);
  if (channel.waiters.load(std::memory_order_acquire) == 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(channel.mutex);
  }
  channel.cv.notify_one();
}

void MemoryNode::notify_storage_owner_maintenance_executor_scan(
    u32 worker_id) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_id >= worker_count) {
    for (u32 owner = 0; owner < worker_count; ++owner) {
      storage_owner_maintenance_wake_channels_[owner]
        .context_scan_requested.store(true, std::memory_order_release);
    }
    notify_storage_owner_maintenance_executors();
    return;
  }
  storage_owner_maintenance_wake_channels_[worker_id]
    .context_scan_requested.store(true, std::memory_order_release);
  notify_storage_owner_maintenance_executor(worker_id);
}

void MemoryNode::notify_storage_owner_maintenance_executors() {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_count == 0) return;
  storage_owner_maintenance_broadcast_wakes_.fetch_add(
    1, std::memory_order_relaxed);
  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    auto& channel = storage_owner_maintenance_wake_channels_[worker_id];
    channel.epoch.fetch_add(1, std::memory_order_release);
    if (channel.waiters.load(std::memory_order_acquire) == 0) {
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(channel.mutex);
    }
    channel.cv.notify_one();
  }
}

void MemoryNode::notify_one_storage_owner_maintenance_executor() {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_count == 0) return;
  storage_owner_maintenance_generic_wakes_.fetch_add(
    1, std::memory_order_relaxed);
  const u32 ticket = storage_owner_maintenance_generic_wake_cursor_.fetch_add(
    1, std::memory_order_relaxed);
  u32 selected = ticket % worker_count;
  // Shared capacity is not owner-specific. Prefer a sleeping executor so the
  // sole edge is not assigned to a busy worker while every blocked context
  // waits for its 1 ms deadline. Rotating the scan start preserves fairness.
  for (u32 offset = 0; offset < worker_count; ++offset) {
    const u32 candidate = (selected + offset) % worker_count;
    if (storage_owner_maintenance_wake_channels_[candidate].waiters.load(
          std::memory_order_acquire) != 0) {
      selected = candidate;
      break;
    }
  }
  auto& channel = storage_owner_maintenance_wake_channels_[selected];
  channel.epoch.fetch_add(1, std::memory_order_release);
  if (channel.waiters.load(std::memory_order_acquire) == 0) return;
  {
    std::lock_guard<std::mutex> lock(channel.mutex);
  }
  // A fresh queue descriptor is not worker-owned; any executor can admit it.
  // Waking one avoids an eight-way herd. Context/RDMA completions continue to
  // route to the exact owner recorded by the response registry.
  channel.cv.notify_one();
}

memory_node_storage_owner_maintenance_detail::Stage2ContextOwnerKey
MemoryNode::current_storage_owner_maintenance_context_owner() const {
  return current_storage_owner_maintenance_context_owner_;
}

bool MemoryNode::enqueue_storage_owner_maintenance_context_ready(
    const memory_node_storage_owner_maintenance_detail::
      Stage2ContextOwnerKey& owner,
    memory_node_storage_owner_maintenance_detail::
      Stage2ContextReadyReason reason) {
  if (owner.worker_id >=
      storage_owner_maintenance_ready_queue_active_.size()) {
    storage_owner_maintenance_ready_stale_notifications_.fetch_add(
      1, std::memory_order_relaxed);
    return false;
  }
  StorageOwnerReadyContextQueue* queue =
    storage_owner_maintenance_ready_queue_active_[owner.worker_id].load(
      std::memory_order_acquire);
  if (queue == nullptr || !queue->notify(owner, reason)) {
    storage_owner_maintenance_ready_stale_notifications_.fetch_add(
      1, std::memory_order_relaxed);
    return false;
  }
  storage_owner_maintenance_ready_notifications_.fetch_add(
    1, std::memory_order_relaxed);
  return true;
}

bool MemoryNode::notify_storage_owner_maintenance_context_ready(
    const memory_node_storage_owner_maintenance_detail::
      Stage2ContextOwnerKey& owner,
    memory_node_storage_owner_maintenance_detail::
      Stage2ContextReadyReason reason) {
  if (!enqueue_storage_owner_maintenance_context_ready(owner, reason)) {
    return false;
  }
  notify_storage_owner_maintenance_executor(owner.worker_id);
  return true;
}

bool MemoryNode::mark_current_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters) {
  if (!current_storage_owner_maintenance_worker_ ||
      current_storage_owner_thread_ == nullptr) {
    return false;
  }
  const u32 worker_id = current_storage_owner_thread_->id;
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_id >= worker_count) return false;
  if (current_storage_owner_maintenance_waiter_registrations_ != nullptr) {
    auto& registrations =
      *current_storage_owner_maintenance_waiter_registrations_;
    if (std::find(registrations.begin(), registrations.end(), &waiters) ==
        registrations.end()) {
      registrations.push_back(&waiters);
      waiters.refs[worker_id].fetch_add(1, std::memory_order_acq_rel);
    }
  }
  const size_t word = worker_id / 64;
  const u64 bit = u64{1} << (worker_id % 64);
  waiters.words[word].fetch_or(bit, std::memory_order_release);
  return true;
}

void MemoryNode::clear_current_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters) {
  if (!current_storage_owner_maintenance_worker_ ||
      current_storage_owner_thread_ == nullptr) {
    return;
  }
  const u32 worker_id = current_storage_owner_thread_->id;
  auto* registrations =
    current_storage_owner_maintenance_waiter_registrations_;
  if (registrations == nullptr) return;
  const auto found = std::find(
    registrations->begin(), registrations->end(), &waiters);
  if (found == registrations->end()) return;
  *found = registrations->back();
  registrations->pop_back();
  const u32 previous = waiters.refs[worker_id].fetch_sub(
    1, std::memory_order_acq_rel);
  lib_assert(previous != 0,
             "maintenance resource waiter reference underflow");
  if (previous == 1) {
    clear_storage_owner_maintenance_waiter(waiters, worker_id);
  } else {
    // A release consumes the runnable hint, not a context subscription. Keep
    // the owner visible while any sibling context still needs this resource.
    const size_t word = worker_id / 64;
    const u64 bit = u64{1} << (worker_id % 64);
    waiters.words[word].fetch_or(bit, std::memory_order_release);
  }
}

void MemoryNode::clear_all_current_storage_owner_maintenance_waiters() {
  if (!current_storage_owner_maintenance_worker_ ||
      current_storage_owner_thread_ == nullptr ||
      current_storage_owner_maintenance_waiter_registrations_ == nullptr) {
    return;
  }
  auto& registrations =
    *current_storage_owner_maintenance_waiter_registrations_;
  while (!registrations.empty()) {
    StorageOwnerMaintenanceWaiterSet* waiters = registrations.back();
    clear_current_storage_owner_maintenance_waiter(*waiters);
  }
}

void MemoryNode::clear_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters, u32 worker_id) {
  if (worker_id >= CPU_SETSIZE) return;
  const size_t word = worker_id / 64;
  const u64 bit = u64{1} << (worker_id % 64);
  waiters.words[word].fetch_and(~bit, std::memory_order_acq_rel);
}

void MemoryNode::take_storage_owner_maintenance_waiters(
    StorageOwnerMaintenanceWaiterSet& waiters,
    StorageOwnerMaintenanceWaiterMask& owners) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  // Consume at least the first word even after the executor has stopped. Peer
  // CQ progress can outlive maintenance and must not retain stale owners into
  // a later runtime start.
  const size_t words = std::min<size_t>(
    kStorageOwnerMaintenanceWaiterWords,
    std::max<size_t>(1, (static_cast<size_t>(worker_count) + 63) / 64));
  for (size_t word = 0; word < words; ++word) {
    owners[word] |= waiters.words[word].exchange(
      0, std::memory_order_acq_rel);
  }
}

std::optional<u32> MemoryNode::take_one_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters,
    u32 avoid_worker) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  if (worker_count == 0) return std::nullopt;
  const u32 begin = waiters.cursor.fetch_add(1, std::memory_order_relaxed) %
    worker_count;
  bool avoided_present = false;
  for (u32 offset = 0; offset < worker_count; ++offset) {
    const u32 worker_id = (begin + offset) % worker_count;
    if (worker_id == avoid_worker) {
      const size_t word = worker_id / 64;
      const u64 bit = u64{1} << (worker_id % 64);
      avoided_present =
        (waiters.words[word].load(std::memory_order_acquire) & bit) != 0;
      continue;
    }
    const size_t word = worker_id / 64;
    const u64 bit = u64{1} << (worker_id % 64);
    const u64 previous = waiters.words[word].fetch_and(
      ~bit, std::memory_order_acq_rel);
    if ((previous & bit) != 0) return worker_id;
  }
  if (avoided_present && avoid_worker < worker_count) {
    const size_t word = avoid_worker / 64;
    const u64 bit = u64{1} << (avoid_worker % 64);
    const u64 previous = waiters.words[word].fetch_and(
      ~bit, std::memory_order_acq_rel);
    if ((previous & bit) != 0) return avoid_worker;
  }
  return std::nullopt;
}

void MemoryNode::notify_storage_owner_maintenance_waiters(
    const StorageOwnerMaintenanceWaiterMask& owners) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  for (size_t word = 0;
       word < kStorageOwnerMaintenanceWaiterWords; ++word) {
    u64 remaining = owners[word];
    while (remaining != 0) {
      const u32 bit = static_cast<u32>(std::countr_zero(remaining));
      const size_t worker_id = word * 64 + bit;
      remaining &= remaining - 1;
      if (worker_id < worker_count) {
        notify_storage_owner_maintenance_executor(
          static_cast<u32>(worker_id));
      }
    }
  }
}

void MemoryNode::notify_storage_owner_maintenance_waiters_scan(
    const StorageOwnerMaintenanceWaiterMask& owners) {
  const u32 worker_count =
    storage_owner_maintenance_wake_worker_count_.load(
      std::memory_order_acquire);
  for (size_t word = 0;
       word < kStorageOwnerMaintenanceWaiterWords; ++word) {
    u64 remaining = owners[word];
    while (remaining != 0) {
      const u32 bit = static_cast<u32>(std::countr_zero(remaining));
      const size_t worker_id = word * 64 + bit;
      remaining &= remaining - 1;
      if (worker_id < worker_count) {
        notify_storage_owner_maintenance_executor_scan(
          static_cast<u32>(worker_id));
      }
    }
  }
}

void MemoryNode::reset_storage_owner_maintenance_waiters(
    StorageOwnerMaintenanceWaiterSet& waiters) {
  for (auto& word : waiters.words) {
    word.store(0, std::memory_order_relaxed);
  }
  for (auto& ref : waiters.refs) {
    ref.store(0, std::memory_order_relaxed);
  }
  waiters.cursor.store(0, std::memory_order_relaxed);
}

bool MemoryNode::try_acquire_storage_owner_search_lane_lease(
    bool& waiter_registered) {
  lib_assert(current_storage_owner_maintenance_worker_ &&
               current_storage_owner_thread_ != nullptr,
             "Stage2 search-lane lease acquired outside maintenance");
  const u32 worker_id = current_storage_owner_thread_->id;
  lib_assert(worker_id < CPU_SETSIZE,
             "Stage2 search-lane lease owner is out of range");
  u32 grants = storage_owner_search_lane_grants_[worker_id].load(
    std::memory_order_acquire);
  while (grants != 0) {
    if (storage_owner_search_lane_grants_[worker_id].compare_exchange_weak(
          grants, grants - 1, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      cancel_storage_owner_search_lane_waiter(waiter_registered);
      if (storage_owner_search_lane_blocked_contexts_[worker_id].load(
            std::memory_order_acquire) != 0) {
        // A grant belongs to the worker's runnable set, not to the first
        // context that originally published the bit. Preserve visibility for
        // any sibling context that remains blocked after this consumption.
        // This is only a runnable-hint republish: registering through the
        // current context would incorrectly subscribe the context that just
        // acquired the lane.
        const size_t word = worker_id / 64;
        const u64 bit = u64{1} << (worker_id % 64);
        storage_owner_search_lane_waiters_.words[word].fetch_or(
          bit, std::memory_order_release);
      }
      return true;
    }
  }
  const auto try_once = [&]() {
    const u32 limit = storage_owner_search_lane_lease_limit_.load(
      std::memory_order_acquire);
    u32 active = storage_owner_search_lane_leases_.load(
      std::memory_order_acquire);
    while (active < limit) {
      if (storage_owner_search_lane_leases_.compare_exchange_weak(
            active, active + 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        u32 peak = storage_owner_search_lane_lease_peak_.load(
          std::memory_order_relaxed);
        while (peak < active + 1 &&
               !storage_owner_search_lane_lease_peak_.compare_exchange_weak(
                 peak, active + 1, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
        }
        return true;
      }
    }
    return false;
  };

  if (try_once()) {
    cancel_storage_owner_search_lane_waiter(waiter_registered);
    return true;
  }
  if (!mark_current_storage_owner_maintenance_waiter(
        storage_owner_search_lane_waiters_)) {
    return false;
  }
  if (!waiter_registered) {
    storage_owner_search_lane_blocked_contexts_[worker_id].fetch_add(
      1, std::memory_order_acq_rel);
    waiter_registered = true;
  }
  storage_owner_search_lane_lease_blocked_.fetch_add(
    1, std::memory_order_relaxed);
  // Register-before-recheck closes release -> mark. If a lane returned in that
  // interval the second CAS observes it; otherwise the releasing owner takes
  // this bit after publishing the credit.
  if (try_once()) {
    cancel_storage_owner_search_lane_waiter(waiter_registered);
    return true;
  }
  return false;
}

void MemoryNode::cancel_storage_owner_search_lane_waiter(
    bool& waiter_registered) {
  if (!waiter_registered) return;
  lib_assert(current_storage_owner_maintenance_worker_ &&
               current_storage_owner_thread_ != nullptr,
             "Stage2 search-lane waiter canceled outside maintenance");
  const u32 worker_id = current_storage_owner_thread_->id;
  const u32 previous =
    storage_owner_search_lane_blocked_contexts_[worker_id].fetch_sub(
      1, std::memory_order_acq_rel);
  lib_assert(previous != 0,
             "Stage2 search-lane blocked-context count underflow");
  waiter_registered = false;
  clear_current_storage_owner_maintenance_waiter(
    storage_owner_search_lane_waiters_);
  if (previous == 1) {
    // A release can select this worker immediately before cancellation. The
    // releasing thread publishes its grant before rechecking the blocked
    // count, while this side reclaims after publishing zero. These two checks
    // close both sides of the race without binding a token to a dead context.
    return_storage_owner_search_lane_grants(worker_id);
  }
}

void MemoryNode::handoff_or_release_storage_owner_search_lane_lease(
    u32 avoid_worker) {
  for (;;) {
    const auto owner = storage_owner_maintenance_shutdown_.load(
                         std::memory_order_acquire)
      ? std::optional<u32>{}
      : take_one_storage_owner_maintenance_waiter(
          storage_owner_search_lane_waiters_, avoid_worker);
    if (!owner.has_value()) {
      const u32 previous = storage_owner_search_lane_leases_.fetch_sub(
        1, std::memory_order_acq_rel);
      lib_assert(previous != 0,
                 "Stage2 global search-lane lease underflow");
      return;
    }

    storage_owner_search_lane_grants_[*owner].fetch_add(
      1, std::memory_order_release);
    if (storage_owner_search_lane_blocked_contexts_[*owner].load(
          std::memory_order_acquire) != 0) {
      notify_storage_owner_maintenance_executor_scan(*owner);
      return;
    }

    // The selected context canceled between take_one and grant publication.
    // Reclaim exactly one still-unconsumed token and pass the baton. If the
    // worker raced in and consumed it, ownership has already transferred.
    u32 grants = storage_owner_search_lane_grants_[*owner].load(
      std::memory_order_acquire);
    bool reclaimed = false;
    while (grants != 0) {
      if (storage_owner_search_lane_grants_[*owner].compare_exchange_weak(
            grants, grants - 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        reclaimed = true;
        break;
      }
    }
    if (!reclaimed) return;
    avoid_worker = *owner;
  }
}

void MemoryNode::return_storage_owner_search_lane_grants(u32 worker_id) {
  lib_assert(worker_id < CPU_SETSIZE,
             "Stage2 returned search-lane grant owner is out of range");
  const u32 grants = storage_owner_search_lane_grants_[worker_id].exchange(
    0, std::memory_order_acq_rel);
  for (u32 grant = 0; grant < grants; ++grant) {
    handoff_or_release_storage_owner_search_lane_lease(worker_id);
  }
}

void MemoryNode::retire_storage_owner_search_lane_grants(u32 worker_id) {
  lib_assert(worker_id < CPU_SETSIZE,
             "Stage2 search-lane grant owner is out of range");
  const u32 grants = storage_owner_search_lane_grants_[worker_id].exchange(
    0, std::memory_order_acq_rel);
  if (grants == 0) return;
  const u32 previous = storage_owner_search_lane_leases_.fetch_sub(
    grants, std::memory_order_acq_rel);
  lib_assert(previous >= grants,
             "Stage2 retired search-lane grants underflowed leases");
}

void MemoryNode::release_storage_owner_search_lane_lease() {
  const u32 releasing_worker =
    current_storage_owner_maintenance_worker_ &&
        current_storage_owner_thread_ != nullptr
      ? current_storage_owner_thread_->id
      : std::numeric_limits<u32>::max();
  handoff_or_release_storage_owner_search_lane_lease(releasing_worker);
}


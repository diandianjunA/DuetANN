#include "memory_node/storage_owner_maintenance/worker_internal.hh"

namespace memory_node_storage_owner_maintenance_detail {

Stage2Worker::Stage2Worker(MemoryNode& node, u32 worker_id_)
    : node_(node), worker_id_(worker_id_) {
  lib_assert(node_.peer_rpc_runtime_.message_bytes >
               sizeof(service::storage_owner::PeerRpcHeader),
             "peer RPC slot has no reconciliation payload capacity");
  lib_assert(worker_id_ < node_.storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  lib_assert(worker_id_ < node_.storage_owner_maintenance_wake_channels_.size(),
             "storage-owner maintenance wake channel missing");
  MemoryNode::current_storage_owner_thread_ = &thread;
  MemoryNode::current_storage_owner_maintenance_worker_ = true;
  MemoryNode::current_storage_owner_maintenance_context_owner_ = {};
  lib_assert(ready_context_queue != nullptr,
             "storage-owner maintenance ready queue missing");
  fallback_waiter_registrations.reserve(
    5 + static_cast<size_t>(node_.num_storage_nodes_) *
          (static_cast<size_t>(node_.peer_qps_per_peer_) + 1));
  MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
    &fallback_waiter_registrations;
  lib_assert(node_.num_storage_nodes_ > 0 && node_.num_storage_nodes_ <= 64,
             "asynchronous stage2 supports at most 64 storage shards");
  snapshot_storage.reserve(
    continuation_capacity +
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  snapshot_by_raw.reserve(
    continuation_capacity +
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  identity_storage.reserve(
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  stable_identity_targets.reserve(
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  reverse_completion_wake_owners.reserve(
    node_.storage_owner_reverse_completions_.size());
  reverse_response_payload.reserve(node_.peer_rpc_runtime_.message_bytes);
  overflow_ready_events.reserve(context_capacity);

  for (Stage2Context& context : contexts) {
    context.waiter_registrations.reserve(
      5 + static_cast<size_t>(node_.num_storage_nodes_) *
            (static_cast<size_t>(node_.peer_qps_per_peer_) + 1));
    context.tasks.reserve(config.storage_owner_batch_max);
    context.targets.reserve(config.storage_owner_batch_max);
    context.continued_candidates_by_task.reserve(
      config.storage_owner_batch_max);
    context.live_stage2_neighbors_by_task.resize(
      config.storage_owner_batch_max);
    context.finalize_subphases.resize(
      config.storage_owner_batch_max, Stage2FinalizeSubphase::prepare);
    context.reconcile_batch.reserve(
      reconcile_op_capacity, reconcile_chunk_capacity);
    context.reconcile_items.reserve(config.storage_owner_batch_max);
    context.remote_ops_by_peer.resize(node_.num_storage_nodes_);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.reserve(static_cast<size_t>(config.R) *
                  config.storage_owner_batch_max);
    }
    context.reverse_request_ids.resize(node_.num_storage_nodes_);
  }

}

void Stage2Worker::run() {
  for (;;) {
    // Worker-wide producers (currently reverse-outbox posting) rebuild their
    // resource dependencies on every scheduler pass. Remove subscriptions
    // whose operation either succeeded or was canceled before retrying; the
    // failure path registers again under the same slot mutex/counter recheck.
    MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
      &fallback_waiter_registrations;
    node_.clear_all_current_storage_owner_maintenance_waiters();
    // Snapshot before scanning contexts. Any completion/event published after
    // node_ load either becomes visible to the scan or changes the predicate
    // below. This closes the classic notify-before-wait window without busy
    // polling and without shortening the batching deadline.
    const u64 observed_wake_epoch =
      wake_channel.epoch.load(std::memory_order_acquire);
    const bool resource_context_scan_requested =
      wake_channel.context_scan_requested.exchange(
        false, std::memory_order_acq_rel);
    const bool refinement_progressed = consolidate_finalized_insertions();
    if (refinement_progressed && !storage_owner_pipeline_refinement_enabled()) {
      continue;
    }
    if (node_.storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      if (refinement_progressed) continue;
      flush_idle_timing();
      if (node_.storage_owner_reverse_outbox_ != nullptr) {
        (void)node_.storage_owner_reverse_outbox_->erase_queued_worker(worker_id_);
        for (;;) {
          const auto wire_request_id =
            node_.storage_owner_reverse_outbox_->discard_owned_aggregate(worker_id_);
          if (!wire_request_id.has_value()) break;
          node_.cancel_peer_rpc_response(*wire_request_id);
        }
      }
      // Registered lane scratch cannot be reset while the HCA may still be
      // writing into it. Shutdown therefore drains every outstanding lane
      // before releasing context ownership. This path is bounded by the
      // transport's already-posted work only; it never submits new Stage2
      // reads after shutdown has been observed.
      for (;;) {
        bool lane_rdma_pending = false;
        for (const Stage2Context& context : contexts) {
          if (!context.active || !context.search_lane.has_value()) continue;
          lane_rdma_pending = lane_rdma_pending ||
            !thread.is_ready(*context.search_lane);
        }
        if (!lane_rdma_pending) break;
        node_.poll_peer_send_cq();
        std::unique_lock<std::mutex> lock(node_.storage_owner_maintenance_mutex_);
        node_.storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
      for (Stage2Context& context : contexts) {
        if (!context.active) {
          continue;
        }
        MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
          &context.waiter_registrations;
        finish_reverse_prepare_timing(context);
        cancel_context_reconcile(context);
        node_.cancel_storage_owner_search_lane_waiter(
          context.search_lane_wait_registered);
        node_.clear_all_current_storage_owner_maintenance_waiters();
        release_context_lane(context);
        // Invalidate every logical home-RPC member before invalidating its
        // OwnerKey. A late aggregate response then suppresses this member
        // instead of publishing an owned payload to a retired context slot.
        for (size_t rpc_index = 0;
             rpc_index < context.search_io.home_expand_rpc_count;
             ++rpc_index) {
          const Stage2HomeExpandRpc& rpc =
            context.search_io.home_expand_rpcs[rpc_index];
          if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
            node_.cancel_peer_rpc_response(rpc.request_id);
          }
        }
        for (size_t rpc_index = 0;
             rpc_index < context.search_io.score_home_rpc_count;
             ++rpc_index) {
          const Stage2HomeExpandRpc& rpc =
            context.search_io.score_home_rpcs[rpc_index];
          if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
            node_.cancel_peer_rpc_response(rpc.request_id);
          }
        }
        lib_assert(ready_context_queue->deactivate(context.ready_owner),
                   "stage2 shutdown deactivated a stale ready owner");
        context.ready_owner = {};
        node_.storage_owner_maintenance_active_workers_.fetch_sub(
          1, std::memory_order_acq_rel);
      }
      node_.cancel_storage_owner_search_lane_waiter(
        refinement_lane_wait_registered);
      node_.retire_storage_owner_search_lane_grants(worker_id_);
      MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
        &fallback_waiter_registrations;
      node_.clear_all_current_storage_owner_maintenance_waiters();
      MemoryNode::current_storage_owner_maintenance_waiter_registrations_ = nullptr;
      MemoryNode::current_storage_owner_maintenance_context_owner_ = {};
      MemoryNode::current_storage_owner_maintenance_worker_ = false;
      MemoryNode::current_storage_owner_thread_ = nullptr;
      return;
    }

    // Completion producers route directly to a generation-fenced context
    // ticket. Drain those O(1) events before worker-wide producers so a ready
    // continuation does not wait behind unrelated slot scans or aggregation.
    auto [ready_progressed, ready_count] = drain_ready_contexts();
    bool progressed = refinement_progressed || ready_progressed;

    // Drain every currently sendable per-peer descriptor. A second pass below
    // catches work produced by pruning in this iteration; neither pass waits
    // for a timer to form a batch.
    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;
    auto [late_ready_progressed, late_ready_count] = drain_ready_contexts();
    progressed = late_ready_progressed || progressed;
    ready_count += late_ready_count;

    // Resource-credit wakes do not yet carry a context key, and local lock /
    // retry timers have no external producer. Recover them with a bounded
    // scan only when an unmatched worker wake requests one or once per 10 ms.
    // Ordinary RDMA/RPC/reverse completions stay on the ticket path.
    const size_t context_count = contexts.size();
    lib_assert(context_count != 0,
               "maintenance worker has no Stage2 context slots");
    const auto scan_now = std::chrono::steady_clock::now();
    const bool periodic_scan_due = scan_now >= next_fallback_context_scan;
    const bool scan_unmatched_wake =
      resource_context_scan_requested ||
      (fallback_context_scan_requested && ready_count == 0);
    fallback_context_scan_requested = false;
    if (periodic_scan_due || scan_unmatched_wake) {
      const size_t scan_begin = active_context_cursor;
      unpublished_context_slots_scanned += context_count;
      node_.storage_owner_maintenance_ready_fallback_scans_.fetch_add(
        1, std::memory_order_relaxed);
      if (unpublished_context_slots_scanned >= 4096) {
        node_.storage_owner_maintenance_context_slots_scanned_.fetch_add(
          unpublished_context_slots_scanned, std::memory_order_relaxed);
        unpublished_context_slots_scanned = 0;
      }
      for (size_t offset = 0; offset < context_count; ++offset) {
        const size_t context_index = stage2_round_robin_context_index(
          scan_begin, offset, context_count);
        Stage2Context& context = contexts[context_index];
        if (context.active) {
          const bool context_progressed = drive_owned_context(context);
          progressed = context_progressed || progressed;
        }
      }
      active_context_cursor = scan_begin + 1 == context_count
        ? 0 : scan_begin + 1;
      next_fallback_context_scan =
        scan_now + kFallbackContextAuditInterval;
    }

    // Admission is intentionally a bounded scheduler action.  With a hot
    // shard, an unbounded admit-and-immediately-finish stream can keep every
    // executor inside this loop indefinitely, starving ready completions,
    // reverse ACKs, timeout progress, and even the observation heartbeat.
    // Eight workers still admit up to eight contexts per scheduler pass; each
    // pass must first service already-owned dependency chains.
    constexpr size_t kAdmissionBurstPerSchedulerPass = 1;
    size_t admission_burst = 0;
    while (admission_burst < kAdmissionBurstPerSchedulerPass) {
      Stage2Context* context = try_admit_context();
      if (context == nullptr) break;
      ++admission_burst;
      progressed = true;
      (void)drive_owned_context(*context);
    }
    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;

    node_.maybe_log_storage_owner_maintenance_observation();
    if (!progressed) {
      const auto idle_started = std::chrono::steady_clock::now();
      auto wake_at = std::min(
        next_fallback_context_scan,
        idle_started + kFallbackContextAuditInterval);
      // Deferred contexts are worker-private here: clamp the sleep to the
      // earliest task retry deadline instead of waiting for the 10 ms
      // fallback audit to rediscover them.
      for (const Stage2Context& context : contexts) {
        if (!context.active) continue;
        for (const StorageOwnerMaintenanceTask& task : context.tasks) {
          if (task.retry_not_before !=
                std::chrono::steady_clock::time_point{} &&
              task.retry_not_before < wake_at) {
            wake_at = task.retry_not_before;
          }
        }
      }
      {
        std::lock_guard<std::mutex> queue_lock(
          node_.storage_owner_maintenance_mutex_);
        if (!node_.storage_owner_stage2_tasks_.empty()) {
          const auto parameters = node_.storage_owner_stage2_packing_.parameters();
          const size_t batch_limit =
            std::max<size_t>(1, config.storage_owner_batch_max);
          const Stage2PackingDecision decision = decide_stage2_packing(
            node_.storage_owner_stage2_tasks_.size(),
            batch_limit,
            node_.storage_owner_stage2_tasks_.front().queued_at,
            idle_started,
            config.storage_owner_stage2_batch_max_wait_us,
            parameters.estimated_arrival_interval_us);
          const auto not_before =
            node_.storage_owner_stage2_tasks_.front().queued_at +
            std::chrono::milliseconds(
              config.storage_owner_stage2_initial_delay_ms);
          if (not_before > idle_started && not_before < wake_at) {
            wake_at = not_before;
          } else if (not_before <= idle_started &&
                     decision.deadline.has_value() &&
                     *decision.deadline < wake_at) {
            wake_at = *decision.deadline;
          }
        }
      }
      const u64 idle_started_ns = steady_now_ns();
      std::unique_lock<std::mutex> wake_lock(wake_channel.mutex);
      wake_channel.waiters.fetch_add(
        1, std::memory_order_acq_rel);
      const auto unregister_waiter = [&]() {
        const u32 previous =
          wake_channel.waiters.fetch_sub(
            1, std::memory_order_acq_rel);
        lib_assert(previous != 0,
                   "maintenance wake waiter registration underflow");
      };
      const bool raced_before_wait =
        wake_channel.epoch.load(
          std::memory_order_acquire) != observed_wake_epoch;
      if (raced_before_wait) {
        node_.storage_owner_maintenance_lost_wake_avoided_.fetch_add(
          1, std::memory_order_relaxed);
      } else {
        (void)wake_channel.cv.wait_until(
          wake_lock, wake_at, [&]() {
            return node_.storage_owner_maintenance_shutdown_.load(
                     std::memory_order_acquire) ||
              wake_channel.epoch.load(
                std::memory_order_acquire) != observed_wake_epoch;
          });
      }
      fallback_context_scan_requested =
        wake_channel.epoch.load(std::memory_order_acquire) !=
          observed_wake_epoch;
      if (!fallback_context_scan_requested) {
        // A deadline-timed wake carries no event epoch. Request the bounded
        // scan that drives the now-due deferred context; without it the next
        // iteration would recompute a past wake_at and spin until the 10 ms
        // fallback audit.
        const auto woken_at = std::chrono::steady_clock::now();
        for (const Stage2Context& context : contexts) {
          if (!context.active) continue;
          for (const StorageOwnerMaintenanceTask& task : context.tasks) {
            if (task.retry_not_before !=
                  std::chrono::steady_clock::time_point{} &&
                task.retry_not_before <= woken_at) {
              fallback_context_scan_requested = true;
              break;
            }
          }
          if (fallback_context_scan_requested) break;
        }
      }
      unregister_waiter();
      wake_lock.unlock();
      unpublished_idle_wait_ns += steady_now_ns() - idle_started_ns;
      ++unpublished_idle_waits;
      // Publishing in small batches avoids an atomic operation on every idle
      // loop while keeping the five-second observation error below one worker
      // times 64 ms.
      if (unpublished_idle_waits >= 64) flush_idle_timing();
    }
  }

}

}  // namespace memory_node_storage_owner_maintenance_detail

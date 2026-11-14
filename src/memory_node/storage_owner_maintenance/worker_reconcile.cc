#include "memory_node/storage_owner_maintenance/worker_internal.hh"

#include <cstdlib>

namespace memory_node_storage_owner_maintenance_detail {

bool Stage2Worker::build_reconcile_barrier_ops(
      Stage2Context& context, size_t item, Stage2ReconcileBarrier barrier,
      vec<ReconcileReverseOp>& selected_ops){
    vec<ReconcileReverseOp> promotion_ops;
    vec<ReconcileReverseOp> stable_ops;
    vec<ReconcileReverseOp> removal_ops;
    StorageOwnerMaintenanceTask& task = context.tasks[item];
    lib_assert(item < context.live_stage2_neighbors_by_task.size(),
               "Stage2 reconcile lost its persisted planner input");
    if (!append_stage2_reconcile_ops(
          task,
          span<const RemotePtr>{context.live_stage2_neighbors_by_task[item]},
          promotion_ops, stable_ops, removal_ops)) {
      return false;
    }
    switch (barrier) {
      case Stage2ReconcileBarrier::install:
        // This ordering is semantic, not merely a packing preference. The
        // target-run packer below groups a target without reordering its ops,
        // so every receiver executes all ordinary stable proposals before any
        // mandatory promotion for that target and audits the mandatory set on
        // the resulting final adjacency before ACKing the install barrier.
        stable_ops.insert(stable_ops.end(),
                          std::make_move_iterator(promotion_ops.begin()),
                          std::make_move_iterator(promotion_ops.end()));
        if (!config.storage_owner_lease_merge) {
          stable_ops.insert(stable_ops.end(),
                            std::make_move_iterator(removal_ops.begin()),
                            std::make_move_iterator(removal_ops.end()));
        }
        selected_ops = std::move(stable_ops);
        return true;
      case Stage2ReconcileBarrier::removal:
        if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
            task.id < 100000016) {
          std::cerr << "[stage2-bridge-debug] removal id=" << task.id
                    << " count=" << removal_ops.size();
          for (const auto& op : removal_ops) {
            std::cerr << " " << op.target_raw;
          }
          std::cerr << std::endl;
        }
        selected_ops = std::move(removal_ops);
        return true;
      case Stage2ReconcileBarrier::release: {
        const vec<RemotePtr> parents = stage2_certificate_release_parents(
          task.stage2_promotion_parent,
          span<const RemotePtr>{task.stage2_replaced_certificate_parents});
        for (const RemotePtr parent : parents) {
          selected_ops.push_back(make_reconcile_op(
            task, parent, RemotePtr{}, task.final_target,
            ReconcileReverseOpKind::release_certificate_lease));
        }
        return true;
      }
      case Stage2ReconcileBarrier::none:
        return false;
    }
    return false;
  };

  // Apply the local part immediately, then retain one immutable copy of every
  // remote chunk in the context.  Posting is intentionally separate so one
  // peer with no send credit cannot prevent all other peers from being posted.



bool Stage2Worker::begin_reconcile_barrier(
      Stage2Context& context, Stage2ReconcileBarrier barrier,
      span<const ReconcileReverseOp> ops) {
    const auto debug_failure = [&](const char* reason) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") == nullptr) return;
      const auto now = std::chrono::steady_clock::now();
      const bool stalled = std::any_of(
        context.tasks.begin(), context.tasks.end(), [&](const auto& task) {
          return now - task.queued_at > std::chrono::seconds(5);
        });
      if (!stalled) return;
      static std::atomic<u32> debug_count{0};
      if (debug_count.fetch_add(1, std::memory_order_relaxed) >= 64) return;
      std::cerr << "[stage2-reconcile-stall] reason=" << reason
                << " worker=" << worker_id_
                << " barrier=" << static_cast<u32>(barrier)
                << " ops=" << ops.size()
                << " first_id=" << (context.tasks.empty()
                  ? 0 : context.tasks.front().id) << std::endl;
    };
    vec<ReconcileReverseOp> local_ops;
    dense_hashmap_t<u32, vec<ReconcileReverseOp>> remote_ops;
    local_ops.reserve(ops.size());
    for (const ReconcileReverseOp& op : ops) {
      const RemotePtr target{op.target_raw};
      if (target.is_null() || target.memory_node() >= node_.num_storage_nodes_) {
        debug_failure("invalid_target");
        return false;
      }
      if (node_.local_shard(target.memory_node())) {
        local_ops.push_back(op);
      } else {
        remote_ops[target.memory_node()].push_back(op);
      }
    }

    vec<protocol::ReconcileReverseResult> local_results;
    if (!local_ops.empty() &&
        !node_.reconcile_local_reverse_ops(
          span<const ReconcileReverseOp>{local_ops}, config, local_results)) {
      debug_failure("local_apply");
      return false;
    }
    if (local_results.size() != local_ops.size()) {
      debug_failure("local_result_count");
      return false;
    }
    for (size_t index = 0; index < local_ops.size(); ++index) {
      if (!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(
              local_ops[index], local_results[index])) {
        debug_failure("local_postcondition");
        return false;
      }
    }

    const size_t payload_bytes = node_.peer_rpc_runtime_.message_bytes -
      sizeof(protocol::PeerRpcHeader);
    const u32 wire_capacity = static_cast<u32>(
      payload_bytes / sizeof(ReconcileReverseOp));
    lib_assert(wire_capacity != 0,
               "peer RPC slot cannot hold a reverse reconciliation op");
    context.reconcile_batch.begin(context.handle, barrier);
    for (const auto& [target_shard, peer_ops] : remote_ops) {
      const auto packed = pack_stage2_reconcile_target_runs(
        span<const ReconcileReverseOp>{peer_ops}, wire_capacity);
      // The whole context shares one target transaction. An oversize run is a
      // semantic retry: splitting it could expose ordinary work after a
      // promotion or ACK only a subset of the mandatory certificate set.
      if (!packed.has_value()) {
        debug_failure("oversize_target_run");
        return false;
      }
      for (const auto& chunk_ops : *packed) {
        const size_t count = chunk_ops.size();
        const u64 request_id = node_.allocate_peer_request_id();
        const bool appended = context.reconcile_batch.append_chunk(
          request_id, target_shard,
          std::span<const ReconcileReverseOp>{chunk_ops.data(), count});
        lib_assert(appended,
                   "bounded Stage2 reconcile chunk could not be persisted");
      }
    }
    for (const size_t item : context.reconcile_items) {
      context.finalize_subphases[item] =
        stage2_reconcile_wait_subphase(barrier);
    }
    return true;
  };

  // One nonblocking transport pass over every peer/chunk.  Retries preserve
  // request_id and the exact byte payload.  An ACK is accepted only when both
  // the context generation/barrier epoch and every per-op postcondition match.



Stage2Worker::ReconcilePollResult Stage2Worker::poll_reconcile_barrier(
      Stage2Context& context) {
    if (!context.reconcile_batch.active() ||
        context.reconcile_batch.context() != context.handle) {
      return ReconcilePollResult::retry;
    }
    constexpr u32 kTransportAttempts = 3;
    const u32 epoch = context.reconcile_batch.epoch();
    vec<Stage2ReconcileChunk>& chunks =
      context.reconcile_batch.chunks();
    for (size_t chunk_index = 0; chunk_index < chunks.size();
         ++chunk_index) {
      Stage2ReconcileChunk& chunk = chunks[chunk_index];
      if (chunk.complete) continue;
      if (!chunk.correlates(context.handle, epoch)) {
        return ReconcilePollResult::retry;
      }

      u64 now_ns = steady_now_ns();
      if (!chunk.attempt_active) {
        if (chunk.attempts_started == kTransportAttempts) {
          return ReconcilePollResult::retry;
        }
        ++chunk.attempts_started;
        chunk.attempt_active = true;
        chunk.posted = false;
        chunk.deadline_ns = now_ns + rpc_timeout_ns;
      }

      const std::span<const ReconcileReverseOp> payload =
        context.reconcile_batch.payload(chunk);
      if (payload.size() != chunk.item_count) {
        return ReconcilePollResult::retry;
      }
      if (!chunk.posted) {
        const size_t request_bytes =
          protocol::reconcile_reverse_request_bytes(chunk.item_count);
        chunk.posted = node_.try_post_peer_rpc_request_attempt(
          chunk.target_shard, PeerRpcType::reconcile_reverse_request,
          PeerRpcType::reconcile_reverse_response, chunk.request_id,
          chunk.item_count, payload.data(),
          payload.size_bytes(), request_bytes,
          PeerRpcSendClass::graph_update);
        now_ns = steady_now_ns();
        if (!chunk.posted && now_ns >= chunk.deadline_ns) {
          node_.cancel_peer_rpc_response(chunk.request_id);
          chunk.attempt_active = false;
        }
        if (!chunk.posted) continue;
      }

      protocol::PeerRpcHeader response_header{};
      PeerResponseLease response_lease{};
      const TryPeerResponse response = node_.try_consume_peer_rpc_response(
        chunk.request_id, chunk.target_shard,
        PeerRpcType::reconcile_reverse_response, chunk.item_count,
        response_header, reverse_response_payload, response_lease);
      now_ns = steady_now_ns();
      if (response == TryPeerResponse::pending) {
        if (now_ns >= chunk.deadline_ns) {
          node_.cancel_peer_rpc_response(chunk.request_id);
          chunk.attempt_active = false;
          chunk.posted = false;
        }
        continue;
      }
      if (response == TryPeerResponse::stale) {
        chunk.attempt_active = false;
        chunk.posted = false;
        continue;
      }

      const size_t expected_bytes =
        protocol::reconcile_reverse_response_bytes(chunk.item_count);
      bool valid = response == TryPeerResponse::success &&
        reverse_response_payload.size() == expected_bytes &&
        response_header.magic == protocol::kPeerRpcMagic &&
        response_header.version == protocol::kPeerRpcVersion &&
        response_header.type == static_cast<u32>(
          PeerRpcType::reconcile_reverse_response) &&
        response_header.source_shard == chunk.target_shard &&
        response_header.item_count == chunk.item_count &&
        response_header.request_id == chunk.request_id &&
        response_header.status == static_cast<u32>(protocol::InsertStatus::ok) &&
        response_header.reserved == 0;
      if (valid) {
        const auto* results = protocol::reconcile_reverse_results(
          reverse_response_payload.data());
        for (u32 index = 0; index < chunk.item_count; ++index) {
          const protocol::ReconcileReverseResult& result = results[index];
          valid = result.accepted <= 1 && result.replaced <= 1 &&
            result.removed <= 1 && result.stale <= 1 &&
            result.reserved == 0 &&
            memory_node_storage_owner_index_detail::
              reconcile_reverse_postcondition_holds(payload[index], result);
          if (!valid) break;
        }
      }

      if (valid) {
        lib_assert(node_.acknowledge_peer_rpc_response(response_lease),
                   "validated async reconcile response lost its lease");
        lib_assert(context.reconcile_batch.mark_complete(
                     chunk_index, context.handle, epoch),
                   "late reconcile ACK crossed a context/barrier fence");
      } else {
        if (response_lease.valid()) {
          lib_assert(node_.rearm_peer_rpc_response(response_lease),
                     "invalid async reconcile response lost its lease");
        }
        chunk.attempt_active = false;
        chunk.posted = false;
      }
    }
    return context.reconcile_batch.complete()
      ? ReconcilePollResult::complete
      : ReconcilePollResult::waiting;
  };




void Stage2Worker::defer_stage2_item_retry(
      Stage2Context& context, size_t item, bool permanent) {
    StorageOwnerMaintenanceTask& task = context.tasks[item];
    const auto now = std::chrono::steady_clock::now();
    if (permanent) {
      if (task.stage2_first_permanent_defer_at ==
          std::chrono::steady_clock::time_point{}) {
        task.stage2_first_permanent_defer_at = now;
      }
      if (config.storage_owner_stage2_max_retry_ms != 0 &&
          now - task.stage2_first_permanent_defer_at >
            std::chrono::milliseconds(config.storage_owner_stage2_max_retry_ms)) {
        const char* stale_reason = "not_attempted";
        if (!complete_stale_stage2(task, &stale_reason)) {
          static std::atomic<u32> terminate_cleanup_logs{0};
          if (terminate_cleanup_logs.fetch_add(1, std::memory_order_relaxed) <
              16) {
            std::cerr << "[storage-owner] stage2 item termination cleanup "
                         "incomplete self_shard=" << node_.storage_id_
                      << " sequence=" << task.maintenance_sequence
                      << " id=" << task.id
                      << " generation=" << task.generation
                      << " reason=" << stale_reason
                      << " placement_committed=" << task.placement_committed
                      << " centroid_committed=" << task.centroid_committed
                      << " final_target=" << task.final_target.raw_address
                      << " target=" << task.target.raw_address
                      << std::endl;
          }
          node_.storage_owner_maintenance_failed_.fetch_add(
            1, std::memory_order_relaxed);
        }
        node_.storage_owner_maintenance_stage2_terminated_.fetch_add(
          1, std::memory_order_relaxed);
        atomic_utils::update_max_relaxed(
          node_.storage_owner_maintenance_stage2_last_terminated_sequence_,
          task.maintenance_sequence);
        task.maintenance_sequence = 0;
        return;
      }
    }
    task.retry_not_before = now + std::chrono::milliseconds(1);
  };

  // Placement authority and centroid membership complete independently for
  // each sealed item after its own reconciliation barriers have ACKed.



bool Stage2Worker::finish_stage2_after_reconcile(
      Stage2Context& context, size_t item) {
    StorageOwnerMaintenanceTask& task = context.tasks[item];
    if (task.maintenance_sequence == 0) return true;
    const auto failed = [&](const char* reason) {
      static std::atomic<u32> debug_count{0};
      const auto task_age = std::chrono::steady_clock::now() - task.queued_at;
      if (((std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
            task_age > std::chrono::seconds(5)) ||
           task_age > std::chrono::seconds(60)) &&
          debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        std::cerr << "[stage2-finish-debug] self=" << node_.storage_id_
                  << " id=" << task.id
                  << " sequence=" << task.maintenance_sequence
                  << " reason=" << reason << std::endl;
      }
      return false;
    };
    lib_assert(!config.storage_owner_lease_merge || !task.certificate_released,
               "Stage2 placement started after certificate release");
    RemotePtr parent = task.stage2_promotion_parent;
    if (parent.is_null() && !task.stage2_neighbors.empty()) {
      parent = task.stage2_neighbors.front();
    }
    if (parent.is_null() && !task.stage1_backlink_targets.empty()) {
      parent = task.stage1_backlink_targets.front();
    }
    lib_assert(!parent.is_null(),
               "Stage2 placement has no certificate parent");
    const auto trace = [&](const char* event, RemotePtr attempted = RemotePtr{}) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") == nullptr) return;
      std::string message = std::string("[certificate-parent-trace] event=") +
        event + " self=" + std::to_string(node_.storage_id_) +
        " id=" + std::to_string(task.id) +
        " sequence=" + std::to_string(task.maintenance_sequence) +
        " child=" + std::to_string(task.final_target.raw_address) +
        " current=" + std::to_string(task.stage2_promotion_parent.raw_address) +
        " attempted=" + std::to_string(attempted.raw_address) + " history=";
      for (const RemotePtr replaced : task.stage2_replaced_certificate_parents) {
        message += std::to_string(replaced.raw_address) + ",";
      }
      std::cerr << message << std::endl;
    };
    trace("enter", parent);
    const auto acquire = [&](RemotePtr candidate) {
      const ReconcileReverseOp op = make_reconcile_op(
        task, candidate, RemotePtr{}, task.final_target,
        stage2_reconcile_certificate_kind(config.storage_owner_lease_merge));
      return apply_reconcile_ops(span<const ReconcileReverseOp>{&op, 1});
    };
    if (!acquire(parent)) {
      if (!config.storage_owner_lease_merge) {
        return failed("certificate_acquire");
      }
      stage2_record_certificate_parent_attempt(
        parent, task.stage2_replaced_certificate_parents);
      const vec<RemotePtr> recovery = node_.local_centroid_route_entries();
      vec<RemotePtr> candidates = stage2_certificate_parent_candidates(
        span<const RemotePtr>{task.stage1_backlink_targets},
        span<const RemotePtr>{task.stage2_neighbors}, parent,
        span<const RemotePtr>{recovery});
      candidates.erase(
        std::remove_if(
          candidates.begin(), candidates.end(),
          [&](RemotePtr candidate) {
            return candidate == task.final_target ||
              !node_.storage_node_pointer_addressable(candidate);
          }),
        candidates.end());
      vec<std::pair<RemotePtr, u64>> identities;
      vec<StableNodeSnapshotState> states;
      node_.read_node_identity_headers_batched_into(
        span<const RemotePtr>{candidates}, config, identities, &states);
      lib_assert(states.size() == candidates.size(),
                 "certificate parent identity wave lost input alignment");
      dense_hashmap_t<u64, u64> headers;
      headers.reserve(identities.size());
      for (const auto& [candidate, header] : identities) {
        headers.emplace(candidate.raw_address, header);
      }
      bool acquired = false;
      for (size_t index = 0; index < candidates.size(); ++index) {
        const RemotePtr candidate = candidates[index];
        const auto header = headers.find(candidate.raw_address);
        const bool eligible = states[index] == StableNodeSnapshotState::stable &&
          header != headers.end() &&
          stage2_parent_is_stable(
            header->second, (header->second & VamanaNode::HEADER_DELETED) != 0);
        if (candidate == parent) {
          if (eligible || states[index] == StableNodeSnapshotState::retryable) {
            defer_stage2_item_retry(context, item, /*permanent=*/true);
            return failed("certificate_acquire_current");
          }
          continue;
        }
        if (!eligible) continue;
        if (std::find(task.stage2_neighbors.begin(), task.stage2_neighbors.end(),
                      candidate) == task.stage2_neighbors.end() &&
            !publish_stage2_recovery_parent(task, candidate)) {
          continue;
        }
        const bool candidate_acquired = acquire(candidate);
        stage2_replace_certificate_parent(
          candidate, task.stage2_promotion_parent,
          task.stage2_replaced_certificate_parents);
        parent = candidate;
        if (candidate_acquired) {
          acquired = true;
          break;
        }
      }
      if (!acquired) {
        defer_stage2_item_retry(context, item, /*permanent=*/true);
        return failed("certificate_acquire_fallback");
      }
    }

    Stage2PhaseAttemptTimer placement_timer(
      node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::placement_authority)], 1);
    if (!task.placement_committed) {
      const protocol::AuthorityPlacementItem placement{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .id = task.id,
        .generation = task.generation,
        .expected_raw = task.target.raw_address,
        .desired_raw = task.final_target.raw_address,
        .expected_placement_version = task.initial_placement_version,
      };
      protocol::AuthorityPlacementResult result;
      if (!node_.relocate_via_authority(
            task.authority_shard, placement, result, config)) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_item_retry(context, item, /*permanent=*/true);
        return false;
      }
      const auto status = static_cast<protocol::AuthorityPlacementStatus>(
        result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_item_retry(context, item);
        return false;
      }
      if (status == protocol::AuthorityPlacementStatus::stale) {
        if (!complete_stale_stage2(task)) {
          defer_stage2_item_retry(context, item, /*permanent=*/true);
          return false;
        }
        task.maintenance_sequence = 0;
        return true;
      }
      lib_assert(
        status == protocol::AuthorityPlacementStatus::committed ||
          status == protocol::AuthorityPlacementStatus::replay,
        "authority rejected a structurally valid Stage2 placement token");
      const u64 expected_version = task.final_target == task.target
        ? task.initial_placement_version : task.initial_placement_version + 1;
      lib_assert(result.resulting_placement_version == expected_version,
                 "authority placement returned an unexpected version");
      task.placement_committed = true;
    }

    if (!task.centroid_committed) {
      const CentroidMembershipOp add = make_centroid_membership_op(
        task, task.final_target, CentroidMembershipKind::add);
      if (!node_.apply_centroid_membership_fanout_and_wait(
            span<const CentroidMembershipOp>{&add, 1}, config)) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_item_retry(context, item, /*permanent=*/true);
        return false;
      }
      task.centroid_committed = true;
    }

    if (task.final_target != task.target && !task.allocation_settled) {
      lib_assert(migrated_source_tombstone_allowed(
                   task.placement_committed, task.centroid_committed),
                 "migrated source tombstoned before final centroid publication");
      const u64 source_header = node_.load_local_node_header_acquire(task.target);
      lib_assert((source_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0,
                 "Stage1 source was counted before final placement");
      (void)node_.mark_node_deleted(task.target, task.generation);
      if (!settle_dynamic_allocation(task)) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_item_retry(context, item, /*permanent=*/true);
        return false;
      }
      task.allocation_settled = true;
    }
    if (config.storage_owner_lease_merge) {
      const vec<RemotePtr> parents = stage2_certificate_release_parents(
        task.stage2_promotion_parent,
        span<const RemotePtr>{task.stage2_replaced_certificate_parents});
      trace("release");
      vec<ReconcileReverseOp> releases;
      releases.reserve(parents.size());
      for (const RemotePtr release_parent : parents) {
        releases.push_back(make_reconcile_op(
          task, release_parent, RemotePtr{}, task.final_target,
          ReconcileReverseOpKind::release_certificate_lease));
      }
      if (!apply_reconcile_ops(span<const ReconcileReverseOp>{releases})) {
        return failed("certificate_release");
      }
      task.certificate_released = true;
    }
    return true;
  };




void Stage2Worker::reset_reconcile_for_semantic_retry(Stage2Context& context) {
    cancel_context_reconcile(context);
    context.item_pipeline_started = false;
    for (const size_t item : context.reconcile_items) {
      context.finalize_subphases[item] = Stage2FinalizeSubphase::prepare;
      if (context.tasks[item].maintenance_sequence != 0) {
        defer_stage2_item_retry(context, item, /*permanent=*/true);
      } else {
        context.finalize_subphases[item] = Stage2FinalizeSubphase::complete;
      }
    }
    context.reconcile_items.clear();
    node_.storage_owner_stage2_reconcile_retries_.fetch_add(
      1, std::memory_order_relaxed);
    node_.storage_owner_maintenance_pressure_yields_.fetch_add(
      1, std::memory_order_relaxed);
  };




bool Stage2Worker::start_reconcile_barrier(Stage2Context& context) {
    lib_assert(!context.reconcile_batch.active(),
               "Stage2 batch barrier restarted before its ACK fence");
    context.reconcile_items.clear();
    Stage2ReconcileBarrier barrier = Stage2ReconcileBarrier::none;
    vec<ReconcileReverseOp> ops;
    const auto now = std::chrono::steady_clock::now();
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0 || task.retry_not_before > now ||
          context.finalize_subphases[item] ==
            Stage2FinalizeSubphase::placement_ready ||
          context.finalize_subphases[item] == Stage2FinalizeSubphase::complete) {
        continue;
      }
      const Stage2ReconcileBarrier item_barrier =
        !task.stage2_promotion_committed
        ? Stage2ReconcileBarrier::install
        : (!task.reverse_reconciled
             ? Stage2ReconcileBarrier::removal
             : Stage2ReconcileBarrier::none);
      if (item_barrier == Stage2ReconcileBarrier::none) continue;
      if (barrier == Stage2ReconcileBarrier::none) {
        barrier = item_barrier;
      } else if (barrier != item_barrier) {
        continue;
      }
      if (!build_reconcile_barrier_ops(context, item, barrier, ops)) {
        reset_reconcile_for_semantic_retry(context);
        return false;
      }
      context.reconcile_items.push_back(item);
    }
    if (barrier == Stage2ReconcileBarrier::none) return false;
    lib_assert(!context.reconcile_items.empty(),
               "Stage2 batch barrier has no task");
    if (barrier == Stage2ReconcileBarrier::install) {
      order_stage2_install_ops(ops);
    }
    if (!begin_reconcile_barrier(
          context, barrier, span<const ReconcileReverseOp>{ops})) {
      reset_reconcile_for_semantic_retry(context);
      return false;
    }
    return true;
  };




bool Stage2Worker::advance_reconcile_pipeline(Stage2Context& context) {
    const ReconcilePollResult poll = poll_reconcile_barrier(context);
    if (poll == ReconcilePollResult::waiting) return false;
    if (poll == ReconcilePollResult::retry) {
      reset_reconcile_for_semantic_retry(context);
      return false;
    }

    const Stage2ReconcileBarrier completed = context.reconcile_batch.barrier();
    if (completed == Stage2ReconcileBarrier::none) {
      reset_reconcile_for_semantic_retry(context);
      return false;
    }
    const Stage2ReconcileBarrier next = stage2_reconcile_next_barrier(
      completed, config.storage_owner_lease_merge);
    cancel_context_reconcile(context);
    for (const size_t item : context.reconcile_items) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (completed == Stage2ReconcileBarrier::install) {
        task.stage2_promotion_committed = true;
        if (!config.storage_owner_lease_merge) {
          task.reverse_reconciled = true;
          task.certificate_released = true;
        }
      } else if (completed == Stage2ReconcileBarrier::removal) {
        task.reverse_reconciled = true;
      } else {
        task.certificate_released = true;
      }
      context.finalize_subphases[item] = next == Stage2ReconcileBarrier::none
        ? Stage2FinalizeSubphase::placement_ready
        : Stage2FinalizeSubphase::prepare;
    }
    context.reconcile_items.clear();
    return true;
  };




}  // namespace memory_node_storage_owner_maintenance_detail

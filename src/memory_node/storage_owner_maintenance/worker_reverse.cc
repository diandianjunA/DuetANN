#include "memory_node/storage_owner_maintenance/worker_internal.hh"

#include <cstdlib>
#include <mutex>
#include <syncstream>
#include <unordered_map>

namespace memory_node_storage_owner_maintenance_detail {

u64 Stage2Worker::reverse_retry_backoff(u32 attempts) {
  return reverse_retry_backoff_ns(
    attempts, retry_backoff_ns, reverse_retry_backoff_max_ns);
}

void Stage2Worker::release_context_lane(Stage2Context& context) {
    if (!context.search_lane.has_value()) return;
    const u32 lane = *context.search_lane;
    lib_assert(search_lanes.owns(lane, context.handle),
               "stage2 context lost ownership of its search lane");
    const bool rdma_ready = thread.is_ready(lane);
    lib_assert(rdma_ready,
               "stage2 context released scratch with RDMA still in flight");
    lib_assert(search_lanes.release(lane, context.handle, rdma_ready),
               "stage2 search lane release violated context generation");
    context.search_lane.reset();
    node_.release_storage_owner_search_lane_lease();
  };




bool Stage2Worker::bind_context_lane(Stage2Context& context) {
    const auto debug_failure = [&](const char* reason) {
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") == nullptr ||
          context.tasks.empty() ||
          std::chrono::steady_clock::now() - context.tasks.front().queued_at <=
            std::chrono::seconds(5)) {
        return;
      }
      static std::atomic<u32> debug_count{0};
      if (debug_count.fetch_add(1, std::memory_order_relaxed) >= 64) return;
      std::osyncstream(std::cerr) << "[stage2-lane-bind-failure] reason=" << reason
                << " worker=" << worker_id_
                << " first_id=" << context.tasks.front().id
                << " active_leases=" << node_.storage_owner_search_lane_leases_.load(
                  std::memory_order_acquire)
                << " limit=" << node_.storage_owner_search_lane_lease_limit_.load(
                  std::memory_order_acquire) << std::endl;
    };
    if (!context.search_lane.has_value()) {
      if (!node_.try_acquire_storage_owner_search_lane_lease(
            context.search_lane_wait_registered)) {
        debug_failure("global_lease");
        return false;
      }
      const auto lane = search_lanes.try_acquire(context.handle);
      if (!lane.has_value()) {
        debug_failure("local_lane");
        // With one physical lane per bounded local context node_ is reachable
        // only if ownership bookkeeping is inconsistent. Keep the release
        // path defensive so an unexpected retry cannot leak the node-wide
        // lease and starve every other worker.
        node_.release_storage_owner_search_lane_lease();
        return false;
      }
      context.search_lane = *lane;
    }
    lib_assert(search_lanes.owns(*context.search_lane, context.handle),
               "stage2 search lane belongs to another context generation");
    thread.set_current_coroutine(*context.search_lane);
    return true;
  };




bool Stage2Worker::release_rebindable_context_lane(Stage2Context& context) {
    if (!context.search_lane.has_value()) return true;
    const u32 lane = *context.search_lane;
    lib_assert(search_lanes.owns(lane, context.handle),
               "stage2 rebind check observed a foreign search lane");
    if (!stage2_search_lane_rebindable(
          thread.is_ready(lane), context.search_io.scratch_rebindable())) {
      return false;
    }
    release_context_lane(context);
    return true;
  };




void Stage2Worker::reset_context(Stage2Context& context) {
    if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
        context.active && !context.tasks.empty()) {
      std::osyncstream(std::cerr) << "[stage2-reset] worker=" << worker_id_
                << " slot=" << context.handle.slot
                << " generation=" << context.handle.generation
                << " first_sequence=" << context.tasks.front().maintenance_sequence
                << " tasks=" << context.tasks.size()
                << " input_prepared=" << context.search_input_prepared
                << std::endl;
    }
    auto* const previous_waiter_registrations =
      MemoryNode::current_storage_owner_maintenance_waiter_registrations_;
    MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
      &context.waiter_registrations;
    lib_assert(!context.search_lane.has_value(),
               "stage2 context reset before releasing its search lane");
    node_.cancel_storage_owner_search_lane_waiter(
      context.search_lane_wait_registered);
    node_.clear_all_current_storage_owner_maintenance_waiters();
    for (const Stage2ReconcileChunk& chunk :
         context.reconcile_batch.chunks()) {
      if (!chunk.complete && chunk.request_id != 0) {
        node_.cancel_peer_rpc_response(chunk.request_id);
      }
    }
    for (size_t rpc_index = 0;
         rpc_index < context.search_io.home_expand_rpc_count; ++rpc_index) {
      const Stage2HomeExpandRpc& rpc =
        context.search_io.home_expand_rpcs[rpc_index];
      if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
        node_.cancel_peer_rpc_response(rpc.request_id);
      }
    }
    for (size_t rpc_index = 0;
         rpc_index < context.search_io.score_home_rpc_count; ++rpc_index) {
      const Stage2HomeExpandRpc& rpc =
        context.search_io.score_home_rpcs[rpc_index];
      if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
        node_.cancel_peer_rpc_response(rpc.request_id);
      }
    }
    // Cancel every transport member while its exact OwnerKey is still live.
    // A concurrent aggregate completion can then either publish to node_
    // context or observe the cancellation; it can never target a reused slot.
    if (context.ready_owner.token != 0) {
      lib_assert(ready_context_queue->deactivate(context.ready_owner),
                 "stage2 context ready owner was stale at reset");
      context.ready_owner = {};
    }
    context.search_io.reset();
    context.reconcile_batch.clear();
    context.reconcile_items.clear();
    std::fill(context.finalize_subphases.begin(),
              context.finalize_subphases.end(),
              Stage2FinalizeSubphase::prepare);
    context.active = false;
    context.tasks.clear();
    context.targets.clear();
    context.targets.resize(context.tasks.size());
    for (vec<RemotePtr>& candidates :
         context.continued_candidates_by_task) {
      candidates.clear();
    }
    for (vec<RemotePtr>& neighbors :
         context.live_stage2_neighbors_by_task) {
      neighbors.clear();
    }
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }
    std::fill(context.reverse_request_ids.begin(),
              context.reverse_request_ids.end(), 0);
    context.completion_handoff_started_ns = 0;
    context.search_input_prepared = false;
    context.search_timing_recorded = false;
    context.search_stall_diagnostic_emitted = false;
    context.search_started_ns = 0;
    context.reverse_prepare_started_ns = 0;
    context.reverse_prepare_timing_active = false;
    context.item_pipeline_started = false;
    MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
      previous_waiter_registrations;
  };




void Stage2Worker::materialize_stable_snapshot_wave(size_t task_count) {
    lib_assert(task_count <= snapshot_candidates_by_task.size(),
               "Stage2 snapshot wave exceeded worker batch capacity");
    snapshot_plan.build(span<const vec<RemotePtr>>{
      snapshot_candidates_by_task.data(), task_count});
    const size_t snapshot_count = node_.read_node_snapshots_batched_into(
      span<const RemotePtr>{snapshot_plan.targets}, config,
      snapshot_storage, "stage2_freeze_prune_wave", &snapshot_states);
    lib_assert(snapshot_states.size() == snapshot_plan.targets.size(),
               "Stage2 snapshot wave lost per-target read state");
    node_.storage_owner_stage2_prune_contention_excluded_.fetch_add(
      static_cast<u64>(std::count(
        snapshot_states.begin(), snapshot_states.end(),
        StableNodeSnapshotState::retryable)),
      std::memory_order_relaxed);
    // Pending candidates are eligible for graph quality, but not for the
    // independent reachability certificate preserved by the final prune.
    // This wave builds a RobustPrune candidate set; it is not a commit
    // barrier.  Requiring every independent candidate to be unlocked in the
    // same observation creates an all-or-nothing livelock on hub-heavy
    // graphs: one hot neighbor makes the complete semantic B8 discard all of
    // its successful snapshots forever.  A contended record is conservatively
    // ineligible for node_ prune snapshot, exactly like a record that retired
    // before the observation.  Reachability does not depend on retaining it:
    // the independently revalidated promotion certificate below is mandatory,
    // and the reconciliation receiver rechecks every selected parent while
    // holding its incarnation lock.
    snapshot_by_raw.clear();
    snapshot_by_raw.reserve(snapshot_count);
    for (size_t snapshot_index = 0; snapshot_index < snapshot_count;
         ++snapshot_index) {
      const NodeSnapshot& snapshot = snapshot_storage[snapshot_index];
      const bool pending = !snapshot.deleted &&
        (snapshot.header & (VamanaNode::HEADER_PROVISIONAL |
                            VamanaNode::HEADER_RETIRING |
                            VamanaNode::HEADER_STAGE2_FROZEN)) ==
          VamanaNode::HEADER_PROVISIONAL;
      if (!snapshot.deleted &&
          (VamanaNode::stable_graph_mutation_allowed(snapshot.header) || pending)) {
        snapshot_by_raw.emplace(snapshot.rptr.raw_address, &snapshot);
      }
    }

    if (snapshots_by_task.size() < task_count) {
      snapshots_by_task.resize(task_count);
    }
    for (size_t task = 0; task < task_count; ++task) {
      vec<const NodeSnapshot*>& snapshots = snapshots_by_task[task];
      snapshots.clear();
      snapshots.reserve(snapshot_plan.task_target_indices[task].size());
      for (const u32 target_index :
           snapshot_plan.task_target_indices[task]) {
        lib_assert(target_index < snapshot_plan.targets.size(),
                   "Stage2 snapshot wave task index is invalid");
        const RemotePtr target = snapshot_plan.targets[target_index];
        const auto found = snapshot_by_raw.find(target.raw_address);
        if (found != snapshot_by_raw.end()) snapshots.push_back(found->second);
      }
    }
  };




void Stage2Worker::materialize_stable_identity_wave(size_t task_count) {
    lib_assert(task_count <= snapshot_candidates_by_task.size(),
               "Stage2 identity wave exceeded worker batch capacity");
    snapshot_plan.build(span<const vec<RemotePtr>>{
      snapshot_candidates_by_task.data(), task_count});
    const size_t identity_count = node_.read_node_identity_headers_batched_into(
      span<const RemotePtr>{snapshot_plan.targets}, config,
      identity_storage, &identity_states);
    lib_assert(identity_states.size() == snapshot_plan.targets.size(),
               "Stage2 identity wave lost per-target read state");
    node_.storage_owner_stage2_parent_contention_excluded_.fetch_add(
      static_cast<u64>(std::count(
        identity_states.begin(), identity_states.end(),
        StableNodeSnapshotState::retryable)),
      std::memory_order_relaxed);
    // Parent selection is also conservative: a neighbor whose coherent
    // identity cannot be captured now must not hold the whole context behind
    // a global retry barrier.  Exclude it from node_ planner view and choose a
    // different live parent.  The receiver-side incarnation/lifecycle check
    // remains the authoritative publication boundary, so contention is never
    // mistaken for a successful edge mutation.
    stable_identity_targets.clear();
    stable_identity_targets.reserve(identity_count);
    for (size_t index = 0; index < identity_count; ++index) {
      const auto& [pointer, header] = identity_storage[index];
      if (stage2_parent_is_stable(
            header, (header & VamanaNode::HEADER_DELETED) != 0)) {
        stable_identity_targets.insert(pointer);
      }
    }
    static std::atomic<u32> identity_debug_count{0};
    if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
        identity_debug_count.fetch_add(1, std::memory_order_relaxed) < 12) {
      std::osyncstream(std::cerr) << "[stage2-identity-debug] worker=" << worker_id_
                << " targets=" << snapshot_plan.targets.size()
                << " identities=" << identity_count
                << " stable=" << stable_identity_targets.size() << std::endl;
      for (size_t index = 0; index < std::min<size_t>(4, identity_count); ++index) {
        const auto& [pointer, header] = identity_storage[index];
        std::osyncstream(std::cerr) << "[stage2-identity-debug] raw=" << pointer.raw_address
                  << " header=" << header
                  << " incarnation=" << pointer.incarnation()
                  << " eligible=" << stage2_parent_is_stable(
                    header, (header & VamanaNode::HEADER_DELETED) != 0)
                  << std::endl;
      }
      for (size_t task = 0; task < std::min<size_t>(2, task_count); ++task) {
        std::osyncstream(std::cerr) << "[stage2-identity-debug] task=" << task
                  << " live=" << live_stage2_neighbors_by_task[task].size()
                  << " final=" << snapshot_plan.task_target_indices[task].size()
                  << std::endl;
      }
    }
    for (size_t task = 0; task < task_count; ++task) {
      vec<RemotePtr>& live = live_stage2_neighbors_by_task[task];
      live.clear();
      live.reserve(snapshot_plan.task_target_indices[task].size());
      for (const u32 target_index :
           snapshot_plan.task_target_indices[task]) {
        lib_assert(target_index < snapshot_plan.targets.size(),
                   "Stage2 identity wave task index is invalid");
        const RemotePtr target = snapshot_plan.targets[target_index];
        if (stable_identity_targets.contains(target)) {
          live.push_back(target);
        }
      }
    }
  };




void Stage2Worker::record_finalized_live(
      std::chrono::steady_clock::time_point queued_at) {
    const u64 latency_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - queued_at).count());
    node_.storage_owner_maintenance_finalize_latency_ns_.fetch_add(
      latency_ns, std::memory_order_relaxed);
    node_.storage_owner_maintenance_finalize_latency_buckets_[
      finalize_latency_bucket(latency_ns)].fetch_add(
        1, std::memory_order_relaxed);
    atomic_utils::update_max_relaxed(
      node_.storage_owner_maintenance_finalize_max_latency_ns_, latency_ns);
    node_.storage_owner_maintenance_finalized_live_.fetch_add(
      1, std::memory_order_relaxed);
  };

  using ReconcileReverseOp = service::storage_owner::ReconcileReverseOp;
  using ReconcileReverseOpKind =
    service::storage_owner::ReconcileReverseOpKind;
  using CentroidMembershipOp =
    service::storage_owner::CentroidMembershipOp;
  using CentroidMembershipKind =
    service::storage_owner::CentroidMembershipKind;




ReconcileReverseOp Stage2Worker::make_reconcile_op(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr graph_target,
      RemotePtr old_candidate,
      RemotePtr new_candidate,
      ReconcileReverseOpKind kind) {
    return ReconcileReverseOp{
      .target_raw = graph_target.raw_address,
      .old_candidate_raw = old_candidate.raw_address,
      .new_candidate_raw = new_candidate.raw_address,
      .placement_sequence = task.maintenance_sequence,
      .id = task.id,
      .generation = task.generation,
      .kind = static_cast<u32>(kind),
    };
  };




CentroidMembershipOp Stage2Worker::make_centroid_membership_op(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr pointer,
      CentroidMembershipKind kind) {
    return CentroidMembershipOp{
      .node_raw = pointer.raw_address,
      .maintenance_sequence = task.maintenance_sequence,
      .id = task.id,
      .generation = task.generation,
      .kind = static_cast<u32>(kind),
    };
  };

  // One maintenance context emits at most one RPC stream per destination,
  // independent of the number of inserted nodes.  The peer helper posts all
  // chunks before waiting for ACKs, preserving cross-shard fanout parallelism.



bool Stage2Worker::apply_reconcile_ops(
      span<const ReconcileReverseOp> ops,
      const Configuration* prune_config, vec<u8>* retry_mask) {
    const auto debug_failure = [&](const char* reason,
                                   span<const ReconcileReverseOp> failed_ops) {
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") == nullptr) return;
      static std::mutex debug_mutex;
      static std::unordered_map<u64, u32> debug_counts;
      for (const auto& op : failed_ops) {
        u32 count = 0;
        {
          std::lock_guard<std::mutex> lock(debug_mutex);
          count = ++debug_counts[op.id];
        }
        if (count > 16 && (count & (count - 1)) != 0) continue;
        std::string message = "[reconcile-op-debug] reason=" +
          std::string(reason) + " worker=" + std::to_string(worker_id_) +
          " kind=" + std::to_string(op.kind) +
          " target=" + std::to_string(op.target_raw) +
          " old=" + std::to_string(op.old_candidate_raw) +
          " new=" + std::to_string(op.new_candidate_raw) +
          " id=" + std::to_string(op.id) +
          " generation=" + std::to_string(op.generation) +
          " sequence=" + std::to_string(op.placement_sequence);
        const RemotePtr target{op.target_raw};
        if (node_.local_shard(target.memory_node()) &&
            node_.storage_node_pointer_addressable(target)) {
          message += " target_valid=" + std::to_string(
            node_.valid_local_storage_node_pointer(target)) +
            " target_header=" + std::to_string(
              node_.load_local_node_header_acquire(target));
        }
        std::osyncstream(std::cerr) << message << std::endl;
      }
    };
    if (retry_mask != nullptr) retry_mask->assign(ops.size(), 0);
    const Configuration& active_config =
      prune_config == nullptr ? config : *prune_config;
    vec<ReconcileReverseOp> local_ops;
    vec<size_t> local_op_indices;
    dense_hashmap_t<u32, vec<ReconcileReverseOp>> remote_ops;
    dense_hashmap_t<u32, vec<size_t>> remote_op_indices;
    local_ops.reserve(ops.size());
    local_op_indices.reserve(ops.size());
    for (size_t index = 0; index < ops.size(); ++index) {
      const ReconcileReverseOp& op = ops[index];
      const RemotePtr target{op.target_raw};
      if (target.is_null() || target.memory_node() >= node_.num_storage_nodes_) {
        debug_failure("invalid_target", span<const ReconcileReverseOp>{&op, 1});
        return false;
      }
      if (node_.local_shard(target.memory_node())) {
        local_ops.push_back(op);
        local_op_indices.push_back(index);
      } else {
        remote_ops[target.memory_node()].push_back(op);
        remote_op_indices[target.memory_node()].push_back(index);
      }
    }

    const auto result_accepted = [&](size_t op_index,
                                     const ReconcileReverseOp& op,
                                     const protocol::ReconcileReverseResult& result,
                                     const char* failure_reason) {
      if (retry_mask != nullptr &&
          memory_node_storage_owner_index_detail::
            reconcile_reverse_refinement_retryable(op, result)) {
        (*retry_mask)[op_index] = 1;
        return true;
      }
      if (memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(op, result)) {
        return true;
      }
      debug_failure(failure_reason,
                    span<const ReconcileReverseOp>{&op, 1});
      return false;
    };

    vec<service::storage_owner::ReconcileReverseResult> local_results;
    const bool local_succeeded = local_ops.empty() ||
      node_.reconcile_local_reverse_ops(
        span<const ReconcileReverseOp>{local_ops}, active_config, local_results);
    if (!local_succeeded && retry_mask == nullptr) {
      debug_failure("local_apply", span<const ReconcileReverseOp>{local_ops});
      return false;
    }
    lib_assert(local_results.size() == local_ops.size(),
               "local reconciliation lost per-operation results");
    for (size_t index = 0; index < local_ops.size(); ++index) {
      if (!result_accepted(local_op_indices[index], local_ops[index],
                           local_results[index], "local_postcondition")) {
        return false;
      }
    }
    if (remote_ops.empty()) return true;
    vec<service::storage_owner::ReconcileReverseResult> remote_results;
    const bool succeeded = node_.send_reconcile_reverse_fanout_and_wait(
      remote_ops, remote_results, active_config);
    if (!succeeded) {
      for (const auto& [target_shard, peer_ops] : remote_ops) {
        (void)target_shard;
        debug_failure("remote_apply", span<const ReconcileReverseOp>{peer_ops});
      }
      return false;
    }
    size_t remote_result_offset = 0;
    for (const auto& [target_shard, peer_ops] : remote_ops) {
      const auto& op_indices = remote_op_indices.find(target_shard)->second;
      for (size_t index = 0; index < peer_ops.size(); ++index) {
        lib_assert(remote_result_offset < remote_results.size(),
                   "remote reconciliation lost per-operation results");
        if (!result_accepted(op_indices[index], peer_ops[index],
                             remote_results[remote_result_offset++],
                             "remote_postcondition")) {
          return false;
        }
      }
    }
    lib_assert(remote_result_offset == remote_results.size(),
               "remote reconciliation returned unexpected result count");
    return true;
  };




Stage2Worker::CleanupParentQuiesceResult Stage2Worker::quiesce_cleanup_parent(
      StorageOwnerMaintenanceTask& task) {
    if (task.cleanup_repair_only || task.cleanup_retiring) {
      return CleanupParentQuiesceResult::ready;
    }
    lib_assert(!task.target.is_null() &&
                 node_.local_shard(task.target.memory_node()) &&
                 node_.storage_node_pointer_addressable(task.target),
               "cleanup parent is not an addressable local physical node");

    const IncarnationLockResult target_lock = node_.try_lock_node(task.target);
    if (target_lock == IncarnationLockResult::busy) {
      return CleanupParentQuiesceResult::busy;
    }
    if (target_lock == IncarnationLockResult::stale) {
      // Reuse is fenced behind completion of the original cleanup sequence.
      // Observing a different incarnation therefore means a duplicate/late
      // descriptor whose old-incarnation postcondition is already durable.
      return CleanupParentQuiesceResult::stale;
    }
    const u64 header = node_.load_local_node_header_acquire(task.target);
    const byte_t* record = node_.index_buffer_.get_full_buffer() +
      task.target.byte_offset();
    const node_t observed_id = *reinterpret_cast<const node_t*>(
      record + VamanaNode::offset_id());
    const u32 observed_generation = *reinterpret_cast<const u32*>(
      record + VamanaNode::offset_generation());
    if (VamanaNode::header_incarnation(header) !=
          task.target.incarnation() ||
        observed_id != task.id || observed_generation != task.generation) {
      node_.unlock_node(task.target);
      return CleanupParentQuiesceResult::stale;
    }

    GraphAdjacency adjacency;
    if (!node_.read_graph_adjacency(task.target, adjacency)) {
      node_.unlock_node(task.target);
      return CleanupParentQuiesceResult::busy;
    }
    if ((header & VamanaNode::HEADER_DELETED) != 0 ||
        adjacency.deleted) {
      // An idempotent duplicate can observe the postcondition produced by the
      // original cleanup. It must not attempt to reparent from a tombstone.
      task.cleanup_retiring = true;
      task.cleanup_protected_reparented = true;
      node_.unlock_node(task.target);
      return CleanupParentQuiesceResult::ready;
    }

    auto* header_ptr = reinterpret_cast<u64*>(
      node_.index_buffer_.get_full_buffer() +
      vamana::StorageLayoutResolver::header(task.target).offset);
    std::atomic_ref<u64>(*header_ptr).fetch_or(
      static_cast<u64>(VamanaNode::HEADER_RETIRING),
      std::memory_order_acq_rel);
    task.cleanup_protected_children = adjacency.provisional;
    task.cleanup_replacement_parents.assign(
      task.cleanup_protected_children.size(), RemotePtr{});
    task.cleanup_retiring = true;
    node_.unlock_node(task.target);
    return CleanupParentQuiesceResult::ready;
  };




bool Stage2Worker::reparent_cleanup_children(
      StorageOwnerMaintenanceTask& task) {
    if (task.cleanup_repair_only || task.cleanup_protected_reparented) {
      return true;
    }
    lib_assert(task.cleanup_retiring,
               "protected-child reparent ran before parent quiescence");
    lib_assert(task.cleanup_protected_children.size() ==
                 task.cleanup_replacement_parents.size(),
               "protected-child replacement state lost correlation");

    vec<ReconcileReverseOp> reservations;
    reservations.reserve(task.cleanup_protected_children.size());
    for (size_t child_index = 0;
         child_index < task.cleanup_protected_children.size();
         ++child_index) {
      const RemotePtr child = task.cleanup_protected_children[child_index];
      NodeSnapshot child_snapshot;
      if (!node_.read_node_snapshot(child, child_snapshot)) {
        // A reused tagged slot is no longer this protected child. A
        // same-incarnation lock is only transient, so retain it and retry.
        u64 child_header = 0;
        if (node_.local_shard(child.memory_node())) {
          if (!node_.valid_local_storage_node_pointer(child)) continue;
          child_header = node_.load_local_node_header_acquire(child);
        } else if (child.memory_node() < node_.num_storage_nodes_) {
          const auto address =
            vamana::StorageLayoutResolver::header(child);
          if (address.offset > node_.mn_memory_bytes_ ||
              sizeof(child_header) > node_.mn_memory_bytes_ - address.offset) {
            continue;
          }
          node_.remote_read_bytes(child.memory_node(), address.offset,
                            &child_header, sizeof(child_header), 0);
        } else {
          continue;
        }
        if (VamanaNode::header_incarnation(child_header) ==
            child.incarnation()) {
          return false;
        }
        continue;
      }
      if (child_snapshot.deleted) continue;

      GraphAdjacency child_adjacency;
      if (!node_.read_graph_adjacency(child, child_adjacency) ||
          child_adjacency.deleted) {
        return false;
      }
      const vec<RemotePtr> candidates =
        order_protected_reparent_candidates(
          child, task.target,
          span<const RemotePtr>{child_adjacency.stable});

      RemotePtr& replacement =
        task.cleanup_replacement_parents[child_index];
      const auto usable_replacement = [&](RemotePtr candidate) {
        if (candidate.is_null() || candidate == task.target ||
            candidate == child) {
          return false;
        }
        NodeSnapshot candidate_snapshot;
        if (!node_.read_node_snapshot(candidate, candidate_snapshot) ||
            candidate_snapshot.deleted ||
            !VamanaNode::stable_graph_mutation_allowed(
              candidate_snapshot.header)) {
          return false;
        }
        if (config.storage_owner_lease_merge) {
          return (candidate_snapshot.header &
                  VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
        }
        GraphAdjacency candidate_adjacency;
        if (!node_.read_graph_adjacency(candidate, candidate_adjacency) ||
            candidate_adjacency.deleted) {
          return false;
        }
        if (protected_reparent_target_has_capacity(
              child,
              span<const RemotePtr>{candidate_adjacency.provisional},
              VamanaNode::provisional_slots())) {
          return true;
        }
        for (const RemotePtr protected_child :
             candidate_adjacency.provisional) {
          NodeSnapshot protected_snapshot;
          if (node_.read_node_snapshot(protected_child, protected_snapshot)) {
            if (protected_snapshot.deleted) return true;
            continue;
          }
          u64 protected_header = 0;
          if (node_.local_shard(protected_child.memory_node())) {
            if (!node_.valid_local_storage_node_pointer(protected_child)) {
              return true;
            }
            protected_header =
              node_.load_local_node_header_acquire(protected_child);
          } else if (protected_child.memory_node() < node_.num_storage_nodes_) {
            const auto protected_address =
              vamana::StorageLayoutResolver::header(protected_child);
            if (protected_address.offset > node_.mn_memory_bytes_ ||
                sizeof(protected_header) >
                  node_.mn_memory_bytes_ - protected_address.offset) {
              return true;
            }
            node_.remote_read_bytes(
              protected_child.memory_node(), protected_address.offset,
              &protected_header, sizeof(protected_header), 0);
          } else {
            return true;
          }
          if (VamanaNode::header_incarnation(protected_header) !=
              protected_child.incarnation()) {
            return true;
          }
        }
        return false;
      };

      if (!usable_replacement(replacement)) {
        if (!replacement.is_null() &&
            std::find(task.cleanup_replaced_certificate_parents.begin(),
                      task.cleanup_replaced_certificate_parents.end(),
                      replacement) ==
              task.cleanup_replaced_certificate_parents.end()) {
          task.cleanup_replaced_certificate_parents.push_back(replacement);
        }
        replacement.reset();
        for (const RemotePtr candidate : candidates) {
          if (usable_replacement(candidate)) {
            replacement = candidate;
            break;
          }
        }
      }
      if (replacement.is_null()) {
        static std::atomic<u32> debug_count{0};
        const auto task_age =
          std::chrono::steady_clock::now() - task.queued_at;
        if ((std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr ||
             task_age > std::chrono::seconds(60)) &&
            debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::osyncstream(std::cerr) << "[cleanup-reparent-debug] self=" << node_.storage_id_
                    << " worker=" << worker_id_ << " reason=no_replacement"
                    << " id=" << task.id << " generation=" << task.generation
                    << " sequence=" << task.maintenance_sequence
                    << " child=" << child.raw_address
                    << " stable_neighbors=" << child_adjacency.stable.size()
                    << " candidates=" << candidates.size() << std::endl;
          static std::atomic<u32> candidate_debug_count{0};
          for (const RemotePtr candidate : candidates) {
            if (candidate_debug_count.fetch_add(1, std::memory_order_relaxed) >=
                128) {
              break;
            }
            NodeSnapshot candidate_snapshot;
            const bool snapshot_read =
              node_.read_node_snapshot(candidate, candidate_snapshot);
            std::osyncstream(std::cerr)
              << "[cleanup-replacement-debug] self=" << node_.storage_id_
              << " worker=" << worker_id_ << " id=" << task.id
              << " child=" << child.raw_address
              << " candidate=" << candidate.raw_address
              << " snapshot_read=" << snapshot_read
              << " header=" << (snapshot_read ? candidate_snapshot.header : 0)
              << " deleted=" << (snapshot_read && candidate_snapshot.deleted)
              << " stable=" << (snapshot_read &&
                VamanaNode::stable_graph_mutation_allowed(
                  candidate_snapshot.header))
              << " accounted=" << (snapshot_read &&
                (candidate_snapshot.header &
                 VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0)
              << std::endl;
          }
        }
        return false;
      }
      reservations.push_back(ReconcileReverseOp{
        .target_raw = replacement.raw_address,
        .old_candidate_raw = 0,
        .new_candidate_raw = child.raw_address,
        .placement_sequence = task.maintenance_sequence,
        .id = child_snapshot.id,
        .generation = child_snapshot.generation,
        .kind = static_cast<u32>(config.storage_owner_lease_merge
          ? ReconcileReverseOpKind::acquire_certificate_lease
          : ReconcileReverseOpKind::ensure_reachable),
      });
    }

    if (!apply_reconcile_ops(span<const ReconcileReverseOp>{reservations})) {
      return false;
    }
    task.cleanup_protected_reparented = true;
    return true;
  };




bool Stage2Worker::publish_stage2_recovery_parent(
      StorageOwnerMaintenanceTask& task, RemotePtr recovery_parent,
      vec<RemotePtr>* live_neighbors) {
    if (recovery_parent.is_null() || recovery_parent == task.final_target) {
      return false;
    }
    lib_assert(node_.storage_node_pointer_addressable(task.final_target),
               "Stage2 recovery target is structurally invalid");
    const IncarnationLockResult final_lock =
      node_.try_lock_node(task.final_target);
    if (final_lock != IncarnationLockResult::locked) return false;

    u64 final_header = 0;
    node_t final_id = 0;
    u32 final_generation = 0;
    GraphAdjacency adjacency;
    const bool final_current =
      node_.read_locked_node_identity(
        task.final_target, final_header, final_id, final_generation) &&
      final_id == task.id && final_generation == task.generation &&
      (final_header & (VamanaNode::HEADER_DELETED |
                       VamanaNode::HEADER_PROVISIONAL |
                       VamanaNode::HEADER_RETIRING)) == 0 &&
      node_.read_graph_adjacency(task.final_target, adjacency) &&
      !adjacency.deleted && adjacency.generation == task.generation;
    if (!final_current) {
      lib_assert(node_.publish_locked_node_header(
                   task.final_target, final_header, 0, 0),
                 "failed to release invalid Stage2 recovery target");
      return false;
    }

    vec<RemotePtr> candidates = adjacency.stable;
    candidates.erase(
      std::remove_if(
        candidates.begin(), candidates.end(),
        [&](RemotePtr candidate) {
          return candidate.is_null() || candidate == task.final_target ||
            !node_.storage_node_pointer_addressable(candidate);
        }),
      candidates.end());
    if (std::find(candidates.begin(), candidates.end(), recovery_parent) ==
        candidates.end()) {
      candidates.push_back(recovery_parent);
    }
    vec<StableNodeSnapshotState> recovery_states;
    const vec<NodeSnapshot> snapshots = node_.read_node_snapshots_batched(
      candidates, config, "stage2_recovery_outgoing", &recovery_states);
    hashset_t<RemotePtr> eligible;
    eligible.reserve(snapshots.size());
    for (const NodeSnapshot& snapshot : snapshots) {
      if (snapshot.rptr != task.final_target &&
          stage2_parent_is_stable(snapshot.header, snapshot.deleted)) {
        eligible.insert(snapshot.rptr);
      }
    }

    if (live_neighbors != nullptr) {
      live_neighbors->clear();
      for (const RemotePtr neighbor : adjacency.stable) {
        if (eligible.contains(neighbor)) live_neighbors->push_back(neighbor);
      }
      if (!live_neighbors->empty()) {
        lib_assert(node_.publish_locked_node_header(
                     task.final_target, final_header, 0, 0),
                   "failed to release Stage2 recovery target with live parents");
        return true;
      }
    }
    if (!eligible.contains(recovery_parent)) {
      lib_assert(node_.publish_locked_node_header(
                   task.final_target, final_header, 0, 0),
                 "failed to release ineligible Stage2 recovery target");
      return false;
    }
    if (std::find(adjacency.stable.begin(), adjacency.stable.end(),
                  recovery_parent) == adjacency.stable.end()) {
      if (adjacency.stable.size() < config.R) {
        adjacency.stable.push_back(recovery_parent);
      } else {
        const auto stale = std::find_if(
          adjacency.stable.begin(), adjacency.stable.end(),
          [&](RemotePtr neighbor) { return !eligible.contains(neighbor); });
        if (stale == adjacency.stable.end()) {
          lib_assert(node_.publish_locked_node_header(
                       task.final_target, final_header, 0, 0),
                     "failed to release saturated Stage2 recovery target");
          return false;
        }
        *stale = recovery_parent;
      }
      node_.write_graph_adjacency(
        task.final_target, adjacency.stable, adjacency.provisional,
        task.generation, false);
      task.stage2_neighbors = adjacency.stable;
    }
    lib_assert(node_.publish_locked_node_header(
                 task.final_target, final_header, 0, 0),
               "failed to publish Stage2 recovery outgoing graph");
    if (live_neighbors != nullptr) {
      live_neighbors->push_back(recovery_parent);
    }
    return true;
  };


bool Stage2Worker::append_stage2_reconcile_ops(
      StorageOwnerMaintenanceTask& task,
      span<const RemotePtr> live_final_neighbors,
      vec<ReconcileReverseOp>& promotion_ops,
      vec<ReconcileReverseOp>& stable_ops,
      vec<ReconcileReverseOp>& removal_ops) {
    const auto plan = plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{task.stage1_backlink_targets},
      live_final_neighbors,
      task.stage2_promotion_committed
        ? task.stage2_promotion_parent : RemotePtr{});
    if (plan.promotion_target.is_null()) {
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
          std::chrono::steady_clock::now() - task.queued_at >
            std::chrono::seconds(5)) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::osyncstream(std::cerr) << "[stage2-reconcile-stall] reason=no_promotion_parent"
                    << " id=" << task.id
                    << " sequence=" << task.maintenance_sequence
                    << " final_neighbors=" << task.stage2_neighbors.size()
                    << " live_neighbors=" << live_final_neighbors.size()
                    << " stage1_bridges=" << task.stage1_backlink_targets.size()
                    << std::endl;
        }
      }
      return false;
    }

    static std::atomic<u32> planner_debug_count{0};
    if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
        planner_debug_count.fetch_add(1, std::memory_order_relaxed) < 12) {
      std::osyncstream(std::cerr) << "[stage2-planner-debug] id=" << task.id
                << " final=" << task.stage2_neighbors.size()
                << " live=" << live_final_neighbors.size()
                << " promotion=" << plan.promotion_target.raw_address
                << " ordinary=" << plan.ordinary_stable_targets.size()
                << std::endl;
    }
    stage2_replace_certificate_parent(
      plan.promotion_target, task.stage2_promotion_parent,
      task.stage2_replaced_certificate_parents);
    task.stage2_promotion_parent = plan.promotion_target;

    promotion_ops.push_back(make_reconcile_op(
      task, plan.promotion_target,
      plan.promotion_consumes_stage1_bridge ? task.target : RemotePtr{},
      task.final_target,
      stage2_reconcile_certificate_kind(config.storage_owner_lease_merge)));
    stable_ops.reserve(
      stable_ops.size() + plan.ordinary_stable_targets.size());
    const bool defer_ordinary = storage_owner_consolidation_enabled();
    for (const auto& target : plan.ordinary_stable_targets) {
      const bool replace = target.had_stage1_bridge &&
        task.final_target != task.target;
      if (defer_ordinary && replace) continue;
      if (replace && config.storage_owner_lease_merge) {
        stable_ops.push_back(make_reconcile_op(
          task, target.target, task.target, RemotePtr{},
          ReconcileReverseOpKind::remove_if_present));
      }
      stable_ops.push_back(make_reconcile_op(
        task, target.target,
        replace && !config.storage_owner_lease_merge
          ? task.target : RemotePtr{},
        task.final_target,
        replace && !config.storage_owner_lease_merge
          ? ReconcileReverseOpKind::replace_or_add
          : defer_ordinary
            ? ReconcileReverseOpKind::add_if_capacity
            : ReconcileReverseOpKind::add));
    }
    if (config.storage_owner_lease_merge) {
      removal_ops.reserve(
        removal_ops.size() + task.stage1_backlink_targets.size());
      hashset_t<RemotePtr> removal_targets;
      for (const RemotePtr target : task.stage1_backlink_targets) {
        if (!target.is_null() && removal_targets.insert(target).second) {
          removal_ops.push_back(make_reconcile_op(
            task, target, task.target, RemotePtr{},
            ReconcileReverseOpKind::remove_provisional_if_present));
        }
      }
    } else {
      removal_ops.reserve(
        removal_ops.size() + plan.obsolete_stage1_bridges.size());
      for (const RemotePtr target : plan.obsolete_stage1_bridges) {
        removal_ops.push_back(make_reconcile_op(
          task, target, task.target, RemotePtr{},
          ReconcileReverseOpKind::remove_if_present));
      }
    }
    return true;
  };




bool Stage2Worker::remove_candidate_backlinks(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr candidate,
      span<const RemotePtr> targets) {
    if (candidate.is_null() || targets.empty()) return true;
    hashset_t<RemotePtr> unique_targets;
    unique_targets.reserve(targets.size());
    vec<ReconcileReverseOp> ops;
    ops.reserve(targets.size());
    for (const RemotePtr target : targets) {
      if (target.is_null() || !unique_targets.insert(target).second) continue;
      ops.push_back(make_reconcile_op(
        task, target, candidate, RemotePtr{},
        ReconcileReverseOpKind::remove_if_present));
    }
    return apply_reconcile_ops(span<const ReconcileReverseOp>{ops});
  };




bool Stage2Worker::settle_dynamic_allocation(
      const StorageOwnerMaintenanceTask& task) {
    if (task.final_target.is_null() ||
        task.final_target == task.target) {
      return true;
    }
    const protocol::DynamicNodeControlItem settlement{
      .token = {
        .source_client = task.source_client,
        .item_index = task.operation_item_index,
        .client_batch_id = task.operation_batch_id,
      },
      .node_raw = task.target.raw_address,
      .allocated_raw = task.final_target.raw_address,
      .id = task.id,
      .generation = task.generation,
      .authority_shard = task.authority_shard,
      .action = static_cast<u32>(
        protocol::DynamicNodeControlAction::settle_allocation),
    };
    protocol::DynamicNodeControlResult result;
    if (!node_.control_dynamic_node_on_shard(
          task.final_target.memory_node(), settlement, result, config)) {
      return false;
    }
    return static_cast<protocol::DynamicNodeControlStatus>(result.status) ==
      protocol::DynamicNodeControlStatus::ok;
  };




bool Stage2Worker::complete_stale_stage2(StorageOwnerMaintenanceTask& task,
                                         const char** failure_reason) {
    const auto failed = [&](const char* reason, int status = -1) {
      if (failure_reason != nullptr) *failure_reason = reason;
      static std::atomic<u32> debug_count{0};
      const auto task_age =
        std::chrono::steady_clock::now() - task.queued_at;
      if (((std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
            task_age > std::chrono::seconds(5)) ||
           task_age > std::chrono::seconds(60)) &&
          debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        std::osyncstream(std::cerr) << "[stage2-stale-debug] self=" << node_.storage_id_
                  << " id=" << task.id
                  << " sequence=" << task.maintenance_sequence
                  << " reason=" << reason;
        if (status >= 0) std::osyncstream(std::cerr) << " status=" << status;
        std::osyncstream(std::cerr) << std::endl;
      }
      return false;
    };
    if (!task.stage1_receipt_released) {
      const RemotePtr gate_target = task.placement_committed
        ? task.final_target : task.target;
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      const protocol::AuthorityPlacementItem gate{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .id = task.id,
        .generation = task.generation,
        .expected_raw = gate_target.raw_address,
        .desired_raw = gate_target.raw_address,
        .expected_placement_version = gate_version,
      };
      protocol::AuthorityPlacementResult gate_result;
      if (!node_.relocate_via_authority(
            task.authority_shard, gate, gate_result, config)) {
        return failed("placement_transport");
      }
      const auto status = static_cast<
        protocol::AuthorityPlacementStatus>(gate_result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) {
        return failed("placement_busy", static_cast<int>(status));
      }
      if (status != protocol::AuthorityPlacementStatus::committed &&
          status != protocol::AuthorityPlacementStatus::replay &&
          status != protocol::AuthorityPlacementStatus::stale) {
        return failed("placement_conflict", static_cast<int>(status));
      }
      if (!node_.release_resolved_local_stage1_receipt(task, config)) {
        return failed("stage1_release");
      }
      task.stage1_receipt_released = true;
    }
    if (config.storage_owner_lease_merge &&
        stage2_stale_certificate_release_required(
          task.stage2_promotion_committed, task.stage2_promotion_parent,
          span<const RemotePtr>{task.stage2_replaced_certificate_parents}) &&
        !task.certificate_released && !task.final_target.is_null()) {
      vec<ReconcileReverseOp> releases;
      const vec<RemotePtr> parents = stage2_certificate_release_parents(
        task.stage2_promotion_parent,
        span<const RemotePtr>{task.stage2_replaced_certificate_parents});
      releases.reserve(parents.size());
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr) {
        std::string message = std::string("[certificate-stale-release] self=") +
          std::to_string(node_.storage_id_) + " id=" + std::to_string(task.id) +
          " sequence=" + std::to_string(task.maintenance_sequence) +
          " child=" + std::to_string(task.final_target.raw_address) + " parents=";
        for (const RemotePtr parent : parents) {
          message += std::to_string(parent.raw_address) + ",";
        }
        std::osyncstream(std::cerr) << message << std::endl;
      }
      for (const RemotePtr parent : parents) {
        releases.push_back(make_reconcile_op(
          task, parent, RemotePtr{}, task.final_target,
          ReconcileReverseOpKind::release_certificate_lease));
      }
      if (!apply_reconcile_ops(span<const ReconcileReverseOp>{releases})) {
        return failed("certificate_release");
      }
      task.certificate_released = true;
    }

    // A successor can win while Stage2 is searching or after part of an
    // idempotent reconciliation retry. Remove both physical incarnations
    // before retiring an uncommitted migrated record. Unlike the legacy
    // cleanup path, node_ explicitly addresses the provisional backlink plane.
    vec<RemotePtr> original_targets = task.stage1_backlink_targets;
    original_targets.insert(
      original_targets.end(), task.stage2_replaced_certificate_parents.begin(),
      task.stage2_replaced_certificate_parents.end());
    original_targets.push_back(task.stage2_promotion_parent);
    if (task.final_target.is_null() || task.final_target == task.target) {
      original_targets.insert(original_targets.end(),
                              task.stage2_neighbors.begin(),
                              task.stage2_neighbors.end());
    }
    bool cleaned = remove_candidate_backlinks(
      task, task.target, span<const RemotePtr>{original_targets});
    if (cleaned && !task.final_target.is_null() &&
        task.final_target != task.target) {
      vec<RemotePtr> final_targets = task.stage2_neighbors;
      final_targets.insert(
        final_targets.end(), task.stage2_replaced_certificate_parents.begin(),
        task.stage2_replaced_certificate_parents.end());
      final_targets.push_back(task.stage2_promotion_parent);
      cleaned = remove_candidate_backlinks(
        task, task.final_target, span<const RemotePtr>{final_targets});
    }
    if (!cleaned) return failed("backlink_removal");

    // A timeout can occur after final membership was published but before the
    // worker observed its ACK. Withdraw that exact tagged generation while it
    // is still readable; only then may the uncommitted destination be
    // tombstoned. The accounted bit makes node_ a no-op when publication never
    // happened or a successor cleanup already removed it.
    const RemotePtr accounted_candidate = task.final_target.is_null()
      ? task.target : task.final_target;
    const CentroidMembershipOp centroid_remove =
      make_centroid_membership_op(
        task, accounted_candidate, CentroidMembershipKind::remove);
    if (!node_.apply_centroid_membership_fanout_and_wait(
          span<const CentroidMembershipOp>{&centroid_remove, 1}, config)) {
      return failed("centroid_removal");
    }
    if (!task.final_target.is_null() && task.final_target != task.target) {
      if (stale_stage2_owns_source_retirement(
            task.placement_committed, task.target, task.final_target)) {
        // Placement made node_.final_target the authority predecessor seen by every
        // successor. If that successor wins before this worker tombstones the
        // Stage1 source, nobody else owns source cleanup. The source graph was
        // frozen before rebase, both backlink incarnations are gone above,
        // and it was never centroid-accounted, so retire it here before
        // settling the destination allocation receipt.
        NodeSnapshot source_snapshot;
        if (!node_.read_node_snapshot(task.target, source_snapshot) ||
            source_snapshot.id != task.id ||
            source_snapshot.generation != task.generation) {
          return failed("source_identity");
        }
        if (!source_snapshot.deleted &&
            !node_.mark_node_deleted(task.target, task.generation)) {
          return failed("source_tombstone");
        }
        node_.retire_local_dynamic_node(
          task.target, task.maintenance_sequence);
      }
      const protocol::DynamicNodeControlItem retirement{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .node_raw = task.final_target.raw_address,
        .id = task.id,
        .generation = task.generation,
        .authority_shard = task.authority_shard,
        .action = static_cast<u32>(
          protocol::DynamicNodeControlAction::retire),
      };
      protocol::DynamicNodeControlResult retirement_result;
      if (!node_.control_dynamic_node_on_shard(
            task.final_target.memory_node(), retirement,
            retirement_result, config)) {
        return failed("destination_retirement");
      }
      const auto retirement_status = static_cast<
        protocol::DynamicNodeControlStatus>(retirement_result.status);
      if (retirement_status != protocol::DynamicNodeControlStatus::ok &&
          retirement_status != protocol::DynamicNodeControlStatus::stale) {
        return failed("destination_retirement_status",
                      static_cast<int>(retirement_status));
      }
      if (!settle_dynamic_allocation(task)) {
        return failed("allocation_settlement");
      }
      task.allocation_settled = true;
    }
    node_.storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    node_.storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
    node_.complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    return true;
  };




bool Stage2Worker::complete_stale_cleanup(const StorageOwnerMaintenanceTask& task) {
    if (config.storage_owner_lease_merge && !task.cleanup_repair_only) {
      lib_assert(task.cleanup_protected_children.size() ==
                   task.cleanup_replacement_parents.size(),
                 "stale cleanup lost protected child/parent correlation");
      vec<ReconcileReverseOp> releases;
      for (size_t index = 0; index < task.cleanup_protected_children.size();
           ++index) {
        const RemotePtr child = task.cleanup_protected_children[index];
        if (child.is_null()) continue;
        const vec<RemotePtr> parents = stage2_certificate_release_parents(
          task.cleanup_replacement_parents[index],
          span<const RemotePtr>{task.cleanup_replaced_certificate_parents});
        for (const RemotePtr parent : parents) {
          releases.push_back(make_reconcile_op(
            task, parent, RemotePtr{}, child,
            ReconcileReverseOpKind::release_certificate_lease));
        }
      }
      if (!releases.empty() &&
          !apply_reconcile_ops(span<const ReconcileReverseOp>{releases})) {
        return false;
      }
    }
    node_.storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    node_.storage_owner_maintenance_cleanup_processed_.fetch_add(
      1, std::memory_order_relaxed);
    node_.complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    return true;
  };




PeerRpcType Stage2Worker::reverse_request_type(const Stage2Context& context) {
    return context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node
      ? PeerRpcType::cleanup_deleted_request
      : PeerRpcType::reverse_update_request;
  };




Stage2ReverseEnqueueResult Stage2Worker::enqueue_reverse_dispatch(Stage2Context& context,
                                             u32 shard,
                                             u64 ready_at_ns) {
    lib_assert(node_.storage_owner_reverse_outbox_ != nullptr,
               "stage2 reverse outbox is not initialized");
    lib_assert(shard < node_.num_storage_nodes_ && shard != node_.storage_id_,
               "stage2 reverse dispatch targets an invalid peer");
    lib_assert(context.remote_ops_by_peer[shard].size() <=
                 std::numeric_limits<u32>::max(),
               "stage2 reverse dispatch item count exceeds wire bound");
    const u32 item_count = static_cast<u32>(
      context.remote_ops_by_peer[shard].size());
    const Stage2ReverseDispatch dispatch{
      .logical_request_id = context.reverse_request_ids[shard],
      .context = context.handle,
      .worker_id = worker_id_,
      .peer_index = shard,
      .request_type = reverse_request_type(context),
      .item_count = item_count,
      .ops = context.remote_ops_by_peer[shard].data(),
      .ready_at_ns = ready_at_ns,
    };
    const Stage2ReverseEnqueueResult result =
      node_.storage_owner_reverse_outbox_->try_enqueue(dispatch);
    lib_assert(result == Stage2ReverseEnqueueResult::enqueued ||
                 result == Stage2ReverseEnqueueResult::duplicate,
               "bounded stage2 reverse outbox capacity/correlation invariant failed");
    if (result == Stage2ReverseEnqueueResult::enqueued) {
      // This worker immediately drives the shared outbox below; no executor
      // wake is needed for its own publication.
    }
    return result;
  };




void Stage2Worker::register_reverse_requests(Stage2Context& context,
                                             u64 expected_mask) {
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < node_.num_storage_nodes_; ++shard) {
      const u64 bit = u64{1} << shard;
      if ((expected_mask & bit) == 0) {
        continue;
      }
      const u64 request_id = node_.allocate_peer_request_id();
      context.reverse_request_ids[shard] = request_id;
      // Reverse retry/deadline state belongs to the aggregate wire request;
      // node_ logical record exists only for ACK fan-out into the context mask.
      const auto result = requests.try_register(
        request_id, context.handle, Stage2RequestKind::reverse_update, shard,
        now, std::numeric_limits<u64>::max(), states);
      lib_assert(result == Stage2RequestRegisterResult::registered,
                 "stage2 reverse request tracker capacity invariant failed");
      const auto queued = enqueue_reverse_dispatch(context, shard, now);
      lib_assert(queued == Stage2ReverseEnqueueResult::enqueued,
                 "new stage2 reverse request was already present in outbox");
    }
  };




PeerRpcType Stage2Worker::aggregate_response_type(PeerRpcType request_type) {
    return request_type == PeerRpcType::cleanup_deleted_request
      ? PeerRpcType::cleanup_deleted_response
      : PeerRpcType::reverse_update_response;
  };




bool Stage2Worker::poll_owned_reverse_aggregates() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const auto aggregate =
        node_.storage_owner_reverse_outbox_->claim_awaiting_response(
          worker_id_, cursor);
      if (!aggregate.has_value()) break;

      service::storage_owner::PeerRpcHeader header{};
      reverse_response_payload.clear();
      const PeerRpcType response_type =
        aggregate_response_type(aggregate->request_type);
      PeerResponseLease response_lease{};
      TryPeerResponse response = node_.try_consume_peer_rpc_response(
        aggregate->wire_request_id, aggregate->peer_index, response_type,
        aggregate->item_count, header, reverse_response_payload,
        response_lease);
      const u64 now = steady_now_ns();
      if (response == TryPeerResponse::success) {
        if (!node_.acknowledge_peer_rpc_response(response_lease)) {
          response = TryPeerResponse::stale;
        }
      }
      if (response == TryPeerResponse::success) {
        const auto completion_count =
          node_.storage_owner_reverse_outbox_->copy_completions(
            worker_id_, aggregate->wire_request_id,
            std::span<Stage2ReverseCompletion>{
              reverse_completion_scratch.data(),
              reverse_completion_scratch.size()});
        lib_assert(completion_count.has_value() &&
                     *completion_count == aggregate->logical_count,
                   "stage2 reverse aggregate lost ACK fan-out metadata");
        // The copied completions are value snapshots. Release every logical
        // outbox entry before making an ACK visible to a destination worker:
        // that worker may consume its final ACK, reuse the context slot, and
        // enqueue replacement work immediately. Keeping the old entries until
        // after fan-out would create a transient false-full at exact capacity.
        lib_assert(node_.storage_owner_reverse_outbox_->finish_success(
                     worker_id_, aggregate->wire_request_id),
                   "stage2 reverse aggregate ACK release failed");
        reverse_completion_wake_owners.clear();
        for (size_t index = 0; index < *completion_count; ++index) {
          const Stage2ReverseCompletion& completion =
            reverse_completion_scratch[index];
          lib_assert(completion.worker_id <
                       node_.storage_owner_reverse_completions_.size() &&
                       node_.storage_owner_reverse_completions_[
                         completion.worker_id] != nullptr,
                     "stage2 reverse completion targets an invalid worker");
          lib_assert(node_.storage_owner_reverse_completions_[
                       completion.worker_id]->try_push(completion),
                     "bounded stage2 reverse completion capacity invariant failed");
          if (reverse_completion_worker_marked[completion.worker_id] == 0) {
            reverse_completion_worker_marked[completion.worker_id] = 1;
            reverse_completion_wake_owners.push_back(completion.worker_id);
          }
        }
        // One aggregate may contain logical requests owned by several
        // executors. Wake exactly the distinct owners that received a queue
        // entry instead of making unrelated executors rescan every context.
        for (const u32 wake_owner : reverse_completion_wake_owners) {
          reverse_completion_worker_marked[wake_owner] = 0;
          node_.notify_storage_owner_maintenance_executor(wake_owner);
        }
        progressed = true;
        continue;
      }

      if (response == TryPeerResponse::failure ||
          response == TryPeerResponse::stale) {
        if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr &&
            aggregate->retry_attempts >= 3 &&
            ((aggregate->retry_attempts + 1) &
             aggregate->retry_attempts) == 0) {
          std::string message = std::string("[reverse-failure-debug] self=") +
            std::to_string(node_.storage_id_) + " worker=" +
            std::to_string(worker_id_) + " peer=" +
            std::to_string(aggregate->peer_index) + " wire_request_id=" +
            std::to_string(aggregate->wire_request_id) + " type=" +
            std::to_string(static_cast<u32>(aggregate->request_type)) +
            " items=" + std::to_string(aggregate->item_count) +
            " logical=" + std::to_string(aggregate->logical_count) +
            " retry=" + std::to_string(aggregate->retry_attempts + 1) +
            " response=" + std::to_string(static_cast<u32>(response)) +
            " status=" + std::to_string(header.status);
          reverse_wire_ops.resize(aggregate->item_count);
          lib_assert(node_.storage_owner_reverse_outbox_->copy_ops(
                       worker_id_, aggregate->wire_request_id,
                       std::span<ReverseUpdateOp>{reverse_wire_ops}),
                     "failed reverse request lost its immutable payload");
          for (size_t index = 0;
               index < std::min<size_t>(reverse_wire_ops.size(), 16); ++index) {
            const auto& op = reverse_wire_ops[index];
            message += " op(" + std::to_string(op.target_raw) + "," +
              std::to_string(op.candidate_raw) + "," +
              std::to_string(op.target_id) + "," +
              std::to_string(op.target_generation) + "," +
              std::to_string(op.candidate_id) + "," +
              std::to_string(op.candidate_generation) + ")";
          }
          std::osyncstream(std::cerr) << message << std::endl;
        }
        if (response == TryPeerResponse::stale) {
          static std::atomic<u32> debug_count{0};
          if (debug_count.fetch_add(1, std::memory_order_relaxed) < 128) {
            const auto metadata = node_.peer_async_responses_
              ->debug_response_metadata(aggregate->wire_request_id);
            std::osyncstream(std::cerr) << std::string("[reverse-stale-debug] self=") +
              std::to_string(node_.storage_id_) + " peer=" +
              std::to_string(aggregate->peer_index) + " request_id=" +
              std::to_string(aggregate->wire_request_id) + " expected_type=" +
              std::to_string(static_cast<u32>(response_type)) +
              " expected_items=" + std::to_string(aggregate->item_count) +
              " registry_state=" + std::to_string(metadata.state) +
              " registry_request_id=" + std::to_string(metadata.request_id) +
              " registry_peer=" + std::to_string(metadata.expected_shard) +
              " registry_type=" + std::to_string(metadata.expected_type) +
              " registry_items=" + std::to_string(metadata.expected_item_count)
                      << std::endl;
          }
        }
        if (response == TryPeerResponse::failure && response_lease.valid()) {
          (void)node_.rearm_peer_rpc_response(response_lease);
        }
        if (response == TryPeerResponse::stale) {
          node_.cancel_peer_rpc_response(aggregate->wire_request_id);
        }
        lib_assert(node_.storage_owner_reverse_outbox_->release_poll(
                     worker_id_, aggregate->wire_request_id, true,
                     now + reverse_retry_backoff(aggregate->retry_attempts)),
                   "stage2 reverse aggregate failure retry release failed");
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        progressed = true;
        continue;
      }

      const bool timed_out = now >= aggregate->deadline_ns;
      static std::atomic<u32> debug_count{0};
      if ((response == TryPeerResponse::failure ||
           response == TryPeerResponse::stale || timed_out) &&
          (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr ||
           aggregate->retry_attempts >= 3) &&
          debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        std::osyncstream(std::cerr) << "[reverse-ack-debug] self=" << node_.storage_id_
                  << " worker=" << worker_id_
                  << " peer=" << aggregate->peer_index
                  << " wire_request_id=" << aggregate->wire_request_id
                  << " request_type=" << static_cast<u32>(aggregate->request_type)
                  << " items=" << aggregate->item_count
                  << " logical=" << aggregate->logical_count
                  << " retries=" << aggregate->retry_attempts
                  << " response=" << static_cast<u32>(response) << std::endl;
      }
      if (timed_out) {
        node_.storage_owner_maintenance_rpc_timeouts_.fetch_add(
          1, std::memory_order_relaxed);
      }
      lib_assert(node_.storage_owner_reverse_outbox_->release_poll(
                   worker_id_, aggregate->wire_request_id, timed_out,
                   timed_out
                     ? now + reverse_retry_backoff(aggregate->retry_attempts)
                     : now),
                 "stage2 reverse aggregate poll release failed");
      progressed = timed_out || progressed;
    }
    return progressed;
  };




bool Stage2Worker::form_reverse_aggregates() {
    bool progressed = false;
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < node_.num_storage_nodes_; ++shard) {
      if (shard == node_.storage_id_) continue;
      for (;;) {
        if (!node_.storage_owner_reverse_outbox_->can_form_aggregate(shard, now)) {
          break;
        }
        const u64 wire_request_id = node_.allocate_peer_request_id();
        const auto aggregate = node_.storage_owner_reverse_outbox_->form_aggregate(
          shard, worker_id_, wire_request_id, now);
        if (!aggregate.has_value()) break;
        node_.storage_owner_reverse_aggregate_batches_.fetch_add(
          1, std::memory_order_relaxed);
        node_.storage_owner_reverse_aggregate_logical_requests_.fetch_add(
          aggregate->logical_count, std::memory_order_relaxed);
        node_.storage_owner_reverse_aggregate_ops_.fetch_add(
          aggregate->item_count, std::memory_order_relaxed);
        progressed = true;
      }
    }
    return progressed;
  };




void Stage2Worker::log_reverse_send_snapshot(
    u32 peer, u64 request_id, u32 item_count) {
    std::array<size_t, 3> free_slots{};
    {
      std::lock_guard<std::mutex> lock(node_.peer_rpc_send_slots_mutex_);
      for (size_t send_class = 0; send_class < free_slots.size(); ++send_class) {
        free_slots[send_class] =
          node_.peer_rpc_free_send_slots_[peer][send_class].size();
      }
    }
    size_t pending_sends = 0;
    {
      std::lock_guard<std::mutex> lock(node_.peer_completion_mutex_);
      for (const auto& [wr_id, pending] : node_.peer_pending_sends_) {
        pending_sends += pending.release_rpc_slot && pending.target_shard == peer;
      }
    }
    const auto& registry = node_.peer_async_responses_;
    std::osyncstream(std::cerr) << std::string("[reverse-resource-debug] self=") +
      std::to_string(node_.storage_id_) + " worker=" +
      std::to_string(worker_id_) + " peer=" + std::to_string(peer) +
      " request_id=" + std::to_string(request_id) +
      " items=" + std::to_string(item_count) +
      " request_bytes=" + std::to_string(
        service::storage_owner::reverse_update_request_bytes(item_count)) +
      " registry_state=" + std::to_string(
        registry == nullptr ? -2 : registry->debug_response_state(request_id)) +
      " registry_size=" + std::to_string(
        registry == nullptr ? 0 : registry->size()) +
      " registry_capacity=" + std::to_string(
        registry == nullptr ? 0 : registry->capacity()) +
      " message_bytes=" + std::to_string(
        node_.peer_rpc_runtime_.message_bytes) +
      " free_stage1=" + std::to_string(free_slots[0]) +
      " free_graph=" + std::to_string(free_slots[1]) +
      " free_control=" + std::to_string(free_slots[2]) +
      " pending_sends=" + std::to_string(pending_sends) << std::endl;
  };



bool Stage2Worker::post_owned_reverse_aggregates() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const u64 now = steady_now_ns();
      const auto aggregate =
        node_.storage_owner_reverse_outbox_->claim_ready_to_post(
          worker_id_, now, cursor);
      if (!aggregate.has_value()) break;

      reverse_wire_ops.resize(aggregate->item_count);
      lib_assert(node_.storage_owner_reverse_outbox_->copy_ops(
                   worker_id_, aggregate->wire_request_id,
                   std::span<ReverseUpdateOp>{reverse_wire_ops.data(),
                                              reverse_wire_ops.size()}),
                 "stage2 reverse aggregate payload copy failed");
      u32 item_count = 0;
      const bool sent = node_.post_peer_op_batch_async(
        aggregate->peer_index, reverse_wire_ops, aggregate->request_type,
        aggregate->wire_request_id, item_count, config);
      lib_assert(item_count == aggregate->item_count,
                 "stage2 reverse aggregate post changed item_count");
      const u64 posted_at = steady_now_ns();
      constexpr u64 reverse_ack_timeout_ns = 1'000'000'000ull;
      lib_assert(node_.storage_owner_reverse_outbox_->finish_post(
                   worker_id_, aggregate->wire_request_id, sent,
                   posted_at + (sent ? std::min(rpc_timeout_ns,
                                                reverse_ack_timeout_ns)
                                     : reverse_retry_backoff(
                                         aggregate->retry_attempts))),
                 "stage2 reverse aggregate post release failed");
      if (sent) {
        progressed = true;
      } else {
        if (aggregate->retry_attempts >= 8) {
          log_reverse_send_snapshot(
            aggregate->peer_index, aggregate->wire_request_id,
            aggregate->item_count);
        }
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
    reverse_wire_ops.resize(reverse_wire_max_ops);
    return progressed;
  };




bool Stage2Worker::drive_reverse_outbox() {
    bool progressed = poll_owned_reverse_aggregates();
    progressed = form_reverse_aggregates() || progressed;
    progressed = post_owned_reverse_aggregates() || progressed;
    return progressed;
  };




bool Stage2Worker::drain_reverse_completions() {
    bool progressed = false;
    Stage2ReverseCompletion completion;
    auto& completion_queue = *node_.storage_owner_reverse_completions_[worker_id_];
    while (completion_queue.try_pop(completion)) {
      lib_assert(completion.worker_id == worker_id_,
                 "stage2 reverse completion reached the wrong worker");
      const auto metadata = requests.find(completion.logical_request_id);
      lib_assert(metadata.has_value() &&
                   metadata->context == completion.context &&
                   metadata->kind == Stage2RequestKind::reverse_update &&
                   metadata->peer_index == completion.peer_index,
                 "stage2 reverse completion lost logical correlation");
      const Stage2EventResult result = requests.record_response(
        completion.logical_request_id, states);
      lib_assert(result == Stage2EventResult::accepted ||
                   result == Stage2EventResult::ready_to_finalize ||
                   result == Stage2EventResult::duplicate,
                 "stage2 rejected an aggregate reverse-update ACK");
      lib_assert(completion.context.slot < contexts.size(),
                 "stage2 reverse completion context slot is out of range");
      Stage2Context& ready_context = contexts[completion.context.slot];
      lib_assert(ready_context.active &&
                   ready_context.handle == completion.context,
                 "stage2 reverse completion reached a stale context");
      lib_assert(node_.enqueue_storage_owner_maintenance_context_ready(
                   ready_context.ready_owner,
                   Stage2ContextReadyReason::reverse_completion),
                 "stage2 reverse completion failed ready-queue handoff");
      lib_assert(requests.erase(completion.logical_request_id),
                 "stage2 reverse completion request release failed");
      completion = {};
      progressed = true;
    }
    return progressed;
  };



}  // namespace memory_node_storage_owner_maintenance_detail

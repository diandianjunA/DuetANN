#include "memory_node/storage_owner_maintenance/worker_internal.hh"

#include <cstdlib>
#include <syncstream>

namespace memory_node_storage_owner_maintenance_detail {

bool Stage2Worker::consolidate_finalized_insertions() {
  const auto set_backpressure = [&](bool active) {
    auto& flag = node_.storage_owner_refinement_backpressure_[worker_id_];
    const bool was_active = flag.exchange(
      static_cast<u8>(active), std::memory_order_acq_rel) != 0;
    if (was_active && !active) {
      node_.storage_owner_maintenance_cv_.notify_all();
      node_.wake_peer_stage1_admission_waiters();
    }
  };
  const bool pipeline = storage_owner_pipeline_refinement_enabled();
  const auto now = std::chrono::steady_clock::now();
  if (now < refinement_retry_not_before) return false;
  {
    std::lock_guard<std::mutex> lock(node_.storage_owner_deferred_refinements_mutex_);
    if (node_.storage_owner_deferred_refinements_.empty()) {
      set_backpressure(false);
      return false;
    }
    if (node_.storage_owner_deferred_refinements_.size() <
          (pipeline ? 256 : 4096) &&
        !node_.storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) &&
        now - node_.storage_owner_deferred_refinement_started_ <
          (pipeline ? std::chrono::milliseconds(5)
                    : std::chrono::milliseconds(500))) {
      return false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(node_.storage_owner_maintenance_mutex_);
    const bool cleanup_pending = !node_.storage_owner_cleanup_tasks_.empty();
    const bool cleanup_ready = pipeline && std::any_of(
      node_.storage_owner_cleanup_tasks_.begin(),
      node_.storage_owner_cleanup_tasks_.end(), [&](const auto& task) {
        return node_.storage_owner_cleanup_ready(task, config);
      });
    if ((!pipeline && !node_.storage_owner_stage2_tasks_.empty()) ||
        !refinement_cleanup_gate_open(
          pipeline, cleanup_pending, cleanup_ready) ||
        (node_.storage_owner_repair_tasks_ != nullptr &&
         !node_.storage_owner_repair_tasks_->empty()) ||
        (!pipeline &&
         node_.storage_owner_maintenance_active_workers_.load(
           std::memory_order_acquire) != 0)) {
      return false;
    }
  }

  std::optional<u32> refinement_lane;
  if (pipeline) {
    for (size_t lane = 0; lane < thread.post_balances.size(); ++lane) {
      const bool occupied = std::any_of(
        contexts.begin(), contexts.end(), [&](const Stage2Context& context) {
          return context.search_lane == lane;
        });
      if (occupied || !thread.is_ready(lane)) continue;
      if (!node_.try_acquire_storage_owner_search_lane_lease(
            refinement_lane_wait_registered)) {
        return false;
      }
      refinement_lane = static_cast<u32>(lane);
      break;
    }
    if (!refinement_lane.has_value()) return false;
  }

  const size_t worker_count = node_.storage_owner_maintenance_workers_.size();
  lib_assert(worker_count != 0 && worker_id_ < worker_count,
               "invalid maintenance worker for stable refinement");
  vec<StorageOwnerMaintenanceTask> batch;
  std::deque<StorageOwnerMaintenanceTask> remaining;
  {
    std::lock_guard<std::mutex> lock(node_.storage_owner_deferred_refinements_mutex_);
    if (node_.storage_owner_deferred_refinements_.empty()) {
      if (refinement_lane.has_value()) {
        node_.release_storage_owner_search_lane_lease();
      }
      return false;
    }
    if (pipeline) {
      constexpr size_t task_limit = 256;
      batch.reserve(std::min(task_limit,
        node_.storage_owner_deferred_refinements_.size()));
      while (batch.size() < task_limit &&
             !node_.storage_owner_deferred_refinements_.empty()) {
        batch.push_back(std::move(
          node_.storage_owner_deferred_refinements_.front()));
        node_.storage_owner_deferred_refinements_.pop_front();
      }
    } else {
      batch.reserve(node_.storage_owner_deferred_refinements_.size() /
                      worker_count + 1);
      for (StorageOwnerMaintenanceTask& task :
           node_.storage_owner_deferred_refinements_) {
        if (task.maintenance_sequence % worker_count == worker_id_) {
          batch.push_back(std::move(task));
        } else {
          remaining.push_back(std::move(task));
        }
      }
      node_.storage_owner_deferred_refinements_.swap(remaining);
    }
    if (node_.storage_owner_deferred_refinements_.empty()) {
      node_.storage_owner_deferred_refinement_started_ = {};
    }
  }
  if (batch.empty()) {
    if (refinement_lane.has_value()) {
      node_.release_storage_owner_search_lane_lease();
    }
    return false;
  }
  const u32 active_lane = refinement_lane.value_or(0);
  thread.set_current_coroutine(active_lane);
  lib_assert(thread.is_ready(active_lane),
             "stable refinement inherited a busy RDMA lane");
  vec<StorageOwnerMaintenanceTask> retry;
  const size_t chunk_size = pipeline ? batch.size() : 32;
  for (size_t begin = 0; begin < batch.size(); begin += chunk_size) {
    const size_t end = std::min(batch.size(), begin + chunk_size);
    vec<StorageOwnerMaintenanceTask> chunk_tasks;
    vec<RemotePtr> parents;
    chunk_tasks.reserve(end - begin);
    for (size_t index = begin; index < end; ++index) {
      StorageOwnerMaintenanceTask& task = batch[index];
      NodeSnapshot snapshot;
      const auto state = node_.storage_owner_physical_node_state(
        task.id, task.generation, task.target, &snapshot);
      if (state == StableNodeSnapshotState::retryable ||
          (state == StableNodeSnapshotState::stable &&
           !stage2_parent_is_stable(snapshot.header, snapshot.deleted))) {
        if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr) {
          static std::atomic<u32> debug_count{0};
          if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
            std::osyncstream(std::cerr) << "[stage2-refine-stall] reason=target_state"
                      << " id=" << task.id
                      << " sequence=" << task.maintenance_sequence
                      << " state=" << static_cast<u32>(state)
                      << " header=" << snapshot.header
                      << " deleted=" << snapshot.deleted << std::endl;
          }
        }
        retry.push_back(std::move(task));
        continue;
      }
      if (state == StableNodeSnapshotState::terminal) {
        node_.complete_storage_owner_maintenance_sequence(
          task.maintenance_sequence);
        continue;
      }
      for (const RemotePtr parent : task.stage2_neighbors) {
        if (!parent.is_null() && parent != task.target &&
            node_.storage_node_pointer_addressable(parent)) {
          parents.push_back(parent);
        }
      }
      chunk_tasks.push_back(std::move(task));
    }
    std::sort(parents.begin(), parents.end(),
              [](RemotePtr lhs, RemotePtr rhs) {
                return lhs.raw_address < rhs.raw_address;
              });
    parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
    vec<std::pair<RemotePtr, GraphAdjacency>> graph_snapshots;
    node_.read_graph_adjacencies_batched_into(
      span<const RemotePtr>{parents}, config, graph_snapshots);
    dense_hashmap_t<u64, const GraphAdjacency*> parent_graphs;
    parent_graphs.reserve(graph_snapshots.size());
    for (const auto& [pointer, adjacency] : graph_snapshots) {
      parent_graphs.emplace(pointer.raw_address, &adjacency);
    }
    struct RefinementOp {
      ReconcileReverseOp wire;
      size_t task_index;
    };
    vec<RefinementOp> refinement_ops;
    for (size_t task_index = 0; task_index < chunk_tasks.size(); ++task_index) {
      const StorageOwnerMaintenanceTask& task = chunk_tasks[task_index];
      for (const RemotePtr parent : task.stage2_neighbors) {
        if (parent.is_null() || parent == task.target ||
            !node_.storage_node_pointer_addressable(parent)) {
          continue;
        }
        const auto found = parent_graphs.find(parent.raw_address);
        if (found != parent_graphs.end()) {
          const GraphAdjacency& adjacency = *found->second;
          if (adjacency.deleted) continue;
          if (std::find(adjacency.stable.begin(), adjacency.stable.end(),
                        task.target) != adjacency.stable.end()) {
            continue;
          }
        }
        refinement_ops.push_back(RefinementOp{
          make_reconcile_op(
            task, parent, RemotePtr{}, task.target,
            pipeline ? ReconcileReverseOpKind::refine_add
                     : ReconcileReverseOpKind::add),
          task_index,
        });
      }
    }
    std::stable_sort(
      refinement_ops.begin(), refinement_ops.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.wire.target_raw < rhs.wire.target_raw;
      });
    vec<ReconcileReverseOp> ops;
    vec<size_t> op_task_indices;
    ops.reserve(refinement_ops.size());
    op_task_indices.reserve(refinement_ops.size());
    for (const RefinementOp& op : refinement_ops) {
      ops.push_back(op.wire);
      op_task_indices.push_back(op.task_index);
    }
    Configuration prune_config = config;
    prune_config.alpha = -std::abs(prune_config.alpha);
    vec<u8> retry_mask;
    const bool applied = ops.empty() || apply_reconcile_ops(
      span<const ReconcileReverseOp>{ops}, &prune_config, &retry_mask);
    if (!applied) {
      retry_mask.assign(ops.size(), 1);
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::osyncstream(std::cerr) << "[stage2-refine-stall] reason=apply"
                    << " worker=" << worker_id_
                    << " tasks=" << chunk_tasks.size()
                    << " ops=" << ops.size()
                    << " first_id=" << (chunk_tasks.empty()
                      ? 0 : chunk_tasks.front().id)
                    << " first_sequence=" << (chunk_tasks.empty()
                      ? 0 : chunk_tasks.front().maintenance_sequence)
                    << std::endl;
        }
      }
    }
    lib_assert(retry_mask.size() == ops.size(),
               "refinement retry mask lost operation alignment");
    vec<vec<RemotePtr>> remaining_parents(chunk_tasks.size());
    for (size_t op_index = 0; op_index < ops.size(); ++op_index) {
      if (retry_mask[op_index] != 0) {
        remaining_parents[op_task_indices[op_index]].emplace_back(
          ops[op_index].target_raw);
      }
    }
    for (size_t task_index = 0; task_index < chunk_tasks.size(); ++task_index) {
      StorageOwnerMaintenanceTask& task = chunk_tasks[task_index];
      if (remaining_parents[task_index].empty()) {
        node_.complete_storage_owner_maintenance_sequence(
          task.maintenance_sequence);
        continue;
      }
      task.stage2_neighbors = std::move(remaining_parents[task_index]);
      retry.push_back(std::move(task));
    }
  }
  if (!retry.empty()) {
    set_backpressure(true);
    const auto retry_now = std::chrono::steady_clock::now();
    for (StorageOwnerMaintenanceTask& task : retry) {
      if (task.refinement_first_retry_at ==
          std::chrono::steady_clock::time_point{}) {
        task.refinement_first_retry_at = retry_now;
      }
      if (refinement_retry_deadline_exceeded(
            task.refinement_first_retry_at, retry_now,
            config.storage_owner_stage2_max_retry_ms)) {
        node_.storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        lib_failure(
          "stable refinement exceeded its retry deadline: worker=" +
          std::to_string(worker_id_) + " id=" + std::to_string(task.id) +
          " sequence=" + std::to_string(task.maintenance_sequence) +
          " remaining_parents=" + std::to_string(task.stage2_neighbors.size()));
      }
    }
    const u64 backoff_ns = reverse_retry_backoff(refinement_retry_attempts_++);
    refinement_retry_not_before = retry_now +
      std::chrono::nanoseconds(backoff_ns);
    std::lock_guard<std::mutex> lock(node_.storage_owner_deferred_refinements_mutex_);
    if (node_.storage_owner_deferred_refinements_.empty()) {
      node_.storage_owner_deferred_refinement_started_ = retry_now;
    }
    node_.storage_owner_deferred_refinements_.insert(
      node_.storage_owner_deferred_refinements_.end(),
      std::make_move_iterator(retry.begin()),
      std::make_move_iterator(retry.end()));
  } else {
    set_backpressure(false);
    refinement_retry_attempts_ = 0;
    refinement_retry_not_before = {};
  }
  if (refinement_lane.has_value()) {
    lib_assert(thread.is_ready(active_lane),
               "stable refinement returned with RDMA still in flight");
    node_.release_storage_owner_search_lane_lease();
  }
  return retry.size() < batch.size();
}

bool Stage2Worker::finalize_context(Stage2Context& context) {
    if (config.storage_owner_lease_merge &&
        context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      vec<ReconcileReverseOp> releases;
      for (const StorageOwnerMaintenanceTask& task : context.tasks) {
        if (task.cleanup_repair_only) continue;
        lib_assert(task.cleanup_protected_children.size() ==
                     task.cleanup_replacement_parents.size(),
                   "cleanup lease release lost child/parent correlation");
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
      }
      if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr) {
        for (const ReconcileReverseOp& op : releases) {
          std::osyncstream(std::cerr) << "[cleanup-certificate-release] self=" << node_.storage_id_
                    << " id=" << op.id << " sequence=" << op.placement_sequence
                    << " target=" << op.target_raw << " child=" << op.new_candidate_raw
                    << std::endl;
        }
      }
      if (!releases.empty() &&
          !apply_reconcile_ops(span<const ReconcileReverseOp>{releases})) {
        const auto retry_at = std::chrono::steady_clock::now() +
          std::chrono::milliseconds(1);
        for (StorageOwnerMaintenanceTask& task : context.tasks) {
          task.retry_not_before = retry_at;
        }
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
    }
    const Stage2EventResult transition = states.finalize(context.handle);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 finalized before all reverse ACKs");

    if (context.kind == StorageOwnerMaintenanceKind::finalize_insert) {
      lib_assert(context.completion_handoff_started_ns != 0,
                 "Stage2 finalization lost its completion-handoff timestamp");
      auto& completion_handoff_timing =
        node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
          StorageOwnerStage2TimingPhase::completion_handoff)];
      completion_handoff_timing.attempts.fetch_add(
        1, std::memory_order_relaxed);
      completion_handoff_timing.task_attempts.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
      completion_handoff_timing.elapsed_ns.fetch_add(
        steady_now_ns() - context.completion_handoff_started_ns,
        std::memory_order_relaxed);
      context.completion_handoff_started_ns = 0;
      Stage2PhaseAttemptTimer finalize_timer(
        node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
          StorageOwnerStage2TimingPhase::finalize)],
        context.tasks.size());
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        if (task.maintenance_sequence == 0) continue;
        lib_assert(task.outgoing_committed && task.reverse_reconciled &&
                     task.placement_committed && task.centroid_committed &&
                     (task.final_target == task.target ||
                      task.allocation_settled),
                   "Stage2 finalized before physical placement commit");
        record_finalized_live(task.queued_at);
        u64 final_edges = 0;
        u64 cross_edges_from_stage1_home = 0;
        u64 cross_edges_from_final_home = 0;
        for (const RemotePtr neighbor : task.stage2_neighbors) {
          if (neighbor.is_null() ||
              neighbor.memory_node() >= node_.num_storage_nodes_) {
            continue;
          }
          ++final_edges;
          cross_edges_from_stage1_home +=
            neighbor.memory_node() != task.target.memory_node();
          cross_edges_from_final_home +=
            neighbor.memory_node() != task.final_target.memory_node();
        }
        node_.storage_owner_stage2_final_edges_.fetch_add(
          final_edges, std::memory_order_relaxed);
        node_.storage_owner_stage2_cross_edges_stage1_home_.fetch_add(
          cross_edges_from_stage1_home, std::memory_order_relaxed);
        node_.storage_owner_stage2_cross_edges_final_home_.fetch_add(
          cross_edges_from_final_home, std::memory_order_relaxed);
        if (task.final_target != task.target) {
          node_.storage_owner_stage2_migrations_.fetch_add(
            1, std::memory_order_relaxed);
        }
        node_.storage_owner_maintenance_processed_.fetch_add(
          1, std::memory_order_relaxed);
        if (task.final_target != task.target) {
          // Reverse reconciliation and the authority placement ACK are both
          // complete. Retire the old Stage1 incarnation under node_ Stage2
          // sequence; reuse still waits for local durability and every
          // compute client's RCU acknowledgement.
          node_.retire_local_dynamic_node(
            task.target, task.maintenance_sequence);
        }
        if (storage_owner_consolidation_enabled()) {
          StorageOwnerMaintenanceTask deferred;
          deferred.id = task.id;
          deferred.generation = task.generation;
          deferred.target = task.final_target;
          deferred.maintenance_sequence = task.maintenance_sequence;
          deferred.stage2_neighbors = task.stage2_neighbors;
          std::lock_guard<std::mutex> lock(
            node_.storage_owner_deferred_refinements_mutex_);
          lib_assert(node_.storage_owner_deferred_refinements_.size() <
                       config.storage_owner_maintenance_queue_depth,
                     "deferred refinement exceeded bounded Stage2 admission");
          if (node_.storage_owner_deferred_refinements_.empty()) {
            node_.storage_owner_deferred_refinement_started_ =
              std::chrono::steady_clock::now();
          }
          node_.storage_owner_deferred_refinements_.push_back(
            std::move(deferred));
        } else {
          node_.complete_storage_owner_maintenance_sequence(
            task.maintenance_sequence);
        }
      }
    } else {
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        lib_assert(task.centroid_committed,
                   "cleanup finalized before centroid membership removal");
        if (!task.cleanup_repair_only) {
          node_.retire_local_dynamic_node(task.target, task.maintenance_sequence);
        }
        node_.storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        node_.complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      }
    }

    const Stage2ContextHandle handle = context.handle;
    release_context_lane(context);
    reset_context(context);
    lib_assert(states.release(handle),
               "stage2 context release violated finalized generation");
    node_.storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    node_.notify_storage_owner_maintenance_capacity();
    return true;
  };




void Stage2Worker::defer_cleanup_context(Stage2Context& context) {
    lib_assert(context.kind ==
                 StorageOwnerMaintenanceKind::cleanup_deleted_node,
               "only cleanup work can use dependency-aware deferral");
    const auto retry_at = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(1);
    {
      std::lock_guard<std::mutex> lock(node_.storage_owner_maintenance_mutex_);
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        if (task.maintenance_sequence == 0) continue;
        task.retry_not_before = retry_at;
        cleanup_schedule_push(
          node_.storage_owner_cleanup_tasks_, std::move(task));
      }
    }
    const Stage2ContextHandle handle = context.handle;
    release_context_lane(context);
    reset_context(context);
    lib_assert(states.release_retryable(handle),
               "cleanup deferral retained an asynchronous request");
    node_.storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    node_.notify_storage_owner_maintenance_capacity();
  };




bool Stage2Worker::drive_context(Stage2Context& context) {
    bool progressed = false;
    for (;;) {
      const auto snapshot = states.snapshot(context.handle);
      lib_assert(snapshot.has_value(), "active stage2 context became stale");
      const auto now = std::chrono::steady_clock::now();
      const bool deferred = std::any_of(
        context.tasks.begin(), context.tasks.end(),
        [&](const StorageOwnerMaintenanceTask& task) {
          return task.retry_not_before > now;
        });
      if (deferred && !context.item_pipeline_started) {
        if (snapshot->phase == Stage2Phase::prune_ready) {
          (void)release_rebindable_context_lane(context);
        }
        return progressed;
      }
      const bool prune_needs_lane =
        snapshot->phase == Stage2Phase::prune_ready &&
        (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node ||
         std::any_of(
           context.finalize_subphases.begin(),
           context.finalize_subphases.begin() + context.tasks.size(),
           [](Stage2FinalizeSubphase subphase) {
             return stage2_finalize_subphase_needs_lane(subphase);
           }));
      if (snapshot->phase == Stage2Phase::local_ready ||
          snapshot->phase == Stage2Phase::remote_search_pending ||
          prune_needs_lane) {
        if (!bind_context_lane(context)) return progressed;
      }
      switch (snapshot->phase) {
        case Stage2Phase::local_ready: {
          const Stage2SearchAdvanceResult local_result =
            prepare_local(context);
          if (local_result != Stage2SearchAdvanceResult::complete) {
            // Before search initialization, a transient target lock can return
            // waiting_rdma with neither live continuation state nor a posted
            // WR.  Reuse that otherwise-idle lane; an initialized search fails
            // the search_state_idle half of the predicate and remains pinned.
            (void)release_rebindable_context_lane(context);
            return progressed ||
              local_result == Stage2SearchAdvanceResult::posted_rdma;
          }
          progressed = true;
          // candidate_search copied every result into the context and reset its
          // lane state before reporting complete.  Yield at node_ semantic
          // boundary so another ready context can hide this context's upcoming
          // synchronous prune/control work.
          lib_assert(release_rebindable_context_lane(context),
                     "completed Stage2 search retained live lane state");
          return true;
        }
        case Stage2Phase::remote_search_pending:
          lib_failure(
            "home-executed Stage2 continuation cannot await legacy shard-search ACKs");
          return progressed;
        case Stage2Phase::prune_ready: {
          const bool prepared =
            context.kind == StorageOwnerMaintenanceKind::finalize_insert
              ? prepare_stage2_reverse(context)
              : prepare_cleanup_reverse(context);
          if (!prepared) {
            if (context.kind ==
                StorageOwnerMaintenanceKind::cleanup_deleted_node) {
              defer_cleanup_context(context);
              return true;
            }
            (void)release_rebindable_context_lane(context);
            return progressed;
          }
          progressed = true;
          continue;
        }
        case Stage2Phase::reverse_pending:
          release_context_lane(context);
          if (states.snapshot(context.handle)->phase ==
              Stage2Phase::reverse_pending &&
              states.snapshot(context.handle)->completed_reverse_mask !=
                states.snapshot(context.handle)->expected_reverse_mask) {
            return progressed;
          }
          if (!finalize_context(context)) return progressed;
          return true;
        case Stage2Phase::finalized:
          lib_failure("stage2 context remained active after finalization");
      }
    }
  };




bool Stage2Worker::drive_owned_context(Stage2Context& context) {
    const bool protocol_stall_debug =
      std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr ||
      std::any_of(context.tasks.begin(), context.tasks.end(),
        [](const StorageOwnerMaintenanceTask& task) {
          return std::chrono::steady_clock::now() - task.queued_at >
            std::chrono::seconds(60);
        });
    const u64 debug_now_ns = steady_now_ns();
    if (protocol_stall_debug && context.active && !context.tasks.empty() &&
        debug_now_ns - last_protocol_stall_log_ns > 10'000'000'000ull &&
        std::any_of(context.tasks.begin(), context.tasks.end(),
          [&](const StorageOwnerMaintenanceTask& task) {
            return std::chrono::steady_clock::now() - task.queued_at >
              std::chrono::seconds(10);
          })) {
      last_protocol_stall_log_ns = debug_now_ns;
      const auto snapshot = states.snapshot(context.handle);
      std::osyncstream(std::cerr) << "[stage2-debug] worker=" << worker_id_
                << " slot=" << context.handle.slot
                << " phase=" << (snapshot.has_value()
                  ? static_cast<u32>(snapshot->phase) : 255)
                << " item_pipeline=" << context.item_pipeline_started
                << " reconcile_active=" << context.reconcile_batch.active()
                << " expected_mask=" << (snapshot.has_value()
                  ? snapshot->expected_reverse_mask : 0)
                << " completed_mask=" << (snapshot.has_value()
                  ? snapshot->completed_reverse_mask : 0)
                << std::endl;
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        const StorageOwnerMaintenanceTask& task = context.tasks[item];
        std::osyncstream(std::cerr) << "[stage2-debug] item=" << item
                  << " id=" << task.id
                  << " sequence=" << task.maintenance_sequence
                  << " subphase=" << static_cast<u32>(
                    context.finalize_subphases[item])
                  << " authority=" << task.authority_shard
                  << " target=" << task.target.raw_address
                  << " final=" << task.final_target.raw_address
                  << " prepared=" << task.stage2_prepared
                  << " outgoing=" << task.outgoing_committed
                  << " promotion=" << task.stage2_promotion_committed
                  << " placement=" << task.placement_committed
                  << " centroid=" << task.centroid_committed
                  << " reversed=" << task.reverse_reconciled
                  << " rebase=" << task.stage2_rebase_replayed
                  << " frozen=" << task.stage2_source_frozen
                  << " parents=" << task.stage2_neighbors.size()
                  << " bridges=" << task.stage1_backlink_targets.size()
                  << " retry=" << (task.retry_not_before >
                    std::chrono::steady_clock::now())
                  << std::endl;
      }
      if (snapshot.has_value() &&
          snapshot->phase == Stage2Phase::reverse_pending) {
        for (u32 peer = 0; peer < node_.num_storage_nodes_; ++peer) {
          const u64 bit = u64{1} << peer;
          if ((snapshot->expected_reverse_mask & bit) == 0 ||
              (snapshot->completed_reverse_mask & bit) != 0) {
            continue;
          }
          const u64 request_id = context.reverse_request_ids[peer];
          const auto metadata = requests.find(request_id);
          const int response_state = node_.peer_async_responses_ == nullptr
            ? -2 : node_.peer_async_responses_->debug_response_state(request_id);
          std::osyncstream(std::cerr) << "[reverse-context-debug] self=" << node_.storage_id_
                    << " worker=" << worker_id_
                    << " slot=" << context.handle.slot
                    << " generation=" << context.handle.generation
                    << " peer=" << peer
                    << " logical_request_id=" << request_id
                    << " tracked=" << metadata.has_value()
                    << " response_state=" << response_state
                    << " completion_queue="
                    << node_.storage_owner_reverse_completions_[worker_id_]
                      ->approximate_size()
                    << " outbox_size="
                    << node_.storage_owner_reverse_outbox_->size()
                    << " aggregate_size="
                    << node_.storage_owner_reverse_outbox_->aggregate_size()
                    << std::endl;
          for (const auto& entry :
               node_.storage_owner_reverse_outbox_->debug_snapshot()) {
            if (entry.logical_request_id != request_id) continue;
            std::osyncstream(std::cerr) << "[reverse-outbox-debug] self=" << node_.storage_id_
                      << " logical_request_id=" << entry.logical_request_id
                      << " wire_request_id=" << entry.wire_request_id
                      << " owner=" << entry.owner_worker_id
                      << " peer=" << entry.peer_index
                      << " type=" << entry.request_type
                      << " context_slot=" << entry.context_slot
                      << " context_generation=" << entry.context_generation
                      << " state=" << entry.state
                      << " leased=" << entry.leased
                      << " retries=" << entry.retry_attempts
                      << " now=" << debug_now_ns
                      << " ready_at=" << entry.ready_at_ns
                      << " deadline=" << entry.deadline_ns << std::endl;
          }
          log_reverse_send_snapshot(
            peer, request_id, context.remote_ops_by_peer[peer].size());
        }
      }
    }
    auto* const previous_waiter_registrations =
      MemoryNode::current_storage_owner_maintenance_waiter_registrations_;
    MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
      &context.waiter_registrations;
    const Stage2ContextOwnerKey previous_context_owner =
      MemoryNode::current_storage_owner_maintenance_context_owner_;
    MemoryNode::current_storage_owner_maintenance_context_owner_ = context.ready_owner;
    const bool progressed = drive_context(context);
    MemoryNode::current_storage_owner_maintenance_context_owner_ =
      previous_context_owner;
    MemoryNode::current_storage_owner_maintenance_waiter_registrations_ =
      previous_waiter_registrations;
    return progressed;
  };

bool Stage2Worker::drive_ready_event(const Stage2ReadyContextEvent& event) {
    lib_assert(event.owner.worker_id == worker_id_ &&
                 event.owner.slot < contexts.size(),
               "stage2 ready event targets the wrong executor");
    Stage2Context& context = contexts[event.owner.slot];
    if (!context.active || context.ready_owner != event.owner) {
      return false;
    }
    return drive_owned_context(context);
  };



std::pair<bool, size_t> Stage2Worker::drain_ready_contexts() {
    bool progressed = false;
    size_t drained = 0;
    Stage2ReadyContextEvent event;
    // One active generation contributes at most one ordinary queued ticket;
    // the 2x bound also covers stale prior-generation tickets during reuse.
    const size_t ticket_budget = context_capacity * 2;
    while (drained < ticket_budget && ready_context_queue->try_pop(event)) {
      ++drained;
      progressed = drive_ready_event(event) || progressed;
    }
    if (drained != 0) {
      node_.storage_owner_maintenance_ready_tickets_drained_.fetch_add(
        drained, std::memory_order_relaxed);
    }

    if (ready_context_queue->overflowed()) {
      overflow_ready_events.clear();
      const size_t recovered =
        ready_context_queue->recover_overflow(overflow_ready_events);
      node_.storage_owner_maintenance_ready_overflow_scans_.fetch_add(
        1, std::memory_order_relaxed);
      node_.storage_owner_maintenance_context_slots_scanned_.fetch_add(
        context_capacity, std::memory_order_relaxed);
      node_.storage_owner_maintenance_ready_tickets_drained_.fetch_add(
        recovered, std::memory_order_relaxed);
      drained += recovered;
      for (const Stage2ReadyContextEvent& recovered_event :
           overflow_ready_events) {
        progressed = drive_ready_event(recovered_event) || progressed;
      }
    }
    return std::pair<bool, size_t>{progressed, drained};
  };




Stage2Worker::Stage2Context* Stage2Worker::try_admit_context() {
    const Stage2AdmissionDecision admission = decide_stage2_admission(
      states.full(),
      node_.storage_owner_maintenance_shutdown_.load(std::memory_order_acquire));
    if (admission == Stage2AdmissionDecision::unavailable) {
      return nullptr;
    }
    const size_t local_context_limit =
      stage2_worker_context_admission_limit(
        config.storage_owner_rpc_depth);
    if (states.size() >= local_context_limit) {
      // Do not let this worker monopolize the process-wide context allowance.
      // Its contexts can use only this worker's registered search lanes;
      // leaving the remaining global permits for sibling workers is what
      // exposes the already-allocated RDMA concurrency.
      node_.storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }
    if (storage_owner_consolidation_enabled()) {
      const u64 active = node_.storage_owner_maintenance_active_workers_.load(
        std::memory_order_acquire);
      std::lock_guard<std::mutex> deferred_lock(
        node_.storage_owner_deferred_refinements_mutex_);
      const u64 reserved = active * config.storage_owner_batch_max;
      const u64 queued = node_.storage_owner_deferred_refinements_.size();
      if (queued + reserved >= config.storage_owner_maintenance_queue_depth) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return nullptr;
      }
    }
    if (!node_.try_acquire_storage_owner_maintenance_slot(config)) {
      node_.storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }

    const size_t batch_limit =
      std::max<size_t>(1, config.storage_owner_batch_max);
    const auto acquire_context = [&]() {
      const auto handle = states.try_acquire();
      lib_assert(handle.has_value(),
                 "stage2 context tracker unexpectedly exhausted");
      Stage2Context& context = contexts[handle->slot];
      reset_context(context);
      context.handle = *handle;
      context.ready_owner = ready_context_queue->activate(handle->slot);
      context.active = true;
      return &context;
    };

    // Repair continuations own an already-admitted maintenance sequence and
    // therefore take priority over new Stage2 work. This removes the stale
    // Stage2 finalization's attempted backlinks before advancing the watermark and proves
    // the dedicated queue cannot grow across successive admission waves.
    if (node_.storage_owner_repair_tasks_ != nullptr) {
      StorageOwnerMaintenanceTask repair;
      if (node_.storage_owner_repair_tasks_->try_pop(repair)) {
        if (node_.storage_owner_cleanup_ready(repair, config)) {
          Stage2Context& context = *acquire_context();
          context.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
          context.tasks.push_back(std::move(repair));
          repair = StorageOwnerMaintenanceTask{};
          while (context.tasks.size() < batch_limit &&
                 node_.storage_owner_repair_tasks_->try_pop(repair)) {
            if (!node_.storage_owner_cleanup_ready(repair, config)) {
              lib_assert(node_.storage_owner_repair_tasks_->try_push(
                           std::move(repair)),
                         "failed to return a not-yet-ready repair descriptor");
              break;
            }
            context.tasks.push_back(std::move(repair));
            repair = StorageOwnerMaintenanceTask{};
          }
          node_.notify_storage_owner_maintenance_capacity();
          return &context;
        }
        lib_assert(node_.storage_owner_repair_tasks_->try_push(std::move(repair)),
                   "failed to return a not-yet-ready repair descriptor");
      }
    }

    std::unique_lock<std::mutex> lock(node_.storage_owner_maintenance_mutex_);
    const auto admission_now = std::chrono::steady_clock::now();
    const bool cleanup_ready = std::any_of(
      node_.storage_owner_cleanup_tasks_.begin(), node_.storage_owner_cleanup_tasks_.end(),
      [&](const StorageOwnerMaintenanceTask& task) {
        return node_.storage_owner_cleanup_ready(task, config) &&
          admission_now >= task.retry_not_before;
      });
    const auto packing_parameters =
      node_.storage_owner_stage2_packing_.parameters();
    Stage2PackingDecision packing_decision;
    if (!node_.storage_owner_stage2_tasks_.empty()) {
      packing_decision = decide_stage2_packing(
        node_.storage_owner_stage2_tasks_.size(), batch_limit,
        node_.storage_owner_stage2_tasks_.front().queued_at, admission_now,
        config.storage_owner_stage2_batch_max_wait_us,
        packing_parameters.estimated_arrival_interval_us);
    }
    const auto stage2_not_before = node_.storage_owner_stage2_tasks_.empty()
      ? admission_now
      : node_.storage_owner_stage2_tasks_.front().queued_at +
          std::chrono::milliseconds(
            config.storage_owner_stage2_initial_delay_ms);
    const bool stage2_ready = packing_decision.ready &&
      admission_now >= stage2_not_before;
    const bool choose_stage2 =
      stage2_ready &&
      (!cleanup_ready || node_.storage_owner_stage2_tasks_.front().queued_at <=
                            node_.storage_owner_cleanup_tasks_.front().queued_at);
    if (!choose_stage2 && !cleanup_ready) {
      static std::atomic<u32> cleanup_debug_count{0};
      const bool cleanup_stuck =
        !node_.storage_owner_cleanup_tasks_.empty() &&
        admission_now - node_.storage_owner_cleanup_tasks_.front().queued_at >
          std::chrono::seconds(60);
      if ((std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr ||
           cleanup_stuck) &&
          !node_.storage_owner_cleanup_tasks_.empty() &&
          cleanup_debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        const auto& task = node_.storage_owner_cleanup_tasks_.front();
        std::osyncstream(std::cerr) << "[cleanup-admission-debug] self=" << node_.storage_id_
                  << " worker=" << worker_id_
                  << " pending=" << node_.storage_owner_cleanup_tasks_.size()
                  << " dependencies=" <<
              node_.storage_owner_maintenance_dependencies_.size()
                  << " id=" << task.id << " generation=" << task.generation
                  << " target=" << task.target.raw_address
                  << " sequence=" << task.maintenance_sequence
                  << " repair_only=" << task.cleanup_repair_only
                  << " retry_ms=" << std::chrono::duration_cast<
                      std::chrono::milliseconds>(
                        task.retry_not_before - admission_now).count()
                  << std::endl;
      }
      node_.storage_owner_maintenance_active_workers_.fetch_sub(
        1, std::memory_order_acq_rel);
      return nullptr;
    }

    Stage2Context* admitted_context = nullptr;
    if (choose_stage2) {
      const size_t queued_at_admission =
        node_.storage_owner_stage2_tasks_.size();
      const auto oldest_queued_at =
        node_.storage_owner_stage2_tasks_.front().queued_at;
      const u64 oldest_wait_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::max<std::chrono::steady_clock::duration>(
            std::chrono::steady_clock::duration::zero(),
            admission_now - oldest_queued_at)).count());
      const size_t selected_pop_limit = std::min(
        batch_limit, std::max<size_t>(1, packing_decision.pop_limit));
      const size_t pop_limit = stage2_execution_slice_limit(
        selected_pop_limit, batch_limit);
      const size_t actual_pop_count = std::min(
        pop_limit, node_.storage_owner_stage2_tasks_.size());
      lib_assert(actual_pop_count != 0 &&
                   actual_pop_count <= std::numeric_limits<u32>::max(),
                 "Stage2 execution slice exceeds its bounded context");
      admitted_context = acquire_context();
      Stage2Context& context = *admitted_context;
      context.kind = StorageOwnerMaintenanceKind::finalize_insert;
      while (!node_.storage_owner_stage2_tasks_.empty() &&
             context.tasks.size() < pop_limit) {
        context.tasks.push_back(
          std::move(node_.storage_owner_stage2_tasks_.front()));
        node_.storage_owner_stage2_tasks_.pop_front();
      }
      lib_assert(context.tasks.size() == actual_pop_count,
                 "Stage2 queue pop diverged from its selected B8 slice");
      node_.storage_owner_stage2_packing_.observe_admission(
        packing_decision.reason, context.tasks.size(), oldest_wait_ns,
        packing_decision.wait_budget_us,
        packing_decision.target_batch, queued_at_admission);
      node_.storage_owner_stage2_batches_.fetch_add(1, std::memory_order_relaxed);
      node_.storage_owner_stage2_batched_items_.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
    } else {
      admitted_context = acquire_context();
      Stage2Context& context = *admitted_context;
      context.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
      node_.storage_owner_stage2_packing_.observe_admission(
        Stage2PackingFlushReason::cleanup, 0, 0, 0);
      vec<StorageOwnerMaintenanceTask> blocked;
      while (!node_.storage_owner_cleanup_tasks_.empty() &&
             context.tasks.size() < batch_limit) {
        StorageOwnerMaintenanceTask next =
          cleanup_schedule_pop(node_.storage_owner_cleanup_tasks_);
        if (node_.storage_owner_cleanup_ready(next, config) &&
            std::chrono::steady_clock::now() >= next.retry_not_before) {
          context.tasks.push_back(std::move(next));
        } else {
          blocked.push_back(std::move(next));
        }
      }
      for (StorageOwnerMaintenanceTask& task : blocked) {
        cleanup_schedule_push(node_.storage_owner_cleanup_tasks_, std::move(task));
      }
    }
    lib_assert(admitted_context != nullptr,
               "Stage2 admission lost its context");
    Stage2Context& context = *admitted_context;
    lock.unlock();
    // Removing descriptors from the bounded runnable queue is an admission
    // edge just like completing a Stage2 sequence. Wake a parked Stage1 arm
    // even when queue capacity, rather than completion credit, was the last
    // exhausted resource.
    node_.wake_peer_stage1_admission_waiters();
    node_.notify_storage_owner_maintenance_capacity();
    lib_assert(!context.tasks.empty(),
               "stage2 admitted an empty maintenance context");
    if (std::getenv("DUETANN_STAGE2_STALL_DEBUG") != nullptr) {
      std::osyncstream(std::cerr) << "[stage2-admit] worker=" << worker_id_
                << " slot=" << context.handle.slot
                << " generation=" << context.handle.generation
                << " first_sequence=" << context.tasks.front().maintenance_sequence
                << " last_sequence=" << context.tasks.back().maintenance_sequence
                << " tasks=" << context.tasks.size()
                << " active=" << node_.storage_owner_maintenance_active_workers_.load(
                  std::memory_order_acquire) << std::endl;
    }
    return &context;
  };

void Stage2Worker::flush_idle_timing() {
    if (unpublished_idle_waits != 0) {
      node_.storage_owner_maintenance_worker_idle_ns_.fetch_add(
        unpublished_idle_wait_ns, std::memory_order_relaxed);
      node_.storage_owner_maintenance_worker_idle_waits_.fetch_add(
        unpublished_idle_waits, std::memory_order_relaxed);
      unpublished_idle_wait_ns = 0;
      unpublished_idle_waits = 0;
    }
    if (unpublished_context_slots_scanned != 0) {
      node_.storage_owner_maintenance_context_slots_scanned_.fetch_add(
        unpublished_context_slots_scanned, std::memory_order_relaxed);
      unpublished_context_slots_scanned = 0;
    }
  };



}  // namespace memory_node_storage_owner_maintenance_detail

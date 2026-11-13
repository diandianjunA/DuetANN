#include "memory_node/storage_owner_maintenance/worker_internal.hh"

#include <cstdlib>

namespace memory_node_storage_owner_maintenance_detail {

Stage2SearchAdvanceResult Stage2Worker::prepare_local(Stage2Context& context) {
    const auto debug_wait = [&](const char* reason,
                                const StorageOwnerMaintenanceTask& task,
                                u64 header = 0) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") == nullptr ||
          std::chrono::steady_clock::now() - task.queued_at <=
            std::chrono::seconds(5)) {
        return;
      }
      static std::atomic<u32> debug_count{0};
      if (debug_count.fetch_add(1, std::memory_order_relaxed) >= 64) return;
      std::cerr << "[stage2-search-wait] reason=" << reason
                << " worker=" << worker_id_
                << " id=" << task.id
                << " sequence=" << task.maintenance_sequence
                << " target=" << task.target.raw_address
                << " header=" << header << std::endl;
    };
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
        !context.tasks.empty() &&
        std::chrono::steady_clock::now() - context.tasks.front().queued_at >
          std::chrono::seconds(5)) {
      static std::atomic<u32> debug_count{0};
      if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        const auto& io = context.search_io;
        std::cerr << "[stage2-search-state] worker=" << worker_id_
                  << " input_prepared=" << context.search_input_prepared
                  << " started_ns=" << context.search_started_ns
                  << " initialized=" << io.initialized
                  << " score_phase=" << static_cast<u32>(io.score_phase)
                  << " graph_phase=" << static_cast<u32>(io.graph_phase)
                  << " pending_vectors=" << io.pending_vectors.size()
                  << " score_rpcs=" << io.score_home_rpc_count
                  << " graph_rpcs=" << io.home_expand_rpc_count
                  << " first_id=" << context.tasks.front().id << std::endl;
      }
    }
    if (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      const auto transition = states.begin_remote_search(context.handle, 0);
      lib_assert(transition == Stage2EventResult::phase_advanced,
                 "cleanup stage2 failed to enter prune_ready");
      return Stage2SearchAdvanceResult::complete;
    }

    if (!context.search_stall_diagnostic_emitted &&
        context.search_started_ns != 0 &&
        steady_now_ns() - context.search_started_ns > 10'000'000'000ull) {
      context.search_stall_diagnostic_emitted = true;
      std::cerr << "[storage-owner] Stage2 continuation stalled"
                << " shard=" << node_.storage_id_
                << " context=" << context.handle.slot
                << " score_phase="
                << static_cast<u32>(context.search_io.score_phase)
                << " graph_phase="
                << static_cast<u32>(context.search_io.graph_phase)
                << " initialized=" << context.search_io.initialized
                << " seeded=" << std::count(
                     context.search_io.search_seeded.begin(),
                     context.search_io.search_seeded.end(), u8{1})
                << " score_consumers=" << context.search_io.score_consumers.size()
                << " graph_consumers=" << context.search_io.graph_consumers.size()
                << " pending_vectors=" << context.search_io.pending_vectors.size()
                << " score_rpcs=" << context.search_io.score_home_rpc_count
                << " graph_rpcs=" << context.search_io.home_expand_rpc_count
                << std::endl;
    }

    if (!context.search_input_prepared) {
      context.search_started_ns = steady_now_ns();
      context.targets.clear();
      context.targets.resize(context.tasks.size());

      // Retire stale Stage1 records before reading construction inputs. Once
      // asynchronous continuation starts, its private beam resumes across I/O
      // waits. Pending candidates can improve this graph, while the later
      // certificate planner independently requires an already-published parent.
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.maintenance_sequence == 0) continue;
        lib_assert(node_.local_shard(task.target.memory_node()),
                   "Stage2 must execute on the Stage1 physical shard");

        NodeSnapshot target_snapshot;
        const StableNodeSnapshotState target_state =
          node_.storage_owner_physical_node_state(
            task.id, task.generation, task.target, &target_snapshot);
        if (target_state == StableNodeSnapshotState::retryable) {
          debug_wait("target_retryable", task, target_snapshot.header);
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
        if (target_state == StableNodeSnapshotState::terminal) {
          if (!complete_stale_stage2(task)) {
            debug_wait("stale_cleanup", task, target_snapshot.header);
            return Stage2SearchAdvanceResult::waiting_rdma;
          }
          task.maintenance_sequence = 0;
          continue;
        }
        context.targets[item] = std::move(target_snapshot);
      }

      size_t ready = 0;
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        if (context.tasks[item].maintenance_sequence == 0) continue;
        if (ready != item) {
          context.tasks[ready] = std::move(context.tasks[item]);
          context.targets[ready] = std::move(context.targets[item]);
        }
        ++ready;
      }
      context.tasks.resize(ready);
      context.targets.resize(ready);

      if (!context.tasks.empty()) {
        const vec<RemotePtr> entries = node_.local_centroid_route_entries();
        if (entries.empty()) return Stage2SearchAdvanceResult::waiting_rdma;
        const bool reuse_beam = storage_owner_reuse_stage1_beam();
        for (size_t item = 0; item < context.tasks.size(); ++item) {
          StorageOwnerMaintenanceTask& task = context.tasks[item];
          if (reuse_beam && !task.stage1_beam.empty()) continue;
          const NodeSnapshot& target = context.targets[item];
          vec<element_t> decoded;
          if (VamanaNode::vector_dtype() == VectorDType::float32) {
            decoded.resize(VamanaNode::DIM);
            decode_storage_vector_to_float(
              target.vector_data.data(), VamanaNode::vector_dtype(),
              VamanaNode::DIM, decoded.data());
          }
          vec<memory_node_detail::BeamEntry> refreshed_beam;
          vec<RemotePtr> refreshed_frontier;
          const vec<RemotePtr> refreshed_candidates =
            node_.partition_local_search_candidates(
              span<const element_t>{decoded}, entries, config, nullptr,
              target.vector_data.data(), &refreshed_beam,
              &refreshed_frontier, false, nullptr, true);
          if (refreshed_candidates.empty() || refreshed_beam.empty()) {
            debug_wait("candidate_refresh", task);
            return Stage2SearchAdvanceResult::waiting_rdma;
          }
          task.stage1_beam = std::move(refreshed_beam);
          task.stage1_remote_frontier = std::move(refreshed_frontier);
          task.search_dependencies = {};
        }
      }
      context.search_input_prepared = true;
    }

    // Advance every independent continuation by one dependency step per
    // wave. Graph and vector reads that are ready at the same time are issued
    // across the complete context batch, eliminating the per-task RDMA RTT
    // chain while preserving each task's private beam and convergence state.
    Stage2SearchIoState& search_io = context.search_io;
    const Stage2SearchAdvanceResult search_result =
      node_.advance_stage2_search_candidates_batched(
        span<const StorageOwnerMaintenanceTask>{context.tasks},
        span<const NodeSnapshot>{context.targets},
        context.continued_candidates_by_task, search_io, config);
    if (search_result != Stage2SearchAdvanceResult::complete) {
      return search_result;
    }
    // Record the continuation at its semantic completion boundary.  The
    // prune phase can retry authority, lock, placement, or reverse work many
    // times; measuring there both double-counted one search and charged those
    // unrelated waits to continuation_search.
    lib_assert(!context.search_timing_recorded &&
                 context.search_started_ns != 0,
               "Stage2 continuation completion timing was recorded twice");
    auto& search_timing =
      node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::continuation_search)];
    search_timing.attempts.fetch_add(1, std::memory_order_relaxed);
    search_timing.task_attempts.fetch_add(
      context.tasks.size(), std::memory_order_relaxed);
    search_timing.elapsed_ns.fetch_add(
      steady_now_ns() - context.search_started_ns,
      std::memory_order_relaxed);
    context.search_timing_recorded = true;
    lib_assert(context.continued_candidates_by_task.size() ==
                 context.tasks.size(),
               "Stage2 continuation lost context/task correlation");
    for (const vec<RemotePtr>& candidates :
         context.continued_candidates_by_task) {
      lib_assert(candidates.size() <= construction_width,
                 "stage2 continuation exceeded construction width L");
    }

    // The Stage1 owner remains the sole owner of one logical beam. Remote
    // homes execute only the selected expansion plus same-home scoring; they
    // never restart or recursively advance a shard-local search. There is
    // therefore no remote-search ACK mask to wait for.
    constexpr u64 expected_mask = 0;
    const auto transition =
      states.begin_remote_search(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 failed to enter remote_search_pending");
    return Stage2SearchAdvanceResult::complete;
  };

  // Termination releases execution resources, but an incomplete cleanup is an
  // explicit permanent failure. Its sequence and physical dependency remain
  // unfinished so the durable watermark can never certify partial maintenance.



void Stage2Worker::terminate_stuck_stage2(Stage2Context& context) {
    if (!release_rebindable_context_lane(context)) {
      const auto retry_at = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(1);
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        task.retry_not_before = retry_at;
      }
      return;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.maintenance_sequence == 0) continue;
      const char* stale_reason = "not_attempted";
      if (!complete_stale_stage2(task, &stale_reason)) {
        static std::atomic<u32> terminate_cleanup_logs{0};
        const u32 log_index =
          terminate_cleanup_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 16) {
          std::cerr << "[storage-owner] stage2 termination cleanup incomplete"
                    << " self_shard=" << node_.storage_id_
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
    }
    const Stage2ContextHandle handle = context.handle;
    reset_context(context);
    lib_assert(states.release_retryable(handle),
               "terminated stage2 context retained an asynchronous request");
    node_.storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    node_.notify_storage_owner_maintenance_capacity();
  };




void Stage2Worker::defer_stage2_retry(Stage2Context& context,
                                      bool permanent) {
    const auto now = std::chrono::steady_clock::now();
    if (permanent) {
      auto first_permanent_defer =
        std::chrono::steady_clock::time_point::max();
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        if (task.stage2_first_permanent_defer_at ==
            std::chrono::steady_clock::time_point{}) {
          task.stage2_first_permanent_defer_at = now;
        }
        first_permanent_defer = std::min(
          first_permanent_defer, task.stage2_first_permanent_defer_at);
      }
      if (config.storage_owner_stage2_max_retry_ms != 0 &&
          now - first_permanent_defer >
            std::chrono::milliseconds(
              config.storage_owner_stage2_max_retry_ms)) {
        terminate_stuck_stage2(context);
        return;
      }
    }
    const auto retry_at = now + std::chrono::milliseconds(1);
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.retry_not_before = retry_at;
    }
  };




void Stage2Worker::cancel_context_reconcile(Stage2Context& context) {
    auto& batch = context.reconcile_batch;
    for (const Stage2ReconcileChunk& chunk : batch.chunks()) {
      if (!chunk.complete && chunk.request_id != 0) {
        node_.cancel_peer_rpc_response(chunk.request_id);
      }
    }
    batch.clear();
  };




void Stage2Worker::finish_reverse_prepare_timing(Stage2Context& context) {
    if (!context.reverse_prepare_timing_active) return;
    auto& timing = node_.storage_owner_stage2_phase_timing_[
      static_cast<size_t>(StorageOwnerStage2TimingPhase::reverse_prepare)];
    timing.attempts.fetch_add(1, std::memory_order_relaxed);
    timing.task_attempts.fetch_add(
      context.tasks.size(), std::memory_order_relaxed);
    timing.elapsed_ns.fetch_add(
      steady_now_ns() - context.reverse_prepare_started_ns,
      std::memory_order_relaxed);
    context.reverse_prepare_timing_active = false;
    context.reverse_prepare_started_ns = 0;
  };


}  // namespace memory_node_storage_owner_maintenance_detail

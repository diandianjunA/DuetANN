#include "memory_node/storage_owner_maintenance/worker_internal.hh"

#include <cstdlib>

namespace memory_node_storage_owner_maintenance_detail {

bool Stage2Worker::prepare_stage2_reverse(Stage2Context& context) {
    const auto debug_event = [&](u32 event, const StorageOwnerMaintenanceTask& task,
                                 u64 first, u64 second = 0) {
      static const bool enabled =
        std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr;
      if (!enabled || event >= 32 || (debug_parent_events_ & (u32{1} << event)) ||
          std::none_of(context.tasks.begin(), context.tasks.end(),
            [](const StorageOwnerMaintenanceTask& candidate) {
              return std::chrono::steady_clock::now() - candidate.queued_at >
                std::chrono::seconds(10);
            })) {
        return;
      }
      debug_parent_events_ |= u32{1} << event;
      std::cerr << "[stage2-parent-debug] worker=" << worker_id_
                << " event=" << event << " id=" << task.id
                << " first=" << first << " second=" << second << std::endl;
    };
    if (context.item_pipeline_started) {
      vec<size_t> ready_items;
      vec<u32> placement_authorities;
      vec<protocol::AuthorityPlacementItem> placements;
      vec<size_t> placement_items;
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        if (context.tasks[item].maintenance_sequence == 0 ||
            context.finalize_subphases[item] !=
              Stage2FinalizeSubphase::placement_ready) {
          continue;
        }
        const auto action = stage2_item_pipeline_action(
          context.finalize_subphases[item],
          context.tasks[item].retry_not_before,
          std::chrono::steady_clock::now());
        if (action != Stage2ItemPipelineAction::finish_placement) continue;
        ready_items.push_back(item);
        const StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.placement_committed) continue;
        placement_items.push_back(item);
        placement_authorities.push_back(task.authority_shard);
        placements.push_back(protocol::AuthorityPlacementItem{
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
        });
      }
      if (!placements.empty()) {
        Stage2PhaseAttemptTimer placement_timer(
          node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
            StorageOwnerStage2TimingPhase::placement_authority)],
          placements.size());
        vec<protocol::AuthorityPlacementResult> results;
        const bool transported = node_.relocate_batch_via_authority(
          span<const u32>{placement_authorities},
          span<const protocol::AuthorityPlacementItem>{placements},
          results, config);
        if (!transported) {
          for (const size_t item : ready_items) {
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_item_retry(context, item, /*permanent=*/true);
          }
        } else {
          lib_assert(results.size() == placements.size(),
                     "batched Stage2 placement lost an item result");
          for (size_t slot = 0; slot < results.size(); ++slot) {
            const size_t item = placement_items[slot];
            StorageOwnerMaintenanceTask& task = context.tasks[item];
            const auto status = static_cast<protocol::AuthorityPlacementStatus>(
              results[slot].status);
            if (status == protocol::AuthorityPlacementStatus::busy) {
              node_.storage_owner_maintenance_pressure_yields_.fetch_add(
                1, std::memory_order_relaxed);
              defer_stage2_item_retry(context, item);
              continue;
            }
            if (status == protocol::AuthorityPlacementStatus::stale) {
              if (!complete_stale_stage2(task)) {
                node_.storage_owner_maintenance_pressure_yields_.fetch_add(
                  1, std::memory_order_relaxed);
                defer_stage2_item_retry(context, item, /*permanent=*/true);
              } else {
                task.maintenance_sequence = 0;
                context.finalize_subphases[item] =
                  Stage2FinalizeSubphase::complete;
              }
              continue;
            }
            lib_assert(
              status == protocol::AuthorityPlacementStatus::committed ||
                status == protocol::AuthorityPlacementStatus::replay,
              "authority rejected a structurally valid Stage2 placement token");
            const u64 expected_version = task.final_target == task.target
              ? task.initial_placement_version
              : task.initial_placement_version + 1;
            lib_assert(results[slot].resulting_placement_version ==
                         expected_version,
                       "authority placement returned an unexpected version");
            task.placement_committed = true;
          }
        }
      }
      vec<size_t> centroid_items;
      vec<CentroidMembershipOp> centroid_adds;
      for (const size_t item : ready_items) {
        StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.maintenance_sequence == 0 || !task.placement_committed ||
            task.centroid_committed) {
          continue;
        }
        centroid_items.push_back(item);
        centroid_adds.push_back(make_centroid_membership_op(
          task, task.final_target, CentroidMembershipKind::add));
      }
      if (!centroid_adds.empty()) {
        if (!node_.apply_centroid_membership_fanout_and_wait(
              span<const CentroidMembershipOp>{centroid_adds}, config)) {
          for (const size_t item : centroid_items) {
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_item_retry(context, item, /*permanent=*/true);
          }
        } else {
          for (const size_t item : centroid_items) {
            context.tasks[item].centroid_committed = true;
          }
        }
      }
      vec<size_t> settlement_items;
      vec<u32> settlement_shards;
      vec<protocol::DynamicNodeControlItem> settlements;
      for (const size_t item : ready_items) {
        StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.maintenance_sequence == 0 || !task.placement_committed ||
            !task.centroid_committed || task.final_target == task.target ||
            task.allocation_settled) {
          continue;
        }
        lib_assert(migrated_source_tombstone_allowed(
                     task.placement_committed, task.centroid_committed),
                   "migrated source tombstoned before final centroid publication");
        const u64 source_header = node_.load_local_node_header_acquire(task.target);
        lib_assert((source_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0,
                   "Stage1 source was counted before final placement");
        (void)node_.mark_node_deleted(task.target, task.generation);
        settlement_items.push_back(item);
        settlement_shards.push_back(task.final_target.memory_node());
        settlements.push_back(protocol::DynamicNodeControlItem{
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
        });
      }
      if (!settlements.empty()) {
        vec<protocol::DynamicNodeControlResult> results;
        const bool transported = node_.control_dynamic_nodes_on_shards(
          span<const u32>{settlement_shards},
          span<const protocol::DynamicNodeControlItem>{settlements},
          results, config);
        lib_assert(!transported || results.size() == settlements.size(),
                   "batched allocation settlement lost an item result");
        for (size_t slot = 0; slot < settlement_items.size(); ++slot) {
          const size_t item = settlement_items[slot];
          if (transported && static_cast<protocol::DynamicNodeControlStatus>(
                results[slot].status) == protocol::DynamicNodeControlStatus::ok) {
            context.tasks[item].allocation_settled = true;
            continue;
          }
          node_.storage_owner_maintenance_pressure_yields_.fetch_add(
            1, std::memory_order_relaxed);
          defer_stage2_item_retry(context, item, /*permanent=*/true);
        }
      }
      const auto suffix_now = std::chrono::steady_clock::now();
      for (const size_t item : ready_items) {
        StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.maintenance_sequence == 0 || !task.placement_committed ||
            task.retry_not_before > suffix_now) {
          continue;
        }
        if (finish_stage2_after_reconcile(context, item)) {
          context.finalize_subphases[item] = Stage2FinalizeSubphase::complete;
        }
      }
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        if (context.tasks[item].maintenance_sequence == 0) {
          context.finalize_subphases[item] = Stage2FinalizeSubphase::complete;
        }
      }
      if (context.reconcile_batch.active()) {
        (void)advance_reconcile_pipeline(context);
      } else {
        (void)start_reconcile_barrier(context);
      }
      const bool all_complete = stage2_context_items_complete(
        span<const Stage2FinalizeSubphase>{
          context.finalize_subphases.data(), context.tasks.size()});
      if (!all_complete) return false;
      finish_reverse_prepare_timing(context);
      context.completion_handoff_started_ns = steady_now_ns();
      constexpr u64 expected_mask = 0;
      const Stage2EventResult transition =
        states.begin_reverse(context.handle, expected_mask);
      lib_assert(transition == Stage2EventResult::phase_advanced ||
                   transition == Stage2EventResult::ready_to_finalize,
                 "Stage2 finalization failed to enter reverse_pending");
      return true;
    }
    Stage2PhaseAttemptTimer phase_timer(
      node_.storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::freeze_prune)],
      context.tasks.size());
    // First finish every node record. No reverse edge or directory entry is
    // allowed to expose a destination whose vector/graph/PQ record is partial.
    vec<u32> gate_authorities;
    vec<protocol::AuthorityPlacementItem> gates;
    vec<size_t> gate_task_indices;
    gate_authorities.reserve(context.tasks.size());
    gates.reserve(context.tasks.size());
    gate_task_indices.reserve(context.tasks.size());
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      const RemotePtr gate_target = task.placement_committed
        ? task.final_target : task.target;
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      gate_authorities.push_back(task.authority_shard);
      gate_task_indices.push_back(item);
      gates.push_back(protocol::AuthorityPlacementItem{
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
      });
    }
    vec<protocol::AuthorityPlacementResult> gate_results;
    if (!gates.empty() && !node_.relocate_batch_via_authority(
          span<const u32>{gate_authorities},
          span<const protocol::AuthorityPlacementItem>{gates},
          gate_results, config)) {
      node_.storage_owner_stage2_gate_transport_retries_.fetch_add(
        1, std::memory_order_relaxed);
      node_.storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context, /*permanent=*/true);
      return false;
    }
    lib_assert(gate_results.size() == gate_task_indices.size(),
               "batched Stage2 gate lost a task result");
    // Do not freeze the first items in a batch if a later authority lease is
    // still busy.  A retry must cross node_ batch boundary as one unit.
    for (const protocol::AuthorityPlacementResult& gate_result :
         gate_results) {
      const auto status = static_cast<
        protocol::AuthorityPlacementStatus>(gate_result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) {
        node_.storage_owner_stage2_gate_busy_retries_.fetch_add(
          1, std::memory_order_relaxed);
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
    }

    const vec<RemotePtr> durable_route_entries =
      node_.local_centroid_route_entries();
    if (snapshots_by_task.size() < context.tasks.size()) {
      snapshots_by_task.resize(context.tasks.size());
    }
    if (snapshot_task_active.size() < context.tasks.size()) {
      snapshot_task_active.resize(context.tasks.size());
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      snapshot_candidates_by_task[item].clear();
      snapshots_by_task[item].clear();
      snapshot_task_active[item] = false;
    }
    for (size_t gate_slot = 0;
         gate_slot < gate_task_indices.size(); ++gate_slot) {
      const size_t item = gate_task_indices[gate_slot];
      StorageOwnerMaintenanceTask& task = context.tasks[item];

      // Arm publishes a runnable descriptor before the foreground authority
      // commit. Use a no-op placement CAS as the gate: while that mutation's
      // lease is pending it returns busy; after abort it returns stale; only a
      // committed current generation may perform any Stage2-visible graph
      // mutation. On retries after a real relocation, validate the resulting
      // physical pointer/version instead of the original Stage1 address.
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      const protocol::AuthorityPlacementResult& gate_result =
        gate_results[gate_slot];
      const auto gate_status = static_cast<
        service::storage_owner::AuthorityPlacementStatus>(
          gate_result.status);
      lib_assert(gate_status !=
                   service::storage_owner::AuthorityPlacementStatus::busy,
                 "Stage2 authority preflight lost a busy result");
      if (gate_status ==
          service::storage_owner::AuthorityPlacementStatus::stale) {
        if (!task.stage1_receipt_released) {
          if (!node_.release_resolved_local_stage1_receipt(task, config)) {
            return false;
          }
          task.stage1_receipt_released = true;
        }
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }
      if (gate_status !=
            service::storage_owner::AuthorityPlacementStatus::committed &&
          gate_status !=
            service::storage_owner::AuthorityPlacementStatus::replay) {
        node_.storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      lib_assert(gate_result.resulting_placement_version == gate_version,
                 "authority Stage2 gate returned an unexpected version");
      if (!task.stage1_receipt_released) {
        if (!node_.release_resolved_local_stage1_receipt(task, config)) {
          return false;
        }
        task.stage1_receipt_released = true;
      }

      const RemotePtr current_physical = task.placement_committed
        ? task.final_target : task.target;
      const StableNodeSnapshotState current_physical_state =
        node_.storage_owner_physical_node_state(
          task.id, task.generation, current_physical);
      if (current_physical_state == StableNodeSnapshotState::retryable) {
        node_.storage_owner_stage2_physical_retries_.fetch_add(
          1, std::memory_order_relaxed);
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
      if (current_physical_state == StableNodeSnapshotState::terminal) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }

      if (!task.stage2_prepared) {
        lib_assert(item < context.continued_candidates_by_task.size(),
                   "Stage2 context lost its continuation candidates");
        // Freeze the graph mutation plane at the same locked boundary as the
        // rebase snapshot. Queries still traverse node_ record, but every
        // ordinary reverse mutation retries after observing FROZEN. Thus an
        // edge ACKed before node_ boundary is in observed_adjacency and no edge
        // can be ACKed in the snapshot-to-publication window.
        GraphAdjacency observed_adjacency;
        bool target_current = false;
        bool adjacency_retryable = false;
        if (task.stage2_source_frozen) {
          // A different item in node_ batched context may have hit a fallible
          // authority/RPC boundary after node_ source was frozen.  Re-entry is
          // an ordinary state-machine retry, not a stale-node condition.
          const u64 frozen_header =
            node_.load_local_node_header_acquire(task.target);
          const byte_t* record = node_.index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const bool frozen_identity_current =
            VamanaNode::header_incarnation(frozen_header) ==
              task.target.incarnation() &&
            *reinterpret_cast<const node_t*>(
              record + VamanaNode::offset_id()) == task.id &&
            *reinterpret_cast<const u32*>(
              record + VamanaNode::offset_generation()) == task.generation &&
            (frozen_header & VamanaNode::HEADER_PROVISIONAL) != 0 &&
            (frozen_header & VamanaNode::HEADER_STAGE2_FROZEN) != 0 &&
            (frozen_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING)) == 0;
          if (frozen_identity_current) {
            if (!node_.read_graph_adjacency(task.target, observed_adjacency)) {
              adjacency_retryable = true;
            } else {
              target_current = !observed_adjacency.deleted &&
                observed_adjacency.generation == task.generation;
            }
          }
        } else {
          lib_assert(node_.storage_node_pointer_addressable(task.target),
                     "Stage2 freeze target is structurally invalid");
          const IncarnationLockResult target_lock =
            node_.try_lock_node(task.target);
          if (target_lock == IncarnationLockResult::busy) {
            node_.storage_owner_stage2_source_lock_retries_.fetch_add(
              1, std::memory_order_relaxed);
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          if (target_lock == IncarnationLockResult::stale) {
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          const u64 locked_header =
            node_.load_local_node_header_acquire(task.target);
          const byte_t* locked_record = node_.index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const node_t locked_id = *reinterpret_cast<const node_t*>(
            locked_record + VamanaNode::offset_id());
          const u32 locked_generation = *reinterpret_cast<const u32*>(
            locked_record + VamanaNode::offset_generation());
          const bool locked_identity_matches =
            VamanaNode::header_incarnation(locked_header) ==
              task.target.incarnation() &&
            locked_id == task.id &&
            locked_generation == task.generation;
          if (locked_identity_matches &&
              (locked_header & (VamanaNode::HEADER_DELETED |
                                VamanaNode::HEADER_RETIRING |
                                VamanaNode::HEADER_STAGE2_FROZEN)) == 0) {
            lib_assert((locked_header & VamanaNode::HEADER_PROVISIONAL) != 0,
                       "Stage1 source lost PROVISIONAL before Stage2 freeze");
          }
          const bool locked_target_current =
            locked_identity_matches &&
            (locked_header & VamanaNode::HEADER_PROVISIONAL) != 0 &&
            (locked_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING |
                              VamanaNode::HEADER_STAGE2_FROZEN)) == 0;
          if (locked_target_current) {
            if (!node_.read_graph_adjacency(task.target, observed_adjacency)) {
              adjacency_retryable = true;
            } else {
              target_current = !observed_adjacency.deleted &&
                observed_adjacency.generation == task.generation;
            }
          }
          if (target_current) {
            auto* header_ptr = reinterpret_cast<u64*>(
              node_.index_buffer_.get_full_buffer() +
              vamana::StorageLayoutResolver::header(task.target).offset);
            std::atomic_ref<u64>(*header_ptr).fetch_or(
              static_cast<u64>(VamanaNode::HEADER_STAGE2_FROZEN),
              std::memory_order_acq_rel);
            task.stage2_source_frozen = true;
          }
          node_.unlock_node(task.target);
        }
        if (adjacency_retryable) {
          node_.storage_owner_stage2_source_adjacency_retries_.fetch_add(
            1, std::memory_order_relaxed);
          node_.storage_owner_maintenance_pressure_yields_.fetch_add(
            1, std::memory_order_relaxed);
          defer_stage2_retry(context);
          return false;
        }
        if (!target_current) {
          if (!complete_stale_stage2(task)) return false;
          task.maintenance_sequence = 0;
          continue;
        }
        task.stage2_protected_children = observed_adjacency.provisional;
        lib_assert(task.stage2_protected_children.empty(),
                   "query-ineligible Stage1 source accepted a protected child");

        // The outgoing set combines global co-refinement candidates with the
        // stable Stage1 bridges needed for an independent reachability parent.
        // Concurrent reverse additions are replayed separately after pruning.
        snapshot_candidates_by_task[item] =
          context.continued_candidates_by_task[item];
        for (const RemotePtr candidate : task.stage1_backlink_targets) {
          if (!candidate.is_null() &&
              std::find(snapshot_candidates_by_task[item].begin(),
                        snapshot_candidates_by_task[item].end(), candidate) ==
                snapshot_candidates_by_task[item].end()) {
            snapshot_candidates_by_task[item].push_back(candidate);
          }
        }
        task.stage2_rebase_additions = select_stage2_rebase_additions(
          span<const RemotePtr>{context.continued_candidates_by_task[item]},
          span<const RemotePtr>{task.stage1_base_neighbors},
          span<const RemotePtr>{observed_adjacency.stable});
        if (storage_owner_consolidation_enabled()) {
          task.stage2_rebase_additions.clear();
        }
        snapshot_task_active[item] = true;
      }

    }

    materialize_stable_snapshot_wave(context.tasks.size());
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (!snapshot_task_active[item]) continue;
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      hashset_t<RemotePtr> skip;
      skip.insert(task.target);
      task.stage2_neighbors = node_.robust_prune_snapshot_refs_cpu(
        context.targets[item].vector_data.data(),
        VamanaNode::vector_dtype(),
        span<const NodeSnapshot* const>{snapshots_by_task[item]}, skip,
        config, config.R, true);
      vec<RemotePtr> stable_parents;
      for (const NodeSnapshot* candidate : snapshots_by_task[item]) {
        if (stage2_parent_is_stable(candidate->header, candidate->deleted)) {
          stable_parents.push_back(candidate->rptr);
        }
      }
      lib_assert(stable_parents.empty() ||
                   preserve_reconcile_mandatory_candidates(
                     span<const RemotePtr>{snapshot_candidates_by_task[item]},
                     span<const RemotePtr>{stable_parents.data(),
                       stable_parents.empty() ? size_t{0} : size_t{1}},
                     config.R, task.stage2_neighbors),
                 "Stage2 co-refinement lost every stable reachability parent");
      lib_assert(task.stage2_neighbors.size() <= config.R,
                 "online Stage2 finalization exceeded graph degree");
      task.final_home = task.target.memory_node();
      task.stage2_revalidated_home = task.final_home;
      task.stage2_prepared = true;
    }

    // Co-refinement publishes in place: a pending parent must never leave a
    // dangling source pointer in another node's adjacency. The source lock is
    // released only after the complete outgoing adjacency has been written.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      lib_assert(task.stage2_prepared,
                 "Stage2 outgoing publication bypassed frozen prune");
      if (task.final_target.is_null()) {
        if (task.final_home == task.target.memory_node()) {
          task.final_target = task.target;
        } else {
          const protocol::DynamicNodeControlItem allocation{
            .token = {
              .source_client = task.source_client,
              .item_index = task.operation_item_index,
              .client_batch_id = task.operation_batch_id,
            },
            .node_raw = task.target.raw_address,
            .id = task.id,
            .generation = task.generation,
            .authority_shard = task.authority_shard,
            .action = static_cast<u32>(
              protocol::DynamicNodeControlAction::allocate),
          };
          protocol::DynamicNodeControlResult allocation_result;
          if (!node_.control_dynamic_node_on_shard(
                task.final_home, allocation, allocation_result, config)) {
            node_.storage_owner_stage2_allocation_retries_.fetch_add(
              1, std::memory_order_relaxed);
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context, /*permanent=*/true);
            return false;
          }
          if (static_cast<protocol::DynamicNodeControlStatus>(
                allocation_result.status) !=
                protocol::DynamicNodeControlStatus::ok) {
            node_.storage_owner_stage2_allocation_retries_.fetch_add(
              1, std::memory_order_relaxed);
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context, /*permanent=*/true);
            return false;
          }
          const RemotePtr allocated{allocation_result.node_raw};
          if (allocated.is_null() ||
              allocated.memory_node() != task.final_home) {
            node_.storage_owner_maintenance_failed_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          task.final_target = allocated;
        }
      }

      if (!task.outgoing_committed) {
        if (task.final_target != task.target) {
          vec<element_t> components(VamanaNode::DIM);
          decode_storage_vector_to_float(
            context.targets[item].vector_data.data(),
            VamanaNode::vector_dtype(), VamanaNode::DIM,
            components.data());
          // The allocation header remains the publication lock until vector,
          // graph and PQ code are complete. The authority directory still
          // names the frozen source, so ordinary graph updates cannot discover
          // node_ destination during materialization.
          node_.write_new_node_on_shard(
            task.final_target, task.id,
            span<const element_t>{components}, task.stage2_neighbors,
            task.generation, false);
          lib_assert(task.stage2_protected_children.empty(),
                     "migration cannot bypass protected Stage1 children");
        } else {
          lib_assert(node_.storage_node_pointer_addressable(task.target),
                     "Stage2 publish target is structurally invalid");
          const IncarnationLockResult target_lock =
            node_.try_lock_node(task.target);
          if (target_lock == IncarnationLockResult::busy) {
            node_.storage_owner_stage2_publish_lock_retries_.fetch_add(
              1, std::memory_order_relaxed);
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          if (target_lock == IncarnationLockResult::stale) {
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          const u64 locked_header =
            node_.load_local_node_header_acquire(task.target);
          const byte_t* locked_record = node_.index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const bool current =
            VamanaNode::header_incarnation(locked_header) ==
              task.target.incarnation() &&
            *reinterpret_cast<const node_t*>(
              locked_record + VamanaNode::offset_id()) == task.id &&
            *reinterpret_cast<const u32*>(
              locked_record + VamanaNode::offset_generation()) ==
                task.generation &&
            (locked_header & VamanaNode::HEADER_STAGE2_FROZEN) != 0 &&
            (locked_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING)) == 0;
          if (!current) {
            node_.unlock_node(task.target);
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          GraphAdjacency adjacency;
          lib_assert(node_.read_graph_adjacency(task.target, adjacency),
                     "Stage2 target adjacency became unreadable");
          node_.write_graph_adjacency(
            task.target, task.stage2_neighbors,
            task.stage2_protected_children,
            task.generation, false);
          auto* header_ptr = reinterpret_cast<u64*>(
            node_.index_buffer_.get_full_buffer() +
            vamana::StorageLayoutResolver::header(task.target).offset);
          std::atomic_ref<u64>(*header_ptr).fetch_and(
            ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL |
                              VamanaNode::HEADER_STAGE2_FROZEN),
            std::memory_order_acq_rel);
          node_.unlock_node(task.target);
          task.stage2_source_frozen = false;
        }
        task.outgoing_committed = true;
        task.stage2_plan_sealed = true;
      }
    }

    vec<ReconcileReverseOp> rebase_ops;
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.maintenance_sequence == 0 || task.stage2_rebase_replayed) {
        continue;
      }
      for (const RemotePtr addition : task.stage2_rebase_additions) {
        rebase_ops.push_back(make_reconcile_op(
          task, task.final_target, RemotePtr{}, addition,
          ReconcileReverseOpKind::add));
      }
    }
    if (!rebase_ops.empty()) {
      debug_event(8, context.tasks.front(), rebase_ops.size());
    }
    if (!rebase_ops.empty() &&
        !apply_reconcile_ops(span<const ReconcileReverseOp>{rebase_ops})) {
      defer_stage2_retry(context, /*permanent=*/true);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.maintenance_sequence == 0 || task.stage2_rebase_replayed) {
        continue;
      }
      if (!task.stage2_rebase_additions.empty()) {
        GraphAdjacency final_adjacency;
        if (!node_.read_graph_adjacency(
              task.final_target, final_adjacency) || final_adjacency.deleted) {
          defer_stage2_retry(context, /*permanent=*/true);
          return false;
        }
        task.stage2_neighbors = final_adjacency.stable;
      }
      task.stage2_rebase_replayed = true;
    }
    if (!rebase_ops.empty()) {
      debug_event(9, context.tasks.front(), rebase_ops.size());
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

    phase_timer.finish();
    if (!context.reverse_prepare_timing_active) {
      context.reverse_prepare_started_ns = steady_now_ns();
      context.reverse_prepare_timing_active = true;
    }

    // The source lock freezes concurrent additions and captures the rebase
    // set, but only global continuation candidates decide the initial outgoing
    // prune. Those captured additions have now been replayed as ordinary
    // post-publication updates. A second freeze/prune cannot make parents live;
    // reachability churn is handled by the promotion certificate below.
    for (const StorageOwnerMaintenanceTask& task : context.tasks) {
      lib_assert(task.stage2_plan_sealed && task.outgoing_committed &&
                   !task.final_target.is_null(),
                 "Stage2 reached parent validation with an open placement plan");
      lib_assert(task.final_target != task.target ||
                   !task.stage2_source_frozen,
                 "in-place Stage2 publication left its source frozen");
    }

    // A sealed outgoing edge may retire before reverse reconciliation. It
    // remains part of the already-published adjacency (normal cleanup owns
    // that stale edge), but it cannot be selected as the mandatory promotion
    // parent. Revalidate all sealed neighbors in one shared wave and retain a
    // worker-local, order-preserving planner view. This also guarantees that a
    // retry advances past a dead first choice instead of livelocking on it.
    if (snapshots_by_task.size() < context.tasks.size()) {
      snapshots_by_task.resize(context.tasks.size());
    }
    if (snapshot_task_active.size() < context.tasks.size()) {
      snapshot_task_active.resize(context.tasks.size());
    }
    if (live_stage2_neighbors_by_task.size() < context.tasks.size()) {
      live_stage2_neighbors_by_task.resize(context.tasks.size());
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      snapshot_candidates_by_task[item].clear();
      snapshots_by_task[item].clear();
      live_stage2_neighbors_by_task[item].clear();
      snapshot_task_active[item] = !context.tasks[item].reverse_reconciled;
      if (snapshot_task_active[item]) {
        snapshot_candidates_by_task[item] =
          context.tasks[item].stage2_neighbors;
      }
    }
    // Liveness revalidation needs only the coherent header/incarnation pair,
    // not another copy of every D-byte vector. The compact identity wave also
    // fits several times more requests in the same registered scratch plane.
    materialize_stable_identity_wave(context.tasks.size());

    const auto publish_recovery_outgoing = [&](
        StorageOwnerMaintenanceTask& task,
        RemotePtr recovery_parent,
        vec<RemotePtr>& live_neighbors) {
      return publish_stage2_recovery_parent(
        task, recovery_parent, &live_neighbors);
    };

    // A Stage1 parent can be tombstoned while Stage2 is queued. Revalidate the
    // acknowledged protected slots before atomically promoting one final
    // stable bridge. No protected slot survives finalization, and Stage2 must
    // never silently commit an unreachable node.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.reverse_reconciled) continue;
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
          task.id < 100000016) {
        std::cerr << "[stage2-bridge-debug] id=" << task.id
                  << " target=" << task.target.raw_address
                  << " bridges=" << task.stage1_backlink_targets.size();
        for (const RemotePtr parent : task.stage1_backlink_targets) {
          std::cerr << " " << parent.raw_address;
        }
        std::cerr << std::endl;
      }
      auto expected_plan = plan_stage2_backlink_reconciliation(
        span<const RemotePtr>{task.stage1_backlink_targets},
        span<const RemotePtr>{live_stage2_neighbors_by_task[item]},
        task.stage2_promotion_committed
          ? task.stage2_promotion_parent : RemotePtr{});
      if (task.stage2_first_permanent_defer_at !=
            std::chrono::steady_clock::time_point{} &&
          !task.stage2_promotion_committed) {
        expected_plan.promotion_target = {};
      }
      if (expected_plan.promotion_target.is_null()) {
        debug_event(1, task, live_stage2_neighbors_by_task[item].size(),
                    task.stage1_backlink_targets.size());
        // All parents selected by the sealed plan can legitimately retire
        // before reverse reconciliation. After a permanent retry, prefer the
        // durable route over equally short-lived new parents. Under the child
        // lock, preserve concurrent live additions or replace one proven-stale
        // outgoing slot. Placement stays sealed, while future deletion cleanup
        // can still discover and remove the resulting incoming certificate.
        vec<RemotePtr> recovery_candidates = durable_route_entries;
        recovery_candidates.reserve(recovery_candidates.size() +
                                    task.stage1_base_neighbors.size());
        for (const RemotePtr candidate : task.stage1_base_neighbors) {
          if (std::find(recovery_candidates.begin(),
                        recovery_candidates.end(), candidate) ==
              recovery_candidates.end()) {
            recovery_candidates.push_back(candidate);
          }
        }
        recovery_candidates.erase(
          std::remove_if(
            recovery_candidates.begin(), recovery_candidates.end(),
            [&](RemotePtr candidate) {
              return candidate.is_null() ||
                candidate == task.target ||
                candidate == task.final_target ||
                !node_.storage_node_pointer_addressable(candidate);
            }),
          recovery_candidates.end());
        vec<StableNodeSnapshotState> recovery_states;
        const vec<NodeSnapshot> recovery_snapshots =
          node_.read_node_snapshots_batched(
            recovery_candidates, config,
            "stage2_reachability_recovery", &recovery_states);
        // Select from the coherent subset.  One hot route entry must not
        // suppress another independently verified recovery parent.
        RemotePtr recovery_parent;
        for (const NodeSnapshot& candidate : recovery_snapshots) {
          if (!stage2_parent_is_stable(
                candidate.header, candidate.deleted)) {
            continue;
          }
          recovery_parent = candidate.rptr;
          break;
        }
        debug_event(2, task, recovery_parent.raw_address,
                    recovery_snapshots.size());
        if (!recovery_parent.is_null() &&
            !publish_recovery_outgoing(
              task, recovery_parent,
              live_stage2_neighbors_by_task[item])) {
          defer_stage2_retry(context, /*permanent=*/true);
          return false;
        }
        expected_plan = plan_stage2_backlink_reconciliation(
          span<const RemotePtr>{task.stage1_backlink_targets},
          span<const RemotePtr>{live_stage2_neighbors_by_task[item]},
          task.stage2_promotion_committed
            ? task.stage2_promotion_parent : RemotePtr{});
      }
      const vec<RemotePtr> candidate_parents =
        stage2_revalidation_parents(
          span<const RemotePtr>{task.stage1_backlink_targets},
          task.stage2_promotion_committed
            ? task.stage2_promotion_parent : RemotePtr{},
          expected_plan.promotion_target);
      // Only Stage1 backlink targets can own a provisional reachability slot.
      // On a retry, the one selected promotion target may instead already own
      // the final stable certificate.  Ordinary base/final neighbors cannot
      // satisfy either postcondition, so scanning O(2R+L) of them (including a
      // full vector snapshot and a synchronous adjacency read apiece) was
      // pure work and dominated Stage2 latency.
      debug_event(6, task, candidate_parents.size(),
                  expected_plan.promotion_target.raw_address);
      vec<StableNodeSnapshotState> parent_states;
      const vec<NodeSnapshot> parent_snapshots =
        node_.read_node_snapshots_batched(
          candidate_parents, config, "stage2_parent_revalidation",
          &parent_states);
      if (std::find(parent_states.begin(), parent_states.end(),
                    StableNodeSnapshotState::retryable) !=
          parent_states.end()) {
        defer_stage2_retry(context);
        return false;
      }
      vec<RemotePtr> protected_parents;
      protected_parents.reserve(parent_snapshots.size());
      bool promotion_postcondition_holds = false;
      bool parent_adjacency_retryable = false;
      RemotePtr observed_promotion_parent;
      for (const NodeSnapshot& parent : parent_snapshots) {
        if (stage2_parent_is_stable(parent.header, parent.deleted)) {
          GraphAdjacency parent_adjacency;
          if (!node_.read_graph_adjacency(parent.rptr, parent_adjacency)) {
            // The vector/header observation proved node_ parent current. A
            // checksum/torn adjacency miss is therefore contention, not
            // evidence that its Stage1 bridge disappeared. Preserve every
            // persisted reconciliation field and retry the whole boundary.
            parent_adjacency_retryable = true;
            break;
          }
          if (parent_adjacency.deleted) {
            continue;
          }
          const bool protects_old = std::find(
            parent_adjacency.provisional.begin(),
            parent_adjacency.provisional.end(), task.target) !=
              parent_adjacency.provisional.end();
          const bool protects_final = !task.final_target.is_null() &&
            std::find(parent_adjacency.provisional.begin(),
                      parent_adjacency.provisional.end(),
                      task.final_target) !=
              parent_adjacency.provisional.end();
          if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
              task.id < 100000016) {
            std::cerr << "[stage2-bridge-debug] id=" << task.id
                      << " parent=" << parent.rptr.raw_address
                      << " old=" << protects_old << " final=" << protects_final
                      << " provisional=";
            for (const RemotePtr child : parent_adjacency.provisional) {
              std::cerr << " " << child.raw_address;
            }
            std::cerr << std::endl;
          }
          const bool stable_final_present =
            !task.final_target.is_null() &&
            std::find(parent_adjacency.stable.begin(),
                      parent_adjacency.stable.end(),
                      task.final_target) != parent_adjacency.stable.end();
          const bool lease_final_present =
            node_.storage_owner_certificate_leases_ != nullptr &&
            node_.storage_owner_certificate_leases_->contains(
              parent.rptr.raw_address, task.final_target.raw_address,
              task.maintenance_sequence);
          // An ordinary stable edge is not proof of this operation's mandatory
          // certificate. LeaseMerge must check the exact parent/child/sequence.
          const bool has_promoted_final =
            stage2_promotion_certificate_matches(
              config.storage_owner_lease_merge, stable_final_present,
              lease_final_present);
          if (has_promoted_final &&
              (!promotion_postcondition_holds ||
               parent.rptr == expected_plan.promotion_target)) {
            promotion_postcondition_holds = true;
            observed_promotion_parent = parent.rptr;
          }
          if (protects_old || protects_final) {
            protected_parents.push_back(parent.rptr);
          }
        }
      }
      if (parent_adjacency_retryable) {
        defer_stage2_retry(context);
        return false;
      }
      // Retired/reused parents already satisfy the remove postcondition and
      // must not poison a fixed retry payload. Keep only exact live protected
      // slots, even after promotion; the separately persisted stable
      // certificate carries the idempotent retry anchor.
      task.stage1_backlink_targets = std::move(protected_parents);
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
          task.id < 100000016) {
        std::cerr << "[stage2-bridge-debug] id=" << task.id
                  << " protected=" << protected_parents.size() << std::endl;
      }
      if (promotion_postcondition_holds) {
        stage2_replace_certificate_parent(
          observed_promotion_parent, task.stage2_promotion_parent,
          task.stage2_replaced_certificate_parents);
      } else {
        stage2_record_certificate_parent_attempt(
          task.stage2_promotion_parent,
          task.stage2_replaced_certificate_parents);
      }
      task.stage2_promotion_committed = promotion_postcondition_holds;
      task.stage2_promotion_parent = promotion_postcondition_holds
        ? observed_promotion_parent : RemotePtr{};
      debug_event(7, task, promotion_postcondition_holds,
                  parent_snapshots.size());
      // Every original protected parent may retire while Stage2 is queued.
      // If any live protected parent remains, the planner consumes it even
      // when final pruning produced no outgoing edge. Otherwise it establishes
      // one bounded stable bridge at a durable final parent. The promotion ACK
      // remains the gate for all later cleanup.
    }

    // Promotion is the durable reachability certificate. Persist the exact
    // liveness view before yielding, then install ordinary stable additions
    // followed by promotion in one audited per-target transaction. Obsolete
    // Stage1 removals remain a second barrier. This retains promotion-last
    // semantics without paying a separate promotion RTT.
    lib_assert(context.live_stage2_neighbors_by_task.size() >=
                 context.tasks.size(),
               "Stage2 context cannot persist its reconciliation plan");
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      context.live_stage2_neighbors_by_task[item] =
        live_stage2_neighbors_by_task[item];
    }
    context.item_pipeline_started = true;
    return false;
  };




bool Stage2Worker::prepare_cleanup_reverse(Stage2Context& context) {
    const auto failed = [&](const char* reason,
                            const StorageOwnerMaintenanceTask& task) {
      static std::atomic<u32> debug_count{0};
      const auto task_age = std::chrono::steady_clock::now() - task.queued_at;
      if ((std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr ||
           task_age > std::chrono::seconds(60)) &&
          debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
        std::cerr << "[cleanup-prepare-debug] self=" << node_.storage_id_
                  << " worker=" << worker_id_ << " reason=" << reason
                  << " id=" << task.id << " generation=" << task.generation
                  << " target=" << task.target.raw_address
                  << " sequence=" << task.maintenance_sequence
                  << " repair_only=" << task.cleanup_repair_only
                  << " authority_retired=" << task.cleanup_authority_retired
                  << " retiring=" << task.cleanup_retiring
                  << " children=" << task.cleanup_protected_children.size()
                  << std::endl;
      }
      return false;
    };
    vec<vec<RemotePtr>> cleanup_neighbors_by_task;
    cleanup_neighbors_by_task.reserve(context.tasks.size());
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }

    // Cleanup descriptors are deliberately runnable before the foreground
    // authority commit. First cross that token-fenced authority barrier; an
    // aborted mutation simply cancels its descriptor without ever quiescing
    // the still-current old generation.
    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.target.is_null()) {
        node_.storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        node_.complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
        task.maintenance_sequence = 0;
        continue;
      }
      if (!task.cleanup_repair_only) {
        if (!task.cleanup_authority_retired) {
          const protocol::AuthorityPlacementItem retirement_gate{
            .token = {
              .source_client = task.source_client,
              .item_index = task.operation_item_index,
              .client_batch_id = task.operation_batch_id,
            },
            .id = task.id,
            // desired_raw == 0 interprets node_ as a read-only proof that a
            // strictly newer authority generation retired target.
            .generation = task.generation,
            .expected_raw = task.target.raw_address,
            .desired_raw = 0,
            .expected_placement_version = 0,
          };
          protocol::AuthorityPlacementResult gate_result;
          if (!node_.relocate_via_authority(
                task.authority_shard, retirement_gate, gate_result, config)) {
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            return failed("authority_transport", task);
          }
          const auto gate_status = static_cast<
            protocol::AuthorityPlacementStatus>(gate_result.status);
          if (gate_status == protocol::AuthorityPlacementStatus::busy) {
            node_.storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            return failed("authority_busy", task);
          }
          if (gate_status == protocol::AuthorityPlacementStatus::stale) {
            if (!complete_stale_cleanup(task)) {
              return failed("stale_cleanup_release", task);
            }
            task.maintenance_sequence = 0;
            continue;
          }
          if (gate_status != protocol::AuthorityPlacementStatus::committed &&
              gate_status != protocol::AuthorityPlacementStatus::replay) {
            node_.storage_owner_maintenance_failed_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          task.cleanup_authority_retired = true;
        }
      } else {
        task.centroid_committed = true;
      }
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      ++ready;
    }
    context.tasks.resize(ready);

    // Do not quiesce an earlier item in the batch while a later item's
    // authority lease is still pending. Once every item has crossed its
    // barrier, protected-child reparenting may proceed independently and be
    // retried from the task's bounded progress vectors.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.cleanup_repair_only) continue;
      const CleanupParentQuiesceResult quiesce_result =
        quiesce_cleanup_parent(task);
      if (quiesce_result == CleanupParentQuiesceResult::stale) {
        if (!complete_stale_cleanup(task)) {
          return failed("stale_cleanup_release", task);
        }
        task.maintenance_sequence = 0;
        continue;
      }
      if (quiesce_result == CleanupParentQuiesceResult::busy) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return failed("parent_quiesce", task);
      }
      if (!reparent_cleanup_children(task)) {
        node_.storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return failed("reparent", task);
      }
    }

    // Stale incarnations above are complete, not retryable. Remove them
    // before centroid/removal vectors are built so task-index correlation is
    // preserved for the remainder of node_ cleanup context.
    ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (context.tasks[item].maintenance_sequence == 0) continue;
      if (ready != item) context.tasks[ready] = std::move(context.tasks[item]);
      ++ready;
    }
    context.tasks.resize(ready);

    // Withdraw the exact tagged identity, update the FP64 sum/count, elect
    // replacement entries and publish the complete route before any matching
    // node can become DELETED. A lost response is harmless: the accounted bit
    // turns retry into a no-op, while the node remains RETIRING and readable.
    vec<CentroidMembershipOp> centroid_removes;
    centroid_removes.reserve(context.tasks.size());
    for (const StorageOwnerMaintenanceTask& task : context.tasks) {
      if (!task.centroid_committed) {
        centroid_removes.push_back(make_centroid_membership_op(
          task, task.target, CentroidMembershipKind::remove));
      }
    }
    if (!centroid_removes.empty() &&
        !node_.apply_centroid_membership_fanout_and_wait(
          span<const CentroidMembershipOp>{centroid_removes}, config)) {
      node_.storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return failed("centroid_removal", context.tasks.front());
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.centroid_committed = true;
    }

    ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (!task.cleanup_repair_only) {
        NodeSnapshot before_delete;
        if (!node_.read_node_snapshot(task.target, before_delete)) {
          return failed("before_delete_snapshot", task);
        }
        if (!before_delete.deleted) {
          if (before_delete.id != task.id ||
              before_delete.generation != task.generation ||
              (before_delete.header & VamanaNode::HEADER_RETIRING) == 0 ||
              !cleanup_tombstone_allowed(
                task.cleanup_authority_retired, task.cleanup_retiring,
                task.centroid_committed) ||
              !node_.mark_node_deleted(task.target, task.generation)) {
            return failed("tombstone", task);
          }
        }
      }
      NodeSnapshot deleted_snapshot;
      const bool readable = node_.read_node_snapshot(task.target, deleted_snapshot);
      lib_assert(readable, "local cleanup snapshot was unreadable");
      if (!deleted_snapshot.deleted && !task.cleanup_repair_only) {
        if (!complete_stale_cleanup(task)) {
          return failed("stale_cleanup_release", task);
        }
        task.maintenance_sequence = 0;
        continue;
      }
      task.id = deleted_snapshot.id;
      task.generation = deleted_snapshot.generation;
      // A prepared Stage2 finalization can become stale only after a later erase/upsert
      // tombstones the same physical node. That later mutation owns an
      // ordinary cleanup intent for the preserved adjacency. This repair must
      // therefore undo only the backlinks attempted by the stale Stage2 finalization; a
      // preserved+supplemental union can contain 2R operations per item and
      // cannot fit one bounded R*batch peer message.
      lib_assert(!task.cleanup_repair_only || deleted_snapshot.deleted,
                 "stale Stage2 finalization repair requires a successor tombstone cleanup");
      lib_assert(task.cleanup_repair_only || task.cleanup_neighbors.empty(),
                 "ordinary tombstone cleanup unexpectedly carried repair neighbors");
      vec<RemotePtr> preserved_neighbors;
      if (deleted_snapshot.deleted && !task.cleanup_repair_only) {
        preserved_neighbors = node_.read_preserved_neighbor_list(task.target);
      }
      vec<RemotePtr> old_neighbors = select_cleanup_neighbors(
        task.cleanup_repair_only,
        span<const RemotePtr>{preserved_neighbors.data(),
                              preserved_neighbors.size()},
        task.cleanup_repair_only
          ? span<const RemotePtr>{task.cleanup_neighbors.data(),
                                  task.cleanup_neighbors.size()}
          : span<const RemotePtr>{task.cleanup_protected_children.data(),
                                  task.cleanup_protected_children.size()});
      lib_assert(old_neighbors.size() <= VamanaNode::graph_entry_capacity(),
                 "stage2 cleanup exceeded the per-item wire bound");
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      cleanup_neighbors_by_task.push_back(std::move(old_neighbors));
      ++ready;
    }
    context.tasks.resize(ready);

    // Snapshot every reverse target in one batch before constructing the
    // cleanup wire payload. Both ends carry ID/generation so a delayed retry
    // cannot mutate an unrelated node after either physical slot is reused.
    vec<RemotePtr> cleanup_targets;
    for (const vec<RemotePtr>& neighbors : cleanup_neighbors_by_task) {
      for (RemotePtr neighbor : neighbors) {
        if (!neighbor.is_null() &&
            neighbor.memory_node() < node_.num_storage_nodes_) {
          cleanup_targets.push_back(neighbor);
        }
      }
    }
    std::sort(cleanup_targets.begin(), cleanup_targets.end(),
              [](RemotePtr lhs, RemotePtr rhs) {
                return lhs.raw_address < rhs.raw_address;
              });
    cleanup_targets.erase(
      std::unique(cleanup_targets.begin(), cleanup_targets.end()),
      cleanup_targets.end());
    vec<StableNodeSnapshotState> cleanup_target_states;
    const vec<NodeSnapshot> cleanup_target_snapshots =
      node_.read_node_snapshots_batched(
        cleanup_targets, config, "cleanup_target_snapshot",
        &cleanup_target_states);
    if (std::find(cleanup_target_states.begin(),
                  cleanup_target_states.end(),
                  StableNodeSnapshotState::retryable) !=
        cleanup_target_states.end()) {
      return false;
    }
    dense_hashmap_t<u64, const NodeSnapshot*> cleanup_target_by_raw;
    cleanup_target_by_raw.reserve(cleanup_target_snapshots.size());
    for (const NodeSnapshot& snapshot : cleanup_target_snapshots) {
      cleanup_target_by_raw.emplace(snapshot.rptr.raw_address, &snapshot);
    }

    vec<ReverseUpdateOp> local_cleanup_ops;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      for (RemotePtr neighbor : cleanup_neighbors_by_task[item]) {
        const auto found = cleanup_target_by_raw.find(neighbor.raw_address);
        if (found == cleanup_target_by_raw.end()) continue;
        const NodeSnapshot& target = *found->second;
        const ReverseUpdateOp op{
          .target_raw = neighbor.raw_address,
          .candidate_raw = task.target.raw_address,
          .target_id = target.id,
          .target_generation = target.generation,
          .candidate_id = task.id,
          .candidate_generation = task.generation,
        };
        if (node_.local_shard(neighbor.memory_node())) {
          local_cleanup_ops.push_back(op);
        } else {
          context.remote_ops_by_peer[neighbor.memory_node()].push_back(op);
        }
      }
    }

    if (!node_.remove_local_neighbors_identity_fenced(
          span<const ReverseUpdateOp>{local_cleanup_ops}, config)) {
      node_.storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }

    u64 expected_mask = 0;
    for (u32 shard = 0; shard < node_.num_storage_nodes_; ++shard) {
      if (shard != node_.storage_id_ &&
          !context.remote_ops_by_peer[shard].empty()) {
        expected_mask |= u64{1} << shard;
      }
    }
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "cleanup stage2 failed to enter reverse_pending");
    register_reverse_requests(context, expected_mask);
    return true;
  };




}  // namespace memory_node_storage_owner_maintenance_detail

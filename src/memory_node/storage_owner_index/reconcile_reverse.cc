#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>

using namespace memory_node_storage_owner_index_detail;

namespace {

bool same_reconcile_neighbors(const vec<RemotePtr>& lhs,
                              const vec<RemotePtr>& rhs) {
  return lhs.size() == rhs.size() &&
    std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

bool MemoryNode::reconcile_local_reverse_ops(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const Configuration& config,
    vec<service::storage_owner::ReconcileReverseResult>& results) {
  ReverseSnapshotReuseScope snapshot_reuse(ops.size());
  bool lease_merge = false;
  vec<RemotePtr> identity_candidates;
  identity_candidates.reserve(ops.size());
  hashset_t<RemotePtr> seen_identities;
  seen_identities.reserve(ops.size());
  for (const auto& op : ops) {
    const RemotePtr target{op.target_raw};
    if (target.is_null() || !target.is_well_formed() ||
        target.memory_node() != storage_id_) {
      results.assign(ops.size(), {});
      for (size_t index = 0; index < ops.size(); ++index) {
        results[index].placement_sequence = ops[index].placement_sequence;
        results[index].stale = 1;
      }
      return false;
    }
    const auto kind = static_cast<service::storage_owner::ReconcileReverseOpKind>(
      op.kind);
    const RemotePtr candidate{op.new_candidate_raw};
    if (reconcile_kind_needs_new_identity(kind) && !candidate.is_null() &&
        candidate != target && seen_identities.insert(candidate).second) {
      identity_candidates.push_back(candidate);
    }
    lease_merge = lease_merge ||
      (config.storage_owner_lease_merge &&
       kind == service::storage_owner::ReconcileReverseOpKind::refine_add) ||
      kind == service::storage_owner::ReconcileReverseOpKind::
        acquire_certificate_lease ||
      kind == service::storage_owner::ReconcileReverseOpKind::
        release_certificate_lease ||
      kind == service::storage_owner::ReconcileReverseOpKind::
        remove_provisional_if_present;
  }
  if (!identity_candidates.empty()) {
    vec<NodeSnapshot> snapshots;
    vec<StableNodeSnapshotState> states;
    read_node_snapshots_batched_into(
      span<const RemotePtr>{identity_candidates}, config, snapshots,
      "reconcile_identity_prefetch", &states);
    for (const NodeSnapshot& snapshot : snapshots) {
      remember_reverse_snapshot(snapshot);
    }
  }
  if (lease_merge) return reconcile_lease_merge_ops(ops, config, results);
  if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
    std::cerr << std::string("[reconcile-stage] self=") +
      std::to_string(storage_id_) + " mode=one_sided ops=" +
      std::to_string(ops.size()) << std::endl;
  }
  return reconcile_reverse_ops_one_sided(ops, config, results);
}

bool MemoryNode::reconcile_reverse_ops_one_sided(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const Configuration& config,
    vec<service::storage_owner::ReconcileReverseResult>& results) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  results.assign(ops.size(), {});
  bool structurally_valid = true;
  dense_hashmap_t<u64, vec<size_t>> grouped;
  grouped.reserve(ops.size());
  vec<u64> target_order;
  target_order.reserve(ops.size());
  for (size_t op_index = 0; op_index < ops.size(); ++op_index) {
    const auto& op = ops[op_index];
    ReconcileReverseResult& result = results[op_index];
    result.placement_sequence = op.placement_sequence;

    const RemotePtr target{op.target_raw};
    if (target.is_null() || !target.is_well_formed() ||
        target.memory_node() >= num_storage_nodes_ ||
        !VamanaNode::hot_graph_entry_available(target)) {
      result.stale = 1;
      structurally_valid = false;
      continue;
    }
    if (local_shard(target.memory_node()) &&
        !valid_local_storage_node_pointer(target)) {
      // The wire pointer is structurally valid, but its tagged physical
      // incarnation is already gone.  Complete only optional/absence
      // postconditions here; mandatory reachability operations remain stale
      // and make Stage2 reselect a live parent.
      result = reconcile_retired_target_result(op);
      continue;
    }

    auto position = grouped.find(target.raw_address);
    if (position == grouped.end()) {
      target_order.push_back(target.raw_address);
      position = grouped.emplace(target.raw_address, vec<size_t>{}).first;
    }
    position->second.push_back(op_index);
  }

  for (const u64 target_raw : target_order) {
    const RemotePtr target{target_raw};
    const vec<size_t>& op_indices = grouped.find(target_raw)->second;
    bool has_promotion = false;
    for (const size_t op_index : op_indices) {
      has_promotion = has_promotion ||
        static_cast<ReconcileReverseOpKind>(ops[op_index].kind) ==
          ReconcileReverseOpKind::promote_stable_bridge;
    }
    vec<size_t> reordered_indices;
    span<const size_t> execution_indices{op_indices};
    if (has_promotion &&
        !reconcile_reverse_target_execution_order(
          ops, span<const size_t>{op_indices}, reordered_indices)) {
      structurally_valid = false;
      continue;
    }
    if (has_promotion) execution_indices = reordered_indices;

    vec<RemotePtr> mandatory_promotions;
    mandatory_promotions.reserve(op_indices.size());
    for (const size_t op_index : op_indices) {
      const auto kind = static_cast<ReconcileReverseOpKind>(
        ops[op_index].kind);
      if (kind != ReconcileReverseOpKind::promote_stable_bridge) continue;
      const RemotePtr candidate{ops[op_index].new_candidate_raw};
      if (candidate.is_null() || candidate == target ||
          std::find(mandatory_promotions.begin(),
                    mandatory_promotions.end(), candidate) !=
            mandatory_promotions.end()) {
        continue;
      }
      mandatory_promotions.push_back(candidate);
    }
    if (mandatory_promotions.size() > config.R) {
      // The complete mandatory set cannot fit in this target's bounded graph.
      // Valid Stage2 contexts are capped below R; treat an oversized wire
      // transaction as malformed without publishing a partial certificate.
      structurally_valid = false;
      continue;
    }
    const auto pointer_sane = [&](const RemotePtr candidate) {
      if (candidate.is_null() ||
          candidate.memory_node() >= num_storage_nodes_ ||
          !VamanaNode::hot_graph_entry_available(candidate)) {
        return false;
      }
      if (local_shard(candidate.memory_node()) &&
          !valid_local_storage_node_pointer(candidate)) {
        return false;
      }
      const auto vector = vamana::StorageLayoutResolver::vector(candidate);
      return vector.offset <= mn_memory_bytes_ &&
        vector.size <= mn_memory_bytes_ - vector.offset;
    };

    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      std::cerr << std::string("[reconcile-stage] self=") +
        std::to_string(storage_id_) + " target=" +
        std::to_string(target_raw) + " point=before_lock" << std::endl;
    }
    const IncarnationLockResult target_lock = try_lock_node(target);
    if (target_lock == IncarnationLockResult::stale) {
      // The operation names a retired physical identity. Never acquire or
      // mutate its replacement; optional/absence operations can terminate,
      // while promotion/ensure force a fresh live-parent plan.
      for (const size_t op_index : op_indices) {
        results[op_index] = reconcile_retired_target_result(ops[op_index]);
      }
      continue;
    }
    if (target_lock == IncarnationLockResult::busy) {
      // Preserve transient contention as an RPC retry instead of pretending
      // that the reconciliation postcondition was reached.
      structurally_valid = false;
      continue;
    }
    u64 target_header = 0;
    node_t target_id = 0;
    u32 target_generation = 0;
    if (!read_locked_node_identity(
          target, target_header, target_id, target_generation)) {
      structurally_valid = false;
      unlock_node(target);
      continue;
    }
    (void)target_id;
    (void)target_generation;
    if (memory_node_storage_owner_maintenance_detail::
          stage2_retired_target_cleanup_allowed(target_header)) {
      for (const size_t op_index : op_indices) {
        results[op_index] = reconcile_retired_target_result(ops[op_index]);
      }
      unlock_node(target);
      continue;
    }
    const bool target_stable =
      VamanaNode::stable_graph_mutation_allowed(target_header);
    const bool target_route_accounted =
      (target_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
    GraphAdjacency adjacency;
    if (!read_graph_adjacency(target, adjacency)) {
      // We own the target identity lock, so a bounded checksum miss is not a
      // stable stale-target observation. Return a retryable RPC failure and
      // leave the authoritative adjacency byte-for-byte unchanged.
      structurally_valid = false;
      unlock_node(target);
      continue;
    }
    const vec<RemotePtr> before_stable = adjacency.stable;
    const vec<RemotePtr> before_provisional = adjacency.provisional;

    bool reclaimed_provisional = false;
    const auto reclaim_stale_provisional = [&]() {
      if (reclaimed_provisional) return;
      reclaimed_provisional = true;
      // Protected slots are bounded long-lived structural state. Reclaim
      // deleted/reused tagged children lazily at the next reservation so a
      // stale protected pointer cannot permanently consume capacity under a
      // high-frequency update workload. Live provisional Stage1 children are
      // intentionally retained.
      adjacency.provisional.erase(
        std::remove_if(
          adjacency.provisional.begin(), adjacency.provisional.end(),
          [this](RemotePtr child) {
            NodeSnapshot child_snapshot;
            if (read_node_snapshot(child, child_snapshot)) {
              return child_snapshot.deleted;
            }
            // A locked same-incarnation child is uncertain and must be
            // retained. A tag mismatch, in contrast, proves that this edge
            // names a reclaimed slot and is safe to drop.
            u64 header = 0;
            if (local_shard(child.memory_node())) {
              if (!valid_local_storage_node_pointer(child)) return true;
              header = load_local_node_header_acquire(child);
            } else if (child.memory_node() < num_storage_nodes_) {
              const auto address =
                vamana::StorageLayoutResolver::header(child);
              if (address.offset > mn_memory_bytes_ ||
                  sizeof(header) > mn_memory_bytes_ - address.offset) {
                return true;
              }
              remote_read_bytes(child.memory_node(), address.offset,
                                &header, sizeof(header), 0);
            } else {
              return true;
            }
            return VamanaNode::header_incarnation(header) !=
              child.incarnation();
          }),
        adjacency.provisional.end());
    };

    const auto target_vector =
      vamana::StorageLayoutResolver::vector(target);
    lib_assert(target_vector.offset <= mn_memory_bytes_ &&
                 target_vector.size <= mn_memory_bytes_ -
                   target_vector.offset &&
                 target_vector.size == VamanaNode::vector_bytes(),
               "reconcile reverse target vector exceeds shard bounds");
    thread_local vec<byte_t> remote_target_vector;
    const byte_t* target_vector_data = nullptr;
    if (local_shard(target.memory_node())) {
      target_vector_data =
        index_buffer_.get_full_buffer() + target_vector.offset;
    } else {
      remote_target_vector.resize(target_vector.size);
      remote_read_bytes(target.memory_node(), target_vector.offset,
                        remote_target_vector.data(),
                        remote_target_vector.size(), 0);
      target_vector_data = remote_target_vector.data();
    }

    Configuration prune_config = config;
    if (std::any_of(op_indices.begin(), op_indices.end(), [&](size_t index) {
          return static_cast<ReconcileReverseOpKind>(ops[index].kind) ==
            ReconcileReverseOpKind::refine_add;
        })) {
      prune_config.alpha = -std::abs(prune_config.alpha);
    }
    bool robust_prune_retryable = false;
    const auto robust_prune = [&](const vec<RemotePtr>& candidates,
                                  size_t existing_count = 0) {
      vec<StableNodeSnapshotState> snapshot_states;
      vec<NodeSnapshot> snapshots =
        read_node_snapshots_batched(
          candidates, config, "reconcile_reverse_ops_one_sided",
          &snapshot_states);
      if (std::find(snapshot_states.begin(), snapshot_states.end(),
                    StableNodeSnapshotState::retryable) !=
          snapshot_states.end()) {
        // The caller still owns the target lock. Preserve the complete
        // pre-reconciliation adjacency and make the RPC retry instead of
        // pruning from a set that merely omitted a contended live neighbor.
        robust_prune_retryable = true;
        return candidates;
      }
      hashset_t<RemotePtr> skip;
      skip.insert(target);
      vec<RemotePtr> selected = robust_prune_snapshots_cpu(
        target_vector_data,
        VamanaNode::vector_dtype(),
        span<const NodeSnapshot>{snapshots.data(), snapshots.size()},
        skip, prune_config, config.R, false, existing_count);
      vec<RemotePtr> eligible_candidates;
      eligible_candidates.reserve(snapshots.size());
      for (const NodeSnapshot& snapshot : snapshots) {
        eligible_candidates.push_back(snapshot.rptr);
      }
      u64 forced = 0;
      if (!preserve_reconcile_mandatory_candidates(
            span<const RemotePtr>{eligible_candidates},
            span<const RemotePtr>{mandatory_promotions}, config.R,
            selected, &forced)) {
        robust_prune_retryable = true;
        return candidates;
      }
      storage_owner_stage2_mandatory_promotions_preserved_.fetch_add(
        forced, std::memory_order_relaxed);
      return selected;
    };

    bool publish_allowed = false;
    size_t group_position = 0;
    while (group_position < execution_indices.size()) {
      const size_t op_index = execution_indices[group_position];
      const auto& op = ops[op_index];
      const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);

      // Reachability certificates may only land on a node that still belongs
      // to the versioned centroid routing backbone while its incarnation lock
      // is held.  This closes the selection-to-RPC retirement window without
      // preventing ordinary removal cleanup on a node leaving the route.
      const bool mandatory_reachability =
        kind == ReconcileReverseOpKind::promote_stable_bridge ||
        kind == ReconcileReverseOpKind::ensure_reachable;
      const bool reachability_target_stable = target_stable &&
        (!mandatory_reachability || target_route_accounted);

      if (reconcile_kind_is_ordinary_add(kind)) {
        // Preserve the receiver-selected per-target order. Stronger
        // reconciliation operations delimit compatible ordinary-add runs;
        // this keeps mandatory/removal operation boundaries intact while
        // collapsing the common hot-target fan-in case to one RobustPrune.
        if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        std::cerr << std::string("[reconcile-stage] self=") +
          std::to_string(storage_id_) + " target=" +
          std::to_string(target_raw) + " point=before_snapshot" << std::endl;
      }
      const size_t run_end = reconcile_reverse_add_run_end(
          ops, execution_indices, group_position);
        lib_assert(run_end > group_position,
                   "ordinary reverse add produced an empty compatible run");
        if (run_end == group_position + 1) {
          // Avoid temporary vectors on the overwhelmingly common no-contention
          // path. The scalar policy below is identical to the pre-batching
          // implementation; union pruning is reserved for actual fan-in.
        } else {

          vec<service::storage_owner::ReconcileReverseOp> add_ops;
          vec<u8> new_identity_stable;
          add_ops.reserve(run_end - group_position);
          new_identity_stable.reserve(run_end - group_position);
          for (size_t run_position = group_position;
               run_position < run_end; ++run_position) {
            const auto& add_op = ops[execution_indices[run_position]];
            add_ops.push_back(add_op);
            const RemotePtr candidate{add_op.new_candidate_raw};
            NodeSnapshot snapshot;
            bool live = false;
            if (pointer_sane(candidate) &&
                read_node_snapshot(candidate, snapshot)) {
              live = !snapshot.deleted &&
                (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
                snapshot.id == add_op.id &&
                snapshot.generation == add_op.generation;
            }
            new_identity_stable.push_back(live ? 1 : 0);
          }

          vec<ReconcileReverseResult> add_results;
          reconcile_reverse_add_batch(
            span<const service::storage_owner::ReconcileReverseOp>{add_ops},
            span<const u8>{new_identity_stable},
            target_stable, config.R,
            VamanaNode::provisional_slots(), adjacency.stable,
            adjacency.provisional, add_results, robust_prune);
          for (size_t run_position = group_position;
               run_position < run_end; ++run_position) {
            ReconcileReverseResult& result =
              results[execution_indices[run_position]];
            result = add_results[run_position - group_position];
            publish_allowed = publish_allowed || result.stale == 0;
          }
          group_position = run_end;
          continue;
        }
      }

      if (kind == ReconcileReverseOpKind::ensure_reachable) {
        reclaim_stale_provisional();
      }
      const RemotePtr old_candidate{op.old_candidate_raw};
      const RemotePtr new_candidate{op.new_candidate_raw};
      const bool old_present = !old_candidate.is_null() &&
        (reconcile_contains(adjacency.stable, old_candidate) ||
         reconcile_contains(adjacency.provisional, old_candidate));
      const bool needs_new = reconcile_kind_needs_new_identity(kind);
      if ((kind == ReconcileReverseOpKind::remove_if_present ||
           kind == ReconcileReverseOpKind::remove_provisional_if_present) &&
          old_present) {
        const bool pointer_valid = pointer_sane(old_candidate);
        NodeSnapshot stale_snapshot;
        const bool snapshot_read = pointer_valid &&
          read_node_snapshot(old_candidate, stale_snapshot);
        const bool identity_matches = snapshot_read &&
          stale_snapshot.id == op.id &&
          stale_snapshot.generation == op.generation;
        bool terminal = reconcile_stale_removal_completed(
          old_present, pointer_valid, snapshot_read, identity_matches);
        if (!terminal && !snapshot_read) {
          vec<std::pair<RemotePtr, u64>> identity;
          vec<StableNodeSnapshotState> states;
          read_node_identity_headers_batched_into(
            span<const RemotePtr>{&old_candidate, 1}, config, identity, &states);
          lib_assert(states.size() == 1,
                     "stale removal identity observation lost its input");
          terminal = states.front() == StableNodeSnapshotState::terminal;
        }
        if (terminal) {
          results[op_index].placement_sequence = op.placement_sequence;
          results[op_index].removed = 1;
          ++group_position;
          continue;
        }
      }

      NodeSnapshot old_snapshot;
      NodeSnapshot new_snapshot;
      bool old_identity_matches = !old_present;
      if (old_present && pointer_sane(old_candidate) &&
          read_node_snapshot(old_candidate, old_snapshot)) {
        old_identity_matches =
          old_snapshot.id == op.id &&
          old_snapshot.generation == op.generation;
      }

      bool new_identity_live = !needs_new;
      if (needs_new && pointer_sane(new_candidate) &&
          read_node_snapshot(new_candidate, new_snapshot)) {
        new_identity_live =
          !new_snapshot.deleted &&
          (kind == ReconcileReverseOpKind::ensure_reachable ||
           (new_snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0) &&
          new_snapshot.id == op.id &&
          new_snapshot.generation == op.generation;
      }

      const bool replacement_equivalent =
        old_present && old_identity_matches && new_identity_live &&
        old_snapshot.vector_data.size() == new_snapshot.vector_data.size() &&
        old_snapshot.vector_data.size() >= VamanaNode::vector_bytes() &&
        std::memcmp(old_snapshot.vector_data.data(),
                    new_snapshot.vector_data.data(),
                    VamanaNode::vector_bytes()) == 0;

      ReconcileReverseResult& result = results[op_index];
      result = reconcile_reverse_adjacency(
        op, reachability_target_stable,
        old_identity_matches, new_identity_live,
        replacement_equivalent, config.R, VamanaNode::provisional_slots(),
        adjacency.stable, adjacency.provisional, robust_prune);
      publish_allowed = publish_allowed || result.stale == 0;
      ++group_position;
    }

    if (robust_prune_retryable) {
      adjacency.stable = before_stable;
      adjacency.provisional = before_provisional;
      structurally_valid = false;
      unlock_node(target);
      continue;
    }

    // Do not publish an intermediate accepted bit whose certificate was
    // displaced by a later operation in this target transaction. The sender
    // still owns its Stage1 bridge on this failure and can safely replan.
    if (!reconcile_reverse_final_reachability_holds(
          ops, execution_indices,
          span<const ReconcileReverseResult>{results},
          span<const RemotePtr>{adjacency.stable},
          span<const RemotePtr>{adjacency.provisional})) {
      adjacency.stable = before_stable;
      adjacency.provisional = before_provisional;
      structurally_valid = false;
      unlock_node(target);
      continue;
    }

    const bool changed =
      !same_reconcile_neighbors(before_stable, adjacency.stable) ||
      !same_reconcile_neighbors(before_provisional,
                                adjacency.provisional);
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      std::cerr << std::string("[reconcile-stage] self=") +
        std::to_string(storage_id_) + " target=" +
        std::to_string(target_raw) + " point=before_unlock" << std::endl;
    }
    if (publish_allowed && changed) {
      write_graph_adjacency(target, adjacency.stable,
                            adjacency.provisional,
                            adjacency.generation, false);
    }
    unlock_node(target);
  }
  return structurally_valid;
}

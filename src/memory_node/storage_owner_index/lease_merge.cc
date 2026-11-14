#include "memory_node/storage_owner_index/detail.hh"

#include "memory_node/storage_owner_index/atomic_cut_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <cstdlib>

using namespace memory_node_storage_owner_index_detail;

bool MemoryNode::reconcile_lease_merge_ops(
    span<const service::storage_owner::ReconcileReverseOp> ops,
    const Configuration& config,
    vec<service::storage_owner::ReconcileReverseResult>& results) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  const auto debug_failure = [&](const char* reason,
                                 const service::storage_owner::ReconcileReverseOp& op) {
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") == nullptr) return;
    static std::atomic<u32> debug_count{0};
    if (debug_count.fetch_add(1, std::memory_order_relaxed) >= 128) return;
    std::string message = "[lease-input-debug] reason=" +
      std::string(reason) + " kind=" + std::to_string(op.kind) +
      " target=" + std::to_string(op.target_raw) +
      " old=" + std::to_string(op.old_candidate_raw) +
      " new=" + std::to_string(op.new_candidate_raw) +
      " id=" + std::to_string(op.id) +
      " generation=" + std::to_string(op.generation) +
      " sequence=" + std::to_string(op.placement_sequence);
    const RemotePtr target{op.target_raw};
    if (storage_node_pointer_addressable(target)) {
      message += " target_valid=" + std::to_string(
        valid_local_storage_node_pointer(target)) +
        " target_header=" + std::to_string(
          load_local_node_header_acquire(target));
    }
    const RemotePtr old{op.old_candidate_raw};
    if (!old.is_null() && storage_node_pointer_addressable(old)) {
      message += " old_valid=" + std::to_string(
        valid_local_storage_node_pointer(old)) +
        " old_header=" + std::to_string(load_local_node_header_acquire(old));
    }
    std::cerr << message << std::endl;
  };
  results.assign(ops.size(), {});
  vec<u8> terminal_removals(ops.size(), 0);
  vec<u8> rejected_additions(ops.size(), 0);
  for (size_t index = 0; index < ops.size(); ++index) {
    results[index].placement_sequence = ops[index].placement_sequence;
    results[index].stale = 1;
  }
  if (ops.empty()) return true;
  if (storage_owner_certificate_leases_ == nullptr) return false;
  if (storage_owner_maintenance_completion_ring_ != nullptr) {
    storage_owner_certificate_leases_->discard_released_through(
      storage_owner_maintenance_completion_ring_->finalized());
  }

  dense_hashmap_t<u64, vec<size_t>> grouped;
  grouped.reserve(ops.size());
  vec<u64> target_order;
  target_order.reserve(ops.size());
  for (size_t index = 0; index < ops.size(); ++index) {
    const auto& op = ops[index];
    const RemotePtr target{op.target_raw};
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    const bool supported =
      reconcile_kind_is_ordinary_add(kind) ||
      kind == ReconcileReverseOpKind::add_if_capacity ||
      kind == ReconcileReverseOpKind::remove_if_present ||
      kind == ReconcileReverseOpKind::acquire_certificate_lease ||
      kind == ReconcileReverseOpKind::release_certificate_lease ||
      kind == ReconcileReverseOpKind::remove_provisional_if_present;
    const bool target_addressable = local_shard(target.memory_node()) &&
      storage_node_pointer_addressable(target);
    const auto admission = reconcile_target_admission(
      target_addressable,
      target_addressable && valid_local_storage_node_pointer(target));
    if (!supported || op.reserved != 0 || op.placement_sequence == 0 ||
        admission == ReconcileTargetAdmission::malformed) {
      debug_failure("invalid_input", op);
      return false;
    }
    if (kind == ReconcileReverseOpKind::remove_if_present ||
        kind == ReconcileReverseOpKind::remove_provisional_if_present) {
      if (op.old_candidate_raw == 0) return false;
    } else if (op.new_candidate_raw == 0) {
      return false;
    }
    if (admission == ReconcileTargetAdmission::retired) {
      (void)storage_owner_certificate_leases_->release_target(
        target.raw_address);
      results[index] = reconcile_retired_target_result(op);
      continue;
    }
    auto position = grouped.find(target.raw_address);
    if (position == grouped.end()) {
      target_order.push_back(target.raw_address);
      position = grouped.emplace(target.raw_address, vec<size_t>{}).first;
    }
    position->second.push_back(index);
  }

  bool valid = true;
  for (const u64 target_raw : target_order) {
    const RemotePtr target{target_raw};
    const vec<size_t>& indices = grouped.find(target_raw)->second;
    const bool release_only = std::all_of(
      indices.begin(), indices.end(), [&](size_t index) {
        return static_cast<ReconcileReverseOpKind>(ops[index].kind) ==
          ReconcileReverseOpKind::release_certificate_lease;
      });
    if (release_only &&
        memory_node_storage_owner_maintenance_detail::
          certificate_release_can_bypass_target_lock(
            load_local_node_header_acquire(target))) {
      vec<std::pair<u64, u64>> released;
      released.reserve(indices.size());
      for (const size_t index : indices) {
        released.emplace_back(ops[index].new_candidate_raw,
                              ops[index].placement_sequence);
      }
      if (!storage_owner_certificate_leases_->commit(
            target.raw_address, {}, released)) {
        valid = false;
        continue;
      }
      for (const size_t index : indices) {
        results[index].placement_sequence = ops[index].placement_sequence;
        results[index].accepted = 1;
        results[index].stale = 0;
      }
      continue;
    }
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      std::cerr << std::string("[reconcile-stage] self=") +
        std::to_string(storage_id_) + " target=" +
        std::to_string(target_raw) + " point=before_lease_lock" << std::endl;
    }
    const IncarnationLockResult lock = try_lock_node(target);
    if (lock == IncarnationLockResult::stale) {
      (void)storage_owner_certificate_leases_->release_target(
        target.raw_address);
      for (const size_t index : indices) {
        results[index] = reconcile_retired_target_result(ops[index]);
      }
      continue;
    }
    if (lock == IncarnationLockResult::busy) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::cerr << "[stage2-lease-failure] reason=lock_busy"
                    << " target=" << target.raw_address << std::endl;
        }
      }
      valid = false;
      continue;
    }
    if (release_only) {
      vec<std::pair<u64, u64>> released;
      released.reserve(indices.size());
      for (const size_t index : indices) {
        released.emplace_back(ops[index].new_candidate_raw,
                              ops[index].placement_sequence);
      }
      if (!storage_owner_certificate_leases_->commit(
            target.raw_address, {}, released)) {
        unlock_node(target);
        valid = false;
        continue;
      }
      for (const size_t index : indices) {
        results[index].placement_sequence = ops[index].placement_sequence;
        results[index].accepted = 1;
        results[index].stale = 0;
      }
      unlock_node(target);
      continue;
    }

    u64 target_header = 0;
    node_t target_id = 0;
    u32 target_generation = 0;
    if (!read_locked_node_identity(
          target, target_header, target_id, target_generation)) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::cerr << "[stage2-lease-failure] reason=identity_read"
                    << " target=" << target.raw_address << std::endl;
        }
      }
      unlock_node(target);
      valid = false;
      continue;
    }
    if (memory_node_storage_owner_maintenance_detail::
          stage2_retired_target_cleanup_allowed(target_header)) {
      (void)storage_owner_certificate_leases_->release_target(
        target.raw_address);
      for (const size_t index : indices) {
        results[index] = reconcile_retired_target_result(ops[index]);
      }
      unlock_node(target);
      continue;
    }
    GraphAdjacency adjacency;
    if (!read_graph_adjacency(target, adjacency)) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        std::cerr << "[stage2-cleanup-debug] target=" << target.raw_address
                  << " header=" << target_header << " id=" << target_id
                  << " generation=" << target_generation
                  << " reason=adjacency_read" << std::endl;
      }
      unlock_node(target);
      valid = false;
      continue;
    }
    const bool target_stable =
      VamanaNode::stable_graph_mutation_allowed(target_header);
    const bool target_accounted =
      (target_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
    const vec<RemotePtr> before_stable = adjacency.stable;
    const vec<RemotePtr> before_provisional = adjacency.provisional;

    const auto lease_rows =
      storage_owner_certificate_leases_->leases_at(target.raw_address);
    vec<std::pair<RemotePtr, u64>> active;
    active.reserve(lease_rows.size());
    for (const auto& [child_raw, sequence] : lease_rows) {
      active.emplace_back(RemotePtr{child_raw}, sequence);
    }

    bool identities_valid = true;
    const auto pointer_sane = [&](RemotePtr candidate) {
      if (candidate.is_null() || candidate == target ||
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
    vec<std::pair<RemotePtr, u64>> expired_leases;
    for (const auto& [child, sequence] : active) {
      const bool released = std::any_of(
        indices.begin(), indices.end(), [&](size_t index) {
          const auto& op = ops[index];
          return op.kind == static_cast<u32>(
                   ReconcileReverseOpKind::release_certificate_lease) &&
            RemotePtr{op.new_candidate_raw} == child &&
            op.placement_sequence == sequence;
        });
      if (released) continue;
      const bool pointer_valid = pointer_sane(child);
      NodeSnapshot snapshot;
      const bool snapshot_read = pointer_valid &&
        read_node_snapshot(child, snapshot);
      auto identity_state = StableNodeSnapshotState::retryable;
      u64 identity_header = snapshot_read ? snapshot.header : 0;
      if (!snapshot_read) {
        vec<std::pair<RemotePtr, u64>> identity;
        vec<StableNodeSnapshotState> states;
        read_node_identity_headers_batched_into(
          span<const RemotePtr>{&child, 1}, config, identity, &states);
        lib_assert(states.size() == 1,
                   "certificate child identity observation lost its input");
        identity_state = states.front();
        if (!identity.empty()) identity_header = identity.front().second;
      }
      const CertificateLeaseChildState child_state = certificate_lease_child_state(
        pointer_valid, snapshot_read, snapshot_read && snapshot.deleted,
        identity_state, identity_header);
      if (child_state == CertificateLeaseChildState::live) continue;
      if (child_state == CertificateLeaseChildState::terminal) {
        expired_leases.emplace_back(child, sequence);
        if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
          std::cerr << "[stage2-lease-reclaim] target=" << target.raw_address
                    << " child=" << child.raw_address
                    << " sequence=" << sequence
                    << " header=" << identity_header << std::endl;
        }
        continue;
      }
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        std::cerr << "[stage2-lease-child-debug] target=" << target.raw_address
                  << " child=" << child.raw_address
                  << " sequence=" << sequence
                  << " pointer_valid=" << pointer_valid
                  << " snapshot_read=" << snapshot_read
                  << " identity_state=" << static_cast<u32>(identity_state)
                  << " header=" << identity_header << std::endl;
      }
      identities_valid = false;
      break;
    }
    active.erase(
      std::remove_if(
        active.begin(), active.end(), [&](const auto& lease) {
          return std::find(expired_leases.begin(), expired_leases.end(), lease) !=
            expired_leases.end();
        }),
      active.end());
    for (const size_t index : indices) {
      if (!identities_valid) break;
      const auto& op = ops[index];
      const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
      if (kind == ReconcileReverseOpKind::remove_if_present ||
          kind == ReconcileReverseOpKind::remove_provisional_if_present) {
        const RemotePtr old{op.old_candidate_raw};
        const bool old_present =
          std::find(adjacency.stable.begin(), adjacency.stable.end(), old) !=
            adjacency.stable.end() ||
          (kind == ReconcileReverseOpKind::remove_if_present &&
           std::find(adjacency.provisional.begin(), adjacency.provisional.end(),
                     old) != adjacency.provisional.end());
        if (old_present) {
          const bool pointer_valid = pointer_sane(old);
          NodeSnapshot snapshot;
          const bool snapshot_read = pointer_valid &&
            read_node_snapshot(old, snapshot);
          const bool identity_matches = snapshot_read &&
            snapshot.id == op.id && snapshot.generation == op.generation;
          bool terminal = reconcile_stale_removal_completed(
            old_present, pointer_valid, snapshot_read, identity_matches);
          if (!terminal && !snapshot_read) {
            vec<std::pair<RemotePtr, u64>> identity;
            vec<StableNodeSnapshotState> states;
            read_node_identity_headers_batched_into(
              span<const RemotePtr>{&old, 1}, config, identity, &states);
            lib_assert(states.size() == 1,
                       "stale removal identity observation lost its input");
            terminal = states.front() == StableNodeSnapshotState::terminal;
          }
          if (terminal) {
            terminal_removals[index] = 1;
          } else if (!snapshot_read) {
            if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
              std::cerr << "[stage2-cleanup-debug] target=" << target.raw_address
                        << " header=" << target_header << " id=" << target_id
                        << " generation=" << target_generation
                        << " old=" << old.raw_address
                        << " old_present=1 reason=old_identity" << std::endl;
            }
            identities_valid = false;
          }
        }
        continue;
      }
      if (!reconcile_kind_is_ordinary_add(kind) &&
          kind != ReconcileReverseOpKind::add_if_capacity &&
          kind != ReconcileReverseOpKind::acquire_certificate_lease) {
        continue;
      }
      const RemotePtr candidate{op.new_candidate_raw};
      const bool optional_addition = reconcile_kind_is_ordinary_add(kind) ||
        kind == ReconcileReverseOpKind::add_if_capacity;
      NodeSnapshot snapshot;
      const bool candidate_valid = pointer_sane(candidate) &&
        read_node_snapshot(candidate, snapshot) && !snapshot.deleted &&
        snapshot.id == op.id && snapshot.generation == op.generation &&
        (!optional_addition ||
         (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0) &&
        target_stable &&
        (kind != ReconcileReverseOpKind::acquire_certificate_lease ||
         target_accounted);
      if (!candidate_valid) {
        if (optional_addition) {
          rejected_additions[index] = 1;
          results[index].accepted = 0;
          results[index].stale = 0;
        } else {
          identities_valid = false;
        }
      }
    }
    if (!identities_valid) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 32) {
          std::cerr << "[stage2-lease-debug] target=" << target.raw_address
                    << " header=" << target_header
                    << " target_id=" << target_id
                    << " target_generation=" << target_generation
                    << " stable=" << target_stable
                    << " accounted=" << target_accounted;
          for (const auto& [child, sequence] : active) {
            std::cerr << " active(" << child.raw_address << "," << sequence
                      << ")";
          }
          for (const size_t op_index : indices) {
            const auto& op = ops[op_index];
            std::cerr << " op(kind=" << op.kind << ",id=" << op.id
                      << ",sequence=" << op.placement_sequence << ")";
          }
          std::cerr << std::endl;
        }
      }
      unlock_node(target);
      valid = false;
      continue;
    }

    Configuration prune_config = config;
    if (std::any_of(indices.begin(), indices.end(), [&](size_t index) {
          return static_cast<ReconcileReverseOpKind>(ops[index].kind) ==
            ReconcileReverseOpKind::refine_add;
        })) {
      prune_config.alpha = -std::abs(prune_config.alpha);
    }
    vec<RemotePtr> required_snapshots;
    for (const auto& [child, sequence] : active) {
      const bool released = std::any_of(
        indices.begin(), indices.end(), [&](size_t index) {
          const auto& op = ops[index];
          return op.kind == static_cast<u32>(
                   ReconcileReverseOpKind::release_certificate_lease) &&
            RemotePtr{op.new_candidate_raw} == child &&
            op.placement_sequence == sequence;
        });
      if (!released) required_snapshots.push_back(child);
    }
    for (const size_t index : indices) {
      if (static_cast<ReconcileReverseOpKind>(ops[index].kind) ==
          ReconcileReverseOpKind::acquire_certificate_lease) {
        required_snapshots.emplace_back(ops[index].new_candidate_raw);
      }
    }
    const auto target_vector = vamana::StorageLayoutResolver::vector(target);
    const byte_t* target_vector_data =
      index_buffer_.get_full_buffer() + target_vector.offset;
    bool prune_retryable = false;
    auto robust_prune = [&](const vec<RemotePtr>& candidates,
                            size_t existing_count = 0) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        std::cerr << std::string("[reconcile-stage] self=") +
          std::to_string(storage_id_) + " target=" +
          std::to_string(target_raw) + " point=before_lease_snapshot" << std::endl;
      }
      vec<StableNodeSnapshotState> states;
      const vec<NodeSnapshot> snapshots = read_node_snapshots_batched(
        candidates, config, "lease_merge_atomic_cut", &states);
      lib_assert(states.size() == candidates.size(),
                 "LeaseMerge prune lost candidate snapshot states");
      for (size_t index = 0; index < candidates.size(); ++index) {
        if (lease_merge_snapshot_requires_retry(
              candidates[index], states[index],
              span<const RemotePtr>{required_snapshots})) {
          prune_retryable = true;
          return candidates;
        }
      }
      hashset_t<RemotePtr> skip;
      skip.insert(target);
      vec<RemotePtr> selected = robust_prune_snapshots_cpu(
        target_vector_data, VamanaNode::vector_dtype(),
        span<const NodeSnapshot>{snapshots}, skip, prune_config, config.R,
        false, existing_count);
      static std::atomic<u32> debug_prunes{0};
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
          debug_prunes.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::cerr << "[stage2-prune-debug] target=" << target.raw_address
                  << " candidates=" << candidates.size()
                  << " selected=" << selected.size() << std::endl;
      }
      return selected;
    };
    vec<service::storage_owner::ReconcileReverseOp> target_ops;
    vec<size_t> target_op_indices;
    target_ops.reserve(indices.size() + expired_leases.size());
    target_op_indices.reserve(indices.size());
    for (const size_t index : indices) {
      if (terminal_removals[index] != 0) {
        results[index].placement_sequence = ops[index].placement_sequence;
        results[index].removed = 1;
        continue;
      }
      if (rejected_additions[index] != 0) continue;
      target_op_indices.push_back(index);
      target_ops.push_back(ops[index]);
    }
    for (const auto& [child, sequence] : expired_leases) {
      target_ops.push_back(service::storage_owner::ReconcileReverseOp{
        .target_raw = target.raw_address,
        .old_candidate_raw = 0,
        .new_candidate_raw = child.raw_address,
        .placement_sequence = sequence,
        .id = 0,
        .generation = 0,
        .kind = static_cast<u32>(
          ReconcileReverseOpKind::release_certificate_lease),
      });
    }
    auto plan = plan_atomic_cut(
      span<const service::storage_owner::ReconcileReverseOp>{target_ops},
      span<const RemotePtr>{before_stable},
      span<const RemotePtr>{before_provisional},
      span<const std::pair<RemotePtr, u64>>{active}, config.R, robust_prune);
    if (prune_retryable || !plan.valid) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 64) {
          std::cerr << "[stage2-lease-failure] reason=plan"
                    << " target=" << target.raw_address
                    << " prune_retryable=" << prune_retryable
                    << " before=" << before_stable.size()
                    << " active=" << active.size()
                    << " ops=" << target_ops.size() << std::endl;
        }
      }
      unlock_node(target);
      valid = false;
      continue;
    }
    static std::atomic<u32> debug_plans{0};
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr &&
        debug_plans.fetch_add(1, std::memory_order_relaxed) < 16) {
      size_t adds = 0;
      size_t accepted = 0;
      for (size_t slot = 0; slot < target_ops.size(); ++slot) {
        if (static_cast<ReconcileReverseOpKind>(target_ops[slot].kind) ==
            ReconcileReverseOpKind::add) {
          ++adds;
          accepted += plan.results[slot].accepted != 0;
        }
      }
      std::cerr << "[stage2-plan-debug] target=" << target.raw_address
                << " before=" << before_stable.size()
                << " after=" << plan.stable.size()
                << " adds=" << adds << " accepted=" << accepted
                << " active=" << active.size() << std::endl;
    }

    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      for (const auto& op : target_ops) {
        if (static_cast<ReconcileReverseOpKind>(op.kind) ==
              ReconcileReverseOpKind::remove_provisional_if_present &&
            op.id < 100000016) {
          std::cerr << "[stage2-remove-debug] target=" << target.raw_address
                    << " child=" << op.old_candidate_raw
                    << " before=" << before_provisional.size()
                    << " after=" << plan.provisional.size()
                    << " removed=" << static_cast<u32>(
                      plan.results[&op - target_ops.data()].removed)
                    << std::endl;
        }
      }
    }
    const bool changed = before_stable != plan.stable ||
      before_provisional != plan.provisional;
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      std::cerr << std::string("[reconcile-stage] self=") +
        std::to_string(storage_id_) + " target=" +
        std::to_string(target_raw) + " point=before_lease_unlock" << std::endl;
    }
    if (changed) {
      write_graph_adjacency(target, plan.stable, plan.provisional,
                            adjacency.generation, false);
    }
    vec<std::pair<u64, u64>> acquired;
    vec<std::pair<u64, u64>> released;
    acquired.reserve(plan.acquired.size());
    released.reserve(plan.released.size());
    for (const auto& [child, sequence] : plan.acquired) {
      acquired.emplace_back(child.raw_address, sequence);
    }
    for (const auto& [child, sequence] : plan.released) {
      released.emplace_back(child.raw_address, sequence);
    }
    if (!storage_owner_certificate_leases_->commit(
          target.raw_address, acquired, released)) {
      if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        static std::atomic<u32> debug_count{0};
        if (debug_count.fetch_add(1, std::memory_order_relaxed) < 16) {
          std::cerr << "[stage2-lease-commit-debug] target=" << target.raw_address;
          for (const auto& [child, sequence] : acquired) {
            std::cerr << " acquire(" << child << "," << sequence << ")";
          }
          for (const auto& [child, sequence] : released) {
            std::cerr << " release(" << child << "," << sequence << ")";
          }
          std::cerr << std::endl;
        }
      }
      if (changed) {
        write_graph_adjacency(target, before_stable, before_provisional,
                              adjacency.generation, false);
      }
      unlock_node(target);
      valid = false;
      continue;
    }
    storage_owner_lease_merge_atomic_cuts_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_lease_merge_target_ops_.fetch_add(
      indices.size(), std::memory_order_relaxed);
    for (size_t slot = 0; slot < target_op_indices.size(); ++slot) {
      results[target_op_indices[slot]] = plan.results[slot];
    }
    unlock_node(target);
  }
  return valid;
}

#include <array>
#include <cassert>
#include <cstdint>

#include "memory_node/storage_owner_index/atomic_cut_policy.hh"
#include "memory_node/storage_owner_maintenance/certificate_lease.hh"
#include "memory_node/storage_owner_maintenance/dependency_scheduler.hh"

namespace {

using memory_node_storage_owner_maintenance_detail::CertificateAcquireResult;
using memory_node_storage_owner_maintenance_detail::CertificateLeaseTable;
using memory_node_storage_owner_maintenance_detail::MaintenanceDependencyTracker;

void test_lease_lifecycle_capacity_and_replay() {
  CertificateLeaseTable leases(2);
  assert(leases.acquire(10, 100, 1) == CertificateAcquireResult::acquired);
  assert(leases.acquire(10, 100, 1) == CertificateAcquireResult::replay);
  assert(leases.contains(10, 100, 1));
  assert(leases.active_at(10) == 1);

  assert(leases.acquire(10, 101, 2) == CertificateAcquireResult::acquired);
  assert(leases.acquire(10, 102, 3) ==
         CertificateAcquireResult::capacity_exhausted);
  assert(leases.acquire(11, 102, 3) == CertificateAcquireResult::acquired);
  assert(leases.size() == 3);
  assert(leases.leases_at(10).size() == 2);
  assert(leases.commit(20, {{200, 4}}, {}));
  assert(leases.size() == 4);
  assert(leases.commit(20, {{200, 4}}, {}));
  assert(leases.size() == 4);
  assert(leases.commit(20, {}, {{200, 4}}));
  assert(leases.size() == 3);
  assert(leases.commit(20, {{201, 5}}, {{200, 9}}));
  assert(leases.contains(20, 201, 5));
  assert(leases.size() == 4);
  const auto snapshot = leases.snapshot();
  assert(snapshot.size() == 4);
  assert((snapshot[0] == std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>{10, 100, 1}));
  assert((snapshot[3] == std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>{20, 201, 5}));

  assert(leases.release(10, 100, 9));
  assert(leases.contains(10, 100, 1));
  assert(leases.release(10, 100, 1));
  assert(leases.release(10, 100, 1));
  assert(!leases.contains(10, 100));
  assert(leases.acquire(10, 100, 2) == CertificateAcquireResult::acquired);
  assert(leases.commit(10, {}, {{100, 1}}));
  assert(leases.contains(10, 100, 2));
  assert(leases.commit(10, {}, {{100, 2}}));
  assert(!leases.contains(10, 100));
  assert(leases.acquire(10, 102, 3) == CertificateAcquireResult::acquired);
}

void test_concurrent_cleanup_sequences_are_independent() {
  CertificateLeaseTable leases(2);
  constexpr std::uint64_t target = 10;
  constexpr std::uint64_t child = 100;
  assert(leases.acquire(target, child, 1) == CertificateAcquireResult::acquired);
  assert(leases.acquire(target, child, 2) == CertificateAcquireResult::acquired);
  assert(leases.acquire(target, child, 2) == CertificateAcquireResult::replay);
  assert(leases.size() == 1);
  assert(leases.active_at(target) == 1);
  assert(leases.contains(target, child, 1));
  assert(leases.contains(target, child, 2));
  assert(leases.commit(target, {}, {{child, 1}}));
  assert(!leases.contains(target, child, 1));
  assert(leases.contains(target, child, 2));
  assert(leases.commit(target, {}, {{child, 1}}));
  assert(leases.contains(target, child, 2));
  assert(leases.commit(target, {{child, 2}}, {}));
  assert(leases.size() == 1);
  assert(leases.release(target, child, 2));
  assert(!leases.contains(target, child));
  assert(leases.acquire(target, child, 2) == CertificateAcquireResult::replay);

  assert(leases.acquire(target, child, 3) == CertificateAcquireResult::acquired);
  assert(leases.acquire(target, child, 4) == CertificateAcquireResult::acquired);
  assert(leases.release(target, child, 3));
  assert(leases.contains(target, child, 4));
  assert(leases.release(target, child, 4));
  assert(!leases.contains(target, child));
}

void test_retired_target_cleanup_reclaims_lease() {
  using memory_node_storage_owner_index_detail::ReconcileTargetAdmission;
  using memory_node_storage_owner_index_detail::reconcile_retired_target_result;
  using memory_node_storage_owner_index_detail::reconcile_reverse_postcondition_holds;
  using memory_node_storage_owner_index_detail::reconcile_target_admission;
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  constexpr std::uint64_t target = 10;
  constexpr std::uint64_t child = 100;
  CertificateLeaseTable leases(2);
  assert(leases.acquire(target, child, 7) == CertificateAcquireResult::acquired);

  const auto admission = reconcile_target_admission(true, false);
  assert(admission == ReconcileTargetAdmission::retired);
  const ReconcileReverseOp removal{
    .target_raw = target,
    .old_candidate_raw = child,
    .placement_sequence = 7,
    .id = 42,
    .generation = 3,
    .kind = static_cast<std::uint32_t>(
      ReconcileReverseOpKind::remove_if_present),
  };
  const auto result = reconcile_retired_target_result(removal);
  assert(reconcile_reverse_postcondition_holds(removal, result));

  assert(leases.acquire(target + 1, child, 8) ==
         CertificateAcquireResult::acquired);
  assert(leases.release_target(target) == 1);
  assert(leases.release_target(target) == 0);
  assert(!leases.contains(target, child));
  assert(leases.contains(target + 1, child, 8));
  assert(leases.size() == 1);
}

void test_late_acquire_replay_does_not_reinsert_released_lease() {
  CertificateLeaseTable leases(2);
  constexpr std::uint64_t target = 20;
  constexpr std::uint64_t child = 200;
  constexpr std::uint64_t sequence = 9;
  assert(leases.commit(target, {}, {{child, sequence}}));
  assert(leases.size() == 0);
  assert(leases.commit(target, {{child, sequence}}, {}));
  assert(!leases.contains(target, child, sequence));
  assert(leases.size() == 0);
  assert(leases.commit(target, {{child, sequence}}, {}));
  assert(leases.size() == 0);
  assert(leases.release_target(target) == 0);
  assert(leases.acquire(target, child, sequence) ==
         CertificateAcquireResult::replay);
  assert(leases.size() == 0);

  leases.discard_released_through(sequence);
  assert(leases.commit(target, {{child, sequence}}, {}));
  assert(leases.size() == 0);
  assert(leases.acquire(target, child, sequence + 1) ==
         CertificateAcquireResult::acquired);
  assert(leases.size() == 1);
  assert(leases.release(target, child, sequence + 1));
  assert(leases.acquire(target, child, sequence + 1) ==
         CertificateAcquireResult::replay);
  assert(leases.size() == 0);
}

void test_cleanup_depends_only_on_matching_incarnation() {
  MaintenanceDependencyTracker dependencies;
  assert(dependencies.register_finalize(7, 42, 3, 1000));
  assert(!dependencies.register_finalize(8, 42, 3, 1000));
  assert(!dependencies.cleanup_ready(42, 3, 1000));
  assert(dependencies.cleanup_ready(42, 4, 1000));
  assert(dependencies.cleanup_ready(42, 3, 1001));
  assert(dependencies.cleanup_ready(43, 3, 1000));
  assert(dependencies.size() == 1);

  assert(dependencies.complete_finalize(7));
  assert(!dependencies.complete_finalize(7));
  assert(dependencies.cleanup_ready(42, 3, 1000));
  assert(dependencies.size() == 0);
}

void test_atomic_cut_pins_leases_and_commits_certificate_sets() {
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  const auto op = [](ReconcileReverseOpKind kind, std::uint64_t old_raw,
                     std::uint64_t new_raw, std::uint64_t sequence) {
    return ReconcileReverseOp{
      .target_raw = 10,
      .old_candidate_raw = old_raw,
      .new_candidate_raw = new_raw,
      .placement_sequence = sequence,
      .id = 42,
      .generation = 7,
      .kind = static_cast<std::uint32_t>(kind),
    };
  };
  const RemotePtr existing{1, 0x1000};
  const RemotePtr ordinary{1, 0x2000};
  const RemotePtr leased{1, 0x3000};
  const RemotePtr acquiring{1, 0x4000};
  const vec<ReconcileReverseOp> ops{
    op(ReconcileReverseOpKind::add, 0, ordinary.raw_address, 9),
    op(ReconcileReverseOpKind::acquire_certificate_lease, 0,
       acquiring.raw_address, 2),
    op(ReconcileReverseOpKind::remove_provisional_if_present,
       leased.raw_address, 0, 1),
  };
  const vec<RemotePtr> stable{existing};
  const vec<RemotePtr> provisional{leased, acquiring};
  const vec<std::pair<RemotePtr, std::uint64_t>> active{
    std::pair{leased, std::uint64_t{1}}};
  std::uint32_t prune_calls = 0;
  auto prune = [&](const vec<RemotePtr>&) {
    ++prune_calls;
    return vec<RemotePtr>{existing, ordinary};
  };
  const auto plan = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{ops}, span<const RemotePtr>{stable},
    span<const RemotePtr>{provisional},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, u32{2}, prune);
  assert(plan.valid);
  assert(prune_calls == 1);
  assert(plan.stable.size() == 2);
  assert(std::find(plan.stable.begin(), plan.stable.end(), leased) !=
         plan.stable.end());
  assert(std::find(plan.stable.begin(), plan.stable.end(), acquiring) !=
         plan.stable.end());
  assert(plan.provisional.empty());
  assert(plan.acquired.size() == 1);
  assert(plan.released.empty());
  assert(plan.results[1].accepted == 1);
  assert(plan.results[2].removed == 1);

  const vec<ReconcileReverseOp> release{
    op(ReconcileReverseOpKind::release_certificate_lease, 0,
       leased.raw_address, 1)};
  const auto released = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{release},
    span<const RemotePtr>{plan.stable},
    span<const RemotePtr>{plan.provisional},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, 2,
    [&](const vec<RemotePtr>& candidates) { return candidates; });
  assert(released.valid);
  assert(released.released.size() == 1);
  assert(released.results[0].accepted == 1);
}

void test_atomic_cut_capacity_eviction_is_bounded_and_replayable() {
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  const RemotePtr first{1, 0x1000};
  const RemotePtr second{1, 0x2000};
  const RemotePtr incoming{1, 0x3000};
  const vec<ReconcileReverseOp> ops{ReconcileReverseOp{
    .target_raw = 10,
    .old_candidate_raw = 0,
    .new_candidate_raw = incoming.raw_address,
    .placement_sequence = 3,
    .id = 42,
    .generation = 7,
    .kind = static_cast<std::uint32_t>(
      ReconcileReverseOpKind::acquire_certificate_lease),
  }};
  const vec<std::pair<RemotePtr, std::uint64_t>> active{
    std::pair{first, std::uint64_t{1}},
    std::pair{second, std::uint64_t{2}},
  };
  const vec<RemotePtr> stable{first, second};
  const auto prune = [&](const vec<RemotePtr>&) {
    return vec<RemotePtr>{second, incoming};
  };
  const auto plan = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{ops}, span<const RemotePtr>{stable}, {},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, 2, prune);
  assert(plan.valid);
  assert(plan.released.size() == 1);
  assert(plan.released[0].first == first);
  assert(plan.released[0].second == 1);
  assert(plan.stable.size() == 2);
  assert(plan.results[0].accepted == 1);

  const vec<std::pair<RemotePtr, std::uint64_t>> replay_active{
    std::pair{second, std::uint64_t{2}},
    std::pair{incoming, std::uint64_t{3}},
  };
  const auto replay = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{ops}, span<const RemotePtr>{plan.stable}, {},
    span<const std::pair<RemotePtr, std::uint64_t>>{replay_active}, 2, prune);
  assert(replay.valid);
  assert(replay.released.empty());
  assert(replay.results[0].accepted == 1);
}

void test_atomic_cut_capacity_only_add_preserves_certificate() {
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  const RemotePtr first{1, 0x1000};
  const RemotePtr second{1, 0x2000};
  const RemotePtr child = RemotePtr{1, 0x3000}.with_incarnation(1);
  const auto op = [&](ReconcileReverseOpKind kind, std::uint64_t sequence) {
    return ReconcileReverseOp{
      .target_raw = 10,
      .old_candidate_raw = 0,
      .new_candidate_raw = child.raw_address,
      .placement_sequence = sequence,
      .id = 42,
      .generation = 7,
      .kind = static_cast<std::uint32_t>(kind),
    };
  };
  const vec<RemotePtr> stable{first, second};
  const vec<RemotePtr> provisional{child};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>&) {
    ++prune_calls;
    return vec<RemotePtr>{second};
  };
  const vec<ReconcileReverseOp> optional{op(
    ReconcileReverseOpKind::add_if_capacity, 3)};
  const auto plan = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{optional}, span<const RemotePtr>{stable},
    span<const RemotePtr>{provisional}, {}, 2, prune);
  assert(plan.valid);
  assert(plan.results[0].accepted == 0);
  assert(plan.stable == stable);
  assert(plan.provisional == provisional);
  assert(plan.acquired.empty());
  assert(prune_calls == 0);
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(
             optional.front(), plan.results.front()));

  const vec<ReconcileReverseOp> acquire{op(
    ReconcileReverseOpKind::acquire_certificate_lease, 4)};
  const auto promoted = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{acquire}, span<const RemotePtr>{plan.stable},
    span<const RemotePtr>{plan.provisional}, {}, 2,
    [&](const vec<RemotePtr>&) { return vec<RemotePtr>{second, child}; });
  assert(promoted.valid);
  assert(promoted.results[0].accepted == 1);
  assert(promoted.provisional.empty());
  assert(promoted.acquired.size() == 1);
  assert(promoted.acquired.front().first == child);
  assert(promoted.acquired.front().second == 4);
}

void test_incremental_refinement_preserves_existing_leases() {
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  const RemotePtr certified{1, 0x1000};
  const RemotePtr ordinary{1, 0x2000};
  const RemotePtr incoming{1, 0x3000};
  const vec<RemotePtr> stable{certified, ordinary};
  const vec<std::pair<RemotePtr, std::uint64_t>> active{{certified, 7}};
  const vec<ReconcileReverseOp> ops{ReconcileReverseOp{
    .target_raw = 10,
    .new_candidate_raw = incoming.raw_address,
    .placement_sequence = 9,
    .id = 42,
    .generation = 1,
    .kind = static_cast<std::uint32_t>(ReconcileReverseOpKind::refine_add),
  }};
  const auto prune = [&](const vec<RemotePtr>& candidates,
                         size_t existing_count) {
    assert(candidates.size() == 3);
    assert(existing_count == stable.size());
    return vec<RemotePtr>{incoming};
  };
  const auto plan = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{ops}, span<const RemotePtr>{stable}, {},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, 2, prune);
  assert(plan.valid);
  assert(plan.results.front().accepted == 1);
  assert(plan.acquired.empty() && plan.released.empty());
  assert(std::find(plan.stable.begin(), plan.stable.end(), certified) !=
         plan.stable.end());
  assert(std::find(plan.stable.begin(), plan.stable.end(), incoming) !=
         plan.stable.end());
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(ops.front(), plan.results.front()));
}

void test_ordinary_snapshot_contention_does_not_block_atomic_cut() {
  using memory_node_storage_owner_index_detail::
    lease_merge_snapshot_requires_retry;
  using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
  const RemotePtr ordinary{1, 0x1000};
  const RemotePtr required{1, 0x2000};
  const vec<RemotePtr> mandatory{required};
  const span<const RemotePtr> required_view{mandatory};
  assert(!lease_merge_snapshot_requires_retry(
    ordinary, StableNodeSnapshotState::retryable, required_view));
  assert(!lease_merge_snapshot_requires_retry(
    ordinary, StableNodeSnapshotState::terminal, required_view));
  assert(!lease_merge_snapshot_requires_retry(
    required, StableNodeSnapshotState::stable, required_view));
  assert(lease_merge_snapshot_requires_retry(
    required, StableNodeSnapshotState::retryable, required_view));
  assert(lease_merge_snapshot_requires_retry(
    required, StableNodeSnapshotState::terminal, required_view));
}

}  // namespace

int main() {
  test_incremental_refinement_preserves_existing_leases();
  test_atomic_cut_capacity_only_add_preserves_certificate();
  test_lease_lifecycle_capacity_and_replay();
  test_concurrent_cleanup_sequences_are_independent();
  test_retired_target_cleanup_reclaims_lease();
  test_late_acquire_replay_does_not_reinsert_released_lease();
  test_cleanup_depends_only_on_matching_incarnation();
  test_atomic_cut_pins_leases_and_commits_certificate_sets();
  test_atomic_cut_capacity_eviction_is_bounded_and_replayable();
  test_ordinary_snapshot_contention_does_not_block_atomic_cut();
}

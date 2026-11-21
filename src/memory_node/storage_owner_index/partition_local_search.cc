#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/candidate_search_budget.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/ordered_commit_policy.hh"

#include <numeric>
#include <syncstream>

using namespace memory_node_storage_owner_index_detail;
using namespace memory_node_storage_owner_maintenance_detail;

namespace {

// Pending records can participate in construction, but they are not
// reachability certificates. Stage1 still prunes and protects only stable
// parents; Stage2 retains a published parent before adding pending edges.
StableNodeSnapshotState classify_stage1_traversal_snapshot(
    RemotePtr pointer,
    u64 before,
    u64 after,
    u32 slot_incarnation,
    bool include_provisional) {
  return include_provisional
    ? classify_construction_node_snapshot(
        pointer, before, after, slot_incarnation)
    : classify_stable_node_snapshot(
        pointer, before, after, slot_incarnation);
}

}  // namespace

vec<RemotePtr> MemoryNode::partition_local_search_candidates(
    const span<const element_t> query,
    const vec<RemotePtr>& entry_points,
    const Configuration& config,
    InsertBreakdownCounters* breakdown,
    const byte_t* integral_raw_query,
    vec<BeamEntry>* stage1_beam,
    vec<RemotePtr>* remote_frontier,
    bool record_pipeline_telemetry,
    SearchDependencySet* search_dependencies,
    bool include_provisional) {
  if (search_dependencies != nullptr) {
    search_dependencies->expanded.clear();
    search_dependencies->scored.clear();
  }
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<RemotePtr> local_neighbors;
  vec<byte_t> local_neighbor_entry;
  vec<byte_t> local_neighbor_decoded;
  if (scratch != nullptr) {
    scratch->neighbors.clear();
  }
  vec<RemotePtr>& neighbors = scratch != nullptr ? scratch->neighbors : local_neighbors;
  vec<byte_t>& neighbor_entry = scratch != nullptr
                                  ? scratch->neighbor_entry
                                  : local_neighbor_entry;
  vec<byte_t>& neighbor_decoded = scratch != nullptr
                                    ? scratch->neighbor_decoded
                                    : local_neighbor_decoded;
  neighbors.reserve(config.R);

  const auto trace_retry = [&](const char* boundary, RemotePtr candidate,
                               u64 before, u64 after, u32 slot_incarnation,
                               std::chrono::steady_clock::time_point started) {
    if (std::getenv("DUETANN_SEARCH_STALL_DEBUG") == nullptr) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - started < std::chrono::seconds(5)) return;
    thread_local u64 last_log_ns = 0;
    const u64 now_ns = steady_now_ns();
    if (now_ns - last_log_ns < 10'000'000'000ull) return;
    last_log_ns = now_ns;
    std::osyncstream(std::cerr) << std::string("[partition-search-wait] boundary=") + boundary +
      " candidate=" + std::to_string(candidate.raw_address) +
      " expected_incarnation=" + std::to_string(candidate.incarnation()) +
      " slot_incarnation=" + std::to_string(slot_incarnation) +
      " before=" + std::to_string(before) +
      " after=" + std::to_string(after) << std::endl;
  };

  const u32 construction_width = storage_owner_construction_width(config);
  const VectorDType dtype = VamanaNode::vector_dtype();
  const bool exact_integral_query = integral_raw_query != nullptr &&
    (dtype == VectorDType::uint8 || dtype == VectorDType::int8);
  auto score = [&](RemotePtr candidate) -> std::optional<distance_t> {
    auto started = std::chrono::steady_clock::now();
    if (!storage_node_pointer_addressable(candidate)) {
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns +=
          elapsed_ns_since(started);
      }
      return std::nullopt;
    }

    const byte_t* node = local_node_ptr(candidate);
    const byte_t* vector = node + VamanaNode::offset_vector();
    for (;;) {
      const u64 before = load_local_node_header_acquire(candidate);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      if ((before & (VamanaNode::HEADER_NODE_LOCK |
                     VamanaNode::HEADER_DELETED)) != 0 ||
          VamanaNode::header_incarnation(before) != candidate.incarnation() ||
          slot_incarnation != candidate.incarnation()) {
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(candidate);
        const StableNodeSnapshotState state =
          classify_stage1_traversal_snapshot(
            candidate, before, after, slot_incarnation, include_provisional);
        if (state == StableNodeSnapshotState::terminal) {
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_snapshot_read_ns +=
              elapsed_ns_since(started);
          }
          return std::nullopt;
        }
        trace_retry("score", candidate, before, after, slot_incarnation,
                    started);
        std::this_thread::yield();
        continue;
      }

      const auto distance_started = std::chrono::steady_clock::now();
      // The canonical insert bytes are available on both stages. Use the same
      // chunked integer SIMD reduction for every uint8/int8 dimension so the
      // local beam, remote continuation and final prune compare one distance
      // semantics. A normal query still uses the float-query path because it
      // may contain genuinely fractional coordinates.
      const distance_t distance = exact_integral_query
        ? typed_l2_distance(
            integral_raw_query, dtype, vector, dtype, config.dim)
        : distance_to_stored_vector(query, vector, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_distance_ns +=
          elapsed_ns_since(distance_started);
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(candidate);
      const StableNodeSnapshotState state =
        classify_stage1_traversal_snapshot(
          candidate, before, after, slot_incarnation, include_provisional);
      if (state == StableNodeSnapshotState::stable) {
        if (search_dependencies != nullptr) {
          search_dependencies->scored.push_back(NodeDependencyFingerprint{
            candidate, dependency_hash_seed,
            dependency_hash_bytes(
              span<const byte_t>{vector, VamanaNode::vector_bytes()}),
            after});
        }
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns +=
            elapsed_ns_since(started);
        }
        return distance;
      }
      if (state == StableNodeSnapshotState::terminal) {
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns +=
            elapsed_ns_since(started);
        }
        return std::nullopt;
      }
      // NODE_LOCK or any optimistic before/after mismatch is contention, not
      // evidence that the candidate is absent.  There is intentionally no
      // retry count that converts this state into successful convergence.
      std::this_thread::yield();
    }
  };
  auto expand = [&](RemotePtr candidate, auto&& visit) {
    const auto started = std::chrono::steady_clock::now();
    const byte_t* node = local_node_ptr(candidate);
    const auto validate_decoded_neighbors = [&] {
      const u32 edge_count =
        VamanaNode::decoded_neighbor_count(neighbor_decoded.data());
      const auto* slots = reinterpret_cast<const RemotePtr*>(
        neighbor_decoded.data() +
          VamanaNode::neighbor_payload_offset_in_read());
      for (u32 index = 0;
           index < edge_count &&
             index < VamanaNode::graph_entry_capacity();
           ++index) {
        if (!slots[index].is_null() &&
            !storage_node_pointer_addressable(slots[index])) {
          lib_failure(
            "stable Stage1 graph contains a malformed remote pointer");
        }
      }
    };
    u32 stable_optimistic_misses = 0;
    constexpr u32 kStableOptimisticMissesBeforeLock = 3;
    for (;;) {
      const u64 before = load_local_node_header_acquire(candidate);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const bool decoded = read_local_neighbor_list(
        candidate, neighbors, neighbor_entry, neighbor_decoded);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(candidate);
      const StableNodeSnapshotState state =
        classify_stage1_traversal_snapshot(
          candidate, before, after, slot_incarnation, include_provisional);
      if (state == StableNodeSnapshotState::terminal) {
        neighbors.clear();
        return;
      }
      if (state == StableNodeSnapshotState::stable && decoded) {
        validate_decoded_neighbors();
        if (search_dependencies != nullptr) {
          u64 adjacency_hash = dependency_hash_seed;
          for (const RemotePtr neighbor : neighbors) {
            adjacency_hash = dependency_hash_mix(
              adjacency_hash, neighbor.raw_address);
          }
          search_dependencies->expanded.push_back(NodeDependencyFingerprint{
            candidate, adjacency_hash,
            dependency_hash_bytes(span<const byte_t>{
              node + VamanaNode::offset_vector(), VamanaNode::vector_bytes()}),
            after});
        }
        break;
      }

      if (state != StableNodeSnapshotState::stable) {
        // A changing/locked header explains this miss; it is ordinary
        // contention rather than evidence of persistent graph corruption.
        stable_optimistic_misses = 0;
        trace_retry("expand", candidate, before, after, slot_incarnation,
                    started);
        std::this_thread::yield();
        continue;
      }

      ++stable_optimistic_misses;
      if (stable_optimistic_misses <
          kStableOptimisticMissesBeforeLock) {
        std::this_thread::yield();
        continue;
      }

      // Repeated checksum failure under one coherent physical identity needs
      // a race-closing diagnosis. The short count is not a search budget: it
      // only distinguishes optimistic publication tearing from a durable
      // malformed adjacency. Once the incarnation lock is held, a second
      // decode cannot race a graph writer and therefore must succeed.
      const IncarnationLockResult lock = try_lock_node(candidate);
      if (lock == IncarnationLockResult::busy) {
        trace_retry("expand_lock", candidate, before, before,
                    slot_incarnation, started);
        std::this_thread::yield();
        continue;
      }
      if (lock == IncarnationLockResult::stale) {
        neighbors.clear();
        return;
      }
      const u64 locked_header = load_local_node_header_acquire(candidate);
      if ((locked_header & VamanaNode::HEADER_DELETED) != 0) {
        neighbors.clear();
        unlock_node(candidate);
        return;
      }
      const bool locked_decoded = read_local_neighbor_list(
        candidate, neighbors, neighbor_entry, neighbor_decoded);
      unlock_node(candidate);
      if (!locked_decoded) {
        lib_failure(
          "stable Stage1 graph remains malformed while incarnation-locked");
      }
      validate_decoded_neighbors();
      break;
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    for (const RemotePtr neighbor : neighbors) {
      visit(neighbor);
    }
  };

  // This wrapper never suspends, so one reusable state per OS thread cannot
  // be observed by another coroutine while a search is in progress.
  thread_local PartitionLocalSearchBeam reusable_search(0, 1);
  const PartitionSearchBudget search_budget =
    stage1_partition_search_budget(
      construction_width, entry_points.size(),
      VamanaNode::graph_entry_capacity());
  vec<PartitionLocalSearchEntry>& final_beam =
    partition_local_construction_search_into(
      reusable_search, span<const RemotePtr>{entry_points}, storage_id_,
      construction_width, search_budget, score, expand);
  if (record_pipeline_telemetry && reusable_search.budget_exhausted()) {
    storage_owner_stage1_search_budget_exhausted_.fetch_add(
      1, std::memory_order_relaxed);
  }

  filter_final_partition_local_beam(
    final_beam, [&](RemotePtr candidate) {
      const auto validation_started = std::chrono::steady_clock::now();
      const byte_t* node = local_node_ptr(candidate);
      for (;;) {
        const u64 before = load_local_node_header_acquire(candidate);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          node + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(candidate);
        const StableNodeSnapshotState state = include_provisional
          ? classify_construction_node_snapshot(
              candidate, before, after, slot_incarnation)
          : classify_stable_node_snapshot(
              candidate, before, after, slot_incarnation);
        if (state != StableNodeSnapshotState::retryable) {
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_snapshot_read_ns +=
              elapsed_ns_since(validation_started);
          }
          return state == StableNodeSnapshotState::stable;
        }
        trace_retry("filter", candidate, before, after, slot_incarnation,
                    validation_started);
        std::this_thread::yield();
      }
    });

  if (stage1_beam != nullptr) {
    stage1_beam->clear();
    stage1_beam->reserve(final_beam.size());
    for (const PartitionLocalSearchEntry& entry : final_beam) {
      stage1_beam->push_back(
        BeamEntry{entry.rptr, entry.distance, true});
    }
  }
  if (remote_frontier != nullptr) {
    remote_frontier->assign(reusable_search.remote_frontier().begin(),
                            reusable_search.remote_frontier().end());
  }

  const auto started = std::chrono::steady_clock::now();
  vec<RemotePtr> candidates;
  candidates.reserve(final_beam.size());
  for (const PartitionLocalSearchEntry& entry : final_beam) {
    candidates.push_back(entry.rptr);
  }
  if (breakdown != nullptr) {
    // Beam insertion keeps results ordered, so this field now covers only the
    // final result materialization rather than a redundant sort.
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  if (search_dependencies != nullptr) search_dependencies->finalize();
  // All three externally visible Stage1 products have now been copied.
  // Release only exceptional thread-local high-water capacity; this is a
  // completed-search retention policy and never constrains an active search.
  reusable_search.trim_oversized_capacity(
    std::max<size_t>(1024, static_cast<size_t>(construction_width) * 8));
  return candidates;
}


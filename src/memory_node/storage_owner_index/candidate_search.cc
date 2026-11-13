#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/candidate_search_budget.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/detail.hh"

#include <numeric>

using namespace memory_node_storage_owner_index_detail;
using namespace memory_node_storage_owner_maintenance_detail;

const vec<RemotePtr>& MemoryNode::continue_stage2_search_candidates(
    const StorageOwnerMaintenanceTask& task,
    const NodeSnapshot& target,
    const Configuration& config,
    bool record_stage2_telemetry) {
  lib_assert(!task.stage1_beam.empty(),
             "stage2 continuation requires the exact Stage1 beam");
  lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
             "stage2 continuation target vector is incomplete");

  const auto classify_inherited_stage1_entry = [&](RemotePtr pointer) {
    if (!storage_node_pointer_addressable(pointer) ||
        !local_shard(pointer.memory_node())) {
      return StableNodeSnapshotState::terminal;
    }
    const byte_t* node = local_node_ptr(pointer);
    const u64 before = load_local_node_header_acquire(pointer);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 after = load_local_node_header_acquire(pointer);
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  thread_local vec<PartitionLocalSearchEntry> local_beam;
  local_beam.clear();
  local_beam.reserve(task.stage1_beam.size());
  for (const BeamEntry& entry : task.stage1_beam) {
    // Stage1 already computed the exact distance. Revalidation needs only the
    // stable physical identity, not another D-byte vector materialization.
    // A transient lock/torn observation is retried without a fixed count;
    // only a coherent stale/deleted/provisional identity is omitted.
    for (;;) {
      const StableNodeSnapshotState state =
        classify_inherited_stage1_entry(entry.rptr);
      if (state == StableNodeSnapshotState::stable) {
        local_beam.push_back(
          PartitionLocalSearchEntry{entry.rptr, entry.distance, true});
        break;
      }
      if (state == StableNodeSnapshotState::terminal) break;
      std::this_thread::yield();
    }
  }

  thread_local vec<element_t> query;
  const VectorDType dtype = VamanaNode::vector_dtype();
  if (dtype == VectorDType::float32) {
    query.resize(VamanaNode::DIM);
    decode_storage_vector_to_float(
      target.vector_data.data(), dtype, VamanaNode::DIM, query.data());
  } else {
    // Integer Stage2 uses the canonical raw bytes on both sides; decoding an
    // unused float query would add O(D) work to every insertion.
    query.clear();
  }

  const auto score_batch = [&](span<const RemotePtr> pointers, auto&& emit) {
    // Count requested records rather than only live emissions: this is the
    // actual remote snapshot traffic Stage2 paid for under churn.
    if (record_stage2_telemetry) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        pointers.size(), std::memory_order_relaxed);
    }
    const vec<BeamEntry>& scores = score_stable_node_vectors_batched(
      pointers, target.vector_data.data(), span<const element_t>{query},
      config);
    for (const BeamEntry& score : scores) {
      emit(score.rptr, score.distance);
    }
  };
  const auto expand = [&](RemotePtr pointer, auto&& visit) {
    // Reuse adjacency storage across the O(L*R) continuation loop. Returning
    // a fresh neighbor vector here would otherwise allocate once per remote
    // expansion even though the consumer only streams its elements.
    thread_local GraphAdjacency adjacency;
    if (!read_graph_adjacency(pointer, adjacency) || adjacency.deleted) {
      return;
    }
    for (const RemotePtr neighbor : adjacency.stable) visit(neighbor);
  };

  bool budget_exhausted = false;
  u64 remote_expansions = 0;
  const u32 construction_width =
    storage_owner_construction_width(config);
  if (record_stage2_telemetry) {
    storage_owner_stage2_continuations_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stage2_remote_frontier_items_.fetch_add(
      task.stage1_remote_frontier.size(), std::memory_order_relaxed);
  }
  const vec<PartitionLocalSearchEntry>& final_beam =
    continue_partition_construction_search_into(
      span<const PartitionLocalSearchEntry>{local_beam},
      span<const RemotePtr>{task.stage1_remote_frontier}, storage_id_,
      construction_width,
      stage2_search_budget_for_backlog(
        config, storage_owner_maintenance_incomplete(), construction_width),
      score_batch, expand, &budget_exhausted, &remote_expansions);
  if (record_stage2_telemetry) {
    storage_owner_stage2_remote_expansions_.fetch_add(
      remote_expansions, std::memory_order_relaxed);
  }
  if (record_stage2_telemetry && budget_exhausted) {
    storage_owner_stage2_search_budget_exhausted_.fetch_add(
      1, std::memory_order_relaxed);
  }
  thread_local vec<RemotePtr> candidates;
  candidates.clear();
  candidates.reserve(final_beam.size());
  for (const PartitionLocalSearchEntry& entry : final_beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}

void MemoryNode::continue_stage2_search_candidates_batched(
    span<const StorageOwnerMaintenanceTask> tasks,
    span<const NodeSnapshot> targets,
    vec<vec<RemotePtr>>& candidates_by_task,
    const Configuration& config) {
  lib_assert(tasks.size() == targets.size(),
             "batched Stage2 continuation lost task/target correlation");
  // Each resumable Stage2 context owns this output. Preserve active inner
  // vector capacity across context reuse instead of destroying every row on
  // each batch; the final-beam scatter below clears the rows it writes.
  candidates_by_task.resize(tasks.size());
  if (tasks.empty()) return;

  const u32 construction_width = storage_owner_construction_width(config);
  const PartitionSearchBudget budget = stage2_search_budget_for_backlog(
    config, storage_owner_maintenance_incomplete(), construction_width);

  // Keep independent beams/visited sets but retain their capacity across
  // high-frequency batches on this maintenance OS worker.
  thread_local vec<vec<PartitionLocalSearchEntry>> local_beams;
  thread_local vec<PartitionContinuationSeed> seeds;
  thread_local PartitionContinuationBatch continuation_batch;
  thread_local vec<bool> budget_exhausted;
  thread_local vec<u64> expansion_counts;
  thread_local vec<RemotePtr> score_unique;
  thread_local vec<size_t> score_order;
  thread_local vec<size_t> score_group_offsets;
  thread_local vec<RemotePtr> graph_unique;
  thread_local vec<std::pair<RemotePtr, GraphAdjacency>> graph_snapshots;
  thread_local dense_hashmap_t<u64, const GraphAdjacency*> graph_by_pointer;
  local_beams.resize(tasks.size());
  seeds.resize(tasks.size());

  const auto classify_inherited_stage1_entry = [&](RemotePtr pointer) {
    if (!storage_node_pointer_addressable(pointer) ||
        !local_shard(pointer.memory_node())) {
      return StableNodeSnapshotState::terminal;
    }
    const byte_t* node = local_node_ptr(pointer);
    const u64 before = load_local_node_header_acquire(pointer);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 after = load_local_node_header_acquire(pointer);
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  u64 frontier_items = 0;
  for (size_t item = 0; item < tasks.size(); ++item) {
    const StorageOwnerMaintenanceTask& task = tasks[item];
    const NodeSnapshot& target = targets[item];
    lib_assert(!task.stage1_beam.empty(),
               "stage2 task lost its Stage1 continuation beam");
    lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
               "stage2 continuation target vector is incomplete");
    vec<PartitionLocalSearchEntry>& local = local_beams[item];
    local.clear();
    local.reserve(task.stage1_beam.size());
    for (const BeamEntry& entry : task.stage1_beam) {
      // Stage1 distances remain authoritative for its local beam. Identity
      // validation is local to the Stage1 owner and therefore adds no RDMA
      // serialization to the cross-task wavefront.
      for (;;) {
        const StableNodeSnapshotState state =
          classify_inherited_stage1_entry(entry.rptr);
        if (state == StableNodeSnapshotState::stable) {
          local.push_back({entry.rptr, entry.distance, true});
          break;
        }
        if (state == StableNodeSnapshotState::terminal) break;
        std::this_thread::yield();
      }
    }
    seeds[item] = {
      .local_beam = span<const PartitionLocalSearchEntry>{local},
      .remote_frontier = span<const RemotePtr>{task.stage1_remote_frontier},
    };
    frontier_items += task.stage1_remote_frontier.size();
  }
  storage_owner_stage2_continuations_.fetch_add(
    tasks.size(), std::memory_order_relaxed);
  storage_owner_stage2_remote_frontier_items_.fetch_add(
    frontier_items, std::memory_order_relaxed);

  const auto score_wave = [&](
      span<const PartitionContinuationScoreRequest> requests,
      auto&& emit) {
    storage_owner_stage2_scored_candidates_.fetch_add(
      requests.size(), std::memory_order_relaxed);
    score_order.resize(requests.size());
    std::iota(score_order.begin(), score_order.end(), size_t{0});
    std::sort(score_order.begin(), score_order.end(),
              [&](size_t lhs, size_t rhs) {
                const auto& left = requests[lhs];
                const auto& right = requests[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                return left.search_index < right.search_index;
              });
    score_unique.clear();
    score_group_offsets.clear();
    for (size_t position = 0; position < score_order.size(); ++position) {
      const RemotePtr pointer = requests[score_order[position]].pointer;
      if (score_unique.empty() || score_unique.back() != pointer) {
        score_unique.push_back(pointer);
        score_group_offsets.push_back(position);
      }
    }
    score_group_offsets.push_back(score_order.size());
    storage_owner_stage2_vector_read_waves_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stage2_vector_unique_reads_.fetch_add(
      score_unique.size(), std::memory_order_relaxed);

    const VectorDType dtype = VamanaNode::vector_dtype();
    const auto emit_group = [&](size_t group_index,
                                const byte_t* candidate_vector) {
      for (size_t position = score_group_offsets[group_index];
           position < score_group_offsets[group_index + 1]; ++position) {
        const auto& request = requests[score_order[position]];
        const distance_t distance = distance_between_vectors(
          targets[request.search_index].vector_data.data(), dtype,
          candidate_vector, dtype, config);
        emit(request.search_index, request.pointer, distance);
      }
    };

    StorageOwnerThread* thread = current_storage_owner_thread_;
    if (thread == nullptr || !thread->has_peer_scratch()) {
      for (size_t group = 0; group < score_unique.size(); ++group) {
        NodeSnapshot snapshot;
        if (read_node_snapshot(score_unique[group], snapshot) &&
            !snapshot.deleted &&
            (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
            snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
          emit_group(group, snapshot.vector_data.data());
        }
      }
      return;
    }

    struct PendingVectorRead {
      size_t group_index{};
      RemotePtr pointer;
      byte_t* buffer{};
      byte_t* after_header{};
      u64 before{};
      u32 slot_incarnation{};
    };
    thread_local vec<PendingVectorRead> pending;
    thread_local vec<PeerReadRequest> read_requests;
    thread_local vec<PeerReadPairRequest> read_pairs;
    const size_t snapshot_stride = aligned_snapshot_bytes();
    const size_t validation_offset =
      memory_node_detail::storage_owner_snapshot_validation_offset();
    const size_t max_batch =
      storage_owner_snapshot_batch_size(config, thread);
    const bool ordered_pair_supported =
      memory_node_detail::peer_rdma_read_pair_group_limit(
        peer_rdma_read_credit_plan()) != 0;
    lib_assert(validation_offset + VamanaNode::HEADER_SIZE <=
                 snapshot_stride,
               "storage-owner snapshot slot lost after-header scratch");
    pending.reserve(max_batch);
    for (size_t begin = 0; begin < score_unique.size();
         begin += max_batch) {
      const size_t end = std::min(score_unique.size(), begin + max_batch);
      pending.clear();
      read_requests.clear();
      read_pairs.clear();
      u32 remote_slot = 0;
      for (size_t group = begin; group < end; ++group) {
        const RemotePtr pointer = score_unique[group];
        if (!storage_node_pointer_addressable(pointer)) {
          if (!pointer.is_null()) {
            report_rejected_graph_pointer(
              "stage2_continuation_score", pointer, RemotePtr{}, group);
          }
          continue;
        }
        if (local_shard(pointer.memory_node())) {
          NodeSnapshot snapshot;
          if (read_node_snapshot(pointer, snapshot) && !snapshot.deleted &&
              (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
              snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
            emit_group(group, snapshot.vector_data.data());
          }
          continue;
        }
        const size_t scratch_offset =
          static_cast<size_t>(remote_slot++) * snapshot_stride;
        lib_assert(scratch_offset + validation_offset +
                     VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                   "storage-owner scratch cannot hold Stage2 score wave");
        byte_t* buffer = thread->coroutine_scratch(scratch_offset);
        const PeerReadRequest full_snapshot{
          .shard_id = pointer.memory_node(),
          .remote_offset = pointer.byte_offset(),
          .destination = buffer,
          .bytes = VamanaNode::size_until_vector_end(),
        };
        byte_t* after_header = buffer + validation_offset;
        if (ordered_pair_supported) {
          read_pairs.push_back(PeerReadPairRequest{
            .full_snapshot = full_snapshot,
            .after_header = PeerReadRequest{
              .shard_id = pointer.memory_node(),
              .remote_offset = pointer.byte_offset(),
              .destination = after_header,
              .bytes = VamanaNode::HEADER_SIZE,
            },
          });
        } else {
          read_requests.push_back(full_snapshot);
        }
        pending.push_back({group, pointer, buffer, after_header});
      }
      if (ordered_pair_supported) {
        post_peer_read_pairs_async(
          *thread, span<const PeerReadPairRequest>{read_pairs});
      } else {
        post_peer_reads_async(
          *thread, span<const PeerReadRequest>{read_requests});
      }
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }

      size_t valid_count = 0;
      read_requests.clear();
      for (PendingVectorRead& read : pending) {
        read.before = *reinterpret_cast<const u64*>(read.buffer);
        read.slot_incarnation = *reinterpret_cast<const u32*>(
          read.buffer + VamanaNode::offset_slot_incarnation());
        if (!stable_vector_snapshot_valid(
              read.pointer, read.before, read.before,
              read.slot_incarnation)) {
          continue;
        }
        if (valid_count != static_cast<size_t>(&read - pending.data())) {
          pending[valid_count] = read;
        }
        PendingVectorRead& accepted = pending[valid_count++];
        if (!ordered_pair_supported) {
          read_requests.push_back(PeerReadRequest{
            .shard_id = accepted.pointer.memory_node(),
            .remote_offset = accepted.pointer.byte_offset(),
            .destination = accepted.buffer,
            .bytes = VamanaNode::HEADER_SIZE,
          });
        }
      }
      pending.resize(valid_count);
      if (ordered_pair_supported) {
        for (const PendingVectorRead& read : pending) {
          const u64 after = *reinterpret_cast<const u64*>(
            read.after_header);
          if (stable_vector_snapshot_valid(
                read.pointer, read.before, after,
                read.slot_incarnation)) {
            emit_group(read.group_index,
                       read.buffer + VamanaNode::offset_vector());
          }
        }
        continue;
      }

      // A transport with only one atomic/WQE credit cannot reserve an
      // ordered two-read chain. Preserve the original two-wave validation;
      // never weaken a stable snapshot to a single read.
      post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }
      for (const PendingVectorRead& read : pending) {
        const u64 after = *reinterpret_cast<const u64*>(read.buffer);
        if (stable_vector_snapshot_valid(
              read.pointer, read.before, after,
              read.slot_incarnation)) {
          emit_group(read.group_index,
                     read.buffer + VamanaNode::offset_vector());
        }
      }
    }
  };

  const auto expand_wave = [&](
      span<const PartitionContinuationExpandRequest> requests,
      auto&& emit) {
    graph_unique.clear();
    graph_unique.reserve(requests.size());
    for (const auto& request : requests) {
      graph_unique.push_back(request.pointer);
    }
    std::sort(graph_unique.begin(), graph_unique.end(),
              [](RemotePtr lhs, RemotePtr rhs) {
                return lhs.raw_address < rhs.raw_address;
              });
    graph_unique.erase(
      std::unique(graph_unique.begin(), graph_unique.end()),
      graph_unique.end());
    storage_owner_stage2_graph_read_waves_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stage2_graph_unique_reads_.fetch_add(
      graph_unique.size(), std::memory_order_relaxed);
    const size_t graph_snapshot_count = read_graph_adjacencies_batched_into(
      span<const RemotePtr>{graph_unique}, config, graph_snapshots);
    graph_by_pointer.clear();
    graph_by_pointer.reserve(graph_snapshot_count);
    for (size_t snapshot_index = 0;
         snapshot_index < graph_snapshot_count; ++snapshot_index) {
      const auto& [pointer, adjacency] = graph_snapshots[snapshot_index];
      if (!adjacency.deleted) {
        graph_by_pointer.emplace(pointer.raw_address, &adjacency);
      }
    }
    for (const auto& request : requests) {
      const auto found = graph_by_pointer.find(request.pointer.raw_address);
      if (found == graph_by_pointer.end()) continue;
      const GraphAdjacency& adjacency = *found->second;
      for (const RemotePtr neighbor : adjacency.stable) {
        emit(request.search_index, neighbor);
      }
    }
  };

  const auto& final_beams = continuation_batch.run(
    span<const PartitionContinuationSeed>{seeds}, storage_id_,
    construction_width, budget, score_wave, expand_wave,
    &budget_exhausted, &expansion_counts);
  u64 total_expansions = 0;
  u64 exhausted = 0;
  for (size_t item = 0; item < final_beams.size(); ++item) {
    total_expansions += expansion_counts[item];
    exhausted += budget_exhausted[item];
    vec<RemotePtr>& candidates = candidates_by_task[item];
    candidates.clear();
    candidates.reserve(final_beams[item].size());
    for (const PartitionLocalSearchEntry& entry : final_beams[item]) {
      candidates.push_back(entry.rptr);
    }
  }
  storage_owner_stage2_remote_expansions_.fetch_add(
    total_expansions, std::memory_order_relaxed);
  storage_owner_stage2_search_budget_exhausted_.fetch_add(
    exhausted, std::memory_order_relaxed);
}

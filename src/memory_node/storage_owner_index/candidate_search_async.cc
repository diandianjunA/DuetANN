#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/candidate_search_budget.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/detail.hh"

#include <numeric>

using namespace memory_node_storage_owner_index_detail;
using namespace memory_node_storage_owner_maintenance_detail;

Stage2SearchAdvanceResult
MemoryNode::advance_stage2_search_candidates_batched(
    span<const StorageOwnerMaintenanceTask> tasks,
    span<const NodeSnapshot> targets,
    vec<vec<RemotePtr>>& candidates_by_task,
    Stage2SearchIoState& state,
    const Configuration& config) {
  lib_assert(tasks.size() == targets.size(),
             "asynchronous Stage2 continuation lost task/target correlation");
  candidates_by_task.resize(tasks.size());
  if (tasks.empty()) {
    state.reset();
    return Stage2SearchAdvanceResult::complete;
  }

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    // Unit/single-node callers have no registered lane scratch.  Production
    // maintenance always enters the resumable path below.
    continue_stage2_search_candidates_batched(
      tasks, targets, candidates_by_task, config);
    state.reset();
    return Stage2SearchAdvanceResult::complete;
  }

  const u32 construction_width = storage_owner_construction_width(config);
  const PartitionSearchBudget budget = stage2_search_budget_for_backlog(
    config, storage_owner_maintenance_incomplete(), construction_width);
  const auto& credit_plan = peer_rdma_read_credit_plan();
  const auto ordinary_dispatch_limits =
    memory_node_detail::peer_rdma_read_dispatch_limits(credit_plan);
  const auto pair_dispatch_limits =
    memory_node_detail::peer_rdma_read_pair_dispatch_limits(credit_plan);
  const auto snapshot_dispatch_limits =
    memory_node_detail::peer_rdma_snapshot_dispatch_limits(credit_plan);
  lib_assert(ordinary_dispatch_limits.global_items != 0 &&
               ordinary_dispatch_limits.per_peer_items != 0,
             "asynchronous Stage2 has no RDMA READ credit");
  const bool ordered_pairs = pair_dispatch_limits.global_items != 0 &&
    pair_dispatch_limits.per_peer_items != 0;
  const bool mixed_snapshots = ordered_pairs &&
    snapshot_dispatch_limits.per_peer_pairs != 0;
  // Scratch bounds distinct remote physical records, not logical consumers.
  // Local/terminal work and multiple searches sharing one pointer consume no
  // additional registered bytes and can retire in this dispatcher turn. RDMA
  // quota is enforced independently below: global_items caps the whole
  // physical wave and per_peer_items caps each destination shard.
  const size_t score_dispatch_limit =
    storage_owner_snapshot_batch_size(config, thread);
  const size_t graph_dispatch_limit =
    storage_owner_graph_batch_size(config, thread);
  lib_assert(score_dispatch_limit != 0 && graph_dispatch_limit != 0,
             "asynchronous Stage2 scratch cannot hold one RDMA record");

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
    return classify_construction_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  if (!state.initialized) {
    lib_assert(state.score_phase == Stage2ScoreIoPhase::idle &&
                 state.graph_phase == Stage2GraphIoPhase::idle,
               "new Stage2 continuation inherited pending lane I/O");
    state.local_beams.resize(tasks.size());
    state.seeds.resize(tasks.size());
    state.search_seeded.assign(tasks.size(), 0);
    state.score_collect_cursors.assign(tasks.size(), {});
    state.graph_prefetch_cache.resize(tasks.size());
    state.continuation.initialize(
      tasks.size(), storage_id_, construction_width, budget);
    state.round_robin_search = 0;
    state.initialized = true;

    u64 frontier_items = 0;
    for (const StorageOwnerMaintenanceTask& task : tasks) {
      frontier_items += task.stage1_remote_frontier.size();
    }
    storage_owner_stage2_continuations_.fetch_add(
      tasks.size(), std::memory_order_relaxed);
    storage_owner_stage2_remote_frontier_items_.fetch_add(
      frontier_items, std::memory_order_relaxed);
  }

  // Activate every readable Stage1 handoff independently. A transient lock in
  // one inherited local beam leaves only that search uninitialized; searches
  // already activated below continue to issue and retire RDMA dispatches.
  for (size_t item = 0; item < tasks.size(); ++item) {
    if (state.search_seeded[item] != 0) continue;
    const StorageOwnerMaintenanceTask& task = tasks[item];
    const NodeSnapshot& target = targets[item];
    lib_assert(!task.stage1_beam.empty(),
               "stage2 task lost its Stage1 continuation beam");
    lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
               "stage2 continuation target vector is incomplete");

    vec<PartitionLocalSearchEntry>& local = state.local_beams[item];
    local.clear();
    local.reserve(task.stage1_beam.size());
    bool retryable = false;
    for (const BeamEntry& entry : task.stage1_beam) {
      const StableNodeSnapshotState entry_state =
        classify_inherited_stage1_entry(entry.rptr);
      if (entry_state == StableNodeSnapshotState::retryable) {
        retryable = true;
        break;
      }
      if (entry_state == StableNodeSnapshotState::stable) {
        local.push_back({entry.rptr, entry.distance, true});
      }
    }
    if (retryable) continue;

    state.seeds[item] = {
      .local_beam = span<const PartitionLocalSearchEntry>{local},
      .remote_frontier = span<const RemotePtr>{task.stage1_remote_frontier},
    };
    state.continuation.initialize_search(item, state.seeds[item]);
    state.search_seeded[item] = 1;
  }

  const auto classify_vector_snapshot = [](
      RemotePtr pointer, u64 before, u64 after, u32 slot_incarnation) {
    return classify_construction_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  const auto clear_score_dispatch = [&] {
    state.clear_score_dispatch();
  };
  const auto clear_graph_dispatch = [&] {
    state.clear_graph_dispatch();
  };

  const auto resolve_score_group = [&] (
      size_t group_index,
      const byte_t* candidate_vector) -> size_t {
    lib_assert(group_index + 1 < state.score_group_offsets.size(),
               "Stage2 score dispatch group is out of range");
    const VectorDType dtype = VamanaNode::vector_dtype();
    size_t resolved = 0;
    for (size_t position = state.score_group_offsets[group_index];
         position < state.score_group_offsets[group_index + 1]; ++position) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[state.score_order[position]];
      const distance_t distance = distance_between_vectors(
        targets[consumer.search_index].vector_data.data(), dtype,
        candidate_vector, dtype, config);
      resolved += state.continuation.resolve_score_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        std::optional<distance_t>{distance});
    }
    if (resolved != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        resolved, std::memory_order_relaxed);
    }
    return resolved;
  };

  const auto resolve_terminal_score_group = [&] (
      size_t group_index) -> size_t {
    lib_assert(group_index + 1 < state.score_group_offsets.size(),
               "Stage2 terminal score group is out of range");
    size_t resolved = 0;
    for (size_t position = state.score_group_offsets[group_index];
         position < state.score_group_offsets[group_index + 1]; ++position) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[state.score_order[position]];
      resolved += state.continuation.resolve_score_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        std::nullopt);
    }
    if (resolved != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        resolved, std::memory_order_relaxed);
    }
    return resolved;
  };

  const auto resolve_graph_group = [&] (
      size_t group_index, span<const RemotePtr> neighbors) -> size_t {
    lib_assert(group_index + 1 < state.graph_group_offsets.size(),
               "Stage2 graph dispatch group is out of range");
    size_t resolved = 0;
    for (size_t position = state.graph_group_offsets[group_index];
         position < state.graph_group_offsets[group_index + 1]; ++position) {
      const Stage2GraphConsumer& consumer =
        state.graph_consumers[state.graph_order[position]];
      if (consumer.speculative) continue;
      resolved += state.continuation.resolve_expand_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        neighbors);
    }
    return resolved;
  };

  const auto collect_score_dispatch = [&] {
    state.score_consumers.clear();
    state.score_selected_remote.clear();
    state.score_selected_remote.reserve(score_dispatch_limit);
    for (Stage2ScoreRoundRobinCursor& cursor :
         state.score_collect_cursors) {
      cursor.begin_dispatch();
    }
    thread_local vec<u32> remote_wrs_by_peer;
    thread_local vec<u32> remote_pairs_by_peer;
    thread_local vec<u32> score_many_items_by_peer;
    thread_local vec<size_t> score_many_pending_by_peer;
    thread_local vec<u8> score_many_peer_eligible;
    remote_wrs_by_peer.resize(num_storage_nodes_);
    remote_pairs_by_peer.resize(num_storage_nodes_);
    score_many_items_by_peer.resize(num_storage_nodes_);
    score_many_pending_by_peer.assign(num_storage_nodes_, 0);
    score_many_peer_eligible.assign(num_storage_nodes_, 0);
    const u32 score_many_item_limit =
      std::max<u32>(1, config.storage_owner_search_snapshot_batch);
    state.score_many_dispatch = false;
    if (config.storage_owner_stage2_score_many) {
      for (size_t search_index = 0; search_index < tasks.size();
           ++search_index) {
        if (state.search_seeded[search_index] == 0) continue;
        for (const PartitionContinuationScoreRequest& request :
             state.continuation.pending_score_requests(search_index)) {
          if (storage_node_pointer_addressable(request.pointer) &&
              !local_shard(request.pointer.memory_node())) {
            ++score_many_pending_by_peer[request.pointer.memory_node()];
          }
        }
      }
      for (u32 peer = 0; peer < num_storage_nodes_; ++peer) {
        score_many_peer_eligible[peer] =
          stage2_score_many_peer_eligible(
            score_many_pending_by_peer[peer], score_many_item_limit,
            config.storage_owner_stage2_score_many_min_items);
        state.score_many_dispatch |= score_many_peer_eligible[peer] != 0;
      }
    }
    memory_node_detail::PeerRdmaVectorSnapshotDispatchQuota quota;
    // This collector prepares a new dispatch, so choose its accounting from
    // the transport capability, not from state.ordered_snapshot_pairs (which
    // describes the previously prepared/in-flight dispatch and is cleared at
    // retirement). Dynamic records must consume both WR credits before this
    // wave is materialized.
    quota.reset(
      mixed_snapshots, snapshot_dispatch_limits,
      ordinary_dispatch_limits,
      std::span<u32>{
        remote_wrs_by_peer.data(), remote_wrs_by_peer.size()},
      std::span<u32>{
        remote_pairs_by_peer.data(), remote_pairs_by_peer.size()});
    Stage2ScoreManyDispatchQuota score_many_quota;
    score_many_quota.reset(
      std::span<u32>{score_many_items_by_peer.data(),
                     score_many_items_by_peer.size()},
      score_many_item_limit);
    const size_t search_count = tasks.size();
    size_t last_search = state.round_robin_search % search_count;
    size_t physical_reads = 0;
    bool selected_any = false;
    // Visit every logical request at most once. In the one-sided path only
    // distinct remote records consume lane scratch/read credit; score-many
    // instead counts logical wire items per destination RPC. In either mode a
    // full peer cannot hide local work or a different peer in this pass.
    for (;;) {
      bool examined_this_round = false;
      for (size_t offset = 0; offset < search_count; ++offset) {
        const size_t search_index =
          (state.round_robin_search + offset) % search_count;
        if (state.search_seeded[search_index] == 0) continue;
        const auto requests =
          state.continuation.pending_score_requests(search_index);
        const std::optional<size_t> position =
          state.score_collect_cursors[search_index].take(requests.size());
        if (!position.has_value()) continue;
        examined_this_round = true;
        const PartitionContinuationScoreRequest& request =
          requests[*position];
        const bool remote = storage_node_pointer_addressable(
          request.pointer) && !local_shard(request.pointer.memory_node());
        const bool duplicate_remote = remote &&
          !state.score_many_dispatch &&
          state.score_selected_remote.contains(request.pointer);
        const bool distinct_remote = remote && !duplicate_remote;
        const bool requires_after_header = distinct_remote &&
          !VamanaNode::immutable_base_record(request.pointer);
        const bool accepted = state.score_many_dispatch
          ? (!remote ||
             (score_many_peer_eligible[request.pointer.memory_node()] != 0 &&
              score_many_quota.try_accept(
                request.pointer.memory_node(), true)))
          : stage2_consumer_fits_physical_scratch(
              distinct_remote, physical_reads, score_dispatch_limit) &&
            quota.try_accept(request.pointer.memory_node(), distinct_remote,
                             requires_after_header);
        if (!accepted) {
          // The cursor has advanced, but the continuation request remains
          // unresolved.  This prevents a hot peer at its quota from hiding a
          // later request for another peer, and revisits the skipped request
          // in a subsequent finite dispatch without duplicating it here.
          continue;
        }
        state.score_consumers.push_back({
          request.search_index, request.generation, request.pointer});
        if (!state.score_many_dispatch && distinct_remote) {
          state.score_selected_remote.insert(request.pointer);
          ++physical_reads;
        }
        last_search = search_index;
        selected_any = true;
      }
      if (!examined_this_round) break;
    }
    if (selected_any) {
      state.round_robin_search = (last_search + 1) % search_count;
    }
    return selected_any;
  };

  const auto collect_graph_dispatch = [&] {
    state.graph_consumers.clear();
    state.graph_selected_remote.clear();
    state.graph_selected_remote.reserve(graph_dispatch_limit);
    thread_local vec<u32> remote_items_by_peer;
    thread_local vec<u32> wire_items_by_peer;
    remote_items_by_peer.resize(num_storage_nodes_);
    wire_items_by_peer.assign(num_storage_nodes_, 0);
    memory_node_detail::PeerRdmaReadDispatchQuota quota;
    quota.reset(
      ordinary_dispatch_limits,
      std::span<u32>{
        remote_items_by_peer.data(), remote_items_by_peer.size()});
    const size_t search_count = tasks.size();
    size_t last_search = state.round_robin_search % search_count;
    size_t physical_reads = 0;
    bool selected_any = false;
    for (size_t offset = 0; offset < search_count; ++offset) {
      const size_t search_index =
        (state.round_robin_search + offset) % search_count;
      if (state.search_seeded[search_index] == 0) continue;
      const auto request =
        state.continuation.pending_expand_request(search_index);
      if (!request.has_value()) continue;
      const bool remote = storage_node_pointer_addressable(
        request->pointer) && !local_shard(request->pointer.memory_node());
      const bool duplicate_remote = remote &&
        state.graph_selected_remote.contains(request->pointer);
      const bool distinct_remote = remote && !duplicate_remote;
      if (!stage2_consumer_fits_physical_scratch(
            distinct_remote, physical_reads, graph_dispatch_limit) ||
          !quota.try_accept(request->pointer.memory_node(), distinct_remote)) {
        continue;
      }
      state.graph_consumers.push_back({
        request->search_index, request->generation, request->pointer,
        false});
      if (remote) ++wire_items_by_peer[request->pointer.memory_node()];
      if (distinct_remote) {
        state.graph_selected_remote.insert(request->pointer);
        ++physical_reads;
      }
      last_search = search_index;
      selected_any = true;
    }
    // Ordered issue is piggyback-only: it may occupy unused items in any peer
    // RPC already selected by this context wave, but it can neither create a
    // new destination RPC nor split an existing one.  Ranking candidates
    // globally avoids wasting a high-confidence preview slot merely because
    // the next nearest candidate lives on a different already-active peer.
    const u64 prefetch_hits =
      storage_owner_stage2_graph_prefetch_hits_.load(
        std::memory_order_relaxed);
    const u64 prefetch_wasted =
      storage_owner_stage2_graph_prefetch_wasted_.load(
        std::memory_order_relaxed);
    const u64 prefetch_outcomes = prefetch_hits + prefetch_wasted;
    u64 next_feedback_outcome =
      storage_owner_stage2_graph_feedback_next_outcome_.load(
        std::memory_order_relaxed);
    if (prefetch_outcomes >= next_feedback_outcome &&
        storage_owner_stage2_graph_feedback_next_outcome_
          .compare_exchange_strong(
            next_feedback_outcome, prefetch_outcomes + 512,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
      const u64 base_hits =
        storage_owner_stage2_graph_feedback_base_hits_.exchange(
          prefetch_hits, std::memory_order_relaxed);
      const u64 base_wasted =
        storage_owner_stage2_graph_feedback_base_wasted_.exchange(
          prefetch_wasted, std::memory_order_relaxed);
      storage_owner_stage2_graph_issue_width_current_.store(
        stage2_ordered_issue_width(
          prefetch_hits >= base_hits ? prefetch_hits - base_hits : 0,
          prefetch_wasted >= base_wasted
            ? prefetch_wasted - base_wasted : 0,
          config.storage_owner_stage2_graph_issue_width),
        std::memory_order_relaxed);
    }
    const u32 issue_width = std::min(
      config.storage_owner_stage2_graph_issue_width,
      storage_owner_stage2_graph_issue_width_current_.load(
        std::memory_order_relaxed));
    const size_t cache_capacity =
      static_cast<size_t>(config.storage_owner_stage2_graph_issue_width);
    u64 speculative_added = 0;
    if (issue_width > 1) {
      thread_local vec<RemotePtr> prefetch_candidates;
      const size_t authoritative_count = state.graph_consumers.size();
      for (size_t consumer_index = 0;
           consumer_index < authoritative_count; ++consumer_index) {
        const Stage2GraphConsumer authoritative =
          state.graph_consumers[consumer_index];
        if (!storage_node_pointer_addressable(authoritative.pointer) ||
            local_shard(authoritative.pointer.memory_node())) {
          continue;
        }
        const size_t cached =
          state.graph_prefetch_size(authoritative.search_index);
        if (cached >= cache_capacity) continue;
        const size_t per_search_limit = std::min<size_t>(
          static_cast<size_t>(issue_width - 1),
          cache_capacity - cached);
        if (per_search_limit == 0) continue;
        prefetch_candidates.clear();
        state.continuation.append_expand_prefetch_candidates(
          authoritative.search_index, construction_width,
          prefetch_candidates);
        size_t admitted = 0;
        for (const RemotePtr pointer : prefetch_candidates) {
          if (admitted == per_search_limit) break;
          if (!storage_node_pointer_addressable(pointer) ||
              local_shard(pointer.memory_node())) continue;
          const u32 peer = pointer.memory_node();
          // Do not turn previewing into a new transport dependency. The
          // per-peer exact combiner can relax this restriction later; today
          // an authoritative item in this wave must already own the RPC.
          if (wire_items_by_peer[peer] == 0 ||
              wire_items_by_peer[peer] >= config.storage_owner_batch_max) {
            continue;
          }
          if (state.graph_prefetch_contains(
                authoritative.search_index, pointer)) {
            continue;
          }
          state.graph_consumers.push_back({
            authoritative.search_index, authoritative.generation, pointer,
            true});
          ++wire_items_by_peer[peer];
          ++admitted;
          ++speculative_added;
        }
      }
    }
    if (speculative_added != 0) {
      storage_owner_stage2_graph_prefetch_issued_.fetch_add(
        speculative_added, std::memory_order_relaxed);
    }
    if (selected_any) {
      state.round_robin_search = (last_search + 1) % search_count;
    }
    return selected_any;
  };

  const auto prepare_score_dispatch = [&] {
    if (!collect_score_dispatch()) return std::pair{false, false};
    state.score_order.resize(state.score_consumers.size());
    std::iota(state.score_order.begin(), state.score_order.end(), size_t{0});
    std::sort(state.score_order.begin(), state.score_order.end(),
              [&](size_t lhs, size_t rhs) {
                const Stage2ScoreConsumer& left =
                  state.score_consumers[lhs];
                const Stage2ScoreConsumer& right =
                  state.score_consumers[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                if (left.search_index != right.search_index) {
                  return left.search_index < right.search_index;
                }
                return left.generation < right.generation;
              });
    state.score_unique.clear();
    state.score_group_offsets.clear();
    for (size_t position = 0; position < state.score_order.size();
         ++position) {
      const RemotePtr pointer =
        state.score_consumers[state.score_order[position]].pointer;
      if (state.score_unique.empty() ||
          state.score_unique.back() != pointer) {
        state.score_unique.push_back(pointer);
        state.score_group_offsets.push_back(position);
      }
    }
    state.score_group_offsets.push_back(state.score_order.size());
    state.pending_vectors.clear();
    const size_t snapshot_stride = aligned_snapshot_bytes();
    const size_t validation_offset =
      memory_node_detail::storage_owner_snapshot_validation_offset();
    u32 remote_slot = 0;
    bool progressed = false;
    for (size_t group = 0; group < state.score_unique.size(); ++group) {
      const RemotePtr pointer = state.score_unique[group];
      if (!storage_node_pointer_addressable(pointer)) {
        if (!pointer.is_null()) {
          report_rejected_graph_pointer(
            "stage2_async_score/input", pointer, RemotePtr{}, group);
        }
        progressed |= resolve_terminal_score_group(group) != 0;
        continue;
      }
      if (local_shard(pointer.memory_node())) {
        NodeSnapshot snapshot;
        if (read_node_snapshot(pointer, snapshot)) {
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              pointer, snapshot.header, snapshot.header,
              snapshot.slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable &&
              snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
            progressed |= resolve_score_group(
              group, snapshot.vector_data.data()) != 0;
          } else if (disposition == StableNodeSnapshotState::terminal) {
            progressed |= resolve_terminal_score_group(group) != 0;
          }
          continue;
        }
        const u64 before = load_local_node_header_acquire(pointer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (classify_construction_node_snapshot(
              pointer, before, after, slot_incarnation) ==
            StableNodeSnapshotState::terminal) {
          progressed |= resolve_terminal_score_group(group) != 0;
        }
        continue;
      }

      if (!state.score_many_dispatch &&
          VamanaNode::immutable_base_record(pointer)) {
        const size_t scratch_offset =
          static_cast<size_t>(remote_slot++) * snapshot_stride;
        lib_assert(scratch_offset + validation_offset +
                     VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                   "Stage2 base-vector dispatch exceeded lane scratch");
        byte_t* buffer = thread->coroutine_scratch(scratch_offset);
        state.pending_vectors.push_back(Stage2PendingVectorRead{
          .group_index = group,
          .pointer = pointer,
          .buffer = buffer,
          .after_header = buffer + validation_offset,
          .requires_after_header = false,
        });
      }
      // Mutable/recyclable records are scored at their physical home below.
      // Their home RPC retains the incarnation validation, while immutable
      // base records above need only one authoritative one-sided READ.
    }

    // Resolve the cheap immutable wave first.  Dynamic requests remain in the
    // continuation and are selected by the next finite dispatch; this avoids
    // imposing an all-RPC barrier on base candidates and keeps one CQ wave
    // bounded by the existing credit calculation.
    if (!state.pending_vectors.empty()) {
      storage_owner_stage2_vector_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_vector_unique_reads_.fetch_add(
        state.pending_vectors.size(), std::memory_order_relaxed);
      state.ordered_snapshot_pairs = mixed_snapshots;
      state.score_phase = Stage2ScoreIoPhase::body_ready;
      return std::pair{progressed, true};
    }

    state.score_home_rpc_count = 0;
    for (Stage2HomeExpandRpc& rpc : state.score_home_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.deadline_ns = 0;
      rpc.request.clear();
    }
    thread_local vec<vec<size_t>> consumers_by_shard;
    consumers_by_shard.clear();
    consumers_by_shard.resize(num_storage_nodes_);
    for (size_t consumer_index = 0;
         consumer_index < state.score_consumers.size(); ++consumer_index) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[consumer_index];
      if (storage_node_pointer_addressable(consumer.pointer) &&
          !local_shard(consumer.pointer.memory_node()) &&
          (state.score_many_dispatch ||
           !VamanaNode::immutable_base_record(consumer.pointer))) {
        consumers_by_shard[consumer.pointer.memory_node()].push_back(
          consumer_index);
      }
    }
    const size_t rpc_item_limit = state.score_many_dispatch
      ? std::max<size_t>(1, config.storage_owner_search_snapshot_batch)
      : std::max<size_t>(1, config.storage_owner_batch_max);
    thread_local vec<size_t> score_many_query_searches;
    thread_local vec<u32> score_many_query_indexes;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const vec<size_t>& shard_consumers = consumers_by_shard[shard];
      for (size_t begin = 0; begin < shard_consumers.size();
           begin += rpc_item_limit) {
        const u32 item_count = static_cast<u32>(std::min(
          rpc_item_limit, shard_consumers.size() - begin));
        if (state.score_home_rpc_count == state.score_home_rpcs.size()) {
          state.score_home_rpcs.emplace_back();
        }
        Stage2HomeExpandRpc& rpc =
          state.score_home_rpcs[state.score_home_rpc_count++];
        rpc.target_shard = shard;
        rpc.item_count = item_count;
        rpc.request_id = allocate_peer_request_id();
        if (state.score_many_dispatch) {
          score_many_query_searches.clear();
          score_many_query_indexes.assign(
            tasks.size(), std::numeric_limits<u32>::max());
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            lib_assert(consumer.search_index < tasks.size(),
                       "Stage2 score-many consumer escaped its context");
            u32& query_index =
              score_many_query_indexes[consumer.search_index];
            if (query_index == std::numeric_limits<u32>::max()) {
              query_index = static_cast<u32>(
                score_many_query_searches.size());
              score_many_query_searches.push_back(consumer.search_index);
            }
          }
          const u32 query_count = static_cast<u32>(
            score_many_query_searches.size());
          rpc.request.resize(
            service::storage_owner::stage2_score_many_request_bytes(
              item_count, query_count));
          std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
          auto* own_header =
            service::storage_owner::stage2_score_many_header(
              rpc.request.data());
          own_header->query_count = query_count;
          auto* items = service::storage_owner::stage2_score_many_items(
            rpc.request.data());
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            items[item] = service::storage_owner::Stage2ScoreManyItem{
              .pointer_raw = consumer.pointer.raw_address,
              .generation = consumer.generation,
              .search_index = static_cast<u32>(consumer.search_index),
              .query_index =
                score_many_query_indexes[consumer.search_index],
            };
          }
          byte_t* queries =
            service::storage_owner::stage2_score_many_queries(
              rpc.request.data(), item_count);
          for (u32 query_index = 0; query_index < query_count;
               ++query_index) {
            const size_t search_index =
              score_many_query_searches[query_index];
            std::memcpy(
              queries + static_cast<size_t>(query_index) *
                VamanaNode::vector_bytes(),
              targets[search_index].vector_data.data(),
              VamanaNode::vector_bytes());
          }
        } else {
          rpc.request.resize(
            service::storage_owner::stage2_expand_score_request_bytes(
              item_count));
          std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
          auto* items = service::storage_owner::stage2_expand_score_items(
            rpc.request.data());
          byte_t* queries =
            service::storage_owner::stage2_expand_score_queries(
              rpc.request.data(), item_count);
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            items[item] = service::storage_owner::Stage2ExpandScoreItem{
              .pointer_raw = consumer.pointer.raw_address,
              .generation = consumer.generation,
              .search_index = static_cast<u32>(consumer.search_index),
              .operation = static_cast<u32>(
                service::storage_owner::Stage2HomeOperation::score_only),
            };
            std::memcpy(
              queries + static_cast<size_t>(item) *
                VamanaNode::vector_bytes(),
              targets[consumer.search_index].vector_data.data(),
              VamanaNode::vector_bytes());
          }
        }
      }
    }
    if (state.score_home_rpc_count != 0) {
      u64 rpc_items = 0;
      for (size_t rpc_index = 0;
           rpc_index < state.score_home_rpc_count; ++rpc_index) {
        const Stage2HomeExpandRpc& rpc = state.score_home_rpcs[rpc_index];
        rpc_items += rpc.item_count;
      }
      storage_owner_stage2_vector_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_vector_unique_reads_.fetch_add(
        rpc_items, std::memory_order_relaxed);
      state.score_phase = Stage2ScoreIoPhase::home_pending;
    }
    return std::pair{progressed, state.score_home_rpc_count != 0};
  };

  const auto prepare_graph_dispatch = [&] {
    if (!collect_graph_dispatch()) return std::pair{false, false};
    state.graph_order.resize(state.graph_consumers.size());
    std::iota(state.graph_order.begin(), state.graph_order.end(), size_t{0});
    std::sort(state.graph_order.begin(), state.graph_order.end(),
              [&](size_t lhs, size_t rhs) {
                const Stage2GraphConsumer& left =
                  state.graph_consumers[lhs];
                const Stage2GraphConsumer& right =
                  state.graph_consumers[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                if (left.search_index != right.search_index) {
                  return left.search_index < right.search_index;
                }
                return left.generation < right.generation;
              });
    state.graph_unique.clear();
    state.graph_group_offsets.clear();
    for (size_t position = 0; position < state.graph_order.size();
         ++position) {
      const RemotePtr pointer =
        state.graph_consumers[state.graph_order[position]].pointer;
      if (state.graph_unique.empty() ||
          state.graph_unique.back() != pointer) {
        state.graph_unique.push_back(pointer);
        state.graph_group_offsets.push_back(position);
      }
    }
    state.graph_group_offsets.push_back(state.graph_order.size());
    if (state.graph_neighbors.size() < state.graph_unique.size()) {
      state.graph_neighbors.resize(state.graph_unique.size());
    }
    for (size_t group = 0; group < state.graph_unique.size(); ++group) {
      state.graph_neighbors[group].clear();
    }
    bool progressed = false;
    for (size_t group = 0; group < state.graph_unique.size(); ++group) {
      const RemotePtr pointer = state.graph_unique[group];
      if (!storage_node_pointer_addressable(pointer)) {
        if (!pointer.is_null()) {
          report_rejected_graph_pointer(
            "stage2_async_graph/input", pointer, RemotePtr{}, group);
        }
        progressed |= resolve_graph_group(
          group, span<const RemotePtr>{}) != 0;
        continue;
      }
      if (local_shard(pointer.memory_node())) {
        GraphAdjacency adjacency;
        if (read_graph_adjacency(pointer, adjacency)) {
          vec<RemotePtr>& neighbors = state.graph_neighbors[group];
          if (!adjacency.deleted) {
            neighbors.insert(neighbors.end(), adjacency.stable.begin(),
                             adjacency.stable.end());
          }
          progressed |= resolve_graph_group(
            group, span<const RemotePtr>{neighbors}) != 0;
          continue;
        }
        const u64 before = load_local_node_header_acquire(pointer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (classify_construction_node_snapshot(
              pointer, before, after, slot_incarnation) ==
            StableNodeSnapshotState::terminal) {
          progressed |= resolve_graph_group(
            group, span<const RemotePtr>{}) != 0;
        }
        continue;
      }

      // Remote expansion is executed at the pointer's physical home below.
      // Do not consume one-sided graph scratch here: the home response also
      // carries scores for same-home neighbors and removes the following
      // vector-read round trips without changing the beam's chosen pointer.
    }
    state.home_expand_rpc_count = 0;
    for (Stage2HomeExpandRpc& rpc : state.home_expand_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.deadline_ns = 0;
      rpc.request.clear();
      rpc.graph_consumer_indexes.clear();
    }
    thread_local vec<u32> home_counts;
    thread_local vec<size_t> rpc_by_shard;
    home_counts.assign(num_storage_nodes_, 0);
    rpc_by_shard.assign(num_storage_nodes_, std::numeric_limits<size_t>::max());
    for (const Stage2GraphConsumer& consumer : state.graph_consumers) {
      if (storage_node_pointer_addressable(consumer.pointer) &&
          !local_shard(consumer.pointer.memory_node())) {
        ++home_counts[consumer.pointer.memory_node()];
      }
    }
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (home_counts[shard] == 0) continue;
      if (state.home_expand_rpc_count == state.home_expand_rpcs.size()) {
        state.home_expand_rpcs.emplace_back();
      }
      Stage2HomeExpandRpc& rpc =
        state.home_expand_rpcs[state.home_expand_rpc_count];
      rpc.target_shard = shard;
      rpc.item_count = home_counts[shard];
      rpc.request_id = allocate_peer_request_id();
      rpc.request.resize(
        service::storage_owner::stage2_expand_score_request_bytes(
          rpc.item_count));
      rpc.graph_consumer_indexes.resize(rpc.item_count);
      std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
      rpc_by_shard[shard] = state.home_expand_rpc_count++;
    }
    home_counts.assign(num_storage_nodes_, 0);
    for (size_t consumer_index = 0;
         consumer_index < state.graph_consumers.size(); ++consumer_index) {
      const Stage2GraphConsumer& consumer =
        state.graph_consumers[consumer_index];
      if (!storage_node_pointer_addressable(consumer.pointer) ||
          local_shard(consumer.pointer.memory_node())) {
        continue;
      }
      const u32 shard = consumer.pointer.memory_node();
      Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_by_shard[shard]];
      const u32 item_index = home_counts[shard]++;
      rpc.graph_consumer_indexes[item_index] = consumer_index;
      auto* items = service::storage_owner::stage2_expand_score_items(
        rpc.request.data());
      items[item_index] = service::storage_owner::Stage2ExpandScoreItem{
        .pointer_raw = consumer.pointer.raw_address,
        .generation = consumer.generation,
        .search_index = static_cast<u32>(consumer.search_index),
      };
      byte_t* queries = service::storage_owner::stage2_expand_score_queries(
        rpc.request.data(), rpc.item_count);
      std::memcpy(
        queries + static_cast<size_t>(item_index) *
          VamanaNode::vector_bytes(),
        targets[consumer.search_index].vector_data.data(),
        VamanaNode::vector_bytes());
    }
    if (state.home_expand_rpc_count != 0) {
      storage_owner_stage2_graph_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_graph_unique_reads_.fetch_add(
        state.graph_consumers.size(), std::memory_order_relaxed);
      state.graph_phase = Stage2GraphIoPhase::home_pending;
    }
    return std::pair{progressed, state.home_expand_rpc_count != 0};
  };

  thread_local vec<PeerReadRequest> read_requests;
  thread_local vec<PeerReadSnapshotRequest> snapshot_requests;
  thread_local vec<byte_t> home_response_payload;
  thread_local vec<RemotePtr> home_neighbors;

  // Authoritative Stage2-home work is transported through the node-wide
  // combiner. The context keeps its original logical request ID and consumes
  // an ordinary logical response; only the wire transaction is shared across
  // contexts. No timer is introduced here and no search work is added.
  const auto enqueue_authoritative_home_rpc = [&] (
      const Stage2HomeExpandRpc& rpc,
      service::storage_owner::PeerRpcType request_type) {
    const auto owner = current_storage_owner_maintenance_context_owner();
    lib_assert(owner.runtime_epoch != 0 && owner.token != 0,
               "Stage2 home RPC has no generation-fenced context owner");
    const auto result = try_enqueue_stage2_home_rpc_request(
      rpc.target_shard, request_type, rpc.request_id, rpc.item_count,
      span<const byte_t>{rpc.request.data(), rpc.request.size()},
      PeerResponseCompletionTarget{.context_owner = owner});
    return result == Stage2HomeRpcEnqueueResult::enqueued ||
      result == Stage2HomeRpcEnqueueResult::duplicate;
  };

  const auto commit_prefetched_graph = [&] {
    u64 hits = 0;
    u64 wasted = 0;
    u64 scored = 0;
    u64 home_scored = 0;
    bool progressed = false;
    for (size_t search_index = 0; search_index < tasks.size();
         ++search_index) {
      if (state.search_seeded[search_index] == 0) continue;
      const auto request =
        state.continuation.pending_expand_request(search_index);
      if (!request.has_value()) continue;
      auto cached = state.take_graph_prefetch(
        search_index, request->pointer);
      if (!cached.has_value()) continue;
      const auto disposition = static_cast<
        service::storage_owner::Stage2HomeDisposition>(
          cached->disposition);
      if (disposition !=
            service::storage_owner::Stage2HomeDisposition::stable &&
          disposition !=
            service::storage_owner::Stage2HomeDisposition::terminal) {
        ++wasted;
        continue;
      }
      home_neighbors.clear();
      if (disposition ==
          service::storage_owner::Stage2HomeDisposition::stable) {
        home_neighbors.reserve(cached->neighbors.size());
        for (const Stage2PrefetchedGraphNeighbor& neighbor :
             cached->neighbors) {
          home_neighbors.push_back(neighbor.pointer);
        }
      }
      if (!state.continuation.resolve_expand_request(
            search_index, request->generation, request->pointer,
            span<const RemotePtr>{home_neighbors})) {
        ++wasted;
        continue;
      }
      ++hits;
      progressed = true;
      const u64 score_generation =
        state.continuation.generation(search_index);
      for (const Stage2PrefetchedGraphNeighbor& neighbor :
           cached->neighbors) {
        const auto neighbor_disposition = static_cast<
          service::storage_owner::Stage2HomeDisposition>(
            neighbor.disposition);
        bool resolved = false;
        if (neighbor_disposition ==
            service::storage_owner::Stage2HomeDisposition::stable) {
          resolved = state.continuation.resolve_score_request(
            search_index, score_generation, neighbor.pointer,
            std::optional<distance_t>{neighbor.distance});
          home_scored += resolved;
        } else if (neighbor_disposition ==
                   service::storage_owner::Stage2HomeDisposition::terminal) {
          resolved = state.continuation.resolve_score_request(
            search_index, score_generation, neighbor.pointer,
            std::nullopt);
        }
        scored += resolved;
      }
    }
    if (hits != 0) {
      storage_owner_stage2_graph_prefetch_hits_.fetch_add(
        hits, std::memory_order_relaxed);
    }
    if (wasted != 0) {
      storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
        wasted, std::memory_order_relaxed);
    }
    if (scored != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        scored, std::memory_order_relaxed);
    }
    if (home_scored != 0) {
      storage_owner_stage2_home_scored_neighbors_.fetch_add(
        home_scored, std::memory_order_relaxed);
    }
    return progressed;
  };
  const auto record_home_plane_wait = [&](bool score_plane) {
    u64& started_ns = score_plane
      ? state.score_home_wait_started_ns : state.graph_home_wait_started_ns;
    if (started_ns == 0) return;
    auto& total_ns = score_plane
      ? storage_owner_stage2_score_home_wait_ns_
      : storage_owner_stage2_graph_home_wait_ns_;
    total_ns.fetch_add(
      steady_now_ns() - started_ns, std::memory_order_relaxed);
  };
  const auto note_dual_plane_inflight = [&] {
    if (state.dual_plane_wait_recorded ||
        state.score_phase != Stage2ScoreIoPhase::home_pending ||
        state.graph_phase != Stage2GraphIoPhase::home_pending) {
      return;
    }
    const auto any_posted = [](const vec<Stage2HomeExpandRpc>& rpcs,
                               size_t count) {
      return std::any_of(rpcs.begin(), rpcs.begin() + count,
                         [](const Stage2HomeExpandRpc& rpc) {
                           return rpc.posted && !rpc.complete;
                         });
    };
    if (!any_posted(state.score_home_rpcs, state.score_home_rpc_count) ||
        !any_posted(state.home_expand_rpcs, state.home_expand_rpc_count)) {
      return;
    }
    state.dual_plane_wait_recorded = true;
    storage_owner_stage2_dual_plane_waits_.fetch_add(
      1, std::memory_order_relaxed);
  };
  u8 idle_attempt_mask = 0;

  // A direct-read dispatch is synchronized on its lane CQ. Independent home
  // RPC planes instead retire through their own logical response leases.
  // Only a bounded transport dispatch is synchronized on one CQ.  Once its
  // WRs retire, every stable/terminal consumer is resolved immediately and
  // retryable consumers simply remain in their own search generation.
  for (;;) {
    if (state.continuation.all_complete()) {
      const auto& final_beams = state.continuation.results();
      const auto& exhausted =
        state.continuation.budget_exhausted_results();
      const auto& expansions =
        state.continuation.expansion_count_results();
      u64 total_expansions = 0;
      u64 exhausted_count = 0;
      lib_assert(final_beams.size() == tasks.size(),
                 "asynchronous Stage2 result count changed");
      for (size_t item = 0; item < final_beams.size(); ++item) {
        total_expansions += expansions[item];
        exhausted_count += exhausted[item];
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
        exhausted_count, std::memory_order_relaxed);
      const u64 unused_prefetch = state.graph_prefetch_entry_count();
      if (unused_prefetch != 0) {
        storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
          unused_prefetch, std::memory_order_relaxed);
      }
      const size_t retained_capacity = std::max<size_t>(
        1024, static_cast<size_t>(construction_width) * 8);
      state.reset_completed(retained_capacity);
      return Stage2SearchAdvanceResult::complete;
    }

    // pending_expand_request() is a per-search dependency fence: a search is
    // in either score or expand phase, never both. Do not gate cache commit on
    // every other search in this batched context having no pending scores.
    // That global gate caused 130809 to refetch an authoritative graph for A
    // while unrelated B was scoring, wasting every ordered cache entry and
    // disabling the controller during warm-up.
    if (state.graph_phase == Stage2GraphIoPhase::idle &&
        commit_prefetched_graph()) {
      idle_attempt_mask = 0;
      continue;
    }

    if (state.score_phase == Stage2ScoreIoPhase::body_pending) {
      if (!thread->is_ready(thread->running_coroutine)) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      if (state.ordered_snapshot_pairs) {
        for (Stage2PendingVectorRead& read : state.pending_vectors) {
          const u64 before = *reinterpret_cast<const u64*>(read.buffer);
          const u32 slot_incarnation = *reinterpret_cast<const u32*>(
            read.buffer + VamanaNode::offset_slot_incarnation());
          const u64 after = read.requires_after_header
            ? *reinterpret_cast<const u64*>(read.after_header)
            : before;
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              read.pointer, before, after, slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable) {
            resolve_score_group(
              read.group_index,
              read.buffer + VamanaNode::offset_vector());
          } else if (disposition == StableNodeSnapshotState::terminal) {
            resolve_terminal_score_group(read.group_index);
          } else {
            lib_assert(read.attempt != std::numeric_limits<u32>::max(),
                       "Stage2 vector retry counter overflow");
          }
        }
        clear_score_dispatch();
        idle_attempt_mask = 0;
        continue;
      }
      size_t dynamic_count = 0;
      for (Stage2PendingVectorRead& read : state.pending_vectors) {
        read.before = *reinterpret_cast<const u64*>(read.buffer);
        read.slot_incarnation = *reinterpret_cast<const u32*>(
          read.buffer + VamanaNode::offset_slot_incarnation());
        if (!read.requires_after_header) {
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              read.pointer, read.before, read.before,
              read.slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable) {
            resolve_score_group(
              read.group_index,
              read.buffer + VamanaNode::offset_vector());
          } else if (disposition == StableNodeSnapshotState::terminal) {
            resolve_terminal_score_group(read.group_index);
          }
          continue;
        }
        if (dynamic_count !=
            static_cast<size_t>(&read - state.pending_vectors.data())) {
          state.pending_vectors[dynamic_count] = read;
        }
        ++dynamic_count;
      }
      state.pending_vectors.resize(dynamic_count);
      if (state.pending_vectors.empty()) {
        clear_score_dispatch();
        idle_attempt_mask = 0;
        continue;
      }
      state.score_phase = Stage2ScoreIoPhase::header_ready;
      continue;
    }

    if (state.score_phase == Stage2ScoreIoPhase::header_pending) {
      if (!thread->is_ready(thread->running_coroutine)) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      for (Stage2PendingVectorRead& read : state.pending_vectors) {
        const u64 after = *reinterpret_cast<const u64*>(read.buffer);
        const StableNodeSnapshotState disposition = classify_vector_snapshot(
          read.pointer, read.before, after, read.slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          resolve_score_group(
            read.group_index,
            read.buffer + VamanaNode::offset_vector());
        } else if (disposition == StableNodeSnapshotState::terminal) {
          resolve_terminal_score_group(read.group_index);
        } else {
          lib_assert(read.attempt != std::numeric_limits<u32>::max(),
                     "Stage2 vector retry counter overflow");
        }
      }
      clear_score_dispatch();
      idle_attempt_mask = 0;
      continue;
    }

    if (state.score_phase == Stage2ScoreIoPhase::header_ready) {
      read_requests.clear();
      for (const Stage2PendingVectorRead& read : state.pending_vectors) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = read.pointer.memory_node(),
          .remote_offset = read.pointer.byte_offset(),
          .destination = read.buffer,
          .bytes = VamanaNode::HEADER_SIZE,
        });
      }
      lib_assert(read_requests.size() <=
                   ordinary_dispatch_limits.global_items,
                 "Stage2 header dispatch exceeds RDMA credit");
      if (!try_post_peer_reads_async(
            *thread, span<const PeerReadRequest>{read_requests})) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      state.score_phase = Stage2ScoreIoPhase::header_pending;
      return Stage2SearchAdvanceResult::posted_rdma;
    }

    if (state.score_phase == Stage2ScoreIoPhase::body_ready) {
      if (state.ordered_snapshot_pairs) {
        snapshot_requests.clear();
        for (const Stage2PendingVectorRead& read : state.pending_vectors) {
          const PeerReadRequest body{
            .shard_id = read.pointer.memory_node(),
            .remote_offset = read.pointer.byte_offset(),
            .destination = read.buffer,
            .bytes = VamanaNode::size_until_vector_end(),
          };
          snapshot_requests.push_back(PeerReadSnapshotRequest{
            .full_snapshot = body,
            .after_header = read.requires_after_header
              ? std::optional<PeerReadRequest>{PeerReadRequest{
                  .shard_id = read.pointer.memory_node(),
                  .remote_offset = read.pointer.byte_offset(),
                  .destination = read.after_header,
                  .bytes = VamanaNode::HEADER_SIZE,
                }}
              : std::nullopt,
          });
        }
        if (!try_post_peer_snapshot_reads_async(
              *thread,
              span<const PeerReadSnapshotRequest>{snapshot_requests})) {
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
      } else {
        read_requests.clear();
        for (const Stage2PendingVectorRead& read : state.pending_vectors) {
          read_requests.push_back(PeerReadRequest{
            .shard_id = read.pointer.memory_node(),
            .remote_offset = read.pointer.byte_offset(),
            .destination = read.buffer,
            .bytes = VamanaNode::size_until_vector_end(),
          });
        }
        lib_assert(read_requests.size() <=
                     ordinary_dispatch_limits.global_items,
                   "Stage2 vector dispatch exceeds RDMA credit");
        if (!try_post_peer_reads_async(
              *thread, span<const PeerReadRequest>{read_requests})) {
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
      }
      state.score_phase = Stage2ScoreIoPhase::body_pending;
      return Stage2SearchAdvanceResult::posted_rdma;
    }

    if (state.score_phase == Stage2ScoreIoPhase::home_pending) {
      bool all_complete = true;
      for (size_t rpc_index = 0;
           rpc_index < state.score_home_rpc_count; ++rpc_index) {
        Stage2HomeExpandRpc& rpc = state.score_home_rpcs[rpc_index];
        if (rpc.complete) continue;
        all_complete = false;
        if (!rpc.posted) {
          const auto request_type = state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_request
            : service::storage_owner::PeerRpcType::stage2_expand_score_request;
          rpc.posted = enqueue_authoritative_home_rpc(rpc, request_type);
          if (rpc.posted) {
            if (state.score_home_wait_started_ns == 0) {
              state.score_home_wait_started_ns = steady_now_ns();
              storage_owner_stage2_score_home_dispatches_.fetch_add(
                1, std::memory_order_relaxed);
            }
            note_dual_plane_inflight();
            rpc.deadline_ns = steady_now_ns() +
              static_cast<u64>(config.storage_owner_rpc_timeout_ms) *
                1000ull * 1000ull;
          }
          if (!rpc.posted) continue;
        }

        service::storage_owner::PeerRpcHeader response_header{};
        PeerResponseLease response_lease{};
        home_response_payload.clear();
        const TryPeerResponse response = try_consume_peer_rpc_response(
          rpc.request_id, rpc.target_shard,
          state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_response
            : service::storage_owner::PeerRpcType::stage2_expand_score_response,
          rpc.item_count, response_header, home_response_payload,
          response_lease);
        if (response == TryPeerResponse::pending) {
          // The outer aggregate owns same-wire-ID timeout/retry. Cancelling a
          // logical member here would separate its response-registry cell
          // from the retained aggregate and defeat exact fan-out.
          continue;
        }
        const size_t expected_bytes = state.score_many_dispatch
          ? service::storage_owner::stage2_score_many_response_bytes(
              rpc.item_count)
          : service::storage_owner::stage2_expand_score_response_bytes(
              rpc.item_count, 0);
        const auto expected_response_type =
          state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_response
            : service::storage_owner::PeerRpcType::stage2_expand_score_response;
        bool valid = response == TryPeerResponse::success &&
          home_response_payload.size() == expected_bytes &&
          response_header.magic == service::storage_owner::kPeerRpcMagic &&
          response_header.version == service::storage_owner::kPeerRpcVersion &&
          response_header.type == static_cast<u32>(
            expected_response_type) &&
          response_header.source_shard == rpc.target_shard &&
          response_header.item_count == rpc.item_count &&
          response_header.request_id == rpc.request_id &&
          response_header.status == static_cast<u32>(
            service::storage_owner::InsertStatus::ok) &&
          response_header.reserved == 0;
        const auto* legacy_results = valid &&
            !state.score_many_dispatch
          ? service::storage_owner::stage2_expand_score_results(
              home_response_payload.data())
          : nullptr;
        const auto* score_many_results = valid &&
            state.score_many_dispatch
          ? service::storage_owner::stage2_score_many_results(
              home_response_payload.data())
          : nullptr;
        if (state.score_many_dispatch) {
          const auto* request_items =
            service::storage_owner::stage2_score_many_items(
              rpc.request.data());
          for (u32 item = 0; valid && item < rpc.item_count; ++item) {
            const auto& request = request_items[item];
            const auto& result = score_many_results[item];
            valid = result.pointer_raw == request.pointer_raw &&
              result.generation == request.generation &&
              result.search_index == request.search_index &&
              result.search_index < tasks.size() && result.reserved == 0 &&
              result.disposition <= static_cast<u32>(
                service::storage_owner::Stage2HomeDisposition::terminal);
          }
        } else {
          const auto* request_items =
            service::storage_owner::stage2_expand_score_items(
              rpc.request.data());
          for (u32 item = 0; valid && item < rpc.item_count; ++item) {
            const auto& request = request_items[item];
            const auto& result = legacy_results[item];
            valid = request.operation == static_cast<u32>(
                      service::storage_owner::Stage2HomeOperation::score_only) &&
              result.operation == request.operation &&
              result.pointer_raw == request.pointer_raw &&
              result.generation == request.generation &&
              result.search_index == request.search_index &&
              result.search_index < tasks.size() &&
              result.neighbor_count == 0 && result.neighbor_offset == 0 &&
              result.disposition <= static_cast<u32>(
                service::storage_owner::Stage2HomeDisposition::terminal);
          }
        }
        if (!valid) {
          if (response_lease.valid()) {
            lib_assert(rearm_peer_rpc_response(response_lease),
                       "invalid Stage2 score response lost its lease");
          }
          rpc.posted = false;
          continue;
        }
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated Stage2 score response lost its lease");
        for (u32 item = 0; item < rpc.item_count; ++item) {
          const u32 result_search_index =
            state.score_many_dispatch
              ? score_many_results[item].search_index
              : legacy_results[item].search_index;
          const u64 result_generation =
            state.score_many_dispatch
              ? score_many_results[item].generation
              : legacy_results[item].generation;
          const u64 result_pointer_raw =
            state.score_many_dispatch
              ? score_many_results[item].pointer_raw
              : legacy_results[item].pointer_raw;
          const u32 result_disposition =
            state.score_many_dispatch
              ? score_many_results[item].disposition
              : legacy_results[item].disposition;
          const distance_t result_distance =
            state.score_many_dispatch
              ? score_many_results[item].distance
              : legacy_results[item].distance;
          const auto disposition = static_cast<
            service::storage_owner::Stage2HomeDisposition>(
              result_disposition);
          bool resolved = false;
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::stable) {
            resolved = state.continuation.resolve_score_request(
              result_search_index, result_generation,
              RemotePtr{result_pointer_raw},
              std::optional<distance_t>{result_distance});
          } else if (disposition ==
                     service::storage_owner::Stage2HomeDisposition::terminal) {
            resolved = state.continuation.resolve_score_request(
              result_search_index, result_generation,
              RemotePtr{result_pointer_raw}, std::nullopt);
          }
          if (resolved) {
            storage_owner_stage2_scored_candidates_.fetch_add(
              1, std::memory_order_relaxed);
            storage_owner_stage2_home_scored_neighbors_.fetch_add(
              disposition ==
                  service::storage_owner::Stage2HomeDisposition::stable,
              std::memory_order_relaxed);
          }
        }
        rpc.complete = true;
      }
      all_complete = std::all_of(
        state.score_home_rpcs.begin(),
        state.score_home_rpcs.begin() + state.score_home_rpc_count,
        [](const Stage2HomeExpandRpc& rpc) { return rpc.complete; });
      if (all_complete) {
        record_home_plane_wait(true);
        clear_score_dispatch();
        idle_attempt_mask = 0;
        continue;
      }
    }

    if (state.graph_phase == Stage2GraphIoPhase::home_pending) {
      bool all_complete = true;
      for (size_t rpc_index = 0;
           rpc_index < state.home_expand_rpc_count; ++rpc_index) {
        Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_index];
        if (rpc.complete) continue;
        all_complete = false;
        if (!rpc.posted) {
          rpc.posted = enqueue_authoritative_home_rpc(
            rpc,
            service::storage_owner::PeerRpcType::stage2_expand_score_request);
          if (rpc.posted) {
            if (state.graph_home_wait_started_ns == 0) {
              state.graph_home_wait_started_ns = steady_now_ns();
              storage_owner_stage2_graph_home_dispatches_.fetch_add(
                1, std::memory_order_relaxed);
            }
            note_dual_plane_inflight();
            rpc.deadline_ns = steady_now_ns() +
              static_cast<u64>(config.storage_owner_rpc_timeout_ms) *
                1000ull * 1000ull;
          }
          if (!rpc.posted) continue;
        }

        service::storage_owner::PeerRpcHeader response_header{};
        PeerResponseLease response_lease{};
        home_response_payload.clear();
        const TryPeerResponse response = try_consume_peer_rpc_response(
          rpc.request_id, rpc.target_shard,
          service::storage_owner::PeerRpcType::stage2_expand_score_response,
          rpc.item_count, response_header, home_response_payload,
          response_lease);
        if (response == TryPeerResponse::pending) {
          // The combined outer request, not an individual logical member,
          // owns transport retry and its immutable wire image.
          continue;
        }
        const size_t minimum_bytes =
          service::storage_owner::stage2_expand_score_response_bytes(
            rpc.item_count, 0);
        const size_t maximum_bytes =
          service::storage_owner::stage2_expand_score_response_bytes(
            rpc.item_count);
        const size_t response_bytes = home_response_payload.size();
        const bool compact_size_valid = response_bytes >= minimum_bytes &&
          response_bytes <= maximum_bytes &&
          (response_bytes - minimum_bytes) %
              sizeof(service::storage_owner::Stage2ExpandScoreNeighbor) == 0;
        const size_t compact_neighbor_count = compact_size_valid
          ? (response_bytes - minimum_bytes) /
              sizeof(service::storage_owner::Stage2ExpandScoreNeighbor)
          : 0;
        bool valid = response == TryPeerResponse::success &&
          compact_size_valid &&
          response_header.magic == service::storage_owner::kPeerRpcMagic &&
          response_header.version == service::storage_owner::kPeerRpcVersion &&
          response_header.type == static_cast<u32>(
            service::storage_owner::PeerRpcType::stage2_expand_score_response) &&
          response_header.source_shard == rpc.target_shard &&
          response_header.item_count == rpc.item_count &&
          response_header.request_id == rpc.request_id &&
          response_header.status == static_cast<u32>(
            service::storage_owner::InsertStatus::ok) &&
          response_header.reserved == 0;
        const auto* request_items =
          service::storage_owner::stage2_expand_score_items(
            rpc.request.data());
        const auto* results = valid
          ? service::storage_owner::stage2_expand_score_results(
              home_response_payload.data())
          : nullptr;
        const auto* neighbors = valid
          ? service::storage_owner::stage2_expand_score_neighbors(
              home_response_payload.data(), rpc.item_count)
          : nullptr;
        const size_t neighbor_stride = VamanaNode::graph_entry_capacity();
        size_t expected_neighbor_offset = 0;
        for (u32 item_index = 0; valid && item_index < rpc.item_count;
             ++item_index) {
          const auto& request = request_items[item_index];
          const auto& result = results[item_index];
          valid = result.pointer_raw == request.pointer_raw &&
            result.generation == request.generation &&
            result.search_index == request.search_index &&
            result.search_index < tasks.size() &&
            request.operation == static_cast<u32>(
              service::storage_owner::Stage2HomeOperation::expand_score) &&
            result.operation == request.operation &&
            result.neighbor_count <= neighbor_stride &&
            result.neighbor_offset == expected_neighbor_offset &&
            result.neighbor_offset <= compact_neighbor_count &&
            result.neighbor_count <=
              compact_neighbor_count - result.neighbor_offset &&
            result.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::terminal);
          for (u32 neighbor_index = 0;
               valid && neighbor_index < result.neighbor_count;
               ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(result.neighbor_offset) + neighbor_index];
            valid = neighbor.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::unscored);
          }
          expected_neighbor_offset += result.neighbor_count;
        }
        valid = valid && expected_neighbor_offset == compact_neighbor_count;
        if (!valid) {
          if (response_lease.valid()) {
            lib_assert(rearm_peer_rpc_response(response_lease),
                       "invalid Stage2 home response lost its lease");
          }
          rpc.posted = false;
          continue;
        }

        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated Stage2 home response lost its lease");
        for (u32 item_index = 0; item_index < rpc.item_count; ++item_index) {
          const auto& result = results[item_index];
          lib_assert(item_index < rpc.graph_consumer_indexes.size() &&
                       rpc.graph_consumer_indexes[item_index] <
                         state.graph_consumers.size(),
                     "Stage2 graph response lost consumer metadata");
          const Stage2GraphConsumer& consumer = state.graph_consumers[
            rpc.graph_consumer_indexes[item_index]];
          const auto disposition = static_cast<
            service::storage_owner::Stage2HomeDisposition>(
              result.disposition);
          if (consumer.speculative) {
            bool retained = false;
            if (disposition !=
                service::storage_owner::Stage2HomeDisposition::retryable) {
              Stage2PrefetchedGraphExpansion cached{
                .pointer = RemotePtr{result.pointer_raw},
                .disposition = result.disposition,
                .neighbors = {},
              };
              cached.neighbors.reserve(result.neighbor_count);
              for (u32 neighbor_index = 0;
                   neighbor_index < result.neighbor_count;
                   ++neighbor_index) {
                const auto& neighbor = neighbors[
                  static_cast<size_t>(result.neighbor_offset) +
                  neighbor_index];
                cached.neighbors.push_back({
                  .pointer = RemotePtr{neighbor.pointer_raw},
                  .distance = neighbor.distance,
                  .disposition = neighbor.disposition,
                });
              }
              retained = state.insert_graph_prefetch(
                consumer.search_index, std::move(cached),
                static_cast<size_t>(
                  config.storage_owner_stage2_graph_issue_width));
            }
            if (!retained) {
              storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
                1, std::memory_order_relaxed);
            }
            continue;
          }
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::retryable) {
            continue;
          }
          home_neighbors.clear();
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::stable) {
            home_neighbors.reserve(result.neighbor_count);
            for (u32 neighbor_index = 0;
                 neighbor_index < result.neighbor_count; ++neighbor_index) {
              home_neighbors.push_back(RemotePtr{neighbors[
                static_cast<size_t>(result.neighbor_offset) +
                neighbor_index].pointer_raw});
            }
          }
          if (!state.continuation.resolve_expand_request(
                result.search_index, result.generation,
                RemotePtr{result.pointer_raw},
                span<const RemotePtr>{home_neighbors})) {
            continue;
          }
          const u64 score_generation =
            state.continuation.generation(result.search_index);
          for (u32 neighbor_index = 0;
               neighbor_index < result.neighbor_count; ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(result.neighbor_offset) + neighbor_index];
            const RemotePtr pointer{neighbor.pointer_raw};
            const auto neighbor_disposition = static_cast<
              service::storage_owner::Stage2HomeDisposition>(
                neighbor.disposition);
            bool resolved = false;
            if (neighbor_disposition ==
                service::storage_owner::Stage2HomeDisposition::stable) {
              resolved = state.continuation.resolve_score_request(
                result.search_index, score_generation, pointer,
                std::optional<distance_t>{neighbor.distance});
              if (resolved) {
                storage_owner_stage2_home_scored_neighbors_.fetch_add(
                  1, std::memory_order_relaxed);
              }
            } else if (neighbor_disposition ==
                       service::storage_owner::Stage2HomeDisposition::terminal) {
              resolved = state.continuation.resolve_score_request(
                result.search_index, score_generation, pointer,
                std::nullopt);
            }
            if (resolved) {
              storage_owner_stage2_scored_candidates_.fetch_add(
                1, std::memory_order_relaxed);
            }
          }
        }
        rpc.complete = true;
      }
      all_complete = std::all_of(
        state.home_expand_rpcs.begin(),
        state.home_expand_rpcs.begin() + state.home_expand_rpc_count,
        [](const Stage2HomeExpandRpc& rpc) { return rpc.complete; });
      if (all_complete) {
        record_home_plane_wait(false);
        clear_graph_dispatch();
        idle_attempt_mask = 0;
        continue;
      }
    }

    lib_assert(
      (state.score_phase == Stage2ScoreIoPhase::idle ||
       state.score_phase == Stage2ScoreIoPhase::home_pending) &&
      (state.graph_phase == Stage2GraphIoPhase::idle ||
       state.graph_phase == Stage2GraphIoPhase::home_pending),
      "Stage2 dispatcher entered an invalid I/O phase");
    const auto schedule = stage2_split_plane_schedule(
      !state.continuation.pending_score_requests().empty(),
      !state.continuation.pending_expand_requests().empty(),
      state.score_phase, state.graph_phase, idle_attempt_mask,
      config.storage_owner_stage2_split_plane, state.prefer_graph);
    if (!schedule.dispatch_graph && !schedule.dispatch_score) {
      // An in-flight response on the other plane is independent work, not a
      // reason to spin or to discard this plane's retry-only consumers.
      note_dual_plane_inflight();
      return Stage2SearchAdvanceResult::waiting_rdma;
    }

    std::pair<bool, bool> graph_prepared{false, false};
    std::pair<bool, bool> score_prepared{false, false};
    if (schedule.dispatch_graph) {
      idle_attempt_mask |= 2u;
      graph_prepared = prepare_graph_dispatch();
      if (!config.storage_owner_stage2_split_plane) state.prefer_graph = false;
      if (!graph_prepared.second) clear_graph_dispatch();
    }
    if (schedule.dispatch_score &&
        (!graph_prepared.second ||
         state.graph_phase == Stage2GraphIoPhase::home_pending)) {
      // Graph expansion feeds the next score generation. Prepare it first,
      // then fill an independent score plane without clearing graph state.
      // A defensive direct graph dispatch retains exclusive lane-CQ issue.
      idle_attempt_mask |= 1u;
      score_prepared = prepare_score_dispatch();
      if (!config.storage_owner_stage2_split_plane) state.prefer_graph = true;
      if (!score_prepared.second) clear_score_dispatch();
    }
    if (graph_prepared.second && score_prepared.second &&
        state.graph_phase == Stage2GraphIoPhase::home_pending &&
        state.score_phase == Stage2ScoreIoPhase::home_pending) {
      storage_owner_stage2_dual_plane_dispatches_.fetch_add(
        1, std::memory_order_relaxed);
    }
    if (graph_prepared.second || score_prepared.second) {
      continue;
    }
    if (graph_prepared.first || score_prepared.first) {
      idle_attempt_mask = 0;
    }
  }
}

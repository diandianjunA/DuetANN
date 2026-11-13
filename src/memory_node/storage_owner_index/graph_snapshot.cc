#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/locked_node_publication.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "gpu_search/graph_record_validation.hh"

using namespace memory_node_storage_owner_index_detail;

bool MemoryNode::read_graph_adjacency(RemotePtr rptr,
                                      GraphAdjacency& adjacency) {
  adjacency.stable.clear();
  adjacency.provisional.clear();
  adjacency.generation = 0;
  adjacency.deleted = false;
  if (!storage_node_pointer_addressable(rptr)) {
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("read_graph_adjacency/input", rptr);
    }
    return false;
  }
  thread_local vec<byte_t> local_entry;
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = nullptr;
  if (local_shard(rptr.memory_node())) {
    local_entry.resize(VamanaNode::hot_graph_entry_size());
    read_buffer = local_entry.data();
  } else {
    read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                    ? owner_thread->coroutine_scratch()
                    : peer_scratch_buffer_.get_full_buffer();
  }
  const u32 full_bytes = VamanaNode::hot_graph_entry_size();
  u32 transfer_bytes = full_bytes;
  const u32 maximum_attempts = graph_extent_enabled_ ? 4u : 3u;
  for (u32 attempt = 0; attempt < maximum_attempts; ++attempt) {
    if (local_shard(rptr.memory_node())) {
      transfer_bytes = full_bytes;
      std::memcpy(read_buffer,
                  index_buffer_.get_full_buffer() + VamanaNode::hot_graph_entry_offset(rptr),
                  full_bytes);
    } else {
      if (attempt == 0) {
        transfer_bytes = graph_extent_cache_.transport_bytes(
          rptr, full_bytes, VamanaNode::graph_entry_capacity(),
          graph_extent_enabled_);
      } else if (transfer_bytes < full_bytes) {
        graph_extent_cache_.note_fallback();
        transfer_bytes = full_bytes;
      }
      remote_read_bytes(rptr.memory_node(),
                        VamanaNode::hot_graph_entry_offset(rptr),
                        read_buffer, transfer_bytes, 0);
    }
    const bool partial = transfer_bytes < full_bytes;
    const auto snapshot = partial
      ? gpu_search::graph_record_validation::classify_zero_extended_snapshot(
          read_buffer, transfer_bytes, full_bytes, VamanaNode::R,
          VamanaNode::graph_entry_capacity(), rptr.incarnation())
      : gpu_search::graph_record_validation::classify_snapshot(
          read_buffer, full_bytes, VamanaNode::R,
          VamanaNode::graph_entry_capacity(), rptr.incarnation());
    if (snapshot != gpu_search::graph_record_validation::SnapshotState::valid) {
      std::this_thread::yield();
      continue;
    }
    const u8 stable_count = read_buffer[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(read_buffer);
    if (!local_shard(rptr.memory_node()) && graph_extent_enabled_) {
      u32 required_bytes = 0;
      lib_assert(gpu_search::graph_record_validation::required_live_extent_bytes(
                   read_buffer, transfer_bytes, VamanaNode::R,
                   VamanaNode::graph_entry_capacity(), required_bytes),
                 "validated graph snapshot has an invalid extent header");
      graph_extent_cache_.learn(
        rptr, required_bytes, VamanaNode::graph_entry_capacity());
    }

    adjacency.stable.clear();
    adjacency.provisional.clear();
    adjacency.generation = vamana::hot_graph::load_u32_le(read_buffer + 4);
    adjacency.deleted =
      (read_buffer[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
    adjacency.stable.reserve(stable_count);
    adjacency.provisional.reserve(provisional_count);
    bool malformed_neighbor = false;
    for (u32 index = 0; index < stable_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        read_buffer + vamana::hot_graph::neighbor_offset(index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        // The count is part of the checksummed publication. A null inside
        // that counted prefix is structural corruption, not unused capacity.
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/stable_null", neighbor, rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/stable", neighbor, rptr, index);
        continue;
      }
      adjacency.stable.push_back(neighbor);
    }
    for (u32 index = 0; index < provisional_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        read_buffer + vamana::hot_graph::neighbor_offset(stable_count + index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/provisional_null", neighbor, rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/provisional", neighbor, rptr, index);
        continue;
      }
      adjacency.provisional.push_back(neighbor);
    }
    // A malformed pointer can be a transient torn graph snapshot, so retry
    // optimistic reads. It must never be accepted after filtering: mutation
    // maintenance could otherwise acknowledge a graph with silently missing
    // authoritative edges. Repeated checksum-consistent damage is terminal.
    if (!malformed_neighbor) return true;
    if (attempt + 1 == maximum_attempts) {
      lib_failure(
        "persistent malformed counted graph edge: parent_raw=" +
        std::to_string(rptr.raw_address));
    }
    std::this_thread::yield();
  }
  adjacency.stable.clear();
  adjacency.provisional.clear();
  adjacency.generation = 0;
  adjacency.deleted = false;
  return false;
}

vec<std::pair<RemotePtr, MemoryNode::GraphAdjacency>>
MemoryNode::read_graph_adjacencies_batched(
    span<const RemotePtr> rptrs,
    const Configuration& config) {
  vec<std::pair<RemotePtr, GraphAdjacency>> results;
  const size_t result_count =
    read_graph_adjacencies_batched_into(rptrs, config, results);
  results.resize(result_count);
  return results;
}

size_t MemoryNode::read_graph_adjacencies_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<std::pair<RemotePtr, GraphAdjacency>>& results) {
  results.reserve(rptrs.size());
  size_t result_count = 0;
  if (rptrs.empty()) return result_count;

  const auto next_result = [&]() -> std::pair<RemotePtr, GraphAdjacency>& {
    if (result_count == results.size()) {
      results.emplace_back();
    }
    return results[result_count];
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (const RemotePtr rptr : rptrs) {
      auto& slot = next_result();
      if (read_graph_adjacency(rptr, slot.second)) {
        slot.first = rptr;
        ++result_count;
      }
    }
    return result_count;
  }

  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
    u32 transfer_bytes{};
  };

  enum class GraphDecodeResult : u8 {
    invalid_snapshot,
    valid,
    malformed_pointer,
  };
  const auto decode = [&](RemotePtr rptr, const byte_t* entry,
                          u32 transfer_bytes, GraphAdjacency& adjacency) {
    adjacency.stable.clear();
    adjacency.provisional.clear();
    adjacency.generation = 0;
    adjacency.deleted = false;
    const u32 full_bytes = VamanaNode::hot_graph_entry_size();
    const auto snapshot = transfer_bytes < full_bytes
      ? gpu_search::graph_record_validation::classify_zero_extended_snapshot(
          entry, transfer_bytes, full_bytes, VamanaNode::R,
          VamanaNode::graph_entry_capacity(), rptr.incarnation())
      : gpu_search::graph_record_validation::classify_snapshot(
          entry, full_bytes, VamanaNode::R,
          VamanaNode::graph_entry_capacity(), rptr.incarnation());
    if (snapshot != gpu_search::graph_record_validation::SnapshotState::valid) {
      return GraphDecodeResult::invalid_snapshot;
    }
    const u8 stable_count = entry[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(entry);
    adjacency.generation = vamana::hot_graph::load_u32_le(entry + 4);
    adjacency.deleted =
      (entry[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
    adjacency.stable.reserve(stable_count);
    adjacency.provisional.reserve(provisional_count);
    bool malformed_neighbor = false;
    for (u32 index = 0; index < stable_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        entry + vamana::hot_graph::neighbor_offset(index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/stable_null", neighbor, rptr,
          index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/stable", neighbor, rptr, index);
        continue;
      }
      adjacency.stable.push_back(neighbor);
    }
    for (u32 index = 0; index < provisional_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        entry + vamana::hot_graph::neighbor_offset(stable_count + index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/provisional_null", neighbor,
          rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/provisional", neighbor, rptr,
          index);
        continue;
      }
      adjacency.provisional.push_back(neighbor);
    }
    return malformed_neighbor ? GraphDecodeResult::malformed_pointer
                              : GraphDecodeResult::valid;
  };

  const size_t scratch_stride = aligned_graph_entry_bytes();
  const size_t max_batch = storage_owner_graph_batch_size(config, thread);
  const u32 maximum_attempts = graph_extent_enabled_ ? 4u : 3u;
  thread_local vec<PendingRead> pending;
  thread_local vec<PendingRead> retry;
  thread_local vec<PeerReadRequest> read_requests;
  pending.reserve(max_batch);
  retry.reserve(max_batch);
  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    u32 remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr rptr = rptrs[index];
      if (!storage_node_pointer_addressable(rptr)) {
        if (!rptr.is_null()) {
          report_rejected_graph_pointer(
            "read_graph_adjacencies_batched/input", rptr);
        }
        continue;
      }
      if (local_shard(rptr.memory_node())) {
        auto& slot = next_result();
        if (read_graph_adjacency(rptr, slot.second)) {
          slot.first = rptr;
          ++result_count;
        }
        continue;
      }
      const size_t scratch_offset =
        static_cast<size_t>(remote_slot++) * scratch_stride;
      lib_assert(scratch_offset + VamanaNode::hot_graph_entry_size() <=
                   thread->scratch_stride,
                 "storage-owner scratch cannot hold batched graph reads");
      pending.push_back({rptr, thread->coroutine_scratch(scratch_offset), 0});
    }

    for (u32 attempt = 0;
         attempt < maximum_attempts && !pending.empty(); ++attempt) {
      read_requests.clear();
      for (PendingRead& read : pending) {
        if (attempt == 0) {
          read.transfer_bytes = graph_extent_cache_.transport_bytes(
            read.rptr, VamanaNode::hot_graph_entry_size(),
            VamanaNode::graph_entry_capacity(), graph_extent_enabled_);
        }
        read_requests.push_back(PeerReadRequest{
          .shard_id = read.rptr.memory_node(),
          .remote_offset = VamanaNode::hot_graph_entry_offset(read.rptr),
          .destination = read.buffer,
          .bytes = read.transfer_bytes,
        });
      }
      post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }

      retry.clear();
      for (PendingRead& read : pending) {
        auto& slot = next_result();
        const GraphDecodeResult decoded = decode(
          read.rptr, read.buffer, read.transfer_bytes, slot.second);
        if (decoded == GraphDecodeResult::valid) {
          if (graph_extent_enabled_) {
            u32 required_bytes = 0;
            lib_assert(
              gpu_search::graph_record_validation::required_live_extent_bytes(
                read.buffer, read.transfer_bytes, VamanaNode::R,
                VamanaNode::graph_entry_capacity(), required_bytes),
              "validated batched graph snapshot has an invalid extent header");
            graph_extent_cache_.learn(
              read.rptr, required_bytes, VamanaNode::graph_entry_capacity());
          }
          slot.first = read.rptr;
          ++result_count;
        } else if (decoded == GraphDecodeResult::malformed_pointer &&
                   attempt + 1 == maximum_attempts) {
          lib_failure(
            "persistent malformed counted graph edge in batched read: "
            "parent_raw=" + std::to_string(read.rptr.raw_address));
        } else if (attempt + 1 < maximum_attempts) {
          if (read.transfer_bytes < VamanaNode::hot_graph_entry_size()) {
            graph_extent_cache_.note_fallback();
            read.transfer_bytes = VamanaNode::hot_graph_entry_size();
          }
          retry.push_back(read);
        }
      }
      pending.swap(retry);
    }
  }
  return result_count;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
  GraphAdjacency adjacency;
  if (!read_graph_adjacency(rptr, adjacency) || adjacency.deleted) {
    return {};
  }
  vec<RemotePtr> neighbors;
  neighbors.reserve(adjacency.stable.size() + adjacency.provisional.size());
  neighbors.insert(neighbors.end(), adjacency.stable.begin(),
                   adjacency.stable.end());
  neighbors.insert(neighbors.end(), adjacency.provisional.begin(),
                   adjacency.provisional.end());
  return neighbors;
}

vec<RemotePtr> MemoryNode::read_stable_neighbor_list(RemotePtr rptr) {
  GraphAdjacency adjacency;
  if (!read_graph_adjacency(rptr, adjacency) || adjacency.deleted) {
    return {};
  }
  return std::move(adjacency.stable);
}

bool MemoryNode::read_local_neighbor_list(RemotePtr rptr,
                                          vec<RemotePtr>& neighbors,
                                          vec<byte_t>& entry,
                                          vec<byte_t>& decoded) const {
  if (!local_shard(rptr.memory_node()) ||
      !storage_node_pointer_addressable(rptr)) {
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("read_local_neighbor_list", rptr);
    }
    neighbors.clear();
    return false;
  }
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  if (hot_offset > mn_memory_bytes_ ||
      entry_size > mn_memory_bytes_ - hot_offset) {
    report_rejected_graph_pointer("read_local_neighbor_list.hot", rptr);
    neighbors.clear();
    return false;
  }

  entry.resize(entry_size);
  decoded.resize(VamanaNode::neighbor_read_size());
  neighbors.clear();
  constexpr u32 kMaxReadAttempts = 3;
  bool decoded_ok = false;
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    std::memcpy(entry.data(), index_buffer_.get_full_buffer() + hot_offset,
                entry_size);
    decoded_ok = VamanaNode::decode_hot_graph_entry(
      entry.data(), decoded.data(), rptr.incarnation());
    if (decoded_ok) {
      break;
    }
    std::this_thread::yield();
  }
  if (!decoded_ok) {
    return false;
  }

  const u32 edge_count = VamanaNode::decoded_neighbor_count(decoded.data());
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  neighbors.reserve(edge_count);
  for (u32 index = 0; index < edge_count &&
                       index < VamanaNode::graph_entry_capacity(); ++index) {
    const RemotePtr neighbor = slots[index];
    if (neighbor.is_null()) continue;
    if (!storage_node_pointer_addressable(neighbor)) {
      report_rejected_graph_pointer(
        "read_local_neighbor_list", neighbor, rptr, index);
      continue;
    }
    neighbors.push_back(neighbor);
  }
  return true;
}

vec<MemoryNode::NodeSnapshot> MemoryNode::read_node_snapshots_batched(
    const vec<RemotePtr>& rptrs,
    const Configuration& config,
    const char* boundary,
    vec<StableNodeSnapshotState>* states) {
  vec<NodeSnapshot> snapshots;
  const size_t snapshot_count = read_node_snapshots_batched_into(
    span<const RemotePtr>{rptrs}, config, snapshots, boundary, states);
  snapshots.resize(snapshot_count);
  return snapshots;
}

size_t MemoryNode::read_node_snapshots_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<NodeSnapshot>& snapshots,
    const char* boundary,
    vec<StableNodeSnapshotState>* states) {
  snapshots.reserve(rptrs.size());
  size_t snapshot_count = 0;
  if (states != nullptr) {
    states->assign(rptrs.size(), StableNodeSnapshotState::terminal);
  }
  if (rptrs.empty()) {
    return snapshot_count;
  }

  const auto set_state = [&](size_t input_index,
                             StableNodeSnapshotState state) {
    if (states != nullptr) (*states)[input_index] = state;
  };

  // If a complete D-byte optimistic snapshot did not stabilize, a compact
  // identity observation distinguishes a stable replacement from transient
  // lock/torn contention. A current identity still means retryable here: the
  // caller requested the vector payload and must never receive a fabricated
  // or partially observed one.
  const auto classify_failed_snapshot = [&](size_t input_index,
                                            RemotePtr pointer) {
    vec<std::pair<RemotePtr, u64>> identity;
    vec<StableNodeSnapshotState> identity_state;
    read_node_identity_headers_batched_into(
      span<const RemotePtr>{&pointer, 1}, config, identity, &identity_state);
    lib_assert(identity_state.size() == 1,
               "single snapshot retry classification lost input alignment");
    set_state(
      input_index,
      identity_state.front() == StableNodeSnapshotState::stable
        ? StableNodeSnapshotState::retryable
        : identity_state.front());
  };

  const auto next_snapshot = [&]() -> NodeSnapshot& {
    if (snapshot_count == snapshots.size()) snapshots.emplace_back();
    return snapshots[snapshot_count];
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (size_t input_index = 0; input_index < rptrs.size(); ++input_index) {
      const RemotePtr rptr = rptrs[input_index];
      NodeSnapshot& snapshot = next_snapshot();
      if (!rptr.is_null() && read_node_snapshot(rptr, snapshot)) {
        set_state(input_index, StableNodeSnapshotState::stable);
        ++snapshot_count;
      } else if (!rptr.is_null() && storage_node_pointer_addressable(rptr)) {
        classify_failed_snapshot(input_index, rptr);
      }
    }
    return snapshot_count;
  }

  struct PendingRead {
    size_t input_index{};
    RemotePtr rptr;
    byte_t* buffer{};
    u64 before{};
    u32 slot_incarnation{};
  };

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);
  thread_local vec<PeerReadRequest> read_requests;

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    thread_local vec<PendingRead> pending;
    pending.reserve(end - begin);
    pending.clear();
    read_requests.clear();
    u32 remote_slot = 0;

    for (size_t idx = begin; idx < end; ++idx) {
      const RemotePtr& rptr = rptrs[idx];
      if (rptr.is_null()) {
        continue;
      }

      if (!storage_node_pointer_addressable(rptr)) {
        report_rejected_graph_pointer(boundary, rptr);
        continue;
      }

      if (!local_shard(rptr.memory_node())) {
        NodeSnapshot& snapshot = next_snapshot();
        if (reuse_reverse_snapshot(rptr, snapshot)) {
          set_state(idx, StableNodeSnapshotState::stable);
          ++snapshot_count;
          continue;
        }
      }

      if (local_shard(rptr.memory_node())) {
        NodeSnapshot& snapshot = next_snapshot();
        if (read_node_snapshot(rptr, snapshot)) {
          set_state(idx, StableNodeSnapshotState::stable);
          ++snapshot_count;
        } else {
          classify_failed_snapshot(idx, rptr);
        }
        continue;
      }

      const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + snapshot_size <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for snapshot batch: "
                 "offset=" + std::to_string(scratch_offset) +
                 " snapshot=" + std::to_string(snapshot_size) +
                 " stride=" + std::to_string(thread->scratch_stride) +
                 " remote_slot=" + std::to_string(remote_slot) +
                 " chunk=" + std::to_string(end - begin));
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      read_requests.push_back(PeerReadRequest{
        .shard_id = rptr.memory_node(),
        .remote_offset = rptr.byte_offset(),
        .destination = buffer,
        .bytes = VamanaNode::size_until_vector_end(),
      });
      pending.push_back(PendingRead{idx, rptr, buffer});
      ++remote_slot;
    }

    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    // A structurally verified base record has an immutable identity/vector
    // payload, so its first unlocked body header is already a linearization
    // point. Dynamic slots can be recycled while this READ crosses the body;
    // retain their second header read after saving the first identity fields.
    // Equality closes that overwrite window without another scratch plane.
    read_requests.clear();
    size_t dynamic_count = 0;
    for (PendingRead& read : pending) {
      read.before = *reinterpret_cast<const u64*>(read.buffer);
      read.slot_incarnation = *reinterpret_cast<const u32*>(
        read.buffer + VamanaNode::offset_slot_incarnation());
      if (VamanaNode::immutable_base_record(read.rptr)) {
        const StableNodeSnapshotState disposition =
          classify_physical_node_snapshot(
            read.rptr, read.before, read.before,
            read.slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          NodeSnapshot& snapshot = next_snapshot();
          lib_assert(parse_remote_snapshot(
                       read.rptr, read.buffer, snapshot),
                     "immutable base snapshot failed to parse");
          set_state(read.input_index, StableNodeSnapshotState::stable);
          remember_reverse_snapshot(snapshot);
          ++snapshot_count;
        } else {
          set_state(read.input_index, disposition);
        }
        continue;
      }
      if (dynamic_count !=
          static_cast<size_t>(&read - pending.data())) {
        pending[dynamic_count] = read;
      }
      PendingRead& dynamic = pending[dynamic_count++];
      read_requests.push_back(PeerReadRequest{
        .shard_id = dynamic.rptr.memory_node(),
        .remote_offset = dynamic.rptr.byte_offset(),
        .destination = dynamic.buffer,
        .bytes = VamanaNode::HEADER_SIZE,
      });
    }
    pending.resize(dynamic_count);
    if (!read_requests.empty()) {
      post_peer_reads_async(
        *thread, span<const PeerReadRequest>{read_requests});
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }
    }
    for (const PendingRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          read.rptr, read.before, after, read.slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        NodeSnapshot& snapshot = next_snapshot();
        lib_assert(parse_remote_snapshot(read.rptr, read.buffer, snapshot),
                   "coherent remote snapshot failed to parse");
        set_state(read.input_index, StableNodeSnapshotState::stable);
        ++snapshot_count;
      } else {
        set_state(read.input_index, disposition);
      }
    }
  }

  return snapshot_count;
}

const vec<MemoryNode::BeamEntry>&
MemoryNode::score_stable_node_vectors_batched(
    span<const RemotePtr> rptrs,
    const byte_t* stored_query,
    span<const element_t> decoded_query,
    const Configuration& config) {
  struct PendingVectorRead {
    RemotePtr rptr;
    byte_t* buffer{};
    byte_t* after_header{};
    u64 before{};
    u32 slot_incarnation{};
  };

  // Stage2 is synchronous on one storage-owner OS worker, just like its
  // continuation beam. Keeping these containers thread-local preserves their
  // capacity across high-frequency insertions without sharing mutable state
  // across workers or allocating once per L*R candidate batch.
  thread_local vec<BeamEntry> scored;
  thread_local vec<PendingVectorRead> pending;
  scored.clear();
  pending.clear();
  if (rptrs.empty()) return scored;

  const VectorDType dtype = VamanaNode::vector_dtype();
  const bool integral =
    dtype == VectorDType::uint8 || dtype == VectorDType::int8;
  lib_assert(stored_query != nullptr &&
               (integral || decoded_query.size() >= VamanaNode::DIM),
             "stage2 vector scoring query is incomplete");
  scored.reserve(rptrs.size());

  const auto score_vector = [&](const byte_t* vector) {
    return integral
      ? typed_l2_distance(stored_query, dtype, vector, dtype, config.dim)
      : distance_to_stored_vector(decoded_query, vector, config);
  };
  const auto validate_pointer = [&](RemotePtr rptr) {
    if (storage_node_pointer_addressable(rptr)) return true;
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("stage2_vector_scoring", rptr);
    }
    return false;
  };
  const auto score_local = [&](RemotePtr rptr) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      rptr.byte_offset();
    constexpr u32 kMaxReadAttempts = 3;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if (VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        return;
      }
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      const distance_t distance = score_vector(
        record + VamanaNode::offset_vector());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (stable_vector_snapshot_valid(
            rptr, before, after, slot_incarnation)) {
        scored.push_back(BeamEntry{rptr, distance, false});
        return;
      }
      if (before == after ||
          VamanaNode::header_incarnation(after) != rptr.incarnation()) {
        return;
      }
      std::this_thread::yield();
    }
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    byte_t* read_buffer = peer_scratch_buffer_.get_full_buffer();
    constexpr u32 kMaxReadAttempts = 3;
    for (const RemotePtr rptr : rptrs) {
      if (!validate_pointer(rptr)) continue;
      if (local_shard(rptr.memory_node())) {
        score_local(rptr);
        continue;
      }
      for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
        remote_read_bytes(
          rptr.memory_node(), rptr.byte_offset(), read_buffer,
          VamanaNode::size_until_vector_end(), 0);
        const u64 before = *reinterpret_cast<const u64*>(read_buffer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          read_buffer + VamanaNode::offset_slot_incarnation());
        if (VamanaNode::header_incarnation(before) !=
              rptr.incarnation() ||
            slot_incarnation != rptr.incarnation()) {
          break;
        }
        if ((before & VamanaNode::HEADER_NODE_LOCK) != 0) {
          std::this_thread::yield();
          continue;
        }
        u64 after = 0;
        remote_read_bytes(rptr.memory_node(), rptr.byte_offset(),
                          &after, sizeof(after), 0);
        if (stable_vector_snapshot_valid(
              rptr, before, after, slot_incarnation)) {
          scored.push_back(BeamEntry{
            rptr,
            score_vector(read_buffer + VamanaNode::offset_vector()),
            false});
          break;
        }
        if (before == after ||
            VamanaNode::header_incarnation(after) !=
              rptr.incarnation()) {
          break;
        }
        std::this_thread::yield();
      }
    }
    return scored;
  }

  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t validation_offset =
    memory_node_detail::storage_owner_snapshot_validation_offset();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);
  thread_local vec<PeerReadRequest> read_requests;
  thread_local vec<PeerReadPairRequest> read_pairs;
  const bool ordered_pair_supported =
    memory_node_detail::peer_rdma_read_pair_group_limit(
      peer_rdma_read_credit_plan()) != 0;
  lib_assert(validation_offset + VamanaNode::HEADER_SIZE <=
               snapshot_stride,
             "storage-owner snapshot slot lost after-header scratch");
  pending.reserve(max_batch);
  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    read_requests.clear();
    read_pairs.clear();
    u32 remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr rptr = rptrs[index];
      if (!validate_pointer(rptr)) continue;
      if (local_shard(rptr.memory_node())) {
        score_local(rptr);
        continue;
      }
      const size_t scratch_offset =
        static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + validation_offset +
                   VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for "
                 "stage2 vector scoring");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      const PeerReadRequest full_snapshot{
        .shard_id = rptr.memory_node(),
        .remote_offset = rptr.byte_offset(),
        .destination = buffer,
        .bytes = VamanaNode::size_until_vector_end(),
      };
      byte_t* after_header = buffer + validation_offset;
      if (ordered_pair_supported) {
        read_pairs.push_back(PeerReadPairRequest{
          .full_snapshot = full_snapshot,
          .after_header = PeerReadRequest{
            .shard_id = rptr.memory_node(),
            .remote_offset = rptr.byte_offset(),
            .destination = after_header,
            .bytes = VamanaNode::HEADER_SIZE,
          },
        });
      } else {
        read_requests.push_back(full_snapshot);
      }
      pending.push_back(PendingVectorRead{rptr, buffer, after_header});
      ++remote_slot;
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
            read.rptr, read.before, read.before,
            read.slot_incarnation)) {
        continue;
      }
      if (valid_count != static_cast<size_t>(&read - pending.data())) {
        pending[valid_count] = read;
      }
      PendingVectorRead& accepted = pending[valid_count++];
      if (!ordered_pair_supported) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = accepted.rptr.memory_node(),
          .remote_offset = accepted.rptr.byte_offset(),
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
        if (!stable_vector_snapshot_valid(
              read.rptr, read.before, after,
              read.slot_incarnation)) {
          continue;
        }
        scored.push_back(BeamEntry{
          read.rptr,
          score_vector(read.buffer + VamanaNode::offset_vector()),
          false});
      }
      continue;
    }

    // Keep the original two-wave path when the transport cannot reserve two
    // ordered READ credits. A stable vector snapshot is never reduced to a
    // single body read.
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    for (const PendingVectorRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      if (!stable_vector_snapshot_valid(
            read.rptr, read.before, after,
            read.slot_incarnation)) {
        continue;
      }
      scored.push_back(BeamEntry{
        read.rptr,
        score_vector(read.buffer + VamanaNode::offset_vector()),
        false});
    }
  }
  return scored;
}


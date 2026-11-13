#include "memory_node/peer_rpc/detail.hh"
#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
#include "memory_node/storage_owner_index/stage1_reachability_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

namespace authority = memory_node_storage_owner_index_detail;

bool MemoryNode::handle_peer_stage2_expand_score_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
  using memory_node_storage_owner_index_detail::classify_construction_node_snapshot;
  if (payload == nullptr || source_shard >= num_storage_nodes_ ||
      source_shard == storage_id_ || header.item_count == 0 ||
      header.item_count > config.storage_owner_batch_max) {
    return false;
  }

  const size_t response_capacity =
    stage2_expand_score_response_bytes(header.item_count);
  if (response_capacity > peer_rpc_runtime_.message_bytes) return false;
  // The asynchronous sender recycles bounded high-water buffers after copying
  // them into registered send slots. Stage2 emits many medium-sized responses;
  // avoiding a malloc/free pair for each graph wave is important at high QPS.
  vec<byte_t> response = acquire_peer_graph_response_buffer(response_capacity);
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage2_expand_score_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(InsertStatus::ok);
  response_header->reserved = 0;

  const auto* items = stage2_expand_score_items(payload);
  const byte_t* queries = stage2_expand_score_queries(
    payload, header.item_count);
  auto* results = stage2_expand_score_results(response.data());
  auto* neighbors = stage2_expand_score_neighbors(
    response.data(), header.item_count);
  const size_t neighbor_stride = VamanaNode::graph_entry_capacity();
  const VectorDType dtype = VamanaNode::vector_dtype();
  GraphAdjacency adjacency;
  u32 compact_neighbor_count = 0;

  // Home expansion needs only a readable vector, not an owning NodeSnapshot.
  // Pending candidates use the same incarnation/coherence fence as stable ones.
  // Score directly from the local registered node under the same header /
  // incarnation seqlock used by read_node_snapshot(). This removes one vector
  // allocation and 128-byte copy per neighbor while preserving the exact
  // stable/terminal/retryable classification.
  const auto score_local_neighbor = [&] (
      RemotePtr neighbor, const byte_t* query,
      Stage2ExpandScoreNeighbor& output) {
    constexpr u32 kMaxReadAttempts = 3;
    const byte_t* node = local_node_ptr(neighbor);
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(neighbor);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) !=
            neighbor.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const distance_t distance = distance_between_vectors(
        query, dtype, node + VamanaNode::offset_vector(), dtype, config);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(neighbor);
      const StableNodeSnapshotState state = classify_construction_node_snapshot(
        neighbor, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::stable) {
        output.distance = distance;
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::stable);
        return;
      }
      if (state == StableNodeSnapshotState::terminal) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::terminal);
        return;
      }
      std::this_thread::yield();
    }
    output.disposition = static_cast<u32>(
      Stage2HomeDisposition::retryable);
  };

  for (u32 item_index = 0; item_index < header.item_count; ++item_index) {
    const Stage2ExpandScoreItem& item = items[item_index];
    Stage2ExpandScoreResult& result = results[item_index];
    result = {};
    result.pointer_raw = item.pointer_raw;
    result.generation = item.generation;
    result.search_index = item.search_index;
    result.neighbor_offset = compact_neighbor_count;
    result.operation = item.operation;
    const RemotePtr pointer{item.pointer_raw};
    if (item.operation > static_cast<u32>(Stage2HomeOperation::score_only)) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::terminal);
      continue;
    }
    if (!valid_local_storage_node_pointer(pointer)) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::terminal);
      continue;
    }

    const byte_t* query = queries +
      static_cast<size_t>(item_index) * VamanaNode::vector_bytes();
    if (item.operation == static_cast<u32>(Stage2HomeOperation::score_only)) {
      Stage2ExpandScoreNeighbor score{};
      score_local_neighbor(pointer, query, score);
      result.distance = score.distance;
      result.disposition = score.disposition;
      continue;
    }

    adjacency.stable.clear();
    adjacency.provisional.clear();
    if (!read_graph_adjacency(pointer, adjacency)) {
      const u64 before = load_local_node_header_acquire(pointer);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState state = classify_construction_node_snapshot(
        pointer, before, after, slot_incarnation);
      result.disposition = static_cast<u32>(
        state == StableNodeSnapshotState::terminal
          ? Stage2HomeDisposition::terminal
          : Stage2HomeDisposition::retryable);
      continue;
    }

    result.disposition = static_cast<u32>(Stage2HomeDisposition::stable);
    if (adjacency.deleted) continue;
    const size_t total_neighbors = std::min(
      neighbor_stride, adjacency.stable.size());
    result.neighbor_count = static_cast<u32>(total_neighbors);
    size_t output_index = 0;
    const auto emit_neighbor = [&](RemotePtr neighbor) {
      if (output_index >= total_neighbors) return;
      Stage2ExpandScoreNeighbor& output =
        neighbors[static_cast<size_t>(compact_neighbor_count) +
                  output_index++];
      output = {};
      output.pointer_raw = neighbor.raw_address;
      if (!storage_node_pointer_addressable(neighbor)) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::terminal);
        return;
      }
      if (!local_shard(neighbor.memory_node())) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::unscored);
        return;
      }
      score_local_neighbor(neighbor, query, output);
    };
    for (RemotePtr neighbor : adjacency.stable) emit_neighbor(neighbor);
    compact_neighbor_count += result.neighbor_count;
  }

  response.resize(stage2_expand_score_response_bytes(
    header.item_count, compact_neighbor_count));

  PeerReverseUpdateResponse outbound;
  outbound.destination_shard = source_shard;
  outbound.header = *response_header;
  outbound.payload = std::move(response);
  outbound.graph_response = true;
  outbound.queued_at = std::chrono::steady_clock::now();
  const bool queued = try_enqueue_peer_reverse_update_response(
    std::move(outbound));
  if (!queued) {
    peer_stage2_home_response_queue_drops_.fetch_add(
      1, std::memory_order_relaxed);
  }
  return queued;
}

bool MemoryNode::handle_peer_stage2_score_many_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
  using memory_node_storage_owner_index_detail::classify_construction_node_snapshot;
  if (payload == nullptr || source_shard >= num_storage_nodes_ ||
      source_shard == storage_id_ || header.item_count == 0 ||
      header.item_count > std::max<u32>(
        1, config.storage_owner_search_snapshot_batch)) {
    return false;
  }
  const auto* own_header = stage2_score_many_header(payload);
  if (own_header->query_count == 0 ||
      own_header->query_count > header.item_count ||
      own_header->reserved != 0) {
    return false;
  }

  const size_t response_capacity =
    stage2_score_many_response_bytes(header.item_count);
  if (response_capacity > peer_rpc_runtime_.message_bytes) return false;
  vec<byte_t> response = acquire_peer_graph_response_buffer(response_capacity);
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage2_score_many_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(InsertStatus::ok);
  response_header->reserved = 0;

  const auto* items = stage2_score_many_items(payload);
  const byte_t* queries = stage2_score_many_queries(
    payload, header.item_count);
  auto* results = stage2_score_many_results(response.data());
  const VectorDType dtype = VamanaNode::vector_dtype();
  constexpr u32 kMaxReadAttempts = 3;
  for (u32 item_index = 0; item_index < header.item_count; ++item_index) {
    const Stage2ScoreManyItem& item = items[item_index];
    Stage2ScoreManyResult& result = results[item_index];
    result = {};
    result.pointer_raw = item.pointer_raw;
    result.generation = item.generation;
    result.search_index = item.search_index;
    result.disposition = static_cast<u32>(
      Stage2HomeDisposition::terminal);
    const RemotePtr pointer{item.pointer_raw};
    if (item.query_index >= own_header->query_count ||
        !valid_local_storage_node_pointer(pointer)) {
      continue;
    }

    const byte_t* query = queries +
      static_cast<size_t>(item.query_index) * VamanaNode::vector_bytes();
    const byte_t* node = local_node_ptr(pointer);
    bool classified = false;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(pointer);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != pointer.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const distance_t distance = distance_between_vectors(
        query, dtype, node + VamanaNode::offset_vector(), dtype, config);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState state = classify_construction_node_snapshot(
        pointer, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::stable) {
        result.distance = distance;
        result.disposition = static_cast<u32>(
          Stage2HomeDisposition::stable);
        classified = true;
        break;
      }
      if (state == StableNodeSnapshotState::terminal) {
        classified = true;
        break;
      }
      std::this_thread::yield();
    }
    if (!classified) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::retryable);
    }
  }

  PeerReverseUpdateResponse outbound;
  outbound.destination_shard = source_shard;
  outbound.header = *response_header;
  outbound.payload = std::move(response);
  outbound.graph_response = true;
  outbound.queued_at = std::chrono::steady_clock::now();
  const bool queued = try_enqueue_peer_reverse_update_response(
    std::move(outbound));
  if (!queued) {
    peer_stage2_home_response_queue_drops_.fetch_add(
      1, std::memory_order_relaxed);
  }
  return queued;
}

bool MemoryNode::try_send_peer_stage1_retry_response(
    u32 destination_shard,
    const service::storage_owner::PeerRpcHeader& header,
    span<const byte_t> request) {
  using namespace service::storage_owner;
  const auto request_type = static_cast<PeerRpcType>(header.type);
  const size_t response_bytes = request_type ==
      PeerRpcType::stage1_execute_request
    ? stage1_execute_response_bytes(header.item_count)
    : request_type == PeerRpcType::stage1_arm_request
      ? stage1_arm_response_bytes(header.item_count) : 0;
  if (destination_shard >= num_storage_nodes_ ||
      destination_shard == storage_id_ || response_bytes == 0 ||
      response_bytes > peer_rpc_runtime_.message_bytes) {
    return false;
  }

  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(
        destination_shard, PeerRpcSendClass::control, slot_id)) {
    return false;
  }
  byte_t* destination = peer_rpc_runtime_.buffer.get_full_buffer() +
    peer_rpc_async_send_offset(destination_shard, slot_id);
  if (!memory_node_peer_rpc_detail::write_stage1_retry_response(
        storage_id_, header, request,
        span<byte_t>{destination, response_bytes})) {
    release_peer_rpc_send_slot(destination_shard, slot_id);
    return false;
  }
  // This path runs on the CQ progress thread. Use an asynchronous registered
  // send slot; send_peer_rpc_message() intentionally forbids its blocking
  // completion wait here.
  post_peer_rpc_send_slot(destination_shard, slot_id, response_bytes);
  return true;
}


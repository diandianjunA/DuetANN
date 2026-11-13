#pragma once

#include "memory_node/memory_node.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "memory_node/storage_owner_index/graph_pointer_validation.hh"
#include "memory_node/storage_owner_index/snapshot_reuse_policy.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"

namespace memory_node_storage_owner_index_detail {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerCoroutineScratch =
  memory_node_detail::StorageOwnerCoroutineScratch;
using StorageOwnerPruneCandidateInfo =
  memory_node_detail::StorageOwnerPruneCandidateInfo;
using StorageOwnerScoredSnapshot =
  memory_node_detail::StorageOwnerScoredSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
using ReverseSnapshotCache = SnapshotReuseCache<NodeSnapshot>;

inline thread_local ReverseSnapshotCache* reverse_snapshot_cache = nullptr;

class ReverseSnapshotReuseScope {
public:
  explicit ReverseSnapshotReuseScope(size_t expected)
      : previous_(reverse_snapshot_cache) {
    cache_.reserve(expected);
    reverse_snapshot_cache = &cache_;
  }
  ReverseSnapshotReuseScope(const ReverseSnapshotReuseScope&) = delete;
  ReverseSnapshotReuseScope& operator=(const ReverseSnapshotReuseScope&) = delete;
  ~ReverseSnapshotReuseScope() { reverse_snapshot_cache = previous_; }

private:
  ReverseSnapshotCache cache_;
  ReverseSnapshotCache* previous_;
};

inline bool reuse_reverse_snapshot(RemotePtr pointer, NodeSnapshot& snapshot) {
  if (reverse_snapshot_cache == nullptr) return false;
  const NodeSnapshot* cached = reverse_snapshot_cache->find(pointer);
  if (cached == nullptr) return false;
  snapshot = *cached;
  return true;
}

inline void remember_reverse_snapshot(const NodeSnapshot& snapshot) {
  if (reverse_snapshot_cache != nullptr) {
    reverse_snapshot_cache->put(snapshot.rptr, snapshot);
  }
}

inline size_t snapshot_buffer_bytes() {
  return memory_node_detail::storage_owner_snapshot_bytes();
}

inline size_t aligned_snapshot_bytes() {
  return memory_node_detail::storage_owner_snapshot_stride();
}

inline size_t aligned_graph_entry_bytes() {
  return memory_node_storage_owner_index_detail::graph_read_slot_stride();
}

inline u32 storage_owner_construction_width(const Configuration& config) {
  return config.resolved_storage_owner_construction_width();
}

inline u32 storage_owner_snapshot_batch_size(
    const Configuration& config, const StorageOwnerThread* thread = nullptr) {
  const u32 configured =
    std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = aligned_snapshot_bytes();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0,
             "storage-owner coroutine scratch cannot hold one snapshot: "
             "snapshot_stride=" +
               std::to_string(stride) + " scratch_stride=" +
               std::to_string(thread->scratch_stride));
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

inline u32 storage_owner_graph_batch_size(
    const Configuration& config, const StorageOwnerThread* thread = nullptr) {
  const u32 configured =
    std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = aligned_graph_entry_bytes();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0,
             "storage-owner coroutine scratch cannot hold one graph entry: "
             "graph_stride=" +
               std::to_string(stride) + " scratch_stride=" +
               std::to_string(thread->scratch_stride));
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

inline bool parse_remote_snapshot(RemotePtr remote_pointer, const byte_t* data,
                                  NodeSnapshot& snapshot) {
  snapshot.rptr = remote_pointer;
  snapshot.header = *reinterpret_cast<const u64*>(data);
  snapshot.id = *reinterpret_cast<const u32*>(data + VamanaNode::offset_id());
  snapshot.generation =
    *reinterpret_cast<const u32*>(data + VamanaNode::offset_generation());
  snapshot.slot_incarnation = *reinterpret_cast<const u32*>(
    data + VamanaNode::offset_slot_incarnation());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  if ((snapshot.header & VamanaNode::HEADER_NODE_LOCK) != 0 ||
      VamanaNode::header_incarnation(snapshot.header) !=
      remote_pointer.incarnation() ||
      snapshot.slot_incarnation != remote_pointer.incarnation()) {
    snapshot.rptr.reset();
    snapshot.header = 0;
    snapshot.id = 0;
    snapshot.generation = 0;
    snapshot.slot_incarnation = 0;
    snapshot.deleted = false;
    snapshot.vector_data.clear();
    return false;
  }
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), data + VamanaNode::offset_vector(),
              VamanaNode::vector_bytes());
  return true;
}

}  // namespace memory_node_storage_owner_index_detail

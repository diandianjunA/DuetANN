#pragma once

#include <algorithm>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

template <class Snapshot>
class SnapshotReuseCache {
public:
  explicit SnapshotReuseCache(size_t capacity = 16384) : capacity_(capacity) {}

  void reserve(size_t count) { snapshots_.reserve(std::min(count, capacity_)); }

  const Snapshot* find(RemotePtr pointer) const {
    const auto found = snapshots_.find(pointer.raw_address);
    return found == snapshots_.end() ? nullptr : &found->second;
  }

  void put(RemotePtr pointer, const Snapshot& snapshot) {
    if (snapshots_.size() < capacity_) {
      snapshots_.emplace(pointer.raw_address, snapshot);
    }
  }

private:
  size_t capacity_;
  dense_hashmap_t<u64, Snapshot> snapshots_;
};

}  // namespace memory_node_storage_owner_index_detail

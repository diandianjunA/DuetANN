#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/locked_node_publication.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <cstdlib>
#include <execinfo.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

using namespace memory_node_storage_owner_index_detail;

void MemoryNode::report_rejected_graph_pointer(
    const char* boundary,
    RemotePtr pointer,
    RemotePtr parent,
    u64 context) const {
  static std::atomic<u64> rejected{0};
  const u64 count = rejected.fetch_add(1, std::memory_order_relaxed) + 1;
  // Malformed pointers are never a normal churn outcome. Keep diagnostics
  // useful without turning a damaged edge into an unbounded logging attack.
  if (count <= 16 || (count & (count - 1)) == 0) {
    std::cerr << "[storage-owner] rejected malformed graph pointer"
              << " boundary=" << boundary
              << " raw=0x" << std::hex << pointer.raw_address << std::dec
              << " shard=" << pointer.memory_node()
              << " offset=" << pointer.byte_offset()
              << " incarnation=" << pointer.incarnation()
              << " configured_shards=" << num_storage_nodes_
              << " shard_bytes=" << mn_memory_bytes_;
    if (!parent.is_null()) {
      std::cerr << " parent_raw=0x" << std::hex << parent.raw_address
                << std::dec
                << " parent_shard=" << parent.memory_node()
                << " parent_offset=" << parent.byte_offset()
                << " parent_incarnation=" << parent.incarnation();
    }
    if (context != std::numeric_limits<u64>::max()) {
      std::cerr << " context=" << context;
    }
    std::cerr
              << " count=" << count << '\n';
  }
}

IncarnationLockResult MemoryNode::try_lock_node(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return IncarnationLockResult::stale;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return IncarnationLockResult::stale;
  }

  // A local validity check rejects unallocated dynamic addresses and malformed
  // static/dynamic tag combinations.  It is only a fast prefilter: the CAS
  // below is the actual race-closing identity boundary.
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return IncarnationLockResult::stale;
  }

  constexpr u32 kMaxAttempts = 8;
  for (u32 attempt = 0; attempt < kMaxAttempts; ++attempt) {
    IncarnationLockResult result = IncarnationLockResult::busy;
    if (local_shard(rptr.memory_node())) {
      auto* header = reinterpret_cast<u64*>(
        index_buffer_.get_full_buffer() + header_address.offset);
      result = try_lock_header_once(*header, rptr.incarnation());
    } else {
      const auto [locked, observed] = try_lock_remote_header(rptr);
      if (locked) {
        result = IncarnationLockResult::locked;
      } else if (VamanaNode::header_incarnation(observed) !=
                 rptr.incarnation()) {
        result = IncarnationLockResult::stale;
      }
    }
    if (result != IncarnationLockResult::busy) {
      if (result == IncarnationLockResult::locked &&
          std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
        std::lock_guard<std::mutex> lock(debug_node_lock_owners_mutex_);
        debug_node_lock_owners_[rptr.raw_address] = std::this_thread::get_id();
        debug_node_lock_owner_tids_[rptr.raw_address] =
          static_cast<pid_t>(::syscall(SYS_gettid));
        debug_node_lock_owner_workers_[rptr.raw_address] = {
          current_storage_owner_thread_ == nullptr
            ? std::numeric_limits<u32>::max()
            : current_storage_owner_thread_->id,
          current_storage_owner_maintenance_worker_};
        if (debug_node_lock_backtraces_.size() < 128 &&
            debug_node_lock_backtraces_.insert(rptr.raw_address).second) {
          void* frames[16];
          const int count = backtrace(frames, 16);
          char** symbols = backtrace_symbols(frames, count);
          if (symbols != nullptr) {
            std::string trace;
            for (int index = 2; index < count; ++index) {
              trace += symbols[index];
              if (index + 1 < count) trace += " | ";
            }
            std::cerr << "[node-lock-acquire] target=" << rptr.raw_address
                      << " worker="
                      << debug_node_lock_owner_workers_[rptr.raw_address].first
                      << " maintenance="
                      << debug_node_lock_owner_workers_[rptr.raw_address].second
                      << " stack=" << trace << std::endl;
          }
          free(symbols);
        }
      }
      return result;
    }
    std::this_thread::yield();
  }
  if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
    std::lock_guard<std::mutex> lock(debug_node_lock_owners_mutex_);
    if (debug_node_lock_busy_logged_.insert(rptr.raw_address).second) {
      const auto owner = debug_node_lock_owners_.find(rptr.raw_address);
      std::cerr << "[node-lock-busy] target=" << rptr.raw_address;
      if (owner != debug_node_lock_owners_.end()) {
        std::cerr << " owner=" << owner->second;
      } else {
        std::cerr << " owner=untracked";
      }
      const auto owner_tid = debug_node_lock_owner_tids_.find(rptr.raw_address);
      if (owner_tid != debug_node_lock_owner_tids_.end()) {
        std::cerr << " owner_tid=" << owner_tid->second;
      }
      const auto owner_worker = debug_node_lock_owner_workers_.find(rptr.raw_address);
      if (owner_worker != debug_node_lock_owner_workers_.end()) {
        std::cerr << " owner_worker=" << owner_worker->second.first
                  << " owner_maintenance=" << owner_worker->second.second;
      }
      std::cerr << std::endl;
    }
  }
  return IncarnationLockResult::busy;
}

bool MemoryNode::read_locked_node_identity(RemotePtr rptr,
                                           u64& header,
                                           node_t& id,
                                           u32& generation) {
  header = 0;
  id = 0;
  generation = 0;
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  constexpr size_t identity_bytes =
    memory_node_storage_owner_index_detail::kLockedNodeIdentityBytes;
  if (rptr.byte_offset() > mn_memory_bytes_ ||
      identity_bytes > mn_memory_bytes_ - rptr.byte_offset()) {
    return false;
  }

  memory_node_storage_owner_index_detail::LockedNodeIdentity identity;
  const bool valid = memory_node_storage_owner_index_detail::
    read_and_validate_locked_node_identity(
      rptr,
      [&](byte_t* destination, size_t bytes) {
        if (local_shard(rptr.memory_node())) {
          const u64 local_header = load_local_node_header_acquire(rptr);
          std::memcpy(destination, &local_header, sizeof(local_header));
          std::memcpy(destination + VamanaNode::HEADER_SIZE,
                      index_buffer_.get_full_buffer() +
                        rptr.byte_offset() + VamanaNode::HEADER_SIZE,
                      VamanaNode::COMPACT_META_SIZE);
          std::atomic_thread_fence(std::memory_order_acquire);
        } else {
          remote_read_bytes(rptr.memory_node(), rptr.byte_offset(),
                            destination, bytes, 0);
        }
        return true;
      },
      identity);
  header = identity.header;
  id = identity.id;
  generation = identity.generation;
  return valid;
}

bool MemoryNode::publish_locked_node_header(RemotePtr rptr,
                                            u64 observed_header,
                                            u64 set_flags,
                                            u64 clear_flags) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const u64 header_offset =
    vamana::StorageLayoutResolver::header(rptr).offset;
  if (header_offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_offset) {
    return false;
  }
  if (local_shard(rptr.memory_node())) {
    auto* storage = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + header_offset);
    std::atomic_ref<u64> header_ref(*storage);
    return memory_node_storage_owner_index_detail::
      publish_locked_node_header_transition(
        rptr, observed_header, set_flags, clear_flags,
        [&](u64 expected, u64 desired) {
          const u64 original = expected;
          if (header_ref.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_acquire)) {
            return original;
          }
          return expected;
        });
  }
  return memory_node_storage_owner_index_detail::
    publish_locked_node_header_transition(
      rptr, observed_header, set_flags, clear_flags,
      [&](u64 expected, u64 desired) {
        return remote_compare_and_swap(
          rptr.memory_node(), header_offset, expected, desired,
          align_up(sizeof(expected)));
      });
}

bool MemoryNode::storage_node_pointer_addressable(RemotePtr rptr) const {
  return memory_node_storage_owner_index_detail::storage_pointer_addressable(
    rptr, num_storage_nodes_, mn_memory_bytes_);
}

bool MemoryNode::read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot) {
  const auto clear_snapshot = [&]() {
    snapshot.rptr.reset();
    snapshot.header = 0;
    snapshot.id = 0;
    snapshot.generation = 0;
    snapshot.slot_incarnation = 0;
    snapshot.deleted = false;
    snapshot.vector_data.clear();
  };
  if (!storage_node_pointer_addressable(rptr)) {
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("read_node_snapshot", rptr);
    }
    clear_snapshot();
    return false;
  }
  if (reuse_reverse_snapshot(rptr, snapshot)) return true;
  const auto vector_addr = vamana::StorageLayoutResolver::vector(rptr);
  clear_snapshot();
  snapshot.rptr = rptr;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());

  constexpr u32 kMaxReadAttempts = 3;
  if (local_shard(rptr.memory_node())) {
    const byte_t* base = index_buffer_.get_full_buffer();
    const byte_t* ptr = base + rptr.byte_offset();
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      snapshot.id = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_id());
      snapshot.generation = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_generation());
      snapshot.slot_incarnation = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_slot_incarnation());
      std::memcpy(snapshot.vector_data.data(), base + vector_addr.offset,
                  VamanaNode::vector_bytes());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (before == after &&
          snapshot.slot_incarnation == rptr.incarnation()) {
        snapshot.header = after;
        snapshot.deleted =
          (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
        remember_reverse_snapshot(snapshot);
        return true;
      }
      std::this_thread::yield();
    }
    clear_snapshot();
    return false;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->coroutine_scratch()
                          : peer_scratch_buffer_.get_full_buffer();
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), read_buffer,
                      VamanaNode::size_until_vector_end(), 0);
    if (!parse_remote_snapshot(rptr, read_buffer, snapshot)) {
      std::this_thread::yield();
      continue;
    }
    u64 after = 0;
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &after,
                      sizeof(after), 0);
    if (snapshot.header == after &&
        (after & VamanaNode::HEADER_NODE_LOCK) == 0 &&
        VamanaNode::header_incarnation(after) == rptr.incarnation()) {
      remember_reverse_snapshot(snapshot);
      return true;
    }
    std::this_thread::yield();
  }
  clear_snapshot();
  return false;
}

bool MemoryNode::valid_local_storage_node_pointer(RemotePtr rptr) const {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      !local_shard(rptr.memory_node()) ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }
  if (rptr.byte_offset() < gpu_dynamic_node_base_) {
    return rptr.incarnation() == 0 &&
      VamanaNode::header_incarnation(
        load_local_node_header_acquire(rptr)) == 0 &&
      *reinterpret_cast<const u32*>(
        index_buffer_.get_full_buffer() + rptr.byte_offset() +
          VamanaNode::offset_slot_incarnation()) == 0;
  }
  if (rptr.incarnation() == 0) return false;
  const auto* control = reinterpret_cast<const
    gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 high_watermark = std::atomic_ref<const u64>(
    control->dynamic_high_watermark).load(std::memory_order_acquire);
  const u64 node_bytes = VamanaNode::allocation_size();
  if (rptr.byte_offset() > high_watermark ||
      node_bytes > high_watermark - rptr.byte_offset()) {
    return false;
  }
  const byte_t* node = index_buffer_.get_full_buffer() +
    rptr.byte_offset();
  return VamanaNode::header_incarnation(
           load_local_node_header_acquire(rptr)) == rptr.incarnation() &&
    *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation()) == rptr.incarnation();
}

bool MemoryNode::storage_owner_node_live(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_) {
    return false;
  }
  if (!VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  u64 after = 0;
  if (local_shard(rptr.memory_node())) {
    const u64 before = load_local_node_header_acquire(rptr);
    std::memcpy(identity, &before, sizeof(before));
    std::memcpy(identity + VamanaNode::HEADER_SIZE,
                index_buffer_.get_full_buffer() + rptr.byte_offset() +
                  VamanaNode::HEADER_SIZE,
                VamanaNode::COMPACT_META_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    after = load_local_node_header_acquire(rptr);
  } else {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
  }
  const u64 before = *reinterpret_cast<const u64*>(identity);
  const u32 incarnation = *reinterpret_cast<const u32*>(
    identity + VamanaNode::offset_slot_incarnation());
  return before == after &&
    (after & (VamanaNode::HEADER_NODE_LOCK |
              VamanaNode::HEADER_DELETED)) == 0 &&
    VamanaNode::header_incarnation(after) == rptr.incarnation() &&
    incarnation == rptr.incarnation();
}

bool MemoryNode::storage_owner_node_stable(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  u64 after = 0;
  if (local_shard(rptr.memory_node())) {
    const u64 before = load_local_node_header_acquire(rptr);
    std::memcpy(identity, &before, sizeof(before));
    std::memcpy(identity + VamanaNode::HEADER_SIZE,
                index_buffer_.get_full_buffer() + rptr.byte_offset() +
                  VamanaNode::HEADER_SIZE,
                VamanaNode::COMPACT_META_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    after = load_local_node_header_acquire(rptr);
  } else {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
  }
  const u64 before = *reinterpret_cast<const u64*>(identity);
  const u32 incarnation = *reinterpret_cast<const u32*>(
    identity + VamanaNode::offset_slot_incarnation());
  return before == after &&
    (after & (VamanaNode::HEADER_NODE_LOCK |
              VamanaNode::HEADER_DELETED |
              VamanaNode::HEADER_PROVISIONAL)) == 0 &&
    VamanaNode::header_incarnation(after) == rptr.incarnation() &&
    incarnation == rptr.incarnation();
}

bool MemoryNode::read_stable_node_identity(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE >
        mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  constexpr u32 kMaxReadAttempts = 3;
  if (local_shard(rptr.memory_node())) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      rptr.byte_offset();
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (before == after && slot_incarnation == rptr.incarnation()) {
        return stable_vector_snapshot_valid(
          rptr, before, after, slot_incarnation);
      }
      std::this_thread::yield();
    }
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE +
                  VamanaNode::COMPACT_META_SIZE]{};
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    const u64 before = *reinterpret_cast<const u64*>(identity);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      identity + VamanaNode::offset_slot_incarnation());
    if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
        VamanaNode::header_incarnation(before) != rptr.incarnation() ||
        slot_incarnation != rptr.incarnation()) {
      std::this_thread::yield();
      continue;
    }
    u64 after = 0;
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
    if (before == after) {
      return stable_vector_snapshot_valid(
        rptr, before, after, slot_incarnation);
    }
    std::this_thread::yield();
  }
  return false;
}

size_t MemoryNode::read_node_identity_headers_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<std::pair<RemotePtr, u64>>& identities,
    vec<StableNodeSnapshotState>* states) {
  identities.reserve(rptrs.size());
  size_t identity_count = 0;
  if (states != nullptr) {
    states->assign(rptrs.size(), StableNodeSnapshotState::terminal);
  }
  if (rptrs.empty()) return identity_count;

  const auto set_state = [&](size_t input_index,
                             StableNodeSnapshotState state) {
    if (states != nullptr) (*states)[input_index] = state;
  };
  const auto next_identity = [&](size_t input_index,
                                 RemotePtr pointer,
                                 u64 header) {
    if (identity_count == identities.size()) {
      identities.emplace_back(pointer, header);
    } else {
      identities[identity_count] = {pointer, header};
    }
    set_state(input_index, StableNodeSnapshotState::stable);
    ++identity_count;
  };
  const auto pointer_valid = [&](RemotePtr pointer) {
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() >= num_storage_nodes_ ||
        !VamanaNode::hot_graph_entry_available(pointer)) {
      return false;
    }
    const auto header = vamana::StorageLayoutResolver::header(pointer);
    constexpr size_t kIdentityBytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    return header.offset <= mn_memory_bytes_ &&
      kIdentityBytes <= mn_memory_bytes_ - header.offset;
  };
  const auto read_local = [&](size_t input_index, RemotePtr pointer) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      pointer.byte_offset();
    constexpr u32 kMaxReadAttempts = 3;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(pointer);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          pointer, before, after, slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        next_identity(input_index, pointer, before);
        return;
      }
      if (disposition == StableNodeSnapshotState::terminal) {
        return;
      }
      std::this_thread::yield();
    }
    set_state(input_index, StableNodeSnapshotState::retryable);
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    constexpr size_t kIdentityBytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    byte_t identity[kIdentityBytes]{};
    for (size_t input_index = 0; input_index < rptrs.size(); ++input_index) {
      const RemotePtr pointer = rptrs[input_index];
      if (!pointer_valid(pointer)) continue;
      if (local_shard(pointer.memory_node())) {
        read_local(input_index, pointer);
        continue;
      }
      constexpr u32 kMaxReadAttempts = 3;
      bool resolved = false;
      for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
        remote_read_bytes(pointer.memory_node(), pointer.byte_offset(),
                          identity, sizeof(identity), 0);
        const u64 before = *reinterpret_cast<const u64*>(identity);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          identity + VamanaNode::offset_slot_incarnation());
        u64 after = 0;
        remote_read_bytes(pointer.memory_node(), pointer.byte_offset(),
                          &after, sizeof(after), 0);
        const StableNodeSnapshotState disposition =
          classify_physical_node_snapshot(
            pointer, before, after, slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          next_identity(input_index, pointer, before);
          resolved = true;
          break;
        }
        if (disposition == StableNodeSnapshotState::terminal) {
          resolved = true;
          break;
        }
        std::this_thread::yield();
      }
      if (!resolved) {
        set_state(input_index, StableNodeSnapshotState::retryable);
      }
    }
    return identity_count;
  }

  struct PendingIdentityRead {
    size_t input_index{};
    RemotePtr pointer;
    byte_t* buffer{};
    u64 before{};
    u32 slot_incarnation{};
  };
  constexpr size_t kIdentityBytes =
    VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
  const size_t identity_stride = align_up(kIdentityBytes);
  // A compact header permits far more records to fit in scratch than a full
  // vector snapshot, but scratch capacity is not a transport credit. Keep the
  // wave inside the configured/tested request window; the linked-read batch
  // still stripes bounded chains over the available data QPs.
  const size_t max_batch = std::max<size_t>(1, std::min<size_t>(
    thread->scratch_stride / identity_stride,
    storage_owner_snapshot_batch_size(config, thread)));
  thread_local vec<PendingIdentityRead> pending;
  thread_local vec<PeerReadRequest> read_requests;
  pending.reserve(max_batch);

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    read_requests.clear();
    size_t remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr pointer = rptrs[index];
      if (!pointer_valid(pointer)) continue;
      if (local_shard(pointer.memory_node())) {
        read_local(index, pointer);
        continue;
      }
      const size_t scratch_offset = remote_slot++ * identity_stride;
      lib_assert(scratch_offset + kIdentityBytes <= thread->scratch_stride,
                 "storage-owner scratch cannot hold identity wave");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      read_requests.push_back(PeerReadRequest{
        .shard_id = pointer.memory_node(),
        .remote_offset = pointer.byte_offset(),
        .destination = buffer,
        .bytes = kIdentityBytes,
      });
      pending.push_back({index, pointer, buffer});
    }
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    read_requests.clear();
    for (PendingIdentityRead& read : pending) {
      read.before = *reinterpret_cast<const u64*>(read.buffer);
      read.slot_incarnation = *reinterpret_cast<const u32*>(
        read.buffer + VamanaNode::offset_slot_incarnation());
      read_requests.push_back(PeerReadRequest{
        .shard_id = read.pointer.memory_node(),
        .remote_offset = read.pointer.byte_offset(),
        .destination = read.buffer,
        .bytes = VamanaNode::HEADER_SIZE,
      });
    }
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
    for (const PendingIdentityRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          read.pointer, read.before, after, read.slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        next_identity(read.input_index, read.pointer, read.before);
      } else {
        set_state(read.input_index, disposition);
      }
    }
  }
  return identity_count;
}

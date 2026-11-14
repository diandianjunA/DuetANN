#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/locked_node_publication.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace memory_node_storage_owner_index_detail;

void MemoryNode::write_hot_graph_entry(
    RemotePtr rptr,
    const vec<RemotePtr>& neighbors,
    std::optional<u32> generation_override,
    std::optional<bool> deleted_override) {
  GraphAdjacency previous;
  const bool previous_valid = read_graph_adjacency(rptr, previous);
  write_graph_adjacency(
    rptr, neighbors,
    previous_valid ? previous.provisional : vec<RemotePtr>{},
    generation_override.has_value()
      ? generation_override
      : (previous_valid
           ? std::optional<u32>{previous.generation}
           : std::nullopt),
    deleted_override.has_value()
      ? deleted_override
      : (previous_valid
           ? std::optional<bool>{previous.deleted}
           : std::nullopt));
}

void MemoryNode::write_graph_adjacency(
    RemotePtr rptr,
    const vec<RemotePtr>& stable,
    const vec<RemotePtr>& provisional,
    std::optional<u32> generation_override,
    std::optional<bool> deleted_override) {
  if (!storage_node_pointer_addressable(rptr)) return;
  lib_assert(VamanaNode::hot_graph_record_kind_matches(rptr),
             "graph publication handle kind does not match its physical "
             "record plane");

  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  lib_assert(hot_offset <= mn_memory_bytes_ &&
               entry_size <= mn_memory_bytes_ - hot_offset,
             "hot graph write exceeds shard bounds");

  u32 generation = generation_override.value_or(0);
  bool deleted = deleted_override.value_or(false);
  vec<byte_t> previous(entry_size, 0);
  if (local_shard(rptr.memory_node())) {
    std::memcpy(previous.data(),
                index_buffer_.get_full_buffer() + hot_offset, entry_size);
  } else {
    remote_read_bytes(
      rptr.memory_node(), hot_offset, previous.data(), previous.size(), 0);
  }
  const u32 previous_version = vamana::hot_graph::load_u32_le(
    previous.data() + 12);
  const bool previous_valid = (previous_version & 1u) == 0 &&
    vamana::hot_graph::load_u16_le(previous.data() + 2) ==
      vamana::hot_graph::checksum16(previous.data(), previous.size());
  if (previous_valid) {
    if (!generation_override.has_value()) {
      generation = vamana::hot_graph::load_u32_le(previous.data() + 4);
    }
    if (!deleted_override.has_value()) {
      deleted = (previous[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
    }
  } else {
    byte_t prefix[VamanaNode::HEADER_SIZE +
                  VamanaNode::COMPACT_META_SIZE]{};
    if (local_shard(rptr.memory_node())) {
      const u64 header = load_local_node_header_acquire(rptr);
      std::memcpy(prefix, &header, sizeof(header));
      std::memcpy(prefix + VamanaNode::HEADER_SIZE,
                  index_buffer_.get_full_buffer() + rptr.byte_offset() +
                    VamanaNode::HEADER_SIZE,
                  VamanaNode::COMPACT_META_SIZE);
    } else {
      remote_read_bytes(
        rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
    }
    if (!deleted_override.has_value()) {
      deleted = (*reinterpret_cast<const u64*>(prefix) &
                 VamanaNode::HEADER_DELETED) != 0;
    }
    if (!generation_override.has_value()) {
      generation = *reinterpret_cast<const u32*>(
        prefix + VamanaNode::offset_generation());
    }
  }

  vec<RemotePtr> bounded_stable;
  bounded_stable.reserve(std::min<size_t>(stable.size(), VamanaNode::R));
  for (const RemotePtr candidate : stable) {
    if (candidate.is_null() || candidate == rptr ||
        std::find(bounded_stable.begin(), bounded_stable.end(), candidate) !=
          bounded_stable.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(candidate)) {
      report_rejected_graph_pointer(
        "write_graph_adjacency/stable", candidate, rptr);
      continue;
    }
    bounded_stable.push_back(candidate);
    if (bounded_stable.size() == VamanaNode::R) break;
  }
  vec<RemotePtr> bounded_provisional;
  bounded_provisional.reserve(std::min<size_t>(
    provisional.size(), VamanaNode::provisional_slots()));
  for (const RemotePtr candidate : provisional) {
    if (candidate.is_null() || candidate == rptr ||
        std::find(bounded_stable.begin(), bounded_stable.end(), candidate) !=
          bounded_stable.end() ||
        std::find(bounded_provisional.begin(), bounded_provisional.end(),
                  candidate) != bounded_provisional.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(candidate)) {
      report_rejected_graph_pointer(
        "write_graph_adjacency/provisional", candidate, rptr);
      continue;
    }
    bounded_provisional.push_back(candidate);
    if (bounded_provisional.size() == VamanaNode::provisional_slots()) break;
  }

  if (graph_extent_enabled_) {
    const u32 required_bytes = vamana::graph_extent::kHeaderBytes +
      static_cast<u32>(bounded_stable.size() + bounded_provisional.size()) *
        vamana::graph_extent::kPointerBytes;
    graph_extent_cache_.learn(
      rptr, required_bytes, VamanaNode::graph_entry_capacity());
  }

  const bool publish_dynamic_extent = rptr.is_dynamic();
  if (publish_dynamic_extent) {
    lib_assert(VamanaNode::dynamic_graph_publication_layout_valid(),
               "dynamic graph and navigation tag must be contiguous");
  }
  const size_t publication_size = publish_dynamic_extent
    ? VamanaNode::dynamic_graph_publication_size() : entry_size;
  lib_assert(hot_offset <= mn_memory_bytes_ &&
               publication_size <= mn_memory_bytes_ - hot_offset,
             "hot graph publication exceeds shard bounds");
  vec<byte_t> entry(publication_size, 0);
  const u8 edge_count = static_cast<u8>(bounded_stable.size());
  if (publish_dynamic_extent) {
    lib_assert(VamanaNode::encode_dynamic_graph_publication(
                 entry.data(), entry.size(), bounded_stable.data(),
                 bounded_stable.size(), bounded_provisional.data(),
                 bounded_provisional.size(), generation, deleted,
                 rptr.incarnation(), VamanaNode::HOT_GRAPH_SHARD_BITS,
                 graph_extent_enabled_),
               "failed to serialize dynamic graph publication");
  } else {
    VamanaNode::encode_hot_graph_entry(
      entry.data(), edge_count, bounded_stable.data(), edge_count,
      VamanaNode::HOT_GRAPH_SHARD_BITS, generation, false,
      bounded_provisional.data(), bounded_provisional.size(),
      rptr.incarnation());
    if (deleted) {
      // Deleted nodes retain their preserved adjacency for cleanup, while GPU
      // readers still observe the tombstone and ignore the payload.
      entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
      const u16 checksum =
        vamana::hot_graph::checksum16(entry.data(), entry_size);
      vamana::hot_graph::store_u16_le(entry.data() + 2, checksum);
    }
  }
  const u32 publishing_version = (previous_version + 1) | 1u;
  const u32 committed_version = publishing_version + 1;
  vamana::hot_graph::store_u32_le(entry.data() + 12, publishing_version);
  vamana::hot_graph::store_u16_le(
    entry.data() + 2,
    vamana::hot_graph::checksum16(entry.data(), entry_size));
  if (local_shard(rptr.memory_node())) {
    // Readers reject the odd publication version until the final even store.
    std::memcpy(index_buffer_.get_full_buffer() + hot_offset,
                entry.data(), publication_size);
  } else {
    remote_write_bytes(
      rptr.memory_node(), hot_offset, entry.data(), publication_size, 0);
  }

  vamana::hot_graph::store_u32_le(entry.data() + 12, committed_version);
  const u16 committed_checksum =
    vamana::hot_graph::checksum16(entry.data(), entry_size);
  vamana::hot_graph::store_u16_le(entry.data() + 2, committed_checksum);
  byte_t* target = index_buffer_.get_full_buffer() + hot_offset;
  if (local_shard(rptr.memory_node())) {
    std::atomic_ref<u16>(*reinterpret_cast<u16*>(target + 2)).store(
      committed_checksum, std::memory_order_release);
    std::atomic_ref<u32>(*reinterpret_cast<u32*>(target + 12)).store(
      committed_version, std::memory_order_release);
    return;
  }
  remote_write_bytes(rptr.memory_node(), hot_offset + 2,
                     entry.data() + 2, sizeof(committed_checksum), 0);
  remote_write_bytes(rptr.memory_node(), hot_offset + 12,
                     entry.data() + 12, sizeof(committed_version), 0);
}

void MemoryNode::dump_final_graph_for_audit() {
  const char* path = std::getenv("DVSTOR_GRAPH_AUDIT");
  if (path == nullptr) return;
  std::FILE* output = std::fopen(path, "a");
  lib_assert(output != nullptr, "cannot open final graph audit output");
  size_t records = 0;
  size_t errors = 0;
  GraphAdjacency adjacency;
  const auto write_record = [&](const char* kind, u64 identity,
                                bool include_static, u64 self_raw = 0) {
    if (!include_static &&
        std::none_of(adjacency.stable.begin(), adjacency.stable.end(),
                     [](RemotePtr pointer) { return pointer.is_dynamic(); }) &&
        std::none_of(adjacency.provisional.begin(), adjacency.provisional.end(),
                     [](RemotePtr pointer) { return pointer.is_dynamic(); })) {
      return;
    }
    std::string record = std::string(kind) + " " + std::to_string(identity);
    if (self_raw != 0) record += " " + std::to_string(self_raw);
    for (const RemotePtr neighbor : adjacency.stable) {
      if (include_static || neighbor.is_dynamic()) {
        record += " " + std::to_string(neighbor.raw_address);
      }
    }
    record += " |";
    for (const RemotePtr neighbor : adjacency.provisional) {
      if (include_static || neighbor.is_dynamic()) {
        record += " " + std::to_string(neighbor.raw_address);
      }
    }
    record += "\n";
    std::fwrite(record.data(), 1, record.size(), output);
    if ((++records & 1023u) == 0) std::fflush(output);
  };

  for (u64 ordinal = 0; ordinal < gpu_static_node_count_; ++ordinal) {
    const RemotePtr pointer{
      storage_id_, gpu_search::format::kNodeBaseOffset +
        ordinal * VamanaNode::total_size(), 0};
    if (!read_graph_adjacency(pointer, adjacency)) {
      ++errors;
      continue;
    }
    if (adjacency.deleted) continue;
    write_record("final_base", pointer.raw_address, false);
  }

  std::cerr << "[graph-audit] static complete records=" << records << std::endl;
  const auto* control = reinterpret_cast<
    const gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 high_watermark = std::atomic_ref<const u64>(
    control->dynamic_high_watermark).load(std::memory_order_acquire);
  const u64 stride = (VamanaNode::allocation_size() + alignof(u64) - 1) &
    ~(alignof(u64) - 1);
  lib_assert(stride != 0 && high_watermark >= gpu_dynamic_node_base_,
             "invalid dynamic graph audit range");
  for (u64 offset = gpu_dynamic_node_base_; offset < high_watermark;
       offset += stride) {
    const u64 header = *reinterpret_cast<const u64*>(
      index_buffer_.get_full_buffer() + offset);
    const u32 incarnation = VamanaNode::header_incarnation(header);
    if (incarnation == 0 ||
        (header & (VamanaNode::HEADER_NODE_LOCK |
                   VamanaNode::HEADER_DELETED)) != 0) {
      continue;
    }
    const RemotePtr pointer{storage_id_, offset, incarnation};
    const byte_t* node = index_buffer_.get_full_buffer() + offset;
    if (*reinterpret_cast<const u32*>(
          node + VamanaNode::offset_slot_incarnation()) != incarnation) {
      ++errors;
      continue;
    }
    if (!read_graph_adjacency(pointer, adjacency)) {
      ++errors;
      continue;
    }
    if (adjacency.deleted) continue;
    const node_t id = *reinterpret_cast<const node_t*>(
      node + VamanaNode::offset_id());
    write_record("final_node", id, true, pointer.raw_address);
  }
  std::fclose(output);
  std::cerr << "[graph-audit] final records=" << records
            << " errors=" << errors << std::endl;
}

void MemoryNode::write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in write_neighbor_list: " + std::to_string(rptr.memory_node()));
  const auto neighbor_addr = vamana::StorageLayoutResolver::neighbor_slots(rptr);
  lib_assert(neighbor_addr.offset + neighbor_addr.size <= mn_memory_bytes_,
             "neighbor-list write exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(neighbor_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  write_hot_graph_entry(rptr, neighbors);
}

void MemoryNode::write_dynamic_navigation_code(
    RemotePtr rptr, const span<const element_t> components) {
  lib_assert(local_shard(rptr.memory_node()) &&
               rptr.byte_offset() >= gpu_dynamic_node_base_,
             "dynamic PQ code must be written to a local dynamic node");
  thread_local vec<f32> transformed;
  transformed.resize(gpu_navigation_model_.dim);
  byte_t* destination = index_buffer_.get_full_buffer() +
    VamanaNode::dynamic_navigation_code_offset(rptr);
  vamana::dynamic_navigation_code::store_u32_le(
    destination,
    VamanaNode::pack_dynamic_navigation_tag(
      rptr.incarnation(), VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN));
  gpu_search::pq::encode(
    gpu_navigation_model_,
    std::span<const f32>{components.data(), components.size()},
    std::span<u8>{destination + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
                  gpu_navigation_model_.code_bytes()}, transformed);
  const u32 checksum = vamana::dynamic_navigation_code::checksum(
    vamana::dynamic_navigation_code::load_u32_le(destination),
    destination + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
    gpu_navigation_model_.code_bytes());
  vamana::dynamic_navigation_code::store_u32_le(
    destination + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      gpu_navigation_model_.code_bytes(),
    checksum);
}

void MemoryNode::write_new_node(RemotePtr rptr,
                    node_t id,
                    const span<const element_t> components,
                    const vec<RemotePtr>& neighbors,
                    u32 generation,
                    bool provisional) {
  byte_t* ptr = local_node_ptr(rptr);
  const u64 publishing_header = VamanaNode::make_header(
    rptr.incarnation(), VamanaNode::HEADER_NODE_LOCK);
  const u64 initial_header = VamanaNode::make_header(
    rptr.incarnation(), provisional ? VamanaNode::HEADER_PROVISIONAL : 0);
  auto* header_storage = reinterpret_cast<u64*>(ptr);
  std::atomic_ref<u64> header_ref(*header_storage);
  lib_assert(header_ref.load(std::memory_order_acquire) == publishing_header &&
               *reinterpret_cast<const u32*>(
                 ptr + VamanaNode::offset_slot_incarnation()) ==
                 rptr.incarnation(),
             "stale local allocation handle cannot materialize a reused slot");
  std::memset(ptr + VamanaNode::HEADER_SIZE, 0,
              VamanaNode::allocation_size() - VamanaNode::HEADER_SIZE);
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_slot_incarnation()) =
    rptr.incarnation();
  encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                                 ptr + VamanaNode::offset_vector());
  write_dynamic_navigation_code(rptr, components);
  write_hot_graph_entry(rptr, neighbors, generation, false);
  header_ref.store(initial_header, std::memory_order_release);
}

void MemoryNode::write_new_node_on_shard(
    RemotePtr rptr,
    node_t id,
    const span<const element_t> components,
    const vec<RemotePtr>& neighbors,
    u32 generation,
    bool provisional) {
  lib_assert(rptr.memory_node() < num_storage_nodes_ &&
               components.size() == VamanaNode::DIM,
             "invalid remote-node materialization request");
  if (local_shard(rptr.memory_node())) {
    write_new_node(rptr, id, components, neighbors,
                   generation, provisional);
    return;
  }

  vec<byte_t> record(VamanaNode::allocation_size(), 0);
  const u64 publishing_header = VamanaNode::make_header(
    rptr.incarnation(), VamanaNode::HEADER_NODE_LOCK);
  const u64 final_header = VamanaNode::make_header(
    rptr.incarnation(), provisional ? VamanaNode::HEADER_PROVISIONAL : 0);
  const u32 incarnation = rptr.incarnation();
  std::memcpy(record.data(), &publishing_header, sizeof(publishing_header));
  std::memcpy(
    record.data() + VamanaNode::offset_id(), &id, sizeof(id));
  std::memcpy(record.data() + VamanaNode::offset_generation(),
              &generation, sizeof(generation));
  std::memcpy(record.data() + VamanaNode::offset_slot_incarnation(),
              &incarnation, sizeof(incarnation));
  encode_float_vector_to_storage(
    components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
    record.data() + VamanaNode::offset_vector());

  vec<RemotePtr> bounded_neighbors;
  bounded_neighbors.reserve(std::min<size_t>(neighbors.size(), VamanaNode::R));
  for (const RemotePtr neighbor : neighbors) {
    if (neighbor.is_null() || neighbor == rptr ||
        std::find(bounded_neighbors.begin(), bounded_neighbors.end(),
                  neighbor) != bounded_neighbors.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(neighbor)) {
      report_rejected_graph_pointer(
        "write_new_node_on_shard", neighbor, rptr);
      continue;
    }
    bounded_neighbors.push_back(neighbor);
    if (bounded_neighbors.size() == VamanaNode::R) break;
  }
  lib_assert(VamanaNode::dynamic_graph_publication_layout_valid(),
             "dynamic graph and navigation tag must be contiguous");
  lib_assert(VamanaNode::encode_dynamic_graph_publication(
               record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
               record.size() - VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
               bounded_neighbors.data(), bounded_neighbors.size(),
               nullptr, 0, generation, false, rptr.incarnation(),
               VamanaNode::HOT_GRAPH_SHARD_BITS,
               graph_extent_enabled_),
             "failed to serialize remote dynamic graph publication");

  thread_local vec<f32> transformed;
  transformed.resize(gpu_navigation_model_.dim);
  gpu_search::pq::encode(
    gpu_navigation_model_,
    std::span<const f32>{components.data(), components.size()},
    std::span<u8>{
      record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET +
        VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
      gpu_navigation_model_.code_bytes()},
    transformed);
  byte_t* dynamic_code =
    record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET;
  const u32 dynamic_code_checksum =
    vamana::dynamic_navigation_code::checksum(
      vamana::dynamic_navigation_code::load_u32_le(dynamic_code),
      dynamic_code + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
      gpu_navigation_model_.code_bytes());
  vamana::dynamic_navigation_code::store_u32_le(
    dynamic_code + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      gpu_navigation_model_.code_bytes(),
    dynamic_code_checksum);
  // Publish remote records header-last. A reader either observes the old
  // tombstone, the publishing lock, or a complete new incarnation.
  const u64 observed = remote_compare_and_swap(
    rptr.memory_node(), rptr.byte_offset(), publishing_header,
    publishing_header, align_up(sizeof(publishing_header)));
  lib_assert(observed == publishing_header,
             "stale remote allocation handle cannot materialize a reused slot");
  remote_write_bytes(
    rptr.memory_node(), rptr.byte_offset() + VamanaNode::HEADER_SIZE,
    record.data() + VamanaNode::HEADER_SIZE,
    record.size() - VamanaNode::HEADER_SIZE, 0);
  remote_write_bytes(rptr.memory_node(), rptr.byte_offset(),
                     &final_header, sizeof(final_header), 0);
}

bool MemoryNode::set_node_provisional(RemotePtr rptr, bool provisional) {
  if (rptr.is_null() || rptr.memory_node() >= num_storage_nodes_) {
    return false;
  }
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  if (header_addr.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_addr.offset) {
    return false;
  }
  if (local_shard(rptr.memory_node())) {
    auto* header = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header);
    for (u32 attempt = 0; attempt < 8; ++attempt) {
      u64 observed = ref.load(std::memory_order_acquire);
      if (VamanaNode::header_incarnation(observed) !=
          rptr.incarnation()) {
        return false;
      }
      if ((observed & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u64 desired = provisional
        ? observed | static_cast<u64>(VamanaNode::HEADER_PROVISIONAL)
        : observed & ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
      if (ref.compare_exchange_weak(
            observed, desired, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  if (try_lock_node(rptr) != IncarnationLockResult::locked) {
    return false;
  }
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), header_addr.offset,
                    &header, sizeof(header), 0);
  if (provisional) {
    header |= static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
  } else {
    header &= ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
  }
  remote_write_bytes(rptr.memory_node(), header_addr.offset,
                     &header, sizeof(header), 0);
  unlock_node(rptr);
  return true;
}

void MemoryNode::lock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    for (;;) {
      u64 header = ref.load(std::memory_order_acquire);
      lib_assert(VamanaNode::header_incarnation(header) ==
                   rptr.incarnation(),
                 "stale tagged handle attempted to lock a reused local slot");
      if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
      if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  for (;;) {
    auto [success, header] = try_lock_remote_header(rptr);
    if (success) {
      return;
    }
    lib_assert(VamanaNode::header_incarnation(header) ==
                 rptr.incarnation(),
               "stale tagged handle attempted to lock a reused remote slot");
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      std::this_thread::yield();
    }
  }
}

void MemoryNode::unlock_node(RemotePtr rptr) {
  if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
    std::lock_guard<std::mutex> lock(debug_node_lock_owners_mutex_);
    debug_node_lock_owners_.erase(rptr.raw_address);
  }
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_and(~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK), std::memory_order_acq_rel);
    return;
  }

  const byte_t unlock = 0;
  remote_write_bytes(rptr.memory_node(),
                     vamana::StorageLayoutResolver::header(rptr).offset +
                       VamanaNode::HEADER_UNTIL_LOCK,
                     &unlock, 1, 0);
}

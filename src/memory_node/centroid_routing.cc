#include "memory_node/memory_node.hh"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"
#include "vamana/centroid_seed_policy.hh"
#include "vamana/centroid_state.hh"
#include "vamana/storage_layout_resolver.hh"

void MemoryNode::initialize_storage_centroid_route() {
  const filepath_t path = index_path::centroid_state_file(
    index_prefix_, storage_id_ + 1, num_storage_nodes_);
  std::ifstream input(path, std::ios::binary);
  lib_assert(input.good(), "missing physical centroid sidecar: " +
                                path.string());

  vamana::centroid_state::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
               header.magic == vamana::centroid_state::kMagic &&
               header.version == vamana::centroid_state::kVersion &&
               header.header_bytes == sizeof(header) &&
               vamana::centroid_state::valid_header_checksum(header),
             "invalid physical centroid sidecar envelope: " +
               path.string());

  const u64 shard_fingerprint = *reinterpret_cast<const u64*>(
    index_buffer_.get_full_buffer() +
      vamana::centroid_state::kShardFingerprintOffset);
  const u64 expected_payload_bytes =
    vamana::centroid_state::payload_bytes(
      header.dim, header.entry_count);
  std::error_code file_error;
  const std::uintmax_t raw_file_bytes =
    std::filesystem::file_size(path, file_error);
  lib_assert(!file_error &&
               raw_file_bytes <= std::numeric_limits<u64>::max(),
             "cannot size physical centroid sidecar: " + path.string());
  const u64 file_bytes = static_cast<u64>(raw_file_bytes);
  lib_assert(header.build_fingerprint == gpu_index_build_fingerprint_ &&
               header.shard_fingerprint == gpu_shard_build_fingerprint_ &&
               header.shard_fingerprint == shard_fingerprint &&
               header.shard == storage_id_ &&
               header.shard_count == num_storage_nodes_ &&
               header.dim == VamanaNode::DIM &&
               header.max_degree == VamanaNode::R &&
               header.vector_count == gpu_static_node_count_ &&
               header.entry_count >= 1 &&
               header.entry_count <=
                 vamana::centroid_state::kMaxLiveEntries &&
               header.entry_count <= header.vector_count &&
               header.node_base_offset ==
                 gpu_search::format::kNodeBaseOffset &&
               header.vector_dtype ==
                 static_cast<u32>(VamanaNode::vector_dtype()) &&
               header.vector_component_size ==
                 VamanaNode::vector_component_size() &&
               header.metadata_schema_version ==
                 gpu_search::format::kMetadataSchemaVersion &&
               header.node_size == VamanaNode::total_size() &&
               header.vector_offset == VamanaNode::offset_vector() &&
               header.vector_bytes == VamanaNode::vector_bytes() &&
               header.slot_incarnation_offset ==
                 VamanaNode::offset_slot_incarnation() &&
               header.hot_graph_version == vamana::hot_graph::kVersion3 &&
               header.hot_graph_entry_size ==
                 VamanaNode::hot_graph_entry_size() &&
               header.hot_graph_pointer_bytes ==
                 vamana::hot_graph::kCompactPointerBytes &&
               header.hot_graph_shard_bits ==
                 VamanaNode::HOT_GRAPH_SHARD_BITS &&
               header.remote_ptr_format_version ==
                 vamana::centroid_state::kRemotePtrFormatVersion &&
               header.remote_ptr_alignment_log2 ==
                 RemotePtr::OFFSET_ALIGNMENT_LOG2 &&
               header.remote_ptr_offset_bits == RemotePtr::OFFSET_UNIT_BITS &&
               header.remote_ptr_shard_bits == RemotePtr::MEMORY_NODE_BITS &&
               header.remote_ptr_incarnation_bits ==
                 RemotePtr::INCARNATION_BITS &&
               header.static_incarnation == 0 &&
               header.payload_bytes == expected_payload_bytes &&
               header.payload_bytes <=
                 std::numeric_limits<size_t>::max() &&
               header.payload_bytes <= static_cast<u64>(
                 std::numeric_limits<std::streamsize>::max()) &&
               header.payload_bytes <=
                 std::numeric_limits<u64>::max() - sizeof(header) &&
               file_bytes == sizeof(header) + header.payload_bytes,
             "physical centroid sidecar does not match the loaded index "
             "build/layout: " + path.string());
  vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
  input.read(reinterpret_cast<char*>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
  lib_assert(input.gcount() == static_cast<std::streamsize>(payload.size()) &&
               vamana::centroid_state::checksum(payload) ==
                 header.payload_checksum,
             "physical centroid sidecar checksum mismatch: " +
               path.string());

  const auto* sums = reinterpret_cast<const f64*>(payload.data());
  for (u32 dimension = 0; dimension < header.dim; ++dimension) {
    lib_assert(floating_value_is_finite(sums[dimension]),
               "centroid sidecar contains a non-finite sum");
  }
  const auto* stored_entries = reinterpret_cast<
    const vamana::centroid_state::Entry*>(
      payload.data() + static_cast<size_t>(header.dim) * sizeof(f64));
  vec<vamana::routing::CentroidRouter::LiveEntry> entries;
  entries.reserve(header.entry_count);
  std::unordered_set<u64> unique_entries;
  unique_entries.reserve(header.entry_count);
  vec<byte_t> decoded_graph(VamanaNode::neighbor_read_size());
  for (u32 index = 0; index < header.entry_count; ++index) {
    const RemotePtr pointer{stored_entries[index].remote_node};
    const bool valid_static_slot = !pointer.is_null() &&
      pointer.is_well_formed() && pointer.incarnation() == 0 &&
      pointer.memory_node() == storage_id_ &&
      pointer.byte_offset() >= header.node_base_offset &&
      (pointer.byte_offset() - header.node_base_offset) %
          header.node_size == 0 &&
      (pointer.byte_offset() - header.node_base_offset) /
          header.node_size < gpu_static_node_count_;
    lib_assert(valid_static_slot && stored_entries[index].generation == 0 &&
                 stored_entries[index].reserved == 0 &&
                 unique_entries.insert(pointer.raw_address).second &&
                 pointer.memory_node() == storage_id_ &&
                 local_live_vector(pointer) != nullptr,
               "centroid sidecar contains a stale graph entry");

    const byte_t* node = local_node_ptr(pointer);
    const u64 node_header = load_local_node_header_acquire(pointer);
    const u32 node_generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    const u64 disallowed_flags = VamanaNode::HEADER_NODE_LOCK |
      VamanaNode::HEADER_DELETED | VamanaNode::HEADER_PROVISIONAL |
      VamanaNode::HEADER_RETIRING;
    const u64 compact_offset = VamanaNode::hot_graph_entry_offset(pointer);
    lib_assert((node_header & disallowed_flags) == 0 &&
                 (node_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0 &&
                 VamanaNode::header_incarnation(node_header) == 0 &&
                 node_generation == stored_entries[index].generation &&
                 VamanaNode::hot_graph_entry_available(pointer) &&
                 compact_offset <= mn_memory_bytes_ &&
                 VamanaNode::hot_graph_entry_size() <=
                   mn_memory_bytes_ - compact_offset &&
                 (index_buffer_.get_full_buffer()[compact_offset + 1] &
                    VamanaNode::HOT_GRAPH_DELETED) == 0 &&
                 VamanaNode::decode_hot_graph_entry(
                   index_buffer_.get_full_buffer() + compact_offset,
                   decoded_graph.data(), 0),
               "centroid sidecar entry has an invalid static graph record");
    entries.push_back({pointer, stored_entries[index].generation});
  }

  storage_centroid_router_ =
    std::make_unique<vamana::routing::CentroidRouter>(
      VamanaNode::DIM, num_storage_nodes_);
  storage_centroid_static_live_bitmap_.assign(
    static_cast<size_t>((gpu_static_node_count_ + 63) / 64), ~u64{0});
  if (!storage_centroid_static_live_bitmap_.empty() &&
      gpu_static_node_count_ % 64 != 0) {
    storage_centroid_static_live_bitmap_.back() =
      (u64{1} << (gpu_static_node_count_ % 64)) - 1;
  }
  storage_centroid_dynamic_live_bitmap_.clear();
  storage_centroid_static_cursor_ = 0;
  storage_centroid_dynamic_cursor_ = 0;
  lib_assert(storage_centroid_router_->restore_shard_state(
               storage_id_, header.vector_count,
               span<const f64>{sums, header.dim}, entries, 1),
             "failed to restore compensated physical centroid state");
  lib_assert(storage_centroid_router_->publish(),
             "failed to publish restored physical centroid state");
  publish_storage_centroid_route();
  print_status(
    "compensated physical centroid restored on shard " +
    std::to_string(storage_id_) + ": vectors=" +
    std::to_string(header.vector_count) + " entries=" +
    std::to_string(header.entry_count));
}

void MemoryNode::publish_storage_centroid_route() {
  if (storage_centroid_router_ == nullptr) return;
  const auto snapshot = storage_centroid_router_->snapshot();
  lib_assert(snapshot != nullptr && storage_id_ < snapshot->shards.size(),
             "physical centroid snapshot is unavailable");
  const auto& shard = snapshot->shards[storage_id_];

  thread_local vec<gpu_search::format::StorageCentroidRouteEntry> entries;
  entries.clear();
  entries.reserve(shard.live_entry_count);
  for (const auto& source : shard.entries()) {
    entries.push_back({
      .remote_node = source.pointer.raw_address,
      .generation = source.generation,
    });
  }

  const auto* control = reinterpret_cast<
    const gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const auto descriptor = control->centroid_route;
  lib_assert(descriptor.centroid_scalar_type == static_cast<u32>(
               gpu_search::format::CentroidScalarType::float32) &&
               (shard.count == 0 || shard.centroid.size() == VamanaNode::DIM),
             "centroid route publication is not canonical FP32");
  thread_local vec<byte_t> next;
  next.resize(static_cast<size_t>(descriptor.publication_bytes));
  thread_local vec<f32> route_centroid;
  route_centroid.clear();
  if (shard.count != 0) {
    route_centroid.reserve(shard.centroid.size());
    for (f64 coordinate : shard.centroid) {
      route_centroid.push_back(static_cast<f32>(coordinate));
    }
  }
  thread_local str error;
  error.clear();
  lib_assert(gpu_search::format::prepare_storage_centroid_route_publication(
               next, storage_id_, VamanaNode::DIM,
               static_cast<gpu_search::format::CentroidScalarType>(
                 descriptor.centroid_scalar_type),
               descriptor.live_entry_capacity,
               std::max<u64>(1, shard.version), shard.count,
               shard.count == 0 ? nullptr : route_centroid.data(), entries,
               &error),
             error);

  std::lock_guard<std::mutex> lock(storage_centroid_publication_mutex_);
  auto* destination = index_buffer_.get_full_buffer() +
    descriptor.remote_offset;
  auto* destination_header = reinterpret_cast<
    gpu_search::format::StorageCentroidRoutePublicationHeader*>(destination);
  std::atomic_ref<u64> sequence(destination_header->sequence);
  const u64 current = sequence.load(std::memory_order_relaxed);
  const u64 odd = (current & ~u64{1}) + 1;
  const u64 even = odd + 1;
  sequence.store(odd, std::memory_order_release);
  std::memcpy(
    destination + sizeof(u64), next.data() + sizeof(u64),
    next.size() - sizeof(u64));
  std::atomic_thread_fence(std::memory_order_release);
  sequence.store(even, std::memory_order_release);
}

vec<RemotePtr> MemoryNode::local_centroid_route_entries() const {
  vec<RemotePtr> entries;
  if (storage_centroid_router_ == nullptr) return entries;
  const auto snapshot = storage_centroid_router_->snapshot();
  if (snapshot == nullptr || storage_id_ >= snapshot->shards.size()) {
    return entries;
  }
  const auto& shard = snapshot->shards[storage_id_];
  entries.reserve(shard.live_entry_count);
  for (const auto& entry : shard.entries()) {
    if (local_live_vector(entry.pointer) != nullptr) {
      entries.push_back(entry.pointer);
    }
  }
  return entries;
}

vec<vamana::routing::CentroidRouter::LiveEntry>
MemoryNode::select_local_centroid_live_entries(
    span<const RemotePtr> preferred) {
  using LiveEntry = vamana::routing::CentroidRouter::LiveEntry;
  thread_local vec<LiveEntry> candidates;
  candidates.clear();
  candidates.reserve(
    preferred.size() + vamana::routing::CentroidRouter::kMaxLiveEntries);
  thread_local hashset_t<RemotePtr> seen;
  seen.clear();
  seen.reserve(candidates.capacity());
  const auto append = [&](RemotePtr pointer) {
    if (pointer.is_null() || pointer.memory_node() != storage_id_ ||
        !valid_local_storage_node_pointer(pointer)) {
      return false;
    }
    if (seen.contains(pointer)) return true;
    const u64 header = load_local_node_header_acquire(pointer);
    if ((header & (VamanaNode::HEADER_DELETED |
                   VamanaNode::HEADER_PROVISIONAL |
                   VamanaNode::HEADER_RETIRING)) != 0 ||
        VamanaNode::header_incarnation(header) != pointer.incarnation() ||
        (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0) {
      return false;
    }
    const byte_t* node = local_node_ptr(pointer);
    if (*reinterpret_cast<const u32*>(
          node + VamanaNode::offset_slot_incarnation()) !=
        pointer.incarnation()) {
      return false;
    }
    seen.insert(pointer);
    candidates.push_back(LiveEntry{
      pointer,
      *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_generation()),
    });
    return true;
  };

  for (const RemotePtr pointer : preferred) append(pointer);
  if (storage_centroid_router_ != nullptr) {
    const auto snapshot = storage_centroid_router_->snapshot();
    if (snapshot != nullptr && storage_id_ < snapshot->shards.size()) {
      for (const LiveEntry& entry :
           snapshot->shards[storage_id_].entries()) {
        append(entry.pointer);
      }
    }
  }

  const u64 authoritative_count =
    storage_centroid_router_->authoritative_count(storage_id_);
  const size_t desired_entries = static_cast<size_t>(std::min<u64>(
    authoritative_count, vamana::routing::CentroidRouter::kMaxLiveEntries));
  if (desired_entries == 0) return {};

  const auto sample_bitmap = [&](vec<u64>& bitmap,
                                 u64 valid_bits,
                                 u64 base,
                                 u64 stride,
                                 bool dynamic,
                                 u64& cursor,
                                 size_t sample_budget) {
    if (bitmap.empty() || valid_bits == 0 || sample_budget == 0) {
      return size_t{0};
    }
    const size_t before = candidates.size();
    // Up to four existing roots can be encountered first. Probe past those
    // duplicates, but retain a fixed per-plane word and set-bit budget.
    const size_t candidate_probe_budget = sample_budget +
      vamana::routing::CentroidRouter::kMaxLiveEntries;
    thread_local vec<u64> ordinals;
    vamana::routing::bounded_rotating_live_samples_into(
      span<const u64>{bitmap}, valid_bits, cursor,
      candidate_probe_budget, ordinals);
    for (const u64 ordinal : ordinals) {
      if (candidates.size() - before >= sample_budget) break;
      const u64 mask = u64{1} << (ordinal % 64);
      const u64 offset = base + ordinal * stride;
      const u32 incarnation = dynamic
        ? *reinterpret_cast<const u32*>(
            index_buffer_.get_full_buffer() + offset +
            VamanaNode::offset_slot_incarnation())
        : 0;
      if (!append(RemotePtr{storage_id_, offset, incarnation})) {
        // Membership metadata is authoritative. Repair a stale bit lazily;
        // the fixed probe budget still bounds this batch under sparse reuse.
        bitmap[ordinal / 64] &= ~mask;
      }
    }
    return candidates.size() - before;
  };

  const u64 dynamic_bits = static_cast<u64>(
    storage_centroid_dynamic_live_bitmap_.size()) * 64;
  const u64 dynamic_stride =
    (VamanaNode::allocation_size() + alignof(u64) - 1) &
    ~u64{alignof(u64) - 1};
  // Explore four additional live identities on every membership batch even
  // when all four published roots remain valid. Split the first probes across
  // immutable and dynamic planes so neither population can starve, then spend
  // any unused quota on whichever plane can supply it. All scans are bounded
  // by constants and advance independent cursors through sparse regions.
  constexpr size_t kInitialPlaneQuota =
    vamana::routing::kLiveSeedExplorationSamples / 2;
  size_t sampled = sample_bitmap(
    storage_centroid_static_live_bitmap_, gpu_static_node_count_,
    gpu_search::format::kNodeBaseOffset, VamanaNode::total_size(), false,
    storage_centroid_static_cursor_, kInitialPlaneQuota);
  sampled += sample_bitmap(
    storage_centroid_dynamic_live_bitmap_, dynamic_bits,
    gpu_dynamic_node_base_, dynamic_stride, true,
    storage_centroid_dynamic_cursor_, kInitialPlaneQuota);
  size_t remaining = vamana::routing::kLiveSeedExplorationSamples - sampled;
  if (remaining != 0) {
    const size_t extra_static = sample_bitmap(
      storage_centroid_static_live_bitmap_, gpu_static_node_count_,
      gpu_search::format::kNodeBaseOffset, VamanaNode::total_size(), false,
      storage_centroid_static_cursor_, remaining);
    remaining -= extra_static;
  }
  if (remaining != 0) {
    (void)sample_bitmap(
      storage_centroid_dynamic_live_bitmap_, dynamic_bits,
      gpu_dynamic_node_base_, dynamic_stride, true,
      storage_centroid_dynamic_cursor_, remaining);
  }

  thread_local vec<f64> centroid;
  centroid.resize(VamanaNode::DIM);
  const bool have_centroid =
    storage_centroid_router_->copy_authoritative_centroid(
      storage_id_, span<f64>{centroid});
  lib_assert(have_centroid &&
               candidates.size() >= desired_entries,
             "non-empty centroid membership lost every live route seed");
  struct RankedEntry {
    LiveEntry entry;
    vamana::routing::CentroidSeedRank rank;
  };
  thread_local vec<RankedEntry> ranked;
  ranked.clear();
  ranked.reserve(candidates.size());
  thread_local vec<element_t> decoded;
  decoded.resize(VamanaNode::DIM);
  for (const LiveEntry& candidate : candidates) {
    const byte_t* node = local_node_ptr(candidate.pointer);
    decode_storage_vector_to_float(
      node + VamanaNode::offset_vector(), VamanaNode::vector_dtype(),
      VamanaNode::DIM, decoded.data());
    ranked.push_back({
      .entry = candidate,
      .rank = {
        .squared_l2 = vamana::routing::centroid_seed_squared_l2(
          span<const f32>{decoded}, span<const f64>{centroid}),
        .pointer_raw = candidate.pointer.raw_address,
      },
    });
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedEntry& lhs, const RankedEntry& rhs) {
              return vamana::routing::centroid_seed_rank_less(
                lhs.rank, rhs.rank);
            });
  vec<LiveEntry> entries;
  entries.reserve(desired_entries);
  for (size_t index = 0; index < desired_entries; ++index) {
    entries.push_back(ranked[index].entry);
  }
  return entries;
}

bool MemoryNode::apply_local_centroid_membership_ops(
    span<const service::storage_owner::CentroidMembershipOp> ops) {
  using Kind = service::storage_owner::CentroidMembershipKind;
  if (ops.empty()) return true;
  std::lock_guard<std::mutex> update_lock(
    storage_centroid_update_mutex_);
  thread_local vec<RemotePtr> preferred;
  preferred.clear();
  preferred.reserve(ops.size());
  bool changed = false;
  bool success = true;
  thread_local vec<element_t> decoded;
  decoded.resize(VamanaNode::DIM);
  const auto valid_centroid_slot_address = [&](RemotePtr pointer) {
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() != storage_id_ ||
        !VamanaNode::hot_graph_entry_available(pointer)) {
      return false;
    }
    const auto header_address =
      vamana::StorageLayoutResolver::header(pointer);
    if (header_address.offset > mn_memory_bytes_ ||
        sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
      return false;
    }
    const u64 offset = pointer.byte_offset();
    if (offset < gpu_dynamic_node_base_) {
      return pointer.incarnation() == 0 &&
        offset >= gpu_search::format::kNodeBaseOffset &&
        (offset - gpu_search::format::kNodeBaseOffset) %
            VamanaNode::total_size() == 0 &&
        (offset - gpu_search::format::kNodeBaseOffset) /
            VamanaNode::total_size() < gpu_static_node_count_;
    }
    if (pointer.incarnation() == 0) return false;
    const u64 stride =
      (VamanaNode::allocation_size() + alignof(u64) - 1) &
      ~u64{alignof(u64) - 1};
    if ((offset - gpu_dynamic_node_base_) % stride != 0) return false;
    const auto* control = reinterpret_cast<const
      gpu_search::format::StorageControlBlock*>(
        index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
    const u64 high_watermark = std::atomic_ref<const u64>(
      control->dynamic_high_watermark).load(std::memory_order_acquire);
    return offset <= high_watermark &&
      VamanaNode::allocation_size() <= high_watermark - offset;
  };
  const auto update_live_bitmap = [&](RemotePtr pointer, bool live) {
    const u64 offset = pointer.byte_offset();
    u64 ordinal = 0;
    vec<u64>* bitmap = nullptr;
    if (offset >= gpu_dynamic_node_base_) {
      const u64 stride =
        (VamanaNode::allocation_size() + alignof(u64) - 1) &
        ~u64{alignof(u64) - 1};
      lib_assert((offset - gpu_dynamic_node_base_) % stride == 0,
                 "centroid membership references a misaligned dynamic node");
      ordinal = (offset - gpu_dynamic_node_base_) / stride;
      bitmap = &storage_centroid_dynamic_live_bitmap_;
      const size_t required_words = static_cast<size_t>(ordinal / 64 + 1);
      if (bitmap->size() < required_words) bitmap->resize(required_words, 0);
    } else {
      lib_assert(offset >= gpu_search::format::kNodeBaseOffset &&
                   (offset - gpu_search::format::kNodeBaseOffset) %
                     VamanaNode::total_size() == 0,
                 "centroid membership references a misaligned static node");
      ordinal = (offset - gpu_search::format::kNodeBaseOffset) /
        VamanaNode::total_size();
      lib_assert(ordinal < gpu_static_node_count_,
                 "centroid membership static ordinal exceeds the shard");
      bitmap = &storage_centroid_static_live_bitmap_;
    }
    const u64 mask = u64{1} << (ordinal % 64);
    if (live) {
      (*bitmap)[ordinal / 64] |= mask;
    } else {
      (*bitmap)[ordinal / 64] &= ~mask;
    }
  };

  // Reject malformed envelopes before touching any node. Identity races are
  // handled independently below: valid operations in the same wire batch are
  // still published, while the caller retries the stale item idempotently.
  for (const auto& op : ops) {
    const RemotePtr pointer{op.node_raw};
    const auto kind = static_cast<Kind>(op.kind);
    if (op.maintenance_sequence == 0 || op.reserved != 0 ||
        (kind != Kind::add && kind != Kind::remove) ||
        !valid_centroid_slot_address(pointer)) {
      return false;
    }
  }

  for (const auto& op : ops) {
    const RemotePtr pointer{op.node_raw};
    const auto kind = static_cast<Kind>(op.kind);

    const auto pointer_lock = try_lock_node(pointer);
    if (pointer_lock != memory_node_storage_owner_index_detail::
                          IncarnationLockResult::locked) {
      // A structurally valid remove whose tagged incarnation is already gone
      // has reached its postcondition.  It must not retry forever and, most
      // importantly, must never debit the slot's new occupant.
      bool old_incarnation_absent = false;
      if (kind == Kind::remove &&
          pointer_lock == memory_node_storage_owner_index_detail::
                            IncarnationLockResult::stale) {
        const u64 before = load_local_node_header_acquire(pointer);
        const byte_t* observed = local_node_ptr(pointer);
        const node_t observed_id = *reinterpret_cast<const node_t*>(
          observed + VamanaNode::offset_id());
        const u32 observed_generation = *reinterpret_cast<const u32*>(
          observed + VamanaNode::offset_generation());
        const u32 observed_slot_incarnation =
          *reinterpret_cast<const u32*>(
            observed + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (before == after &&
            (after & VamanaNode::HEADER_NODE_LOCK) == 0) {
          old_incarnation_absent =
            memory_node_storage_owner_maintenance_detail::
              classify_centroid_remove_identity(
                pointer.incarnation(),
                VamanaNode::header_incarnation(after),
                observed_slot_incarnation,
                observed_id == op.id &&
                  observed_generation == op.generation) ==
              memory_node_storage_owner_maintenance_detail::
                CentroidRemoveIdentityDecision::already_absent;
        }
      }
      if (!old_incarnation_absent) success = false;
      continue;
    }
    auto* node = local_node_ptr(pointer);
    auto* header_storage = reinterpret_cast<u64*>(node);
    std::atomic_ref<u64> header_ref(*header_storage);
    const u64 header = header_ref.load(std::memory_order_acquire);
    const node_t id = *reinterpret_cast<const node_t*>(
      node + VamanaNode::offset_id());
    const u32 generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    const bool accounted =
      (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
    const bool id_and_generation_match =
      id == op.id && generation == op.generation;
    const bool identity_matches = id_and_generation_match &&
      VamanaNode::header_incarnation(header) == pointer.incarnation() &&
      slot_incarnation == pointer.incarnation();
    const bool addable =
      (header & (VamanaNode::HEADER_DELETED |
                 VamanaNode::HEADER_PROVISIONAL |
                 VamanaNode::HEADER_RETIRING)) == 0;
    if (kind == Kind::remove) {
      const auto decision =
        memory_node_storage_owner_maintenance_detail::
          classify_centroid_remove_identity(
            pointer.incarnation(), VamanaNode::header_incarnation(header),
            slot_incarnation, id_and_generation_match);
      if (decision == memory_node_storage_owner_maintenance_detail::
                        CentroidRemoveIdentityDecision::already_absent) {
        unlock_node(pointer);
        continue;
      }
      if (decision != memory_node_storage_owner_maintenance_detail::
                        CentroidRemoveIdentityDecision::apply_exact) {
        unlock_node(pointer);
        success = false;
        continue;
      }
    } else if (!identity_matches || !addable) {
      unlock_node(pointer);
      success = false;
      continue;
    }
    if ((kind == Kind::add && accounted) ||
        (kind == Kind::remove && !accounted)) {
      unlock_node(pointer);
      continue;
    }

    decode_storage_vector_to_float(
      node + VamanaNode::offset_vector(), VamanaNode::vector_dtype(),
      VamanaNode::DIM, decoded.data());
    const bool applied = kind == Kind::add
      ? storage_centroid_router_->insert(
          storage_id_, span<const element_t>{decoded})
      : storage_centroid_router_->erase(
          storage_id_, span<const element_t>{decoded});
    if (!applied) {
      unlock_node(pointer);
      success = false;
      continue;
    }
    if (kind == Kind::add) {
      static std::FILE* graph_audit = [] {
        const char* path = std::getenv("DUETANN_GRAPH_AUDIT");
        return path == nullptr ? nullptr : std::fopen(path, "a");
      }();
      if (graph_audit != nullptr && pointer.is_dynamic() &&
          std::getenv("DUETANN_GRAPH_AUDIT_FINAL_ONLY") == nullptr) {
        GraphAdjacency adjacency;
        lib_assert(read_graph_adjacency(pointer, adjacency),
                   "graph audit could not read the final local adjacency");
        std::fprintf(graph_audit, "node %llu %u %llu",
                     static_cast<unsigned long long>(id), storage_id_,
                     static_cast<unsigned long long>(pointer.raw_address));
        for (const RemotePtr neighbor : adjacency.stable) {
          std::fprintf(graph_audit, " %llu",
                       static_cast<unsigned long long>(neighbor.raw_address));
        }
        std::fprintf(graph_audit, "\n");
        std::fflush(graph_audit);
      }
      header_ref.fetch_or(
        static_cast<u64>(VamanaNode::HEADER_CENTROID_ACCOUNTED),
        std::memory_order_release);
      update_live_bitmap(pointer, true);
      preferred.push_back(pointer);
    } else {
      header_ref.fetch_and(
        ~static_cast<u64>(VamanaNode::HEADER_CENTROID_ACCOUNTED),
        std::memory_order_release);
      update_live_bitmap(pointer, false);
    }
    unlock_node(pointer);
    changed = true;
  }

  if (!changed) return success;
  const auto entries = select_local_centroid_live_entries(
    span<const RemotePtr>{preferred});
  // Empty membership is legal only when the authoritative count became zero;
  // erase() canonicalizes that state and clears its entries itself.
  if (!entries.empty()) {
    (void)storage_centroid_router_->replace_live_entries(
      storage_id_, entries);
  }
  const u64 authoritative_count =
    storage_centroid_router_->authoritative_count(storage_id_);
  lib_assert(!entries.empty() || authoritative_count == 0,
             "non-empty shard lost every dynamic centroid route entry");
  lib_assert(storage_centroid_router_->publish(),
             "centroid membership mutation was not publishable");
  publish_storage_centroid_route();
  return success;
}


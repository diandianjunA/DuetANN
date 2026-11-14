#include <array>
#include <cassert>
#include <cstdint>

#include "gpu_search/graph_record_validation.hh"
#include "remote_pointer.hh"
#include "vamana/graph_extent_cache.hh"
#include "vamana/graph_extent_policy.hh"

namespace {

constexpr u32 kCapacity = 102;
constexpr u32 kRecordBytes = 16 + kCapacity * sizeof(u64);
constexpr u32 kQuantum = vamana::graph_extent::kQuantumEdges;

constexpr u8 level(u32 edges) {
  return vamana::graph_extent::level_for_neighbors(edges, kCapacity);
}

constexpr u32 bytes(u32 edges) {
  return vamana::graph_extent::bytes_for_level(
    level(edges), kRecordBytes, kCapacity);
}

void set_degree(std::array<byte_t, kRecordBytes>& record, u32 stable,
                u32 incarnation) {
  record.fill(0);
  record[0] = static_cast<byte_t>(stable);
  record[8] = static_cast<byte_t>(incarnation);
  record[9] = static_cast<byte_t>(incarnation >> 8);
  record[10] = static_cast<byte_t>(incarnation >> 16);
  record[11] = static_cast<byte_t>(incarnation >> 24);
  const u16 checksum = gpu_search::graph_record_validation::checksum16(
    record.data(), record.size());
  record[2] = static_cast<byte_t>(checksum);
  record[3] = static_cast<byte_t>(checksum >> 8);
}

}  // namespace

int main() {
  namespace extent = vamana::graph_extent;
  namespace validation = gpu_search::graph_record_validation;
  assert(level(0) == 0);
  assert(level(kQuantum) == 1);
  assert(level(kQuantum + 1) == 2);
  assert(level(102) == (102 + kQuantum - 1) / kQuantum);
  assert(extent::level_for_neighbors(136, 136) ==
         (136 + kQuantum - 1) / kQuantum);
  assert(extent::level_for_neighbors(103, kCapacity) == extent::kUnknownLevel);
  assert(extent::bytes_for_level(0, kRecordBytes, kCapacity) == 16);
  assert(extent::bytes_for_level(2, kRecordBytes, kCapacity) ==
         16 + 2 * kQuantum * sizeof(u64));
  assert(extent::bytes_for_level(level(102), kRecordBytes, kCapacity) ==
         kRecordBytes);
  assert(extent::bytes_for_level(
           extent::kUnknownLevel, kRecordBytes, kCapacity) == kRecordBytes);

  auto transition = extent::adapt_level(extent::kUnknownLevel, 2);
  assert(transition.adaptation == extent::Adaptation::refined_unknown &&
         transition.level == 2);
  transition = extent::adapt_level(2, 5);
  assert(transition.adaptation == extent::Adaptation::promoted &&
         transition.level == 5);
  transition = extent::adapt_level(5, 4);
  assert(transition.adaptation == extent::Adaptation::none &&
         transition.level == 5);
  transition = extent::adapt_level(5, 2);
  assert(transition.adaptation == extent::Adaptation::demoted &&
         transition.level == 3);

  extent::Cache cache{8};
  const RemotePtr base{1, 0x4000, 0};
  const RemotePtr dynamic{1, 0x8000, 17};
  for (const RemotePtr handle : {base, dynamic}) {
    const u32 required = 16 + 9 * sizeof(u64);
    assert(cache.lookup(handle) == extent::kUnknownLevel);
    assert(cache.transport_bytes(handle, kRecordBytes, kCapacity) ==
           kRecordBytes);
    assert(cache.learn(handle, required, kCapacity) ==
           extent::Adaptation::refined_unknown);
    assert(cache.lookup(handle) == level(9));
    assert(cache.transport_bytes(handle, kRecordBytes, kCapacity) == bytes(9));
    assert(cache.learn(handle, 16 + 33 * sizeof(u64), kCapacity) ==
           extent::Adaptation::promoted);
    assert(cache.lookup(handle) == level(33));
    assert(cache.learn(handle, required, kCapacity) ==
           extent::Adaptation::demoted);
    assert(cache.lookup(handle) == level(9) + 1);
    cache.invalidate(handle);
    assert(cache.lookup(handle) == extent::kUnknownLevel);
  }

  const RemotePtr recycled{1, 0x8000, 18};
  assert(cache.learn(recycled, 16 + 33 * sizeof(u64), kCapacity) ==
         extent::Adaptation::refined_unknown);
  assert(cache.learn(dynamic, 16 + 65 * sizeof(u64), kCapacity) ==
         extent::Adaptation::none);
  assert(cache.lookup(dynamic) == extent::kUnknownLevel);
  assert(cache.lookup(recycled) == level(33));

  std::array<byte_t, kRecordBytes> record{};
  extent::Cache read_cache{8};
  set_degree(record, 49, 17);
  read_cache.learn(dynamic, 16 + 49 * sizeof(u64), kCapacity);
  u32 transfer = read_cache.transport_bytes(dynamic, kRecordBytes, kCapacity);
  assert(transfer == bytes(49));
  assert(validation::classify_zero_extended_snapshot(
           record.data(), transfer, kRecordBytes, 96, kCapacity, 17) ==
         validation::SnapshotState::valid);

  set_degree(record, 80, 17);
  transfer = read_cache.transport_bytes(dynamic, kRecordBytes, kCapacity);
  assert(validation::classify_zero_extended_snapshot(
           record.data(), transfer, kRecordBytes, 96, kCapacity, 17) ==
         validation::SnapshotState::invalid);
  read_cache.note_fallback();
  read_cache.invalidate(dynamic);
  transfer = read_cache.transport_bytes(dynamic, kRecordBytes, kCapacity);
  assert(transfer == kRecordBytes);
  assert(validation::classify_snapshot(
           record.data(), kRecordBytes, 96, kCapacity, 17) ==
         validation::SnapshotState::valid);
  read_cache.learn(dynamic, 16 + 80 * sizeof(u64), kCapacity);
  assert(read_cache.lookup(dynamic) == level(80));

  set_degree(record, 9, 17);
  read_cache.learn(dynamic, 16 + 9 * sizeof(u64), kCapacity);
  assert(read_cache.lookup(dynamic) == level(9) + 1);
  set_degree(record, 9, 18);
  assert(validation::classify_snapshot(
           record.data(), kRecordBytes, 96, kCapacity, 17) ==
         validation::SnapshotState::stale_incarnation);

  const auto stats = read_cache.stats();
  assert(stats.short_reads == 2 && stats.full_reads == 1 &&
         stats.fallbacks == 1);
  assert(read_cache.transport_bytes(dynamic, kRecordBytes, kCapacity, false) ==
         kRecordBytes);
  return 0;
}

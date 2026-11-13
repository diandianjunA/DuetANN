#pragma once

#include <cstdint>

#include "common/constants.hh"

namespace vamana::graph_extent {

inline constexpr std::uint32_t kHeaderBytes = 16;
inline constexpr std::uint32_t kPointerBytes = 8;
inline constexpr std::uint32_t kQuantumEdges = DVSTOR_GRAPH_EXTENT_QUANTUM_EDGES;
inline constexpr std::uint8_t kUnknownLevel = 0xffu;

#if defined(__CUDACC__)
#define DVSTOR_GRAPH_EXTENT_HD __host__ __device__ __forceinline__
#else
#define DVSTOR_GRAPH_EXTENT_HD constexpr
#endif

enum class Adaptation : std::uint8_t {
  none,
  refined_unknown,
  promoted,
  demoted,
};

struct AdaptationResult {
  std::uint8_t level{kUnknownLevel};
  Adaptation adaptation{Adaptation::none};
};

DVSTOR_GRAPH_EXTENT_HD std::uint8_t level_for_neighbors(
    std::uint32_t live_neighbors, std::uint32_t graph_entry_capacity) {
  if (live_neighbors > graph_entry_capacity) return kUnknownLevel;
  const std::uint32_t level = live_neighbors / kQuantumEdges +
    (live_neighbors % kQuantumEdges != 0 ? 1u : 0u);
  return level < kUnknownLevel ? static_cast<std::uint8_t>(level)
                               : kUnknownLevel;
}

DVSTOR_GRAPH_EXTENT_HD std::uint32_t bytes_for_level(
    std::uint8_t level, std::uint32_t record_bytes,
    std::uint32_t graph_entry_capacity) {
  if (record_bytes < kHeaderBytes || level == kUnknownLevel) {
    return record_bytes;
  }
  const std::uint32_t maximum_level =
    (graph_entry_capacity + kQuantumEdges - 1u) / kQuantumEdges;
  if (level > maximum_level) return record_bytes;
  const std::uint32_t covered_edges =
    static_cast<std::uint32_t>(level) * kQuantumEdges;
  const std::uint32_t bounded_edges =
    covered_edges < graph_entry_capacity ? covered_edges : graph_entry_capacity;
  const std::uint32_t extent_bytes =
    kHeaderBytes + bounded_edges * kPointerBytes;
  return extent_bytes < record_bytes ? extent_bytes : record_bytes;
}

DVSTOR_GRAPH_EXTENT_HD std::uint8_t level_for_required_bytes(
    std::uint32_t required_bytes, std::uint32_t graph_entry_capacity) {
  if (required_bytes < kHeaderBytes) return kUnknownLevel;
  const std::uint32_t payload_bytes = required_bytes - kHeaderBytes;
  if ((payload_bytes % kPointerBytes) != 0) return kUnknownLevel;
  return level_for_neighbors(payload_bytes / kPointerBytes,
                             graph_entry_capacity);
}

DVSTOR_GRAPH_EXTENT_HD AdaptationResult adapt_level(
    std::uint8_t cached_level, std::uint8_t observed_level) {
  if (observed_level == kUnknownLevel) return {cached_level, Adaptation::none};
  if (cached_level == kUnknownLevel) {
    return {observed_level, Adaptation::refined_unknown};
  }
  if (observed_level > cached_level) {
    return {observed_level, Adaptation::promoted};
  }
  if (static_cast<std::uint32_t>(observed_level) + 2u <= cached_level) {
    return {static_cast<std::uint8_t>(observed_level + 1u),
            Adaptation::demoted};
  }
  return {cached_level, Adaptation::none};
}

#undef DVSTOR_GRAPH_EXTENT_HD

}  // namespace vamana::graph_extent

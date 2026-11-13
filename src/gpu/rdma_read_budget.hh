#pragma once

#include <bit>
#include <cstdint>

namespace gpu {

constexpr std::uint32_t kPerStorageReadAtomicBudget = 2048;

constexpr std::uint32_t read_atomic_depth(
  std::uint32_t client_count,
  std::uint32_t qps_per_node,
  std::uint32_t device_limit) {
  if (client_count == 0 || qps_per_node == 0 || device_limit == 0) return 0;
  const std::uint64_t total =
    static_cast<std::uint64_t>(client_count) * qps_per_node;
  const std::uint64_t budgeted = kPerStorageReadAtomicBudget / total;
  const std::uint32_t capped = budgeted < device_limit
    ? static_cast<std::uint32_t>(budgeted) : device_limit;
  if (capped == 0) return 1;
  // DOCA accepts only power-of-two RDMA atomic depths.
  return std::bit_floor(capped);
}

}  // namespace gpu

#include <bit>
#include <cassert>
#include <cstdint>

#include "gpu/rdma_read_budget.hh"

int main() {
  using gpu::kPerStorageReadAtomicBudget;
  using gpu::read_atomic_depth;

  assert(read_atomic_depth(1, 32, 16) == 16);
  assert(read_atomic_depth(4, 32, 16) == 16);
  assert(read_atomic_depth(5, 32, 16) == 8);
  assert(read_atomic_depth(6, 32, 16) == 8);
  assert(read_atomic_depth(7, 32, 16) == 8);
  assert(read_atomic_depth(8, 32, 16) == 8);
  assert(read_atomic_depth(8, 16, 16) == 16);
  assert(read_atomic_depth(8, 32, 4) == 4);
  assert(read_atomic_depth(256, 32, 16) == 1);

  assert(read_atomic_depth(4, 32, 16) * 4 * 32 <= kPerStorageReadAtomicBudget);
  assert(read_atomic_depth(7, 32, 16) * 7 * 32 <= kPerStorageReadAtomicBudget);
  assert(read_atomic_depth(8, 32, 16) * 8 * 32 <= kPerStorageReadAtomicBudget);

  for (std::uint32_t clients = 1; clients <= 64; ++clients) {
    const auto depth = read_atomic_depth(clients, 32, 16);
    assert(depth > 0);
    assert(std::has_single_bit(depth));
  }

  assert(read_atomic_depth(0, 32, 16) == 0);
  assert(read_atomic_depth(8, 0, 16) == 0);
  assert(read_atomic_depth(8, 32, 0) == 0);
  return 0;
}

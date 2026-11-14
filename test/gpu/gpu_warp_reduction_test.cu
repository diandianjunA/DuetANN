#include <cuda_runtime.h>

#include "cuda_test_utils.hh"

using dvstor_test::check_cuda;

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/merge/warp_reduction.cuh"

namespace {

using gpu_search::persistent_kernel_detail::warp_reduce_add_sync;
using gpu_search::persistent_kernel_detail::warp_reduce_min_sync;
using u32 = std::uint32_t;

constexpr u32 kWarpWidth = 32;
constexpr u32 kThreadCount = 128;
constexpr u32 kFullWarp = 0xffffffffu;

__global__ void reduction_kernel(const u32* input, u32* full_sums,
                                 u32* full_minimums, u32* group_sums,
                                 u32* group_minimums) {
  const u32 index = threadIdx.x;
  const u32 lane = index & (kWarpWidth - 1u);
  const u32 value = input[index];
  full_sums[index] = warp_reduce_add_sync(kFullWarp, value);
  full_minimums[index] = warp_reduce_min_sync(kFullWarp, value);

  const bool active = lane % 5u != 4u;
  const u32 active_mask = __ballot_sync(kFullWarp, active);
  if (active) {
    const u32 group = __match_any_sync(active_mask, lane & 3u);
    group_sums[index] = warp_reduce_add_sync(group, value);
    group_minimums[index] = warp_reduce_min_sync(group, value);
  }
}

void expect_equal(u32 actual, u32 expected, const char* output, u32 index) {
  if (actual != expected) {
    throw std::runtime_error(
      std::string(output) + " mismatch at thread " + std::to_string(index) +
      ": actual=" + std::to_string(actual) +
      " expected=" + std::to_string(expected));
  }
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    std::vector<u32> input(kThreadCount);
    for (u32 index = 0; index < kThreadCount; ++index) {
      input[index] = (index * 73u + 19u) % 251u;
    }
    input[17] = 0;
    input[45] = ~u32{0};
    input[93] = 0x80000000u;

    u32* device_input = nullptr;
    u32* full_sums = nullptr;
    u32* full_minimums = nullptr;
    u32* group_sums = nullptr;
    u32* group_minimums = nullptr;
    const auto allocate = [](u32** pointer) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer),
                            kThreadCount * sizeof(u32)), "cudaMalloc");
    };
    allocate(&device_input);
    allocate(&full_sums);
    allocate(&full_minimums);
    allocate(&group_sums);
    allocate(&group_minimums);
    check_cuda(cudaMemcpy(device_input, input.data(), input.size() * sizeof(u32),
                          cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    check_cuda(cudaMemset(group_sums, 0, kThreadCount * sizeof(u32)),
               "cudaMemset group sums");
    check_cuda(cudaMemset(group_minimums, 0, kThreadCount * sizeof(u32)),
               "cudaMemset group minimums");

    reduction_kernel<<<1, kThreadCount>>>(
      device_input, full_sums, full_minimums, group_sums, group_minimums);
    check_cuda(cudaGetLastError(), "reduction kernel launch");
    check_cuda(cudaDeviceSynchronize(), "reduction kernel");

    std::vector<u32> actual_full_sums(kThreadCount);
    std::vector<u32> actual_full_minimums(kThreadCount);
    std::vector<u32> actual_group_sums(kThreadCount);
    std::vector<u32> actual_group_minimums(kThreadCount);
    const auto download = [](const u32* source, std::vector<u32>& destination) {
      check_cuda(cudaMemcpy(destination.data(), source,
                            destination.size() * sizeof(u32),
                            cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    };
    download(full_sums, actual_full_sums);
    download(full_minimums, actual_full_minimums);
    download(group_sums, actual_group_sums);
    download(group_minimums, actual_group_minimums);

    for (u32 index = 0; index < kThreadCount; ++index) {
      const u32 warp_begin = index & ~(kWarpWidth - 1u);
      const u32 lane = index & (kWarpWidth - 1u);
      u32 expected_full_sum = 0;
      u32 expected_full_minimum = ~u32{0};
      u32 expected_group_sum = 0;
      u32 expected_group_minimum = ~u32{0};
      for (u32 source_lane = 0; source_lane < kWarpWidth; ++source_lane) {
        const u32 source = input[warp_begin + source_lane];
        expected_full_sum += source;
        expected_full_minimum = std::min(expected_full_minimum, source);
        if (lane % 5u != 4u && source_lane % 5u != 4u &&
            (source_lane & 3u) == (lane & 3u)) {
          expected_group_sum += source;
          expected_group_minimum = std::min(expected_group_minimum, source);
        }
      }
      expect_equal(actual_full_sums[index], expected_full_sum,
                   "full sum", index);
      expect_equal(actual_full_minimums[index], expected_full_minimum,
                   "full minimum", index);
      if (lane % 5u == 4u) {
        expected_group_sum = 0;
        expected_group_minimum = 0;
      }
      expect_equal(actual_group_sums[index], expected_group_sum,
                   "group sum", index);
      expect_equal(actual_group_minimums[index], expected_group_minimum,
                   "group minimum", index);
    }

    (void)cudaFree(device_input);
    (void)cudaFree(full_sums);
    (void)cudaFree(full_minimums);
    (void)cudaFree(group_sums);
    (void)cudaFree(group_minimums);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

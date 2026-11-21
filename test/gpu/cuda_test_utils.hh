#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace duetann_test {

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace duetann_test

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu_search/types.hh"

int main() {
  gpu_search::TelemetryCounter counter{5};
  if (counter.fetch_add_single_writer(7, std::memory_order_relaxed) != 5 ||
      counter.load(std::memory_order_relaxed) != 12) {
    throw std::runtime_error("single-writer telemetry accumulation is incorrect");
  }

  std::vector<std::thread> writers;
  for (int thread = 0; thread < 4; ++thread) {
    writers.emplace_back([&counter] {
      for (int iteration = 0; iteration < 10000; ++iteration) {
        counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& writer : writers) writer.join();
  if (counter.load(std::memory_order_relaxed) != 40012) {
    throw std::runtime_error("ordinary atomic telemetry accumulation changed behavior");
  }
  std::cout << "telemetry counter: PASS\n";
}

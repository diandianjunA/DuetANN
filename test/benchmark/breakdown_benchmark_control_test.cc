#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tools/breakdown_benchmark/args.hh"
#include "tools/breakdown_benchmark/dynamic_id_pool.hh"
#include "tools/breakdown_benchmark/progress.hh"

namespace {

tools::breakdown_benchmark::Args parse(std::vector<std::string> values) {
  std::vector<char*> argv;
  argv.reserve(values.size());
  for (auto& value : values) argv.push_back(value.data());
  return tools::breakdown_benchmark::parse_args(
    static_cast<int>(argv.size()), argv.data());
}

bool parse_rejected(std::vector<std::string> values) {
  try {
    (void)parse(std::move(values));
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

void test_strict_numeric_args(const std::string& config_path) {
  const auto prefix = std::vector<std::string>{
    "benchmark", "--service-config", config_path,
    "--workload", "insert", "--report-json", "report.json",
  };
  auto with_option = [&](const std::string& option,
                         const std::string& value) {
    auto values = prefix;
    values.push_back(option);
    values.push_back(value);
    return values;
  };
  assert(parse_rejected(with_option("--client-threads", "-1")));
  assert(parse_rejected(with_option("--client-threads", "8threads")));
  assert(parse_rejected(with_option(
    "--client-threads", "184467440737095516160")));
  assert(parse_rejected(with_option("--read-ratio", "nan")));
  assert(parse_rejected(with_option("--read-ratio", "0.5tail")));
}

void test_strict_service_config() {
  const auto duplicate_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_duplicate_config.ini";
  {
    std::ofstream config(duplicate_path, std::ios::trunc);
    config << "[index]\nport = 1234\n[transport]\nport = 1235\n";
  }
  bool rejected = false;
  try {
    (void)tools::breakdown_benchmark::read_config(duplicate_path.string());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  std::filesystem::remove(duplicate_path);

  const auto boolean_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_boolean_config.ini";
  {
    std::ofstream config(boolean_path, std::ios::trunc);
    config << "initiator = perhaps\n";
  }
  rejected = false;
  try {
    (void)tools::breakdown_benchmark::build_service_argv(
      boolean_path.string());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  std::filesystem::remove(boolean_path);

  const auto profile_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_native_profile.ini";
  {
    std::ofstream config(profile_path, std::ios::trunc);
    config << "[profile]\nprofile = full\n"
              "[C1:update]\nc1 = off\n"
              "[C2:transfer]\nc2 = on\n"
              "[C3:query]\nc3 = off\n"
              "[cluster]\nis-server = false\nnode = compute:0\n"
              "[compute:0]\nhost = 192.168.6.125\nport = 2234\ngpu-device = 0\n"
              "[storage:0]\nhost = 192.168.6.202\nport = 1234\ngpu-device = 1\nis-server = true\n"
              "[storage:1]\nhost = 192.168.6.127\nport = 1235\n";
  }
  const auto args = tools::breakdown_benchmark::build_service_argv(
    profile_path.string());
  assert(!args.empty() && args.front() == "dvstor_breakdown_benchmark");
  const std::vector<std::pair<std::string, std::string>> expected{
    {"--c1", "off"}, {"--c2", "on"}, {"--c3", "off"}, {"--profile", "full"}};
  for (const auto& [option, value] : expected) {
    const auto position = std::find(args.begin(), args.end(), option);
    assert(position != args.end() && position + 1 != args.end());
    assert(*(position + 1) == value);
  }
  assert(std::find(args.begin(), args.end(), "--is-server") == args.end());
  auto gpu = std::find(args.begin(), args.end(), "--gpu-device");
  assert(gpu != args.end() && gpu + 1 != args.end() && *(gpu + 1) == "0");
  const auto servers = std::find(args.begin(), args.end(), "--servers");
  assert(servers != args.end() && servers + 2 != args.end());
  assert(*(servers + 1) == "192.168.6.202:1234");
  assert(*(servers + 2) == "192.168.6.127:1235");
  const auto peers = std::find(args.begin(), args.end(), "--storage-peers");
  assert(peers != args.end() && peers + 2 != args.end());
  assert(*(peers + 1) == "192.168.6.202:1234");
  assert(*(peers + 2) == "192.168.6.127:1235");
  {
    std::ofstream config(profile_path, std::ios::trunc);
    config << "node = storage:0\n"
              "[compute:0]\nhost = 192.168.6.125\nport = 2234\ngpu-device = 0\n"
              "[storage:0]\nhost = 192.168.6.202\nport = 1234\ngpu-device = 1\nis-server = true\n"
              "[storage:1]\nhost = 192.168.6.127\nport = 1235\n";
  }
  const auto storage_args = tools::breakdown_benchmark::build_service_argv(
    profile_path.string());
  gpu = std::find(storage_args.begin(), storage_args.end(), "--gpu-device");
  assert(gpu != storage_args.end() && gpu + 1 != storage_args.end() &&
         *(gpu + 1) == "1");
  assert(std::find(storage_args.begin(), storage_args.end(), "--is-server") !=
         storage_args.end());
  std::filesystem::remove(profile_path);

  const auto malformed_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_malformed_section.ini";
  {
    std::ofstream config(malformed_path, std::ios::trunc);
    config << "[gpu\n";
  }
  rejected = false;
  try {
    (void)tools::breakdown_benchmark::build_service_argv(
      malformed_path.string());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  std::filesystem::remove(malformed_path);
}

void test_rate_limited_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--profile-name", "04_gpu_persistent_gpunetio_baseline",
    "--system-variant-label", "baseline",
    "--workload", "mixed",
    "--warmup-seconds", "60",
    "--measure-seconds", "900",
    "--client-threads", "64",
    "--mixed-mode", "rate_limited",
    "--target-query-qps", "5000",
    "--target-insert-qps", "1000",
    "--performance-query-file", "queries.u8bin",
    "--write-insert-ratio", "1",
    "--write-upsert-ratio", "0",
    "--write-delete-ratio", "0",
    "--report-json", "report.json",
  });
  assert(args.client_threads == 64);
  assert(args.profile_name == "04_gpu_persistent_gpunetio_baseline");
  assert(args.system_variant_label == "baseline");
  assert(args.mixed_mode == "rate_limited");
  assert(args.target_query_qps == 5000.0);
  assert(args.target_write_qps == 1000.0);
}

void test_write_only_mixed_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--workload", "mixed",
    "--read-ratio", "0",
    "--mixed-mode", "fixed_threads",
    "--client-threads", "24",
    "--report-json", "report.json",
  });
  assert(args.read_ratio == 0.0);
  assert(args.performance_query_file.empty());
}

void test_write_rate_limited_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--workload", "mixed",
    "--warmup-seconds", "5",
    "--measure-seconds", "20",
    "--client-threads", "272",
    "--mixed-mode", "write_rate_limited",
    "--write-threads", "16",
    "--target-write-qps", "500",
    "--performance-query-file", "queries.u8bin",
    "--performance-query-mode", "cyclic",
    "--report-json", "report.json",
  });
  assert(args.mixed_mode == "write_rate_limited");
  assert(args.performance_query_mode == "cyclic");
  assert(args.write_threads == 16);
  assert(args.target_query_qps == 0.0);
  assert(args.target_write_qps == 500.0);
  auto invalid_mode = std::vector<std::string>{
    "benchmark", "--service-config", config_path,
    "--workload", "query", "--performance-query-file", "queries.u8bin",
    "--performance-query-mode", "random", "--report-json", "report.json"};
  assert(parse_rejected(std::move(invalid_mode)));
}

void test_removed_threshold_args_are_rejected(const std::string& config_path) {
  bool threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "insert",
      "--min-insert-qps", "-0.5",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "insert",
      "--max-stage2-p99-ms", "5000",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "mixed",
      "--warmup-seconds", "1",
      "--measure-seconds", "1",
      "--performance-query-file", "queries.u8bin",
      "--query-baseline-qps", "5000",
      "--min-query-baseline-ratio", "0.9",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void test_base_only_recall_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--workload", "mixed",
    "--warmup-seconds", "1",
    "--measure-seconds", "1",
    "--performance-query-file", "queries.u8bin",
    "--recall-query-file", "recall.u8bin",
    "--groundtruth-file", "truth.bin",
    "--recall-mode", "base_only",
    "--recall-base-id-limit", "100000000",
    "--report-json", "report.json",
  });
  assert(args.recall_mode == "base_only");
  assert(args.recall_base_id_limit == 100000000u);
}

void test_paced_dispatcher() {
  using namespace std::chrono_literals;
  using tools::breakdown_benchmark::PacedOperationDispatcher;
  using tools::breakdown_benchmark::PacedOperationKind;

  assert(PacedOperationDispatcher::scheduled_count(5000.0, 900) == 4'500'000);
  assert(PacedOperationDispatcher::scheduled_count(1000.0, 900) == 900'000);

  PacedOperationDispatcher dispatcher(30.0, 20.0);
  const auto start = std::chrono::steady_clock::now() + 5ms;
  dispatcher.start(start, start + 90ms);
  size_t queries = 0;
  size_t writes = 0;
  while (const auto claim = dispatcher.claim()) {
    if (claim->kind == PacedOperationKind::query) {
      ++queries;
    } else {
      ++writes;
    }
  }
  assert(queries == 3);
  assert(writes == 2);

  PacedOperationDispatcher write_only(0.0, 20.0);
  const auto write_start = std::chrono::steady_clock::now() + 5ms;
  write_only.start(write_start, write_start + 90ms);
  writes = 0;
  while (const auto claim = write_only.claim()) {
    assert(claim->kind == PacedOperationKind::write);
    ++writes;
  }
  assert(writes == 2);
}

void test_progress_deadline_records_zero_tail_and_excludes_drain() {
  using namespace std::chrono_literals;
  using tools::breakdown_benchmark::ProgressReporter;

  std::atomic<size_t> completed{1};
  std::atomic<size_t> writes{1};
  ProgressReporter reporter(
    "deadline-sampling", completed, 0, 1, nullptr, &writes, 250ms);

  const auto wait_deadline = std::chrono::steady_clock::now() + 2s;
  while (reporter.samples().size() < 4 &&
         std::chrono::steady_clock::now() < wait_deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const auto at_deadline = reporter.samples();
  assert(at_deadline.size() == 4);
  assert(at_deadline.front().interval_ops == 1);
  assert(at_deadline.back().elapsed_seconds == 1.0);
  assert(at_deadline.back().interval_ops == 0);
  assert(at_deadline.back().interval_writes == 0);

  // A synchronous operation completing while the caller drains after the
  // deadline must affect aggregate completion counters, not stability windows.
  completed.store(2, std::memory_order_relaxed);
  writes.store(2, std::memory_order_relaxed);
  reporter.finish();
  const auto after_drain = reporter.samples();
  assert(after_drain.size() == at_deadline.size());
  assert(after_drain.back().completed_ops == 1);
}

void test_query_latency_windows() {
  using namespace std::chrono_literals;
  using tools::breakdown_benchmark::QueryLatencyRecorder;

  const auto start = std::chrono::steady_clock::now();
  QueryLatencyRecorder recorder(2, start, 10);
  recorder.record(0, start + 100ms, start + 101ms);
  recorder.record(1, start + 200ms, start + 203ms);
  recorder.record(0, start + 5100ms, start + 5107ms);
  recorder.record(1, start + 10100ms, start + 10200ms);
  const auto windows = recorder.windows(5);
  assert(windows.size() == 2);
  assert(windows[0].samples == 2);
  assert(windows[0].average_ns == 2'000'000.0);
  assert(windows[0].p50_ns == 1'000'000);
  assert(windows[0].p95_ns == 1'000'000);
  assert(windows[0].p99_ns == 1'000'000);
  assert(windows[1].samples == 1);
  assert(windows[1].p99_ns == 7'000'000);

  const auto fill_reservoir = [&](QueryLatencyRecorder& output) {
    for (size_t sample = 0; sample < 1000; ++sample) {
      output.record(
        0, start + 100ms, start + 100ms +
          std::chrono::microseconds(sample + 1));
    }
  };
  QueryLatencyRecorder sampled_a(2, start, 10);
  QueryLatencyRecorder sampled_b(2, start, 10);
  fill_reservoir(sampled_a);
  fill_reservoir(sampled_b);
  const auto sampled_windows_a = sampled_a.windows(5);
  const auto sampled_windows_b = sampled_b.windows(5);
  assert(sampled_windows_a[0].samples == 1000);
  assert(sampled_windows_a[0].average_ns == 500'500.0);
  assert(sampled_windows_a[0].p50_ns > 350'000);
  assert(sampled_windows_a[0].p50_ns < 650'000);
  assert(sampled_windows_a[0].p99_ns > 800'000);
  assert(sampled_windows_a[0].p50_ns == sampled_windows_b[0].p50_ns);
  assert(sampled_windows_a[0].p95_ns == sampled_windows_b[0].p95_ns);
  assert(sampled_windows_a[0].p99_ns == sampled_windows_b[0].p99_ns);
  bool rejected_window = false;
  try {
    (void)sampled_a.windows(1);
  } catch (const std::invalid_argument&) {
    rejected_window = true;
  }
  assert(rejected_window);
}

void test_dynamic_delete_args(const std::string& config_path) {
  const auto state_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_dynamic_args.bin";
  const auto values = std::vector<std::string>{
    "benchmark", "--service-config", config_path,
    "--workload", "mixed", "--warmup-seconds", "1", "--measure-seconds", "1",
    "--client-threads", "8", "--read-ratio", "0.5", "--mixed-mode", "fixed_threads",
    "--performance-query-file", "queries.u8bin",
    "--write-insert-ratio", "0.5", "--write-upsert-ratio", "0",
    "--write-delete-ratio", "0.5", "--mixed-delete-mode", "inserted",
    "--dynamic-state-file", state_path.string(), "--seed-insert-count", "16",
    "--hotspot-id-count", "4096",
    "--maintenance-timeout-ms", "900000", "--skip-maintenance-drain",
    "--report-json", "report.json",
  };
  const auto args = parse(values);
  assert(args.mixed_delete_mode == "inserted");
  assert(args.dynamic_state_file == state_path.string());
  assert(args.seed_insert_count == 16);
  assert(args.hotspot_id_count == 4096);
  assert(args.maintenance_timeout_ms == 900000);
  assert(args.skip_maintenance_drain);
  auto rate_limited_values = values;
  for (auto& value : rate_limited_values) {
    if (value == "fixed_threads") value = "write_rate_limited";
  }
  rate_limited_values.insert(rate_limited_values.end(), {
    "--write-threads", "2", "--target-write-qps", "200"});
  const auto rate_limited_args = parse(rate_limited_values);
  assert(rate_limited_args.mixed_mode == "write_rate_limited");
  assert(rate_limited_args.mixed_delete_mode == "inserted");
  auto rejected = values;
  for (auto& value : rejected) {
    if (value == "fixed_threads") value = "probability";
  }
  assert(parse_rejected(std::move(rejected)));
  rejected = values;
  for (auto& value : rejected) {
    if (value == "0.5") value = "0";
  }
  assert(parse_rejected(std::move(rejected)));
}

void test_single_reader_args(const std::string& config_path) {
  const auto state_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_single_reader_args.bin";
  const auto values = std::vector<std::string>{
    "benchmark", "--service-config", config_path,
    "--workload", "mixed", "--warmup-seconds", "0", "--measure-seconds", "0",
    "--warmup-ops", "64", "--measure-ops", "256",
    "--client-threads", "8", "--read-ratio", "0.5",
    "--mixed-mode", "single_reader", "--single-reader-write-cycles-per-query", "4",
    "--performance-query-file", "queries.u8bin",
    "--write-insert-ratio", "0.5", "--write-upsert-ratio", "0",
    "--write-delete-ratio", "0.5", "--mixed-delete-mode", "inserted",
    "--dynamic-state-file", state_path.string(), "--seed-insert-count", "4096",
    "--hotspot-id-count", "4096", "--report-json", "report.json",
  };
  const auto args = parse(values);
  assert(args.mixed_mode == "single_reader");
  assert(args.single_reader_write_cycles_per_query == 4);
  assert(args.client_threads == 8);
  assert(args.measure_seconds == 0 && args.measure_ops == 256);

  auto rejected = values;
  for (auto& value : rejected) {
    if (value == "single_reader") value = "fixed_threads";
  }
  assert(parse_rejected(std::move(rejected)));
  rejected = values;
  for (auto& value : rejected) {
    if (value == "4096") {
      value = "4095";
      break;
    }
  }
  assert(parse_rejected(std::move(rejected)));
  rejected = values;
  for (auto& value : rejected) {
    if (value == "4") value = "0";
  }
  assert(parse_rejected(std::move(rejected)));
  rejected = values;
  for (auto& value : rejected) {
    if (value == "4") value = "4097";
  }
  assert(parse_rejected(std::move(rejected)));
}

void test_dynamic_id_pool() {
  const auto path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_dynamic_id_pool_test.bin";
  std::filesystem::remove(path);
  tools::breakdown_benchmark::DynamicIdPool pool(path);
  assert(!pool.state_file_exists());
  pool.push(100);
  pool.push(101);
  assert(pool.pop(std::chrono::steady_clock::now()) == 100);
  pool.save();
  assert(pool.state_file_exists());
  tools::breakdown_benchmark::DynamicIdPool restored(path);
  restored.load(1000);
  assert(restored.size() == 1);
  restored.push(102);
  assert(restored.pop(std::chrono::steady_clock::now()) == 101);
  assert(restored.pop(std::chrono::steady_clock::now()) == 102);
  assert(!restored.pop(std::chrono::steady_clock::now()).has_value());
  restored.push(103);
  restored.push(104);
  assert(restored.replace(103, 105));
  assert(!restored.replace(999, 106));
  assert(restored.pop(std::chrono::steady_clock::now()) == 105);
  assert(restored.pop(std::chrono::steady_clock::now()) == 104);
  restored.save();
  std::filesystem::remove(path);
}

}  // namespace

int main() {
  const auto config_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_benchmark_control_test.ini";
  {
    std::ofstream config(config_path, std::ios::trunc);
    config << "insert-start-id = 100000000\n";
  }
  test_rate_limited_args(config_path.string());
  test_write_only_mixed_args(config_path.string());
  test_write_rate_limited_args(config_path.string());
  test_removed_threshold_args_are_rejected(config_path.string());
  test_base_only_recall_args(config_path.string());
  test_strict_numeric_args(config_path.string());
  test_strict_service_config();
  test_paced_dispatcher();
  test_progress_deadline_records_zero_tail_and_excludes_drain();
  test_query_latency_windows();
  test_dynamic_delete_args(config_path.string());
  test_single_reader_args(config_path.string());
  test_dynamic_id_pool();
  std::filesystem::remove(config_path);
  return 0;
}

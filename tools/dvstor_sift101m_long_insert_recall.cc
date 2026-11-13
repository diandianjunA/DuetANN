#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execinfo.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "common/configuration.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "service/compute_service.hh"
#include "../experiment/contribution/c1_locality/locality_sampler.hh"
#include "tools/breakdown_benchmark/args.hh"
#include "tools/breakdown_benchmark/progress.hh"

#ifdef DVSTOR_LONG_INSERT_WITH_FAISS_CPU
#include <faiss/IndexFlat.h>
#endif

namespace {

using tools::breakdown_benchmark::ProgressReporter;

struct Args {
  std::string service_config_path;
  std::filesystem::path insert_file;
  uint64_t insert_start_id{100000000ull};
  size_t insert_count{1000000};
  size_t insert_row_offset{0};
  size_t insert_threads{16};
  size_t insert_batch_size{16};
  size_t reset_breakdown_every{50000};

  std::filesystem::path query_file;
  std::filesystem::path baseline_groundtruth_file;
  std::filesystem::path groundtruth_file;
  std::filesystem::path base_vector_file;
  bool dynamic_groundtruth{false};
  size_t recall_queries{10000};
  u32 recall_k{10};
  size_t recall_sample_interval_seconds{0};
  std::filesystem::path recall_samples_json_path;
  std::filesystem::path locality_samples_json_path;
  size_t locality_sample_interval_ms{0};
  std::filesystem::path groundtruth_output_dir;
  size_t groundtruth_precompute_step{500000};
  bool groundtruth_precompute_only{false};
  bool require_precomputed_groundtruth{false};
  size_t settle_seconds{300};
  size_t self_recall_queries{100};
  u32 self_recall_k{10};

  std::filesystem::path report_json_path;
  std::filesystem::path report_text_path;
};

struct VectorRows {
  VectorDType dtype{VectorDType::float32};
  u32 dim{};
  size_t count{};
  size_t vector_bytes{};
  size_t row_stride{};
  size_t row_payload_offset{};
  std::vector<byte_t> raw;

  const byte_t* row_payload(size_t index) const {
    return raw.data() + index * row_stride + row_payload_offset;
  }

  vec<element_t> decode_row(size_t index) const {
    vec<element_t> values(dim);
    decode_storage_vector_to_float(row_payload(index), dtype, dim, values.data());
    return values;
  }
};

struct GroundTruth {
  u32 rows{};
  u32 top_k{};
  std::vector<u32> ids;

  const u32* row(size_t index) const {
    return ids.data() + static_cast<size_t>(top_k) * index;
  }
};

struct RecallResult {
  double recall{};
  double base_recall{};
  double inserted_recall{};
  size_t queries{};
  u32 k{};
  size_t base_truth{};
  size_t inserted_truth{};
};

struct SelfRecallSnapshot {
  RecallResult summary;
  std::vector<std::vector<node_t>> result_ids;
  std::vector<node_t> missed_insert_ids;
};

struct TimedRecallSample {
  size_t index{};
  double scheduled_elapsed_seconds{};
  double start_elapsed_seconds{};
  double end_elapsed_seconds{};
  size_t inserted_ack_at_start{};
  size_t inserted_ack_at_end{};
  size_t groundtruth_inserted_count{};
  double groundtruth_build_seconds{};
  bool groundtruth_reused{};
  RecallResult recall;
};

struct InsertStats {
  size_t attempted{};
  size_t inserted{};
  size_t failed{};
  double duration_seconds{};
  double inserts_per_second{};
  std::vector<TimedRecallSample> recall_samples;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void signal_handler(int signal) {
  void* frames[64];
  const int count = backtrace(frames, 64);
  const char header[] = "\n[sift101m-long-insert] fatal signal, backtrace:\n";
  const ssize_t ignored = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  _exit(128 + signal);
}

std::string extension_lower(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext;
}

uint64_t path_size(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    fail("cannot stat " + path.string() + ": " + ec.message());
  }
  return size;
}

u32 read_u32(std::ifstream& input, const std::string& what) {
  u32 value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    fail("failed to read " + what);
  }
  return value;
}

u32 load_u32_unaligned(const byte_t* data) {
  u32 value{};
  std::memcpy(&value, data, sizeof(value));
  return value;
}

void read_exact(std::ifstream& input, void* dst, size_t bytes, const std::string& what) {
  input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!input) {
    fail("failed to read " + what);
  }
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  if (path.empty()) {
    return;
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output(path);
  output << text;
  if (!output) {
    fail("failed to write " + path.string());
  }
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& root) {
  if (path.empty()) {
    fail("--report-json is required");
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output(path);
  output << root.dump(2) << '\n';
  if (!output) {
    fail("failed to write " + path.string());
  }
}

void write_groundtruth_file(
    const std::filesystem::path& path, const GroundTruth& groundtruth) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) fail("failed to open dynamic groundtruth output");
  output.write(reinterpret_cast<const char*>(&groundtruth.rows), sizeof(u32));
  output.write(reinterpret_cast<const char*>(&groundtruth.top_k), sizeof(u32));
  output.write(reinterpret_cast<const char*>(groundtruth.ids.data()),
               groundtruth.ids.size() * sizeof(u32));
  if (!output) fail("failed to write dynamic groundtruth");
}

VectorRows read_header_vector_rows(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open vector file: " + path.string());
  }

  const u32 count = read_u32(input, "vector count");
  const u32 dim = read_u32(input, "vector dim");
  if (count == 0 || dim == 0) {
    fail("invalid vector header in " + path.string());
  }

  VectorRows rows;
  const auto inferred_dtype = infer_vector_dtype_from_path(filepath_t{path});
  if (!inferred_dtype.has_value()) {
    fail("ambiguous or unsupported vector suffix; use .fbin, .u8bin, or "
         ".i8bin: " + path.string());
  }
  rows.dtype = *inferred_dtype;
  rows.dim = dim;
  rows.count = count;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  rows.row_stride = rows.vector_bytes;
  rows.row_payload_offset = 0;

  const uint64_t expected_size = 2ull * sizeof(u32) + static_cast<uint64_t>(rows.count) * rows.vector_bytes;
  const uint64_t actual_size = path_size(path);
  if (actual_size != expected_size) {
    fail("unexpected vector file size for " + path.string() + ": got " + std::to_string(actual_size) +
         ", expected " + std::to_string(expected_size));
  }

  rows.raw.resize(static_cast<size_t>(rows.count) * rows.vector_bytes);
  read_exact(input, rows.raw.data(), rows.raw.size(), "vector payload");
  return rows;
}

VectorRows read_vecs_vector_rows(const std::filesystem::path& path, VectorDType dtype) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open vecs file: " + path.string());
  }

  const uint64_t size = path_size(path);
  const u32 dim = read_u32(input, "vecs dim");
  if (dim == 0) {
    fail("invalid vecs dim in " + path.string());
  }

  const size_t vector_bytes = vector_dtype_bytes(dtype, dim);
  const size_t row_stride = sizeof(u32) + vector_bytes;
  if (size % row_stride != 0) {
    fail("vecs file size is not divisible by row size: " + path.string());
  }

  VectorRows rows;
  rows.dtype = dtype;
  rows.dim = dim;
  rows.count = static_cast<size_t>(size / row_stride);
  rows.vector_bytes = vector_bytes;
  rows.row_stride = row_stride;
  rows.row_payload_offset = sizeof(u32);
  rows.raw.resize(static_cast<size_t>(size));

  input.seekg(0, std::ios::beg);
  read_exact(input, rows.raw.data(), rows.raw.size(), "vecs payload");
  for (size_t row = 0; row < rows.count; ++row) {
    const u32 row_dim = load_u32_unaligned(rows.raw.data() + row * row_stride);
    if (row_dim != dim) {
      fail("vecs row dim mismatch in " + path.string() + " row " + std::to_string(row));
    }
  }
  return rows;
}

VectorRows read_vector_rows(const std::filesystem::path& path) {
  const std::string ext = extension_lower(path);
  if (ext == ".bvecs") {
    return read_vecs_vector_rows(path, VectorDType::uint8);
  }
  if (ext == ".fvecs") {
    return read_vecs_vector_rows(path, VectorDType::float32);
  }
  if (ext == ".u8bin" || ext == ".i8bin" || ext == ".fbin" || ext == ".bin") {
    return read_header_vector_rows(path);
  }
  fail("unsupported vector file extension: " + path.string());
}

GroundTruth read_project_groundtruth_bin(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open groundtruth file: " + path.string());
  }

  GroundTruth gt;
  gt.rows = read_u32(input, "groundtruth rows");
  gt.top_k = read_u32(input, "groundtruth top_k");
  if (gt.rows == 0 || gt.top_k == 0) {
    fail("invalid groundtruth header: " + path.string());
  }

  const uint64_t expected_size = 2ull * sizeof(u32) +
                                 static_cast<uint64_t>(gt.rows) * gt.top_k * sizeof(u32);
  const uint64_t actual_size = path_size(path);
  if (actual_size != expected_size) {
    fail("unexpected groundtruth file size for " + path.string() + ": got " + std::to_string(actual_size) +
         ", expected " + std::to_string(expected_size));
  }

  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);
  read_exact(input, gt.ids.data(), gt.ids.size() * sizeof(u32), "groundtruth ids");
  return gt;
}

GroundTruth read_ivecs_groundtruth(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open ivecs groundtruth file: " + path.string());
  }
  const uint64_t size = path_size(path);
  const u32 top_k = read_u32(input, "ivecs top_k");
  if (top_k == 0) {
    fail("invalid ivecs top_k in " + path.string());
  }
  const uint64_t row_bytes = sizeof(u32) + static_cast<uint64_t>(top_k) * sizeof(u32);
  if (size % row_bytes != 0) {
    fail("ivecs file size is not divisible by row size: " + path.string());
  }

  GroundTruth gt;
  gt.rows = static_cast<u32>(size / row_bytes);
  gt.top_k = top_k;
  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);

  input.seekg(0, std::ios::beg);
  for (u32 row = 0; row < gt.rows; ++row) {
    const u32 row_top_k = read_u32(input, "ivecs row top_k");
    if (row_top_k != top_k) {
      fail("ivecs top_k mismatch in row " + std::to_string(row));
    }
    read_exact(input, gt.ids.data() + static_cast<size_t>(row) * gt.top_k,
               static_cast<size_t>(gt.top_k) * sizeof(u32), "ivecs row ids");
  }
  return gt;
}

GroundTruth read_groundtruth(const std::filesystem::path& path) {
  const std::string ext = extension_lower(path);
  if (ext == ".ivecs") {
    return read_ivecs_groundtruth(path);
  }
  if (ext == ".bin") {
    return read_project_groundtruth_bin(path);
  }
  fail("unsupported groundtruth file extension: " + path.string());
}

double recall_at_k(const std::vector<u32>& results, const u32* truth, u32 k,
                   u32 first_inserted_id,
                   size_t* base_hits, size_t* inserted_hits,
                   size_t* base_truth, size_t* inserted_truth) {
  u32 hits = 0;
  *base_hits = 0;
  *inserted_hits = 0;
  *base_truth = 0;
  *inserted_truth = 0;
  for (u32 j = 0; j < k; ++j) {
    ++(truth[j] >= first_inserted_id ? *inserted_truth : *base_truth);
  }
  const size_t result_count = std::min<size_t>(results.size(), k);
  for (size_t i = 0; i < result_count; ++i) {
    for (u32 j = 0; j < k; ++j) {
      if (results[i] == truth[j]) {
        ++hits;
        ++(truth[j] >= first_inserted_id ? *inserted_hits : *base_hits);
        break;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --service-config <ini> --insert-file <path> "
      << "--report-json <path> [options]\n"
      << "  --insert-start-id <id>          First inserted ID, default 100000000\n"
      << "  --insert-count <n>              Number of rows to insert, default 1000000; 0 means all rows\n"
      << "  --insert-row-offset <n>         First row in insert file, default 0\n"
      << "  --insert-threads <n>            Concurrent application threads, default 16\n"
      << "  --insert-batch-size <n>         Vectors per ComputeService::insert call, default 16\n"
      << "  --baseline-groundtruth-file <p> Optional pre-insert GT, e.g. gnd/idx_100M.ivecs\n"
      << "  --base-vector-file <path>       Base vectors used to refresh old GT distances\n"
      << "  --dynamic-groundtruth           Recompute exact Top-K against base GT and inserted vectors\n"
      << "  --recall-queries <n>            Queries for recall, default 10000; 0 means all\n"
      << "  --recall-k <k>                  Recall K, default 10\n"
      << "  --recall-sample-interval-seconds <n> Sample baseline-GT recall during insertion; 0 disables\n"
      << "  --recall-samples-json <path>    Atomic checkpoint for periodic recall samples\n"
      << "  --locality-samples-json <path>  Atomic checkpoint for physical-locality telemetry\n"
      << "  --locality-sample-interval-ms <n> Locality sample interval; 0 disables, default 0\n"
      << "  --groundtruth-output-dir <path> Optional dynamic GT files for independent audit\n"
      << "  --groundtruth-precompute-step <n> Inserted-vector interval for ahead-of-time dynamic GT, default 500000\n"
      << "  --groundtruth-precompute-only   Generate dynamic GT without starting the storage service\n"
      << "  --require-precomputed-groundtruth  Fail rather than recomputing a missing dynamic GT file\n"
      << "  --settle-seconds <n>            Sleep after insertion before post recall, default 300\n"
      << "  --self-recall-queries <n>       Inserted vectors sampled for self-hit, default 100; 0 disables\n"
      << "  --self-recall-k <k>             Search K for inserted-vector self-hit, default 10\n"
      << "  --reset-breakdown-every <n>     Clear samples every n attempted inserts, default 50000; 0 disables\n"
      << "  --report-text <path>            Optional text summary\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fail(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (flag == "--service-config") {
      args.service_config_path = require_value("--service-config");
    } else if (flag == "--insert-file") {
      args.insert_file = require_value("--insert-file");
    } else if (flag == "--insert-start-id") {
      args.insert_start_id = std::stoull(require_value("--insert-start-id"));
    } else if (flag == "--insert-count") {
      args.insert_count = std::stoull(require_value("--insert-count"));
    } else if (flag == "--insert-row-offset") {
      args.insert_row_offset = std::stoull(require_value("--insert-row-offset"));
    } else if (flag == "--insert-threads") {
      args.insert_threads = std::stoull(require_value("--insert-threads"));
    } else if (flag == "--insert-batch-size") {
      args.insert_batch_size = std::stoull(require_value("--insert-batch-size"));
    } else if (flag == "--query-file") {
      args.query_file = require_value("--query-file");
    } else if (flag == "--baseline-groundtruth-file") {
      args.baseline_groundtruth_file = require_value("--baseline-groundtruth-file");
    } else if (flag == "--groundtruth-file") {
      args.groundtruth_file = require_value("--groundtruth-file");
    } else if (flag == "--base-vector-file") {
      args.base_vector_file = require_value("--base-vector-file");
    } else if (flag == "--dynamic-groundtruth") {
      args.dynamic_groundtruth = true;
    } else if (flag == "--recall-queries") {
      args.recall_queries = std::stoull(require_value("--recall-queries"));
    } else if (flag == "--recall-k") {
      args.recall_k = static_cast<u32>(std::stoul(require_value("--recall-k")));
    } else if (flag == "--recall-sample-interval-seconds") {
      args.recall_sample_interval_seconds =
        std::stoull(require_value("--recall-sample-interval-seconds"));
    } else if (flag == "--recall-samples-json") {
      args.recall_samples_json_path = require_value("--recall-samples-json");
    } else if (flag == "--locality-samples-json") {
      args.locality_samples_json_path = require_value("--locality-samples-json");
    } else if (flag == "--locality-sample-interval-ms") {
      args.locality_sample_interval_ms =
        std::stoull(require_value("--locality-sample-interval-ms"));
    } else if (flag == "--groundtruth-output-dir") {
      args.groundtruth_output_dir = require_value("--groundtruth-output-dir");
    } else if (flag == "--groundtruth-precompute-step") {
      args.groundtruth_precompute_step =
        std::stoull(require_value("--groundtruth-precompute-step"));
    } else if (flag == "--groundtruth-precompute-only") {
      args.groundtruth_precompute_only = true;
    } else if (flag == "--require-precomputed-groundtruth") {
      args.require_precomputed_groundtruth = true;
    } else if (flag == "--settle-seconds") {
      args.settle_seconds = std::stoull(require_value("--settle-seconds"));
    } else if (flag == "--self-recall-queries") {
      args.self_recall_queries =
        std::stoull(require_value("--self-recall-queries"));
    } else if (flag == "--self-recall-k") {
      args.self_recall_k = static_cast<u32>(
        std::stoul(require_value("--self-recall-k")));
    } else if (flag == "--reset-breakdown-every") {
      args.reset_breakdown_every = std::stoull(require_value("--reset-breakdown-every"));
    } else if (flag == "--report-json") {
      args.report_json_path = require_value("--report-json");
    } else if (flag == "--report-text") {
      args.report_text_path = require_value("--report-text");
    } else if (flag == "--help" || flag == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      fail("unknown argument: " + flag);
    }
  }

  if (args.service_config_path.empty()) fail("--service-config is required");
  if (args.insert_file.empty()) fail("--insert-file is required");
  if ((!args.groundtruth_file.empty() ||
       !args.baseline_groundtruth_file.empty()) && args.query_file.empty()) {
    fail("--query-file is required when a groundtruth file is used");
  }
  if (args.report_json_path.empty()) fail("--report-json is required");
  if (args.dynamic_groundtruth &&
      (args.baseline_groundtruth_file.empty() || args.base_vector_file.empty())) {
    fail("dynamic groundtruth requires --baseline-groundtruth-file and "
         "--base-vector-file");
  }
  if (args.dynamic_groundtruth && args.groundtruth_precompute_step == 0) {
    fail("--groundtruth-precompute-step must be > 0");
  }
  if ((args.groundtruth_precompute_only ||
       args.require_precomputed_groundtruth) &&
      (!args.dynamic_groundtruth || args.groundtruth_output_dir.empty())) {
    fail("precomputed groundtruth requires --dynamic-groundtruth and "
         "--groundtruth-output-dir");
  }
  if (args.groundtruth_precompute_only &&
      args.require_precomputed_groundtruth) {
    fail("--groundtruth-precompute-only cannot be combined with "
         "--require-precomputed-groundtruth");
  }
#ifndef DVSTOR_LONG_INSERT_WITH_FAISS_CPU
  if (args.dynamic_groundtruth) {
    fail("dynamic groundtruth requires a build with FaissCPU");
  }
#endif
  if (args.recall_sample_interval_seconds != 0 &&
      (args.baseline_groundtruth_file.empty() ||
       args.recall_samples_json_path.empty())) {
    fail("periodic recall requires --baseline-groundtruth-file and "
         "--recall-samples-json");
  }
  if ((args.locality_sample_interval_ms == 0) !=
      args.locality_samples_json_path.empty()) {
    fail("--locality-samples-json and --locality-sample-interval-ms must be "
         "provided together");
  }
  if (args.insert_threads == 0) fail("--insert-threads must be > 0");
  if (args.insert_batch_size == 0) fail("--insert-batch-size must be > 0");
  if (args.recall_k == 0) fail("--recall-k must be > 0");
  if (args.self_recall_k == 0) fail("--self-recall-k must be > 0");
  if (args.insert_start_id > std::numeric_limits<node_t>::max()) {
    fail("--insert-start-id exceeds node_t range");
  }
  return args;
}

RecallResult run_recall(ComputeService& service,
                        const std::string& label,
                        const VectorRows& queries,
                        const GroundTruth& gt,
                        size_t requested_queries,
                        u32 requested_k,
                        u32 first_inserted_id) {
  if (queries.dim != service.config().dim) {
    fail("query dim mismatch: query=" + std::to_string(queries.dim) +
         " config=" + std::to_string(service.config().dim));
  }
  if (gt.rows > queries.count) {
    fail("query/groundtruth row count mismatch: query=" + std::to_string(queries.count) +
         " gt=" + std::to_string(gt.rows));
  }
  if (requested_k > gt.top_k) {
    fail("recall k exceeds groundtruth top_k");
  }

  const size_t recall_queries = requested_queries == 0
                                  ? queries.count
                                  : std::min<size_t>(requested_queries, queries.count);
  if (gt.rows < recall_queries) {
    fail("groundtruth has fewer rows than the recall query count");
  }
  std::atomic<size_t> completed{0};
  ProgressReporter reporter(label, completed, recall_queries, 0);
  double total_recall = 0.0;
  size_t total_base_hits = 0;
  size_t total_inserted_hits = 0;
  size_t total_base_truth = 0;
  size_t total_inserted_truth = 0;
  for (size_t qi = 0; qi < recall_queries; ++qi) {
    const auto results = service.search_raw(queries.dtype, queries.row_payload(qi),
                                            queries.dim, requested_k);
    std::vector<u32> result_ids;
    result_ids.reserve(results.size());
    for (const node_t id : results) {
      result_ids.push_back(static_cast<u32>(id));
    }
    size_t base_hits = 0;
    size_t inserted_hits = 0;
    size_t base_truth = 0;
    size_t inserted_truth = 0;
    total_recall += recall_at_k(
      result_ids, gt.row(qi), requested_k, first_inserted_id,
      &base_hits, &inserted_hits, &base_truth, &inserted_truth);
    total_base_hits += base_hits;
    total_inserted_hits += inserted_hits;
    total_base_truth += base_truth;
    total_inserted_truth += inserted_truth;
    completed.fetch_add(1, std::memory_order_relaxed);
  }
  reporter.finish();

  RecallResult result;
  result.queries = recall_queries;
  result.k = requested_k;
  result.recall = recall_queries == 0 ? 0.0 : total_recall / static_cast<double>(recall_queries);
  result.base_truth = total_base_truth;
  result.inserted_truth = total_inserted_truth;
  result.base_recall = total_base_truth == 0 ? 0.0
    : static_cast<double>(total_base_hits) / static_cast<double>(total_base_truth);
  result.inserted_recall = total_inserted_truth == 0 ? 0.0
    : static_cast<double>(total_inserted_hits) / static_cast<double>(total_inserted_truth);
  std::cerr << "[sift101m-long-insert][recall] " << label << " recall@" << requested_k
            << "=" << result.recall << " base=" << result.base_recall
            << " inserted=" << result.inserted_recall
            << " queries=" << result.queries << std::endl;
  return result;
}

SelfRecallSnapshot run_inserted_self_recall(
    ComputeService& service, const std::string& label,
    const VectorRows& insert_rows, size_t row_offset,
    uint64_t insert_start_id, size_t inserted_count,
    size_t requested_queries, u32 requested_k) {
  const size_t query_count = std::min(requested_queries, inserted_count);
  std::atomic<size_t> completed{0};
  ProgressReporter reporter(label, completed, query_count, 0);
  size_t hits = 0;
  std::vector<std::vector<node_t>> captured_results;
  std::vector<node_t> missed_insert_ids;
  captured_results.reserve(query_count);
  for (size_t sample = 0; sample < query_count; ++sample) {
    const size_t logical_row = query_count == inserted_count
      ? sample
      : sample * inserted_count / query_count;
    const node_t expected = static_cast<node_t>(insert_start_id + logical_row);
    const auto results = service.search_raw(
      insert_rows.dtype, insert_rows.row_payload(row_offset + logical_row),
      insert_rows.dim, requested_k);
    if (std::find(results.begin(), results.end(), expected) != results.end()) {
      ++hits;
    } else {
      missed_insert_ids.push_back(expected);
    }
    captured_results.emplace_back(results.begin(), results.end());
    completed.fetch_add(1, std::memory_order_relaxed);
  }
  reporter.finish();
  const RecallResult result{
    .recall = query_count == 0 ? 0.0 :
      static_cast<double>(hits) / static_cast<double>(query_count),
    .queries = query_count,
    .k = requested_k,
  };
  std::cerr << "[sift101m-long-insert][self-hit] " << label
            << " hit@" << requested_k << "=" << result.recall
            << " queries=" << result.queries << std::endl;
  return SelfRecallSnapshot{
    .summary = result,
    .result_ids = std::move(captured_results),
    .missed_insert_ids = std::move(missed_insert_ids),
  };
}

double result_overlap_at_k(const SelfRecallSnapshot& first,
                           const SelfRecallSnapshot& second) {
  if (first.result_ids.size() != second.result_ids.size()) {
    fail("self-recall result snapshots have different query counts");
  }
  if (first.result_ids.empty()) return 0.0;
  double total = 0.0;
  for (size_t query = 0; query < first.result_ids.size(); ++query) {
    const auto& lhs = first.result_ids[query];
    const auto& rhs = second.result_ids[query];
    const size_t denominator = std::min(lhs.size(), rhs.size());
    if (denominator == 0) continue;
    size_t matches = 0;
    for (const node_t id : lhs) {
      matches += std::find(rhs.begin(), rhs.end(), id) != rhs.end();
    }
    total += static_cast<double>(matches) /
      static_cast<double>(denominator);
  }
  return total / static_cast<double>(first.result_ids.size());
}

#ifdef DVSTOR_LONG_INSERT_WITH_FAISS_CPU
GroundTruth build_dynamic_groundtruth(
    const VectorRows& queries, const VectorRows& base_rows,
    const VectorRows& insert_rows, size_t insert_row_offset,
    uint64_t insert_start_id, size_t inserted_count,
    const GroundTruth& base_groundtruth, size_t requested_queries, u32 k) {
  if (requested_queries == 0) requested_queries = queries.count;
  if (queries.dim != base_rows.dim || queries.dim != insert_rows.dim) {
    fail("dynamic groundtruth dimension mismatch");
  }
  if (inserted_count == 0 || requested_queries == 0 ||
      requested_queries > queries.count ||
      requested_queries > base_groundtruth.rows || k == 0 ||
      k > base_groundtruth.top_k) {
    fail("dynamic groundtruth query/K range is invalid");
  }
  if (insert_row_offset + inserted_count > insert_rows.count) {
    fail("dynamic groundtruth insert range exceeds its input");
  }
  if (inserted_count > std::numeric_limits<faiss::idx_t>::max()) {
    fail("dynamic groundtruth exceeds Faiss index capacity");
  }
  const size_t dim = queries.dim;
  const size_t query_components = requested_queries * dim;
  const size_t inserted_components = inserted_count * dim;
  if (dim != 0 && (query_components / dim != requested_queries ||
                   inserted_components / dim != inserted_count)) {
    fail("dynamic groundtruth matrix size overflows");
  }
  std::vector<float> query_values(query_components);
  for (size_t query = 0; query < requested_queries; ++query) {
    decode_storage_vector_to_float(
      queries.row_payload(query), queries.dtype, dim,
      query_values.data() + query * dim);
  }
  std::vector<float> inserted_values(inserted_components);
  for (size_t row = 0; row < inserted_count; ++row) {
    decode_storage_vector_to_float(
      insert_rows.row_payload(insert_row_offset + row), insert_rows.dtype, dim,
      inserted_values.data() + row * dim);
  }

  faiss::IndexFlatL2 inserted_index(dim);
  if (inserted_count != 0) {
    inserted_index.add(inserted_count, inserted_values.data());
  }
  std::vector<float> inserted_distances(requested_queries * k);
  std::vector<faiss::idx_t> inserted_labels(requested_queries * k);
  if (inserted_count != 0) {
    inserted_index.search(
      requested_queries, query_values.data(), k,
      inserted_distances.data(), inserted_labels.data());
  }

  GroundTruth result;
  result.rows = requested_queries;
  result.top_k = k;
  result.ids.resize(requested_queries * k);
  std::vector<float> base_values(dim);
  std::vector<std::pair<float, u32>> candidates;
  candidates.reserve(2 * k);
  for (size_t query = 0; query < requested_queries; ++query) {
    candidates.clear();
    const float* query_values_row = query_values.data() + query * dim;
    for (u32 rank = 0; rank < k; ++rank) {
      const u32 id = base_groundtruth.row(query)[rank];
      if (id >= base_rows.count) {
        fail("base groundtruth ID exceeds base vector count");
      }
      decode_storage_vector_to_float(
        base_rows.row_payload(id), base_rows.dtype, dim, base_values.data());
      float distance = 0.0f;
      for (size_t component = 0; component < dim; ++component) {
        const float delta = query_values_row[component] - base_values[component];
        distance += delta * delta;
      }
      candidates.emplace_back(distance, id);
    }
    for (u32 rank = 0; rank < k; ++rank) {
      const faiss::idx_t label = inserted_labels[query * k + rank];
      if (label < 0 || static_cast<uint64_t>(label) >= inserted_count) {
        fail("Faiss returned an invalid inserted-vector label");
      }
      candidates.emplace_back(
        inserted_distances[query * k + rank],
        static_cast<u32>(insert_start_id + static_cast<uint64_t>(label)));
    }
    std::sort(candidates.begin(), candidates.end());
    for (u32 rank = 0; rank < k; ++rank) {
      result.ids[query * k + rank] = candidates[rank].second;
    }
  }
  return result;
}
#endif

GroundTruth load_dynamic_groundtruth(
    const VectorRows& queries, const VectorRows& base_rows,
    const VectorRows& insert_rows, const GroundTruth& base_groundtruth,
    const Args& args, size_t inserted_count,
    const std::filesystem::path& path, bool& reused) {
  const bool exists = std::filesystem::exists(path);
  if (args.require_precomputed_groundtruth && !exists) {
    fail("missing required precomputed groundtruth: " + path.string());
  }
  if (!args.groundtruth_output_dir.empty() && exists) {
    reused = true;
    return read_groundtruth(path);
  }
#ifdef DVSTOR_LONG_INSERT_WITH_FAISS_CPU
  reused = false;
  GroundTruth result = build_dynamic_groundtruth(
    queries, base_rows, insert_rows, args.insert_row_offset,
    args.insert_start_id, inserted_count, base_groundtruth,
    args.recall_queries, args.recall_k);
  if (!args.groundtruth_output_dir.empty()) {
    write_groundtruth_file(path, result);
  }
  return result;
#else
  fail("dynamic groundtruth requires a build with FaissCPU");
#endif
}

int run_groundtruth_precompute_only(const Args& args) {
  std::cerr << "[sift101m-long-insert] loading insert vectors: "
            << args.insert_file << std::endl;
  const VectorRows insert_rows = read_vector_rows(args.insert_file);
  std::cerr << "[sift101m-long-insert] loading queries: "
            << args.query_file << std::endl;
  const VectorRows queries = read_vector_rows(args.query_file);
  std::cerr << "[sift101m-long-insert] loading base vectors: "
            << args.base_vector_file << std::endl;
  const VectorRows base_rows = read_vector_rows(args.base_vector_file);
  const GroundTruth base_groundtruth =
    read_groundtruth(args.baseline_groundtruth_file);
  if (args.insert_row_offset > insert_rows.count) {
    fail("--insert-row-offset exceeds insert-file row count");
  }
  const size_t available = insert_rows.count - args.insert_row_offset;
  const size_t inserted_count = args.insert_count == 0
    ? available : std::min(args.insert_count, available);
  if (inserted_count == 0) fail("effective insert count is zero");
  std::filesystem::create_directories(args.groundtruth_output_dir);

  nlohmann::json files = nlohmann::json::array();
  size_t sample_index = 0;
  for (size_t checkpoint = args.groundtruth_precompute_step;
       checkpoint <= inserted_count;
       checkpoint += args.groundtruth_precompute_step, ++sample_index) {
    const auto started = std::chrono::steady_clock::now();
    const std::filesystem::path path = args.groundtruth_output_dir /
      ("sample_" + std::to_string(sample_index) + ".bin");
    bool reused = false;
    const GroundTruth result = load_dynamic_groundtruth(
      queries, base_rows, insert_rows, base_groundtruth, args, checkpoint,
      path, reused);
    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    files.push_back({
      {"file", path.filename().string()},
      {"inserted_count", checkpoint},
      {"rows", result.rows},
      {"top_k", result.top_k},
      {"reused", reused},
      {"seconds", elapsed},
    });
    std::cerr << "[sift101m-long-insert][groundtruth] sample=" << sample_index
              << " inserted=" << checkpoint << " reused=" << reused
              << " seconds=" << elapsed << std::endl;
  }

  const auto started = std::chrono::steady_clock::now();
  const std::filesystem::path final_path =
    args.groundtruth_output_dir / "post_stage2_drain.bin";
  bool final_reused = false;
  const GroundTruth final_result = load_dynamic_groundtruth(
    queries, base_rows, insert_rows, base_groundtruth, args, inserted_count,
    final_path, final_reused);
  const double final_elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  files.push_back({
    {"file", final_path.filename().string()},
    {"inserted_count", inserted_count},
    {"rows", final_result.rows},
    {"top_k", final_result.top_k},
    {"reused", final_reused},
    {"seconds", final_elapsed},
  });
  write_json_file(args.groundtruth_output_dir / "manifest.json", {
    {"mode", "exact_base_and_inserted_topk"},
    {"insert_file", args.insert_file.string()},
    {"query_file", args.query_file.string()},
    {"baseline_groundtruth_file", args.baseline_groundtruth_file.string()},
    {"base_vector_file", args.base_vector_file.string()},
    {"insert_start_id", args.insert_start_id},
    {"insert_row_offset", args.insert_row_offset},
    {"insert_count", inserted_count},
    {"precompute_step", args.groundtruth_precompute_step},
    {"recall_queries", args.recall_queries},
    {"recall_k", args.recall_k},
    {"files", files},
  });
  std::cerr << "[sift101m-long-insert][groundtruth] complete files="
            << files.size() << " output_dir=" << args.groundtruth_output_dir
            << std::endl;
  return EXIT_SUCCESS;
}

nlohmann::json recall_samples_json(
    const std::vector<TimedRecallSample>& samples) {
  nlohmann::json root = nlohmann::json::array();
  for (const TimedRecallSample& sample : samples) {
    root.push_back({
      {"index", sample.index},
      {"scheduled_elapsed_seconds", sample.scheduled_elapsed_seconds},
      {"start_elapsed_seconds", sample.start_elapsed_seconds},
      {"end_elapsed_seconds", sample.end_elapsed_seconds},
      {"inserted_ack_at_start", sample.inserted_ack_at_start},
      {"inserted_ack_at_end", sample.inserted_ack_at_end},
      {"groundtruth_inserted_count", sample.groundtruth_inserted_count},
      {"groundtruth_build_seconds", sample.groundtruth_build_seconds},
      {"groundtruth_reused", sample.groundtruth_reused},
      {"recall", sample.recall.recall},
      {"base_recall", sample.recall.base_recall},
      {"inserted_recall", sample.recall.inserted_recall},
      {"base_truth", sample.recall.base_truth},
      {"inserted_truth", sample.recall.inserted_truth},
      {"queries", sample.recall.queries},
      {"k", sample.recall.k},
    });
  }
  return root;
}

void checkpoint_recall_samples(
    const std::filesystem::path& path,
    const std::vector<TimedRecallSample>& samples) {
  if (path.empty()) return;
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  write_json_file(temporary, {
    {"samples", recall_samples_json(samples)},
  });
  std::filesystem::rename(temporary, path);
}

InsertStats run_insert_phase(ComputeService& service,
                             const VectorRows& insert_rows,
                             const VectorRows* queries,
                             const VectorRows* base_rows,
                             const GroundTruth* baseline_groundtruth,
                             const Args& args,
                             size_t effective_insert_count,
                             std::atomic<size_t>* inserted_progress) {
  if (insert_rows.dim != service.config().dim) {
    fail("insert dim mismatch: insert=" + std::to_string(insert_rows.dim) +
         " config=" + std::to_string(service.config().dim));
  }
  if (args.insert_row_offset + effective_insert_count > insert_rows.count) {
    fail("insert range exceeds insert-file row count");
  }
  if (args.insert_start_id + effective_insert_count > static_cast<uint64_t>(std::numeric_limits<node_t>::max()) + 1ull) {
    fail("insert id range exceeds node_t range");
  }
  if (args.recall_sample_interval_seconds != 0 &&
      (queries == nullptr || baseline_groundtruth == nullptr)) {
    fail("periodic recall has no query or baseline groundtruth input");
  }
  if (args.dynamic_groundtruth && base_rows == nullptr) {
    fail("dynamic groundtruth has no base vectors");
  }

  std::atomic<size_t> next_row{0};
  std::atomic<size_t> attempted{0};
  std::atomic<size_t> local_inserted{0};
  std::atomic<size_t>& inserted = inserted_progress == nullptr
    ? local_inserted : *inserted_progress;
  inserted.store(0, std::memory_order_relaxed);
  std::atomic<size_t> failed{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> sampling_stop{false};
  std::exception_ptr first_error;
  std::mutex error_mutex;
  std::mutex reset_mutex;
  std::mutex sampling_mutex;
  std::condition_variable sampling_cv;
  std::mutex insert_gate_mutex;
  std::condition_variable insert_gate_cv;
  bool pause_insert_claims = false;
  size_t active_insert_batches = 0;
  size_t next_recall_checkpoint = args.groundtruth_precompute_step;
  std::vector<TimedRecallSample> recall_samples;
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.insert_threads));
  std::vector<std::thread> threads;
  threads.reserve(args.insert_threads);

  ProgressReporter reporter("insert-100m-to-101m", attempted, effective_insert_count, 0);
  const auto start = std::chrono::steady_clock::now();
  const auto record_error = [&]() {
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!first_error) first_error = std::current_exception();
    }
    stop.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(insert_gate_mutex);
      pause_insert_claims = false;
    }
    sampling_cv.notify_all();
    insert_gate_cv.notify_all();
  };

  std::thread sampler;
  if (args.recall_sample_interval_seconds != 0) {
    sampler = std::thread([&]() {
      try {
        const auto interval = std::chrono::seconds(
          args.recall_sample_interval_seconds);
        auto deadline = start + interval;
        size_t sample_index = 0;
        for (;;) {
          size_t checkpoint = 0;
          std::optional<GroundTruth> dynamic_groundtruth;
          double groundtruth_build_seconds = 0.0;
          bool reused = false;
          auto sample_start = std::chrono::steady_clock::now();
          if (args.dynamic_groundtruth) {
            {
              std::lock_guard<std::mutex> lock(insert_gate_mutex);
              checkpoint = next_recall_checkpoint;
            }
            if (checkpoint > effective_insert_count ||
                stop.load(std::memory_order_acquire) ||
                sampling_stop.load(std::memory_order_acquire)) {
              break;
            }
            const auto groundtruth_start = std::chrono::steady_clock::now();
            dynamic_groundtruth = load_dynamic_groundtruth(
              *queries, *base_rows, insert_rows, *baseline_groundtruth, args,
              checkpoint,
              args.groundtruth_output_dir /
                ("sample_" + std::to_string(sample_index) + ".bin"),
              reused);
            groundtruth_build_seconds = reused ? 0.0 :
              std::chrono::duration<double>(
                std::chrono::steady_clock::now() - groundtruth_start).count();
            std::cerr << "[sift101m-long-insert][groundtruth] sample="
                      << sample_index << " inserted=" << checkpoint
                      << " reused=" << reused
                      << " build_seconds=" << groundtruth_build_seconds
                      << std::endl;
          } else {
            std::unique_lock<std::mutex> lock(sampling_mutex);
            sampling_cv.wait_until(lock, deadline, [&]() {
              return stop.load(std::memory_order_acquire) ||
                sampling_stop.load(std::memory_order_acquire);
            });
            if (stop.load(std::memory_order_acquire) ||
                sampling_stop.load(std::memory_order_acquire)) {
              break;
            }
            sample_start = std::chrono::steady_clock::now();
          }
          if (args.dynamic_groundtruth) {
            {
              std::unique_lock<std::mutex> lock(insert_gate_mutex);
              insert_gate_cv.wait(lock, [&]() {
                return next_row.load(std::memory_order_relaxed) >= checkpoint ||
                  stop.load(std::memory_order_acquire) ||
                  sampling_stop.load(std::memory_order_acquire);
              });
              pause_insert_claims = true;
              insert_gate_cv.wait(lock, [&]() {
                return active_insert_batches == 0 ||
                  stop.load(std::memory_order_acquire);
              });
            }
          }
          if (stop.load(std::memory_order_acquire) ||
              sampling_stop.load(std::memory_order_acquire)) {
            {
              std::lock_guard<std::mutex> lock(insert_gate_mutex);
              pause_insert_claims = false;
            }
            insert_gate_cv.notify_all();
            break;
          }
          const size_t inserted_at_start = inserted.load(
            std::memory_order_relaxed);
          if (args.dynamic_groundtruth &&
              (inserted_at_start != checkpoint ||
               attempted.load(std::memory_order_relaxed) != inserted_at_start)) {
            fail("insert prefix does not match the precomputed groundtruth");
          }
          const GroundTruth* recall_groundtruth = args.dynamic_groundtruth
            ? &*dynamic_groundtruth : baseline_groundtruth;
          const RecallResult recall = run_recall(
            service, "insert-sample-" + std::to_string(sample_index),
            *queries, *recall_groundtruth, args.recall_queries, args.recall_k,
            static_cast<u32>(args.insert_start_id));
          const auto sample_end = std::chrono::steady_clock::now();
          recall_samples.push_back({
            .index = sample_index,
            .scheduled_elapsed_seconds = std::chrono::duration<double>(
              sample_start - start).count(),
            .start_elapsed_seconds = std::chrono::duration<double>(
              sample_start - start).count(),
            .end_elapsed_seconds = std::chrono::duration<double>(
              sample_end - start).count(),
            .inserted_ack_at_start = inserted_at_start,
            .inserted_ack_at_end = inserted.load(std::memory_order_relaxed),
            .groundtruth_inserted_count = checkpoint,
            .groundtruth_build_seconds = groundtruth_build_seconds,
            .groundtruth_reused = reused,
            .recall = recall,
          });
          checkpoint_recall_samples(
            args.recall_samples_json_path, recall_samples);
          if (args.dynamic_groundtruth) {
            {
              std::lock_guard<std::mutex> lock(insert_gate_mutex);
              next_recall_checkpoint = checkpoint +
                args.groundtruth_precompute_step;
              pause_insert_claims = false;
            }
            insert_gate_cv.notify_all();
          } else {
            deadline += interval;
          }
          ++sample_index;
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(insert_gate_mutex);
          pause_insert_claims = false;
        }
        insert_gate_cv.notify_all();
        record_error();
      }
    });
  }

  for (size_t tid = 0; tid < args.insert_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      (void)tid;
      bool batch_active = false;
      try {
        start_barrier.arrive_and_wait();
        vec<ComputeService::InsertItem> batch;
        batch.reserve(args.insert_batch_size);
        while (!stop.load(std::memory_order_acquire)) {
          size_t begin = 0;
          {
            std::unique_lock<std::mutex> lock(insert_gate_mutex);
            if (args.dynamic_groundtruth &&
                next_recall_checkpoint <= effective_insert_count &&
                next_row.load(std::memory_order_relaxed) >=
                  next_recall_checkpoint) {
              pause_insert_claims = true;
              insert_gate_cv.notify_all();
            }
            insert_gate_cv.wait(lock, [&]() {
              return !pause_insert_claims ||
                stop.load(std::memory_order_acquire);
            });
            if (stop.load(std::memory_order_acquire)) break;
            begin = next_row.fetch_add(
              args.insert_batch_size, std::memory_order_relaxed);
            if (begin >= effective_insert_count) break;
            ++active_insert_batches;
            batch_active = true;
          }
          const size_t end = std::min(begin + args.insert_batch_size, effective_insert_count);
          batch.clear();
          for (size_t row = begin; row < end; ++row) {
            const uint64_t id64 = args.insert_start_id + row;
            batch.push_back({
              static_cast<node_t>(id64),
              insert_rows.decode_row(args.insert_row_offset + row),
            });
          }

          const size_t ok = service.insert(batch);
          inserted.fetch_add(ok, std::memory_order_relaxed);
          if (ok != batch.size()) {
            failed.fetch_add(batch.size() - ok, std::memory_order_relaxed);
          }
          const size_t before = attempted.fetch_add(batch.size(), std::memory_order_relaxed);
          const size_t after = before + batch.size();
          if (args.reset_breakdown_every > 0 &&
              before / args.reset_breakdown_every != after / args.reset_breakdown_every) {
            std::lock_guard<std::mutex> lock(reset_mutex);
            service.reset_breakdown_state();
          }
          {
            std::lock_guard<std::mutex> lock(insert_gate_mutex);
            --active_insert_batches;
            batch_active = false;
            if (active_insert_batches == 0) insert_gate_cv.notify_all();
          }
        }
      } catch (...) {
        if (batch_active) {
          std::lock_guard<std::mutex> lock(insert_gate_mutex);
          --active_insert_batches;
          if (active_insert_batches == 0) insert_gate_cv.notify_all();
        }
        record_error();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  const auto stop_time = std::chrono::steady_clock::now();
  reporter.finish();
  sampling_stop.store(true, std::memory_order_release);
  sampling_cv.notify_all();
  if (sampler.joinable()) sampler.join();
  if (first_error) {
    std::rethrow_exception(first_error);
  }

  InsertStats stats;
  stats.attempted = attempted.load(std::memory_order_relaxed);
  stats.inserted = inserted.load(std::memory_order_relaxed);
  stats.failed = failed.load(std::memory_order_relaxed);
  stats.duration_seconds = std::chrono::duration<double>(stop_time - start).count();
  stats.inserts_per_second = stats.duration_seconds <= 0.0
                               ? 0.0
                               : static_cast<double>(stats.inserted) / stats.duration_seconds;
  stats.recall_samples = std::move(recall_samples);
  return stats;
}

void wait_for_settle(size_t seconds) {
  if (seconds == 0) {
    return;
  }
  std::atomic<size_t> elapsed{0};
  ProgressReporter reporter("settle-after-insert", elapsed, 0, seconds);
  for (size_t i = 0; i < seconds; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    elapsed.fetch_add(1, std::memory_order_relaxed);
  }
  reporter.finish();
}

nlohmann::json recall_json(const RecallResult& result,
                           const std::filesystem::path& query_file,
                           const std::filesystem::path& gt_file) {
  return {
    {"query_file", query_file.string()},
    {"groundtruth_file", gt_file.string()},
    {"queries", result.queries},
    {"k", result.k},
    {"recall", result.recall},
    {"base_recall", result.base_recall},
    {"inserted_recall", result.inserted_recall},
    {"base_truth", result.base_truth},
    {"inserted_truth", result.inserted_truth},
  };
}

nlohmann::json insert_stats_json(const InsertStats& stats) {
  return {
    {"attempted", stats.attempted},
    {"inserted", stats.inserted},
    {"failed", stats.failed},
    {"duration_seconds", stats.duration_seconds},
    {"inserts_per_second", stats.inserts_per_second},
  };
}

int run_with_service(ComputeService& service, const Args& args) {
  std::cerr << "[sift101m-long-insert] loading insert vectors: " << args.insert_file << std::endl;
  const VectorRows insert_rows = read_vector_rows(args.insert_file);
  std::cerr << "[sift101m-long-insert] insert rows=" << insert_rows.count
            << " dim=" << insert_rows.dim
            << " dtype=" << vector_dtype_name(insert_rows.dtype)
            << " vector_bytes=" << insert_rows.vector_bytes << std::endl;

  std::optional<VectorRows> queries;
  if (!args.query_file.empty()) {
    std::cerr << "[sift101m-long-insert] loading queries: "
              << args.query_file << std::endl;
    queries = read_vector_rows(args.query_file);
    std::cerr << "[sift101m-long-insert] query rows=" << queries->count
              << " dim=" << queries->dim
              << " dtype=" << vector_dtype_name(queries->dtype) << std::endl;
  }

  std::optional<VectorRows> base_rows;
  if (args.dynamic_groundtruth) {
    std::cerr << "[sift101m-long-insert] loading base vectors: "
              << args.base_vector_file << std::endl;
    base_rows = read_vector_rows(args.base_vector_file);
    std::cerr << "[sift101m-long-insert] base rows=" << base_rows->count
              << " dim=" << base_rows->dim
              << " dtype=" << vector_dtype_name(base_rows->dtype) << std::endl;
  }

  if (args.insert_row_offset > insert_rows.count) {
    fail("--insert-row-offset exceeds insert-file row count");
  }
  const size_t available_insert_rows = insert_rows.count - args.insert_row_offset;
  const size_t effective_insert_count = args.insert_count == 0
                                          ? available_insert_rows
                                          : std::min(args.insert_count, available_insert_rows);
  if (effective_insert_count == 0) {
    fail("effective insert count is zero");
  }

  nlohmann::json root;
  root["meta"] = {
    {"service_config", args.service_config_path},
    {"insert_file", args.insert_file.string()},
    {"insert_start_id", args.insert_start_id},
    {"insert_count_requested", args.insert_count},
    {"insert_count_effective", effective_insert_count},
    {"insert_row_offset", args.insert_row_offset},
    {"insert_threads", args.insert_threads},
    {"insert_batch_size", args.insert_batch_size},
    {"query_file", args.query_file.string()},
    {"baseline_groundtruth_file", args.baseline_groundtruth_file.empty() ? "" : args.baseline_groundtruth_file.string()},
    {"post_groundtruth_file", args.groundtruth_file.string()},
    {"base_vector_file", args.base_vector_file.string()},
    {"groundtruth_output_dir", args.groundtruth_output_dir.string()},
    {"groundtruth_precompute_step", args.groundtruth_precompute_step},
    {"require_precomputed_groundtruth", args.require_precomputed_groundtruth},
    {"groundtruth_mode", args.dynamic_groundtruth
      ? "dynamic_exact_base_and_inserted" : "fixed_base_groundtruth"},
    {"recall_queries", args.recall_queries},
    {"recall_k", args.recall_k},
    {"recall_sample_interval_seconds", args.recall_sample_interval_seconds},
    {"recall_samples_json", args.recall_samples_json_path.string()},
    {"recall_time_semantics", args.dynamic_groundtruth
      ? "consistent_insert_prefix_after_ack" : "fixed_base_groundtruth_during_concurrent_insert"},
    {"settle_seconds", args.settle_seconds},
    {"self_recall_queries", args.self_recall_queries},
    {"self_recall_k", args.self_recall_k},
    {"reset_breakdown_every", args.reset_breakdown_every},
    {"dim", service.config().dim},
    {"max_vectors_config", service.config().max_vectors},
    {"storage_owner_update_protocol",
     service.config().synchronous_exact_updates_enabled()
       ? "single_owner_synchronous_exact" : "centroid_home_two_stage"},
    {"storage_owner_maintenance_workers", service.config().storage_owner_maintenance_workers},
    {"fine_grained_breakdown_enabled", service.config().enable_breakdown},
    {"search", "gpu_persistent_opq_pq"},
    {"navigation_quantizer", "opq_pq"},
  };
  root["input"] = {
    {"insert_rows", insert_rows.count},
    {"insert_dim", insert_rows.dim},
    {"insert_dtype", vector_dtype_name(insert_rows.dtype)},
    {"query_rows", queries.has_value() ? queries->count : 0},
    {"query_dim", queries.has_value() ? queries->dim : 0},
    {"query_dtype", queries.has_value()
      ? vector_dtype_name(queries->dtype) : "none"},
  };

  std::optional<RecallResult> baseline_recall;
  std::optional<GroundTruth> baseline_groundtruth;
  if (!args.baseline_groundtruth_file.empty()) {
    std::cerr << "[sift101m-long-insert] loading baseline groundtruth: "
              << args.baseline_groundtruth_file << std::endl;
    baseline_groundtruth = read_groundtruth(args.baseline_groundtruth_file);
    baseline_recall = run_recall(
      service, "baseline-100m", *queries, *baseline_groundtruth,
      args.recall_queries, args.recall_k,
      static_cast<u32>(args.insert_start_id));
    root["baseline_recall"] = recall_json(*baseline_recall, args.query_file,
                                          args.baseline_groundtruth_file);
    service.reset_breakdown_state();
  }

  std::cerr << "[sift101m-long-insert] inserting ID range [" << args.insert_start_id
            << ", " << (args.insert_start_id + effective_insert_count) << ")" << std::endl;
  std::atomic<size_t> inserted_ack{0};
  c1_locality::Sampler locality_sampler(
    args.locality_samples_json_path,
    std::chrono::milliseconds(args.locality_sample_interval_ms), inserted_ack,
    service.config().storage_peers.size(), [&service]() {
      return service.storage_maintenance_telemetry();
    });
  root["meta"]["locality_samples_json"] =
    args.locality_samples_json_path.string();
  root["meta"]["locality_sample_interval_ms"] =
    args.locality_sample_interval_ms;
  root["meta"]["locality_metric_scope"] =
    "stage2_finalized_new_node_outgoing_edges";
  locality_sampler.capture("before_insert");
  const InsertStats insert_stats = run_insert_phase(
    service, insert_rows, queries.has_value() ? &*queries : nullptr,
    base_rows.has_value() ? &*base_rows : nullptr,
    baseline_groundtruth.has_value() ? &*baseline_groundtruth : nullptr,
    args, effective_insert_count, &inserted_ack);
  locality_sampler.set_phase("post_insert");
  root["insert"] = insert_stats_json(insert_stats);
  root["recall_time_series"] = recall_samples_json(
    insert_stats.recall_samples);
  service.reset_breakdown_state();

  if (insert_stats.failed != 0 || insert_stats.inserted != effective_insert_count) {
    root["execution_error"] = "one or more inserts failed";
    std::ostringstream text;
    text << "insert failed\n"
         << "  attempted: " << insert_stats.attempted << '\n'
         << "  inserted: " << insert_stats.inserted << '\n'
         << "  failed: " << insert_stats.failed << '\n';
    write_json_file(args.report_json_path, root);
    write_text_file(args.report_text_path, text.str());
    std::cout << text.str();
    return EXIT_FAILURE;
  }

  std::optional<SelfRecallSnapshot> stage1_self_recall;
  const auto stage1_recall_finished_before = std::chrono::steady_clock::now();
  if (args.self_recall_queries != 0) {
    stage1_self_recall = run_inserted_self_recall(
      service, "stage1-only", insert_rows, args.insert_row_offset,
      args.insert_start_id, effective_insert_count,
      args.self_recall_queries, args.self_recall_k);
    root["stage1_only_self_recall"] = {
      {"queries", stage1_self_recall->summary.queries},
      {"k", stage1_self_recall->summary.k},
      {"hit_rate", stage1_self_recall->summary.recall},
    };
  }
  const double stage1_window_elapsed_seconds = insert_stats.duration_seconds +
    std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                  stage1_recall_finished_before).count();
  const double configured_stage2_delay_seconds =
    static_cast<double>(
      service.config().storage_owner_stage2_initial_delay_ms) / 1000.0;
  root["stage1_only_window"] = {
    {"configured_stage2_delay_seconds", configured_stage2_delay_seconds},
    {"insert_plus_recall_seconds", stage1_window_elapsed_seconds},
    {"valid", configured_stage2_delay_seconds >
              stage1_window_elapsed_seconds},
  };

  wait_for_settle(args.settle_seconds);

  const auto maintenance_drain_started = std::chrono::steady_clock::now();
  vec<u64> maintenance_targets;
  vec<u64> maintenance_durable;
  const bool maintenance_complete = service.wait_for_storage_maintenance(
    std::chrono::milliseconds(
      std::max<u32>(1000, service.config().storage_owner_rpc_timeout_ms)),
    &maintenance_targets, &maintenance_durable);
  const double maintenance_drain_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - maintenance_drain_started).count();
  root["stage2_finalization"] = {
    {"complete", maintenance_complete},
    {"drain_seconds_after_settle", maintenance_drain_seconds},
    {"target_sequences", maintenance_targets},
    {"durable_sequences", maintenance_durable},
  };
  if (!maintenance_complete) {
    fail("Stage2 did not reach its durable watermark before final recall");
  }
  if (args.locality_sample_interval_ms != 0) {
    const auto telemetry_deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(15);
    for (;;) {
      const auto snapshots = service.storage_maintenance_telemetry();
      u64 finalized = 0;
      bool settled = snapshots.size() == service.config().storage_peers.size();
      for (const auto& snapshot : snapshots) {
        if (!snapshot.has_value()) {
          settled = false;
          continue;
        }
        finalized += snapshot->stage2_finalized_live;
        settled = settled && snapshot->remaining == 0;
      }
      settled = settled && finalized == effective_insert_count;
      if (settled) break;
      if (std::chrono::steady_clock::now() >= telemetry_deadline) {
        fail("locality telemetry did not publish the drained graph");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    locality_sampler.set_phase("post_stage2_drain");
    locality_sampler.capture("post_stage2_drain");
  }

  std::optional<SelfRecallSnapshot> final_self_recall;
  if (args.self_recall_queries != 0) {
    final_self_recall = run_inserted_self_recall(
      service, "stage2-finalized", insert_rows, args.insert_row_offset,
      args.insert_start_id, effective_insert_count,
      args.self_recall_queries, args.self_recall_k);
    root["finalized_self_recall"] = {
      {"queries", final_self_recall->summary.queries},
      {"k", final_self_recall->summary.k},
      {"hit_rate", final_self_recall->summary.recall},
      {"missed_insert_ids", final_self_recall->missed_insert_ids},
    };
    const double overlap = result_overlap_at_k(
      *stage1_self_recall, *final_self_recall);
    root["self_recall_delta"] = {
      {"final_minus_stage1", final_self_recall->summary.recall -
        stage1_self_recall->summary.recall},
      {"stage1_minus_final", stage1_self_recall->summary.recall -
        final_self_recall->summary.recall},
      {"stage1_final_result_overlap_at_k", overlap},
    };
  }

  std::optional<RecallResult> post_recall;
  if (!args.groundtruth_file.empty() || args.dynamic_groundtruth) {
    std::optional<GroundTruth> post_gt;
    bool groundtruth_reused = false;
    if (args.dynamic_groundtruth) {
      const auto groundtruth_start = std::chrono::steady_clock::now();
      post_gt = load_dynamic_groundtruth(
        *queries, *base_rows, insert_rows, *baseline_groundtruth, args,
        effective_insert_count,
        args.groundtruth_output_dir / "post_stage2_drain.bin",
        groundtruth_reused);
      const double build_seconds = groundtruth_reused ? 0.0 :
        std::chrono::duration<double>(
          std::chrono::steady_clock::now() - groundtruth_start).count();
      std::cerr << "[sift101m-long-insert][groundtruth] post-drain inserted="
                << effective_insert_count << " reused=" << groundtruth_reused
                << " build_seconds=" << build_seconds << std::endl;
    } else {
      std::cerr << "[sift101m-long-insert] loading post-insert groundtruth: "
                << args.groundtruth_file << std::endl;
      post_gt = read_groundtruth(args.groundtruth_file);
    }
    post_recall = run_recall(service, "post-101m", *queries, *post_gt,
                             args.recall_queries, args.recall_k,
                             static_cast<u32>(args.insert_start_id));
    root["post_insert_recall"] = recall_json(
      *post_recall, args.query_file, args.groundtruth_file);
    root["post_insert_recall"]["groundtruth_mode"] = args.dynamic_groundtruth
      ? "dynamic_exact_base_and_inserted" : "fixed_groundtruth_file";
    root["post_insert_recall"]["groundtruth_inserted_count"] =
      effective_insert_count;
    root["post_insert_recall"]["groundtruth_reused"] = groundtruth_reused;
  }

  if (baseline_recall.has_value() && post_recall.has_value()) {
    const double drop = baseline_recall->recall - post_recall->recall;
    root["recall_delta"] = {
      {"baseline_minus_post", drop},
      {"post_minus_baseline", post_recall->recall - baseline_recall->recall},
    };
  }

  std::ostringstream text;
  text << std::fixed << std::setprecision(6);
  text << "sift101m long insert recall\n";
  text << "  inserted: " << insert_stats.inserted << "/" << effective_insert_count
       << " in " << insert_stats.duration_seconds << "s"
       << " (" << insert_stats.inserts_per_second << " inserts/s)\n";
  if (baseline_recall.has_value()) {
    text << "  baseline recall@" << baseline_recall->k << ": "
         << baseline_recall->recall << " (" << baseline_recall->queries << " queries)\n";
  }
  if (!insert_stats.recall_samples.empty()) {
    text << "  periodic recall samples: " << insert_stats.recall_samples.size()
         << '\n';
  }
  if (stage1_self_recall.has_value() && final_self_recall.has_value()) {
    text << "  stage1-only self-hit@" << stage1_self_recall->summary.k << ": "
         << stage1_self_recall->summary.recall << '\n';
    text << "  finalized self-hit@" << final_self_recall->summary.k << ": "
         << final_self_recall->summary.recall << '\n';
    text << "  Stage1/final result overlap@"
         << stage1_self_recall->summary.k << ": "
         << result_overlap_at_k(*stage1_self_recall, *final_self_recall)
         << '\n';
  }
  if (post_recall.has_value()) {
    text << "  post recall@" << post_recall->k << ": "
         << post_recall->recall << " (" << post_recall->queries
         << " queries)\n";
  }
  if (baseline_recall.has_value() && post_recall.has_value()) {
    text << "  recall drop: "
         << (baseline_recall->recall - post_recall->recall) << '\n';
  }

  const auto gpu_telemetry = service.gpu_search_telemetry();
  root["gpu_search_telemetry"] = {
    {"dynamic_code_candidates", gpu_telemetry.dynamic_code_candidates},
    {"dynamic_code_reads", gpu_telemetry.dynamic_code_reads},
    {"dynamic_code_incarnation_rejects",
      gpu_telemetry.dynamic_code_incarnation_rejects},
    {"exact_vector_reads", gpu_telemetry.exact_vector_reads},
  };
  std::cerr << "[sift101m-long-insert][gpu-telemetry] dynamic_candidates="
            << gpu_telemetry.dynamic_code_candidates
            << " dynamic_reads=" << gpu_telemetry.dynamic_code_reads
            << " incarnation_rejects="
            << gpu_telemetry.dynamic_code_incarnation_rejects
            << " exact_reads=" << gpu_telemetry.exact_vector_reads << std::endl;

  locality_sampler.stop();
  root["locality_sampling"] = locality_sampler.status();
  write_json_file(args.report_json_path, root);
  write_text_file(args.report_text_path, text.str());
  std::cout << text.str();
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, signal_handler);
  signal(SIGABRT, signal_handler);

  try {
    const Args args = parse_args(argc, argv);
    if (args.groundtruth_precompute_only) {
      return run_groundtruth_precompute_only(args);
    }
    auto service_args = tools::breakdown_benchmark::build_service_argv(args.service_config_path);
    auto service_argv = tools::breakdown_benchmark::make_argv(service_args);
    configuration::IndexConfiguration config(static_cast<int>(service_argv.size()), service_argv.data());

    ComputeService service(config);
    return run_with_service(service, args);
  } catch (const std::exception& e) {
    std::cerr << "sift101m long insert recall failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}

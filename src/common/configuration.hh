#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include <library/configuration.hh>

#include "constants.hh"
#include "gpu_search/adaptive_frontier.hh"
#include "types.hh"
#include "vector_dtype.hh"

namespace configuration {

struct Parameters {
  u32 num_threads{};
  u32 gpu_rdma_qps{};
};

class IndexConfiguration : public Configuration {
public:
  filepath_t index_prefix{};
  filepath_t server_index_file{};
  u32 num_threads{};
  i32 seed{1234};
  bool disable_thread_pinning{};

  u32 dim{};
  u32 max_vectors{1'000'000};
  // Exclusive upper bound for logical node IDs. Zero during option parsing
  // means "derive from max_vectors" for backward-compatible configurations.
  u32 vector_id_namespace_size{};
  u32 k{};
  u32 R{64};
  u32 beam_width_construction{200};
  f64 alpha{1.2};
  str vector_data_type{"auto"};

  u32 gpu_device{};
  bool enable_breakdown{true};
  u32 gpu_query_slots{256};
  u32 gpu_memory_limit_gb{40};
  u32 gpu_memory_reserve_gb{4};
  u32 gpu_bootstrap_window_mb{64};
  u32 gpu_bootstrap_windows{2};
  u32 gpu_graph_commit_width{32};
  // Native system profiles. Custom preserves the legacy defaults and all
  // lower-level switches; baseline/full set every contribution at once.
  str profile{"custom"};
  str c1{"auto"};
  str c2{"auto"};
  str c3{"auto"};
  // Zero selects the workspace/beam-limited speculative issue cap.
  u32 gpu_graph_issue_width{};
  // Contribution-level graph-access mode. "manual" keeps the lower-level
  // switches available for engineering-only partial ablations; the formal
  // fixed/adaptive modes resolve them as one indivisible mechanism.
  str dynamic_graph_access_mode{"manual"};
  str gpu_query_graph_read_policy{"fixed"};
  // Keep dynamic PQ-tag bootstrap independently ablatable from the base-node
  // sidecar bootstrap. Both feed the same quantized-extent adaptation policy.
  bool gpu_dynamic_graph_extent{true};
  // Sidecar and dynamic PQ-tag bootstrap hints are independently ablatable;
  // disabling them does not disable validated online extent adaptation.
  bool gpu_dynamic_graph_extent_hints{true};
  // Contribution-level GPU-RDMA search-progression mode. Formal modes own
  // both the frontier widths and Beam merge implementation so an experiment
  // cannot accidentally retain a half-enabled ASFE path.
  str gpu_rdma_search_progression_mode{"manual"};
  // Motivation/causal-ablation switch: enable exact mandatory-frontier
  // issue-before-commit without requiring issue_width > commit_width.  This
  // permits a strict persistent+Stable-Run A/B in which both variants read
  // exactly commit_width graph records and speculative tail reads are absent.
  bool gpu_exact_frontier_early_issue{false};
  bool gpu_terminal_exact_prefetch{true};
  str gpu_query_beam_merge_policy{"legacy"};
  str query_rdma_trace_mode{"off"};
  u32 query_rdma_trace_sample_rate{1000};
  filepath_t query_rdma_trace_output{};
  u32 query_rdma_trace_events_per_query{1024};
  u32 gpu_traversal_beam_width{128};
  u32 gpu_final_rerank_width{64};
  u32 gpu_max_expansions{384};
  u32 gpu_centroid_route_fanout{1};
  u32 gpu_rdma_qps{4};
  // A CQE can be delayed well beyond ordinary read latency when GPU reads and
  // CPU-posted mutation traffic share the NIC.  This is a liveness bound, not
  // a latency target: transport errors still fail immediately.
  u32 gpu_direct_timeout_ms{250};
  u32 gpu_persistent_blocks_per_sm{4};
  // Per-thread device stack floor for the out-of-line ASFE/CUB call chain.
  // Keep this explicit so a toolkit-specific requirement is diagnosable and
  // adjustable without changing kernel code.
  u32 gpu_device_stack_bytes{4096};
  // Compute-side mutation support is optional for query-only deployments.
  // Keep it enabled by default so existing service configurations preserve
  // their insert/upsert/erase behavior.
  bool enable_updates{true};
  // Contribution-level update completion contract. Coupled executes one
  // complete graph mutation before ACK. Decoupled enables the Stage1/Stage2
  // pipeline and all companion maintenance optimizations. Manual is reserved
  // for engineering ablations with independently controlled sub-switches.
  str storage_owner_update_completion_mode{"decoupled"};

  u32 storage_id{};
  vec<str> storage_peers;
  u32 storage_owner_batch_max{16};
  // Give synchronous writers enough time to form one useful foreground
  // microbatch. A full batch is sent immediately; this only bounds additional
  // coalescing while an RPC slot is available, not remote service time.
  u32 storage_owner_batch_max_wait_us{2'000};
  // Stage2 is query-invisible background maintenance. Its compaction horizon
  // must not be tied to the foreground Stage1 batching latency.
  // Stage2 batching is coordinated across already-pending work. This bound
  // caps the arrival-aware tail deadline; a complete B8 runs immediately.
  u32 storage_owner_stage2_batch_max_wait_us{25'000};
  // Motivation/diagnostic gate: do not admit a Stage2 descriptor until it
  // has aged this long. Zero is the production behavior.
  u32 storage_owner_stage2_initial_delay_ms{0};
  bool storage_owner_stage2_score_many{true};
  u32 storage_owner_stage2_score_many_min_items{128};
  // Combine ready logical Stage2 home requests that share a peer and wire
  // class. False preserves the same protocol and retry semantics but sends
  // each logical request directly, which is the architecture baseline.
  bool storage_owner_stage2_home_rpc_combining{true};
  // False serializes the two home-RPC planes without changing update semantics.
  bool storage_owner_stage2_split_plane{true};
  // LeaseMerge is the production graph-maintenance protocol. False retains the
  // legacy promotion and globally ordered cleanup path for ablation.
  bool storage_owner_lease_merge{true};
  // Maximum ordered graph issue width. Width one is the exact legacy path;
  // larger values are promotion-gated and only fill spare items in an RPC
  // that an authoritative expansion already requires.
  u32 storage_owner_stage2_graph_issue_width{16};
  u32 storage_owner_peer_qps_per_peer{8};
  u32 storage_owner_peer_rdma_tokens{16};
  u32 storage_owner_rpc_depth{8};
  u32 storage_owner_rpc_timeout_ms{30'000};
  // Zero keeps the legacy unbounded synchronous-exact retry loops.
  u32 storage_owner_exact_max_attempts{0};
  u32 storage_owner_search_snapshot_batch{256};
  // Stage2 is part of the only supported update protocol, so at least one
  // maintenance executor must exist even when the caller does not tune it.
  u32 storage_owner_maintenance_workers{1};
  u32 storage_owner_maintenance_queue_depth{65'536};
  // Permanent Stage2 failures are retried with the nominal 1 ms deferral for
  // at most this long before execution resources are released and the failure
  // is reported. Unfinished cleanup never advances the durable watermark.
  u32 storage_owner_stage2_max_retry_ms{300'000};
  // Backlog-aware Stage2 continuation search budget. Both watermarks zero
  // disables the controller and preserves the unbounded production budget.
  // Below the low watermark the budget stays unbounded; at the high watermark
  // the configured tight values apply; between them the budget shrinks
  // linearly from the search's natural single-pass scale.
  u32 storage_owner_stage2_budget_backlog_low{0};
  u32 storage_owner_stage2_budget_backlog_high{0};
  // Tightest per-dimension Stage2 continuation budget at the high watermark;
  // zero keeps the corresponding dimension unbounded at every backlog.
  u64 storage_owner_stage2_budget_max_expansions{0};
  u64 storage_owner_stage2_budget_max_candidate_visits{0};
  // Target time from a Stage1 completion to its observed durable repair.
  // Zero disables deadline-based admission and deadline-miss accounting.
  u64 storage_owner_repair_deadline_ms{0};
  // Maintenance watermark refresh interval used by deadline admission.
  u32 storage_owner_flow_control_poll_ms{5};
  u32 storage_owner_reverse_queue_depth{65'536};
  u32 storage_owner_reverse_coalesce_max{256};
  // Optimistic reverse-update revalidation gives up after this many observed
  // adjacency conflicts on one target; the sender retries with backoff.
  u32 storage_owner_reverse_conflict_limit{32};
  // Reverse aggregate resends start at the 1 ms base backoff and double per
  // failed attempt up to this cap.
  u32 storage_owner_reverse_retry_backoff_max_us{100'000};

  u32 mn_memory_gb{10};

  IndexConfiguration(int argc, char** argv);


  filepath_t resolved_index_prefix() const {
    return index_prefix;
  }

  VectorDType resolved_vector_dtype() const {
    return vector_data_type == "auto"
      ? VectorDType::float32 : parse_vector_dtype(vector_data_type);
  }

  u32 resolved_storage_owner_construction_width() const {
    return beam_width_construction;
  }

  bool adaptive_dynamic_graph_access_enabled() const {
    return gpu_query_graph_read_policy == "live-extent" &&
      gpu_dynamic_graph_extent;
  }

  bool decoupled_gpu_rdma_search_progression_enabled() const {
    return gpu_rdma_search_progression_mode == "decoupled" ||
      (gpu_rdma_search_progression_mode == "manual" &&
       (gpu_exact_frontier_early_issue ||
        gpu_graph_issue_width > gpu_graph_commit_width));
  }

  bool synchronous_exact_updates_enabled() const {
    return storage_owner_update_completion_mode == "coupled";
  }

private:
  str feature_bundle_conflict_;
  bool native_profile_active_{};

  static bool command_line_has_option(
      int argc, char** argv, std::string_view option) {
    for (int index = 1; index < argc; ++index) {
      if (argv[index] == nullptr) continue;
      const std::string_view argument{argv[index]};
      if (argument == option ||
          (argument.size() > option.size() &&
           argument.starts_with(option) &&
           argument[option.size()] == '=')) {
        return true;
      }
    }
    return false;
  }

  void record_feature_bundle_conflict(str message) {
    if (feature_bundle_conflict_.empty()) {
      feature_bundle_conflict_ = message;
    }
  }

  void resolve_contribution_profile(int argc, char** argv);
  void relax_contribution_overrides(int argc, char** argv);

  void resolve_feature_bundles(
      int argc, char** argv,
      bool graph_read_policy_explicit,
      bool dynamic_graph_extent_explicit,
      bool graph_issue_width_explicit,
      bool beam_merge_policy_explicit);


  static str normalize_mode(str value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return value;
  }

  void add_options();


  void validate(char** argv) const;

public:
  friend std::ostream& operator<<(
      std::ostream& output, const IndexConfiguration& config);

};

}  // namespace configuration

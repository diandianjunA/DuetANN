#include <cassert>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "common/configuration.hh"

namespace {

configuration::IndexConfiguration parse_config(
    std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) argv.push_back(argument.data());
  return configuration::IndexConfiguration(
    static_cast<int>(argv.size()), argv.data());
}

configuration::IndexConfiguration make_config(
    bool explicit_disable, bool explicit_namespace = false,
    std::string beam_merge_policy = {},
    std::string graph_read_policy = {},
    std::string dynamic_graph_extent = {},
    std::string home_rpc_combining = {},
    std::string dynamic_graph_access_mode = {},
    std::string search_progression_mode = {},
    std::string graph_commit_width = {},
    std::string graph_issue_width = {},
    std::string update_completion_mode = {}) {
  std::vector<std::string> arguments{
    "configuration_update_protocol_test",
    "--servers", "127.0.0.1:1234",
    "--storage-peers", "127.0.0.1:1234",
    "--index-prefix", "/tmp/index",
    "--threads", "1",
    "--dim", "128",
    "--k", "10",
  };
  if (explicit_disable) {
    arguments.emplace_back("--enable-updates");
    arguments.emplace_back("false");
  }
  if (explicit_namespace) {
    arguments.emplace_back("--vector-id-namespace-size");
    arguments.emplace_back("2000000");
  }
  if (!beam_merge_policy.empty()) {
    arguments.emplace_back("--gpu-query-beam-merge-policy");
    arguments.emplace_back(std::move(beam_merge_policy));
  }
  if (!graph_read_policy.empty()) {
    arguments.emplace_back("--gpu-query-graph-read-policy");
    arguments.emplace_back(std::move(graph_read_policy));
  }
  if (!dynamic_graph_extent.empty()) {
    arguments.emplace_back("--gpu-dynamic-graph-extent");
    arguments.emplace_back(std::move(dynamic_graph_extent));
  }
  if (!home_rpc_combining.empty()) {
    arguments.emplace_back("--storage-owner-stage2-home-rpc-combining");
    arguments.emplace_back(std::move(home_rpc_combining));
  }
  if (!dynamic_graph_access_mode.empty()) {
    arguments.emplace_back("--gpu-dynamic-graph-access-mode");
    arguments.emplace_back(std::move(dynamic_graph_access_mode));
  }
  if (!search_progression_mode.empty()) {
    arguments.emplace_back("--gpu-rdma-search-progression-mode");
    arguments.emplace_back(std::move(search_progression_mode));
  }
  if (!graph_commit_width.empty()) {
    arguments.emplace_back("--gpu-graph-commit-width");
    arguments.emplace_back(std::move(graph_commit_width));
  }
  if (!graph_issue_width.empty()) {
    arguments.emplace_back("--gpu-graph-issue-width");
    arguments.emplace_back(std::move(graph_issue_width));
  }
  if (!update_completion_mode.empty()) {
    arguments.emplace_back("--storage-owner-update-completion-mode");
    arguments.emplace_back(std::move(update_completion_mode));
  }
  return parse_config(std::move(arguments));
}

configuration::IndexConfiguration make_config_with_extra(
    std::vector<std::string> extra) {
  std::vector<std::string> arguments{
    "configuration_update_protocol_test",
    "--servers", "127.0.0.1:1234",
    "--storage-peers", "127.0.0.1:1234",
    "--index-prefix", "/tmp/index",
    "--threads", "1",
    "--dim", "128",
    "--k", "10",
  };
  for (auto& argument : extra) {
    arguments.push_back(std::move(argument));
  }
  return parse_config(std::move(arguments));
}

template <typename Function>
void expect_configuration_rejected(Function&& function) {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    function();
    _exit(EXIT_SUCCESS);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) != EXIT_SUCCESS);
}

}  // namespace

int main() {
  const auto default_config = make_config(false);
  assert(default_config.enable_updates);
  assert(default_config.vector_id_namespace_size == default_config.max_vectors);
  // Mixed CPU/GPU traffic may transiently delay a CQE; the default must not
  // retain the old 20 ms false-failure threshold.
  assert(default_config.gpu_direct_timeout_ms == 250);
  assert(default_config.gpu_query_beam_merge_policy == "legacy");
  assert(default_config.dynamic_graph_access_mode == "manual");
  assert(default_config.gpu_query_graph_read_policy == "fixed");
  assert(default_config.gpu_dynamic_graph_extent);
  assert(default_config.gpu_rdma_search_progression_mode == "manual");
  assert(default_config.storage_owner_update_completion_mode ==
         "decoupled");
  assert(!default_config.synchronous_exact_updates_enabled());

  const auto coupled_updates = make_config(
    false, false, {}, {}, {}, {}, {}, {}, {}, {}, "COUPLED");
  assert(coupled_updates.storage_owner_update_completion_mode == "coupled");
  assert(coupled_updates.synchronous_exact_updates_enabled());

  const auto stable_run_config =
    make_config(false, false, "STABLE-RUN");
  assert(stable_run_config.gpu_query_beam_merge_policy == "stable-run");
  const auto live_extent_config =
    make_config(false, false, {}, "LIVE-EXTENT");
  assert(live_extent_config.gpu_query_graph_read_policy == "live-extent");
  const auto header_neighbor_config =
    make_config(false, false, {}, "HEADER-NEIGHBOR");
  assert(header_neighbor_config.gpu_query_graph_read_policy ==
         "header-neighbor");
  const auto static_only_extent_config =
    make_config(false, false, {}, "LIVE-EXTENT", "false");
  assert(static_only_extent_config.gpu_query_graph_read_policy ==
         "live-extent");
  assert(!static_only_extent_config.gpu_dynamic_graph_extent);

  const auto fixed_graph_access = make_config(
    false, false, {}, {}, {}, {}, "FIXED");
  assert(fixed_graph_access.dynamic_graph_access_mode == "fixed");
  assert(fixed_graph_access.gpu_query_graph_read_policy == "fixed");
  assert(!fixed_graph_access.gpu_dynamic_graph_extent);
  assert(!fixed_graph_access.adaptive_dynamic_graph_access_enabled());

  const auto adaptive_graph_access = make_config(
    false, false, {}, {}, {}, {}, "ADAPTIVE");
  assert(adaptive_graph_access.dynamic_graph_access_mode == "adaptive");
  assert(adaptive_graph_access.gpu_query_graph_read_policy == "live-extent");
  assert(adaptive_graph_access.gpu_dynamic_graph_extent);
  assert(adaptive_graph_access.adaptive_dynamic_graph_access_enabled());

  const auto coupled_progression = make_config(
    false, false, {}, {}, {}, {}, {}, "COUPLED", "16", "0");
  assert(coupled_progression.gpu_rdma_search_progression_mode == "coupled");
  assert(coupled_progression.gpu_graph_commit_width == 16);
  assert(coupled_progression.gpu_graph_issue_width == 16);
  assert(coupled_progression.gpu_query_beam_merge_policy == "legacy");
  assert(!coupled_progression.decoupled_gpu_rdma_search_progression_enabled());

  const auto decoupled_progression = make_config(
    false, false, {}, {}, {}, {}, {}, "DECOUPLED", "16", "32");
  assert(decoupled_progression.gpu_rdma_search_progression_mode ==
         "decoupled");
  assert(decoupled_progression.gpu_graph_commit_width == 16);
  assert(decoupled_progression.gpu_graph_issue_width == 32);
  assert(decoupled_progression.gpu_query_beam_merge_policy == "stable-run");
  assert(decoupled_progression.decoupled_gpu_rdma_search_progression_enabled());

  assert(decoupled_progression.gpu_exact_frontier_early_issue);
  assert(coupled_updates.storage_owner_stage2_graph_issue_width == 1);
  assert(!coupled_updates.storage_owner_stage2_score_many);
  assert(!coupled_updates.storage_owner_stage2_home_rpc_combining);
  assert(!coupled_updates.storage_owner_stage2_split_plane);
  assert(!coupled_updates.storage_owner_lease_merge);
  assert(coupled_updates.storage_owner_repair_deadline_ms == 0);

  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-stage2-score-many", "false"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-stage2-graph-issue-width", "1"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-repair-deadline-ms", "0"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--gpu-rdma-search-progression-mode", "decoupled",
      "--gpu-graph-commit-width", "16",
      "--gpu-graph-issue-width", "32",
      "--gpu-exact-frontier-early-issue", "false"});
  });

  // Formal contribution modes reject an explicitly requested half-enabled
  // lower-level configuration. Manual mode above remains available for the
  // static-only and other engineering ablations.
  expect_configuration_rejected([] {
    (void)make_config(
      false, false, {}, "live-extent", {}, {}, "fixed");
  });
  expect_configuration_rejected([] {
    (void)make_config(
      false, false, {}, {}, {}, {}, {}, "decoupled", "16", "16");
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({"--alpha", "nan"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({"--alpha", "0.999"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({"--gpu-device", "2147483648"});
  });

  const auto expanded_namespace = make_config(false, true);
  assert(expanded_namespace.max_vectors == 1'000'000);
  assert(expanded_namespace.vector_id_namespace_size == 2'000'000);

  const auto query_only_config = make_config(true);
  assert(!query_only_config.enable_updates);

  assert(default_config.storage_owner_stage2_home_rpc_combining);
  const auto direct_home_rpc_config =
    make_config(false, false, {}, {}, {}, "false", {}, {}, {}, {}, "manual");
  assert(!direct_home_rpc_config.storage_owner_stage2_home_rpc_combining);

  // Stage2 is mandatory for the sole supported update pipeline.
  assert(default_config.storage_owner_maintenance_workers > 0);

  // Backlog-budget controls default off; decoupled supplies its repair target.
  assert(default_config.storage_owner_stage2_budget_backlog_low == 0);
  assert(default_config.storage_owner_stage2_budget_backlog_high == 0);
  assert(default_config.storage_owner_stage2_budget_max_expansions == 0);
  assert(default_config.storage_owner_stage2_budget_max_candidate_visits == 0);
  assert(default_config.storage_owner_repair_deadline_ms == 1000);
  assert(default_config.storage_owner_flow_control_poll_ms == 5);

  const auto budgeted = make_config_with_extra({
    "--storage-owner-stage2-budget-backlog-low", "1000",
    "--storage-owner-stage2-budget-backlog-high", "8000",
    "--storage-owner-stage2-budget-max-expansions", "8",
    "--storage-owner-stage2-budget-max-candidate-visits", "512"});
  assert(budgeted.storage_owner_stage2_budget_backlog_low == 1000);
  assert(budgeted.storage_owner_stage2_budget_backlog_high == 8000);
  assert(budgeted.storage_owner_stage2_budget_max_expansions == 8);
  assert(budgeted.storage_owner_stage2_budget_max_candidate_visits == 512);

  const auto deadline_controlled = make_config_with_extra({
    "--storage-owner-repair-deadline-ms", "1000"});
  assert(deadline_controlled.storage_owner_repair_deadline_ms == 1000);

  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-ack-backlog-limit", "4096"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--gpu-graph-prefetch-depth", "16"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-stage2-budget-backlog-low", "1000"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-stage2-budget-backlog-low", "8000",
      "--storage-owner-stage2-budget-backlog-high", "1000"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-flow-control-backlog-low", "2000"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-flow-control-backlog-high", "16000"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-flow-control-max-delay-us", "20000"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({
      "--storage-owner-flow-control-poll-ms", "0"});
  });

  // Stability bounds for the OOM/retry hardening changes.
  assert(default_config.storage_owner_stage2_max_retry_ms == 300'000);
  assert(default_config.storage_owner_reverse_conflict_limit == 32);
  assert(default_config.storage_owner_reverse_retry_backoff_max_us ==
         100'000);
  assert(default_config.storage_owner_exact_max_attempts == 0);
  assert(default_config.storage_owner_lease_merge);
  const auto legacy_merge = make_config_with_extra({
    "--storage-owner-update-completion-mode", "manual",
    "--storage-owner-lease-merge", "false"});
  assert(!legacy_merge.storage_owner_lease_merge);

  const auto stability_bounds = make_config_with_extra({
    "--storage-owner-stage2-max-retry-ms", "60000",
    "--storage-owner-reverse-conflict-limit", "4",
    "--storage-owner-reverse-retry-backoff-max-us", "5000",
    "--storage-owner-exact-max-attempts", "7"});
  assert(stability_bounds.storage_owner_stage2_max_retry_ms == 60'000);
  assert(stability_bounds.storage_owner_reverse_conflict_limit == 4);
  assert(stability_bounds.storage_owner_reverse_retry_backoff_max_us == 5'000);
  assert(stability_bounds.storage_owner_exact_max_attempts == 7);

  // Zero is a legal explicit value: unbounded Stage2 retry window and
  // unbounded synchronous-exact attempts.
  const auto unbounded_retry = make_config_with_extra({
    "--storage-owner-stage2-max-retry-ms", "0",
    "--storage-owner-exact-max-attempts", "0"});
  assert(unbounded_retry.storage_owner_stage2_max_retry_ms == 0);
  assert(unbounded_retry.storage_owner_exact_max_attempts == 0);

  const auto full_profile = make_config_with_extra({"--profile", "full"});
  assert(full_profile.c1 == "on" && full_profile.c2 == "on" &&
         full_profile.c3 == "on");
  assert(full_profile.storage_owner_update_completion_mode == "decoupled");
  assert(full_profile.storage_owner_stage2_score_many);
  assert(full_profile.storage_owner_stage2_home_rpc_combining);
  assert(full_profile.storage_owner_stage2_split_plane);
  assert(full_profile.storage_owner_lease_merge);
  assert(full_profile.storage_owner_batch_max_wait_us == 10'000);
  assert(full_profile.storage_owner_stage2_batch_max_wait_us == 25'000);
  assert(full_profile.storage_owner_reverse_coalesce_max == 256);
  assert(full_profile.dynamic_graph_access_mode == "adaptive");
  assert(full_profile.gpu_query_graph_read_policy == "live-extent");
  assert(full_profile.gpu_dynamic_graph_extent_hints);
  assert(full_profile.gpu_rdma_search_progression_mode == "decoupled");
  assert(full_profile.gpu_query_beam_merge_policy == "stable-run");
  assert(full_profile.gpu_exact_frontier_early_issue);
  assert(full_profile.gpu_graph_issue_width == 32);

  const auto baseline_profile = make_config_with_extra({"--profile", "baseline"});
  assert(baseline_profile.c1 == "off" && baseline_profile.c2 == "off" &&
         baseline_profile.c3 == "off");
  assert(baseline_profile.storage_owner_update_completion_mode == "coupled");
  assert(!baseline_profile.storage_owner_stage2_score_many);
  assert(!baseline_profile.storage_owner_stage2_home_rpc_combining);
  assert(!baseline_profile.storage_owner_stage2_split_plane);
  assert(!baseline_profile.storage_owner_lease_merge);
  assert(baseline_profile.storage_owner_batch_max_wait_us == 0);
  assert(baseline_profile.storage_owner_stage2_batch_max_wait_us == 0);
  assert(baseline_profile.storage_owner_reverse_coalesce_max == 1);
  assert(baseline_profile.dynamic_graph_access_mode == "fixed");
  assert(baseline_profile.gpu_query_graph_read_policy == "fixed");
  assert(!baseline_profile.gpu_dynamic_graph_extent_hints);
  assert(baseline_profile.gpu_rdma_search_progression_mode == "manual");
  assert(baseline_profile.gpu_query_beam_merge_policy == "legacy");
  assert(!baseline_profile.gpu_exact_frontier_early_issue);
  assert(baseline_profile.gpu_graph_issue_width ==
         baseline_profile.gpu_graph_commit_width);

  const auto all_off = make_config_with_extra(
    {"--c1", "off", "--c2", "off", "--c3", "off"});
  assert(all_off.storage_owner_update_completion_mode ==
         baseline_profile.storage_owner_update_completion_mode);
  assert(all_off.dynamic_graph_access_mode ==
         baseline_profile.dynamic_graph_access_mode);
  assert(all_off.gpu_rdma_search_progression_mode ==
         baseline_profile.gpu_rdma_search_progression_mode);
  assert(all_off.gpu_query_beam_merge_policy == "legacy");

  const auto all_on = make_config_with_extra(
    {"--c1", "on", "--c2", "on", "--c3", "on"});
  assert(all_on.storage_owner_update_completion_mode ==
         full_profile.storage_owner_update_completion_mode);
  assert(all_on.dynamic_graph_access_mode ==
         full_profile.dynamic_graph_access_mode);
  assert(all_on.gpu_rdma_search_progression_mode ==
         full_profile.gpu_rdma_search_progression_mode);
  assert(all_on.gpu_query_beam_merge_policy == "stable-run");

  const auto c1_only_off = make_config_with_extra(
    {"--profile", "full", "--c1", "off"});
  assert(c1_only_off.storage_owner_update_completion_mode == "coupled");
  assert(c1_only_off.dynamic_graph_access_mode == "adaptive");
  assert(c1_only_off.gpu_rdma_search_progression_mode == "decoupled");
  const auto c2_only_off = make_config_with_extra(
    {"--profile", "full", "--c2", "off"});
  assert(c2_only_off.storage_owner_update_completion_mode == "decoupled");
  assert(c2_only_off.dynamic_graph_access_mode == "fixed");
  assert(c2_only_off.gpu_rdma_search_progression_mode == "decoupled");
  const auto c3_only_off = make_config_with_extra(
    {"--profile", "full", "--c3", "off"});
  assert(c3_only_off.storage_owner_update_completion_mode == "decoupled");
  assert(c3_only_off.dynamic_graph_access_mode == "adaptive");
  assert(c3_only_off.gpu_rdma_search_progression_mode == "manual");
  assert(c3_only_off.gpu_query_beam_merge_policy == "legacy");

  const auto full_override = make_config_with_extra(
    {"--profile", "full", "--gpu-terminal-exact-prefetch", "true"});
  assert(full_override.gpu_terminal_exact_prefetch);
  expect_configuration_rejected([] {
    (void)make_config_with_extra({"--profile", "unknown"});
  });
  expect_configuration_rejected([] {
    (void)make_config_with_extra({"--c2", "invalid"});
  });
  const auto baseline_override = make_config_with_extra({
    "--profile", "baseline", "--gpu-query-beam-merge-policy", "stable-run"});
  assert(baseline_override.c3 == "custom");
  assert(baseline_override.gpu_query_beam_merge_policy == "stable-run");
  const auto c1_override = make_config_with_extra({
    "--profile", "full", "--storage-owner-stage2-score-many", "false"});
  assert(c1_override.c1 == "custom");
  assert(!c1_override.storage_owner_stage2_score_many);
  const auto c2_override = make_config_with_extra({
    "--profile", "full", "--gpu-dynamic-graph-extent-hints", "false"});
  assert(c2_override.c2 == "custom");
  assert(!c2_override.gpu_dynamic_graph_extent_hints);
  assert(full_override.c3 == "custom");
  return 0;
}

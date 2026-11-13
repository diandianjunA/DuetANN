#include "common/configuration.hh"

namespace configuration {

  IndexConfiguration::IndexConfiguration(int argc, char** argv) {
    add_options();
    const bool graph_read_policy_explicit = command_line_has_option(
      argc, argv, "--gpu-query-graph-read-policy");
    const bool dynamic_graph_extent_explicit = command_line_has_option(
      argc, argv, "--gpu-dynamic-graph-extent");
    const bool graph_issue_width_explicit = command_line_has_option(
      argc, argv, "--gpu-graph-issue-width");
    const bool beam_merge_policy_explicit = command_line_has_option(
      argc, argv, "--gpu-query-beam-merge-policy");
    process_program_options(argc, argv);
    if (vector_id_namespace_size == 0) {
      vector_id_namespace_size = max_vectors;
    }
    vector_data_type = normalize_mode(vector_data_type);
    dynamic_graph_access_mode = normalize_mode(dynamic_graph_access_mode);
    gpu_query_graph_read_policy =
      normalize_mode(gpu_query_graph_read_policy);
    gpu_rdma_search_progression_mode =
      normalize_mode(gpu_rdma_search_progression_mode);
    gpu_query_beam_merge_policy =
      normalize_mode(gpu_query_beam_merge_policy);
    query_rdma_trace_mode = normalize_mode(query_rdma_trace_mode);
    storage_owner_update_completion_mode =
      normalize_mode(storage_owner_update_completion_mode);
    resolve_feature_bundles(
      argc, argv, graph_read_policy_explicit, dynamic_graph_extent_explicit,
      graph_issue_width_explicit, beam_merge_policy_explicit);
    validate(argv);
    operator<<(std::cerr, *this);
  }

  void IndexConfiguration::resolve_feature_bundles(
      int argc, char** argv,
      bool graph_read_policy_explicit,
      bool dynamic_graph_extent_explicit,
      bool graph_issue_width_explicit,
      bool beam_merge_policy_explicit) {
    if (dynamic_graph_access_mode == "fixed") {
      if (graph_read_policy_explicit &&
          gpu_query_graph_read_policy != "fixed") {
        record_feature_bundle_conflict(
          "--gpu-dynamic-graph-access-mode=fixed conflicts with "
          "--gpu-query-graph-read-policy");
      }
      if (dynamic_graph_extent_explicit && gpu_dynamic_graph_extent) {
        record_feature_bundle_conflict(
          "--gpu-dynamic-graph-access-mode=fixed conflicts with "
          "--gpu-dynamic-graph-extent=true");
      }
      gpu_query_graph_read_policy = "fixed";
      gpu_dynamic_graph_extent = false;
    } else if (dynamic_graph_access_mode == "adaptive") {
      if (graph_read_policy_explicit &&
          gpu_query_graph_read_policy != "live-extent") {
        record_feature_bundle_conflict(
          "--gpu-dynamic-graph-access-mode=adaptive conflicts with "
          "--gpu-query-graph-read-policy");
      }
      if (dynamic_graph_extent_explicit && !gpu_dynamic_graph_extent) {
        record_feature_bundle_conflict(
          "--gpu-dynamic-graph-access-mode=adaptive conflicts with "
          "--gpu-dynamic-graph-extent=false");
      }
      gpu_query_graph_read_policy = "live-extent";
      gpu_dynamic_graph_extent = true;
    }

    const bool coupled_progression =
      gpu_rdma_search_progression_mode == "coupled";
    const bool decoupled_progression =
      gpu_rdma_search_progression_mode == "decoupled";
    if (coupled_progression) {
      if (graph_issue_width_explicit && gpu_graph_issue_width != 0 &&
          gpu_graph_issue_width != gpu_graph_commit_width) {
        record_feature_bundle_conflict(
          "--gpu-rdma-search-progression-mode=coupled requires "
          "--gpu-graph-issue-width to equal the commit width");
      }
      if (beam_merge_policy_explicit &&
          gpu_query_beam_merge_policy != "legacy") {
        record_feature_bundle_conflict(
          "--gpu-rdma-search-progression-mode=coupled conflicts with "
          "--gpu-query-beam-merge-policy");
      }
      gpu_graph_issue_width = gpu_graph_commit_width;
      gpu_query_beam_merge_policy = "legacy";
    } else {
      if (gpu_graph_issue_width == 0) {
        gpu_graph_issue_width =
          gpu_search::adaptive_frontier::automatic_max_issue_width(
            gpu_traversal_beam_width);
        // A historical configuration could legally use a prefetch depth
        // larger than its traversal beam. Keep that manual configuration
        // valid instead of silently narrowing the authoritative width.
        if (!decoupled_progression &&
            gpu_graph_issue_width < gpu_graph_commit_width) {
          gpu_graph_issue_width = gpu_graph_commit_width;
        }
      }
      if (decoupled_progression) {
        if (gpu_graph_issue_width <= gpu_graph_commit_width) {
          record_feature_bundle_conflict(
            "--gpu-rdma-search-progression-mode=decoupled requires an "
            "issue width greater than the commit width");
        }
        if (beam_merge_policy_explicit &&
            gpu_query_beam_merge_policy != "stable-run") {
          record_feature_bundle_conflict(
            "--gpu-rdma-search-progression-mode=decoupled conflicts with "
            "--gpu-query-beam-merge-policy");
        }
        if (command_line_has_option(
              argc, argv, "--gpu-exact-frontier-early-issue") &&
            !gpu_exact_frontier_early_issue) {
          record_feature_bundle_conflict(
            "--gpu-rdma-search-progression-mode=decoupled conflicts with "
            "--gpu-exact-frontier-early-issue=false");
        }
        gpu_query_beam_merge_policy = "stable-run";
        gpu_exact_frontier_early_issue = true;
      } else if (coupled_progression) {
        if (command_line_has_option(
              argc, argv, "--gpu-exact-frontier-early-issue") &&
            gpu_exact_frontier_early_issue) {
          record_feature_bundle_conflict(
            "--gpu-rdma-search-progression-mode=coupled conflicts with "
            "--gpu-exact-frontier-early-issue=true");
        }
        gpu_exact_frontier_early_issue = false;
      }
    }

    const auto require_update_switch = [&](const char* option, bool value,
                                           bool expected, const char* mode) {
      if (command_line_has_option(argc, argv, option) && value != expected) {
        record_feature_bundle_conflict(
          std::string("--storage-owner-update-completion-mode=") + mode +
          " conflicts with " + option);
      }
    };
    if (storage_owner_update_completion_mode == "decoupled") {
      require_update_switch("--storage-owner-stage2-score-many",
                            storage_owner_stage2_score_many, true, "decoupled");
      require_update_switch("--storage-owner-stage2-home-rpc-combining",
                            storage_owner_stage2_home_rpc_combining, true,
                            "decoupled");
      require_update_switch("--storage-owner-stage2-split-plane",
                            storage_owner_stage2_split_plane, true, "decoupled");
      require_update_switch("--storage-owner-lease-merge",
                            storage_owner_lease_merge, true, "decoupled");
      if (command_line_has_option(
            argc, argv, "--storage-owner-stage2-graph-issue-width") &&
          storage_owner_stage2_graph_issue_width <= 1) {
        record_feature_bundle_conflict(
          "--storage-owner-update-completion-mode=decoupled requires a "
          "Stage2 graph issue width greater than one");
      }
      if (command_line_has_option(
            argc, argv, "--storage-owner-repair-deadline-ms") &&
          storage_owner_repair_deadline_ms == 0) {
        record_feature_bundle_conflict(
          "--storage-owner-update-completion-mode=decoupled requires a "
          "positive repair deadline");
      }
      storage_owner_stage2_score_many = true;
      storage_owner_stage2_home_rpc_combining = true;
      storage_owner_stage2_split_plane = true;
      storage_owner_lease_merge = true;
      if (storage_owner_stage2_graph_issue_width <= 1) {
        storage_owner_stage2_graph_issue_width = 16;
      }
      if (storage_owner_repair_deadline_ms == 0) {
        storage_owner_repair_deadline_ms = 1000;
      }
    } else if (storage_owner_update_completion_mode == "coupled") {
      require_update_switch("--storage-owner-stage2-score-many",
                            storage_owner_stage2_score_many, false, "coupled");
      require_update_switch("--storage-owner-stage2-home-rpc-combining",
                            storage_owner_stage2_home_rpc_combining, false,
                            "coupled");
      require_update_switch("--storage-owner-stage2-split-plane",
                            storage_owner_stage2_split_plane, false, "coupled");
      require_update_switch("--storage-owner-lease-merge",
                            storage_owner_lease_merge, false, "coupled");
      storage_owner_stage2_score_many = false;
      storage_owner_stage2_home_rpc_combining = false;
      storage_owner_stage2_split_plane = false;
      storage_owner_lease_merge = false;
      storage_owner_stage2_graph_issue_width = 1;
      storage_owner_repair_deadline_ms = 0;
    }

  }

  void IndexConfiguration::add_options() {
    desc.add_options()
      ("index-prefix", po::value<filepath_t>(&index_prefix),
       "Prefix shared by metadata, graph shards, PQ model, and PQ code shards.")
      ("server-index-file", po::value<filepath_t>(&server_index_file),
       "Local graph shard loaded by this storage node before serving requests.")
      ("threads,t", po::value<u32>(&num_threads),
       "CPU control/update threads represented by this process.")
      ("disable-thread-pinning,p",
       po::bool_switch(&disable_thread_pinning)->default_value(false),
       "Disable CPU thread pinning.")
      ("seed", po::value<i32>(&seed)->default_value(seed),
       "Deterministic random seed.")
      ("dim", po::value<u32>(&dim), "Vector dimension.")
      ("max-vectors", po::value<u32>(&max_vectors)->default_value(max_vectors),
       "Number of immutable base vectors in the loaded index.")
      ("vector-id-namespace-size",
       po::value<u32>(&vector_id_namespace_size)
         ->default_value(vector_id_namespace_size),
       "Exclusive upper bound for base and dynamically assigned vector IDs.")
      ("k", po::value<u32>(&k), "Requested nearest-neighbor count.")
      ("R", po::value<u32>(&R)->default_value(R), "Maximum graph out-degree.")
      ("beam-width-construction",
       po::value<u32>(&beam_width_construction)->default_value(beam_width_construction),
       "Beam width used by storage-side online graph maintenance.")
      ("alpha", po::value<f64>(&alpha)->default_value(alpha),
       "RobustPrune diversity factor.")
      ("vector-data-type",
       po::value<str>(&vector_data_type)->default_value(vector_data_type),
       "Exact-vector storage type: auto, float32, uint8, or int8.")

      ("gpu-device", po::value<u32>(&gpu_device)->default_value(gpu_device),
       "CUDA device used by the persistent query engine.")
      ("enable-breakdown",
       po::value<bool>(&enable_breakdown)->default_value(enable_breakdown),
       "Collect per-request breakdown samples.")
      ("gpu-query-slots",
       po::value<u32>(&gpu_query_slots)->default_value(gpu_query_slots),
       "Maximum concurrent GPU query slots.")
      ("gpu-memory-limit-gb",
       po::value<u32>(&gpu_memory_limit_gb)->default_value(gpu_memory_limit_gb),
       "Hard limit for explicit query-engine GPU allocations.")
      ("gpu-memory-reserve-gb",
       po::value<u32>(&gpu_memory_reserve_gb)->default_value(gpu_memory_reserve_gb),
       "GPU memory reserved for CUDA and transport runtime state.")
      ("gpu-bootstrap-window-mb",
       po::value<u32>(&gpu_bootstrap_window_mb)->default_value(gpu_bootstrap_window_mb),
       "Maximum one-time PQ bootstrap RDMA read size.")
      ("gpu-bootstrap-windows",
       po::value<u32>(&gpu_bootstrap_windows)->default_value(gpu_bootstrap_windows),
       "Concurrent one-time PQ bootstrap reads.")
      ("gpu-graph-commit-width",
       po::value<u32>(&gpu_graph_commit_width)->default_value(gpu_graph_commit_width),
       "Authoritative graph expansion width; defaults to 32.")
      ("gpu-graph-issue-width",
       po::value<u32>(&gpu_graph_issue_width)->default_value(gpu_graph_issue_width),
       "Maximum speculative graph read width; zero derives from frontier capacity and traversal beam.")
      ("gpu-dynamic-graph-access-mode",
       po::value<str>(&dynamic_graph_access_mode)
         ->default_value(dynamic_graph_access_mode),
       "Contribution mode: fixed, adaptive, or manual for lower-level ablations.")
      ("gpu-query-graph-read-policy",
       po::value<str>(&gpu_query_graph_read_policy)
         ->default_value(gpu_query_graph_read_policy),
       "GPU graph-record transfer policy: fixed, header-neighbor, or live-extent.")
      ("gpu-dynamic-graph-extent",
       po::value<bool>(&gpu_dynamic_graph_extent)
         ->default_value(gpu_dynamic_graph_extent),
       "Use dynamic PQ tags as quantized-extent bootstrap hints when "
       "live-extent graph reads are enabled.")
      ("gpu-dynamic-graph-extent-hints",
       po::value<bool>(&gpu_dynamic_graph_extent_hints)
         ->default_value(gpu_dynamic_graph_extent_hints),
       "Enable sidecar and dynamic PQ-tag cold-start hints while preserving "
       "validated online extent adaptation.")
      ("gpu-rdma-search-progression-mode",
       po::value<str>(&gpu_rdma_search_progression_mode)
         ->default_value(gpu_rdma_search_progression_mode),
       "Contribution mode: coupled, decoupled, or manual for lower-level ablations.")
      ("gpu-exact-frontier-early-issue",
       po::value<bool>(&gpu_exact_frontier_early_issue)
         ->default_value(gpu_exact_frontier_early_issue),
       "Issue the exact mandatory next frontier before full Beam publication; "
       "manual mode may use issue-width equal to commit-width to disable the "
       "speculative tail.")
      ("gpu-terminal-exact-prefetch",
       po::value<bool>(&gpu_terminal_exact_prefetch)
         ->default_value(gpu_terminal_exact_prefetch),
       "Prefetch terminal exact snapshots after the final graph frontier is issued.")
      ("gpu-query-beam-merge-policy",
       po::value<str>(&gpu_query_beam_merge_policy)
         ->default_value(gpu_query_beam_merge_policy),
       "GPU query Beam merge policy: legacy or stable-run.")
      ("query-rdma-trace-mode",
       po::value<str>(&query_rdma_trace_mode)->default_value(query_rdma_trace_mode),
       "Shard-batch RDMA trace mode: off, sampled, or full.")
      ("query-rdma-trace-sample-rate",
       po::value<u32>(&query_rdma_trace_sample_rate)
         ->default_value(query_rdma_trace_sample_rate),
       "In sampled mode, trace one of every N request IDs.")
      ("query-rdma-trace-output",
       po::value<filepath_t>(&query_rdma_trace_output)
         ->default_value(query_rdma_trace_output),
       "JSONL output for detailed shard-batch RDMA trace events.")
      ("query-rdma-trace-events-per-query",
       po::value<u32>(&query_rdma_trace_events_per_query)
         ->default_value(query_rdma_trace_events_per_query),
       "Preallocated per-query-slot trace event capacity.")
      ("gpu-traversal-beam-width",
       po::value<u32>(&gpu_traversal_beam_width)->default_value(gpu_traversal_beam_width),
       "OPQ/PQ beam width for GPU graph navigation.")
      ("gpu-final-rerank-width",
       po::value<u32>(&gpu_final_rerank_width)->default_value(gpu_final_rerank_width),
       "Exact vectors fetched for final reranking.")
      ("gpu-max-expansions",
       po::value<u32>(&gpu_max_expansions)->default_value(gpu_max_expansions),
       "Maximum graph expansions per query.")
      ("gpu-centroid-route-fanout",
       po::value<u32>(&gpu_centroid_route_fanout)->default_value(gpu_centroid_route_fanout),
       "Number of nearest centroid shards used as query entry points.")
      ("gpu-rdma-qps",
       po::value<u32>(&gpu_rdma_qps)->default_value(gpu_rdma_qps),
       "GPU-initiated GPUNetIO QPs per storage node.")
      ("gpu-direct-timeout-ms",
       po::value<u32>(&gpu_direct_timeout_ms)->default_value(gpu_direct_timeout_ms),
       "Maximum wait for one GPU-initiated RDMA completion before fail-stop.")
      ("gpu-persistent-blocks-per-sm",
       po::value<u32>(&gpu_persistent_blocks_per_sm)->default_value(gpu_persistent_blocks_per_sm),
       "Maximum unified persistent CTAs per GPU SM; hardware occupancy may be lower.")
      ("gpu-device-stack-bytes",
       po::value<u32>(&gpu_device_stack_bytes)->default_value(gpu_device_stack_bytes),
       "Minimum per-thread CUDA stack for the ASFE persistent kernel.")
      ("enable-updates",
       po::value<bool>(&enable_updates)->default_value(enable_updates),
       "Enable compute-side insert, upsert, and erase submission.")
      ("storage-owner-update-completion-mode",
       po::value<str>(&storage_owner_update_completion_mode)
         ->default_value(storage_owner_update_completion_mode),
       "Update contract: coupled baseline, decoupled full Stage1/Stage2 "
       "optimization bundle, or manual engineering ablation.")
      ("storage-id", po::value<u32>(&storage_id)->default_value(storage_id),
       "Zero-based storage shard identifier.")
      ("storage-peers", po::value<vec<str>>(&storage_peers)->multitoken(),
       "Ordered storage-node endpoints.")
      ("storage-owner-batch-max",
       po::value<u32>(&storage_owner_batch_max)->default_value(storage_owner_batch_max),
       "Maximum mutations in one storage RPC batch.")
      ("storage-owner-batch-max-wait-us",
       po::value<u32>(&storage_owner_batch_max_wait_us)
         ->default_value(storage_owner_batch_max_wait_us),
       "Maximum foreground wait while an announced initial mutation batch is "
       "still being published; zero runs that partial batch immediately.")
      ("storage-owner-stage2-batch-max-wait-us",
       po::value<u32>(&storage_owner_stage2_batch_max_wait_us)
         ->default_value(storage_owner_stage2_batch_max_wait_us),
       "Maximum background Stage2 compaction wait for a partial batch; zero "
       "runs partial Stage2 batches immediately.")
      ("storage-owner-stage2-initial-delay-ms",
       po::value<u32>(&storage_owner_stage2_initial_delay_ms)
         ->default_value(storage_owner_stage2_initial_delay_ms),
       "Diagnostic minimum descriptor age before Stage2 admission; zero "
       "disables the gate.")
      ("storage-owner-stage2-score-many",
       po::value<bool>(&storage_owner_stage2_score_many)
         ->default_value(storage_owner_stage2_score_many),
       "Score remote Stage2 candidates at their physical home using a "
       "query-deduplicated exact score-many RPC.")
      ("storage-owner-stage2-score-many-min-items",
       po::value<u32>(&storage_owner_stage2_score_many_min_items)
         ->default_value(storage_owner_stage2_score_many_min_items),
       "Minimum pending candidates per peer before using score-many RPC.")
      ("storage-owner-stage2-home-rpc-combining",
       po::value<bool>(&storage_owner_stage2_home_rpc_combining)
         ->default_value(storage_owner_stage2_home_rpc_combining),
       "Combine compatible logical Stage2 home requests into one wire RPC.")
      ("storage-owner-stage2-split-plane",
       po::value<bool>(&storage_owner_stage2_split_plane)
         ->default_value(storage_owner_stage2_split_plane),
       "Allow independent Stage2 score and graph home RPC planes; false "
       "serializes the two planes for ablation without changing updates.")
      ("storage-owner-lease-merge",
       po::value<bool>(&storage_owner_lease_merge)
         ->default_value(storage_owner_lease_merge),
       "Use cross-context certificate-lease Atomic Cuts; false runs the legacy "
       "promotion and globally ordered cleanup ablation.")
      ("storage-owner-stage2-graph-issue-width",
       po::value<u32>(&storage_owner_stage2_graph_issue_width)
         ->default_value(storage_owner_stage2_graph_issue_width),
       "Maximum ordered speculative Stage2 graph issue width; one disables "
       "speculation and preserves the legacy request path.")
      ("storage-owner-peer-rdma-tokens",
       po::value<u32>(&storage_owner_peer_rdma_tokens)->default_value(storage_owner_peer_rdma_tokens),
       "Outstanding peer reads allowed per storage data QP.")
      ("storage-owner-peer-qps-per-peer",
       po::value<u32>(&storage_owner_peer_qps_per_peer)
         ->default_value(storage_owner_peer_qps_per_peer),
       "Storage-to-storage RC QPs per peer, including one ordered control QP.")
      ("storage-owner-rpc-depth",
       po::value<u32>(&storage_owner_rpc_depth)->default_value(storage_owner_rpc_depth),
       "In-flight mutation batches per storage node.")
      ("storage-owner-rpc-timeout-ms",
       po::value<u32>(&storage_owner_rpc_timeout_ms)->default_value(storage_owner_rpc_timeout_ms),
       "Mutation RPC timeout.")
      ("storage-owner-exact-max-attempts",
       po::value<u32>(&storage_owner_exact_max_attempts)
         ->default_value(storage_owner_exact_max_attempts),
       "Maximum parent-discovery and bridge retries for one synchronous exact "
       "mutation; zero retries without a bound.")
      ("storage-owner-search-snapshot-batch",
       po::value<u32>(&storage_owner_search_snapshot_batch)
         ->default_value(storage_owner_search_snapshot_batch),
       "Concurrent node snapshots during update search.")
      ("storage-owner-maintenance-workers",
       po::value<u32>(&storage_owner_maintenance_workers)
         ->default_value(storage_owner_maintenance_workers),
       "Background exact-finalization workers.")
      ("storage-owner-maintenance-queue-depth",
       po::value<u32>(&storage_owner_maintenance_queue_depth)
         ->default_value(storage_owner_maintenance_queue_depth),
       "Bounded graph-maintenance backlog; writers backpressure at the limit.")
      ("storage-owner-stage2-max-retry-ms",
       po::value<u32>(&storage_owner_stage2_max_retry_ms)
         ->default_value(storage_owner_stage2_max_retry_ms),
       "Maximum permanent-failure retry window for one Stage2 context or "
       "stable-refinement task before execution stops with an explicit "
       "failure; zero retries forever.")
      ("storage-owner-stage2-budget-backlog-low",
       po::value<u32>(&storage_owner_stage2_budget_backlog_low)
         ->default_value(storage_owner_stage2_budget_backlog_low),
       "Stage2 backlog below which the continuation search budget stays "
       "unbounded; zero disables backlog-aware budgeting.")
      ("storage-owner-stage2-budget-backlog-high",
       po::value<u32>(&storage_owner_stage2_budget_backlog_high)
         ->default_value(storage_owner_stage2_budget_backlog_high),
       "Stage2 backlog at which the continuation search budget reaches its "
       "configured tightest values.")
      ("storage-owner-stage2-budget-max-expansions",
       po::value<u64>(&storage_owner_stage2_budget_max_expansions)
         ->default_value(storage_owner_stage2_budget_max_expansions),
       "Stage2 continuation expansion budget at the high backlog watermark; "
       "zero keeps expansions unbounded.")
      ("storage-owner-stage2-budget-max-candidate-visits",
       po::value<u64>(&storage_owner_stage2_budget_max_candidate_visits)
         ->default_value(storage_owner_stage2_budget_max_candidate_visits),
       "Stage2 continuation candidate-visit budget at the high backlog "
       "watermark; zero keeps visits unbounded.")
      ("storage-owner-repair-deadline-ms",
       po::value<u64>(&storage_owner_repair_deadline_ms)
         ->default_value(storage_owner_repair_deadline_ms),
       "Target milliseconds from Stage1 completion to observed durable repair; "
       "zero disables deadline admission and miss accounting.")
      ("storage-owner-flow-control-poll-ms",
       po::value<u32>(&storage_owner_flow_control_poll_ms)
         ->default_value(storage_owner_flow_control_poll_ms),
       "Compute-side storage maintenance watermark refresh period.")
      ("storage-owner-reverse-queue-depth",
       po::value<u32>(&storage_owner_reverse_queue_depth)
         ->default_value(storage_owner_reverse_queue_depth),
       "Maximum queued reverse updates.")
      ("storage-owner-reverse-coalesce-max",
       po::value<u32>(&storage_owner_reverse_coalesce_max)
         ->default_value(storage_owner_reverse_coalesce_max),
       "Maximum reverse updates per coalesced batch.")
      ("storage-owner-reverse-conflict-limit",
       po::value<u32>(&storage_owner_reverse_conflict_limit)
         ->default_value(storage_owner_reverse_conflict_limit),
       "Optimistic reverse-update conflicts tolerated on one target before "
       "the receiver reports failure and the sender retries with backoff.")
      ("storage-owner-reverse-retry-backoff-max-us",
       po::value<u32>(&storage_owner_reverse_retry_backoff_max_us)
         ->default_value(storage_owner_reverse_retry_backoff_max_us),
       "Cap for the exponential reverse-aggregate resend backoff.")
      ("mn-memory", po::value<u32>(&mn_memory_gb)->default_value(mn_memory_gb),
       "Storage-node registered-memory capacity in GiB.");
  }

  void IndexConfiguration::validate(char** argv) const {
    const auto fail = [&](const str& message) {
      std::cerr << "[ERROR]: " << message << std::endl;
      exit_with_help_message(argv);
    };

    if (!feature_bundle_conflict_.empty()) {
      fail(feature_bundle_conflict_);
    }

    if (index_prefix.empty()) fail("--index-prefix is required");
    if (num_server_nodes() == 0 || num_server_nodes() > kMaxStorageShards) {
      fail("--servers must list between 1 and " +
           std::to_string(kMaxStorageShards) + " storage nodes");
    }
    if (mn_memory_gb > kMaxStorageRegionGiB) {
      fail("--mn-memory exceeds the tagged RemotePtr 256 GiB shard limit");
    }
    if (num_threads == 0 || dim == 0 || max_vectors == 0 || k == 0 ||
        R == 0 || beam_width_construction == 0 || mn_memory_gb == 0) {
      fail("threads, dim, max-vectors, k, R, beam-width-construction, and mn-memory must be > 0");
    }
    if (vector_id_namespace_size < max_vectors) {
      fail("--vector-id-namespace-size must be >= --max-vectors");
    }
    if (R > kMaxSupportedGraphDegree) {
      fail("--R must be <= " + std::to_string(kMaxSupportedGraphDegree));
    }
    if (!std::isfinite(alpha) || alpha < 1.0) {
      fail("--alpha must be finite and >= 1.0");
    }
    if (gpu_device > static_cast<u32>(std::numeric_limits<int>::max())) {
      fail("--gpu-device exceeds the CUDA runtime device-ordinal range");
    }
    if (k > gpu_final_rerank_width) {
      fail("--k must not exceed --gpu-final-rerank-width");
    }
    try {
      if (vector_data_type != "auto") (void)parse_vector_dtype(vector_data_type);
    } catch (const std::exception& error) {
      fail(str{"invalid --vector-data-type: "} + error.what());
    }

    if (query_rdma_trace_mode != "off" &&
        query_rdma_trace_mode != "sampled" &&
        query_rdma_trace_mode != "full") {
      fail("--query-rdma-trace-mode must be off, sampled, or full");
    }
    if (query_rdma_trace_sample_rate == 0 ||
        query_rdma_trace_events_per_query == 0 ||
        query_rdma_trace_events_per_query > 65'536) {
      fail("invalid query RDMA trace sampling rate or event capacity");
    }
    if (query_rdma_trace_mode != "off" && query_rdma_trace_output.empty()) {
      fail("--query-rdma-trace-output is required when tracing is enabled");
    }
    if (dynamic_graph_access_mode != "manual" &&
        dynamic_graph_access_mode != "fixed" &&
        dynamic_graph_access_mode != "adaptive") {
      fail("--gpu-dynamic-graph-access-mode must be fixed, adaptive, or manual");
    }
    if (gpu_query_graph_read_policy != "fixed" &&
        gpu_query_graph_read_policy != "header-neighbor" &&
        gpu_query_graph_read_policy != "live-extent") {
      fail("--gpu-query-graph-read-policy must be fixed, header-neighbor, "
           "or live-extent");
    }
    if (gpu_rdma_search_progression_mode != "manual" &&
        gpu_rdma_search_progression_mode != "coupled" &&
        gpu_rdma_search_progression_mode != "decoupled") {
      fail("--gpu-rdma-search-progression-mode must be coupled, decoupled, "
           "or manual");
    }
    if (gpu_rdma_search_progression_mode != "coupled" &&
        gpu_traversal_beam_width > kMaxPersistentTraversalBeam) {
      fail("persistent GPU traversal beam exceeds the supported maximum " +
           std::to_string(kMaxPersistentTraversalBeam));
    }
    if (gpu_query_beam_merge_policy != "legacy" &&
        gpu_query_beam_merge_policy != "stable-run") {
      fail("--gpu-query-beam-merge-policy must be legacy or stable-run");
    }
    if (storage_owner_update_completion_mode != "coupled" &&
        storage_owner_update_completion_mode != "decoupled" &&
        storage_owner_update_completion_mode != "manual") {
      fail("--storage-owner-update-completion-mode must be coupled, decoupled, "
           "or manual");
    }
    if (gpu_query_slots == 0 || gpu_query_slots > 4096 ||
        gpu_memory_limit_gb == 0 ||
        gpu_memory_reserve_gb >= gpu_memory_limit_gb ||
        gpu_bootstrap_window_mb == 0 ||
        gpu_bootstrap_window_mb >
          (std::numeric_limits<u32>::max() >> 20u) ||
        gpu_bootstrap_windows == 0 ||
        gpu_bootstrap_windows > 16 ||
        gpu_graph_commit_width == 0 ||
        gpu_graph_commit_width > gpu_graph_issue_width ||
        gpu_graph_issue_width >
          gpu_search::adaptive_frontier::kFrontierCapacity ||
        gpu_traversal_beam_width < k || gpu_traversal_beam_width > 256 ||
        gpu_final_rerank_width < k || gpu_final_rerank_width > 512 ||
        gpu_max_expansions < gpu_traversal_beam_width ||
        gpu_max_expansions > 4096 ||
        gpu_centroid_route_fanout == 0 || gpu_centroid_route_fanout > 64 ||
        gpu_rdma_qps == 0 || gpu_rdma_qps > 128 ||
        gpu_direct_timeout_ms < 20 || gpu_direct_timeout_ms > 5'000 ||
        gpu_persistent_blocks_per_sm == 0 ||
        gpu_persistent_blocks_per_sm > 16 ||
        gpu_device_stack_bytes < 2048 ||
        gpu_device_stack_bytes > 65'536 ||
        gpu_device_stack_bytes % 256 != 0) {
      fail("invalid persistent GPU query configuration");
    }

    if (storage_peers.size() != num_server_nodes()) {
      fail("--storage-peers must list exactly one endpoint per storage node");
    }
    if (storage_id >= num_server_nodes() ||
        storage_owner_batch_max == 0 ||
        storage_owner_peer_qps_per_peer == 0 ||
        storage_owner_peer_qps_per_peer > kMaxPeerQps ||
        storage_owner_peer_rdma_tokens == 0 ||
        storage_owner_rpc_depth == 0 ||
        storage_owner_rpc_timeout_ms == 0 ||
        storage_owner_search_snapshot_batch == 0 ||
        storage_owner_stage2_score_many_min_items == 0 ||
        storage_owner_stage2_score_many_min_items >
          storage_owner_search_snapshot_batch ||
        storage_owner_stage2_graph_issue_width == 0 ||
        storage_owner_stage2_graph_issue_width > storage_owner_batch_max ||
        storage_owner_maintenance_workers == 0 ||
        storage_owner_maintenance_queue_depth == 0 ||
        storage_owner_reverse_queue_depth == 0 ||
        storage_owner_reverse_coalesce_max == 0) {
      fail("invalid storage-side update configuration");
    }
    if (storage_owner_batch_max > std::numeric_limits<u32>::max() / R) {
      fail("storage-owner batch invalidation capacity exceeds u32");
    }
    if (static_cast<u64>(storage_owner_maintenance_queue_depth) <
        static_cast<u64>(storage_owner_batch_max) * 2) {
      fail("stage2 maintenance queue depth must cover two intents per RPC batch");
    }
    if ((storage_owner_stage2_budget_backlog_low == 0) !=
        (storage_owner_stage2_budget_backlog_high == 0)) {
      fail("--storage-owner-stage2-budget-backlog-low and "
           "--storage-owner-stage2-budget-backlog-high must be set together");
    }
    if (storage_owner_stage2_budget_backlog_low != 0 &&
        storage_owner_stage2_budget_backlog_low >=
          storage_owner_stage2_budget_backlog_high) {
      fail("--storage-owner-stage2-budget-backlog-low must be smaller than "
           "--storage-owner-stage2-budget-backlog-high");
    }
    if (storage_owner_flow_control_poll_ms == 0) {
      fail("--storage-owner-flow-control-poll-ms must be positive");
    }
    if (is_server && server_index_file.empty()) {
      fail("storage node requires --server-index-file");
    }
  }

std::ostream& operator<<(
    std::ostream& output, const IndexConfiguration& config) {
    output << static_cast<const Configuration&>(config);
    constexpr i32 width = 34;
    constexpr i32 line_width = 68;
    output << std::left << std::setfill(' ');

    if (config.is_initiator) {
      output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
      output << std::setw(width) << "threads: " << config.num_threads << '\n';
      output << std::setw(width) << "dimension: " << config.dim << '\n';
      output << std::setw(width) << "max vectors: " << config.max_vectors << '\n';
      output << std::setw(width) << "vector ID namespace [0,N): "
             << config.vector_id_namespace_size << '\n';
      output << std::setw(width) << "K / R: "
             << config.k << " / " << config.R << '\n';
      output << std::setw(width) << "vector data type: "
             << config.vector_data_type << '\n';
      output << std::setfill('-') << std::setw(line_width) << "" << '\n';
      output << std::setfill(' ');
      const bool persistent_gpu_engine =
        config.gpu_rdma_search_progression_mode != "coupled";
      output << std::setw(width) << "query engine: "
             << (persistent_gpu_engine
                   ? "persistent_gpu_opq_pq"
                   : "host_orchestrated_cpu_rdma_finite_cuda")
             << '\n';
      output << std::setw(width) << "remote transport: "
             << (persistent_gpu_engine
                   ? "GPU-initiated GPUNetIO"
                   : "CPU-posted one-sided RDMA")
             << '\n';
      output << std::setw(width) << "GPU device: " << config.gpu_device << '\n';
      output << std::setw(width) << "GPU concurrent query slots: "
             << config.gpu_query_slots << '\n';
      output << std::setw(width) << "GPU memory limit/reserve GiB: "
             << config.gpu_memory_limit_gb << "/"
             << config.gpu_memory_reserve_gb << '\n';
      output << std::setw(width) << "GPU traversal/rerank width: "
             << config.gpu_traversal_beam_width << "/"
             << config.gpu_final_rerank_width << '\n';
      output << std::setw(width) << "GPU max expansions: "
             << config.gpu_max_expansions << '\n';
      output << std::setw(width) << "GPU device stack bytes/thread: "
             << config.gpu_device_stack_bytes << '\n';
      output << std::setw(width) << "GPU graph commit/issue width: "
             << config.gpu_graph_commit_width << "/"
             << config.gpu_graph_issue_width << '\n';
      output << std::setw(width) << "dynamic graph access mode: "
             << config.dynamic_graph_access_mode << '\n';
      output << std::setw(width) << "GPU graph read policy: "
             << config.gpu_query_graph_read_policy << '\n';
      output << std::setw(width) << "GPU dynamic graph extent: "
             << config.gpu_dynamic_graph_extent << '\n';
      output << std::setw(width) << "GPU-RDMA progression mode: "
             << config.gpu_rdma_search_progression_mode << '\n';
      output << std::setw(width) << "GPU Beam merge policy: "
             << config.gpu_query_beam_merge_policy << '\n';
      output << std::setw(width) << "query RDMA trace mode/rate: "
             << config.query_rdma_trace_mode << "/"
             << config.query_rdma_trace_sample_rate << '\n';
      output << std::setw(width) << "query RDMA trace output: "
             << config.query_rdma_trace_output << '\n';
      output << std::setw(width) << "GPU RDMA QPs per storage node: "
             << config.gpu_rdma_qps << '\n';
      output << std::setw(width) << "GPU direct CQ timeout ms: "
             << config.gpu_direct_timeout_ms << '\n';
      output << std::setw(width) << "GPU persistent blocks/SM cap: "
             << config.gpu_persistent_blocks_per_sm << '\n';
      output << std::setw(width) << "compute updates enabled: "
             << std::boolalpha << config.enable_updates << '\n';
      output << std::setw(width) << "storage update completion mode: "
             << config.storage_owner_update_completion_mode << '\n';
      output << std::setw(width) << "storage update protocol: "
             << (config.synchronous_exact_updates_enabled()
                   ? "single-owner synchronous exact"
                   : "centroid-home two-stage") << '\n';
      output << std::setw(width) << "storage RPC depth/batch: "
             << config.storage_owner_rpc_depth << "/"
             << config.storage_owner_batch_max << '\n';
      output << std::setw(width) << "storage batch max wait us: "
             << config.storage_owner_batch_max_wait_us << '\n';
      output << std::setw(width) << "storage exact max attempts: "
             << config.storage_owner_exact_max_attempts << '\n';
      output << std::setw(width) << "storage Stage2 batch max wait us: "
             << config.storage_owner_stage2_batch_max_wait_us << '\n';
      output << std::setw(width) << "storage Stage2 initial delay ms: "
             << config.storage_owner_stage2_initial_delay_ms << '\n';
      output << std::setw(width) << "Stage2 exact score-many: "
             << config.storage_owner_stage2_score_many << '\n';
      output << std::setw(width) << "Stage2 score-many min items: "
             << config.storage_owner_stage2_score_many_min_items << '\n';
      output << std::setw(width) << "Stage2 home RPC combining: "
             << config.storage_owner_stage2_home_rpc_combining << '\n';
      output << std::setw(width) << "Stage2 Split-Plane: "
             << config.storage_owner_stage2_split_plane << '\n';
      output << std::setw(width) << "Stage2 LeaseMerge: "
             << config.storage_owner_lease_merge << '\n';
      output << std::setw(width) << "Stage2 graph issue width: "
             << config.storage_owner_stage2_graph_issue_width << '\n';
      output << std::setw(width) << "storage peer QPs per peer: "
             << config.storage_owner_peer_qps_per_peer << '\n';
      output << std::setw(width) << "storage stage2 maintenance: "
             << config.storage_owner_maintenance_workers << " workers, backlog "
             << config.storage_owner_maintenance_queue_depth << '\n';
      output << std::setw(width) << "storage Stage2 max retry ms: "
             << config.storage_owner_stage2_max_retry_ms << '\n';
      output << std::setw(width) << "Stage2 budget backlog low/high: "
             << config.storage_owner_stage2_budget_backlog_low << "/"
             << config.storage_owner_stage2_budget_backlog_high << '\n';
      output << std::setw(width) << "Stage2 budget tight exp/visits: "
             << config.storage_owner_stage2_budget_max_expansions << "/"
             << config.storage_owner_stage2_budget_max_candidate_visits
             << '\n';
      output << std::setw(width) << "repair deadline ms: "
             << config.storage_owner_repair_deadline_ms << '\n';
      output << std::setw(width) << "maintenance watermark poll ms: "
             << config.storage_owner_flow_control_poll_ms << '\n';
      output << std::setw(width) << "storage reverse conflict limit: "
             << config.storage_owner_reverse_conflict_limit << '\n';
      output << std::setw(width) << "storage reverse retry backoff max us: "
             << config.storage_owner_reverse_retry_backoff_max_us << '\n';
      output << std::setfill('=') << std::setw(line_width) << "" << '\n';
    } else if (config.is_server) {
      output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
      output << std::setw(width) << "storage shard: "
             << config.server_index_file << '\n';
      output << std::setw(width) << "storage id: " << config.storage_id << '\n';
      output << std::setw(width) << "base vectors: " << config.max_vectors << '\n';
      output << std::setw(width) << "vector ID namespace [0,N): "
             << config.vector_id_namespace_size << '\n';
      output << std::setw(width) << "registered memory GiB: "
             << config.mn_memory_gb << '\n';
      output << std::setfill('=') << std::setw(line_width) << "" << '\n';
    }
    return output;
  }
}  // namespace configuration

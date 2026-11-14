#include "memory_node/memory_node.hh"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "vamana/storage_layout_resolver.hh"

MemoryNode* dvstor_active_memory_node = nullptr;

MemoryNode::MemoryNode(Configuration& config)
    : context_(config), cm_(context_, config), num_clients_(config.num_clients),
      storage_id_(config.storage_id),
      num_storage_nodes_(config.storage_peers.empty() ? config.num_server_nodes()
                                                      : static_cast<u32>(config.storage_peers.size())),
      vector_id_namespace_size_(config.vector_id_namespace_size),
      storage_owner_peer_rdma_tokens_(std::max<u32>(1, config.storage_owner_peer_rdma_tokens)),
      graph_extent_enabled_(config.adaptive_dynamic_graph_access_enabled()),
      index_region_(context_),
      peer_rdma_read_outstanding_(num_storage_nodes_),
      mn_memory_bytes_(static_cast<u64>(config.mn_memory_gb) * 1073741824ul) {
  dvstor_active_memory_node = this;
  lib_assert(num_storage_nodes_ > 0 &&
               num_storage_nodes_ <= RemotePtr::MEMORY_NODE_MASK + 1,
             "tagged RemotePtr supports between 1 and 64 storage shards");
  lib_assert(mn_memory_bytes_ > 0 &&
               mn_memory_bytes_ <= RemotePtr::BYTE_OFFSET_CAPACITY,
             "storage region exceeds the 256 GiB tagged RemotePtr capacity");
  for (auto& credit : peer_rdma_read_outstanding_) {
    credit.store(0, std::memory_order_relaxed);
  }
  bool index_initialized = false;
  do {
    cm_.connect_to_clients();

  // Keep the lifecycle thread unpinned. Storage workers are created later and
  // must not inherit a one-CPU affinity mask.

  // receive runtimes parameters from initiator
  configuration::Parameters p{};
  LocalMemoryRegion region{context_, &p, sizeof(configuration::Parameters)};

  cm_.initiator_qp->post_receive(region);
  context_.receive();

  num_compute_threads_ = p.num_threads;
  const u32 gpu_rdma_qps = p.gpu_rdma_qps;
  if (!index_initialized) {
  const filepath_t index_prefix = config.resolved_index_prefix();
  index_prefix_ = index_prefix;
  VectorDType startup_dtype = config.resolved_vector_dtype();
  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
  lib_assert(!index_prefix.empty(), "GPU storage node requires --index-prefix");
  lib_assert(std::filesystem::exists(meta_file),
             "GPU index metadata does not exist: " + meta_file.string());
  {
    service::index_metadata::Metadata metadata;
    str metadata_error;
    lib_assert(service::index_metadata::load_metadata(index_prefix, metadata, &metadata_error), metadata_error);
    lib_assert(metadata.dim == config.dim, "index metadata dim mismatch on storage node");
    lib_assert(metadata.R == config.R, "index metadata R mismatch on storage node");
    lib_assert(metadata.num_vectors == config.max_vectors,
               "index metadata base-vector count mismatch on storage node");
    lib_assert(metadata.num_memory_nodes == num_storage_nodes_, "index metadata storage-node count mismatch");
    gpu_stream_layout_ = metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
      metadata.node_layout == "plain" &&
      metadata.storage_format == "vamana_tagged_v2" &&
      metadata.navigation_quantizer == "opq_pq" &&
      metadata.navigation_format == "opq_pq_graph_v1";
    lib_assert(gpu_stream_layout_,
               "storage node requires a schema-16 tagged OPQ/PQ index");
    lib_assert(storage_id_ < num_storage_nodes_, "invalid GPU storage shard id");
    lib_assert(metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "GPU storage metadata has invalid static shard counts");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid dynamic shard offsets");
    lib_assert(metadata.index_build_fingerprint != 0 &&
                 metadata.shard_build_fingerprints.size() ==
                   num_storage_nodes_ &&
                 metadata.shard_build_fingerprints[storage_id_] != 0,
               "GPU storage metadata has no bound build identity");
    lib_assert(metadata.storage_control_remote_offsets.size() == num_storage_nodes_ &&
                 metadata.dynamic_node_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid control/dynamic-node offsets");
    gpu_static_node_count_ = metadata.hot_graph_entry_counts[storage_id_];
    gpu_static_dynamic_base_ = metadata.hot_graph_dynamic_base_offsets[storage_id_];
    gpu_storage_control_offset_ = metadata.storage_control_remote_offsets[storage_id_];
    gpu_dynamic_node_base_ = metadata.dynamic_node_base_offsets[storage_id_];
    gpu_navigation_code_bytes_ = metadata.navigation_code_bytes;
    lib_assert(gpu_navigation_code_bytes_ > 0,
               "navigation PQ width must be positive");
    gpu_navigation_model_checksum_ = metadata.navigation_model_checksum;
    gpu_index_build_fingerprint_ = metadata.index_build_fingerprint;
    gpu_shard_build_fingerprint_ =
      metadata.shard_build_fingerprints[storage_id_];
    lib_assert(floating_value_is_finite(metadata.partition_cross_shard_ratio) &&
                 metadata.partition_cross_shard_ratio >= 0.0 &&
                 metadata.partition_cross_shard_ratio <= 1.0,
               "index metadata partition_cross_shard_ratio is invalid");
    if (config.vector_data_type != "auto" && config.resolved_vector_dtype() != metadata.vector_dtype) {
      lib_failure("configured vector-data-type=" + config.vector_data_type +
                  " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
    }
    startup_dtype = metadata.vector_dtype;
    config.vector_data_type = vector_dtype_name(startup_dtype);
    VamanaNode::disable_hot_graph();
    VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
    lib_assert(metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
                 metadata.node_layout == "plain",
               "index storage format is obsolete; rebuild with the current offline builder");
    lib_assert(metadata.vector_component_size == VamanaNode::vector_component_size(),
               "index metadata vector component size mismatch on storage node");
    lib_assert(metadata.vector_bytes == VamanaNode::vector_bytes(),
               "index metadata vector byte size mismatch on storage node");
    lib_assert(metadata.node_size == VamanaNode::total_size(), "index metadata node size mismatch on storage node");
    lib_assert(metadata.graph_hot_bytes == VamanaNode::graph_hot_bytes() &&
               metadata.vector_offset == VamanaNode::offset_vector() &&
               metadata.slot_incarnation_offset ==
                 VamanaNode::offset_slot_incarnation() &&
               metadata.remote_ptr_format ==
                 "tagged_inc24_shard6_off34x16_v1",
               "index metadata storage offsets mismatch on storage node");
    lib_assert(metadata.hot_graph_pointer_bytes == vamana::hot_graph::kCompactPointerBytes &&
               metadata.hot_graph_entry_size == VamanaNode::hot_graph_entry_size() &&
               metadata.hot_graph_offsets.size() == num_storage_nodes_ &&
               metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "index hot graph metadata mismatch on storage node");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_ &&
               metadata.dynamic_node_base_offsets.size() == num_storage_nodes_ &&
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
               metadata.hot_graph_dynamic_hot_offset >= VamanaNode::total_size() &&
               metadata.dynamic_navigation_code_offset ==
                 metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
               metadata.dynamic_navigation_code_validation_bytes ==
                 VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES &&
               metadata.dynamic_navigation_code_checksum_bytes ==
                 VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES &&
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.dynamic_navigation_code_offset +
                   metadata.dynamic_navigation_code_validation_bytes +
                   metadata.navigation_code_bytes +
                   metadata.dynamic_navigation_code_checksum_bytes,
               "index dynamic hot graph metadata mismatch on storage node");
    VamanaNode::configure_hot_graph(metadata.hot_graph_offsets,
                                    metadata.hot_graph_entry_counts,
                                    metadata.hot_graph_entry_size,
                                    metadata.hot_graph_shard_bits,
                                    metadata.dynamic_node_base_offsets,
                                    metadata.hot_graph_dynamic_record_bytes,
                                    metadata.hot_graph_dynamic_hot_offset,
                                    metadata.dynamic_navigation_code_offset,
                                    metadata.navigation_code_bytes);
    lib_assert(VamanaNode::HAS_HOT_GRAPH, "failed to enable compact hot graph on storage node");
    lib_assert(gpu_search::pq::read_model(
                 index_path::navigation_model_file(index_prefix,
                                                   metadata.pq_subquantizers),
                 gpu_navigation_model_, &metadata_error),
               metadata_error);
    lib_assert(gpu_navigation_model_.checksum() == metadata.navigation_model_checksum &&
                 gpu_navigation_model_.code_bytes() == metadata.navigation_code_bytes &&
                 gpu_navigation_model_.dim == metadata.dim,
               "storage-node PQ model does not match index metadata");
    lib_assert(metadata.idmap_format == "owner_sharded_v2_bound",
               "index has no build-bound owner idmap v2; rebuild it");
    owner_idmap_required_ = true;
    lib_assert(metadata.centroid_state_format ==
                 "physical_shard_centroid_v2_bound",
               "index has no exact physical-shard centroid state; rebuild it");
    print_status("loaded index metadata from " + index_prefix.string() +
                 " (layout=" + VamanaNode::layout_name() +
                 ", vector_data_type=" + VamanaNode::vector_dtype_name() + ")");
    print_status(
      "storage-owner stage2 candidate width L=" +
      std::to_string(config.resolved_storage_owner_construction_width()) +
      " per shard");
  }
  allocate_memory();

  // free-ptr is initialized to 16 (points to first free address in the buffer)
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;

  lib_assert(!config.server_index_file.empty(),
             "GPU storage node requires --server-index-file");
  const auto [success, message] = load_index_file(config.server_index_file.string());
  lib_assert(success, message);
  if (owner_idmap_required_) {
    lib_assert(load_owner_idmap(index_prefix_), "failed to load owner-sharded idmap");
  }

  initialize_storage_centroid_route();

  print_status("register memory and distribute access token");
  index_region_.register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);
  index_initialized = true;
  }
  MemoryRegionToken token = index_region_.createToken();

  // send access token to all compute nodes
  for (QP& qp : cm_.client_qps) {
    qp->post_send_inlined(std::addressof(token), sizeof(token), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }

  // connect for each compute thread a new QP
  print_status("connect QPs of compute threads");
  vec<u_ptr<DetachedQP>> qps;

  // note: no need for QP sharing on the memory server side
  const u32 qps_per_node = gpu_rdma_qps;
  if (gpu_rdma_qps > 0) {
    print_status("reserving " + std::to_string(gpu_rdma_qps) +
                 " GPU/bootstrap QPs per compute node");
  }
  qps.reserve(num_clients_ * qps_per_node);

  for (QP& client_qp : cm_.client_qps) {
    for (u32 thread_id = 0; thread_id < qps_per_node; ++thread_id) {
      auto& qp = qps.emplace_back(std::make_unique<DetachedQP>(context_));
      qp->connect(context_, context_.get_lid(), client_qp);
    }
  }

  // notify compute nodes that we are ready
  cm_.synchronize();

  const storage_startup::Response startup_response =
    receive_storage_start_signal(config);
  setup_storage_peers(config);
  setup_insert_runtime(config);
  storage_worker_config_ = std::make_unique<Configuration>(config);
  start_peer_reverse_update_runtime(config);
  start_storage_owner_maintenance_runtime(config);
  start_storage_owner_insert_workers(config);
  if (!config.disable_thread_pinning) {
    pin_main_thread(core_assignment_.get_available_core());
  }
  const char* reuse_connections = std::getenv("DVSTOR_MN_REUSE_CONNECTIONS");
  if (reuse_connections != nullptr && reuse_connections[0] == '1') {
    reconnection_thread_ = std::thread([this] {
      cm_.accept_reconnection(storage_insert_shutdown_);
    });
  }
  service_storage_runtime(config, startup_response);

  storage_insert_shutdown_.store(true, std::memory_order_release);
  if (storage_insert_tasks_) storage_insert_tasks_->notify_all();
  // Foreground mutations can be waiting on a same-token authority lease or
  // on peer/send/maintenance capacity.  Wake every such wait before joining
  // the foreground pool; peer_reverse_shutdown_ is intentionally set only
  // after these workers have stopped, so it cannot be the only exit signal.
  for (auto& shard : dynamic_freshness_shards_) {
    shard.changed.notify_all();
  }
  peer_completion_cv_.notify_all();
  notify_storage_owner_maintenance();
  for (auto& worker : storage_insert_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  stop_storage_owner_maintenance_runtime();
  stop_peer_reverse_update_runtime();

  if (reuse_connections == nullptr || reuse_connections[0] != '1') break;
  cm_.get_context().close_server_socket();
  if (reconnection_thread_.joinable()) reconnection_thread_.join();
  print_status("memory node session complete; accepting next client group");
  cm_.reset_for_reconnect();
  reset_storage_session_runtime();
  storage_insert_shutdown_.store(false, std::memory_order_release);
  } while (true);

  dump_final_graph_for_audit();
  const auto extent_stats = graph_extent_cache_.stats();
  print_status(
    "quantized extent reads short/full/fallback/refine/promote/demote/conflict=" +
    std::to_string(extent_stats.short_reads) + "/" +
    std::to_string(extent_stats.full_reads) + "/" +
    std::to_string(extent_stats.fallbacks) + "/" +
    std::to_string(extent_stats.refined_unknown) + "/" +
    std::to_string(extent_stats.promotions) + "/" +
    std::to_string(extent_stats.demotions) + "/" +
    std::to_string(extent_stats.conflicts));
  print_status("memory node shutting down");
  std::cout << timing_ << std::endl;
}

u64 MemoryNode::elapsed_ns_since(const std::chrono::steady_clock::time_point start) {
  return static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void MemoryNode::request_shutdown() {
  storage_insert_shutdown_.store(true, std::memory_order_release);
  if (storage_insert_tasks_) storage_insert_tasks_->notify_all();
  peer_completion_cv_.notify_all();
  notify_storage_owner_maintenance();
}

u64 MemoryNode::scale_ns(const u64 value, const u32 part, const u32 total) {
  if (value == 0 || part == 0 || total == 0) {
    return 0;
  }
  const u64 quotient = value / total;
  const u64 remainder = value % total;
  return quotient * part + (remainder * part) / total;
}

MemoryNode::InsertBreakdownCounters MemoryNode::scale_breakdown(const InsertBreakdownCounters& counters,
                                               const u32 part,
                                               const u32 total) {
  InsertBreakdownCounters out{};
  out.storage_owner_queue_wait_ns = scale_ns(counters.storage_owner_queue_wait_ns, part, total);
  out.storage_owner_stage1_execute_wait_ns =
    scale_ns(counters.storage_owner_stage1_execute_wait_ns, part, total);
  out.storage_owner_search_ns = scale_ns(counters.storage_owner_search_ns, part, total);
  out.storage_owner_prune_ns = scale_ns(counters.storage_owner_prune_ns, part, total);
  out.storage_owner_write_node_ns = scale_ns(counters.storage_owner_write_node_ns, part, total);
  out.storage_owner_local_reverse_ns = scale_ns(counters.storage_owner_local_reverse_ns, part, total);
  out.storage_owner_remote_reverse_ns = scale_ns(counters.storage_owner_remote_reverse_ns, part, total);
  out.storage_owner_peer_reverse_apply_ns =
    scale_ns(counters.storage_owner_peer_reverse_apply_ns, part, total);
  out.storage_owner_response_send_ns = scale_ns(counters.storage_owner_response_send_ns, part, total);
  out.storage_owner_prepare_mutation_ns =
    scale_ns(counters.storage_owner_prepare_mutation_ns, part, total);
  out.storage_owner_allocate_node_ns =
    scale_ns(counters.storage_owner_allocate_node_ns, part, total);
  out.storage_owner_publish_mutation_ns =
    scale_ns(counters.storage_owner_publish_mutation_ns, part, total);
  out.storage_owner_schedule_maintenance_ns =
    scale_ns(counters.storage_owner_schedule_maintenance_ns, part, total);
  out.storage_owner_response_build_ns =
    scale_ns(counters.storage_owner_response_build_ns, part, total);
  out.storage_owner_stage1_arm_wait_ns =
    scale_ns(counters.storage_owner_stage1_arm_wait_ns, part, total);
  out.storage_owner_stage1_release_wait_ns =
    scale_ns(counters.storage_owner_stage1_release_wait_ns, part, total);
  out.storage_owner_cleanup_control_wait_ns =
    scale_ns(counters.storage_owner_cleanup_control_wait_ns, part, total);
  out.storage_owner_search_select_ns = scale_ns(counters.storage_owner_search_select_ns, part, total);
  out.storage_owner_search_neighbor_read_ns =
    scale_ns(counters.storage_owner_search_neighbor_read_ns, part, total);
  out.storage_owner_search_snapshot_read_ns =
    scale_ns(counters.storage_owner_search_snapshot_read_ns, part, total);
  out.storage_owner_search_distance_ns = scale_ns(counters.storage_owner_search_distance_ns, part, total);
  out.storage_owner_search_beam_update_ns =
    scale_ns(counters.storage_owner_search_beam_update_ns, part, total);
  out.storage_owner_search_result_sort_ns =
    scale_ns(counters.storage_owner_search_result_sort_ns, part, total);
  out.storage_owner_prune_snapshot_read_ns =
    scale_ns(counters.storage_owner_prune_snapshot_read_ns, part, total);
  out.storage_owner_prune_distance_ns =
    scale_ns(counters.storage_owner_prune_distance_ns, part, total);
  out.storage_owner_prune_sort_ns = scale_ns(counters.storage_owner_prune_sort_ns, part, total);
  out.storage_owner_prune_pair_distance_ns =
    scale_ns(counters.storage_owner_prune_pair_distance_ns, part, total);
  return out;
}

void MemoryNode::allocate_memory() {
  const auto t_allocate = timing_.create_enroll("allocate_index_buffer");
  std::cerr << "allocation size: " << mn_memory_bytes_ << std::endl;

  t_allocate->start();
  const size_t available_memory = index_buffer_.get_memory_size();
  lib_assert(mn_memory_bytes_ <= available_memory, "allocation failed");

  index_buffer_.allocate(mn_memory_bytes_);
  index_buffer_.touch_memory();
  t_allocate->stop();
}

storage_startup::Response MemoryNode::receive_storage_start_signal(
    const Configuration& config) {
  print_status("waiting for compute-node startup barrier");
  storage_startup::Request request{};
  LocalMemoryRegion region{context_, &request, sizeof(request)};
  cm_.initiator_qp->post_receive(region);
  context_.receive();
  const u32 local_feature_modes = storage_startup::encode_feature_modes(
    config.storage_owner_update_completion_mode,
    config.dynamic_graph_access_mode,
    config.gpu_rdma_search_progression_mode);
  return storage_startup::evaluate_request(
    request, local_feature_modes,
    gpu_search::format::kMetadataSchemaVersion, storage_id_,
    num_storage_nodes_, vector_id_namespace_size_,
    gpu_index_build_fingerprint_, gpu_shard_build_fingerprint_);
}

void MemoryNode::publish_storage_start_response(
    const storage_startup::Response& response) {
  cm_.initiator_qp->post_send_inlined(
    &response, sizeof(response), IBV_WR_SEND);
  context_.poll_send_cq_until_completion();
  lib_assert(response.ready != 0,
             "compute/storage startup contract mismatch: flags=" +
               std::to_string(response.mismatch_flags) +
               " storage_modes=" +
               std::to_string(response.feature_modes) +
               " storage_schema=" +
               std::to_string(response.schema_version) +
               " storage_shard=" + std::to_string(response.shard));
}

std::pair<bool, str> MemoryNode::load_index_file(const str& path) {
  std::ifstream file{path, std::ios::binary};
  if (!file.good()) {
    return {false, "file \"" + path + "\" does not exist"};
  }

  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  const std::streamoff file_end = file.tellg();
  if (file_end < 0 ||
      static_cast<u64>(file_end) >
        static_cast<u64>(std::numeric_limits<size_t>::max()) ||
      static_cast<u64>(file_end) >
        static_cast<u64>(std::numeric_limits<std::streamsize>::max())) {
    return {false, "cannot determine a supported index file size for " + path};
  }
  const size_t file_size = static_cast<size_t>(file_end);
  file.seekg(0, std::ios::beg);

  if (file_size < gpu_search::format::kNodeBaseOffset ||
      file_size > index_buffer_.buffer_size) {
    return {false, "buffer too small for index file"};
  }

  // Reject a truncated shard, a shard from another data set/build, or a
  // malformed layout header before spending minutes reading the full file.
  if (!gpu_stream_layout_ || gpu_static_node_count_ == 0 ||
      gpu_static_dynamic_base_ == 0) {
    return {false, "GPU storage metadata cannot materialize the PQ stream"};
  }
  const u64 node_size = VamanaNode::total_size();
  if (gpu_static_dynamic_base_ < gpu_search::format::kNodeBaseOffset ||
      node_size == 0 ||
      gpu_static_node_count_ >
        (gpu_static_dynamic_base_ - gpu_search::format::kNodeBaseOffset) /
          node_size) {
    return {false, "GPU storage fixed-node range overflows its shard layout"};
  }
  const u64 fixed_nodes_end = gpu_search::format::kNodeBaseOffset +
    gpu_static_node_count_ * node_size;
  if (fixed_nodes_end > gpu_static_dynamic_base_ ||
      gpu_static_dynamic_base_ != file_size) {
    return {false,
            "GPU storage shard size does not exactly match its metadata"};
  }

  u64 persisted_free_pointer = 0;
  u64 persisted_shard_fingerprint = 0;
  file.read(reinterpret_cast<char*>(&persisted_free_pointer),
            sizeof(persisted_free_pointer));
  file.read(reinterpret_cast<char*>(&persisted_shard_fingerprint),
            sizeof(persisted_shard_fingerprint));
  if (file.gcount() !=
        static_cast<std::streamsize>(sizeof(persisted_shard_fingerprint)) ||
      persisted_free_pointer != gpu_static_dynamic_base_ ||
      persisted_shard_fingerprint == 0 ||
      persisted_shard_fingerprint != gpu_shard_build_fingerprint_) {
    return {false,
            "index shard identity does not match the loaded metadata build"};
  }

  const u64 graph_offset =
    VamanaNode::HOT_GRAPH_ENTRY_OFFSETS[storage_id_];
  if (graph_offset < sizeof(vamana::hot_graph::Header) ||
      graph_offset > file_size ||
      sizeof(vamana::hot_graph::Header) >
        file_size - (graph_offset - sizeof(vamana::hot_graph::Header))) {
    return {false, "index shard graph header range is invalid"};
  }
  vamana::hot_graph::Header graph_header;
  file.seekg(static_cast<std::streamoff>(
    graph_offset - sizeof(vamana::hot_graph::Header)));
  file.read(reinterpret_cast<char*>(&graph_header), sizeof(graph_header));
  if (file.gcount() != static_cast<std::streamsize>(sizeof(graph_header)) ||
      graph_header.magic != vamana::hot_graph::kMagic ||
      graph_header.version != vamana::hot_graph::kVersion3 ||
      graph_header.header_bytes != sizeof(graph_header) ||
      graph_header.entry_bytes != VamanaNode::hot_graph_entry_size() ||
      graph_header.max_degree != VamanaNode::R ||
      graph_header.compact_pointer_bytes !=
        vamana::hot_graph::kCompactPointerBytes ||
      graph_header.compact_pointer_shard_bits !=
        VamanaNode::HOT_GRAPH_SHARD_BITS ||
      graph_header.flags != 0 ||
      graph_header.entry_count != gpu_static_node_count_ ||
      graph_header.node_base_offset != gpu_search::format::kNodeBaseOffset ||
      graph_header.reserved0 != gpu_static_dynamic_base_ ||
      // The graph header is committed at schema 15. PQ indexing upgrades the
      // runtime allocation record in metadata without rewriting the large
      // shard, so reserved1 remains the graph-only dynamic record width.
      graph_header.reserved1 != VamanaNode::dynamic_record_size() ||
      graph_header.reserved2 != VamanaNode::total_size()) {
    return {false, "index shard compact-graph header is incompatible"};
  }

  if (gpu_storage_control_offset_ >
        std::numeric_limits<u64>::max() -
          gpu_search::format::kStorageControlBytes ||
      gpu_navigation_code_bytes_ == 0 ||
      gpu_static_node_count_ >
        std::numeric_limits<u64>::max() / gpu_navigation_code_bytes_) {
    return {false, "GPU PQ stream layout overflows"};
  }
  const u64 remote_offset = gpu_storage_control_offset_ +
    gpu_search::format::kStorageControlBytes;
  const u64 payload_bytes =
    gpu_static_node_count_ * gpu_navigation_code_bytes_;
  if (remote_offset == 0 || remote_offset > index_buffer_.buffer_size ||
      payload_bytes > index_buffer_.buffer_size - remote_offset) {
    return {false, "buffer too small for GPU PQ stream"};
  }

  const filepath_t code_path = index_path::navigation_code_for_shard(
    path, gpu_navigation_code_bytes_);
  gpu_search::format::CodeHeader header;
  str error;
  if (!gpu_search::format::read_code_header(code_path, header, &error) ||
      header.memory_node != storage_id_ ||
      header.node_size != VamanaNode::total_size() ||
      header.code_bytes != gpu_navigation_code_bytes_ ||
      header.vector_dtype != static_cast<u32>(VamanaNode::vector_dtype()) ||
      header.model_checksum != gpu_navigation_model_checksum_ ||
      header.build_fingerprint != gpu_index_build_fingerprint_ ||
      header.shard_fingerprint != gpu_shard_build_fingerprint_ ||
      header.shard_fingerprint != persisted_shard_fingerprint ||
      header.entry_count != gpu_static_node_count_ ||
      header.remote_offset != remote_offset ||
      header.payload_bytes != payload_bytes) {
    return {false, error.empty()
      ? "incompatible GPU PQ sidecar " + code_path.string()
      : error};
  }

  file.clear();
  file.seekg(0, std::ios::beg);
  if (!file.good()) {
    return {false, "failed to rewind index shard after preflight: " + path};
  }

  print_status("loading index (" + std::to_string(file_size) + " Bytes) from " + path);
  auto t_read = timing_.create_enroll("read_index_buffer");
  t_read->start();
  file.read(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), file_size);
  t_read->stop();

  if (!file) {
    return {false, "read failed for " + path};
  }
  std::ifstream codes{code_path, std::ios::binary};
  codes.seekg(static_cast<std::streamoff>(sizeof(header)));
  constexpr size_t chunk_bytes = 64ull << 20;
  u64 checksum = gpu_search::format::checksum64_initial();
  for (u64 offset = 0; offset < header.payload_bytes; offset += chunk_bytes) {
    const size_t bytes = static_cast<size_t>(
      std::min<u64>(chunk_bytes, header.payload_bytes - offset));
    byte_t* destination = index_buffer_.get_full_buffer() + header.remote_offset + offset;
    codes.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(codes.gcount()) != bytes) {
      return {false, "short read from " + code_path.string()};
    }
    checksum = gpu_search::format::checksum64_update(checksum, destination, bytes);
  }
  if (checksum != header.payload_checksum) {
    return {false, "GPU PQ code sidecar payload checksum mismatch: " + code_path.string()};
  }
  print_status("loaded GPU PQ codes (" + std::to_string(header.payload_bytes) +
               " Bytes) at remote offset " + std::to_string(header.remote_offset));

  const u64 region_end = remote_offset + payload_bytes;
  if (gpu_storage_control_offset_ !=
        gpu_search::format::align_up(gpu_static_dynamic_base_, 64) ||
      gpu_dynamic_node_base_ < region_end ||
      (gpu_dynamic_node_base_ - gpu_static_dynamic_base_) %
        VamanaNode::allocation_size() != 0 ||
      gpu_dynamic_node_base_ > index_buffer_.buffer_size) {
    return {false, "GPU storage control/dynamic-node layout is inconsistent"};
  }

  // CentroidRouter retains compensated FP64 sums for stable maintenance. The
  // published route is canonical FP32 because GPU queries consume FP32 and
  // update-home selection must make the identical routing decision.
  const auto centroid_scalar =
    gpu_search::format::CentroidScalarType::float32;
  const u64 centroid_publication_bytes =
    gpu_search::format::storage_centroid_route_publication_bytes(
      VamanaNode::DIM, centroid_scalar,
      gpu_search::format::kStorageCentroidRouteMaxLiveEntries);
  if (centroid_publication_bytes == 0 ||
      centroid_publication_bytes > index_buffer_.buffer_size) {
    return {false, "invalid variable-length centroid publication layout"};
  }
  const u64 centroid_publication_offset =
    (index_buffer_.buffer_size - centroid_publication_bytes) & ~u64{63};
  const u64 dynamic_stride =
    (VamanaNode::allocation_size() + alignof(u64) - 1) &
    ~u64{alignof(u64) - 1};
  if (centroid_publication_offset < gpu_dynamic_node_base_ ||
      dynamic_stride > centroid_publication_offset - gpu_dynamic_node_base_) {
    return {false,
            "storage memory has no dynamic-node headroom before the centroid publication"};
  }
  dynamic_allocation_limit_ = centroid_publication_offset;

  std::memset(index_buffer_.get_full_buffer() + gpu_storage_control_offset_, 0,
              gpu_search::format::kStorageControlBytes);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u32 dynamic_navigation_code_payload_bytes =
    VamanaNode::dynamic_navigation_code_payload_bytes();
  lib_assert(dynamic_navigation_code_payload_bytes ==
               gpu_navigation_code_bytes_,
             "dynamic navigation PQ payload width does not match the index model");
  *control = gpu_search::format::StorageControlBlock{
    .shard_id = storage_id_,
    .dynamic_record_bytes = static_cast<u32>(VamanaNode::allocation_size()),
    .dynamic_hot_offset = VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
    .dynamic_code_offset = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET,
    .code_bytes = dynamic_navigation_code_payload_bytes,
    .dynamic_high_watermark = gpu_dynamic_node_base_,
    .centroid_route = {
      .remote_offset = centroid_publication_offset,
      .publication_bytes = centroid_publication_bytes,
      .dim = VamanaNode::DIM,
      .centroid_scalar_type = static_cast<u32>(centroid_scalar),
      .shard_count = num_storage_nodes_,
    },
  };
  str centroid_error;
  const bool initialized_centroid_publication =
    gpu_search::format::prepare_storage_centroid_route_publication(
      span<byte_t>{index_buffer_.get_full_buffer() +
                     centroid_publication_offset,
                   static_cast<size_t>(centroid_publication_bytes)},
      storage_id_, VamanaNode::DIM, centroid_scalar,
      gpu_search::format::kStorageCentroidRouteMaxLiveEntries,
      1, 0, nullptr,
      span<const gpu_search::format::StorageCentroidRouteEntry>{},
      &centroid_error);
  if (!initialized_centroid_publication) {
    return {false, centroid_error};
  }
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = gpu_dynamic_node_base_;

  return {true, ""};
}

size_t MemoryNode::align_up(size_t value, size_t alignment) {
  while (value % alignment != 0) {
    ++value;
  }
  return value;
}

distance_t MemoryNode::distance_to_stored_vector(const span<const element_t> query,
                                                const byte_t* stored,
                                                const Configuration& config) const {
  return typed_l2_distance_float_query(
    query, stored, VamanaNode::vector_dtype(), config.dim);
}

distance_t MemoryNode::distance_between_vectors(const byte_t* lhs,
                                                VectorDType lhs_dtype,
                                                const byte_t* rhs,
                                                VectorDType rhs_dtype,
                                                const Configuration& config) const {
  return typed_l2_distance(lhs, lhs_dtype, rhs, rhs_dtype, config.dim);
}

u64 MemoryNode::load_local_node_header_acquire(RemotePtr rptr) const {
  lib_assert(local_shard(rptr.memory_node()),
             "local header lookup received a remote pointer");
  const auto header = vamana::StorageLayoutResolver::header(rptr);
  lib_assert(header.offset <= mn_memory_bytes_ &&
               sizeof(u64) <= mn_memory_bytes_ - header.offset,
             "local header lookup exceeds shard bounds");
  auto* storage = reinterpret_cast<u64*>(
    const_cast<byte_t*>(index_buffer_.get_full_buffer()) + header.offset);
  return std::atomic_ref<u64>(*storage).load(std::memory_order_acquire);
}

const byte_t* MemoryNode::local_live_vector(RemotePtr rptr) const {
  lib_assert(local_shard(rptr.memory_node()),
             "local vector lookup received a remote pointer");
  const auto vector = vamana::StorageLayoutResolver::vector(rptr);
  lib_assert(vector.offset + vector.size <= mn_memory_bytes_,
             "local vector lookup exceeds shard bounds");
  const u64 header = load_local_node_header_acquire(rptr);
  if ((header & (VamanaNode::HEADER_NODE_LOCK |
                 VamanaNode::HEADER_DELETED)) != 0 ||
      VamanaNode::header_incarnation(header) != rptr.incarnation()) {
    return nullptr;
  }
  const byte_t* node = index_buffer_.get_full_buffer() + rptr.byte_offset();
  if (*reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation()) !=
      rptr.incarnation()) {
    return nullptr;
  }
  return index_buffer_.get_full_buffer() + vector.offset;
}

bool MemoryNode::local_shard(u32 shard_id) const { return shard_id == storage_id_; }

byte_t* MemoryNode::local_node_ptr(const RemotePtr& rptr) {
  return index_buffer_.get_full_buffer() + rptr.byte_offset();
}

const byte_t* MemoryNode::local_node_ptr(const RemotePtr& rptr) const {
  return index_buffer_.get_full_buffer() + rptr.byte_offset();
}

void MemoryNode::insert_into_beam(vec<BeamEntry>& beam, const RemotePtr& rptr, distance_t dist, u32 max_beam_width) {
  auto it = std::lower_bound(
    beam.begin(), beam.end(), dist, [](const BeamEntry& entry, distance_t value) { return entry.distance < value; });
  beam.insert(it, {rptr, dist, false});
  if (beam.size() > max_beam_width) {
    beam.resize(max_beam_width);
  }
}

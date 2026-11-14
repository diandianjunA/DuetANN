#pragma once

#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_maintenance/item_pipeline_policy.hh"
#include "memory_node/storage_owner_maintenance/reconcile_batch_state.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/search_lane_pool.hh"
#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"
#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <algorithm>
#include <iterator>
#include <limits>

namespace memory_node_storage_owner_maintenance_detail {

using namespace memory_node_storage_owner_index_detail;
namespace protocol = service::storage_owner;
using ReverseUpdateOp = service::storage_owner::ReverseUpdateOp;
using PeerRpcType = service::storage_owner::PeerRpcType;
using ReconcileReverseOp = service::storage_owner::ReconcileReverseOp;
using ReconcileReverseOpKind = service::storage_owner::ReconcileReverseOpKind;
using CentroidMembershipOp = service::storage_owner::CentroidMembershipOp;
using CentroidMembershipKind = service::storage_owner::CentroidMembershipKind;

inline bool storage_owner_consolidation_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("DVSTOR_CONSOLIDATE_ON_IDLE");
    return value == nullptr || (value[0] == '1' && value[1] == '\0');
  }();
  return enabled;
}

inline bool storage_owner_pipeline_refinement_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("DVSTOR_PIPELINE_REFINEMENT");
    return value == nullptr || (value[0] == '1' && value[1] == '\0');
  }();
  return storage_owner_consolidation_enabled() && enabled;
}

inline bool storage_owner_reuse_stage1_beam() {
  static const bool enabled = []() {
    const char* value = std::getenv("DVSTOR_REUSE_STAGE1_BEAM");
    return value == nullptr || (value[0] == '1' && value[1] == '\0');
  }();
  return enabled;
}

class Stage2Worker {
public:
  Stage2Worker(MemoryNode& node, u32 worker_id);
  void run();

private:
  using Configuration = MemoryNode::Configuration;
  using StorageOwnerMaintenanceKind = MemoryNode::StorageOwnerMaintenanceKind;
  using StorageOwnerMaintenanceTask = MemoryNode::StorageOwnerMaintenanceTask;
  using NodeSnapshot = MemoryNode::NodeSnapshot;
  using GraphAdjacency = MemoryNode::GraphAdjacency;
  using StorageOwnerThread = MemoryNode::StorageOwnerThread;
  using StorageOwnerReadyContextQueue = MemoryNode::StorageOwnerReadyContextQueue;
  using StorageOwnerMaintenanceWaiterRegistrations =
    MemoryNode::StorageOwnerMaintenanceWaiterRegistrations;
  using StorageOwnerStage2TimingCounters =
    MemoryNode::StorageOwnerStage2TimingCounters;
  using StorageOwnerStage2TimingPhase = MemoryNode::StorageOwnerStage2TimingPhase;
  using PeerRpcSendClass = MemoryNode::PeerRpcSendClass;
  using PeerResponseLease = MemoryNode::PeerResponseLease;
  using TryPeerResponse = MemoryNode::TryPeerResponse;

  MemoryNode& node_;
  const u32 worker_id_;
  StorageOwnerThread& thread =
    *node_.storage_owner_maintenance_worker_states_[worker_id_];
  MemoryNode::StorageOwnerMaintenanceWakeChannel& wake_channel =
    node_.storage_owner_maintenance_wake_channels_[worker_id_];
  StorageOwnerReadyContextQueue* const ready_context_queue =
    node_.storage_owner_maintenance_ready_queue_active_[worker_id_].load(
      std::memory_order_acquire);
  StorageOwnerMaintenanceWaiterRegistrations fallback_waiter_registrations;
  const Configuration& config = *node_.storage_worker_config_;

  struct Stage2Context {
    bool active{};
    Stage2ContextHandle handle{};
    Stage2ContextOwnerKey ready_owner{};
    StorageOwnerMaintenanceKind kind{
      StorageOwnerMaintenanceKind::finalize_insert};
    vec<StorageOwnerMaintenanceTask> tasks;
    vec<NodeSnapshot> targets;
    // Continuation is consumed after the context has crossed a fallible
    // authority/RPC gate.  It therefore belongs to this resumable context,
    // not to worker-wide scratch that another active context may overwrite.
    vec<vec<RemotePtr>> continued_candidates_by_task;
    // Parent liveness is revalidated immediately before reconciliation.  The
    // resulting exact planner input must survive scheduler yields while each
    // item independently crosses its install/removal/placement pipeline.
    vec<vec<RemotePtr>> live_stage2_neighbors_by_task;
    vec<Stage2FinalizeSubphase> finalize_subphases;
    Stage2ReconcileBatchState reconcile_batch;
    vec<size_t> reconcile_items;
    bool item_pipeline_started{};
    vec<vec<ReverseUpdateOp>> remote_ops_by_peer;
    vec<u64> reverse_request_ids;
    // Timestamped only after all asynchronous reconciliation barriers and the
    // synchronous placement/membership suffix have completed. The outer state
    // tracker has no reverse ACK mask at this point; node_ measures only the
    // state-machine/context handoff into finalization. Remote reconciliation
    // latency is charged to the context-spanning reverse_prepare timer.
    u64 completion_handoff_started_ns{};
    // Logical search state belongs to the resumable context. Registered RDMA
    // scratch belongs to search_lane only while a posted one-sided operation
    // can still touch it. This lets a home-RPC wait release and later rebind a
    // lane without restarting the continuation.
    Stage2SearchIoState search_io;
    std::optional<u32> search_lane;
    bool search_lane_wait_registered{};
    // Exact context-owned subscriptions for SEND/RDMA/lane capacity. Release
    // consumes only a worker runnable bit; success/reset removes node_
    // context's references without hiding sibling contexts on the same worker.
    StorageOwnerMaintenanceWaiterRegistrations waiter_registrations;
    bool search_input_prepared{};
    bool search_timing_recorded{};
    bool search_stall_diagnostic_emitted{};
    u64 search_started_ns{};
    u64 reverse_prepare_started_ns{};
    bool reverse_prepare_timing_active{};
  };

  // One timer update represents a whole context phase attempt.  In
  // particular, no timing atomic appears in a candidate, edge, or RDMA-read
  // loop.  The destructor records failed/deferred attempts as well, which is
  // essential when diagnosing a phase that never reaches finalization.
  struct Stage2PhaseAttemptTimer {
    StorageOwnerStage2TimingCounters* counters{};
    u64 started_ns{};
    u64 task_attempts{};

    Stage2PhaseAttemptTimer(StorageOwnerStage2TimingCounters& value,
                            size_t tasks)
        : counters(&value),
          started_ns(steady_now_ns()),
          task_attempts(static_cast<u64>(tasks)) {}

    Stage2PhaseAttemptTimer(const Stage2PhaseAttemptTimer&) = delete;
    Stage2PhaseAttemptTimer& operator=(
      const Stage2PhaseAttemptTimer&) = delete;

    ~Stage2PhaseAttemptTimer() { finish(); }

    void transition(StorageOwnerStage2TimingCounters& next,
                    size_t tasks) {
      finish();
      counters = &next;
      started_ns = steady_now_ns();
      task_attempts = static_cast<u64>(tasks);
    }

    void finish() {
      if (counters == nullptr) return;
      const u64 elapsed_ns = steady_now_ns() - started_ns;
      counters->attempts.fetch_add(1, std::memory_order_relaxed);
      counters->task_attempts.fetch_add(
        task_attempts, std::memory_order_relaxed);
      counters->elapsed_ns.fetch_add(elapsed_ns,
                                     std::memory_order_relaxed);
      counters = nullptr;
    }
  };

  // A worker owns its contexts and both trackers, so the response path needs
  // no lock beyond the bounded peer-response registry. Global admission below
  // limits all workers together to the dedicated peer RPC depth.
  const size_t context_capacity =
    std::max<size_t>(1, config.storage_owner_rpc_depth);
  const size_t remote_peer_count = node_.num_storage_nodes_ - 1;
  const u32 construction_width =
    config.resolved_storage_owner_construction_width();
  const size_t request_capacity =
    context_capacity * std::max<size_t>(1, remote_peer_count);
  Stage2StateTracker states{context_capacity, node_.num_storage_nodes_};
  Stage2RequestTracker requests{request_capacity};
  vec<Stage2Context> contexts{context_capacity};
  Stage2SearchLanePool search_lanes{thread.post_balances.size()};
  // Stage2 resumes one global beam; it does not collect an independent L-set
  // from every shard. Its memory footprint is therefore O(batch * L), not
  // O(batch * shard_count * L).
  const size_t candidate_capacity_per_item = construction_width;
  const size_t reconcile_op_capacity =
    (2 * static_cast<size_t>(config.R) + construction_width + 1) *
      config.storage_owner_batch_max;
  const size_t reconcile_payload_bytes = node_.peer_rpc_runtime_.message_bytes -
    sizeof(service::storage_owner::PeerRpcHeader);
  const size_t reconcile_wire_capacity = std::max<size_t>(
    1, reconcile_payload_bytes /
         sizeof(service::storage_owner::ReconcileReverseOp));
  const size_t reconcile_chunk_capacity = node_.num_storage_nodes_ +
    (reconcile_op_capacity + reconcile_wire_capacity - 1) /
      reconcile_wire_capacity;
  // Continuation pointers and all later snapshot waves are consumed
  // synchronously by this worker. Reusing their O(batch*L) capacity here avoids
  // multiplying full-vector scratch by every in-flight state-machine context.
  const size_t continuation_capacity =
    static_cast<size_t>(config.storage_owner_batch_max) *
    candidate_capacity_per_item;
  // Only the authoritative source-freeze boundary materializes continuation
  // vectors. Shared physical candidates are read once across the whole batch;
  // ordered pointer references are then scattered into per-task prune views.
  vec<vec<RemotePtr>> snapshot_candidates_by_task =
    vec<vec<RemotePtr>>(config.storage_owner_batch_max);
  vec<vec<const NodeSnapshot*>> snapshots_by_task =
    vec<vec<const NodeSnapshot*>>(config.storage_owner_batch_max);
  vec<bool> snapshot_task_active = vec<bool>(config.storage_owner_batch_max);
  Stage2SnapshotWavePlan snapshot_plan;
  vec<NodeSnapshot> snapshot_storage;
  vec<StableNodeSnapshotState> snapshot_states;
  dense_hashmap_t<u64, const NodeSnapshot*> snapshot_by_raw;
  vec<vec<RemotePtr>> live_stage2_neighbors_by_task =
    vec<vec<RemotePtr>>(config.storage_owner_batch_max);
  vec<std::pair<RemotePtr, u64>> identity_storage;
  vec<StableNodeSnapshotState> identity_states;
  hashset_t<RemotePtr> stable_identity_targets;
  const u32 reverse_wire_max_ops =
    node_.storage_owner_reverse_outbox_->wire_max_ops();
  vec<ReverseUpdateOp> reverse_wire_ops =
    vec<ReverseUpdateOp>(reverse_wire_max_ops);
  vec<Stage2ReverseCompletion> reverse_completion_scratch =
    vec<Stage2ReverseCompletion>(reverse_wire_max_ops);
  vec<u8> reverse_completion_worker_marked =
    vec<u8>(node_.storage_owner_reverse_completions_.size(), u8{0});
  vec<u32> reverse_completion_wake_owners;
  vec<byte_t> reverse_response_payload;
  vec<Stage2ReadyContextEvent> overflow_ready_events;

  const u64 rpc_timeout_ns =
    static_cast<u64>(config.storage_owner_rpc_timeout_ms) * 1000ull * 1000ull;
  const u64 retry_backoff_ns = std::min<u64>(
    rpc_timeout_ns, 1000ull * 1000ull);
  const u64 reverse_retry_backoff_max_ns = static_cast<u64>(
    config.storage_owner_reverse_retry_backoff_max_us) * 1000ull;

  size_t active_context_cursor{};
  u64 last_protocol_stall_log_ns{};
  u32 debug_parent_events_{};
  bool fallback_context_scan_requested{};
  bool refinement_lane_wait_registered{};
  u32 refinement_retry_attempts_{};
  std::chrono::steady_clock::time_point refinement_retry_not_before{};
  std::chrono::steady_clock::time_point next_fallback_context_scan =
    std::chrono::steady_clock::now();
  size_t unpublished_context_slots_scanned{};
  u64 unpublished_idle_wait_ns{};
  size_t unpublished_idle_waits{};
  static constexpr auto kFallbackContextAuditInterval =
    std::chrono::milliseconds(10);

  enum class CleanupParentQuiesceResult : u8 {
    ready,
    busy,
    stale,
  };

  enum class ReconcilePollResult : u8 {
    waiting,
    complete,
    retry,
  };

  u64 reverse_retry_backoff(u32 attempts);
  void release_context_lane(Stage2Context& context);
  bool bind_context_lane(Stage2Context& context);
  bool release_rebindable_context_lane(Stage2Context& context);
  void reset_context(Stage2Context& context);
  bool consolidate_finalized_insertions();
  void materialize_stable_snapshot_wave(size_t task_count);
  void materialize_stable_identity_wave(size_t task_count);
  void record_finalized_live(std::chrono::steady_clock::time_point queued_at);
  ReconcileReverseOp make_reconcile_op(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr graph_target,
      RemotePtr old_candidate,
      RemotePtr new_candidate,
      ReconcileReverseOpKind kind);
  CentroidMembershipOp make_centroid_membership_op(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr pointer,
      CentroidMembershipKind kind);
  bool apply_reconcile_ops(span<const ReconcileReverseOp> ops,
                           const Configuration* prune_config = nullptr,
                           vec<u8>* retry_mask = nullptr);
  CleanupParentQuiesceResult quiesce_cleanup_parent(
      StorageOwnerMaintenanceTask& task);
  bool reparent_cleanup_children(StorageOwnerMaintenanceTask& task);
  bool append_stage2_reconcile_ops(
      StorageOwnerMaintenanceTask& task,
      span<const RemotePtr> live_final_neighbors,
      vec<ReconcileReverseOp>& promotion_ops,
      vec<ReconcileReverseOp>& stable_ops,
      vec<ReconcileReverseOp>& removal_ops);
  bool publish_stage2_recovery_parent(
      StorageOwnerMaintenanceTask& task, RemotePtr recovery_parent,
      vec<RemotePtr>* live_neighbors = nullptr);
  bool remove_candidate_backlinks(
      const StorageOwnerMaintenanceTask& task,
      RemotePtr candidate,
      span<const RemotePtr> targets);
  bool settle_dynamic_allocation(const StorageOwnerMaintenanceTask& task);
  bool complete_stale_stage2(StorageOwnerMaintenanceTask& task,
                             const char** failure_reason = nullptr);
  bool complete_stale_cleanup(const StorageOwnerMaintenanceTask& task);
  PeerRpcType reverse_request_type(const Stage2Context& context);
  Stage2ReverseEnqueueResult enqueue_reverse_dispatch(
      Stage2Context& context, u32 shard, u64 ready_at_ns);
  void register_reverse_requests(Stage2Context& context, u64 expected_mask);
  PeerRpcType aggregate_response_type(PeerRpcType request_type);
  bool poll_owned_reverse_aggregates();
  bool form_reverse_aggregates();
  bool post_owned_reverse_aggregates();
  void log_reverse_send_snapshot(u32 peer, u64 request_id, u32 item_count);
  bool drive_reverse_outbox();
  bool drain_reverse_completions();
  Stage2SearchAdvanceResult prepare_local(Stage2Context& context);
  void terminate_stuck_stage2(Stage2Context& context);
  void defer_stage2_retry(Stage2Context& context, bool permanent = false);
  void cancel_context_reconcile(Stage2Context& context);
  void finish_reverse_prepare_timing(Stage2Context& context);
  bool build_reconcile_barrier_ops(
      Stage2Context& context, size_t item, Stage2ReconcileBarrier barrier,
      vec<ReconcileReverseOp>& selected_ops);
  bool begin_reconcile_barrier(
      Stage2Context& context, Stage2ReconcileBarrier barrier,
      span<const ReconcileReverseOp> ops);
  ReconcilePollResult poll_reconcile_barrier(Stage2Context& context);
  void defer_stage2_item_retry(
      Stage2Context& context, size_t item, bool permanent = false);
  bool finish_stage2_after_reconcile(Stage2Context& context, size_t item);
  void reset_reconcile_for_semantic_retry(Stage2Context& context);
  bool start_reconcile_barrier(Stage2Context& context);
  bool advance_reconcile_pipeline(Stage2Context& context);
  bool prepare_stage2_reverse(Stage2Context& context);
  bool prepare_cleanup_reverse(Stage2Context& context);
  bool finalize_context(Stage2Context& context);
  void defer_cleanup_context(Stage2Context& context);
  bool drive_context(Stage2Context& context);
  bool drive_owned_context(Stage2Context& context);
  bool drive_ready_event(const Stage2ReadyContextEvent& event);
  std::pair<bool, size_t> drain_ready_contexts();
  Stage2Context* try_admit_context();
  void flush_idle_timing();

};

}  // namespace memory_node_storage_owner_maintenance_detail

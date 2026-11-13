#pragma once

#include "common/configuration.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_index_detail {

// The backlog-aware budget reads the accepted-but-unfinished Stage2 sequence
// count (a lock-free approximation) once per continuation batch. Below the
// low watermark the policy returns the historical unbounded budget.
inline PartitionSearchBudget stage2_search_budget_for_backlog(
    const configuration::IndexConfiguration& config,
    u64 backlog,
    u32 construction_width) {
  return stage2_partition_search_budget(
    construction_width, VamanaNode::graph_entry_capacity(), backlog,
    Stage2BudgetBacklogPolicy{
      .backlog_low = config.storage_owner_stage2_budget_backlog_low,
      .backlog_high = config.storage_owner_stage2_budget_backlog_high,
      .max_expansions = config.storage_owner_stage2_budget_max_expansions,
      .max_candidate_visits =
        config.storage_owner_stage2_budget_max_candidate_visits,
    });
}

}  // namespace memory_node_storage_owner_index_detail

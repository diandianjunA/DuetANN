#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>

#include "memory_node/storage_owner_maintenance/ordered_commit_policy.hh"

namespace {

using memory_node_storage_owner_maintenance_detail::CommitDependencyStatus;
using memory_node_storage_owner_maintenance_detail::GlobalCommitTicket;
using memory_node_storage_owner_maintenance_detail::NodeDependencyFingerprint;
using memory_node_storage_owner_maintenance_detail::OrderedCommitAction;
using memory_node_storage_owner_maintenance_detail::OrderedCommitGate;
using memory_node_storage_owner_maintenance_detail::SearchDependencySet;

struct ToyNode {
  u32 id{};
  std::array<float, 2> vector{};
  vec<u32> neighbors;

  bool operator==(const ToyNode&) const = default;
};

using ToyGraph = std::map<u32, ToyNode>;

struct ToyUpdate {
  ToyNode node;
  vec<u32> parents;
  SearchDependencySet dependencies;
};

float distance(const ToyNode& lhs, const ToyNode& rhs) {
  const float x = lhs.vector[0] - rhs.vector[0];
  const float y = lhs.vector[1] - rhs.vector[1];
  return std::sqrt(x * x + y * y);
}

NodeDependencyFingerprint fingerprint(const ToyNode& node) {
  u64 adjacency = memory_node_storage_owner_maintenance_detail::
    dependency_hash_seed;
  for (const u32 neighbor : node.neighbors) {
    adjacency = memory_node_storage_owner_maintenance_detail::
      dependency_hash_mix(adjacency, neighbor);
  }
  u64 vector = memory_node_storage_owner_maintenance_detail::
    dependency_hash_seed;
  for (const float component : node.vector) {
    vector = memory_node_storage_owner_maintenance_detail::
      dependency_hash_mix(vector, static_cast<u32>(component));
  }
  return NodeDependencyFingerprint{
    RemotePtr{node.id, static_cast<u64>(node.id) * 16},
    adjacency, vector, 0};
}

SearchDependencySet dependencies(const ToyGraph& graph) {
  SearchDependencySet result;
  for (const auto& [id, node] : graph) {
    result.expanded.push_back(fingerprint(node));
    result.scored.push_back(fingerprint(node));
  }
  result.finalize();
  return result;
}

ToyUpdate prepare(const ToyGraph& graph, const ToyNode& node) {
  ToyUpdate update{
    .node = node, .parents = {}, .dependencies = dependencies(graph)};
  vec<const ToyNode*> candidates;
  for (const auto& [id, candidate] : graph) candidates.push_back(&candidate);
  std::sort(candidates.begin(), candidates.end(),
            [&](const ToyNode* lhs, const ToyNode* rhs) {
    if (distance(node, *lhs) != distance(node, *rhs)) {
      return distance(node, *lhs) < distance(node, *rhs);
    }
    return lhs->id < rhs->id;
  });
  for (const ToyNode* candidate : candidates) {
    update.parents.push_back(candidate->id);
    if (update.parents.size() == 2) break;
  }
  return update;
}

void apply_update(ToyGraph& graph, const ToyUpdate& update) {
  graph[update.node.id] = update.node;
  graph[update.node.id].neighbors = update.parents;
  for (const u32 parent_id : update.parents) {
    ToyNode& parent = graph.at(parent_id);
    parent.neighbors.push_back(update.node.id);
    std::sort(parent.neighbors.begin(), parent.neighbors.end(),
              [&](u32 lhs, u32 rhs) {
      if (distance(parent, graph.at(lhs)) != distance(parent, graph.at(rhs))) {
        return distance(parent, graph.at(lhs)) <
          distance(parent, graph.at(rhs));
      }
      return lhs < rhs;
    });
    if (parent.neighbors.size() > 2) parent.neighbors.resize(2);
  }
}

ToyGraph initial_graph() {
  ToyGraph graph;
  graph.emplace(1, ToyNode{1, {0.0F, 0.0F}, {2}});
  graph.emplace(2, ToyNode{2, {10.0F, 10.0F}, {1}});
  return graph;
}

void test_noncommuting_updates_match_sequential_reference() {
  const ToyNode a{3, {1.0F, 1.0F}, {}};
  const ToyNode b{4, {2.0F, 2.0F}, {}};

  ToyGraph direct_ab = initial_graph();
  apply_update(direct_ab, prepare(direct_ab, a));
  apply_update(direct_ab, prepare(direct_ab, b));

  ToyGraph direct_ba = initial_graph();
  apply_update(direct_ba, prepare(direct_ba, b));
  apply_update(direct_ba, prepare(direct_ba, a));
  assert(direct_ab != direct_ba);

  const ToyGraph initial = initial_graph();
  const ToyUpdate prepared_a = prepare(initial, a);
  const ToyUpdate prepared_b = prepare(initial, b);
  OrderedCommitGate gate(3, 7);

  assert(gate.evaluate({3, 8}, CommitDependencyStatus::unchanged) ==
         OrderedCommitAction::wait);
  assert(gate.next_sequence() == 7);

  ToyGraph ordered = initial;
  assert(gate.evaluate({3, 7}, CommitDependencyStatus::unchanged) ==
         OrderedCommitAction::reuse);
  apply_update(ordered, prepared_a);
  gate.complete({3, 7});

  assert(gate.evaluate({3, 8}, CommitDependencyStatus::changed) ==
         OrderedCommitAction::recompute);
  const ToyUpdate rebased_b = prepare(ordered, b);
  apply_update(ordered, rebased_b);
  gate.complete({3, 8});
  assert(ordered == direct_ab);

  bool duplicate_rejected = false;
  try {
    gate.complete({3, 7});
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  assert(duplicate_rejected);
  assert(gate.next_sequence() == 9);
}

void test_unavailable_dependencies_wait_without_advancing() {
  OrderedCommitGate gate(2, 11);
  assert(gate.evaluate({2, 11}, CommitDependencyStatus::unavailable) ==
         OrderedCommitAction::wait);
  assert(gate.next_sequence() == 11);
  assert(gate.evaluate({3, 11}, CommitDependencyStatus::unchanged) ==
         OrderedCommitAction::wait);
  assert(gate.next_sequence() == 11);
}

void test_ticket_batches_are_contiguous() {
  memory_node_storage_owner_maintenance_detail::GlobalCommitTicketAllocator
    allocator(5, 41);
  const auto tickets = allocator.allocate_batch(3);
  assert(tickets.size() == 3);
  assert(tickets[0].epoch == 5 && tickets[0].sequence == 41);
  assert(tickets[1].sequence == 42);
  assert(tickets[2].sequence == 43);
  assert(allocator.next_sequence() == 44);
  assert(allocator.allocate_batch(0).empty());
}

void test_dependency_normalization_is_deterministic() {
  const ToyNode node{7, {1.0F, 2.0F}, {9, 8}};
  SearchDependencySet lhs;
  lhs.expanded = {fingerprint(node), fingerprint(node)};
  lhs.scored = {fingerprint(node)};
  lhs.finalize();
  assert(lhs.expanded.size() == 1);
  assert(lhs.scored.size() == 1);

  ToyNode changed = node;
  changed.neighbors = {8, 9};
  assert(fingerprint(node) != fingerprint(changed));
  changed = node;
  changed.vector = {2.0F, 1.0F};
  assert(fingerprint(node) != fingerprint(changed));
}

}  // namespace

int main() {
  test_noncommuting_updates_match_sequential_reference();
  test_unavailable_dependencies_wait_without_advancing();
  test_ticket_batches_are_contiguous();
  test_dependency_normalization_is_deterministic();
  return 0;
}

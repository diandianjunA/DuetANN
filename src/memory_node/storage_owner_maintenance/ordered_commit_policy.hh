#pragma once

#include <algorithm>
#include <stdexcept>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_maintenance_detail {

struct GlobalCommitTicket {
  u64 epoch{};
  u64 sequence{};
};

enum class CommitDependencyStatus {
  unchanged,
  changed,
  unavailable,
};

enum class OrderedCommitAction {
  reuse,
  recompute,
  wait,
};

inline OrderedCommitAction ordered_commit_action(
    GlobalCommitTicket ticket,
    u64 expected_epoch,
    u64 expected_sequence,
    CommitDependencyStatus dependency_status) {
  if (ticket.epoch != expected_epoch || ticket.sequence != expected_sequence ||
      dependency_status == CommitDependencyStatus::unavailable) {
    return OrderedCommitAction::wait;
  }
  return dependency_status == CommitDependencyStatus::changed
    ? OrderedCommitAction::recompute
    : OrderedCommitAction::reuse;
}

class OrderedCommitGate {
public:
  explicit OrderedCommitGate(u64 epoch, u64 first_sequence = 1)
      : epoch_(epoch), next_sequence_(first_sequence) {}

  OrderedCommitAction evaluate(
      GlobalCommitTicket ticket,
      CommitDependencyStatus dependency_status) const {
    return ordered_commit_action(
      ticket, epoch_, next_sequence_, dependency_status);
  }

  void complete(GlobalCommitTicket ticket) {
    if (ticket.epoch != epoch_ || ticket.sequence != next_sequence_) {
      throw std::logic_error("ordered commit completed out of sequence");
    }
    ++next_sequence_;
  }

  u64 epoch() const { return epoch_; }
  u64 next_sequence() const { return next_sequence_; }

private:
  u64 epoch_;
  u64 next_sequence_;
};

class GlobalCommitTicketAllocator {
public:
  explicit GlobalCommitTicketAllocator(u64 epoch, u64 first_sequence = 1)
      : epoch_(epoch), next_sequence_(first_sequence) {}

  GlobalCommitTicket allocate() {
    if (next_sequence_ == 0) {
      throw std::overflow_error("global commit sequence exhausted");
    }
    return GlobalCommitTicket{epoch_, next_sequence_++};
  }

  vec<GlobalCommitTicket> allocate_batch(size_t count) {
    vec<GlobalCommitTicket> tickets;
    tickets.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      tickets.push_back(allocate());
    }
    return tickets;
  }

  u64 epoch() const { return epoch_; }
  u64 next_sequence() const { return next_sequence_; }

private:
  u64 epoch_;
  u64 next_sequence_;
};

inline constexpr u64 dependency_hash_mix(u64 hash, u64 value) {
  hash ^= value;
  hash *= 1099511628211ULL;
  return hash;
}

inline constexpr u64 dependency_hash_seed = 14695981039346656037ULL;

inline u64 dependency_hash_bytes(span<const byte_t> bytes) {
  u64 hash = dependency_hash_seed;
  for (const byte_t value : bytes) hash = dependency_hash_mix(hash, value);
  return hash;
}

struct NodeDependencyFingerprint {
  RemotePtr pointer;
  u64 adjacency_hash{};
  u64 vector_hash{};
  u64 lifecycle_state{};

  bool operator==(const NodeDependencyFingerprint&) const = default;
};

struct SearchDependencySet {
  vec<NodeDependencyFingerprint> expanded;
  vec<NodeDependencyFingerprint> scored;

  void finalize() {
    const auto normalize = [](vec<NodeDependencyFingerprint>& values) {
      std::sort(values.begin(), values.end(),
                [](const auto& lhs, const auto& rhs) {
        return lhs.pointer.raw_address < rhs.pointer.raw_address;
      });
      values.erase(std::unique(values.begin(), values.end()), values.end());
    };
    normalize(expanded);
    normalize(scored);
  }

  bool operator==(const SearchDependencySet&) const = default;
};

}  // namespace memory_node_storage_owner_maintenance_detail

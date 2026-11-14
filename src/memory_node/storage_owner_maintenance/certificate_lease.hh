#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/types.hh"

namespace memory_node_storage_owner_maintenance_detail {

enum class CertificateAcquireResult : std::uint8_t {
  acquired,
  replay,
  capacity_exhausted,
  conflict,
};

// Target-local ownership of the stable incoming edges that must survive until
// every temporary bridge for the same child has been removed. Independent
// cleanup sequences can protect the same child concurrently; releasing one
// sequence must not remove another sequence's protection. The table only
// stores protocol state; graph adjacency remains authoritative for the edge.
class CertificateLeaseTable {
 public:
  explicit CertificateLeaseTable(std::size_t degree_limit)
      : degree_limit_(degree_limit) {}

  CertificateAcquireResult acquire(std::uint64_t target_raw,
                                   std::uint64_t child_raw,
                                   std::uint64_t sequence) {
    if (target_raw == 0 || child_raw == 0 || sequence == 0) {
      return CertificateAcquireResult::conflict;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (sequence <= finalized_sequence_) {
      return CertificateAcquireResult::replay;
    }
    const auto target_released = released_.find(target_raw);
    if (target_released != released_.end()) {
      const auto released = target_released->second.find(child_raw);
      if (released != target_released->second.end() &&
          released->second >= sequence) {
        return CertificateAcquireResult::replay;
      }
    }
    auto& target = leases_[target_raw];
    const auto found = target.find(child_raw);
    if (found != target.end()) {
      return found->second.insert(sequence).second
        ? CertificateAcquireResult::acquired
        : CertificateAcquireResult::replay;
    }
    if (target.size() >= degree_limit_) {
      return CertificateAcquireResult::capacity_exhausted;
    }
    target.emplace(child_raw, SequenceSet{sequence});
    ++size_;
    return CertificateAcquireResult::acquired;
  }

  bool release(std::uint64_t target_raw, std::uint64_t child_raw,
               std::uint64_t sequence) {
    return commit(target_raw, {}, {{child_raw, sequence}});
  }

  bool commit(
      std::uint64_t target_raw,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& acquired,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>>& released) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = leases_.find(target_raw);
    std::unordered_map<std::uint64_t, SequenceSet> updated =
      existing == leases_.end()
        ? std::unordered_map<std::uint64_t, SequenceSet>{}
        : existing->second;
    const auto existing_released = released_.find(target_raw);
    std::unordered_map<std::uint64_t, std::uint64_t> completed =
      existing_released == released_.end()
        ? std::unordered_map<std::uint64_t, std::uint64_t>{}
        : existing_released->second;
    for (const auto& [child, sequence] : released) {
      const auto found = updated.find(child);
      if (found != updated.end()) {
        if (sequence == 0) {
          updated.erase(found);
        } else {
          const auto active = found->second.find(sequence);
          if (active == found->second.end()) continue;
          found->second.erase(active);
          if (found->second.empty()) updated.erase(found);
        }
      }
      const auto [position, inserted] = completed.emplace(child, sequence);
      if (!inserted) position->second = std::max(position->second, sequence);
    }
    for (const auto& [child, sequence] : acquired) {
      if (child == 0 || sequence == 0) return false;
      if (sequence <= finalized_sequence_) continue;
      const auto released = completed.find(child);
      if (released != completed.end() && released->second >= sequence) {
        continue;
      }
      const auto found = updated.find(child);
      if (found == updated.end() && updated.size() >= degree_limit_) {
        return false;
      }
      updated[child].insert(sequence);
    }
    if (std::getenv("DVSTOR_STAGE2_STALL_DEBUG") != nullptr) {
      const auto trace = [&](const char* action, std::uint64_t child,
                             std::uint64_t sequence) {
        const auto found = updated.find(child);
        std::string message = std::string("[lease-lifecycle] action=") +
          action + " target=" + std::to_string(target_raw) +
          " child=" + std::to_string(child) +
          " sequence=" + std::to_string(sequence) + " remaining=";
        message += found == updated.end()
          ? std::to_string(-1)
          : std::to_string(*std::max_element(
              found->second.begin(), found->second.end()));
        std::cerr << message << std::endl;
      };
      for (const auto& [child, sequence] : acquired) {
        trace("acquire", child, sequence);
      }
      for (const auto& [child, sequence] : released) {
        trace("release", child, sequence);
      }
    }
    const std::size_t previous =
      existing == leases_.end() ? 0 : existing->second.size();
    const std::size_t next = updated.size();
    if (updated.empty()) {
      if (existing != leases_.end()) leases_.erase(existing);
    } else {
      leases_[target_raw] = std::move(updated);
    }
    if (completed.empty()) {
      released_.erase(target_raw);
    } else {
      released_[target_raw] = std::move(completed);
    }
    size_ = size_ - previous + next;
    return true;
  }

  std::size_t release_target(std::uint64_t target_raw) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    const std::size_t released = target == leases_.end()
      ? 0 : target->second.size();
    size_ -= released;
    if (target != leases_.end()) leases_.erase(target);
    return released;
  }

  void discard_released_through(std::uint64_t finalized_sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_sequence <= finalized_sequence_) return;
    finalized_sequence_ = finalized_sequence;
    for (auto target = released_.begin(); target != released_.end();) {
      for (auto child = target->second.begin();
           child != target->second.end();) {
        if (child->second <= finalized_sequence_) {
          child = target->second.erase(child);
        } else {
          ++child;
        }
      }
      if (target->second.empty()) {
        target = released_.erase(target);
      } else {
        ++target;
      }
    }
  }

  bool contains(std::uint64_t target_raw, std::uint64_t child_raw,
                std::uint64_t sequence = 0) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    if (target == leases_.end()) return false;
    const auto found = target->second.find(child_raw);
    return found != target->second.end() &&
      (sequence == 0 || found->second.contains(sequence));
  }

  std::size_t active_at(std::uint64_t target_raw) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto target = leases_.find(target_raw);
    return target == leases_.end() ? 0 : target->second.size();
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> leases_at(
      std::uint64_t target_raw) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    const auto target = leases_.find(target_raw);
    if (target == leases_.end()) return result;
    result.reserve(target->second.size());
    for (const auto& [child, sequences] : target->second) {
      result.emplace_back(child, *std::max_element(
        sequences.begin(), sequences.end()));
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> result;
    result.reserve(size_);
    for (const auto& [target, children] : leases_) {
      for (const auto& [child, sequences] : children) {
        for (const auto sequence : sequences) {
          result.emplace_back(target, child, sequence);
        }
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    leases_.clear();
    released_.clear();
    finalized_sequence_ = 0;
    size_ = 0;
  }

 private:
  using SequenceSet = std::unordered_set<std::uint64_t>;

  std::size_t degree_limit_;
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t,
                     std::unordered_map<std::uint64_t, SequenceSet>> leases_;
  std::unordered_map<std::uint64_t,
                     std::unordered_map<std::uint64_t, std::uint64_t>> released_;
  std::uint64_t finalized_sequence_{};
  std::size_t size_{};
};

}  // namespace memory_node_storage_owner_maintenance_detail

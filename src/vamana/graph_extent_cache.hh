#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "remote_pointer.hh"
#include "vamana/graph_extent_policy.hh"

namespace vamana::graph_extent {

struct CacheStats {
  std::uint64_t short_reads{};
  std::uint64_t full_reads{};
  std::uint64_t fallbacks{};
  std::uint64_t refined_unknown{};
  std::uint64_t promotions{};
  std::uint64_t demotions{};
  std::uint64_t conflicts{};
};

class Cache {
public:
  static constexpr std::size_t kDefaultCapacity = std::size_t{1} << 20;

  explicit Cache(std::size_t capacity = kDefaultCapacity)
      : entries_(capacity), capacity_mask_(capacity - 1) {
    if (capacity == 0 || (capacity & capacity_mask_) != 0) {
      throw std::invalid_argument("graph extent cache capacity must be a power of two");
    }
  }

  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  std::uint8_t lookup(RemotePtr handle) const {
    if (handle.is_null() || !handle.is_well_formed()) return kUnknownLevel;
    const Entry& entry = entries_[slot(handle)];
    const std::uint64_t observed_handle =
      entry.handle.load(std::memory_order_acquire);
    if (observed_handle != handle.raw_address) return kUnknownLevel;
    const std::uint32_t observed_state =
      entry.state.load(std::memory_order_acquire);
    if (entry.handle.load(std::memory_order_acquire) != observed_handle ||
        incarnation(observed_state) != handle.incarnation()) {
      return kUnknownLevel;
    }
    return level(observed_state);
  }

  std::uint32_t transport_bytes(
      RemotePtr handle, std::uint32_t record_bytes,
      std::uint32_t graph_entry_capacity, bool enabled = true) {
    if (!enabled) return record_bytes;
    const std::uint32_t bytes = bytes_for_level(
      lookup(handle), record_bytes, graph_entry_capacity);
    if (bytes < record_bytes) {
      stats_.short_reads.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats_.full_reads.fetch_add(1, std::memory_order_relaxed);
    }
    return bytes;
  }

  Adaptation learn(RemotePtr handle, std::uint32_t required_bytes,
                   std::uint32_t graph_entry_capacity) {
    if (handle.is_null() || !handle.is_well_formed()) return Adaptation::none;
    const std::uint8_t observed_level =
      level_for_required_bytes(required_bytes, graph_entry_capacity);
    if (observed_level == kUnknownLevel) return Adaptation::none;

    Entry& entry = entries_[slot(handle)];
    for (;;) {
      std::uint64_t observed_handle =
        entry.handle.load(std::memory_order_acquire);
      if (observed_handle != handle.raw_address) {
        if (same_physical_slot(observed_handle, handle.raw_address) &&
            RemotePtr{observed_handle}.incarnation() > handle.incarnation()) {
          return Adaptation::none;
        }
        entry.state.store(
          pack_state(handle.incarnation(), kUnknownLevel),
          std::memory_order_release);
        if (observed_handle != 0) {
          stats_.conflicts.fetch_add(1, std::memory_order_relaxed);
        }
        if (!entry.handle.compare_exchange_weak(
              observed_handle, handle.raw_address,
              std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
      }

      std::uint32_t observed_state =
        entry.state.load(std::memory_order_acquire);
      if (incarnation(observed_state) != handle.incarnation()) {
        const std::uint32_t desired =
          pack_state(handle.incarnation(), kUnknownLevel);
        if (!entry.state.compare_exchange_weak(
              observed_state, desired,
              std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        observed_state = desired;
      }
      const AdaptationResult transition = adapt_level(
        level(observed_state), observed_level);
      if (transition.adaptation == Adaptation::none) return Adaptation::none;
      const std::uint32_t desired =
        pack_state(handle.incarnation(), transition.level);
      if (!entry.state.compare_exchange_weak(
            observed_state, desired,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      switch (transition.adaptation) {
        case Adaptation::refined_unknown:
          stats_.refined_unknown.fetch_add(1, std::memory_order_relaxed);
          break;
        case Adaptation::promoted:
          stats_.promotions.fetch_add(1, std::memory_order_relaxed);
          break;
        case Adaptation::demoted:
          stats_.demotions.fetch_add(1, std::memory_order_relaxed);
          break;
        case Adaptation::none:
          break;
      }
      return transition.adaptation;
    }
  }

  void note_fallback() {
    stats_.fallbacks.fetch_add(1, std::memory_order_relaxed);
  }

  void invalidate(RemotePtr handle) {
    if (handle.is_null() || !handle.is_well_formed()) return;
    Entry& entry = entries_[slot(handle)];
    if (entry.handle.load(std::memory_order_acquire) != handle.raw_address) {
      return;
    }
    std::uint32_t observed = entry.state.load(std::memory_order_acquire);
    while (incarnation(observed) == handle.incarnation() &&
           level(observed) != kUnknownLevel) {
      const std::uint32_t desired =
        pack_state(handle.incarnation(), kUnknownLevel);
      if (entry.state.compare_exchange_weak(
            observed, desired,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  CacheStats stats() const {
    return CacheStats{
      .short_reads = stats_.short_reads.load(std::memory_order_relaxed),
      .full_reads = stats_.full_reads.load(std::memory_order_relaxed),
      .fallbacks = stats_.fallbacks.load(std::memory_order_relaxed),
      .refined_unknown = stats_.refined_unknown.load(std::memory_order_relaxed),
      .promotions = stats_.promotions.load(std::memory_order_relaxed),
      .demotions = stats_.demotions.load(std::memory_order_relaxed),
      .conflicts = stats_.conflicts.load(std::memory_order_relaxed),
    };
  }

private:
  struct Entry {
    std::atomic<std::uint64_t> handle{0};
    std::atomic<std::uint32_t> state{0};
  };

  struct MutableStats {
    std::atomic<std::uint64_t> short_reads{0};
    std::atomic<std::uint64_t> full_reads{0};
    std::atomic<std::uint64_t> fallbacks{0};
    std::atomic<std::uint64_t> refined_unknown{0};
    std::atomic<std::uint64_t> promotions{0};
    std::atomic<std::uint64_t> demotions{0};
    std::atomic<std::uint64_t> conflicts{0};
  };

  static constexpr std::uint64_t physical_slot(std::uint64_t raw) {
    return raw & ((std::uint64_t{1} << RemotePtr::INCARNATION_SHIFT) - 1u);
  }

  static constexpr bool same_physical_slot(std::uint64_t lhs,
                                            std::uint64_t rhs) {
    return lhs != 0 && physical_slot(lhs) == physical_slot(rhs);
  }

  static constexpr std::uint32_t pack_state(std::uint32_t incarnation,
                                             std::uint8_t level) {
    return (incarnation & ((UINT32_C(1) << RemotePtr::INCARNATION_BITS) - 1u)) |
      (static_cast<std::uint32_t>(level) << RemotePtr::INCARNATION_BITS);
  }

  static constexpr std::uint32_t incarnation(std::uint32_t state) {
    return state & ((UINT32_C(1) << RemotePtr::INCARNATION_BITS) - 1u);
  }

  static constexpr std::uint8_t level(std::uint32_t state) {
    return static_cast<std::uint8_t>(state >> RemotePtr::INCARNATION_BITS);
  }

  std::size_t slot(RemotePtr handle) const {
    const std::uint64_t mixed = physical_slot(handle.raw_address) *
      UINT64_C(11400714819323198485);
    return static_cast<std::size_t>(mixed) & capacity_mask_;
  }

  std::vector<Entry> entries_;
  std::size_t capacity_mask_;
  mutable MutableStats stats_;
};

}  // namespace vamana::graph_extent

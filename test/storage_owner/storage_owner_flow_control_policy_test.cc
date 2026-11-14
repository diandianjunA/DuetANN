#include <cassert>

#include "service/compute_service/storage_owner/batch_policy.hh"

namespace detail = compute_service_detail;

namespace {

void test_repair_deadline_accounts_for_age_and_drain_rate() {
  assert(detail::storage_owner_repair_backlog(1, 0, 0) == 0);
  assert(detail::storage_owner_repair_backlog(1, 100, 50) == 50);
  assert(detail::storage_owner_repair_backlog(101, 80, 50) == 50);
  assert(detail::storage_owner_repair_backlog(50, 100, 50) == 50);
  assert(detail::storage_owner_repair_backlog(0, 100, 120) == 0);

  constexpr u64 kSecond = 1'000'000'000;
  assert(!detail::storage_owner_repair_deadline_blocked(
    0, 0, 0.0, kSecond));
  assert(!detail::storage_owner_repair_deadline_blocked(
    0, 4096, 0.0, 0));
  assert(detail::storage_owner_repair_deadline_blocked(
    kSecond / 2, 4096, 0.0, kSecond));
  assert(detail::storage_owner_repair_deadline_blocked(
    kSecond + 1, 1, 0.0, kSecond));
  assert(detail::storage_owner_repair_deadline_blocked(
    kSecond / 5, 100, 100.0, kSecond));
  assert(!detail::storage_owner_repair_deadline_blocked(
    kSecond / 5, 100, 200.0, kSecond));
  assert(detail::storage_owner_repair_deadline_exceeded(
    kSecond + 1, kSecond));
  assert(!detail::storage_owner_repair_deadline_exceeded(
    kSecond, kSecond));
}

}  // namespace

int main() {
  test_repair_deadline_accounts_for_age_and_drain_rate();
  return 0;
}

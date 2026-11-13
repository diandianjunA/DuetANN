#include "memory_node/storage_owner_maintenance/worker_internal.hh"

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  memory_node_storage_owner_maintenance_detail::Stage2Worker worker(*this, worker_id);
  worker.run();
}

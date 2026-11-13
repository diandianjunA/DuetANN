#include "common/configuration.hh"
#include "memory_node/memory_node.hh"

#include <csignal>
#include <cstdlib>

extern MemoryNode* dvstor_active_memory_node;

extern "C" void handle_memory_node_term(int) {
  if (dvstor_active_memory_node != nullptr) {
    dvstor_active_memory_node->request_shutdown();
  }
}

int main(int argc, char** argv) {
  std::signal(SIGTERM, handle_memory_node_term);
  std::signal(SIGUSR1, MemoryNode::enable_mutation_completion_trace);
  configuration::IndexConfiguration config{argc, argv};
  config.is_server = true;
  MemoryNode memory_node{config};
  return EXIT_SUCCESS;
}

#pragma once

#include <cstddef>
#include <string>
#include <vector>

// NUMA topology discovery for the CPU backend, used by --numa split.
// Linux only, everything reports an empty topology on other platforms.

namespace ggml::cpu::numa {

struct node {
    int id = -1;

    std::vector<int> cpus;      // usable logical CPUs, after the process affinity mask
    int              n_cores = 0; // usable physical cores

    size_t mem_total     = 0;
    size_t mem_available = 0;
};

// nodes that are online and have both usable CPUs and memory, empty if NUMA is not usable here
const std::vector<node> & topology();

// re-read the memory counters of a node, they change while the process runs
void refresh_memory(node & n);

// parse a sysfs tree directly, only the CPUs in cpu_mask are considered (empty means all).
// exposed for tests, use topology() otherwise
std::vector<node> parse_topology(const std::string & sysfs_root, const std::vector<int> & cpu_mask);

// parse a sysfs "0,2-4" style list
std::vector<int> parse_list(const std::string & list);

// Allocate memory that is guaranteed to be on the given node, or fail.
// The pages are bound with mbind, faulted by a thread pinned to the node and then checked, so that
// --numa split can never end up running on memory that silently landed somewhere else.
// Returns nullptr and fills error on failure, the caller is expected to log it and give up.
void * alloc_onnode(size_t size, int node, std::string & error);
void   free_onnode (void * ptr, size_t size);

// node the page at addr is on, -1 if that cannot be determined
int page_node(const void * addr);

// pin the calling thread to the given CPUs, best effort
bool bind_current_thread(const std::vector<int> & cpus);

} // namespace ggml::cpu::numa

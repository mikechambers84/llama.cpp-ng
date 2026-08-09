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

} // namespace ggml::cpu::numa

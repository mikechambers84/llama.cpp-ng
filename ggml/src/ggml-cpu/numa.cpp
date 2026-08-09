#include "numa.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(__gnu_linux__)
#include <sched.h>
#endif

namespace ggml::cpu::numa {

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::vector<int> parse_list(const std::string & list) {
    std::vector<int> ids;

    size_t pos = 0;
    while (pos < list.size()) {
        const size_t end = list.find(',', pos);
        std::string  tok = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end == std::string::npos ? list.size() : end + 1;

        // strip whitespace and newlines
        tok.erase(std::remove_if(tok.begin(), tok.end(), [](unsigned char c) { return std::isspace(c); }), tok.end());
        if (tok.empty()) {
            continue;
        }

        const size_t dash = tok.find('-');
        try {
            if (dash == std::string::npos) {
                ids.push_back(std::stoi(tok));
            } else {
                const int lo = std::stoi(tok.substr(0, dash));
                const int hi = std::stoi(tok.substr(dash + 1));
                for (int i = lo; i <= hi; i++) {
                    ids.push_back(i);
                }
            }
        } catch (const std::exception &) {
            return {};
        }
    }

    return ids;
}

// value of a "Node 0 MemTotal:  123 kB" style line, in bytes
static size_t parse_meminfo(const std::string & meminfo, const std::string & key) {
    std::istringstream ss(meminfo);
    std::string        line;
    while (std::getline(ss, line)) {
        const size_t k = line.find(key + ":");
        if (k == std::string::npos) {
            continue;
        }
        try {
            return (size_t) std::stoull(line.substr(k + key.size() + 1)) * 1024;
        } catch (const std::exception &) {
            return 0;
        }
    }
    return 0;
}

static int count_cores(const std::string & sysfs_root, const std::vector<int> & cpus) {
    std::vector<int> cores;
    cores.reserve(cpus.size());

    for (int cpu : cpus) {
        std::string siblings;
        if (!read_file(sysfs_root + "/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list", siblings)) {
            // no topology information, assume the CPU is its own core
            cores.push_back(cpu);
            continue;
        }
        const std::vector<int> ids = parse_list(siblings);
        cores.push_back(ids.empty() ? cpu : *std::min_element(ids.begin(), ids.end()));
    }

    std::sort(cores.begin(), cores.end());
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());

    return (int) cores.size();
}

std::vector<node> parse_topology(const std::string & sysfs_root, const std::vector<int> & cpu_mask) {
    const std::string node_root = sysfs_root + "/devices/system/node";

    std::string online;
    if (!read_file(node_root + "/online", online)) {
        return {};
    }

    std::vector<node> nodes;
    for (int id : parse_list(online)) {
        const std::string node_dir = node_root + "/node" + std::to_string(id);

        std::string cpulist;
        if (!read_file(node_dir + "/cpulist", cpulist)) {
            continue;
        }

        node n;
        n.id = id;

        for (int cpu : parse_list(cpulist)) {
            if (cpu_mask.empty() || std::find(cpu_mask.begin(), cpu_mask.end(), cpu) != cpu_mask.end()) {
                n.cpus.push_back(cpu);
            }
        }

        std::string meminfo;
        read_file(node_dir + "/meminfo", meminfo);
        n.mem_total     = parse_meminfo(meminfo, "MemTotal");
        n.mem_available = parse_meminfo(meminfo, "MemFree");

        // a node is only usable if it can both run threads and hold weights
        if (n.cpus.empty() || n.mem_total == 0) {
            continue;
        }

        n.n_cores = count_cores(sysfs_root, n.cpus);

        nodes.push_back(std::move(n));
    }

    return nodes;
}

void refresh_memory(node & n) {
    std::string meminfo;
    if (!read_file("/sys/devices/system/node/node" + std::to_string(n.id) + "/meminfo", meminfo)) {
        return;
    }
    n.mem_total     = parse_meminfo(meminfo, "MemTotal");
    n.mem_available = parse_meminfo(meminfo, "MemFree");
}

#if defined(__gnu_linux__)
static std::vector<int> process_cpu_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return {};
    }

    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &set)) {
            cpus.push_back(i);
        }
    }

    return cpus;
}

const std::vector<node> & topology() {
    static const std::vector<node> nodes = parse_topology("/sys", process_cpu_mask());
    return nodes;
}
#else
const std::vector<node> & topology() {
    static const std::vector<node> nodes;
    return nodes;
}
#endif

} // namespace ggml::cpu::numa

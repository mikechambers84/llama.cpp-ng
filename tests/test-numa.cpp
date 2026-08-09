// tests for --numa split: the topology parser and the CPU device defaults
//
// the parser is pure, so it runs against generated sysfs trees and needs no NUMA hardware

#include "../ggml/src/ggml-cpu/numa.h"

#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ggml::cpu::numa;

namespace fs = std::filesystem;

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-58s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

static void write_file(const fs::path & path, const std::string & content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << content;
}

struct fake_node {
    int         id;
    std::string cpulist;
    size_t      mem_total_kb;
};

// build a sysfs tree, every CPU gets a sibling at cpu+n_cpus_per_core_offset to emulate SMT
static fs::path make_sysfs(const std::string & name, const std::string & online, const std::vector<fake_node> & nodes, int smt_offset) {
    const fs::path root = fs::temp_directory_path() / ("ggml-numa-test-" + name);
    fs::remove_all(root);

    write_file(root / "devices/system/node/online", online + "\n");

    for (const auto & n : nodes) {
        const fs::path dir = root / ("devices/system/node/node" + std::to_string(n.id));
        write_file(dir / "cpulist", n.cpulist + "\n");
        write_file(dir / "meminfo",
                "Node " + std::to_string(n.id) + " MemTotal:       " + std::to_string(n.mem_total_kb) + " kB\n"
                "Node " + std::to_string(n.id) + " MemFree:        " + std::to_string(n.mem_total_kb / 2) + " kB\n");

        for (int cpu : parse_list(n.cpulist)) {
            const int sibling = cpu < smt_offset ? cpu + smt_offset : cpu - smt_offset;
            write_file(root / ("devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list"),
                    std::to_string(std::min(cpu, sibling)) + "," + std::to_string(std::max(cpu, sibling)) + "\n");
        }
    }

    return root;
}

int main() {
    printf("parse_list\n");
    {
        check(parse_list("0-3") == std::vector<int>({0, 1, 2, 3}), "range");
        check(parse_list("0,2,4") == std::vector<int>({0, 2, 4}), "comma separated");
        check(parse_list("0-1,4,6-7\n") == std::vector<int>({0, 1, 4, 6, 7}), "mixed with newline");
        check(parse_list("5") == std::vector<int>({5}), "single value");
        check(parse_list("").empty(), "empty");
        check(parse_list("garbage").empty(), "garbage");
    }

    printf("parse_topology\n");
    {
        // this machine: 2 nodes, interleaved CPU numbering, hyperthreads at +48
        const fs::path root = make_sysfs("2node", "0-1", {{0, "0-1,48-49", 8000}, {1, "2-3,50-51", 8000}}, 48);

        const auto nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "two nodes found");
        if (nodes.size() == 2) {
            check(nodes[0].id == 0 && nodes[1].id == 1, "node ids");
            check(nodes[0].cpus == std::vector<int>({0, 1, 48, 49}), "node 0 cpus");
            check(nodes[1].cpus == std::vector<int>({2, 3, 50, 51}), "node 1 cpus");
            check(nodes[0].n_cores == 2 && nodes[1].n_cores == 2, "hyperthreads counted as one core");
            check(nodes[0].mem_total == 8000ull * 1024, "memory in bytes");
        }
    }
    {
        const fs::path root  = make_sysfs("1node", "0", {{0, "0-3", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1, "single node");
    }
    {
        // node 1 offline, so the ids have a hole
        const fs::path root  = make_sysfs("hole", "0,2", {{0, "0-1", 8000}, {2, "4-5", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "hole in node numbering");
        if (nodes.size() == 2) {
            check(nodes[1].id == 2, "second node keeps its id");
        }
    }
    {
        // a memory-only node, e.g. CXL, cannot run threads
        const fs::path root  = make_sysfs("nocpu", "0-1", {{0, "0-1", 8000}, {1, "", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without cpus dropped");
    }
    {
        const fs::path root  = make_sysfs("nomem", "0-1", {{0, "0-1", 8000}, {1, "2-3", 0}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without memory dropped");
    }
    {
        const fs::path root = make_sysfs("affinity", "0-1", {{0, "0-1", 8000}, {1, "2-3", 8000}}, 2);

        const auto pinned = parse_topology(root.string(), {2, 3});
        check(pinned.size() == 1 && pinned[0].id == 1, "affinity mask leaves one node");

        const auto partial = parse_topology(root.string(), {0, 2});
        check(partial.size() == 2 && partial[0].cpus == std::vector<int>({0}), "affinity mask trims cpus");
    }
    {
        const auto nodes = parse_topology("/nonexistent-sysfs", {});
        check(nodes.empty(), "missing sysfs is not an error");
    }
    {
        std::vector<fake_node> many;
        for (int i = 0; i < 16; i++) {
            many.push_back({i, std::to_string(2 * i) + "-" + std::to_string(2 * i + 1), 8000});
        }
        const fs::path root  = make_sysfs("16node", "0-15", many, 1);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 16, "sixteen nodes");
    }

    printf("node local allocation\n");
    if (topology().size() < 2) {
        printf("  skipped, this machine has fewer than 2 usable NUMA nodes\n");
    } else {
        const size_t size = 64ull << 20;

        for (const auto & n : topology()) {
            std::string error;

            void * data = alloc_onnode(size, n.id, error);
            check(data != nullptr, "allocated 64 MiB on node " + std::to_string(n.id) + (data ? "" : ": " + error));
            if (data == nullptr) {
                continue;
            }

            // the allocator only samples pages, so check the whole range here
            size_t n_remote = 0;
            for (size_t off = 0; off < size; off += 2u << 20) {
                ((char *) data)[off] = 1;
                if (page_node((char *) data + off) != n.id) {
                    n_remote++;
                }
            }
            check(n_remote == 0, "every page of node " + std::to_string(n.id) + " is local");

            free_onnode(data, size);
        }

        std::string error;
        check(alloc_onnode(size, 999, error) == nullptr && !error.empty(), "allocation on an unusable node fails");
    }

    // without --numa split the CPU backend must look exactly like it always did
    printf("cpu device defaults\n");
    {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        check(cpu != nullptr, "cpu device found");

        if (cpu != nullptr) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);

            check(ggml_backend_reg_dev_count(reg) == 1, "cpu reg exposes a single device");
            check(ggml_backend_reg_dev_get(reg, 0) == cpu, "and it is the one dev_by_type returns");
            check(std::string(ggml_backend_dev_name(cpu)) == "CPU", "named CPU");
            check(ggml_backend_dev_buffer_type(cpu) == ggml_backend_cpu_buffer_type(), "uses the shared cpu buffer type");

            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(cpu, &props);
            check(!props.caps.async, "synchronous");
            check(props.caps.buffer_from_host_ptr, "maps host pointers, so mmap is used for weights");
            check(props.device_id == nullptr, "no device id");

            auto numa_node_fn = (ggml_backend_dev_get_numa_node_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_numa_node");
            check(numa_node_fn != nullptr && numa_node_fn(cpu) == -1, "not bound to a numa node");

            auto n_threads_max_fn = (ggml_backend_dev_get_n_threads_max_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_n_threads_max");
            check(n_threads_max_fn != nullptr && n_threads_max_fn(cpu) == 0, "no thread limit of its own");
        }
    }

    printf("topology of this machine\n");
    for (const auto & n : topology()) {
        printf("  node %d: %zu cpus, %d cores, %zu MiB total, %zu MiB free\n",
                n.id, n.cpus.size(), n.n_cores, n.mem_total >> 20, n.mem_available >> 20);
    }

    printf("%s\n", n_fail == 0 ? "test-numa: all tests OK" : "test-numa: FAILED");

    return n_fail == 0 ? 0 : 1;
}

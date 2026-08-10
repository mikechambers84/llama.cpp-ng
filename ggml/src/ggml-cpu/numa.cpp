#include "numa.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(__gnu_linux__)
#include <cerrno>
#include <cstring>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// from linux/mempolicy.h, not exposed by glibc
#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
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
        // there is no per node MemAvailable; free plus reclaimable page cache is the usable estimate,
        // MemFree alone would report a node full of cached file pages as having no memory
        n.mem_available = parse_meminfo(meminfo, "MemFree") + parse_meminfo(meminfo, "Inactive(file)");

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
    n.mem_available = parse_meminfo(meminfo, "MemFree") + parse_meminfo(meminfo, "Inactive(file)");
}

#if defined(__gnu_linux__)

static size_t page_size() {
    static const size_t size = (size_t) sysconf(_SC_PAGESIZE);
    return size;
}

static size_t page_align(size_t size) {
    const size_t ps = page_size();
    return (size + ps - 1) & ~(ps - 1);
}

int page_node(const void * addr) {
    void * pages[1] = { const_cast<void *>(addr) };
    int    status[1] = { -1 };

    if (syscall(__NR_move_pages, 0, 1, pages, nullptr, status, 0) != 0) {
        return -1;
    }

    return status[0];
}

// write to one byte of every page in [first, last) so that the pages are really allocated
static void touch_pages(char * addr, size_t first, size_t last, size_t stride) {
    for (size_t off = first; off < last; off += stride) {
        addr[off] = 0;
    }
}

// fault the mapping from a thread pinned to the node, then check where the pages ended up
static bool touch_and_verify(void * addr, size_t len, int node_id, bool bound, std::string & error) {
    const std::vector<node> & nodes = topology();

    const auto it = std::find_if(nodes.begin(), nodes.end(), [node_id](const struct node & n) { return n.id == node_id; });
    if (it == nodes.end()) {
        error = "node " + std::to_string(node_id) + " is not usable";
        return false;
    }

    cpu_set_t prev;
    CPU_ZERO(&prev);
    const bool have_prev = sched_getaffinity(0, sizeof(prev), &prev) == 0;

    cpu_set_t on_node;
    CPU_ZERO(&on_node);
    for (int cpu : it->cpus) {
        CPU_SET(cpu, &on_node);
    }
    const bool pinned = sched_setaffinity(0, sizeof(on_node), &on_node) == 0;
    if (!pinned && !bound) {
        error = "cannot pin a thread to node " + std::to_string(node_id) + " to place its memory: " + strerror(errno);
        return false;
    }

    // when mbind took, the policy places every page, so only a sample has to be faulted here.
    // without it first touch is the only placement mechanism, so every page has to be faulted now
    const size_t ps      = page_size();
    const size_t n_pages = len / ps;
    const size_t stride  = bound ? std::max<size_t>(1, n_pages / 64) * ps : ps;

    touch_pages((char *) addr, 0, len, stride);

    // check a sample spread over the whole mapping, independent of how much was faulted above -
    // sampling only the beginning would miss pages that landed elsewhere further in
    const size_t vstride = std::max<size_t>(1, n_pages / 64) * ps;

    std::vector<void *> pages;
    std::vector<int>    status;
    for (size_t off = 0; off < len && pages.size() < 64; off += vstride) {
        pages .push_back((char *) addr + off);
        status.push_back(-1);
    }

    bool ok = true;
    if (syscall(__NR_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) == 0) {
        size_t n_remote = 0;
        int    seen     = -1;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] >= 0 && status[i] != node_id) {
                n_remote++;
                seen = status[i];
            }
        }
        if (n_remote > 0) {
            error = "asked for node " + std::to_string(node_id) + " but " + std::to_string(n_remote) + " of " +
                    std::to_string(status.size()) + " sampled pages are on node " + std::to_string(seen);
            ok = false;
        }
    }

    if (have_prev) {
        sched_setaffinity(0, sizeof(prev), &prev);
    }

    return ok;
}

void * alloc_onnode(size_t size, int node_id, std::string & error) {
    if (size == 0) {
        error = "zero size allocation";
        return nullptr;
    }

    const size_t len = page_align(size);

    // MPOL_BIND reports an over-commit as an OOM when the pages are faulted, which would be a crash
    // in the middle of loading, so an allocation that cannot fit in the node at all is refused here.
    // merely exceeding the free + reclaimable estimate is not an error: the estimate is imprecise
    // and the kernel reclaims cache as the pages fault in (the caller warns about it)
    struct node n;
    n.id = node_id;
    refresh_memory(n);
    if (n.mem_total > 0 && len > n.mem_total) {
        error = "node " + std::to_string(node_id) + " has " + std::to_string(n.mem_total >> 20) +
                " MiB in total, need " + std::to_string(len >> 20) + " MiB";
        return nullptr;
    }

    void * addr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        error = std::string("mmap failed: ") + strerror(errno);
        return nullptr;
    }

    // bind before the first fault, the policy is applied when the pages are allocated
    const size_t                bits_per_word = 8 * sizeof(unsigned long);
    std::vector<unsigned long>  nodemask(node_id / bits_per_word + 1, 0);
    nodemask[node_id / bits_per_word] = 1UL << (node_id % bits_per_word);

    bool bound = syscall(__NR_mbind, addr, len, MPOL_BIND, nodemask.data(), nodemask.size() * bits_per_word, 0) == 0;
    if (!bound) {
        const int err = errno;
        if (err != ENOSYS && err != EPERM) {
            error = std::string("mbind failed: ") + strerror(err);
            munmap(addr, len);
            return nullptr;
        }

        // without mbind the only way to place pages is to fault them from the node_id, which the kernel
        // only honours if the process did not inherit a different policy
        int mode = MPOL_DEFAULT;
        if (syscall(__NR_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0 || mode != MPOL_DEFAULT) {
            error = std::string("mbind is unavailable (") + strerror(err) +
                    ") and the process runs under a non default memory policy, so pages cannot be placed";
            munmap(addr, len);
            return nullptr;
        }
    }

    // best effort, large pages help a lot at these sizes
    madvise(addr, len, MADV_HUGEPAGE);

    if (!touch_and_verify(addr, len, node_id, bound, error)) {
        munmap(addr, len);
        return nullptr;
    }

    return addr;
}

void free_onnode(void * ptr, size_t size) {
    if (ptr != nullptr) {
        munmap(ptr, page_align(size));
    }
}

bool bind_current_thread(const std::vector<int> & cpus) {
    if (cpus.empty()) {
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &set);
        }
    }

    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

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

int page_node(const void * addr) {
    return -1;

    (void) addr;
}

void * alloc_onnode(size_t size, int node, std::string & error) {
    error = "NUMA memory placement is only implemented on Linux";
    return nullptr;

    (void) size;
    (void) node;
}

void free_onnode(void * ptr, size_t size) {
    (void) ptr;
    (void) size;
}

bool bind_current_thread(const std::vector<int> & cpus) {
    return false;

    (void) cpus;
}

#endif

} // namespace ggml::cpu::numa

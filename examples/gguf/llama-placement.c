
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <sys/stat.h>

#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-placement-memory.h"

#ifdef __linux__
#include <dirent.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

#ifndef MAX_GPUS
#define MAX_GPUS 32
#endif

#ifndef MAX_NUMA_NODES
#define MAX_NUMA_NODES 64
#endif

#define UNKNOWN_NUMA_NODE (-1)
#define DEFAULT_DENSE_VRAM_RESERVE_MIB 768.0
#define DEFAULT_MOE_VRAM_RESERVE_MIB   896.0

struct hw_gpu {
    char name[256];
    char backend_name[128];
    char device_id[128];
    uint64_t total_bytes;
    uint64_t free_bytes;
    int numa_node;
};

struct hw_numa_node {
    int id;
    uint64_t memory_total_bytes;
    uint64_t memory_free_bytes;
    int logical_cpus;
    char cpulist[512];
};

struct hw_info {
    int gpu_count;
    struct hw_gpu gpus[MAX_GPUS];
    int cpu_count;
    char cpu_names[GGML_CPU_NUMA_SPLIT_MAX_DEVICES][128];
    int numa_count;
    struct hw_numa_node numa[MAX_NUMA_NODES];
};

struct placement_plan {
    int valid;
    int ngl;
    int ncmoe;          /* -1 for dense */
    int gpu_slots[MAX_GPUS];
    int cpu_slots;
    uint64_t gpu_bytes[MAX_GPUS];
    uint64_t cpu_expert_bytes;
};

struct planner_options {
    const char * model_path;
    uint32_t context;
    uint32_t batch;
    uint32_t ubatch;
    int threads;
    int measure;
};

/* ------------------------------------------------------------------------- */
/* GGUF helpers                                                              */
/* ------------------------------------------------------------------------- */

static int64_t get_int(const struct gguf_context * ctx, const char * key, int64_t def) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return def;

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:               return def;
    }
}

static const char * get_str(const struct gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return NULL;
    return gguf_get_val_str(ctx, id);
}

static int tensor_layer(const char * name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;

    char * end = NULL;
    long n = strtol(name + 4, &end, 10);

    if (end == name + 4 || *end != '.' || n < 0 || n > 100000) return -1;
    return (int) n;
}

/*
 * Mirrors llama.cpp-ng:
 * LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
 */
static int is_routed_expert_tensor(const char * name) {
    static const char * patterns[] = {
        ".ffn_up_exps",
        ".ffn_down_exps",
        ".ffn_gate_exps",
        ".ffn_gate_up_exps",
        ".ffn_up_chexps",
        ".ffn_down_chexps",
        ".ffn_gate_chexps",
        ".ffn_gate_up_chexps",
        NULL
    };

    for (int i = 0; patterns[i] != NULL; ++i) {
        if (strstr(name, patterns[i]) != NULL) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* CPU-list parsing                                                         */
/* ------------------------------------------------------------------------- */

static int count_cpu_list(const char * s) {
    int count = 0;

    while (*s) {
        while (*s == ',' || isspace((unsigned char) *s)) ++s;
        if (!*s) break;

        char * end = NULL;
        long first = strtol(s, &end, 10);
        if (end == s) break;
        s = end;

        if (*s == '-') {
            ++s;
            long last = strtol(s, &end, 10);
            if (end != s && last >= first) {
                count += (int) (last - first + 1);
                s = end;
            } else {
                ++count;
            }
        } else {
            ++count;
        }

        while (*s && *s != ',') ++s;
    }

    return count;
}

/* ------------------------------------------------------------------------- */
/* GPU discovery through ggml                                               */
/* ------------------------------------------------------------------------- */

static void detect_ggml_gpus(struct hw_info * hw) {
    ggml_backend_load_all();

    size_t ndev = ggml_backend_dev_count();

    for (size_t i = 0; i < ndev && hw->gpu_count < MAX_GPUS; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        struct ggml_backend_dev_props props;
        memset(&props, 0, sizeof(props));
        ggml_backend_dev_get_props(dev, &props);

        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        int duplicate = 0;
        for (int j = 0; props.device_id && j < hw->gpu_count; ++j) {
            if (strcmp(hw->gpus[j].device_id, props.device_id) == 0) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;

        struct hw_gpu * gpu = &hw->gpus[hw->gpu_count];
        memset(gpu, 0, sizeof(*gpu));
        gpu->numa_node = UNKNOWN_NUMA_NODE;

        const char * desc = props.description ? props.description : props.name;
        if (!desc) desc = "unknown";

        snprintf(gpu->name, sizeof(gpu->name), "%s", desc);
        snprintf(gpu->backend_name, sizeof(gpu->backend_name), "%s", props.name);

        if (props.device_id) {
            snprintf(gpu->device_id, sizeof(gpu->device_id), "%s", props.device_id);
        }

        size_t mem_free = 0;
        size_t mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);

        gpu->free_bytes = (uint64_t) mem_free;
        gpu->total_bytes = (uint64_t) mem_total;

        ++hw->gpu_count;
    }
}

/* ------------------------------------------------------------------------- */
/* Linux NUMA / PCI locality                                                */
/* ------------------------------------------------------------------------- */

#ifdef __linux__

static int read_text_file(const char * path, char * buf, size_t buf_size) {
    FILE * f = fopen(path, "r");
    if (!f) return 0;

    if (!fgets(buf, (int) buf_size, f)) {
        fclose(f);
        return 0;
    }

    fclose(f);
    buf[strcspn(buf, "\r\n")] = '\0';
    return 1;
}

static void linux_read_node_memory(
        int node,
        uint64_t * total_bytes,
        uint64_t * free_bytes) {

    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/node/node%d/meminfo", node);

    FILE * f = fopen(path, "r");
    if (!f) return;

    char line[512];
    uint64_t total_kib = 0;
    uint64_t free_kib = 0;

    while (fgets(line, sizeof(line), f)) {
        unsigned long long value = 0;

        if (sscanf(line, "Node %*d MemTotal: %llu kB", &value) == 1) {
            total_kib = (uint64_t) value;
            continue;
        }

        if (sscanf(line, "Node %*d MemFree: %llu kB", &value) == 1) {
            free_kib = (uint64_t) value;
            continue;
        }
    }

    fclose(f);

    *total_bytes = total_kib * 1024ULL;
    *free_bytes  = free_kib  * 1024ULL;
}

static void detect_linux_numa(struct hw_info * hw) {
    for (int node = 0; node < MAX_NUMA_NODES; ++node) {
        char node_path[256];
        snprintf(node_path, sizeof(node_path),
                 "/sys/devices/system/node/node%d", node);

        if (access(node_path, F_OK) != 0) continue;
        if (hw->numa_count >= MAX_NUMA_NODES) break;

        struct hw_numa_node * out = &hw->numa[hw->numa_count];
        memset(out, 0, sizeof(*out));
        out->id = node;

        char cpulist_path[256];
        snprintf(cpulist_path, sizeof(cpulist_path),
                 "/sys/devices/system/node/node%d/cpulist", node);

        if (read_text_file(cpulist_path, out->cpulist, sizeof(out->cpulist))) {
            out->logical_cpus = count_cpu_list(out->cpulist);
        }

        linux_read_node_memory(
            node,
            &out->memory_total_bytes,
            &out->memory_free_bytes);

        ++hw->numa_count;
    }

    for (int i = 0; i < hw->gpu_count; ++i) {
        struct hw_gpu * gpu = &hw->gpus[i];

        if (gpu->device_id[0] == '\0') continue;

        char path[512];
        snprintf(path, sizeof(path),
                 "/sys/bus/pci/devices/%s/numa_node",
                 gpu->device_id);

        FILE * f = fopen(path, "r");
        if (!f) continue;

        int node = UNKNOWN_NUMA_NODE;
        if (fscanf(f, "%d", &node) == 1) gpu->numa_node = node;
        fclose(f);
    }
}

#endif

/* ------------------------------------------------------------------------- */
/* Windows topology                                                         */
/* ------------------------------------------------------------------------- */

#ifdef _WIN32

static int popcount_uintptr(ULONG_PTR v) {
    int n = 0;
    while (v) {
        n += (int) (v & 1);
        v >>= 1;
    }
    return n;
}

static void detect_windows_numa(struct hw_info * hw) {
    ULONG highest = 0;
    if (!GetNumaHighestNodeNumber(&highest)) return;

    for (ULONG node = 0;
         node <= highest && hw->numa_count < MAX_NUMA_NODES;
         ++node) {

        GROUP_AFFINITY affinity;
        memset(&affinity, 0, sizeof(affinity));

        if (!GetNumaNodeProcessorMaskEx((USHORT) node, &affinity)) continue;

        struct hw_numa_node * out = &hw->numa[hw->numa_count];
        memset(out, 0, sizeof(*out));

        out->id = (int) node;
        out->logical_cpus = popcount_uintptr(affinity.Mask);

        ULONGLONG available = 0;
        if (GetNumaAvailableMemoryNodeEx((USHORT) node, &available)) {
            out->memory_free_bytes = (uint64_t) available;
        }

        snprintf(out->cpulist, sizeof(out->cpulist),
                 "processor-group=%u", (unsigned) affinity.Group);

        ++hw->numa_count;
    }
}

#endif

/* ------------------------------------------------------------------------- */
/* macOS topology                                                           */
/* ------------------------------------------------------------------------- */

#ifdef __APPLE__

static uint64_t apple_sysctl_u64(const char * name) {
    uint64_t value = 0;
    size_t size = sizeof(value);

    if (sysctlbyname(name, &value, &size, NULL, 0) != 0) return 0;
    return value;
}

static void detect_apple_topology(struct hw_info * hw) {
    if (hw->numa_count >= MAX_NUMA_NODES) return;

    struct hw_numa_node * out = &hw->numa[hw->numa_count];
    memset(out, 0, sizeof(*out));

    out->id = 0;
    out->memory_total_bytes = apple_sysctl_u64("hw.memsize");
    out->logical_cpus = (int) apple_sysctl_u64("hw.logicalcpu");

    snprintf(out->cpulist, sizeof(out->cpulist),
             "0-%d", out->logical_cpus > 0 ? out->logical_cpus - 1 : 0);

    ++hw->numa_count;
}

#endif

/* ------------------------------------------------------------------------- */
/* Generic fallback                                                         */
/* ------------------------------------------------------------------------- */

#if !defined(__linux__) && !defined(_WIN32) && !defined(__APPLE__)

#include <unistd.h>

static void detect_generic_topology(struct hw_info * hw) {
    if (hw->numa_count >= MAX_NUMA_NODES) return;

    struct hw_numa_node * out = &hw->numa[hw->numa_count];
    memset(out, 0, sizeof(*out));
    out->id = 0;

#ifdef _SC_NPROCESSORS_ONLN
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus > 0) out->logical_cpus = (int) cpus;
#endif

#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);

    if (pages > 0 && page_size > 0) {
        out->memory_total_bytes =
            (uint64_t) pages * (uint64_t) page_size;
    }
#endif

    ++hw->numa_count;
}

#endif

static void detect_cpu_devices(struct hw_info * hw) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
    ggml_backend_cpu_numa_split_init_t init = NULL;
    void * address = ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_split_init");
    if (sizeof(init) != sizeof(address)) {
        return;
    }
    memcpy(&init, &address, sizeof(init));
    if (!init) {
        return;
    }
    ggml_backend_dev_t nodes[GGML_CPU_NUMA_SPLIT_MAX_DEVICES];
    size_t count = GGML_CPU_NUMA_SPLIT_MAX_DEVICES;
    if (init(nodes, &count) != GGML_NUMA_SPLIT_STATUS_SUCCESS) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        snprintf(hw->cpu_names[i], sizeof(hw->cpu_names[i]), "%s", ggml_backend_dev_name(nodes[i]));
        ggml_backend_device_register(nodes[i]);
    }
    hw->cpu_count = (int) count;
}

static void detect_hardware(struct hw_info * hw) {
    memset(hw, 0, sizeof(*hw));

    detect_ggml_gpus(hw);
    detect_cpu_devices(hw);

#ifdef __linux__
    detect_linux_numa(hw);
#elif defined(_WIN32)
    detect_windows_numa(hw);
#elif defined(__APPLE__)
    detect_apple_topology(hw);
#else
    detect_generic_topology(hw);
#endif
}

static void print_hardware(const struct hw_info * hw) {
    const double MiB = 1024.0 * 1024.0;
    const double GiB = 1024.0 * 1024.0 * 1024.0;

    printf("\n");
    printf("hardware.gpus=%d\n", hw->gpu_count);

    for (int i = 0; i < hw->gpu_count; ++i) {
        const struct hw_gpu * gpu = &hw->gpus[i];

        printf("gpu.%d.name=%s\n", i, gpu->name);
        printf("gpu.%d.total_mib=%.0f\n", i, (double) gpu->total_bytes / MiB);
        printf("gpu.%d.free_mib=%.0f\n", i, (double) gpu->free_bytes / MiB);
        printf("gpu.%d.total_gib=%.3f\n", i, (double) gpu->total_bytes / GiB);
        printf("gpu.%d.free_gib=%.3f\n", i, (double) gpu->free_bytes / GiB);

        if (gpu->device_id[0]) {
            printf("gpu.%d.device_id=%s\n", i, gpu->device_id);
        }

        printf("gpu.%d.numa_node=%d\n", i, gpu->numa_node);
    }

    printf("hardware.numa_nodes=%d\n", hw->numa_count);

    for (int i = 0; i < hw->numa_count; ++i) {
        const struct hw_numa_node * node = &hw->numa[i];

        printf("numa.%d.logical_cpus=%d\n", node->id, node->logical_cpus);

        if (node->cpulist[0]) {
            printf("numa.%d.cpus=%s\n", node->id, node->cpulist);
        }

        if (node->memory_total_bytes) {
            printf("numa.%d.memory_total_gib=%.3f\n",
                   node->id,
                   (double) node->memory_total_bytes / GiB);
        }

        if (node->memory_free_bytes) {
            printf("numa.%d.memory_free_gib=%.3f\n",
                   node->id,
                   (double) node->memory_free_bytes / GiB);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Placement planner                                                        */
/* ------------------------------------------------------------------------- */

static uint64_t gpu_budget_bytes(const struct hw_gpu * gpu, double reserve_mib) {
    uint64_t reserve = (uint64_t) (reserve_mib * 1024.0 * 1024.0);
    return gpu->free_bytes > reserve ? gpu->free_bytes - reserve : 0;
}

static uint64_t cpu_expert_bytes_for_ncmoe(int ncmoe, const uint64_t * expert_bytes) {
    uint64_t total = 0;
    for (int i = 0; i < ncmoe; ++i) {
        total += expert_bytes[i];
    }
    return total;
}

static struct placement_plan plan_layers(
        const struct hw_info * hw,
        int block_count,
        int main_layers,
        const uint64_t * layer_bytes,
        const uint64_t * expert_bytes,
        uint64_t nonlayer_bytes,
        int ncmoe,
        double reserve_mib,
        int require_all_gpu) {
    struct placement_plan plan = {0};
    const int slots = block_count + 1;
    const size_t width = (size_t) slots + 1;
    uint64_t * prefix = calloc(width, sizeof(*prefix));
    int * previous = malloc(((size_t) hw->gpu_count + 1) * width * sizeof(*previous));
    double * cost = malloc(2 * width * sizeof(*cost));
    if (!prefix || !previous || !cost) {
        fprintf(stderr, "out of memory in placement planner\n");
        free(prefix);
        free(previous);
        free(cost);
        return plan;
    }

    const int reverse = hw->cpu_count == 0 && hw->gpu_count > 0;
    for (int slot = 0; slot < slots; ++slot) {
        const int i = reverse ? slots - 1 - slot : slot;
        uint64_t bytes = i < main_layers ? layer_bytes[i] : 0;
        if (i < ncmoe) {
            bytes -= expert_bytes[i];
        }
        if (i == block_count) {
            bytes = nonlayer_bytes;
        }
        prefix[slot + 1] = prefix[slot] + bytes;
    }

    for (int i = 0; i <= slots; ++i) {
        cost[i] = i == 0 ? 0.0 : HUGE_VAL;
    }
    // Each GPU receives a contiguous range. Minimize squared slot counts to break capacity ties.
    for (int gpu = 0; gpu < hw->gpu_count; ++gpu) {
        double * before = cost + (gpu % 2) * width;
        double * after = cost + ((gpu + 1) % 2) * width;
        const uint64_t budget = gpu_budget_bytes(&hw->gpus[reverse ? hw->gpu_count - 1 - gpu : gpu], reserve_mib);
        for (int end = 0; end <= slots; ++end) {
            after[end] = HUGE_VAL;
            previous[(gpu + 1) * width + end] = -1;
            for (int start = end; start >= 0; --start) {
                if (prefix[end] - prefix[start] > budget) {
                    break;
                }
                const double count = end - start;
                const double candidate = before[start] + count * count;
                if (candidate < after[end]) {
                    after[end] = candidate;
                    previous[(gpu + 1) * width + end] = start;
                }
            }
        }
    }

    double * final = cost + (hw->gpu_count % 2) * width;
    int end = slots;
    while (end > 0 && !isfinite(final[end])) {
        --end;
    }
    if ((!require_all_gpu || end == slots) && isfinite(final[end])) {
        plan.valid = 1;
        plan.ngl = slots;
        plan.ncmoe = ncmoe;
        plan.cpu_slots = slots - end;
        plan.cpu_expert_bytes = ncmoe < 0 ? 0 : cpu_expert_bytes_for_ncmoe(ncmoe, expert_bytes);
        for (int gpu = hw->gpu_count - 1; gpu >= 0; --gpu) {
            int start = previous[(gpu + 1) * width + end];
            const int index = reverse ? hw->gpu_count - 1 - gpu : gpu;
            plan.gpu_slots[index] = end - start;
            plan.gpu_bytes[index] = prefix[end] - prefix[start];
            end = start;
        }
    }
    free(prefix);
    free(previous);
    free(cost);
    return plan;
}

static void print_layer_split(const struct hw_info * hw, const struct placement_plan * plan) {
    for (int gpu = 0; gpu < hw->gpu_count; ++gpu) {
        printf("%s%d", gpu ? "/" : "", plan->gpu_slots[gpu]);
    }
    if (hw->cpu_count || !hw->gpu_count) {
        printf("%s%d", hw->gpu_count ? "/" : "", plan->cpu_slots);
    }
}

static void print_plan(
        int is_moe,
        const struct hw_info * hw,
        const struct placement_plan * plan,
        const char * label,
        double reserve_mib,
        const struct planner_options * options) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    printf("\nplan.%s.valid=%d\n", label, plan->valid);
    if (!plan->valid) {
        return;
    }
    const int numa = hw->cpu_count >= 2;
    const int explicit_cpu = hw->cpu_count > 0;
    const int ngl = explicit_cpu ? plan->ngl : plan->ngl - plan->cpu_slots;
    printf("plan.%s.ngl=%d\n", label, ngl);
    printf("plan.%s.threads=%d\n", label, options->threads);
    printf("plan.%s.context=%u\n", label, options->context);
    if (is_moe) {
        printf("plan.%s.ncmoe=%d\n", label, plan->ncmoe);
        printf("plan.%s.cpu_expert_gib=%.3f\n", label, (double) plan->cpu_expert_bytes / GiB);
    }
    printf("plan.%s.cpu_slots=%d\n", label, plan->cpu_slots);
    printf("plan.%s.layer_split=", label);
    print_layer_split(hw, plan);
    printf("\n");
    if (numa) {
        printf("plan.%s.cpu_tensor_split=", label);
        for (int i = 0; i < hw->cpu_count; ++i) {
            printf("%s1", i ? "/" : "");
        }
        printf("\n");
    }
    printf("plan.%s.devices=", label);
    for (int i = 0; i < hw->gpu_count; ++i) {
        printf("%s%s", i ? "," : "", hw->gpus[i].backend_name);
    }
    for (int i = 0; i < hw->cpu_count; ++i) {
        printf("%s%s", hw->gpu_count || i ? "," : "", hw->cpu_names[i]);
    }
    if (!hw->gpu_count && !explicit_cpu) {
        printf("none");
    }
    printf("\n");
    for (int i = 0; i < hw->gpu_count; ++i) {
        printf("plan.%s.gpu%d_est_gib=%.3f\n", label, i, (double) plan->gpu_bytes[i] / GiB);
        printf("plan.%s.gpu%d_budget_gib=%.3f\n", label, i, (double) gpu_budget_bytes(&hw->gpus[i], reserve_mib) / GiB);
    }
    const char * commands[] = { "llama-bench", "llama-cli" };
    const char * keys[] = { "benchmark", "cli" };
    for (int i = 0; i < 2; ++i) {
        printf("plan.%s.%s=%s", label, keys[i], commands[i]);
        if (numa) {
            printf(" --numa split");
        }
        printf(" -dev ");
        for (int gpu = 0; gpu < hw->gpu_count; ++gpu) {
            printf("%s%s", gpu ? (i == 0 ? "/" : ",") : "", hw->gpus[gpu].backend_name);
        }
        for (int cpu = 0; cpu < hw->cpu_count; ++cpu) {
            printf("%s%s", hw->gpu_count || cpu ? (i == 0 ? "/" : ",") : "", hw->cpu_names[cpu]);
        }
        if (!hw->gpu_count && !explicit_cpu) {
            printf("none");
        }
        printf(" -sm layer");
        if (ngl > 0 && (hw->gpu_count || explicit_cpu)) {
            printf(" --layer-split ");
            print_layer_split(hw, plan);
        }
        printf(" -t %d -b %u -ub %u", options->threads, options->batch, options->ubatch);
        if (i == 1) {
            printf(" -c %u --no-op-offload", options->context);
        } else {
            printf(" -nopo 1");
        }
        printf(" -ngl %d", ngl);
        if (is_moe) {
            printf(" -ncmoe %d", plan->ncmoe);
        }
        printf("\n");
    }
}

static void print_measured_plan(
        int is_moe, const struct hw_info * hw, const struct placement_plan * plan,
        const char * label, double reserve_mib, const struct planner_options * options) {
    print_plan(is_moe, hw, plan, label, reserve_mib, options);
    if (!options->measure || !plan->valid) {
        return;
    }
    const char * devices[MAX_GPUS + GGML_CPU_NUMA_SPLIT_MAX_DEVICES];
    float split[MAX_GPUS + GGML_CPU_NUMA_SPLIT_MAX_DEVICES] = {0};
    int count = 0;
    for (int i = 0; i < hw->gpu_count; ++i) {
        devices[count] = hw->gpus[i].backend_name;
        split[count++] = (float) plan->gpu_slots[i];
    }
    for (int i = 0; i < hw->cpu_count; ++i) {
        devices[count] = hw->cpu_names[i];
        split[count++] = i == 0 ? (float) plan->cpu_slots : 0.0f;
    }
    struct placement_plan measured = *plan;
    memset(measured.gpu_bytes, 0, sizeof(measured.gpu_bytes));
    const int ngl = hw->cpu_count ? plan->ngl : plan->ngl - plan->cpu_slots;
    int ok = placement_measure_memory(options->model_path, label, devices, count, split,
        ngl, plan->ncmoe, options->context, options->batch, options->ubatch,
        options->threads, measured.gpu_bytes, hw->gpu_count);
    for (int i = 0; i < hw->gpu_count; ++i) {
        if (measured.gpu_bytes[i] > gpu_budget_bytes(&hw->gpus[i], reserve_mib)) {
            measured.valid = 0;
        }
    }
    char measured_label[64];
    snprintf(measured_label, sizeof(measured_label), "llama_%s", label);
    printf("llama.%s.status=%s\n", label, !ok ? "error" : measured.valid ? "fits" : "no_fit");
    if (!ok) {
        measured.valid = 0;
    }
    print_plan(is_moe, hw, &measured, measured_label, reserve_mib, options);
}

static void plan_and_print(
        int64_t block_count,
        int64_t main_layers,
        int64_t expert_count,
        const uint64_t * layer_bytes,
        const uint64_t * expert_bytes,
        uint64_t nonlayer_bytes,
        const struct hw_info * hw,
        double reserve_mib,
        const struct planner_options * options) {
    printf("\nplanner.reserve_mib=%.0f\n", reserve_mib);
    printf("planner.split_mode=layer\n");
    printf("planner.memory_estimate=weights_plus_fixed_reserve\n");
    printf("planner.threads=%d\n", options->threads);
    printf("planner.context=%u\n", options->context);
    printf("planner.llama_memory_estimate=%s\n", options->measure ? "weights_cache_state_compute" : "disabled");
    printf("planner.llama_fit_scope=gpu_memory\n");
    if (hw->cpu_count >= 2) {
        printf("planner.numa=split\n");
    }
    // The runtime has a fixed backend limit, even when NUMA nodes form one Meta device.
    if (hw->gpu_count + hw->cpu_count > GGML_BACKEND_META_MAX_DEVICES - 1) {
        printf("planner.status=too_many_backend_devices\n");
        return;
    }
    const int is_moe = expert_count > 0;
    if (!is_moe || !hw->gpu_count) {
        struct placement_plan plan = plan_layers(hw, (int) block_count, (int) main_layers,
            layer_bytes, expert_bytes, nonlayer_bytes, is_moe ? (int) main_layers : -1, reserve_mib, 0);
        printf("planner.status=%s\n", plan.valid ? "ok" : "no_fit");
        print_measured_plan(is_moe, hw, &plan, "primary", reserve_mib, options);
        return;
    }
    struct placement_plan aggressive = {0};
    for (int ncmoe = 0; ncmoe <= main_layers; ++ncmoe) {
        aggressive = plan_layers(hw, (int) block_count, (int) main_layers,
            layer_bytes, expert_bytes, nonlayer_bytes, ncmoe, reserve_mib, 1);
        if (aggressive.valid) {
            break;
        }
    }
    if (!aggressive.valid) {
        aggressive = plan_layers(hw, (int) block_count, (int) main_layers,
            layer_bytes, expert_bytes, nonlayer_bytes, (int) main_layers, reserve_mib, 0);
    }
    printf("planner.status=%s\n", aggressive.valid ? "ok" : "no_fit");
    if (aggressive.valid && aggressive.ncmoe < main_layers) {
        struct placement_plan safe = aggressive;
        safe.ncmoe++;
        safe.cpu_expert_bytes += expert_bytes[aggressive.ncmoe];
        const int slot = hw->cpu_count == 0 && hw->gpu_count > 0 ? (int) block_count - aggressive.ncmoe : aggressive.ncmoe;
        int start = 0;
        for (int gpu = 0; gpu < hw->gpu_count; ++gpu) {
            const int index = hw->cpu_count == 0 ? hw->gpu_count - 1 - gpu : gpu;
            const int stop = start + safe.gpu_slots[index];
            if (slot >= start && slot < stop) {
                safe.gpu_bytes[index] -= expert_bytes[aggressive.ncmoe];
                break;
            }
            start = stop;
        }
        print_measured_plan(1, hw, &safe, "safe", reserve_mib, options);
    }
    print_measured_plan(1, hw, &aggressive, "aggressive", reserve_mib, options);
}

/* ------------------------------------------------------------------------- */
/* Main                                                                     */
/* ------------------------------------------------------------------------- */

int main(int argc, char ** argv) {
    int verbose = 0;
    int show_hardware = 1;
    int do_plan = 1;

    double reserve_mib = -1.0;
    struct planner_options options = {NULL, 2048, 128, 64, 0, 1};

    int first_file = 1;

    while (first_file < argc) {
        if (strcmp(argv[first_file], "-v") == 0 ||
            strcmp(argv[first_file], "--verbose") == 0) {
            verbose = 1;
            ++first_file;
            continue;
        }

        if (strcmp(argv[first_file], "--no-hardware") == 0) {
            show_hardware = 0;
            ++first_file;
            continue;
        }

        if (strcmp(argv[first_file], "--no-plan") == 0) {
            do_plan = 0;
            ++first_file;
            continue;
        }

        if (strcmp(argv[first_file], "--no-memory-estimate") == 0) {
            options.measure = 0;
            ++first_file;
            continue;
        }
        const char * option = argv[first_file];
        if (!strcmp(option, "-c") || !strcmp(option, "--ctx-size") ||
            !strcmp(option, "-t") || !strcmp(option, "--threads") ||
            !strcmp(option, "-b") || !strcmp(option, "--batch-size") ||
            !strcmp(option, "-ub") || !strcmp(option, "--ubatch-size")) {
            char * end = NULL;
            long value = first_file + 1 < argc ? strtol(argv[first_file + 1], &end, 10) : 0;
            if (!end || end == argv[first_file + 1] || *end || value <= 0 || value > INT32_MAX) {
                fprintf(stderr, "%s requires a positive integer\n", option);
                return 1;
            }
            if (!strcmp(option, "-c") || !strcmp(option, "--ctx-size")) options.context = (uint32_t) value;
            if (!strcmp(option, "-t") || !strcmp(option, "--threads")) options.threads = (int) value;
            if (!strcmp(option, "-b") || !strcmp(option, "--batch-size")) options.batch = (uint32_t) value;
            if (!strcmp(option, "-ub") || !strcmp(option, "--ubatch-size")) options.ubatch = (uint32_t) value;
            first_file += 2;
            continue;
        }

        if (strcmp(argv[first_file], "--reserve-mib") == 0) {
            if (first_file + 1 >= argc) {
                fprintf(stderr, "--reserve-mib requires a value\n");
                return 1;
            }

            char * end = NULL;
            reserve_mib = strtod(argv[first_file + 1], &end);
            if (end == argv[first_file + 1] || *end || !isfinite(reserve_mib) || reserve_mib < 0 || reserve_mib >= (double) UINT64_MAX / (1024.0 * 1024.0)) {
                fprintf(stderr, "invalid --reserve-mib value\n");
                return 1;
            }

            first_file += 2;
            continue;
        }


        break;
    }

    const int help = argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"));
    if (help || first_file >= argc) {
        fprintf(help ? stdout : stderr,
            "usage: %s [-v] [--no-hardware] [--no-plan]\n"
            "          [--reserve-mib N] [-c N] [-t N] [-b N] [-ub N]\n"
            "          [--no-memory-estimate]\n"
            "          model.gguf [model-shard2.gguf ...]\n",
            argv[0]);
        return help ? 0 : 1;
    }

    options.model_path = argv[first_file];
    if (!options.threads) options.threads = placement_physical_cores();
    if (options.threads < 1) options.threads = 1;
    const int shard_count = argc - first_file;

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = NULL,
    };

    struct gguf_context * first =
        gguf_init_from_file(argv[first_file], params);

    if (!first) {
        fprintf(stderr, "failed to open: %s\n", argv[first_file]);
        return 1;
    }

    const char * arch = get_str(first, "general.architecture");

    if (!arch) {
        fprintf(stderr, "general.architecture not found\n");
        gguf_free(first);
        return 1;
    }

    char arch_copy[128];
    snprintf(arch_copy, sizeof(arch_copy), "%s", arch);

    char key[256];

#define GET_ARCH_INT(var, def)                          \
    snprintf(key, sizeof(key), "%s.%s", arch, #var);   \
    int64_t var = get_int(first, key, def)

    GET_ARCH_INT(block_count,          -1);
    GET_ARCH_INT(context_length,       -1);
    GET_ARCH_INT(embedding_length,     -1);
    GET_ARCH_INT(expert_count,          0);
    GET_ARCH_INT(expert_used_count,     0);
    GET_ARCH_INT(nextn_predict_layers,  0);

    if (block_count <= 0 || block_count > 4096) {
        fprintf(stderr, "invalid block_count: %lld\n",
                (long long) block_count);
        gguf_free(first);
        return 1;
    }

    if (nextn_predict_layers < 0 ||
        nextn_predict_layers > block_count) {
        nextn_predict_layers = 0;
    }

    int64_t main_layers = block_count - nextn_predict_layers;

    if (main_layers <= 0) {
        fprintf(stderr, "invalid main layer count\n");
        gguf_free(first);
        return 1;
    }

    gguf_free(first);

    uint64_t * layer_bytes =
        calloc((size_t) block_count, sizeof(uint64_t));

    uint64_t * expert_bytes =
        calloc((size_t) block_count, sizeof(uint64_t));

    if (!layer_bytes || !expert_bytes) {
        fprintf(stderr, "out of memory\n");
        free(layer_bytes);
        free(expert_bytes);
        return 1;
    }

    uint64_t disk_bytes = 0;
    uint64_t tensor_bytes = 0;
    uint64_t nonlayer_bytes = 0;
    uint64_t routed_expert_bytes = 0;
    uint64_t skipped_mtp_bytes = 0;
    int64_t total_tensors = 0;

    for (int arg = first_file; arg < argc; ++arg) {
        struct stat st;

        if (stat(argv[arg], &st) == 0) {
            disk_bytes += (uint64_t) st.st_size;
        }

        struct gguf_context * ctx =
            gguf_init_from_file(argv[arg], params);

        if (!ctx) {
            fprintf(stderr, "failed to open shard: %s\n", argv[arg]);
            free(layer_bytes);
            free(expert_bytes);
            return 1;
        }

        int64_t nt = gguf_get_n_tensors(ctx);
        total_tensors += nt;

        for (int64_t i = 0; i < nt; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            uint64_t size =
                (uint64_t) gguf_get_tensor_size(ctx, i);

            tensor_bytes += size;

            int layer = tensor_layer(name);

            if (layer >= 0 && layer < block_count) {
                layer_bytes[layer] += size;
                if (layer >= main_layers) {
                    skipped_mtp_bytes += size;
                }

                if (is_routed_expert_tensor(name)) {
                    expert_bytes[layer] += size;
                    routed_expert_bytes += size;
                }
            } else {
                nonlayer_bytes += size;
            }
        }

        gguf_free(ctx);
    }

    uint64_t layer_total = 0;
    uint64_t shared_layer_total = 0;
    uint64_t min_layer = UINT64_MAX;
    uint64_t max_layer = 0;

    for (int64_t i = 0; i < block_count; ++i) {
        layer_total += layer_bytes[i];
        shared_layer_total += layer_bytes[i] - expert_bytes[i];

        if (layer_bytes[i] != 0 && layer_bytes[i] < min_layer) {
            min_layer = layer_bytes[i];
        }

        if (layer_bytes[i] > max_layer) {
            max_layer = layer_bytes[i];
        }
    }

    if (min_layer == UINT64_MAX) min_layer = 0;

    uint64_t main_layer_bytes = 0;
    uint64_t main_expert_bytes = 0;
    uint64_t main_shared_bytes = 0;

    for (int64_t i = 0; i < main_layers; ++i) {
        main_layer_bytes += layer_bytes[i];
        main_expert_bytes += expert_bytes[i];
        main_shared_bytes += layer_bytes[i] - expert_bytes[i];
    }

    const double GiB = 1024.0 * 1024.0 * 1024.0;

    printf("file=%s\n", argv[first_file]);
    printf("shards=%d\n", shard_count);
    printf("arch=%s\n", arch_copy);
    printf("model_type=%s\n", expert_count > 0 ? "moe" : "dense");
    printf("layers=%lld\n", (long long) block_count);
    printf("main_layers=%lld\n", (long long) main_layers);
    printf("nextn_layers=%lld\n", (long long) nextn_predict_layers);
    printf("skipped_mtp_gib=%.3f\n", (double) skipped_mtp_bytes / GiB);
    printf("context=%lld\n", (long long) context_length);
    printf("embedding=%lld\n", (long long) embedding_length);
    printf("experts=%lld\n", (long long) expert_count);
    printf("experts_used=%lld\n", (long long) expert_used_count);
    printf("tensors=%lld\n", (long long) total_tensors);
    printf("disk_gib=%.3f\n", (double) disk_bytes / GiB);
    printf("tensor_gib=%.3f\n", (double) tensor_bytes / GiB);
    printf("layer_gib=%.3f\n", (double) layer_total / GiB);
    printf("nonlayer_gib=%.3f\n", (double) nonlayer_bytes / GiB);
    printf("routed_expert_gib=%.3f\n",
           (double) routed_expert_bytes / GiB);
    printf("shared_layer_gib=%.3f\n",
           (double) shared_layer_total / GiB);
    printf("avg_layer_gib=%.3f\n",
           ((double) layer_total / (double) block_count) / GiB);
    printf("min_layer_gib=%.3f\n", (double) min_layer / GiB);
    printf("max_layer_gib=%.3f\n", (double) max_layer / GiB);
    printf("main_layer_gib=%.3f\n", (double) main_layer_bytes / GiB);
    printf("main_expert_gib=%.3f\n", (double) main_expert_bytes / GiB);
    printf("main_shared_gib=%.3f\n", (double) main_shared_bytes / GiB);
    printf("avg_main_layer_gib=%.3f\n",
           ((double) main_layer_bytes / (double) main_layers) / GiB);
    printf("avg_main_expert_layer_gib=%.3f\n",
           ((double) main_expert_bytes / (double) main_layers) / GiB);
    printf("avg_main_shared_layer_gib=%.3f\n",
           ((double) main_shared_bytes / (double) main_layers) / GiB);

    if (verbose) {
        printf("\n");

        for (int64_t i = 0; i < block_count; ++i) {
            uint64_t shared = layer_bytes[i] - expert_bytes[i];

            printf("layer.%lld total=%.3f expert=%.3f shared=%.3f%s\n",
                   (long long) i,
                   (double) layer_bytes[i] / GiB,
                   (double) expert_bytes[i] / GiB,
                   (double) shared / GiB,
                   i >= main_layers ? " extra=1" : "");
        }
    }

    struct hw_info hw;
    detect_hardware(&hw);

    if (show_hardware) {
        print_hardware(&hw);
    }

    /*
     * The MoE path needs a little more runtime headroom than dense
     * whole-layer offload on the hardware we have validated. Users
     * can override either default with --reserve-mib.
     */
    double effective_reserve_mib = reserve_mib >= 0.0
        ? reserve_mib
        : (expert_count > 0
            ? DEFAULT_MOE_VRAM_RESERVE_MIB
            : DEFAULT_DENSE_VRAM_RESERVE_MIB);

    if (do_plan) {
        plan_and_print(
            block_count,
            main_layers,
            expert_count,
            layer_bytes,
            expert_bytes,
            nonlayer_bytes,
            &hw,
            effective_reserve_mib, &options);
    }

    free(layer_bytes);
    free(expert_bytes);

    return 0;
}

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "numa.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

struct ggml_backend_cpu_device_context;

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device);

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    // per node repack buffer types are not in the global list, match them by kind
    return ggml_backend_cpu_buft_is_repack(buft);
}

// CPU backend - backend (stream)

static int ggml_backend_cpu_device_get_n_threads_max(ggml_backend_dev_t dev);
static ggml_threadpool_t ggml_backend_cpu_device_threadpool(ggml_backend_dev_t dev, int n_threads);

// graph_compute_async must consume the graph before it returns: the caller reuses the memory that
// holds the graph and its tensor metadata for the next graph while this one still runs. the tensor
// structs are therefore cloned here at hand-off. only the metadata is copied, the data pointers stay,
// writes to them are ordered by the synchronization the scheduler already does between backends
struct ggml_backend_cpu_graph_copy {
    std::deque<struct ggml_tensor>                                        tensors;
    std::vector<struct ggml_tensor *>                                     nodes;
    std::unordered_map<const struct ggml_tensor *, struct ggml_tensor *>  map;
    std::vector<int32_t>                                                  use_counts;
    struct ggml_hash_set                                                  hash  = {};
    struct ggml_cgraph                                                    graph = {};

    ~ggml_backend_cpu_graph_copy() {
        if (hash.size > 0) {
            ggml_hash_set_free(&hash);
        }
    }

    // a struct copy that keeps pointer identity between clones through the map, but does not follow the
    // tensor's own src pointers. that is enough for everything outside the node list: those are leafs,
    // weights and the outputs of earlier graphs, whose fields are read but whose links are not. following
    // the links would walk back through everything computed so far on every hand-off
    struct ggml_tensor * clone(const struct ggml_tensor * t) {
        if (t == NULL) {
            return NULL;
        }
        const auto it = map.find(t);
        if (it != map.end()) {
            return it->second;
        }
        tensors.push_back(*t);
        struct ggml_tensor * c = &tensors.back();
        map.emplace(t, c);
        return c;
    }

    // fills this with a self-owned copy of cgraph that stays valid while it is being computed
    void take(struct ggml_cgraph * cgraph) {
        nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            // in evaluation order, so a tensor cloned as a src here gets its links set when its own turn comes
            struct ggml_tensor * c = clone(cgraph->nodes[i]);
            c->view_src = clone(c->view_src);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                c->src[j] = clone(c->src[j]);
            }
            nodes.push_back(c);
        }

        // op fusion looks tensors up in the graph's hash set to check their use counts, so the copy
        // needs one over the clones. the counts come from the source graph: recounting inside this
        // graph alone would allow fusing over tensors that a later part of the split graph still reads
        hash = ggml_hash_set_new(2*map.size() + 1);
        use_counts.assign(hash.size, 0);

        const bool src_counts = cgraph->visited_hash_set.size > 0 && cgraph->use_counts != NULL;
        for (const auto & [orig, copy] : map) {
            const size_t pos = ggml_hash_insert(&hash, copy);
            if (src_counts) {
                const size_t src_pos = ggml_hash_find(&cgraph->visited_hash_set, orig);
                if (ggml_bitset_get(cgraph->visited_hash_set.used, src_pos)) {
                    use_counts[pos] = cgraph->use_counts[src_pos];
                }
            }
        }

        graph                  = {};
        graph.size             = cgraph->n_nodes;
        graph.n_nodes          = cgraph->n_nodes;
        graph.nodes            = nodes.data();
        graph.use_counts       = use_counts.data();
        graph.visited_hash_set = hash;
        graph.order            = cgraph->order;
        graph.uid              = cgraph->uid;
    }
};

// one unit of asynchronous work for the dispatcher, executed in FIFO order
struct ggml_backend_cpu_async_op {
    enum { COMPUTE } kind = COMPUTE;

    std::unique_ptr<ggml_backend_cpu_graph_copy> graph; // COMPUTE
};

// a NUMA node backend computes on its own dispatcher thread. the dispatcher is pinned to the node, so the
// OpenMP team and the thread pool it drives stay on the node instead of following the scheduler thread
// across sockets, and graph_compute becomes asynchronous, which lets several node backends overlap. the
// work queue holds any number of graphs, so a pipelining scheduler can keep several in flight.
//
// as with every asynchronous backend, freeing a buffer an in-flight graph still uses is the caller's
// responsibility (synchronize first); llama pre-reserves its compute buffers, so this does not come up there
struct ggml_backend_cpu_async {
    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;

    std::deque<ggml_backend_cpu_async_op> queue;
    bool                    running = false;       // an op popped off the queue is being executed
    enum ggml_status        status  = GGML_STATUS_SUCCESS; // of the last failed graph, kept until a caller sees it
    bool                    status_logged = false; // a pending failure was already reported by synchronize
    bool                    stop    = false;

    bool idle() const {
        return queue.empty() && !running;
    }
};

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    // owned by a NUMA node backend, it is pinned to the CPUs of that node
    ggml_threadpool_t   own_threadpool;
    int                 own_threadpool_size;

    // owned by a NUMA node backend, NULL means graph_compute runs synchronously on the calling thread
    struct ggml_backend_cpu_async * async;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation
};

static ggml_threadpool_t ggml_backend_cpu_threadpool(const struct ggml_backend_cpu_context * ctx) {
    return ctx->threadpool != NULL ? ctx->threadpool : ctx->own_threadpool;
}

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    if (cpu_ctx->async != NULL) {
        {
            std::lock_guard<std::mutex> lock(cpu_ctx->async->mutex);
            cpu_ctx->async->stop = true;
            cpu_ctx->async->cv.notify_all();
        }
        cpu_ctx->async->thread.join(); // drains the queued work first
        delete cpu_ctx->async;
    }
    if (cpu_ctx->own_threadpool != NULL) {
        ggml_threadpool_free(cpu_ctx->own_threadpool);
    }
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, ggml_backend_cpu_threadpool(cpu_ctx));
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute_impl(struct ggml_backend_cpu_context * cpu_ctx, struct ggml_cgraph * cgraph) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, ggml_backend_cpu_threadpool(cpu_ctx));

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static void ggml_backend_cpu_async_loop(struct ggml_backend_cpu_context * cpu_ctx, std::vector<int> cpus) {
    // before anything runs here, so that the threads serving this backend inherit the node
    ggml::cpu::numa::bind_current_thread(cpus);

    struct ggml_backend_cpu_async * a = cpu_ctx->async;
    for (;;) {
        struct ggml_backend_cpu_async_op op;
        {
            std::unique_lock<std::mutex> lock(a->mutex);
            a->cv.wait(lock, [a] { return !a->queue.empty() || a->stop; });
            // work that was handed over before the stop still runs, it must not be dropped silently
            if (a->queue.empty()) {
                return;
            }
            op = std::move(a->queue.front());
            a->queue.pop_front();
            a->running = true;
        }

        enum ggml_status status = GGML_STATUS_SUCCESS;
        switch (op.kind) {
            case ggml_backend_cpu_async_op::COMPUTE:
                status = ggml_backend_cpu_graph_compute_impl(cpu_ctx, &op.graph->graph);
                break;
        }

        {
            std::lock_guard<std::mutex> lock(a->mutex);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_LOG_ERROR("%s: graph computation failed with status %d\n", __func__, status);
            }
            if (a->status == GGML_STATUS_SUCCESS) {
                a->status = status; // kept until a caller sees it
            }
            a->running = false;
            a->cv.notify_all();
        }
    }
}

// a failure of an earlier asynchronous graph that no graph_compute call has consumed yet
static bool ggml_backend_cpu_async_pending_error(struct ggml_backend_cpu_context * cpu_ctx) {
    struct ggml_backend_cpu_async * a = cpu_ctx->async;
    if (a == NULL) {
        return false;
    }
    std::lock_guard<std::mutex> lock(a->mutex);
    return a->status != GGML_STATUS_SUCCESS;
}

static void ggml_backend_cpu_async_wait_idle(struct ggml_backend_cpu_context * cpu_ctx) {
    struct ggml_backend_cpu_async * a = cpu_ctx->async;
    if (a == NULL) {
        return;
    }
    std::unique_lock<std::mutex> lock(a->mutex);
    a->cv.wait(lock, [a] { return a->idle(); });
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_cpu_async * a = cpu_ctx->async;
    if (a == NULL) {
        return ggml_backend_cpu_graph_compute_impl(cpu_ctx, cgraph);
    }

    struct ggml_backend_cpu_async_op op;
    op.kind  = ggml_backend_cpu_async_op::COMPUTE;
    op.graph = std::make_unique<ggml_backend_cpu_graph_copy>();
    op.graph->take(cgraph);

    std::lock_guard<std::mutex> lock(a->mutex);

    // as with other asynchronous backends, a failure surfaces on the next call. the new graph
    // still runs: dropping it here would silently desynchronize a caller that continues after
    // the error
    const enum ggml_status status = a->status;
    a->status        = GGML_STATUS_SUCCESS;
    a->status_logged = false;

    a->queue.push_back(std::move(op));
    a->cv.notify_all();

    return status;
}

static void ggml_backend_cpu_synchronize(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_cpu_async * a = cpu_ctx->async;
    if (a == NULL) {
        return;
    }

    std::unique_lock<std::mutex> lock(a->mutex);
    a->cv.wait(lock, [a] { return a->idle(); });

    // synchronize cannot return a status, so a failed graph is reported here and stays pending
    // for the next graph_compute to return. GGML_STATUS_ABORTED is regular abort callback flow
    if (a->status != GGML_STATUS_SUCCESS && a->status != GGML_STATUS_ABORTED && !a->status_logged) {
        GGML_LOG_ERROR("%s: an asynchronous graph failed with status %d\n", __func__, a->status);
        a->status_logged = true;
    }
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_cpu_synchronize,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->own_threadpool      = NULL;
    ctx->own_threadpool_size = 0;
    ctx->async               = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    // a node backend cannot use more threads than its node has CPUs
    const int n_threads_max = ggml_backend_cpu_device_get_n_threads_max(backend_cpu->device);
    if (n_threads_max > 0) {
        n_threads = std::min(n_threads, n_threads_max);
    }

    // callers re-apply the thread count on every graph, and waiting for the queue on every
    // call would drain a pipelined scheduler; only an actual change has to wait
    if (n_threads == ctx->n_threads) {
        return;
    }

    ggml_backend_cpu_async_wait_idle(ctx);

    // the node pool starts at one thread per physical core; a larger explicit request (SMT
    // threads) recreates it at the requested size. it only ever grows: a pool larger than the
    // active thread count costs a little (idle workers spin the poll window once per graph
    // before sleeping), so the growth only happens when the user asks for the threads
    if (ctx->own_threadpool != NULL && n_threads > ctx->own_threadpool_size) {
        ggml_threadpool_t tp = ggml_backend_cpu_device_threadpool(backend_cpu->device, n_threads);
        if (tp != NULL) {
            ggml_threadpool_free(ctx->own_threadpool);
            ctx->own_threadpool      = tp;
            ctx->own_threadpool_size = n_threads;
        } else {
            GGML_LOG_WARN("%s: failed to grow the thread pool to %d threads\n", __func__, n_threads);
            n_threads = ctx->own_threadpool_size;
        }
    }

    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool == threadpool) {
        return;
    }

    ggml_backend_cpu_async_wait_idle(ctx);

    if (ctx->threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    ggml_backend_cpu_async_wait_idle(ctx);
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    ggml_backend_cpu_async_wait_idle(ctx);
    ctx->use_ref = use_ref;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    // set only for the per-node devices created by --numa split, see ggml_backend_cpu_numa_split_init
    int              numa_node = -1;
    std::string      name      = "CPU";
    std::string      device_id;
    std::vector<int> cpus;
    int              n_cores = 0;

    // the node local buffer type of this device, its name is the device name so that it stays unique
    ggml_backend_buffer_type buft = {};

    // NULL terminated extra buffer types of a node device (its own node local repack), see get_extra_buffers_type
    std::vector<ggml_backend_buffer_type_t> extra_bufts;

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

// CPU backend - NUMA node local buffer type

static const char * ggml_backend_cpu_numa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)buft->device->context;

    return ctx->name.c_str();
}

static void ggml_backend_cpu_numa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml::cpu::numa::free_onnode(buffer->context, buffer->size);
}

static ggml_backend_buffer_t ggml_backend_cpu_numa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)buft->device->context;

    std::string error;

    // the free + reclaimable estimate is imprecise and the kernel reclaims page cache as the pages
    // fault in, so exceeding it is only worth a warning, not a refusal
    {
        ggml::cpu::numa::node n;
        n.id = ctx->numa_node;
        ggml::cpu::numa::refresh_memory(n);
        if (n.mem_available > 0 && size > n.mem_available) {
            GGML_LOG_WARN("%s: %s has about %zu MiB available but %zu MiB are requested, faulting the memory in may fail\n",
                    __func__, ctx->name.c_str(), n.mem_available >> 20, size >> 20);
        }
    }

    void * data = ggml::cpu::numa::alloc_onnode(size, ctx->numa_node, error);
    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu MiB on %s: %s\n", __func__, size >> 20, ctx->name.c_str(), error.c_str());
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, size);
    if (buffer == NULL) {
        ggml::cpu::numa::free_onnode(data, size);
        return NULL;
    }

    buffer->buft              = buft;
    buffer->iface.free_buffer = ggml_backend_cpu_numa_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cpu_numa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_numa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static void ggml_backend_cpu_numa_buffer_type_init(ggml_backend_cpu_device_context * ctx, ggml_backend_dev_t dev) {
    ctx->buft = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_numa_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_numa_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_numa_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .is_host          = */ ggml_backend_cpu_numa_buffer_type_is_host,
        },
        /* .device  = */ dev,
        /* .context = */ NULL,
    };

    // a node device offers its own node local repack, and only that. AMX and KleidiAI stay on the plain device
    ctx->extra_bufts.clear();
    for (auto * global : ggml_backend_cpu_get_extra_buffer_types()) {
        if (ggml_backend_cpu_buft_is_repack(global)) {
            ctx->extra_bufts.push_back(ggml_backend_cpu_repack_buffer_type_for_device(dev));
        }
    }
    ctx->extra_bufts.push_back(NULL);
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)device->context;

    if (ctx->numa_node >= 0) {
        return ctx->extra_bufts.data();
    }

    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();
}

// CPU backend - device interface

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    if (ctx->numa_node >= 0) {
        ggml::cpu::numa::node n;
        n.id = ctx->numa_node;
        ggml::cpu::numa::refresh_memory(n);

        // as below, report all of the memory as free
        *total = n.mem_total;
        *free  = n.mem_total;
        return;
    }

#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    props->device_id   = ctx->numa_node < 0 ? nullptr : ctx->device_id.c_str();
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        // a node device computes on its own dispatcher thread
        /* .async                 = */ ctx->numa_node >= 0,
        /* .host_buffer           = */ false,
        // a node device must not map file pages, its buffers have to be bound to the node
        /* .buffer_from_host_ptr  = */ ctx->numa_node < 0,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

// a thread pool restricted to the CPUs of one node. it is created paused so that the affinity is applied
// to whichever thread ends up driving the graph, not to the thread that happens to build the backend
static ggml_threadpool_t ggml_backend_cpu_numa_threadpool(const ggml_backend_cpu_device_context * ctx, int n_threads) {
    struct ggml_threadpool_params tpp;

    // default is one thread per physical core: extra threads would spend the poll window spinning
    // on the SMT siblings of the compute threads. an explicit larger thread count grows the pool,
    // see ggml_backend_cpu_set_n_threads
    if (n_threads <= 0) {
        n_threads = ctx->n_cores > 0 ? ctx->n_cores : (int) ctx->cpus.size();
    }
    ggml_threadpool_params_init(&tpp, n_threads);

    int n_masked = 0;
    for (int cpu : ctx->cpus) {
        if (cpu < GGML_MAX_N_THREADS) {
            tpp.cpumask[cpu] = true;
            n_masked++;
        }
    }
    if (n_masked < (int) ctx->cpus.size()) {
        GGML_LOG_WARN("%s: %zu CPUs of %s are above the supported maximum of %d and cannot be pinned\n",
                __func__, ctx->cpus.size() - n_masked, ctx->name.c_str(), GGML_MAX_N_THREADS);
        if (n_masked == 0) {
            // an empty mask would mean "no affinity", i.e. a pool that silently floats off its node
            return NULL;
        }
    }

    // the pools are pinned to disjoint CPU sets, so an idle pool does not take cores from a busy one
    tpp.paused = true;

    // one worker per CPU while the pool fits on distinct physical cores: floating workers can be
    // co-scheduled onto SMT siblings while a core idles, which straggles every barrier. once the
    // pool spills onto SMT siblings anyway, the fixed assignment loads the cores unevenly and
    // letting the scheduler balance within the node is slightly better
    tpp.strict_cpu = n_threads <= ctx->n_cores;

    return ggml_threadpool_new(&tpp);
}

static ggml_threadpool_t ggml_backend_cpu_device_threadpool(ggml_backend_dev_t dev, int n_threads) {
    const struct ggml_backend_cpu_device_context * dev_ctx = (const struct ggml_backend_cpu_device_context *)dev->context;
    if (dev_ctx->numa_node < 0) {
        return NULL;
    }
    return ggml_backend_cpu_numa_threadpool(dev_ctx, n_threads);
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    struct ggml_backend_cpu_device_context * dev_ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == NULL) {
        return NULL;
    }

    backend->device = dev;

    if (dev_ctx->numa_node >= 0) {
        struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend->context;

        ctx->own_threadpool      = ggml_backend_cpu_numa_threadpool(dev_ctx, 0);
        ctx->own_threadpool_size = dev_ctx->n_cores;
        if (ctx->own_threadpool == NULL) {
            GGML_LOG_ERROR("%s: failed to create the thread pool of %s\n", __func__, dev_ctx->name.c_str());
            ggml_backend_free(backend);
            return NULL;
        }

        ctx->n_threads = dev_ctx->n_cores;

        ctx->async = new ggml_backend_cpu_async;
        try {
            ctx->async->thread = std::thread(ggml_backend_cpu_async_loop, ctx, dev_ctx->cpus);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: failed to start the dispatcher thread of %s: %s\n", __func__, dev_ctx->name.c_str(), e.what());
            delete ctx->async;
            ctx->async = NULL;
            ggml_backend_free(backend);
            return NULL;
        }
    }

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    if (ctx->numa_node >= 0) {
        return &ctx->buft;
    }

    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static int ggml_backend_cpu_device_get_numa_node(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->numa_node;
}

static int ggml_backend_cpu_device_get_n_threads_max(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    // only a node device is limited to a subset of the machine
    return ctx->numa_node < 0 ? 0 : (int) ctx->cpus.size();
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    if (ctx->numa_node >= 0) {
        // a node device only takes its own buffers, otherwise the scheduler would give it the tensors of
        // another node, which it would then compute on remote memory
        if (buft == ggml_backend_dev_buffer_type(dev)) {
            return true;
        }
        for (auto * extra : ctx->extra_bufts) {
            if (extra == buft) {
                return true;
            }
        }
        return false;
    }

    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// --numa split: expose every usable NUMA node as its own device.
// See the contract next to ggml_numa_split_status in ggml-cpu.h. Nothing externally visible is
// created unless every node passes, so a failure leaves everything untouched and a caller that
// recovers from it never sees a half built configuration.
static enum ggml_numa_split_status ggml_backend_cpu_numa_split_init(ggml_backend_dev_t * devices, size_t * n_devices) {
    static std::mutex mutex;
    static int        status = -1; // latched ggml_numa_split_status after the first call

    std::lock_guard<std::mutex> lock(mutex);

    const size_t capacity = *n_devices;
    *n_devices = 0;

    if (status >= 0) {
        GGML_LOG_INFO("%s: NUMA already initialized\n", __func__);
        return (enum ggml_numa_split_status) status;
    }

    std::vector<ggml::cpu::numa::node> nodes = ggml::cpu::numa::topology();

    if (nodes.size() < 2) {
        GGML_LOG_WARN("%s: --numa split needs 2 or more usable NUMA nodes (found %zu), continuing without NUMA optimizations\n",
                __func__, nodes.size());
        status = GGML_NUMA_SPLIT_STATUS_UNAVAILABLE;
        return (enum ggml_numa_split_status) status;
    }

    // a machine with more nodes than the backend scheduler can hold is still usable with the
    // first nodes; the cap is never silent
    const size_t max_devices = std::min(capacity, (size_t) GGML_CPU_NUMA_SPLIT_MAX_DEVICES);
    if (nodes.size() > max_devices) {
        GGML_LOG_WARN("%s: found %zu NUMA nodes but at most %zu devices are supported, using the first %zu nodes\n",
                __func__, nodes.size(), max_devices, max_devices);
        nodes.resize(max_devices);
    }

    // phase 1: build and check every node, without registering anything
    std::vector<std::unique_ptr<ggml_backend_cpu_device_context>> ctxs;
    std::vector<std::unique_ptr<ggml_backend_device>>             devs;

    for (const auto & node : nodes) {
        auto ctx = std::make_unique<ggml_backend_cpu_device_context>();

        ctx->numa_node   = node.id;
        ctx->name        = "CPU" + std::to_string(node.id);
        ctx->device_id   = "numa:" + std::to_string(node.id);
        ctx->cpus        = node.cpus;
        ctx->n_cores     = node.n_cores;
        ctx->description = ctx->description + " (NUMA node " + std::to_string(node.id) + ")";

        auto dev = std::make_unique<ggml_backend_device>();

        dev->iface   = ggml_backend_cpu_device_i;
        dev->reg     = ggml_backend_cpu_reg();
        dev->context = ctx.get();

        ggml_backend_cpu_numa_buffer_type_init(ctx.get(), dev.get());

        // prove that this node can really hand out node local memory before anything depends on it
        ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(&ctx->buft, 2u << 20);
        if (probe == NULL) {
            GGML_LOG_ERROR("%s: node %d cannot provide node local memory, --numa split is not available\n",
                    __func__, node.id);
            // the devices built so far are about to be destroyed; drop the repack buffer types
            // that were created for them, they would keep dangling device pointers otherwise
            ggml_backend_cpu_repack_buffer_type_forget_device(dev.get());
            for (const auto & d : devs) {
                ggml_backend_cpu_repack_buffer_type_forget_device(d.get());
            }
            status = GGML_NUMA_SPLIT_STATUS_FAILED;
            return (enum ggml_numa_split_status) status;
        }
        ggml_backend_buffer_free(probe);

        GGML_LOG_INFO("%s: node %d: %zu CPUs, %d cores, %zu MiB\n",
                __func__, node.id, node.cpus.size(), node.n_cores, node.mem_total >> 20);

        ctxs.push_back(std::move(ctx));
        devs.push_back(std::move(dev));
    }

    // phase 2: commit, the caller registers the devices
    static std::vector<std::unique_ptr<ggml_backend_cpu_device_context>> ctxs_created;
    static std::vector<std::unique_ptr<ggml_backend_device>>             devs_created;

    std::string names;
    for (size_t i = 0; i < devs.size(); i++) {
        devices[i] = devs[i].get();

        names += names.empty() ? "" : ", ";
        names += ctxs[i]->name;

        ctxs_created.push_back(std::move(ctxs[i]));
        devs_created.push_back(std::move(devs[i]));
    }
    *n_devices = devs_created.size();

    GGML_LOG_INFO("%s: devices: %s\n", __func__, names.c_str());

    status = GGML_NUMA_SPLIT_STATUS_SUCCESS;
    return (enum ggml_numa_split_status) status;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// CPU backend - native allreduce between NUMA node backends
//
// The Meta backend reduces the partial results of every tensor-parallel boundary, preferring a
// backend-native allreduce (this) over its generic fallback, which builds the reduction out of
// cross-backend copies and ADD ops and pays several dispatcher round trips per boundary. CPU node
// backends share one address space, so small tensors are cheaper to reduce right here on the
// calling thread: wait for the partials, one pass over the data, done. Batch-1 decode reduces a
// few tens of KiB per boundary, where the fallback's fixed dispatch cost dominates the token time.

// prompt-sized boundary tensors are reduced faster by the fallback, whose ADDs run on the node
// thread pools; a single thread only wins while the data is small
static const size_t GGML_CPU_COMM_MAX_BYTES = 1024*1024;

struct ggml_backend_cpu_comm {
    std::vector<ggml_backend_t> backends;
};

static void * ggml_backend_cpu_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2) {
        return NULL;
    }
    for (size_t j = 0; j < n_backends; j++) {
        // only between NUMA node backends, whose buffers are all in one address space
        if (!ggml_backend_is_cpu(backends[j]) || ggml_backend_cpu_device_get_numa_node(backends[j]->device) < 0) {
            return NULL;
        }
    }

    struct ggml_backend_cpu_comm * comm = new ggml_backend_cpu_comm;
    comm->backends.assign(backends, backends + n_backends);
    return comm;
}

static bool ggml_backend_cpu_comm_allreduce_tensor(void * comm_ctx, struct ggml_tensor ** tensors) {
    struct ggml_backend_cpu_comm * comm = (struct ggml_backend_cpu_comm *)comm_ctx;
    const size_t n_backends = comm->backends.size();

    const size_t nbytes = ggml_nbytes(tensors[0]);
    if (nbytes > GGML_CPU_COMM_MAX_BYTES) {
        return false;
    }
    for (size_t j = 0; j < n_backends; j++) {
        if (tensors[j]->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensors[j]) || ggml_nbytes(tensors[j]) != nbytes) {
            return false;
        }
    }

    // the partials were dispatched to the backends just before this call
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_synchronize(comm->backends[j]);
    }

    // if a backend failed, its partial is garbage and the computation is already lost; skip the
    // reduction and let the failure surface from that backend's next graph_compute call
    for (size_t j = 0; j < n_backends; j++) {
        if (ggml_backend_cpu_async_pending_error((struct ggml_backend_cpu_context *)comm->backends[j]->context)) {
            return true;
        }
    }

    // sum the partials of the backends that computed one (a backend whose slice of the producing
    // matmul was empty leaves garbage in its tensor), then give every backend the sum
    const int64_t ne  = ggml_nelements(tensors[0]);
    float *       acc = NULL;
    for (size_t j = 0; j < n_backends; j++) {
        if ((tensors[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        float * data = (float *)tensors[j]->data;
        if (acc == NULL) {
            acc = data;
        } else {
            for (int64_t i = 0; i < ne; i++) {
                acc[i] += data[i];
            }
        }
    }
    for (size_t j = 0; j < n_backends; j++) {
        float * data = (float *)tensors[j]->data;
        if (data == acc) {
            continue;
        }
        if (acc != NULL) {
            memcpy(data, acc, nbytes);
        } else {
            memset(data, 0, nbytes);
        }
    }

    return true;
}

static void ggml_backend_cpu_comm_free(void * comm_ctx) {
    delete (struct ggml_backend_cpu_comm *)comm_ctx;
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_split_init") == 0) {
        return (void *)ggml_backend_cpu_numa_split_init;
    }
    if (strcmp(name, "ggml_backend_dev_get_numa_node") == 0) {
        return (void *)ggml_backend_cpu_device_get_numa_node;
    }
    if (strcmp(name, "ggml_backend_dev_get_n_threads_max") == 0) {
        return (void *)ggml_backend_cpu_device_get_n_threads_max;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        ggml_backend_comm_init_t fct = ggml_backend_cpu_comm_init;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ggml_backend_comm_allreduce_tensor_t fct = ggml_backend_cpu_comm_allreduce_tensor;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        ggml_backend_comm_free_t fct = ggml_backend_cpu_comm_free;
        return (void *)fct;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)

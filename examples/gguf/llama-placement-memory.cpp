#include "llama-placement-memory.h"

#include "common.h"
#include "fit.h"
#include "ggml-backend.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

int placement_physical_cores(void) {
    return common_cpu_get_num_physical_cores();
}

int placement_measure_memory(const char * path, const char * label,
        const char * const * devices, int n_devices, const float * layer_split,
        int ngl, int ncmoe, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
        int threads, uint64_t * gpu_bytes, int n_gpus) {
    ggml_log_callback original_logger;
    void * original_data;
    llama_log_get(&original_logger, &original_data);
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }, nullptr);
    try {
        std::vector<ggml_backend_dev_t> selected;
        for (int i = 0; i < n_devices; ++i) {
            auto dev = ggml_backend_dev_by_name(devices[i]);
            if (!dev) {
                throw std::runtime_error(std::string("device not found: ") + devices[i]);
            }
            selected.push_back(dev);
        }
        selected.push_back(nullptr);
        std::vector<std::string> patterns;
        for (int i = 0; i < ncmoe; ++i) {
            patterns.push_back(llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX));
        }
        std::vector<llama_model_tensor_buft_override> overrides;
        for (const auto & pattern : patterns) {
            overrides.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        }
        overrides.push_back({nullptr, nullptr});
        auto mparams = llama_model_default_params();
        mparams.devices = selected.data();
        mparams.n_gpu_layers = ngl;
        mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
        std::vector<float> shares(llama_max_devices(), 0.0f);
        for (int i = 0; i < n_devices; ++i) {
            shares[i] = layer_split[i];
        }
        mparams.layer_split = n_devices > 0 ? shares.data() : nullptr;
        mparams.tensor_buft_overrides = overrides.data();
        auto cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_ubatch;
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
        cparams.op_offload = false;
        cparams.swa_full = false;
        std::vector<ggml_backend_dev_t> measured_devices;
        uint32_t layers = 0, context = 0, experts = 0;
        auto memory = common_get_device_memory_data(path, &mparams, &cparams,
            measured_devices, layers, context, experts, GGML_LOG_LEVEL_ERROR);
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        for (size_t i = 0; i < memory.size(); ++i) {
            const char * name = i < measured_devices.size()
                ? ggml_backend_dev_name(measured_devices[i]) : "Host";
            const auto & mb = memory[i];
            printf("llama.%s.%s.model_gib=%.6f\n", label, name, mb.model / GiB);
            printf("llama.%s.%s.cache_state_gib=%.6f\n", label, name, mb.context / GiB);
            printf("llama.%s.%s.compute_gib=%.6f\n", label, name, mb.compute / GiB);
            for (int gpu = 0; gpu < n_gpus; ++gpu) {
                if (std::string(name) == devices[gpu]) {
                    gpu_bytes[gpu] = mb.model + mb.context + mb.compute;
                }
            }
        }
        llama_log_set(original_logger, original_data);
        return 1;
    } catch (const std::exception & e) {
        llama_log_set(original_logger, original_data);
        fprintf(stderr, "memory estimate for %s failed: %s\n", label, e.what());
        return 0;
    }
}

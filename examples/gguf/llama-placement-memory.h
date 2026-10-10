#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int placement_physical_cores(void);
int placement_measure_memory(const char * path, const char * label,
    const char * const * devices, int n_devices, const float * layer_split,
    int ngl, int ncmoe, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
    int threads, uint64_t * gpu_bytes, int n_gpus);

#ifdef __cplusplus
}
#endif

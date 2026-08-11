#pragma once

#include "ggml.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

// Per-thread scratch (in floats, excluding cache-line padding) for the split-KV
// flash-attention decode path: converted Q for all heads, one score tile per
// head, F32 VKQ accumulators, running max/sum per head, one converted V row and
// a converted mask tile.
#define GGML_FA_DECODE_SCRATCH_F32(n_head, DK, DV) \
    ((n_head)*((DK) + GGML_FA_TILE_KV + (DV) + 2) + (DV) + GGML_FA_TILE_KV)

#ifdef __cplusplus

#include <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float i32_to_f32(int32_t x) {
    return x;
}

static inline int32_t f32_to_i32(float x) {
    return x;
}

static inline float f32_to_f32(float x) {
    return x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}

// Split an (nr rows x nc columns) op over the threads. With at least as many
// rows as threads each thread gets a range of whole rows, like
// get_thread_range. With fewer rows than threads the rows are shared instead:
// every thread visits all rows but only a slice of the columns, so an op with
// a single long row still uses the whole team. Slices are kept at least
// min_cols wide so that a small op is not scattered over many threads.
struct ggml_thread_tile {
    int64_t ir0, ir1; // row range
    int64_t ic0, ic1; // column range
};

static inline ggml_thread_tile get_thread_tile(const struct ggml_compute_params * params,
        int64_t nr, int64_t nc, int64_t min_cols = 1024) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    if (nr >= nth || nc < 2*min_cols) {
        const int64_t dr  = (nr + nth - 1)/nth;
        const int64_t ir0 = MIN(dr*ith, nr);
        return {ir0, MIN(ir0 + dr, nr), 0, nc};
    }

    // keep the slice starts aligned to the widest SIMD step, so that the
    // per-element code path (vector body vs scalar tail) and therefore the
    // result of a vectorized op match the unsliced computation exactly
    const int64_t dc  = GGML_PAD(MAX((nc + nth - 1)/nth, min_cols), 64);
    const int64_t ic0 = MIN(dc*ith, nc);
    return {0, nr, ic0, MIN(ic0 + dc, nc)};
}

struct ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif

# NUMA placement and planner release

This release helps run large GGUF models on machines with multiple CPU sockets and limited GPU memory. It provides placement recommendations, separates GPU/CPU layer placement from tensor splitting inside CPU NUMA devices, and fixes Meta backend failures found during large-model testing.

This is primarily for multi-socket NUMA machines running CPU-only inference or offloading only part of a model to GPUs. The largest benefits in our tests came from larger models with substantial CPU-resident work, especially MoE models whose expert weights remain on CPU. As models outgrow GPU memory and leave more layers or expert tensors on CPU, NUMA-aware placement and CPU tensor splitting have more opportunity to improve performance relative to vanilla. Models that fit mostly on GPUs showed much smaller gains. The benefit depends on architecture and hardware; it does not necessarily increase with every increase in model size or CPU offload.

The intended workflow is to run the planner, choose a memory-validated candidate, add the model path to its command fragment, and run inference. Users do not need to benchmark every thread count before starting.

## Build and use

Build from a clean checkout using CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-cli llama-bench llama-placement -j 8
```

Omit CUDA support on machines without a supported CUDA installation. Build documentation covers other backends and platforms.

Run the planner with the context you intend to use. For a split model, supply all shards:

```sh
./build/bin/llama-placement -c 2048 /models/example/*.gguf
```

The planner prints hardware information, candidate placements, memory estimates, and command fragments. Add `-m /path/to/model.gguf` to a fragment; use the first shard as the model path for inference.

## Placement controls

- `--layer-split` controls layer proportions in selected device order. In layer mode, selected CPU NUMA devices are combined into one CPU Meta placement target after the GPUs.
- `--cpu-tensor-split` controls tensor proportions inside CPU Meta, independently of layer proportions. Equal shares are the default.
- `--numa split` and the explicit device list are included in planner commands when CPU Meta is used.
- `--n-cpu-moe` / `-ncmoe` keeps routed expert weights from the first N layers on CPU. Attention and other shared weights can remain on GPUs.

For example, with `CUDA0,CUDA1,CPU0,CPU1`, a layer split of `31/31/0` assigns layer slots to the two GPUs, while CPU Meta can still hold expert tensors selected by `-ncmoe`. The two CPU entries are one layer-placement target, not two extra layer shares. The output layer counts as a slot.

`--tensor-split` retains its legacy layer-mode meaning. Use either it or `--layer-split` for layer placement, not both. Explicit layer shares require layer mode. CPU tensor shares must be positive and finite.

## Planner settings and memory estimates

| Input | Default | Purpose |
|---|---|---|
| `-t`, `--threads` | Physical core count | Generation thread recommendation |
| `-c`, `--ctx-size` | 2048 | Requested CLI context |
| `-b`, `--batch-size` | 128 | Logical batch size |
| `-ub`, `--ubatch-size` | 64 | Physical batch size |
| `--reserve-mib` | 896 for MoE, 768 for dense | GPU headroom |
| `--no-memory-estimate` | Disabled | Skip llama allocation simulation |

Physical core count is a practical generation default, not a measured optimum on every machine. It recommends `-t 20` on the tested dual-socket server. Prefill can benefit from a different thread count; this release does not tune `-tb` automatically.

There are two sets of outputs:

1. Existing `plan.safe`, `plan.aggressive`, and `plan.primary` candidates retain the weight-based calculation and fixed GPU reserve.
2. `llama.<candidate>` measures those candidates using llama's existing allocation simulation. `plan.llama_<candidate>` includes the same placement when its simulated GPU memory fits after the reserve; otherwise it reports `valid=0`.

The simulation reads model metadata and constructs a context without loading tensor weights or running inference. It accounts for model buffers, KV caches, recurrent state, and compute buffers through the runtime's own architecture-specific logic. It therefore supports the cache layouts used by ordinary attention, MLA, and hybrid/recurrent models without duplicating their sizing formulas in the planner.

Prefer a valid `plan.llama_<candidate>` for the requested settings. If none is valid, try a smaller context or different placement. The planner does not silently reduce context and does not yet search for a replacement placement after simulation rejects a candidate. Host and CPU Meta memory are reported, but the fit decision checks GPU budgets; sufficient system RAM is still required.

CLI fragments include context, batch sizes, threads, and disabled operation offload. Benchmark fragments include batch sizes and threads; llama-bench derives context from its benchmark workload. Llama can pad the requested context internally. Cache types and other runtime settings remain at their defaults unless changed separately; changing them can invalidate the estimates.

## Correctness fixes

- Meta subgraphs are allocated using retained maximum graph capacity, preventing invalid graph/hash-set state when graph sizes change.
- The final subgraph is reduced when its output is PARTIAL. MIRRORED outputs are left alone to avoid double-counting replicated values.
- For supported non-DeepSeek-V4 MLA models, CPU Meta mirrors attention tensors and the shared latent cache. FFN and expert tensors retain their split rules. This fixes the incompatible cache-write and attention layouts encountered with Kimi K2 CPU Meta.
- Skipped MTP expert tensors are handled without creating unused expert weights. MTP tensors are not needed for ordinary generation.

The planner changes where computation runs, not the model's weights or quantization. CPU Meta divides supported tensor operations across NUMA devices and combines their results, which changes the order of floating-point additions. Floating-point arithmetic has finite precision, so adding the same values in a different order can produce slightly different results. Quantized execution can amplify a small difference when a value crosses a rounding boundary. If two next-token scores are close, their ranking can switch, even with greedy sampling. Once a different token is selected, the model continues from a different input and the wording can diverge substantially. Different wording alone therefore does not establish a correctness problem.

In the Kimi investigation, a tiny activation-scale change crossed a quantization rounding boundary and changed the next-token ranking. Isolated CPU replay reproduced the discrepancy without Meta execution, and the inspected mirrored cache replicas agreed exactly. This explains the investigated divergence; it is not a correctness guarantee for every model or prompt. Crashes, invalid values, and persistent runaway repetition still require investigation.

## Measured results

The test machine has two Xeon E5-2670 v2 CPUs, 20 physical cores in total, approximately 384 GB installed DDR3 RAM, and two Tesla P4 GPUs with 8 GB each. Models were stored on NFS. Thermal throttling and uncontrolled cache state affected early tests; those results were excluded from the final comparisons.

The following completed Qwen3.5-397B-A17B Q6_K runs used `-t 20`, the same prompt/generation workloads, three repetitions, warmup enabled, and cache clearing before each run. Weight loading was kept outside the inference throughput measurement.

| Configuration | Prefill tokens/s | Generation tokens/s |
|---|---:|---:|
| Vanilla CPU distribute | 8.62 | 1.57 |
| Vanilla CPU isolate | 5.59 | 1.48 |
| Fork CPU split | 9.19 | 2.78 |
| Vanilla GPU defaults | 4.99 | 2.62 |
| Vanilla GPU planner + distribute | 5.36 | 2.42 |
| Vanilla GPU planner + isolate | 9.36 | 2.67 |
| Fork GPU planner | 13.72 | 4.18 |

For this model, fork CPU-only generation improved by 77% over the best tested vanilla CPU configuration. Fork CPU/GPU generation improved by 56% over the best tested vanilla GPU configuration and 60% over vanilla default placement. GPU prefill improved by 47% over the best tested vanilla GPU configuration.

The compared implementations use different upstream revisions: vanilla `fc07d781e` and fork base `431b74ac7` with local changes. These results compare the tested implementations as a whole; they do not isolate the contribution of each patch. They are specific to this hardware, model quantization, thread count, and workload, not a universal speedup claim.

Allocation-only checks covered 12 model files/variants and 22 candidate placements matched to existing benchmark runs. GPU model and cache/state estimates agreed with recorded allocations within their printed precision, about 0.005 MiB. Compute differences were below 2 MiB and within the configured reserve. These checks reused existing results rather than repeating inference benchmarks.

The context-aware checks reject the existing DeepSeek candidates at context 2048. Qwen235 Instruct and Thinking aggressive candidates also exceed the reserved budget at that context, while their safe candidates fit. A weight-only candidate being valid does not mean its cache and execution buffers fit.

## Release validation and remaining limits

The final optimized release build completed with CUDA, native CPU support, and LTO. Library paths were checked, the rebuilt planner passed its Qwen397 allocation check, and the CUDA0 ADD correctness check passed 99 cases. Earlier model response checks completed without runaway repetition. The final release Kimi CPU Meta response check also completed successfully, produced the requested two sentences without runaway repetition, and exited normally.

This release focuses on layer-mode GPU placement with CPU Meta tensor splitting. Preserving CPU Meta placement alongside GPU tensor splitting is deferred. Testing was performed on a dual-socket machine; higher socket counts and other hardware need independent validation. Keep Flash Attention at its automatic default: forcing it off triggered a KV-write assertion in a tested Mistral CPU Meta configuration.

Benchmark logs and diagnostic probes remain in the development checkout. They are not required to build or use this release.

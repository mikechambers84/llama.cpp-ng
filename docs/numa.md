# NUMA

On multi-socket systems (and some single-socket systems with multiple memory
controllers), memory is divided into NUMA nodes: each CPU accesses its own
node's memory at full speed and the other nodes' memory more slowly. A CPU
inference run that ignores this leaves part of the machine's memory bandwidth
unused.

`llama.cpp` has several NUMA strategies, selected with `--numa TYPE`:

- `distribute`: spread execution evenly over all nodes
- `isolate`: only spawn threads on CPUs of the node that execution started on
- `numactl`: use the CPU map provided by numactl
- `split`: expose every NUMA node as its own device (see below)

## --numa split

`--numa split` (Linux only) probes the NUMA topology at startup and registers
one backend device per node: `CPU0`, `CPU1`, ... Each device:

- allocates its buffers from its own node's memory (verified with `mbind`),
  including its own node-local copy of repacked weights
- computes with a thread pool pinned to its own node's CPUs
- runs asynchronously on its own dispatcher thread, so the nodes can compute
  concurrently

The node devices participate in the regular multi-device machinery, so the
same flags that place a model across GPUs work across NUMA nodes:

```sh
# split the layers across the nodes (each node holds a contiguous range)
llama-server -m model.gguf --numa split -sm layer

# split every weight matrix across the nodes (all nodes work on every op)
llama-server -m model.gguf --numa split -sm tensor

# adjust the per-node fraction, restrict the devices, or pin tensors manually
llama-cli -m model.gguf --numa split -sm layer -ts 60,40
llama-cli -m model.gguf --numa split -dev CPU1
llama-cli -m model.gguf --numa split -ot "blk\.[0-9]\.=CPU0"
llama-cli --numa split --list-devices
```

`-sm row` is not supported for the CPU node devices.

### Split modes

- `-sm tensor` shards every weight matrix across the nodes, so all nodes work
  on every operation and contribute their memory bandwidth to it. The partial
  results are reduced between the nodes after each sharded operation.
- `-sm layer` keeps whole layers on one node. A single generation stream
  executes one layer at a time and therefore runs on one node at a time.
  Batches are processed as a pipeline of micro-batches, with different nodes
  working on different micro-batches concurrently; a prompt only fills the
  pipeline when it is split into more than one micro-batch, so `-ub` values
  smaller than the typical prompt length allow shorter prompts to use the
  pipeline, while smaller micro-batches also do less work per operation.
  Which effect wins depends on the model and the prompt length.

Which mode is faster depends on the model, the batch sizes and the machine;
benchmark both on your own workload.

### Threads

With `--numa split`, `-t` is the total thread budget: it is divided over the
node devices in proportion to their core counts (for example, `-t 48` on two
24-core nodes gives each node 24 threads). Each device's pool is pinned to
its node and defaults to one thread per physical core.

`-t` and `-tb` set the thread counts for generation and batch processing
separately. The best values depend on the model; thread counts well below
the total core count can be worth trying for generation.

### Notes

- The model is loaded without `mmap` into node-bound memory. On a box whose
  page cache is full of the model file, the loader counts reclaimable cache
  as available; if a node still looks too small it warns and proceeds rather
  than refusing.
- `-sm tensor` changes the reduction order of floating point sums between the
  nodes, so logits differ from a single-device run by the usual run-to-run
  float noise; `-sm layer` output is bit-identical to the plain CPU backend.
- `--numa split` only affects CPU inference. It can be combined with GPU
  offload in principle (the node devices behave like any other devices), but
  the main use case is CPU-only inference on multi-socket servers.

# GPU Kernel and Inference Performance Engineering

> GPU kernel and inference performance engineering is the discipline of squeezing maximum throughput and minimum latency out of AI models by optimizing how computation maps onto GPU hardware. As model-serving costs dominate AI budgets in 2026, kernel-level optimization has become one of the highest-leverage and most in-demand skills in the industry.

## Table of Contents

- [Why This Discipline Matters in 2026](#why-this-discipline-matters-in-2026)
- [The GPU Execution Model](#the-gpu-execution-model)
- [Where Performance Is Won and Lost](#where-performance-is-won-and-lost)
- [The Optimization Stack](#the-optimization-stack)
- [Key Metrics](#key-metrics)
- [Roofline Model and Compute vs Memory Bound](#roofline-model-and-compute-vs-memory-bound)
- [Career and Demand Signal](#career-and-demand-signal)
- [Cross-References](#cross-references)

---

## Why This Discipline Matters in 2026

The economics of AI have shifted. Training a frontier model is a one-time (or few-times) cost; **serving** it runs 24/7 across millions of requests. By 2026, inference compute vastly outweighs training compute in aggregate cloud spend. A 20% kernel speedup on a fleet of thousands of GPUs translates directly into millions of dollars saved and lower user-facing latency.

Three forces make kernel engineering central:

1. **GPU scarcity and cost** — H100/H200/B200 class accelerators remain expensive and supply-constrained. Getting more tokens/second per GPU is a direct margin lever.
2. **Latency-sensitive agentic workloads** — Multi-step agents (see `03-Agents/`) issue many sequential LLM calls; each millisecond of time-to-first-token (TTFT) and inter-token latency compounds.
3. **Long-context and multimodal models** — Attention cost scales with sequence length; efficient kernels (FlashAttention and successors) are the difference between feasible and infeasible.

| Era | Dominant cost | Key skill |
|-----|---------------|-----------|
| 2020–2022 | Training | Distributed training, data pipelines |
| 2023–2024 | Model access | Prompt engineering, API integration |
| 2025–2026 | **Inference serving** | **Kernel optimization, quantization, batching** |

This category complements `62-Edge-AI-and-On-Device-Inference/` (device-side efficiency) and `41-AI-Cost-Optimization-and-Enterprise-ROI/` (financial framing) by focusing on the **datacenter GPU kernel layer**.

---

## The GPU Execution Model

To optimize kernels you must understand how a GPU actually runs work. Using NVIDIA terminology (AMD ROCm is analogous):

### Hardware Hierarchy

```
GPU
├── SMs (Streaming Multiprocessors)   ← e.g. 132 on H100
│   ├── Warp schedulers               ← issue 32-thread warps
│   ├── CUDA cores (FP32/INT)
│   ├── Tensor Cores                  ← matrix-multiply-accumulate units
│   ├── Register file                 ← fastest, per-thread
│   └── Shared memory / L1            ← per-block, low latency
├── L2 cache                          ← shared across SMs
└── HBM (High Bandwidth Memory)       ← large, high latency (~3 TB/s on H100)
```

### Software Hierarchy

| Level | Unit | Maps to |
|-------|------|---------|
| Grid | Whole kernel launch | GPU |
| Block (CTA) | Group of threads sharing shared memory | 1 SM |
| Warp | 32 threads executing in lockstep (SIMT) | Warp scheduler |
| Thread | Single lane of execution | CUDA core |

The fundamental performance rule: **keep the Tensor Cores fed**. Every stall — waiting on HBM, poor occupancy, warp divergence — is wasted silicon.

### SIMT and Warp Divergence

Threads in a warp execute the same instruction. A branch where some threads go one way and others another (`if (threadIdx.x % 2)`) causes **warp divergence**: the warp serially executes both paths, halving throughput. Good kernels keep control flow uniform within a warp.

---

## Where Performance Is Won and Lost

Most naive GPU code is **memory-bound**, not compute-bound. The GPU can do far more FLOPs than HBM can feed it bytes.

### The Three Big Levers

1. **Memory coalescing** — Adjacent threads should access adjacent memory addresses so the hardware fuses them into one wide transaction. Strided or random access wastes bandwidth.
2. **Kernel fusion** — Fusing element-wise ops (e.g. `bias + GELU + dropout`) into one kernel avoids round-trips to HBM between each op. This is the single biggest win for transformer inference.
3. **Tiling and shared memory** — Load a tile of data into fast shared memory once, reuse it many times, then write back. This is the core idea behind fast GEMM and FlashAttention.

### Illustrative Impact

```
Naive attention (materialize N×N matrix):
  HBM traffic  = O(N²)   ← dominated by memory
  Result: memory-bound, slow, OOM on long context

FlashAttention (tiled, never materialize full matrix):
  HBM traffic  = O(N)    ← fused, streamed through SRAM
  Result: 2-4x faster, linear memory
```

---

## The Optimization Stack

Performance engineers work across multiple abstraction layers. Pick the highest layer that meets your target before dropping lower.

| Layer | Tools | When to use |
|-------|-------|-------------|
| Serving framework | vLLM, TensorRT-LLM, SGLang | First stop — batching, paged KV cache |
| Graph compiler | torch.compile, TensorRT, XLA | Auto-fusion, no manual kernels |
| Kernel DSL | Triton, CUTLASS, ThunderKittens | Custom fused kernels in Python-ish syntax |
| Low-level | Raw CUDA C++, PTX, inline SASS | Last mile, hardware-specific tuning |

The 2026 sweet spot for most teams is **Triton** (see `04-Tools-and-Frameworks.md`): near-CUDA performance with dramatically better productivity. Detailed treatment in `03-Technical-Deep-Dive.md`.

---

## Key Metrics

You cannot optimize what you don't measure. The core inference metrics:

| Metric | Definition | Why it matters |
|--------|-----------|----------------|
| **TTFT** | Time to first token | User-perceived responsiveness (prefill cost) |
| **ITL / TPOT** | Inter-token latency / time per output token | Streaming smoothness (decode cost) |
| **Throughput** | Tokens/sec across all concurrent requests | Cost efficiency |
| **Goodput** | Throughput meeting an SLA | Real usable capacity |
| **MFU** | Model FLOPs Utilization | % of theoretical peak achieved |
| **Occupancy** | Active warps / max warps per SM | Latency hiding capacity |

**Prefill vs decode** is the central inference dichotomy:
- **Prefill** (processing the prompt) is compute-bound, large batched matmuls — high MFU achievable.
- **Decode** (generating one token at a time) is memory-bound, dominated by loading model weights and KV cache — low MFU, needs batching to amortize.

Disaggregating prefill and decode onto different GPU pools is a major 2025–2026 architecture trend (covered in `02-Core-Topics.md`).

---

## Roofline Model and Compute vs Memory Bound

The **roofline model** tells you whether a kernel is limited by compute or memory.

```
Attainable FLOPs/s = min( Peak FLOPs/s,
                          Peak Bandwidth × Arithmetic Intensity )

Arithmetic Intensity (AI) = FLOPs performed / Bytes moved
```

- Low AI (e.g. element-wise add, AI ≈ 0.25) → **memory-bound** → fuse kernels, reduce HBM traffic.
- High AI (e.g. large GEMM) → **compute-bound** → use Tensor Cores, better tiling.

On an H100 (~989 TFLOPs FP16, ~3.35 TB/s HBM), the "ridge point" is around AI ≈ 295 FLOPs/byte. Below that, you're memory-bound. LLM decode sits far to the left — which is exactly why batching, quantization, and KV-cache optimization dominate inference engineering.

```python
# Quick arithmetic-intensity sanity check
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved

# fp16 matmul MxNxK
M, N, K = 4096, 4096, 4096
flops = 2 * M * N * K
bytes_moved = 2 * (M*K + K*N + M*N)  # fp16 = 2 bytes
print(arithmetic_intensity(flops, bytes_moved))  # high -> compute bound
```

---

## Career and Demand Signal

GPU performance engineering is among the best-compensated and scarcest specializations in AI as of 2026:

- **Hiring pull** — Every foundation-model lab, inference startup (Fireworks, Together, Baseten, Modal), and hyperscaler is hiring "inference performance" / "GPU kernel" engineers.
- **Scarcity** — The intersection of deep-learning knowledge and low-level systems programming is rare; CUDA/Triton fluency commands a premium.
- **Leverage** — Individual contributors here directly move company-wide COGS, giving outsized impact and visibility.

Learning path (detailed in `05-Future-Outlook.md`):
1. Master PyTorch profiling and `torch.compile`.
2. Learn Triton — write a fused kernel, benchmark against PyTorch eager.
3. Study FlashAttention and paged-attention internals.
4. Drop into CUDA C++ / CUTLASS for the last mile.
5. Learn a serving framework (vLLM or TensorRT-LLM) end to end.

---

## Cross-References

- `62-Edge-AI-and-On-Device-Inference/` — device-side quantization and NPU inference
- `41-AI-Cost-Optimization-and-Enterprise-ROI/` — financial framing of inference savings
- `02-LLMs/` — transformer architecture that these kernels accelerate
- `29-Reasoning-and-Inference-Scaling/` — test-time compute that amplifies inference cost
- `36-Long-Context-AI/` — where attention-kernel efficiency is decisive
- `05-Enterprise/03-Fine-Tuning-Enterprise.md` — training-side performance considerations

Continue with:
- `02-Core-Topics.md` — batching, KV cache, quantization, speculative decoding
- `03-Technical-Deep-Dive.md` — writing fused kernels, FlashAttention internals
- `04-Tools-and-Frameworks.md` — Triton, CUTLASS, vLLM, TensorRT-LLM, profilers
- `05-Future-Outlook.md` — new hardware, compilers, and the road ahead

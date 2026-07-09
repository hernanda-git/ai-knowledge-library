# Tools and Frameworks

> The 2026 ecosystem for GPU performance engineering: kernel DSLs (Triton, CUTLASS), serving engines (vLLM, TensorRT-LLM, SGLang), compilers (torch.compile, TensorRT), quantization toolkits, and profilers. This is the practitioner's map.

## Table of Contents

- [Landscape Overview](#landscape-overview)
- [Kernel Authoring](#kernel-authoring)
- [Serving Engines](#serving-engines)
- [Compilers and Graph Optimizers](#compilers-and-graph-optimizers)
- [Quantization Toolkits](#quantization-toolkits)
- [Profiling and Debugging](#profiling-and-debugging)
- [Choosing Your Stack](#choosing-your-stack)
- [Cross-References](#cross-references)

---

## Landscape Overview

| Category | Tools | Role |
|----------|-------|------|
| Kernel DSL | Triton, CUTLASS/CuTe, ThunderKittens, Mojo | Write custom fused kernels |
| Serving engine | vLLM, TensorRT-LLM, SGLang, LMDeploy, Hugging Face TGI | End-to-end LLM serving |
| Compiler | torch.compile (Inductor), TensorRT, Apache TVM, XLA/OpenXLA | Auto-fuse and optimize graphs |
| Quantization | AutoAWQ, AutoGPTQ, llm-compressor, TensorRT Model Optimizer, bitsandbytes | Compress models |
| Profiler | Nsight Systems, Nsight Compute, PyTorch Profiler, Perfetto | Find bottlenecks |
| Attention kernels | FlashAttention, xFormers, FlashInfer | Drop-in fast attention |

---

## Kernel Authoring

### Triton
The default choice for custom kernels in 2026. Python syntax, block-level programming, autotuning, cross-vendor (NVIDIA + growing AMD support). Powers many kernels inside PyTorch Inductor and vLLM.
- **Pros:** productivity, readable, autotuned, integrates with PyTorch.
- **Cons:** less control than CUDA for the absolute last 5–10%.

### CUTLASS / CuTe
NVIDIA's C++ template library for high-performance GEMM and convolution, plus the CuTe layout algebra. This is what you use when you need peak Tensor Core utilization and full control of tiling/pipelining.
- **Pros:** peak performance, composable collectives, Hopper/Blackwell features (TMA, WGMMA).
- **Cons:** steep C++ learning curve.

### FlashInfer
A specialized library of attention and KV-cache kernels optimized for LLM serving (paged KV, prefill/decode, speculative). Increasingly used as the attention backend in vLLM and SGLang.

### Emerging
- **ThunderKittens** — Stanford tile-based CUDA embedded DSL for readable high-performance kernels.
- **Mojo** (Modular) — a Python-superset systems language targeting portable accelerator programming.

---

## Serving Engines

### vLLM
The most widely adopted open-source LLM serving engine. Originated PagedAttention; now includes continuous batching, chunked prefill, speculative decoding, prefix caching, FP8/INT4 quant, tensor/pipeline/expert parallelism, and disaggregated serving.

```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          tensor_parallel_size=2,
          gpu_memory_utilization=0.90,
          enable_prefix_caching=True,
          quantization="fp8")
params = SamplingParams(temperature=0.7, max_tokens=256)
print(llm.generate(["Summarize PagedAttention."], params)[0].outputs[0].text)
```

Serve an OpenAI-compatible endpoint:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 --enable-chunked-prefill --quantization fp8
```

### TensorRT-LLM
NVIDIA's engine built on TensorRT. Compiles per-model, per-GPU engines for peak NVIDIA performance. In-flight batching, FP8/INT4, paged KV, speculative decoding.
- **Pros:** often the fastest on NVIDIA hardware, deep Hopper/Blackwell support.
- **Cons:** engine build step, NVIDIA-only, less flexible than vLLM.

### SGLang
Serving engine with **RadixAttention** for aggressive prefix-cache reuse — excellent for agents and structured/branching generation. Strong for high-throughput and complex prompting workflows.

### Others
- **Hugging Face TGI** — production-ready, easy HF integration.
- **LMDeploy** — from the InternLM team, strong quantization and TurboMind backend.

| Engine | Best for | Vendor |
|--------|----------|--------|
| vLLM | General open-source serving, flexibility | Any |
| TensorRT-LLM | Peak NVIDIA performance | NVIDIA |
| SGLang | Prefix-heavy / agentic / structured | Any |
| TGI | HF ecosystem, ops simplicity | Any |

---

## Compilers and Graph Optimizers

### torch.compile
The first optimization to try — one line, big wins. Uses TorchDynamo to capture the graph and TorchInductor to generate fused Triton kernels.

```python
model = torch.compile(model, mode="max-autotune")
# modes: default | reduce-overhead | max-autotune
```
- `reduce-overhead` uses CUDA graphs to cut launch overhead (great for small-batch decode).
- `max-autotune` sweeps kernel configs for best throughput.

### TensorRT
Ahead-of-time compiler producing optimized inference engines: layer fusion, precision calibration, kernel auto-tuning. Standard for NVIDIA production CV and non-LLM models.

### TVM / OpenXLA
Cross-hardware compilers (including non-NVIDIA accelerators, TPUs). XLA underpins JAX and TF; OpenXLA is the vendor-neutral evolution.

---

## Quantization Toolkits

| Tool | Method | Notes |
|------|--------|-------|
| AutoAWQ | AWQ (weight-only INT4) | Fast, accuracy-preserving, popular for decode |
| AutoGPTQ | GPTQ (weight-only INT4/3) | Mature, wide model support |
| llm-compressor | INT8/FP8/INT4, structured sparsity | vLLM-native (Neural Magic lineage) |
| TensorRT Model Optimizer | FP8/INT4 + QAT | Best for TensorRT-LLM pipelines |
| bitsandbytes | INT8/NF4 | Easy, training + inference (QLoRA) |

Typical flow: quantize once offline (calibration set), save, then load into vLLM/TensorRT-LLM with the matching `quantization=` flag. Always re-evaluate quality — see `52-AI-Hallucination-Detection-and-Mitigation/`.

---

## Profiling and Debugging

| Tool | Scope | Use |
|------|-------|-----|
| PyTorch Profiler | Op-level | Find hot ops, CPU-GPU gaps |
| Nsight Systems (`nsys`) | System timeline | Kernel overlap, stalls, memcpy, launch overhead |
| Nsight Compute (`ncu`) | Single kernel | Roofline, occupancy, stall reasons |
| Perfetto / Chrome trace | Trace viewer | Visualize profiler output |
| `nvidia-smi` / DCGM | Fleet | Utilization, memory, power, thermals |

```bash
# System timeline
nsys profile -o trace python serve.py
# Deep single-kernel analysis
ncu --set full -k my_fused_kernel -o kreport python bench.py
```

Rule: profile before and after every change; keep a benchmark harness with fixed inputs, warmup, and `torch.cuda.synchronize()`.

---

## Choosing Your Stack

A pragmatic decision tree:

1. **Just serving an open model?** Start with **vLLM** (or TensorRT-LLM if NVIDIA-only and peak perf matters).
2. **Custom model / ops slow?** Try **torch.compile(max-autotune)** first.
3. **Still a hot custom op?** Write it in **Triton**, benchmark vs eager.
4. **Need the last 10% on NVIDIA?** Drop to **CUTLASS / CUDA C++**.
5. **Memory-bound / too big?** Add **quantization** (FP8 or AWQ INT4).
6. **Prefix-heavy agents?** Consider **SGLang** for RadixAttention.

Most teams never need to write raw CUDA — the framework + Triton layer captures the majority of gains. Reserve hand-written kernels for proven, profiled bottlenecks.

---

## Cross-References

- `01-Overview.md` — the optimization stack framing
- `02-Core-Topics.md` — the techniques these tools implement
- `03-Technical-Deep-Dive.md` — writing the kernels these tools generate
- `62-Edge-AI-and-On-Device-Inference/` — device-side runtimes (ONNX Runtime, TensorRT, Core ML)
- `20-Agent-Infrastructure-and-Observability/` — serving in production
- `48-MCP-Cloud-Infrastructure-Agent-as-a-Service/` — hosted inference platforms

# Core Topics in Inference Performance

> The practical techniques that make large-model serving fast and cheap: continuous batching, KV-cache management, quantization, speculative decoding, and prefill/decode disaggregation. These are the levers you reach for before hand-writing kernels.

## Table of Contents

- [Continuous (In-Flight) Batching](#continuous-in-flight-batching)
- [KV Cache and PagedAttention](#kv-cache-and-pagedattention)
- [Quantization for Inference](#quantization-for-inference)
- [Speculative Decoding](#speculative-decoding)
- [Prefill/Decode Disaggregation](#prefilldecode-disaggregation)
- [Tensor, Pipeline, and Expert Parallelism](#tensor-pipeline-and-expert-parallelism)
- [Chunked Prefill](#chunked-prefill)
- [Putting It Together](#putting-it-together)
- [Cross-References](#cross-references)

---

## Continuous (In-Flight) Batching

Static batching waits to assemble a full batch, then runs all sequences to completion — wasting GPU cycles when sequences finish at different lengths. **Continuous batching** (a.k.a. in-flight batching, popularized by Orca and vLLM) processes the batch at the **iteration** level: as soon as one sequence emits its end-of-sequence token, its slot is freed and a queued request takes its place.

```
Static batching (poor utilization):
  req A [========>       ] done, GPU idle waiting
  req B [==============>] done
  step  ^^^^^^^^^^^^^^^^

Continuous batching (high utilization):
  req A [========] done -> slot reused by req C immediately
  req B [==============]
  req C          [=======]
```

Impact: often **2–4x throughput** improvement over static batching with no model change. This is the single most important serving optimization and is built into vLLM, TensorRT-LLM, and SGLang.

---

## KV Cache and PagedAttention

During autoregressive decoding, the keys and values of all previous tokens are cached to avoid recomputation. This **KV cache** grows linearly with sequence length and is often the dominant memory consumer.

```
KV cache size = 2 (K and V)
              × num_layers
              × num_kv_heads × head_dim
              × seq_len
              × batch_size
              × bytes_per_element
```

For a 70B model at long context, the KV cache can exceed the model weights themselves.

### The Fragmentation Problem

Naive contiguous KV allocation reserves max-length buffers per request → massive internal fragmentation and low batch sizes.

### PagedAttention (vLLM)

Borrowing OS virtual-memory ideas, **PagedAttention** stores KV in fixed-size **blocks** (pages) that need not be contiguous. A block table maps logical positions to physical blocks.

| Benefit | Mechanism |
|---------|-----------|
| Near-zero fragmentation | Fixed-size blocks packed tightly |
| Prefix sharing | Shared system prompts point to same blocks (copy-on-write) |
| Higher batch size | More requests fit in the same HBM |

Related techniques:
- **GQA / MQA** (grouped/multi-query attention) — fewer KV heads shrink the cache directly.
- **KV cache quantization** — store KV in INT8/FP8 to halve or quarter its footprint.
- **Prefix / prompt caching** — reuse KV for shared prefixes across requests (huge for agent system prompts; see `32-Agent-Memory-Systems/`).

---

## Quantization for Inference

Quantization reduces the numeric precision of weights and/or activations, shrinking memory and boosting memory-bound decode.

| Scheme | Precision | Notes |
|--------|-----------|-------|
| FP16 / BF16 | 16-bit | Baseline for modern inference |
| FP8 (E4M3/E5M2) | 8-bit | Native on Hopper/Blackwell Tensor Cores; strong accuracy |
| INT8 (SmoothQuant, LLM.int8) | 8-bit | Widely supported; needs outlier handling |
| INT4 (GPTQ, AWQ) | 4-bit | Weight-only; excellent for memory-bound decode |
| KV-cache quant | INT8/FP8 | Orthogonal, targets cache not weights |

Key distinctions:
- **Weight-only** (GPTQ, AWQ) — dequantize on the fly; great for decode where weight loading dominates.
- **Weight + activation** (SmoothQuant, FP8) — enables faster matmul on quantized Tensor Cores; better for compute-bound prefill.

```python
# Example: loading an AWQ 4-bit model in vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
    max_model_len=4096,
)
out = llm.generate(["Explain PagedAttention."], SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

Trade-off: aggressive quantization can degrade quality; always evaluate with task-specific benchmarks (see `06-Advanced/` and `52-AI-Hallucination-Detection-and-Mitigation/`). FP8 has become the 2026 default sweet spot on Hopper/Blackwell.

---

## Speculative Decoding

Decode is memory-bound: the GPU loads all model weights to produce one token. **Speculative decoding** amortizes this by having a small **draft model** propose several tokens, then verifying them in a single forward pass of the large **target model**. Accepted tokens are kept; the first rejection triggers a correction.

```
Draft model (fast): proposes  t1 t2 t3 t4
Target model (1 pass): verifies -> accepts t1 t2, rejects t3
Net: 2+ tokens for the cost of ~1 target forward pass
```

Variants:
| Method | Draft source |
|--------|--------------|
| Standard spec decode | Separate small model |
| Self-speculative (Medusa) | Extra heads on the same model |
| EAGLE / EAGLE-2 | Feature-level autoregression, high accept rate |
| Lookahead decoding | N-gram guesses, no draft model |
| Prompt lookup | Copy spans from the prompt (great for RAG/code) |

Speedups of **2–3x** on decode are common with negligible quality change (verification is exact). Ties into `29-Reasoning-and-Inference-Scaling/` where output length is large.

---

## Prefill/Decode Disaggregation

Prefill (compute-bound) and decode (memory-bound) have opposite hardware profiles. Running them on the same GPU pool means neither is optimal and they interfere (a long prefill stalls ongoing decodes).

**Disaggregated serving** (DistServe, Splitwise, and adopted in vLLM/TensorRT-LLM 2026) routes:
- **Prefill** to a pool tuned for throughput (large batches, high MFU).
- **Decode** to a pool tuned for latency (many concurrent slots).

KV cache is transferred from prefill to decode nodes over fast interconnect (NVLink/InfiniBand).

| Benefit | Reason |
|---------|--------|
| Better SLA compliance | Prefill spikes don't hurt decode latency |
| Independent scaling | Scale each pool to its bottleneck |
| Higher goodput | Each stage tuned for its regime |

---

## Tensor, Pipeline, and Expert Parallelism

When a model doesn't fit on one GPU, split it:

| Strategy | Splits | Communication | Best for |
|----------|--------|---------------|----------|
| **Tensor parallel (TP)** | Each layer's matmul across GPUs | All-reduce per layer (heavy, needs NVLink) | Latency, within a node |
| **Pipeline parallel (PP)** | Different layers on different GPUs | Point-to-point between stages | Cross-node, throughput |
| **Expert parallel (EP)** | MoE experts across GPUs | All-to-all routing | Mixture-of-Experts models |

Modern MoE serving (see `30-Small-Language-Models/` and mixture-of-experts) combines TP + EP. Choosing the split is a bandwidth-vs-latency optimization: intra-node NVLink favors TP; slower inter-node links favor PP.

---

## Chunked Prefill

A long prompt's prefill can monopolize the GPU and starve decode requests. **Chunked prefill** breaks a long prefill into smaller chunks and interleaves them with ongoing decode steps in the same batch, smoothing latency.

```
Without chunking: [====== long prefill ======] then decodes resume (spiky ITL)
With chunking:    [pf][dec][pf][dec][pf][dec]  (smooth ITL)
```

This is now default in vLLM for balancing TTFT and ITL under mixed workloads.

---

## Putting It Together

A production 2026 LLM serving stack typically layers:

1. **Continuous batching** — the foundation.
2. **PagedAttention KV cache** + prefix caching.
3. **FP8 or INT4 quantization** matched to the workload.
4. **Speculative decoding** for latency-critical paths.
5. **Chunked prefill** for smooth mixed-traffic latency.
6. **Disaggregation + TP/PP/EP** at scale.

Each layer multiplies with the others; combined, they routinely deliver **10x+** cost/throughput improvement over a naive `model.generate()` loop.

---

## Cross-References

- `01-Overview.md` — GPU execution model and metrics
- `03-Technical-Deep-Dive.md` — how the underlying kernels are written
- `04-Tools-and-Frameworks.md` — vLLM, TensorRT-LLM, SGLang configuration
- `32-Agent-Memory-Systems/` — prefix caching for agent system prompts
- `36-Long-Context-AI/` — KV cache pressure at long context
- `30-Small-Language-Models/` — draft models and MoE serving
- `41-AI-Cost-Optimization-and-Enterprise-ROI/` — translating these into dollars

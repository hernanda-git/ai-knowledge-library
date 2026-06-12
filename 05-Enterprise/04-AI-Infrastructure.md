# AI Infrastructure: Compute, Storage, and Networking
## Table of Contents
1. [Introduction](#1-introduction)
2. [GPU Architecture](#2-gpu)
   - 2.1 [Current Generation (2025-2026)](#21-current-generation-2025-2026)
   - 2.2 [GPU Memory Configurations & Tradeoffs](#22-gpu-memory-configurations--tradeoffs)
   - 2.3 [Bottleneck: Memory Bandwidth](#23-bottleneck-memory-bandwidth)
3. [Training Infrastructure](#3-training)
   - 3.1 [Cluster Design](#31-cluster-design)
   - 3.2 [Parallelism Strategies](#32-parallelism-strategies)
   - 3.3 [Choosing a Parallelism Strategy](#33-choosing-a-parallelism-strategy)
   - 3.4 [FSDP Configuration Example](#34-fsdp-configuration-example)
   - 3.5 [Practical Cluster Planning](#35-practical-cluster-planning)
4. [Inference Infrastructure](#4-inference)
   - 4.1 [Serving Comparison](#41-serving-comparison)
   - 4.2 [vLLM Deployment Example](#42-vllm-deployment-example)
   - 4.3 [Inference Hardware Selection](#43-inference-hardware-selection)
   - 4.4 [Kubernetes Deployment Example](#44-kubernetes-deployment-example)
5. [Storage for AI](#5-storage)
   - 5.1 [Storage Tier Design](#51-storage-tier-design)
   - 5.2 [Data Pipeline Storage Patterns](#52-data-pipeline-storage-patterns)
   - 5.3 [Dataset Loading Code Example](#53-dataset-loading-code-example)
6. [Networking](#6-networking)
   - 6.1 [Interconnect Comparison](#61-interconnect-comparison)
   - 6.2 [Network Topology Design](#62-network-topology-design)
   - 6.3 [Network Planning Guide](#63-network-planning-guide)
7. [Cross-References](#7-cross-references)
---

## 1. Introduction

AI infrastructure is the physical and virtual foundation for training and deploying ML models. It encompasses GPUs, storage systems, networking, and orchestration software. As models grow to trillion-plus parameters, infrastructure design becomes a critical competitive advantage.

This document provides a practitioner's guide to selecting, provisioning, and operating AI infrastructure. It covers GPU architecture, training and inference cluster design, parallelism strategies, storage tiers, networking topologies, and includes executable code examples for FSDP, vLLM, and Kubernetes deployment.

**Key design principles:**
- **Match the bottleneck:** Training is compute-bound; inference is memory-bandwidth-bound.
- **Balance the system:** CPU, GPU, memory, storage, and network must be co-designed — the weakest link determines throughput.
- **Plan for utilization:** A GPU cluster at <60% utilization wastes capital. Use orchestration and bin-packing to maximize efficiency.

---

## 2. GPU Architecture

### 2.1 Current Generation (2025-2026)

| GPU | Memory | BW | FP8 TFLOPS | Interconnect | Power |
|-----|:-----:|:--:|:----------:|:-----------:|:-----:|
| **H100 SXM** | 80GB HBM3 | 3.35 TB/s | 1,979 | NVLink 4 (900 GB/s) | 700W |
| **H200** | 141GB HBM3e | 4.8 TB/s | 1,979 | NVLink 4 | 700W |
| **B200** | 192GB HBM3e | 8 TB/s | 4,500 | NVLink 5 (1.8 TB/s) | 1,000W |
| **MI300X (AMD)** | 192GB HBM3 | 5.3 TB/s | 1,307 | Infinity Fabric | 750W |

**GPU-to-Memory Ratio (key metric for inference capacity):**
| GPU | Total HBM | Max Model Size (FP16/BF16) | Max Model Size (FP8/INT8) | Max Model Size (FP4/INT4) |
|-----|:---------:|:--------------------------:|:-------------------------:|:-------------------------:|
| H100 SXM | 80 GB | 40B params | 80B params | 160B params |
| H200 | 141 GB | 70B params | 140B params | 280B params |
| B200 | 192 GB | 96B params | 192B params | 384B params |
| MI300X | 192 GB | 96B params | 192B params | 384B params |

*Assumes 2 bytes/param for FP16/BF16, 1 byte/param for FP8/INT8, 0.5 bytes/param for FP4/INT4. KV cache overhead excluded.*

### 2.2 GPU Memory Configurations & Tradeoffs

| Factor | Impact | Consideration |
|--------|--------|---------------|
| **HBM capacity** | Determines max model size per GPU | Larger capacity reduces TP/PP degree needed |
| **HBM bandwidth** | Dominates inference latency | Higher BW = faster token generation |
| **Interconnect bandwidth** | Dominates training scaling efficiency | NVLink > InfiniBand > Ethernet |
| **TDP / Cooling** | Affects datacenter density | 1,000W GPUs require liquid cooling |
| **FP8/FP4 support** | Enables lower-precision training | Newer GPUs have native FP8/FP4 tensor cores |

**Cooling requirements by GPU TDP:**
| TDP Range | Cooling Method | Rack Density | CAPEX |
|:---------:|----------------|:------------:|:-----:|
| < 350W | Air-cooled (standard) | 10-20 kW/rack | Low |
| 350-700W | Air-cooled (high-flow) | 20-40 kW/rack | Medium |
| 700-1000W | Direct-to-chip liquid | 40-80 kW/rack | High |
| > 1000W | Immersion cooling | 80-150+ kW/rack | Very High |

### 2.3 Bottleneck: Memory Bandwidth

For inference, the bottleneck is **memory bandwidth** (loading model weights), not **compute** (matrix multiplication). This is why:

- **Quantization helps** — less data to load per token (FP8 loads half the bytes of FP16)
- **Batch processing helps** — amortizes the weight-loading cost across multiple sequences
- **Speculative decoding helps** — fewer autoregressive generation steps
- **KV cache optimization** — reduces memory pressure for long-context inference

**Compute-bound vs Memory-bound regimes:**

| Scenario | Bound By | Symptoms | Solution |
|----------|:--------:|----------|----------|
| Training (large batch) | Compute | GPU compute util ~95%+ | Add more GPUs/TP degree |
| Training (small batch) | Memory BW | Low SM utilization | Increase batch size |
| Inference (small batch ≤ 4) | Memory BW | Low compute util | Batch requests, use flash attention |
| Inference (large batch ≥ 32) | Compute | High arithmetic intensity | Reduce precision, increase GPU count |
| Long context (32K+ tokens) | Memory capacity | OOM or KV cache thrashing | Use GQA, sliding window, context caching |

---

## 3. Training Infrastructure

### 3.1 Cluster Design

| Scale | GPUs | Model Size | Training Time | Example |
|:-----:|:----:|:----------:|:-------------:|---------|
| **Single** | 1-8 | <7B | Hours | Fine-tuning |
| **Small** | 8-64 | 7-70B | Days | LLaMA 70B |
| **Medium** | 64-1024 | 70-400B | Weeks | GPT-4, Claude |
| **Large** | 1024-16384 | 400B-1T+ | Months | Frontier models |
| **Frontier** | 16K-100K+ | >1T | Months | Next-gen systems |

### 3.2 Parallelism Strategies

| Strategy | Split | Communication | Best For |
|----------|:-----:|:-------------:|----------|
| **Data Parallel (DP)** | Data across GPUs | Gradient sync | Small models, large data |
| **Tensor Parallel (TP)** | Layers across GPUs | High (NVLink) | Large models, single node |
| **Pipeline Parallel (PP)** | Layer groups | Moderate | Cross-node scaling |
| **Expert Parallel (EP)** | MoE experts | All-to-all | Mixture of Experts |
| **Sequence Parallel (SP)** | Sequence dimension | High | Long sequences |
| **FSDP (Fully Sharded DP)** | Params + optimizer states | Moderate | Training efficiency |

### 3.3 Choosing a Parallelism Strategy

**Decision matrix — which parallelism to use when:**

| Model Size | GPU Count | Recommended Strategy | Reasoning |
|:----------:|:---------:|---------------------|-----------|
| < 1B | 1-4 GPUs | DP or FSDP | Fits in single GPU memory. No TP/PP needed. |
| 1B-7B | 4-8 GPUs | FSDP | Sharding fits model across GPUs efficiently. |
| 7B-13B | 8-32 GPUs | FSDP + TP (degree 2-4) | TP handles intra-node, FSDP handles inter-node. |
| 13B-70B | 32-256 GPUs | TP(8) + PP + DP | 3D parallelism. TP inside node, PP across nodes. |
| 70B-400B | 256-1024 GPUs | TP(8) + PP(4-8) + DP | 3D parallelism + activation checkpointing. |
| > 400B | 1024+ GPUs | TP(8) + PP(8-16) + DP + SP | 4D parallelism. Sequence parallel for long ctx. |
| MoE models | Any | EP + TP + DP | Expert parallel for MoE layers, TP for dense layers. |

**General guidance:**
1. **FSDP is the default** for most fine-tuning and small-to-medium training. Start here.
2. **TP is required** when a single layer does not fit on one GPU (models > 13B on H100).
3. **TP degree should not exceed 8** on NVLink — beyond that, cross-node TP is communication-bound.
4. **PP is a last resort** for cross-node scaling when TP alone is insufficient. PP creates pipeline bubbles.
5. **EP is mandatory for MoE** models — each expert must fit on one GPU.
6. **Always enable activation checkpointing** for models > 7B to reduce memory footprint.

### 3.4 FSDP Configuration Example

```python
# config/fsdp_config.py — PyTorch FSDP configuration for training Llama 3 8B
# Usage: torchrun --nproc_per_node=8 train.py --fsdp-config config/fsdp_config.py

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch

# --- Sharding ---
# HYBRID_SHARD: shards within node, replicates across nodes (best for multi-node)
# FULL_SHARD: shards across all GPUs (best for single-node)
sharding_strategy = ShardingStrategy.HYBRID_SHARD

# --- Mixed Precision ---
# Keep master weights in FP32, compute in BF16, reduce in FP32
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

# --- Auto-Wrap Policy ---
# Wrap each transformer block into its own FSDP unit
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

# --- Memory Optimizations ---
backward_prefetch = BackwardPrefetch.BACKWARD_PRE      # Prefetch next shard
forward_prefetch = True                                  # Prefetch forward params
limit_all_gathers = True                                 # Limit in-flight all-gathers
cpu_offload = CPUOffload(offload_params=False)           # Keep params on GPU

# --- Activation Checkpointing ---
# Reduces memory by 50-70% at ~20% throughput cost
activation_checkpointing = True

# --- Config dict for trainer ---
fsdp_config = {
    "sharding_strategy": sharding_strategy,
    "mixed_precision": mixed_precision,
    "auto_wrap_policy": auto_wrap_policy,
    "backward_prefetch": backward_prefetch,
    "forward_prefetch": forward_prefetch,
    "limit_all_gathers": limit_all_gathers,
    "cpu_offload": cpu_offload,
    "activation_checkpointing": activation_checkpointing,
}
```

### 3.5 Practical Cluster Planning

**Estimated training cost (H100 GPU-hours):**

| Model Size | GPUs | Duration | GPU-Hours | Est. Cost ($3/GPU-hr) |
|:----------:|:----:|:--------:|:---------:|:---------------------:|
| 7B (fine-tune) | 8 | 4 hours | 32 | $96 |
| 7B (pre-train) | 64 | 5 days | 7,680 | $23,040 |
| 13B (pre-train) | 128 | 10 days | 30,720 | $92,160 |
| 70B (pre-train) | 512 | 30 days | 368,640 | $1.1M |
| 405B (pre-train) | 8,192 | 60 days | 11,796,480 | $35.4M |
| 1T+ (pre-train) | 16,384 | 90 days | 35,389,440 | ~$106M |

*Note: Costs assume 65% MFU (Model FLOPS Utilization) and include overhead. Real costs vary by cluster efficiency, parallelism config, and cloud vs on-prem pricing.*

**Key planning metrics:**
- **MFU (Model FLOPS Utilization):** Target 50-65% for large-scale training. Below 40% indicates a parallelism or I/O bottleneck.
- **HFU (Hardware FLOPS Utilization):** Typically 5-10 points higher than MFU. Measures raw hardware throughput.
- **Memory reserved per GPU:** Model weights + optimizer states + activations + KV cache. Use activation checkpointing to keep this below 90% of HBM.
- **Communication-to-compute ratio:** If > 30%, increase local computation (larger batch, higher TP degree).

---

## 4. Inference Infrastructure

### 4.1 Serving Comparison

| Feature | vLLM | TGI | Triton Inference Server | TensorRT-LLM |
|---------|:----:|:---:|:-----------------------:|:------------:|
| **Engine** | Python + PagedAttention | Rust + Python | C++ + GPU | C++ + CUDA |
| **Throughput** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| **Latency (P50)** | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **HF Integration** | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| **Multi-framework** | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ |
| **Quantization** | AWQ, GPTQ, FP8 | AWQ, GPTQ, bitsandbytes | All (via plugins) | FP8, INT4, INT8 |
| **Continous Batching** | ✅ (native) | ✅ (native) | ✅ | ✅ |
| **PagedAttention** | ✅ (inventor) | ❌ (static cache) | ❌ | ✅ (inflight batching) |
| **LoRA Adapters** | ✅ (hot-swap) | ✅ | ✅ | ⚠️ (limited) |
| **Streaming** | ✅ | ✅ | ✅ | ✅ |
| **Multi-node** | ⚠️ (experimental) | ❌ | ✅ | ✅ |
| **Open Source** | ✅ (Apache 2.0) | ✅ (Apache 2.0) | ✅ (BSD-3) | ❌ (source-available) |
| **Best For** | High-throughput chat, API serving | Quick HF deployment | Multi-model enterprise | Maximum GPU perf |

**Recommendations:**
- **Start with vLLM** for most LLM serving use cases. It delivers the best overall throughput with continuous batching and PagedAttention.
- **Use TGI** when deep Hugging Face ecosystem integration is needed (e.g., custom pipelines, tokenizers).
- **Use Triton** in enterprise environments serving multiple model types (LLMs + vision + tabular) behind a single endpoint.
- **Use TensorRT-LLM** for peak performance on NVIDIA hardware in latency-critical applications (e.g., real-time chatbots).

### 4.2 vLLM Deployment Example

```python
# serve_vllm.py — Start a vLLM server with continuous batching and quantization
# Install: pip install vllm
# Run: python serve_vllm.py

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import uvicorn
import argparse

def create_engine(model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                  tensor_parallel_size: int = 1,
                  max_model_len: int = 8192,
                  gpu_memory_utilization: float = 0.90,
                  quantization: str = None,
                  dtype: str = "bfloat16",
                  enable_lora: bool = False,
                  max_loras: int = 8):
    """
    Create an optimized vLLM async engine.
    
    Args:
        model_name: HF model name or local path
        tensor_parallel_size: Number of GPUs for TP
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use
        quantization: "awq", "gptq", "fp8", or None
        dtype: Model precision
    """
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        dtype=dtype,
        enable_lora=enable_lora,
        max_loras=max_loras,
        # Performance tuning
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        # Memory optimization
        enforce_eager=False,           # Use CUDA graph (faster)
        enable_prefix_caching=True,    # Reuse KV cache for common prefixes
        # Scheduling
        scheduler="vllm_default",
        use_v2_block_manager=True,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quant", type=str, default=None)
    args = parser.parse_args()

    engine = create_engine(
        model_name=args.model,
        tensor_parallel_size=args.tp,
        quantization=args.quant,
    )
    # Serve OpenAI-compatible API
    run_server(engine, host="0.0.0.0", port=args.port)
```

```python
# client_example.py — Call the vLLM server
# Run: python client_example.py

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-not-needed",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain AI inference infrastructure in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=256,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 4.3 Inference Hardware Selection

**Hardware selection by deployment scenario:**

| Scenario | Hardware | Memory Needed | QPS Target | Est. Cost/Request |
|----------|----------|:-------------:|:----------:|:-----------------:|
| **Small model (7B), low traffic** | 1× L4 (24GB) | ~14 GB | < 10 | $0.0001 |
| **Medium model (13B), API** | 1× A100-80GB | ~26 GB | 10-100 | $0.0003 |
| **Large model (70B), chat** | 4× A100-80GB (TP=4) | ~140 GB | 50-500 | $0.001 |
| **Large model (70B), batch** | 1× H200 (141GB) | ~140 GB | 100-1000 | $0.0005 |
| **Frontier (405B), API** | 8× H100 (80GB, TP=8) | ~810 GB | 100-500 | $0.005-0.01 |
| **Edge (7B), on-device** | Apple M3 Max / RTX 4090 | ~14 GB | < 5 | $0 |
| **Real-time (7-13B), <100ms** | NVIDIA L40S (48GB) | ~14-26 GB | 50-200 | $0.0002 |

**Key deployment formulas:**

```
VRAM required ≈ (model_parameters × bytes_per_param) + KV_cache + overhead

KV cache per token per layer ≈ 2 × hidden_dim × num_layers × bytes_per_elem

Example (Llama 3.1 70B, FP16, 4K context):
  Weights: 70B × 2 bytes = 140 GB
  KV cache: 2 × 8192 × 80 × 2 bytes × 2 (K+V) × 4096 tokens ≈ 21 GB (per batch of 1)
  Total: 140 GB + 21 GB + ~10 GB overhead ≈ 171 GB → needs 3× H100 (80 GB) or 2× H200 (141 GB)
```

### 4.4 Kubernetes Deployment Example

```yaml
# k8s/vllm-deployment.yaml — Deploy vLLM on Kubernetes with GPU scheduling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-8b
  namespace: ai-serving
  labels:
    app: vllm
    model: llama-3.1-8b
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0   # zero-downtime updates
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # Prevent scheduling multiple large models on the same node
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: vllm
              topologyKey: kubernetes.io/hostname
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-3.1-8B-Instruct"
        - "--tensor-parallel-size"
        - "1"
        - "--max-model-len"
        - "8192"
        - "--gpu-memory-utilization"
        - "0.90"
        - "--dtype"
        - "bfloat16"
        - "--enable-prefix-caching"
        - "--port"
        - "8000"
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
        ports:
        - containerPort: 8000
          protocol: TCP
        resources:
          limits:
            nvidia.com/gpu: 1    # Request 1 GPU
            memory: "64Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "48Gi"
            cpu: "4"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ai-serving
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: ClusterIP
---
# Horizontal Pod Autoscaler — scale based on GPU utilization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: ai-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama-8b
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 5. Storage for AI

### 5.1 Storage Tier Design

| Tier | Type | Example | Speed | Cost/GB | Use Case |
|------|------|---------|:-----:|:-------:|----------|
| **Hot** | NVMe SSD | 8x Samsung PM9A3 | 20 GB/s | ~$0.10 | Training data (active), checkpoints |
| **Warm** | Object storage | S3, GCS | 1-10 GB/s | ~$0.02 | Datasets, model registry |
| **Cold** | Tape/Archive | Glacier, Azure Archive | Minutes | ~$0.001 | Long-term backups |
| **ML Cache** | Distributed cache | Alluxio, JuiceFS | Varies | Varies | Fast access to cloud data |

**End-to-end data pipeline latency budget (for training at 50 GB/s effective throughput):**

| Stage | Target Latency | Storage Layer | Bottleneck |
|-------|:--------------:|---------------|:----------:|
| Raw data → Preprocessed | < 1 hour | Object storage (warm) | CPU throughput |
| Preprocessed → Shuffled | < 10 min | NVMe scratch (hot) | I/O bandwidth |
| Shuffled → GPU | < 1 ms/batch | GPU memory (HBM) | PCIe bandwidth |
| GPU → Checkpoint | < 5 min | NVMe + Object storage | Network + I/O |
| Checkpoint → Archive | < 24 hours | Object storage (cold) | Upload bandwidth |

**Recommended checkpointing strategy:**
- **Asynchronous checkpointing:** Write checkpoints in the background without blocking training.
- **Distributed checkpointing:** Each GPU saves its shard independently (enabled by FSDP).
- **Checkpoint frequency:** Every N steps where N = max(100, total_steps / 100).
- **Checkpoint retention:** Keep last 5 checkpoints + best checkpoint + every 1000-step checkpoint for debugging.

### 5.2 Data Pipeline Storage Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Local SSD caching** | Pre-fetch dataset to local NVMe | Small datasets (< 500 GB) that fit on each node |
| **Shared filesystem (NFS/Lustre)** | All nodes read from shared storage | Medium datasets, easy to manage |
| **Object store streaming** | Stream data from S3/GCS via mount | Large datasets, elastic cloud training |
| **Distributed cache (Alluxio)** | Cache hot data from object store | Hybrid cloud, multi-region training |
| **Semi-structured (WebDataset)** | Sharded tar files with index | Large-scale training with high throughput |

### 5.3 Dataset Loading Code Example

```python
# data_pipeline.py — Efficient data loading for large-scale training
# Demonstrates three patterns: local caching, streaming, and WebDataset shards

import os
import io
import random
from typing import Iterator, Dict
import torch
from torch.utils.data import IterableDataset, DataLoader

# --- Pattern 1: Streaming from object store with local cache ---
class S3StreamingDataset(IterableDataset):
    """
    Streams training data from S3 with a local SSD cache.
    Assumes data is pre-tokenized and stored as .npy or .arrow shards.
    """
    def __init__(self, 
                 s3_uri: str,
                 local_cache_dir: str = "/mnt/cache/data",
                 shard_glob: str = "shard-*.npy",
                 max_shards_in_cache: int = 8):
        self.s3_uri = s3_uri
        self.local_cache_dir = local_cache_dir
        self.shard_glob = shard_glob
        self.max_shards_in_cache = max_shards_in_cache
        os.makedirs(local_cache_dir, exist_ok=True)

    def _fetch_shard(self, shard_name: str) -> str:
        """Download a shard to local cache if not present."""
        local_path = os.path.join(self.local_cache_dir, shard_name)
        if not os.path.exists(local_path):
            # Example: aws s3 cp (replace with boto3 for production)
            os.system(f"aws s3 cp {self.s3_uri}/{shard_name} {local_path}")
        return local_path

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Shuffle shard order for each epoch
        import glob
        shard_files = glob.glob(os.path.join(self.local_cache_dir, self.shard_glob))
        # Fallback: list from S3 if cache is empty
        if not shard_files:
            shard_files = [f"shard-{i:05d}.npy" for i in range(1000)]
        random.shuffle(shard_files)

        for shard in shard_files:
            local = self._fetch_shard(os.path.basename(shard))
            data = torch.from_numpy(__import__('numpy').load(local))
            # Shuffle within shard
            idx = torch.randperm(len(data))
            data = data[idx]
            for sample in data:
                yield {"input_ids": sample[:-1], "labels": sample[1:]}

# --- Pattern 2: WebDataset for high-throughput sharded loading ---
def create_webdataset_loader(
    data_pattern: str,
    batch_size: int = 8,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """
    Create a DataLoader using WebDataset format.
    
    Data should be structured as:
        /data/train/shard-000000.tar
        /data/train/shard-000001.tar
        ...
    Each tar contains:
        sample_000000.npy, sample_000000.json
        sample_000001.npy, sample_000001.json
    
    Requires: pip install webdataset
    """
    import webdataset as wds

    dataset = (
        wds.WebDataset(data_pattern, shardshuffle=True)
        .shuffle(1000)                              # Buffer shuffle
        .decode()                                    # Auto-decode
        .to_tuple("npy", "json")                     # Extract npy + json
        .batched(batch_size, partial=False)          # Batch
    )

    loader = DataLoader(
        dataset,
        batch_size=None,                             # Already batched
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    return loader

# --- Usage ---
if __name__ == "__main__":
    # Pattern 1: S3 streaming
    dataset = S3StreamingDataset(
        s3_uri="s3://my-bucket/training-data",
        local_cache_dir="/mnt/cache/data",
    )
    loader = DataLoader(dataset, batch_size=8, num_workers=2)
    for batch in loader:
        print(f"Batch shapes: {batch['input_ids'].shape}")
        break

    # Pattern 2: WebDataset
    wds_loader = create_webdataset_loader(
        data_pattern="/mnt/data/train/shard-{000000..000999}.tar",
        batch_size=32,
        num_workers=8,
    )
    for tokens, meta in wds_loader:
        print(f"Tokens shape: {tokens.shape}")
        break
```

---

## 6. Networking

### 6.1 Interconnect Comparison

| Interconnect | Speed | Topology | Latency | Effective BW | Best For |
|-------------|:-----:|:--------:|:-------:|:------------:|----------|
| **NVLink 4** | 900 GB/s (GPU-GPU) | Fully connected | <1µs | ~850 GB/s | Intra-node TP communication |
| **NVLink 5** | 1.8 TB/s (GPU-GPU) | Fully connected | <1µs | ~1.7 TB/s | Intra-node TP (B200) |
| **InfiniBand NDR 400** | 400 Gb/s (Node-Node) | Fat tree | 1-2µs | ~46 GB/s | Inter-node gradient sync |
| **InfiniBand XDR** | 800 Gb/s (Node-Node) | Fat tree | 1-2µs | ~92 GB/s | Inter-node (next-gen) |
| **RoCE v2** | 200-400 Gb/s | Various | 3-5µs | ~37 GB/s (200G) | Cost-sensitive clusters |
| **Ethernet** | 100-800 Gb/s | Various | 5-50µs | ~90 GB/s (800G) | General purpose, inference |

**Bandwidth hierarchy in an 8-GPU node (H100 DGX):**
```
GPU 0 ←—— NVLink (900 GB/s) ——→ GPU 1,2,3,4,5,6,7
  ↑                                   ↑
  |                                   |
  |  PCIe Gen5 (128 GB/s per GPU)     |
  |                                   |
GPU NIC ←—— InfiniBand NDR (400 Gb/s) ———→ NICs on other nodes
```

### 6.2 Network Topology Design

| Topology | Description | Best Scale | Pros | Cons |
|----------|-------------|:----------:|------|------|
| **Fat Tree** | Multiple spine switches, leaf switches connect nodes | 64-1024 GPUs | Predictable bisection bandwidth, proven | Higher latency with more hops |
| **Dragonfly** | Groups interconnected in all-to-all pattern | 1024-65536+ GPUs | Low diameter, high scalability | Complex routing, potential congestion |
| **Torus** | 3D/4D mesh of compute nodes | 4096+ (TPU pods) | Simple cabling, good for specific patterns | Higher latency for non-local traffic |
| **Fully connected** (NVSwitch) | All GPUs in node talk at NVLink speed | 8-576 (NVSwitch domain) | Maximum bandwidth for TP | Expensive, limited scale |

**Bisection bandwidth recommendations:**
| Training Scale | Min Bisection BW | Interconnect | Cost Impact |
|:--------------:|:----------------:|:------------:|:-----------:|
| < 64 GPUs | 200 Gb/s per node | RoCE v2 or NDR 200 | Low |
| 64-256 GPUs | 400 Gb/s per node | InfiniBand NDR 400 | Medium |
| 256-1024 GPUs | 800 Gb/s per node | InfiniBand XDR (dual) | High |
| > 1024 GPUs | 1600 Gb/s per node | InfiniBand XDR (quad) | Very High |

### 6.3 Network Planning Guide

**Communication volume estimation:**

```python
# comm_estimation.py — Estimate network bandwidth requirements for training

def estimate_communication(
    model_size_b: float,    # Model size in billions of parameters
    dp_degree: int,         # Data parallelism degree
    tp_degree: int,         # Tensor parallelism degree
    pp_degree: int,         # Pipeline parallelism degree
    gradient_bits: int = 16, # Bits per gradient element (FP16 = 16)
    micro_batch_size: int = 1,
    seq_len: int = 4096,
    hidden_dim: int = 8192,
) -> dict:
    """
    Estimate per-step communication volume for each parallelism strategy.
    """
    params = model_size_b * 1e9  # Total parameters

    # DP: All-reduce of gradients
    dp_comm = params * gradient_bits / 8  # Bytes per GPU per step
    dp_comm_gb = dp_comm / 1e9

    # TP: All-reduce of activations in forward/backward
    activations_per_layer = 2 * micro_batch_size * seq_len * hidden_dim * gradient_bits / 8
    num_tp_layers = 80  # Approximate for 70B-class model
    tp_comm = activations_per_layer * num_tp_layers * 2  # Forward + backward
    tp_comm_gb = tp_comm / 1e9

    # PP: Point-to-point send/recv of activations
    pp_comm = activations_per_layer * num_tp_layers  # Send activations between stages
    pp_comm_gb = pp_comm / 1e9

    return {
        "dp_comm_gb_per_step": round(dp_comm_gb, 2),
        "tp_comm_gb_per_step": round(tp_comm_gb, 2),
        "pp_comm_gb_per_step": round(pp_comm_gb, 2),
        "total_comm_gb_per_step": round(dp_comm_gb + tp_comm_gb + pp_comm_gb, 2),
        "recommended_network_bw": f"{'NDR 400' if dp_comm_gb + tp_comm_gb > 10 else 'RoCE 200'}",
    }

# Example: Llama 3.1 70B with DP=8, TP=8, PP=1
result = estimate_communication(
    model_size_b=70,
    dp_degree=8,
    tp_degree=8,
    pp_degree=1,
)
print(result)
# Output: {"dp_comm_gb_per_step": 140.0, "tp_comm_gb_per_step": 53.68, ...}
```

**Network configuration best practices:**
1. **NIC-to-GPU ratio:** For training nodes, use 1 NIC per 4 GPUs minimum. Recommended: 1 NIC per 2 GPUs.
2. **MTU:** Set jumbo frames (MTU 9000) on all data-plane interfaces. This reduces per-packet overhead by ~6×.
3. **Flow control:** Enable Priority Flow Control (PFC) for RoCE to prevent packet loss. InfiniBand handles this natively.
4. **Congestion management:** Use Adaptive Routing (AR) on InfiniBand switches for optimal load balancing.
5. **Multiple rails:** Configure separate network rails to avoid bandwidth contention between training traffic and storage traffic.
6. **NCCL tuning:** Set `NCCL_IB_TIMEOUT=22`, `NCCL_IB_SL=1`, and `NCCL_ALGO=Ring` for InfiniBand clusters.

```python
# nccl_tuning.py — NCCL environment variables for optimal performance
# Source these before launching training jobs

import os

def configure_nccl_for_training(
    num_gpus: int = 8,
    interconnect: str = "ib",  # "ib" for InfiniBand, "roce" for RoCE
    use_tc: bool = True,       # Use NVLink texture caching
):
    """Set NCCL environment variables for optimal multi-GPU communication."""
    
    # Core NCCL settings
    os.environ["NCCL_DEBUG"] = "WARN"  # Only log warnings/errors
    os.environ["NCCL_IB_DISABLE"] = "0" if interconnect == "ib" else "1"
    
    if interconnect == "ib":
        os.environ["NCCL_IB_TIMEOUT"] = "22"
        os.environ["NCCL_IB_SL"] = "1"
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "8"
    
    if interconnect == "roce":
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Adjust to your RoCE interface
        os.environ["NCCL_NET_PLUGIN"] = "0"         # Default RoCE path
    
    # Ring vs Tree algorithm selection
    os.environ["NCCL_ALGO"] = "Ring"  # Ring: consistent BW, Tree: lower latency
    os.environ["NCCL_PROTO"] = "Simple"
    
    # Multi-node tuning
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"   # Enable GPU Direct RDMA
    os.environ["NCCL_P2P_DISABLE"] = "0"     # Enable P2P (NVLink)
    os.environ["NCCL_NVLS_ENABLE"] = "0"     # NVLink SHARP (DGX only)
    
    # Memory and performance
    os.environ["NCCL_BUFFSIZE"] = str(4 * 1024 * 1024)  # 4 MB NCCL buffer
    os.environ["NCCL_MAX_NCHANNELS"] = str(num_gpus * 4)  # Multiple channels
    os.environ["NCCL_MIN_NCHANNELS"] = str(num_gpus)
    
    if use_tc and num_gpus > 1:
        os.environ["NCCL_GRAPH_DUMP_FILE"] = "/tmp/nccl_graph.json"
    
    print(f"NCCL configured: {num_gpus} GPUs, interconnect={interconnect}")
```

---

## 7. Cross-References

| Reference | Description |
|-----------|-------------|
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production deployment, K8s, serving patterns |
| [05-Enterprise/02-MLOps.md] | MLOps pipeline, model registry, CI/CD for ML |
| [02-LLMs/04-Quantization.md] | Model compression for inference efficiency |
| [02-LLMs/02-Training.md] | Training recipes, loss curves, scaling laws |
| [02-LLMs/06-Fine-Tuning.md] | Fine-tuning methods, LoRA, QLoRA |
| [08-Reference/02-AI-Roadmap.md] | Hardware roadmap, GPU release calendar |
| [08-Reference/01-Glossary.md] | Infrastructure and parallelism terms |
| [08-Reference/04-Cost-Analysis.md] | Cloud vs on-prem cost comparison |

---

## 8. Cloud Provider Comparison for AI Workloads

### 8.1 GPU Instance Comparison

| Provider | Instance Type | GPU | GPU Memory | Interconnect | Price/hr (on-demand) | Price/hr (reserved 1yr) | Best For |
|:---------|:------------|:---:|:----------:|:------------:|:-------------------:|:----------------------:|----------|
| **AWS** | p5.48xlarge | 8× H100 SXM | 80 GB each | NVLink + EFA | $96.00 | $61.44 | Large-scale training, hybrid cloud |
| **AWS** | p4d.24xlarge | 8× A100 40GB | 40 GB each | NVLink + EFA | $32.77 | $19.66 | Medium training, fine-tuning |
| **AWS** | g6.12xlarge | 4× L4 | 24 GB each | — | $5.67 | $3.40 | Small inference, batch processing |
| **Azure** | ND H100 v5 | 8× H100 SXM | 80 GB each | NVLink + InfiniBand | $96.00 | $57.60 | Large training (best NVIDIA partnership) |
| **Azure** | ND A100 v4 | 8× A100 80GB | 80 GB each | NVLink + InfiniBand | $44.16 | $26.50 | Versatile training + inference |
| **Azure** | NC A100 v4 | 4× A100 80GB | 80 GB each | NVLink | $22.08 | $13.25 | Fine-tuning, medium inference |
| **GCP** | a3-megagpu-8g | 8× H100 SXM | 80 GB each | NVLink + GPUDirect-TCPX | $92.40 | $55.44 | Large training (best network) |
| **GCP** | a2-ultragpu-8g | 8× A100 80GB | 80 GB each | NVLink | $55.36 | $33.22 | Training + inference flexibility |
| **GCP** | g2-standard-96 | 4× L4 | 24 GB each | — | $5.68 | $3.41 | Cost-effective inference |
| **Lambda Labs** | 8× H100 | 8× H100 SXM | 80 GB each | NVLink + InfiniBand | $55.00 | — | Best price for bare-metal H100 |
| **CoreWeave** | 8× H100 HGX | 8× H100 SXM | 80 GB each | NVLink + InfiniBand | $44.00 | — | Most cost-effective cloud GPU |

### 8.2 Price-Performance Comparison

| Workload | Best Value | Runner-Up | Cost Difference | Rationale |
|:---------|:-----------|:----------|:---------------:|:----------|
| **Pre-training (7B)** | CoreWeave H100 | Lambda Labs H100 | ~20% cheaper | Specialized GPU clouds undercut hyperscalers by 40-50% on base compute |
| **Pre-training (70B+)** | GCP a3-megagpu (TCPX) | Azure ND H100 v5 | ~4% cheaper | GCP's GPUDirect-TCPX gives 2-5% better training throughput |
| **Fine-tuning (LoRA)** | AWS g6 (L4) | GCP g2 (L4) | Comparable | L4 GPUs are 3-5× cheaper than H100 for fine-tuning workloads |
| **Inference (real-time)** | Azure NC A100 v4 | AWS p4d | ~30% cheaper | A100 80GB provides best cost/performance for serving |
| **Inference (batch)** | Lambda Labs H100 | CoreWeave H100 | ~5% cheaper | Bare-metal reduces overhead for sustained batch jobs |
| **Development/test** | AWS g6 (L4) | GCP g2 (L4) | $5/hr class | L4 is the cheapest GPU with FP8 support, adequate for dev workloads |

### 8.3 Key Differentiators

| Factor | AWS | Azure | GCP | Specialized (Lambda, CoreWeave, RunPod) |
|:-------|:---:|:-----:|:---:|:---------------------------------------:|
| **GPU availability** | Moderate (quotas) | Moderate (quotas) | High (preemptible) | Very high (no quotas) |
| **Preemptible/spot pricing** | 60-70% discount | 60-70% discount | 60-91% discount | 50-70% discount |
| **Inter-node networking** | EFA (100 Gb/s) | InfiniBand (400 Gb/s) | GPUDirect-TCPX (200 Gb/s) | InfiniBand (400-800 Gb/s) |
| **Managed training** | SageMaker | Azure ML | Vertex AI | None (raw compute) |
| **Storage integration** | FSx for Lustre, S3 | Azure Blob + Lustre | Filestore, GCS | Object store (S3-compatible) |
| **Kubernetes integration** | EKS + Karpenter | AKS | GKE (best-in-class) | Manual or Kubernetes-only |
| **Support quality** | Enterprise (slow) | Enterprise (medium) | Enterprise + dedicated | Community + Slack |
| **Hidden costs** | Data egress ($0.09/GB) | Data egress ($0.05/GB) | Data egress ($0.12/GB) | Minimal egress costs |

### 8.4 Multi-Cloud Strategy Code Example

```python
# cloud_provisioning.py — Programmatic GPU instance selection across clouds
# Demonstrates cost-aware instance selection for training jobs

from dataclasses import dataclass
from typing import Optional

@dataclass
class GPUInstance:
    provider: str
    instance_type: str
    gpu_count: int
    gpu_type: str
    gpu_memory_gb: int
    interconnect: str
    hourly_cost: float
    spot_hourly_cost: Optional[float]

# Common GPU configurations across clouds
INSTANCE_CATALOG: list[GPUInstance] = [
    # Hyper scalers
    GPUInstance("aws", "p5.48xlarge", 8, "H100", 80, "nvlink+efa", 96.00, 28.80),
    GPUInstance("azure", "ND H100 v5", 8, "H100", 80, "nvlink+ib", 96.00, 28.80),
    GPUInstance("gcp", "a3-megagpu-8g", 8, "H100", 80, "nvlink+tcpx", 92.40, 18.48),
    # Specialized providers
    GPUInstance("coreweave", "8xH100-HGX", 8, "H100", 80, "nvlink+ib", 44.00, None),
    GPUInstance("lambdalabs", "8xH100-SXM", 8, "H100", 80, "nvlink+ib", 55.00, None),
    # Budget options
    GPUInstance("aws", "g6.12xlarge", 4, "L4", 24, "none", 5.67, 1.70),
    GPUInstance("gcp", "g2-standard-96", 4, "L4", 24, "none", 5.68, 1.14),
]

def select_best_instance(
    gpu_type: str = "H100",
    min_gpus: int = 8,
    max_hourly_budget: float = 100.0,
    use_spot: bool = False,
    require_interconnect: bool = True,
) -> list[GPUInstance]:
    \"\"\"Select the best GPU instances matching constraints, sorted by cost.\"\"\"
    candidates = []
    for inst in INSTANCE_CATALOG:
        cost = inst.spot_hourly_cost if use_spot and inst.spot_hourly_cost else inst.hourly_cost
        if not inst.gpu_type.startswith(gpu_type):
            continue
        if inst.gpu_count < min_gpus:
            continue
        if cost is not None and cost > max_hourly_budget:
            continue
        if require_interconnect and inst.interconnect == "none":
            continue
        candidates.append((cost, inst))
    
    candidates.sort(key=lambda x: x[0] if x[0] else float('inf'))
    return [inst for _, inst in candidates]

# Example: find cheapest H100 cluster for pre-training
best = select_best_instance(gpu_type="H100", min_gpus=8, max_hourly_budget=100)
print("Cheapest H100 clusters:")
for inst in best[:5]:
    cost = inst.spot_hourly_cost or inst.hourly_cost
    print(f"  {inst.provider:12s} {inst.instance_type:20s} ${cost:.2f}/hr ({'spot' if inst.spot_hourly_cost else 'on-demand'})")
# Output:
#   coreweave   8xH100-HGX           $44.00/hr (on-demand)
#   lambdalabs  8xH100-SXM           $55.00/hr (on-demand)
#   gcp         a3-megagpu-8g        $18.48/hr (spot)
#   aws         p5.48xlarge          $28.80/hr (spot)
#   azure       ND H100 v5           $28.80/hr (spot)
```

### 8.5 Cost Optimization Strategies by Workload Type

| Strategy | Description | Savings | Best For | Caveat |
|:---------|:------------|:-------:|:---------|:-------|
| **Spot/preemptible instances** | Use spare capacity at 60-90% discount | 60-90% | Pre-training, fine-tuning (fault-tolerant) | Instances can terminate with 30s notice; checkpoint frequently |
| **Reserved instances (1yr)** | Commit to steady-state usage | 35-40% | Production inference, continuous training | Requires accurate capacity planning |
| **Multi-cloud bidding** | Run on cheapest provider at any moment | 10-30% | Elastic training jobs | Orchestration complexity; data transfer costs |
| **GPU pooling** | Share GPU pool across teams via K8s | 20-40% | Enterprise with multiple ML teams | Requires robust cluster management |
| **Preemption-aware training** | Save checkpoint every N steps, resume on preemption | — | Long-running training jobs | Adds checkpoint storage costs (+5-10%) |
| **Right-sizing** | Choose correct GPU type for workload | 30-70% | Inference (L4 vs H100) | Over-provisioning is the #1 cost waste in AI infra |

### 8.6 Cloud Provider Selection Flowchart

```
Q: What's your primary workload?
├── Pre-training > 70B params → GCP a3-megagpu (best network) or CoreWeave (best price)
├── Pre-training < 70B params → CoreWeave / Lambda Labs (cost) or Azure (managed)
├── Fine-tuning (LoRA/QLoRA) → AWS g6 (L4) or any L4 instance (cheapest)
├── Production inference
│   ├── Real-time (< 100ms) → Azure NC A100 v4 or dedicated H200
│   ├── High-throughput → GCP a2 (A100, large batch) or vLLM on any H100
│   └── Cost-sensitive → AWS g6 (L4, FP8) or quantized on A10G
├── Dev / experimentation → Spot instances (any cloud) or Colab (free T4)
└── Enterprise with compliance → Azure (best enterprise features) or AWS (most compliance certs)
```

### 8.7 Cross-References for Cloud

| Reference | Description |
|-----------|-------------|
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production deployment patterns, K8s, serving |
| [08-Reference/04-Cost-Analysis.md] | Detailed cloud vs on-prem cost analysis |
| [08-Reference/02-AI-Roadmap.md] | GPU release calendar, hardware evolution |
| [02-LLMs/04-Quantization.md] | Model compression to reduce GPU needs |

---

*Document version: 2.5 — June 2026 | Expanded: added §8 Cloud Provider Comparison — GPU instance comparison tables, price-performance analysis, multi-cloud code example, cost optimization strategies, selection flowchart. Previously v2.0.*
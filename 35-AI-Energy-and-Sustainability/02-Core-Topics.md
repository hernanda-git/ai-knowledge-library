# AI Energy and Sustainability: Core Topics

> **Category:** 35 — AI Energy and Sustainability  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](./01-Overview.md), [30-Small-Language-Models/](../30-Small-Language-Models/), [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/), [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Energy Hierarchy of AI Workloads](#2-the-energy-hierarchy-of-ai-workloads)
3. [Efficient Model Architectures](#3-efficient-model-architectures)
4. [Quantization and Precision](#4-quantization-and-precision)
5. [Distillation and Pruning](#5-distillation-and-pruning)
6. [Mixture-of-Experts (MoE)](#6-mixture-of-experts-moe)
7. [Small Language Models (SLMs)](#7-small-language-models-slms)
8. [Serving Efficiency: Batching and KV-Cache](#8-serving-efficiency-batching-and-kv-cache)
9. [Scheduling and Utilization](#9-scheduling-and-utilization)
10. [Lifecycle Energy: Train vs Inference](#10-lifecycle-energy-train-vs-inference)
11. [Comparison Tables](#11-comparison-tables)
12. [Practical Decision Framework](#12-practical-decision-framework)

---

## 1. Executive Summary

The cheapest watt is the one you never spend. Core sustainability work in AI is about **doing less compute for the same outcome**. This document surveys the model- and system-level levers, with concrete techniques and tradeoffs.

```
ENERGY REDUCTION LEVERS (roughly largest → smallest impact):
─────────────────────────────────────────────────────────────
1. Use a smaller model for the task        (10–100×)
2. Quantize (FP16 → INT4)                  (2–4×)
3. Distill / prune                          (1.5–3×)
4. MoE (activate subset of params)         (2–5× at inference)
5. Batch + KV-cache reuse                   (2–10×)
6. Better scheduling / utilization         (1.2–2×)
7. Cleaner power (PPA/SMR)                  (carbon, not watt)
─────────────────────────────────────────────────────────────
```

---

## 2. The Energy Hierarchy of AI Workloads

Not all tokens cost the same:

| Workload | Energy driver | Optimization target |
|----------|---------------|---------------------|
| Pretraining | total FLOPs | architecture + hardware |
| Fine-tuning | dataset × epochs | LoRA / parameter-efficient |
| Embedding / RAG retrieve | vector search | indexing, caching |
| Chat inference | decode steps × model size | quantization, batching |
| Long-context (see [36-Long-Context-AI/](../36-Long-Context-AI/)) | KV-cache memory | KV quantization, eviction |

---

## 3. Efficient Model Architectures

### 3.1 Why architecture matters

Energy ≈ FLOPs × energy-per-FLOP. Reducing FLOPs per token is the highest-leverage move.

| Technique | FLOPs impact | Quality impact | Complexity |
|-----------|--------------|----------------|------------|
| Smaller model | −90% | −5–20% | low |
| MoE | −50–80% (inference) | ~0 | high |
| Sparse attention | −30–60% | −1–5% | med |
| Rotary/optimized attention | −10–20% | 0 | low |

---

## 4. Quantization and Precision

Quantization reduces the bits per weight/activation. Lower precision = less memory bandwidth = less energy.

```python
# PyTorch dynamic quantization (INT8) — quick win for inference
import torch
from torch.quantization import quantize_dynamic

model = MyLLM().eval()
qmodel = quantize_dynamic(
    model,
    {torch.nn.Linear},       # layers to quantize
    dtype=torch.qint8
)
# Memory ↓ ~4× (FP32→INT8), energy ↓ via reduced bandwidth
```

| Precision | Bits | Relative energy* | Use |
|-----------|------|------------------|-----|
| FP32 | 32 | 1.0 (baseline) | training (rare now) |
| FP16/BF16 | 16 | ~0.5 | training/inference |
| INT8 | 8 | ~0.25 | inference |
| INT4 | 4 | ~0.13 | inference (aggressive) |
| INT2/ternary | 2–3 | ~0.07 | research/extreme |

\*Relative to FP32 memory-bandwidth proxy; actual energy depends on hardware MAC efficiency.

### 4.1 Pitfalls
- **Accuracy cliffs** below INT4 for some models — always eval.
- **Unsupported ops** may dequantize silently, killing savings.
- **KV-cache quantization** often safer than weight quantization for quality.

---

## 5. Distillation and Pruning

### Knowledge distillation
Train a small "student" to mimic a large "teacher":

```python
# Conceptual distillation loss
loss = alpha * CE(student_logits, targets) \
     + (1 - alpha) * KL_div(softmax(teacher / T), softmax(student / T))
# Student is 5–10× smaller → far lower serving energy
```

### Structured pruning
Remove whole channels/heads; keeps hardware efficiency (unlike unstructured).

| Method | Energy gain | Quality risk | Effort |
|--------|-------------|--------------|--------|
| Distillation | 2–10× | low–med | med |
| Structured pruning | 1.5–3× | low | med |
| Unstructured pruning | 2–4× | med | high (needs sparse kernels) |

---

## 6. Mixture-of-Experts (MoE)

MoE activates only a subset of experts per token → same parameter count, far fewer FLOPs/token.

```python
# Pseudocode: routed expert computation
def moe_forward(x, experts, router):
    gates, indices = router(x)            # route to top-k experts
    out = 0
    for k in range(top_k):
        e = experts[indices[k]]
        out += gates[k] * e(x)            # only top_k experts run
    return out                            # ~ top_k/total FLOPs used
```

| Model class | Active params | Energy/token |
|-------------|---------------|--------------|
| Dense 70B | 70B | high |
| MoE 8×7B (top-2) | ~14B active | ~5× less |

---

## 7. Small Language Models (SLMs)

On-device and task-specific SLMs (see [30-Small-Language-Models/](../30-Small-Language-Models/)) slash energy by running locally — no data-center round trip, no large model.

| SLM | Params | Typical use | Energy |
|-----|--------|-------------|--------|
| Phi-3-mini | 3.8B | edge copilots | very low |
| Gemma 2 2B | 2B | on-device | very low |
| Llama 3.2 1B/3B | 1–3B | mobile | very low |

Edge inference also cuts **network energy** and latency — a double win (see [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/)).

---

## 8. Serving Efficiency: Batching and KV-Cache

### Continuous batching
Serving frameworks (vLLM, TensorRT-LLM, TGI) batch many requests, raising GPU utilization from ~10% to ~80%+.

```
Without batching:  GPU 12% utilized  → most watts wasted
With continuous batch: GPU 85% utilized → same watts, 7× throughput
```

### KV-cache management
Long contexts blow up KV-cache memory (and energy to move it). Techniques:
- PagedAttention (vLLM) — memory efficiency.
- KV quantization (INT8/INT4).
- Eviction / windowed attention for long docs (see [36-Long-Context-AI/](../36-Long-Context-AI/)).

---

## 9. Scheduling and Utilization

The biggest *wasted* energy is idle but powered-on hardware.

| Strategy | Effect |
|----------|--------|
| Consolidation (bin-pack jobs) | fewer idle GPUs |
| Preemptible/spot training | use otherwise-idle capacity |
| Autoscaling inference | scale to zero when quiet |
| Carbon-aware scheduling | run flexible jobs on clean hours |

```yaml
# Example: carbon-aware training cron (conceptual)
schedule:
  flexible_jobs:
    run_when: grid_cfe > 0.8   # only when clean
    pause_when: grid_cfe < 0.4
  latency_critical:
    always_on: true            # separate tier
```

---

## 10. Lifecycle Energy: Train vs Inference

| Phase | One-time? | Share of lifetime CO₂ |
|-------|-----------|------------------------|
| Pretraining | yes | 10–40% |
| Fine-tuning | occasional | 5–15% |
| Inference | continuous | 50–85% |

**Implication:** For deployed products, inference dominates. Optimize the served model hard.

---

## 11. Comparison Tables

### Technique scorecard

| Technique | Energy ↓ | Quality ↓ | Effort | When to use |
|-----------|----------|-----------|--------|-------------|
| Smaller model | ★★★★★ | ★★ | ★ | always first option |
| INT8 quant | ★★★ | ★ | ★ | most inference |
| INT4 quant | ★★★★ | ★★ | ★★ | tolerant tasks |
| Distillation | ★★★★ | ★★ | ★★★ | have teacher |
| MoE | ★★★★ | ★ | ★★★★ | large serving |
| LoRA ft | ★★ | ★ | ★★ | adaptation |
| Edge SLM | ★★★★★ | ★★ | ★★★ | on-device |

---

## 12. Practical Decision Framework

```
START
  │
  ├─ Can a smaller model do it? ──YES──► use SLM / smaller LLM (biggest win)
  │                                     quantize to INT8/INT4
  ├─ Must use large model?
  │     ├─ Inference heavy? ──► batch, KV-quant, continuous batching
  │     ├─ Train heavy?   ──► LoRA/PEFT, carbon-aware scheduling
  │     └─ Long context?  ──► KV eviction, windowed attention
  └─ Report: measure kWh + carbon, disclose (compliance)
```

**Rule of thumb:** A 7B model served at INT8 on a single efficient GPU often beats a 70B model on a cluster for 80% of enterprise tasks — at ~20× less energy (see [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/)).

---

*See also: [01-Overview.md](./01-Overview.md) · [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md) · [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md)*
